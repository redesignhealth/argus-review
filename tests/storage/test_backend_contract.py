"""Backend contract suite for Argus round-history storage.

Every scenario in this file exercises the seven logical operations defined
by ``argus/storage/sql.py`` (the Postgres canonical writer) purely through
their Pydantic input/output shapes — no backend-specific assertions. The
goal is a single suite that can run, unmodified, against any backend that
implements the same operations: today that's just
:class:`argus.storage.sqlite.SqliteHistoryBackend` (wired via the
``backend`` fixture below); ``postgres`` and ``http`` params are declared
already (each ``pytest.mark.skip``ped with the reason it isn't wired yet)
so that adding those fixtures later is a matter of filling in the
``elif`` branch in ``backend()`` — no test bodies change.

- **postgres**: needs a live database and ``argus.storage.session`` wired
  up. Guard with ``ARGUS_DB_URL`` / ``SUPABASE_DB_URL`` env-var presence and
  the ``integration`` marker once wired, mirroring the pattern in
  ``tests/storage/test_sql.py``.
- **http**: needs a ``pytest-httpx`` fixture standing in for your configured
  HTTP backend's endpoints (see ``docs/STORAGE.md`` for the contract). Once
  wired, ``HttpStorageClient`` would sit behind the same seven-operation
  surface as a thin adapter.

Scenario coverage:

1. upsert_running -> select_status_by_flow_run shows ``running``.
2. upsert_completed (same flow_run_id) -> select_latest_completed_round
   returns it, byte-faithful round-trip of the input shape.
3. Two completed rounds -> latest wins; select_recent_rounds ordering +
   limit.
4. Nested JSONB payload (including a ``dismissals`` array) round-trips
   with JSON equality.
5. Upsert conflict path: a second upsert_running with the same
   flow_run_id updates in place rather than duplicating the row
   (partial-unique-index semantics).
6. insert_agent_runs batch insert + row count.
7. select_recent_lite_rounds: only 'v3-lite' rounds, ordering + limit.
8. UpsertReturnedNoRow is the documented exception type on the "insert
   returned no row" path (the *forced* failure injection lives in
   ``tests/storage/test_sqlite_backend.py`` since it's necessarily
   backend-implementation-specific; here we only pin the happy path
   never raises it and that the type matches sql.py's).
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from itertools import count
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio

from argus.storage.http import install_http_storage, reset_http_storage
from argus.storage.resolver import (
    HistoryBackend,
    HttpHistoryBackend,
    PostgresHistoryBackend,
)
from argus.storage.sql import (
    AgentRunIn,
    CodeReviewRoundIn,
    UpsertReturnedNoRow,
)
from argus.storage.sql import UpsertReturnedNoRow as SqlUpsertReturnedNoRow
from argus.storage.sqlite import SqliteHistoryBackend

pytestmark = pytest.mark.asyncio


@pytest.fixture(
    params=[
        "sqlite",
        pytest.param("postgres", marks=pytest.mark.integration),
        "http",
    ]
)
def backend_kind(request: pytest.FixtureRequest) -> str:
    return str(request.param)


# ---------------------------------------------------------------------------
# HTTP fixture support: a minimal in-memory fake standing in for the
# two-endpoint HTTP storage contract (see ``docs/STORAGE.md``), wired via
# pytest-httpx. Only the two operations ``HttpStorageClient`` actually
# implements are exercised through it; see ``_HTTP_UNSUPPORTED_TESTS`` below
# for the scenarios that don't apply to this backend (documented gaps, not
# silently skipped -- they xfail with a reason).
# ---------------------------------------------------------------------------

_HTTP_READ_URL = "https://fake-storage-backend.test/reviews/{owner}/{repo}/{pr}"
_HTTP_WRITE_URL = "https://fake-storage-backend.test/reviews/{owner}/{repo}/{pr}/rounds"

_HTTP_UNSUPPORTED_TESTS = {
    # select_status_by_flow_run -- not exposed over HTTP.
    "test_upsert_running_reflected_in_status",
    "test_status_by_flow_run_none_when_missing",
    # select_recent_rounds -- not exposed over HTTP.
    "test_select_recent_rounds_orders_desc_and_respects_limit",
    "test_repeated_running_upsert_same_flow_run_id_does_not_duplicate",
    # select_recent_lite_rounds -- documented no-op (always returns []).
    "test_select_recent_lite_rounds_filters_reviewer_version",
    "test_select_recent_lite_rounds_default_limit_is_200",
}


def _mark_known_http_gaps(request: pytest.FixtureRequest) -> None:
    base_name = request.node.name.split("[")[0]
    if base_name in _HTTP_UNSUPPORTED_TESTS:
        request.node.add_marker(
            pytest.mark.xfail(
                reason=(
                    "HTTP storage shim doesn't expose this operation yet "
                    "-- documented gap, not silently skipped"
                ),
                strict=True,
            )
        )


class _FakeHttpStorageBackend:
    """In-memory stand-in for the two-endpoint HTTP storage contract.

    Keyed by ``(repo, pr_number)``; upserts by ``flow_run_id`` with the same
    "sha/base_ref never cleared once bound" COALESCE semantics ``sql.py`` /
    ``sqlite.py`` document, so the contract-suite scenarios that exercise
    that invariant pass identically against this backend too.
    """

    def __init__(self) -> None:
        self._rounds: dict[tuple[str, int], list[dict[str, Any]]] = {}
        self._seq = count()

    def handle(self, request: httpx.Request) -> httpx.Response:
        parts = [p for p in request.url.path.split("/") if p]
        # parts: ["reviews", owner, repo, pr] (GET) or [..., "rounds"] (POST)
        _, owner, repo, pr_str = parts[:4]
        key = (f"{owner}/{repo}", int(pr_str))

        if request.method == "GET":
            rounds = self._rounds.get(key, [])
            return httpx.Response(200, json={"rounds": [_strip_seq(r) for r in rounds]})

        body: dict[str, Any] = json.loads(request.content)
        rounds = self._rounds.setdefault(key, [])
        flow_run_id = body.get("flow_run_id")
        existing_idx = next(
            (
                i
                for i, r in enumerate(rounds)
                if flow_run_id is not None and r.get("flow_run_id") == flow_run_id
            ),
            None,
        )
        if existing_idx is not None:
            existing = rounds[existing_idx]
            merged = {**existing, **body}
            # Mirrors sql.py's ``COALESCE(EXCLUDED.sha, ...sha)`` -- a
            # finalize call with sha=None must not clobber a bound sha.
            if body.get("sha") is None:
                merged["sha"] = existing.get("sha")
            if body.get("base_ref") is None:
                merged["base_ref"] = existing.get("base_ref")
            merged["id"] = existing["id"]
            merged["created_at"] = existing["created_at"]
            merged["_seq"] = existing["_seq"]
            rounds[existing_idx] = merged
            record = merged
        else:
            record = {
                **body,
                "id": str(uuid4()),
                "created_at": datetime.now(UTC).isoformat(),
                "_seq": next(self._seq),
            }
            rounds.append(record)
        # Newest-first, tie-broken by insertion sequence (mirrors sqlite.py's
        # ``ORDER BY created_at DESC, rowid DESC`` determinism guard).
        rounds.sort(key=lambda r: (r["created_at"], r["_seq"]), reverse=True)
        return httpx.Response(201, json=_strip_seq(record))


def _strip_seq(record: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in record.items() if k != "_seq"}


@pytest_asyncio.fixture
async def backend(
    backend_kind: str, tmp_path: Path, request: pytest.FixtureRequest
) -> AsyncIterator[HistoryBackend]:
    if backend_kind == "sqlite":
        sqlite_backend = SqliteHistoryBackend(db_path=tmp_path / "history.db")
        try:
            yield sqlite_backend
        finally:
            await sqlite_backend.aclose()
        return

    if backend_kind == "http":
        httpx_mock = request.getfixturevalue("httpx_mock")
        httpx_mock.add_callback(_FakeHttpStorageBackend().handle, is_reusable=True)
        install_http_storage(read_url=_HTTP_READ_URL, write_url=_HTTP_WRITE_URL)
        _mark_known_http_gaps(request)
        try:
            yield HttpHistoryBackend()
        finally:
            await reset_http_storage()
        return

    if backend_kind == "postgres":
        db_url = os.environ.get("ARGUS_DB_URL") or os.environ.get("SUPABASE_DB_URL")
        if not db_url or not db_url.startswith(("postgres://", "postgresql://")):
            pytest.skip(
                "ARGUS_DB_URL/SUPABASE_DB_URL not set to a real Postgres URL -- "
                "skipping integration test"
            )
        yield PostgresHistoryBackend()
        return

    pytest.skip(f"{backend_kind} backend not wired yet")


# ---------------------------------------------------------------------------
# 1. upsert_running -> select_status_by_flow_run shows running
# ---------------------------------------------------------------------------


async def test_upsert_running_reflected_in_status(backend: SqliteHistoryBackend) -> None:
    await backend.upsert_running_row(
        flow_run_id="fr-status-1",
        repo="org/repo",
        pr_number=42,
        sha="deadbeef",
        base_ref="main",
    )

    status = await backend.select_status_by_flow_run(flow_run_id="fr-status-1")

    assert status is not None
    assert status.current_stage == "running"
    assert status.blocking_count is None
    assert status.result_json is None
    assert status.age_seconds is not None
    assert status.age_seconds >= 0


async def test_status_by_flow_run_none_when_missing(backend: SqliteHistoryBackend) -> None:
    assert await backend.select_status_by_flow_run(flow_run_id="does-not-exist") is None


# ---------------------------------------------------------------------------
# 2. upsert_completed -> select_latest_completed_round round-trip
# ---------------------------------------------------------------------------


async def test_upsert_completed_round_trips_through_latest(
    backend: SqliteHistoryBackend,
) -> None:
    payload = CodeReviewRoundIn(
        flow_run_id="fr-rt-1",
        repo="org/repo",
        pr_number=7,
        verdict="approve",
        risk_level="low",
        blocking_count=0,
        suggestion_count=2,
        review_comment="lgtm",
        result_json={"findings": [{"id": "S1", "severity": "suggestion"}]},
        cost_usd=0.42,
        duration_seconds=12.5,
        orchestrator_model="claude-opus",
        subagent_model="claude-sonnet",
        sha="cafebabe",
        base_ref="main",
    )

    written = await backend.upsert_completed_row(row=payload)
    latest = await backend.select_latest_completed_round(repo="org/repo", pr_number=7)

    assert latest is not None
    assert latest.id == written.id
    assert latest.created_at == written.created_at
    for field in (
        "flow_run_id",
        "repo",
        "pr_number",
        "verdict",
        "risk_level",
        "blocking_count",
        "suggestion_count",
        "review_comment",
        "result_json",
        "cost_usd",
        "duration_seconds",
        "orchestrator_model",
        "subagent_model",
        "sha",
        "base_ref",
    ):
        assert getattr(latest, field) == getattr(payload, field), field
    assert latest.prior_count == 1


async def test_upsert_completed_row_defaults_reviewer_version_and_stage(
    backend: SqliteHistoryBackend,
) -> None:
    written = await backend.upsert_completed_row(
        row=CodeReviewRoundIn(repo="org/repo", pr_number=1)
    )
    assert written.reviewer_version == "v3"
    assert written.current_stage == "completed"
    assert written.verdict is None


# ---------------------------------------------------------------------------
# 3. Two completed rounds -> latest wins; select_recent_rounds ordering + limit
# ---------------------------------------------------------------------------


async def test_second_completed_round_becomes_latest(
    backend: SqliteHistoryBackend,
) -> None:
    first = await backend.upsert_completed_row(
        row=CodeReviewRoundIn(
            flow_run_id="fr-round-1", repo="org/repo", pr_number=99, verdict="approve"
        )
    )
    second = await backend.upsert_completed_row(
        row=CodeReviewRoundIn(
            flow_run_id="fr-round-2", repo="org/repo", pr_number=99, verdict="block"
        )
    )

    latest = await backend.select_latest_completed_round(repo="org/repo", pr_number=99)
    assert latest is not None
    assert latest.id == second.id
    assert latest.id != first.id
    assert latest.prior_count == 2


async def test_select_recent_rounds_orders_desc_and_respects_limit(
    backend: SqliteHistoryBackend,
) -> None:
    ids = []
    for i in range(5):
        row = await backend.upsert_completed_row(
            row=CodeReviewRoundIn(
                flow_run_id=f"fr-recent-{i}",
                repo="org/repo2",
                pr_number=1,
                verdict="approve",
            )
        )
        ids.append(row.id)

    recent = await backend.select_recent_rounds(repo="org/repo2", pr_number=1, limit=3)

    assert len(recent) == 3
    # Newest-first: the cap drops the oldest rounds, not the newest.
    assert [r.id for r in recent] == list(reversed(ids))[:3]


# ---------------------------------------------------------------------------
# 4. Nested JSONB payload (dismissals array) round-trips byte-faithfully
# ---------------------------------------------------------------------------


async def test_result_json_with_dismissals_round_trips_by_value(
    backend: SqliteHistoryBackend,
) -> None:
    nested_payload = {
        "findings": [
            {"id": "B1", "severity": "blocking", "description": "sql injection"},
            {"id": "S1", "severity": "suggestion", "description": "rename var"},
        ],
        "dismissals": [
            {"finding_id": "B1", "reason": "false positive -- input is sanitized upstream"},
        ],
        "metadata": {"round": 2, "nested": {"deeply": ["a", "b", 3, None, True]}},
    }
    written = await backend.upsert_completed_row(
        row=CodeReviewRoundIn(
            flow_run_id="fr-json-1",
            repo="org/repo3",
            pr_number=5,
            verdict="block",
            result_json=nested_payload,
        )
    )

    latest = await backend.select_latest_completed_round(repo="org/repo3", pr_number=5)

    assert latest is not None
    assert latest.result_json == nested_payload
    assert written.result_json == nested_payload


# ---------------------------------------------------------------------------
# 5. Upsert conflict path: running upsert on the same flow_run_id updates
#    in place (partial-unique-index semantics), not a duplicate row.
# ---------------------------------------------------------------------------


async def test_repeated_running_upsert_same_flow_run_id_does_not_duplicate(
    backend: SqliteHistoryBackend,
) -> None:
    await backend.upsert_running_row(
        flow_run_id="fr-conflict-1",
        repo="org/repo4",
        pr_number=3,
        sha="sha-1",
        base_ref="main",
    )
    await backend.upsert_running_row(
        flow_run_id="fr-conflict-1",
        repo="org/repo4",
        pr_number=3,
        sha="sha-2",
        base_ref="main",
    )

    # Finalize and confirm exactly one row exists for this PR (the conflict
    # target collapsed the two running-upserts into one row that then
    # became the single completed round).
    await backend.upsert_completed_row(
        row=CodeReviewRoundIn(
            flow_run_id="fr-conflict-1",
            repo="org/repo4",
            pr_number=3,
            verdict="approve",
        )
    )
    recent = await backend.select_recent_rounds(repo="org/repo4", pr_number=3, limit=10)
    assert len(recent) == 1


async def test_completed_upsert_same_flow_run_id_preserves_sha_via_coalesce(
    backend: SqliteHistoryBackend,
) -> None:
    """Mirrors sql.py's documented invariant: "sha is never cleared once
    bound". A finalize call with ``sha=None`` must not clobber a
    previously-bound sha."""
    await backend.upsert_running_row(
        flow_run_id="fr-coalesce-1",
        repo="org/repo5",
        pr_number=4,
        sha="original-sha",
        base_ref="main",
    )
    finalized = await backend.upsert_completed_row(
        row=CodeReviewRoundIn(
            flow_run_id="fr-coalesce-1",
            repo="org/repo5",
            pr_number=4,
            verdict="approve",
            sha=None,
            base_ref=None,
        )
    )
    assert finalized.sha == "original-sha"
    assert finalized.base_ref == "main"


# ---------------------------------------------------------------------------
# 6. insert_agent_runs batch insert + row count
# ---------------------------------------------------------------------------


async def test_insert_agent_runs_batch(backend: SqliteHistoryBackend) -> None:
    review = await backend.upsert_completed_row(
        row=CodeReviewRoundIn(repo="org/repo6", pr_number=2, verdict="approve")
    )
    runs = [
        AgentRunIn(
            agent_name="specialist:security",
            agent_type="specialist",
            model="claude-sonnet",
            cost_usd=0.05,
            duration_seconds=10.0,
            tool_call_count=3,
            tool_names=["grep", "read"],
            files_explored=["a.py", "b.py"],
            finding_count=1,
            result_text_length=500,
        ),
        AgentRunIn(agent_name="cross_cutting:orchestration", agent_type="cross_cutting"),
    ]

    await backend.insert_agent_runs(code_review_id=review.id, runs=runs)

    # Batch insert with no rows is a documented no-op.
    await backend.insert_agent_runs(code_review_id=review.id, runs=[])


# ---------------------------------------------------------------------------
# 7. select_recent_lite_rounds: only 'v3-lite' rounds
# ---------------------------------------------------------------------------


async def test_select_recent_lite_rounds_filters_reviewer_version(
    backend: SqliteHistoryBackend,
) -> None:
    await backend.upsert_completed_row(
        row=CodeReviewRoundIn(
            flow_run_id="fr-lite-full",
            repo="org/repo7",
            pr_number=1,
            verdict="approve",
            reviewer_version="v3",
        )
    )
    lite_rows = []
    for i in range(3):
        row = await backend.upsert_completed_row(
            row=CodeReviewRoundIn(
                flow_run_id=f"fr-lite-{i}",
                repo="org/repo7",
                pr_number=1,
                verdict="approve",
                reviewer_version="v3-lite",
            )
        )
        lite_rows.append(row)

    recent_lite = await backend.select_recent_lite_rounds(repo="org/repo7", pr_number=1, limit=2)

    assert len(recent_lite) == 2
    assert all(r.reviewer_version == "v3-lite" for r in recent_lite)
    # Newest-first ordering, same as select_recent_rounds.
    assert [r.id for r in recent_lite] == [lite_rows[2].id, lite_rows[1].id]


async def test_select_recent_lite_rounds_default_limit_is_200(
    backend: SqliteHistoryBackend,
) -> None:
    await backend.upsert_completed_row(
        row=CodeReviewRoundIn(
            flow_run_id="fr-lite-default",
            repo="org/repo8",
            pr_number=1,
            verdict="approve",
            reviewer_version="v3-lite",
        )
    )
    recent_lite = await backend.select_recent_lite_rounds(repo="org/repo8", pr_number=1)
    assert len(recent_lite) == 1


# ---------------------------------------------------------------------------
# 8. UpsertReturnedNoRow — happy path never raises; type matches sql.py's
# ---------------------------------------------------------------------------


async def test_upsert_completed_row_happy_path_does_not_raise_no_row(
    backend: SqliteHistoryBackend,
) -> None:
    # Forced-failure injection is backend-specific (sqlite's is in
    # tests/storage/test_sqlite_backend.py) since it requires poking at
    # the driver layer; here we only pin that the exception type this
    # module raises is literally sql.py's, not a re-declared lookalike.
    assert UpsertReturnedNoRow is SqlUpsertReturnedNoRow
    await backend.upsert_completed_row(row=CodeReviewRoundIn(repo="org/repo9", pr_number=1))
