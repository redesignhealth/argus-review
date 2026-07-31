"""SQLite-backend-specific edge cases for ``argus.storage.sqlite``.

Complements ``tests/storage/test_backend_contract.py`` (backend-agnostic
operation semantics) with cases that only make sense against this specific
implementation: file bootstrap, concurrent-open behavior, timestamp/epoch
math, JSON fidelity for deeply nested payloads, and the
``UpsertReturnedNoRow`` forced-failure path.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from argus.storage.sql import AgentRunIn, CodeReviewRoundIn, UpsertReturnedNoRow
from argus.storage.sqlite import DEFAULT_DB_PATH, SqliteHistoryBackend

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fresh-file DDL bootstrap + parent-dir creation
# ---------------------------------------------------------------------------


async def test_fresh_file_creates_parent_dirs_and_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "dirs" / "history.db"
    assert not db_path.parent.exists()

    backend = SqliteHistoryBackend(db_path=db_path)
    try:
        # Any operation triggers lazy bootstrap.
        result = await backend.select_latest_completed_round(repo="org/repo", pr_number=1)
        assert result is None
    finally:
        await backend.aclose()

    assert db_path.exists()
    assert db_path.parent.is_dir()

    # Schema landed for real: introspect with a plain stdlib connection.
    conn = sqlite3.connect(db_path)
    try:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert {"code_reviews", "agent_runs"} <= tables
        indexes = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
        }
        assert "uq_code_reviews_flow_run_id" in indexes
    finally:
        conn.close()


async def test_default_db_path_lives_under_local_share_argus() -> None:
    assert DEFAULT_DB_PATH == Path.home() / ".local" / "share" / "argus" / "history.db"


async def test_reopening_existing_file_preserves_data(tmp_path: Path) -> None:
    db_path = tmp_path / "history.db"
    first = SqliteHistoryBackend(db_path=db_path)
    try:
        await first.upsert_completed_row(
            row=CodeReviewRoundIn(
                flow_run_id="fr-persist-1", repo="org/repo", pr_number=1, verdict="approve"
            )
        )
    finally:
        await first.aclose()

    second = SqliteHistoryBackend(db_path=db_path)
    try:
        latest = await second.select_latest_completed_round(repo="org/repo", pr_number=1)
        assert latest is not None
        assert latest.flow_run_id == "fr-persist-1"
    finally:
        await second.aclose()


async def test_pre_existing_file_without_failure_reason_column_self_heals(
    tmp_path: Path,
) -> None:
    """A local history.db created before this column existed must not need a
    manual migration -- opening it should add the column automatically.

    CREATE TABLE IF NOT EXISTS is a no-op once the table already exists, so
    without the self-healing ALTER TABLE in
    ``SqliteHistoryBackend._migrate_agent_runs_failure_reason``, a
    pre-existing local file would never pick up ``failure_reason`` at all.
    """
    db_path = tmp_path / "legacy.db"
    review_id = "11111111-1111-1111-1111-111111111111"

    # Build the table exactly as it looked before failure_reason existed.
    # code_reviews only needs enough columns to satisfy agent_runs' FK --
    # the real column_reviews shape isn't what this test is about.
    conn = sqlite3.connect(db_path)
    try:
        # executescript() auto-commits its own DDL; the conn.commit() below
        # is for the parameterized INSERT that follows, not for this script.
        conn.executescript("""
            CREATE TABLE code_reviews (
                id TEXT PRIMARY KEY,
                flow_run_id TEXT,
                repo TEXT NOT NULL,
                pr_number INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE agent_runs (
                id TEXT PRIMARY KEY,
                code_review_id TEXT NOT NULL REFERENCES code_reviews (id) ON DELETE CASCADE,
                agent_name TEXT NOT NULL,
                agent_type TEXT NOT NULL,
                model TEXT,
                cost_usd REAL DEFAULT 0,
                duration_seconds REAL DEFAULT 0,
                started_at TEXT,
                finished_at TEXT,
                tool_call_count INTEGER DEFAULT 0,
                tool_names TEXT,
                context7_call_count INTEGER DEFAULT 0,
                files_explored TEXT,
                finding_count INTEGER DEFAULT 0,
                result_text_length INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            );
        """)
        # Parameter binding (not f-string/executescript) establishes the
        # safe pattern even though review_id is a hardcoded literal here.
        conn.execute(
            "INSERT INTO code_reviews (id, repo, pr_number, created_at) VALUES (?, ?, ?, ?)",
            (review_id, "org/repo", 1, "2026-01-01T00:00:00Z"),
        )
        conn.commit()
        columns_before = {row[1] for row in conn.execute("PRAGMA table_info(agent_runs)")}
        assert "failure_reason" not in columns_before
    finally:
        conn.close()

    backend = SqliteHistoryBackend(db_path=db_path)
    try:
        # Would raise sqlite3.OperationalError ("no such column") if the
        # self-healing migration hadn't run.
        await backend.insert_agent_runs(
            code_review_id=review_id,
            runs=[AgentRunIn(agent_name="system:x", agent_type="system", failure_reason="timeout")],
        )
    finally:
        await backend.aclose()

    conn = sqlite3.connect(db_path)
    try:
        columns_after = {row[1] for row in conn.execute("PRAGMA table_info(agent_runs)")}
        assert "failure_reason" in columns_after
        row = conn.execute("SELECT failure_reason FROM agent_runs").fetchone()
        assert row[0] == "timeout"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Concurrent open: two backend instances, same file
# ---------------------------------------------------------------------------


async def test_two_backends_same_file_see_each_others_writes(tmp_path: Path) -> None:
    db_path = tmp_path / "shared.db"
    writer = SqliteHistoryBackend(db_path=db_path)
    reader = SqliteHistoryBackend(db_path=db_path)
    try:
        await writer.upsert_completed_row(
            row=CodeReviewRoundIn(
                flow_run_id="fr-shared-1", repo="org/repo", pr_number=1, verdict="approve"
            )
        )
        seen = await reader.select_latest_completed_round(repo="org/repo", pr_number=1)
        assert seen is not None
        assert seen.flow_run_id == "fr-shared-1"

        await reader.upsert_completed_row(
            row=CodeReviewRoundIn(
                flow_run_id="fr-shared-2", repo="org/repo", pr_number=1, verdict="block"
            )
        )
        seen_by_writer = await writer.select_latest_completed_round(repo="org/repo", pr_number=1)
        assert seen_by_writer is not None
        assert seen_by_writer.flow_run_id == "fr-shared-2"
    finally:
        await writer.aclose()
        await reader.aclose()


async def test_concurrent_writes_from_one_backend_do_not_corrupt_state(
    tmp_path: Path,
) -> None:
    """``asyncio.gather`` over many upserts on one backend instance exercises
    the internal lock: every write must land, none silently dropped."""
    db_path = tmp_path / "concurrent.db"
    backend = SqliteHistoryBackend(db_path=db_path)
    try:
        await asyncio.gather(
            *[
                backend.upsert_completed_row(
                    row=CodeReviewRoundIn(
                        flow_run_id=f"fr-concurrent-{i}",
                        repo="org/repo",
                        pr_number=1,
                        verdict="approve",
                    )
                )
                for i in range(10)
            ]
        )
        recent = await backend.select_recent_rounds(repo="org/repo", pr_number=1, limit=50)
        assert len(recent) == 10
        assert len({r.flow_run_id for r in recent}) == 10
    finally:
        await backend.aclose()


# ---------------------------------------------------------------------------
# In-memory database
# ---------------------------------------------------------------------------


async def test_in_memory_db_path_works_without_touching_filesystem(tmp_path: Path) -> None:
    backend = SqliteHistoryBackend(db_path=":memory:")
    try:
        await backend.upsert_completed_row(
            row=CodeReviewRoundIn(repo="org/repo", pr_number=1, verdict="approve")
        )
        recent = await backend.select_recent_rounds(repo="org/repo", pr_number=1, limit=10)
        assert len(recent) == 1
    finally:
        await backend.aclose()
    # No file materialized anywhere under tmp_path (the default constructor
    # arg wasn't touched, and ":memory:" never resolves to a real path).
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Timestamp / epoch math correctness
# ---------------------------------------------------------------------------


async def test_age_seconds_reflects_real_elapsed_time(tmp_path: Path) -> None:
    backend = SqliteHistoryBackend(db_path=tmp_path / "age.db")
    try:
        await backend.upsert_running_row(
            flow_run_id="fr-age-1", repo="org/repo", pr_number=1, sha=None, base_ref=None
        )
        await asyncio.sleep(0.05)
        status = await backend.select_status_by_flow_run(flow_run_id="fr-age-1")
        assert status is not None
        assert status.age_seconds is not None
        assert status.age_seconds >= 0.05
        # Sanity bound: this test doesn't take minutes to run.
        assert status.age_seconds < 30
    finally:
        await backend.aclose()


async def test_created_at_survives_round_trip_as_aware_utc_datetime(
    tmp_path: Path,
) -> None:
    backend = SqliteHistoryBackend(db_path=tmp_path / "tz.db")
    try:
        before = datetime.now(UTC)
        written = await backend.upsert_completed_row(
            row=CodeReviewRoundIn(repo="org/repo", pr_number=1, verdict="approve")
        )
        after = datetime.now(UTC)

        assert written.created_at.tzinfo is not None
        assert before - timedelta(seconds=1) <= written.created_at <= after + timedelta(seconds=1)
    finally:
        await backend.aclose()


async def test_created_at_and_id_preserved_across_upsert_conflict(
    tmp_path: Path,
) -> None:
    """The DO UPDATE SET list in _UPSERT_COMPLETED_SQL deliberately omits
    ``id``/``created_at`` — a second upsert for the same flow_run_id must not
    mint a new id or move created_at forward."""
    backend = SqliteHistoryBackend(db_path=tmp_path / "stable-id.db")
    try:
        first = await backend.upsert_completed_row(
            row=CodeReviewRoundIn(
                flow_run_id="fr-stable-1", repo="org/repo", pr_number=1, verdict="approve"
            )
        )
        await asyncio.sleep(0.01)
        second = await backend.upsert_completed_row(
            row=CodeReviewRoundIn(
                flow_run_id="fr-stable-1", repo="org/repo", pr_number=1, verdict="block"
            )
        )
        assert second.id == first.id
        assert second.created_at == first.created_at
        assert second.verdict == "block"
    finally:
        await backend.aclose()


# ---------------------------------------------------------------------------
# JSON fidelity for nested payloads
# ---------------------------------------------------------------------------


async def test_deeply_nested_result_json_round_trips_exactly(tmp_path: Path) -> None:
    backend = SqliteHistoryBackend(db_path=tmp_path / "json.db")
    try:
        payload = {
            "findings": [
                {
                    "id": "B1",
                    "severity": "blocking",
                    "tags": ["security", "sql"],
                    "location": {"file": "a.py", "line": 10, "extra": None},
                },
            ],
            "dismissals": [{"finding_id": "B1", "reason": "x", "count": 3, "confident": False}],
            "unicode": "naïve café ☃",
            "empty_list": [],
            "empty_dict": {},
        }
        written = await backend.upsert_completed_row(
            row=CodeReviewRoundIn(
                repo="org/repo", pr_number=1, verdict="block", result_json=payload
            )
        )
        latest = await backend.select_latest_completed_round(repo="org/repo", pr_number=1)
        assert latest is not None
        assert latest.result_json == payload
        assert written.result_json == payload
    finally:
        await backend.aclose()


async def test_null_result_json_stays_none_not_json_null_string(tmp_path: Path) -> None:
    backend = SqliteHistoryBackend(db_path=tmp_path / "null.db")
    try:
        written = await backend.upsert_completed_row(
            row=CodeReviewRoundIn(repo="org/repo", pr_number=1, verdict="approve")
        )
        assert written.result_json is None
        latest = await backend.select_latest_completed_round(repo="org/repo", pr_number=1)
        assert latest is not None
        assert latest.result_json is None
    finally:
        await backend.aclose()


# ---------------------------------------------------------------------------
# UpsertReturnedNoRow forced-failure path
# ---------------------------------------------------------------------------


async def test_upsert_completed_row_raises_upsert_returned_no_row_when_forced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Practically unreachable through normal use (this backend serializes
    every write through its own lock, and the DO UPDATE clause has no WHERE
    narrowing it), but sql.py exposes this exception for the analogous
    Postgres race and callers may branch on it — pin the raise by forcing
    the RETURNING cursor to come back empty."""

    # sqlite3.Connection is a C-implemented, attribute-immutable type — it
    # can't be monkeypatched directly (``setattr`` raises). Subclassing it
    # via the ``factory=`` kwarg to ``sqlite3.connect`` (the documented
    # extension point) and swapping in the subclass through a patched
    # ``sqlite3.connect`` is the supported way to intercept a single
    # statement's result.
    class _NoRowOnReturningConnection(sqlite3.Connection):
        def execute(  # type: ignore[override]
            self, sql: str, parameters: dict[str, Any] | None = None
        ) -> sqlite3.Cursor:
            cursor = (
                super().execute(sql, parameters) if parameters is not None else super().execute(sql)
            )
            if "RETURNING" in sql:
                empty_cursor = self.cursor()
                empty_cursor.execute("SELECT 1 WHERE 0")
                return empty_cursor
            return cursor

    real_connect = sqlite3.connect

    def _fake_connect(database: str | Path, **kwargs: Any) -> sqlite3.Connection:
        kwargs["factory"] = _NoRowOnReturningConnection
        conn = real_connect(database, **kwargs)
        assert isinstance(conn, sqlite3.Connection)
        return conn

    monkeypatch.setattr(sqlite3, "connect", _fake_connect)

    backend = SqliteHistoryBackend(db_path=tmp_path / "norow.db")
    try:
        with pytest.raises(UpsertReturnedNoRow, match="returned no row"):
            await backend.upsert_completed_row(
                row=CodeReviewRoundIn(repo="org/repo", pr_number=1, verdict="approve")
            )
    finally:
        await backend.aclose()


# ---------------------------------------------------------------------------
# ping(): connectivity probe used by validate_history_backend_connectivity
# ---------------------------------------------------------------------------


async def test_ping_creates_file_and_schema(tmp_path: Path) -> None:
    """``ping()`` forces the same connect-and-bootstrap a real write would
    trigger, just callable ahead of time to validate the configured path."""
    db_path = tmp_path / "nested" / "history.db"
    backend = SqliteHistoryBackend(db_path=db_path)

    await backend.ping()

    assert db_path.exists()
    # Verify the bootstrap DDL actually ran, not just that the file exists.
    conn = sqlite3.connect(db_path)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert "code_reviews" in tables

    await backend.aclose()


async def test_ping_is_idempotent(tmp_path: Path) -> None:
    """Calling ``ping()`` twice on the same backend must not re-run the
    bootstrap DDL or open a second connection -- it's the same lazy-connect
    path a real write would take, which is itself idempotent."""
    backend = SqliteHistoryBackend(db_path=tmp_path / "history.db")

    await backend.ping()
    first_conn = backend._conn
    await backend.ping()

    assert backend._conn is first_conn
    await backend.aclose()


async def test_ping_default_path_when_none_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``db_path=None`` (the production default, used when
    ARGUS_HISTORY_DB_PATH is unset) must resolve to DEFAULT_DB_PATH and
    succeed, not raise on a missing path argument. Monkeypatches
    DEFAULT_DB_PATH to a tmp dir rather than touching the real
    ``~/.local/share/argus/history.db`` this test would otherwise create
    as a side effect on the machine running it."""
    import argus.storage.sqlite as sqlite_module

    fake_default = tmp_path / "history.db"
    monkeypatch.setattr(sqlite_module, "DEFAULT_DB_PATH", fake_default)

    backend = SqliteHistoryBackend()
    assert backend.db_path == str(fake_default)

    await backend.ping()

    assert fake_default.exists()
    await backend.aclose()


async def test_ping_raises_on_unwritable_parent(tmp_path: Path) -> None:
    if hasattr(os, "getuid") and os.getuid() == 0:
        pytest.skip("chmod has no effect as root (common in Docker-based CI)")
    unwritable = tmp_path / "readonly"
    unwritable.mkdir(mode=0o500)
    backend = SqliteHistoryBackend(db_path=unwritable / "sub" / "history.db")

    with pytest.raises(OSError):
        await backend.ping()


async def test_db_path_property_matches_constructor_arg(tmp_path: Path) -> None:
    db_path = tmp_path / "history.db"
    backend = SqliteHistoryBackend(db_path=db_path)
    assert backend.db_path == str(db_path)


async def test_db_path_property_preserves_relative_path_string() -> None:
    """``db_path`` is a plain passthrough -- a relative path string is
    stored and returned as-is, not resolved to an absolute path."""
    backend = SqliteHistoryBackend(db_path="relative/history.db")
    assert backend.db_path == "relative/history.db"
