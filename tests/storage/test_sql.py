"""Parity tests for ``argus.storage.sql``.

These tests use a fake ``AsyncSession`` that records the executed SQL
text + bind parameters. Goal: pin the SQL byte-for-byte so future
edits to the canonical writer module surface as test diffs rather
than silent behavior changes for the 7 (now 8) call sites that
share this code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest

from argus.storage.sql import (
    AgentRunIn,
    CodeReviewRoundIn,
    CodeReviewRoundRow,
    CodeReviewStatusRow,
    insert_agent_runs,
    select_latest_completed_round,
    select_recent_rounds,
    select_status_by_flow_run,
    upsert_completed_row,
    upsert_running_row,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _Call:
    sql: str
    params: dict[str, Any]


class _FakeMappingResult:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def first(self) -> dict[str, Any] | None:
        return self._rows[0] if self._rows else None

    def all(self) -> list[dict[str, Any]]:
        return list(self._rows)


class _FakeResult:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def mappings(self) -> _FakeMappingResult:
        return _FakeMappingResult(self._rows)


@dataclass
class FakeSession:
    """Records every execute call; returns programmed rows in order."""

    rows_per_call: list[list[dict[str, Any]]] = field(default_factory=list)
    calls: list[_Call] = field(default_factory=list)

    async def execute(
        self,
        statement: Any,
        params: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> _FakeResult:
        # Capture compiled SQL text + bind params for byte-equivalence assertions
        try:
            sql_text = str(statement.compile(compile_kwargs={"literal_binds": False}))
        except Exception:
            sql_text = str(statement)
        if isinstance(params, list):
            # executemany-style batch — record each payload as its own
            # call so call-by-call assertions stay ergonomic.
            for payload in params:
                self.calls.append(_Call(sql=sql_text, params=dict(payload)))
        else:
            self.calls.append(_Call(sql=sql_text, params=dict(params or {})))
        rows = self.rows_per_call.pop(0) if self.rows_per_call else []
        return _FakeResult(rows)


# ---------------------------------------------------------------------------
# Read tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_select_latest_completed_round_emits_expected_sql_and_params() -> None:
    sample_id = uuid4()
    session = FakeSession(
        rows_per_call=[
            [
                {
                    "id": sample_id,
                    "flow_run_id": "fr-1",
                    "repo": "org/repo",
                    "pr_number": 42,
                    "verdict": "approve",
                    "risk_level": "low",
                    "blocking_count": 0,
                    "suggestion_count": 1,
                    "review_comment": "lgtm",
                    "result_json": {"findings": []},
                    "cost_usd": 0.12,
                    "duration_seconds": 3.4,
                    "reviewer_version": "v3",
                    "orchestrator_model": "claude-opus",
                    "subagent_model": "claude-sonnet",
                    "sha": "abc1234",
                    "base_ref": "main",
                    "current_stage": "completed",
                    "created_at": datetime(2026, 5, 19, 12, 0, tzinfo=UTC),
                    "prior_count": 2,
                }
            ]
        ]
    )

    row = await select_latest_completed_round(session, repo="org/repo", pr_number=42)

    assert row is not None
    assert row.id == sample_id
    assert row.prior_count == 2
    assert len(session.calls) == 1
    call = session.calls[0]
    assert "FROM review_service.code_reviews" in call.sql
    assert "verdict IS NOT NULL" in call.sql
    # ``prior_count`` subquery uses the same ``verdict IS NOT NULL``
    # filter as the outer SELECT — sha is preserved by the COALESCE
    # write path and excluding sha=NULL rows from the count would
    # undercount for legacy rows.
    assert "sha IS NOT NULL" not in call.sql
    assert "ORDER BY created_at DESC" in call.sql
    assert "LIMIT 1" in call.sql
    assert call.params == {"repo": "org/repo", "pr_number": 42}


@pytest.mark.asyncio
async def test_select_latest_completed_round_returns_none_when_no_row() -> None:
    session = FakeSession(rows_per_call=[[]])
    row = await select_latest_completed_round(session, repo="org/repo", pr_number=1)
    assert row is None


@pytest.mark.asyncio
async def test_select_recent_rounds_uses_limit_param() -> None:
    session = FakeSession(rows_per_call=[[]])
    rows = await select_recent_rounds(session, repo="org/repo", pr_number=7, limit=200)
    assert rows == []
    call = session.calls[0]
    assert call.params == {"repo": "org/repo", "pr_number": 7, "limit": 200}
    assert "ORDER BY created_at DESC" in call.sql
    assert "LIMIT :limit" in call.sql


@pytest.mark.asyncio
async def test_select_status_by_flow_run_emits_status_columns() -> None:
    session = FakeSession(
        rows_per_call=[
            [
                {
                    "result_json": '{"a": 1}',
                    "current_stage": "running",
                    "blocking_count": 0,
                    "age_seconds": 12.0,
                }
            ]
        ]
    )
    row = await select_status_by_flow_run(session, flow_run_id="fr-xyz")
    assert isinstance(row, CodeReviewStatusRow)
    # JSONB-as-str coercion lives on the model
    assert row.result_json == {"a": 1}
    call = session.calls[0]
    assert "result_json" in call.sql and "current_stage" in call.sql
    assert "EXTRACT(EPOCH FROM (now() - created_at))" in call.sql
    assert call.params == {"fid": "fr-xyz"}


# ---------------------------------------------------------------------------
# Write tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_running_row_params_match_legacy() -> None:
    session = FakeSession(rows_per_call=[[]])
    await upsert_running_row(
        session,
        flow_run_id="fr-1",
        repo="org/repo",
        pr_number=10,
        sha="deadbeef",
        base_ref="main",
    )
    call = session.calls[0]
    assert "INSERT INTO review_service.code_reviews" in call.sql
    assert "'running'" in call.sql and "'v3'" in call.sql
    assert "ON CONFLICT (flow_run_id) WHERE flow_run_id IS NOT NULL" in call.sql
    assert call.params == {
        "flow_run_id": "fr-1",
        "repo": "org/repo",
        "pr_number": 10,
        "sha": "deadbeef",
        "base_ref": "main",
    }


@pytest.mark.asyncio
async def test_upsert_completed_row_returns_row_and_jsonifies_result_json() -> None:
    persisted_id = uuid4()
    persisted_row = {
        "id": persisted_id,
        "flow_run_id": "fr-1",
        "repo": "org/repo",
        "pr_number": 11,
        "verdict": "block",
        "risk_level": "high",
        "blocking_count": 2,
        "suggestion_count": 3,
        "review_comment": "see findings",
        "result_json": {"findings": [{"severity": "BLOCKING"}]},
        "cost_usd": 1.5,
        "duration_seconds": 30.0,
        "reviewer_version": "v3",
        "orchestrator_model": "claude-opus",
        "subagent_model": "claude-sonnet",
        "sha": "cafebabe",
        "base_ref": "main",
        "current_stage": "completed",
        "created_at": datetime(2026, 5, 19, 13, 0, tzinfo=UTC),
    }
    session = FakeSession(rows_per_call=[[persisted_row]])
    payload = CodeReviewRoundIn(
        flow_run_id="fr-1",
        repo="org/repo",
        pr_number=11,
        verdict="block",
        risk_level="high",
        blocking_count=2,
        suggestion_count=3,
        review_comment="see findings",
        result_json={"findings": [{"severity": "BLOCKING"}]},
        cost_usd=1.5,
        duration_seconds=30.0,
        orchestrator_model="claude-opus",
        subagent_model="claude-sonnet",
        sha="cafebabe",
        base_ref="main",
    )
    out = await upsert_completed_row(session, row=payload)
    assert isinstance(out, CodeReviewRoundRow)
    assert out.id == persisted_id

    call = session.calls[0]
    assert "INSERT INTO review_service.code_reviews" in call.sql
    assert "ON CONFLICT (flow_run_id) WHERE flow_run_id IS NOT NULL" in call.sql
    assert "RETURNING id" in call.sql
    # result_json must be json-serialized on the wire (CAST :result_json AS JSONB)
    assert "CAST(:result_json AS JSONB)" in call.sql
    assert isinstance(call.params["result_json"], str)
    assert json.loads(call.params["result_json"]) == {"findings": [{"severity": "BLOCKING"}]}
    # All 17 columns present in bind params
    assert set(call.params.keys()) == {
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
        "reviewer_version",
        "orchestrator_model",
        "subagent_model",
        "sha",
        "base_ref",
        "current_stage",
    }


@pytest.mark.asyncio
async def test_upsert_completed_row_preserves_sha_via_coalesce() -> None:
    """Argus round 1, finding B1: the finalize ON CONFLICT must not
    overwrite a non-null persisted ``sha`` with NULL when the caller
    passes ``sha=None``. The fix is ``sha = COALESCE(EXCLUDED.sha,
    review_service.code_reviews.sha)`` — pin it in SQL text so a
    regression surfaces as a test diff.
    """
    persisted_id = uuid4()
    session = FakeSession(
        rows_per_call=[
            [
                {
                    "id": persisted_id,
                    "flow_run_id": "fr-1",
                    "repo": "org/repo",
                    "pr_number": 11,
                    "verdict": "approve",
                    "risk_level": "low",
                    "blocking_count": 0,
                    "suggestion_count": 0,
                    "review_comment": "lgtm",
                    "result_json": {"findings": []},
                    "cost_usd": 0.1,
                    "duration_seconds": 1.0,
                    "reviewer_version": "v3",
                    "orchestrator_model": "claude-opus",
                    "subagent_model": "claude-sonnet",
                    "sha": None,
                    "base_ref": None,
                    "current_stage": "completed",
                    "created_at": datetime(2026, 5, 19, tzinfo=UTC),
                }
            ]
        ]
    )
    await upsert_completed_row(
        session, row=CodeReviewRoundIn(repo="org/repo", pr_number=11, sha=None)
    )
    call = session.calls[0]
    assert "COALESCE(EXCLUDED.sha, review_service.code_reviews.sha)" in call.sql
    assert "COALESCE(EXCLUDED.base_ref, review_service.code_reviews.base_ref)" in call.sql


@pytest.mark.asyncio
async def test_upsert_completed_row_raises_when_no_row_returned() -> None:
    session = FakeSession(rows_per_call=[[]])
    payload = CodeReviewRoundIn(repo="org/repo", pr_number=1)
    with pytest.raises(RuntimeError, match="returned no row"):
        await upsert_completed_row(session, row=payload)


@pytest.mark.asyncio
async def test_insert_agent_runs_batches_with_array_binds() -> None:
    session = FakeSession(rows_per_call=[[], []])
    code_review_id = uuid4()
    runs = [
        AgentRunIn(
            agent_name="system:Foo",
            agent_type="system",
            model="claude-sonnet",
            cost_usd=0.01,
            duration_seconds=1.0,
            tool_call_count=2,
            tool_names=["read", "grep"],
            files_explored=["a.py", "b.py"],
            finding_count=1,
            result_text_length=200,
            failure_reason="timeout",
        ),
        AgentRunIn(
            agent_name="cross_cutting:Bar",
            agent_type="cross_cutting",
        ),
    ]
    await insert_agent_runs(session, code_review_id=code_review_id, runs=runs)
    assert len(session.calls) == 2
    for call in session.calls:
        assert "INSERT INTO review_service.agent_runs" in call.sql
        assert call.params["code_review_id"] == str(code_review_id)
    # First row's array binds round-trip
    assert session.calls[0].params["tool_names"] == ["read", "grep"]
    assert session.calls[0].params["files_explored"] == ["a.py", "b.py"]
    # failure_reason must reach the persisted row -- an unpersisted field is
    # unqueryable, defeating the whole point of adding it.
    assert session.calls[0].params["failure_reason"] == "timeout"
    assert session.calls[1].params["failure_reason"] is None


@pytest.mark.asyncio
async def test_insert_agent_runs_no_op_on_empty_list() -> None:
    session = FakeSession()
    await insert_agent_runs(session, code_review_id=uuid4(), runs=[])
    assert session.calls == []


def test_agent_run_in_rejects_invalid_failure_reason() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AgentRunIn(
            agent_name="system:Foo",
            agent_type="system",
            failure_reason="bad_value",
        )


def test_agent_run_in_accepts_worker_crashed_failure_reason() -> None:
    run = AgentRunIn(
        agent_name="system:Foo",
        agent_type="system",
        failure_reason="worker_crashed",
    )
    assert run.failure_reason == "worker_crashed"


# ---------------------------------------------------------------------------
# Model round-trip
# ---------------------------------------------------------------------------


def test_code_review_round_row_decodes_jsonb_string() -> None:
    row = CodeReviewRoundRow(
        id=UUID("00000000-0000-0000-0000-000000000001"),
        repo="org/repo",
        pr_number=1,
        result_json='{"a": 1}',
        created_at=datetime(2026, 5, 19, tzinfo=UTC),
    )
    assert row.result_json == {"a": 1}


def test_code_review_status_row_decodes_jsonb_string() -> None:
    row = CodeReviewStatusRow(result_json='{"k": "v"}')
    assert row.result_json == {"k": "v"}


def test_code_review_round_in_forbids_extra_fields() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CodeReviewRoundIn(repo="org/repo", pr_number=1, bogus="x")  # type: ignore[call-arg]
