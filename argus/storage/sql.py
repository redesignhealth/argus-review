"""Canonical raw-SQL surface over ``review_service.code_reviews`` /
``review_service.agent_runs``.

Functions take an ``AsyncSession`` so the caller owns the transaction
(commit/rollback). Two **intentional, documented** deviations from a
naive read/write of these tables:

1. The finalize ``ON CONFLICT`` clause uses
   ``sha = COALESCE(EXCLUDED.sha, review_service.code_reviews.sha)``
   (and the same for ``base_ref``) instead of an unconditional
   overwrite. An unconditional ``EXCLUDED.sha`` would clobber a
   non-null persisted ``sha`` with ``NULL`` whenever a caller passed
   ``sha=None`` (the case with no upstream run tracking a commit).
   ``COALESCE`` preserves the invariant "sha is never cleared once
   bound" inside this module's write path. The ``prior_count``
   subquery uses the same ``verdict IS NOT NULL`` filter as the outer
   SELECT so the count is consistent with the row set the caller sees.
2. The finalize INSERT binds ``current_stage`` from the input row
   instead of hardcoding ``'completed'``. ``CodeReviewRoundIn``
   defaults the field to ``'completed'`` so most callers are
   unaffected, while callers that report progress incrementally can
   still set it explicitly.

Notes on filter reconciliation:

- ``select_latest_completed_round`` filters on ``verdict IS NOT NULL``
  only. ``graph._fetch_prior_review`` doesn't separately require
  ``sha IS NOT NULL`` — that would be redundant **given the COALESCE
  invariant above** (the running-state INSERT binds ``sha`` and the
  finalize ``ON CONFLICT`` no longer overwrites it).
- ``graph._fetch_prior_review`` also wants a correlated
  ``prior_count`` subquery alongside the row; we preserve that field
  on ``CodeReviewRoundRow`` so it doesn't need a second round-trip.
"""

from __future__ import annotations

import json
from datetime import datetime  # noqa: TC003 — used at runtime by Pydantic
from typing import Any
from uuid import UUID  # noqa: TC003 — used at runtime by Pydantic

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import ARRAY, TEXT, bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: TC002 — async function param


class UpsertReturnedNoRow(RuntimeError):
    """Raised when an ``INSERT ... ON CONFLICT ... RETURNING`` write
    yields zero rows (partial-unique-index race or similar). Callers
    typically map this to an internal-error response; it is a narrower
    type than the SQLAlchemy errors that bubble up from the driver
    itself so call sites can distinguish "the DB rejected my write"
    from "the DB call never completed".
    """


# =============================================================================
# Pydantic shapes
# =============================================================================


class CodeReviewRoundIn(BaseModel):
    """Write-side input for :func:`upsert_completed_row`.

    Mirrors the 17 columns the legacy ``graph.py`` finalize INSERT
    writes. ``extra="forbid"`` so unknown columns surface as
    validation errors rather than being silently dropped.
    """

    model_config = ConfigDict(extra="forbid")

    flow_run_id: str | None = Field(default=None)
    repo: str
    pr_number: int = Field(ge=0)
    verdict: str | None = Field(default=None)
    risk_level: str | None = Field(default=None)
    blocking_count: int | None = Field(default=None, ge=0)
    suggestion_count: int | None = Field(default=None, ge=0)
    review_comment: str | None = Field(default=None)
    result_json: dict[str, Any] | None = Field(default=None)
    cost_usd: float | None = Field(default=None, ge=0)
    duration_seconds: float | None = Field(default=None, ge=0)
    reviewer_version: str = Field(default="v3")
    orchestrator_model: str | None = Field(default=None)
    subagent_model: str | None = Field(default=None)
    sha: str | None = Field(default=None)
    base_ref: str | None = Field(default=None)
    current_stage: str = Field(default="completed")


class CodeReviewRoundRow(BaseModel):
    """Read-side row shape returned by the latest / list / upsert SELECTs.

    Adds the server-assigned ``id`` + ``created_at`` to the input
    columns, plus the optional ``prior_count`` correlated count used
    by ``graph._fetch_prior_review``. ``extra="ignore"`` so an
    additive server-side schema change (a new column) doesn't break
    older readers.
    """

    model_config = ConfigDict(extra="ignore")

    id: UUID
    flow_run_id: str | None = None
    repo: str
    pr_number: int = Field(ge=0)
    verdict: str | None = None
    risk_level: str | None = None
    blocking_count: int | None = None
    suggestion_count: int | None = None
    review_comment: str | None = None
    result_json: dict[str, Any] | None = None
    cost_usd: float | None = None
    duration_seconds: float | None = None
    reviewer_version: str = "v3"
    orchestrator_model: str | None = None
    subagent_model: str | None = None
    sha: str | None = None
    base_ref: str | None = None
    current_stage: str = "completed"
    created_at: datetime
    # Populated only by ``select_latest_completed_round``; ``None`` from
    # the list / upsert RETURNING paths.
    prior_count: int | None = None

    @field_validator("result_json", mode="before")
    @classmethod
    def _decode_jsonb_string(cls, v: Any) -> Any:
        """Some driver configs return JSONB columns as a raw JSON string."""
        if isinstance(v, str):
            return json.loads(v)
        return v


class CodeReviewStatusRow(BaseModel):
    """Lightweight read shape for ``get_code_review_status``."""

    model_config = ConfigDict(extra="ignore")

    result_json: dict[str, Any] | None = None
    current_stage: str | None = None
    blocking_count: int | None = None
    age_seconds: float | None = None

    @field_validator("result_json", mode="before")
    @classmethod
    def _decode_jsonb_string(cls, v: Any) -> Any:
        if isinstance(v, str):
            return json.loads(v)
        return v


class AgentRunIn(BaseModel):
    """One row's worth of ``review_service.agent_runs`` analytics."""

    model_config = ConfigDict(extra="forbid")

    agent_name: str
    agent_type: str
    model: str | None = None
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    tool_call_count: int = 0
    tool_names: list[str] = Field(default_factory=list)
    context7_call_count: int = 0
    files_explored: list[str] = Field(default_factory=list)
    finding_count: int = 0
    result_text_length: int = 0


# =============================================================================
# SQL constants (verbatim parity with the legacy inline strings)
# =============================================================================

_SELECT_LATEST_COMPLETED_ROUND_SQL = """
    SELECT id, flow_run_id, repo, pr_number, verdict, risk_level,
           blocking_count, suggestion_count, review_comment,
           result_json, cost_usd, duration_seconds, reviewer_version,
           orchestrator_model, subagent_model, sha, base_ref,
           current_stage, created_at,
           (SELECT COUNT(*) FROM review_service.code_reviews
            WHERE repo = :repo AND pr_number = :pr_number
              AND verdict IS NOT NULL) AS prior_count
    FROM review_service.code_reviews
    WHERE repo = :repo
      AND pr_number = :pr_number
      AND verdict IS NOT NULL
    ORDER BY created_at DESC
    LIMIT 1
"""

_SELECT_RECENT_ROUNDS_SQL = """
    SELECT id, flow_run_id, repo, pr_number, verdict, risk_level,
           blocking_count, suggestion_count, review_comment,
           result_json, cost_usd, duration_seconds, reviewer_version,
           orchestrator_model, subagent_model, sha, base_ref,
           current_stage, created_at
    FROM review_service.code_reviews
    WHERE repo = :repo AND pr_number = :pr_number
      AND verdict IS NOT NULL
    ORDER BY created_at DESC
    LIMIT :limit
"""

_SELECT_STATUS_BY_FLOW_RUN_SQL = """
    SELECT result_json, current_stage, blocking_count,
           EXTRACT(EPOCH FROM (now() - created_at)) AS age_seconds
    FROM review_service.code_reviews
    WHERE flow_run_id = :fid
    ORDER BY created_at DESC
    LIMIT 1
"""

_UPSERT_RUNNING_SQL = """
    INSERT INTO review_service.code_reviews (
        flow_run_id, repo, pr_number,
        current_stage, reviewer_version,
        sha, base_ref
    ) VALUES (
        :flow_run_id, :repo, :pr_number,
        'running', 'v3',
        :sha, :base_ref
    )
    ON CONFLICT (flow_run_id) WHERE flow_run_id IS NOT NULL
    DO UPDATE SET
        current_stage = 'running',
        verdict = NULL,
        result_json = NULL
"""

_UPSERT_COMPLETED_SQL = """
    INSERT INTO review_service.code_reviews (
        flow_run_id, repo, pr_number, verdict, risk_level,
        blocking_count, suggestion_count,
        review_comment, result_json, cost_usd,
        duration_seconds, reviewer_version,
        orchestrator_model, subagent_model,
        sha, base_ref, current_stage
    ) VALUES (
        :flow_run_id, :repo, :pr_number, :verdict, :risk_level,
        :blocking_count, :suggestion_count,
        :review_comment, CAST(:result_json AS JSONB), :cost_usd,
        :duration_seconds, :reviewer_version,
        :orchestrator_model, :subagent_model,
        :sha, :base_ref, :current_stage
    )
    ON CONFLICT (flow_run_id) WHERE flow_run_id IS NOT NULL
    DO UPDATE SET
        verdict = EXCLUDED.verdict,
        risk_level = EXCLUDED.risk_level,
        blocking_count = EXCLUDED.blocking_count,
        suggestion_count = EXCLUDED.suggestion_count,
        review_comment = EXCLUDED.review_comment,
        result_json = EXCLUDED.result_json,
        cost_usd = EXCLUDED.cost_usd,
        duration_seconds = EXCLUDED.duration_seconds,
        orchestrator_model = EXCLUDED.orchestrator_model,
        subagent_model = EXCLUDED.subagent_model,
        -- Preserve sha / base_ref bound by the running-state upsert
        -- when the finalize call passes ``sha=None`` / ``base_ref=None``
        -- (e.g. the agent-storage path that has no orchestrator run). This
        -- invariant — "sha is never cleared once bound" — justifies
        -- dropping ``AND sha IS NOT NULL`` from
        -- ``select_latest_completed_round``'s prior_count subquery.
        sha = COALESCE(EXCLUDED.sha, review_service.code_reviews.sha),
        base_ref = COALESCE(EXCLUDED.base_ref, review_service.code_reviews.base_ref),
        current_stage = EXCLUDED.current_stage,
        reviewer_version = EXCLUDED.reviewer_version
    RETURNING id, flow_run_id, repo, pr_number, verdict, risk_level,
              blocking_count, suggestion_count, review_comment,
              result_json, cost_usd, duration_seconds, reviewer_version,
              orchestrator_model, subagent_model, sha, base_ref,
              current_stage, created_at
"""

_INSERT_AGENT_RUN_SQL = """
    INSERT INTO review_service.agent_runs (
        code_review_id, agent_name, agent_type, model,
        cost_usd, duration_seconds, started_at, finished_at,
        tool_call_count, tool_names, context7_call_count,
        files_explored, finding_count, result_text_length
    ) VALUES (
        :code_review_id, :agent_name, :agent_type, :model,
        :cost_usd, :duration_seconds, :started_at, :finished_at,
        :tool_call_count, :tool_names, :context7_call_count,
        :files_explored, :finding_count, :result_text_length
    )
"""


# =============================================================================
# Public async functions
# =============================================================================


async def select_latest_completed_round(
    session: AsyncSession, *, repo: str, pr_number: int
) -> CodeReviewRoundRow | None:
    """Return the most recent completed round for ``(repo, pr_number)``.

    A "completed" round is one with ``verdict IS NOT NULL``. The
    returned row carries a populated ``prior_count`` (the total
    completed-round count for this PR) so callers don't need a second
    round-trip.
    """
    result = await session.execute(
        text(_SELECT_LATEST_COMPLETED_ROUND_SQL),
        {"repo": repo, "pr_number": pr_number},
    )
    row = result.mappings().first()
    if row is None:
        return None
    return CodeReviewRoundRow(**dict(row))


async def select_recent_rounds(
    session: AsyncSession, *, repo: str, pr_number: int, limit: int
) -> list[CodeReviewRoundRow]:
    """Return up to ``limit`` recent completed rounds for ``(repo, pr_number)``.

    Ordered ``created_at DESC`` so the cap drops oldest rounds, not
    newest (the newest is exactly the one Argus's Prior Feedback
    Verification table needs).
    """
    result = await session.execute(
        text(_SELECT_RECENT_ROUNDS_SQL),
        {"repo": repo, "pr_number": pr_number, "limit": limit},
    )
    rows = result.mappings().all()
    return [CodeReviewRoundRow(**dict(r)) for r in rows]


_SELECT_RECENT_LITE_ROUNDS_SQL = """
    SELECT id, flow_run_id, repo, pr_number, verdict, risk_level,
           blocking_count, suggestion_count, review_comment,
           result_json, cost_usd, duration_seconds, reviewer_version,
           orchestrator_model, subagent_model, sha, base_ref,
           current_stage, created_at
    FROM review_service.code_reviews
    WHERE repo = :repo AND pr_number = :pr_number
      AND verdict IS NOT NULL
      AND reviewer_version = 'v3-lite'
    ORDER BY created_at DESC
    LIMIT :limit
"""


async def select_recent_lite_rounds(
    session: AsyncSession, *, repo: str, pr_number: int, limit: int = 200
) -> list[CodeReviewRoundRow]:
    """Return up to ``limit`` most-recent lite-mode rounds for ``(repo, pr_number)``.

    Filters on ``reviewer_version = 'v3-lite'`` in SQL so the LIMIT
    applies to the correct population (not all reviewer versions). Ordered
    ``created_at DESC`` so the cap drops the oldest rounds, not the newest —
    matching ``select_recent_rounds`` semantics. Reverse the result for
    chronological (oldest-first) display.
    """
    result = await session.execute(
        text(_SELECT_RECENT_LITE_ROUNDS_SQL),
        {"repo": repo, "pr_number": pr_number, "limit": limit},
    )
    rows = result.mappings().all()
    return [CodeReviewRoundRow(**dict(r)) for r in rows]


async def select_status_by_flow_run(
    session: AsyncSession, *, flow_run_id: str
) -> CodeReviewStatusRow | None:
    """Lightweight status read by ``flow_run_id`` for the status endpoint."""
    result = await session.execute(
        text(_SELECT_STATUS_BY_FLOW_RUN_SQL),
        {"fid": flow_run_id},
    )
    row = result.mappings().first()
    if row is None:
        return None
    return CodeReviewStatusRow(**dict(row))


async def upsert_running_row(
    session: AsyncSession,
    *,
    flow_run_id: str,
    repo: str,
    pr_number: int,
    sha: str | None,
    base_ref: str | None,
) -> None:
    """Pre-flight upsert: mark a flow as ``current_stage='running'``.

    Idempotent on ``flow_run_id`` via partial unique index. Caller
    owns commit/rollback.
    """
    await session.execute(
        text(_UPSERT_RUNNING_SQL),
        {
            "flow_run_id": flow_run_id,
            "repo": repo,
            "pr_number": pr_number,
            "sha": sha,
            "base_ref": base_ref,
        },
    )


async def upsert_completed_row(
    session: AsyncSession, *, row: CodeReviewRoundIn
) -> CodeReviewRoundRow:
    """Finalize-upsert a review round and return the persisted row.

    Idempotent on ``flow_run_id`` via partial unique index. Caller
    owns commit/rollback. Returns the full row (including the
    server-assigned ``id`` and ``created_at``) — callers that only
    need the id can do ``.id`` on the result.
    """
    params = {
        "flow_run_id": row.flow_run_id,
        "repo": row.repo,
        "pr_number": row.pr_number,
        "verdict": row.verdict,
        "risk_level": row.risk_level,
        "blocking_count": row.blocking_count,
        "suggestion_count": row.suggestion_count,
        "review_comment": row.review_comment,
        "result_json": (json.dumps(row.result_json) if row.result_json is not None else None),
        "cost_usd": row.cost_usd,
        "duration_seconds": row.duration_seconds,
        "reviewer_version": row.reviewer_version,
        "orchestrator_model": row.orchestrator_model,
        "subagent_model": row.subagent_model,
        "sha": row.sha,
        "base_ref": row.base_ref,
        "current_stage": row.current_stage,
    }
    result = await session.execute(text(_UPSERT_COMPLETED_SQL), params)
    returned = result.mappings().first()
    if returned is None:
        raise UpsertReturnedNoRow("upsert_completed_row: INSERT returned no row")
    return CodeReviewRoundRow(**dict(returned))


async def insert_agent_runs(
    session: AsyncSession,
    *,
    code_review_id: UUID | str,
    runs: list[AgentRunIn],
) -> None:
    """Insert per-agent execution analytics for a finalized review.

    Best-effort analytics — callers typically run this in a separate
    transaction from the main review row so a failure here doesn't
    roll back the result write.
    """
    if not runs:
        return
    stmt = text(_INSERT_AGENT_RUN_SQL).bindparams(
        bindparam("tool_names", type_=ARRAY(TEXT)),
        bindparam("files_explored", type_=ARRAY(TEXT)),
    )
    # SQLAlchemy 2.0 ``AsyncSession.execute`` dispatches to
    # ``cursor.executemany`` when given a list of bind dicts — one
    # async round-trip instead of N, and the whole batch commits or
    # rolls back atomically (avoiding partial-state on mid-loop
    # failure).
    payloads = [
        {
            "code_review_id": str(code_review_id),
            "agent_name": run.agent_name,
            "agent_type": run.agent_type,
            "model": run.model,
            "cost_usd": run.cost_usd,
            "duration_seconds": run.duration_seconds,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "tool_call_count": run.tool_call_count,
            "tool_names": run.tool_names,
            "context7_call_count": run.context7_call_count,
            "files_explored": run.files_explored,
            "finding_count": run.finding_count,
            "result_text_length": run.result_text_length,
        }
        for run in runs
    ]
    await session.execute(stmt, payloads)
