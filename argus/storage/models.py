"""Wire-shape Pydantic models for the Argus storage backend.

Mirrors the columns the Postgres INSERT in ``graph.py:1738+`` writes
today. Source-of-truth lives here; the HTTP storage backend's server
side imports this same shape (or re-declares it identically) so the
Postgres and HTTP backends round-trip bytes cleanly.

Forbid-vs-ignore policy:

- **Write inputs (``CodeReviewRound``)**: ``extra="forbid"`` — a
  caller passing an unknown column is almost certainly a bug; surface
  loudly at validation time.
- **Response models (``CodeReviewRoundRecord``,
  ``ListReviewRoundsResponse``)**: ``extra="ignore"`` — the HTTP
  server side may add a column that an older Argus client doesn't
  know about; ignoring lets the older reader keep working until it's
  upgraded.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CodeReviewRound(BaseModel):
    """One row's worth of ``review_service.code_reviews`` data.

    All optional fields default to ``None`` so partial rounds
    (e.g. an upsert of a "running" row that gets finalized later by
    the same flow) round-trip cleanly through the HTTP path.
    """

    model_config = ConfigDict(extra="forbid")

    flow_run_id: str | None = Field(
        default=None,
        description=(
            "Job/run ID for callers driven by an external orchestrator; "
            "``None`` for in-sandbox Argus invocations where no such run "
            "exists. Used as the upsert conflict key for the former."
        ),
    )
    repo: str = Field(description="GitHub repository in owner/repo format.")
    pr_number: int = Field(
        ge=0,
        description=(
            "GitHub PR number, or 0 for SHA-only reviews (the storage "
            "contract supports both — a SHA-only run produces a row "
            "with pr_number=0)."
        ),
    )
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


class CodeReviewRoundRecord(CodeReviewRound):
    """A persisted round — adds the server-assigned ``id`` + ``created_at``.

    Overrides ``extra`` to ``ignore`` so an additive server-side
    schema change (the HTTP backend starts returning a new column)
    doesn't hard-fail older Argus clients reading the response.
    """

    model_config = ConfigDict(extra="ignore")

    id: UUID
    created_at: datetime

    @field_validator("result_json", mode="before")
    @classmethod
    def _decode_jsonb_string(cls, v: Any) -> Any:
        """Handle asyncpg returning JSONB as a raw JSON string.

        Read-path only: some driver configurations return JSONB
        columns as ``str`` rather than a parsed ``dict``. Coerce here
        so any caller — whether reading from Postgres directly or
        from the HTTP storage backend's response — gets the same
        shape. Deliberately NOT placed on
        :class:`CodeReviewRound` so a write-side caller that passes a
        JSON string for ``result_json`` (annotated
        ``dict[str, Any] | None``) trips Pydantic's type validation
        instead of being silently coerced into a dict.
        """
        if isinstance(v, str):
            return json.loads(v)
        return v


class ListReviewRoundsResponse(BaseModel):
    """Response shape returned by the GET endpoint on the HTTP storage backend.

    Mirrors the server-side response model of whatever HTTP service
    implements the storage contract described in ``docs/STORAGE.md``.
    """

    model_config = ConfigDict(extra="ignore")

    rounds: list[CodeReviewRoundRecord] = Field(default_factory=list)
