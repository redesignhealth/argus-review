"""Data contracts for the v3 review pipeline.

These models define the intermediate data structures passed between pipeline
steps (planner -> reviewers -> coverage check -> writer). They are internal
to the pipeline and not exposed in the public API.

The public API contracts live in ``models.py``.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from datetime import datetime, timezone
from typing import Any, Literal, get_args

from pydantic import AwareDatetime, BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Step 1: Planner output
# ---------------------------------------------------------------------------


# Single source of truth for the v3 reviewer's specialist registry.
# Matches the keys of ``runners._SPECIALIST_PROMPT_MAP`` and the names listed
# in the planner prompt at ``manage_prompts.py``.
#
# Constraining ``specialists_needed`` to this Literal makes the planner LLM's
# structured output mode reject hallucinated specialist names (e.g.
# ``test-coverage``, ``test-adequacy``) at parse time, instead of crashing
# downstream in ``runners.run_specialist_reviewer`` with a ValueError.
SpecialistName = Literal[
    "security",
    "sql",
    "infra",
    "orchestration",
    "frontend",
    "slackbot",
    "deployment",
    "llm-patterns",
    "observability",
]


class FileEntry(BaseModel):
    """A single file changed in the PR."""

    path: str = Field(description="File path relative to repo root")
    change_type: str = Field(description="One of: added, modified, deleted")


class SystemGroup(BaseModel):
    """A logical group of files that should be reviewed together."""

    name: str = Field(description="Human-readable group name, e.g. 'backend API endpoints'")
    files: list[str] = Field(description="File paths assigned to this group")
    conventions: str = Field(description="Relevant .cursorrules excerpt or conventions")
    review_focus: str = Field(description="Specific things to check for this group")
    specialists_needed: list[SpecialistName] = Field(
        default_factory=list,
        description=(
            "Specialist co-reviewers to dispatch. Must be one of: "
            + ", ".join(sorted(get_args(SpecialistName)))
            + "."
        ),
    )


class ReviewPlan(BaseModel):
    """Output of the planner step — drives the rest of the pipeline."""

    system_groups: list[SystemGroup] = Field(description="Groups of files to review together")
    cross_cutting_concerns: list[str] = Field(
        description="Issues that span multiple groups (e.g. 'migration ordering')"
    )
    file_manifest: list[FileEntry] = Field(description="All changed files in the PR")

    @field_validator("system_groups", "cross_cutting_concerns", "file_manifest", mode="before")
    @classmethod
    def _parse_stringified_json(cls, v: Any) -> Any:
        """LLMs sometimes return list fields as JSON strings instead of arrays.

        Also fixes invalid escape sequences (e.g. \\n in file paths) that
        cause json.loads to fail.
        """
        if isinstance(v, str):
            # Fix invalid JSON escape sequences from LLM output.
            # Allow \uXXXX (4 hex digits) but escape \u<non-hex> (e.g. C:\utils)
            v = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r"\\\\", v)
            return json.loads(v)
        return v


# ---------------------------------------------------------------------------
# Steps 2+3: Reviewer output
# ---------------------------------------------------------------------------


class RawFinding(BaseModel):
    """A single finding from a system or cross-cutting reviewer.

    These are raw — no severity assignment yet. The writer scores them.
    """

    file: str | None = Field(None, description="File path, if applicable")
    line: int | str | None = Field(None, description="Line number or range, if applicable")
    description: str = Field(description="What the reviewer found")
    context: str | None = Field(None, description="Surrounding code or explanation")


class SystemReviewResult(BaseModel):
    """Output of a single system or cross-cutting reviewer."""

    system_group: str = Field(description="Name of the system group that was reviewed")
    findings: list[RawFinding] = Field(default_factory=list)
    files_explored: list[str] = Field(
        default_factory=list, description="Files the reviewer actually read"
    )
    cost_usd: float = Field(default=0.0, description="Agent session cost in USD")
    timed_out: bool = Field(
        default=False,
        description=(
            "True when the reviewer's subprocess hit the wall-clock timeout and was "
            "killed. Additive field: absent/False for all pre-existing rows means "
            "'not known to have timed out', not 'confirmed completed'."
        ),
    )


# ---------------------------------------------------------------------------
# Step 4a: Coverage check output
# ---------------------------------------------------------------------------


class CoverageGap(BaseModel):
    """A set of files that were not adequately covered by reviewers."""

    files: list[str] = Field(description="File paths with insufficient coverage")
    reason: str = Field(description="Why these files were not covered")


class CoverageResult(BaseModel):
    """Output of the coverage check step."""

    is_covered: bool = Field(description="True if all changed files were reviewed")
    gaps: list[CoverageGap] = Field(default_factory=list, description="Empty if is_covered is True")


# ---------------------------------------------------------------------------
# Multi-round: Prior review context and feedback verification
# ---------------------------------------------------------------------------


class VerificationStatus(str, Enum):
    """Status of a prior finding after verification against the new diff."""

    RESOLVED = "RESOLVED"
    UNRESOLVED = "UNRESOLVED"
    REGRESSED = "REGRESSED"


class PriorFinding(BaseModel):
    """A finding from a prior review round, with its assigned severity."""

    severity: str = Field(description="BLOCKING or SUGGESTION")
    category: str = Field(default="", description="Finding category from prior round")
    file: str | None = None
    line: int | None = None
    description: str = Field(description="Original finding description")
    suggestion: str | None = None


class FeedbackVerificationItem(BaseModel):
    """Verification result for a single prior finding."""

    prior_finding: PriorFinding
    status: VerificationStatus
    rationale: str = Field(description="Why this status was assigned")


class FeedbackVerificationResult(BaseModel):
    """Output of the feedback verifier agent."""

    items: list[FeedbackVerificationItem] = Field(default_factory=list)
    cost_usd: float = Field(default=0.0)


class DismissedFinding(BaseModel):
    """A prior finding that was explicitly dismissed by the PR author."""

    file: str | None = None
    description: str = Field(description="Original finding description (or prefix match)")
    dismissed_by: str = Field(description="GitHub username who dismissed it")
    reason: str = Field(description="Author's reason for dismissing")


class PriorReviewContext(BaseModel):
    """Context from a prior review round, fetched from storage."""

    review_id: str = Field(description="UUID of the prior code_reviews row")
    reviewed_sha: str = Field(description="HEAD SHA that was reviewed in the prior round")
    findings: list[PriorFinding] = Field(default_factory=list)
    dismissed_findings: list[DismissedFinding] = Field(default_factory=list)
    notes_for_next_round: str | None = None
    round_number: int = Field(default=2, description="Which round this will be (prior_count + 1)")
    prior_verdict: str | None = Field(
        default=None,
        description="Verdict from the prior round (APPROVE or BLOCKING) — used by preflight routing",
    )


# ---------------------------------------------------------------------------
# Post-writer: BLOCKING finding validation
# ---------------------------------------------------------------------------


class ValidationVerdict(str, Enum):
    """Whether a BLOCKING finding was confirmed or rejected by the validator."""

    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"


class FindingValidationItem(BaseModel):
    """Validation result for a single BLOCKING finding."""

    index: int = Field(description="0-based index into the BLOCKING findings list")
    verdict: ValidationVerdict
    evidence: str = Field(
        description="File:line citation or explanation proving/disproving the claim"
    )


class FindingValidationResult(BaseModel):
    """Output of the post-writer BLOCKING finding validator."""

    items: list[FindingValidationItem] = Field(default_factory=list)
    cost_usd: float = Field(default=0.0)


# ---------------------------------------------------------------------------
# Per-agent execution data (flows through graph state to storage)
# ---------------------------------------------------------------------------

AgentType = Literal[
    "system",
    "specialist",
    "cross_cutting",
    "tests_and_docs",
    "feedback_verifier",
    "blocking_validator",
]


class AgentRunData(BaseModel):
    """Execution metadata for a single sub-agent run.

    Collected in runners, propagated through the LangGraph state via a
    reducer, and batch-inserted into ``review_service.agent_runs`` after
    the review completes.
    """

    agent_name: str = Field(description="e.g. 'system:Settings & Secrets Loading', 'cross-cutting'")
    agent_type: AgentType = Field(
        description="system, specialist, cross_cutting, tests_and_docs, feedback_verifier, blocking_validator"
    )
    model: str | None = Field(default=None)
    cost_usd: float = Field(default=0.0)
    duration_seconds: float = Field(default=0.0)
    started_at: AwareDatetime | None = Field(default=None)
    finished_at: AwareDatetime | None = Field(default=None)
    tool_call_count: int = Field(default=0)
    tool_names: list[str] = Field(default_factory=list)
    context7_call_count: int = Field(default=0)
    files_explored: list[str] = Field(default_factory=list)
    finding_count: int = Field(default=0)
    result_text_length: int = Field(default=0)
    timed_out: bool = Field(
        default=False,
        description="True when the underlying subprocess hit the wall-clock timeout.",
    )

    @field_validator("started_at", "finished_at", mode="before")
    @classmethod
    def _ensure_tz_aware(cls, v: Any) -> Any:
        """Coerce naive datetimes to UTC instead of rejecting them."""
        if isinstance(v, datetime) and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v
