"""Data contracts for the review service API.

These Pydantic models define the request/response schema for the review API.
They are the I/O contracts that both the GHA workflow (caller) and the
service implementation must agree on.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Verdict(str, Enum):
    """Overall review verdict."""

    APPROVE = "APPROVE"
    BLOCKING = "BLOCKING"


class RiskLevel(str, Enum):
    """Risk classification for the PR."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Severity(str, Enum):
    """Finding severity — only two levels by design.

    BLOCKING = runtime failures and security vulnerabilities only.
    SUGGESTION = everything else.
    """

    BLOCKING = "BLOCKING"
    SUGGESTION = "SUGGESTION"


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class ReviewRequest(BaseModel):
    """POST /review request body.

    Provide either pr_number OR sha (not both). When sha is provided,
    the base_ref is looked up automatically from the PR's merge-base
    if pr_number is also 0, or can be supplied explicitly.
    """

    repo: str = Field(
        ...,
        pattern=r"^[\w.-]+/[\w.-]+$",
        description="Full repo identifier, e.g. 'owner/repo'",
    )
    pr_number: int = Field(0, description="Pull request number to review (0 if using sha mode)")
    sha: Optional[str] = Field(
        None,
        description="Specific commit SHA to review (for v1/v2 comparison testing)",
    )
    base_ref: Optional[str] = Field(
        None,
        description="Base SHA or ref for diff. Auto-resolved from PR merge-base when omitted.",
    )
    dismissals: list[str] = Field(
        default_factory=list,
        description="Local-only dismiss commands (from argus_review_local.py --dismiss). "
        "Not exposed via API or an external orchestrator — those paths always default to [].",
    )


# ---------------------------------------------------------------------------
# Response — Findings
# ---------------------------------------------------------------------------


class Finding(BaseModel):
    """A single review finding."""

    severity: Severity
    category: str = Field(
        ...,
        description="Review category: code-correctness, security, cross-layer, test-adequacy, cross-cutting",
    )
    file: Optional[str] = Field(None, description="File path, if applicable")
    line: Optional[int] = Field(None, description="Line number, if applicable")
    description: str = Field(..., description="Human-readable description of the finding")
    suggestion: Optional[str] = Field(
        None,
        description="Suggested fix or improvement",
    )


# ---------------------------------------------------------------------------
# Response — Coverage map
# ---------------------------------------------------------------------------


class SystemCoverage(BaseModel):
    """Coverage status for a reviewed system group."""

    system: str = Field(description="System group name, e.g. 'Review service orchestrator'")
    files_explored: list[str] = Field(default_factory=list, description="Files the subagent read")
    checks_performed: list[str] = Field(
        default_factory=list,
        description="What conventions/patterns were checked",
    )


# ---------------------------------------------------------------------------
# Response — Token usage and cost tracking
# ---------------------------------------------------------------------------


class TokenUsage(BaseModel):
    """Aggregated token usage across orchestrator + all subagents."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float = Field(0.0, description="Estimated total cost in USD")


# ---------------------------------------------------------------------------
# Response — Top level
# ---------------------------------------------------------------------------


class PriorFeedbackItem(BaseModel):
    """Status of a prior-round finding after verification."""

    severity: str = Field(description="Original severity: BLOCKING or SUGGESTION")
    description: str = Field(description="Original finding description")
    file: Optional[str] = None
    status: str = Field(description="RESOLVED, UNRESOLVED, or REGRESSED")
    rationale: str = Field(default="", description="Why this status was assigned")


class DroppedFinding(BaseModel):
    """A BLOCKING finding that was rejected by the post-writer validator."""

    finding: Finding
    rejection_rationale: str = Field(
        description="Evidence-based explanation of why this finding was rejected"
    )


class ReviewResponse(BaseModel):
    """POST /review response body."""

    verdict: Verdict
    risk_level: RiskLevel
    findings: list[Finding] = Field(default_factory=list)
    dropped_findings: list[DroppedFinding] = Field(
        default_factory=list,
        description="BLOCKING findings rejected by the validator (audit trail, not shown in comment)",
    )
    prior_feedback: list[PriorFeedbackItem] = Field(
        default_factory=list,
        description="Verification status of prior-round findings (empty for round 1)",
    )
    coverage_map: list[SystemCoverage] = Field(default_factory=list)
    notes_for_next_round: Optional[str] = Field(
        None,
        description="Structured notes for the next review round (coverage map + findings with status)",
    )
    review_comment: str = Field(
        ...,
        description="Formatted markdown comment ready to post on the PR",
    )
    review_round: int = Field(
        default=1,
        description="Which review round this is (1 = first review, 2+ = incremental)",
    )
    prior_review_id: Optional[str] = Field(
        None,
        description="UUID of the prior review this round builds on (None for round 1)",
    )
    usage: TokenUsage = Field(default_factory=TokenUsage)
    lite_mode: bool = Field(
        default=False,
        description="True if this review was performed in lite mode (single-pass check)",
    )
    preflight_reason: str = Field(
        default="",
        description="Preflight routing reason — stored in result_json for lite round history",
    )
