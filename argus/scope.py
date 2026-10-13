"""Full-PR scope gate and workload halt helpers (pure; no network, no LLM).

The graph's ``scope_gate`` node asks a model to split the full PR into
independently mergeable behaviors. Everything the model returns is untrusted:
``validate_assessment`` checks it against the pinned file manifest, and any
doubt becomes a halt (always BLOCKING), never an approval.
"""

from __future__ import annotations

import json
import re
from graphlib import CycleError, TopologicalSorter
from typing import Any

from pydantic import BaseModel, Field

from argus.models import (
    HaltCode,
    ReviewHalt,
    ReviewResponse,
    RiskLevel,
    ScopeBehavior,
    ScopeReceipt,
    Verdict,
)
from argus.pipeline_models import ReviewPlan

SCOPE_PROMPT = "pr-review-scope"
MAX_BEHAVIORS = 12  # model output bound; more is treated as unverifiable
MAX_SPLIT_SUGGESTIONS = 8  # slices shown in a halt
PATCH_CHARS = 6000  # per-file patch cap sent to the model
INPUT_CHARS = 150_000  # whole-input cap; beyond it the gate halts instead of guessing
DESCRIPTION_CHARS = 4000
OVERRIDE_REASON_MAX = 500
SHA40 = re.compile(r"[0-9a-f]{40}")


class ScopeUnverified(Exception):  # noqa: N818 -- reads as the halt code it maps to
    """The scope could not be verified; ``str(exc)`` is safe to show in a PR comment."""


class ScopeAssessment(BaseModel):
    """Structured model output for the scope gate."""

    indeterminate: bool = Field(
        description="True when the evidence is too weak to partition the PR confidently"
    )
    behaviors: list[ScopeBehavior]
    rationale: str


def validate_override(sha: str | None, reason: str | None) -> tuple[str, str] | None:
    """Return ``(sha, reason)`` for a valid human override pair, ``None`` if neither is set."""
    if sha is None and reason is None:
        return None
    if sha is None or reason is None:
        raise ValueError("--review-override-sha and --review-override-reason must be set together")
    reason = reason.strip()
    if not SHA40.fullmatch(sha):
        raise ValueError("--review-override-sha must be a 40-character lowercase hex SHA")
    if not 1 <= len(reason) <= OVERRIDE_REASON_MAX:
        raise ValueError(
            f"--review-override-reason must be 1-{OVERRIDE_REASON_MAX} characters after stripping"
        )
    return sha, reason


def build_scope_input(files: list[dict[str, Any]], description: str) -> str:
    """Serialize the manifest as JSON so patch text cannot break out of the data envelope."""
    entries = []
    for f in files:
        patch = f.get("patch") or ""
        if not patch:
            note = "no patch available (binary, oversized, or rename/mode-only): treat as changed"
        elif len(patch) > PATCH_CHARS:
            patch, note = patch[:PATCH_CHARS], "patch truncated"
        else:
            note = ""
        entries.append(
            {
                "filename": f.get("filename", ""),
                "status": f.get("status", "modified"),
                "previous_filename": f.get("previous_filename") or None,
                "patch": patch,
                "patch_note": note,
            }
        )
    payload = json.dumps(
        {"pr_description": description[:DESCRIPTION_CHARS], "files": entries}, ensure_ascii=False
    )
    if len(payload) > INPUT_CHARS:
        raise ScopeUnverified(
            f"PR is too large to scope-check ({len(files)} files, {len(payload)} input chars)"
        )
    return payload


def validate_assessment(assessment: ScopeAssessment, manifest: set[str]) -> None:
    """Raise ``ScopeUnverified`` unless the behaviors exactly partition ``manifest``."""
    behaviors = assessment.behaviors
    if assessment.indeterminate:
        raise ScopeUnverified("scope model could not decide (indeterminate)")
    if not manifest:
        if behaviors:
            raise ScopeUnverified("scope model returned behaviors for an empty PR")
        return
    if not behaviors:
        raise ScopeUnverified("scope model returned no behaviors for a non-empty PR")
    if len(behaviors) > MAX_BEHAVIORS:
        raise ScopeUnverified(
            f"scope model returned {len(behaviors)} behaviors (max {MAX_BEHAVIORS})"
        )
    titles = [b.title for b in behaviors]
    if any(t != t.strip() or not t for t in titles + [d for b in behaviors for d in b.depends_on]):
        raise ScopeUnverified("behavior titles and dependencies must be non-empty and trimmed")
    if len(set(titles)) != len(titles):
        raise ScopeUnverified("behavior titles must be unique")
    if any(not b.files for b in behaviors):
        raise ScopeUnverified("a behavior has no files")
    assigned = [f for b in behaviors for f in b.files]
    if len(assigned) != len(set(assigned)):
        raise ScopeUnverified("a file is assigned to more than one behavior")
    unknown = set(assigned) - manifest
    if unknown:
        raise ScopeUnverified(f"{len(unknown)} assigned file(s) are not in the PR manifest")
    unassigned = manifest - set(assigned)
    if unassigned:
        raise ScopeUnverified(f"{len(unassigned)} PR file(s) are not assigned to any behavior")
    graph: dict[str, set[str]] = {}
    for b in behaviors:
        deps = set(b.depends_on)
        if b.title in deps or not deps <= set(titles):
            raise ScopeUnverified("behavior depends on itself or on an unknown behavior")
        graph[b.title] = deps
    try:
        TopologicalSorter(graph).prepare()
    except CycleError as exc:
        raise ScopeUnverified("behavior dependencies are cyclic") from exc


def dispatch_count(plan: ReviewPlan) -> int:
    """Reviewers the plan would dispatch: each group + its specialists, plus 2 always-on."""
    return sum(1 + len(g.specialists_needed) for g in plan.system_groups) + 2


def require_receipt(scope: ScopeReceipt | None, head_sha: str) -> None:
    """Raise unless ``scope`` is a coherent receipt for ``head_sha`` (APPROVE needs one)."""
    if scope is None or scope.head_sha != head_sha:
        raise RuntimeError("refusing APPROVE: scope receipt missing or not pinned to the head")
    if len(scope.behaviors) > 1 and not scope.override_reason:
        raise RuntimeError("refusing APPROVE: multi-behavior scope without a human override")


def _inline(text: str) -> str:
    """Render untrusted model text as one short code span (no mentions, links, or markup)."""
    return "`" + " ".join(text.split())[:80].replace("`", "'") + "`"


def halt_response(
    code: HaltCode,
    detail: str,
    head_sha: str,
    *,
    review_round: int = 1,
    prior_review_id: str | None = None,
    split_suggestions: list[ScopeBehavior] | None = None,
    dispatch_count: int | None = None,
    dispatch_budget: int | None = None,
) -> ReviewResponse:
    """Build the BLOCKING/HIGH response for a halt; deliberately has no fabricated findings."""
    split = (split_suggestions or [])[:MAX_SPLIT_SUGGESTIONS]
    lines = [
        f"## Code Review — Round {review_round} — Halted (`{code}`)",
        "",
        f"Argus stopped before a full review at `{head_sha[:12]}`: {detail}",
    ]
    if split:
        lines += ["", "Suggested independently mergeable slices:"]
        for b in split:
            dep = f" (after: {', '.join(map(_inline, b.depends_on))})" if b.depends_on else ""
            lines.append(f"- {_inline(b.title)}{dep}: {len(b.files)} file(s)")
    lines += ["", "This is a BLOCKING halt, not a verdict on code quality."]
    return ReviewResponse(
        verdict=Verdict.BLOCKING,
        risk_level=RiskLevel.HIGH,
        review_comment="\n".join(lines),
        review_round=review_round,
        prior_review_id=prior_review_id,
        preflight_reason=f"halt: {code}",
        halt=ReviewHalt(
            code=code,
            detail=detail,
            head_sha=head_sha,
            split_suggestions=split,
            dispatch_count=dispatch_count,
            dispatch_budget=dispatch_budget,
        ),
    )
