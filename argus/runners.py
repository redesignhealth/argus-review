"""Subagent runner functions for the v3 review pipeline.

Each runner creates a standalone ClaudeSDKClient session with Read/Glob/Grep
tools and Context7 MCP for codebase documentation retrieval. Called as
LangGraph @task nodes — retry and timeout are handled by LangGraph
RetryPolicy, not application code.

LangSmith tracing is automatic when LANGCHAIN_TRACING_V2=true (no @traceable shim needed).
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import re
import signal
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from argus.llm.models import CLAUDE_DEFAULT, CLAUDE_FRONTIER
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TaskStartedMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)
from langsmith import traceable

from argus.config import get_settings
from argus.helpers import (
    filter_diff_for_files as _filter_diff_for_files,
)
from argus.helpers import (
    parse_review_result as _parse_review_result,
)
from argus.helpers import (
    sanitize_file_paths as _sanitize_file_paths,
)
from argus.pipeline_models import (
    AgentRunData,
    AgentType,
    FeedbackVerificationItem,
    FeedbackVerificationResult,
    FindingValidationItem,
    FindingValidationResult,
    PriorFinding,
    PriorReviewContext,
    ReviewPlan,
    SpecialistName,
    SystemGroup,
    SystemReviewResult,
    ValidationVerdict,
    VerificationStatus,
)
from argus.prompts_runtime import fetch_prompt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SYSTEM_REVIEWER_MODEL = CLAUDE_DEFAULT
_CROSS_CUTTING_MODEL = CLAUDE_FRONTIER
_MAX_TURNS = 30
# Fallback repo root for ClaudeSDKClient cwd — used when no SHA-pinned
# worktree has been provisioned (e.g. local dev runs, tests, subprocess
# worker). In production, callers pass an explicit repo_root provisioned
# by repo_provision.provisioned_worktree().
_REPO_ROOT = str(Path(__file__).resolve().parents[4])
# Fallback used when no Settings instance is available (e.g. direct unit
# tests calling _run_session_in_subprocess without a settings-derived
# timeout_s). Production call sites pass settings.ARGUS_SESSION_TIMEOUT.
_SUBPROCESS_TIMEOUT_S = 300


def _resolve_repo_root(repo_root: str | None, caller: str) -> str:
    """Return the effective repo root for a reviewer's ``cwd``.

    Falls back to the module-level ``_REPO_ROOT`` (the ambient checkout) when
    no SHA-pinned worktree was provided, logging a warning because in that case
    Read/Glob/Grep resolve against whatever the ambient checkout is on, not the
    reviewed SHA. ``caller`` names the reviewer for the log line.
    """
    if repo_root is None:
        logger.warning(
            "%s: repo_root is None; falling back to _REPO_ROOT; "
            "Read/Glob/Grep will resolve against the ambient checkout, not the reviewed SHA",
            caller,
        )
        return _REPO_ROOT
    return repo_root


# Shared directive constraining reviewer findings to diff-causal issues only.
# Reviewers may explore the full codebase for context but must only report
# issues that are in the diff or directly caused by changes in the diff.
_CAUSAL_SCOPE_DIRECTIVE = """\

## Finding Scope — Changed or Caused by This Diff

Your findings MUST be directly related to the changes in this diff — either \
code that was changed, or existing code that is newly broken or exposed by \
the changes.

**In scope:**
- Code that was added or modified in this diff
- A new call to an existing function that has a bug (the diff introduced the call path)
- An IAM policy that doesn't cover a new SSM key added in this diff (the diff created the need)
- A missing migration for a new column referenced by the diff

**Out of scope:**
- A pre-existing pattern in code that wasn't changed and isn't newly exercised by the diff
- A lockfile, Dockerfile, or config file that wasn't touched and isn't affected by the diff
- Style or convention issues in unchanged adjacent code
- Pre-existing bugs in code the diff doesn't interact with

**The test:** "Would this issue exist regardless of whether this PR lands?" \
If yes, it's pre-existing and out of scope for this review.

You SHOULD explore unchanged files to understand context, verify claims, and \
trace dependencies. But only REPORT findings that pass the causal test above.
"""

# Context7 MCP configuration. The library ID is deployment-specific (which
# codebase's curated docs to query) and comes from ARGUS_CONTEXT7_LIBRARY_ID;
# when unset, Context7 attachment is skipped entirely (same graceful
# degradation as a missing CONTEXT7_API_KEY).
_CONTEXT7_MCP_URL = "https://mcp.context7.com/mcp"


@dataclass
class SessionResult:
    """Rich return from _run_claude_session with timing and tool metadata."""

    result_text: str
    cost_usd: float
    duration_seconds: float
    started_at: datetime
    finished_at: datetime
    tool_call_count: int
    tool_names: list[str] = field(default_factory=list)
    context7_call_count: int = 0
    model: str | None = None
    timed_out: bool = False


def _empty_session_result(
    model: str | None = None,
    duration_seconds: float = 0.0,
    started_at: datetime | None = None,
    timed_out: bool = False,
) -> SessionResult:
    """Default SessionResult for subprocess timeout / worker failure.

    When ``started_at`` is provided (timeout path), ``finished_at`` is computed
    as ``started_at + timedelta(seconds=duration_seconds)`` so the timestamps
    are internally consistent with the reported duration. When omitted, both
    timestamps default to now and duration is 0.

    ``timed_out`` distinguishes "subprocess hit the wall-clock timeout and was
    killed" from other empty-result paths (worker crash, no output) so callers
    can surface a TIMED_OUT status instead of silently reporting 0 findings.
    """
    if started_at is not None:
        finished_at = started_at + timedelta(seconds=duration_seconds)
    else:
        started_at = datetime.now(timezone.utc)
        finished_at = started_at
    return SessionResult(
        result_text="",
        cost_usd=0.0,
        duration_seconds=duration_seconds,
        started_at=started_at,
        finished_at=finished_at,
        tool_call_count=0,
        model=model,
        timed_out=timed_out,
    )


# Sized above graph._MAX_CONCURRENT_REVIEWERS (16) so isolated reviewer
# subprocesses never queue behind the default asyncio thread pool.
_REVIEWER_THREAD_POOL_SIZE = 20
_REVIEWER_EXECUTOR: ThreadPoolExecutor | None = None
_REVIEWER_EXECUTOR_LOCK = threading.Lock()


def _get_reviewer_executor() -> ThreadPoolExecutor:
    """Lazily create the dedicated reviewer-subprocess thread pool.

    Lazy so spawned worker processes (which import this module only to call
    _run_claude_session) do not each allocate a pool they never use.
    """
    global _REVIEWER_EXECUTOR
    if _REVIEWER_EXECUTOR is None:
        with _REVIEWER_EXECUTOR_LOCK:
            if _REVIEWER_EXECUTOR is None:
                _REVIEWER_EXECUTOR = ThreadPoolExecutor(
                    max_workers=_REVIEWER_THREAD_POOL_SIZE,
                    thread_name_prefix="argus-reviewer",
                )
    return _REVIEWER_EXECUTOR


async def _run_session_isolated(**kwargs: Any) -> SessionResult:
    """Run _run_session_in_subprocess on a dedicated thread pool so concurrent
    reviewer sessions do not queue behind the default asyncio executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _get_reviewer_executor(), functools.partial(_run_session_in_subprocess, **kwargs)
    )


def _context7_system_directive(settings: Any) -> str:
    """Return Context7 system prompt directive if configured, else empty string.

    Placed in the system prompt (not user message) so the agent treats it as
    a mandatory step, not an optional suggestion.
    """
    library_id = getattr(settings, "ARGUS_CONTEXT7_LIBRARY_ID", None)
    if getattr(settings, "CONTEXT7_API_KEY", None) and library_id:
        return (
            "\n\n## Required: Context7 Codebase Documentation Lookup\n\n"
            "BEFORE reading any repo files, you MUST query Context7 for relevant conventions:\n"
            f"1. Call `mcp__context7__query-docs` with libraryId `{library_id}` "
            "and a query relevant to the files being reviewed (e.g., 'database access patterns', "
            "'error handling conventions', 'security patterns', 'testing requirements')\n"
            "2. Use the retrieved conventions to inform your review — flag code that violates them\n\n"
            "This is NOT optional. Context7 contains curated coding standards and common review "
            "findings that may not be obvious from reading the code alone.\n"
        )
    return ""


# ---------------------------------------------------------------------------
# Specialist prompt mapping
# ---------------------------------------------------------------------------

_SPECIALIST_PROMPT_MAP: dict[SpecialistName, str] = {
    "security": "pr-review-specialist-security",
    "sql": "pr-review-specialist-sql",
    "infra": "pr-review-specialist-infra",
    "orchestration": "pr-review-specialist-orchestration",
    "frontend": "pr-review-specialist-frontend",
    "slackbot": "pr-review-specialist-slackbot",
    "deployment": "pr-review-specialist-deployment",
    "llm-patterns": "pr-review-specialist-llm-patterns",
    "observability": "pr-review-specialist-observability",
}


# ---------------------------------------------------------------------------
# Agent run metadata helper
# ---------------------------------------------------------------------------


def _build_agent_run(
    *,
    session: SessionResult,
    agent_name: str,
    agent_type: AgentType,
    result: SystemReviewResult | FeedbackVerificationResult | FindingValidationResult | None = None,
) -> AgentRunData:
    """Build an AgentRunData from a SessionResult and optional parsed result."""
    files_explored: list[str] = []
    finding_count = 0
    if isinstance(result, SystemReviewResult):
        files_explored = result.files_explored
        finding_count = len(result.findings)
    elif isinstance(result, FeedbackVerificationResult):
        finding_count = len(result.items)
    elif isinstance(result, FindingValidationResult):
        finding_count = len(result.items)

    return AgentRunData(
        agent_name=agent_name,
        agent_type=agent_type,
        model=session.model,
        cost_usd=session.cost_usd,
        duration_seconds=session.duration_seconds,
        started_at=session.started_at,
        finished_at=session.finished_at,
        tool_call_count=session.tool_call_count,
        tool_names=session.tool_names,
        context7_call_count=session.context7_call_count,
        files_explored=files_explored,
        finding_count=finding_count,
        result_text_length=len(session.result_text),
        timed_out=session.timed_out,
    )


# ---------------------------------------------------------------------------
# Runner functions (called by LangGraph @task nodes)
# ---------------------------------------------------------------------------


async def run_system_reviewer(
    group: SystemGroup,
    diff_text: str,
    settings: Any | None = None,
    *,
    repo_root: str | None = None,
) -> tuple[SystemReviewResult, AgentRunData | None]:
    """Run a system reviewer subagent for a single file group.

    Failures propagate to the caller (LangGraph RetryPolicy handles retry).
    Returns (review_result, agent_run_data) — agent_run_data is None when skipped.

    Args:
        repo_root: Absolute path to the SHA-pinned worktree provisioned by
            ``repo_provision.provisioned_worktree``. Falls back to ``_REPO_ROOT``
            when None (local dev / tests).
    """
    if settings is None:
        settings = get_settings()

    effective_root = _resolve_repo_root(repo_root, "run_system_reviewer")
    safe_files = _sanitize_file_paths(group.files, effective_root)
    filtered_diff = _filter_diff_for_files(diff_text, safe_files)
    if not filtered_diff:
        logger.info("No diff hunks for group '%s', skipping review", group.name)
        return SystemReviewResult(system_group=group.name, findings=[], files_explored=[]), None

    base_prompt = await fetch_prompt("pr-review-subagent")
    prior_art_prompt = await fetch_prompt("pr-review-prior-art")
    logger.info("Prior art prompt loaded: %d chars, group=%s", len(prior_art_prompt), group.name)

    system_prompt = (
        f"{base_prompt}\n\n"
        f"{prior_art_prompt}\n\n"
        f"## Conventions for this group\n\n{group.conventions}\n\n"
        f"## Review focus\n\n{group.review_focus}\n\n"
        f"{_context7_system_directive(settings)}"
        "## Output format\n\n"
        "Return your findings as a JSON object with these keys:\n"
        '- "system_group": the group name\n'
        '- "findings": list of objects with {file, line, description, context}\n'
        '- "files_explored": list of file paths you read for context\n\n'
        "Wrap the JSON in a ```json code block."
    )

    user_message = (
        f"Review the following diff for the **{group.name}** system group.\n\n"
        f"Files in this group: {', '.join(safe_files)}\n\n"
        f"```diff\n{filtered_diff}\n```\n\n"
        "Use Read, Glob, and Grep to explore the surrounding codebase for context. "
        "Focus on correctness, security, and adherence to the conventions above."
        f"{_CAUSAL_SCOPE_DIRECTIVE}"
    )

    session = await _run_session_isolated(
        model=_SYSTEM_REVIEWER_MODEL,
        system_prompt=system_prompt,
        user_message=user_message,
        anthropic_api_key=settings.ANTHROPIC_API_KEY,
        anthropic_auth_token=settings.ANTHROPIC_AUTH_TOKEN,
        context7_key=getattr(settings, "CONTEXT7_API_KEY", None),
        context7_library_id=getattr(settings, "ARGUS_CONTEXT7_LIBRARY_ID", None),
        timeout_s=getattr(settings, "ARGUS_SESSION_TIMEOUT", _SUBPROCESS_TIMEOUT_S),
        cwd=effective_root,
        label=f"system:{group.name}",
    )
    result = _parse_review_result(session.result_text, group.name)
    result.cost_usd = session.cost_usd
    result.timed_out = session.timed_out
    agent_run = _build_agent_run(
        session=session,
        agent_name=f"system:{group.name}",
        agent_type="system",
        result=result,
    )
    return result, agent_run


async def run_specialist_reviewer(
    specialist: SpecialistName,
    group: SystemGroup,
    diff_text: str,
    settings: Any | None = None,
    *,
    repo_root: str | None = None,
) -> tuple[SystemReviewResult, AgentRunData | None]:
    """Run a specialist co-reviewer alongside the general system reviewer.

    Returns (review_result, agent_run_data) — agent_run_data is None when skipped.

    The ``specialist`` argument is typed as ``SpecialistName`` (a Literal), so
    invalid values are caught by Pydantic at planner-output validation time —
    long before they reach this function. The ValueError below is a defense-
    in-depth guard for bypassed validation paths (e.g. callers constructing
    SystemGroup with ``model_construct`` to skip validation).

    Args:
        repo_root: Absolute path to the SHA-pinned worktree. Falls back to
            ``_REPO_ROOT`` when None.
    """
    if settings is None:
        settings = get_settings()

    prompt_name = _SPECIALIST_PROMPT_MAP.get(specialist)
    if not prompt_name:
        raise ValueError(f"Unknown specialist: {specialist!r}")

    effective_root = _resolve_repo_root(repo_root, "run_specialist_reviewer")
    safe_files = _sanitize_file_paths(group.files, effective_root)
    filtered_diff = _filter_diff_for_files(diff_text, safe_files)
    if not filtered_diff:
        return (
            SystemReviewResult(
                system_group=f"{group.name}::{specialist}", findings=[], files_explored=[]
            ),
            None,
        )

    base_prompt = await fetch_prompt(prompt_name)

    system_prompt = (
        f"{base_prompt}\n\n"
        f"{_context7_system_directive(settings)}"
        "## Output format\n\n"
        "Return your findings as a JSON object with these keys:\n"
        f'- "system_group": "{group.name}::{specialist}"\n'
        '- "findings": list of objects with {file, line, description, context}\n'
        '- "files_explored": list of file paths you read for context\n\n'
        "Wrap the JSON in a ```json code block."
    )

    user_message = (
        f"Review the following diff for **{specialist}** issues in the "
        f"**{group.name}** system group.\n\n"
        f"Files: {', '.join(safe_files)}\n\n"
        f"```diff\n{filtered_diff}\n```\n\n"
        "Use Read, Glob, and Grep to explore the codebase for context."
        f"{_CAUSAL_SCOPE_DIRECTIVE}"
    )

    session = await _run_session_isolated(
        model=_SYSTEM_REVIEWER_MODEL,
        system_prompt=system_prompt,
        user_message=user_message,
        anthropic_api_key=settings.ANTHROPIC_API_KEY,
        anthropic_auth_token=settings.ANTHROPIC_AUTH_TOKEN,
        context7_key=getattr(settings, "CONTEXT7_API_KEY", None),
        context7_library_id=getattr(settings, "ARGUS_CONTEXT7_LIBRARY_ID", None),
        timeout_s=getattr(settings, "ARGUS_SESSION_TIMEOUT", _SUBPROCESS_TIMEOUT_S),
        cwd=effective_root,
        label=f"specialist:{group.name}::{specialist}",
    )
    result = _parse_review_result(session.result_text, f"{group.name}::{specialist}")
    result.cost_usd = session.cost_usd
    result.timed_out = session.timed_out
    agent_run = _build_agent_run(
        session=session,
        agent_name=f"specialist:{group.name}::{specialist}",
        agent_type="specialist",
        result=result,
    )
    return result, agent_run


async def run_cross_cutting_reviewer(
    plan: ReviewPlan,
    diff_text: str,
    settings: Any | None = None,
    *,
    repo_root: str | None = None,
) -> tuple[SystemReviewResult, AgentRunData | None]:
    """Run the cross-cutting reviewer subagent (Opus).

    Returns (review_result, agent_run_data).

    Args:
        repo_root: Absolute path to the SHA-pinned worktree. Falls back to
            ``_REPO_ROOT`` when None.
    """
    if settings is None:
        settings = get_settings()

    effective_root = _resolve_repo_root(repo_root, "run_cross_cutting_reviewer")

    base_prompt = await fetch_prompt("pr-review-cross-cutting")
    prior_art_prompt = await fetch_prompt("pr-review-prior-art")
    logger.info("Prior art prompt loaded: %d chars, reviewer=cross-cutting", len(prior_art_prompt))

    system_prompt = (
        f"{base_prompt}\n\n"
        f"{prior_art_prompt}\n\n"
        f"{_context7_system_directive(settings)}"
        "## Output format\n\n"
        "Return your findings as a JSON object with these keys:\n"
        '- "system_group": "cross-cutting"\n'
        '- "findings": list of objects with {file, line, description, context}\n'
        '- "files_explored": list of file paths you read for context\n\n'
        "Wrap the JSON in a ```json code block."
    )

    concerns = "\n".join(f"- {c}" for c in plan.cross_cutting_concerns) or "- None identified"
    all_files = "\n".join(f"- {fe.path} ({fe.change_type})" for fe in plan.file_manifest)
    groups_summary = "\n".join(f"- **{g.name}**: {', '.join(g.files)}" for g in plan.system_groups)

    user_message = (
        "Review this PR for cross-cutting issues that span multiple system groups.\n\n"
        f"## System Groups\n\n{groups_summary}\n\n"
        f"## Cross-Cutting Concerns to Investigate\n\n{concerns}\n\n"
        f"## All Changed Files\n\n{all_files}\n\n"
        f"## Full PR Diff\n\n```diff\n{diff_text}\n```\n\n"
        "Use Read, Glob, and Grep to trace execution paths across files."
        f"{_CAUSAL_SCOPE_DIRECTIVE}"
    )

    session = await _run_session_isolated(
        model=_CROSS_CUTTING_MODEL,
        system_prompt=system_prompt,
        user_message=user_message,
        anthropic_api_key=settings.ANTHROPIC_API_KEY,
        anthropic_auth_token=settings.ANTHROPIC_AUTH_TOKEN,
        context7_key=getattr(settings, "CONTEXT7_API_KEY", None),
        context7_library_id=getattr(settings, "ARGUS_CONTEXT7_LIBRARY_ID", None),
        timeout_s=getattr(settings, "ARGUS_SESSION_TIMEOUT", _SUBPROCESS_TIMEOUT_S),
        cwd=effective_root,
        label="cross-cutting",
    )
    result = _parse_review_result(session.result_text, "cross-cutting")
    result.cost_usd = session.cost_usd
    result.timed_out = session.timed_out
    agent_run = _build_agent_run(
        session=session,
        agent_name="cross-cutting",
        agent_type="cross_cutting",
        result=result,
    )
    return result, agent_run


async def run_tests_and_docs_reviewer(
    plan: ReviewPlan,
    diff_text: str,
    settings: Any | None = None,
    *,
    repo_root: str | None = None,
) -> tuple[SystemReviewResult, AgentRunData | None]:
    """Always-on reviewer for test coverage and documentation compliance.

    Returns (review_result, agent_run_data).

    Args:
        repo_root: Absolute path to the SHA-pinned worktree. Falls back to
            ``_REPO_ROOT`` when None.
    """
    if settings is None:
        settings = get_settings()

    effective_root = _resolve_repo_root(repo_root, "run_tests_and_docs_reviewer")

    base_prompt = await fetch_prompt("pr-review-tests-and-docs")
    system_prompt = f"{base_prompt}\n\n{_context7_system_directive(settings)}"

    all_files = "\n".join(f"- {fe.path} ({fe.change_type})" for fe in plan.file_manifest)

    user_message = (
        "Review this PR for test coverage gaps and documentation compliance.\n\n"
        f"## All Changed Files\n\n{all_files}\n\n"
        f"## Full PR Diff\n\n```diff\n{diff_text}\n```\n\n"
        "Use Read, Glob, and Grep to check test directories and .cursorrules files."
        f"{_CAUSAL_SCOPE_DIRECTIVE}"
    )

    session = await _run_session_isolated(
        model=_SYSTEM_REVIEWER_MODEL,
        system_prompt=system_prompt,
        user_message=user_message,
        anthropic_api_key=settings.ANTHROPIC_API_KEY,
        anthropic_auth_token=settings.ANTHROPIC_AUTH_TOKEN,
        context7_key=getattr(settings, "CONTEXT7_API_KEY", None),
        context7_library_id=getattr(settings, "ARGUS_CONTEXT7_LIBRARY_ID", None),
        timeout_s=getattr(settings, "ARGUS_SESSION_TIMEOUT", _SUBPROCESS_TIMEOUT_S),
        cwd=effective_root,
        label="tests-and-docs",
    )
    result = _parse_review_result(session.result_text, "tests-and-docs")
    result.cost_usd = session.cost_usd
    result.timed_out = session.timed_out
    agent_run = _build_agent_run(
        session=session,
        agent_name="tests-and-docs",
        agent_type="tests_and_docs",
        result=result,
    )
    return result, agent_run


async def run_feedback_verifier(
    prior_context: PriorReviewContext,
    diff_text: str,
    settings: Any | None = None,
    *,
    repo_root: str | None = None,
) -> tuple[FeedbackVerificationResult, AgentRunData | None]:
    """Verify whether prior-round findings have been addressed in the new diff.

    For each prior BLOCKING/SUGGESTION finding, determines: RESOLVED (the diff
    addresses it), UNRESOLVED (not addressed), or REGRESSED (fix introduced a
    new problem).

    Returns (verification_result, agent_run_data) — agent_run_data is None when skipped.

    Args:
        repo_root: Absolute path to the SHA-pinned worktree. Falls back to
            ``_REPO_ROOT`` when None.
    """
    if settings is None:
        settings = get_settings()

    if not prior_context.findings:
        return FeedbackVerificationResult(items=[], cost_usd=0.0), None

    effective_root = _resolve_repo_root(repo_root, "run_feedback_verifier")

    findings_json = json.dumps([f.model_dump() for f in prior_context.findings], indent=2)

    base_prompt = await fetch_prompt("pr-review-feedback-verifier")

    system_prompt = f"{base_prompt}\n\n{_context7_system_directive(settings)}"

    user_message = (
        "## Prior Review Findings\n\n"
        f"```json\n{findings_json}\n```\n\n"
        "## New Changes (diff since last review)\n\n"
        f"```diff\n{diff_text}\n```\n\n"
        "For each prior finding above, determine whether the new changes have "
        "resolved it, left it unresolved, or caused a regression. "
        "Use Read, Glob, and Grep to verify against the actual codebase."
    )

    # Sonnet, not Opus: feedback verifier checks N prior findings in bulk
    # (resolved/unresolved/regressed) - volume-oriented like system reviewers.
    session = await _run_session_isolated(
        model=_SYSTEM_REVIEWER_MODEL,
        system_prompt=system_prompt,
        user_message=user_message,
        anthropic_api_key=settings.ANTHROPIC_API_KEY,
        anthropic_auth_token=settings.ANTHROPIC_AUTH_TOKEN,
        context7_key=getattr(settings, "CONTEXT7_API_KEY", None),
        context7_library_id=getattr(settings, "ARGUS_CONTEXT7_LIBRARY_ID", None),
        timeout_s=getattr(settings, "ARGUS_SESSION_TIMEOUT", _SUBPROCESS_TIMEOUT_S),
        cwd=effective_root,
        label="feedback-verifier",
    )

    result = _parse_verification_result(
        session.result_text, prior_context.findings, session.cost_usd
    )
    agent_run = _build_agent_run(
        session=session,
        agent_name="feedback-verifier",
        agent_type="feedback_verifier",
        result=result,
    )
    return result, agent_run


def _parse_verification_result(
    raw_text: str,
    prior_findings: list[Any],
    cost_usd: float,
) -> FeedbackVerificationResult:
    """Parse the feedback verifier's JSON output into FeedbackVerificationResult."""
    if not raw_text.strip():
        return FeedbackVerificationResult(items=[], cost_usd=cost_usd)

    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    json_str = json_match.group(1) if json_match else raw_text.strip()

    try:
        data = json.loads(json_str)
        items: list[FeedbackVerificationItem] = []

        for item in data.get("items", []):
            idx = item.get("index", -1)
            if not isinstance(idx, int) or idx < 0 or idx >= len(prior_findings):
                continue

            status_str = item.get("status", "UNRESOLVED").upper()
            try:
                status = VerificationStatus(status_str)
            except ValueError:
                status = VerificationStatus.UNRESOLVED

            pf = prior_findings[idx]
            items.append(
                FeedbackVerificationItem(
                    prior_finding=pf
                    if isinstance(pf, PriorFinding)
                    else PriorFinding.model_validate(pf),
                    status=status,
                    rationale=item.get("rationale", ""),
                )
            )

        return FeedbackVerificationResult(items=items, cost_usd=cost_usd)
    except (json.JSONDecodeError, TypeError, KeyError):
        logger.warning("Could not parse feedback verifier output, marking all as UNRESOLVED")
        items = [
            FeedbackVerificationItem(
                prior_finding=pf
                if isinstance(pf, PriorFinding)
                else PriorFinding.model_validate(pf),
                status=VerificationStatus.UNRESOLVED,
                rationale="Verification output could not be parsed",
            )
            for pf in prior_findings
        ]
        return FeedbackVerificationResult(items=items, cost_usd=cost_usd)


# ---------------------------------------------------------------------------
# BLOCKING finding validator
# ---------------------------------------------------------------------------


async def run_blocking_validator(
    blocking_findings: list[dict[str, Any]],
    diff_text: str,
    settings: Any | None = None,
    *,
    repo_root: str | None = None,
) -> tuple[FindingValidationResult, AgentRunData | None]:
    """Verify each BLOCKING finding against the actual codebase.

    Runs the native Claude Agent SDK session in a spawned subprocess to
    isolate it from the parent's asyncio child-watcher state. After
    parallel reviewer sessions on macOS, the parent's ThreadedChildWatcher
    can misreap grandchild processes, causing subsequent CLI subprocesses
    to hang. A fresh process gets a clean event loop and child watcher.

    Returns (validation_result, agent_run_data) — agent_run_data is None when skipped.

    Args:
        repo_root: Absolute path to the SHA-pinned worktree. Passed as ``cwd``
            to the subprocess worker so its Read/Glob/Grep calls resolve against
            the correct commit. Falls back to ``_REPO_ROOT`` when None.
    """
    if settings is None:
        settings = get_settings()

    if not blocking_findings:
        return FindingValidationResult(items=[], cost_usd=0.0), None

    effective_root = _resolve_repo_root(repo_root, "run_blocking_validator")

    base_prompt = await fetch_prompt("pr-review-blocking-validator")
    system_prompt = f"{base_prompt}\n\n{_context7_system_directive(settings)}"
    findings_json = json.dumps(blocking_findings, indent=2)

    user_message = (
        "## BLOCKING Findings to Validate\n\n"
        f"```json\n{findings_json}\n```\n\n"
        "## PR Diff\n\n"
        f"```diff\n{diff_text}\n```\n\n"
        "For each BLOCKING finding above, read the actual code at the referenced "
        "file:line and determine whether the claim is true. Use Read, Glob, and "
        "Grep to verify. Return your validation results."
    )

    session = await _run_session_isolated(
        model=_SYSTEM_REVIEWER_MODEL,
        system_prompt=system_prompt,
        user_message=user_message,
        anthropic_api_key=settings.ANTHROPIC_API_KEY,
        anthropic_auth_token=settings.ANTHROPIC_AUTH_TOKEN,
        context7_key=getattr(settings, "CONTEXT7_API_KEY", None),
        context7_library_id=getattr(settings, "ARGUS_CONTEXT7_LIBRARY_ID", None),
        timeout_s=getattr(settings, "ARGUS_SESSION_TIMEOUT", _SUBPROCESS_TIMEOUT_S),
        cwd=effective_root,
        label="blocking-validator",
    )

    result = _parse_validation_result(session.result_text, len(blocking_findings), session.cost_usd)

    agent_run = _build_agent_run(
        session=session,
        agent_name="blocking-validator",
        agent_type="blocking_validator",
        result=result,
    )
    return result, agent_run


def _kill_process_group(p: Any) -> None:
    """SIGKILL the worker's whole process group so the spawned claude CLI
    grandchild is reaped too, not just the Python worker PID.

    SAFETY: only kill the group if the worker actually formed its own group
    (pgid == its own pid, i.e. os.setsid ran). If it has not yet (a startup
    race), the pgid would still be the PARENT's group - never kill that;
    fall back to killing just the worker PID.
    """
    pid = p.pid
    if pid is None:
        return
    try:
        pgid = os.getpgid(pid)
    except (ProcessLookupError, OSError):
        return
    try:
        if pgid == pid:  # worker is its own group leader (setsid ran)
            os.killpg(pgid, signal.SIGKILL)
        else:  # startup race: setsid not yet run - do NOT kill parent's group
            p.kill()
    except (ProcessLookupError, OSError):
        pass


def _run_session_in_subprocess(
    *,
    model: str,
    system_prompt: str,
    user_message: str,
    anthropic_api_key: str | None = None,
    anthropic_auth_token: str | None = None,
    context7_key: str | None,
    context7_library_id: str | None = None,
    cwd: str,
    label: str,
    timeout_s: int = _SUBPROCESS_TIMEOUT_S,
) -> SessionResult:
    """Run a Claude Agent SDK session in a spawned process (clean event loop).

    ``timeout_s`` defaults to the module constant for callers (mostly tests)
    that don't have a Settings instance; production runners pass
    ``settings.ARGUS_SESSION_TIMEOUT`` explicitly.
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    result_pipe_recv, result_pipe_send = ctx.Pipe(duplex=False)

    p = ctx.Process(
        target=_validator_worker,
        args=(
            result_pipe_send,
            model,
            system_prompt,
            user_message,
            anthropic_api_key,
            anthropic_auth_token,
            context7_key,
            context7_library_id,
            cwd,
            label,
        ),
    )
    start = datetime.now(timezone.utc)
    p.start()
    result_pipe_send.close()  # parent only reads

    p.join(timeout=timeout_s)

    if p.is_alive():
        _kill_process_group(p)
        p.join()
        logger.warning(
            "Validator subprocess [%s] timed out after %ds; killed process group", label, timeout_s
        )
        return _empty_session_result(
            model, duration_seconds=float(timeout_s), started_at=start, timed_out=True
        )

    if result_pipe_recv.poll():
        return result_pipe_recv.recv()  # type: ignore[no-any-return]

    logger.warning("Validator subprocess exited with code %d, no result", p.exitcode or -1)
    return _empty_session_result(model)


def _validator_worker(
    result_pipe: Any,
    model: str,
    system_prompt: str,
    user_message: str,
    anthropic_api_key: str | None,
    anthropic_auth_token: str | None,
    context7_key: str | None,
    context7_library_id: str | None,
    cwd: str,
    label: str,
) -> None:
    """Worker process: runs _run_claude_session with a fresh event loop."""
    import asyncio as _asyncio
    import contextlib
    import os

    # Become a session/process-group leader so the parent can reap this worker
    # AND its grandchildren (the spawned claude CLI) as one group on timeout.
    with contextlib.suppress(OSError, AttributeError):
        os.setsid()

    # Set whichever credential the caller actually configured -- forcing
    # everything to ANTHROPIC_API_KEY would send a gateway/proxy bearer
    # token as an x-api-key, which not every gateway accepts. Clear the
    # other var explicitly (rather than leaving a stale inherited value)
    # so get_settings() below re-resolves to exactly this one credential.
    if anthropic_auth_token:
        os.environ["ANTHROPIC_AUTH_TOKEN"] = anthropic_auth_token
        os.environ.pop("ANTHROPIC_API_KEY", None)
    else:
        assert anthropic_api_key
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
        os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
    if context7_key:
        os.environ["CONTEXT7_API_KEY"] = context7_key
    # Must accompany CONTEXT7_API_KEY: get_settings() is called fresh in this
    # subprocess, and if ARGUS_CONTEXT7_LIBRARY_ID was only present in the
    # parent's .env file (not os.environ), the attachment guard below would
    # silently skip Context7 even though the baked-in system prompt still
    # instructs the agent to call it.
    if context7_library_id:
        os.environ["ARGUS_CONTEXT7_LIBRARY_ID"] = context7_library_id

    async def _run() -> "SessionResult":
        # Import fresh in the subprocess — no shared state with parent
        from argus.config import get_settings
        from argus.runners import _run_claude_session

        session = await _run_claude_session(
            model=model,
            system_prompt=system_prompt,
            user_message=user_message,
            settings=get_settings(),
            label=label,
            repo_root=cwd,
        )
        return session

    try:
        result = _asyncio.run(_run())
        result_pipe.send(result)
    except Exception as e:
        logging.getLogger(__name__).error("Validator worker failed: %s", e)
        from argus.runners import _empty_session_result as _esr

        result_pipe.send(_esr(model))
    finally:
        result_pipe.close()


def _parse_validation_result(
    raw_text: str,
    expected_count: int,
    cost_usd: float,
) -> FindingValidationResult:
    """Parse the validator's JSON output into FindingValidationResult."""
    if not raw_text.strip():
        # If validator returns nothing, conservatively confirm all findings
        return FindingValidationResult(
            items=[
                FindingValidationItem(
                    index=i,
                    verdict=ValidationVerdict.CONFIRMED,
                    evidence="Validator produced no output — finding kept by default",
                )
                for i in range(expected_count)
            ],
            cost_usd=cost_usd,
        )

    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    json_str = json_match.group(1) if json_match else raw_text.strip()

    try:
        data = json.loads(json_str)
        items: list[FindingValidationItem] = []

        for item in data.get("items", []):
            idx = item.get("index", -1)
            if not isinstance(idx, int) or idx < 0 or idx >= expected_count:
                continue

            verdict_str = item.get("verdict", "CONFIRMED").upper()
            try:
                verdict = ValidationVerdict(verdict_str)
            except ValueError:
                # Unknown verdict — conservatively confirm
                verdict = ValidationVerdict.CONFIRMED

            items.append(
                FindingValidationItem(
                    index=idx,
                    verdict=verdict,
                    evidence=item.get("evidence", ""),
                )
            )

        # Any findings not mentioned are conservatively confirmed
        mentioned = {i.index for i in items}
        for i in range(expected_count):
            if i not in mentioned:
                items.append(
                    FindingValidationItem(
                        index=i,
                        verdict=ValidationVerdict.CONFIRMED,
                        evidence="Not mentioned by validator — finding kept by default",
                    )
                )

        items.sort(key=lambda x: x.index)
        return FindingValidationResult(items=items, cost_usd=cost_usd)

    except (json.JSONDecodeError, TypeError, KeyError):
        logger.warning("Could not parse validator output, conservatively confirming all findings")
        return FindingValidationResult(
            items=[
                FindingValidationItem(
                    index=i,
                    verdict=ValidationVerdict.CONFIRMED,
                    evidence="Validator output could not be parsed — finding kept by default",
                )
                for i in range(expected_count)
            ],
            cost_usd=cost_usd,
        )


# ---------------------------------------------------------------------------
# ClaudeSDKClient session wrapper
# ---------------------------------------------------------------------------


@traceable(name="pr_review.claude_session")
async def _run_claude_session(
    *,
    model: str,
    system_prompt: str,
    user_message: str,
    settings: Any,
    label: str = "",
    repo_root: str | None = None,
) -> SessionResult:
    """Run a single ClaudeSDKClient session. Returns a SessionResult with timing and tool metadata.

    Args:
        repo_root: Working directory for the agent's Read/Glob/Grep calls.
            Should be the path to a SHA-pinned worktree provisioned by
            ``repo_provision.provisioned_worktree``. Falls back to the
            module-level ``_REPO_ROOT`` constant when None. None is reached
            both in tests and in production: graph.py invokes the pipeline
            with worktree_path=None for review requests that carry neither a
            SHA nor a PR number, and that None flows through to every reviewer.
    """
    # Reviewer callers already resolve repo_root, so this is normally a no-op;
    # routing through the same helper keeps the fallback + warning consistent
    # for the SHA-less production path and for any direct caller that passes None.
    effective_root = _resolve_repo_root(repo_root, "_run_claude_session")

    # Context7 MCP: gives reviewers access to curated codebase documentation
    # via the same retrieval layer implementation agents use. Requires both
    # CONTEXT7_API_KEY and ARGUS_CONTEXT7_LIBRARY_ID; either missing degrades
    # gracefully to no Context7 access.
    context7_api_key: str | None = getattr(settings, "CONTEXT7_API_KEY", None)
    context7_library_id: str | None = getattr(settings, "ARGUS_CONTEXT7_LIBRARY_ID", None)
    mcp_servers: dict[str, Any] = {}
    context7_tools: list[str] = []
    if context7_api_key and context7_library_id:
        mcp_servers["context7"] = {
            "type": "http",
            "url": _CONTEXT7_MCP_URL,
            "headers": {"CONTEXT7_API_KEY": context7_api_key},
        }
        context7_tools = [
            "mcp__context7__resolve-library-id",
            "mcp__context7__query-docs",
        ]
    else:
        logger.warning(
            "CONTEXT7_API_KEY / ARGUS_CONTEXT7_LIBRARY_ID not configured — "
            "review agents will not have Context7 access"
        )

    def _stderr_handler(line: str) -> None:
        logger.warning("[cli-stderr:%s] %s", label, line[:300])

    options = ClaudeAgentOptions(
        cwd=effective_root,
        allowed_tools=["Read", "Glob", "Grep"] + context7_tools,
        mcp_servers=mcp_servers,
        permission_mode="default",
        model=model,
        system_prompt=system_prompt,
        max_turns=_MAX_TURNS,
        env=dict([settings.anthropic_credential]),
        stderr=_stderr_handler,
    )

    started_at = datetime.now(timezone.utc)
    async with ClaudeSDKClient(options=options) as client:
        await client.query(user_message)
        result_text = ""
        cost_usd = 0.0
        tool_calls: list[str] = []
        async for message in client.receive_response():
            if isinstance(message, TaskStartedMessage):
                logger.info("Agent session started: %s model=%s", label or "unlabeled", model)
            elif isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        tool_calls.append(f"{block.name}({json.dumps(block.input)[:100]})")
                    elif isinstance(block, ThinkingBlock):
                        logger.debug("Agent thinking: %s...", (block.thinking or "")[:200])
                    elif isinstance(block, TextBlock):
                        logger.debug("Agent text: %s...", (block.text or "")[:200])
            elif isinstance(message, ResultMessage):
                result_text = message.result or ""
                cost_usd = getattr(message, "total_cost_usd", 0.0) or 0.0
    finished_at = datetime.now(timezone.utc)

    tool_names = [tc.split("(")[0] for tc in tool_calls]
    unique_tool_names = sorted(set(tool_names))
    context7_calls = [t for t in tool_names if "context7" in t.lower() or "mcp" in t.lower()]

    if tool_calls:
        logger.info(
            "Agent done [%s]: %d tool calls (tools: %s), cost=$%.4f",
            label or "unlabeled",
            len(tool_calls),
            ", ".join(unique_tool_names),
            cost_usd,
        )
        if context7_calls:
            logger.info("Context7 calls: %s", context7_calls)
        elif context7_api_key and context7_library_id:
            logger.warning("Context7 was available but agent made 0 Context7 calls")

    return SessionResult(
        result_text=result_text,
        cost_usd=cost_usd,
        duration_seconds=(finished_at - started_at).total_seconds(),
        started_at=started_at,
        finished_at=finished_at,
        tool_call_count=len(tool_calls),
        tool_names=unique_tool_names,
        context7_call_count=len(context7_calls),
        model=model,
    )
