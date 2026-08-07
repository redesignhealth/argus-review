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
from typing import Any, Literal

from argus.llm.models import ALIAS_MAP, CLAUDE_DEFAULT, CLAUDE_OPUS
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TaskStartedMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from langsmith import traceable
from langsmith.run_helpers import LangSmithExtra, get_current_run_tree

from argus.config import DEFAULT_ARGUS_SESSION_TIMEOUT_S, get_settings
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
# Opus, not frontier (Fable): evals showed no measurable quality gain from
# frontier on this stage, at ~2x the per-token cost.
_CROSS_CUTTING_MODEL = CLAUDE_OPUS
# Whether the system reviewer's current model (following any
# --specialist-model/ARGUS_SPECIALIST_MODEL override) still matches
# ALIAS_MAP["claude-default"] -- the fixed pin the 1M-context beta (see
# _run_claude_session) was empirically verified against. Computed once here
# from the two constants above, not re-derived per-call from `model` string
# equality -- see _run_claude_session's is_system_reviewer_role docstring
# for why a string-equality gate can't be made sound once overrides exist.
#
# Caveat shared with argus/graph.py's _TEMPERATURE_UNSUPPORTED_MODELS
# comment: the empirical verification below was run against whatever
# ALIAS_MAP["claude-default"] resolved to on 2026-07-31 (claude-sonnet-5 at
# the time), NOT specifically against claude-sonnet-4-6 (the pin as of this
# comment, after this same diff's default bump). Gating on the alias means
# the beta keeps applying across future pin bumps automatically, without a
# fresh re-verification each time -- the alternative (pinning this gate to
# the literal claude-sonnet-5 string instead) would have silently withheld
# the beta from the system reviewer's new default entirely, reinstating the
# TECH-4734 autocompact-thrashing problem for the common no-override case.
# Tracking the alias was the deliberate tradeoff; re-verify the beta
# empirically against the current pin whenever ALIAS_MAP["claude-default"]
# moves, and correct this comment if a future pin ever fails the probe
# described below.
_SYSTEM_REVIEWER_UNOVERRIDDEN = _SYSTEM_REVIEWER_MODEL == ALIAS_MAP["claude-default"]

# Logged once at import time, not per-session (an earlier per-call version
# of this log was flagged twice in opposite directions: too quiet at debug,
# too loud at info-per-call across 5 reviewer-role call sites x N rounds).
# The withheld case is the one with a real cost/quality consequence
# (TECH-4734 autocompact thrashing), so it's WARNING; the enabled case is
# DEBUG since it's simply confirming the common, unoverridden default.
if not _SYSTEM_REVIEWER_UNOVERRIDDEN:
    logger.warning(
        "1M-context beta withheld for this process: system reviewer model %r "
        "diverges from the validated pin %r (ARGUS_SPECIALIST_MODEL override "
        "active) -- autocompact may thrash on long reviews under this override.",
        _SYSTEM_REVIEWER_MODEL,
        ALIAS_MAP["claude-default"],
    )
else:
    logger.debug(
        "1M-context beta enabled for the system reviewer role (model=%r, unoverridden).",
        _SYSTEM_REVIEWER_MODEL,
    )

# Same one-time-at-import treatment for the other silent-cost-shift this
# override mechanism can cause: --frontier-model/ARGUS_FRONTIER_MODEL
# repoints both CLAUDE_FRONTIER and CLAUDE_OPUS (see argus/llm/models.py),
# so a frontier override picked for planning/coverage purposes also moves
# the cross-cutting reviewer off its cheaper Opus default with no other
# runtime signal that happened.
if _CROSS_CUTTING_MODEL != ALIAS_MAP["claude-opus"]:
    logger.warning(
        "Cross-cutting reviewer moved off its default model %r onto %r due to "
        "ARGUS_FRONTIER_MODEL -- this env var/--frontier-model repoints both "
        "the frontier tier and the cross-cutting model together, so a "
        "frontier override for planning purposes also moves cross-cutting "
        "off its cheaper Opus default.",
        ALIAS_MAP["claude-opus"],
        _CROSS_CUTTING_MODEL,
    )

_MAX_TURNS = 30
# Fallback repo root for ClaudeSDKClient cwd — used when no SHA-pinned
# worktree has been provisioned (e.g. local dev runs, tests, subprocess
# worker). In production, callers pass an explicit repo_root provisioned
# by repo_provision.provisioned_worktree().
_REPO_ROOT = str(Path(__file__).resolve().parents[4])
# Fallback used when no Settings instance is available (e.g. direct unit
# tests calling _run_session_in_subprocess without a settings-derived
# timeout_s). Production call sites pass settings.ARGUS_SESSION_TIMEOUT.
# Imports the same constant Settings.ARGUS_SESSION_TIMEOUT defaults to, so
# the two can never drift out of sync.
_SUBPROCESS_TIMEOUT_S = DEFAULT_ARGUS_SESSION_TIMEOUT_S


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

# Shared directive steering reviewers away from full-file Reads on large
# targets. Root cause of TECH-4643: specialists tracing execution paths into
# a multi-thousand-line file would Read it whole, exhaust a large fraction of
# their context budget in 1-2 tool calls, and then "autocompact thrash" for
# the rest of the session — burning further turns compacting instead of
# reviewing, with the specialist's own findings effectively lost. That
# failure mode was silent: the session still exits normally (not a timeout),
# so it doesn't set failure_reason and isn't surfaced as degraded coverage.
_LARGE_FILE_READ_DIRECTIVE = """\

## Reading Large Files

Before reading any file, check its size (e.g. via a targeted Grep or a \
quick line count). For a file beyond a few hundred lines, do not Read it in \
one call: Grep first for the specific symbol, function, or pattern you need, \
then use Read's `offset`/`limit` to pull just the surrounding lines. A full \
Read of a large file can exhaust a large share of your context budget in a \
single tool call, leaving too little to actually analyze what you read.
"""

# Context7 MCP configuration. The library ID is deployment-specific (which
# codebase's curated docs to query) and comes from ARGUS_CONTEXT7_LIBRARY_ID;
# when unset, Context7 attachment is skipped entirely (same graceful
# degradation as a missing CONTEXT7_API_KEY).
_CONTEXT7_MCP_URL = "https://mcp.context7.com/mcp"

# Context-usage threshold for a WARNING-level log (TECH-4734/TECH-4732).
# ~170k is the pre-1M-beta autocompact-thrashing floor observed in production
# (sessions compacting at ~177-189k preTokens); crossing it is still a useful
# "this session is unusually large" signal now that the 1M-context beta has
# removed autocompact as an implicit ceiling, not just a historical artifact.
_CONTEXT_WARNING_THRESHOLD_TOKENS = 170_000


def _tool_result_size(content: str | list[dict[str, Any]] | None) -> int:
    """Return a size-in-chars estimate for a ToolResultBlock's content.

    ``content`` can be a plain string or a list of content-block dicts
    (``claude_agent_sdk.ToolResultBlock.content: str | list[dict] | None``).
    ``len(str(content))`` on the list form measures a Python repr, not the
    actual content size, and its str() conversion briefly materializes any
    embedded text in memory -- summing per-block text lengths instead avoids
    both problems.
    """
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    total = 0
    for part in content:
        text = part.get("text") if isinstance(part, dict) else None
        total += len(text) if isinstance(text, str) else len(str(part))
    return total


def _context_ledger_path(pid: int | None = None) -> str:
    """Return the per-process local context-usage ledger path (TECH-4734 phase 2).

    Sibling of the ``/tmp/argus-checkpoint-<pid>.db`` LangGraph checkpoint
    (see graph.py). Written unconditionally on every message, regardless of
    whether LangSmith tracing is on or off -- NOT gated on tracing state,
    despite the "fallback for when tracing is off" framing an earlier
    version of this comment used. It's useful either way: even with tracing
    on, this is a plain local file a developer can `tail`/`grep` without a
    LangSmith UI round-trip, and it costs nothing extra to keep writing it.
    Each reviewer session runs in its own spawned subprocess (see
    _run_session_in_subprocess), so os.getpid() here is that subprocess's
    pid, not the parent graph process's -- one ledger file per session.
    Overridable for tests, same pattern as ARGUS_SQLITE_CHECKPOINT_PATH; see
    _validator_worker's atexit cleanup for when the default per-pid path is
    removed.

    ``pid``: defaults to the caller's own pid (os.getpid()), which is the
    only correct choice when this runs inside the ledger-writing subprocess
    itself. The parent process (_run_session_in_subprocess) instead passes
    the *child's* pid explicitly, since it needs this same template to clean
    up a killed child's ledger without inheriting the child's pid as its own.
    """
    return os.environ.get(
        "ARGUS_CONTEXT_LEDGER_PATH",
        f"/tmp/argus-context-ledger-{pid if pid is not None else os.getpid()}.jsonl",  # nosec B108
    )


def _try_unlink_ledger(pid: int) -> None:
    """Best-effort removal of another process's context-usage ledger.

    Used by the parent (_run_session_in_subprocess) to clean up a killed
    child's ledger on paths where the child's own atexit-registered
    _cleanup_ledger_file never ran (SIGKILL, crash) -- mirrors that
    function's FileNotFoundError/other-OSError split so both cleanup sites
    log consistently. A no-op (silently) when ARGUS_CONTEXT_LEDGER_PATH
    pins an operator-chosen path, since that path is presumably meant to
    persist for inspection.
    """
    if "ARGUS_CONTEXT_LEDGER_PATH" in os.environ:
        return
    ledger_path = _context_ledger_path(pid)
    try:
        os.unlink(ledger_path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("Could not unlink context-usage ledger %s: %s", ledger_path, exc)


def _append_context_ledger(record: dict[str, Any]) -> None:
    """Best-effort append of one context-usage record to the local ledger.

    Sizes/counts only (ts, label, msg_index, token counts) -- never prompt
    or response content. IOError must never break a review: a full /tmp, a
    permissions issue, or a concurrent-write race is a diagnostics-only
    degradation, not a review failure.
    """
    try:
        with open(_context_ledger_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        logger.debug("Failed to append to context-usage ledger", exc_info=True)


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
    failure_reason: Literal["timeout", "worker_crashed"] | None = None


def _empty_session_result(
    model: str | None = None,
    duration_seconds: float = 0.0,
    started_at: datetime | None = None,
    *,
    failure_reason: Literal["timeout", "worker_crashed"],
) -> SessionResult:
    """Default SessionResult for subprocess timeout / worker failure.

    When ``started_at`` is provided (timeout path), ``finished_at`` is computed
    as ``started_at + timedelta(seconds=duration_seconds)`` so the timestamps
    are internally consistent with the reported duration. When omitted, both
    timestamps default to now and duration is 0.

    ``failure_reason`` is keyword-only and required (no default): every empty
    result comes from a real failure, and a caller that forgets to pass it is
    a bug, not something that should silently resolve to a guess.
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
        failure_reason=failure_reason,
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
    reviewer sessions do not queue behind the default asyncio executor.

    TECH-4734 phase 2: the reviewer session itself runs in a spawned
    subprocess (see _run_session_in_subprocess) with a fresh interpreter and
    no shared contextvars, so the @traceable span _run_claude_session opens
    there has no way to see the LangGraph node's run tree that is active
    here, in the parent's asyncio task -- without help it starts a new root
    trace instead of nesting under the graph-level trace. get_current_run_tree()
    reads the ambient contextvar directly (no LangGraph/LangChain call
    needed) and returns None when tracing is off, which serializes to no
    headers and is a no-op for the child.

    Empirically verified (2026-08-07) against both this repo's pinned
    versions (langsmith==0.9.8, langgraph==1.2.8, per uv.lock) and a newer
    langsmith==0.10.16 that this contextvar IS populated here even though
    this function -- and the LangGraph node that calls it -- are plain async
    functions, not @traceable-decorated: a minimal repro (a bare async node
    added to a compiled StateGraph, invoked with LANGCHAIN_TRACING_V2=true,
    calling get_current_run_tree() directly inside the node body) returned a
    real RunTree, not None, on both pins. LangGraph's own LangSmith
    integration sets this contextvar per-node automatically; a node/helper
    does not need its own @traceable wrapper to observe it. This directly
    contradicts an earlier, plausible-sounding claim (raised and then
    dismissed with this evidence in Argus's review of this PR) that
    LangGraph only propagates via RunnableConfig/callback handlers and never
    touches LangSmith's own contextvar -- re-verify if this package's
    langsmith/langgraph pins ever change materially, since this behavior is
    an integration detail of those two packages, not something this
    codebase controls.
    """
    parent_run = get_current_run_tree()
    langsmith_parent_headers = parent_run.to_headers() if parent_run is not None else None
    logger.debug(
        "LangSmith trace propagation: parent_run_id=%s has_headers=%s",
        parent_run.id if parent_run is not None else None,
        bool(langsmith_parent_headers),
    )
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _get_reviewer_executor(),
        functools.partial(
            _run_session_in_subprocess,
            langsmith_parent_headers=langsmith_parent_headers,
            **kwargs,
        ),
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
        failure_reason=session.failure_reason,
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
        f"{_LARGE_FILE_READ_DIRECTIVE}"
    )

    session = await _run_session_isolated(
        model=_SYSTEM_REVIEWER_MODEL,
        is_system_reviewer_role=True,
        system_prompt=system_prompt,
        user_message=user_message,
        anthropic_api_key=settings.ANTHROPIC_API_KEY,
        anthropic_auth_token=settings.ANTHROPIC_AUTH_TOKEN,
        context7_key=getattr(settings, "CONTEXT7_API_KEY", None),
        context7_library_id=getattr(settings, "ARGUS_CONTEXT7_LIBRARY_ID", None),
        context7_base_url=getattr(settings, "ARGUS_CONTEXT7_BASE_URL", None),
        timeout_s=getattr(settings, "ARGUS_SESSION_TIMEOUT", _SUBPROCESS_TIMEOUT_S),
        cwd=effective_root,
        label=f"system:{group.name}",
    )
    result = _parse_review_result(session.result_text, group.name)
    result.cost_usd = session.cost_usd
    result.failure_reason = session.failure_reason
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
        f"{_LARGE_FILE_READ_DIRECTIVE}"
    )

    session = await _run_session_isolated(
        model=_SYSTEM_REVIEWER_MODEL,
        is_system_reviewer_role=True,
        system_prompt=system_prompt,
        user_message=user_message,
        anthropic_api_key=settings.ANTHROPIC_API_KEY,
        anthropic_auth_token=settings.ANTHROPIC_AUTH_TOKEN,
        context7_key=getattr(settings, "CONTEXT7_API_KEY", None),
        context7_library_id=getattr(settings, "ARGUS_CONTEXT7_LIBRARY_ID", None),
        context7_base_url=getattr(settings, "ARGUS_CONTEXT7_BASE_URL", None),
        timeout_s=getattr(settings, "ARGUS_SESSION_TIMEOUT", _SUBPROCESS_TIMEOUT_S),
        cwd=effective_root,
        label=f"specialist:{group.name}::{specialist}",
    )
    result = _parse_review_result(session.result_text, f"{group.name}::{specialist}")
    result.cost_usd = session.cost_usd
    result.failure_reason = session.failure_reason
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
        f"{_LARGE_FILE_READ_DIRECTIVE}"
    )

    session = await _run_session_isolated(
        model=_CROSS_CUTTING_MODEL,
        system_prompt=system_prompt,
        user_message=user_message,
        anthropic_api_key=settings.ANTHROPIC_API_KEY,
        anthropic_auth_token=settings.ANTHROPIC_AUTH_TOKEN,
        context7_key=getattr(settings, "CONTEXT7_API_KEY", None),
        context7_library_id=getattr(settings, "ARGUS_CONTEXT7_LIBRARY_ID", None),
        context7_base_url=getattr(settings, "ARGUS_CONTEXT7_BASE_URL", None),
        timeout_s=getattr(settings, "ARGUS_SESSION_TIMEOUT", _SUBPROCESS_TIMEOUT_S),
        cwd=effective_root,
        label="cross-cutting",
    )
    result = _parse_review_result(session.result_text, "cross-cutting")
    result.cost_usd = session.cost_usd
    result.failure_reason = session.failure_reason
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
        f"{_LARGE_FILE_READ_DIRECTIVE}"
    )

    session = await _run_session_isolated(
        model=_SYSTEM_REVIEWER_MODEL,
        is_system_reviewer_role=True,
        system_prompt=system_prompt,
        user_message=user_message,
        anthropic_api_key=settings.ANTHROPIC_API_KEY,
        anthropic_auth_token=settings.ANTHROPIC_AUTH_TOKEN,
        context7_key=getattr(settings, "CONTEXT7_API_KEY", None),
        context7_library_id=getattr(settings, "ARGUS_CONTEXT7_LIBRARY_ID", None),
        context7_base_url=getattr(settings, "ARGUS_CONTEXT7_BASE_URL", None),
        timeout_s=getattr(settings, "ARGUS_SESSION_TIMEOUT", _SUBPROCESS_TIMEOUT_S),
        cwd=effective_root,
        label="tests-and-docs",
    )
    result = _parse_review_result(session.result_text, "tests-and-docs")
    result.cost_usd = session.cost_usd
    result.failure_reason = session.failure_reason
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
        is_system_reviewer_role=True,
        system_prompt=system_prompt,
        user_message=user_message,
        anthropic_api_key=settings.ANTHROPIC_API_KEY,
        anthropic_auth_token=settings.ANTHROPIC_AUTH_TOKEN,
        context7_key=getattr(settings, "CONTEXT7_API_KEY", None),
        context7_library_id=getattr(settings, "ARGUS_CONTEXT7_LIBRARY_ID", None),
        context7_base_url=getattr(settings, "ARGUS_CONTEXT7_BASE_URL", None),
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
        is_system_reviewer_role=True,
        system_prompt=system_prompt,
        user_message=user_message,
        anthropic_api_key=settings.ANTHROPIC_API_KEY,
        anthropic_auth_token=settings.ANTHROPIC_AUTH_TOKEN,
        context7_key=getattr(settings, "CONTEXT7_API_KEY", None),
        context7_library_id=getattr(settings, "ARGUS_CONTEXT7_LIBRARY_ID", None),
        context7_base_url=getattr(settings, "ARGUS_CONTEXT7_BASE_URL", None),
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
    context7_base_url: str | None = None,
    cwd: str,
    label: str,
    timeout_s: int = _SUBPROCESS_TIMEOUT_S,
    is_system_reviewer_role: bool = False,
    langsmith_parent_headers: dict[str, str] | None = None,
) -> SessionResult:
    """Run a Claude Agent SDK session in a spawned process (clean event loop).

    ``timeout_s`` defaults to the module constant for callers (mostly tests)
    that don't have a Settings instance; production runners pass
    ``settings.ARGUS_SESSION_TIMEOUT`` explicitly.

    ``is_system_reviewer_role``: forwarded to ``_run_claude_session`` --
    see that function's docstring for why this must be an explicit,
    caller-declared role flag rather than inferred from ``model``.

    ``langsmith_parent_headers`` (TECH-4734 phase 2) carries the serialized
    parent LangSmith run tree (via RunTree.to_headers()) across the process
    boundary -- plain str/str dict, safe for spawn's pickling of Process
    args -- so the @traceable span opened inside the subprocess nests under
    the caller's trace instead of starting an orphaned root trace. None when
    tracing is off or there was no ambient run tree.
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
            context7_base_url,
            cwd,
            label,
            is_system_reviewer_role,
            langsmith_parent_headers,
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
        # SIGKILL bypasses atexit entirely, so _validator_worker's own ledger
        # cleanup never runs on this path -- unlink here in the parent using
        # the killed child's pid instead.
        if p.pid is not None:
            _try_unlink_ledger(p.pid)
        return _empty_session_result(
            model, duration_seconds=float(timeout_s), started_at=start, failure_reason="timeout"
        )

    if result_pipe_recv.poll():
        return result_pipe_recv.recv()  # type: ignore[no-any-return]

    logger.warning(
        "Validator subprocess [%s] exited with code %d, no result", label, p.exitcode or -1
    )
    # Same atexit-bypass gap as the timeout branch above: a crashed worker
    # (OOM kill, SIGSEGV, external SIGKILL) never runs its own atexit
    # cleanup either, so the ledger leaks unless cleaned up here too.
    if p.pid is not None:
        _try_unlink_ledger(p.pid)
    return _empty_session_result(model, failure_reason="worker_crashed")


def _validator_worker(
    result_pipe: Any,
    model: str,
    system_prompt: str,
    user_message: str,
    anthropic_api_key: str | None,
    anthropic_auth_token: str | None,
    context7_key: str | None,
    context7_library_id: str | None,
    context7_base_url: str | None,
    cwd: str,
    label: str,
    is_system_reviewer_role: bool = False,
    langsmith_parent_headers: dict[str, str] | None = None,
) -> None:
    """Worker process: runs _run_claude_session with a fresh event loop."""
    import asyncio as _asyncio
    import atexit
    import contextlib
    import os

    # Become a session/process-group leader so the parent can reap this worker
    # AND its grandchildren (the spawned claude CLI) as one group on timeout.
    with contextlib.suppress(OSError, AttributeError):
        os.setsid()

    # Clean up this subprocess's context-usage ledger on exit -- mirrors
    # graph.py's _cleanup_sqlite_files() gating (skip cleanup when the path
    # is operator-pinned via the env var, same convention as
    # ARGUS_SQLITE_CHECKPOINT_PATH): an operator who explicitly set
    # ARGUS_CONTEXT_LEDGER_PATH presumably wants the file to persist for
    # inspection, so only the default per-pid /tmp path is auto-removed.
    # Registered here (not in _run_claude_session) since _context_ledger_path()
    # resolves os.getpid() at call time, and this worker's pid is stable for
    # its whole lifetime -- computing the path once, at registration, avoids
    # any risk of the cleanup unlinking a different path than what was
    # actually written to if something in between changed os.environ.
    if "ARGUS_CONTEXT_LEDGER_PATH" not in os.environ:
        _ledger_path_to_clean = _context_ledger_path()

        def _cleanup_ledger_file() -> None:
            try:
                os.unlink(_ledger_path_to_clean)
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning(
                    "Could not unlink context-usage ledger %s: %s", _ledger_path_to_clean, exc
                )

        atexit.register(_cleanup_ledger_file)

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
    # Same reasoning as ARGUS_CONTEXT7_LIBRARY_ID above: must be forwarded
    # explicitly so a fresh get_settings() in this subprocess sees it even
    # when the parent only had it via .env, not os.environ.
    if context7_base_url:
        os.environ["ARGUS_CONTEXT7_BASE_URL"] = context7_base_url

    async def _run() -> "SessionResult":
        # Import fresh in the subprocess — no shared state with parent
        from argus.config import get_settings
        from argus.runners import _run_claude_session

        # TECH-4734 phase 2: re-attach the parent LangSmith run tree (if any)
        # so the @traceable span below nests under the caller's trace instead
        # of opening an orphaned root trace -- this subprocess has no shared
        # contextvars with the parent. langsmith_extra["parent"] accepts a
        # headers mapping directly (RunTree.from_headers under the hood); a
        # None value is a no-op, same as omitting langsmith_extra entirely.
        langsmith_extra: LangSmithExtra | None = (
            {"parent": langsmith_parent_headers} if langsmith_parent_headers else None
        )
        session = await _run_claude_session(
            model=model,
            system_prompt=system_prompt,
            user_message=user_message,
            settings=get_settings(),
            label=label,
            repo_root=cwd,
            is_system_reviewer_role=is_system_reviewer_role,
            langsmith_extra=langsmith_extra,
        )
        return session

    try:
        result = _asyncio.run(_run())
        result_pipe.send(result)
    except Exception as e:
        logging.getLogger(__name__).error("Validator worker [%s] failed: %s", label, e)
        from argus.runners import _empty_session_result as _esr

        result_pipe.send(_esr(model, failure_reason="worker_crashed"))
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
    is_system_reviewer_role: bool = False,
    langsmith_extra: LangSmithExtra | None = None,
) -> SessionResult:
    """Run a single ClaudeSDKClient session. Returns a SessionResult with timing and tool metadata.

    Args:
        langsmith_extra: NOT read by this function's body -- ``@traceable``'s
            own wrapper intercepts this reserved kwarg name before the call
            reaches here (the LangSmith SDK contract; see ``_validator_worker``,
            which builds ``{"parent": langsmith_parent_headers}`` and passes it
            here to reparent this span under the caller's trace). Declared
            explicitly, rather than left implicit, so a future refactor of
            this function's signature (e.g. to ``**kwargs``) or removal of
            ``@traceable`` fails loudly at the call site instead of a
            subprocess silently swallowing a ``TypeError`` as
            ``worker_crashed`` and re-orphaning traces with no signal.
        repo_root: Working directory for the agent's Read/Glob/Grep calls.
            Should be the path to a SHA-pinned worktree provisioned by
            ``repo_provision.provisioned_worktree``. Falls back to the
            module-level ``_REPO_ROOT`` constant when None. None is reached
            both in tests and in production: graph.py invokes the pipeline
            with worktree_path=None for review requests that carry neither a
            SHA nor a PR number, and that None flows through to every reviewer.
        is_system_reviewer_role: Whether this call is one of the system-
            reviewer-role sessions (system reviewer, specialist reviewer,
            coverage/dismissal-matching validators -- see call sites passing
            ``model=_SYSTEM_REVIEWER_MODEL``), as opposed to the
            cross-cutting/opus session. Declared explicitly by the caller
            rather than inferred by comparing ``model`` against
            ``_SYSTEM_REVIEWER_MODEL`` inside this function: once
            ARGUS_SPECIALIST_MODEL/ARGUS_FRONTIER_MODEL can repoint either
            role's model, two DIFFERENT roles' resolved model strings can
            collide (e.g. --frontier-model set to the same value as the
            unoverridden system-reviewer pin), and a string-equality check
            can no longer tell them apart -- exactly the two round-3 Argus
            findings this parameter exists to fix. Only the caller genuinely
            knows its own role; only the caller can pass a decision that a
            model-value collision cannot fool.
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
    context7_base_url: str = getattr(settings, "ARGUS_CONTEXT7_BASE_URL", None) or _CONTEXT7_MCP_URL
    mcp_servers: dict[str, Any] = {}
    context7_tools: list[str] = []
    if context7_api_key and context7_library_id:
        mcp_servers["context7"] = {
            "type": "http",
            "url": context7_base_url,
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

    # TECH-4732 / TECH-4643: strict_mcp_config removes the disease, the 1M
    # beta raises the ceiling for what's left. Deliberately paired.
    #
    # Disease: reviewer sessions carry a huge fixed prefix that autocompact
    # can never shrink (it only compresses *conversation*), so sessions
    # compacted 2-3x per run for zero benefit ("autocompact thrashing",
    # TECH-4643) and lost findings along the way. Root cause, verified live
    # through Argus's real code path via a request-capture sink 2026-07-31:
    # the CLI inherits every MCP server from the *operator's own* Claude Code
    # config -- `allowed_tools` above only gates invocation permission, not
    # which schemas get sent on the wire. The captured request carried 334
    # tool schemas (~419KB, ~105k tokens). `strict_mcp_config=True` makes the
    # CLI ignore that inherited config (project .mcp.json, user/global
    # settings, plugin-provided servers) and send only the 25 built-in tool
    # schemas (~82KB, ~20k tokens) plus whatever is passed explicitly via the
    # `mcp_servers=` option below -- i.e. Argus's own Context7 attachment,
    # when configured, is unaffected. See the strict_mcp_config docstring in
    # claude_agent_sdk/types.py (~line 1621). Net: total request dropped from
    # ~121k to ~36k tokens (~85k/request saved), taking the ~180k
    # session-start context that sat right at the 200k autocompact threshold
    # down to ~95k.
    #
    # Remaining ceiling: reviewer sessions carry a ~155k-token fixed prefix
    # measured before this fix (334 tools/~105k tokens, TECH-4734
    # investigation 2026-07-31) -- see above for why strict_mcp_config cuts
    # that down. For reviews that are still genuinely large even at the
    # reduced ~95k floor, the default 200k window remains tight. The
    # Anthropic 1M-context beta moves that ceiling far above any fixed
    # prefix, so autocompact stops firing on it.
    #
    # Verified end-to-end 2026-07-31 through the deployed prod rh-mcp Argus
    # proxy (the path the argus-review-loop skill uses), with the same
    # ANTHROPIC_AUTH_TOKEN bearer-credential mode this function actually runs
    # with in production (not ANTHROPIC_API_KEY): an ~860k-token probe request
    # WITHOUT this beta was rejected with "Prompt is too long"; the identical
    # probe WITH betas=["context-1m-2025-08-07"] was accepted and metered
    # ctx=860,253 tokens. This is an empirical, not just structural,
    # confirmation that the proxy forwards the anthropic-beta header for
    # bearer-token auth specifically -- a live probe, not a read of the
    # proxy's own header-allowlist (which lives in a different repo, rh-mcp's
    # main.py, and could drift independently of this comment).
    #
    # Only applied for _SYSTEM_REVIEWER_MODEL (sonnet), not
    # _CROSS_CUTTING_MODEL (opus): sonnet reviewer sessions are the ones
    # observed thrashing at the ~155-180k fixed-prefix floor (TECH-4734
    # investigation 2026-07-31); cross-cutting/opus sessions have never come
    # close to the 200k ceiling in any production round measured so far
    # (peaked 57k-91k across every real review this fix has been validated
    # against). Applying an unverified-for-opus beta there would add
    # long-context-premium billing risk for zero observed benefit -- verify
    # empirically before ever widening this beyond sonnet.
    #
    # Cost note: tokens beyond 200k bill at the long-context premium; that's
    # acceptable against the status quo of thrashing sessions that cost
    # $1.5-2.2 each and produce zero findings.
    #
    # Gate is role-based (the caller-declared `is_system_reviewer_role`
    # parameter), NOT a `model == _SYSTEM_REVIEWER_MODEL`/
    # `model == _CROSS_CUTTING_MODEL` string comparison. Two earlier
    # string-equality attempts both broke once ARGUS_SPECIALIST_MODEL/
    # ARGUS_FRONTIER_MODEL (argus/llm/models.py) could repoint either
    # role's model at runtime:
    #   - `model == _SYSTEM_REVIEWER_MODEL` alone (the original condition)
    #     would attach this beta to WHATEVER model ARGUS_SPECIALIST_MODEL
    #     currently points at, extending an unverified-for-that-model
    #     billing/compatibility risk far past the one pin this was
    #     validated against.
    #   - `model == ALIAS_MAP["claude-default"]` alone (a later, over-
    #     corrected fix) decoupled the gate entirely from the override: the
    #     system reviewer silently LOST this beta under any
    #     --specialist-model override (reinstating the TECH-4734
    #     autocompact-thrashing problem this beta exists to fix), and worse,
    #     `model == _SYSTEM_REVIEWER_MODEL and model == ALIAS_MAP["claude-default"]`
    #     (the very next fix attempt) could STILL mis-fire for cross-cutting:
    #     if --frontier-model happened to be set to the same value as the
    #     unoverridden system-reviewer pin, `_CROSS_CUTTING_MODEL` collided
    #     with `_SYSTEM_REVIEWER_MODEL` and the cross-cutting call received
    #     the beta anyway, despite the opus-specific risk called out two
    #     paragraphs up. Every one of these was a variant of inferring
    #     "which role is this call" from data (a model string) that
    #     overrides can make ambiguous. Only the caller genuinely knows its
    #     own role; passing that decision down explicitly is the only fix
    #     that can't be defeated by a future override collision.
    _attach_1m_context_beta = is_system_reviewer_role and _SYSTEM_REVIEWER_UNOVERRIDDEN
    # Logged once at module import time (see _SYSTEM_REVIEWER_UNOVERRIDDEN's
    # definition above), not per-call here: this decision is fixed for the
    # whole process, so a per-session log would just repeat the same fact
    # once per reviewer session (5 call sites x N rounds) for zero
    # additional signal -- high-volume noise an earlier version of this
    # comment/log pair was flagged for twice, in opposite directions (too
    # quiet at debug, too loud at info-per-call).
    betas: list[Literal["context-1m-2025-08-07"]] = (
        ["context-1m-2025-08-07"] if _attach_1m_context_beta else []
    )
    options = ClaudeAgentOptions(
        cwd=effective_root,
        allowed_tools=["Read", "Glob", "Grep"] + context7_tools,
        mcp_servers=mcp_servers,
        strict_mcp_config=True,
        permission_mode="default",
        model=model,
        system_prompt=system_prompt,
        max_turns=_MAX_TURNS,
        env=dict([settings.anthropic_credential]),
        stderr=_stderr_handler,
        betas=betas,
    )

    started_at = datetime.now(timezone.utc)
    # TECH-4734: context-usage instrumentation.
    #
    # Specialist reviewer sessions have been observed autocompacting 2-3x per
    # session at ~177-189k preTokens, then refilling within ~3 turns
    # ("autocompact thrashing"). Local transcript forensics (tool-result
    # sizes, thinking-block sizes) cannot account for that context size --
    # one session had ~1KB of visible tool-result content total yet still
    # compacted 3 times at ~181k. That means ~170k/request of context is
    # invisible to any artifact we currently persist, and there is no way to
    # tell which request or which component (system prompt, a specific tool
    # result, cache churn) is responsible.
    #
    # The API reports true usage (input_tokens, cache_read_input_tokens,
    # cache_creation_input_tokens, output_tokens) on every response, and the
    # message loop below already iterates every message -- it just never
    # looked at usage. This block adds observability only: it logs sizes and
    # counts (never prompt/response content, which may include untrusted
    # diff content or credentials) and does not change what tools are
    # allowed, what options are set, or how the session runs.
    peak_context_tokens: int | None = None
    tool_use_names: dict[str, str] = {}  # tool_use_id -> tool name, for correlating results
    # TECH-4734 phase 2: sizes/counts only, mirrored into the LangSmith run's
    # metadata (below) and a local JSONL ledger for when tracing is off.
    # Never content -- see module docstring on why (untrusted diff/PR data).
    usage_records: list[dict[str, Any]] = []
    tool_result_sizes: list[dict[str, Any]] = []
    async with ClaudeSDKClient(options=options) as client:
        await client.query(user_message)
        result_text = ""
        cost_usd = 0.0
        tool_calls: list[str] = []
        message_index = 0
        async for message in client.receive_response():
            if isinstance(message, TaskStartedMessage):
                logger.info("Agent session started: %s model=%s", label or "unlabeled", model)
            elif isinstance(message, AssistantMessage):
                message_index += 1
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        tool_calls.append(f"{block.name}({json.dumps(block.input)[:100]})")
                        tool_use_names[block.id] = block.name
                    elif isinstance(block, ThinkingBlock):
                        logger.debug("Agent thinking: %s...", (block.thinking or "")[:200])
                    elif isinstance(block, TextBlock):
                        logger.debug("Agent text: %s...", (block.text or "")[:200])
                usage = getattr(message, "usage", None) or {}
                input_tokens = usage.get("input_tokens")
                cache_read_tokens = usage.get("cache_read_input_tokens")
                cache_creation_tokens = usage.get("cache_creation_input_tokens")
                output_tokens = usage.get("output_tokens")
                context_total: int | None = None
                if isinstance(input_tokens, int):
                    context_total = (
                        input_tokens
                        + (cache_read_tokens if isinstance(cache_read_tokens, int) else 0)
                        + (cache_creation_tokens if isinstance(cache_creation_tokens, int) else 0)
                    )
                    peak_context_tokens = max(peak_context_tokens or 0, context_total)
                logger.info(
                    "Agent usage [%s] msg=%d input=%s cache_read=%s cache_creation=%s "
                    "output=%s context_total=%s",
                    label or "unlabeled",
                    message_index,
                    input_tokens,
                    cache_read_tokens,
                    cache_creation_tokens,
                    output_tokens,
                    context_total,
                )
                usage_record = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "label": label or "unlabeled",
                    "msg_index": message_index,
                    "input_tokens": input_tokens,
                    "cache_read": cache_read_tokens,
                    "cache_creation": cache_creation_tokens,
                    "output_tokens": output_tokens,
                    "context_total": context_total,
                    "peak": peak_context_tokens,
                }
                usage_records.append(usage_record)
                # asyncio.to_thread, not a bare call: this loop is a plain
                # `async for`, and a synchronous open()+write() here would
                # block the event loop for however long the write takes. No
                # observed starvation today (this subprocess's event loop
                # runs only this one coroutine), but the platform rule is
                # "no sync I/O in async code" regardless of whether today's
                # call graph happens to make it harmless, since that safety
                # margin silently disappears the moment anything else (a
                # heartbeat, a timeout watchdog) shares this loop.
                await asyncio.to_thread(_append_context_ledger, usage_record)
            elif isinstance(message, UserMessage):
                content = message.content
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, ToolResultBlock):
                            tool_name = tool_use_names.get(block.tool_use_id, "unknown")
                            result_size = _tool_result_size(block.content)
                            logger.info(
                                "Agent tool result [%s] msg=%d tool=%s size_chars=%d is_error=%s",
                                label or "unlabeled",
                                message_index,
                                tool_name,
                                result_size,
                                bool(block.is_error),
                            )
                            tool_result_sizes.append(
                                {
                                    "tool": tool_name,
                                    "size_chars": result_size,
                                    "is_error": bool(block.is_error),
                                }
                            )
            elif isinstance(message, ResultMessage):
                result_text = message.result or ""
                cost_usd = getattr(message, "total_cost_usd", 0.0) or 0.0
                logger.info(
                    "Agent result usage [%s] usage=%s model_usage=%s",
                    label or "unlabeled",
                    getattr(message, "usage", None),
                    getattr(message, "model_usage", None),
                )
    finished_at = datetime.now(timezone.utc)

    tool_names = [tc.split("(")[0] for tc in tool_calls]
    unique_tool_names = sorted(set(tool_names))
    context7_calls = [t for t in tool_names if "context7" in t.lower() or "mcp" in t.lower()]

    # Unconditional summary, unlike the `if tool_calls:` block below: a
    # text-only session (no tool calls at all) is exactly the profile
    # implicated in autocompact thrashing, and previously got no session-level
    # peak_context_tokens summary at all -- only scattered per-message
    # "Agent usage" lines.
    logger.info(
        "Agent done [%s]: %d tool calls, cost=$%.4f, peak_context_tokens=%s",
        label or "unlabeled",
        len(tool_calls),
        cost_usd,
        peak_context_tokens if peak_context_tokens is not None else "unknown",
    )
    if peak_context_tokens is not None and peak_context_tokens >= _CONTEXT_WARNING_THRESHOLD_TOKENS:
        beta_note = (
            "the 1M-context beta means this no longer autocompacts, so this "
            "is the only signal for an unusually large session"
            if betas
            else "this session did NOT get the 1M-context beta (opus is not "
            "gated on), so it may also be autocompacting -- check for thrashing"
        )
        logger.warning(
            "Agent [%s] peak_context_tokens=%d crossed warning threshold %d (%s)",
            label or "unlabeled",
            peak_context_tokens,
            _CONTEXT_WARNING_THRESHOLD_TOKENS,
            beta_note,
        )

    if tool_calls:
        logger.info(
            "Agent done [%s]: tools used: %s",
            label or "unlabeled",
            ", ".join(unique_tool_names),
        )
        if context7_calls:
            logger.info("Context7 calls: %s", context7_calls)
        elif context7_api_key and context7_library_id:
            logger.warning("Context7 was available but agent made 0 Context7 calls")

    # TECH-4734 phase 2: attach the usage data collected above to the
    # LangSmith run for this session (this function is @traceable, so a run
    # tree exists whenever tracing is on). get_current_run_tree() returns
    # None when LANGSMITH_API_KEY/LANGCHAIN_TRACING_V2 aren't set -- guarded,
    # never raises. Metadata only: sizes, counts, labels -- no prompt/response
    # content, which may include untrusted diff content.
    run_tree = get_current_run_tree()
    if run_tree is not None:
        try:
            run_tree.add_metadata(
                {
                    "argus_context_usage": {
                        "label": label or "unlabeled",
                        "model": model,
                        "peak_context_tokens": peak_context_tokens,
                        "message_usage": usage_records,
                        "tool_result_sizes": tool_result_sizes,
                    }
                }
            )
        except Exception:  # noqa: BLE001 -- metadata attachment must never break a review
            logger.debug("Failed to attach context-usage metadata to LangSmith run", exc_info=True)

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
