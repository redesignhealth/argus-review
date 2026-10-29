"""Gemini leaf-reviewer runner: an original tool-calling loop against the
public ``google-genai`` SDK.

This is Track 3 of the multi-platform reviewer bench work (see
``argus.bench``'s module docstring). It is written directly against
``google.genai``/``google.genai.types`` -- deliberately NOT ported or
vendored from any internal, non-public codebase; there is no code
lineage between this module and any other implementation of a Gemini
reviewer, only a shared design (explicit context caching via
``argus.gemini_cache.GeminiCacheKeeper``, the same five platform-neutral
tools in ``argus.review_tools``, and the same ``SessionResult`` contract
``argus.runners`` already defines for the Claude Agent SDK path).

NOTE on raw-SDK usage vs. D024 ("no unwrapped LLM client" per RH's
internal ``llm-pipelines.md`` guidance, which some of THIS repo's own
packaged review prompts -- e.g. ``pr-review-specialist-llm-patterns.md``
-- enforce against code Argus reviews): calling ``google.genai`` directly
here, instead of through ``litellm`` or RH's internal ``rh-lib`` LLM
wrappers, is intentional, not an oversight this repo's own dogfooding
should flag against itself. ``argus-review`` is a public, Apache-2.0 OSS
package (see ``pyproject.toml``'s ``license``) that deliberately does NOT
depend on or vendor ``rh-lib`` -- a private, RH-internal package under a
different (non-public) license with no public distribution grant or
semver guarantee. ``.github/workflows/ci.yml``'s ``guard-rh-lib`` job enforces exactly
this: it fails the build on any reference to that package's underscored
import name appearing in ``argus/**/*.py`` or ``tests/**/*.py`` (this
docstring deliberately avoids spelling that literal string here, so as
not to trip its own grep). D024 is a policy this repo's own
prompts apply to the RH-internal codebases Argus reviews; it was never
adopted as a rule this repo enforces against its own source, and no such
check (ruff rule, mypy plugin, or otherwise) exists here today -- so
there is no real suppression marker to add. ``argus.openai_client``'s
raw ``openai`` SDK usage is the same pre-existing pattern for the same
reason.

Architecture, in one paragraph: build the five ``argus.review_tools``
functions into Gemini ``FunctionDeclaration``s, open one
``argus.review_tools.review_session`` for the sandboxed worktree root,
then loop ``client.aio.models.generate_content(...)`` -- executing every
function call the model requests each turn (there can be more than
one), feeding all of their results back as a single follow-up turn --
until the model stops requesting function calls, calls
``finish_review``, or the turn budget (``_MAX_TURNS_GEMINI``, this
module's own budget, independent of the Claude path's
``argus.runners._MAX_TURNS_CLAUDE`` and the OpenAI path's
``argus.openai_runner._MAX_TURNS_OPENAI``) is exhausted. Findings arrive via
``report_finding`` tool calls into ``review_tools``' per-session sink,
not as a JSON blob embedded in the model's own text -- so, to keep this
task's blast radius contained to this file plus ``argus.bench``'s
dispatch table, the final ``SessionResult.result_text`` is built here as
the same fenced ``json`` blob shape ``argus.helpers.parse_review_result``
already knows how to parse, deliberately omitting the ``system_group``
key so that function's own ``group_name`` fallback still applies
(this runner has no access to the caller's ``group.name`` -- only to
``entry.role``, which is a different, bench-level identifier).

Explicit context caching (``entry.caching == "on"``) is
strictly best-effort: Gemini's explicit-cache API has a minimum
content-size requirement a short prompt plus this small a toolset might
not clear, so any exception raised while creating/reusing a cache is
caught, logged, and treated exactly like ``entry.caching == "off"`` --
never allowed to fail the review itself. When a cache IS active, the
per-request ``system_instruction``/``tools``/``tool_config`` are
deliberately NOT also sent -- the live API rejects a request that
carries ANY of the three alongside ``cached_content=`` with
``400 INVALID_ARGUMENT: CachedContent can not be used with
GenerateContent request setting system_instruction, tools or
tool_config``. An earlier version of this module treated
``tool_config`` as a safe-to-combine per-request directive (unlike the
other two) on the theory that it configures *how* the model calls
tools rather than *what* tools/instructions it has -- that assumption
was wrong and was only caught by a real end-to-end run against the
live API (mocked unit tests never exercise this validation). The
forced-ANY-mode directive (see ``_FORCE_TOOL_CONFIG``) is instead baked
into the cache itself at creation time, via
``CreateCachedContentConfig.tool_config`` (which the SDK does support),
so it's still enforced on every cached request even though the
per-request config can no longer carry it.

Timeout: ``asyncio.wait_for`` wraps the whole session against the
effective timeout (an explicit ``timeout_s`` argument, when supplied by
the caller -- see ``argus.bench``'s ``_gemini_runner`` adapter -- else
``settings.ARGUS_SESSION_TIMEOUT``, the same setting the Claude
subprocess path enforces), returning a ``failure_reason="timeout"`` result instead
of raising. Unlike the Claude path, there is no subprocess to
``SIGKILL`` here -- ``asyncio.wait_for`` can only ever cancel at the next
``await`` point, so it CANNOT interrupt an in-flight blocking call inside
the SDK's own HTTP client. The ``genai.Client`` is therefore constructed
with its own ``http_options=types.HttpOptions(timeout=...)`` (in
milliseconds) as a second, lower-level layer of defense that bounds the
underlying HTTP request itself, independent of whether the outer
``asyncio.wait_for`` ever gets a chance to act -- deliberately set to a
MEANINGFULLY SHORTER deadline than the outer ``wait_for`` (see
``_HTTP_TIMEOUT_FRACTION``), so it fires FIRST: a shorter deadline
necessarily elapses before a longer one, raising its own ``httpx``-level
timeout exception well ahead of the outer ``wait_for``'s own deadline.
That exception -- whether it escapes the SDK call directly, or is
observed via ``asyncio.wait_for``'s own cancellation, depending on exact
timing -- is caught either way and translated into the same clean
``failure_reason="timeout"`` result.

Other implementation notes:

- Gemini 3.x attaches an opaque ``thought_signature`` to function-call
  parts that MUST be echoed back verbatim on the next turn. Each turn's
  "model" ``Content`` is therefore appended EXACTLY as
  ``response.candidates[0].content`` came back from the API, never
  hand-rebuilt from ``response.function_calls``.
- Every request forces ``tool_config=types.ToolConfig(function_calling_config=
  types.FunctionCallingConfig(mode="ANY"))`` -- Gemini's default AUTO mode
  otherwise lets the model follow the shared system prompt's "return
  findings as fenced JSON text" instruction (written for the Claude path)
  literally, silently reporting zero findings. On an UNCACHED request this
  is sent directly in the per-request ``GenerateContentConfig``; on a
  CACHED request it cannot be (the live API rejects ``tool_config``
  alongside ``cached_content=``, exactly like ``system_instruction``/
  ``tools`` -- see the caching paragraph below), so it's instead baked
  into the cache itself at creation time via
  ``CreateCachedContentConfig.tool_config``, which the SDK does support.
  As a second, belt-and-suspenders layer covering both paths, a turn with
  no function calls still attempts to parse ``response.text`` as that
  same fenced-JSON shape and feeds any findings found there into the sink
  before treating the turn as terminal.
- Cache activation (a synchronous ``caches.create()`` call, plus file I/O
  under a lock) runs via ``asyncio.to_thread`` so it never blocks this
  event loop -- and, symmetrically, a ``generate_content`` call that
  fails because a previously-valid ``cached_content=`` reference was
  invalidated upstream (TTL drift, manual deletion) invalidates the local
  record and retries the same request once, uncached, rather than failing
  the whole session. That ``caches.create()`` call runs against its own
  short-lived ``genai.Client`` instance, deliberately NOT the session's
  main ``client`` -- ``asyncio.to_thread`` cancellation doesn't stop the
  underlying OS thread, so a shared client could still be in use there at
  the exact moment a session timeout's ``finally`` block closes it; see
  ``_maybe_activate_cache``'s docstring.
- Both the sync and async ``genai.Client`` HTTP clients are closed in a
  ``finally`` covering every exit path (including a timeout/cancellation
  unwinding through it), so connections don't leak across the concurrent
  reviewer fan-out.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from langsmith import traceable

from argus import review_tools
from argus.bench import BenchEntry
from argus.gemini_cache import GeminiCacheKeeper
from argus.llm.models import estimate_cost_usd
from argus.llm.models import resolve as resolve_model_alias
from argus.runners import (
    _MID_BUDGET_NUDGE,
    _NUDGE_TURNS_BEFORE_BUDGET,
    _SUBPROCESS_TIMEOUT_S,
    _TURN_BUDGET_NUDGE,
    _TURN_BUDGET_SYSTEM_PROMPT_LINE,
    SessionResult,
    _compute_mid_budget_nudge_turn,
    _resolve_repo_root,
)

logger = logging.getLogger(__name__)

_MAX_TURNS_GEMINI = 100  # raised from 45 -- see TECH-6558

# Only "on" creates an explicit cache. "auto" (the packaged default) and "off"
# both send the whole request inline and leave caching to Gemini's own
# automatic prefix caching, which is enabled by default on current models.
#
# Why "auto" is not explicit: an explicit cache holds only what is placed in
# it at creation time (here the system prompt, tools and tool config, about
# 1-3k tokens), and a request that references it ("cached_content=") must
# not also carry those three fields. Every turn's growing conversation
# (the diff plus all prior tool results) is then sent as ordinary
# `contents`. In live A/B runs of the same PR,
# explicit caching reported ~7% cached prompt tokens (exactly the cache size
# on every turn) while caching="off" reported ~90%, with the same total
# prompt volume. The graceful-degradation try/except in
# `_maybe_activate_cache` still protects a caller who sets "on" against a
# prompt+toolset that cannot clear Gemini's minimum cache size.
_CACHE_ENABLED_VALUES = frozenset({"on"})

_DEFAULT_READ_LIMIT = 2000  # mirrors argus.review_tools._DEFAULT_READ_LIMIT

# Forces every turn to make a function call rather than freely returning
# plain text. Without this, Gemini's default AUTO function-calling mode
# lets the model follow the shared system prompt's "return findings as
# fenced JSON text" instruction (written for the Claude path, which reads
# result_text directly) literally, silently reporting zero findings via
# report_finding even when it found real issues. This is belt-and-suspenders
# with `_parse_text_findings` below, which recovers findings from
# `response.text` on the rare turn the model still doesn't call a tool.
_FORCE_TOOL_CONFIG = types.ToolConfig(
    function_calling_config=types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.ANY)
)

# The HTTP-client-level timeout (see module docstring) is deliberately a
# MEANINGFULLY SHORTER deadline than the outer `asyncio.wait_for` in
# `run_session_gemini`, so it fires FIRST -- a shorter deadline
# necessarily elapses before a longer one does. That raises its own
# httpx-level timeout exception ahead of the outer `wait_for`'s own
# deadline; `run_session_gemini`'s except clause catches it and
# translates it into the same clean `failure_reason="timeout"` result the outer
# timeout would otherwise produce, rather than letting it escape
# uncaught.
_HTTP_TIMEOUT_FRACTION = 0.8

# ---------------------------------------------------------------------------
# One-shot retry of transient Gemini failures
# ---------------------------------------------------------------------------
# A session is retried AT MOST ONCE, and only when it failed with a transient
# error (HTTP 429/500/502/503/504, RESOURCE_EXHAUSTED/UNAVAILABLE/
# DEADLINE_EXCEEDED/INTERNAL, or an httpx timeout/connect/read/remote-protocol
# error) before the model returned ANY response (i.e. on turn 0), so the retry
# restarts the session cleanly from the beginning. Later failures are not
# restarted; the SDK retries only individual 5xx requests (see
# `_HTTP_RETRY_ATTEMPTS`), and a 5xx it already retried is not restarted again,
# so the restart covers 429 and transport errors only. The cooldown depends on the error kind (see `_retry_decision`):
# 45-60 s jitter for 429/RESOURCE_EXHAUSTED (per-minute quota windows), 5-10 s
# for 5xx/network errors; a server Retry-After/retryDelay >= 5 s is honored;
# never more than 60 s. Daily-quota 429s are not retried. Overridable via
# ARGUS_GEMINI_RETRY_RATE_LIMIT_{MIN,MAX}_S, ARGUS_GEMINI_RETRY_COOLDOWN_{MIN,MAX}_S
# and ARGUS_GEMINI_RETRY_MAX_WAIT_S (read per call, so tests can set them to 0).
# The 60 s worst case is far inside ARGUS_SESSION_TIMEOUT (default 1500 s), and
# the retry is skipped if <25% of that timeout remains.
_RETRY_RATE_LIMIT_MIN_S = 45.0  # 429 / RESOURCE_EXHAUSTED: per-minute windows
_RETRY_RATE_LIMIT_MAX_S = 60.0
_RETRY_SHORT_MIN_S = 5.0  # 5xx / UNAVAILABLE / DEADLINE_EXCEEDED / network
_RETRY_SHORT_MAX_S = 10.0
_RETRY_MAX_WAIT_S = 60.0  # hard cap on any single wait, incl. a server Retry-After
_RETRY_AFTER_MIN_S = 5.0  # a server delay below this is ignored (jitter used)
# A server-suggested delay above this means the quota window is not a
# per-minute one (minutes/hours): fail fast rather than retry.
_RETRY_GIVEUP_DELAY_S = 120.0
# Do not retry when less than this fraction of the session timeout remains
# (e.g. the first attempt already burned its HTTP timeout).
_RETRY_MIN_REMAINING_FRACTION = 0.25
_TRANSIENT_HTTP_CODES = frozenset({429, 500, 502, 503, 504})
_TRANSIENT_STATUSES = frozenset(
    {"RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED", "INTERNAL"}
)
_TRANSIENT_HTTPX = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
)

# SDK-level per-request retry of 5xx at any turn; the request is re-sent
# unchanged, so no tools are replayed. 429s stay in _retry_decision because the
# SDK cannot tell daily quota from per-minute limits or honor Retry-After.
_HTTP_RETRY_ATTEMPTS = 3  # including the initial call
_HTTP_RETRY_STATUS_CODES = (500, 502, 503, 504)
_HTTP_RETRY_INITIAL_DELAY_S = 2.0
_HTTP_RETRY_MAX_DELAY_S = 20.0
_DETAIL_MAX_CHARS = 300
_SECRET_RE = re.compile(
    r"(AIza[0-9A-Za-z_\-]{20,}|(?i:(?:api[_-]?key|key|token)=)[^&\s\"']+|"
    r"(?i:bearer\s+)[A-Za-z0-9._\-]+|sk-[A-Za-z0-9_\-]{16,})"
)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default


@dataclass
class _Progress:
    """Mutable per-session state shared between attempts and the failure handler."""

    responses: int = 0  # successful generate_content responses received
    last_turn: int | None = None  # 0-based index of the turn that last ran
    retried: bool = False
    retry_wait_s: float = 0.0
    # Usage accrued so far, so a session that dies mid-flight still reports it.
    prompt_tokens: int = 0
    cached_tokens: int = 0
    candidates_tokens: int = 0
    thoughts_tokens: int = 0
    tool_use_prompt_tokens: int = 0
    tool_calls: list[str] = field(default_factory=list)

    @property
    def input_tokens(self) -> int:
        # Gemini's prompt_token_count already includes the cached portion, so
        # subtract it to avoid billing it at both input and cache-read rates.
        # Tool-result tokens are billed as input.
        return max(self.prompt_tokens - self.cached_tokens, 0) + self.tool_use_prompt_tokens

    @property
    def output_tokens(self) -> int:
        # Thinking tokens are a separate additive bucket billed as output.
        return self.candidates_tokens + self.thoughts_tokens

    def cost_usd(self, model: str) -> float:
        if self.responses == 0:
            return 0.0
        return estimate_cost_usd(
            model=model,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cached_input_tokens=self.cached_tokens,
        )


def _scrub(text: str) -> str:
    """Redact anything key-shaped and truncate for logging."""
    text = _SECRET_RE.sub("***", text).replace("\n", " ")
    return text[:_DETAIL_MAX_CHARS]


def _error_code_status(exc: BaseException) -> tuple[int | None, str | None]:
    """Best-effort (HTTP code, status string) for an SDK/transport exception."""
    if isinstance(exc, genai_errors.APIError):
        code = getattr(exc, "code", None)
        status = getattr(exc, "status", None)
        return (
            code if isinstance(code, int) else None,
            status if isinstance(status, str) else None,
        )
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code, None
    return None, None


def _is_transient_error(exc: BaseException) -> bool:
    code, status = _error_code_status(exc)
    if code in _TRANSIENT_HTTP_CODES or (status or "").upper() in _TRANSIENT_STATUSES:
        return True
    if isinstance(exc, (genai_errors.APIError, httpx.HTTPStatusError)):
        return False  # a definite API answer that is not transient (400/401/403/404...)
    return isinstance(exc, _TRANSIENT_HTTPX)


def _retry_after_seconds(exc: BaseException) -> float | None:
    """Server-suggested delay from a Retry-After header or RetryInfo.retryDelay."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    try:
        raw = headers.get("retry-after") if headers is not None else None
        if raw is not None:
            return max(0.0, float(raw))
    except (TypeError, ValueError, AttributeError):
        pass
    details = getattr(exc, "details", None)
    err = details.get("error", details) if isinstance(details, dict) else None
    items = err.get("details") if isinstance(err, dict) else None
    for item in items if isinstance(items, list) else []:
        delay = item.get("retryDelay") if isinstance(item, dict) else None
        if isinstance(delay, str) and delay.endswith("s"):
            try:
                return max(0.0, float(delay[:-1]))
            except ValueError:
                pass
    return None


def _is_daily_quota(exc: BaseException, server_delay: float | None) -> bool:
    """True when a 429 reports a per-day quota or a many-minutes retry delay."""
    if server_delay is not None and server_delay > _RETRY_GIVEUP_DELAY_S:
        return True
    return bool(re.search(r"per\s*day|perday", str(exc), re.IGNORECASE))


def _retry_decision(exc: BaseException) -> tuple[str, float] | None:
    """``(kind, wait_seconds)`` if ``exc`` should be retried once, else ``None``.

    Kinds: ``rate_limit`` (429/RESOURCE_EXHAUSTED, 45-60 s jitter),
    ``server`` (5xx/UNAVAILABLE/DEADLINE_EXCEEDED/INTERNAL) and ``network``
    (5-10 s jitter). A server Retry-After/retryDelay >= 5 s replaces the
    jittered value; the wait never exceeds 60 s. A daily-quota 429 is never
    retried.
    """
    if not _is_transient_error(exc):
        return None
    code, status = _error_code_status(exc)
    if isinstance(exc, genai_errors.ServerError) and code in _HTTP_RETRY_STATUS_CODES:
        return None  # the SDK already retried this request; a restart would repeat it
    server_delay = _retry_after_seconds(exc)
    if code == 429 or (status or "").upper() == "RESOURCE_EXHAUSTED":
        if _is_daily_quota(exc, server_delay):
            return None
        kind = "rate_limit"
        lo = _env_float("ARGUS_GEMINI_RETRY_RATE_LIMIT_MIN_S", _RETRY_RATE_LIMIT_MIN_S)
        hi = _env_float("ARGUS_GEMINI_RETRY_RATE_LIMIT_MAX_S", _RETRY_RATE_LIMIT_MAX_S)
    else:
        kind = "server" if code is not None or status else "network"
        lo = _env_float("ARGUS_GEMINI_RETRY_COOLDOWN_MIN_S", _RETRY_SHORT_MIN_S)
        hi = _env_float("ARGUS_GEMINI_RETRY_COOLDOWN_MAX_S", _RETRY_SHORT_MAX_S)
    cap = _env_float("ARGUS_GEMINI_RETRY_MAX_WAIT_S", _RETRY_MAX_WAIT_S)
    wait = random.uniform(lo, max(lo, hi))  # noqa: S311 - jitter, not security
    if server_delay is not None and server_delay >= _RETRY_AFTER_MIN_S:
        wait = server_delay
    return kind, min(wait, cap)


def _failure_detail(exc: BaseException) -> str:
    """Short, key-free label such as ``429 RESOURCE_EXHAUSTED`` or ``ConnectError``."""
    code, status = _error_code_status(exc)
    detail = " ".join(p for p in (str(code) if code is not None else "", status or "") if p)
    detail = detail or type(exc).__name__
    if (code == 429 or (status or "").upper() == "RESOURCE_EXHAUSTED") and _is_daily_quota(
        exc, _retry_after_seconds(exc)
    ):
        detail += " (daily quota)"
    return detail


_INVALID_ARGUMENT_RETRY_MIN_S = 2.0
_INVALID_ARGUMENT_RETRY_MAX_S = 5.0
_SHAPE_TAIL_CONTENTS = 4


def _is_invalid_argument(exc: BaseException) -> bool:
    """True for a Gemini ``400 INVALID_ARGUMENT`` API answer."""
    code, status = _error_code_status(exc)
    return code == 400 and (status or "").upper() == "INVALID_ARGUMENT"


def _part_shape(part: Any) -> str:
    """Kind and size of one content part -- never its text."""
    call = getattr(part, "function_call", None)
    if call is not None:
        size = len(json.dumps(call.args or {}, default=str))
        return f"call:{call.name}({size})"
    resp = getattr(part, "function_response", None)
    if resp is not None:
        # An empty result reports size 0 so _describe_request can flag it.
        size = len(json.dumps(resp.response, default=str)) if resp.response else 0
        return f"result:{resp.name}({size})"
    text = getattr(part, "text", None)
    if text is not None:
        tag = "thought" if getattr(part, "thought", False) else "text"
        return f"{tag}({len(text)})"
    return type(part).__name__


def _describe_request(contents: list[Any], config: Any = None) -> str:
    """Key-free, text-free summary of a request for diagnosing a 400.

    Gemini's INVALID_ARGUMENT body carries no field name, so this records the
    structure that could make a request malformed: roles, part kinds and sizes
    for the last few turns, plus anomalies (empty parts, unanswered function
    calls, consecutive same-role turns).
    """
    total = 0
    biggest = 0
    anomalies: list[str] = []
    prev_role = None
    for i, content in enumerate(contents):
        parts = list(content.parts or [])
        if not parts:
            anomalies.append(f"empty-content@{i}")
        calls = results = 0
        for part in parts:
            shape = _part_shape(part)
            size = int(m.group(1)) if (m := re.search(r"\((\d+)\)$", shape)) else 0
            total += size
            biggest = max(biggest, size)
            if shape.startswith("call:"):
                calls += 1
            elif shape.startswith("result:"):
                results += 1
            if size == 0 and shape.startswith(("text", "thought", "result")):
                anomalies.append(f"zero-size-{re.split(r'[:(]', shape)[0]}@{i}")
        if prev_role is not None and content.role == prev_role:
            anomalies.append(f"repeated-role-{content.role}@{i}")
        prev_role = content.role
        nxt = contents[i + 1] if i + 1 < len(contents) else None
        if calls and nxt is not None:
            answered = sum(
                1 for p in (nxt.parts or []) if getattr(p, "function_response", None) is not None
            )
            if answered != calls:
                anomalies.append(f"calls({calls})!=results({answered})@{i}")
    tail = contents[-_SHAPE_TAIL_CONTENTS:]
    first = len(contents) - len(tail)
    tail_desc = "; ".join(
        f"#{first + j} {c.role}[{', '.join(_part_shape(p) for p in (c.parts or []))}]"
        for j, c in enumerate(tail)
    )
    cfg = ""
    if config is not None:
        cfg = (
            f" tools={len(getattr(config, 'tools', None) or [])}"
            f" system_instruction={bool(getattr(config, 'system_instruction', None))}"
            f" cached_content={bool(getattr(config, 'cached_content', None))}"
        )
    return (
        f"contents={len(contents)} approx_chars={total} largest_part={biggest}{cfg} "
        f"anomalies={anomalies or 'none'} tail: {tail_desc}"
    )


def _safe_describe_request(contents: list[Any], config: Any) -> str:
    """``_describe_request`` that can never mask the error being diagnosed."""
    try:
        return _describe_request(contents, config)
    except Exception as exc:  # noqa: BLE001 - diagnostics must not replace the 400
        return f"unavailable ({type(exc).__name__})"


def _error_details(exc: BaseException) -> str:
    """Scrubbed ``details`` payload of an API error (may name the bad field)."""
    details = getattr(exc, "details", None)
    try:
        return _scrub(json.dumps(details, default=str)) if details else ""
    except (TypeError, ValueError):
        return ""


# Matches the fenced-json extraction argus.helpers.parse_review_result
# already applies for the Claude path.
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

# One manual JSON-schema dict per argus.review_tools function. A small
# hand-written schema per tool (5 known functions) is clearer than
# introspecting each function's signature/docstring into a schema, and
# this same list doubles as the GeminiCacheKeeper `tool_schema` hash
# input (see `_maybe_activate_cache`) -- it's already a plain,
# JSON-serializable structure, no separate serialization needed.
_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": (
            "Read a file, sandboxed to this review's worktree root. Returns "
            "line-numbered content ('<1-based line number>: <content>', one per "
            "output line)."
        ),
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the worktree root.",
                },
                "offset": {
                    "type": "integer",
                    "description": "0-indexed line to start reading from. Defaults to 0.",
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "Maximum number of lines to return. 0 reads to EOF "
                        f"(capped at a hard ceiling). Defaults to {_DEFAULT_READ_LIMIT}."
                    ),
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "glob_files",
        "description": (
            "List files under the worktree root matching a glob pattern, one path "
            "per output line, sorted."
        ),
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": (
                        "A pathlib-style glob pattern relative to the root, e.g. '**/*.py'."
                    ),
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "grep",
        "description": (
            "Search for a regular-expression pattern in files matching a glob under "
            "the worktree root."
        ),
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression to search for.",
                },
                "glob": {
                    "type": "string",
                    "description": "Glob of files to search. Defaults to '**/*'.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["files", "content"],
                    "description": (
                        "'files' (default) returns matching file paths, one per line. "
                        "'content' returns '<file>:<line>: <content>' for every "
                        "matching line."
                    ),
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "report_finding",
        "description": "Record one review finding -- a bug, risk, or issue worth flagging.",
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "file": {
                    "type": ["string", "null"],
                    "description": (
                        "File path the finding is about, or null if not file-specific."
                    ),
                },
                "line": {
                    "type": ["integer", "null"],
                    "description": (
                        "Line number the finding is about, or null if not line-specific."
                    ),
                },
                "description": {
                    "type": "string",
                    "description": "The finding itself, in plain language.",
                },
                "context": {
                    "type": ["string", "null"],
                    "description": "Optional supporting context or quoted code.",
                },
            },
            "required": ["file", "line", "description"],
        },
    },
    {
        "name": "finish_review",
        "description": (
            "Signal that you are done exploring and reporting findings. Call this "
            "exactly once, as your last tool call, when you have nothing further to "
            "report."
        ),
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "files_explored": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "File paths you read for context during this review.",
                },
            },
            "required": ["files_explored"],
        },
    },
]

# Dispatch table: Gemini function-call name -> the argus.review_tools
# implementation it invokes. Deliberately the plain functions themselves
# (not wrapped) -- they already read the active review_session's
# sandboxed root/findings sink from contextvars, so no session/root
# parameter needs threading through here.
_TOOL_FUNCTIONS: dict[str, Any] = {
    "read_file": review_tools.read_file,
    "glob_files": review_tools.glob_files,
    "grep": review_tools.grep,
    "report_finding": review_tools.report_finding,
    "finish_review": review_tools.finish_review,
}


def _build_tools() -> list[types.Tool]:
    """Build the Gemini ``Tool``/``FunctionDeclaration`` set from ``_TOOL_SCHEMAS``."""
    declarations = [
        types.FunctionDeclaration(
            name=schema["name"],
            description=schema["description"],
            parameters_json_schema=schema["parameters_json_schema"],
        )
        for schema in _TOOL_SCHEMAS
    ]
    return [types.Tool(function_declarations=declarations)]


def _execute_tool_call(name: str, args: dict[str, Any]) -> tuple[str, bool]:
    """Dispatch one Gemini function call to its ``review_tools`` implementation.

    Catches every exception: ``name``/``args`` come from an LLM tool call
    this code can't trust to be well-formed (an unknown tool name, a
    missing required argument, a malformed regex), and a bad call must
    degrade to an error string the model can see and recover from in its
    next turn, not crash the whole review session.

    Returns ``(output, is_error)``. The caller uses ``is_error`` to decide
    whether a ``finish_review`` call actually succeeded -- a malformed
    ``finish_review`` (e.g. missing its required ``files_explored``
    argument) must NOT terminate the session; it should be recoverable
    like any other tool error, not treated as a clean finish.
    """
    fn = _TOOL_FUNCTIONS.get(name)
    if fn is None:
        logger.warning("Tool call error: unknown tool %r", name)
        return f"Error: unknown tool {name!r}", True
    try:
        return str(fn(**args)), False
    except Exception as exc:  # noqa: BLE001 - LLM-supplied call, must never crash the loop
        args_repr = repr(args)[:300]
        logger.warning("Tool call error %s(%s): %s", name, args_repr, exc)
        return f"Error calling {name}({args_repr}): {exc}", True


def _build_result_text(findings: list[dict[str, Any]], files_explored: list[str]) -> str:
    """Build a ``result_text`` JSON blob matching what
    ``argus.helpers.parse_review_result`` already parses for the Claude path.

    Deliberately omits the ``system_group`` key: this runner only knows
    ``entry.role`` (a bench-level identifier, e.g. ``"system-generalist"``),
    not the caller's own ``group.name``/``"{group}::{specialist}"`` label,
    and ``parse_review_result`` already falls back to its caller-supplied
    ``group_name`` when the key is absent -- so omitting it here is what
    makes that fallback apply correctly instead of a wrong value winning.
    """
    payload = {"findings": findings, "files_explored": files_explored}
    return "```json\n" + json.dumps(payload) + "\n```"


def _parse_text_findings(text: str) -> tuple[list[dict[str, Any]], list[str]] | None:
    """Best-effort recovery for a reviewer that returned its findings as a
    fenced JSON text blob instead of calling ``report_finding`` (see this
    module's docstring on the AUTO function-calling gap).

    Uses the same fenced-``json`` extraction ``argus.helpers.parse_review_result``
    already applies for the Claude path. Returns ``None`` when ``text``
    doesn't parse as that shape at all (plain prose, e.g. "Nothing to
    report.") -- deliberately does NOT fall back to treating arbitrary text
    as a single finding the way ``parse_review_result``'s own except branch
    does for the Claude path; a Gemini turn can legitimately end with
    unstructured commentary and genuinely no findings, and manufacturing a
    finding out of that would be noise, not recovery.
    """
    if not text:
        return None
    match = _FENCED_JSON_RE.search(text)
    json_str = match.group(1) if match else text.strip()
    try:
        data = json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    raw_findings = data.get("findings")
    if not isinstance(raw_findings, list):
        return None
    findings = [f for f in raw_findings if isinstance(f, dict)]
    raw_files = data.get("files_explored")
    files_explored = [str(f) for f in raw_files] if isinstance(raw_files, list) else []
    return findings, files_explored


def _is_cache_invalid_error(exc: BaseException) -> bool:
    """Best-effort detection of a Gemini "cache not found/invalid" error.

    Deliberately narrow: only a 4xx ``ClientError`` whose message actually
    mentions the cache is treated this way -- an unrelated 4xx (bad auth,
    malformed request, quota exhaustion) must surface normally, not
    silently trigger an uncached retry that would mask the real problem.
    """
    if not isinstance(exc, genai_errors.ClientError):
        return False
    message = (exc.message or str(exc)).lower()
    status = (exc.status or "").upper()
    return "cache" in message and (
        status in {"NOT_FOUND", "FAILED_PRECONDITION"} or exc.code == 404
    )


def _invalidate_cache_record(*, entry: BenchEntry, model: str, system_prompt: str) -> None:
    """Remove the local cache-lifecycle record for this key so a subsequent
    session doesn't keep handing out a ``cache_name`` the API has already
    forgotten about (TTL drift, manual deletion upstream, etc.).

    Best-effort: never raises -- the worst case of a failure here is one
    more redundant (harmless) cache-creation attempt on some future
    session, not a crashed review. Recomputes the exact same key
    :meth:`GeminiCacheKeeper.get_or_create` used to create the record in
    the first place (same role/model/system_prompt/tool_schema inputs).
    """
    try:
        GeminiCacheKeeper().invalidate(
            role=entry.role,
            model=model,
            system_prompt=system_prompt,
            tool_schema=_TOOL_SCHEMAS,
        )
    except Exception as exc:  # noqa: BLE001 - best-effort, see docstring
        logger.warning(
            "Failed to invalidate local Gemini cache record (role=%s): %s", entry.role, exc
        )


def _maybe_activate_cache(
    *,
    api_key: str,
    http_options: types.HttpOptions,
    entry: BenchEntry,
    model: str,
    system_prompt: str,
    gemini_tools: list[types.Tool],
    label: str,
) -> str | None:
    """Best-effort explicit-cache activation. Never raises.

    Returns the active ``cache_name`` to pass as ``cached_content=`` on
    every subsequent ``generate_content`` call, or ``None`` when caching
    is off, or when creating/reusing one failed for any reason (e.g.
    Gemini's minimum-content-size requirement for explicit caching isn't
    cleared by a short prompt + this small a toolset) -- a cache failure
    degrades to an uncached session rather than failing the review.

    Deliberately takes ``api_key``/``http_options`` rather than the
    session's own ``client``: this whole function runs inside a detached
    ``asyncio.to_thread`` worker (see the call site in ``_run_turns``),
    and cancelling that ``asyncio.to_thread`` (e.g. on a session timeout)
    does NOT stop the underlying OS thread -- it keeps running until the
    call it's making returns. If ``_create_fn`` below shared the
    session's own ``client``, that in-flight `caches.create()` call could
    still be executing against it at the exact moment `_run_turns`'
    ``finally`` block closes that same ``client``/``client.aio`` on the
    cancellation path, racing the close. Building a short-lived client
    scoped to just this call -- fully created, used, and closed within
    this one synchronous function -- makes that race impossible: nothing
    outside this function ever touches it.
    """
    if entry.caching not in _CACHE_ENABLED_VALUES:
        return None

    def _create_fn(
        *,
        role: str,
        model: str,
        system_prompt: str,
        tool_schema: Any,
        ttl_seconds: int,
    ) -> str:
        # `tool_schema` here is `_TOOL_SCHEMAS` (passed to get_or_create below,
        # purely for cache-key hashing) -- the actual `tools=` sent upstream is
        # the closed-over `gemini_tools` (the real `types.Tool` objects built
        # from that same schema), not this raw dict form.
        #
        # A dedicated client, NOT the session's own -- see this function's
        # docstring for why sharing it would be unsafe under cancellation.
        # Its entire lifetime is contained within this call.
        cache_client = genai.Client(api_key=api_key, http_options=http_options)
        try:
            cache = cache_client.caches.create(
                model=model,
                config=types.CreateCachedContentConfig(
                    system_instruction=system_prompt,
                    tools=gemini_tools,
                    # Baked into the cache itself, not the per-request
                    # config -- see `_build_config`'s docstring for why a
                    # cached request can no longer carry `tool_config`
                    # directly. `CreateCachedContentConfig` (unlike
                    # `GenerateContentConfig` when `cached_content=` is
                    # set) DOES support `tool_config`, so the forced-ANY
                    # tool-calling directive is still enforced on every
                    # request that references this cache.
                    tool_config=_FORCE_TOOL_CONFIG,
                    ttl=f"{ttl_seconds}s",
                ),
            )
        finally:
            cache_client.close()
        if not cache.name:
            raise ValueError("client.caches.create() returned a cache with no name")
        return cache.name

    try:
        keeper = GeminiCacheKeeper(create_fn=_create_fn)
        record = keeper.get_or_create(
            role=entry.role,
            model=model,
            system_prompt=system_prompt,
            tool_schema=_TOOL_SCHEMAS,
        )
        return record.cache_name
    except Exception as exc:  # noqa: BLE001 - explicit caching is best-effort, see docstring
        logger.warning(
            "Gemini explicit-cache activation failed [%s] (role=%s); proceeding "
            "without a cache: %s",
            label or "unlabeled",
            entry.role,
            exc,
        )
        return None


async def _run_turns(
    *,
    entry: BenchEntry,
    model: str,
    system_prompt: str,
    user_message: str,
    settings: Any,
    label: str,
    repo_root: str,
    timeout_s: float,
    started_at: datetime,
    progress: _Progress,
) -> SessionResult:
    """The actual tool-calling loop. Returns ``failure_reason="turn_budget_exhausted"``
    when the loop ran out of turns without a ``finish_review`` call, else
    ``None`` -- the enclosing ``run_session_gemini`` is what applies the
    timeout and turns a cancellation into a ``failure_reason="timeout"``
    result instead.
    """
    system_prompt = (
        system_prompt + "\n\n" + _TURN_BUDGET_SYSTEM_PROMPT_LINE.format(max_turns=_MAX_TURNS_GEMINI)
    )
    _, api_key = settings.google_credential
    # Second layer of defense: `asyncio.wait_for` in `run_session_gemini`
    # cannot interrupt an in-flight blocking SDK call once it has started
    # (it can only act at the next `await` point), so this bounds the
    # underlying HTTP request itself, independently. Kept MEANINGFULLY
    # SHORTER than the outer `wait_for` deadline (see
    # `_HTTP_TIMEOUT_FRACTION`) so it fires FIRST -- raising its own
    # httpx-level timeout exception, which `run_session_gemini`'s except
    # clause catches and translates into a clean `failure_reason="timeout"` result
    # -- rather than racing the outer deadline so closely that the raw
    # httpx exception could escape uncaught instead.
    #
    # `base_url` is only set when truthy; `None` safely defaults to the
    # standard Gemini endpoint (`generativelanguage.googleapis.com`) --
    # mirrors `OPENAI_BASE_URL`'s exact same optional-proxy pattern in
    # `argus.openai_runner`/`argus.openai_client`. This same `http_options`
    # instance is reused as-is by `_maybe_activate_cache`'s own
    # short-lived `genai.Client` (see that function's docstring for why it
    # needs a separate client instance), so threading `base_url` through
    # here covers both `genai.Client(...)` construction sites in this
    # module without a second read of `settings.GOOGLE_BASE_URL`.
    base_url = getattr(settings, "GOOGLE_BASE_URL", None)
    http_options = types.HttpOptions(
        timeout=int(timeout_s * _HTTP_TIMEOUT_FRACTION * 1000),
        base_url=base_url if isinstance(base_url, str) and base_url else None,
    )
    client = genai.Client(api_key=api_key, http_options=http_options)
    logger.info(
        "Gemini session started: %s model=%s caching=%s timeout=%ss",
        label,
        model,
        entry.caching if entry else "none",
        timeout_s,
    )
    try:
        gemini_tools = _build_tools()

        # `_maybe_activate_cache` performs a synchronous `caches.create()`
        # call (plus file I/O under a lock) -- run it in a worker thread so it
        # never blocks this event loop (and every other concurrent reviewer
        # session sharing it) for the duration of an upstream cache-creation
        # round trip. Deliberately passed `api_key`/`http_options` rather
        # than this function's own `client`: see `_maybe_activate_cache`'s
        # docstring for why sharing it would race this function's own
        # `finally` block on a timeout/cancellation.
        cached_content_name = await asyncio.to_thread(
            _maybe_activate_cache,
            api_key=api_key,
            http_options=http_options,
            entry=entry,
            model=model,
            system_prompt=system_prompt,
            gemini_tools=gemini_tools,
            label=label,
        )

        # Retry only model calls, not detached synchronous cache creation.
        request_http_options = types.HttpOptions(
            retry_options=types.HttpRetryOptions(
                attempts=_HTTP_RETRY_ATTEMPTS,
                initial_delay=_HTTP_RETRY_INITIAL_DELAY_S,
                max_delay=_HTTP_RETRY_MAX_DELAY_S,
                http_status_codes=list(_HTTP_RETRY_STATUS_CODES),
            )
        )

        def _build_config(cache_name: str | None) -> types.GenerateContentConfig:
            # Gemini constraint (confirmed against the live API, not just
            # documentation): a request using `cached_content=` must NOT
            # ALSO carry `system_instruction`, `tools`, OR `tool_config` --
            # all three are rejected outright with `400 INVALID_ARGUMENT:
            # CachedContent can not be used with GenerateContent request
            # setting system_instruction, tools or tool_config`. All three
            # are already part of what the cache represents (see
            # `_maybe_activate_cache`'s `_create_fn`, which now bakes
            # `_FORCE_TOOL_CONFIG` into the cache itself via
            # `CreateCachedContentConfig.tool_config`), so the cached
            # branch here sends `cached_content` alone.
            if cache_name:
                return types.GenerateContentConfig(
                    cached_content=cache_name, http_options=request_http_options
                )
            return types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=gemini_tools,
                tool_config=_FORCE_TOOL_CONFIG,
                http_options=request_http_options,
            )

        config = _build_config(cached_content_name)

        contents: list[types.Content] = [
            types.Content(role="user", parts=[types.Part.from_text(text=user_message)])
        ]

        files_explored: list[str] = []
        exhausted = False

        _mid_budget_nudge_turn = _compute_mid_budget_nudge_turn(_MAX_TURNS_GEMINI)

        with review_tools.review_session(repo_root) as findings_sink:
            for _turn in range(_MAX_TURNS_GEMINI):
                progress.last_turn = _turn
                if (
                    _turn == _MAX_TURNS_GEMINI - _NUDGE_TURNS_BEFORE_BUDGET
                    and contents[-1].parts is not None
                ):
                    contents[-1].parts.append(types.Part.from_text(text=_TURN_BUDGET_NUDGE))
                # Not an `elif` -- these are two independent checkpoints at
                # different turns and must both be able to fire in the same
                # session (see _compute_mid_budget_nudge_turn's docstring
                # for why they can never land on the same turn today).
                if (
                    _mid_budget_nudge_turn is not None
                    and _turn == _mid_budget_nudge_turn
                    and contents[-1].parts is not None
                ):
                    contents[-1].parts.append(types.Part.from_text(text=_MID_BUDGET_NUDGE))
                try:
                    response = await client.aio.models.generate_content(
                        model=model, contents=contents, config=config
                    )
                except Exception as exc:
                    # A locally-recorded-as-unexpired cache can still have
                    # been deleted/invalidated upstream (TTL drift, manual
                    # deletion) -- degrade gracefully by invalidating the
                    # stale local record and retrying this SAME request once,
                    # uncached, rather than failing the whole session.
                    if cached_content_name is None or not _is_cache_invalid_error(exc):
                        logger.error(
                            "Gemini generate_content failed [%s] (role=%s, model=%s, turn=%d): %s",
                            label or "unlabeled",
                            entry.role if entry else "unknown",
                            model,
                            _turn,
                            _scrub(str(exc)),
                        )
                        if not _is_invalid_argument(exc):
                            raise
                        # A 400 INVALID_ARGUMENT names no field, so record the
                        # request's shape, then retry this SAME request once:
                        # whether it recovers tells a transient fault from a
                        # deterministic one.
                        logger.error(
                            "Gemini 400 request shape [%s] (turn=%d) details=%s: %s",
                            label or "unlabeled",
                            _turn,
                            _error_details(exc) or "none",
                            _safe_describe_request(contents, config),
                        )
                        lo = _env_float(
                            "ARGUS_GEMINI_RETRY_INVALID_ARGUMENT_MIN_S",
                            _INVALID_ARGUMENT_RETRY_MIN_S,
                        )
                        hi = _env_float(
                            "ARGUS_GEMINI_RETRY_INVALID_ARGUMENT_MAX_S",
                            _INVALID_ARGUMENT_RETRY_MAX_S,
                        )
                        await asyncio.sleep(
                            random.uniform(lo, max(lo, hi))  # noqa: S311 - jitter, not security
                        )
                        try:
                            response = await client.aio.models.generate_content(
                                model=model, contents=contents, config=config
                            )
                        except Exception as retry_exc:
                            logger.error(
                                "Gemini 400 retry failed [%s] (turn=%d): %s",
                                label or "unlabeled",
                                _turn,
                                _scrub(str(retry_exc)),
                            )
                            raise
                        logger.warning(
                            "Gemini 400 recovered on retry [%s] (turn=%d): the failure "
                            "was transient, not a malformed request",
                            label or "unlabeled",
                            _turn,
                        )
                    else:
                        logger.warning(
                            "Gemini cache [%s] appears invalid upstream (role=%s); "
                            "invalidating the local record and retrying this turn "
                            "uncached: %s",
                            label or "unlabeled",
                            entry.role,
                            exc,
                        )
                        await asyncio.to_thread(
                            _invalidate_cache_record,
                            entry=entry,
                            model=model,
                            system_prompt=system_prompt,
                        )
                        cached_content_name = None
                        config = _build_config(None)
                        response = await client.aio.models.generate_content(
                            model=model, contents=contents, config=config
                        )

                progress.responses += 1
                usage = response.usage_metadata
                if usage is not None:
                    # thoughts and tool-use prompt tokens are billable, separate
                    # additive buckets (not folded into prompt/candidates counts).
                    progress.prompt_tokens += usage.prompt_token_count or 0
                    progress.candidates_tokens += usage.candidates_token_count or 0
                    progress.cached_tokens += usage.cached_content_token_count or 0
                    progress.thoughts_tokens += usage.thoughts_token_count or 0
                    progress.tool_use_prompt_tokens += usage.tool_use_prompt_token_count or 0

                function_calls = response.function_calls or []
                if not function_calls:
                    # Model stopped requesting function calls. Gemini's
                    # forced ANY tool-calling mode (see `_FORCE_TOOL_CONFIG`)
                    # should make this rare, but as a second layer of
                    # defense, recover any findings the model reported as
                    # fenced-JSON text instead of a report_finding call
                    # before treating this as clean termination -- otherwise
                    # they'd be silently discarded.
                    recovered = _parse_text_findings(response.text or "")
                    if recovered is not None:
                        recovered_findings, recovered_files = recovered
                        for finding in recovered_findings:
                            review_tools.report_finding(
                                file=finding.get("file"),
                                line=finding.get("line"),
                                description=str(finding.get("description", "")),
                                context=finding.get("context"),
                            )
                        if recovered_findings:
                            logger.warning(
                                "Gemini session [%s] returned %d finding(s) as text "
                                "instead of calling report_finding; recovered them "
                                "from response.text",
                                label or "unlabeled",
                                len(recovered_findings),
                            )
                        if recovered_files:
                            files_explored = recovered_files
                    break

                # Append the model's own turn EXACTLY as the API returned
                # it -- NOT hand-rebuilt from `function_calls` -- since
                # Gemini 3.x attaches an opaque `thought_signature` to
                # function-call parts that MUST be echoed back verbatim on
                # the next turn, or multi-turn calls can be rejected.
                # `response.candidates[0].content` is guaranteed non-None here:
                # `response.function_calls` (already truthy, or we wouldn't
                # be in this branch) is itself derived from exactly that
                # attribute by the SDK.
                model_turn_content = response.candidates[0].content  # type: ignore[index]
                assert model_turn_content is not None
                contents.append(model_turn_content)

                # Execute ALL of this turn's calls before appending a
                # single follow-up turn bundling every FunctionResponse --
                # handles multiple simultaneous function calls correctly.
                response_parts: list[types.Part] = []
                finished = False
                for call in function_calls:
                    name = call.name or ""
                    args = dict(call.args or {})
                    progress.tool_calls.append(name)
                    output, is_error = await asyncio.to_thread(_execute_tool_call, name, args)
                    if is_error and name != "finish_review":
                        logger.warning("Tool call failed [%s]: %s", name, output[:200])
                    response_parts.append(
                        types.Part.from_function_response(name=name, response={"result": output})
                    )
                    # A malformed finish_review call (e.g. missing its
                    # required files_explored argument) must NOT terminate
                    # the session -- only a call that actually executed
                    # successfully counts as a real finish.
                    if name == "finish_review" and not is_error:
                        finished = True
                        files_explored = list(args.get("files_explored") or [])

                contents.append(types.Content(role="user", parts=response_parts))

                if finished:
                    break
            else:
                # `for...else`: only reached if every iteration executed a
                # function call and none was finish_review, i.e. exhausted.
                logger.warning(
                    "Gemini session [%s] exhausted its %d-turn budget without a finish_review call",
                    label or "unlabeled",
                    _MAX_TURNS_GEMINI,
                )
                exhausted = True

            result_text = _build_result_text(findings_sink, files_explored)

        finished_at = datetime.now(timezone.utc)
        unique_tool_names = sorted(set(progress.tool_calls))
        cost_usd = progress.cost_usd(model)

        tools_str = f" (tools: {', '.join(unique_tool_names)})" if progress.tool_calls else ""
        logger.info(
            "Gemini agent done [%s]: %d tool calls%s, cost=$%.4f input_tokens=%d cached_tokens=%d output_tokens=%d",
            label or "unlabeled",
            len(progress.tool_calls),
            tools_str,
            cost_usd,
            progress.input_tokens,
            progress.cached_tokens,
            progress.output_tokens,
        )

        return SessionResult(
            result_text=result_text,
            cost_usd=cost_usd,
            duration_seconds=(finished_at - started_at).total_seconds(),
            started_at=started_at,
            finished_at=finished_at,
            tool_call_count=len(progress.tool_calls),
            tool_names=unique_tool_names,
            # No Context7 MCP integration in scope for this runner (see module
            # docstring) -- always 0, never derived from tool_calls.
            context7_call_count=0,
            model=model,
            failure_reason="turn_budget_exhausted" if exhausted else None,
        )
    finally:
        # Every session creates both a sync and async HTTP client (the sync
        # one used only by `client.caches.create()` inside
        # `_maybe_activate_cache`) -- close both on every exit path,
        # including a timeout/cancellation unwinding through this `finally`,
        # or connections leak across the concurrent reviewer fan-out.
        # Combine into a single shielded gather with return_exceptions=True
        # so an error in one close does not abort the other.
        await asyncio.shield(
            asyncio.gather(
                asyncio.to_thread(client.close),
                client.aio.aclose(),
                return_exceptions=True,
            )
        )


async def _run_turns_with_retry(
    *, progress: _Progress, timeout_s: float, **kwargs: Any
) -> SessionResult:
    """``_run_turns`` with at most one retry of a transient turn-0 failure."""
    label = kwargs.get("label") or "unlabeled"
    t0 = asyncio.get_running_loop().time()
    try:
        return await _run_turns(timeout_s=timeout_s, progress=progress, **kwargs)
    except (genai_errors.APIError, httpx.HTTPError) as exc:
        remaining = timeout_s - (asyncio.get_running_loop().time() - t0)
        decision = _retry_decision(exc) if progress.responses == 0 else None
        if decision is None or remaining < timeout_s * _RETRY_MIN_REMAINING_FRACTION:
            raise
        kind, wait = decision
        if wait >= remaining:
            raise
        progress.retried = True
        progress.retry_wait_s = wait
        logger.info(
            "gemini session [%s] retrying after %s (kind=%s) in %.1fs",
            label,
            _failure_detail(exc),
            kind,
            wait,
        )
        await asyncio.sleep(wait)
    return await _run_turns(timeout_s=timeout_s, progress=progress, **kwargs)


def _redact_gemini_inputs(inputs: dict[str, Any], **_: Any) -> dict[str, Any]:
    """LangSmith ``process_inputs`` hook: redact credentials from Settings."""
    if not isinstance(inputs, dict):
        return {}
    s = inputs.get("settings")
    if s is not None:
        return {
            **inputs,
            "settings": f"<Settings project={getattr(s, 'LANGSMITH_PROJECT', '?')}>",
        }
    return dict(inputs)


@traceable(name="pr_review.gemini_session", process_inputs=_redact_gemini_inputs)
async def run_session_gemini(
    *,
    entry: BenchEntry,
    system_prompt: str,
    user_message: str,
    settings: Any,
    label: str = "",
    repo_root: str | None = None,
    timeout_s: float | None = None,
) -> SessionResult:
    """Run one Gemini leaf-reviewer session. Returns a ``SessionResult``
    with the same shape/semantics the Claude Agent SDK path returns.

    Args:
        entry: The resolved bench entry (platform/model/caching) driving
            this session. ``entry.model`` is a registry alias (e.g.
            ``"gemini-frontier"``), resolved here via
            ``argus.llm.models.resolve``.
        settings: An ``argus.config.Settings``-shaped object (or a test
            double exposing the same attributes/properties) -- needs
            ``google_credential`` and, optionally, ``ARGUS_SESSION_TIMEOUT``.
        repo_root: Absolute path to the sandboxed worktree root. Falls
            back to ``argus.runners._REPO_ROOT`` (with a warning) when
            ``None``, exactly like the Claude path's ``_resolve_repo_root``.
        timeout_s: Explicit per-call timeout override, taking priority over
            ``settings.ARGUS_SESSION_TIMEOUT`` when supplied -- the shared
            ``RunnerFn`` bench contract's caller (see ``argus.bench``'s
            ``_gemini_runner`` adapter) always resolves its own timeout
            from the caller's own settings object and must win over this
            function silently re-deriving a possibly-different one. Falls
            back to ``settings.ARGUS_SESSION_TIMEOUT`` (then
            ``argus.runners._SUBPROCESS_TIMEOUT_S``) only when left
            ``None``.

    Timeout: wraps the whole session in ``asyncio.wait_for`` against the
    effective timeout described above. On expiry -- or if the underlying
    SDK call raises an ``httpx``-level timeout exception directly, rather
    than via ``asyncio.wait_for``'s own cancellation (the HTTP-client-level
    timeout in ``_run_turns`` is deliberately shorter than this outer
    deadline, so it fires FIRST and this is the exception that actually
    surfaces in practice) -- returns a ``failure_reason="timeout"`` result instead
    of letting the exception propagate.
    """
    effective_root = _resolve_repo_root(repo_root, "run_session_gemini")
    model = resolve_model_alias(entry.model)
    effective_timeout_s = (
        timeout_s
        if timeout_s is not None
        else getattr(settings, "ARGUS_SESSION_TIMEOUT", _SUBPROCESS_TIMEOUT_S)
    )

    started_at = datetime.now(timezone.utc)
    progress = _Progress()
    try:
        return await asyncio.wait_for(
            _run_turns_with_retry(
                progress=progress,
                entry=entry,
                model=model,
                system_prompt=system_prompt,
                user_message=user_message,
                settings=settings,
                label=label,
                repo_root=effective_root,
                timeout_s=effective_timeout_s,
                started_at=started_at,
            ),
            timeout=effective_timeout_s,
        )
    except (TimeoutError, httpx.TimeoutException) as exc:
        finished_at = datetime.now(timezone.utc)
        logger.warning(
            "Gemini session [%s] timed out after %ss (%s), %d turn(s) and %d tool call(s) done",
            label or "unlabeled",
            effective_timeout_s,
            type(exc).__name__,
            progress.responses,
            len(progress.tool_calls),
        )
        return SessionResult(
            result_text="",
            cost_usd=progress.cost_usd(model),
            duration_seconds=(finished_at - started_at).total_seconds(),
            started_at=started_at,
            finished_at=finished_at,
            tool_call_count=len(progress.tool_calls),
            tool_names=sorted(set(progress.tool_calls)),
            context7_call_count=0,
            model=model,
            failure_reason="timeout",
        )
    except (genai_errors.APIError, httpx.HTTPError) as exc:
        # A genuine, non-timeout SDK/transport failure -- a real API error
        # (auth rejection, rate limit, malformed request, 5xx) via
        # `google.genai.errors.APIError` (the SDK's own base for
        # `ClientError`/`ServerError`), or a lower-level transport failure
        # (connection reset, DNS failure, protocol error -- anything that
        # isn't itself a timeout) via `httpx.HTTPError`. `httpx.TimeoutException`
        # is a subclass of `httpx.HTTPError`, but the more specific except
        # clause above already claims it first, so it can never reach here.
        #
        # Deliberately narrower than a bare `except Exception`: a genuine bug
        # in THIS module's own code (a `TypeError`/`AttributeError` from a
        # programming error, say) is NOT caught here and still propagates,
        # rather than being silently relabeled as an ordinary session
        # failure -- see this function's module-level docstring section
        # this except clause was added to fix for the full rationale.
        #
        # Before this except clause existed, only timeout exceptions were
        # translated into a clean SessionResult; every other exception
        # propagated all the way up to `graph.py`'s blanket per-reviewer
        # exception handling, which swallows it into a 0-finding,
        # `failure_reason=None` result -- indistinguishable from "this
        # reviewer ran fine and genuinely found nothing," with zero visible
        # signal that anything went wrong. Reuses the existing
        # `"worker_crashed"` failure_reason value (already a valid member of
        # `SessionResult.failure_reason`'s Literal type, and of the Postgres/
        # SQLite `agent_runs.failure_reason` CHECK constraints) rather than
        # introducing a new value that would need a new DB migration on both
        # storage backends -- the exact distinction that value already
        # exists to carry ("this session did not complete or produce a
        # trustworthy result, for a reason other than timeout") applies
        # identically here, even though this runner has no literal
        # subprocess "worker" the way the Claude Agent SDK path does.
        finished_at = datetime.now(timezone.utc)
        detail = _failure_detail(exc)
        if progress.last_turn is not None:
            detail = f"{detail}, turn {progress.last_turn}"
        logger.warning(
            "Gemini session [%s] failed after %.1fs (%s, %s%s): %s",
            label or "unlabeled",
            (finished_at - started_at).total_seconds(),
            type(exc).__name__,
            detail,
            ", after one retry" if progress.retried else "",
            _scrub(str(exc)),
        )
        return SessionResult(
            result_text="",
            cost_usd=progress.cost_usd(model),
            duration_seconds=(finished_at - started_at).total_seconds(),
            started_at=started_at,
            finished_at=finished_at,
            tool_call_count=len(progress.tool_calls),
            tool_names=sorted(set(progress.tool_calls)),
            context7_call_count=0,
            model=model,
            failure_reason="worker_crashed",
            failure_detail=detail,
        )
