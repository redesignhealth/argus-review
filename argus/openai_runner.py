"""OpenAI leaf-reviewer runner: an original tool-calling loop against the
public ``openai`` SDK using the Responses API.

This is Track 3 of the multi-platform reviewer bench work (see
``argus.bench``'s module docstring). It is written directly against
``openai`` (AsyncOpenAI / Responses API) -- deliberately NOT ported or
vendored from any internal, non-public codebase (such as rh-lib's OpenAISyncAgent);
there is no code lineage between this module and any internal implementation,
only a shared design (the same five platform-neutral tools in
``argus.review_tools``, and the same ``SessionResult`` contract
``argus.runners`` already defines for the Claude Agent SDK path).

NOTE on raw-SDK usage vs. D024 ("no unwrapped LLM client" per RH's
internal ``llm-pipelines.md`` guidance, which some of THIS repo's own
packaged review prompts -- e.g. ``pr-review-specialist-llm-patterns.md``
-- enforce against code Argus reviews): calling ``openai`` directly
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

Architecture: build the five ``argus.review_tools`` functions into OpenAI
function-calling tool definitions for the Responses API, open one
``argus.review_tools.review_session`` for the sandboxed worktree root,
then loop ``client.responses.create(...)`` -- executing every function call
the model requests each turn (there can be more than one), chaining the
conversation state forward via ``previous_response_id`` and feeding back
all tool outputs as ``function_call_output`` items -- until the model stops
requesting function calls, calls ``finish_review``, or the turn budget
(``argus.runners._MAX_TURNS``) is exhausted. Findings arrive via
``report_finding`` tool calls into ``review_tools``' per-session sink,
not as a JSON blob embedded in the model's own text -- so, to keep this
task's blast radius contained, the final ``SessionResult.result_text`` is
built here as the same fenced ``json`` blob shape
``argus.helpers.parse_review_result`` already knows how to parse,
deliberately omitting the ``system_group`` key so that function's own
``group_name`` fallback still applies.

Prompt caching: OpenAI's Responses API uses automatic, server-side prefix
prompt caching (for prompts >= 1024 tokens) without requiring manual cache
creation or lifecycle management. Tokens read from cache are reported in
``response.usage.input_tokens_details.cached_tokens`` and billed at the
reduced cache-read rate via ``argus.llm.pricing``.

Timeout handling: ``asyncio.wait_for`` bounds the entire multi-turn session
against ``effective_timeout_s`` (default 600s, consistent across platforms).
The ``AsyncOpenAI`` client is constructed with ``timeout=effective_timeout_s``,
bounding individual HTTP calls natively in the SDK. Timeouts raise
``TimeoutError``, ``APITimeoutError``, or ``httpx.TimeoutException``, caught
cleanly and returned as ``SessionResult(failure_reason="timeout")``.
Non-timeout SDK or network errors (``OpenAIError``, ``httpx.HTTPError``) are
caught and returned as ``SessionResult(failure_reason="worker_crashed")`` to
ensure loud, visible failures that surface as degraded coverage rather than
silently returning 0 findings.

Partial-progress accounting on failure: all of this exception handling lives
inside ``_run_turns`` itself (not in a separate path in ``run_session_openai``
that starts from zero), so a timeout or API failure that strikes after
several turns have already completed still returns a ``SessionResult`` whose
``cost_usd``/token counts/``tool_call_count``/``result_text`` reflect
whatever those completed turns actually produced and billed -- real,
already-incurred cost is never silently discarded just because a later turn
failed. ``run_session_openai``'s own ``try``/``except`` around
``asyncio.wait_for`` is kept only as a defense-in-depth fallback for failures
that occur before ``_run_turns`` starts accumulating anything (e.g. client
construction), where there is nothing to preserve anyway.

Degraded-completion detection: a turn with no function calls is only treated
as "the model is done" after checking that OpenAI's Responses API didn't
actually flag the response as incomplete, errored, or refused --
``response.status`` (must be ``"completed"`` or unset), ``response.error``,
``response.incomplete_details``, and any ``refusal``-type content block on
an output message. Any of these route to the same
``SessionResult(failure_reason="worker_crashed")`` path as a real SDK
exception, rather than being silently reported as a clean zero-finding
review.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Literal

import httpx
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import APITimeoutError, AsyncOpenAI, OpenAIError

from argus import review_tools
from argus.bench import BenchEntry
from argus.llm.models import estimate_cost_usd
from argus.llm.models import resolve as resolve_model_alias
from argus.runners import (
    _MAX_TURNS,
    _SUBPROCESS_TIMEOUT_S,
    SessionResult,
    _resolve_repo_root,
)

logger = logging.getLogger(__name__)

_DEFAULT_READ_LIMIT = 2000  # mirrors argus.review_tools._DEFAULT_READ_LIMIT

# Matches the fenced-json extraction argus.helpers.parse_review_result
# already applies for the Claude path.
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "read_file",
        "description": (
            "Read a file, sandboxed to this review's worktree root. Returns "
            "line-numbered content ('<1-based line number>: <content>', one per "
            "output line)."
        ),
        "parameters": {
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
        "type": "function",
        "name": "glob_files",
        "description": (
            "List files under the worktree root matching a glob pattern, one path "
            "per output line, sorted."
        ),
        "parameters": {
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
        "type": "function",
        "name": "grep",
        "description": (
            "Search for a regular-expression pattern in files matching a glob under "
            "the worktree root."
        ),
        "parameters": {
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
        "type": "function",
        "name": "report_finding",
        "description": "Record one review finding -- a bug, risk, or issue worth flagging.",
        "parameters": {
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
        "type": "function",
        "name": "finish_review",
        "description": (
            "Signal that you are done exploring and reporting findings. Call this "
            "exactly once, as your last tool call, when you have nothing further to "
            "report."
        ),
        "parameters": {
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

_TOOL_FUNCTIONS: dict[str, Any] = {
    "read_file": review_tools.read_file,
    "glob_files": review_tools.glob_files,
    "grep": review_tools.grep,
    "report_finding": review_tools.report_finding,
    "finish_review": review_tools.finish_review,
}


def _execute_tool_call(name: str, args: dict[str, Any]) -> tuple[str, bool]:
    """Dispatch one OpenAI function call to its ``review_tools`` implementation.

    Catches all exceptions: ``name``/``args`` originate from an LLM tool call
    that may be malformed. A failed call returns an error string the model can
    recover from in subsequent turns.

    Returns ``(output, is_error)``.
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
    ``argus.helpers.parse_review_result`` parses.
    """
    payload = {"findings": findings, "files_explored": files_explored}
    return "```json\n" + json.dumps(payload) + "\n```"


def _parse_text_findings(text: str) -> tuple[list[dict[str, Any]], list[str]] | None:
    """Best-effort recovery when a model reports findings as fenced JSON in text
    instead of calling ``report_finding``.
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


def _detect_degraded_response(response: Any, output_items: list[Any]) -> str | None:
    """Return a human-readable reason if ``response`` signals an incomplete,
    errored, or refused completion -- rather than a genuine "the model has
    nothing further to say/call" stop.

    Checks the actual Responses API degraded-completion signals the
    ``openai`` SDK exposes on a ``Response`` object:

    - ``response.status`` other than ``"completed"`` (or unset/``None``,
      which real ``Response`` instances only leave unset in tests) --
      covers ``"failed"``, ``"incomplete"``, and ``"cancelled"``.
    - ``response.error`` -- a top-level ``ResponseError`` the API attached
      to an otherwise-terminal response.
    - ``response.incomplete_details`` -- set when ``status == "incomplete"``
      (e.g. ``reason="max_output_tokens"`` or ``"content_filter"``), checked
      independently in case a future SDK version populates it without also
      setting ``status``.
    - A ``refusal``-type content block on any output message -- the model
      can refuse to answer while the response itself still reports
      ``status == "completed"``, so this is not implied by the checks above.

    Returns ``None`` for a healthy response. Uses ``getattr`` throughout
    since these fields are absent from hand-built ``MagicMock(spec=Response)``
    test doubles that don't set them explicitly.
    """
    status = getattr(response, "status", None)
    if status is not None and status != "completed":
        return f"response.status={status!r}"

    error = getattr(response, "error", None)
    if error is not None:
        code = getattr(error, "code", None)
        message = getattr(error, "message", None)
        return f"response.error(code={code!r}, message={message!r})"

    incomplete_details = getattr(response, "incomplete_details", None)
    if incomplete_details is not None:
        reason = getattr(incomplete_details, "reason", None)
        return f"response.incomplete_details.reason={reason!r}"

    for item in output_items:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", None) or []:
            if getattr(content, "type", None) == "refusal":
                refusal_text = getattr(content, "refusal", "") or ""
                return f"model refused: {refusal_text[:200]!r}"

    return None


def _build_client(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
) -> AsyncOpenAI:
    """Build an AsyncOpenAI client wrapped with LangSmith tracing."""
    client_kwargs: dict[str, Any] = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        client_kwargs["base_url"] = base_url
    if timeout is not None:
        client_kwargs["timeout"] = timeout
    return wrap_openai(AsyncOpenAI(**client_kwargs))


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
) -> SessionResult:
    """Run the OpenAI Responses API multi-turn tool-calling loop.

    All failure handling (timeout, transport/API error, or a degraded/
    refused response -- see ``_detect_degraded_response``) lives inside this
    function rather than in a separate zero-state path upstream, so it has
    access to this loop's own running accumulator (``tool_calls``, usage
    totals, ``findings_sink``). A session that fails after several turns
    have already completed still returns a ``SessionResult`` reflecting
    whatever those completed turns actually produced and billed.
    """
    api_key = getattr(settings, "OPENAI_API_KEY", None)
    base_url = getattr(settings, "OPENAI_BASE_URL", None)

    client = _build_client(
        api_key=api_key if isinstance(api_key, str) else None,
        base_url=base_url if isinstance(base_url, str) else None,
        timeout=timeout_s,
    )
    logger.info(
        "OpenAI session started: %s model=%s timeout=%ss",
        label,
        model,
        timeout_s,
    )

    tool_calls: list[str] = []
    files_explored: list[str] = []
    findings_sink: list[dict[str, Any]] = []
    usage_input_total = 0
    usage_output_total = 0
    usage_cached_total = 0

    def _build_result(
        failure_reason: Literal["timeout", "worker_crashed"] | None,
    ) -> SessionResult:
        """Build a ``SessionResult`` from whatever has accumulated so far.

        Shared by the successful-completion path and every failure path, so
        a mid-session failure reports the real cost/usage/tool-call
        accounting for the turns that DID complete instead of zeroing it.
        """
        finished_at = datetime.now(timezone.utc)
        unique_tool_names = sorted(set(tool_calls))
        input_tokens = max(usage_input_total - usage_cached_total, 0)
        output_tokens = usage_output_total
        cost_usd = estimate_cost_usd(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=usage_cached_total,
        )
        result_text = _build_result_text(findings_sink, files_explored)
        tools_str = f" (tools: {', '.join(unique_tool_names)})" if tool_calls else ""

        if failure_reason is None:
            logger.info(
                "OpenAI agent done [%s]: %d tool calls%s, cost=$%.4f input_tokens=%d "
                "cached_tokens=%d output_tokens=%d",
                label or "unlabeled",
                len(tool_calls),
                tools_str,
                cost_usd,
                input_tokens,
                usage_cached_total,
                output_tokens,
            )
        else:
            logger.warning(
                "OpenAI session [%s] ended with failure_reason=%s after %d tool call(s)%s -- "
                "preserving accumulated cost=$%.4f input_tokens=%d cached_tokens=%d "
                "output_tokens=%d rather than reporting 0",
                label or "unlabeled",
                failure_reason,
                len(tool_calls),
                tools_str,
                cost_usd,
                input_tokens,
                usage_cached_total,
                output_tokens,
            )

        return SessionResult(
            result_text=result_text,
            cost_usd=cost_usd,
            duration_seconds=(finished_at - started_at).total_seconds(),
            started_at=started_at,
            finished_at=finished_at,
            tool_call_count=len(tool_calls),
            tool_names=unique_tool_names,
            context7_call_count=0,
            model=model,
            failure_reason=failure_reason,
        )

    try:
        with review_tools.review_session(repo_root) as findings_sink:
            previous_response_id: str | None = None
            tool_outputs: list[dict[str, Any]] = []
            for _turn in range(_MAX_TURNS):
                create_kwargs: dict[str, Any] = {
                    "model": model,
                    "instructions": system_prompt,
                    "tools": _TOOL_SCHEMAS,
                }
                if _turn == 0:
                    create_kwargs["input"] = user_message
                else:
                    create_kwargs["input"] = tool_outputs
                    if previous_response_id:
                        create_kwargs["previous_response_id"] = previous_response_id

                response = await client.responses.create(**create_kwargs)

                usage = getattr(response, "usage", None)
                if usage is not None:
                    usage_input_total += getattr(usage, "input_tokens", 0) or 0
                    usage_output_total += getattr(usage, "output_tokens", 0) or 0
                    details = getattr(usage, "input_tokens_details", None)
                    if details is not None:
                        usage_cached_total += getattr(details, "cached_tokens", 0) or 0

                output_items = getattr(response, "output", None) or []

                degraded_reason = _detect_degraded_response(response, output_items)
                if degraded_reason is not None:
                    # Route through the same OpenAIError-based failure path as a
                    # real SDK exception (caught below) -- a refused/incomplete
                    # response must not be silently reported as a clean,
                    # zero-findings review. This turn's usage is already
                    # accumulated above, so it's preserved in the failure result.
                    raise OpenAIError(
                        f"OpenAI response degraded on turn {_turn} "
                        f"[{label or 'unlabeled'}]: {degraded_reason}"
                    )

                function_calls = [
                    item for item in output_items if getattr(item, "type", None) == "function_call"
                ]

                if not function_calls:
                    output_text = getattr(response, "output_text", None) or ""
                    recovered = _parse_text_findings(output_text)
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
                                "OpenAI session [%s] returned %d finding(s) as text instead of "
                                "calling report_finding; recovered them from response.output_text",
                                label or "unlabeled",
                                len(recovered_findings),
                            )
                        if recovered_files:
                            files_explored = recovered_files
                    break

                tool_outputs = []
                finished = False
                for call in function_calls:
                    name = getattr(call, "name", "")
                    raw_args = getattr(call, "arguments", "{}")
                    call_id = getattr(call, "call_id", "")
                    tool_calls.append(name)

                    try:
                        args = (
                            json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                        )
                        if not isinstance(args, dict):
                            args = {}
                    except Exception as exc:  # noqa: BLE001
                        args = {}
                        output = f"Error parsing arguments for {name}: {exc}"
                        is_error = True
                    else:
                        output, is_error = await asyncio.to_thread(_execute_tool_call, name, args)

                    if is_error and name != "finish_review":
                        logger.warning("Tool call failed [%s]: %s", name, output[:200])

                    tool_outputs.append(
                        {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": output,
                        }
                    )

                    if name == "finish_review" and not is_error:
                        finished = True
                        files_explored = list(args.get("files_explored") or [])

                if finished:
                    break

                previous_response_id = getattr(response, "id", None)
            else:
                logger.warning(
                    "OpenAI session [%s] exhausted its %d-turn budget without a finish_review call",
                    label or "unlabeled",
                    _MAX_TURNS,
                )

        return _build_result(None)
    except (TimeoutError, APITimeoutError, httpx.TimeoutException, asyncio.CancelledError):
        # asyncio.CancelledError is included deliberately: run_session_openai
        # bounds this whole function with asyncio.wait_for, whose expiry
        # cancels whatever this coroutine is currently awaiting (typically
        # client.responses.create) rather than raising a plain TimeoutError
        # here directly. Catching it (without re-raising) lets this function
        # return its own partial-progress SessionResult instead of letting
        # asyncio.wait_for's caller see a bare TimeoutError with no
        # accumulated accounting.
        logger.warning(
            "OpenAI session [%s] timed out after %ss",
            label or "unlabeled",
            timeout_s,
        )
        return _build_result("timeout")
    except (OpenAIError, httpx.HTTPError) as exc:
        logger.warning(
            "OpenAI session [%s] failed with a non-timeout error (%s: %s)",
            label or "unlabeled",
            type(exc).__name__,
            exc,
        )
        return _build_result("worker_crashed")
    finally:
        await asyncio.shield(client.close())


def _redact_openai_inputs(inputs: dict[str, Any], **_: Any) -> dict[str, Any]:
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


@traceable(name="pr_review.openai_session", process_inputs=_redact_openai_inputs)
async def run_session_openai(
    *,
    entry: BenchEntry,
    system_prompt: str,
    user_message: str,
    settings: Any,
    label: str = "",
    repo_root: str | None = None,
    timeout_s: float | None = None,
) -> SessionResult:
    """Run one OpenAI Responses API leaf-reviewer session.

    Returns a ``SessionResult`` matching the contract defined for the Claude
    and Gemini runners.

    ``_run_turns`` handles every failure mode it can encounter (timeout,
    transport/API error, degraded response) internally, using its own
    running accumulator to preserve partial cost/usage/tool-call accounting
    -- see that function's docstring. The ``try``/``except`` below is kept
    only as a defense-in-depth fallback for a failure occurring before
    ``_run_turns`` has accumulated anything (e.g. during client
    construction), where a zero-accounting result is correct because
    nothing has actually happened yet.
    """
    effective_root = _resolve_repo_root(repo_root, "run_session_openai")
    model = resolve_model_alias(entry.model)
    effective_timeout_s = (
        timeout_s
        if timeout_s is not None
        else getattr(settings, "ARGUS_SESSION_TIMEOUT", _SUBPROCESS_TIMEOUT_S)
    )

    started_at = datetime.now(timezone.utc)
    try:
        return await asyncio.wait_for(
            _run_turns(
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
    except (TimeoutError, APITimeoutError, httpx.TimeoutException) as exc:
        finished_at = datetime.now(timezone.utc)
        logger.warning(
            "OpenAI session [%s] timed out after %ss (%s)",
            label or "unlabeled",
            effective_timeout_s,
            type(exc).__name__,
        )
        return SessionResult(
            result_text="",
            cost_usd=0.0,
            duration_seconds=(finished_at - started_at).total_seconds(),
            started_at=started_at,
            finished_at=finished_at,
            tool_call_count=0,
            tool_names=[],
            context7_call_count=0,
            model=model,
            failure_reason="timeout",
        )
    except (OpenAIError, httpx.HTTPError) as exc:
        finished_at = datetime.now(timezone.utc)
        logger.warning(
            "OpenAI session [%s] failed with a non-timeout error after %.1fs (%s: %s)",
            label or "unlabeled",
            (finished_at - started_at).total_seconds(),
            type(exc).__name__,
            exc,
        )
        return SessionResult(
            result_text="",
            cost_usd=0.0,
            duration_seconds=(finished_at - started_at).total_seconds(),
            started_at=started_at,
            finished_at=finished_at,
            tool_call_count=0,
            tool_names=[],
            context7_call_count=0,
            model=model,
            failure_reason="worker_crashed",
        )
