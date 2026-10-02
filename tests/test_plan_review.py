"""Unit tests for plan_review — streamed native structured output + GPT-5.4-mini fallback parse.

Opus occasionally emits invalid JSON escape sequences (e.g.
``\\s``, ``\\p``) inside structured-output string fields. The standard
``with_structured_output`` path crashes because the Anthropic SDK strict-parses
the wire response before we see anything usable. ``plan_review`` instead binds
Anthropic native structured output (``output_config.format`` with a
``json_schema`` derived from ``ReviewPlan``) and streams the raw JSON text;
if Pydantic can't parse it, GPT-5.4-mini re-emits it as valid JSON matching
the schema (same pattern as the writer's phase-2 extraction).

Chunk-shape coverage: the stream collector accepts native text chunks
(string ``content``, list text-content blocks in dict/object/bare-string
forms, and chunk-level ``.text``), while retaining legacy
``tool_call_chunks`` collection for chunks with no text (backwards/mock
compatibility). A chunk exposing BOTH channels must contribute its JSON
exactly once.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic import transform_schema
from pydantic import ValidationError

from argus.graph import PlannerTransientError, plan_review
from argus.pipeline_models import ReviewPlan


_VALID_PLAN_DICT: dict[str, Any] = {
    "system_groups": [
        {
            "name": "api-endpoints",
            "files": ["app/api.py"],
            "conventions": "FastAPI; async only",
            "review_focus": "auth + error handling",
            "specialists_needed": [],
        }
    ],
    "cross_cutting_concerns": ["migration ordering"],
    "file_manifest": [{"path": "app/api.py", "change_type": "modified"}],
}


class _TextChunk:
    """Minimal stand-in for a LangChain AIMessageChunk streaming native
    structured output as plain string ``content``."""

    def __init__(self, content: str | list[Any]) -> None:
        self.content = content


class _BareTextChunk:
    """Stand-in for a chunk with no ``content`` at all, only a chunk-level
    ``.text`` (the collector's last-resort text source)."""

    def __init__(self, text: str) -> None:
        self.text = text


class _TextBlockObj:
    """Object form of a text content block (LangChain TextBlock-like)."""

    def __init__(self, text: str, block_type: str | None = "text") -> None:
        self.text = text
        self.type = block_type


class _ToolCallChunk:
    """Minimal stand-in for a legacy LangChain AIMessageChunk with
    tool_call_chunks (dict entries) and no text content."""

    def __init__(self, args: str, *, index: int = 0) -> None:
        self.tool_call_chunks = [{"name": "ReviewPlan", "args": args, "id": "tc_1", "index": index}]


class _ToolCallEntryObj:
    """Attribute (object) form of a single legacy tool_call_chunks entry."""

    def __init__(self, args: str, *, index: int = 0) -> None:
        self.name = "ReviewPlan"
        self.args = args
        self.id = "tc_1"
        self.index = index


class _ToolCallChunkObjEntries:
    """Legacy chunk whose tool_call_chunks entries are attribute-form objects."""

    def __init__(self, args: str, *, index: int = 0) -> None:
        self.tool_call_chunks = [_ToolCallEntryObj(args, index=index)]


class _DualChannelChunk:
    """Stand-in for a chunk exposing BOTH native text content and legacy
    ``tool_call_chunks`` — used to prove the anti-duplication guard."""

    def __init__(self, text: str, tool_call_chunks: list[dict[str, Any]]) -> None:
        self.content = text
        self.tool_call_chunks = tool_call_chunks


def _stream_factory(chunks: list[Any]) -> Any:
    """Return a mock matching the real call chain:
    init_chat_model(...).bind(output_config=...).with_config(...).astream(messages)."""

    async def _astream(_messages: Any) -> Any:
        for c in chunks:
            yield c

    configured = MagicMock()
    configured.astream = _astream

    bound = MagicMock()
    bound.with_config = MagicMock(return_value=configured)

    base = MagicMock()
    base.bind = MagicMock(return_value=bound)
    return base


def _assert_native_structured_output_bind(factory: Any) -> None:
    """Assert the planner binds Anthropic native structured output — an
    ``output_config.format`` of ``type=json_schema`` with the schema derived
    from ``ReviewPlan`` — and that no forced tool-choice API remains
    (no ``bind_tools``, no ``tool_choice``/``tools`` kwargs)."""
    factory.bind.assert_called_once()
    bind_kwargs = factory.bind.call_args.kwargs
    assert set(bind_kwargs) == {"output_config"}, bind_kwargs
    factory.bind_tools.assert_not_called()

    fmt = bind_kwargs["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"] == transform_schema(ReviewPlan)
    # Readable sanity check that the schema really is ReviewPlan's.
    assert "system_groups" in fmt["schema"]["properties"]


@pytest.fixture(autouse=True)
def _fake_settings() -> Any:
    with patch("argus.graph.get_settings") as m:
        s = MagicMock()
        s.ANTHROPIC_API_KEY = "test-key"
        # Explicit None, not an auto-generated (truthy) MagicMock attribute --
        # and anthropic_credential is a real property on Settings that this
        # bare mock can't derive, so it's stubbed directly to match what the
        # real object would return given ANTHROPIC_API_KEY above.
        s.ANTHROPIC_AUTH_TOKEN = None
        s.anthropic_credential = ("ANTHROPIC_API_KEY", "test-key")
        s.OPENAI_API_KEY = "test-openai-key"
        m.return_value = s
        yield m


@pytest.fixture(autouse=True)
def _fake_prompt() -> Any:
    with patch(
        "argus.graph.fetch_prompt",
        new=AsyncMock(return_value="planner-prompt"),
    ) as m:
        yield m


@pytest.mark.asyncio
async def test_plan_review_parses_clean_streamed_native_text() -> None:
    """Happy path: valid JSON streamed as native text chunks (string
    ``content``) parses to ReviewPlan directly; the GPT-5.4-mini fallback is
    not invoked. Also verifies the planner binds native structured output
    (json_schema derived from ReviewPlan) with no forced tool-choice API,
    and keeps the planner stream run naming/tags."""
    raw = json.dumps(_VALID_PLAN_DICT)
    parts = [raw[:20], raw[20:50], raw[50:]]
    factory = _stream_factory([_TextChunk(p) for p in parts])

    with (
        patch("argus.graph.init_chat_model", return_value=factory),
        patch("argus.graph._extract_plan_with_openai") as fallback,
    ):
        plan = await plan_review(diff="diff", description="desc")

    assert fallback.call_count == 0, "fallback should not fire on valid JSON"
    _assert_native_structured_output_bind(factory)
    # Observability: the stream keeps its planner run name and carries the
    # native_structured_output tag (not the retired bind_tools one).
    with_config_kwargs = factory.bind.return_value.with_config.call_args.kwargs
    assert with_config_kwargs["run_name"] == "planner-phase1-stream"
    assert "native_structured_output" in with_config_kwargs["tags"]
    assert "bind_tools" not in with_config_kwargs["tags"]
    assert len(plan.system_groups) == 1
    assert plan.system_groups[0].name == "api-endpoints"
    assert plan.cross_cutting_concerns == ["migration ordering"]
    assert plan.file_manifest[0].path == "app/api.py"


@pytest.mark.asyncio
async def test_plan_review_collects_text_from_list_content_blocks() -> None:
    """Native structured output can also arrive as ``content`` lists — text
    blocks in dict form (with and without an explicit ``type``), bare strings
    inside the list, and object-form blocks (.text/.type) — plus a chunk-level
    ``.text`` when a chunk has no ``content`` at all. Every text-bearing form
    must contribute its slice, and non-text blocks (e.g. tool_use) must be
    ignored."""
    raw = json.dumps(_VALID_PLAN_DICT)
    chunks = [
        _TextChunk([{"type": "text", "text": raw[:12]}]),  # dict block, typed
        _TextChunk([{"text": raw[12:25]}]),  # dict block, typeless (accepted)
        _TextChunk([{"type": "tool_use", "input": {}}]),  # non-text dict block: ignored
        _TextChunk([raw[25:40]]),  # bare string inside a content list
        _TextChunk([_TextBlockObj(raw[40:55])]),  # object-form text block
        _TextChunk([_TextBlockObj("IGNORE-ME", block_type="tool_use")]),  # ignored
        _BareTextChunk(raw[55:]),  # chunk-level .text fallback
    ]
    factory = _stream_factory(chunks)

    with (
        patch("argus.graph.init_chat_model", return_value=factory),
        patch("argus.graph._extract_plan_with_openai") as fallback,
    ):
        plan = await plan_review(diff="diff", description="desc")

    assert fallback.call_count == 0
    assert plan.system_groups[0].name == "api-endpoints"
    assert "IGNORE-ME" not in json.dumps(plan.model_dump())


@pytest.mark.asyncio
async def test_plan_review_collects_json_once_when_chunk_has_both_text_and_tool_chunks() -> None:
    """Anti-duplication: a chunk exposing BOTH native text and legacy
    ``tool_call_chunks`` must contribute its JSON exactly once — the text
    channel wins and that same chunk's tool-call entries are skipped
    entirely, including its non-zero-index entry, which must not even trip
    the dropped-index warning. If both channels were collected, the raw JSON
    would be the plan duplicated back-to-back and the strict parse would
    fail over into the repair fallback."""
    raw = json.dumps(_VALID_PLAN_DICT)
    tool_chunks = [
        {"name": "ReviewPlan", "args": raw, "id": "tc_1", "index": 0},
        {"name": "ReviewPlan", "args": "!!!GARBAGE!!!", "id": "tc_2", "index": 1},
    ]
    factory = _stream_factory([_DualChannelChunk(raw, tool_chunks)])

    with (
        patch("argus.graph.init_chat_model", return_value=factory),
        patch("argus.graph._extract_plan_with_openai") as fallback,
        patch("argus.graph.logger") as mock_log,
    ):
        plan = await plan_review(diff="diff", description="desc")

    assert fallback.call_count == 0, "dual-channel chunk must be collected exactly once"
    assert plan.system_groups[0].name == "api-endpoints"
    warn_calls = [str(c) for c in mock_log.warning.call_args_list]
    assert not any("planner_multiple_tool_calls=true" in s for s in warn_calls), warn_calls


@pytest.mark.asyncio
async def test_plan_review_collects_legacy_tool_chunks_only_when_no_text() -> None:
    """Backwards/mock compatibility: chunks with no text content still
    contribute their JSON via legacy ``tool_call_chunks`` (dict and attribute
    entry forms), interleaved with native text chunks in the same stream."""
    raw = json.dumps(_VALID_PLAN_DICT)
    chunks = [
        _ToolCallChunk(raw[:20]),
        _TextChunk(raw[20:40]),
        _ToolCallChunkObjEntries(raw[40:60]),
        _ToolCallChunk(raw[60:]),
    ]
    factory = _stream_factory(chunks)

    with (
        patch("argus.graph.init_chat_model", return_value=factory),
        patch("argus.graph._extract_plan_with_openai") as fallback,
    ):
        plan = await plan_review(diff="diff", description="desc")

    assert fallback.call_count == 0
    assert plan.system_groups[0].name == "api-endpoints"


@pytest.mark.asyncio
async def test_plan_review_falls_back_on_invalid_escape() -> None:
    """Regression: an invalid \\s escape in a string field must
    trigger the GPT-5.4-mini extraction fallback rather than raising.
    Also asserts the structured warning log marker fires for observability."""
    plan = dict(_VALID_PLAN_DICT)
    plan["system_groups"] = [
        {
            "name": "api-endpoints",
            "files": ["app/api.py"],
            "conventions": r"FastAPI; use \s for whitespace matching in regex",
            "review_focus": "auth",
            "specialists_needed": [],
        }
    ]
    raw = json.dumps(plan).replace(r"\\s", r"\s")

    with pytest.raises(json.JSONDecodeError):
        json.loads(raw)  # sanity: genuinely malformed

    expected = ReviewPlan.model_validate(_VALID_PLAN_DICT)
    with (
        patch(
            "argus.graph.init_chat_model",
            return_value=_stream_factory([_TextChunk(raw)]),
        ),
        patch("argus.graph._extract_plan_with_openai", return_value=expected) as fallback,
        patch("argus.graph.logger") as mock_log,
    ):
        result = await plan_review(diff="diff", description="desc")

    assert fallback.call_count == 1
    assert fallback.call_args.args[0] == raw
    assert result is expected
    warn_calls = [str(c) for c in mock_log.warning.call_args_list]
    assert any("planner_json_parse_fallback=true" in s for s in warn_calls), warn_calls


@pytest.mark.asyncio
async def test_plan_review_falls_back_on_schema_validation_error() -> None:
    """A well-formed JSON blob that doesn't match ReviewPlan (missing required
    fields) triggers ValidationError — the same fallback should fire."""
    # Valid JSON, but not a valid ReviewPlan: missing system_groups etc.
    bad_but_parseable = '{"unrelated": []}'

    expected = ReviewPlan.model_validate(_VALID_PLAN_DICT)
    with (
        patch(
            "argus.graph.init_chat_model",
            return_value=_stream_factory([_TextChunk(bad_but_parseable)]),
        ),
        patch("argus.graph._extract_plan_with_openai", return_value=expected) as fallback,
    ):
        result = await plan_review(diff="diff", description="desc")

    assert fallback.call_count == 1
    assert result is expected


@pytest.mark.asyncio
async def test_plan_review_ignores_non_zero_index_chunks() -> None:
    """Defense-in-depth on the legacy (text-less) collection path: if the
    model emits extra tool-call slots beyond index 0, their args must not be
    concatenated into the primary plan JSON, AND the observability warning
    must fire so operators see the event in log aggregation."""
    raw = json.dumps(_VALID_PLAN_DICT)
    # Add a chunk at index=1 with garbage — must be ignored.
    extras = [_ToolCallChunk("!!!GARBAGE!!!", index=1)]
    factory = _stream_factory([_ToolCallChunk(raw)] + extras)

    with (
        patch("argus.graph.init_chat_model", return_value=factory),
        patch("argus.graph._extract_plan_with_openai") as fallback,
        patch("argus.graph.logger") as mock_log,
    ):
        plan = await plan_review(diff="diff", description="desc")

    assert fallback.call_count == 0
    assert plan.system_groups[0].name == "api-endpoints"
    warn_calls = [str(c) for c in mock_log.warning.call_args_list]
    assert any("planner_multiple_tool_calls=true" in s for s in warn_calls), warn_calls


@pytest.mark.asyncio
async def test_plan_review_raises_when_no_structured_output() -> None:
    """If the model streams nothing collectible, surface that explicitly
    rather than silently invoking the fallback on an empty string. Asserts
    the error is logged before raising for observability."""
    with (
        patch("argus.graph.init_chat_model", return_value=_stream_factory([])),
        patch("argus.graph._extract_plan_with_openai") as fallback,
        patch("argus.graph.logger") as mock_log,
    ):
        with pytest.raises(ValueError, match="no structured output"):
            await plan_review(diff="diff", description="desc")

    assert fallback.call_count == 0
    assert mock_log.error.called, "no-structured-output path must log before raising"


@pytest.mark.asyncio
async def test_plan_review_raises_transient_on_truncated_output() -> None:
    """Truncated stream (not ending with `}`) must raise PlannerTransientError
    — the retry-eligible class — rather than passing partial JSON to the
    repair fallback where it would silently succeed with a hallucinated
    plan. PlannerTransientError is a ValueError subclass so existing
    ValueError matchers still work, but the plan node's RetryPolicy
    scopes retry_on=(PlannerTransientError,) to this path only (not to
    the permanent no-structured-output failure)."""
    truncated = '{"system_groups": [{"name": "x", "files"'  # cut off mid-field

    # Import fresh from sys.modules in case graph was reloaded by another test.
    from argus.graph import PlannerTransientError as _PlannerTransientError

    with (
        patch(
            "argus.graph.init_chat_model",
            return_value=_stream_factory([_TextChunk(truncated)]),
        ),
        patch("argus.graph._extract_plan_with_openai") as fallback,
        patch("argus.graph.logger") as mock_log,
    ):
        with pytest.raises(_PlannerTransientError, match="truncated"):
            await plan_review(diff="diff", description="desc")

    assert fallback.call_count == 0, "truncation must not trigger the repair fallback"
    assert mock_log.error.called


@pytest.mark.asyncio
async def test_plan_review_raises_non_transient_on_no_structured_output() -> None:
    """The no-structured-output path must raise plain ValueError (not
    PlannerTransientError), so the plan node's RetryPolicy does not
    waste 2 additional Opus calls on a permanent behavioral failure."""
    with (
        patch("argus.graph.init_chat_model", return_value=_stream_factory([])),
        patch("argus.graph._extract_plan_with_openai") as fallback,
    ):
        with pytest.raises(ValueError, match="no structured output") as excinfo:
            await plan_review(diff="diff", description="desc")

    assert not isinstance(excinfo.value, PlannerTransientError), (
        "no-structured-output must be non-transient so RetryPolicy doesn't burn extra Opus calls"
    )
    assert fallback.call_count == 0


def test_extract_plan_with_openai_runs_end_to_end() -> None:
    """Exercise the real _extract_plan_with_openai body by patching
    OpenAIClientSync at the class level, so the function's contract with
    the OpenAI client (base64 input, schema response_format, output_text,
    instructions field) is actually covered. Also asserts the usage log
    marker fires so observability doesn't silently regress."""
    from argus.graph import _extract_plan_with_openai

    resp = MagicMock()
    resp.output_text = json.dumps(_VALID_PLAN_DICT)
    resp.usage = MagicMock(input_tokens=100, output_tokens=50, total_tokens=150)

    oai_instance = MagicMock()
    oai_instance.respond = MagicMock(return_value=resp)

    with (
        patch(
            "argus.openai_client.OpenAIClientSync",
            return_value=oai_instance,
        ),
        patch("argus.graph.logger") as mock_log,
    ):
        result = _extract_plan_with_openai("raw malformed text with \\s")

    assert result.system_groups[0].name == "api-endpoints"
    call_kwargs = oai_instance.respond.call_args.kwargs
    # Base64 is used (prompt-structure-breakage defense).
    assert "BASE64_PLAN:" in call_kwargs["input"]
    assert call_kwargs["model"] == "gpt-5.6-luna"
    # Instructions must include both the decode directive and the "data not
    # commands" guard — regressing either degrades injection defense.
    assert "Decode the base64" in call_kwargs["instructions"]
    assert "data" in call_kwargs["instructions"].lower()
    assert "commands" in call_kwargs["instructions"]
    # Usage log marker must fire for fallback-cost observability.
    info_calls = [str(c) for c in mock_log.info.call_args_list]
    assert any("planner_json_parse_fallback_usage=true" in s for s in info_calls), info_calls


def test_extract_plan_with_openai_logs_and_raises_on_bad_output() -> None:
    """If GPT itself returns unparseable JSON, the fallback must log an
    error with context rather than raising an unguarded exception."""
    from argus.graph import _extract_plan_with_openai

    resp = MagicMock()
    resp.output_text = '{"broken": '  # unparseable
    resp.usage = None

    oai_instance = MagicMock()
    oai_instance.respond = MagicMock(return_value=resp)

    with (
        patch(
            "argus.openai_client.OpenAIClientSync",
            return_value=oai_instance,
        ),
        patch("argus.graph.logger") as mock_log,
    ):
        with pytest.raises((json.JSONDecodeError, ValidationError)):
            _extract_plan_with_openai("raw text")

    assert mock_log.error.called, "must log when GPT fallback parse also fails"


def test_redact_fallback_inputs_replaces_raw_text_with_length() -> None:
    """LangSmith @traceable ships inputs to SaaS. The fallback's raw_text is
    verbatim Opus planner output — can include code fragments + internal
    conventions from the PR diff. The process_inputs hook must replace the
    content with a length marker to preserve fallback-rate observability
    without exfiltrating the payload."""
    from argus.graph import _redact_fallback_inputs

    redacted = _redact_fallback_inputs({"raw_text": "secret internal plan"})
    assert redacted["raw_text"] == "<20 chars redacted>"
    # Empty-input edge case must not crash.
    assert _redact_fallback_inputs({})["raw_text"] == "<0 chars redacted>"


def test_redact_fallback_outputs_emits_shape_only() -> None:
    """The repaired ReviewPlan echoes content derived from the PR diff
    (conventions text, review_focus strings, file paths). For parity with
    the input redaction, process_outputs must emit only the plan shape —
    not the actual strings — so LangSmith SaaS traces can't leak
    code-derived content."""
    from argus.graph import _redact_fallback_outputs

    plan = ReviewPlan.model_validate(_VALID_PLAN_DICT)
    out = _redact_fallback_outputs(plan)
    assert out == {
        "fallback_output": {
            "system_groups_count": 1,
            "cross_cutting_concerns_count": 1,
            "file_manifest_count": 1,
        }
    }
    # None of the original content must appear in the redacted output.
    dumped = json.dumps(out)
    assert "FastAPI" not in dumped  # from conventions
    assert "auth + error handling" not in dumped  # from review_focus
    assert "migration ordering" not in dumped  # from cross_cutting_concerns
    assert "app/api.py" not in dumped  # from files / file_manifest

    # Unknown-shape input must redact entirely rather than leak.
    assert _redact_fallback_outputs("something unexpected") == {"fallback_output": "<redacted>"}
    # LangSmith sometimes wraps returned values under "output".
    wrapped = _redact_fallback_outputs({"output": plan})
    assert wrapped["fallback_output"]["system_groups_count"] == 1
