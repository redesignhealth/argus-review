"""Tests for argus.openai_runner: the real OpenAI Responses API leaf-reviewer runner.

Mocks the openai client entirely -- no real API calls. Covers:
single-turn termination with no tool calls, multi-turn tool-calling with
report_finding accumulating into the review_tools findings sink and
appearing in the final result_text JSON blob, multiple simultaneous
function calls handled in one turn, usage/cost accumulation across turns,
the timeout path returning failure_reason="timeout" rather than raising,
non-timeout API and transport failures returning failure_reason="worker_crashed",
fenced-JSON text recovery, and client cleanup on exit.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseUsage,
)
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from argus.bench import BenchEntry
from argus.llm.models import resolve as resolve_alias
from argus.openai_runner import _redact_openai_inputs, run_session_openai

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_entry(role: str = "system-generalist", model: str = "gpt-mini") -> BenchEntry:
    return BenchEntry(
        role=role,
        platform="openai-responses",
        model=model,
        prompt_name="pr-review-subagent",
    )


def _make_settings(session_timeout: float = 300) -> MagicMock:
    settings = MagicMock()
    settings.OPENAI_API_KEY = "test-openai-key"
    settings.ARGUS_SESSION_TIMEOUT = session_timeout
    return settings


def _make_usage(
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_tokens: int = 0,
) -> ResponseUsage:
    return ResponseUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=cached_tokens),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )


def _make_function_call(
    name: str, args: dict[str, Any] | str, call_id: str = "call_1"
) -> ResponseFunctionToolCall:
    args_str = json.dumps(args) if isinstance(args, dict) else args
    return ResponseFunctionToolCall(
        call_id=call_id,
        name=name,
        arguments=args_str,
        type="function_call",
    )


def _make_response(
    calls: list[tuple[str, dict[str, Any]]] | None = None,
    usage: ResponseUsage | None = None,
    text: str = "",
    resp_id: str = "resp_123",
) -> MagicMock:
    """Build a mock Response object for the Responses API."""
    response = MagicMock(spec=Response)
    response.id = resp_id
    response.usage = usage
    response.output_text = text

    output_items: list[Any] = []
    for i, (name, args) in enumerate(calls or []):
        output_items.append(_make_function_call(name, args, call_id=f"call_{i + 1}"))
    if text and not output_items:
        output_items.append(
            ResponseOutputMessage(
                id=f"{resp_id}_msg",
                type="message",
                role="assistant",
                status="completed",
                content=[ResponseOutputText(text=text, type="output_text", annotations=[])],
            )
        )
    response.output = output_items
    return response


def _make_fake_client(responses: list[Any]) -> MagicMock:
    """Build a fake AsyncOpenAI client with .responses.create and .close."""
    client = MagicMock()
    client.responses.create = AsyncMock(side_effect=responses)
    client.close = AsyncMock()
    return client


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestOpenAIRunnerBasicLoop:
    async def test_no_function_calls_terminates_immediately(self, tmp_path: Any) -> None:
        """A response with no tool calls terminates the loop after 1 turn."""
        client = _make_fake_client([_make_response(calls=[], text="Looks good!")])
        entry = _make_entry()

        with patch("argus.openai_runner._build_client", return_value=client):
            result = await run_session_openai(
                entry=entry,
                system_prompt="system instructions",
                user_message="review this diff",
                settings=_make_settings(),
                repo_root=str(tmp_path),
            )

        assert result.failure_reason is None
        assert result.tool_call_count == 0
        assert result.tool_names == []
        assert client.responses.create.call_count == 1
        call_kwargs = client.responses.create.call_args.kwargs
        assert call_kwargs["instructions"] == "system instructions"
        assert call_kwargs["input"] == "review this diff"
        assert call_kwargs["model"] == resolve_alias("gpt-mini")
        assert "tools" in call_kwargs

    async def test_report_finding_accumulates_into_sink_and_result_text(
        self, tmp_path: Any
    ) -> None:
        """Calling report_finding populates review_tools sink and final JSON blob."""
        finding_args = {
            "file": "foo.py",
            "line": 42,
            "description": "SQL injection risk",
            "context": "SELECT * FROM users WHERE id = " + "x",
        }
        finish_args = {"files_explored": ["foo.py"]}

        responses = [
            _make_response(calls=[("report_finding", finding_args)], resp_id="resp_1"),
            _make_response(calls=[("finish_review", finish_args)], resp_id="resp_2"),
        ]
        client = _make_fake_client(responses)
        entry = _make_entry()

        with patch("argus.openai_runner._build_client", return_value=client):
            result = await run_session_openai(
                entry=entry,
                system_prompt="system",
                user_message="user",
                settings=_make_settings(),
                repo_root=str(tmp_path),
            )

        assert result.failure_reason is None
        assert result.tool_call_count == 2
        assert sorted(result.tool_names) == ["finish_review", "report_finding"]

        # Result text parses into the expected finding
        from argus.helpers import parse_review_result

        parsed = parse_review_result(result.result_text, "test-group")
        assert len(parsed.findings) == 1
        assert parsed.findings[0].file == "foo.py"
        assert parsed.findings[0].line == 42
        assert parsed.findings[0].description == "SQL injection risk"
        assert parsed.files_explored == ["foo.py"]

        # Check turn 2 passed previous_response_id
        turn2_kwargs = client.responses.create.call_args_list[1].kwargs
        assert turn2_kwargs["previous_response_id"] == "resp_1"
        assert turn2_kwargs["input"][0]["type"] == "function_call_output"

    async def test_finish_review_stops_the_loop_even_with_turns_remaining(
        self, tmp_path: Any
    ) -> None:
        """finish_review immediately terminates the turn loop."""
        responses = [
            _make_response(calls=[("finish_review", {"files_explored": ["a.py"]})]),
            # Trailing unused response -- should never be called
            _make_response(calls=[]),
        ]
        client = _make_fake_client(responses)
        entry = _make_entry()

        with patch("argus.openai_runner._build_client", return_value=client):
            result = await run_session_openai(
                entry=entry,
                system_prompt="system",
                user_message="user",
                settings=_make_settings(),
                repo_root=str(tmp_path),
            )

        assert result.failure_reason is None
        assert client.responses.create.call_count == 1
        assert result.tool_names == ["finish_review"]

    async def test_multiple_calls_in_one_turn_all_executed_and_bundled(self, tmp_path: Any) -> None:
        """Simultaneous function calls in a single turn are all executed and returned."""
        target = tmp_path / "hello.py"
        target.write_text("print('hello world')\n")

        responses = [
            _make_response(
                calls=[
                    ("read_file", {"path": "hello.py", "offset": 0, "limit": 10}),
                    ("report_finding", {"file": "hello.py", "line": 1, "description": "issue"}),
                ],
                resp_id="resp_1",
            ),
            _make_response(
                calls=[("finish_review", {"files_explored": ["hello.py"]})],
                resp_id="resp_2",
            ),
        ]
        client = _make_fake_client(responses)
        entry = _make_entry()

        with patch("argus.openai_runner._build_client", return_value=client):
            result = await run_session_openai(
                entry=entry,
                system_prompt="system",
                user_message="user",
                settings=_make_settings(),
                repo_root=str(tmp_path),
            )

        assert result.failure_reason is None
        assert result.tool_call_count == 3
        assert sorted(result.tool_names) == ["finish_review", "read_file", "report_finding"]

        # Turn 2 input should have 2 function_call_output items
        turn2_kwargs = client.responses.create.call_args_list[1].kwargs
        outputs = turn2_kwargs["input"]
        assert len(outputs) == 2
        assert outputs[0]["type"] == "function_call_output"
        assert outputs[0]["call_id"] == "call_1"
        assert "hello world" in outputs[0]["output"]
        assert outputs[1]["type"] == "function_call_output"
        assert outputs[1]["call_id"] == "call_2"

    async def test_exhausting_max_turns_still_returns_a_result(self, tmp_path: Any) -> None:
        """Exhausting _MAX_TURNS stops and builds whatever findings were reported."""
        finding_call = ("report_finding", {"file": "f.py", "line": 1, "description": "d"})
        responses = [_make_response(calls=[finding_call], resp_id=f"r_{i}") for i in range(35)]
        client = _make_fake_client(responses)
        entry = _make_entry()

        with patch("argus.openai_runner._build_client", return_value=client):
            result = await run_session_openai(
                entry=entry,
                system_prompt="system",
                user_message="user",
                settings=_make_settings(),
                repo_root=str(tmp_path),
            )

        assert result.failure_reason is None
        from argus.runners import _MAX_TURNS

        assert client.responses.create.call_count == _MAX_TURNS
        from argus.helpers import parse_review_result

        parsed = parse_review_result(result.result_text, "test-group")
        assert len(parsed.findings) == _MAX_TURNS


class TestOpenAIRunnerUsageAndCost:
    async def test_usage_summed_across_every_turn_not_just_the_last(self, tmp_path: Any) -> None:
        """Usage tokens accumulate additively across all turns."""
        u1 = _make_usage(input_tokens=1000, output_tokens=100, cached_tokens=200)
        u2 = _make_usage(input_tokens=1500, output_tokens=150, cached_tokens=500)

        responses = [
            _make_response(
                calls=[("report_finding", {"file": "x.py", "line": 1, "description": "d"})],
                usage=u1,
                resp_id="r1",
            ),
            _make_response(
                calls=[("finish_review", {"files_explored": ["x.py"]})],
                usage=u2,
                resp_id="r2",
            ),
        ]
        client = _make_fake_client(responses)
        entry = _make_entry(model="gpt-mini")

        with patch("argus.openai_runner._build_client", return_value=client):
            result = await run_session_openai(
                entry=entry,
                system_prompt="system",
                user_message="user",
                settings=_make_settings(),
                repo_root=str(tmp_path),
            )

        assert result.failure_reason is None
        assert result.cost_usd > 0.0

        from argus.llm.models import estimate_cost_usd

        # Total input = 2500, total cached = 700, total output = 250
        # uncached input = 2500 - 700 = 1800
        expected_cost = estimate_cost_usd(
            model=resolve_alias("gpt-mini"),
            input_tokens=1800,
            output_tokens=250,
            cached_input_tokens=700,
        )
        assert result.cost_usd == pytest.approx(expected_cost)

    async def test_missing_usage_metadata_on_some_turns_does_not_crash(self, tmp_path: Any) -> None:
        """If response.usage is None on a turn, runner does not crash."""
        responses = [
            _make_response(
                calls=[("report_finding", {"file": "x.py", "line": 1, "description": "d"})],
                usage=None,
                resp_id="r1",
            ),
            _make_response(
                calls=[("finish_review", {"files_explored": []})],
                usage=_make_usage(input_tokens=500, output_tokens=50),
                resp_id="r2",
            ),
        ]
        client = _make_fake_client(responses)
        entry = _make_entry()

        with patch("argus.openai_runner._build_client", return_value=client):
            result = await run_session_openai(
                entry=entry,
                system_prompt="system",
                user_message="user",
                settings=_make_settings(),
                repo_root=str(tmp_path),
            )

        assert result.failure_reason is None
        assert result.cost_usd > 0.0


class TestOpenAIRunnerErrorHandling:
    async def test_unknown_tool_name_reports_error_and_continues(self, tmp_path: Any) -> None:
        """An unknown tool name returns an error string to model without crashing loop."""
        responses = [
            _make_response(calls=[("non_existent_tool", {"foo": "bar"})], resp_id="r1"),
            _make_response(calls=[("finish_review", {"files_explored": []})], resp_id="r2"),
        ]
        client = _make_fake_client(responses)
        entry = _make_entry()

        with patch("argus.openai_runner._build_client", return_value=client):
            result = await run_session_openai(
                entry=entry,
                system_prompt="system",
                user_message="user",
                settings=_make_settings(),
                repo_root=str(tmp_path),
            )

        assert result.failure_reason is None
        turn2_kwargs = client.responses.create.call_args_list[1].kwargs
        tool_output = turn2_kwargs["input"][0]["output"]
        assert "Error: unknown tool 'non_existent_tool'" in tool_output

    async def test_finish_review_missing_required_arg_does_not_end_session(
        self, tmp_path: Any
    ) -> None:
        """A malformed finish_review (missing files_explored) produces error and does not exit."""
        responses = [
            _make_response(calls=[("finish_review", {})], resp_id="r1"),  # missing files_explored
            _make_response(
                calls=[("finish_review", {"files_explored": ["good.py"]})], resp_id="r2"
            ),
        ]
        client = _make_fake_client(responses)
        entry = _make_entry()

        with patch("argus.openai_runner._build_client", return_value=client):
            result = await run_session_openai(
                entry=entry,
                system_prompt="system",
                user_message="user",
                settings=_make_settings(),
                repo_root=str(tmp_path),
            )

        assert result.failure_reason is None
        assert client.responses.create.call_count == 2
        from argus.helpers import parse_review_result

        parsed = parse_review_result(result.result_text, "test-group")
        assert parsed.files_explored == ["good.py"]

    async def test_invalid_json_args_reports_error_and_continues(self, tmp_path: Any) -> None:
        """Invalid JSON string in tool arguments produces error output."""
        bad_call = ResponseFunctionToolCall(
            call_id="call_bad",
            name="read_file",
            arguments="invalid{json",
            type="function_call",
        )
        resp1 = MagicMock(spec=Response)
        resp1.id = "r1"
        resp1.output = [bad_call]
        resp1.output_text = ""
        resp1.usage = None

        resp2 = _make_response(calls=[("finish_review", {"files_explored": []})], resp_id="r2")
        client = _make_fake_client([resp1, resp2])
        entry = _make_entry()

        with patch("argus.openai_runner._build_client", return_value=client):
            result = await run_session_openai(
                entry=entry,
                system_prompt="system",
                user_message="user",
                settings=_make_settings(),
                repo_root=str(tmp_path),
            )

        assert result.failure_reason is None
        turn2_kwargs = client.responses.create.call_args_list[1].kwargs
        assert "Error parsing arguments for read_file" in turn2_kwargs["input"][0]["output"]

    async def test_response_text_fenced_json_findings_recovered_when_no_function_calls(
        self, tmp_path: Any
    ) -> None:
        """When model outputs fenced JSON text instead of report_finding, it is recovered."""
        fenced_json = (
            "```json\n"
            '{\n  "findings": [{"file": "recovered.py", "line": 10, "description": "rec"}],\n'
            '  "files_explored": ["recovered.py"]\n'
            "}\n```"
        )
        client = _make_fake_client([_make_response(calls=[], text=fenced_json)])
        entry = _make_entry()

        with patch("argus.openai_runner._build_client", return_value=client):
            result = await run_session_openai(
                entry=entry,
                system_prompt="system",
                user_message="user",
                settings=_make_settings(),
                repo_root=str(tmp_path),
            )

        assert result.failure_reason is None
        from argus.helpers import parse_review_result

        parsed = parse_review_result(result.result_text, "test-group")
        assert len(parsed.findings) == 1
        assert parsed.findings[0].file == "recovered.py"
        assert parsed.files_explored == ["recovered.py"]

    async def test_plain_prose_text_with_no_function_calls_yields_no_findings(
        self, tmp_path: Any
    ) -> None:
        """Plain conversational text with no tool calls terminates with 0 findings."""
        client = _make_fake_client(
            [_make_response(calls=[], text="I reviewed everything and all looks great!")]
        )
        entry = _make_entry()

        with patch("argus.openai_runner._build_client", return_value=client):
            result = await run_session_openai(
                entry=entry,
                system_prompt="system",
                user_message="user",
                settings=_make_settings(),
                repo_root=str(tmp_path),
            )

        assert result.failure_reason is None
        from argus.helpers import parse_review_result

        parsed = parse_review_result(result.result_text, "test-group")
        assert len(parsed.findings) == 0


class TestOpenAIRunnerTimeoutsAndFailures:
    async def test_timeout_returns_timed_out_result_instead_of_raising(self, tmp_path: Any) -> None:
        """asyncio.TimeoutError translates into failure_reason='timeout'."""

        async def _slow_create(**_: Any) -> Any:
            await asyncio.sleep(5.0)
            return _make_response(calls=[])

        client = MagicMock()
        client.responses.create = AsyncMock(side_effect=_slow_create)
        client.close = AsyncMock()

        entry = _make_entry()
        with patch("argus.openai_runner._build_client", return_value=client):
            result = await run_session_openai(
                entry=entry,
                system_prompt="system",
                user_message="user",
                settings=_make_settings(),
                repo_root=str(tmp_path),
                timeout_s=0.05,
            )

        assert result.failure_reason == "timeout"
        assert result.timed_out is True
        assert result.result_text == ""
        assert result.tool_call_count == 0

    async def test_api_timeout_error_returns_timed_out_result(self, tmp_path: Any) -> None:
        """openai.APITimeoutError from SDK translates into failure_reason='timeout'."""
        client = MagicMock()
        client.responses.create = AsyncMock(side_effect=APITimeoutError(request=MagicMock()))
        client.close = AsyncMock()

        entry = _make_entry()
        with patch("argus.openai_runner._build_client", return_value=client):
            result = await run_session_openai(
                entry=entry,
                system_prompt="system",
                user_message="user",
                settings=_make_settings(),
                repo_root=str(tmp_path),
            )

        assert result.failure_reason == "timeout"
        assert result.timed_out is True

    async def test_httpx_timeout_exception_returns_timed_out_result(self, tmp_path: Any) -> None:
        """httpx.TimeoutException translates into failure_reason='timeout'."""
        client = MagicMock()
        client.responses.create = AsyncMock(side_effect=httpx.ReadTimeout("read timeout"))
        client.close = AsyncMock()

        entry = _make_entry()
        with patch("argus.openai_runner._build_client", return_value=client):
            result = await run_session_openai(
                entry=entry,
                system_prompt="system",
                user_message="user",
                settings=_make_settings(),
                repo_root=str(tmp_path),
            )

        assert result.failure_reason == "timeout"
        assert result.timed_out is True

    async def test_openai_api_error_produces_visible_failure_not_silent_result(
        self, tmp_path: Any
    ) -> None:
        """A 500/APIError translates into failure_reason='worker_crashed'."""
        client = MagicMock()
        client.responses.create = AsyncMock(
            side_effect=APIError(message="internal server error", request=MagicMock(), body=None)
        )
        client.close = AsyncMock()

        entry = _make_entry()
        with patch("argus.openai_runner._build_client", return_value=client):
            result = await run_session_openai(
                entry=entry,
                system_prompt="system",
                user_message="user",
                settings=_make_settings(),
                repo_root=str(tmp_path),
            )

        assert result.failure_reason == "worker_crashed"
        assert result.result_text == ""

    async def test_openai_rate_limit_error_produces_visible_failure(self, tmp_path: Any) -> None:
        """RateLimitError translates into failure_reason='worker_crashed'."""
        client = MagicMock()
        client.responses.create = AsyncMock(
            side_effect=RateLimitError(
                message="rate limit exceeded",
                response=MagicMock(status_code=429),
                body=None,
            )
        )
        client.close = AsyncMock()

        entry = _make_entry()
        with patch("argus.openai_runner._build_client", return_value=client):
            result = await run_session_openai(
                entry=entry,
                system_prompt="system",
                user_message="user",
                settings=_make_settings(),
                repo_root=str(tmp_path),
            )

        assert result.failure_reason == "worker_crashed"

    async def test_transport_level_connect_error_produces_visible_failure(
        self, tmp_path: Any
    ) -> None:
        """Network/connection errors produce failure_reason='worker_crashed'."""
        client = MagicMock()
        client.responses.create = AsyncMock(side_effect=APIConnectionError(request=MagicMock()))
        client.close = AsyncMock()

        entry = _make_entry()
        with patch("argus.openai_runner._build_client", return_value=client):
            result = await run_session_openai(
                entry=entry,
                system_prompt="system",
                user_message="user",
                settings=_make_settings(),
                repo_root=str(tmp_path),
            )

        assert result.failure_reason == "worker_crashed"

    async def test_genuine_bug_still_propagates_rather_than_being_mislabeled(
        self, tmp_path: Any
    ) -> None:
        """A coding bug (TypeError/AttributeError) must propagate, not be swallowed."""
        client = MagicMock()
        client.responses.create = AsyncMock(side_effect=TypeError("coding bug"))
        client.close = AsyncMock()

        entry = _make_entry()
        with patch("argus.openai_runner._build_client", return_value=client):
            with pytest.raises(TypeError, match="coding bug"):
                await run_session_openai(
                    entry=entry,
                    system_prompt="system",
                    user_message="user",
                    settings=_make_settings(),
                    repo_root=str(tmp_path),
                )

    async def test_explicit_timeout_s_overrides_settings_default(self, tmp_path: Any) -> None:
        """Explicit timeout_s overrides Settings.ARGUS_SESSION_TIMEOUT."""
        settings = _make_settings(session_timeout=600)
        client = _make_fake_client([_make_response(calls=[])])
        entry = _make_entry()

        with patch("argus.openai_runner._build_client", return_value=client) as mock_init:
            await run_session_openai(
                entry=entry,
                system_prompt="system",
                user_message="user",
                settings=settings,
                repo_root=str(tmp_path),
                timeout_s=42.0,
            )

        mock_init.assert_called_once()
        assert mock_init.call_args.kwargs["timeout"] == 42.0

    async def test_default_timeout_is_600_seconds(self, tmp_path: Any) -> None:
        """When neither timeout_s nor ARGUS_SESSION_TIMEOUT is passed, defaults to 600s."""
        settings = MagicMock(spec=[])
        settings.OPENAI_API_KEY = "key"
        client = _make_fake_client([_make_response(calls=[])])
        entry = _make_entry()

        with patch("argus.openai_runner._build_client", return_value=client) as mock_init:
            await run_session_openai(
                entry=entry,
                system_prompt="system",
                user_message="user",
                settings=settings,
                repo_root=str(tmp_path),
            )

        mock_init.assert_called_once()
        assert mock_init.call_args.kwargs["timeout"] == 600

    async def test_client_closed_on_normal_completion(self, tmp_path: Any) -> None:
        """client.close is called when session completes normally."""
        client = _make_fake_client([_make_response(calls=[])])
        entry = _make_entry()

        with patch("argus.openai_runner._build_client", return_value=client):
            await run_session_openai(
                entry=entry,
                system_prompt="system",
                user_message="user",
                settings=_make_settings(),
                repo_root=str(tmp_path),
            )

        client.close.assert_called_once()

    async def test_client_closed_on_timeout(self, tmp_path: Any) -> None:
        """client.close is called even when session times out."""

        async def _slow(**_: Any) -> Any:
            await asyncio.sleep(5.0)
            return _make_response(calls=[])

        client = MagicMock()
        client.responses.create = AsyncMock(side_effect=_slow)
        client.close = AsyncMock()

        entry = _make_entry()
        with patch("argus.openai_runner._build_client", return_value=client):
            await run_session_openai(
                entry=entry,
                system_prompt="system",
                user_message="user",
                settings=_make_settings(),
                repo_root=str(tmp_path),
                timeout_s=0.05,
            )

        client.close.assert_called_once()


class TestOpenAIRedactInputs:
    async def test_redact_openai_inputs_redacts_settings(self) -> None:
        settings = MagicMock()
        settings.OPENAI_API_KEY = "sk-secret-12345"
        settings.LANGSMITH_PROJECT = "my-project"
        inputs = {"settings": settings, "other": "val"}

        redacted = _redact_openai_inputs(inputs)
        assert redacted["other"] == "val"
        assert "sk-secret" not in str(redacted["settings"])
        assert "my-project" in str(redacted["settings"])

    async def test_redact_openai_inputs_missing_settings(self) -> None:
        inputs = {"other": "val"}
        assert _redact_openai_inputs(inputs) == inputs
        assert _redact_openai_inputs(cast(dict[str, Any], "not-a-dict")) == {}
