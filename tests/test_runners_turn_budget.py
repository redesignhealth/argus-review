"""Tests for Claude and OpenAI turn budget constants, disclosures, and hooks (TECH-7093).

Covers:
- Claude turn budget constant is 50 and is passed to ClaudeAgentOptions.max_turns.
- OpenAI runner owns a decoupled _MAX_TURNS_OPENAI = 30 constant, while
  argus.runners._MAX_TURNS was removed.
- Claude system prompt includes the budget disclosure line and never mentions finish_review.
- Early completion and session exhaustion do not perform unexpected mid-stream query calls.
- Pure _compute_mid_budget_nudge_turn logic remains verified for runners that use it.
- Tool-budget hooks (_make_tool_budget_nudge_hooks):
  - no nudge below 60 calls;
  - mid nudge fires at >= 60 calls exactly once;
  - final nudge fires at >= 100 calls exactly once;
  - tracks combined PostToolUse and PostToolUseFailure counts;
  - subagent (agent_id) calls are skipped without advancing counter;
  - malformed inputs return {} and cannot raise;
  - hookEventName is properly mirrored in hookSpecificOutput;
  - nudge copy contains tool-call counts, mentions final JSON output block,
    and makes no claim about turns left;
  - thresholds ordered (60 < 100).
- Claude Agent SDK contracts:
  - ClaudeAgentOptions accepts hooks parameter;
  - HookEvent includes PostToolUse and PostToolUseFailure;
  - PostToolUseHookSpecificOutput and PostToolUseFailureHookSpecificOutput expose additionalContext;
  - ClaudeSDKClient hook conversion preserves matchers and callbacks;
  - _convert_hook_output_for_cli preserves hookSpecificOutput.additionalContext;
  - Pinned required claude_agent_sdk.types import paths.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, cast, get_args
from unittest.mock import MagicMock, patch

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    ResultMessage,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from claude_agent_sdk._internal.query import _convert_hook_output_for_cli
from claude_agent_sdk.types import (
    HookContext,
    HookEvent,
    PostToolUseFailureHookSpecificOutput,
    PostToolUseHookSpecificOutput,
    SyncHookJSONOutput,
)

import argus.runners
from argus.llm.models import CLAUDE_DEFAULT
from argus.openai_runner import _MAX_TURNS_OPENAI
from argus.runners import (
    _MAX_TURNS_CLAUDE,
    _TOOL_BUDGET_FINAL_NUDGE,
    _TOOL_BUDGET_FINAL_THRESHOLD,
    _TOOL_BUDGET_MID_NUDGE,
    _TOOL_BUDGET_MID_THRESHOLD,
    _TURN_BUDGET_SYSTEM_PROMPT_LINE_CLAUDE,
    _compute_mid_budget_nudge_turn,
    _make_tool_budget_nudge_hooks,
    _run_claude_session,
)

_RUNNERS_MODULE = "argus.runners"


# ---------------------------------------------------------------------------
# Test Helpers & Fakes
# ---------------------------------------------------------------------------


def _settings() -> MagicMock:
    settings = MagicMock()
    settings.CONTEXT7_API_KEY = None
    settings.ARGUS_CONTEXT7_LIBRARY_ID = None
    settings.anthropic_credential = ("ANTHROPIC_API_KEY", "sk-test")
    return settings


class _FakeClient:
    """Stand-in for ClaudeSDKClient tracking options and client.query calls."""

    def __init__(
        self,
        messages: list[Any],
        options: ClaudeAgentOptions | None = None,
    ) -> None:
        self.options = options
        self._messages = messages
        self.query_calls: list[str] = []

    async def query(self, message: str) -> None:
        self.query_calls.append(message)

    async def receive_response(self) -> AsyncIterator[Any]:
        for message in self._messages:
            yield message

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


async def _run_session_with_fake(
    messages: list[Any],
    system_prompt: str = "You are a code reviewer.",
    user_message: str = "Please review this diff.",
) -> tuple[argus.runners.SessionResult, _FakeClient]:
    captured_clients: list[_FakeClient] = []

    def _fake_client_factory(*args: Any, **kwargs: Any) -> _FakeClient:
        client = _FakeClient(
            messages=messages,
            options=kwargs.get("options"),
        )
        captured_clients.append(client)
        return client

    with patch(f"{_RUNNERS_MODULE}.ClaudeSDKClient", side_effect=_fake_client_factory):
        result = await _run_claude_session(
            model=CLAUDE_DEFAULT,
            system_prompt=system_prompt,
            user_message=user_message,
            settings=_settings(),
            label="turn-budget-test",
            repo_root="/tmp/fake-repo",
        )
    return result, captured_clients[0]


def _make_tool_turn(turn_num: int) -> list[Any]:
    tool_id = f"tool_{turn_num}"
    return [
        AssistantMessage(
            content=[
                ToolUseBlock(
                    id=tool_id,
                    name="Read",
                    input={"path": f"src/file_{turn_num}.py"},
                )
            ],
            model=CLAUDE_DEFAULT,
            usage={"input_tokens": 100, "output_tokens": 50},
        ),
        UserMessage(
            content=[
                ToolResultBlock(
                    tool_use_id=tool_id,
                    content="def test_func(): pass\n",
                )
            ]
        ),
    ]


def _make_result_message(
    result: str = '{"findings": []}', subtype: str = "success"
) -> ResultMessage:
    return ResultMessage(
        subtype=subtype,
        duration_ms=500,
        duration_api_ms=450,
        is_error=False,
        num_turns=50,
        session_id="sess-turn-budget-test",
        total_cost_usd=0.05,
        usage={"input_tokens": 1000, "output_tokens": 500},
        model_usage=None,
        result=result,
    )


# ---------------------------------------------------------------------------
# Test Suites
# ---------------------------------------------------------------------------


class TestTurnBudgetConstants:
    """Verify decoupled constants for Claude and OpenAI runners."""

    def test_claude_turn_budget_is_50(self) -> None:
        assert _MAX_TURNS_CLAUDE == 50

    def test_openai_runner_owns_separate_30_turn_constant(self) -> None:
        assert _MAX_TURNS_OPENAI == 30
        assert not hasattr(argus.runners, "_MAX_TURNS")
        assert _MAX_TURNS_OPENAI != _MAX_TURNS_CLAUDE

    @pytest.mark.asyncio
    async def test_claude_budget_passed_as_options_max_turns(self) -> None:
        messages = [
            *_make_tool_turn(1),
            _make_result_message(),
        ]
        with patch(
            f"{_RUNNERS_MODULE}.ClaudeAgentOptions", wraps=ClaudeAgentOptions
        ) as spy_options:
            _result, client = await _run_session_with_fake(messages)

        assert spy_options.call_args.kwargs["max_turns"] == 50
        assert client.options is not None
        assert client.options.max_turns == 50
        assert client.options.hooks is not None
        assert "PostToolUse" in client.options.hooks
        assert "PostToolUseFailure" in client.options.hooks

    def test_thresholds_ordered(self) -> None:
        assert _TOOL_BUDGET_MID_THRESHOLD < _TOOL_BUDGET_FINAL_THRESHOLD
        assert _TOOL_BUDGET_MID_THRESHOLD == 60
        assert _TOOL_BUDGET_FINAL_THRESHOLD == 100


class TestClaudeSystemPromptDisclosure:
    """Verify Claude system prompt disclosure formatting and content."""

    @pytest.mark.asyncio
    async def test_system_prompt_includes_budget_disclosure_and_no_finish_review(self) -> None:
        base_prompt = "You are an automated reviewer evaluating PR changes."
        messages = [
            *_make_tool_turn(1),
            _make_result_message(),
        ]
        _result, client = await _run_session_with_fake(messages, system_prompt=base_prompt)

        assert client.options is not None
        system_prompt = client.options.system_prompt
        assert isinstance(system_prompt, str)
        assert system_prompt.startswith(base_prompt)

        expected_disclosure = _TURN_BUDGET_SYSTEM_PROMPT_LINE_CLAUDE.format(max_turns=50)
        assert expected_disclosure in system_prompt
        assert "50 turns" in system_prompt
        assert "Emit your final JSON output block" in system_prompt
        # Claude path does not have finish_review tool
        assert "finish_review" not in system_prompt


class TestClaudeSessionExecution:
    """Verify session completion and failure reporting without mid-stream client.query calls."""

    @pytest.mark.asyncio
    async def test_early_completion_no_extra_client_query(self) -> None:
        """A session completing early finishes with 0 extra client query calls."""
        messages: list[Any] = []
        for turn_idx in range(1, 6):
            messages.extend(_make_tool_turn(turn_idx))
        messages.append(_make_result_message(subtype="success"))

        user_msg = "Quick review."
        result, client = await _run_session_with_fake(messages, user_message=user_msg)

        assert result.failure_reason is None
        assert client.query_calls == [user_msg]

    @pytest.mark.asyncio
    async def test_session_exhaustion_sets_turn_budget_exhausted_reason(self) -> None:
        messages: list[Any] = []
        for turn_idx in range(1, 51):
            messages.extend(_make_tool_turn(turn_idx))
        messages.append(_make_result_message(subtype="error_max_turns"))

        user_msg = "Exhaust budget."
        result, client = await _run_session_with_fake(messages, user_message=user_msg)
        assert result.failure_reason == "turn_budget_exhausted"
        assert client.query_calls == [user_msg]


class TestComputeMidBudgetNudgeTurn:
    """Verify pure logic of _compute_mid_budget_nudge_turn for runners that use it (OpenAI/Gemini)."""

    def test_compute_mid_budget_nudge_turn_pure_logic(self) -> None:
        # Standard budgets where checkpoint < emergency
        assert _compute_mid_budget_nudge_turn(50) == 37  # int(37.5) = 37 < 47
        assert _compute_mid_budget_nudge_turn(30) == 22  # int(22.5) = 22 < 27
        assert _compute_mid_budget_nudge_turn(100) == 75  # 75 < 97

        # Small budgets where checkpoint >= emergency (suppression guard fires)
        # max_turns=10: checkpoint=7, emergency=7 -> 7 >= 7 -> None
        assert _compute_mid_budget_nudge_turn(10) is None
        # max_turns=4: checkpoint=3, emergency=1 -> 3 >= 1 -> None
        assert _compute_mid_budget_nudge_turn(4) is None
        # max_turns=3: checkpoint=2, emergency=0 -> 2 >= 0 -> None
        assert _compute_mid_budget_nudge_turn(3) is None


async def _call_hook(
    callback: Any,
    input_data: Any,
    tool_use_id: str | None = None,
    context: HookContext | None = None,
) -> dict[str, Any]:
    ctx = context if context is not None else {"signal": None}
    res = await callback(input_data, tool_use_id, ctx)
    return cast(dict[str, Any], res)


class TestToolBudgetNudgeHooks:
    """Directly exercise _make_tool_budget_nudge_hooks callbacks with synthetic inputs."""

    @pytest.mark.asyncio
    async def test_no_nudge_below_60_calls(self) -> None:
        hooks = _make_tool_budget_nudge_hooks(label="test-below-60")
        callback = hooks["PostToolUse"][0].hooks[0]

        for i in range(1, 60):
            input_data = {"hook_event_name": "PostToolUse", "tool_name": "Read"}
            res = await _call_hook(callback, input_data, f"tu-{i}")
            assert res == {}

    @pytest.mark.asyncio
    async def test_mid_nudge_at_60_exactly_once(self) -> None:
        hooks = _make_tool_budget_nudge_hooks(label="test-mid-60")
        callback = hooks["PostToolUse"][0].hooks[0]

        for i in range(1, 60):
            res = await _call_hook(callback, {"hook_event_name": "PostToolUse"}, f"tu-{i}")
            assert res == {}

        # Call 60: fires mid nudge
        res60 = await _call_hook(callback, {"hook_event_name": "PostToolUse"}, "tu-60")
        assert "hookSpecificOutput" in res60
        hso60 = res60["hookSpecificOutput"]
        assert hso60["hookEventName"] == "PostToolUse"
        assert "60 tool calls" in hso60["additionalContext"]
        assert "final JSON output block" in hso60["additionalContext"]

        # Calls 61 to 99: no additional nudge
        for i in range(61, 100):
            res = await _call_hook(callback, {"hook_event_name": "PostToolUse"}, f"tu-{i}")
            assert res == {}

    @pytest.mark.asyncio
    async def test_final_nudge_at_100_exactly_once(self) -> None:
        hooks = _make_tool_budget_nudge_hooks(label="test-final-100")
        callback = hooks["PostToolUse"][0].hooks[0]

        for i in range(1, 100):
            await _call_hook(callback, {"hook_event_name": "PostToolUse"}, f"tu-{i}")

        # Call 100: fires final nudge
        res100 = await _call_hook(callback, {"hook_event_name": "PostToolUse"}, "tu-100")
        assert "hookSpecificOutput" in res100
        hso100 = res100["hookSpecificOutput"]
        assert hso100["hookEventName"] == "PostToolUse"
        assert "100 tool calls" in hso100["additionalContext"]
        assert "final JSON output block" in hso100["additionalContext"]
        assert "Stop reading and searching" in hso100["additionalContext"]

        # Calls 101 to 110: no additional nudge
        for i in range(101, 111):
            res = await _call_hook(callback, {"hook_event_name": "PostToolUse"}, f"tu-{i}")
            assert res == {}

    @pytest.mark.asyncio
    async def test_combined_success_and_failure_count(self) -> None:
        hooks = _make_tool_budget_nudge_hooks(label="test-combined")
        callback = hooks["PostToolUse"][0].hooks[0]

        # 35 successes + 24 failures = 59 calls
        for i in range(1, 36):
            res = await _call_hook(callback, {"hook_event_name": "PostToolUse"}, f"tu-s-{i}")
            assert res == {}
        for i in range(1, 25):
            res = await _call_hook(callback, {"hook_event_name": "PostToolUseFailure"}, f"tu-f-{i}")
            assert res == {}

        # 60th call is a failure: mid nudge fires with PostToolUseFailure mirrored
        res60 = await _call_hook(callback, {"hook_event_name": "PostToolUseFailure"}, "tu-f-25")
        assert "hookSpecificOutput" in res60
        assert res60["hookSpecificOutput"]["hookEventName"] == "PostToolUseFailure"
        assert "60 tool calls" in res60["hookSpecificOutput"]["additionalContext"]

        # 20 successes + 19 failures = 39 more calls (total 99)
        for i in range(36, 56):
            res = await _call_hook(callback, {"hook_event_name": "PostToolUse"}, f"tu-s-{i}")
            assert res == {}
        for i in range(26, 45):
            res = await _call_hook(callback, {"hook_event_name": "PostToolUseFailure"}, f"tu-f-{i}")
            assert res == {}

        # 100th call is a success: final nudge fires with PostToolUse mirrored
        res100 = await _call_hook(callback, {"hook_event_name": "PostToolUse"}, "tu-s-56")
        assert "hookSpecificOutput" in res100
        assert res100["hookSpecificOutput"]["hookEventName"] == "PostToolUse"
        assert "100 tool calls" in res100["hookSpecificOutput"]["additionalContext"]

    @pytest.mark.asyncio
    async def test_agent_id_subagent_inputs_skipped_without_advancing(self) -> None:
        hooks = _make_tool_budget_nudge_hooks(label="test-subagent")
        callback = hooks["PostToolUse"][0].hooks[0]

        # 59 top-level calls
        for i in range(1, 60):
            await _call_hook(callback, {"hook_event_name": "PostToolUse"}, f"tu-{i}")

        # 20 subagent calls with agent_id set: all return {}
        for i in range(1, 21):
            sub_res = await _call_hook(
                callback,
                {"agent_id": f"subagent-task-{i}", "hook_event_name": "PostToolUse"},
                f"tu-sub-{i}",
            )
            assert sub_res == {}

        # 60th top-level call fires mid nudge with n=60, proving subagents did NOT advance count
        res60 = await _call_hook(callback, {"hook_event_name": "PostToolUse"}, "tu-60")
        assert "hookSpecificOutput" in res60
        assert "60 tool calls" in res60["hookSpecificOutput"]["additionalContext"]

    @pytest.mark.asyncio
    async def test_malformed_input_cannot_raise(self) -> None:
        hooks = _make_tool_budget_nudge_hooks(label="test-malformed")
        callback = hooks["PostToolUse"][0].hooks[0]

        malformed_inputs: list[Any] = [
            None,
            "not-a-dict",
            12345,
            [],
            {},
            {"agent_id": None, "hook_event_name": None},
            {"hook_event_name": 999},
            {"agent_id": ""},
        ]
        for item in malformed_inputs:
            res = await _call_hook(callback, item, None)
            assert res == {}

    @pytest.mark.asyncio
    async def test_mirrored_hook_event_name(self) -> None:
        # Use custom thresholds 1 and 2 to test mirroring on both events
        hooks = _make_tool_budget_nudge_hooks(mid_threshold=1, final_threshold=2)
        callback = hooks["PostToolUse"][0].hooks[0]

        res_success = await _call_hook(callback, {"hook_event_name": "PostToolUse"}, "tu-1")
        assert res_success["hookSpecificOutput"]["hookEventName"] == "PostToolUse"

        res_failure = await _call_hook(callback, {"hook_event_name": "PostToolUseFailure"}, "tu-2")
        assert res_failure["hookSpecificOutput"]["hookEventName"] == "PostToolUseFailure"

    def test_nudge_copy_contains_counts_and_makes_no_claim_about_turns_left(self) -> None:
        # Mid nudge copy
        mid_text = _TOOL_BUDGET_MID_NUDGE.format(n=60, count=60)
        assert "60 tool calls" in mid_text
        assert "final JSON output block" in mid_text
        assert "turns left" not in mid_text
        assert "turn budget" not in mid_text
        assert "turns" not in mid_text
        assert "finish_review" not in mid_text

        # Final nudge copy
        final_text = _TOOL_BUDGET_FINAL_NUDGE.format(n=100, count=100)
        assert "100 tool calls" in final_text
        assert "final JSON output block" in final_text
        assert "turns left" not in final_text
        assert "turn budget" not in final_text
        assert "turns" not in final_text
        assert "finish_review" not in final_text

    @pytest.mark.asyncio
    async def test_custom_thresholds_ordered_and_configurable(self) -> None:
        hooks = _make_tool_budget_nudge_hooks(mid_threshold=3, final_threshold=7)
        callback = hooks["PostToolUse"][0].hooks[0]

        for i in range(1, 3):
            assert await _call_hook(callback, {"hook_event_name": "PostToolUse"}, f"tu-{i}") == {}

        res3 = await _call_hook(callback, {"hook_event_name": "PostToolUse"}, "tu-3")
        assert "3 tool calls" in res3["hookSpecificOutput"]["additionalContext"]

        for i in range(4, 7):
            assert await _call_hook(callback, {"hook_event_name": "PostToolUse"}, f"tu-{i}") == {}

        res7 = await _call_hook(callback, {"hook_event_name": "PostToolUse"}, "tu-7")
        assert "7 tool calls" in res7["hookSpecificOutput"]["additionalContext"]

        assert await _call_hook(callback, {"hook_event_name": "PostToolUse"}, "tu-8") == {}


class TestClaudeAgentSdkHookContract:
    """Real SDK contract tests without network/process spawn."""

    def test_real_claude_agent_options_accepts_hooks(self) -> None:
        hooks = _make_tool_budget_nudge_hooks("test-options")
        options = ClaudeAgentOptions(hooks=hooks)
        assert options.hooks is hooks
        assert "PostToolUse" in options.hooks
        assert "PostToolUseFailure" in options.hooks

    def test_sdk_hook_event_includes_post_tool_use_and_failure(self) -> None:
        raw_events = get_args(HookEvent)
        events = {
            val for item in raw_events for val in (get_args(item) if get_args(item) else (item,))
        }
        assert "PostToolUse" in events
        assert "PostToolUseFailure" in events

    def test_hook_output_typed_dicts_expose_additional_context(self) -> None:
        assert "additionalContext" in PostToolUseHookSpecificOutput.__annotations__
        assert "additionalContext" in PostToolUseFailureHookSpecificOutput.__annotations__

        success_output: PostToolUseHookSpecificOutput = {
            "hookEventName": "PostToolUse",
            "additionalContext": "test context",
        }
        assert success_output["additionalContext"] == "test context"

        failure_output: PostToolUseFailureHookSpecificOutput = {
            "hookEventName": "PostToolUseFailure",
            "additionalContext": "failure context",
        }
        assert failure_output["additionalContext"] == "failure context"

    def test_claude_sdk_client_hook_conversion_preserves_matchers(self) -> None:
        hooks = _make_tool_budget_nudge_hooks("test-client")
        assert isinstance(hooks["PostToolUse"][0], HookMatcher)
        assert isinstance(hooks["PostToolUseFailure"][0], HookMatcher)

        client = ClaudeSDKClient(options=ClaudeAgentOptions(hooks=hooks))
        internal = client._convert_hooks_to_internal_format(hooks)
        assert "PostToolUse" in internal
        assert "PostToolUseFailure" in internal
        assert len(internal["PostToolUse"]) == 1
        assert len(internal["PostToolUseFailure"]) == 1
        assert internal["PostToolUse"][0]["hooks"] == hooks["PostToolUse"][0].hooks
        assert internal["PostToolUseFailure"][0]["hooks"] == hooks["PostToolUseFailure"][0].hooks

    def test_convert_hook_output_for_cli_preserves_additional_context(self) -> None:
        success_json: SyncHookJSONOutput = {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": "test convergence message",
            }
        }
        converted_success = _convert_hook_output_for_cli(success_json)
        assert (
            converted_success["hookSpecificOutput"]["additionalContext"]
            == "test convergence message"
        )
        assert converted_success["hookSpecificOutput"]["hookEventName"] == "PostToolUse"

        failure_json: SyncHookJSONOutput = {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUseFailure",
                "additionalContext": "test failure message",
            }
        }
        converted_failure = _convert_hook_output_for_cli(failure_json)
        assert (
            converted_failure["hookSpecificOutput"]["additionalContext"] == "test failure message"
        )
        assert converted_failure["hookSpecificOutput"]["hookEventName"] == "PostToolUseFailure"

    def test_pinned_claude_agent_sdk_types_import_paths(self) -> None:
        from claude_agent_sdk.types import (
            HookContext,
            HookEvent,
            HookMatcher,
            PostToolUseFailureHookInput,
            PostToolUseFailureHookSpecificOutput,
            PostToolUseHookInput,
            PostToolUseHookSpecificOutput,
            SyncHookJSONOutput,
        )

        # Verify required types are present and exposed directly on claude_agent_sdk.types
        assert HookContext is not None
        assert HookEvent is not None
        assert HookMatcher is not None
        assert PostToolUseHookInput is not None
        assert PostToolUseFailureHookInput is not None
        assert PostToolUseHookSpecificOutput is not None
        assert PostToolUseFailureHookSpecificOutput is not None
        assert SyncHookJSONOutput is not None
