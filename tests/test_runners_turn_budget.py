"""Tests for Claude and OpenAI turn budget constants, disclosures, and nudges (TECH-7093).

Covers:
- Claude turn budget constant is 50 and is passed to ClaudeAgentOptions.max_turns.
- Claude system prompt includes the budget disclosure line and never mentions finish_review.
- OpenAI runner owns a decoupled _MAX_TURNS_OPENAI = 30 constant, while
  argus.runners._MAX_TURNS was removed.
- Claude 75% and final-three-turn nudges inject exactly once at the expected turns
  (turn 37 and turn 47) for a 50-turn tool-using session, with the expected query-call sequence.
- Early completion receives no nudges.
- Text-only checkpoint turns do not inject nudges.
- Existing _compute_mid_budget_nudge_turn collision guard suppresses the 75% mid nudge
  for a small monkeypatched budget.
- Nudge query failure does not abort the review session.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Callable
from unittest.mock import MagicMock, patch

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

import argus.runners
from argus.llm.models import CLAUDE_DEFAULT
from argus.openai_runner import _MAX_TURNS_OPENAI
from argus.runners import (
    _MAX_TURNS_CLAUDE,
    _MID_BUDGET_NUDGE_CLAUDE,
    _NUDGE_TURNS_BEFORE_BUDGET,
    _TURN_BUDGET_NUDGE_CLAUDE,
    _TURN_BUDGET_SYSTEM_PROMPT_LINE_CLAUDE,
    _compute_mid_budget_nudge_turn,
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
    """Stand-in for ClaudeSDKClient tracking options, queries, and turn history."""

    def __init__(
        self,
        messages: list[Any],
        options: ClaudeAgentOptions | None = None,
        query_exc_predicate: Callable[[str], bool] | None = None,
        query_exc: Exception | None = None,
    ) -> None:
        self.options = options
        self._messages = messages
        self.query_calls: list[str] = []
        self.query_turn_history: list[tuple[int, str]] = []
        self._current_turn = 0
        self._query_exc_predicate = query_exc_predicate
        self._query_exc = query_exc

    async def query(self, message: str) -> None:
        self.query_calls.append(message)
        self.query_turn_history.append((self._current_turn, message))
        if self._query_exc and self._query_exc_predicate and self._query_exc_predicate(message):
            raise self._query_exc

    async def receive_response(self) -> AsyncIterator[Any]:
        for message in self._messages:
            if isinstance(message, AssistantMessage):
                self._current_turn += 1
            yield message

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


async def _run_session_with_fake(
    messages: list[Any],
    system_prompt: str = "You are a code reviewer.",
    user_message: str = "Please review this diff.",
    query_exc_predicate: Callable[[str], bool] | None = None,
    query_exc: Exception | None = None,
) -> tuple[argus.runners.SessionResult, _FakeClient]:
    captured_clients: list[_FakeClient] = []

    def _fake_client_factory(*args: Any, **kwargs: Any) -> _FakeClient:
        client = _FakeClient(
            messages=messages,
            options=kwargs.get("options"),
            query_exc_predicate=query_exc_predicate,
            query_exc=query_exc,
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


def _make_text_turn(turn_num: int) -> list[Any]:
    return [
        AssistantMessage(
            content=[TextBlock(text=f"Analyzing turn {turn_num} findings...")],
            model=CLAUDE_DEFAULT,
            usage={"input_tokens": 100, "output_tokens": 50},
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


class TestClaudeNudgeInjection:
    """Verify in-band nudge injection at 75% and final-three-turn thresholds."""

    @pytest.mark.asyncio
    async def test_nudges_injected_exactly_once_at_expected_turns_in_50_turn_session(self) -> None:
        """For a 50-turn tool-using session:
        - mid-budget nudge fires on turn 37 (int(50 * 0.75))
        - final-three-turn nudge fires on turn 47 (50 - 3)
        - client.query sequence has length 3: [initial_user_message, mid_nudge, final_nudge]
        - neither nudge mentions finish_review
        """
        mid_turn = _compute_mid_budget_nudge_turn(50)
        assert mid_turn == 37
        final_turn = 50 - _NUDGE_TURNS_BEFORE_BUDGET
        assert final_turn == 47

        messages: list[Any] = []
        for turn_idx in range(1, 51):
            messages.extend(_make_tool_turn(turn_idx))
        messages.append(_make_result_message())

        user_msg = "Review the changes in this PR."
        result, client = await _run_session_with_fake(messages, user_message=user_msg)

        assert result.failure_reason is None

        # Verify query call sequence and turns
        assert len(client.query_calls) == 3
        assert client.query_calls[0] == user_msg
        assert client.query_calls[1] == _MID_BUDGET_NUDGE_CLAUDE
        assert client.query_calls[2] == _TURN_BUDGET_NUDGE_CLAUDE

        assert client.query_turn_history == [
            (0, user_msg),
            (37, _MID_BUDGET_NUDGE_CLAUDE),
            (47, _TURN_BUDGET_NUDGE_CLAUDE),
        ]

        # Verify nudge wording
        assert "finish_review" not in _MID_BUDGET_NUDGE_CLAUDE
        assert "finish_review" not in _TURN_BUDGET_NUDGE_CLAUDE
        assert "75%" in _MID_BUDGET_NUDGE_CLAUDE
        assert "final JSON output block" in _MID_BUDGET_NUDGE_CLAUDE
        assert "3 turns left" in _TURN_BUDGET_NUDGE_CLAUDE
        assert "final JSON output block" in _TURN_BUDGET_NUDGE_CLAUDE

    @pytest.mark.asyncio
    async def test_session_exhaustion_sets_turn_budget_exhausted_reason(self) -> None:
        messages: list[Any] = []
        for turn_idx in range(1, 51):
            messages.extend(_make_tool_turn(turn_idx))
        messages.append(_make_result_message(subtype="error_max_turns"))

        result, client = await _run_session_with_fake(messages)
        assert result.failure_reason == "turn_budget_exhausted"
        assert len(client.query_calls) == 3

    @pytest.mark.asyncio
    async def test_early_completion_receives_no_nudges(self) -> None:
        """A session completing at turn 5 exits before turns 37 and 47, receiving 0 nudges."""
        messages: list[Any] = []
        for turn_idx in range(1, 6):
            messages.extend(_make_tool_turn(turn_idx))
        messages.append(_make_result_message(subtype="success"))

        user_msg = "Quick review."
        result, client = await _run_session_with_fake(messages, user_message=user_msg)

        assert result.failure_reason is None
        assert client.query_calls == [user_msg]
        assert _MID_BUDGET_NUDGE_CLAUDE not in client.query_calls
        assert _TURN_BUDGET_NUDGE_CLAUDE not in client.query_calls

    @pytest.mark.asyncio
    async def test_text_only_checkpoint_turn_does_not_inject_nudge(self) -> None:
        """If turn 37 is text-only (no ToolUseBlock), the mid nudge must not inject."""
        messages: list[Any] = []
        for turn_idx in range(1, 51):
            if turn_idx == 37:
                messages.extend(_make_text_turn(turn_idx))
            else:
                messages.extend(_make_tool_turn(turn_idx))
        messages.append(_make_result_message())

        user_msg = "Review code."
        result, client = await _run_session_with_fake(messages, user_message=user_msg)

        assert result.failure_reason is None
        # Mid nudge skipped because turn 37 had no tool use; final nudge still fires on turn 47
        assert _MID_BUDGET_NUDGE_CLAUDE not in client.query_calls
        assert _TURN_BUDGET_NUDGE_CLAUDE in client.query_calls
        assert client.query_turn_history == [
            (0, user_msg),
            (47, _TURN_BUDGET_NUDGE_CLAUDE),
        ]

    @pytest.mark.asyncio
    async def test_text_only_final_turn_does_not_inject_nudge(self) -> None:
        """If turn 47 is text-only (no ToolUseBlock), the final nudge must not inject."""
        messages: list[Any] = []
        for turn_idx in range(1, 51):
            if turn_idx == 47:
                messages.extend(_make_text_turn(turn_idx))
            else:
                messages.extend(_make_tool_turn(turn_idx))
        messages.append(_make_result_message())

        user_msg = "Review code."
        result, client = await _run_session_with_fake(messages, user_message=user_msg)

        assert result.failure_reason is None
        # Mid nudge fires on turn 37; final nudge skipped because turn 47 had no tool use
        assert _MID_BUDGET_NUDGE_CLAUDE in client.query_calls
        assert _TURN_BUDGET_NUDGE_CLAUDE not in client.query_calls
        assert client.query_turn_history == [
            (0, user_msg),
            (37, _MID_BUDGET_NUDGE_CLAUDE),
        ]


class TestMidBudgetCollisionGuard:
    """Verify _compute_mid_budget_nudge_turn collision suppression."""

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

    @pytest.mark.asyncio
    async def test_collision_guard_suppresses_mid_nudge_for_small_monkeypatched_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With a small budget (e.g. 4 turns), mid-budget nudge collision guard
        returns None, suppressing the 75% nudge entirely while leaving final nudge
        active on turn 1 (4 - 3)."""
        monkeypatch.setattr(f"{_RUNNERS_MODULE}._MAX_TURNS_CLAUDE", 4)

        messages: list[Any] = []
        for turn_idx in range(1, 5):
            messages.extend(_make_tool_turn(turn_idx))
        messages.append(_make_result_message())

        user_msg = "Review small budget."
        result, client = await _run_session_with_fake(messages, user_message=user_msg)

        assert result.failure_reason is None
        assert _MID_BUDGET_NUDGE_CLAUDE not in client.query_calls
        # Final nudge fires on turn 1 (4 - 3 = 1)
        assert _TURN_BUDGET_NUDGE_CLAUDE in client.query_calls
        assert client.query_turn_history == [
            (0, user_msg),
            (1, _TURN_BUDGET_NUDGE_CLAUDE),
        ]


class TestNudgeQueryFailureResilience:
    """Verify that exceptions raised by client.query during nudge injection do not abort the review."""

    @pytest.mark.asyncio
    async def test_nudge_query_failure_cannot_abort_review(self) -> None:
        messages: list[Any] = []
        for turn_idx in range(1, 51):
            messages.extend(_make_tool_turn(turn_idx))
        expected_json = '{"findings": [{"file": "src/app.py", "line": 42, "description": "Bug"}]}'
        messages.append(_make_result_message(result=expected_json))

        # Raise RuntimeError whenever a nudge query is sent
        def _is_nudge(msg: str) -> bool:
            return msg in (_MID_BUDGET_NUDGE_CLAUDE, _TURN_BUDGET_NUDGE_CLAUDE)

        result, client = await _run_session_with_fake(
            messages,
            query_exc=RuntimeError("Transient SDK query pipe failure"),
            query_exc_predicate=_is_nudge,
        )

        # Review must still succeed and return the output payload
        assert result.failure_reason is None
        assert result.result_text == expected_json
        # Both nudges were attempted despite the errors
        assert len(client.query_calls) == 3
        assert client.query_calls[1] == _MID_BUDGET_NUDGE_CLAUDE
        assert client.query_calls[2] == _TURN_BUDGET_NUDGE_CLAUDE
