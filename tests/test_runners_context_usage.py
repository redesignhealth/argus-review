"""Tests for TECH-4734 context-usage instrumentation in _run_claude_session.

The instrumentation logs per-message token usage (AssistantMessage.usage),
per-tool-result sizes (UserMessage/ToolResultBlock), and the final
ResultMessage usage/model_usage dicts. It must be purely additive: a message
stream where any of these fields is missing (usage=None, empty dict, etc.)
must never raise -- a missing field must never break a review session.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from argus.llm.models import CLAUDE_DEFAULT, CLAUDE_OPUS

_RUNNERS_MODULE = "argus.runners"


def _settings() -> MagicMock:
    settings = MagicMock()
    settings.CONTEXT7_API_KEY = None
    settings.ARGUS_CONTEXT7_LIBRARY_ID = None
    settings.anthropic_credential = ("ANTHROPIC_API_KEY", "sk-test")
    return settings


class _FakeClient:
    """Stand-in for ClaudeSDKClient: async context manager yielding fixed messages."""

    def __init__(self, messages: list[Any]) -> None:
        self._messages = messages

    async def query(self, _message: str) -> None:
        return None

    async def receive_response(self) -> Any:
        for message in self._messages:
            yield message

    async def __aenter__(self) -> "_FakeClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


def _run_session(messages: list[Any]) -> Any:
    import asyncio

    from argus.runners import _run_claude_session

    fake_client = _FakeClient(messages)
    with patch(f"{_RUNNERS_MODULE}.ClaudeSDKClient", return_value=fake_client):
        return asyncio.run(
            _run_claude_session(
                model=CLAUDE_DEFAULT,
                system_prompt="system",
                user_message="review this",
                settings=_settings(),
                label="test-label",
                repo_root="/tmp/fake-repo",
            )
        )


class TestUsageLoggingIsDefensive:
    """No usage field anywhere in the stream must never crash the loop."""

    def test_assistant_message_without_usage_field(self) -> None:
        messages = [
            AssistantMessage(content=[TextBlock(text="hi")], model=CLAUDE_DEFAULT, usage=None),
            ResultMessage(
                subtype="success",
                duration_ms=100,
                duration_api_ms=90,
                is_error=False,
                num_turns=1,
                session_id="sess-1",
                usage=None,
                model_usage=None,
            ),
        ]
        result = _run_session(messages)
        assert result.tool_call_count == 0

    def test_assistant_message_with_empty_usage_dict(self) -> None:
        messages = [
            AssistantMessage(content=[TextBlock(text="hi")], model=CLAUDE_DEFAULT, usage={}),
            ResultMessage(
                subtype="success",
                duration_ms=100,
                duration_api_ms=90,
                is_error=False,
                num_turns=1,
                session_id="sess-1",
                usage={},
                model_usage={},
            ),
        ]
        result = _run_session(messages)
        assert result.tool_call_count == 0

    def test_tool_result_without_content(self) -> None:
        tool_use = ToolUseBlock(id="tu-1", name="Read", input={"file_path": "a.py"})
        tool_result = ToolResultBlock(tool_use_id="tu-1", content=None, is_error=None)
        messages = [
            AssistantMessage(
                content=[tool_use],
                model=CLAUDE_DEFAULT,
                usage={"input_tokens": 100, "output_tokens": 5},
            ),
            UserMessage(content=[tool_result]),
            ResultMessage(
                subtype="success",
                duration_ms=100,
                duration_api_ms=90,
                is_error=False,
                num_turns=1,
                session_id="sess-1",
                usage={"input_tokens": 100},
                model_usage=None,
            ),
        ]
        result = _run_session(messages)
        assert result.tool_call_count == 1
        assert result.tool_names == ["Read"]

    def test_peak_context_tokens_computed_from_usage(self) -> None:
        messages = [
            AssistantMessage(
                content=[TextBlock(text="a")],
                model=CLAUDE_DEFAULT,
                usage={
                    "input_tokens": 1000,
                    "cache_read_input_tokens": 5000,
                    "cache_creation_input_tokens": 200,
                    "output_tokens": 10,
                },
            ),
            AssistantMessage(
                content=[TextBlock(text="b")],
                model=CLAUDE_DEFAULT,
                usage={"input_tokens": 500},
            ),
            ResultMessage(
                subtype="success",
                duration_ms=100,
                duration_api_ms=90,
                is_error=False,
                num_turns=2,
                session_id="sess-1",
                usage={"input_tokens": 500},
                model_usage=None,
            ),
        ]
        with patch(f"{_RUNNERS_MODULE}.logger") as mock_logger:
            _run_session(messages)
        usage_log_calls = [
            call
            for call in mock_logger.info.call_args_list
            if call.args and isinstance(call.args[0], str) and "Agent usage" in call.args[0]
        ]
        assert len(usage_log_calls) == 2
        # First message: 1000 + 5000 + 200 = 6200. Assert the actual computed
        # value, not just that a log call happened -- a wrong sum would
        # previously have passed this test.
        assert usage_log_calls[0].args[-1] == 6200
        # Second message has no cache fields at all: 500 + 0 + 0 = 500, and
        # since 500 < 6200 peak_context_tokens must stay pinned at the first
        # message's value, not regress to the smaller one.
        assert usage_log_calls[1].args[-1] == 500

        # peak_context_tokens (6200) must surface unconditionally in the
        # "Agent done" summary line, even though this session had zero tool
        # calls -- previously that line was nested inside `if tool_calls:`
        # and a text-only session got no session-level summary at all.
        done_log_calls = [
            call
            for call in mock_logger.info.call_args_list
            if call.args and isinstance(call.args[0], str) and "Agent done" in call.args[0]
        ]
        assert len(done_log_calls) == 1
        assert done_log_calls[0].args[-1] == 6200

    def test_context_warning_logged_above_threshold(self) -> None:
        messages = [
            AssistantMessage(
                content=[TextBlock(text="a")],
                model=CLAUDE_DEFAULT,
                usage={"input_tokens": 1000, "cache_read_input_tokens": 200_000},
            ),
            ResultMessage(
                subtype="success",
                duration_ms=100,
                duration_api_ms=90,
                is_error=False,
                num_turns=1,
                session_id="sess-1",
                usage={"input_tokens": 1000},
                model_usage=None,
            ),
        ]
        with patch(f"{_RUNNERS_MODULE}.logger") as mock_logger:
            _run_session(messages)
        warning_calls = [
            call
            for call in mock_logger.warning.call_args_list
            if call.args
            and isinstance(call.args[0], str)
            and "crossed warning threshold" in call.args[0]
        ]
        assert len(warning_calls) == 1

    def test_no_context_warning_below_threshold(self) -> None:
        messages = [
            AssistantMessage(
                content=[TextBlock(text="a")],
                model=CLAUDE_DEFAULT,
                usage={"input_tokens": 1000, "cache_read_input_tokens": 5000},
            ),
            ResultMessage(
                subtype="success",
                duration_ms=100,
                duration_api_ms=90,
                is_error=False,
                num_turns=1,
                session_id="sess-1",
                usage={"input_tokens": 1000},
                model_usage=None,
            ),
        ]
        with patch(f"{_RUNNERS_MODULE}.logger") as mock_logger:
            _run_session(messages)
        threshold_warning_calls = [
            call
            for call in mock_logger.warning.call_args_list
            if call.args
            and isinstance(call.args[0], str)
            and "crossed warning threshold" in call.args[0]
        ]
        assert len(threshold_warning_calls) == 0

    def test_tool_result_size_for_list_content_does_not_use_repr(self) -> None:
        """A list-shaped ToolResultBlock.content must size actual text, not
        len(str(...)), which would count Python-repr punctuation/keys as
        content size and could incidentally embed content fragments."""
        tool_use = ToolUseBlock(id="tu-1", name="Read", input={"file_path": "a.py"})
        tool_result = ToolResultBlock(
            tool_use_id="tu-1",
            content=[{"type": "text", "text": "0123456789"}],
            is_error=None,
        )
        messages = [
            AssistantMessage(
                content=[tool_use],
                model=CLAUDE_DEFAULT,
                usage={"input_tokens": 100},
            ),
            UserMessage(content=[tool_result]),
            ResultMessage(
                subtype="success",
                duration_ms=100,
                duration_api_ms=90,
                is_error=False,
                num_turns=1,
                session_id="sess-1",
                usage={"input_tokens": 100},
                model_usage=None,
            ),
        ]
        with patch(f"{_RUNNERS_MODULE}.logger") as mock_logger:
            _run_session(messages)
        result_log_calls = [
            call
            for call in mock_logger.info.call_args_list
            if call.args and isinstance(call.args[0], str) and "Agent tool result" in call.args[0]
        ]
        assert len(result_log_calls) == 1
        # size_chars is the second-to-last positional arg (msg, tool, size_chars, is_error)
        assert result_log_calls[0].args[-2] == 10


class TestContext1mBetaAndStrictMcpConfig:
    """TECH-4732: sonnet reviewer sessions enable the Anthropic 1M-context
    beta and strict MCP config; the opus cross-cutting session gets
    strict_mcp_config only, not the beta.

    Reviewer sessions carry a large fixed tool-schema prefix that sits at the
    default 200k window's autocompact threshold, causing repeated
    autocompact-thrashing (TECH-4643) with no benefit. strict_mcp_config=True
    stops the CLI from inheriting the operator's own MCP servers (the source
    of most of that fixed prefix); the 1M beta moves the ceiling well above
    whatever fixed prefix remains -- but only sonnet reviewer sessions have
    ever been observed hitting that ceiling. The opus cross-cutting session
    has never come close to it in any measured production round, so it stays
    on the default window rather than taking on unverified-for-opus risk
    (long-context premium billing, beta-header handling) for no benefit.
    """

    def _run(self, model: str) -> MagicMock:
        messages: list[Any] = [
            ResultMessage(
                subtype="success",
                duration_ms=100,
                duration_api_ms=90,
                is_error=False,
                num_turns=0,
                session_id="sess-1",
                usage=None,
                model_usage=None,
            ),
        ]
        fake_client = _FakeClient(messages)
        with (
            patch(f"{_RUNNERS_MODULE}.ClaudeSDKClient", return_value=fake_client),
            patch(f"{_RUNNERS_MODULE}.ClaudeAgentOptions") as mock_options,
        ):
            import asyncio

            from argus.runners import _run_claude_session

            asyncio.run(
                _run_claude_session(
                    model=model,
                    system_prompt="system",
                    user_message="review this",
                    settings=_settings(),
                    label="test-label",
                    repo_root="/tmp/fake-repo",
                )
            )
        return mock_options

    def test_betas_and_strict_mcp_config_for_sonnet(self) -> None:
        mock_options = self._run(CLAUDE_DEFAULT)
        assert mock_options.call_args.kwargs["betas"] == ["context-1m-2025-08-07"]
        assert mock_options.call_args.kwargs["strict_mcp_config"] is True

    def test_strict_mcp_config_but_no_beta_for_opus(self) -> None:
        mock_options = self._run(CLAUDE_OPUS)
        assert mock_options.call_args.kwargs["betas"] == []
        assert mock_options.call_args.kwargs["strict_mcp_config"] is True

    def test_real_claude_agent_options_accepts_betas_and_strict_mcp_config(self) -> None:
        """Construct the REAL ClaudeAgentOptions (not mocked) with the exact
        kwargs _run_claude_session passes, so a claude_agent_sdk version that
        renamed/removed `betas` or `strict_mcp_config` fails this test rather
        than only surfacing as a silent runtime no-op in production."""
        options = ClaudeAgentOptions(
            cwd="/tmp/fake-repo",
            allowed_tools=["Read", "Glob", "Grep"],
            mcp_servers={},
            strict_mcp_config=True,
            permission_mode="default",
            model=CLAUDE_DEFAULT,
            system_prompt="system",
            max_turns=30,
            env={"ANTHROPIC_API_KEY": "sk-test"},
            betas=["context-1m-2025-08-07"],
        )
        assert options.betas == ["context-1m-2025-08-07"]
        assert options.strict_mcp_config is True

    def test_real_claude_agent_options_accepts_empty_betas(self) -> None:
        """The opus cross-cutting call site passes betas=[] (not omitted) --
        construct the REAL ClaudeAgentOptions with that exact shape too, so
        an SDK-level incompatibility with an empty betas list would fail
        this test rather than only surface in production."""
        options = ClaudeAgentOptions(
            cwd="/tmp/fake-repo",
            allowed_tools=["Read", "Glob", "Grep"],
            mcp_servers={},
            strict_mcp_config=True,
            permission_mode="default",
            model=CLAUDE_OPUS,
            system_prompt="system",
            max_turns=30,
            env={"ANTHROPIC_API_KEY": "sk-test"},
            betas=[],
        )
        assert options.betas == []
        assert options.strict_mcp_config is True
