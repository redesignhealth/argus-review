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


class TestLangSmithEnrichmentIsDefensive:
    """TECH-4734 phase 2: LangSmith span enrichment and the local ledger fallback
    must never break a review session, whether tracing is off or a write fails.
    """

    def test_tracing_off_does_not_raise(self) -> None:
        """get_current_run_tree() returning None (tracing off) must be a no-op,
        not an AttributeError from calling .add_metadata() on None."""
        messages = [
            AssistantMessage(
                content=[TextBlock(text="a")],
                model="claude-sonnet-4-6",
                usage={"input_tokens": 100, "output_tokens": 5},
            ),
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
        with patch(f"{_RUNNERS_MODULE}.get_current_run_tree", return_value=None):
            result = _run_session(messages)
        assert result.result_text == ""

    def test_run_tree_add_metadata_failure_does_not_raise(self) -> None:
        """A LangSmith-side failure while attaching metadata (e.g. a client
        error) must be swallowed -- observability enrichment is best-effort."""
        messages = [
            AssistantMessage(
                content=[TextBlock(text="a")],
                model="claude-sonnet-4-6",
                usage={"input_tokens": 100, "output_tokens": 5},
            ),
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
        broken_run_tree = MagicMock()
        broken_run_tree.add_metadata.side_effect = RuntimeError("langsmith unavailable")
        with patch(f"{_RUNNERS_MODULE}.get_current_run_tree", return_value=broken_run_tree):
            result = _run_session(messages)
        assert result.result_text == ""
        broken_run_tree.add_metadata.assert_called_once()

    def test_ledger_write_failure_does_not_raise(self) -> None:
        """A best-effort local-ledger append that hits an OSError (full disk,
        permissions, concurrent-write race) must not surface to the caller."""
        messages = [
            AssistantMessage(
                content=[TextBlock(text="a")],
                model="claude-sonnet-4-6",
                usage={"input_tokens": 100, "output_tokens": 5},
            ),
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
        with patch(f"{_RUNNERS_MODULE}.open", side_effect=OSError("disk full"), create=True):
            result = _run_session(messages)
        assert result.result_text == ""

    def test_ledger_records_usage_fields(self, tmp_path: Any) -> None:
        """The ledger line shape matches the spec: ts, label, msg_index, token
        counts, context_total, peak -- and only ever those fields (no content)."""
        import json

        ledger_path = tmp_path / "ledger.jsonl"
        messages = [
            AssistantMessage(
                content=[TextBlock(text="a")],
                model="claude-sonnet-4-6",
                usage={
                    "input_tokens": 100,
                    "cache_read_input_tokens": 50,
                    "cache_creation_input_tokens": 10,
                    "output_tokens": 5,
                },
            ),
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
        with patch(f"{_RUNNERS_MODULE}._context_ledger_path", return_value=str(ledger_path)):
            _run_session(messages)
        lines = ledger_path.read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["label"] == "test-label"
        assert record["msg_index"] == 1
        assert record["input_tokens"] == 100
        assert record["cache_read"] == 50
        assert record["cache_creation"] == 10
        assert record["output_tokens"] == 5
        assert record["context_total"] == 160
        assert record["peak"] == 160
        assert set(record.keys()) == {
            "ts",
            "label",
            "msg_index",
            "input_tokens",
            "cache_read",
            "cache_creation",
            "output_tokens",
            "context_total",
            "peak",
        }


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

    def _run(self, model: str, *, is_system_reviewer_role: bool = False) -> MagicMock:
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
                    is_system_reviewer_role=is_system_reviewer_role,
                )
            )
        return mock_options

    def test_betas_and_strict_mcp_config_for_sonnet(self) -> None:
        mock_options = self._run(CLAUDE_DEFAULT, is_system_reviewer_role=True)
        assert mock_options.call_args.kwargs["betas"] == ["context-1m-2025-08-07"]
        assert mock_options.call_args.kwargs["strict_mcp_config"] is True

    def test_strict_mcp_config_but_no_beta_for_opus(self) -> None:
        mock_options = self._run(CLAUDE_OPUS, is_system_reviewer_role=False)
        assert mock_options.call_args.kwargs["betas"] == []
        assert mock_options.call_args.kwargs["strict_mcp_config"] is True

    def test_no_beta_for_non_system_reviewer_role_even_with_default_model(self) -> None:
        """A call that passes the system-reviewer's own default model value
        but declares is_system_reviewer_role=False must never get the beta --
        this is what makes the gate role-based rather than model-string-based
        (see _run_claude_session's docstring)."""
        mock_options = self._run(CLAUDE_DEFAULT, is_system_reviewer_role=False)
        assert mock_options.call_args.kwargs["betas"] == []

    def test_no_beta_when_specialist_model_overridden(self, monkeypatch: Any) -> None:
        """Regression test for the round-2 Argus finding on this PR: the
        beta gate must require BOTH the system-reviewer role AND that the
        reviewer model hasn't been overridden away from the validated pin.
        Gating on role alone (ignoring divergence) would attach this beta to
        whatever ARGUS_SPECIALIST_MODEL points at, extending an
        unverified-for-that-model billing/compatibility risk.

        Patches the module-level `_SYSTEM_REVIEWER_UNOVERRIDDEN` flag
        directly rather than reloading argus.llm.models/argus.runners under
        a real ARGUS_SPECIALIST_MODEL env var: a full reload re-executes
        argus.runners' module-level side effects (the _REVIEWER_EXECUTOR
        singleton) well beyond what this test needs -- flagged as fragile in
        Argus's round-3 review of this PR. The env var's actual effect on
        _SYSTEM_REVIEWER_UNOVERRIDDEN's resolution is independently covered
        by tests/test_llm_models_override.py.
        """
        monkeypatch.setattr(f"{_RUNNERS_MODULE}._SYSTEM_REVIEWER_UNOVERRIDDEN", False)
        mock_options = self._run(CLAUDE_DEFAULT, is_system_reviewer_role=True)
        assert mock_options.call_args.kwargs["betas"] == []

    def test_no_beta_for_cross_cutting_even_when_model_collides_with_default_pin(self) -> None:
        """Regression test for a round-3 Argus finding on this PR: an earlier
        gate (`model == _SYSTEM_REVIEWER_MODEL and model ==
        ALIAS_MAP["claude-default"]`) was still a model-string comparison, so
        if --frontier-model happened to be set to the SAME value as the
        unoverridden system-reviewer default pin, `_CROSS_CUTTING_MODEL`
        collided with `_SYSTEM_REVIEWER_MODEL` and the cross-cutting call
        received the beta anyway -- exactly the opus-specific
        billing/compatibility risk the surrounding comment says must never
        happen. The role-based gate (is_system_reviewer_role, declared by
        the caller, never inferred from `model`) cannot be fooled by this
        collision: passing CLAUDE_DEFAULT's own value (simulating the
        collision) with is_system_reviewer_role=False -- exactly what the
        cross-cutting call site always passes, regardless of what model
        value _CROSS_CUTTING_MODEL resolves to -- must still withhold the
        beta. This is mechanically the same assertion as
        test_no_beta_for_non_system_reviewer_role_even_with_default_model
        above; kept as its own named test since it documents a specific,
        previously-real regression rather than a general property.
        """
        mock_options = self._run(CLAUDE_DEFAULT, is_system_reviewer_role=False)
        assert mock_options.call_args.kwargs["betas"] == []

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


class TestOneTimeImportLogs:
    """The 1M-context-beta-withheld and cross-cutting-tier-shift signals are
    logged once at module import time (see argus/runners.py, right after
    _SYSTEM_REVIEWER_UNOVERRIDDEN is computed), not per-session -- an
    earlier per-call version of the beta-withheld log was flagged as both
    too quiet (debug) and too loud (info x 5 call sites x N rounds) across
    successive Argus reviews of this PR. Reloading argus.runners here is
    the correct tool for exactly this case, unlike the per-call gating
    logic tested above: the behavior under test IS the module-import-time
    side effect, not a re-derivable-without-reload runtime decision.
    """

    def test_warns_once_when_specialist_model_overridden(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        import importlib

        import argus.llm.models as models_module
        import argus.runners as runners_module

        monkeypatch.setenv("ARGUS_SPECIALIST_MODEL", "claude-haiku-4-5")
        try:
            importlib.reload(models_module)
            with caplog.at_level("WARNING", logger=_RUNNERS_MODULE):
                importlib.reload(runners_module)
            assert any("1M-context beta withheld" in record.message for record in caplog.records)
        finally:
            monkeypatch.delenv("ARGUS_SPECIALIST_MODEL", raising=False)
            importlib.reload(models_module)
            importlib.reload(runners_module)

    def test_does_not_warn_when_unoverridden(self, caplog: pytest.LogCaptureFixture) -> None:
        import importlib

        import argus.runners as runners_module

        with caplog.at_level("WARNING", logger=_RUNNERS_MODULE):
            importlib.reload(runners_module)
        assert not any("1M-context beta withheld" in record.message for record in caplog.records)

    def test_warns_once_when_frontier_model_moves_cross_cutting_off_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        import importlib

        import argus.llm.models as models_module
        import argus.runners as runners_module

        monkeypatch.setenv("ARGUS_FRONTIER_MODEL", "claude-fable-5")
        try:
            importlib.reload(models_module)
            with caplog.at_level("WARNING", logger=_RUNNERS_MODULE):
                importlib.reload(runners_module)
            assert any(
                "Cross-cutting reviewer moved off its default model" in record.message
                for record in caplog.records
            )
        finally:
            monkeypatch.delenv("ARGUS_FRONTIER_MODEL", raising=False)
            importlib.reload(models_module)
            importlib.reload(runners_module)
