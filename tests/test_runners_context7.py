"""Tests for the Context7 MCP opt-out gating in argus.runners.

ARGUS_CONTEXT7_LIBRARY_ID is deployment-specific; when unset, Context7
attachment must be skipped gracefully (same as a missing CONTEXT7_API_KEY),
matching the graceful-degradation contract for optional Context7 access.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from argus.runners import _context7_system_directive, _validator_worker

_RUNNERS_MODULE = "argus.runners"


def _settings(**overrides: str | None) -> SimpleNamespace:
    base = {
        "CONTEXT7_API_KEY": None,
        "ARGUS_CONTEXT7_LIBRARY_ID": None,
        "ARGUS_CONTEXT7_BASE_URL": None,
        "anthropic_credential": ("ANTHROPIC_API_KEY", "sk-test"),
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_no_directive_when_both_unset() -> None:
    assert _context7_system_directive(_settings()) == ""


def test_no_directive_when_only_api_key_set() -> None:
    settings = _settings(CONTEXT7_API_KEY="key-123")
    assert _context7_system_directive(settings) == ""


def test_no_directive_when_only_library_id_set() -> None:
    settings = _settings(ARGUS_CONTEXT7_LIBRARY_ID="/owner/repo")
    assert _context7_system_directive(settings) == ""


def test_directive_present_when_both_set() -> None:
    settings = _settings(CONTEXT7_API_KEY="key-123", ARGUS_CONTEXT7_LIBRARY_ID="/owner/repo")
    directive = _context7_system_directive(settings)
    assert directive != ""
    assert "/owner/repo" in directive


class TestValidatorWorkerEnvForwarding:
    """B7 regression: ARGUS_CONTEXT7_LIBRARY_ID must reach the subprocess env.

    ``_validator_worker`` runs in a spawned process with a fresh environment;
    it must forward both CONTEXT7_API_KEY and ARGUS_CONTEXT7_LIBRARY_ID (not
    just the API key) so the freshly-loaded settings in that process attach
    Context7 the same way the parent decided to.
    """

    @pytest.fixture(autouse=True)
    def _isolate_worker_env(self, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
        # Never actually call os.setsid() from a test process, and never
        # actually run the Claude session coroutine — just close it so
        # nothing warns about a never-awaited coroutine.
        monkeypatch.setattr(os, "setsid", lambda: None)
        monkeypatch.setattr(asyncio, "run", lambda coro: coro.close())
        monkeypatch.delenv("ARGUS_CONTEXT7_LIBRARY_ID", raising=False)
        monkeypatch.delenv("CONTEXT7_API_KEY", raising=False)
        monkeypatch.delenv("ARGUS_CONTEXT7_BASE_URL", raising=False)
        # _validator_worker mutates os.environ directly (not via monkeypatch),
        # so teardown must delenv again -- otherwise a test that sets one of
        # these leaks it into every test that runs afterward, in this class
        # and beyond, for the rest of the process.
        yield
        monkeypatch.delenv("ARGUS_CONTEXT7_LIBRARY_ID", raising=False)
        monkeypatch.delenv("CONTEXT7_API_KEY", raising=False)
        monkeypatch.delenv("ARGUS_CONTEXT7_BASE_URL", raising=False)

    def test_forwards_library_id_alongside_api_key(self) -> None:
        result_pipe = MagicMock()

        _validator_worker(
            result_pipe,
            "claude-model",
            "system prompt",
            "user message",
            "anthropic-key",
            None,
            "context7-key",
            "/owner/repo",
            None,
            ".",
            "label",
        )

        assert os.environ.get("CONTEXT7_API_KEY") == "context7-key"
        assert os.environ.get("ARGUS_CONTEXT7_LIBRARY_ID") == "/owner/repo"

    def test_does_not_set_library_id_env_when_absent(self) -> None:
        result_pipe = MagicMock()

        _validator_worker(
            result_pipe,
            "claude-model",
            "system prompt",
            "user message",
            "anthropic-key",
            None,
            "context7-key",
            None,
            None,
            ".",
            "label",
        )

        assert "ARGUS_CONTEXT7_LIBRARY_ID" not in os.environ

    def test_forwards_base_url_override_alongside_api_key(self) -> None:
        result_pipe = MagicMock()

        _validator_worker(
            result_pipe=result_pipe,
            model="claude-model",
            system_prompt="system prompt",
            user_message="user message",
            anthropic_api_key="anthropic-key",
            anthropic_auth_token=None,
            context7_key="context7-key",
            context7_library_id="/owner/repo",
            context7_base_url="https://rh-mcp.example.com/argus-proxy/context7",
            cwd=".",
            label="label",
        )

        assert (
            os.environ.get("ARGUS_CONTEXT7_BASE_URL")
            == "https://rh-mcp.example.com/argus-proxy/context7"
        )

    def test_does_not_set_base_url_env_when_absent(self) -> None:
        result_pipe = MagicMock()

        _validator_worker(
            result_pipe=result_pipe,
            model="claude-model",
            system_prompt="system prompt",
            user_message="user message",
            anthropic_api_key="anthropic-key",
            anthropic_auth_token=None,
            context7_key="context7-key",
            context7_library_id="/owner/repo",
            context7_base_url=None,
            cwd=".",
            label="label",
        )

        assert "ARGUS_CONTEXT7_BASE_URL" not in os.environ


class TestContext7McpServerUrlOverride:
    """TECH-4736: ARGUS_CONTEXT7_BASE_URL redirects the Context7 MCP server
    entry so a caller proxying the Context7 key (e.g. through rh-mcp) doesn't
    send that proxy-issued credential to the real mcp.context7.com, which
    wouldn't recognize it."""

    class _FakeClient:
        def __init__(self, messages: list) -> None:
            self._messages = messages

        async def query(self, _message: str) -> None:
            return None

        async def receive_response(self):
            for message in self._messages:
                yield message

        async def __aenter__(self) -> "TestContext7McpServerUrlOverride._FakeClient":
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

    def _run(self, settings: SimpleNamespace) -> MagicMock:
        from claude_agent_sdk import ResultMessage

        from argus.runners import _run_claude_session

        messages = [
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
        fake_client = self._FakeClient(messages)
        with (
            patch(f"{_RUNNERS_MODULE}.ClaudeSDKClient", return_value=fake_client),
            patch(f"{_RUNNERS_MODULE}.ClaudeAgentOptions") as mock_options,
        ):
            asyncio.run(
                _run_claude_session(
                    model="claude-sonnet-4-6",
                    system_prompt="system",
                    user_message="review this",
                    settings=settings,
                    label="test-label",
                    repo_root="/tmp/fake-repo",
                )
            )
        return mock_options

    def test_default_url_used_when_no_override(self) -> None:
        mock_options = self._run(
            _settings(CONTEXT7_API_KEY="key-123", ARGUS_CONTEXT7_LIBRARY_ID="/owner/repo")
        )
        mcp_servers = mock_options.call_args.kwargs["mcp_servers"]
        assert mcp_servers["context7"]["url"] == "https://mcp.context7.com/mcp"

    def test_override_url_used_when_configured(self) -> None:
        mock_options = self._run(
            _settings(
                CONTEXT7_API_KEY="key-123",
                ARGUS_CONTEXT7_LIBRARY_ID="/owner/repo",
                ARGUS_CONTEXT7_BASE_URL="https://rh-mcp.example.com/argus-proxy/context7",
            )
        )
        mcp_servers = mock_options.call_args.kwargs["mcp_servers"]
        assert mcp_servers["context7"]["url"] == "https://rh-mcp.example.com/argus-proxy/context7"
