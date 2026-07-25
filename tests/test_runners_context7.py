"""Tests for the Context7 MCP opt-out gating in argus.runners.

ARGUS_CONTEXT7_LIBRARY_ID is deployment-specific; when unset, Context7
attachment must be skipped gracefully (same as a missing CONTEXT7_API_KEY),
matching the graceful-degradation contract for optional Context7 access.
"""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from argus.runners import _context7_system_directive, _validator_worker


def _settings(**overrides: str | None) -> SimpleNamespace:
    base = {"CONTEXT7_API_KEY": None, "ARGUS_CONTEXT7_LIBRARY_ID": None}
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
    def _isolate_worker_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Never actually call os.setsid() from a test process, and never
        # actually run the Claude session coroutine — just close it so
        # nothing warns about a never-awaited coroutine.
        monkeypatch.setattr(os, "setsid", lambda: None)
        monkeypatch.setattr(asyncio, "run", lambda coro: coro.close())
        monkeypatch.delenv("ARGUS_CONTEXT7_LIBRARY_ID", raising=False)
        monkeypatch.delenv("CONTEXT7_API_KEY", raising=False)

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
            ".",
            "label",
        )

        assert "ARGUS_CONTEXT7_LIBRARY_ID" not in os.environ
