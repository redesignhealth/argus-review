"""Unit tests for the local Argus runner's settings loading and watchdog.

Argus is env-var / ``.env``-only configuration — no AWS, no SSM. These
tests exercise ``_load_settings`` / ``_check_settings`` / the watchdog
helpers directly.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from argus import cli as argus_review_local


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset Settings cache and scrub env vars so tests don't leak state."""
    from argus.config import clear_cache

    clear_cache()
    for var in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GITHUB_TOKEN_RO",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
        "ARGUS_DB_URL",
        "SUPABASE_DB_URL",
        "CONTEXT7_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


def _stub_settings(db_url: str | None = "stub-db-url") -> MagicMock:
    settings = MagicMock()
    for k in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GITHUB_TOKEN_RO",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
        "CONTEXT7_API_KEY",
    ):
        setattr(settings, k, f"stub-{k.lower()}")
    settings.db_url = db_url
    return settings


def test_load_settings_propagates_secrets_to_environ() -> None:
    """Loaded secrets are mirrored into os.environ so subprocess code reads them."""
    with (
        patch("argus.dotenv_utils.load_dotenv_early", return_value=None),
        patch("argus.config.get_settings", return_value=_stub_settings()),
    ):
        argus_review_local._load_settings()
    assert os.environ["ANTHROPIC_API_KEY"] == "stub-anthropic_api_key"
    assert os.environ["SUPABASE_DB_URL"] == "stub-db-url"


def test_check_settings_exits_when_required_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Missing critical secrets → sys.exit(1) with a clear message."""
    import logging

    settings = MagicMock()
    settings.ANTHROPIC_API_KEY = None
    settings.GITHUB_TOKEN_RO = "gh"
    settings.OPENAI_API_KEY = "oa"
    settings.db_url = "url"

    with caplog.at_level(logging.ERROR, logger="argus_review_local"):
        with pytest.raises(SystemExit) as exc:
            argus_review_local._check_settings(settings)
    assert exc.value.code == 1
    assert any("ANTHROPIC_API_KEY" in rec.message for rec in caplog.records)


def test_check_settings_does_not_require_db_url() -> None:
    """A DB URL is NOT required: no ARGUS_DB_URL / SUPABASE_DB_URL
    and no HTTP storage URLs → SQLite history + checkpoints is the default."""
    settings = MagicMock()
    settings.ANTHROPIC_API_KEY = "a"
    settings.GITHUB_TOKEN_RO = "gh"
    settings.OPENAI_API_KEY = "oa"
    settings.db_url = None

    # Should not raise.
    argus_review_local._check_settings(settings)


def test_start_watchdog_starts_named_daemon_thread() -> None:
    """``_start_watchdog`` must spawn a daemon thread named ``argus-watchdog``.

    We patch ``threading.Thread`` so the captured target is never actually run -
    no real ``time.sleep`` of the full timeout and no real ``os._exit``.
    """
    captured: dict[str, object] = {}

    class _FakeThread:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)
            self.daemon = bool(kwargs.get("daemon"))
            self.name = str(kwargs.get("name"))

        def start(self) -> None:
            captured["started"] = True

    with patch.object(argus_review_local.threading, "Thread", _FakeThread):
        thread = argus_review_local._start_watchdog("owner/repo", 123)

    assert captured["daemon"] is True
    assert captured["name"] == "argus-watchdog"
    assert captured["started"] is True
    assert thread.daemon is True
    assert thread.name == "argus-watchdog"


def test_watchdog_closure_body_sweeps_and_exits() -> None:
    """Exercise the ``_watchdog`` closure body without sleeping or really exiting.

    ``_start_watchdog`` creates a daemon thread whose target is the inner
    ``_watchdog`` closure.  We capture that callable and call it directly after
    patching out the three side-effecting calls: ``time.sleep`` (no-op),
    ``os._exit`` (MagicMock so it does NOT exit), and
    ``_sweep_stale_argus_tempdirs`` (MagicMock).  We then assert both
    were called with the expected arguments.
    """
    captured: dict[str, object] = {}

    class _FakeThread:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)
            self.daemon = bool(kwargs.get("daemon"))
            self.name = str(kwargs.get("name"))

        def start(self) -> None:
            pass  # do not start a real thread

    mock_exit = MagicMock()
    mock_sweep = MagicMock()

    with (
        patch.object(argus_review_local.threading, "Thread", _FakeThread),
        patch.object(argus_review_local.time, "sleep"),
        patch.object(argus_review_local.os, "_exit", mock_exit),
        patch.object(argus_review_local, "_sweep_stale_argus_tempdirs", mock_sweep),
    ):
        argus_review_local._start_watchdog("org/repo", 42)
        target = captured["target"]
        assert callable(target)
        target()  # run the closure body directly

    mock_sweep.assert_called_once()
    mock_exit.assert_called_once_with(1)


def test_sweep_stale_argus_tempdirs_removes_only_argus_artifacts(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sweep removes ``argus-worktree-*`` dirs and ``argus-gitcfg-*`` files
    but leaves unrelated entries in the same temp dir untouched."""
    import tempfile as _tempfile
    from pathlib import Path as _Path

    tmp_dir = _Path(str(tmp_path))
    worktree = tmp_dir / "argus-worktree-abc123"
    worktree.mkdir()
    (worktree / "nested.txt").write_text("x", encoding="utf-8")
    gitcfg = tmp_dir / "argus-gitcfg-xyz789"
    gitcfg.write_text("token", encoding="utf-8")
    unrelated = tmp_dir / "keep-me.txt"
    unrelated.write_text("safe", encoding="utf-8")

    monkeypatch.setattr(_tempfile, "gettempdir", lambda: str(tmp_dir))

    argus_review_local._sweep_stale_argus_tempdirs()

    assert not worktree.exists()
    assert not gitcfg.exists()
    assert unrelated.exists()
