"""Unit tests for the preflight checks.

``argus review`` must fail fast — before any network call — when ``git`` or
the ``claude`` CLI are missing from PATH, or when required settings are
absent. This file covers the PATH checks for git/claude and the
required-secrets contract of ``_check_settings`` (three API keys only —
no storage env vars required, given the local SQLite default).
"""

from __future__ import annotations

import logging

import pytest

from argus import cli as argus_cli
from argus.config import Settings, clear_cache


def _which_stub(present: set[str]):
    def _which(name: str) -> str | None:
        return f"/usr/bin/{name}" if name in present else None

    return _which


_STORAGE_ENV_VARS = (
    "ARGUS_DB_URL",
    "SUPABASE_DB_URL",
    "ARGUS_STORAGE_READ_URL",
    "ARGUS_STORAGE_WRITE_URL",
    "ARGUS_STORAGE_AUTH",
    "ARGUS_SQLITE_CHECKPOINT_PATH",
)


def _settings_from_env(monkeypatch: pytest.MonkeyPatch, **extra: str) -> Settings:
    """Build a real Settings object from a controlled environment.

    The conftest autouse fixture sets the three required API keys; this
    scrubs every storage-related var, applies ``extra``, and reloads.
    """
    for var in _STORAGE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    for key, value in extra.items():
        monkeypatch.setenv(key, value)
    clear_cache()
    return Settings()


class TestCheckSettingsRequiredSecrets:
    """Regression tests for the SQLite-default contract: exactly
    three required secrets (ANTHROPIC_API_KEY, GITHUB_TOKEN_RO,
    OPENAI_API_KEY) and NO storage env vars."""

    def test_passes_with_only_three_api_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The live-E2E regression: no DB URL, no HTTP URLs → must pass
        (history + checkpoints default to local SQLite)."""
        settings = _settings_from_env(monkeypatch)
        assert settings.db_url is None
        argus_cli._check_settings(settings)  # must not raise

    def test_passes_in_postgres_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = _settings_from_env(
            monkeypatch, ARGUS_DB_URL="postgresql+asyncpg://user:pw@localhost/argus"
        )
        argus_cli._check_settings(settings)  # must not raise

    def test_passes_in_http_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = _settings_from_env(
            monkeypatch,
            ARGUS_STORAGE_READ_URL="https://api.example.com/reviews/{owner}/{repo}/{pr}",
            ARGUS_STORAGE_WRITE_URL="https://api.example.com/reviews/{owner}/{repo}/{pr}/rounds",
        )
        assert settings.db_url is None
        argus_cli._check_settings(settings)  # must not raise

    @pytest.mark.parametrize(
        "empty_key",
        ["ANTHROPIC_API_KEY", "GITHUB_TOKEN_RO", "OPENAI_API_KEY"],
    )
    def test_each_api_key_is_required(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        empty_key: str,
    ) -> None:
        """An empty value for any of the three keys → exit 1 naming it."""
        settings = _settings_from_env(monkeypatch, **{empty_key: ""})
        with caplog.at_level(logging.ERROR, logger="argus_review_local"):
            with pytest.raises(SystemExit) as exc:
                argus_cli._check_settings(settings)
        assert exc.value.code == 1
        assert any(empty_key in rec.message for rec in caplog.records)


class TestCheckPrerequisites:
    def test_passes_when_both_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import shutil

        monkeypatch.setattr(shutil, "which", _which_stub({"git", "claude"}))
        argus_cli._check_prerequisites()  # must not raise

    def test_exits_when_git_missing(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        import shutil

        monkeypatch.setattr(shutil, "which", _which_stub({"claude"}))
        with caplog.at_level(logging.ERROR, logger="argus_review_local"):
            with pytest.raises(SystemExit) as exc:
                argus_cli._check_prerequisites()
        assert exc.value.code == 1
        assert any("`git` was not found on PATH" in rec.message for rec in caplog.records)

    def test_exits_when_claude_missing(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        import shutil

        monkeypatch.setattr(shutil, "which", _which_stub({"git"}))
        with caplog.at_level(logging.ERROR, logger="argus_review_local"):
            with pytest.raises(SystemExit) as exc:
                argus_cli._check_prerequisites()
        assert exc.value.code == 1
        assert any("`claude` CLI was not found on PATH" in rec.message for rec in caplog.records)

    def test_exits_when_both_missing_reports_both(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        import shutil

        monkeypatch.setattr(shutil, "which", _which_stub(set()))
        with caplog.at_level(logging.ERROR, logger="argus_review_local"):
            with pytest.raises(SystemExit) as exc:
                argus_cli._check_prerequisites()
        assert exc.value.code == 1
        messages = [rec.message for rec in caplog.records]
        assert any("`git` was not found on PATH" in m for m in messages)
        assert any("`claude` CLI was not found on PATH" in m for m in messages)


class TestRunReviewConnectivityErrorExit:
    """A bad ARGUS_DB_URL/ARGUS_HISTORY_DB_PATH must exit cleanly (like a
    missing required secret) rather than let HistoryBackendConnectivityError
    escape as a raw traceback -- the exact UX gap this catch exists to close."""

    def test_exits_cleanly_instead_of_raising(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import argparse
        import shutil
        from unittest.mock import MagicMock, patch

        from argus.storage.resolver import HistoryBackendConnectivityError

        monkeypatch.setattr(shutil, "which", _which_stub({"git", "claude"}))
        args = argparse.Namespace(
            repo_positional="org/repo",
            repo_flag=None,
            storage_read_url=None,
            storage_write_url=None,
            storage_auth=None,
            pr=42,
            sha=None,
            no_prompt_overrides=False,
            base_ref=None,
            dismiss=[],
            post=False,
            commit_status=False,
            output=None,
        )
        parser = MagicMock()

        with (
            patch("argus.cli._start_watchdog"),
            patch(
                "argus.cli.run",
                side_effect=HistoryBackendConnectivityError("bad ARGUS_DB_URL"),
            ),
            caplog.at_level(logging.ERROR, logger="argus_review_local"),
        ):
            with pytest.raises(SystemExit) as exc:
                argus_cli._run_review(parser, args)

        assert exc.value.code == 1
        assert any("bad ARGUS_DB_URL" in rec.message for rec in caplog.records)
