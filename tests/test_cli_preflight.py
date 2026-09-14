"""Unit tests for the preflight checks.

``argus review`` must fail fast — before any network call — when ``git`` or
the ``claude`` CLI are missing from PATH, or when required settings are
absent. This file covers the PATH checks for git/claude, the
required-secrets contract of ``_check_settings`` (four API keys by default —
Anthropic, GitHub, OpenAI, and Google; no storage env vars required, given
the local SQLite default), and bench configuration validation.
"""

from __future__ import annotations

import logging
from pathlib import Path

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

    The conftest autouse fixture sets the required API keys; this
    scrubs every storage-related var, applies ``extra``, and reloads.
    """
    for var in _STORAGE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    for key, value in extra.items():
        monkeypatch.setenv(key, value)
    clear_cache()
    return Settings()


class TestCheckSettingsRequiredSecrets:
    """Regression tests for required secrets: baseline API keys (ANTHROPIC_API_KEY,
    GITHUB_TOKEN_RO, OPENAI_API_KEY), conditional provider credentials based on
    the effective bench (GOOGLE_API_KEY for default Gemini bulk reviewers), and
    NO storage env vars."""

    def test_passes_with_required_api_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
        ["ANTHROPIC_API_KEY", "GITHUB_TOKEN_RO", "OPENAI_API_KEY", "GOOGLE_API_KEY"],
    )
    def test_each_api_key_is_required(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        empty_key: str,
    ) -> None:
        """An empty value for any required key → exit 1 naming it."""
        settings = _settings_from_env(monkeypatch, **{empty_key: ""})
        with caplog.at_level(logging.ERROR, logger="argus_review_local"):
            with pytest.raises(SystemExit) as exc:
                argus_cli._check_settings(settings)
        assert exc.value.code == 1
        assert any(empty_key in rec.message for rec in caplog.records)

    def test_missing_google_key_with_default_config_exits(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Default bench resolves bulk_reviewer to Gemini, so GOOGLE_API_KEY is required."""
        settings = _settings_from_env(monkeypatch, GOOGLE_API_KEY="")
        with caplog.at_level(logging.ERROR, logger="argus_review_local"):
            with pytest.raises(SystemExit) as exc:
                argus_cli._check_settings(settings)
        assert exc.value.code == 1
        assert any("GOOGLE_API_KEY" in rec.message for rec in caplog.records)

    def test_missing_google_key_allowed_when_bench_overridden_to_claude(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When bench overrides bulk_reviewer to claude-sdk, GOOGLE_API_KEY is not required."""
        explicit = tmp_path / "bench.toml"
        explicit.write_text('[bulk_reviewer]\nplatform = "claude-sdk"\nmodel = "claude-default"\n')
        monkeypatch.setenv("ARGUS_BENCH_FILE", str(explicit))
        monkeypatch.delenv("ARGUS_NO_BENCH_OVERRIDES", raising=False)
        settings = _settings_from_env(monkeypatch, GOOGLE_API_KEY="")
        argus_cli._check_settings(settings)  # must not raise

    def test_invalid_bench_config_fails_preflight(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An invalid bench config (e.g. mismatched platform/model) must fail
        fast in preflight with exit 1 and a clear error message, rather than
        failing later mid-review."""
        explicit = tmp_path / "invalid-bench.toml"
        explicit.write_text('[bulk_reviewer]\nplatform = "gemini"\nmodel = "claude-mini"\n')
        monkeypatch.setenv("ARGUS_BENCH_FILE", str(explicit))
        monkeypatch.delenv("ARGUS_NO_BENCH_OVERRIDES", raising=False)
        settings = _settings_from_env(monkeypatch)
        with caplog.at_level(logging.ERROR, logger="argus_review_local"):
            with pytest.raises(SystemExit) as exc:
                argus_cli._check_settings(settings)
        assert exc.value.code == 1
        assert any("Invalid bench configuration" in rec.message for rec in caplog.records)
        assert any(
            "model 'claude-mini' is not compatible with platform 'gemini'" in rec.message
            for rec in caplog.records
        )


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

        from argus.cli import _MODEL_OVERRIDE_UNSET
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
            specialist_model=_MODEL_OVERRIDE_UNSET,
            frontier_model=_MODEL_OVERRIDE_UNSET,
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
