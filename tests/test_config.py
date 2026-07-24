"""Tests for argus.config: env precedence, aliasing, caching."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from argus.config import clear_cache, get_settings


@pytest.fixture(autouse=True)
def _clear_cache_before_and_after() -> Generator[None, None, None]:
    clear_cache()
    yield
    clear_cache()


def _set_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setenv("GITHUB_TOKEN_RO", "github-token")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")


def test_get_settings_reads_required_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required(monkeypatch)
    settings = get_settings()
    assert settings.ANTHROPIC_API_KEY == "anthropic-key"
    assert settings.GITHUB_TOKEN_RO == "github-token"
    assert settings.OPENAI_API_KEY == "openai-key"


def test_missing_required_var_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing ANTHROPIC_API_KEY (or any required var) fails loudly at construction."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN_RO", "github-token")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    with pytest.raises(Exception, match="ANTHROPIC_API_KEY"):
        get_settings()


def test_supabase_db_url_alias_used_when_argus_db_url_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required(monkeypatch)
    monkeypatch.delenv("ARGUS_DB_URL", raising=False)
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://legacy/db")

    settings = get_settings()
    assert settings.db_url == "postgresql://legacy/db"


def test_argus_db_url_wins_when_both_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """ARGUS_DB_URL takes precedence over the SUPABASE_DB_URL back-compat alias."""
    _set_required(monkeypatch)
    monkeypatch.setenv("ARGUS_DB_URL", "postgresql://new/db")
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://legacy/db")

    settings = get_settings()
    assert settings.db_url == "postgresql://new/db"


def test_db_url_none_when_neither_set(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required(monkeypatch)
    monkeypatch.delenv("ARGUS_DB_URL", raising=False)
    monkeypatch.delenv("SUPABASE_DB_URL", raising=False)

    settings = get_settings()
    assert settings.db_url is None


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated calls return the same cached instance until clear_cache()."""
    _set_required(monkeypatch)
    first = get_settings()
    second = get_settings()
    assert first is second


def test_clear_cache_forces_reload(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required(monkeypatch)
    first = get_settings()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "rotated-key")
    clear_cache()
    second = get_settings()

    assert first is not second
    assert second.ANTHROPIC_API_KEY == "rotated-key"


def test_optional_fields_default_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required(monkeypatch)
    for var in (
        "ARGUS_STORAGE_READ_URL",
        "ARGUS_STORAGE_WRITE_URL",
        "ARGUS_STORAGE_AUTH",
        "ARGUS_SQLITE_CHECKPOINT_PATH",
        "ARGUS_PROMPTS_DIR",
        "ARGUS_REPO_CACHE",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
        "CONTEXT7_API_KEY",
        "ARGUS_CONTEXT7_LIBRARY_ID",
    ):
        monkeypatch.delenv(var, raising=False)

    settings = get_settings()
    assert settings.CONTEXT7_API_KEY is None
    assert settings.ARGUS_CONTEXT7_LIBRARY_ID is None
    assert settings.ARGUS_PROMPTS_DIR is None


def test_session_timeout_defaults_to_300(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required(monkeypatch)
    monkeypatch.delenv("ARGUS_SESSION_TIMEOUT", raising=False)
    settings = get_settings()
    assert settings.ARGUS_SESSION_TIMEOUT == 300


def test_session_timeout_override(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required(monkeypatch)
    monkeypatch.setenv("ARGUS_SESSION_TIMEOUT", "120")
    settings = get_settings()
    assert settings.ARGUS_SESSION_TIMEOUT == 120


def test_no_prompt_overrides_defaults_to_false(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required(monkeypatch)
    monkeypatch.delenv("ARGUS_NO_PROMPT_OVERRIDES", raising=False)
    settings = get_settings()
    assert settings.ARGUS_NO_PROMPT_OVERRIDES is False


def test_no_prompt_overrides_parses_truthy_env_value(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required(monkeypatch)
    monkeypatch.setenv("ARGUS_NO_PROMPT_OVERRIDES", "1")
    settings = get_settings()
    assert settings.ARGUS_NO_PROMPT_OVERRIDES is True
