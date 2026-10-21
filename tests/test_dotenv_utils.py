"""Tests for ``ARGUS_NO_DOTENV``: the kill switch for every ``.env`` load."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from argus.config import Settings, clear_cache, get_settings
from argus.dotenv_utils import dotenv_disabled, load_dotenv_early

_SENTINEL = "ARGUS_TEST_DOTENV_SENTINEL"


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "On", "t", "y"])
def test_dotenv_disabled_truthy_values(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARGUS_NO_DOTENV", value)
    assert dotenv_disabled() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "enabled", "2"])
def test_dotenv_disabled_falsy_values(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARGUS_NO_DOTENV", value)
    assert dotenv_disabled() is False


def test_dotenv_disabled_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARGUS_NO_DOTENV", raising=False)
    assert dotenv_disabled() is False


def test_load_dotenv_early_is_a_noop_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text(f"{_SENTINEL}=loaded\n")
    monkeypatch.delenv(_SENTINEL, raising=False)
    monkeypatch.setenv("ARGUS_NO_DOTENV", "1")
    assert load_dotenv_early(start=tmp_path, env_filename=".env") is None
    import os

    assert _SENTINEL not in os.environ


def test_load_dotenv_early_still_loads_when_not_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    (tmp_path / ".env").write_text(f"{_SENTINEL}=loaded\n")
    monkeypatch.delenv(_SENTINEL, raising=False)
    monkeypatch.delenv("ARGUS_NO_DOTENV", raising=False)
    try:
        assert load_dotenv_early(start=tmp_path, env_filename=".env") is not None
        assert os.environ[_SENTINEL] == "loaded"
    finally:
        os.environ.pop(_SENTINEL, None)


class TestSettingsHonorKillSwitch:
    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
        (tmp_path / ".env").write_text("ARGUS_GEMINI_CACHE_TTL=7\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("ARGUS_GEMINI_CACHE_TTL", raising=False)
        clear_cache()
        yield
        clear_cache()

    def test_cached_settings_read_dotenv_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARGUS_NO_DOTENV", raising=False)
        assert get_settings().ARGUS_GEMINI_CACHE_TTL == 7

    def test_cached_settings_ignore_dotenv_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ARGUS_NO_DOTENV", "true")
        default = Settings.model_fields["ARGUS_GEMINI_CACHE_TTL"].default
        assert get_settings().ARGUS_GEMINI_CACHE_TTL == default

    def test_direct_settings_construction_also_ignores_dotenv(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ARGUS_NO_DOTENV", "1")
        default = Settings.model_fields["ARGUS_GEMINI_CACHE_TTL"].default
        assert Settings().ARGUS_GEMINI_CACHE_TTL == default

    def test_real_environment_still_wins_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ARGUS_NO_DOTENV", "1")
        monkeypatch.setenv("ARGUS_GEMINI_CACHE_TTL", "42")
        assert Settings().ARGUS_GEMINI_CACHE_TTL == 42
