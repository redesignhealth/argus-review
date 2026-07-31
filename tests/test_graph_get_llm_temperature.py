"""Unit tests for _get_llm's temperature-stripping guard.

claude-fable-5 and claude-sonnet-5 (the current CLAUDE_FRONTIER / CLAUDE_DEFAULT
resolutions) reject an explicit ``temperature`` kwarg outright
(``invalid_request_error: "temperature is deprecated for this model"``), while
claude-haiku-4-5 (CLAUDE_MINI) accepts it fine. ``_get_llm`` must therefore
drop ``temperature`` from the kwargs it hands to ``init_chat_model`` for the
former and pass it through for the latter.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from argus.graph import _get_llm


def _patched_settings() -> MagicMock:
    settings = MagicMock()
    settings.anthropic_credential = ("x-api-key", "test-anthropic-key")
    return settings


def test_get_llm_strips_temperature_for_claude_sonnet_5() -> None:
    with (
        patch("argus.graph.get_settings", return_value=_patched_settings()),
        patch("argus.graph.init_chat_model") as mock_init_chat_model,
    ):
        _get_llm("anthropic:claude-sonnet-5", temperature=0)

    mock_init_chat_model.assert_called_once()
    _, kwargs = mock_init_chat_model.call_args
    assert "temperature" not in kwargs


def test_get_llm_strips_temperature_for_claude_fable_5() -> None:
    with (
        patch("argus.graph.get_settings", return_value=_patched_settings()),
        patch("argus.graph.init_chat_model") as mock_init_chat_model,
    ):
        _get_llm("anthropic:claude-fable-5", temperature=0.7)

    mock_init_chat_model.assert_called_once()
    _, kwargs = mock_init_chat_model.call_args
    assert "temperature" not in kwargs


def test_get_llm_passes_temperature_for_claude_haiku_4_5() -> None:
    with (
        patch("argus.graph.get_settings", return_value=_patched_settings()),
        patch("argus.graph.init_chat_model") as mock_init_chat_model,
    ):
        _get_llm("anthropic:claude-haiku-4-5", temperature=0)

    mock_init_chat_model.assert_called_once()
    _, kwargs = mock_init_chat_model.call_args
    assert kwargs.get("temperature") == 0


def test_get_llm_omits_temperature_when_none_regardless_of_model() -> None:
    with (
        patch("argus.graph.get_settings", return_value=_patched_settings()),
        patch("argus.graph.init_chat_model") as mock_init_chat_model,
    ):
        _get_llm("anthropic:claude-haiku-4-5", temperature=None)

    mock_init_chat_model.assert_called_once()
    _, kwargs = mock_init_chat_model.call_args
    assert "temperature" not in kwargs
