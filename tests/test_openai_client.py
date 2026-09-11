"""Tests for argus.openai_client: OPENAI_BASE_URL threading.

Covers both OpenAI client factories argus.openai_client exposes --
``OpenAIClientSync.__init__`` (sync) and ``get_async_openai_client()``
(async) -- to confirm ``settings.OPENAI_BASE_URL`` reaches the underlying
``openai`` SDK constructor in both cases. ``argus.openai_runner`` already
has this coverage for its own client construction; this file covers
argus.openai_client's two factories, which previously had none.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from argus.openai_client import OpenAIClientSync, get_async_openai_client


def _make_settings(base_url: str | None) -> MagicMock:
    settings = MagicMock()
    settings.OPENAI_API_KEY = "test-openai-key"
    settings.OPENAI_BASE_URL = base_url
    return settings


class TestOpenAIClientSyncBaseURL:
    def test_settings_base_url_passed_to_openai_constructor(self) -> None:
        """OPENAI_BASE_URL on settings reaches the underlying OpenAI(...) call."""
        settings = _make_settings("https://proxy.example.com/v1")

        with (
            patch("argus.openai_client.get_settings", return_value=settings),
            patch("argus.openai_client.OpenAI") as mock_openai,
            patch("argus.openai_client.wrap_openai", side_effect=lambda client: client),
        ):
            OpenAIClientSync()

        mock_openai.assert_called_once()
        assert mock_openai.call_args.kwargs["base_url"] == "https://proxy.example.com/v1"

    def test_no_base_url_override_passes_none(self) -> None:
        """Without an override, base_url is passed through as None (SDK default)."""
        settings = _make_settings(None)

        with (
            patch("argus.openai_client.get_settings", return_value=settings),
            patch("argus.openai_client.OpenAI") as mock_openai,
            patch("argus.openai_client.wrap_openai", side_effect=lambda client: client),
        ):
            OpenAIClientSync()

        mock_openai.assert_called_once()
        assert mock_openai.call_args.kwargs["base_url"] is None


class TestGetAsyncOpenAIClientBaseURL:
    def test_settings_base_url_passed_to_async_openai_constructor(self) -> None:
        """OPENAI_BASE_URL on settings reaches the underlying AsyncOpenAI(...) call."""
        settings = _make_settings("https://proxy.example.com/v1")

        with (
            patch("argus.openai_client.get_settings", return_value=settings),
            patch("argus.openai_client.AsyncOpenAI") as mock_async_openai,
            patch("argus.openai_client.wrap_openai", side_effect=lambda client: client),
        ):
            get_async_openai_client()

        mock_async_openai.assert_called_once()
        assert mock_async_openai.call_args.kwargs["base_url"] == "https://proxy.example.com/v1"

    def test_no_base_url_override_passes_none(self) -> None:
        """Without an override, base_url is passed through as None (SDK default)."""
        settings = _make_settings(None)

        with (
            patch("argus.openai_client.get_settings", return_value=settings),
            patch("argus.openai_client.AsyncOpenAI") as mock_async_openai,
            patch("argus.openai_client.wrap_openai", side_effect=lambda client: client),
        ):
            get_async_openai_client()

        mock_async_openai.assert_called_once()
        assert mock_async_openai.call_args.kwargs["base_url"] is None
