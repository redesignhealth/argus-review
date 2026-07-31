"""Synchronous OpenAI client using the Responses API with tool support.

This module provides a unified synchronous client for OpenAI API access
using the Responses API (successor to Chat Completions), including support
for function calling (tools) and reasoning models.

Usage:
    from argus.openai_client import OpenAIClientSync

    client = OpenAIClientSync()

    # Simple response
    response = client.respond(
        input="Hello",
        model=GPT_MINI,
    )
    print(response.output_text)

    # With structured output
    response = client.respond(
        input=[{"role": "user", "content": "Extract data..."}],
        model=GPT_MINI,
        text_format={
            "type": "json_schema",
            "name": "output",
            "schema": {...},
        },
    )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import httpx
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI, OpenAI, OpenAIError

from argus.config import get_settings
from argus.llm.models import GPT_MINI

if TYPE_CHECKING:
    from openai.types.responses import Response

logger = logging.getLogger(__name__)


class OpenAIClientError(Exception):
    """Base exception for OpenAI client errors."""

    pass


class OpenAIAPIError(OpenAIClientError):
    """Raised when OpenAI API returns an error."""

    def __init__(self, message: str, status_code: int | None = None):
        self.status_code = status_code
        super().__init__(f"OpenAI API error: {message}")


class OpenAIClientSync:
    """Synchronous OpenAI client using the Responses API.

    This client wraps the OpenAI Python SDK and provides:
    - Simple responses via `respond()`
    - Structured output support via `text_format`
    - Reasoning model support with `reasoning_effort`

    The Responses API offers:
    - Better performance with reasoning models (3% improvement on SWE-bench)
    - Lower costs via improved caching (40-80% improvement)
    - Native agentic tools (web_search, code_interpreter, etc.)
    - Stateful context preservation for multi-turn reasoning

    Attributes:
        DEFAULT_MODEL: Default model to use for completions.
        DEFAULT_TIMEOUT: Default request timeout in seconds.
    """

    DEFAULT_MODEL = GPT_MINI
    DEFAULT_TIMEOUT = 180  # Increased for reasoning models

    def __init__(
        self,
        api_key: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize the OpenAI client.

        Args:
            api_key: OpenAI API key. If not provided, loads from settings.
            timeout: Request timeout in seconds.
        """
        settings = get_settings()
        self._api_key = api_key or settings.OPENAI_API_KEY
        if not self._api_key:
            raise OpenAIClientError(
                "OPENAI_API_KEY not configured. "
                "Set it in environment or pass api_key to constructor."
            )
        self._client = wrap_openai(OpenAI(api_key=self._api_key, timeout=timeout))
        self._timeout = timeout

    @traceable(run_type="llm")
    def respond(
        self,
        input: str | list[dict[str, Any]],
        model: str = DEFAULT_MODEL,
        instructions: str | None = None,
        text_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
        store: bool = True,
        previous_response_id: str | None = None,
        langsmith_extra: dict[str, Any] | None = None,
    ) -> Response:
        """Execute a response request using the Responses API.

        Args:
            input: String prompt or list of message dicts with 'role' and 'content'.
            model: Model identifier (e.g. ``GPT_MINI``).
            instructions: System-level instructions (like system message).
            text_format: Structured output format for text.format parameter.
                Example: {"type": "json_schema", "name": "output", "schema": {...}}
            tools: Optional list of tool definitions for function calling.
                Responses API uses internally-tagged format:
                {"type": "function", "name": "...", "parameters": {...}}
            max_output_tokens: Maximum tokens in response.
            reasoning_effort: For reasoning models: "low", "medium", or "high".
            store: Whether to store the response (default True for context preservation).
            previous_response_id: Chain to a previous response for multi-turn.
            langsmith_extra: Optional dict of LangSmith trace metadata.
                The @traceable decorator automatically picks up this parameter
                to set metadata, tags, and other trace attributes. Example:
                {"metadata": {"user_id": "U123"}, "tags": ["user:U123"]}

        Returns:
            Response object. Use response.output_text for simple text output,
            or response.output for the full list of output items.

        Raises:
            OpenAIAPIError: If the API request fails.
        """
        kwargs: dict[str, Any] = {
            "model": model,
            "input": input,
            "store": store,
        }

        if instructions is not None:
            kwargs["instructions"] = instructions

        if text_format is not None:
            kwargs["text"] = {"format": text_format}

        if tools is not None:
            kwargs["tools"] = tools

        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens

        if reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": reasoning_effort}

        if previous_response_id is not None:
            kwargs["previous_response_id"] = previous_response_id

        logger.debug(
            "OpenAI responses request: model=%s, input_type=%s, tools=%d, reasoning=%s",
            model,
            type(input).__name__,
            len(tools) if tools else 0,
            reasoning_effort,
        )

        try:
            response = self._client.responses.create(**kwargs)
            logger.debug(
                "OpenAI responses response: model=%s, status=%s, output_items=%d",
                response.model,
                response.status,
                len(response.output) if response.output else 0,
            )
            return cast("Response", response)
        except OpenAIError as e:
            logger.error("OpenAI API error: %s", e)
            status_code = getattr(e, "status_code", None)
            raise OpenAIAPIError(str(e), status_code=status_code) from e
        except httpx.HTTPError as e:
            logger.error("Network error in OpenAI request: %s", e, exc_info=True)
            raise OpenAIClientError(f"Network error: {e}") from e
        except (ValueError, TypeError) as e:
            logger.error("Invalid parameters in OpenAI request: %s", e, exc_info=True)
            raise OpenAIClientError(f"Invalid parameters: {e}") from e

    # Backwards compatibility aliases
    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str = DEFAULT_MODEL,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,  # Ignored - kept for signature compatibility
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> Response:
        """Legacy chat method for backwards compatibility.

        Deprecated: Use respond() instead.
        """
        # Convert response_format to text_format
        text_format = None
        if response_format is not None and response_format.get("type") == "json_schema":
            json_schema = response_format.get("json_schema", {})
            text_format = {
                "type": "json_schema",
                "name": json_schema.get("name", "output"),
                "strict": json_schema.get("strict", True),
                "schema": json_schema.get("schema", {}),
            }

        # Convert Chat Completions tools format to Responses format
        responses_tools = None
        if tools:
            responses_tools = self._convert_tools_format(tools)

        return self.respond(
            input=messages,
            model=model,
            text_format=text_format,
            tools=responses_tools,
            max_output_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )

    def _convert_tools_format(self, chat_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Chat Completions tools format to Responses API format.

        Chat Completions (externally tagged):
            {"type": "function", "function": {"name": "...", "parameters": {...}}}

        Responses API (internally tagged):
            {"type": "function", "name": "...", "parameters": {...}}
        """
        responses_tools = []
        for tool in chat_tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                responses_tools.append(
                    {
                        "type": "function",
                        "name": func.get("name"),
                        "description": func.get("description"),
                        "parameters": func.get("parameters"),
                    }
                )
            else:
                # Pass through non-function tools (e.g., web_search)
                responses_tools.append(tool)
        return responses_tools


def get_async_openai_client() -> AsyncOpenAI:
    """Return an AsyncOpenAI client configured from settings.

    Centralises client construction so all async OpenAI callers go through
    a single factory rather than instantiating AsyncOpenAI directly. Always
    creates a fresh client — avoids httpx event-loop binding issues across
    async frameworks.

    The client is wrapped with ``langsmith.wrappers.wrap_openai`` so
    ``client.chat.completions.create`` and ``client.responses.create`` calls
    are auto-traced. The wrapper is a **no-op** when LangSmith env vars are
    not set (``LANGSMITH_API_KEY`` empty), so callers that haven't opted
    into tracing see no behaviour change. Mirrors the sync
    ``OpenAIClientSync`` class above which has wrapped from the start.

    Returns:
        Wrapped AsyncOpenAI client with the API key from settings.
    """
    settings = get_settings()
    return wrap_openai(AsyncOpenAI(api_key=settings.OPENAI_API_KEY))
