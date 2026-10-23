"""Exercise the real google-genai retry layer with an in-memory HTTP transport."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from google import genai

from argus.bench import BenchEntry
from argus.gemini_runner import run_session_gemini


@pytest.mark.asyncio
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize(
    ("status", "recover", "deadline", "expected_reason", "expected_calls"),
    [
        (503, True, False, None, 3),
        (503, False, False, "worker_crashed", 4),
        (429, False, False, "worker_crashed", 2),
        (503, False, True, "timeout", 2),
    ],
)
async def test_mid_session_retry_uses_identical_request_without_replaying_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cached: bool,
    status: int,
    recover: bool,
    deadline: bool,
    expected_reason: str | None,
    expected_calls: int,
) -> None:
    from argus import gemini_runner

    requests: list[dict[str, Any]] = []
    tool_calls: list[str] = []
    client_options: list[Any] = []
    real_execute = gemini_runner._execute_tool_call
    real_client = genai.Client
    (tmp_path / "README.md").write_text("test content")

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        n = len(requests)
        if n == 1 or (recover and n == 3):
            name = "read_file" if n == 1 else "finish_review"
            args: dict[str, Any] = (
                {"path": "README.md"} if n == 1 else {"files_explored": ["README.md"]}
            )
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [{"functionCall": {"name": name, "args": args}}],
                            }
                        }
                    ],
                    "usageMetadata": {"promptTokenCount": 1000, "candidatesTokenCount": 100},
                },
            )
        return httpx.Response(
            status,
            json={
                "error": {
                    "code": status,
                    "status": "UNAVAILABLE" if status == 503 else "RESOURCE_EXHAUSTED",
                    "message": "private-provider-payload",
                }
            },
        )

    def make_client(*, api_key: str, http_options: Any) -> genai.Client:
        http_options.httpx_client = httpx.Client(transport=httpx.MockTransport(handle))
        http_options.httpx_async_client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        client_options.append(http_options)
        return real_client(api_key=api_key, http_options=http_options)

    def execute(name: str, args: dict[str, Any]) -> tuple[Any, bool]:
        tool_calls.append(name)
        return real_execute(name, args)

    monkeypatch.setattr(genai, "Client", make_client)
    monkeypatch.setattr(gemini_runner, "_execute_tool_call", execute)
    if cached:
        monkeypatch.setattr(
            gemini_runner, "_maybe_activate_cache", lambda **_: "cachedContents/test"
        )
    monkeypatch.setattr(gemini_runner, "_HTTP_RETRY_INITIAL_DELAY_S", 30.0 if deadline else 0.001)
    monkeypatch.setattr(gemini_runner, "_HTTP_RETRY_MAX_DELAY_S", 60.0 if deadline else 0.002)
    settings = SimpleNamespace(
        google_credential=("GOOGLE_API_KEY", "test-key"),
        GOOGLE_BASE_URL=None,
        ARGUS_SESSION_TIMEOUT=3 if deadline else 10,
    )
    entry = BenchEntry(
        role="system-generalist",
        platform="gemini",
        model="gemini-frontier",
        prompt_name="pr-review-subagent",
        caching="on" if cached else "off",
    )
    result = await run_session_gemini(
        entry=entry,
        system_prompt="sys",
        user_message="review",
        settings=settings,
        repo_root=str(tmp_path),
    )
    assert result.failure_reason == expected_reason
    # Cache creation shares client options; retries belong to model requests only.
    assert client_options and all(options.retry_options is None for options in client_options)
    if cached:
        assert all(request["cachedContent"] == "cachedContents/test" for request in requests)
    # Deadline case: setup is far inside the timeout, so exactly the first
    # turn plus one failed request are sent before the backoff is interrupted.
    assert len(requests) == expected_calls
    assert all(request == requests[1] for request in requests[1:])
    assert tool_calls == (["read_file", "finish_review"] if recover else ["read_file"])
    assert result.tool_call_count == len(tool_calls)
    assert result.tool_names == sorted(tool_calls)
    assert result.cost_usd > 0  # successful first turn survives a later failure
    if expected_reason == "worker_crashed":
        assert result.failure_detail == (
            "503 UNAVAILABLE, turn 1" if status == 503 else "429 RESOURCE_EXHAUSTED, turn 1"
        )
        assert "private-provider-payload" not in result.failure_detail
    if expected_reason == "timeout":
        # The deadline fires during the (much longer) SDK backoff sleep.
        assert result.duration_seconds < 15
