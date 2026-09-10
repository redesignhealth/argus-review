"""Tests for argus.gemini_runner: the real Gemini leaf-reviewer runner.

Mocks the google-genai client entirely -- no real API calls. Covers:
single-turn termination with no tool calls, multi-turn tool-calling with
report_finding accumulating into the review_tools findings sink and
appearing in the final result_text JSON blob, multiple simultaneous
function calls handled in one turn, explicit-cache activation (verifying
cached_content is set and system_instruction/tools are NOT duplicated on
subsequent requests), graceful degradation when cache creation fails,
usage accumulation across turns, and the timeout path returning
failure_reason="timeout" rather than raising.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# `google-genai` is an OPTIONAL dependency (`[project.optional-dependencies]
# gemini`), not a hard one -- see test_argus_importable_without_gemini_extra
# in tests/test_packaging.py. Skip this whole module cleanly (rather than
# erroring out at collection time) when it isn't installed. This must run
# before any other import that transitively pulls in `google.genai`,
# including `argus.gemini_runner` itself.
pytest.importorskip("google.genai")
from google.genai import errors as genai_errors  # noqa: E402
from google.genai import types  # noqa: E402

from argus.bench import BenchEntry  # noqa: E402
from argus.gemini_runner import run_session_gemini  # noqa: E402
from argus.llm.models import resolve as resolve_alias  # noqa: E402

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_gemini_cache_dir(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Point the file-backed cache keeper at a throwaway directory.

    Every test that activates caching relies on this being isolated from
    a developer's real ``~/.local/share/argus/gemini-cache/``.
    """
    from argus.config import clear_cache as clear_settings_cache

    monkeypatch.setenv("ARGUS_GEMINI_CACHE_DIR", str(tmp_path / "gemini-cache"))
    clear_settings_cache()
    yield
    clear_settings_cache()


def _make_entry(caching: str = "off", role: str = "system-generalist") -> BenchEntry:
    return BenchEntry(
        role=role,
        platform="gemini",
        model="gemini-frontier",
        prompt_name="pr-review-subagent",
        caching=caching,
    )


def _make_settings(session_timeout: float = 300) -> MagicMock:
    settings = MagicMock()
    settings.google_credential = ("GOOGLE_API_KEY", "test-google-key")
    settings.ARGUS_SESSION_TIMEOUT = session_timeout
    return settings


def _make_usage(
    prompt: int = 0,
    candidates: int = 0,
    cached: int = 0,
    thoughts: int = 0,
    tool_use_prompt: int = 0,
    total: int | None = None,
) -> MagicMock:
    usage = MagicMock()
    usage.prompt_token_count = prompt
    usage.candidates_token_count = candidates
    usage.cached_content_token_count = cached
    usage.thoughts_token_count = thoughts
    usage.tool_use_prompt_token_count = tool_use_prompt
    usage.total_token_count = (
        total if total is not None else prompt + candidates + thoughts + tool_use_prompt
    )
    return usage


def _make_function_call(name: str, args: dict[str, Any]) -> MagicMock:
    call = MagicMock()
    call.name = name
    call.args = args
    return call


def _make_response(
    calls: list[tuple[str, dict[str, Any]]] | None = None,
    usage: MagicMock | None = None,
    text: str = "",
    thought_signatures: dict[str, bytes] | None = None,
) -> MagicMock:
    """Build a fake ``GenerateContentResponse``.

    ``response.function_calls`` is set directly (mirroring the SDK's own
    convenience property), and ``response.candidates[0].content`` is built
    as a REAL ``types.Content``/``types.Part.from_function_call(...)``
    object -- not a bare ``MagicMock`` -- so a test can assert on the exact
    object the runner appends to ``contents`` for the next turn, including
    a per-call ``thought_signature`` (via ``thought_signatures``, keyed by
    call name) the way Gemini 3.x actually attaches one.
    """
    response = MagicMock()
    response.function_calls = [_make_function_call(n, a) for n, a in (calls or [])]
    response.usage_metadata = usage
    response.text = text

    parts = []
    for name, args in calls or []:
        part = types.Part.from_function_call(name=name, args=args)
        if thought_signatures and name in thought_signatures:
            part.thought_signature = thought_signatures[name]
        parts.append(part)
    if parts:
        candidate = MagicMock()
        candidate.content = types.Content(role="model", parts=parts)
        response.candidates = [candidate]
    else:
        response.candidates = []
    return response


def _make_fake_client(responses: list[MagicMock], cache_name: str | None = None) -> MagicMock:
    """Build a fake genai.Client with .aio.models.generate_content and
    .caches.create wired up for a test.

    ``client.close()`` and ``client.aio.aclose()`` are also configured
    (the latter as an ``AsyncMock``, since the real SDK's ``aclose`` is a
    coroutine) -- the runner closes both on every exit path.
    """
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(side_effect=responses)
    client.aio.aclose = AsyncMock()
    if cache_name is not None:
        cache = MagicMock()
        cache.name = cache_name
        client.caches.create = MagicMock(return_value=cache)
    return client


def _parse_result_json(result_text: str) -> dict[str, Any]:
    assert result_text.startswith("```json\n")
    assert result_text.endswith("\n```")
    body = result_text[len("```json\n") : -len("\n```")]
    return json.loads(body)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Single-turn, no tool calls
# ---------------------------------------------------------------------------


class TestSingleTurnNoToolCall:
    async def test_no_function_calls_terminates_immediately(self) -> None:
        entry = _make_entry(caching="off")
        settings = _make_settings()
        response = _make_response(calls=[], usage=_make_usage(10, 5), text="Nothing to report.")
        fake_client = _make_fake_client([response])

        with patch("argus.gemini_runner.genai.Client", return_value=fake_client):
            result = await run_session_gemini(
                entry=entry,
                system_prompt="You are a reviewer.",
                user_message="Review this diff.",
                settings=settings,
                label="test",
                repo_root="/tmp/does-not-need-to-exist",
            )

        assert result.failure_reason is None
        assert result.tool_call_count == 0
        assert result.tool_names == []
        assert result.model == resolve_alias("gemini-frontier")
        payload = _parse_result_json(result.result_text)
        assert payload == {"findings": [], "files_explored": []}
        fake_client.aio.models.generate_content.assert_awaited_once()


# ---------------------------------------------------------------------------
# Multi-turn tool-calling
# ---------------------------------------------------------------------------


class TestMultiTurnToolCalling:
    async def test_report_finding_accumulates_into_sink_and_result_text(self) -> None:
        entry = _make_entry(caching="off")
        settings = _make_settings()

        turn1 = _make_response(
            calls=[
                (
                    "report_finding",
                    {
                        "file": "src/app.py",
                        "line": 42,
                        "description": "Off-by-one error",
                        "context": "for i in range(n+1):",
                    },
                )
            ],
            usage=_make_usage(100, 20),
        )
        turn2 = _make_response(
            calls=[("finish_review", {"files_explored": ["src/app.py"]})],
            usage=_make_usage(50, 10),
        )
        fake_client = _make_fake_client([turn1, turn2])

        with patch("argus.gemini_runner.genai.Client", return_value=fake_client):
            result = await run_session_gemini(
                entry=entry,
                system_prompt="You are a reviewer.",
                user_message="Review this diff.",
                settings=settings,
                label="test",
                repo_root="/tmp/does-not-need-to-exist",
            )

        assert result.tool_call_count == 2
        assert result.tool_names == ["finish_review", "report_finding"]
        payload = _parse_result_json(result.result_text)
        assert payload["files_explored"] == ["src/app.py"]
        assert len(payload["findings"]) == 1
        finding = payload["findings"][0]
        assert finding["file"] == "src/app.py"
        assert finding["line"] == 42
        assert finding["description"] == "Off-by-one error"
        assert finding["context"] == "for i in range(n+1):"
        assert fake_client.aio.models.generate_content.await_count == 2

    async def test_finish_review_stops_the_loop_even_with_turns_remaining(self) -> None:
        entry = _make_entry(caching="off")
        settings = _make_settings()

        turn1 = _make_response(calls=[("finish_review", {"files_explored": []})])
        # A third response that must never be consumed, since finish_review
        # in turn1 should stop the loop.
        fake_client = _make_fake_client([turn1, _make_response(calls=[])])

        with patch("argus.gemini_runner.genai.Client", return_value=fake_client):
            result = await run_session_gemini(
                entry=entry,
                system_prompt="sys",
                user_message="msg",
                settings=settings,
                repo_root="/tmp/does-not-need-to-exist",
            )

        assert result.tool_call_count == 1
        assert fake_client.aio.models.generate_content.await_count == 1


class TestMultipleSimultaneousFunctionCalls:
    async def test_multiple_calls_in_one_turn_all_executed_and_bundled(self) -> None:
        entry = _make_entry(caching="off")
        settings = _make_settings()

        turn1 = _make_response(
            calls=[
                ("report_finding", {"file": "a.py", "line": 1, "description": "issue A"}),
                ("report_finding", {"file": "b.py", "line": 2, "description": "issue B"}),
            ],
            usage=_make_usage(10, 5),
        )
        turn2 = _make_response(calls=[("finish_review", {"files_explored": ["a.py", "b.py"]})])
        fake_client = _make_fake_client([turn1, turn2])

        with patch("argus.gemini_runner.genai.Client", return_value=fake_client):
            result = await run_session_gemini(
                entry=entry,
                system_prompt="sys",
                user_message="msg",
                settings=settings,
                repo_root="/tmp/does-not-need-to-exist",
            )

        assert result.tool_call_count == 3
        payload = _parse_result_json(result.result_text)
        assert len(payload["findings"]) == 2
        descriptions = {f["description"] for f in payload["findings"]}
        assert descriptions == {"issue A", "issue B"}

        # Both function calls from turn 1 must have been bundled into a
        # SINGLE follow-up turn (turn1's model+response Content pair),
        # not one round-trip per call -- only 2 generate_content calls
        # total for 3 tool calls across 2 turns.
        assert fake_client.aio.models.generate_content.await_count == 2

        # `contents` is one shared, mutated-in-place list across the whole
        # loop, so every recorded call arg is the SAME (by-reference) final
        # list -- inspect its final state via the last recorded call.
        # Shape: [initial user msg, model(call,call), user(resp,resp),
        #         model(call), user(resp)].
        final_contents = fake_client.aio.models.generate_content.await_args.kwargs["contents"]
        assert len(final_contents) == 5
        bundled_response_turn = final_contents[2]
        assert len(bundled_response_turn.parts) == 2


# ---------------------------------------------------------------------------
# Turn-budget exhaustion
# ---------------------------------------------------------------------------


class TestTurnBudgetExhaustion:
    async def test_exhausting_max_turns_still_returns_a_result(self) -> None:
        from argus.runners import _MAX_TURNS

        entry = _make_entry(caching="off")
        settings = _make_settings()

        # Every turn keeps requesting a function call, never finish_review --
        # the loop must stop after _MAX_TURNS turns rather than looping forever.
        responses = [
            _make_response(
                calls=[("report_finding", {"file": None, "line": None, "description": "x"})]
            )
            for _ in range(_MAX_TURNS)
        ]
        fake_client = _make_fake_client(responses)

        with patch("argus.gemini_runner.genai.Client", return_value=fake_client):
            result = await run_session_gemini(
                entry=entry,
                system_prompt="sys",
                user_message="msg",
                settings=settings,
                repo_root="/tmp/does-not-need-to-exist",
            )

        assert result.failure_reason is None
        assert fake_client.aio.models.generate_content.await_count == _MAX_TURNS
        payload = _parse_result_json(result.result_text)
        assert len(payload["findings"]) == _MAX_TURNS


# ---------------------------------------------------------------------------
# Explicit context caching
# ---------------------------------------------------------------------------


class TestCacheActivation:
    async def test_cache_active_sets_cached_content_and_omits_system_instruction_and_tools(
        self,
    ) -> None:
        entry = _make_entry(caching="on")
        settings = _make_settings()

        response = _make_response(calls=[("finish_review", {"files_explored": []})])
        fake_client = _make_fake_client([response], cache_name="cachedContents/abc123")

        with patch("argus.gemini_runner.genai.Client", return_value=fake_client):
            result = await run_session_gemini(
                entry=entry,
                system_prompt="You are a reviewer.",
                user_message="msg",
                settings=settings,
                repo_root="/tmp/does-not-need-to-exist",
            )

        assert result.failure_reason is None
        fake_client.caches.create.assert_called_once()
        create_kwargs = fake_client.caches.create.call_args.kwargs
        assert create_kwargs["model"] == resolve_alias("gemini-frontier")
        create_config = create_kwargs["config"]
        assert create_config.system_instruction == "You are a reviewer."
        assert create_config.tools is not None
        assert create_config.ttl == "3600s"

        # The forced-ANY tool_config directive must be baked into the
        # CACHE ITSELF, not the per-request config -- the live API rejects
        # tool_config alongside cached_content just like system_instruction/
        # tools (see TestCachedRequestNeverCombinesToolConfig below).
        assert create_config.tool_config is not None
        assert create_config.tool_config.function_calling_config.mode == "ANY"

        generate_kwargs = fake_client.aio.models.generate_content.await_args.kwargs
        request_config = generate_kwargs["config"]
        assert request_config.cached_content == "cachedContents/abc123"
        # Gemini rejects duplicating system_instruction/tools/tool_config
        # alongside cached_content -- none of the three may also be set on
        # this request.
        assert request_config.system_instruction is None
        assert request_config.tools is None
        assert request_config.tool_config is None

    async def test_cache_off_never_calls_caches_create(self) -> None:
        entry = _make_entry(caching="off")
        settings = _make_settings()

        response = _make_response(calls=[("finish_review", {"files_explored": []})])
        fake_client = _make_fake_client([response], cache_name="cachedContents/should-not-be-used")

        with patch("argus.gemini_runner.genai.Client", return_value=fake_client):
            await run_session_gemini(
                entry=entry,
                system_prompt="sys",
                user_message="msg",
                settings=settings,
                repo_root="/tmp/does-not-need-to-exist",
            )

        fake_client.caches.create.assert_not_called()
        generate_kwargs = fake_client.aio.models.generate_content.await_args.kwargs
        assert generate_kwargs["config"].system_instruction == "sys"
        assert generate_kwargs["config"].cached_content is None


class TestCacheCreationFailureDegradesGracefully:
    async def test_cache_create_raises_proceeds_uncached_without_crashing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        entry = _make_entry(caching="on")
        settings = _make_settings()

        response = _make_response(calls=[("finish_review", {"files_explored": []})])
        fake_client = _make_fake_client([response])
        fake_client.caches.create = MagicMock(side_effect=RuntimeError("cache too small"))

        with (
            patch("argus.gemini_runner.genai.Client", return_value=fake_client),
            caplog.at_level("WARNING", logger="argus.gemini_runner"),
        ):
            result = await run_session_gemini(
                entry=entry,
                system_prompt="sys",
                user_message="msg",
                settings=settings,
                repo_root="/tmp/does-not-need-to-exist",
            )

        assert result.failure_reason is None
        fake_client.caches.create.assert_called_once()
        generate_kwargs = fake_client.aio.models.generate_content.await_args.kwargs
        # Fell back to an uncached request.
        assert generate_kwargs["config"].cached_content is None
        assert generate_kwargs["config"].system_instruction == "sys"
        assert any("cache" in record.message.lower() for record in caplog.records)


# ---------------------------------------------------------------------------
# Usage accumulation
# ---------------------------------------------------------------------------


class TestUsageAccumulation:
    async def test_usage_summed_across_every_turn_not_just_the_last(self) -> None:
        entry = _make_entry(caching="off")
        settings = _make_settings()

        turn1 = _make_response(
            calls=[("report_finding", {"file": None, "line": None, "description": "x"})],
            usage=_make_usage(prompt=100, candidates=20, cached=30),
        )
        turn2 = _make_response(
            calls=[("finish_review", {"files_explored": []})],
            usage=_make_usage(prompt=40, candidates=10, cached=5),
        )
        fake_client = _make_fake_client([turn1, turn2])

        with (
            patch("argus.gemini_runner.genai.Client", return_value=fake_client),
            patch("argus.gemini_runner.estimate_cost_usd", return_value=1.23) as mock_estimate_cost,
        ):
            result = await run_session_gemini(
                entry=entry,
                system_prompt="sys",
                user_message="msg",
                settings=settings,
                repo_root="/tmp/does-not-need-to-exist",
            )

        assert result.cost_usd == 1.23
        mock_estimate_cost.assert_called_once()
        cost_kwargs = mock_estimate_cost.call_args.kwargs
        # prompt totals: 100+40=140, cached totals: 30+5=35 -> uncached input
        # is 140-35=105 (cached tokens must not double-count against the
        # full input rate).
        assert cost_kwargs["input_tokens"] == 105
        assert cost_kwargs["cached_input_tokens"] == 35
        assert cost_kwargs["output_tokens"] == 30  # 20 + 10

    async def test_missing_usage_metadata_on_some_turns_does_not_crash(self) -> None:
        entry = _make_entry(caching="off")
        settings = _make_settings()

        turn1 = _make_response(
            calls=[("report_finding", {"file": None, "line": None, "description": "x"})],
            usage=None,
        )
        turn2 = _make_response(
            calls=[("finish_review", {"files_explored": []})],
            usage=_make_usage(prompt=10, candidates=5, cached=0),
        )
        fake_client = _make_fake_client([turn1, turn2])

        with patch("argus.gemini_runner.genai.Client", return_value=fake_client):
            result = await run_session_gemini(
                entry=entry,
                system_prompt="sys",
                user_message="msg",
                settings=settings,
                repo_root="/tmp/does-not-need-to-exist",
            )

        assert result.failure_reason is None


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


class TestTimeout:
    async def test_timeout_returns_timed_out_result_instead_of_raising(self) -> None:
        entry = _make_entry(caching="off")
        settings = _make_settings(session_timeout=0.05)

        async def _hang(*args: Any, **kwargs: Any) -> MagicMock:
            await asyncio.sleep(10)
            return _make_response(calls=[])  # pragma: no cover - never reached

        fake_client = MagicMock()
        fake_client.aio.models.generate_content = AsyncMock(side_effect=_hang)
        fake_client.aio.aclose = AsyncMock()

        with patch("argus.gemini_runner.genai.Client", return_value=fake_client):
            result = await run_session_gemini(
                entry=entry,
                system_prompt="sys",
                user_message="msg",
                settings=settings,
                label="timeout-test",
                repo_root="/tmp/does-not-need-to-exist",
            )

        assert result.failure_reason == "timeout"
        assert result.result_text == ""
        assert result.cost_usd == 0.0
        assert result.tool_call_count == 0
        assert isinstance(result.started_at, datetime)
        assert result.started_at.tzinfo is timezone.utc


# ---------------------------------------------------------------------------
# Unknown tool call / malformed args don't crash the session
# ---------------------------------------------------------------------------


class TestMalformedToolCall:
    async def test_unknown_tool_name_reports_error_and_continues(self) -> None:
        entry = _make_entry(caching="off")
        settings = _make_settings()

        turn1 = _make_response(calls=[("not_a_real_tool", {"foo": "bar"})])
        turn2 = _make_response(calls=[("finish_review", {"files_explored": []})])
        fake_client = _make_fake_client([turn1, turn2])

        with patch("argus.gemini_runner.genai.Client", return_value=fake_client):
            result = await run_session_gemini(
                entry=entry,
                system_prompt="sys",
                user_message="msg",
                settings=settings,
                repo_root="/tmp/does-not-need-to-exist",
            )

        assert result.failure_reason is None
        assert "not_a_real_tool" in result.tool_names

        # `contents` is one shared, mutated-in-place list across the whole
        # loop -- inspect its final state via the last recorded call.
        # Shape: [initial user msg, model(not_a_real_tool), user(error resp),
        #         model(finish_review), user(resp)].
        final_contents = fake_client.aio.models.generate_content.await_args.kwargs["contents"]
        assert len(final_contents) == 5
        first_model_turn = final_contents[1]
        assert first_model_turn.parts[0].function_call.name == "not_a_real_tool"


# ---------------------------------------------------------------------------
# finish_review with a missing required argument must NOT terminate the loop
# ---------------------------------------------------------------------------


class TestMalformedFinishReviewDoesNotTerminate:
    async def test_finish_review_missing_required_arg_does_not_end_session(self) -> None:
        entry = _make_entry(caching="off")
        settings = _make_settings()

        # Missing the required `files_explored` argument -- review_tools.
        # finish_review(files_explored) raises TypeError, so this must be
        # treated as a recoverable tool error, not a clean finish.
        turn1 = _make_response(calls=[("finish_review", {})])
        turn2 = _make_response(calls=[("finish_review", {"files_explored": []})])
        fake_client = _make_fake_client([turn1, turn2])

        with patch("argus.gemini_runner.genai.Client", return_value=fake_client):
            result = await run_session_gemini(
                entry=entry,
                system_prompt="sys",
                user_message="msg",
                settings=settings,
                repo_root="/tmp/does-not-need-to-exist",
            )

        assert result.failure_reason is None
        # The loop must NOT have stopped after turn 1's malformed call --
        # turn 2's well-formed finish_review is what actually ends it.
        assert fake_client.aio.models.generate_content.await_count == 2
        assert result.tool_call_count == 2
        assert result.tool_names == ["finish_review"]


# ---------------------------------------------------------------------------
# Thought signatures (Gemini 3.x) must be preserved verbatim across turns
# ---------------------------------------------------------------------------


class TestThoughtSignaturePreservation:
    async def test_model_turn_appended_verbatim_preserves_thought_signature(self) -> None:
        entry = _make_entry(caching="off")
        settings = _make_settings()

        turn1 = _make_response(
            calls=[("report_finding", {"file": "a.py", "line": 1, "description": "d"})],
            usage=_make_usage(10, 5),
            thought_signatures={"report_finding": b"opaque-signature-bytes"},
        )
        turn2 = _make_response(calls=[("finish_review", {"files_explored": []})])
        fake_client = _make_fake_client([turn1, turn2])

        with patch("argus.gemini_runner.genai.Client", return_value=fake_client):
            result = await run_session_gemini(
                entry=entry,
                system_prompt="sys",
                user_message="msg",
                settings=settings,
                repo_root="/tmp/does-not-need-to-exist",
            )

        assert result.failure_reason is None
        final_contents = fake_client.aio.models.generate_content.await_args.kwargs["contents"]
        # The exact Content object the API returned for turn 1's model turn
        # (final_contents[1]) must be what got appended -- not a hand-
        # rebuilt Content -- so its thought_signature survives verbatim.
        model_turn = final_contents[1]
        assert model_turn is turn1.candidates[0].content
        assert model_turn.parts[0].thought_signature == b"opaque-signature-bytes"


# ---------------------------------------------------------------------------
# Forced function-calling mode (ANY) -- Gemini must not be free to answer
# with plain text instead of a tool call
# ---------------------------------------------------------------------------


class TestForcedFunctionCallingMode:
    async def test_generate_content_config_forces_any_mode(self) -> None:
        entry = _make_entry(caching="off")
        settings = _make_settings()
        response = _make_response(calls=[("finish_review", {"files_explored": []})])
        fake_client = _make_fake_client([response])

        with patch("argus.gemini_runner.genai.Client", return_value=fake_client):
            await run_session_gemini(
                entry=entry,
                system_prompt="sys",
                user_message="msg",
                settings=settings,
                repo_root="/tmp/does-not-need-to-exist",
            )

        generate_kwargs = fake_client.aio.models.generate_content.await_args.kwargs
        tool_config = generate_kwargs["config"].tool_config
        assert tool_config is not None
        assert tool_config.function_calling_config.mode == "ANY"

    async def test_cached_session_forces_any_mode_via_the_cache_itself(self) -> None:
        """A cached request cannot carry `tool_config` directly (the live
        API rejects it alongside `cached_content=`, just like
        `system_instruction`/`tools`) -- so the forced-ANY directive must
        instead be baked into the cache at creation time, via
        `CreateCachedContentConfig.tool_config`."""
        entry = _make_entry(caching="on")
        settings = _make_settings()
        response = _make_response(calls=[("finish_review", {"files_explored": []})])
        fake_client = _make_fake_client([response], cache_name="cachedContents/abc")

        with patch("argus.gemini_runner.genai.Client", return_value=fake_client):
            await run_session_gemini(
                entry=entry,
                system_prompt="sys",
                user_message="msg",
                settings=settings,
                repo_root="/tmp/does-not-need-to-exist",
            )

        create_kwargs = fake_client.caches.create.call_args.kwargs
        create_tool_config = create_kwargs["config"].tool_config
        assert create_tool_config is not None
        assert create_tool_config.function_calling_config.mode == "ANY"

        # The per-request config must NOT also carry tool_config.
        generate_kwargs = fake_client.aio.models.generate_content.await_args.kwargs
        assert generate_kwargs["config"].tool_config is None


class TestToolConfigNeverCombinedWithCachedContent:
    """Regression test for a real (not mocked) 400 INVALID_ARGUMENT the live
    Gemini API returned: 'CachedContent can not be used with GenerateContent
    request setting system_instruction, tools or tool_config.' An earlier
    version of this module treated `tool_config` as safe to combine with
    `cached_content=` (unlike `system_instruction`/`tools`) -- it is not.
    Mirrors the equivalent `system_instruction`/`tools` assertions in
    `TestCacheActivation`."""

    async def test_cached_request_never_carries_tool_config(self) -> None:
        entry = _make_entry(caching="on")
        settings = _make_settings()
        response = _make_response(calls=[("finish_review", {"files_explored": []})])
        fake_client = _make_fake_client([response], cache_name="cachedContents/xyz")

        with patch("argus.gemini_runner.genai.Client", return_value=fake_client):
            result = await run_session_gemini(
                entry=entry,
                system_prompt="sys",
                user_message="msg",
                settings=settings,
                repo_root="/tmp/does-not-need-to-exist",
            )

        assert result.failure_reason is None
        generate_kwargs = fake_client.aio.models.generate_content.await_args.kwargs
        request_config = generate_kwargs["config"]
        assert request_config.cached_content == "cachedContents/xyz"
        assert request_config.system_instruction is None
        assert request_config.tools is None
        assert request_config.tool_config is None

    async def test_uncached_retry_after_cache_invalidation_restores_tool_config(self) -> None:
        """When a stale cache is invalidated mid-session and the runner
        falls back to an uncached retry, that retry must fully restore
        `tool_config` (not just `system_instruction`/`tools`) -- otherwise
        the forced-ANY directive silently disappears for the rest of the
        session the moment a cache goes stale upstream."""
        entry = _make_entry(caching="on")
        settings = _make_settings()

        cache_invalid_error = genai_errors.ClientError(
            404, {"message": "Cache cachedContents/xyz not found", "status": "NOT_FOUND"}
        )
        success_response = _make_response(calls=[("finish_review", {"files_explored": []})])

        fake_client = _make_fake_client([], cache_name="cachedContents/xyz")
        fake_client.aio.models.generate_content = AsyncMock(
            side_effect=[cache_invalid_error, success_response]
        )

        with patch("argus.gemini_runner.genai.Client", return_value=fake_client):
            result = await run_session_gemini(
                entry=entry,
                system_prompt="sys",
                user_message="msg",
                settings=settings,
                repo_root="/tmp/does-not-need-to-exist",
            )

        assert result.failure_reason is None
        retry_kwargs = fake_client.aio.models.generate_content.await_args.kwargs
        retry_config = retry_kwargs["config"]
        assert retry_config.cached_content is None
        assert retry_config.system_instruction == "sys"
        assert retry_config.tools is not None
        assert retry_config.tool_config is not None
        assert retry_config.tool_config.function_calling_config.mode == "ANY"


# ---------------------------------------------------------------------------
# Findings returned as fenced-JSON text instead of a report_finding call
# ---------------------------------------------------------------------------


class TestTextFallbackFindingsRecovery:
    async def test_response_text_fenced_json_findings_recovered_when_no_function_calls(
        self,
    ) -> None:
        entry = _make_entry(caching="off")
        settings = _make_settings()

        text_payload = json.dumps(
            {
                "findings": [
                    {"file": "x.py", "line": 10, "description": "leaked secret", "context": None}
                ],
                "files_explored": ["x.py"],
            }
        )
        response = _make_response(
            calls=[], usage=_make_usage(50, 20), text=f"```json\n{text_payload}\n```"
        )
        fake_client = _make_fake_client([response])

        with patch("argus.gemini_runner.genai.Client", return_value=fake_client):
            result = await run_session_gemini(
                entry=entry,
                system_prompt="sys",
                user_message="msg",
                settings=settings,
                repo_root="/tmp/does-not-need-to-exist",
            )

        assert result.failure_reason is None
        payload = _parse_result_json(result.result_text)
        assert payload["files_explored"] == ["x.py"]
        assert len(payload["findings"]) == 1
        assert payload["findings"][0]["description"] == "leaked secret"

    async def test_plain_prose_text_with_no_function_calls_yields_no_findings(self) -> None:
        """Regression guard: freeform, non-JSON commentary (e.g. "Nothing to
        report.") must NOT be treated as a finding -- this only mirrors the
        fenced-JSON shape `parse_review_result` uses for the Claude path,
        not its raw-text fallback."""
        entry = _make_entry(caching="off")
        settings = _make_settings()

        response = _make_response(calls=[], usage=_make_usage(10, 5), text="Nothing to report.")
        fake_client = _make_fake_client([response])

        with patch("argus.gemini_runner.genai.Client", return_value=fake_client):
            result = await run_session_gemini(
                entry=entry,
                system_prompt="sys",
                user_message="msg",
                settings=settings,
                repo_root="/tmp/does-not-need-to-exist",
            )

        payload = _parse_result_json(result.result_text)
        assert payload == {"findings": [], "files_explored": []}


# ---------------------------------------------------------------------------
# Cost: thoughts_token_count / tool_use_prompt_token_count accumulation
# ---------------------------------------------------------------------------


class TestThinkingAndToolUsePromptTokenAccumulation:
    async def test_thoughts_and_tool_use_prompt_tokens_are_accumulated_and_priced(self) -> None:
        entry = _make_entry(caching="off")
        settings = _make_settings()

        turn1 = _make_response(
            calls=[("report_finding", {"file": None, "line": None, "description": "x"})],
            usage=_make_usage(prompt=100, candidates=20, thoughts=15, tool_use_prompt=8),
        )
        turn2 = _make_response(
            calls=[("finish_review", {"files_explored": []})],
            usage=_make_usage(prompt=40, candidates=10, thoughts=5, tool_use_prompt=2),
        )
        fake_client = _make_fake_client([turn1, turn2])

        with (
            patch("argus.gemini_runner.genai.Client", return_value=fake_client),
            patch("argus.gemini_runner.estimate_cost_usd", return_value=9.99) as mock_estimate_cost,
        ):
            result = await run_session_gemini(
                entry=entry,
                system_prompt="sys",
                user_message="msg",
                settings=settings,
                repo_root="/tmp/does-not-need-to-exist",
            )

        assert result.cost_usd == 9.99
        cost_kwargs = mock_estimate_cost.call_args.kwargs
        # input: prompt totals (140) - cached (0) + tool_use_prompt totals (10)
        assert cost_kwargs["input_tokens"] == 150
        # output: candidates totals (30) + thoughts totals (20)
        assert cost_kwargs["output_tokens"] == 50
        assert cost_kwargs["cached_input_tokens"] == 0


# ---------------------------------------------------------------------------
# HTTP-level (httpx) timeout raised directly by the SDK call
# ---------------------------------------------------------------------------


class TestHttpLevelTimeout:
    async def test_httpx_timeout_exception_from_sdk_call_returns_timed_out(self) -> None:
        entry = _make_entry(caching="off")
        settings = _make_settings(session_timeout=30)

        fake_client = MagicMock()
        fake_client.aio.models.generate_content = AsyncMock(
            side_effect=httpx.ReadTimeout("upstream read timed out")
        )
        fake_client.aio.aclose = AsyncMock()

        with patch("argus.gemini_runner.genai.Client", return_value=fake_client):
            result = await run_session_gemini(
                entry=entry,
                system_prompt="sys",
                user_message="msg",
                settings=settings,
                label="httpx-timeout-test",
                repo_root="/tmp/does-not-need-to-exist",
            )

        assert result.failure_reason == "timeout"
        assert result.result_text == ""
        assert result.cost_usd == 0.0


# ---------------------------------------------------------------------------
# Non-timeout exceptions must produce a visible failure, not vanish
# ---------------------------------------------------------------------------


class TestNonTimeoutExceptionHandling:
    """Regression coverage for a live-dogfood finding: before this fix,
    ``run_session_gemini`` only caught timeout-shaped exceptions
    (``TimeoutError``/``httpx.TimeoutException``); any other exception the
    underlying ``google-genai`` SDK could realistically raise (a genuine API
    error, a network/transport failure) propagated uncaught, and
    ``argus.graph``'s blanket per-reviewer exception handling silently
    turned it into an empty ``{"findings": [], "agent_runs": []}`` result --
    indistinguishable from "this reviewer ran fine and found nothing."

    Each test here simulates a specific non-timeout exception type the SDK
    can realistically raise and asserts it now produces a *visible*
    ``SessionResult`` (``failure_reason="worker_crashed"``, no findings,
    no raised exception) instead of propagating or vanishing silently.
    """

    async def test_genai_server_error_produces_visible_failure_not_silent_result(self) -> None:
        """A 5xx from the Gemini API itself (google.genai.errors.ServerError,
        APIError's other concrete subclass alongside ClientError)."""
        entry = _make_entry(caching="off")
        settings = _make_settings()

        server_error = genai_errors.ServerError(
            503, {"message": "The model is overloaded", "status": "UNAVAILABLE"}
        )
        fake_client = MagicMock()
        fake_client.aio.models.generate_content = AsyncMock(side_effect=server_error)
        fake_client.aio.aclose = AsyncMock()

        with patch("argus.gemini_runner.genai.Client", return_value=fake_client):
            result = await run_session_gemini(
                entry=entry,
                system_prompt="sys",
                user_message="msg",
                settings=settings,
                label="server-error-test",
                repo_root="/tmp/does-not-need-to-exist",
            )

        assert result.failure_reason == "worker_crashed"
        assert result.result_text == ""
        assert result.cost_usd == 0.0

    async def test_genai_client_error_produces_visible_failure_not_silent_result(self) -> None:
        """A 4xx (auth rejection, bad request, quota) with no cache in play
        at all -- the most direct real-world "genuine API error" case."""
        entry = _make_entry(caching="off")
        settings = _make_settings()

        client_error = genai_errors.ClientError(
            401, {"message": "API key not valid", "status": "UNAUTHENTICATED"}
        )
        fake_client = MagicMock()
        fake_client.aio.models.generate_content = AsyncMock(side_effect=client_error)
        fake_client.aio.aclose = AsyncMock()

        with patch("argus.gemini_runner.genai.Client", return_value=fake_client):
            result = await run_session_gemini(
                entry=entry,
                system_prompt="sys",
                user_message="msg",
                settings=settings,
                label="client-error-test",
                repo_root="/tmp/does-not-need-to-exist",
            )

        assert result.failure_reason == "worker_crashed"
        assert result.result_text == ""

    async def test_transport_level_connect_error_produces_visible_failure(self) -> None:
        """A non-timeout httpx transport failure (connection reset, DNS
        failure, etc.) -- httpx.ConnectError is a subclass of
        httpx.HTTPError, not of httpx.TimeoutException, so this exercises a
        genuinely different exception path than TestHttpLevelTimeout above."""
        entry = _make_entry(caching="off")
        settings = _make_settings()

        fake_client = MagicMock()
        fake_client.aio.models.generate_content = AsyncMock(
            side_effect=httpx.ConnectError("connection refused")
        )
        fake_client.aio.aclose = AsyncMock()

        with patch("argus.gemini_runner.genai.Client", return_value=fake_client):
            result = await run_session_gemini(
                entry=entry,
                system_prompt="sys",
                user_message="msg",
                settings=settings,
                label="connect-error-test",
                repo_root="/tmp/does-not-need-to-exist",
            )

        assert result.failure_reason == "worker_crashed"
        assert result.result_text == ""

    async def test_genuine_bug_still_propagates_rather_than_being_mislabeled(self) -> None:
        """A real programming-error-shaped exception (not an SDK/transport
        failure) must NOT be caught by this narrower except clause -- it
        should still propagate, so a real bug in this module's own code
        isn't silently relabeled as an ordinary session failure. Confirms
        the fix is deliberately narrower than a bare ``except Exception``."""
        entry = _make_entry(caching="off")
        settings = _make_settings()

        fake_client = MagicMock()
        fake_client.aio.models.generate_content = AsyncMock(
            side_effect=AttributeError("boom: not an SDK/transport failure")
        )
        fake_client.aio.aclose = AsyncMock()

        with patch("argus.gemini_runner.genai.Client", return_value=fake_client):
            with pytest.raises(AttributeError, match="boom"):
                await run_session_gemini(
                    entry=entry,
                    system_prompt="sys",
                    user_message="msg",
                    settings=settings,
                    label="genuine-bug-test",
                    repo_root="/tmp/does-not-need-to-exist",
                )


# ---------------------------------------------------------------------------
# Explicit timeout_s override wins over settings.ARGUS_SESSION_TIMEOUT
# ---------------------------------------------------------------------------


class TestTimeoutSOverride:
    async def test_explicit_timeout_s_overrides_settings_default(self) -> None:
        entry = _make_entry(caching="off")
        # Settings says a generous 300s timeout, but an explicit timeout_s
        # override must win -- prove it by making the call hang far longer
        # than the tiny explicit override and confirming we still time out
        # quickly rather than waiting out the settings-based default.
        settings = _make_settings(session_timeout=300)

        async def _hang(*args: Any, **kwargs: Any) -> MagicMock:
            await asyncio.sleep(10)
            return _make_response(calls=[])  # pragma: no cover - never reached

        fake_client = MagicMock()
        fake_client.aio.models.generate_content = AsyncMock(side_effect=_hang)
        fake_client.aio.aclose = AsyncMock()

        with patch("argus.gemini_runner.genai.Client", return_value=fake_client):
            result = await run_session_gemini(
                entry=entry,
                system_prompt="sys",
                user_message="msg",
                settings=settings,
                repo_root="/tmp/does-not-need-to-exist",
                timeout_s=0.05,
            )

        assert result.failure_reason == "timeout"


# ---------------------------------------------------------------------------
# SDK clients (sync + async) are closed on every exit path
# ---------------------------------------------------------------------------


class TestClientCleanup:
    async def test_both_clients_closed_on_normal_completion(self) -> None:
        entry = _make_entry(caching="off")
        settings = _make_settings()
        response = _make_response(calls=[("finish_review", {"files_explored": []})])
        fake_client = _make_fake_client([response])

        with patch("argus.gemini_runner.genai.Client", return_value=fake_client):
            await run_session_gemini(
                entry=entry,
                system_prompt="sys",
                user_message="msg",
                settings=settings,
                repo_root="/tmp/does-not-need-to-exist",
            )

        fake_client.close.assert_called_once()
        fake_client.aio.aclose.assert_awaited_once()

    async def test_both_clients_closed_on_timeout(self) -> None:
        entry = _make_entry(caching="off")
        settings = _make_settings(session_timeout=0.05)

        async def _hang(*args: Any, **kwargs: Any) -> MagicMock:
            await asyncio.sleep(10)
            return _make_response(calls=[])  # pragma: no cover - never reached

        fake_client = MagicMock()
        fake_client.aio.models.generate_content = AsyncMock(side_effect=_hang)
        fake_client.aio.aclose = AsyncMock()

        with patch("argus.gemini_runner.genai.Client", return_value=fake_client):
            await run_session_gemini(
                entry=entry,
                system_prompt="sys",
                user_message="msg",
                settings=settings,
                repo_root="/tmp/does-not-need-to-exist",
            )

        fake_client.close.assert_called_once()
        fake_client.aio.aclose.assert_awaited_once()


# ---------------------------------------------------------------------------
# Cache activation runs off the event loop (asyncio.to_thread)
# ---------------------------------------------------------------------------


class TestCacheActivationDoesNotBlockEventLoop:
    async def test_slow_cache_create_does_not_stall_a_concurrent_coroutine(self) -> None:
        entry = _make_entry(caching="on")
        settings = _make_settings()

        def _slow_create(*args: Any, **kwargs: Any) -> MagicMock:
            time.sleep(0.3)  # simulate a slow, blocking upstream call
            cache = MagicMock()
            cache.name = "cachedContents/slow"
            return cache

        response = _make_response(calls=[("finish_review", {"files_explored": []})])
        fake_client = _make_fake_client([response])
        fake_client.caches.create = MagicMock(side_effect=_slow_create)

        concurrent_completed_at: list[float] = []

        async def _concurrent_ping() -> None:
            await asyncio.sleep(0.02)
            concurrent_completed_at.append(time.monotonic())

        with patch("argus.gemini_runner.genai.Client", return_value=fake_client):
            start = time.monotonic()
            await asyncio.gather(
                run_session_gemini(
                    entry=entry,
                    system_prompt="sys",
                    user_message="msg",
                    settings=settings,
                    repo_root="/tmp/does-not-need-to-exist",
                ),
                _concurrent_ping(),
            )

        # The concurrent 20ms sleep must complete well before the 300ms
        # blocking cache-create call does -- proving cache creation ran off
        # the event loop (in a worker thread via asyncio.to_thread), not
        # blocking it.
        assert concurrent_completed_at
        assert concurrent_completed_at[0] - start < 0.15


class TestCacheActivationClientIsolationUnderTimeout:
    """Regression test: a session timeout/cancellation firing WHILE cache
    creation is still in flight must never race the session's own
    `client`/`client.aio` being closed in `_run_turns`' `finally` block.

    `asyncio.to_thread` cancellation doesn't stop the underlying OS thread
    -- a detached cache-creation call can still be executing after the
    coroutine awaiting it has already been cancelled and unwound through
    that `finally`. The fix is for cache creation to use its own,
    separate, short-lived `genai.Client` instead of the session's -- this
    test proves that separation by checking exactly two distinct clients
    get constructed, that the session's own client is never the one used
    for `caches.create()`, and that the cache-creation client is only
    ever closed AFTER its own `caches.create()` call actually completes
    (never torn down out from under an in-flight call).
    """

    async def test_timeout_during_in_flight_cache_creation_does_not_race_a_shared_client(
        self,
    ) -> None:
        entry = _make_entry(caching="on")
        settings = _make_settings(session_timeout=0.05)

        call_order: list[str] = []
        constructed_clients: list[MagicMock] = []

        def _new_client(*args: Any, **kwargs: Any) -> MagicMock:
            index = len(constructed_clients)
            client = MagicMock()
            constructed_clients.append(client)

            # Never actually reached in this test (the session times out
            # while still awaiting cache activation, before any
            # generate_content call), but wired up defensively so a future
            # change to call ordering doesn't hang the test on a bare
            # (non-awaitable) MagicMock.
            client.aio.models.generate_content = AsyncMock(
                side_effect=lambda *a, **k: asyncio.sleep(10)
            )
            client.aio.aclose = AsyncMock(
                side_effect=lambda i=index: call_order.append(f"client[{i}].aio.aclose")
            )
            client.close = MagicMock(
                side_effect=lambda i=index: call_order.append(f"client[{i}].close")
            )

            def _slow_create(*a: Any, i: int = index, **k: Any) -> MagicMock:
                call_order.append(f"client[{i}].caches.create:start")
                time.sleep(0.3)  # still "in flight" well past the session timeout
                call_order.append(f"client[{i}].caches.create:end")
                cache = MagicMock()
                cache.name = f"cachedContents/from-client-{i}"
                return cache

            client.caches.create = MagicMock(side_effect=_slow_create)
            return client

        with patch("argus.gemini_runner.genai.Client", side_effect=_new_client):
            result = await run_session_gemini(
                entry=entry,
                system_prompt="sys",
                user_message="msg",
                settings=settings,
                label="race-test",
                repo_root="/tmp/does-not-need-to-exist",
            )

        assert result.failure_reason == "timeout"

        # Two SEPARATE `genai.Client` instances must have been constructed:
        # one for the session itself (`_run_turns`'s own `client`), and one
        # dedicated to the cache-creation call inside `_maybe_activate_cache`'s
        # `_create_fn`. Under the old (buggy) implementation that shared the
        # session's client with the detached worker, only ONE client would
        # ever be constructed here, and this unpacking would raise.
        assert len(constructed_clients) == 2
        session_client, cache_client = constructed_clients
        assert session_client is not cache_client

        # The cache-creation call must genuinely still be in flight at the
        # moment the session times out -- otherwise this test wouldn't be
        # exercising the race at all.
        assert "client[1].caches.create:start" in call_order
        assert "client[1].caches.create:end" not in call_order

        # The session's OWN client was closed promptly on the timeout, and
        # it was never used for `caches.create()` -- so closing it can
        # never race the (separate) client the in-flight call is using.
        assert "client[0].close" in call_order
        assert "client[0].aio.aclose" in call_order
        assert not any(marker.startswith("client[0].caches.create") for marker in call_order)

        # Let the detached background thread actually finish before this
        # test's event loop tears down.
        await asyncio.sleep(0.5)

        # The cache-creation client is only ever closed AFTER its own
        # `caches.create()` call completes -- i.e. no client was ever used
        # after being closed.
        assert call_order.index("client[1].caches.create:end") < call_order.index("client[1].close")


# ---------------------------------------------------------------------------
# A cache invalidated upstream (but still recorded locally as unexpired)
# degrades gracefully: invalidate the local record, retry once uncached
# ---------------------------------------------------------------------------


class TestCacheInvalidatedUpstreamDegradesGracefully:
    async def test_cache_not_found_error_invalidates_local_record_and_retries_uncached(
        self,
    ) -> None:
        entry = _make_entry(caching="on")
        settings = _make_settings()

        cache_invalid_error = genai_errors.ClientError(
            404, {"message": "Cache cachedContents/abc123 not found", "status": "NOT_FOUND"}
        )
        success_response = _make_response(calls=[("finish_review", {"files_explored": []})])

        fake_client = _make_fake_client([], cache_name="cachedContents/abc123")
        fake_client.aio.models.generate_content = AsyncMock(
            side_effect=[cache_invalid_error, success_response]
        )

        with patch("argus.gemini_runner.genai.Client", return_value=fake_client):
            result = await run_session_gemini(
                entry=entry,
                system_prompt="sys",
                user_message="msg",
                settings=settings,
                repo_root="/tmp/does-not-need-to-exist",
            )

        assert result.failure_reason is None
        assert fake_client.aio.models.generate_content.await_count == 2
        # The retry must have fallen back to an uncached request -- and
        # that uncached request must fully restore system_instruction/
        # tools/tool_config (see TestToolConfigNeverCombinedWithCachedContent
        # for a dedicated test of the tool_config restoration).
        retry_kwargs = fake_client.aio.models.generate_content.await_args.kwargs
        assert retry_kwargs["config"].cached_content is None
        assert retry_kwargs["config"].system_instruction == "sys"
        assert retry_kwargs["config"].tools is not None
        assert retry_kwargs["config"].tool_config is not None

    async def test_unrelated_client_error_is_not_treated_as_cache_invalid(self) -> None:
        """A 4xx that has nothing to do with the cache (bad request, auth,
        quota) must not be silently retried as if the cache were invalid --
        that would mask the real problem by re-sending the exact same
        request uncached and hoping it succeeds for an unrelated reason.

        It must still surface as a visible, non-silent failure overall
        (a ``SessionResult`` with ``failure_reason="worker_crashed"``, per
        ``run_session_gemini``'s broader non-timeout except clause) rather
        than being swallowed into a clean 0-finding result -- see
        TestNonTimeoutExceptionHandling below for that behavior's own
        dedicated coverage. It no longer propagates all the way out of
        ``run_session_gemini`` uncaught (that was the exact round-4 Argus
        finding this broader except clause exists to fix)."""
        entry = _make_entry(caching="on")
        settings = _make_settings()

        unrelated_error = genai_errors.ClientError(
            400, {"message": "Invalid argument: bad request", "status": "INVALID_ARGUMENT"}
        )
        fake_client = _make_fake_client([], cache_name="cachedContents/abc123")
        fake_client.aio.models.generate_content = AsyncMock(side_effect=unrelated_error)

        with patch("argus.gemini_runner.genai.Client", return_value=fake_client):
            result = await run_session_gemini(
                entry=entry,
                system_prompt="sys",
                user_message="msg",
                settings=settings,
                repo_root="/tmp/does-not-need-to-exist",
            )

        # Not retried uncached (that would mean 2 awaits, as in the
        # cache-invalid-recovery test above) -- the error propagated
        # straight out of the cache-retry logic on the first attempt.
        assert fake_client.aio.models.generate_content.await_count == 1
        assert result.failure_reason == "worker_crashed"
        assert result.result_text == ""


class TestGeminiRedactionHooks:
    async def test_redact_gemini_inputs_redacts_settings(self) -> None:
        from argus.gemini_runner import _redact_gemini_inputs

        settings = _make_settings()
        inputs = {"settings": settings, "label": "test", "other": 123}
        redacted = _redact_gemini_inputs(inputs)
        assert redacted["label"] == "test"
        assert redacted["other"] == 123
        assert "Settings" in redacted["settings"]
        assert "secret" not in redacted["settings"]

    async def test_redact_gemini_inputs_missing_settings(self) -> None:
        from argus.gemini_runner import _redact_gemini_inputs

        inputs = {"label": "test"}
        redacted = _redact_gemini_inputs(inputs)
        assert redacted == {"label": "test"}
