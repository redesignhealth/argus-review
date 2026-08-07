"""Tests for run_tests_and_docs_reviewer and _run_session_in_subprocess.

Covers:
- run_tests_and_docs_reviewer: prompt fetch, session call, result parsing
- _run_session_in_subprocess: timeout, normal completion, worker exception
- run_blocking_validator: happy path and timeout flow-through
- _run_session_isolated: dedicated executor routing
"""

from __future__ import annotations

import logging
import signal
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from argus.pipeline_models import (
    FeedbackVerificationResult,
    FileEntry,
    FindingValidationResult,
    PriorFinding,
    PriorReviewContext,
    ReviewPlan,
    SystemGroup,
    SystemReviewResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RUNNERS_MODULE = "argus.runners"


def _make_session_result(result_text: str = "", cost_usd: float = 0.0) -> MagicMock:
    """Build a mock SessionResult matching argus.runners.SessionResult."""
    sr = MagicMock()
    sr.result_text = result_text
    sr.cost_usd = cost_usd
    sr.duration_seconds = 1.0
    sr.started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    sr.finished_at = datetime(2026, 1, 1, 0, 5, 0, tzinfo=timezone.utc)
    sr.tool_call_count = 0
    sr.tool_names = []
    sr.context7_call_count = 0
    sr.model = "claude-sonnet-4-6"
    sr.failure_reason = None
    return sr


_VALID_REVIEW_JSON = (
    '{"system_group": "tests-and-docs", '
    '"findings": [{"file": "src/app.py", "line": 10, "description": "Missing test"}], '
    '"files_explored": ["src/app.py", "tests/test_app.py"]}'
)


def _make_plan() -> ReviewPlan:
    return ReviewPlan(
        system_groups=[
            SystemGroup(
                name="backend",
                files=["src/app.py"],
                conventions="",
                review_focus="",
            ),
        ],
        cross_cutting_concerns=["test coverage"],
        file_manifest=[
            FileEntry(path="src/app.py", change_type="modified"),
            FileEntry(path="tests/test_app.py", change_type="added"),
        ],
    )


# ---------------------------------------------------------------------------
# run_tests_and_docs_reviewer
# ---------------------------------------------------------------------------


class TestRunTestsAndDocsReviewer:
    @pytest.mark.asyncio
    async def test_returns_system_review_result_with_correct_group(self) -> None:
        """Mocks fetch_prompt and _run_session_in_subprocess, verifies result shape."""
        mock_settings = MagicMock()
        mock_settings.CONTEXT7_API_KEY = None

        with (
            patch(
                f"{_RUNNERS_MODULE}.fetch_prompt",
                new_callable=AsyncMock,
                return_value="You are a test reviewer.",
            ) as mock_fetch,
            patch(
                f"{_RUNNERS_MODULE}._run_session_in_subprocess",
                return_value=_make_session_result(_VALID_REVIEW_JSON, 0.05),
            ) as mock_session,
        ):
            from argus.runners import run_tests_and_docs_reviewer

            result_tuple = await run_tests_and_docs_reviewer(
                plan=_make_plan(),
                diff_text="diff --git a/src/app.py ...",
                settings=mock_settings,
            )

        result, agent_run = result_tuple
        assert isinstance(result, SystemReviewResult)
        assert result.system_group == "tests-and-docs"
        assert len(result.findings) == 1
        assert result.findings[0].file == "src/app.py"
        assert result.cost_usd == 0.05

        # Verify _build_agent_run populated AgentRunData correctly
        assert agent_run is not None
        assert agent_run.agent_name == "tests-and-docs"
        assert agent_run.agent_type == "tests_and_docs"
        assert agent_run.finding_count == 1
        assert agent_run.files_explored == ["src/app.py", "tests/test_app.py"]
        assert agent_run.model == "claude-sonnet-4-6"

        mock_fetch.assert_awaited_once_with("pr-review-tests-and-docs")
        mock_session.assert_called_once()
        call_kwargs = mock_session.call_args.kwargs
        assert call_kwargs["label"] == "tests-and-docs"
        # is_system_reviewer_role must reach _run_session_in_subprocess unchanged --
        # a silent drop at this call site would withhold the 1M-context beta from
        # this reviewer role with no other test catching it.
        assert call_kwargs["is_system_reviewer_role"] is True

    @pytest.mark.asyncio
    async def test_timed_out_session_marks_result_and_agent_run(self) -> None:
        """A SessionResult with failure_reason="timeout" flows through
        _parse_review_result's output into SystemReviewResult.failure_reason and
        AgentRunData.failure_reason, so a killed reviewer is distinguishable from
        one that genuinely found nothing.
        """
        mock_settings = MagicMock()
        mock_settings.CONTEXT7_API_KEY = None
        mock_settings.ARGUS_SESSION_TIMEOUT = 300
        timeout_session = _make_session_result("", cost_usd=0.0)
        timeout_session.failure_reason = "timeout"

        with (
            patch(
                f"{_RUNNERS_MODULE}.fetch_prompt",
                new_callable=AsyncMock,
                return_value="You are a test reviewer.",
            ),
            patch(
                f"{_RUNNERS_MODULE}._run_session_in_subprocess",
                return_value=timeout_session,
            ) as mock_session,
        ):
            from argus.runners import run_tests_and_docs_reviewer

            result, agent_run = await run_tests_and_docs_reviewer(
                plan=_make_plan(),
                diff_text="diff --git a/src/app.py ...",
                settings=mock_settings,
            )

        assert result.failure_reason == "timeout"
        assert result.findings == []
        assert agent_run is not None
        assert agent_run.failure_reason == "timeout"
        call_kwargs = mock_session.call_args.kwargs
        assert call_kwargs["timeout_s"] == 300

    @pytest.mark.asyncio
    async def test_user_message_includes_large_file_read_directive(self) -> None:
        """TECH-4643: reviewers that trace across files need explicit
        Grep-first/scoped-Read guidance for large targets, or a full Read of
        a multi-thousand-line file can exhaust enough context budget in one
        tool call to autocompact-thrash for the rest of the session --
        silently, since the session still exits normally rather than timing
        out."""
        mock_settings = MagicMock()
        mock_settings.CONTEXT7_API_KEY = None

        with (
            patch(
                f"{_RUNNERS_MODULE}.fetch_prompt",
                new_callable=AsyncMock,
                return_value="You are a test reviewer.",
            ),
            patch(
                f"{_RUNNERS_MODULE}._run_session_in_subprocess",
                return_value=_make_session_result(_VALID_REVIEW_JSON, 0.05),
            ) as mock_session,
        ):
            from argus.runners import run_tests_and_docs_reviewer

            await run_tests_and_docs_reviewer(
                plan=_make_plan(),
                diff_text="diff --git a/src/app.py ...",
                settings=mock_settings,
            )

        call_kwargs = mock_session.call_args.kwargs
        assert "Reading Large Files" in call_kwargs["user_message"]

    @pytest.mark.asyncio
    async def test_empty_result_text_returns_empty_findings(self) -> None:
        """When the agent returns empty text, findings should be empty."""
        mock_settings = MagicMock()
        mock_settings.CONTEXT7_API_KEY = None

        with (
            patch(
                f"{_RUNNERS_MODULE}.fetch_prompt",
                new_callable=AsyncMock,
                return_value="You are a test reviewer.",
            ),
            patch(
                f"{_RUNNERS_MODULE}._run_session_in_subprocess",
                return_value=_make_session_result("", 0.0),
            ),
        ):
            from argus.runners import run_tests_and_docs_reviewer

            result_tuple = await run_tests_and_docs_reviewer(
                plan=_make_plan(),
                diff_text="some diff",
                settings=mock_settings,
            )

        result, _agent_run = result_tuple
        assert isinstance(result, SystemReviewResult)
        assert result.system_group == "tests-and-docs"
        assert result.findings == []

    @pytest.mark.asyncio
    async def test_uses_default_settings_when_none(self) -> None:
        """When settings=None, get_settings() is called."""
        with (
            patch(
                f"{_RUNNERS_MODULE}.get_settings",
                return_value=MagicMock(CONTEXT7_API_KEY=None),
            ) as mock_get_settings,
            patch(
                f"{_RUNNERS_MODULE}.fetch_prompt",
                new_callable=AsyncMock,
                return_value="prompt",
            ),
            patch(
                f"{_RUNNERS_MODULE}._run_session_in_subprocess",
                return_value=_make_session_result("", 0.0),
            ),
        ):
            from argus.runners import run_tests_and_docs_reviewer

            await run_tests_and_docs_reviewer(
                plan=_make_plan(),
                diff_text="diff",
                settings=None,
            )

        mock_get_settings.assert_called_once()

    @pytest.mark.asyncio
    async def test_forwards_repo_root_as_cwd_to_subprocess(self) -> None:
        """The provided repo_root is forwarded to _run_session_in_subprocess as cwd."""
        mock_settings = MagicMock(CONTEXT7_API_KEY=None)
        with (
            patch(f"{_RUNNERS_MODULE}.fetch_prompt", new_callable=AsyncMock, return_value="p"),
            patch(
                f"{_RUNNERS_MODULE}._run_session_in_subprocess",
                return_value=_make_session_result("", 0.0),
            ) as mock_session,
        ):
            from argus.runners import run_tests_and_docs_reviewer

            await run_tests_and_docs_reviewer(
                plan=_make_plan(),
                diff_text="diff",
                settings=mock_settings,
                repo_root="/wt/abc123",
            )

        assert mock_session.call_args.kwargs["cwd"] == "/wt/abc123"

    @pytest.mark.asyncio
    async def test_routes_through_subprocess_not_direct_session(self) -> None:
        """run_tests_and_docs_reviewer calls _run_session_in_subprocess,
        NOT _run_claude_session directly.
        """
        mock_settings = MagicMock(CONTEXT7_API_KEY=None)
        with (
            patch(f"{_RUNNERS_MODULE}.fetch_prompt", new_callable=AsyncMock, return_value="p"),
            patch(
                f"{_RUNNERS_MODULE}._run_session_in_subprocess",
                return_value=_make_session_result("", 0.0),
            ) as mock_subprocess,
            patch(
                f"{_RUNNERS_MODULE}._run_claude_session",
                new_callable=AsyncMock,
            ) as mock_direct,
        ):
            from argus.runners import run_tests_and_docs_reviewer

            await run_tests_and_docs_reviewer(
                plan=_make_plan(),
                diff_text="diff",
                settings=mock_settings,
            )

        mock_subprocess.assert_called_once()
        mock_direct.assert_not_called()


# ---------------------------------------------------------------------------
# _resolve_repo_root: shared fallback helper
# ---------------------------------------------------------------------------


class TestResolveRepoRoot:
    def test_returns_provided_path_without_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        from argus.runners import _resolve_repo_root

        with caplog.at_level(logging.WARNING, logger="argus.runners"):
            result = _resolve_repo_root("/some/worktree", "run_system_reviewer")

        assert result == "/some/worktree"
        assert not caplog.records  # no warning when an explicit path is given

    def test_falls_back_to_repo_root_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        from argus.runners import _REPO_ROOT, _resolve_repo_root

        with caplog.at_level(logging.WARNING, logger="argus.runners"):
            result = _resolve_repo_root(None, "run_specialist_reviewer")

        assert result == _REPO_ROOT
        messages = [r.getMessage() for r in caplog.records]
        assert any("repo_root is None" in m for m in messages)
        # caller name is interpolated into the warning
        assert any("run_specialist_reviewer" in m for m in messages)


# ---------------------------------------------------------------------------
# _run_session_in_subprocess
# ---------------------------------------------------------------------------


class TestRunSessionInSubprocess:
    """Tests for _run_session_in_subprocess timeout and result handling.

    We mock multiprocessing internals to avoid actually spawning processes.
    """

    def test_timeout_returns_empty(self) -> None:
        """When the subprocess is still alive after join, returns a SessionResult
        with empty result_text, 0.0 cost_usd, and duration_seconds equal to
        _SUBPROCESS_TIMEOUT_S, with consistent timestamps.

        When the worker has formed its own process group (pgid == pid),
        os.killpg is used and p.kill() is NOT called directly.
        """
        mock_process = MagicMock()
        mock_process.is_alive.return_value = True
        mock_process.pid = 12345

        mock_recv = MagicMock()

        mock_ctx = MagicMock()
        mock_ctx.Pipe.return_value = (mock_recv, MagicMock())
        mock_ctx.Process.return_value = mock_process

        with (
            patch("multiprocessing.get_context", return_value=mock_ctx),
            patch("argus.runners.os") as mock_os,
        ):
            mock_os.getpgid.return_value = 12345  # pgid == pid: worker is group leader
            from argus.runners import _SUBPROCESS_TIMEOUT_S, _run_session_in_subprocess

            result = _run_session_in_subprocess(
                model="claude-sonnet-4-6",
                system_prompt="test",
                user_message="test",
                anthropic_api_key="test-key",
                context7_key=None,
                cwd="/tmp/repo",
                label="test",
            )

        assert result.result_text == ""
        assert result.cost_usd == 0.0
        assert result.duration_seconds == float(_SUBPROCESS_TIMEOUT_S)
        # Timestamps must be consistent: finished_at = started_at + timeout budget.
        assert result.finished_at == result.started_at + timedelta(seconds=_SUBPROCESS_TIMEOUT_S)
        mock_os.killpg.assert_called_once_with(12345, signal.SIGKILL)
        mock_process.kill.assert_not_called()
        assert result.failure_reason == "timeout"

    def test_timeout_kills_process_group_when_worker_is_group_leader(self) -> None:
        """On timeout, os.killpg is called with the worker's pgid when
        pgid == pid (setsid ran in the worker), not p.kill() directly.
        duration_seconds is set to the timeout budget on the timeout path,
        and finished_at is consistent with started_at + timeout budget.
        """
        mock_process = MagicMock()
        mock_process.is_alive.return_value = True
        mock_process.pid = 99001

        mock_ctx = MagicMock()
        mock_ctx.Pipe.return_value = (MagicMock(), MagicMock())
        mock_ctx.Process.return_value = mock_process

        with (
            patch("multiprocessing.get_context", return_value=mock_ctx),
            patch("argus.runners.os") as mock_os,
        ):
            mock_os.getpgid.return_value = 99001  # pgid == pid
            from argus.runners import _SUBPROCESS_TIMEOUT_S, _run_session_in_subprocess

            result = _run_session_in_subprocess(
                model="claude-sonnet-4-6",
                system_prompt="test",
                user_message="test",
                anthropic_api_key="test-key",
                context7_key=None,
                cwd="/tmp/repo",
                label="test",
            )

        assert result.result_text == ""
        assert result.cost_usd == 0.0
        assert result.duration_seconds == float(_SUBPROCESS_TIMEOUT_S)
        # Timestamps must be consistent: finished_at = started_at + timeout budget.
        assert result.finished_at == result.started_at + timedelta(seconds=_SUBPROCESS_TIMEOUT_S)
        mock_os.killpg.assert_called_once_with(99001, signal.SIGKILL)
        mock_process.kill.assert_not_called()
        assert result.failure_reason == "timeout"

    def test_timeout_startup_race_falls_back_to_kill(self) -> None:
        """On timeout, when pgid != pid (setsid not yet run - startup race),
        os.killpg is NOT called and p.kill() is used instead to avoid
        killing the parent's process group. Timestamps remain consistent.
        """
        mock_process = MagicMock()
        mock_process.is_alive.return_value = True
        mock_process.pid = 55500

        mock_ctx = MagicMock()
        mock_ctx.Pipe.return_value = (MagicMock(), MagicMock())
        mock_ctx.Process.return_value = mock_process

        with (
            patch("multiprocessing.get_context", return_value=mock_ctx),
            patch("argus.runners.os") as mock_os,
        ):
            mock_os.getpgid.return_value = 1000  # pgid != pid: setsid not yet run
            from argus.runners import _SUBPROCESS_TIMEOUT_S, _run_session_in_subprocess

            result = _run_session_in_subprocess(
                model="claude-sonnet-4-6",
                system_prompt="test",
                user_message="test",
                anthropic_api_key="test-key",
                context7_key=None,
                cwd="/tmp/repo",
                label="test",
            )

        assert result.result_text == ""
        assert result.cost_usd == 0.0
        assert result.duration_seconds == float(_SUBPROCESS_TIMEOUT_S)
        # Timestamps must be consistent: finished_at = started_at + timeout budget.
        assert result.finished_at == result.started_at + timedelta(seconds=_SUBPROCESS_TIMEOUT_S)
        mock_os.killpg.assert_not_called()
        mock_process.kill.assert_called_once()
        assert result.failure_reason == "timeout"

    def test_custom_timeout_s_overrides_default(self) -> None:
        """A caller-supplied ``timeout_s`` (from settings.ARGUS_SESSION_TIMEOUT)
        is passed to ``Process.join`` instead of the module default, and the
        timed-out SessionResult's duration_seconds reflects it.
        """
        mock_process = MagicMock()
        mock_process.is_alive.return_value = True
        mock_process.pid = 42

        mock_ctx = MagicMock()
        mock_ctx.Pipe.return_value = (MagicMock(), MagicMock())
        mock_ctx.Process.return_value = mock_process

        with (
            patch("multiprocessing.get_context", return_value=mock_ctx),
            patch("argus.runners.os") as mock_os,
        ):
            mock_os.getpgid.return_value = 42
            from argus.runners import _run_session_in_subprocess

            result = _run_session_in_subprocess(
                model="claude-sonnet-4-6",
                system_prompt="test",
                user_message="test",
                anthropic_api_key="test-key",
                context7_key=None,
                cwd="/tmp/repo",
                label="test",
                timeout_s=45,
            )

        mock_process.join.assert_any_call(timeout=45)
        assert result.duration_seconds == 45.0
        assert result.failure_reason == "timeout"

    def test_normal_completion_returns_result(self) -> None:
        """When subprocess completes and pipe has data, returns that data."""
        expected = _make_session_result("review output text", 0.12)

        mock_process = MagicMock()
        mock_process.is_alive.return_value = False
        mock_process.exitcode = 0

        mock_recv = MagicMock()
        mock_recv.poll.return_value = True
        mock_recv.recv.return_value = expected

        mock_send = MagicMock()

        mock_ctx = MagicMock()
        mock_ctx.Pipe.return_value = (mock_recv, mock_send)
        mock_ctx.Process.return_value = mock_process

        with patch("multiprocessing.get_context", return_value=mock_ctx):
            from argus.runners import _run_session_in_subprocess

            result = _run_session_in_subprocess(
                model="claude-sonnet-4-6",
                system_prompt="test",
                user_message="test",
                anthropic_api_key="test-key",
                context7_key=None,
                cwd="/tmp/repo",
                label="test",
            )

        assert result is expected
        mock_recv.recv.assert_called_once()

    def test_no_pipe_data_returns_empty(self) -> None:
        """When subprocess exits but pipe has no data (worker exception),
        returns a SessionResult with empty result_text and 0.0 cost_usd.
        """
        mock_process = MagicMock()
        mock_process.is_alive.return_value = False
        mock_process.exitcode = 1

        mock_recv = MagicMock()
        mock_recv.poll.return_value = False

        mock_ctx = MagicMock()
        mock_ctx.Pipe.return_value = (mock_recv, MagicMock())
        mock_ctx.Process.return_value = mock_process

        with patch("multiprocessing.get_context", return_value=mock_ctx):
            from argus.runners import _run_session_in_subprocess

            result = _run_session_in_subprocess(
                model="claude-sonnet-4-6",
                system_prompt="test",
                user_message="test",
                anthropic_api_key="test-key",
                context7_key=None,
                cwd="/tmp/repo",
                label="test",
            )

        assert result.result_text == ""
        assert result.cost_usd == 0.0
        assert result.failure_reason == "worker_crashed"
        mock_process.kill.assert_not_called()


# ---------------------------------------------------------------------------
# _validator_worker
# ---------------------------------------------------------------------------


class TestValidatorWorker:
    def test_exception_sends_empty_result(self) -> None:
        """When _run() raises, the worker sends an empty SessionResult via the pipe."""
        mock_pipe = MagicMock()

        with (
            patch(
                f"{_RUNNERS_MODULE}._run_claude_session",
                new_callable=AsyncMock,
                side_effect=RuntimeError("boom"),
            ),
            patch(f"{_RUNNERS_MODULE}.get_settings", return_value=MagicMock()),
        ):
            from argus.runners import _validator_worker

            _validator_worker(
                result_pipe=mock_pipe,
                model="claude-sonnet-4-6",
                system_prompt="test",
                user_message="test",
                anthropic_api_key="test-key",
                anthropic_auth_token=None,
                context7_key=None,
                context7_library_id=None,
                context7_base_url=None,
                cwd="/tmp/repo",
                label="test",
            )

        # The worker should have sent *something* to the pipe
        mock_pipe.send.assert_called()
        sent_value = mock_pipe.send.call_args[0][0]
        # After the refactor, worker sends a SessionResult on error
        assert sent_value.result_text == ""
        assert sent_value.cost_usd == 0.0
        assert sent_value.failure_reason == "worker_crashed"
        mock_pipe.close.assert_called_once()

    def test_forwards_langsmith_parent_headers_as_langsmith_extra(self) -> None:
        """End-to-end coverage for the full propagation chain (flagged by
        Argus's review of this PR): a non-None langsmith_parent_headers must
        reach _run_claude_session as langsmith_extra={"parent": headers}. A
        positional-argument mismatch in the mp.Process args tuple (e.g. after
        a future hand-merge with other runners.py plumbing) would otherwise
        silently re-orphan traces with no CI signal."""
        mock_pipe = MagicMock()
        expected_headers = {"langsmith-trace": "test-parent-id"}

        with (
            patch(
                f"{_RUNNERS_MODULE}._run_claude_session",
                new_callable=AsyncMock,
                return_value=_make_session_result("ok", 0.01),
            ) as mock_run_claude,
            patch(f"{_RUNNERS_MODULE}.get_settings", return_value=MagicMock()),
        ):
            from argus.runners import _validator_worker

            _validator_worker(
                result_pipe=mock_pipe,
                model="claude-sonnet-4-6",
                system_prompt="test",
                user_message="test",
                anthropic_api_key="test-key",
                anthropic_auth_token=None,
                context7_key=None,
                context7_library_id=None,
                context7_base_url=None,
                cwd="/tmp/repo",
                label="test",
                langsmith_parent_headers=expected_headers,
            )

        assert mock_run_claude.call_args.kwargs["langsmith_extra"] == {"parent": expected_headers}

    def test_omits_langsmith_extra_when_no_parent_headers(self) -> None:
        """When langsmith_parent_headers is None (tracing off, or no ambient
        run tree), langsmith_extra must be an empty dict, not {"parent": None}
        -- a non-empty dict with a None parent would be treated by @traceable
        as an explicit (invalid) parent rather than "no parent"."""
        mock_pipe = MagicMock()

        with (
            patch(
                f"{_RUNNERS_MODULE}._run_claude_session",
                new_callable=AsyncMock,
                return_value=_make_session_result("ok", 0.01),
            ) as mock_run_claude,
            patch(f"{_RUNNERS_MODULE}.get_settings", return_value=MagicMock()),
        ):
            from argus.runners import _validator_worker

            _validator_worker(
                result_pipe=mock_pipe,
                model="claude-sonnet-4-6",
                system_prompt="test",
                user_message="test",
                anthropic_api_key="test-key",
                anthropic_auth_token=None,
                context7_key=None,
                context7_library_id=None,
                context7_base_url=None,
                cwd="/tmp/repo",
                label="test",
                langsmith_parent_headers=None,
            )

        assert mock_run_claude.call_args.kwargs["langsmith_extra"] == {}


# ---------------------------------------------------------------------------
# _run_session_isolated: dedicated executor routing
# ---------------------------------------------------------------------------


class TestRunSessionIsolated:
    @pytest.mark.asyncio
    async def test_routes_through_dedicated_executor(self) -> None:
        """_run_session_isolated passes the dedicated executor (from _get_reviewer_executor)
        as the first positional arg to loop.run_in_executor, NOT asyncio.to_thread.
        This test fails if someone reverts to asyncio.to_thread or passes None/default.
        """
        expected = _make_session_result("isolated result", 0.07)
        sentinel_executor = MagicMock(name="sentinel-executor")

        # Capture what run_in_executor was called with so we can assert the executor arg.
        captured: dict[str, object] = {}

        async def _fake_run_in_executor(executor: object, fn: object) -> object:
            captured["executor"] = executor
            # fn is a functools.partial wrapping _run_session_in_subprocess; call it.
            return expected

        mock_loop = MagicMock()
        mock_loop.run_in_executor = _fake_run_in_executor

        with (
            patch(f"{_RUNNERS_MODULE}._get_reviewer_executor", return_value=sentinel_executor),
            patch("asyncio.get_running_loop", return_value=mock_loop),
        ):
            from argus.runners import _run_session_isolated

            result = await _run_session_isolated(
                model="claude-sonnet-4-6",
                system_prompt="test",
                user_message="hello",
                anthropic_api_key="test-key",
                context7_key=None,
                cwd="/tmp/repo",
                label="test-isolated",
            )

        assert result is expected
        # The dedicated executor sentinel must be passed as first arg to run_in_executor.
        assert captured["executor"] is sentinel_executor

    @pytest.mark.asyncio
    async def test_returns_subprocess_result(self) -> None:
        """_run_session_isolated returns the value produced by _run_session_in_subprocess."""
        expected = _make_session_result("isolated result", 0.07)

        with patch(
            f"{_RUNNERS_MODULE}._run_session_in_subprocess",
            return_value=expected,
        ) as mock_subprocess:
            from argus.runners import _run_session_isolated

            result = await _run_session_isolated(
                model="claude-sonnet-4-6",
                system_prompt="test",
                user_message="hello",
                anthropic_api_key="test-key",
                context7_key=None,
                cwd="/tmp/repo",
                label="test-isolated",
            )

        assert result is expected
        mock_subprocess.assert_called_once_with(
            model="claude-sonnet-4-6",
            system_prompt="test",
            user_message="hello",
            anthropic_api_key="test-key",
            context7_key=None,
            cwd="/tmp/repo",
            label="test-isolated",
            # TECH-4734 phase 2: propagated so the subprocess's @traceable span
            # can nest under the caller's LangSmith trace. None here because
            # tracing is off in tests (no ambient run tree to capture).
            langsmith_parent_headers=None,
        )

    @pytest.mark.asyncio
    async def test_captures_and_forwards_parent_run_headers_when_tracing_active(self) -> None:
        """The non-None get_current_run_tree() path -- the actual core of the
        TECH-4734 phase 2 fix -- had zero test coverage before this test
        (flagged by Argus's review of this PR): every other test here only
        exercises langsmith_parent_headers=None (tracing off). Without this,
        a regression that silently drops the captured headers before they
        reach _run_session_in_subprocess would go undetected by CI."""
        expected_headers = {"langsmith-trace": "test-parent-id"}
        mock_run_tree = MagicMock()
        mock_run_tree.to_headers.return_value = expected_headers

        with (
            patch(f"{_RUNNERS_MODULE}.get_current_run_tree", return_value=mock_run_tree),
            patch(
                f"{_RUNNERS_MODULE}._run_session_in_subprocess",
                return_value=_make_session_result("isolated result", 0.07),
            ) as mock_subprocess,
        ):
            from argus.runners import _run_session_isolated

            await _run_session_isolated(
                model="claude-sonnet-4-6",
                system_prompt="test",
                user_message="hello",
                anthropic_api_key="test-key",
                context7_key=None,
                cwd="/tmp/repo",
                label="test-isolated",
            )

        mock_run_tree.to_headers.assert_called_once()
        assert mock_subprocess.call_args.kwargs["langsmith_parent_headers"] == expected_headers


# ---------------------------------------------------------------------------
# run_blocking_validator
# ---------------------------------------------------------------------------

_VALID_BLOCKING_FINDINGS = [
    {"file": "src/auth.py", "line": 42, "description": "SQL injection risk", "context": "..."},
]

_VALID_VALIDATION_JSON = (
    '{"items": [{"index": 0, "verdict": "CONFIRMED", "evidence": "Code matches the claim"}]}'
)


class TestRunBlockingValidator:
    @pytest.mark.asyncio
    async def test_happy_path_returns_agent_run_with_correct_duration_and_cost(self) -> None:
        """Happy path: session result flows through to AgentRunData.duration_seconds and cost_usd."""
        session = _make_session_result(_VALID_VALIDATION_JSON, cost_usd=0.08)
        session.duration_seconds = 17.5

        mock_settings = MagicMock(CONTEXT7_API_KEY=None)

        with (
            patch(
                f"{_RUNNERS_MODULE}.fetch_prompt",
                new_callable=AsyncMock,
                return_value="You are a validator.",
            ),
            patch(
                f"{_RUNNERS_MODULE}._run_session_isolated",
                new_callable=AsyncMock,
                return_value=session,
            ) as mock_session_isolated,
        ):
            from argus.runners import run_blocking_validator

            result, agent_run = await run_blocking_validator(
                blocking_findings=_VALID_BLOCKING_FINDINGS,
                diff_text="diff --git a/src/auth.py ...",
                settings=mock_settings,
            )

        assert isinstance(result, FindingValidationResult)
        assert len(result.items) == 1
        # is_system_reviewer_role must reach _run_session_isolated unchanged --
        # a silent drop at this call site would withhold the 1M-context beta from
        # this reviewer role with no other test catching it.
        assert mock_session_isolated.call_args.kwargs["is_system_reviewer_role"] is True
        assert agent_run is not None
        # duration and cost must flow from the SessionResult to AgentRunData.
        assert agent_run.duration_seconds == 17.5
        assert agent_run.cost_usd == 0.08
        assert agent_run.agent_name == "blocking-validator"
        assert agent_run.agent_type == "blocking_validator"

    @pytest.mark.asyncio
    async def test_empty_findings_skips_session(self) -> None:
        """When blocking_findings is empty, no session is run and result has 0 items."""
        mock_settings = MagicMock(CONTEXT7_API_KEY=None)

        with patch(
            f"{_RUNNERS_MODULE}._run_session_isolated",
            new_callable=AsyncMock,
        ) as mock_session:
            from argus.runners import run_blocking_validator

            result, agent_run = await run_blocking_validator(
                blocking_findings=[],
                diff_text="diff",
                settings=mock_settings,
            )

        mock_session.assert_not_called()
        assert result.items == []
        assert agent_run is None

    @pytest.mark.asyncio
    async def test_timeout_duration_flows_through_to_agent_run(self) -> None:
        """A timed-out session (duration_seconds == _SUBPROCESS_TIMEOUT_S) flows
        through to AgentRunData.duration_seconds unchanged.
        """
        from argus.runners import _SUBPROCESS_TIMEOUT_S

        timeout_session = _make_session_result("", cost_usd=0.0)
        timeout_session.duration_seconds = float(_SUBPROCESS_TIMEOUT_S)
        timeout_session.started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        timeout_session.finished_at = timeout_session.started_at + timedelta(
            seconds=_SUBPROCESS_TIMEOUT_S
        )
        timeout_session.failure_reason = "timeout"

        mock_settings = MagicMock(CONTEXT7_API_KEY=None)

        with (
            patch(
                f"{_RUNNERS_MODULE}.fetch_prompt",
                new_callable=AsyncMock,
                return_value="validator prompt",
            ),
            patch(
                f"{_RUNNERS_MODULE}._run_session_isolated",
                new_callable=AsyncMock,
                return_value=timeout_session,
            ),
        ):
            from argus.runners import run_blocking_validator

            result, agent_run = await run_blocking_validator(
                blocking_findings=_VALID_BLOCKING_FINDINGS,
                diff_text="diff",
                settings=mock_settings,
            )

        assert agent_run is not None
        assert agent_run.duration_seconds == float(_SUBPROCESS_TIMEOUT_S)
        assert agent_run.failure_reason == "timeout"
        # Timed-out session produces no output text, so validator conservatively confirms.
        assert len(result.items) == len(_VALID_BLOCKING_FINDINGS)


# ---------------------------------------------------------------------------
# TECH-4643: large-file-read directive wiring
# ---------------------------------------------------------------------------


class TestLargeFileReadDirectiveWiring:
    def test_directive_reaches_every_broad_exploration_reviewer(self) -> None:
        """Guards against a future edit to one of these four prompt-building
        functions silently dropping the _LARGE_FILE_READ_DIRECTIVE append --
        each is a reviewer that traces across files (the failure mode
        TECH-4643 describes), unlike the feedback-verifier/blocking-validator,
        which only check specific, already-scoped claims."""
        import inspect

        from argus import runners

        for func in (
            runners.run_system_reviewer,
            runners.run_specialist_reviewer,
            runners.run_cross_cutting_reviewer,
            runners.run_tests_and_docs_reviewer,
        ):
            source = inspect.getsource(func)
            assert "_LARGE_FILE_READ_DIRECTIVE" in source, (
                f"{func.__name__} is missing the large-file-read directive"
            )


# ---------------------------------------------------------------------------
# is_system_reviewer_role forwarding across all six reviewer entry points
# ---------------------------------------------------------------------------


class TestIsSystemReviewerRoleForwarding:
    """Each of the six public run_* reviewer functions must forward (or
    correctly omit) is_system_reviewer_role to _run_session_isolated exactly
    once. run_tests_and_docs_reviewer and run_blocking_validator already have
    this assertion added to their existing happy-path tests above; the
    remaining four (run_system_reviewer, run_specialist_reviewer,
    run_feedback_verifier, run_cross_cutting_reviewer) had no direct test
    coverage at all before this class -- a silent drop at any of these five
    True call sites would withhold the 1M-context beta from that role with
    nothing catching it; a silent True at the one False call site
    (run_cross_cutting_reviewer) would extend an opus-unverified
    billing/compatibility risk to the cross-cutting session.
    """

    def _settings(self) -> MagicMock:
        settings = MagicMock()
        settings.CONTEXT7_API_KEY = None
        settings.ARGUS_CONTEXT7_LIBRARY_ID = None
        settings.ARGUS_CONTEXT7_BASE_URL = None
        settings.ARGUS_SESSION_TIMEOUT = 600
        settings.ANTHROPIC_API_KEY = "sk-test"
        settings.ANTHROPIC_AUTH_TOKEN = None
        return settings

    @pytest.mark.asyncio
    async def test_run_system_reviewer_passes_true(self) -> None:
        group = SystemGroup(name="backend", files=["src/app.py"], conventions="", review_focus="")
        session = _make_session_result("", cost_usd=0.0)

        with (
            patch(f"{_RUNNERS_MODULE}.fetch_prompt", new_callable=AsyncMock, return_value="prompt"),
            patch(
                f"{_RUNNERS_MODULE}._run_session_isolated",
                new_callable=AsyncMock,
                return_value=session,
            ) as mock_session,
        ):
            from argus.runners import run_system_reviewer

            await run_system_reviewer(
                group=group,
                diff_text="diff --git a/src/app.py b/src/app.py\n@@ -1 +1 @@\n-a\n+b\n",
                settings=self._settings(),
                repo_root="/tmp/fake-repo",
            )

        assert mock_session.call_args.kwargs["is_system_reviewer_role"] is True

    @pytest.mark.asyncio
    async def test_run_specialist_reviewer_passes_true(self) -> None:
        group = SystemGroup(name="backend", files=["src/app.py"], conventions="", review_focus="")
        session = _make_session_result("", cost_usd=0.0)

        with (
            patch(f"{_RUNNERS_MODULE}.fetch_prompt", new_callable=AsyncMock, return_value="prompt"),
            patch(
                f"{_RUNNERS_MODULE}._run_session_isolated",
                new_callable=AsyncMock,
                return_value=session,
            ) as mock_session,
        ):
            from argus.runners import run_specialist_reviewer

            await run_specialist_reviewer(
                specialist="security",
                group=group,
                diff_text="diff --git a/src/app.py b/src/app.py\n@@ -1 +1 @@\n-a\n+b\n",
                settings=self._settings(),
                repo_root="/tmp/fake-repo",
            )

        assert mock_session.call_args.kwargs["is_system_reviewer_role"] is True

    @pytest.mark.asyncio
    async def test_run_feedback_verifier_passes_true(self) -> None:
        prior_context = PriorReviewContext(
            review_id="00000000-0000-0000-0000-000000000000",
            reviewed_sha="a" * 40,
            findings=[
                PriorFinding(severity="BLOCKING", description="SQL injection risk"),
            ],
        )
        session = _make_session_result("", cost_usd=0.0)

        with (
            patch(f"{_RUNNERS_MODULE}.fetch_prompt", new_callable=AsyncMock, return_value="prompt"),
            patch(
                f"{_RUNNERS_MODULE}._run_session_isolated",
                new_callable=AsyncMock,
                return_value=session,
            ) as mock_session,
        ):
            from argus.runners import run_feedback_verifier

            result, _ = await run_feedback_verifier(
                prior_context=prior_context,
                diff_text="diff --git a/src/app.py b/src/app.py\n@@ -1 +1 @@\n-a\n+b\n",
                settings=self._settings(),
                repo_root="/tmp/fake-repo",
            )

        assert isinstance(result, FeedbackVerificationResult)
        assert mock_session.call_args.kwargs["is_system_reviewer_role"] is True

    @pytest.mark.asyncio
    async def test_run_cross_cutting_reviewer_omits_true(self) -> None:
        """The one call site that must NOT pass is_system_reviewer_role=True:
        cross-cutting is the opus session this whole role-based gate exists
        to protect from an unverified-for-opus 1M-context beta."""
        plan = ReviewPlan(
            system_groups=[
                SystemGroup(name="backend", files=["src/app.py"], conventions="", review_focus=""),
            ],
            cross_cutting_concerns=["test coverage"],
            file_manifest=[FileEntry(path="src/app.py", change_type="modified")],
        )
        session = _make_session_result("", cost_usd=0.0)

        with (
            patch(f"{_RUNNERS_MODULE}.fetch_prompt", new_callable=AsyncMock, return_value="prompt"),
            patch(
                f"{_RUNNERS_MODULE}._run_session_isolated",
                new_callable=AsyncMock,
                return_value=session,
            ) as mock_session,
        ):
            from argus.runners import run_cross_cutting_reviewer

            await run_cross_cutting_reviewer(
                plan=plan,
                diff_text="diff --git a/src/app.py b/src/app.py\n@@ -1 +1 @@\n-a\n+b\n",
                settings=self._settings(),
                repo_root="/tmp/fake-repo",
            )

        call_kwargs = mock_session.call_args.kwargs
        assert call_kwargs.get("is_system_reviewer_role", False) is False
