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
    FileEntry,
    FindingValidationResult,
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
    sr.timed_out = False
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

    @pytest.mark.asyncio
    async def test_timed_out_session_marks_result_and_agent_run(self) -> None:
        """A SessionResult with timed_out=True flows through _parse_review_result's
        output into SystemReviewResult.timed_out and AgentRunData.timed_out, so a
        killed reviewer is distinguishable from one that genuinely found nothing.
        """
        mock_settings = MagicMock()
        mock_settings.CONTEXT7_API_KEY = None
        mock_settings.ARGUS_SESSION_TIMEOUT = 300
        timeout_session = _make_session_result("", cost_usd=0.0)
        timeout_session.timed_out = True

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

        assert result.timed_out is True
        assert result.findings == []
        assert agent_run is not None
        assert agent_run.timed_out is True
        call_kwargs = mock_session.call_args.kwargs
        assert call_kwargs["timeout_s"] == 300

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
                api_key="test-key",
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
        assert result.timed_out is True

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
                api_key="test-key",
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
        assert result.timed_out is True

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
                api_key="test-key",
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
        assert result.timed_out is True

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
                api_key="test-key",
                context7_key=None,
                cwd="/tmp/repo",
                label="test",
                timeout_s=45,
            )

        mock_process.join.assert_any_call(timeout=45)
        assert result.duration_seconds == 45.0
        assert result.timed_out is True

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
                api_key="test-key",
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
                api_key="test-key",
                context7_key=None,
                cwd="/tmp/repo",
                label="test",
            )

        assert result.result_text == ""
        assert result.cost_usd == 0.0
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
                api_key="test-key",
                context7_key=None,
                context7_library_id=None,
                cwd="/tmp/repo",
                label="test",
            )

        # The worker should have sent *something* to the pipe
        mock_pipe.send.assert_called()
        sent_value = mock_pipe.send.call_args[0][0]
        # After the refactor, worker sends a SessionResult on error
        assert sent_value.result_text == ""
        assert sent_value.cost_usd == 0.0
        mock_pipe.close.assert_called_once()


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
                api_key="test-key",
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
                api_key="test-key",
                context7_key=None,
                cwd="/tmp/repo",
                label="test-isolated",
            )

        assert result is expected
        mock_subprocess.assert_called_once_with(
            model="claude-sonnet-4-6",
            system_prompt="test",
            user_message="hello",
            api_key="test-key",
            context7_key=None,
            cwd="/tmp/repo",
            label="test-isolated",
        )


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
            ),
        ):
            from argus.runners import run_blocking_validator

            result, agent_run = await run_blocking_validator(
                blocking_findings=_VALID_BLOCKING_FINDINGS,
                diff_text="diff --git a/src/auth.py ...",
                settings=mock_settings,
            )

        assert isinstance(result, FindingValidationResult)
        assert len(result.items) == 1
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
        timeout_session.timed_out = True

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
        assert agent_run.timed_out is True
        # Timed-out session produces no output text, so validator conservatively confirms.
        assert len(result.items) == len(_VALID_BLOCKING_FINDINGS)
