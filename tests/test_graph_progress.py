"""Unit tests for run_review progress changes in graph.py.

Tests cover:
- run_review uses flow_run_id as thread_id
- run_review finalization (upsert without old progress columns)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_GRAPH_MODULE = "argus.graph"
_GH_CLIENT_CLASS = "argus.github_client.GitHubClient"
_PROVISIONED_WORKTREE = f"{_GRAPH_MODULE}.provisioned_worktree"

_FAKE_SHA = "a" * 40


def _mock_gh_client() -> MagicMock:
    """Return a mock GitHubClient whose get_pull_request returns a fake SHA."""
    mock_gh = MagicMock()
    mock_gh.get_pull_request.return_value = {"head_sha": _FAKE_SHA}
    return mock_gh


def _mock_provisioned_worktree() -> MagicMock:
    """Return a mock provisioned_worktree context manager yielding a fake path."""
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value="/tmp/fake-worktree")
    mock_ctx.__aexit__ = AsyncMock(return_value=None)
    return mock_ctx


# ---------------------------------------------------------------------------
# run_review: thread_id = flow_run_id
# ---------------------------------------------------------------------------


class TestRunReviewThreadId:
    """Tests that run_review uses flow_run_id as the LangGraph thread_id."""

    @pytest.mark.asyncio
    async def test_flow_run_id_used_as_thread_id(self) -> None:
        """When flow_run_id is provided, it is used as the LangGraph thread_id."""
        from argus.graph import run_review
        from argus.models import (
            ReviewRequest,
            ReviewResponse,
            RiskLevel,
            TokenUsage,
            Verdict,
        )

        request = ReviewRequest(repo="org/repo", pr_number=42)

        mock_response = ReviewResponse(
            verdict=Verdict.APPROVE,
            risk_level=RiskLevel.LOW,
            findings=[],
            coverage_map=[],
            review_comment="LGTM",
            usage=TokenUsage(input_tokens=100, output_tokens=50, cost_usd=0.01),
        )

        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "response": mock_response.model_dump(),
            "findings": [],
        }

        mock_session = AsyncMock()
        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_session_factory = MagicMock(return_value=mock_session_ctx)

        with (
            patch(f"{_GRAPH_MODULE}.build_pipeline") as mock_build,
            patch(
                "argus.storage.resolver.get_async_session_factory",
                return_value=mock_session_factory,
            ),
            patch(_GH_CLIENT_CLASS, return_value=_mock_gh_client()),
            patch(_PROVISIONED_WORKTREE, return_value=_mock_provisioned_worktree()),
        ):
            mock_build.return_value.__aenter__ = AsyncMock(return_value=mock_graph)
            mock_build.return_value.__aexit__ = AsyncMock(return_value=None)

            await run_review(request, flow_run_id="flow-abc-123")

        # Verify ainvoke was called with flow_run_id as thread_id
        call_kwargs = mock_graph.ainvoke.call_args
        config = call_kwargs[1].get("config") or call_kwargs[0][1]
        assert config["configurable"]["thread_id"] == "flow-abc-123"


# ---------------------------------------------------------------------------
# run_review: head_sha + worktree_path threaded into ainvoke config
# ---------------------------------------------------------------------------


class TestRunReviewWorktreeThreading:
    """run_review must thread the resolved head_sha and provisioned
    worktree_path into the LangGraph config so reviewers explore the exact
    reviewed commit."""

    def _common_mocks(self) -> tuple[Any, Any]:
        from argus.models import (
            ReviewResponse,
            RiskLevel,
            TokenUsage,
            Verdict,
        )

        mock_response = ReviewResponse(
            verdict=Verdict.APPROVE,
            risk_level=RiskLevel.LOW,
            findings=[],
            coverage_map=[],
            review_comment="LGTM",
            usage=TokenUsage(input_tokens=100, output_tokens=50, cost_usd=0.01),
        )
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "response": mock_response.model_dump(),
            "findings": [],
            "head_sha": _FAKE_SHA,
        }
        mock_session = AsyncMock()
        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_session_factory = MagicMock(return_value=mock_session_ctx)
        return mock_graph, mock_session_factory

    @pytest.mark.asyncio
    async def test_pr_path_threads_head_sha_and_worktree(self) -> None:
        """PR-number path: head_sha resolved from the GitHub API and the
        provisioned worktree_path are both placed in the graph config."""
        from argus.graph import run_review
        from argus.models import ReviewRequest

        request = ReviewRequest(repo="org/repo", pr_number=42)
        mock_graph, mock_session_factory = self._common_mocks()

        with (
            patch(f"{_GRAPH_MODULE}.build_pipeline") as mock_build,
            patch(
                "argus.storage.resolver.get_async_session_factory",
                return_value=mock_session_factory,
            ),
            patch(_GH_CLIENT_CLASS, return_value=_mock_gh_client()),
            patch(_PROVISIONED_WORKTREE, return_value=_mock_provisioned_worktree()),
        ):
            mock_build.return_value.__aenter__ = AsyncMock(return_value=mock_graph)
            mock_build.return_value.__aexit__ = AsyncMock(return_value=None)

            await run_review(request, flow_run_id="flow-pr")

        config = mock_graph.ainvoke.call_args[1]["config"]
        assert config["configurable"]["head_sha"] == _FAKE_SHA
        assert config["configurable"]["worktree_path"] == "/tmp/fake-worktree"

    @pytest.mark.asyncio
    async def test_sha_only_path_threads_head_sha_and_worktree(self) -> None:
        """SHA-only path (no PR number): head_sha comes straight from the
        request and the worktree is still provisioned and threaded."""
        from argus.graph import run_review
        from argus.models import ReviewRequest

        request = ReviewRequest(repo="org/repo", sha=_FAKE_SHA)
        mock_graph, mock_session_factory = self._common_mocks()

        with (
            patch(f"{_GRAPH_MODULE}.build_pipeline") as mock_build,
            patch(
                "argus.storage.resolver.get_async_session_factory",
                return_value=mock_session_factory,
            ),
            patch(_GH_CLIENT_CLASS, return_value=_mock_gh_client()) as mock_gh,
            patch(_PROVISIONED_WORKTREE, return_value=_mock_provisioned_worktree()) as mock_wt,
        ):
            mock_build.return_value.__aenter__ = AsyncMock(return_value=mock_graph)
            mock_build.return_value.__aexit__ = AsyncMock(return_value=None)

            await run_review(request, flow_run_id="flow-sha")

        config = mock_graph.ainvoke.call_args[1]["config"]
        assert config["configurable"]["head_sha"] == _FAKE_SHA
        assert config["configurable"]["worktree_path"] == "/tmp/fake-worktree"
        # SHA-only path must NOT hit the GitHub PR API to resolve the SHA.
        mock_gh.return_value.get_pull_request.assert_not_called()
        # Worktree provisioned with the request SHA.
        assert mock_wt.call_args[1]["head_sha"] == _FAKE_SHA


# ---------------------------------------------------------------------------
# run_review finalization: upsert
# ---------------------------------------------------------------------------


class TestRunReviewFinalization:
    """Tests for run_review upsert finalization."""

    def _make_mocks(self) -> tuple[AsyncMock, MagicMock, Any]:
        """Create common mocks for run_review tests."""
        from argus.models import (
            ReviewResponse,
            RiskLevel,
            TokenUsage,
            Verdict,
        )

        mock_response = ReviewResponse(
            verdict=Verdict.APPROVE,
            risk_level=RiskLevel.LOW,
            findings=[],
            coverage_map=[],
            review_comment="LGTM",
            usage=TokenUsage(input_tokens=100, output_tokens=50, cost_usd=0.01),
        )

        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "response": mock_response.model_dump(),
            "findings": [],
        }

        mock_session = AsyncMock()
        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_session_factory = MagicMock(return_value=mock_session_ctx)

        return mock_graph, mock_session_factory, mock_session

    @pytest.mark.asyncio
    async def test_finalization_uses_upsert(self) -> None:
        """Final write uses INSERT ... ON CONFLICT to handle retries safely."""
        from argus.graph import run_review
        from argus.models import ReviewRequest, Verdict

        request = ReviewRequest(repo="org/repo", pr_number=42)
        mock_graph, mock_session_factory, mock_session = self._make_mocks()

        with (
            patch(f"{_GRAPH_MODULE}.build_pipeline") as mock_build,
            patch(
                "argus.storage.resolver.get_async_session_factory",
                return_value=mock_session_factory,
            ),
            patch(_GH_CLIENT_CLASS, return_value=_mock_gh_client()),
            patch(_PROVISIONED_WORKTREE, return_value=_mock_provisioned_worktree()),
        ):
            mock_build.return_value.__aenter__ = AsyncMock(return_value=mock_graph)
            mock_build.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await run_review(request, flow_run_id="flow-abc")

        assert result.verdict == Verdict.APPROVE

        # Two DB calls: initial upsert (running row) + final upsert (completed result)
        assert mock_session.execute.await_count == 2

        # Both should be INSERT ... ON CONFLICT upserts
        for call in mock_session.execute.call_args_list:
            sql_text = str(call[0][0].text)
            assert "INSERT INTO review_service.code_reviews" in sql_text
            assert "ON CONFLICT" in sql_text

        # Final upsert should NOT contain blocking_so_far
        final_sql = str(mock_session.execute.call_args_list[-1][0][0].text)
        assert "blocking_so_far" not in final_sql

    @pytest.mark.asyncio
    async def test_finalize_log_names_actual_backend(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The finalize log line must name whichever backend actually
        persisted the round (here Postgres, per the module's default mocked
        backend), not a hardcoded vendor name."""
        from argus.graph import run_review
        from argus.models import ReviewRequest

        request = ReviewRequest(repo="org/repo", pr_number=42)
        mock_graph, mock_session_factory, _mock_session = self._make_mocks()

        with (
            patch(f"{_GRAPH_MODULE}.build_pipeline") as mock_build,
            patch(
                "argus.storage.resolver.get_async_session_factory",
                return_value=mock_session_factory,
            ),
            patch(_GH_CLIENT_CLASS, return_value=_mock_gh_client()),
            patch(_PROVISIONED_WORKTREE, return_value=_mock_provisioned_worktree()),
            caplog.at_level("INFO"),
        ):
            mock_build.return_value.__aenter__ = AsyncMock(return_value=mock_graph)
            mock_build.return_value.__aexit__ = AsyncMock(return_value=None)

            await run_review(request, flow_run_id="flow-abc")

        finalize_messages = [rec.message for rec in caplog.records if "review to" in rec.message]
        assert finalize_messages, "expected a '... review to <backend>' log line"
        assert "Postgres" in finalize_messages[0]
        assert "Supabase" not in finalize_messages[0]

    @pytest.mark.asyncio
    async def test_initial_upsert_no_progress_columns(self) -> None:
        """Initial upsert should not write reviewers_total, reviewers_completed, or blocking_so_far."""
        from argus.graph import run_review
        from argus.models import ReviewRequest

        request = ReviewRequest(repo="org/repo", pr_number=42)
        mock_graph, mock_session_factory, mock_session = self._make_mocks()

        with (
            patch(f"{_GRAPH_MODULE}.build_pipeline") as mock_build,
            patch(
                "argus.storage.resolver.get_async_session_factory",
                return_value=mock_session_factory,
            ),
            patch(_GH_CLIENT_CLASS, return_value=_mock_gh_client()),
            patch(_PROVISIONED_WORKTREE, return_value=_mock_provisioned_worktree()),
        ):
            mock_build.return_value.__aenter__ = AsyncMock(return_value=mock_graph)
            mock_build.return_value.__aexit__ = AsyncMock(return_value=None)

            await run_review(request, flow_run_id="flow-init-test")

        # Initial upsert is the first DB call
        initial_sql = str(mock_session.execute.call_args_list[0][0][0].text)
        assert "reviewers_total" not in initial_sql
        assert "reviewers_completed" not in initial_sql
        assert "blocking_so_far" not in initial_sql

    @pytest.mark.asyncio
    async def test_flow_run_id_not_in_state(self) -> None:
        """flow_run_id should NOT be passed in the graph state dict."""
        from argus.graph import run_review
        from argus.models import ReviewRequest

        request = ReviewRequest(repo="org/repo", pr_number=42)
        mock_graph, mock_session_factory, _mock_session = self._make_mocks()

        with (
            patch(f"{_GRAPH_MODULE}.build_pipeline") as mock_build,
            patch(
                "argus.storage.resolver.get_async_session_factory",
                return_value=mock_session_factory,
            ),
            patch(_GH_CLIENT_CLASS, return_value=_mock_gh_client()),
            patch(_PROVISIONED_WORKTREE, return_value=_mock_provisioned_worktree()),
        ):
            mock_build.return_value.__aenter__ = AsyncMock(return_value=mock_graph)
            mock_build.return_value.__aexit__ = AsyncMock(return_value=None)

            await run_review(request, flow_run_id="flow-state-test")

        # The state dict (first arg to ainvoke) should not contain flow_run_id
        state_dict = mock_graph.ainvoke.call_args[0][0]
        assert "flow_run_id" not in state_dict


# ---------------------------------------------------------------------------
# run_review: agent_runs persistence
# ---------------------------------------------------------------------------


class TestRunReviewAgentRuns:
    """Tests for per-agent execution data persistence in run_review."""

    def _make_agent_run_dict(self) -> dict[str, Any]:
        """Create a sample AgentRunData dict as it appears in graph state."""
        return {
            "agent_name": "system:test-group",
            "agent_type": "system",
            "model": "claude-sonnet-4-6",
            "cost_usd": 0.5,
            "duration_seconds": 120.0,
            "started_at": datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
            "finished_at": datetime(2026, 1, 1, 0, 2, 0, tzinfo=timezone.utc).isoformat(),
            "tool_call_count": 10,
            "tool_names": ["Read", "Grep"],
            "context7_call_count": 1,
            "files_explored": ["src/app.py"],
            "finding_count": 3,
            "result_text_length": 500,
        }

    def _make_mocks(
        self, *, include_agent_runs: bool = True
    ) -> tuple[AsyncMock, MagicMock, AsyncMock, AsyncMock]:
        """Create mocks for run_review with agent_runs support."""
        from argus.models import (
            ReviewResponse,
            RiskLevel,
            TokenUsage,
            Verdict,
        )

        mock_response = ReviewResponse(
            verdict=Verdict.APPROVE,
            risk_level=RiskLevel.LOW,
            findings=[],
            coverage_map=[],
            review_comment="LGTM",
            usage=TokenUsage(input_tokens=100, output_tokens=50, cost_usd=0.01),
        )

        graph_result: dict[str, Any] = {
            "response": mock_response.model_dump(),
            "findings": [],
        }
        if include_agent_runs:
            graph_result["agent_runs"] = [self._make_agent_run_dict()]

        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = graph_result

        # Core session (for code_reviews upsert). The canonical writer reads
        # ``result.mappings().first()`` and constructs a
        # ``CodeReviewRoundRow``; the mock returns the full mapping shape
        # that model expects.
        import datetime as _dt

        review_id = uuid.uuid4()
        mock_mappings = MagicMock()
        mock_mappings.first.return_value = {
            "id": review_id,
            "flow_run_id": "flow-agent-runs",
            "repo": "org/repo",
            "pr_number": 99,
            "verdict": "APPROVE",
            "risk_level": "LOW",
            "blocking_count": 0,
            "suggestion_count": 0,
            "review_comment": "LGTM",
            "result_json": {},
            "cost_usd": 0.0,
            "duration_seconds": 1.0,
            "reviewer_version": "v3",
            "orchestrator_model": "claude-opus",
            "subagent_model": "claude-sonnet",
            "sha": None,
            "base_ref": None,
            "current_stage": "completed",
            "created_at": _dt.datetime.now(_dt.timezone.utc),
        }
        mock_execute_result = MagicMock()
        mock_execute_result.mappings = MagicMock(return_value=mock_mappings)

        mock_core_session = AsyncMock()
        mock_core_session.execute.return_value = mock_execute_result

        mock_core_ctx = MagicMock()
        mock_core_ctx.__aenter__ = AsyncMock(return_value=mock_core_session)
        mock_core_ctx.__aexit__ = AsyncMock(return_value=None)

        # Agent session (for agent_runs inserts)
        mock_agent_session = AsyncMock()
        mock_agent_ctx = MagicMock()
        mock_agent_ctx.__aenter__ = AsyncMock(return_value=mock_agent_session)
        mock_agent_ctx.__aexit__ = AsyncMock(return_value=None)

        # Factory returns core session first, then agent session
        mock_session_factory = MagicMock(side_effect=[mock_core_ctx, mock_core_ctx, mock_agent_ctx])

        return mock_graph, mock_session_factory, mock_core_session, mock_agent_session

    @pytest.mark.asyncio
    async def test_agent_runs_inserted_in_separate_transaction(self) -> None:
        """agent_runs should be inserted in a separate session after code_reviews commits."""
        from argus.graph import run_review
        from argus.models import ReviewRequest

        request = ReviewRequest(repo="org/repo", pr_number=99)
        mock_graph, mock_session_factory, mock_core_session, mock_agent_session = self._make_mocks()

        with (
            patch(f"{_GRAPH_MODULE}.build_pipeline") as mock_build,
            patch(
                "argus.storage.resolver.get_async_session_factory",
                return_value=mock_session_factory,
            ),
            patch(_GH_CLIENT_CLASS, return_value=_mock_gh_client()),
            patch(_PROVISIONED_WORKTREE, return_value=_mock_provisioned_worktree()),
        ):
            mock_build.return_value.__aenter__ = AsyncMock(return_value=mock_graph)
            mock_build.return_value.__aexit__ = AsyncMock(return_value=None)

            await run_review(request, flow_run_id="flow-agent-runs")

        # Core session commits the code_reviews row
        mock_core_session.commit.assert_awaited()

        # Agent session receives the agent_runs INSERT
        assert mock_agent_session.execute.await_count >= 1
        agent_sql = str(mock_agent_session.execute.call_args[0][0].text)
        assert "INSERT INTO review_service.agent_runs" in agent_sql
        mock_agent_session.commit.assert_awaited()

    @pytest.mark.asyncio
    async def test_agent_runs_skipped_on_storage_write_failure(self) -> None:
        """When code_reviews commit fails, agent_runs insert is skipped."""
        from argus.graph import run_review
        from argus.models import ReviewRequest, Verdict

        request = ReviewRequest(repo="org/repo", pr_number=99)
        mock_graph, mock_session_factory, mock_core_session, mock_agent_session = self._make_mocks()

        # Make the core session commit raise an error
        mock_core_session.execute.side_effect = RuntimeError("storage backend down")

        with (
            patch(f"{_GRAPH_MODULE}.build_pipeline") as mock_build,
            patch(
                "argus.storage.resolver.get_async_session_factory",
                return_value=mock_session_factory,
            ),
            patch(_GH_CLIENT_CLASS, return_value=_mock_gh_client()),
            patch(_PROVISIONED_WORKTREE, return_value=_mock_provisioned_worktree()),
        ):
            mock_build.return_value.__aenter__ = AsyncMock(return_value=mock_graph)
            mock_build.return_value.__aexit__ = AsyncMock(return_value=None)

            # Should not raise — error is swallowed
            result = await run_review(request, flow_run_id="flow-fail")

        assert result.verdict == Verdict.APPROVE
        # Agent session should never have been used
        mock_agent_session.execute.assert_not_awaited()


# ---------------------------------------------------------------------------
# build_pipeline: checkpoint pool teardown resilience
# ---------------------------------------------------------------------------


class TestBuildPipelinePoolTeardown:
    """build_pipeline must not let a checkpoint pool-close failure on
    teardown discard a review whose verdict is already computed.

    A PgBouncer-style pooler in front of Postgres reaps idle connections
    during the multi-minute reviewer fan-out, so the pool's close() can hit
    a dead socket. The verdict is fully built by graph.ainvoke() before that
    flush, so the close failure is swallowed as a logged warning.
    """

    @pytest.mark.asyncio
    async def test_pool_close_failure_is_non_fatal(self) -> None:
        """A pool.close() error during teardown does not propagate."""
        import psycopg

        from argus import graph as graph_module

        mock_settings = MagicMock()
        mock_settings.ANTHROPIC_API_KEY = None
        mock_settings.OPENAI_API_KEY = None
        mock_settings.LANGSMITH_API_KEY = None
        mock_settings.LANGSMITH_PROJECT = None
        mock_settings.LANGSMITH_WORKSPACE_ID = None
        mock_settings.SUPABASE_DB_URL = "postgresql://user:pass@host:5432/db"

        # Pool opens cleanly but close() raises — mimics the pooler
        # reaping the connection mid-review.
        mock_pool = MagicMock()
        mock_pool.open = AsyncMock()
        mock_pool.close = AsyncMock(
            side_effect=psycopg.OperationalError("connection unexpectedly closed")
        )

        mock_saver = MagicMock()
        mock_saver.setup = AsyncMock()

        # Stand in for the compiled graph so copy() doesn't validate the
        # mock saver against langgraph internals.
        sentinel_graph = MagicMock(name="compiled_graph")
        mock_review_graph = MagicMock()
        mock_review_graph.copy.return_value = sentinel_graph

        original_flag = graph_module._checkpoint_tables_created
        try:
            with (
                patch("argus.config.get_settings", return_value=mock_settings),
                patch(f"{_GRAPH_MODULE}._review_graph", mock_review_graph),
                patch("psycopg_pool.AsyncConnectionPool", return_value=mock_pool),
                patch(
                    "langgraph.checkpoint.postgres.aio.AsyncPostgresSaver",
                    return_value=mock_saver,
                ),
            ):
                graph_module._checkpoint_tables_created = False
                yielded = None
                # Must not raise despite close() failing on exit.
                async with graph_module.build_pipeline() as graph:
                    yielded = graph
        finally:
            graph_module._checkpoint_tables_created = original_flag

        assert yielded is sentinel_graph
        mock_pool.open.assert_awaited_once()
        mock_pool.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# run_review: max_concurrency cap
# ---------------------------------------------------------------------------


class TestRunReviewMaxConcurrency:
    """max_concurrency must be present in the config passed to graph.ainvoke."""

    @pytest.mark.asyncio
    async def test_max_concurrency_in_ainvoke_config(self) -> None:
        """graph.ainvoke must be called with max_concurrency equal to
        _MAX_CONCURRENT_REVIEWERS as a top-level config key (not inside
        configurable)."""
        from argus.graph import _MAX_CONCURRENT_REVIEWERS, run_review
        from argus.models import (
            ReviewRequest,
            ReviewResponse,
            RiskLevel,
            TokenUsage,
            Verdict,
        )

        request = ReviewRequest(repo="org/repo", pr_number=42)
        mock_response = ReviewResponse(
            verdict=Verdict.APPROVE,
            risk_level=RiskLevel.LOW,
            findings=[],
            coverage_map=[],
            review_comment="LGTM",
            usage=TokenUsage(input_tokens=100, output_tokens=50, cost_usd=0.01),
        )
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "response": mock_response.model_dump(),
            "findings": [],
        }

        mock_session = AsyncMock()
        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_session_factory = MagicMock(return_value=mock_session_ctx)

        with (
            patch(f"{_GRAPH_MODULE}.build_pipeline") as mock_build,
            patch(
                "argus.storage.resolver.get_async_session_factory",
                return_value=mock_session_factory,
            ),
            patch(_GH_CLIENT_CLASS, return_value=_mock_gh_client()),
            patch(_PROVISIONED_WORKTREE, return_value=_mock_provisioned_worktree()),
        ):
            mock_build.return_value.__aenter__ = AsyncMock(return_value=mock_graph)
            mock_build.return_value.__aexit__ = AsyncMock(return_value=None)

            await run_review(request, flow_run_id="flow-concurrency-test")

        call_kwargs = mock_graph.ainvoke.call_args
        config = call_kwargs[1].get("config") or call_kwargs[0][1]
        assert config["max_concurrency"] == _MAX_CONCURRENT_REVIEWERS
        # max_concurrency must NOT be nested inside configurable
        assert "max_concurrency" not in config.get("configurable", {})


# ---------------------------------------------------------------------------
# _edge_fan_out_reviewers: agent-count ceiling
# ---------------------------------------------------------------------------


class TestEdgeFanOutReviewersCeiling:
    """When the plan produces more than _MAX_REVIEWER_FANOUT system groups,
    the fan-out list must be truncated to exactly _MAX_REVIEWER_FANOUT."""

    def _make_state_with_n_groups(self, n: int) -> Any:
        """Build a minimal ReviewState dict with n system groups (no specialists)."""
        from argus.pipeline_models import FileEntry, ReviewPlan, SystemGroup

        groups = [
            SystemGroup(
                name=f"group-{i}",
                files=[f"file_{i}.py"],
                conventions="",
                review_focus="check it",
                specialists_needed=[],
            )
            for i in range(n)
        ]
        plan = ReviewPlan(
            system_groups=groups,
            cross_cutting_concerns=[],
            file_manifest=[
                FileEntry(path=f"file_{i}.py", change_type="modified") for i in range(n)
            ],
        )
        from argus.models import ReviewRequest

        return {
            "plan": plan.model_dump(),
            "diff": "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new",
            "request": ReviewRequest(repo="org/repo", pr_number=1).model_dump(),
            "prior_review": {},
        }

    def test_fanout_truncated_to_cap(self) -> None:
        """With 60 system groups, _edge_fan_out_reviewers must return exactly
        _MAX_REVIEWER_FANOUT Sends, not 60+2 (cross-cutting + tests-and-docs)."""
        from argus.graph import _MAX_REVIEWER_FANOUT, _edge_fan_out_reviewers

        state = self._make_state_with_n_groups(60)
        result = _edge_fan_out_reviewers(state)
        assert len(result) == _MAX_REVIEWER_FANOUT

    def test_fanout_not_truncated_when_under_cap(self) -> None:
        """With fewer groups than the cap, all Sends are returned unchanged."""
        from argus.graph import _MAX_REVIEWER_FANOUT, _edge_fan_out_reviewers

        # 3 groups -> 3 system + 1 cross-cutting + 1 tests-and-docs = 5 sends
        state = self._make_state_with_n_groups(3)
        result = _edge_fan_out_reviewers(state)
        assert len(result) == 5
        assert len(result) <= _MAX_REVIEWER_FANOUT

    def test_fanout_always_preserves_protected_reviewers(self) -> None:
        """BLOCKING regression: with 60 groups (exceeding the cap), the cap must
        never drop the always-on cross_cutting and tests_and_docs reviewers.

        Old sends[:50] logic would truncate from the end and silently remove
        cross_cutting and tests_and_docs (appended last). The fix uses an explicit
        cappable set (_CAPPABLE_REVIEWER_TYPES) so only plan-driven reviewers are
        subject to the ceiling, and always-on singletons are preserved by default.
        """
        from argus.graph import _MAX_REVIEWER_FANOUT, _edge_fan_out_reviewers

        state = self._make_state_with_n_groups(60)
        result = _edge_fan_out_reviewers(state)

        # Total must still respect the cap
        assert len(result) <= _MAX_REVIEWER_FANOUT

        # Always-on reviewers must be present regardless of group count
        reviewer_types = [s.arg.get("reviewer_type") for s in result]
        assert "cross_cutting" in reviewer_types, (
            "cross_cutting reviewer was dropped by fan-out cap - security review missing"
        )
        assert "tests_and_docs" in reviewer_types, (
            "tests_and_docs reviewer was dropped by fan-out cap - test-coverage review missing"
        )

    def test_cappable_set_excludes_feedback_verifier_style_sends(self) -> None:
        """Sends with no reviewer_type key (feedback-verifier style) must be
        classified as protected (not cappable) under the explicit cappable-set logic.

        This guards against the allow-list regression: if the protection logic
        were an allow-list of known protected types, a Send without any
        reviewer_type would become cap-eligible and could be dropped silently.
        The cappable set must be an explicit deny-list so unknown or future
        always-on reviewer types default to protected.
        """
        from argus.graph import _CAPPABLE_REVIEWER_TYPES
        from langgraph.types import Send

        # Feedback-verifier sends carry no reviewer_type key at all
        fv_send = Send("run_reviewer", {"diff": "x", "findings": []})
        assert fv_send.arg.get("reviewer_type") not in _CAPPABLE_REVIEWER_TYPES, (
            "A Send with no reviewer_type must not be in the cappable set - "
            "it would be silently dropped under the fan-out cap"
        )

        # Explicitly verify the known always-on types are also not cappable
        for always_on_type in ("cross_cutting", "tests_and_docs"):
            assert always_on_type not in _CAPPABLE_REVIEWER_TYPES, (
                f"{always_on_type} must not be cappable"
            )

        # And verify the plan-driven types ARE cappable
        for plan_driven_type in ("system", "specialist"):
            assert plan_driven_type in _CAPPABLE_REVIEWER_TYPES, (
                f"{plan_driven_type} must be cappable (plan-driven, potentially unbounded)"
            )


# ---------------------------------------------------------------------------
# _edge_fan_out_gap_fills: agent-count ceiling
# ---------------------------------------------------------------------------


class TestEdgeFanOutGapFillsCeiling:
    """When there are more coverage gaps than _MAX_REVIEWER_FANOUT, the
    gap-fill fan-out list must be truncated to exactly _MAX_REVIEWER_FANOUT."""

    def _make_gap_fill_state(self, n_gaps: int) -> Any:
        """Build a minimal ReviewState dict with n coverage gaps."""
        from argus.pipeline_models import CoverageGap, CoverageResult

        gaps = [
            CoverageGap(files=[f"file_{i}.py"], reason=f"not reviewed ({i})") for i in range(n_gaps)
        ]
        coverage = CoverageResult(is_covered=False, gaps=gaps)
        return {
            "coverage": coverage.model_dump(),
            "diff": "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new",
        }

    def test_gap_fill_truncated_to_cap(self) -> None:
        """With more gaps than _MAX_REVIEWER_FANOUT, the result is capped."""
        from argus.graph import _MAX_REVIEWER_FANOUT, _edge_fan_out_gap_fills

        state = self._make_gap_fill_state(60)
        result = _edge_fan_out_gap_fills(state)
        assert len(result) == _MAX_REVIEWER_FANOUT

    def test_gap_fill_not_truncated_when_under_cap(self) -> None:
        """With fewer gaps than the cap, all gap-fill Sends are returned."""
        from argus.graph import _MAX_REVIEWER_FANOUT, _edge_fan_out_gap_fills

        state = self._make_gap_fill_state(3)
        result = _edge_fan_out_gap_fills(state)
        assert len(result) == 3
        assert len(result) <= _MAX_REVIEWER_FANOUT
