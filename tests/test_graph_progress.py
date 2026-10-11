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
# What a successful pipeline always leaves in graph state: the pinned head and a receipt.
_SCOPED_STATE = {
    "head_sha": _FAKE_SHA,
    "scope": {
        "base_sha": "b" * 40,
        "head_sha": _FAKE_SHA,
        "behaviors": [{"title": "t", "files": ["x.py"], "depends_on": []}],
        "rationale": "one behavior",
    },
}


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
            **_SCOPED_STATE,
        }

        mock_session = AsyncMock()
        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_session_factory = MagicMock(return_value=mock_session_ctx)

        with (
            patch(f"{_GRAPH_MODULE}.build_pipeline") as mock_build,
            patch(
                f"{_GRAPH_MODULE}.validate_history_backend_connectivity",
                new_callable=AsyncMock,
            ),
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


class TestRunReviewConnectivityPreflight:
    """run_review's connectivity check is stated as a propagation contract
    in a comment (HistoryBackendConnectivityError should reach the caller
    unmodified) -- every other test in this file patches it to a no-op, so
    nothing actually proves that contract holds without this test."""

    @pytest.mark.asyncio
    async def test_connectivity_error_propagates_out_of_run_review(self) -> None:
        from argus.graph import run_review
        from argus.models import ReviewRequest
        from argus.storage.resolver import HistoryBackendConnectivityError

        request = ReviewRequest(repo="org/repo", pr_number=42)

        with patch(
            f"{_GRAPH_MODULE}.validate_history_backend_connectivity",
            new_callable=AsyncMock,
            side_effect=HistoryBackendConnectivityError("bad db"),
        ):
            with pytest.raises(HistoryBackendConnectivityError, match="bad db"):
                await run_review(request, flow_run_id=None)


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
            **_SCOPED_STATE,
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
                f"{_GRAPH_MODULE}.validate_history_backend_connectivity",
                new_callable=AsyncMock,
            ),
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
                f"{_GRAPH_MODULE}.validate_history_backend_connectivity",
                new_callable=AsyncMock,
            ),
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
            **_SCOPED_STATE,
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
                f"{_GRAPH_MODULE}.validate_history_backend_connectivity",
                new_callable=AsyncMock,
            ),
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
                f"{_GRAPH_MODULE}.validate_history_backend_connectivity",
                new_callable=AsyncMock,
            ),
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
                f"{_GRAPH_MODULE}.validate_history_backend_connectivity",
                new_callable=AsyncMock,
            ),
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
                f"{_GRAPH_MODULE}.validate_history_backend_connectivity",
                new_callable=AsyncMock,
            ),
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
# _looks_like_missing_schema_object
# ---------------------------------------------------------------------------


class TestLooksLikeMissingSchemaObject:
    def test_postgres_missing_column(self) -> None:
        from argus.graph import _looks_like_missing_schema_object

        exc = Exception('column "failure_reason" of relation "agent_runs" does not exist')
        assert _looks_like_missing_schema_object(exc) is True

    def test_postgres_missing_relation(self) -> None:
        """Strictly worse than a missing column (schema/015 itself was never
        applied) and lacks the word "column" entirely -- must still match."""
        from argus.graph import _looks_like_missing_schema_object

        exc = Exception('relation "review_service.agent_runs" does not exist')
        assert _looks_like_missing_schema_object(exc) is True

    def test_sqlite_missing_column_on_insert(self) -> None:
        """SQLite's actual INSERT-path phrasing is "has no column named",
        NOT "no such column" (that's the SELECT/WHERE-path phrasing)."""
        from argus.graph import _looks_like_missing_schema_object

        exc = Exception("table agent_runs has no column named failure_reason")
        assert _looks_like_missing_schema_object(exc) is True

    def test_sqlite_missing_column_on_select(self) -> None:
        from argus.graph import _looks_like_missing_schema_object

        exc = Exception("no such column: failure_reason")
        assert _looks_like_missing_schema_object(exc) is True

    def test_unrelated_error_does_not_match(self) -> None:
        from argus.graph import _looks_like_missing_schema_object

        exc = Exception("database is locked")
        assert _looks_like_missing_schema_object(exc) is False

    def test_checks_cause_when_message_lives_there(self) -> None:
        """With asyncpg, the real Postgres message sometimes lives on
        __cause__ rather than inline in str(exc)."""
        from argus.graph import _looks_like_missing_schema_object

        cause = Exception('column "failure_reason" does not exist')
        exc = Exception("generic wrapper message")
        exc.__cause__ = cause
        assert _looks_like_missing_schema_object(exc) is True

    def test_checks_cause_for_missing_relation_too(self) -> None:
        """The __cause__ path isn't specific to the missing-column shape --
        a missing-table message living on __cause__ must match too."""
        from argus.graph import _looks_like_missing_schema_object

        cause = Exception('relation "review_service.agent_runs" does not exist')
        exc = Exception("asyncpg wrapper")
        exc.__cause__ = cause
        assert _looks_like_missing_schema_object(exc) is True

    def test_permission_denied_does_not_match_missing_relation(self) -> None:
        """'relation' appears in plenty of non-migration errors too --
        without the 'does not exist' guard, a permission error would
        misleadingly route through the migration-alert path."""
        from argus.graph import _looks_like_missing_schema_object

        exc = Exception("permission denied for relation agent_runs")
        assert _looks_like_missing_schema_object(exc) is False

    def test_unrelated_cause_does_not_match(self) -> None:
        """A __cause__ that itself doesn't match must not produce a false
        positive just because __cause__ traversal happened."""
        from argus.graph import _looks_like_missing_schema_object

        exc = Exception("insert failed")
        exc.__cause__ = Exception("connection timeout")
        assert _looks_like_missing_schema_object(exc) is False

    def test_split_tokens_across_message_and_cause_do_not_false_positive(self) -> None:
        """Regression guard: str(exc) and str(exc.__cause__) must be checked
        independently, not joined into one string -- joining would let
        'column' in one and 'does not exist' in the other combine into a
        false positive even though neither message alone matches."""
        from argus.graph import _looks_like_missing_schema_object

        exc = Exception("something mentions a column")
        exc.__cause__ = Exception("an unrelated timeout does not exist... in this sentence")
        assert _looks_like_missing_schema_object(exc) is False


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
            **_SCOPED_STATE,
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
                f"{_GRAPH_MODULE}.validate_history_backend_connectivity",
                new_callable=AsyncMock,
            ),
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
                f"{_GRAPH_MODULE}.validate_history_backend_connectivity",
                new_callable=AsyncMock,
            ),
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

    @pytest.mark.asyncio
    async def test_agent_runs_missing_column_logs_error_with_count(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A schema-mismatch error (e.g. an unapplied migration) must log at
        ERROR with the dropped-row count, not the generic non-fatal WARNING
        -- silently losing every agent_runs row for a whole deploy window is
        not a one-off transient blip."""
        from sqlalchemy.exc import ProgrammingError

        from argus.graph import run_review
        from argus.models import ReviewRequest, Verdict

        request = ReviewRequest(repo="org/repo", pr_number=99)
        mock_graph, mock_session_factory, mock_core_session, mock_agent_session = self._make_mocks()

        mock_agent_session.execute.side_effect = ProgrammingError(
            "INSERT INTO review_service.agent_runs ...",
            {},
            Exception('column "failure_reason" does not exist'),
        )

        with (
            patch(f"{_GRAPH_MODULE}.build_pipeline") as mock_build,
            patch(
                f"{_GRAPH_MODULE}.validate_history_backend_connectivity",
                new_callable=AsyncMock,
            ),
            patch(
                "argus.storage.resolver.get_async_session_factory",
                return_value=mock_session_factory,
            ),
            patch(_GH_CLIENT_CLASS, return_value=_mock_gh_client()),
            patch(_PROVISIONED_WORKTREE, return_value=_mock_provisioned_worktree()),
            caplog.at_level("WARNING", logger="argus.graph"),
        ):
            mock_build.return_value.__aenter__ = AsyncMock(return_value=mock_graph)
            mock_build.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await run_review(request, flow_run_id="flow-schema-error")

        assert result.verdict == Verdict.APPROVE  # error is swallowed, not propagated
        error_records = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(error_records) == 1
        assert "1 agent_runs" in error_records[0].message
        assert "schema/*.sql migration" in error_records[0].message

    @pytest.mark.asyncio
    async def test_agent_runs_unrelated_db_error_logs_warning_not_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A ProgrammingError NOT shaped like a missing column (e.g. a bad
        bind-parameter count) must stay at the generic non-fatal WARNING --
        the schema-error ERROR message would be misleading for an unrelated
        SQL bug."""
        from sqlalchemy.exc import ProgrammingError

        from argus.graph import run_review
        from argus.models import ReviewRequest, Verdict

        request = ReviewRequest(repo="org/repo", pr_number=99)
        mock_graph, mock_session_factory, mock_core_session, mock_agent_session = self._make_mocks()

        mock_agent_session.execute.side_effect = ProgrammingError(
            "INSERT INTO review_service.agent_runs ...",
            {},
            Exception("syntax error at or near ..."),
        )

        with (
            patch(f"{_GRAPH_MODULE}.build_pipeline") as mock_build,
            patch(
                f"{_GRAPH_MODULE}.validate_history_backend_connectivity",
                new_callable=AsyncMock,
            ),
            patch(
                "argus.storage.resolver.get_async_session_factory",
                return_value=mock_session_factory,
            ),
            patch(_GH_CLIENT_CLASS, return_value=_mock_gh_client()),
            patch(_PROVISIONED_WORKTREE, return_value=_mock_provisioned_worktree()),
            caplog.at_level("WARNING", logger="argus.graph"),
        ):
            mock_build.return_value.__aenter__ = AsyncMock(return_value=mock_graph)
            mock_build.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await run_review(request, flow_run_id="flow-unrelated-error")

        assert result.verdict == Verdict.APPROVE
        assert not [r for r in caplog.records if r.levelname == "ERROR"]
        warning_records = [
            r for r in caplog.records if r.levelname == "WARNING" and "agent_runs" in r.message
        ]
        assert len(warning_records) == 1
        # Pin the actual formatted content, not just that a WARNING fired --
        # a format bug (wrong arg count, swapped args) would still pass a
        # bare "one WARNING record exists" assertion.
        assert "1 agent_runs" in warning_records[0].message
        assert "ProgrammingError" in warning_records[0].message


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
            **_SCOPED_STATE,
        }

        mock_session = AsyncMock()
        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_session_factory = MagicMock(return_value=mock_session_ctx)

        with (
            patch(f"{_GRAPH_MODULE}.build_pipeline") as mock_build,
            patch(
                f"{_GRAPH_MODULE}.validate_history_backend_connectivity",
                new_callable=AsyncMock,
            ),
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
# Dispatch budget / gap-fill cap: stop, never truncate
# ---------------------------------------------------------------------------

_HEAD = "c" * 40


def _plan_with(n_groups: int, specialists: int = 0) -> Any:
    from argus.pipeline_models import FileEntry, ReviewPlan, SystemGroup

    return ReviewPlan(
        system_groups=[
            SystemGroup(
                name=f"group-{i}",
                files=[f"file_{i}.py"],
                conventions="",
                review_focus="check it",
                specialists_needed=["security"] * specialists,
            )
            for i in range(n_groups)
        ],
        cross_cutting_concerns=[],
        file_manifest=[
            FileEntry(path=f"file_{i}.py", change_type="modified") for i in range(n_groups)
        ],
    )


def _plan_state() -> Any:
    from argus.models import ReviewRequest

    return {
        "diff": "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new",
        "description": "d",
        "head_sha": _HEAD,
        "request": ReviewRequest(repo="org/repo", pr_number=1).model_dump(),
        "prior_review": {},
        "scope": {"base_sha": "b" * 40, "head_sha": _HEAD, "behaviors": [], "rationale": ""},
    }


async def _run_plan(plan: Any, *, budget: int = 16, override: bool = False) -> tuple[Any, Any]:
    """Run _node_plan then the fan-out edge exactly as the graph would."""
    from types import SimpleNamespace

    from argus.graph import _edge_fan_out_reviewers, _node_plan

    config: Any = {
        "configurable": {
            "review_override_sha": _HEAD if override else None,
            "review_override_reason": "approved by human" if override else None,
        }
    }
    with (
        patch(f"{_GRAPH_MODULE}.plan_review", AsyncMock(return_value=plan)),
        patch(
            f"{_GRAPH_MODULE}.get_settings",
            return_value=SimpleNamespace(ARGUS_REVIEWER_DISPATCH_BUDGET=budget),
        ),
    ):
        update = await _node_plan(_plan_state(), config)
    merged: Any = {**_plan_state(), **update}
    return update, _edge_fan_out_reviewers(merged)


class TestPlanNodeDispatchHalt:
    @pytest.mark.asyncio
    async def test_over_budget_halts_and_emits_no_reviewer(self) -> None:
        update, routed = await _run_plan(_plan_with(15))  # 15 + 2 = 17 > 16
        halt = update["response"]["halt"]
        assert (halt["code"], halt["dispatch_count"], halt["dispatch_budget"]) == (
            "dispatch_budget_exceeded",
            17,
            16,
        )
        assert halt["head_sha"] == _HEAD
        assert update["response"]["verdict"] == "BLOCKING"
        assert update["response"]["risk_level"] == "HIGH"
        assert update["response"]["findings"] == []
        assert routed == "__end__"

    @pytest.mark.asyncio
    async def test_at_budget_dispatches_everything(self) -> None:
        update, routed = await _run_plan(_plan_with(14))  # 16 == budget
        assert "response" not in update
        assert len(routed) == 16

    @pytest.mark.asyncio
    async def test_specialists_count_toward_budget(self) -> None:
        update, _ = await _run_plan(_plan_with(5, specialists=2))  # 15 + 2 = 17
        assert update["response"]["halt"]["dispatch_count"] == 17

    @pytest.mark.asyncio
    async def test_configured_budget_is_used(self) -> None:
        update, _ = await _run_plan(_plan_with(2), budget=3)  # 4 > 3
        assert update["response"]["halt"]["dispatch_budget"] == 3

    @pytest.mark.asyncio
    async def test_override_allows_up_to_hard_ceiling_without_truncation(self) -> None:
        update, routed = await _run_plan(_plan_with(48), override=True)  # exactly 50
        assert "response" not in update
        assert len(routed) == 50
        assert update["scope"]["override_reason"] == "approved by human"

    @pytest.mark.asyncio
    async def test_over_hard_ceiling_halts_even_with_override(self) -> None:
        for n_groups in (49, 60):
            update, routed = await _run_plan(_plan_with(n_groups), override=True)
            assert update["response"]["halt"]["code"] == "dispatch_budget_exceeded"
            assert routed == "__end__"

    @pytest.mark.asyncio
    async def test_override_pinned_to_other_head_is_ignored(self) -> None:
        from argus.graph import _node_plan

        config: Any = {
            "configurable": {"review_override_sha": "d" * 40, "review_override_reason": "stale"}
        }
        with patch(f"{_GRAPH_MODULE}.plan_review", AsyncMock(return_value=_plan_with(20))):
            update = await _node_plan(_plan_state(), config)
        assert update["response"]["halt"]["code"] == "dispatch_budget_exceeded"


class TestCoverageGapHalt:
    async def _coverage(self, n_gaps: int) -> tuple[Any, Any]:
        from argus.graph import _edge_coverage_decision, _node_check_coverage
        from argus.pipeline_models import CoverageGap, CoverageResult

        coverage = CoverageResult(
            is_covered=False,
            gaps=[CoverageGap(files=[f"f{i}.py"], reason="missed") for i in range(n_gaps)],
        )
        state = {**_plan_state(), "plan": _plan_with(1).model_dump(), "findings": []}
        with patch(f"{_GRAPH_MODULE}.check_coverage", AsyncMock(return_value=coverage)):
            update = await _node_check_coverage(state)
        merged: Any = {**state, **update}
        return update, _edge_coverage_decision(merged)

    @pytest.mark.asyncio
    async def test_over_50_gaps_halts_before_dispatch(self) -> None:
        update, route = await self._coverage(51)
        halt = update["response"]["halt"]
        assert (halt["code"], halt["dispatch_count"], halt["dispatch_budget"]) == (
            "gap_fill_cap_exceeded",
            51,
            50,
        )
        assert route == "__end__"

    @pytest.mark.asyncio
    async def test_exactly_50_gaps_dispatches_all(self) -> None:
        from argus.graph import _edge_fan_out_gap_fills

        update, route = await self._coverage(50)
        assert "response" not in update
        assert route == "fill_gaps"
        sends = _edge_fan_out_gap_fills({"coverage": update["coverage"], "diff": "d"})
        assert len(sends) == 50
