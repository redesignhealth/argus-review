"""Unit tests for the reviewer bench config change guard (TECH-6282).

Covers:
- _is_bench_config_path matching and near-misses
- _detect_bench_config_routing_lines matching and exclusions
- Dynamic injection of ARGUS_BENCH_FILE
- Full-PR diff scoping across multi-round reviews (the two regression tests):
  (1) Round 2 with round-scoped diff that does NOT contain .argus/bench.toml
      but full-PR compare that DOES -> still BLOCKING.
  (2) Round 2 where .argus/bench.toml was removed from the PR -> full-PR compare
      no longer contains it -> APPROVE (no deadlock).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from argus.graph import (
    _detect_bench_config_routing_lines,
    _is_bench_config_path,
    _node_fetch_diff,
    run_review,
)
from argus.helpers import apply_bench_config_change_gate
from argus.models import (
    ReviewRequest,
    ReviewResponse,
    RiskLevel,
    Severity,
    TokenUsage,
    Verdict,
)

_GH_CLIENT_CLASS = "argus.github_client.GitHubClient"
_GRAPH_MODULE = "argus.graph"


def _make_response(
    verdict: Verdict = Verdict.APPROVE,
    risk_level: RiskLevel = RiskLevel.LOW,
    review_comment: str = "## Code Review\n\n**Verdict**: ✅ APPROVE | **Risk**: LOW\n\nAll clear.",
) -> ReviewResponse:
    return ReviewResponse(
        verdict=verdict,
        risk_level=risk_level,
        findings=[],
        coverage_map=[],
        review_comment=review_comment,
        usage=TokenUsage(input_tokens=10, output_tokens=10, cost_usd=0.001),
    )


# ---------------------------------------------------------------------------
# Path matching tests
# ---------------------------------------------------------------------------


class TestIsBenchConfigPath:
    """Tests for _is_bench_config_path."""

    def test_root_bench_toml(self) -> None:
        assert _is_bench_config_path(".argus/bench.toml") is True

    def test_nested_bench_toml(self) -> None:
        assert _is_bench_config_path("sub/.argus/bench.toml") is True
        assert _is_bench_config_path("packages/api/service/.argus/bench.toml") is True

    def test_packaged_default_exact(self) -> None:
        assert _is_bench_config_path("argus/bench_default.toml") is True

    def test_packaged_default_nested_suffix(self) -> None:
        assert _is_bench_config_path("packages/core/argus/bench_default.toml") is True

    def test_near_misses_do_not_match(self) -> None:
        assert _is_bench_config_path(".argus/README.md") is False
        assert _is_bench_config_path("bench.toml") is False
        assert _is_bench_config_path("docs/bench.toml") is False
        assert _is_bench_config_path("argus/bench.py") is False
        assert _is_bench_config_path("argus/bench.toml") is False
        assert _is_bench_config_path("other/.argus/config.toml") is False

    def test_dynamic_injection_dot_prefixed_bench_file(self) -> None:
        """Dot-prefixed dynamic bench paths like .argus/custom.toml must normalize correctly."""
        assert (
            _is_bench_config_path(
                ".argus/custom.toml",
                bench_file_setting=".argus/custom.toml",
            )
            is True
        )
        assert (
            _is_bench_config_path(
                ".argus/custom.toml",
                bench_file_setting="./.argus/custom.toml",
            )
            is True
        )
        assert (
            _is_bench_config_path(
                "sub/.argus/custom.toml",
                bench_file_setting=".argus/custom.toml",
            )
            is True
        )

    def test_dynamic_injection_relative_bench_file(self) -> None:
        assert (
            _is_bench_config_path(
                "custom/my_bench.toml",
                bench_file_setting="custom/my_bench.toml",
            )
            is True
        )
        assert (
            _is_bench_config_path(
                "other/file.py",
                bench_file_setting="custom/my_bench.toml",
            )
            is False
        )

    def test_dynamic_injection_absolute_bench_file_inside_worktree(self) -> None:
        assert (
            _is_bench_config_path(
                "configs/bench.toml",
                bench_file_setting="/tmp/wt/configs/bench.toml",
                worktree_path="/tmp/wt",
            )
            is True
        )

    def test_dynamic_injection_absolute_bench_file_outside_worktree(self) -> None:
        assert (
            _is_bench_config_path(
                "configs/bench.toml",
                bench_file_setting="/etc/other/bench.toml",
                worktree_path="/tmp/wt",
            )
            is False
        )


# ---------------------------------------------------------------------------
# Routing line detection tests
# ---------------------------------------------------------------------------


class TestDetectBenchConfigRoutingLines:
    """Tests for _detect_bench_config_routing_lines."""

    def test_added_bench_file_line(self) -> None:
        diff = "+export ARGUS_BENCH_FILE=configs/bench.toml\n"
        assert _detect_bench_config_routing_lines(diff) == [
            "export ARGUS_BENCH_FILE=configs/bench.toml"
        ]

    def test_added_no_bench_overrides_line(self) -> None:
        diff = "+ARGUS_NO_BENCH_OVERRIDES=1\n"
        assert _detect_bench_config_routing_lines(diff) == ["ARGUS_NO_BENCH_OVERRIDES=1"]

    def test_removed_routing_lines_do_not_match(self) -> None:
        diff = "-export ARGUS_NO_BENCH_OVERRIDES=1\n-ARGUS_BENCH_FILE=old_bench.toml\n"
        assert _detect_bench_config_routing_lines(diff) == []

    def test_normal_added_lines_do_not_match(self) -> None:
        diff = "+import os\n+def foo(): pass\n+result = True\n"
        assert _detect_bench_config_routing_lines(diff) == []

    def test_diff_header_lines_do_not_match(self) -> None:
        diff = "+++ b/.github/workflows/ARGUS_BENCH_FILE.yml\n"
        assert _detect_bench_config_routing_lines(diff) == []


# ---------------------------------------------------------------------------
# Multi-round regression tests (critical correctness properties)
# ---------------------------------------------------------------------------


class TestMultiRoundBenchConfigGuardRegressions:
    """Regression tests for full-PR diff scope vs round-scoped diffs (TECH-6282)."""

    @pytest.mark.asyncio
    async def test_round2_unmodified_bench_config_still_blocks(self) -> None:
        """Regression test 1:
        In round 1, .argus/bench.toml was added.
        In round 2, the author pushed a new commit touching only src/app.py.
        The round-scoped diff (prior..head) does NOT contain .argus/bench.toml.
        Evaluating against full-PR compare (base..head) MUST detect .argus/bench.toml
        and force verdict to BLOCKING.
        """
        req = ReviewRequest(repo="org/repo", pr_number=42)
        round_diff = (
            "diff --git a/src/app.py b/src/app.py\n"
            "--- a/src/app.py\n"
            "+++ b/src/app.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )

        mock_prior = MagicMock()
        mock_prior.reviewed_sha = "prior12345678"
        mock_prior.review_id = "rev-1"
        mock_prior.findings = []
        mock_prior.dismissed_findings = []
        mock_prior.model_dump.return_value = {"reviewed_sha": "prior12345678"}

        mock_gh = MagicMock()
        # Full-PR compare includes the bench config file added in round 1
        mock_gh.get_compare_files.return_value = (
            [
                {
                    "filename": ".argus/bench.toml",
                    "status": "added",
                    "patch": "+model = 'gemini'\n",
                },
                {"filename": "src/app.py", "status": "modified", "patch": "+new\n"},
            ],
            False,
        )

        state = {"request": req.model_dump()}
        config: dict[str, Any] = {"configurable": {}}

        with (
            patch(
                "argus.graph._fetch_prior_review", new_callable=AsyncMock, return_value=mock_prior
            ),
            patch("argus.graph._fetch_dismissed_findings", new_callable=AsyncMock, return_value=[]),
            patch(
                "argus.graph._fetch_pr_diff_and_description",
                new_callable=AsyncMock,
                return_value=(round_diff, "desc", "head12345678", "main"),
            ),
            patch(_GH_CLIENT_CLASS, return_value=mock_gh),
        ):
            node_result = await _node_fetch_diff(state, config)

        assert ".argus/bench.toml" in node_result["bench_config_changes"]

        # Now verify gate enforcement on this result
        response = _make_response(verdict=Verdict.APPROVE)
        fired = apply_bench_config_change_gate(response, node_result["bench_config_changes"])
        assert fired is True
        assert response.verdict == Verdict.BLOCKING
        assert response.risk_level == RiskLevel.HIGH
        assert len(response.findings) == 1
        assert response.findings[0].category == "argus-self-config"
        assert response.findings[0].severity == Severity.BLOCKING

    @pytest.mark.asyncio
    async def test_round2_removed_bench_config_allows_approve(self) -> None:
        """Regression test 2:
        In round 1, .argus/bench.toml was accidentally added (or flagged).
        In round 2, the author removed .argus/bench.toml from the PR.
        Between base_branch and head_sha, the file is no longer changed.
        Full-PR compare returns only src/app.py.
        The gate must NOT fire -> verdict remains APPROVE (no deadlock).
        """
        req = ReviewRequest(repo="org/repo", pr_number=42)
        # Round 2 diff might show the deletion of .argus/bench.toml
        round_diff = (
            "diff --git a/.argus/bench.toml b/.argus/bench.toml\n"
            "deleted file mode 100644\n"
            "--- a/.argus/bench.toml\n"
            "+++ /dev/null\n"
            "@@ -1 +0,0 @@\n"
            "-model = 'gemini'\n"
            "diff --git a/src/app.py b/src/app.py\n"
            "--- a/src/app.py\n"
            "+++ b/src/app.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )

        mock_prior = MagicMock()
        mock_prior.reviewed_sha = "prior12345678"
        mock_prior.review_id = "rev-1"
        mock_prior.findings = []
        mock_prior.dismissed_findings = []
        mock_prior.model_dump.return_value = {"reviewed_sha": "prior12345678"}

        mock_gh = MagicMock()
        # Full-PR compare from main..head has NO .argus/bench.toml
        mock_gh.get_compare_files.return_value = (
            [{"filename": "src/app.py", "status": "modified", "patch": "+new\n"}],
            False,
        )

        state = {"request": req.model_dump()}
        config: dict[str, Any] = {"configurable": {}}

        with (
            patch(
                "argus.graph._fetch_prior_review", new_callable=AsyncMock, return_value=mock_prior
            ),
            patch("argus.graph._fetch_dismissed_findings", new_callable=AsyncMock, return_value=[]),
            patch(
                "argus.graph._fetch_pr_diff_and_description",
                new_callable=AsyncMock,
                return_value=(round_diff, "desc", "head12345678", "main"),
            ),
            patch(_GH_CLIENT_CLASS, return_value=mock_gh),
        ):
            node_result = await _node_fetch_diff(state, config)

        assert node_result["bench_config_changes"] == []

        # Gate does NOT fire; verdict stays APPROVE
        response = _make_response(verdict=Verdict.APPROVE)
        fired = apply_bench_config_change_gate(response, node_result["bench_config_changes"])
        assert fired is False
        assert response.verdict == Verdict.APPROVE
        assert response.findings == []

    @pytest.mark.asyncio
    async def test_deleting_existing_bench_config_on_base_branch_does_not_block(self) -> None:
        """A PR that deletes an existing .argus/bench.toml present on the base branch
        must NOT trigger the blocking guard. Removing a bench override is a safe action."""
        req = ReviewRequest(repo="org/repo", pr_number=42)
        diff = (
            "diff --git a/.argus/bench.toml b/.argus/bench.toml\n"
            "deleted file mode 100644\n"
            "--- a/.argus/bench.toml\n"
            "+++ /dev/null\n"
            "@@ -1,5 +0,0 @@\n"
            "-model = 'gemini'\n"
        )
        mock_gh = MagicMock()
        mock_gh.get_compare_files.return_value = (
            [{"filename": ".argus/bench.toml", "status": "removed"}],
            False,
        )

        state = {"request": req.model_dump()}
        config: dict[str, Any] = {"configurable": {}}

        with (
            patch("argus.graph._fetch_prior_review", new_callable=AsyncMock, return_value=None),
            patch("argus.graph._fetch_dismissed_findings", new_callable=AsyncMock, return_value=[]),
            patch(
                "argus.graph._fetch_pr_diff_and_description",
                new_callable=AsyncMock,
                return_value=(diff, "desc", "head12345678", "main"),
            ),
            patch(_GH_CLIENT_CLASS, return_value=mock_gh),
        ):
            node_result = await _node_fetch_diff(state, config)

        assert node_result["bench_config_changes"] == []
        assert node_result["bench_config_unconfirmed"] is None

        response = _make_response(verdict=Verdict.APPROVE)
        fired = apply_bench_config_change_gate(
            response, node_result["bench_config_changes"], node_result["bench_config_unconfirmed"]
        )
        assert fired is False
        assert response.verdict == Verdict.APPROVE

    @pytest.mark.asyncio
    async def test_round2_routing_lines_in_full_pr_detected(self) -> None:
        """Round 2 diff only changes README.md, but full-PR changed files has
        .github/workflows/ci.yml with a patch adding ARGUS_BENCH_FILE -> must BLOCK."""
        req = ReviewRequest(repo="org/repo", pr_number=42)
        round_diff = (
            "diff --git a/README.md b/README.md\n"
            "--- a/README.md\n"
            "+++ b/README.md\n"
            "@@ -1 +1 @@\n"
            "-old doc\n"
            "+new doc\n"
        )
        mock_prior = MagicMock()
        mock_prior.reviewed_sha = "prior12345678"
        mock_prior.review_id = "rev-1"
        mock_prior.findings = []
        mock_prior.dismissed_findings = []
        mock_prior.model_dump.return_value = {"reviewed_sha": "prior12345678"}

        mock_gh = MagicMock()
        mock_gh.get_compare_files.return_value = (
            [
                {
                    "filename": ".github/workflows/ci.yml",
                    "status": "modified",
                    "patch": "@@ -10,3 +10,4 @@\n+export ARGUS_BENCH_FILE=configs/custom.toml\n",
                },
                {"filename": "README.md", "status": "modified", "patch": "+new doc\n"},
            ],
            False,
        )

        state = {"request": req.model_dump()}
        config: dict[str, Any] = {"configurable": {}}

        with (
            patch(
                "argus.graph._fetch_prior_review", new_callable=AsyncMock, return_value=mock_prior
            ),
            patch("argus.graph._fetch_dismissed_findings", new_callable=AsyncMock, return_value=[]),
            patch(
                "argus.graph._fetch_pr_diff_and_description",
                new_callable=AsyncMock,
                return_value=(round_diff, "desc", "head12345678", "main"),
            ),
            patch(_GH_CLIENT_CLASS, return_value=mock_gh),
        ):
            node_result = await _node_fetch_diff(state, config)

        assert any("ARGUS_BENCH_FILE" in c for c in node_result["bench_config_changes"])

        response = _make_response(verdict=Verdict.APPROVE)
        fired = apply_bench_config_change_gate(response, node_result["bench_config_changes"])
        assert fired is True
        assert response.verdict == Verdict.BLOCKING

    @pytest.mark.asyncio
    async def test_round1_diff_truncated_but_api_detects_bench_file(self) -> None:
        """In round 1, diff does NOT contain bench file (e.g. past 5000-line diff truncation),
        but GitHub API compare returns .argus/bench.toml -> must BLOCK."""
        req = ReviewRequest(repo="org/repo", pr_number=42)
        # diff truncated, only shows unrelated file
        diff = "diff --git a/src/huge_file.py b/src/huge_file.py\n+lots of lines\n"

        mock_gh = MagicMock()
        mock_gh.get_compare_files.return_value = (
            [
                {"filename": "src/huge_file.py", "status": "modified"},
                {"filename": ".argus/bench.toml", "status": "added"},
            ],
            False,
        )

        state = {"request": req.model_dump()}
        config: dict[str, Any] = {"configurable": {}}

        with (
            patch("argus.graph._fetch_prior_review", new_callable=AsyncMock, return_value=None),
            patch("argus.graph._fetch_dismissed_findings", new_callable=AsyncMock, return_value=[]),
            patch(
                "argus.graph._fetch_pr_diff_and_description",
                new_callable=AsyncMock,
                return_value=(diff, "desc", "head12345678", "main"),
            ),
            patch(_GH_CLIENT_CLASS, return_value=mock_gh),
        ):
            node_result = await _node_fetch_diff(state, config)

        assert ".argus/bench.toml" in node_result["bench_config_changes"]

        response = _make_response(verdict=Verdict.APPROVE)
        fired = apply_bench_config_change_gate(response, node_result["bench_config_changes"])
        assert fired is True
        assert response.verdict == Verdict.BLOCKING

    @pytest.mark.asyncio
    async def test_api_failure_fails_closed(self) -> None:
        """When GitHub comparison API fails, the guard fails CLOSED with an unconfirmed finding."""
        req = ReviewRequest(repo="org/repo", pr_number=42)
        diff = "diff --git a/src/app.py b/src/app.py\n+new code\n"

        mock_gh = MagicMock()
        mock_gh.get_compare_files.side_effect = RuntimeError("500 Internal Server Error")

        state = {"request": req.model_dump()}
        config: dict[str, Any] = {"configurable": {}}

        with (
            patch("argus.graph._fetch_prior_review", new_callable=AsyncMock, return_value=None),
            patch("argus.graph._fetch_dismissed_findings", new_callable=AsyncMock, return_value=[]),
            patch(
                "argus.graph._fetch_pr_diff_and_description",
                new_callable=AsyncMock,
                return_value=(diff, "desc", "head12345678", "main"),
            ),
            patch(_GH_CLIENT_CLASS, return_value=mock_gh),
        ):
            node_result = await _node_fetch_diff(state, config)

        assert node_result["bench_config_unconfirmed"] is not None

        response = _make_response(verdict=Verdict.APPROVE)
        fired = apply_bench_config_change_gate(
            response, node_result["bench_config_changes"], node_result["bench_config_unconfirmed"]
        )
        assert fired is True
        assert response.verdict == Verdict.BLOCKING
        assert len(response.findings) == 1
        assert response.findings[0].category == "argus-self-config"
        assert "Unable to verify" in response.findings[0].description
        assert "fails closed" in response.findings[0].description
        assert "Verdict forced to BLOCKING" in response.review_comment

    @pytest.mark.asyncio
    async def test_modified_file_with_none_patch_fails_closed(self) -> None:
        """A modified, non-removed file with patch=None in the API response must fail closed."""
        req = ReviewRequest(repo="org/repo", pr_number=42)
        diff = "diff --git a/src/app.py b/src/app.py\n"

        mock_gh = MagicMock()
        mock_gh.get_compare_files.return_value = (
            [{"filename": "src/app.py", "status": "modified", "patch": None}],
            False,
        )

        state = {"request": req.model_dump()}
        config: dict[str, Any] = {"configurable": {}}

        with (
            patch("argus.graph._fetch_prior_review", new_callable=AsyncMock, return_value=None),
            patch("argus.graph._fetch_dismissed_findings", new_callable=AsyncMock, return_value=[]),
            patch(
                "argus.graph._fetch_pr_diff_and_description",
                new_callable=AsyncMock,
                return_value=(diff, "desc", "head12345678", "main"),
            ),
            patch(_GH_CLIENT_CLASS, return_value=mock_gh),
        ):
            node_result = await _node_fetch_diff(state, config)

        assert node_result["bench_config_unconfirmed"] is not None
        assert "Diff patch content missing or empty" in node_result["bench_config_unconfirmed"]

        response = _make_response(verdict=Verdict.APPROVE)
        fired = apply_bench_config_change_gate(
            response, node_result["bench_config_changes"], node_result["bench_config_unconfirmed"]
        )
        assert fired is True
        assert response.verdict == Verdict.BLOCKING
        assert any("Unable to verify" in f.description for f in response.findings)

    @pytest.mark.asyncio
    async def test_modified_file_with_empty_patch_fails_closed(self) -> None:
        """A modified, non-removed file with patch='' in the API response must fail closed."""
        req = ReviewRequest(repo="org/repo", pr_number=42)
        diff = "diff --git a/src/app.py b/src/app.py\n"

        mock_gh = MagicMock()
        mock_gh.get_compare_files.return_value = (
            [{"filename": "src/app.py", "status": "modified", "patch": ""}],
            False,
        )

        state = {"request": req.model_dump()}
        config: dict[str, Any] = {"configurable": {}}

        with (
            patch("argus.graph._fetch_prior_review", new_callable=AsyncMock, return_value=None),
            patch("argus.graph._fetch_dismissed_findings", new_callable=AsyncMock, return_value=[]),
            patch(
                "argus.graph._fetch_pr_diff_and_description",
                new_callable=AsyncMock,
                return_value=(diff, "desc", "head12345678", "main"),
            ),
            patch(_GH_CLIENT_CLASS, return_value=mock_gh),
        ):
            node_result = await _node_fetch_diff(state, config)

        assert node_result["bench_config_unconfirmed"] is not None
        assert "Diff patch content missing or empty" in node_result["bench_config_unconfirmed"]

        response = _make_response(verdict=Verdict.APPROVE)
        fired = apply_bench_config_change_gate(
            response, node_result["bench_config_changes"], node_result["bench_config_unconfirmed"]
        )
        assert fired is True
        assert response.verdict == Verdict.BLOCKING

    @pytest.mark.asyncio
    async def test_removed_file_with_no_patch_does_not_fail_closed(self) -> None:
        """A file with status='removed' legitimately has no patch; must NOT fail closed."""
        req = ReviewRequest(repo="org/repo", pr_number=42)
        diff = "diff --git a/.argus/bench.toml b/.argus/bench.toml\ndeleted file\n"

        mock_gh = MagicMock()
        mock_gh.get_compare_files.return_value = (
            [{"filename": ".argus/bench.toml", "status": "removed", "patch": None}],
            False,
        )

        state = {"request": req.model_dump()}
        config: dict[str, Any] = {"configurable": {}}

        with (
            patch("argus.graph._fetch_prior_review", new_callable=AsyncMock, return_value=None),
            patch("argus.graph._fetch_dismissed_findings", new_callable=AsyncMock, return_value=[]),
            patch(
                "argus.graph._fetch_pr_diff_and_description",
                new_callable=AsyncMock,
                return_value=(diff, "desc", "head12345678", "main"),
            ),
            patch(_GH_CLIENT_CLASS, return_value=mock_gh),
        ):
            node_result = await _node_fetch_diff(state, config)

        assert node_result["bench_config_unconfirmed"] is None
        assert node_result["bench_config_changes"] == []

        response = _make_response(verdict=Verdict.APPROVE)
        fired = apply_bench_config_change_gate(
            response, node_result["bench_config_changes"], node_result["bench_config_unconfirmed"]
        )
        assert fired is False
        assert response.verdict == Verdict.APPROVE


class TestRunReviewBenchGuardIntegration:
    """Integration test for run_review executing the bench config change gate."""

    @pytest.mark.asyncio
    async def test_run_review_forces_blocking_when_bench_config_changed(self) -> None:
        req = ReviewRequest(repo="org/repo", pr_number=42)
        mock_response = _make_response(verdict=Verdict.APPROVE)

        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "response": mock_response.model_dump(),
            "findings": [],
            "bench_config_changes": [".argus/bench.toml"],
        }

        mock_session = AsyncMock()
        mock_execute_result = MagicMock()
        mock_execute_result.scalar_one_or_none.return_value = "code-review-1"
        mock_session.execute = AsyncMock(return_value=mock_execute_result)
        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_session_factory = MagicMock(return_value=mock_session_ctx)

        mock_gh = MagicMock()
        mock_gh.get_pull_request.return_value = {"head_sha": "abc123def456"}

        mock_worktree_ctx = MagicMock()
        mock_worktree_ctx.__aenter__ = AsyncMock(return_value="/tmp/worktree")
        mock_worktree_ctx.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(f"{_GRAPH_MODULE}.build_pipeline") as mock_build,
            patch(f"{_GRAPH_MODULE}.validate_history_backend_connectivity", new_callable=AsyncMock),
            patch(
                "argus.storage.resolver.get_async_session_factory",
                return_value=mock_session_factory,
            ),
            patch(_GH_CLIENT_CLASS, return_value=mock_gh),
            patch(f"{_GRAPH_MODULE}.provisioned_worktree", return_value=mock_worktree_ctx),
        ):
            mock_build.return_value.__aenter__ = AsyncMock(return_value=mock_graph)
            mock_build.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await run_review(req, flow_run_id="flow-bench-test")

        assert result.verdict == Verdict.BLOCKING
        assert result.risk_level == RiskLevel.HIGH
        assert any(f.category == "argus-self-config" for f in result.findings)
        assert "Verdict forced to BLOCKING" in result.review_comment

    @pytest.mark.asyncio
    async def test_run_review_forces_blocking_when_detection_unconfirmed(self) -> None:
        req = ReviewRequest(repo="org/repo", pr_number=42)
        mock_response = _make_response(verdict=Verdict.APPROVE)

        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "response": mock_response.model_dump(),
            "findings": [],
            "bench_config_changes": [],
            "bench_config_unconfirmed": "GitHub API comparison failed: 500 error",
        }

        mock_session = AsyncMock()
        mock_execute_result = MagicMock()
        mock_execute_result.scalar_one_or_none.return_value = "code-review-1"
        mock_session.execute = AsyncMock(return_value=mock_execute_result)
        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_session_factory = MagicMock(return_value=mock_session_ctx)

        mock_gh = MagicMock()
        mock_gh.get_pull_request.return_value = {"head_sha": "abc123def456"}

        mock_worktree_ctx = MagicMock()
        mock_worktree_ctx.__aenter__ = AsyncMock(return_value="/tmp/worktree")
        mock_worktree_ctx.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(f"{_GRAPH_MODULE}.build_pipeline") as mock_build,
            patch(f"{_GRAPH_MODULE}.validate_history_backend_connectivity", new_callable=AsyncMock),
            patch(
                "argus.storage.resolver.get_async_session_factory",
                return_value=mock_session_factory,
            ),
            patch(_GH_CLIENT_CLASS, return_value=mock_gh),
            patch(f"{_GRAPH_MODULE}.provisioned_worktree", return_value=mock_worktree_ctx),
        ):
            mock_build.return_value.__aenter__ = AsyncMock(return_value=mock_graph)
            mock_build.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await run_review(req, flow_run_id="flow-unconfirmed-test")

        assert result.verdict == Verdict.BLOCKING
        assert result.risk_level == RiskLevel.HIGH
        assert any(f.category == "argus-self-config" for f in result.findings)
        assert any("Unable to verify" in f.description for f in result.findings)
        assert "Verdict forced to BLOCKING" in result.review_comment
