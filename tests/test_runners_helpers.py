"""Unit tests for pure helper functions in helpers.py.

These are security controls and parsing logic with no LLM dependency.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from argus.helpers import (
    _RISK_LEVEL_ORDER,
    append_degraded_coverage_section,
    apply_bench_config_change_gate,
    apply_precheck_scanner_failure_gate,
    build_degraded_coverage_labels,
    collect_reviewed_files,
    extract_changed_files,
    failed_reviewer_labels,
    filter_diff_for_files,
    parse_review_result,
    sanitize_file_paths,
)
from argus.models import ReviewResponse, RiskLevel, Severity, Verdict
from argus.pipeline_models import RawFinding, SystemReviewResult


def _response(
    verdict: Verdict = Verdict.APPROVE,
    risk_level: RiskLevel = RiskLevel.LOW,
    review_comment: str = "## Code Review\n\n**Verdict**: ✅ APPROVE | **Risk**: LOW\n\nLooks good.",
) -> ReviewResponse:
    return ReviewResponse(
        verdict=verdict,
        risk_level=risk_level,
        review_comment=review_comment,
    )


# ---------------------------------------------------------------------------
# filter_diff_for_files
# ---------------------------------------------------------------------------


class TestFilterDiffForFiles:
    SAMPLE_DIFF = (
        "diff --git a/src/main.py b/src/main.py\n"
        "--- a/src/main.py\n"
        "+++ b/src/main.py\n"
        "@@ -1,3 +1,4 @@\n"
        "+import os\n"
        " def main():\n"
        "     pass\n"
        "diff --git a/src/utils.py b/src/utils.py\n"
        "--- a/src/utils.py\n"
        "+++ b/src/utils.py\n"
        "@@ -1 +1,2 @@\n"
        "+# new comment\n"
        " def helper():\n"
        "diff --git a/tests/test_main.py b/tests/test_main.py\n"
        "--- a/tests/test_main.py\n"
        "+++ b/tests/test_main.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )

    def test_filters_to_matching_files(self) -> None:
        result = filter_diff_for_files(self.SAMPLE_DIFF, ["src/main.py"])
        assert "src/main.py" in result
        assert "src/utils.py" not in result
        assert "tests/test_main.py" not in result

    def test_multiple_files(self) -> None:
        result = filter_diff_for_files(self.SAMPLE_DIFF, ["src/main.py", "tests/test_main.py"])
        assert "src/main.py" in result
        assert "tests/test_main.py" in result
        assert "src/utils.py" not in result

    def test_no_matching_files(self) -> None:
        result = filter_diff_for_files(self.SAMPLE_DIFF, ["nonexistent.py"])
        assert result == ""

    def test_empty_diff(self) -> None:
        assert filter_diff_for_files("", ["src/main.py"]) == ""

    def test_empty_files(self) -> None:
        assert filter_diff_for_files(self.SAMPLE_DIFF, []) == ""


# ---------------------------------------------------------------------------
# extract_changed_files
# ---------------------------------------------------------------------------


class TestExtractChangedFiles:
    def test_extracts_both_sides_of_a_rename(self) -> None:
        diff = (
            "diff --git a/old_name.py b/new_name.py\n"
            "--- a/old_name.py\n"
            "+++ b/new_name.py\n"
            "@@ -1 +1 @@\n"
            "-x\n"
            "+y\n"
        )
        assert extract_changed_files(diff) == ["new_name.py", "old_name.py"]

    def test_dedupes_and_sorts(self) -> None:
        result = extract_changed_files(TestFilterDiffForFiles.SAMPLE_DIFF)
        assert result == ["src/main.py", "src/utils.py", "tests/test_main.py"]

    def test_empty_diff_returns_empty_list(self) -> None:
        assert extract_changed_files("") == []

    def test_diff_with_no_file_headers_returns_empty_list(self) -> None:
        assert extract_changed_files("not a real diff\njust text\n") == []


# ---------------------------------------------------------------------------
# parse_review_result
# ---------------------------------------------------------------------------


class TestParseReviewResult:
    def test_json_in_code_block(self) -> None:
        raw = '```json\n{"system_group": "test", "findings": [{"file": "a.py", "line": 10, "description": "bug"}], "files_explored": ["a.py"]}\n```'
        result = parse_review_result(raw, "fallback-name")
        assert result.system_group == "test"
        assert len(result.findings) == 1
        assert result.findings[0].file == "a.py"
        assert result.findings[0].line == 10

    def test_raw_json(self) -> None:
        raw = '{"system_group": "raw", "findings": [], "files_explored": ["b.py"]}'
        result = parse_review_result(raw, "fallback")
        assert result.system_group == "raw"
        assert result.files_explored == ["b.py"]

    def test_malformed_json_fallback(self) -> None:
        raw = "This is not JSON at all, just a text finding."
        result = parse_review_result(raw, "my-group")
        assert result.system_group == "my-group"
        assert len(result.findings) == 1
        assert "not JSON" in result.findings[0].description

    def test_empty_input(self) -> None:
        result = parse_review_result("", "empty")
        assert result.system_group == "empty"
        assert result.findings == []

    def test_whitespace_only(self) -> None:
        result = parse_review_result("   \n\n  ", "ws")
        assert result.findings == []

    def test_missing_keys(self) -> None:
        raw = '{"findings": [{"description": "no file or line"}]}'
        result = parse_review_result(raw, "partial")
        assert len(result.findings) == 1
        assert result.findings[0].file is None
        assert result.findings[0].line is None

    def test_string_line_number(self) -> None:
        raw = '{"findings": [{"file": "a.py", "line": "52-55", "description": "range"}]}'
        result = parse_review_result(raw, "range")
        assert result.findings[0].line == "52-55"

    def test_numeric_string_line_coerced(self) -> None:
        raw = '{"findings": [{"file": "a.py", "line": "42", "description": "coerce"}]}'
        result = parse_review_result(raw, "coerce")
        assert result.findings[0].line == 42


# ---------------------------------------------------------------------------
# sanitize_file_paths
# ---------------------------------------------------------------------------


class TestSanitizeFilePaths:
    def test_normal_paths_pass_through(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            Path(root, "src").mkdir()
            Path(root, "src/main.py").touch()
            result = sanitize_file_paths(["src/main.py"], root)
            assert result == ["src/main.py"]

    def test_path_traversal_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            result = sanitize_file_paths(["../../../etc/passwd"], root)
            assert result == []

    def test_absolute_path_stripped_to_relative(self) -> None:
        """Absolute paths have leading / stripped — they become repo-relative."""
        with tempfile.TemporaryDirectory() as root:
            result = sanitize_file_paths(["/etc/passwd"], root)
            assert result == ["etc/passwd"]  # Leading / stripped, stays in repo

    def test_leading_slash_stripped(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            Path(root, "src").mkdir()
            Path(root, "src/file.py").touch()
            result = sanitize_file_paths(["/src/file.py"], root)
            assert result == ["src/file.py"]

    def test_mixed_paths(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            Path(root, "good.py").touch()
            result = sanitize_file_paths(
                ["good.py", "../../evil.py"],
                root,
            )
            assert result == ["good.py"]


# ---------------------------------------------------------------------------
# collect_reviewed_files
# ---------------------------------------------------------------------------


class TestCollectReviewedFiles:
    def test_from_findings(self) -> None:
        results = [
            SystemReviewResult(
                system_group="test",
                findings=[RawFinding(file="a.py", line=1, description="x")],
                files_explored=[],
            )
        ]
        assert collect_reviewed_files(results) == {"a.py"}

    def test_from_files_explored(self) -> None:
        results = [
            SystemReviewResult(
                system_group="test",
                findings=[],
                files_explored=["b.py", "c.py"],
            )
        ]
        assert collect_reviewed_files(results) == {"b.py", "c.py"}

    def test_combined(self) -> None:
        results = [
            SystemReviewResult(
                system_group="g1",
                findings=[RawFinding(file="a.py", line=1, description="x")],
                files_explored=["b.py"],
            ),
            SystemReviewResult(
                system_group="g2",
                findings=[RawFinding(file="c.py", line=2, description="y")],
                files_explored=["a.py"],
            ),
        ]
        assert collect_reviewed_files(results) == {"a.py", "b.py", "c.py"}

    def test_empty(self) -> None:
        assert collect_reviewed_files([]) == set()

    def test_none_file_ignored(self) -> None:
        results = [
            SystemReviewResult(
                system_group="test",
                findings=[RawFinding(file=None, line=None, description="general")],
                files_explored=[],
            )
        ]
        assert collect_reviewed_files(results) == set()


# ---------------------------------------------------------------------------
# failed_reviewer_labels / append_degraded_coverage_section
# ---------------------------------------------------------------------------


class TestTimedOutReviewerLabels:
    def test_no_failures_returns_empty(self) -> None:
        results = [
            SystemReviewResult(system_group="g1", findings=[], files_explored=[]),
            SystemReviewResult(system_group="g2", findings=[], files_explored=[]),
        ]
        assert failed_reviewer_labels(results) == []

    def test_collects_only_failed_labels_in_order(self) -> None:
        results = [
            SystemReviewResult(system_group="g1", findings=[], files_explored=[]),
            SystemReviewResult(
                system_group="specialist/orchestration::g2",
                findings=[],
                files_explored=[],
                failure_reason="timeout",
            ),
            SystemReviewResult(system_group="g3", findings=[], files_explored=[]),
            SystemReviewResult(
                system_group="g4",
                findings=[],
                files_explored=[],
                failure_reason="worker_crashed",
            ),
        ]
        assert failed_reviewer_labels(results) == [
            ("specialist/orchestration::g2", "timeout"),
            ("g4", "worker_crashed"),
        ]

    def test_empty_results(self) -> None:
        assert failed_reviewer_labels([]) == []


class TestBuildDegradedCoverageLabels:
    """Regression coverage for the exact key lookup
    (``graph_result.get("precheck_scanner_failures", [])``) that a typo on
    either the producer side (``graph._node_precheck_rules``) or this read
    side would otherwise let slip past the full test suite undetected --
    see this function's own docstring for why it was pulled out of
    ``graph.run_review`` specifically to make this testable in isolation.
    """

    def test_no_failures_of_either_kind_returns_empty(self) -> None:
        results = [SystemReviewResult(system_group="g1", findings=[], files_explored=[])]
        assert build_degraded_coverage_labels(results, {}) == []

    def test_missing_key_treated_as_no_precheck_failures(self) -> None:
        results = [SystemReviewResult(system_group="g1", findings=[], files_explored=[])]
        assert build_degraded_coverage_labels(results, {"unrelated": "value"}) == []

    def test_precheck_scanner_failures_become_labeled_and_reasoned(self) -> None:
        results: list[SystemReviewResult] = []
        graph_result = {"precheck_scanner_failures": ["zizmor", "trivy"]}
        assert build_degraded_coverage_labels(results, graph_result) == [
            ("precheck:zizmor", "scanner did not complete this round"),
            ("precheck:trivy", "scanner did not complete this round"),
        ]

    def test_reviewer_and_precheck_failures_combine_reviewer_first(self) -> None:
        results = [
            SystemReviewResult(
                system_group="specialist/orchestration::g2",
                findings=[],
                files_explored=[],
                failure_reason="timeout",
            ),
        ]
        graph_result = {"precheck_scanner_failures": ["zizmor"]}
        assert build_degraded_coverage_labels(results, graph_result) == [
            ("specialist/orchestration::g2", "timeout"),
            ("precheck:zizmor", "scanner did not complete this round"),
        ]

    def test_missing_scanners_get_a_distinct_reason_from_crashed_scanners(self) -> None:
        """A never-installed scanner (precheck_missing_scanners) must not
        collapse into the same reason text as a crashed one
        (precheck_scanner_failures) -- the reader needs to tell "install
        it" apart from "investigate a failure"."""
        results: list[SystemReviewResult] = []
        graph_result = {
            "precheck_scanner_failures": ["zizmor"],
            "precheck_missing_scanners": ["trivy"],
        }
        assert build_degraded_coverage_labels(results, graph_result) == [
            ("precheck:zizmor", "scanner did not complete this round"),
            ("precheck:trivy", "scanner not installed"),
        ]


class TestApplyPrecheckScannerFailureGate:
    """ARGUS_PRECHECK_BLOCK_ON_SCANNER_FAILURE is opt-in and off by default
    -- see its docstring in argus/config.py for the fail-open-vs-fail-closed
    tradeoff. This gate must only ever make the verdict stricter, never
    looser, and must be a true no-op (no mutation) in every case where it
    doesn't fire.
    """

    def test_noop_when_flag_off(self) -> None:
        response = _response()
        original_comment = response.review_comment
        fired = apply_precheck_scanner_failure_gate(response, ["zizmor"], block_on_failure=False)
        assert fired is False
        assert response.verdict == Verdict.APPROVE
        assert response.findings == []
        assert response.review_comment == original_comment

    def test_noop_when_no_failures(self) -> None:
        response = _response()
        original_comment = response.review_comment
        fired = apply_precheck_scanner_failure_gate(response, [], block_on_failure=True)
        assert fired is False
        assert response.verdict == Verdict.APPROVE
        assert response.findings == []
        assert response.review_comment == original_comment

    def test_noop_when_already_blocking(self) -> None:
        """Never touches a review that's already BLOCKING for its own
        reasons -- this flag only ever strengthens, never re-derives, the
        verdict.
        """
        response = _response(verdict=Verdict.BLOCKING)
        original_comment = response.review_comment
        fired = apply_precheck_scanner_failure_gate(response, ["zizmor"], block_on_failure=True)
        assert fired is False
        assert response.verdict == Verdict.BLOCKING
        assert response.findings == []
        assert response.review_comment == original_comment

    def test_forces_blocking_when_flag_on_and_failures_present(self) -> None:
        response = _response()
        fired = apply_precheck_scanner_failure_gate(
            response, ["zizmor", "trivy"], block_on_failure=True
        )
        assert fired is True
        assert response.verdict == Verdict.BLOCKING
        assert response.risk_level == RiskLevel.HIGH
        assert len(response.findings) == 1
        finding = response.findings[0]
        assert finding.severity.value == "BLOCKING"
        assert finding.category == "deterministic-precheck"
        # Sorted, not insertion order -- deterministic regardless of which
        # scanner's coroutine happened to finish first.
        assert "trivy" in finding.description
        assert "zizmor" in finding.description
        assert finding.description.index("trivy") < finding.description.index("zizmor")

    def test_rewrites_the_rendered_comments_verdict_line(self) -> None:
        """Regression test: the structured verdict/risk_level/findings were
        the only things mutated by an earlier version of this gate, leaving
        the human-visible comment -- what cli.py actually posts to the PR
        and persists to the DB -- still reading APPROVE. The comment must
        reflect BLOCKING too, not just the structured response fields.
        """
        response = _response(
            review_comment="## Code Review\n\n**Verdict**: ✅ APPROVE | **Risk**: LOW\n\nLooks good."
        )
        apply_precheck_scanner_failure_gate(response, ["zizmor"], block_on_failure=True)

        assert "**Verdict**: 🚫 BLOCKING" in response.review_comment
        assert "✅ APPROVE" not in response.review_comment
        # The explanation must actually reach the PR-visible comment, not
        # just the structured Finding -- that was the other half of the
        # original bug.
        assert "zizmor" in response.review_comment
        assert "ARGUS_PRECHECK_BLOCK_ON_SCANNER_FAILURE" in response.review_comment
        # Original body content preserved, not replaced wholesale.
        assert "Looks good." in response.review_comment

    def test_risk_level_only_raised_never_downgraded(self) -> None:
        """A response that already carries CRITICAL (e.g. from
        _node_validate_blockings leaving APPROVE+CRITICAL) must not be
        silently weakened to HIGH just because this gate also fired --
        RiskLevel.CRITICAL ranks above HIGH, and this gate's own contract
        is "only ever stricter, never looser."
        """
        response = _response(risk_level=RiskLevel.CRITICAL)
        apply_precheck_scanner_failure_gate(response, ["zizmor"], block_on_failure=True)
        assert response.risk_level == RiskLevel.CRITICAL

    def test_risk_level_raised_from_below_high(self) -> None:
        response = _response(risk_level=RiskLevel.MEDIUM)
        apply_precheck_scanner_failure_gate(response, ["zizmor"], block_on_failure=True)
        assert response.risk_level == RiskLevel.HIGH

    def test_risk_level_already_high_stays_high(self) -> None:
        """Exact-boundary case: HIGH is neither raised nor downgraded."""
        response = _response(risk_level=RiskLevel.HIGH)
        apply_precheck_scanner_failure_gate(response, ["zizmor"], block_on_failure=True)
        assert response.risk_level == RiskLevel.HIGH

    def test_warns_and_still_appends_note_when_comment_has_no_verdict_header(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """If review_comment doesn't contain a '**Verdict**:'-shaped line,
        the re.subn rewrite can't find anything to rewrite -- this must not
        silently no-op; it should log a warning and still append the
        explanatory section so the forced-BLOCKING reason is visible
        somewhere in the comment.
        """
        response = _response(review_comment="## Code Review\n\nNo header here at all.")
        with caplog.at_level(logging.WARNING):
            apply_precheck_scanner_failure_gate(response, ["zizmor"], block_on_failure=True)
        assert "No header here at all." in response.review_comment
        assert "Verdict forced to BLOCKING" in response.review_comment
        assert any(
            "no '**Verdict**:'-shaped line found" in record.message for record in caplog.records
        )


class TestApplyBenchConfigChangeGate:
    """Unit tests for apply_bench_config_change_gate (TECH-6282)."""

    def test_noop_when_empty_changes(self) -> None:
        response = _response()
        original_comment = response.review_comment
        fired = apply_bench_config_change_gate(response, [])
        assert fired is False
        assert response.verdict == Verdict.APPROVE
        assert response.findings == []
        assert response.review_comment == original_comment

    def test_forces_blocking_on_bench_changes(self) -> None:
        response = _response()
        fired = apply_bench_config_change_gate(response, [".argus/bench.toml"])
        assert fired is True
        assert response.verdict == Verdict.BLOCKING
        assert response.risk_level == RiskLevel.HIGH
        assert len(response.findings) == 1
        finding = response.findings[0]
        assert finding.severity == Severity.BLOCKING
        assert finding.category == "argus-self-config"
        assert finding.file == ".argus/bench.toml"
        assert ".argus/bench.toml" in finding.description
        assert "human sign-off" in finding.description
        assert "escalate rather than dismiss" in (finding.suggestion or "")
        assert "Verdict forced to BLOCKING" in response.review_comment
        assert ".argus/bench.toml" in response.review_comment

    def test_risk_level_monotonic_raised_from_low(self) -> None:
        response = _response(risk_level=RiskLevel.LOW)
        apply_bench_config_change_gate(response, [".argus/bench.toml"])
        assert response.risk_level == RiskLevel.HIGH

    def test_risk_level_monotonic_raised_from_medium(self) -> None:
        response = _response(risk_level=RiskLevel.MEDIUM)
        apply_bench_config_change_gate(response, [".argus/bench.toml"])
        assert response.risk_level == RiskLevel.HIGH

    def test_risk_level_already_high_stays_high(self) -> None:
        response = _response(risk_level=RiskLevel.HIGH)
        apply_bench_config_change_gate(response, [".argus/bench.toml"])
        assert response.risk_level == RiskLevel.HIGH

    def test_risk_level_critical_never_downgraded(self) -> None:
        response = _response(risk_level=RiskLevel.CRITICAL)
        apply_bench_config_change_gate(response, [".argus/bench.toml"])
        assert response.risk_level == RiskLevel.CRITICAL

    def test_rewrites_rendered_comment_verdict_line_when_verdict_changed(self) -> None:
        response = _response(
            review_comment="## Code Review\n\n**Verdict**: ✅ APPROVE | **Risk**: LOW\n\nLooks good."
        )
        apply_bench_config_change_gate(response, [".argus/bench.toml"])
        assert "**Verdict**: 🚫 BLOCKING | **Risk**: HIGH" in response.review_comment
        assert "✅ APPROVE" not in response.review_comment
        assert "Verdict forced to BLOCKING" in response.review_comment
        assert "Looks good." in response.review_comment

    def test_header_rewrite_skipped_when_already_blocking(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        response = _response(
            verdict=Verdict.BLOCKING,
            risk_level=RiskLevel.HIGH,
            review_comment="## Code Review\n\nNo header here at all.",
        )
        with caplog.at_level(logging.WARNING):
            apply_bench_config_change_gate(response, [".argus/bench.toml"])
        assert not any(
            "no '**Verdict**:'-shaped line found" in record.message for record in caplog.records
        )
        assert "Verdict forced to BLOCKING" in response.review_comment

    def test_warns_and_still_appends_note_when_comment_has_no_verdict_header_and_verdict_changed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        response = _response(
            verdict=Verdict.APPROVE,
            review_comment="## Code Review\n\nNo header here at all.",
        )
        with caplog.at_level(logging.WARNING):
            apply_bench_config_change_gate(response, [".argus/bench.toml"])
        assert "No header here at all." in response.review_comment
        assert "Verdict forced to BLOCKING" in response.review_comment
        assert any(
            "no '**Verdict**:'-shaped line found" in record.message for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_ordering_invariant_in_run_review(self) -> None:
        """Integration-level invariant: in run_review,
        apply_precheck_gate_and_surface_degraded_coverage must be called
        BEFORE apply_bench_config_change_gate.

        If bench gate were called first, it would force verdict to BLOCKING
        before the precheck gate runs, which would cause the precheck gate to
        early-return and create a spurious coverage-gap finding instead of
        proper precheck finding accounting.
        """
        from argus.graph import run_review
        from argus.models import ReviewRequest, TokenUsage

        req = ReviewRequest(repo="org/repo", pr_number=42)
        mock_response = ReviewResponse(
            verdict=Verdict.APPROVE,
            risk_level=RiskLevel.LOW,
            findings=[],
            coverage_map=[],
            review_comment="## Code Review\n\n**Verdict**: ✅ APPROVE | **Risk**: LOW\n\nLooks good.",
            usage=TokenUsage(input_tokens=10, output_tokens=10, cost_usd=0.001),
        )

        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "response": mock_response.model_dump(),
            "findings": [],
            "precheck_scanner_failures": ["zizmor"],
            "bench_config_changes": [".argus/bench.toml"],
        }

        call_order: list[str] = []

        def spy_precheck_gate(*args: Any, **kwargs: Any) -> tuple[bool, list[tuple[str, str]]]:
            call_order.append("precheck_gate")
            from argus.helpers import (
                apply_precheck_gate_and_surface_degraded_coverage as real_precheck,
            )

            return real_precheck(*args, **kwargs)

        def spy_bench_gate(*args: Any, **kwargs: Any) -> bool:
            call_order.append("bench_gate")
            from argus.helpers import apply_bench_config_change_gate as real_bench

            return real_bench(*args, **kwargs)

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
            patch("argus.graph.build_pipeline") as mock_build,
            patch("argus.graph.validate_history_backend_connectivity", new_callable=AsyncMock),
            patch(
                "argus.storage.resolver.get_async_session_factory",
                return_value=mock_session_factory,
            ),
            patch("argus.github_client.GitHubClient", return_value=mock_gh),
            patch("argus.graph.provisioned_worktree", return_value=mock_worktree_ctx),
            patch(
                "argus.graph.apply_precheck_gate_and_surface_degraded_coverage",
                side_effect=spy_precheck_gate,
            ),
            patch("argus.graph.apply_bench_config_change_gate", side_effect=spy_bench_gate),
            patch.dict(os.environ, {"ARGUS_PRECHECK_BLOCK_ON_SCANNER_FAILURE": "1"}),
        ):
            mock_build.return_value.__aenter__ = AsyncMock(return_value=mock_graph)
            mock_build.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await run_review(req, flow_run_id="flow-order-test")

        # 1. Assert call order in run_review
        assert call_order == ["precheck_gate", "bench_gate"]

        # 2. Assert real end-to-end outcome: both gates fired and no spurious coverage-gap
        assert result.verdict == Verdict.BLOCKING
        categories = [f.category for f in result.findings]
        assert "deterministic-precheck" in categories
        assert "argus-self-config" in categories
        assert "coverage-gap" not in categories


class TestApplyPrecheckGateAndSurfaceDegradedCoverage:
    """Integration-level coverage for the round-3 BLOCKING-fix ordering
    invariant: apply_precheck_scanner_failure_gate must run BEFORE the
    coverage-gap findings are built, since coverage_gap_findings_for_round
    needs the gate's real return value (not a hand-supplied boolean) to
    decide whether to filter precheck: labels. Testing each half in
    isolation (as TestApplyPrecheckScannerFailureGate and
    TestCoverageGapFindingsForRound already do) cannot catch a refactor
    that reorders the two real call sites in graph.run_review; calling the
    combined function end-to-end, as done here, can.
    """

    def test_non_blocking_config_still_surfaces_precheck_failure_as_finding(self) -> None:
        """ARGUS_PRECHECK_BLOCK_ON_SCANNER_FAILURE unset (the default): the
        gate adds nothing, so the crashed precheck scanner must still get
        a SUGGESTION/coverage-gap finding -- the exact round-3 bug was
        this ending up with zero findings in this configuration."""
        from argus.helpers import apply_precheck_gate_and_surface_degraded_coverage

        response = _response()
        graph_result = {"precheck_scanner_failures": ["zizmor"]}
        gate_added_precheck_finding, failed_labels = (
            apply_precheck_gate_and_surface_degraded_coverage(
                response, findings_models=[], graph_result=graph_result, block_on_failure=False
            )
        )
        assert gate_added_precheck_finding is False
        assert failed_labels == [("precheck:zizmor", "scanner did not complete this round")]
        assert response.verdict == Verdict.APPROVE
        assert len(response.findings) == 1
        assert response.findings[0].category == "coverage-gap"
        assert "precheck:zizmor" in response.findings[0].description
        assert "⚠ Degraded coverage" in response.review_comment

    def test_blocking_config_surfaces_gate_finding_without_duplicate_coverage_gap(self) -> None:
        """ARGUS_PRECHECK_BLOCK_ON_SCANNER_FAILURE set and the gate fires:
        the same precheck failure must get exactly the gate's own
        BLOCKING/deterministic-precheck finding, not also a SUGGESTION/
        coverage-gap finding for the identical scanner (double-reporting,
        the bug this whole split exists to prevent)."""
        from argus.helpers import apply_precheck_gate_and_surface_degraded_coverage

        response = _response()
        graph_result = {"precheck_scanner_failures": ["zizmor"]}
        gate_added_precheck_finding, failed_labels = (
            apply_precheck_gate_and_surface_degraded_coverage(
                response, findings_models=[], graph_result=graph_result, block_on_failure=True
            )
        )
        assert gate_added_precheck_finding is True
        assert failed_labels == [("precheck:zizmor", "scanner did not complete this round")]
        assert response.verdict == Verdict.BLOCKING
        assert len(response.findings) == 1
        assert response.findings[0].category == "deterministic-precheck"
        # The markdown "Degraded coverage" section is still rendered
        # (it's a separate, additive surface from the structured findings).
        assert "⚠ Degraded coverage" in response.review_comment

    def test_no_failures_at_all_is_a_full_noop(self) -> None:
        from argus.helpers import apply_precheck_gate_and_surface_degraded_coverage

        response = _response()
        original_comment = response.review_comment
        gate_added_precheck_finding, failed_labels = (
            apply_precheck_gate_and_surface_degraded_coverage(
                response, findings_models=[], graph_result={}, block_on_failure=True
            )
        )
        assert gate_added_precheck_finding is False
        assert failed_labels == []
        assert response.verdict == Verdict.APPROVE
        assert response.findings == []
        assert response.review_comment == original_comment

    def test_missing_scanner_default_config_surfaces_but_does_not_block(self) -> None:
        """The bug this whole change fixes: a never-installed scanner must
        become visible (a coverage-gap finding + the markdown section) even
        though ARGUS_PRECHECK_BLOCK_ON_SCANNER_FAILURE is unset (the
        default) -- and the default-off gate must remain a true no-op, same
        as for a crashed scanner."""
        from argus.helpers import apply_precheck_gate_and_surface_degraded_coverage

        response = _response()
        graph_result = {"precheck_missing_scanners": ["trivy"]}
        gate_added_precheck_finding, failed_labels = (
            apply_precheck_gate_and_surface_degraded_coverage(
                response, findings_models=[], graph_result=graph_result, block_on_failure=False
            )
        )
        assert gate_added_precheck_finding is False
        assert failed_labels == [("precheck:trivy", "scanner not installed")]
        assert response.verdict == Verdict.APPROVE
        assert len(response.findings) == 1
        assert response.findings[0].category == "coverage-gap"
        assert "is not installed" in response.findings[0].description
        assert "⚠ Degraded coverage" in response.review_comment

    def test_missing_scanner_feeds_the_gate_when_block_on_failure_is_set(self) -> None:
        """When the opt-in flag IS set, a never-installed scanner must be
        treated the same as a crashed one for gating purposes -- both mean
        "no confirmed coverage from this scanner this round" -- forcing
        BLOCKING exactly like TestApplyPrecheckGateAndSurfaceDegradedCoverage's
        crashed-scanner equivalent above."""
        from argus.helpers import apply_precheck_gate_and_surface_degraded_coverage

        response = _response()
        graph_result = {"precheck_missing_scanners": ["trivy"]}
        gate_added_precheck_finding, failed_labels = (
            apply_precheck_gate_and_surface_degraded_coverage(
                response, findings_models=[], graph_result=graph_result, block_on_failure=True
            )
        )
        assert gate_added_precheck_finding is True
        assert failed_labels == [("precheck:trivy", "scanner not installed")]
        assert response.verdict == Verdict.BLOCKING
        assert len(response.findings) == 1
        assert response.findings[0].category == "deterministic-precheck"

    def test_mixed_timed_out_reviewer_and_precheck_failure_same_round(self) -> None:
        """Production-realistic mixed shape: a timed-out LLM reviewer
        session AND a crashed precheck scanner in the same round, with the
        gate off (the default). Both failure classes must be surfaced --
        two coverage-gap findings plus the markdown section -- since
        reviewer-session failures and precheck-scanner failures are
        different failure classes this flag was never scoped to unify.

        The reviewer failure also forces BLOCKING (see
        TestApplyReviewerFailureGate) -- only the reviewer's own
        coverage-gap finding is promoted, not the precheck scanner's."""
        from argus.helpers import apply_precheck_gate_and_surface_degraded_coverage

        response = _response()
        timed_out_reviewer = SystemReviewResult(
            system_group="backend",
            findings=[],
            files_explored=[],
            cost_usd=0.0,
            timed_out=True,
        )
        graph_result = {"precheck_scanner_failures": ["zizmor"]}

        gate_added_precheck_finding, failed_labels = (
            apply_precheck_gate_and_surface_degraded_coverage(
                response,
                findings_models=[timed_out_reviewer],
                graph_result=graph_result,
                block_on_failure=False,
            )
        )

        assert gate_added_precheck_finding is False
        assert set(failed_labels) == {
            ("backend", "timeout"),
            ("precheck:zizmor", "scanner did not complete this round"),
        }
        assert response.verdict == Verdict.BLOCKING
        assert len(response.findings) == 2
        categories = {f.category for f in response.findings}
        assert categories == {"coverage-gap"}
        descriptions = [f.description for f in response.findings]
        assert any("backend" in d for d in descriptions)
        assert any("precheck:zizmor" in d for d in descriptions)
        assert "⚠ Degraded coverage" in response.review_comment

        reviewer_finding = next(f for f in response.findings if "'backend'" in f.description)
        precheck_finding = next(
            f for f in response.findings if "'precheck:zizmor'" in f.description
        )
        assert reviewer_finding.severity == Severity.BLOCKING
        assert precheck_finding.severity == Severity.SUGGESTION


class TestApplyReviewerFailureGate:
    """Task: a review with a failed reviewer may not APPROVE -- covers both
    directions so a future change can't quietly widen or drop the rule."""

    def test_reviewer_failure_forces_blocking_even_with_no_other_findings(self) -> None:
        from argus.helpers import apply_precheck_gate_and_surface_degraded_coverage

        response = _response()
        crashed_reviewer = SystemReviewResult(
            system_group="auth",
            findings=[],
            files_explored=[],
            cost_usd=0.0,
            failure_reason="worker_crashed",
        )
        apply_precheck_gate_and_surface_degraded_coverage(
            response, findings_models=[crashed_reviewer], graph_result={}, block_on_failure=False
        )
        assert response.verdict == Verdict.BLOCKING
        assert response.risk_level == RiskLevel.HIGH
        assert any(
            f.category == "coverage-gap" and f.severity == Severity.BLOCKING
            for f in response.findings
        )
        assert "🚫 BLOCKING" in response.review_comment

    def test_no_reviewer_failures_leaves_approve_unchanged(self) -> None:
        from argus.helpers import apply_precheck_gate_and_surface_degraded_coverage

        response = _response()
        gate_added_precheck_finding, failed_labels = (
            apply_precheck_gate_and_surface_degraded_coverage(
                response, findings_models=[], graph_result={}, block_on_failure=False
            )
        )
        assert failed_labels == []
        assert response.verdict == Verdict.APPROVE
        assert response.risk_level == RiskLevel.LOW
        assert response.findings == []


class TestRiskLevelOrderExhaustive:
    def test_covers_every_risk_level(self) -> None:
        assert set(_RISK_LEVEL_ORDER) == set(RiskLevel)


class TestAppendDegradedCoverageSection:
    def test_no_labels_returns_comment_unchanged(self) -> None:
        comment = "## Review\n\nLooks good."
        assert append_degraded_coverage_section(comment, []) == comment

    def test_labels_appended_as_visible_section(self) -> None:
        comment = "## Review\n\nLooks good."
        result = append_degraded_coverage_section(
            comment, [("specialist/orchestration::g2", "timeout")]
        )
        assert result.startswith(comment)
        assert "Degraded coverage" in result
        assert "specialist/orchestration::g2" in result
        assert "not reviewed" in result
        # The reason itself (not just the label) must be visible, so a
        # reader can tell a timeout apart from a worker crash.
        assert "(timeout)" in result

    def test_multiple_labels_each_get_a_bullet(self) -> None:
        result = append_degraded_coverage_section(
            "body", [("a", "timeout"), ("b", "worker_crashed"), ("c", "timeout")]
        )
        assert "- a" in result
        assert "- b" in result
        assert "- c" in result
        # Each label's own reason must appear next to it, not just the first.
        assert "a (timeout)" in result
        assert "b (worker_crashed)" in result
        assert "c (timeout)" in result
