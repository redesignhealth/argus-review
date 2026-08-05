"""Unit tests for pure helper functions in helpers.py.

These are security controls and parsing logic with no LLM dependency.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pytest

from argus.helpers import (
    _RISK_LEVEL_ORDER,
    append_degraded_coverage_section,
    apply_precheck_scanner_failure_gate,
    build_degraded_coverage_labels,
    collect_reviewed_files,
    extract_changed_files,
    filter_diff_for_files,
    parse_review_result,
    sanitize_file_paths,
    failed_reviewer_labels,
)
from argus.models import ReviewResponse, RiskLevel, Verdict
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
