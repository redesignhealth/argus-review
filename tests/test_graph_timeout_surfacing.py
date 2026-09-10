"""Unit tests for surfacing failed reviewer sessions in _node_run_reviewer.

Covers surfacing failed reviewers: a killed/timed-out or crashed reviewer
subprocess must be logged distinctly from a reviewer that ran to completion
and genuinely found nothing (both currently produce a 0-finding
SystemReviewResult).
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, patch

import pytest

from argus.pipeline_models import AgentRunData, SystemReviewResult

_GRAPH_MODULE = "argus.graph"


def _make_group_dict() -> dict[str, object]:
    return {
        "name": "Storage engine caching & SQLite concurrency",
        "files": ["argus/storage/sqlite.py"],
        "conventions": "",
        "review_focus": "",
        "specialists_needed": [],
    }


class TestNodeRunReviewerTimeoutLogging:
    @pytest.mark.asyncio
    async def test_timed_out_result_logs_warning_not_info_done_line(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A SystemReviewResult with failure_reason="timeout" produces a WARNING
        log naming the reviewer instead of the normal INFO 'done: N findings' line.
        """
        from argus.graph import _node_run_reviewer

        timed_out_result = SystemReviewResult(
            system_group="Storage engine caching & SQLite concurrency",
            findings=[],
            files_explored=[],
            cost_usd=0.0,
            failure_reason="timeout",
        )
        agent_run = AgentRunData(
            agent_name="system:Storage engine caching & SQLite concurrency",
            agent_type="system",
            duration_seconds=300.0,
            failure_reason="timeout",
        )

        inputs = {
            "reviewer_type": "system",
            "group": _make_group_dict(),
            "specialist": "",
            "diff": "diff --git a/x b/x",
            "plan": {},
        }

        with (
            patch(
                f"{_GRAPH_MODULE}.review_system_group",
                new_callable=AsyncMock,
                return_value=(timed_out_result, agent_run),
            ),
            caplog.at_level(logging.INFO, logger=_GRAPH_MODULE),
        ):
            state_update = await _node_run_reviewer(inputs, {"configurable": {}})

        # Result is preserved in state regardless of failure status.
        assert state_update["findings"][0]["failure_reason"] == "timeout"

        warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any("FAILED" in r.message for r in warning_records)
        info_records = [r for r in caplog.records if r.levelname == "INFO"]
        assert not any("done:" in r.message for r in info_records)

    @pytest.mark.asyncio
    async def test_non_timed_out_result_still_logs_normal_done_line(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A normal (non-failed) result keeps the existing INFO 'done:' line."""
        from argus.graph import _node_run_reviewer

        normal_result = SystemReviewResult(
            system_group="Storage engine caching & SQLite concurrency",
            findings=[],
            files_explored=[],
            cost_usd=0.05,
            failure_reason=None,
        )
        agent_run = AgentRunData(
            agent_name="system:Storage engine caching & SQLite concurrency",
            agent_type="system",
            duration_seconds=42.0,
            failure_reason=None,
        )

        inputs = {
            "reviewer_type": "system",
            "group": _make_group_dict(),
            "specialist": "",
            "diff": "diff --git a/x b/x",
            "plan": {},
        }

        with (
            patch(
                f"{_GRAPH_MODULE}.review_system_group",
                new_callable=AsyncMock,
                return_value=(normal_result, agent_run),
            ),
            caplog.at_level(logging.INFO, logger=_GRAPH_MODULE),
        ):
            state_update = await _node_run_reviewer(inputs, {"configurable": {}})

        assert state_update["findings"][0]["failure_reason"] is None
        info_records = [r for r in caplog.records if r.levelname == "INFO"]
        assert any("done:" in r.message for r in info_records)
        warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert not any("TIMED OUT" in r.message for r in warning_records)

    @pytest.mark.asyncio
    async def test_reviewer_exception_surfaces_as_worker_crashed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When a reviewer raises an unhandled exception, _node_run_reviewer
        must return a SystemReviewResult with failure_reason='worker_crashed'
        rather than dropping findings silently."""
        from argus.graph import _node_run_reviewer

        inputs = {
            "reviewer_type": "system",
            "group": _make_group_dict(),
            "specialist": "",
            "diff": "diff --git a/x b/x",
            "plan": {},
        }

        with (
            patch(
                f"{_GRAPH_MODULE}.review_system_group",
                new_callable=AsyncMock,
                side_effect=RuntimeError("unexpected crash"),
            ),
            caplog.at_level(logging.ERROR, logger=_GRAPH_MODULE),
        ):
            state_update = await _node_run_reviewer(inputs, {"configurable": {}})

        assert len(state_update["findings"]) == 1
        result = state_update["findings"][0]
        assert result["failure_reason"] == "worker_crashed"
        assert result["system_group"] == "Storage engine caching & SQLite concurrency"
        assert len(state_update["agent_runs"]) == 1
        assert state_update["agent_runs"][0]["failure_reason"] == "worker_crashed"

    async def test_malformed_group_dict_falls_back_to_raw_name(self) -> None:
        """When SystemGroup.model_validate(inputs["group"]) itself raises
        (e.g. a missing required field), `group` is never assigned, so the
        branches keyed on `group` being truthy never match. The except
        block must still recover a meaningful group name from the raw,
        unvalidated input dict rather than collapsing to the bare
        reviewer_type string ("system")."""
        from argus.graph import _node_run_reviewer

        malformed_group = {"name": "backend-api"}  # missing required "files"/"conventions"/etc.
        inputs = {
            "reviewer_type": "system",
            "group": malformed_group,
            "specialist": "",
            "diff": "diff --git a/x b/x",
            "plan": {},
        }

        state_update = await _node_run_reviewer(inputs, {"configurable": {}})

        assert len(state_update["findings"]) == 1
        result = state_update["findings"][0]
        assert result["failure_reason"] == "worker_crashed"
        assert result["system_group"] == "backend-api"
        assert len(state_update["agent_runs"]) == 1
        assert state_update["agent_runs"][0]["agent_name"] == "system:backend-api"

    async def test_unrecognized_reviewer_type_skips_agent_run_insert(self) -> None:
        """An anomalous (non-Literal) reviewer_type -- e.g. a typo'd Send
        arg -- must not be coerced into any of AgentType's real values in
        the crash-marker telemetry: AgentType has no "unknown" member, and
        picking an existing one would misattribute this anomalous event's
        telemetry to a real reviewer kind. The crashed_result finding must
        still be returned; only the agent_runs insert is skipped."""
        from argus.graph import _node_run_reviewer

        inputs = {
            "reviewer_type": "not_a_real_type",
            "group": {},
            "specialist": "",
            "diff": "diff --git a/x b/x",
            "plan": {},
        }

        state_update = await _node_run_reviewer(inputs, {"configurable": {}})

        assert len(state_update["findings"]) == 1
        result = state_update["findings"][0]
        assert result["failure_reason"] == "worker_crashed"
        assert result["system_group"] == "not_a_real_type"
        assert state_update["agent_runs"] == []


class TestDegradedCoverageFindings:
    def test_build_degraded_coverage_findings_creates_suggestions(self) -> None:
        from argus.helpers import build_degraded_coverage_findings
        from argus.models import Severity

        failed_labels = [
            ("system/backend", "timeout"),
            ("specialist/security", "worker_crashed"),
        ]
        findings = build_degraded_coverage_findings(failed_labels)
        assert len(findings) == 2
        for f in findings:
            assert f.severity == Severity.SUGGESTION
            assert f.category == "coverage-gap"

        assert "system/backend" in findings[0].description
        assert "timed out" in findings[0].description
        assert "specialist/security" in findings[1].description
        assert "worker_crashed" in findings[1].description

    def test_build_degraded_coverage_findings_words_precheck_labels_distinctly(self) -> None:
        """A `precheck:`-prefixed label (a failed deterministic scanner, not
        a crashed LLM reviewer session) must be worded as a scanner, not a
        'Reviewer session' -- see build_degraded_coverage_findings' own
        docstring for the double-reporting concern this defense in depth
        guards against if a future caller passes an unfiltered label list."""
        from argus.helpers import build_degraded_coverage_findings

        findings = build_degraded_coverage_findings(
            [("precheck:zizmor", "scanner did not complete this round")]
        )
        assert len(findings) == 1
        assert "Precheck scanner" in findings[0].description
        assert "Reviewer session" not in findings[0].description
        assert "precheck:zizmor" in findings[0].description

    def test_compute_persisted_finding_counts_excludes_coverage_gap(self) -> None:
        """A synthetic coverage-gap SUGGESTION finding (from a crashed/
        timed-out reviewer session) must not inflate the persisted
        blocking_count/suggestion_count columns -- these are infra-failure
        markers, not real review findings, and would otherwise corrupt
        historical trend analysis over those columns."""
        from argus.helpers import compute_persisted_finding_counts
        from argus.models import Finding, Severity

        findings = [
            Finding(
                severity=Severity.BLOCKING,
                category="security",
                file=None,
                line=None,
                description="real bug",
                suggestion=None,
            ),
            Finding(
                severity=Severity.SUGGESTION,
                category="code-quality",
                file=None,
                line=None,
                description="nit",
                suggestion=None,
            ),
            Finding(
                severity=Severity.SUGGESTION,
                category="coverage-gap",
                file=None,
                line=None,
                description="reviewer session crashed",
                suggestion=None,
            ),
        ]
        blocking_count, suggestion_count = compute_persisted_finding_counts(findings)
        assert blocking_count == 1
        assert suggestion_count == 1

    def test_reviewer_only_labels_drops_precheck_prefixed_entries(self) -> None:
        """The graph.run_review call site relies on this filter to avoid
        double-reporting a failed precheck scanner as both a SUGGESTION/
        coverage-gap finding and a BLOCKING/deterministic-precheck
        finding (the latter via apply_precheck_scanner_failure_gate)."""
        from argus.helpers import reviewer_only_labels

        mixed = [
            ("system/backend", "timeout"),
            ("precheck:zizmor", "scanner did not complete this round"),
            ("specialist/security", "worker_crashed"),
            ("precheck:trivy", "scanner did not complete this round"),
        ]
        assert reviewer_only_labels(mixed) == [
            ("system/backend", "timeout"),
            ("specialist/security", "worker_crashed"),
        ]

    def test_failed_reviewer_labels_recognizes_timed_out_field(self) -> None:
        from argus.helpers import failed_reviewer_labels

        result1 = SystemReviewResult(
            system_group="group1",
            findings=[],
            files_explored=[],
            cost_usd=0.0,
            timed_out=True,
        )
        assert result1.failure_reason == "timeout"
        labels = failed_reviewer_labels([result1])
        assert labels == [("group1", "timeout")]


class TestSyncTimedOutAndFailureReasonContract:
    """Direct coverage for the shared pipeline_models._sync_timed_out_and_failure_reason
    helper's own documented contract, independent of any one caller
    (SystemReviewResult/AgentRunData's validators, SessionResult's
    __post_init__, or runners.py's 4 call sites)."""

    def test_forward_sync_timed_out_true_sets_failure_reason(self) -> None:
        from argus.pipeline_models import _sync_timed_out_and_failure_reason

        timed_out, failure_reason = _sync_timed_out_and_failure_reason(True, None)
        assert timed_out is True
        assert failure_reason == "timeout"

    def test_backward_sync_failure_reason_timeout_sets_timed_out(self) -> None:
        from argus.pipeline_models import _sync_timed_out_and_failure_reason

        timed_out, failure_reason = _sync_timed_out_and_failure_reason(False, "timeout")
        assert timed_out is True
        assert failure_reason == "timeout"

    def test_worker_crashed_alone_is_unaffected(self) -> None:
        from argus.pipeline_models import _sync_timed_out_and_failure_reason

        timed_out, failure_reason = _sync_timed_out_and_failure_reason(False, "worker_crashed")
        assert timed_out is False
        assert failure_reason == "worker_crashed"

    def test_contradictory_state_raises(self) -> None:
        """timed_out=True paired with a failure_reason other than 'timeout'
        can only mean a caller bug -- neither field existed before this PR
        introduced them together, so no legitimate historical-data path
        could produce this combination."""
        from argus.pipeline_models import _sync_timed_out_and_failure_reason

        with pytest.raises(ValueError, match="Contradictory failure state"):
            _sync_timed_out_and_failure_reason(True, "worker_crashed")

    def test_contradictory_state_raises_via_system_review_result_constructor(self) -> None:
        """The same guard fires through SystemReviewResult's own
        mode='before' validator, not just when calling the helper
        directly."""
        with pytest.raises(ValueError, match="Contradictory failure state"):
            SystemReviewResult(
                system_group="group1",
                findings=[],
                files_explored=[],
                cost_usd=0.0,
                timed_out=True,
                failure_reason="worker_crashed",
            )

    def test_contradictory_state_raises_via_agent_run_data_constructor(self) -> None:
        with pytest.raises(ValueError, match="Contradictory failure state"):
            AgentRunData(
                agent_name="system:group1",
                agent_type="system",
                timed_out=True,
                failure_reason="worker_crashed",
            )


def _make_plan_dict() -> dict[str, object]:
    """A minimal ReviewPlan with 1 system group (0 specialists) -- expected
    reviewer count = 2 (cross-cutting + tests-and-docs) + 1 (system) = 3.
    """
    return {
        "system_groups": [
            {
                "name": "backend",
                "files": ["a.py"],
                "conventions": "",
                "review_focus": "",
                "specialists_needed": [],
            }
        ],
        "cross_cutting_concerns": [],
        "file_manifest": [{"path": "a.py", "change_type": "modified"}],
    }


class TestNodeCollectFindingsAllCrashedGuard:
    """Regression coverage for the all-reviewers-failed guard in
    _node_collect_findings: a crashed/timed-out reviewer still contributes a
    (marker) SystemReviewResult to state["findings"], so a plain
    `len(findings)` check cannot tell "everyone crashed" apart from
    "everyone succeeded".
    """

    @pytest.mark.asyncio
    async def test_all_reviewers_crashed_raises_instead_of_proceeding(self) -> None:
        """Every reviewer result carries a failure_reason -- must raise, not
        silently continue toward a clean-looking verdict with zero real
        coverage.
        """
        from argus.graph import _node_collect_findings

        crashed = SystemReviewResult(
            system_group="backend",
            findings=[],
            files_explored=[],
            cost_usd=0.0,
            failure_reason="worker_crashed",
        )
        timed_out = SystemReviewResult(
            system_group="cross-cutting",
            findings=[],
            files_explored=[],
            cost_usd=0.0,
            failure_reason="timeout",
        )
        another_crashed = SystemReviewResult(
            system_group="tests-and-docs",
            findings=[],
            files_explored=[],
            cost_usd=0.0,
            failure_reason="worker_crashed",
        )

        state = {
            "plan": _make_plan_dict(),
            "findings": [
                crashed.model_dump(),
                timed_out.model_dump(),
                another_crashed.model_dump(),
            ],
        }

        with pytest.raises(RuntimeError, match="All reviewers failed"):
            await _node_collect_findings(state)

    @pytest.mark.asyncio
    async def test_empty_findings_still_raises(self) -> None:
        """The pre-existing "nothing came back at all" case must still raise."""
        from argus.graph import _node_collect_findings

        state = {"plan": _make_plan_dict(), "findings": []}

        with pytest.raises(RuntimeError, match="All reviewers failed"):
            await _node_collect_findings(state)

    @pytest.mark.asyncio
    async def test_mixed_success_and_crash_proceeds_and_counts_only_real_failures(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """At least one genuinely successful reviewer keeps the pipeline
        going, and the partial-failure count reflects only the
        crashed/timed-out entries plus any missing results -- not
        `expected - len(findings)`.
        """
        from argus.graph import _node_collect_findings

        succeeded = SystemReviewResult(
            system_group="backend",
            findings=[],
            files_explored=[],
            cost_usd=0.01,
            failure_reason=None,
        )
        crashed = SystemReviewResult(
            system_group="cross-cutting",
            findings=[],
            files_explored=[],
            cost_usd=0.0,
            failure_reason="worker_crashed",
        )

        state = {
            "plan": _make_plan_dict(),
            "findings": [succeeded.model_dump(), crashed.model_dump()],
        }

        with caplog.at_level(logging.WARNING, logger=_GRAPH_MODULE):
            result = await _node_collect_findings(state)

        assert result == {}
        warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
        # expected=3, succeeded=1 (only the real success counts) -> failed=2,
        # even though state["findings"] has 2 entries (not 1 as a naive
        # len(findings)-based "succeeded" count would have implied).
        assert any("Partial reviewer failure: 2/3" in r.message for r in warning_records)
