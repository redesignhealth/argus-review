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
