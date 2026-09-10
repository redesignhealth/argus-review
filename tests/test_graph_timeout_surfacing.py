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
