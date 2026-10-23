"""Regression coverage for runner failures surviving the 200-character summaries."""

from datetime import datetime, timezone

import httpx
import pytest

from argus.gemini_runner import _TRANSIENT_HTTPX
from argus.helpers import (
    apply_precheck_gate_and_surface_degraded_coverage,
    failed_reviewer_labels,
)
from argus.models import ReviewResponse, RiskLevel, Verdict
from argus.pipeline_models import SystemReviewResult
from argus.runners import SessionResult, _apply_session_failure


def test_provider_cause_survives_round_trip_and_short_service_summary() -> None:
    now = datetime.now(timezone.utc)
    session = SessionResult(
        result_text="",
        cost_usd=0.1,
        duration_seconds=200,
        started_at=now,
        finished_at=now,
        tool_call_count=18,
        failure_reason="worker_crashed",
        failure_detail="503 UNAVAILABLE, turn 45",
    )
    result = SystemReviewResult(system_group="dependency-lockfile-" * 30 + "::deployment")
    _apply_session_failure(result, session)
    # LangGraph serializes and reconstructs reviewer results between nodes.
    restored = SystemReviewResult.model_validate_json(result.model_dump_json())
    response = ReviewResponse(
        verdict=Verdict.APPROVE, risk_level=RiskLevel.LOW, review_comment="Review"
    )
    _, labels = apply_precheck_gate_and_surface_degraded_coverage(response, [restored], {}, False)
    assert labels == [(result.system_group, "worker_crashed: 503 UNAVAILABLE, turn 45")]
    assert response.verdict == Verdict.BLOCKING
    assert len(response.findings) == 1
    finding = response.findings[0]
    assert finding.category == "coverage-gap"
    # api-main and rh-mcp both retain only the first 200 characters.
    summary = finding.description[:200]
    assert "503 UNAVAILABLE, turn 45" in summary
    assert "worker_crashed" in summary
    assert "Area:" in summary
    assert "503 UNAVAILABLE, turn 45" in response.review_comment
    assert result.system_group in response.review_comment


@pytest.mark.parametrize(
    "detail",
    [
        "503 UNAVAILABLE, api_key=not-for-the-report",
        "503 UNAVAILABLE\nBearer private-token",
        "503 " + "x" * 1000,
        "503 INTERNAL https://private.example",
        "UNAVAILABLE",
        "502 bad gateway",
        "502 Bad Gateway at https://private.example",
        "502 " + "Bad " * 11,
        "WriteError api_key=x",
        "",
        None,
    ],
)
def test_noncanonical_failure_detail_is_never_rendered(detail: str | None) -> None:
    result = SystemReviewResult(
        system_group="deployment", failure_reason="worker_crashed", failure_detail=detail
    )
    assert failed_reviewer_labels([result]) == [("deployment", "worker_crashed")]
    response = ReviewResponse(
        verdict=Verdict.APPROVE, risk_level=RiskLevel.LOW, review_comment="Review"
    )
    apply_precheck_gate_and_surface_degraded_coverage(response, [result], {}, False)
    assert "did not complete (worker_crashed)" in response.findings[0].description
    assert "private" not in response.model_dump_json()
    assert "api_key" not in response.model_dump_json()


@pytest.mark.parametrize(
    "detail",
    [
        "500 INTERNAL, turn 18",
        "429 RESOURCE_EXHAUSTED (daily quota), turn 0",
        "ConnectError",
        "502 Bad Gateway, turn 0",
        "504 Gateway Timeout",
        "503 Service Unavailable, turn 3",
        "WriteError, turn 1",
        "ReadTimeout",
    ],
)
def test_canonical_failure_detail_is_rendered(detail: str) -> None:
    result = SystemReviewResult(
        system_group="security", failure_reason="worker_crashed", failure_detail=detail
    )
    assert failed_reviewer_labels([result]) == [("security", f"worker_crashed: {detail}")]


def test_success_does_not_turn_into_a_failure_due_to_detail() -> None:
    result = SystemReviewResult(system_group="deployment", failure_detail="500 INTERNAL")
    assert failed_reviewer_labels([result]) == []


def _subclasses(base: type[BaseException]) -> set[type[BaseException]]:
    found: set[type[BaseException]] = {base}
    for sub in base.__subclasses__():
        found |= _subclasses(sub)
    return found


@pytest.mark.parametrize(
    "exc_type",
    sorted(
        _subclasses(httpx.TransportError).union(*map(_subclasses, _TRANSIENT_HTTPX)),
        key=repr,
    ),
)
def test_every_runner_transport_exception_name_is_rendered(
    exc_type: type[BaseException],
) -> None:
    result = SystemReviewResult(
        system_group="security", failure_reason="worker_crashed", failure_detail=exc_type.__name__
    )
    assert failed_reviewer_labels([result]) == [
        ("security", f"worker_crashed: {exc_type.__name__}")
    ]
