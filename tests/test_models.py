"""Smoke tests for data contracts — validates Pydantic models parse correctly."""

from __future__ import annotations

from argus.models import (
    Finding,
    ReviewRequest,
    ReviewResponse,
    RiskLevel,
    Severity,
    SystemCoverage,
    TokenUsage,
    Verdict,
)


def test_review_request_minimal() -> None:
    req = ReviewRequest(repo="owner/repo", pr_number=2802)
    assert req.sha is None
    assert req.base_ref is None


def test_review_request_full() -> None:
    req = ReviewRequest(
        repo="owner/repo",
        pr_number=2802,
        sha="abc123def456",
        base_ref="main",
    )
    assert req.sha == "abc123def456"
    assert req.base_ref == "main"


def test_finding_blocking() -> None:
    f = Finding(
        severity=Severity.BLOCKING,
        category="security",
        file="app/config/settings.py",
        line=42,
        description="Hardcoded API key",
    )
    assert f.severity == Severity.BLOCKING


def test_review_response_approve() -> None:
    resp = ReviewResponse(
        verdict=Verdict.APPROVE,
        risk_level=RiskLevel.LOW,
        findings=[],
        coverage_map=[
            SystemCoverage(
                system="MCP service",
                files_explored=["services/api/main.py"],
                checks_performed=["async patterns", "error handling"],
            ),
        ],
        review_comment="## Review\n\nAll clear.",
        usage=TokenUsage(input_tokens=1000, output_tokens=500, cost_usd=0.03),
    )
    assert resp.verdict == Verdict.APPROVE
    assert len(resp.coverage_map) == 1


def test_review_response_blocking() -> None:
    resp = ReviewResponse(
        verdict=Verdict.BLOCKING,
        risk_level=RiskLevel.HIGH,
        findings=[
            Finding(
                severity=Severity.BLOCKING,
                category="security",
                description="SQL injection via string interpolation",
                file="projects/backend-api/backend_api/endpoints/search.py",
                line=55,
            ),
        ],
        coverage_map=[],
        notes_for_next_round="Focus on search.py SQL queries",
        review_comment="## Review\n\n1 blocking issue found.",
    )
    assert resp.verdict == Verdict.BLOCKING
    assert resp.findings[0].severity == Severity.BLOCKING
