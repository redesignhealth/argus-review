"""Tests for multi-round review iteration.

Covers: prior review fetch, scoped diff, feedback verifier parsing,
round 2 writer messages, and model contracts.
"""

from __future__ import annotations

import importlib
import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from argus.models import (
    PriorFeedbackItem,
    ReviewResponse,
    RiskLevel,
    Verdict,
)
from argus.pipeline_models import (
    DismissedFinding,
    FeedbackVerificationItem,
    FeedbackVerificationResult,
    PriorFinding,
    PriorReviewContext,
    VerificationStatus,
)

_GH_CLIENT_CLASS = "argus.github_client.GitHubClient"


# ---------------------------------------------------------------------------
# Pipeline model tests
# ---------------------------------------------------------------------------


class TestPriorReviewModels:
    def test_verification_status_values(self) -> None:
        assert VerificationStatus.RESOLVED.value == "RESOLVED"
        assert VerificationStatus.UNRESOLVED.value == "UNRESOLVED"
        assert VerificationStatus.REGRESSED.value == "REGRESSED"

    def test_prior_finding_from_result_json(self) -> None:
        """PriorFinding can be constructed from result_json finding data."""
        data = {
            "severity": "BLOCKING",
            "category": "security",
            "file": "src/auth.py",
            "line": 42,
            "description": "SQL injection",
            "suggestion": "Use parameterized queries",
        }
        pf = PriorFinding.model_validate(data)
        assert pf.severity == "BLOCKING"
        assert pf.file == "src/auth.py"

    def test_prior_review_context(self) -> None:
        ctx = PriorReviewContext(
            review_id="abc-123",
            reviewed_sha="deadbeef1234",
            findings=[
                PriorFinding(severity="BLOCKING", description="issue 1"),
                PriorFinding(severity="SUGGESTION", description="issue 2"),
            ],
            notes_for_next_round="focus on auth",
        )
        assert len(ctx.findings) == 2
        assert ctx.reviewed_sha == "deadbeef1234"

    def test_verification_result_empty(self) -> None:
        result = FeedbackVerificationResult(items=[], cost_usd=0.0)
        assert result.items == []

    def test_verification_result_with_items(self) -> None:
        pf = PriorFinding(severity="BLOCKING", description="SQL injection")
        item = FeedbackVerificationItem(
            prior_finding=pf,
            status=VerificationStatus.RESOLVED,
            rationale="Parameterized query added at line 45",
        )
        result = FeedbackVerificationResult(items=[item], cost_usd=0.05)
        assert result.items[0].status == VerificationStatus.RESOLVED
        assert result.cost_usd == 0.05


# ---------------------------------------------------------------------------
# ReviewResponse round 2 fields
# ---------------------------------------------------------------------------


class TestReviewResponseRound2:
    def test_round1_defaults(self) -> None:
        resp = ReviewResponse(
            verdict=Verdict.APPROVE,
            risk_level=RiskLevel.LOW,
            review_comment="All clear",
        )
        assert resp.review_round == 1
        assert resp.prior_review_id is None
        assert resp.prior_feedback == []

    def test_round2_fields(self) -> None:
        resp = ReviewResponse(
            verdict=Verdict.APPROVE,
            risk_level=RiskLevel.MEDIUM,
            review_comment="Round 2 review",
            review_round=2,
            prior_review_id="abc-123",
            prior_feedback=[
                PriorFeedbackItem(
                    severity="BLOCKING",
                    description="SQL injection",
                    file="auth.py",
                    status="RESOLVED",
                    rationale="Fixed with parameterized query",
                ),
            ],
        )
        assert resp.review_round == 2
        assert resp.prior_review_id == "abc-123"
        assert len(resp.prior_feedback) == 1
        assert resp.prior_feedback[0].status == "RESOLVED"

    def test_round2_serialization_roundtrip(self) -> None:
        resp = ReviewResponse(
            verdict=Verdict.BLOCKING,
            risk_level=RiskLevel.HIGH,
            review_comment="Issues remain",
            review_round=2,
            prior_review_id="def-456",
            prior_feedback=[
                PriorFeedbackItem(
                    severity="BLOCKING",
                    description="issue",
                    status="UNRESOLVED",
                    rationale="not addressed",
                ),
            ],
        )
        data = resp.model_dump(mode="json")
        restored = ReviewResponse.model_validate(data)
        assert restored.review_round == 2
        assert restored.prior_feedback[0].status == "UNRESOLVED"


# ---------------------------------------------------------------------------
# Scoped diff (round 2+)
# ---------------------------------------------------------------------------


def _reload_graph():
    mod = "argus.graph"
    if mod in sys.modules:
        importlib.reload(sys.modules[mod])
    from argus.graph import _fetch_pr_diff_and_description

    return _fetch_pr_diff_and_description


class TestScopedDiff:
    @pytest.mark.asyncio
    async def test_round2_uses_prior_sha_for_diff(self) -> None:
        """When prior_sha is provided, diff is scoped to prior_sha..head_sha."""
        mock_gh = MagicMock()
        mock_gh.get_pull_request.return_value = {
            "base_branch": "main",
            "head_sha": "aabbccdd1234",
            "body": "PR description",
        }
        # status="ahead" → prior_sha is still an ancestor of head, so the
        # merge-base/clamp path is not taken. merge_base_sha is intentionally
        # left unused-looking to make that obvious.
        mock_gh.get_compare_metadata.return_value = {
            "status": "ahead",
            "merge_base_sha": "unused-when-ahead",
            "ahead_by": 2,
            "behind_by": 0,
        }
        mock_gh.get_compare_diff.return_value = "diff --git a/fix.py b/fix.py\n+fixed"

        with patch(_GH_CLIENT_CLASS, return_value=mock_gh):
            fn = _reload_graph()
            from argus.models import ReviewRequest

            req = ReviewRequest(repo="org/repo", pr_number=42)
            diff, desc, head_sha = await fn(req, prior_sha="aabb00112233")

        assert diff == "diff --git a/fix.py b/fix.py\n+fixed"
        assert head_sha == "aabbccdd1234"
        # Should use prior_sha as base, not "main"
        mock_gh.get_compare_diff.assert_called_once_with(
            "org/repo", "aabb00112233", "aabbccdd1234", max_lines=5000
        )
        # Non-clamp path: only the initial metadata call, no second clamp call.
        # (Symmetric with the clamp tests, which assert call_count == 2.)
        assert mock_gh.get_compare_metadata.call_count == 1

    @pytest.mark.asyncio
    async def test_round2_clamps_to_merge_base_on_rebase(self) -> None:
        """If prior_sha is no longer an ancestor of head_sha (rebase detected),
        the diff base clamps to merge-base(base_branch, head_sha) to avoid
        including rebase noise from main."""
        mock_gh = MagicMock()
        mock_gh.get_pull_request.return_value = {
            "base_branch": "main",
            "head_sha": "aabbccdd1234",
            "body": "PR description",
        }
        # First call (prior_sha vs head) → diverged (rebase)
        # Second call (base_branch vs head) → merge-base is the new post-rebase branch point
        mock_gh.get_compare_metadata.side_effect = [
            {
                "status": "diverged",
                "merge_base_sha": "deadbeef0000",
                "ahead_by": 3,
                "behind_by": 12,
            },
            {
                "status": "ahead",
                "merge_base_sha": "ccccdddd5678",
                "ahead_by": 3,
                "behind_by": 0,
            },
        ]
        mock_gh.get_compare_diff.return_value = "diff --git a/fix.py b/fix.py\n+fixed"

        with patch(_GH_CLIENT_CLASS, return_value=mock_gh):
            fn = _reload_graph()
            from argus.models import ReviewRequest

            req = ReviewRequest(repo="org/repo", pr_number=42)
            diff, _, head_sha = await fn(req, prior_sha="aabb00112233")

        assert diff == "diff --git a/fix.py b/fix.py\n+fixed"
        assert head_sha == "aabbccdd1234"
        # Diff should use the merge-base SHA as the base, not the orphaned prior_sha
        mock_gh.get_compare_diff.assert_called_once_with(
            "org/repo", "ccccdddd5678", "aabbccdd1234", max_lines=5000
        )
        # Two compare_metadata calls: prior->head, then base_branch->head
        assert mock_gh.get_compare_metadata.call_count == 2
        mock_gh.get_compare_metadata.assert_any_call("org/repo", "aabb00112233", "aabbccdd1234")
        mock_gh.get_compare_metadata.assert_any_call("org/repo", "main", "aabbccdd1234")

    @pytest.mark.asyncio
    async def test_round2_clamps_to_merge_base_on_force_push_rewind(self) -> None:
        """status='behind' (force-push that rewound the branch) is treated
        the same as 'diverged' — clamp to merge-base(base_branch, head)."""
        mock_gh = MagicMock()
        mock_gh.get_pull_request.return_value = {
            "base_branch": "main",
            "head_sha": "aabbccdd1234",
            "body": "PR description",
        }
        mock_gh.get_compare_metadata.side_effect = [
            {
                "status": "behind",
                "merge_base_sha": "deadbeef0000",
                "ahead_by": 0,
                "behind_by": 5,
            },
            {
                "status": "ahead",
                "merge_base_sha": "ccccdddd5678",
                "ahead_by": 1,
                "behind_by": 0,
            },
        ]
        mock_gh.get_compare_diff.return_value = "diff --git a/fix.py b/fix.py\n+fixed"

        with patch(_GH_CLIENT_CLASS, return_value=mock_gh):
            fn = _reload_graph()
            from argus.models import ReviewRequest

            req = ReviewRequest(repo="org/repo", pr_number=42)
            await fn(req, prior_sha="aabb00112233")

        mock_gh.get_compare_diff.assert_called_once_with(
            "org/repo", "ccccdddd5678", "aabbccdd1234", max_lines=5000
        )

    @pytest.mark.asyncio
    async def test_round2_raises_when_merge_base_is_empty(self) -> None:
        """If GitHub returns an empty merge_base_sha during clamp, raise
        rather than silently fall back to the orphaned prior_sha."""
        mock_gh = MagicMock()
        mock_gh.get_pull_request.return_value = {
            "base_branch": "main",
            "head_sha": "aabbccdd1234",
            "body": "PR description",
        }
        mock_gh.get_compare_metadata.side_effect = [
            {
                "status": "diverged",
                "merge_base_sha": "deadbeef0000",
                "ahead_by": 3,
                "behind_by": 12,
            },
            # Pathological: GitHub couldn't compute a merge-base
            {
                "status": "diverged",
                "merge_base_sha": "",
                "ahead_by": 0,
                "behind_by": 0,
            },
        ]

        with patch(_GH_CLIENT_CLASS, return_value=mock_gh):
            fn = _reload_graph()
            from argus.models import ReviewRequest

            req = ReviewRequest(repo="org/repo", pr_number=42)
            with pytest.raises(ValueError, match="no merge-base"):
                await fn(req, prior_sha="aabb00112233")

        mock_gh.get_compare_diff.assert_not_called()

    @pytest.mark.parametrize("short_sha", ["a", "ab", "abc", "abcd", "abcde", "abcdef"])
    @pytest.mark.asyncio
    async def test_round2_rejects_too_short_prior_sha(self, short_sha: str) -> None:
        """_SHA_RE requires ≥7 hex chars — catches short hex strings that
        would have slipped through the old ``[0-9a-fA-F]+`` regex."""
        mock_gh = MagicMock()
        mock_gh.get_pull_request.return_value = {
            "base_branch": "main",
            "head_sha": "aabbccdd1234",
            "body": "",
        }

        with patch(_GH_CLIENT_CLASS, return_value=mock_gh):
            fn = _reload_graph()
            from argus.models import ReviewRequest

            req = ReviewRequest(repo="org/repo", pr_number=42)
            with pytest.raises(ValueError, match="Invalid prior_sha"):
                await fn(req, prior_sha=short_sha)

    @pytest.mark.asyncio
    async def test_round2_raises_on_non_hex_merge_base(self) -> None:
        """If GitHub returns a non-hex merge_base_sha, reject it before
        interpolating into a compare URL."""
        mock_gh = MagicMock()
        mock_gh.get_pull_request.return_value = {
            "base_branch": "main",
            "head_sha": "aabbccdd1234567",
            "body": "",
        }
        mock_gh.get_compare_metadata.side_effect = [
            {
                "status": "diverged",
                "merge_base_sha": "aabbccdd1234567",
                "ahead_by": 3,
                "behind_by": 12,
            },
            {
                "status": "diverged",
                "merge_base_sha": "not-a-sha!",
                "ahead_by": 0,
                "behind_by": 0,
            },
        ]

        with patch(_GH_CLIENT_CLASS, return_value=mock_gh):
            fn = _reload_graph()
            from argus.models import ReviewRequest

            req = ReviewRequest(repo="org/repo", pr_number=42)
            with pytest.raises(ValueError, match="Invalid merge_base_sha"):
                await fn(req, prior_sha="aabbccdd1234500")

        mock_gh.get_compare_diff.assert_not_called()

    @pytest.mark.asyncio
    async def test_round2_identical_status_uses_prior_sha(self) -> None:
        """status='identical' (no new commits since last review) flows through
        the non-rewrite path and uses prior_sha as the diff base."""
        mock_gh = MagicMock()
        mock_gh.get_pull_request.return_value = {
            "base_branch": "main",
            "head_sha": "aabb00112233",  # head == prior_sha
            "body": "",
        }
        mock_gh.get_compare_metadata.return_value = {
            "status": "identical",
            "merge_base_sha": "aabb00112233",
            "ahead_by": 0,
            "behind_by": 0,
        }
        mock_gh.get_compare_diff.return_value = ""

        with patch(_GH_CLIENT_CLASS, return_value=mock_gh):
            fn = _reload_graph()
            from argus.models import ReviewRequest

            req = ReviewRequest(repo="org/repo", pr_number=42)
            await fn(req, prior_sha="aabb00112233")

        # Only the initial metadata call; no clamp call needed
        assert mock_gh.get_compare_metadata.call_count == 1
        mock_gh.get_compare_diff.assert_called_once_with(
            "org/repo", "aabb00112233", "aabb00112233", max_lines=5000
        )

    @pytest.mark.asyncio
    async def test_round1_uses_base_branch(self) -> None:
        """Without prior_sha, diff uses base_branch (round 1 behavior)."""
        mock_gh = MagicMock()
        mock_gh.get_pull_request.return_value = {
            "base_branch": "main",
            "head_sha": "abc123def456",
            "body": "",
        }
        mock_gh.get_compare_diff.return_value = "diff"

        with patch(_GH_CLIENT_CLASS, return_value=mock_gh):
            fn = _reload_graph()
            from argus.models import ReviewRequest

            req = ReviewRequest(repo="org/repo", pr_number=42)
            await fn(req, prior_sha=None)

        mock_gh.get_compare_diff.assert_called_once_with(
            "org/repo", "main", "abc123def456", max_lines=5000
        )

    @pytest.mark.asyncio
    async def test_round2_rejects_invalid_prior_sha(self) -> None:
        """prior_sha must be valid hex."""
        mock_gh = MagicMock()
        mock_gh.get_pull_request.return_value = {
            "base_branch": "main",
            "head_sha": "abc123def456",
            "body": "",
        }

        with patch(_GH_CLIENT_CLASS, return_value=mock_gh):
            fn = _reload_graph()
            from argus.models import ReviewRequest

            req = ReviewRequest(repo="org/repo", pr_number=42)
            with pytest.raises(ValueError, match="Invalid prior_sha"):
                await fn(req, prior_sha="not-hex!")


# ---------------------------------------------------------------------------
# Feedback verifier parsing
# ---------------------------------------------------------------------------


class TestFeedbackVerifierParsing:
    def test_parse_valid_json(self) -> None:
        from argus.runners import _parse_verification_result

        prior_findings = [
            PriorFinding(severity="BLOCKING", description="SQL injection", file="auth.py"),
            PriorFinding(severity="SUGGESTION", description="Add tests"),
        ]
        raw = json.dumps(
            {
                "items": [
                    {"index": 0, "status": "RESOLVED", "rationale": "Fixed with params"},
                    {"index": 1, "status": "UNRESOLVED", "rationale": "No test added"},
                ]
            }
        )
        raw_text = f"```json\n{raw}\n```"
        result = _parse_verification_result(raw_text, prior_findings, 0.05)

        assert len(result.items) == 2
        assert result.items[0].status == VerificationStatus.RESOLVED
        assert result.items[0].prior_finding.description == "SQL injection"
        assert result.items[1].status == VerificationStatus.UNRESOLVED
        assert result.cost_usd == 0.05

    def test_parse_regressed(self) -> None:
        from argus.runners import _parse_verification_result

        prior_findings = [PriorFinding(severity="BLOCKING", description="issue")]
        raw_text = '{"items": [{"index": 0, "status": "REGRESSED", "rationale": "new bug"}]}'
        result = _parse_verification_result(raw_text, prior_findings, 0.0)

        assert result.items[0].status == VerificationStatus.REGRESSED

    def test_parse_invalid_index_skipped(self) -> None:
        from argus.runners import _parse_verification_result

        prior_findings = [PriorFinding(severity="BLOCKING", description="issue")]
        raw_text = '{"items": [{"index": 99, "status": "RESOLVED", "rationale": "bad index"}]}'
        result = _parse_verification_result(raw_text, prior_findings, 0.0)

        assert len(result.items) == 0

    def test_parse_malformed_json_marks_all_unresolved(self) -> None:
        from argus.runners import _parse_verification_result

        prior_findings = [
            PriorFinding(severity="BLOCKING", description="issue 1"),
            PriorFinding(severity="SUGGESTION", description="issue 2"),
        ]
        result = _parse_verification_result("not json at all", prior_findings, 0.0)

        assert len(result.items) == 2
        assert all(i.status == VerificationStatus.UNRESOLVED for i in result.items)

    def test_parse_empty_text(self) -> None:
        from argus.runners import _parse_verification_result

        result = _parse_verification_result("", [], 0.0)
        assert result.items == []

    def test_parse_unknown_status_defaults_to_unresolved(self) -> None:
        from argus.runners import _parse_verification_result

        prior_findings = [PriorFinding(severity="BLOCKING", description="issue")]
        raw_text = '{"items": [{"index": 0, "status": "UNKNOWN", "rationale": "bad"}]}'
        result = _parse_verification_result(raw_text, prior_findings, 0.0)

        assert result.items[0].status == VerificationStatus.UNRESOLVED


# ---------------------------------------------------------------------------
# Writer message building (round 2)
# ---------------------------------------------------------------------------


class TestWriterMessagesRound2:
    def test_round1_no_verification_section(self) -> None:
        from argus.graph import _build_writer_messages
        from argus.pipeline_models import (
            FileEntry,
            ReviewPlan,
            SystemGroup,
            SystemReviewResult,
        )

        plan = ReviewPlan(
            system_groups=[SystemGroup(name="g1", files=["a.py"], conventions="", review_focus="")],
            cross_cutting_concerns=[],
            file_manifest=[FileEntry(path="a.py", change_type="modified")],
        )
        findings = [SystemReviewResult(system_group="g1", findings=[], files_explored=["a.py"])]

        messages = _build_writer_messages(findings, plan, "diff", "desc")
        content = messages[0]["content"]

        assert "Prior Review Feedback Verification" not in content
        assert "round 2" not in content.lower()

    def test_round2_includes_verification_section(self) -> None:
        from argus.graph import _build_writer_messages
        from argus.pipeline_models import (
            FileEntry,
            ReviewPlan,
            SystemGroup,
            SystemReviewResult,
        )

        plan = ReviewPlan(
            system_groups=[SystemGroup(name="g1", files=["a.py"], conventions="", review_focus="")],
            cross_cutting_concerns=[],
            file_manifest=[FileEntry(path="a.py", change_type="modified")],
        )
        findings = [SystemReviewResult(system_group="g1", findings=[], files_explored=["a.py"])]
        verification = FeedbackVerificationResult(
            items=[
                FeedbackVerificationItem(
                    prior_finding=PriorFinding(severity="BLOCKING", description="SQL injection"),
                    status=VerificationStatus.RESOLVED,
                    rationale="Fixed",
                ),
            ],
            cost_usd=0.05,
        )

        messages = _build_writer_messages(findings, plan, "diff", "desc", verification)
        content = messages[0]["content"]

        assert "Prior Review Feedback Verification" in content
        assert "round 2" in content.lower()
        assert "RESOLVED" in content
        assert "SQL injection" in content


# ---------------------------------------------------------------------------
# BLOCKING finding validator parsing
# ---------------------------------------------------------------------------


class TestBlockingValidatorParsing:
    def test_parse_confirmed_and_rejected(self) -> None:
        from argus.runners import _parse_validation_result

        raw = json.dumps(
            {
                "items": [
                    {"index": 0, "verdict": "CONFIRMED", "evidence": "Bug exists at auth.py:42"},
                    {
                        "index": 1,
                        "verdict": "REJECTED",
                        "evidence": "IAM policy covers this at iam.tf:68",
                    },
                ]
            }
        )
        raw_text = f"```json\n{raw}\n```"
        result = _parse_validation_result(raw_text, 2, 0.03)

        from argus.pipeline_models import ValidationVerdict

        assert len(result.items) == 2
        assert result.items[0].verdict == ValidationVerdict.CONFIRMED
        assert result.items[1].verdict == ValidationVerdict.REJECTED
        assert "iam.tf:68" in result.items[1].evidence
        assert result.cost_usd == 0.03

    def test_parse_empty_text_confirms_all(self) -> None:
        from argus.runners import _parse_validation_result

        result = _parse_validation_result("", 3, 0.0)
        from argus.pipeline_models import ValidationVerdict

        assert len(result.items) == 3
        assert all(i.verdict == ValidationVerdict.CONFIRMED for i in result.items)

    def test_parse_malformed_json_confirms_all(self) -> None:
        from argus.runners import _parse_validation_result

        result = _parse_validation_result("not json", 2, 0.0)
        from argus.pipeline_models import ValidationVerdict

        assert len(result.items) == 2
        assert all(i.verdict == ValidationVerdict.CONFIRMED for i in result.items)

    def test_unmentioned_findings_confirmed_by_default(self) -> None:
        """Findings not mentioned by the validator are kept (confirmed)."""
        from argus.runners import _parse_validation_result

        raw_text = '{"items": [{"index": 1, "verdict": "REJECTED", "evidence": "false positive"}]}'
        result = _parse_validation_result(raw_text, 3, 0.0)
        from argus.pipeline_models import ValidationVerdict

        assert len(result.items) == 3
        assert result.items[0].verdict == ValidationVerdict.CONFIRMED  # index 0: not mentioned
        assert result.items[1].verdict == ValidationVerdict.REJECTED  # index 1: rejected
        assert result.items[2].verdict == ValidationVerdict.CONFIRMED  # index 2: not mentioned

    def test_invalid_index_skipped(self) -> None:
        from argus.runners import _parse_validation_result

        raw_text = '{"items": [{"index": 99, "verdict": "REJECTED", "evidence": "bad"}]}'
        result = _parse_validation_result(raw_text, 2, 0.0)
        from argus.pipeline_models import ValidationVerdict

        # Both should be confirmed (99 is out of range, ignored)
        assert len(result.items) == 2
        assert all(i.verdict == ValidationVerdict.CONFIRMED for i in result.items)

    def test_unknown_verdict_defaults_to_confirmed(self) -> None:
        from argus.runners import _parse_validation_result

        raw_text = '{"items": [{"index": 0, "verdict": "MAYBE", "evidence": "unsure"}]}'
        result = _parse_validation_result(raw_text, 1, 0.0)
        from argus.pipeline_models import ValidationVerdict

        assert result.items[0].verdict == ValidationVerdict.CONFIRMED


# ---------------------------------------------------------------------------
# DroppedFinding model
# ---------------------------------------------------------------------------


class TestDroppedFinding:
    def test_dropped_finding_in_response(self) -> None:
        from argus.models import (
            DroppedFinding,
            Finding,
            ReviewResponse,
            RiskLevel,
            Severity,
            Verdict,
        )

        resp = ReviewResponse(
            verdict=Verdict.APPROVE,
            risk_level=RiskLevel.LOW,
            review_comment="All clear",
            dropped_findings=[
                DroppedFinding(
                    finding=Finding(
                        severity=Severity.BLOCKING,
                        category="security",
                        file="iam.tf",
                        line=68,
                        description="Missing IAM permission",
                    ),
                    rejection_rationale="permission exists at iam.tf:68",
                ),
            ],
        )
        assert len(resp.dropped_findings) == 1
        assert resp.dropped_findings[0].finding.description == "Missing IAM permission"

        # Round-trip serialization
        data = resp.model_dump(mode="json")
        restored = ReviewResponse.model_validate(data)
        assert len(restored.dropped_findings) == 1
        assert restored.dropped_findings[0].rejection_rationale == "permission exists at iam.tf:68"


# ---------------------------------------------------------------------------
# Dismiss mechanism
# ---------------------------------------------------------------------------


class TestDismissedFindingModel:
    def test_dismissed_finding_fields(self) -> None:
        d = DismissedFinding(
            file="graph.py",
            description="except Exception swallows DB errors",
            dismissed_by="octocat",
            reason="intentional design choice",
        )
        assert d.dismissed_by == "octocat"
        assert d.file == "graph.py"

    def test_prior_review_context_with_dismissed(self) -> None:
        ctx = PriorReviewContext(
            review_id="abc-123",
            reviewed_sha="deadbeef1234",
            findings=[PriorFinding(severity="BLOCKING", description="real issue")],
            dismissed_findings=[
                DismissedFinding(
                    file="graph.py",
                    description="except Exception",
                    dismissed_by="user1",
                    reason="intentional",
                ),
            ],
        )
        assert len(ctx.findings) == 1
        assert len(ctx.dismissed_findings) == 1


class TestDismissCommentParsing:
    @pytest.mark.asyncio
    async def test_parse_dismiss_comments(self) -> None:
        """_fetch_dismissed_findings stores full comment body for /dismiss comments."""
        mock_gh = MagicMock()
        mock_gh.list_issue_comments.return_value = [
            {
                "id": 1,
                "user": "octocat",
                "body": "/dismiss except Exception — intentional design choice",
                "created_at": "2026-04-08T12:00:00Z",
            },
            {
                "id": 2,
                "user": "otheruser",
                "body": "This looks good to me",
                "created_at": "2026-04-08T13:00:00Z",
            },
        ]

        with patch(_GH_CLIENT_CLASS, return_value=mock_gh):
            _reload_graph()
            from argus.graph import _fetch_dismissed_findings

            dismissed = await _fetch_dismissed_findings("org/repo", 42)

        # Only comment containing /dismiss is included
        assert len(dismissed) == 1
        assert "/dismiss except Exception" in dismissed[0].description
        assert dismissed[0].dismissed_by == "octocat"
        assert dismissed[0].reason == "from PR comment"

    @pytest.mark.asyncio
    async def test_cli_dismissals_passed_through(self) -> None:
        """extra_dismissals from --dismiss CLI args are included as-is."""
        mock_gh = MagicMock()
        mock_gh.list_issue_comments.return_value = []

        with patch(_GH_CLIENT_CLASS, return_value=mock_gh):
            _reload_graph()
            from argus.graph import _fetch_dismissed_findings

            dismissed = await _fetch_dismissed_findings(
                "org/repo", 42, extra_dismissals=["the retry-wrapper thing is pre-existing"]
            )

        assert len(dismissed) == 1
        assert dismissed[0].description == "the retry-wrapper thing is pre-existing"
        assert dismissed[0].dismissed_by == "local"

    @pytest.mark.asyncio
    async def test_no_dismiss_comments(self) -> None:
        mock_gh = MagicMock()
        mock_gh.list_issue_comments.return_value = [
            {"id": 1, "user": "u", "body": "LGTM", "created_at": "2026-04-08T12:00:00Z"},
        ]

        with patch(_GH_CLIENT_CLASS, return_value=mock_gh):
            _reload_graph()
            from argus.graph import _fetch_dismissed_findings

            dismissed = await _fetch_dismissed_findings("org/repo", 42)

        assert dismissed == []


class TestWriterMessagesWithDismissed:
    def test_dismissed_section_included(self) -> None:
        from argus.graph import _build_writer_messages
        from argus.pipeline_models import (
            FileEntry,
            ReviewPlan,
            SystemGroup,
            SystemReviewResult,
        )

        plan = ReviewPlan(
            system_groups=[SystemGroup(name="g1", files=["a.py"], conventions="", review_focus="")],
            cross_cutting_concerns=[],
            file_manifest=[FileEntry(path="a.py", change_type="modified")],
        )
        findings = [SystemReviewResult(system_group="g1", findings=[], files_explored=["a.py"])]
        dismissed = [
            DismissedFinding(
                file="graph.py",
                description="except Exception swallows errors",
                dismissed_by="octocat",
                reason="intentional",
            ),
        ]

        messages = _build_writer_messages(findings, plan, "diff", "desc", dismissed=dismissed)
        content = messages[0]["content"]

        assert "Dismissed Findings" in content
        assert "except Exception" in content
        assert "@octocat" in content
        assert "intentional" in content

    def test_no_dismissed_section_when_empty(self) -> None:
        from argus.graph import _build_writer_messages
        from argus.pipeline_models import (
            FileEntry,
            ReviewPlan,
            SystemGroup,
            SystemReviewResult,
        )

        plan = ReviewPlan(
            system_groups=[SystemGroup(name="g1", files=["a.py"], conventions="", review_focus="")],
            cross_cutting_concerns=[],
            file_manifest=[FileEntry(path="a.py", change_type="modified")],
        )
        findings = [SystemReviewResult(system_group="g1", findings=[], files_explored=["a.py"])]

        messages = _build_writer_messages(findings, plan, "diff", "desc")
        content = messages[0]["content"]

        assert "Dismissed Findings" not in content


# ---------------------------------------------------------------------------
# _node_validate_blockings
# ---------------------------------------------------------------------------


class TestNodeValidateBlockings:
    """Tests for the BLOCKING finding validator node."""

    @pytest.mark.asyncio
    async def test_no_blockings_skips_validation(self) -> None:
        """Node returns empty validation when no BLOCKING findings exist."""
        from argus.models import Finding, ReviewResponse, RiskLevel, Severity, Verdict

        response = ReviewResponse(
            verdict=Verdict.APPROVE,
            risk_level=RiskLevel.LOW,
            review_comment="## Code Review — Round 1\n\n**Verdict**: APPROVE",
            findings=[
                Finding(
                    severity=Severity.SUGGESTION, category="style", description="Use snake_case"
                ),
            ],
        )
        state = {"response": response.model_dump(), "diff": "diff"}

        with patch(_GH_CLIENT_CLASS, return_value=MagicMock()):
            _reload_graph()
            from argus.graph import _node_validate_blockings

            result = await _node_validate_blockings(state, {})

        assert result["validation"] == {}

    @pytest.mark.asyncio
    async def test_all_blockings_rejected_flips_to_approve(self) -> None:
        """When validator rejects all BLOCKINGs, verdict flips to APPROVE."""
        from argus.models import Finding, ReviewResponse, RiskLevel, Severity, Verdict
        from argus.pipeline_models import (
            FindingValidationItem,
            FindingValidationResult,
            ValidationVerdict,
        )

        response = ReviewResponse(
            verdict=Verdict.BLOCKING,
            risk_level=RiskLevel.HIGH,
            review_comment="## Code Review — Round 1\n\n**Verdict**: BLOCKING",
            findings=[
                Finding(
                    severity=Severity.BLOCKING,
                    category="security",
                    file="iam.tf",
                    line=68,
                    description="Missing IAM permission",
                ),
                Finding(severity=Severity.SUGGESTION, category="style", description="Naming"),
            ],
        )
        state = {"response": response.model_dump(), "diff": "diff"}

        mock_validation = FindingValidationResult(
            items=[
                FindingValidationItem(
                    index=0,
                    verdict=ValidationVerdict.REJECTED,
                    evidence="permission exists at iam.tf:68",
                ),
            ],
            cost_usd=0.05,
        )

        _reload_graph()
        from argus.graph import _node_validate_blockings

        with (
            patch(_GH_CLIENT_CLASS, return_value=MagicMock()),
            patch(
                "argus.graph.run_blocking_validator_session",
                new_callable=AsyncMock,
                return_value=(mock_validation, None),
            ),
            patch("argus.graph.get_settings", return_value=MagicMock()),
        ):
            result = await _node_validate_blockings(state, {})

        updated = ReviewResponse.model_validate(result["response"])
        assert updated.verdict == Verdict.APPROVE
        assert len(updated.dropped_findings) == 1
        assert updated.dropped_findings[0].finding.description == "Missing IAM permission"
        # SUGGESTION should survive
        assert any(f.description == "Naming" for f in updated.findings)
        # BLOCKING should be removed
        assert not any(f.description == "Missing IAM permission" for f in updated.findings)

    @pytest.mark.asyncio
    async def test_some_blockings_confirmed_stays_blocking(self) -> None:
        """When validator confirms some BLOCKINGs, verdict stays BLOCKING."""
        from argus.models import Finding, ReviewResponse, RiskLevel, Severity, Verdict
        from argus.pipeline_models import (
            FindingValidationItem,
            FindingValidationResult,
            ValidationVerdict,
        )

        response = ReviewResponse(
            verdict=Verdict.BLOCKING,
            risk_level=RiskLevel.HIGH,
            review_comment="## Code Review — Round 1\n\n**Verdict**: BLOCKING",
            findings=[
                Finding(
                    severity=Severity.BLOCKING,
                    category="security",
                    file="auth.py",
                    description="SQL injection",
                ),
                Finding(
                    severity=Severity.BLOCKING,
                    category="infra",
                    file="iam.tf",
                    description="Missing permission",
                ),
            ],
        )
        state = {"response": response.model_dump(), "diff": "diff"}

        mock_validation = FindingValidationResult(
            items=[
                FindingValidationItem(
                    index=0, verdict=ValidationVerdict.CONFIRMED, evidence="confirmed"
                ),
                FindingValidationItem(
                    index=1, verdict=ValidationVerdict.REJECTED, evidence="false positive"
                ),
            ],
            cost_usd=0.05,
        )

        _reload_graph()
        from argus.graph import _node_validate_blockings

        with (
            patch(_GH_CLIENT_CLASS, return_value=MagicMock()),
            patch(
                "argus.graph.run_blocking_validator_session",
                new_callable=AsyncMock,
                return_value=(mock_validation, None),
            ),
            patch("argus.graph.get_settings", return_value=MagicMock()),
        ):
            result = await _node_validate_blockings(state, {})

        updated = ReviewResponse.model_validate(result["response"])
        assert updated.verdict == Verdict.BLOCKING
        assert len(updated.dropped_findings) == 1
        assert len([f for f in updated.findings if f.severity == Severity.BLOCKING]) == 1

    @pytest.mark.asyncio
    async def test_agent_run_data_propagated_when_returned(self) -> None:
        """When validator returns AgentRunData, it appears in state agent_runs."""
        from argus.models import Finding, ReviewResponse, RiskLevel, Severity, Verdict
        from argus.pipeline_models import (
            AgentRunData,
            FindingValidationItem,
            FindingValidationResult,
            ValidationVerdict,
        )

        response = ReviewResponse(
            verdict=Verdict.BLOCKING,
            risk_level=RiskLevel.HIGH,
            review_comment="## Code Review — Round 1\n\n**Verdict**: BLOCKING",
            findings=[
                Finding(
                    severity=Severity.BLOCKING,
                    category="security",
                    file="auth.py",
                    description="SQL injection",
                ),
            ],
        )
        state = {"response": response.model_dump(), "diff": "diff"}

        mock_validation = FindingValidationResult(
            items=[
                FindingValidationItem(
                    index=0, verdict=ValidationVerdict.CONFIRMED, evidence="confirmed"
                ),
            ],
            cost_usd=0.05,
        )
        mock_agent_run_data = AgentRunData(
            agent_name="blocking_validator",
            agent_type="blocking_validator",
            model="claude-sonnet-4-6",
            cost_usd=0.05,
            duration_seconds=2.0,
        )

        _reload_graph()
        from argus.graph import _node_validate_blockings

        with (
            patch(_GH_CLIENT_CLASS, return_value=MagicMock()),
            patch(
                "argus.graph.run_blocking_validator_session",
                new_callable=AsyncMock,
                return_value=(mock_validation, mock_agent_run_data),
            ),
            patch("argus.graph.get_settings", return_value=MagicMock()),
        ):
            result = await _node_validate_blockings(state, {})

        assert "agent_runs" in result
        assert len(result["agent_runs"]) == 1
        assert result["agent_runs"][0]["agent_name"] == "blocking_validator"
        assert result["agent_runs"][0]["agent_type"] == "blocking_validator"
        assert result["agent_runs"][0]["cost_usd"] == 0.05


# ---------------------------------------------------------------------------
# _apply_dismissals
# ---------------------------------------------------------------------------


class TestApplyDismissals:
    """Tests for LLM-based dismiss matching."""

    @pytest.mark.asyncio
    async def test_empty_dismissals_returns_all_findings(self) -> None:
        _reload_graph()
        from argus.graph import _apply_dismissals

        findings = [PriorFinding(severity="BLOCKING", description="issue 1")]
        remaining, matched = await _apply_dismissals(findings, [])
        assert remaining == findings
        assert matched == []

    @pytest.mark.asyncio
    async def test_successful_match(self) -> None:
        """LLM structured output matches one dismiss to one finding."""
        _reload_graph()
        from argus.graph import _apply_dismissals

        findings = [
            PriorFinding(severity="BLOCKING", file="auth.py", description="SQL injection"),
            PriorFinding(severity="BLOCKING", file="graph.py", description="except Exception"),
        ]
        dismissals = [
            DismissedFinding(
                description="except Exception",
                dismissed_by="user1",
                reason="intentional",
            ),
        ]

        # Mock the _get_llm -> with_structured_output chain
        from pydantic import BaseModel

        class DismissMatch(BaseModel):
            dismiss_index: int
            finding_index: int

        class DismissMatches(BaseModel):
            matches: list[DismissMatch]

        mock_result = DismissMatches(matches=[DismissMatch(dismiss_index=0, finding_index=1)])

        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value=mock_result)

        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured)

        with patch("argus.graph._get_llm", return_value=mock_llm):
            remaining, matched = await _apply_dismissals(findings, dismissals)

        assert len(remaining) == 1
        assert remaining[0].description == "SQL injection"
        assert len(matched) == 1
        assert matched[0].description == "except Exception"
        assert matched[0].dismissed_by == "user1"

    @pytest.mark.asyncio
    async def test_out_of_range_indices_ignored(self) -> None:
        """Matches with out-of-range indices are silently skipped."""
        _reload_graph()
        from argus.graph import _apply_dismissals

        from pydantic import BaseModel

        class DismissMatch(BaseModel):
            dismiss_index: int
            finding_index: int

        class DismissMatches(BaseModel):
            matches: list[DismissMatch]

        findings = [PriorFinding(severity="BLOCKING", description="issue")]
        dismissals = [
            DismissedFinding(description="issue", dismissed_by="u", reason="r"),
        ]

        mock_result = DismissMatches(matches=[DismissMatch(dismiss_index=0, finding_index=99)])
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value=mock_result)
        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured)

        with patch("argus.graph._get_llm", return_value=mock_llm):
            remaining, matched = await _apply_dismissals(findings, dismissals)

        assert len(remaining) == 1
        assert matched == []


# ---------------------------------------------------------------------------
# _fetch_prior_review (JSON parsing and BLOCKING filter)
# ---------------------------------------------------------------------------


class TestFetchPriorReviewParsing:
    """Tests for _fetch_prior_review JSON parsing and finding filter logic."""

    @pytest.mark.asyncio
    async def test_filters_to_blocking_only(self) -> None:
        """Only BLOCKING findings are carried forward from prior review."""
        # ``_fetch_prior_review`` goes through
        # ``argus.storage.sql.select_latest_completed_round``
        # which reads ``result.mappings().first()`` — mock the full
        # mapping shape (every column the canonical SELECT returns).
        from datetime import datetime, timezone
        from uuid import UUID

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_mappings = MagicMock()
        mock_mappings.first.return_value = {
            "id": UUID("00000000-0000-0000-0000-000000000123"),
            "flow_run_id": None,
            "repo": "org/repo",
            "pr_number": 42,
            "verdict": "BLOCK",
            "risk_level": "HIGH",
            "blocking_count": 2,
            "suggestion_count": 1,
            "review_comment": "see findings",
            "result_json": {
                "findings": [
                    {
                        "severity": "BLOCKING",
                        "category": "security",
                        "file": "auth.py",
                        "description": "SQL injection",
                    },
                    {
                        "severity": "SUGGESTION",
                        "category": "style",
                        "file": "main.py",
                        "description": "Use snake_case",
                    },
                    {
                        "severity": "BLOCKING",
                        "category": "infra",
                        "file": "iam.tf",
                        "description": "Missing permission",
                    },
                ],
                "notes_for_next_round": "focus on auth",
            },
            "cost_usd": 0.1,
            "duration_seconds": 5.0,
            "reviewer_version": "v3",
            "orchestrator_model": "claude-opus",
            "subagent_model": "claude-sonnet",
            "sha": "aabbccdd1234",
            "base_ref": "main",
            "current_stage": "completed",
            "created_at": datetime(2026, 5, 19, tzinfo=timezone.utc),
            "prior_count": 3,
        }
        mock_result.mappings = MagicMock(return_value=mock_mappings)
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        _reload_graph()
        with (
            patch(_GH_CLIENT_CLASS, return_value=MagicMock()),
            patch("argus.storage.resolver.get_async_session_factory", return_value=mock_factory),
        ):
            from argus.graph import _fetch_prior_review

            result = await _fetch_prior_review("org/repo", 42)

        assert result is not None
        assert len(result.findings) == 2  # Only BLOCKINGs
        assert all(f.severity == "BLOCKING" for f in result.findings)
        assert result.findings[0].description == "SQL injection"
        assert result.findings[1].description == "Missing permission"
        assert result.round_number == 4  # prior_count (3) + 1
        assert result.notes_for_next_round == "focus on auth"

    @pytest.mark.asyncio
    async def test_no_prior_review_returns_none(self) -> None:
        """Returns None when no prior review exists."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_mappings = MagicMock()
        mock_mappings.first.return_value = None
        mock_result.mappings = MagicMock(return_value=mock_mappings)
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__ = MagicMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = MagicMock(return_value=False)

        with (
            patch(_GH_CLIENT_CLASS, return_value=MagicMock()),
            patch("argus.storage.resolver.get_async_session_factory", return_value=mock_factory),
        ):
            _reload_graph()
            from argus.graph import _fetch_prior_review

            result = await _fetch_prior_review("org/repo", 42)

        assert result is None

    @pytest.mark.asyncio
    async def test_db_error_returns_none(self) -> None:
        """Database errors are caught and return None (graceful degradation)."""
        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__ = MagicMock(
            side_effect=Exception("connection refused")
        )
        mock_factory.return_value.__aexit__ = MagicMock(return_value=False)

        with (
            patch(_GH_CLIENT_CLASS, return_value=MagicMock()),
            patch("argus.storage.resolver.get_async_session_factory", return_value=mock_factory),
        ):
            _reload_graph()
            from argus.graph import _fetch_prior_review

            result = await _fetch_prior_review("org/repo", 42)

        assert result is None
