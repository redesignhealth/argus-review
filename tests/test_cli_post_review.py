"""Tests for argus.cli._post_review: the --post / --commit-status write path.

This function has GitHub side effects (upsert a PR comment, set a commit
status) and previously had zero test coverage. Covers the six code paths
called out in Argus's own self-review: GitHubClientError -> sys.exit(1),
--post without --pr, upsert_pr_comment exception, SHA-from-PR fetch via
get_pull_request(), unresolvable SHA guard, and set_commit_status exception.
"""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest

from argus.cli import _post_review
from argus.models import ReviewResponse, RiskLevel, Verdict


def _response(verdict: Verdict = Verdict.APPROVE) -> ReviewResponse:
    return ReviewResponse(
        verdict=verdict,
        risk_level=RiskLevel.LOW,
        review_comment="## Argus Review\nLooks good.",
    )


def _args(**overrides: object) -> argparse.Namespace:
    base = {"post": False, "commit_status": False, "pr": None, "sha": None}
    base.update(overrides)
    return argparse.Namespace(**base)


class TestPostReviewClientConstruction:
    def test_exits_when_no_write_token_available(self) -> None:
        from argus.github_client import GitHubClientError

        with patch(
            "argus.github_client.GitHubClient.for_writes", side_effect=GitHubClientError("no token")
        ):
            with pytest.raises(SystemExit) as exc_info:
                _post_review("owner/repo", _args(post=True, pr=1), _response())
        assert exc_info.value.code == 1


class TestPostFlag:
    def test_post_without_pr_exits(self) -> None:
        mock_client = MagicMock()
        with patch("argus.github_client.GitHubClient.for_writes", return_value=mock_client):
            with pytest.raises(SystemExit) as exc_info:
                _post_review("owner/repo", _args(post=True, pr=None), _response())
        assert exc_info.value.code == 1
        mock_client.upsert_pr_comment.assert_not_called()

    def test_post_happy_path_upserts_comment(self) -> None:
        mock_client = MagicMock()
        response = _response()
        with patch("argus.github_client.GitHubClient.for_writes", return_value=mock_client):
            _post_review("owner/repo", _args(post=True, pr=42), response)
        mock_client.upsert_pr_comment.assert_called_once_with(
            "owner/repo", 42, response.review_comment
        )

    def test_post_exception_exits(self) -> None:
        mock_client = MagicMock()
        mock_client.upsert_pr_comment.side_effect = RuntimeError("GitHub API error")
        with patch("argus.github_client.GitHubClient.for_writes", return_value=mock_client):
            with pytest.raises(SystemExit) as exc_info:
                _post_review("owner/repo", _args(post=True, pr=42), _response())
        assert exc_info.value.code == 1


class TestCommitStatusFlag:
    def test_commit_status_with_explicit_sha(self) -> None:
        mock_client = MagicMock()
        response = _response(Verdict.APPROVE)
        with patch("argus.github_client.GitHubClient.for_writes", return_value=mock_client):
            _post_review("owner/repo", _args(commit_status=True, sha="abc1234"), response)
        mock_client.set_commit_status.assert_called_once()
        mock_client.get_pull_request.assert_not_called()
        call_args = mock_client.set_commit_status.call_args.args
        assert call_args[:2] == ("owner/repo", "abc1234")
        assert call_args[2] == "success"

    def test_commit_status_fetches_sha_from_pr(self) -> None:
        mock_client = MagicMock()
        mock_client.get_pull_request.return_value = {"head_sha": "deadbeef"}
        response = _response(Verdict.BLOCKING)
        with patch("argus.github_client.GitHubClient.for_writes", return_value=mock_client):
            _post_review("owner/repo", _args(commit_status=True, pr=7), response)
        mock_client.get_pull_request.assert_called_once_with("owner/repo", 7)
        call_args = mock_client.set_commit_status.call_args.args
        assert call_args[1] == "deadbeef"
        assert call_args[2] == "failure"

    def test_commit_status_unresolvable_sha_exits(self) -> None:
        mock_client = MagicMock()
        with patch("argus.github_client.GitHubClient.for_writes", return_value=mock_client):
            with pytest.raises(SystemExit) as exc_info:
                _post_review("owner/repo", _args(commit_status=True), _response())
        assert exc_info.value.code == 1
        mock_client.set_commit_status.assert_not_called()

    def test_commit_status_exception_exits(self) -> None:
        mock_client = MagicMock()
        mock_client.set_commit_status.side_effect = RuntimeError("GitHub API error")
        with patch("argus.github_client.GitHubClient.for_writes", return_value=mock_client):
            with pytest.raises(SystemExit) as exc_info:
                _post_review("owner/repo", _args(commit_status=True, sha="abc1234"), _response())
        assert exc_info.value.code == 1
