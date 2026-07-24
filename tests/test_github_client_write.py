"""Tests for the additive GitHubClient write methods (--post,
--commit-status). These are new methods only — the read methods above them
in ``github_client.py`` are untouched and already covered elsewhere.
"""

from __future__ import annotations

import pytest
from pytest_httpx import HTTPXMock

from argus.github_client import GitHubClient, GitHubClientError


@pytest.fixture(autouse=True)
def _clear_github_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN_RO", raising=False)


class TestForWrites:
    def test_prefers_github_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "write-token")
        monkeypatch.setenv("GITHUB_TOKEN_RO", "ro-token")
        client = GitHubClient.for_writes()
        assert client.token == "write-token"
        assert client._using_ro_fallback is False

    def test_falls_back_to_github_token_ro(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GITHUB_TOKEN_RO", "ro-token")
        client = GitHubClient.for_writes()
        assert client.token == "ro-token"
        assert client._using_ro_fallback is True

    def test_raises_when_no_token_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(GitHubClientError):
            GitHubClient.for_writes()


class TestUpsertPrComment:
    def test_creates_new_comment_when_no_prior_marker(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            method="GET",
            url="https://api.github.com/repos/owner/repo/issues/42/comments?per_page=100&page=1",
            json=[{"id": 1, "body": "unrelated comment"}],
        )
        httpx_mock.add_response(
            method="POST",
            url="https://api.github.com/repos/owner/repo/issues/42/comments",
            json={"id": 99, "body": "tagged"},
        )
        client = GitHubClient(token="write-token")
        client._using_ro_fallback = False

        result = client.upsert_pr_comment("owner/repo", 42, "## Argus Review\nAPPROVE")

        assert result["id"] == 99
        post_request = httpx_mock.get_requests(method="POST")[0]
        assert b"<!-- argus-review: owner/repo#42 -->" in post_request.content

    def test_updates_existing_comment_by_marker(self, httpx_mock: HTTPXMock) -> None:
        marker = "<!-- argus-review: owner/repo#42 -->"
        httpx_mock.add_response(
            method="GET",
            url="https://api.github.com/repos/owner/repo/issues/42/comments?per_page=100&page=1",
            json=[{"id": 7, "body": f"{marker}\nold review body"}],
        )
        httpx_mock.add_response(
            method="PATCH",
            url="https://api.github.com/repos/owner/repo/issues/comments/7",
            json={"id": 7, "body": "updated"},
        )
        client = GitHubClient(token="write-token")
        client._using_ro_fallback = False

        result = client.upsert_pr_comment("owner/repo", 42, "## Argus Review\nBLOCKING")

        assert result["id"] == 7
        patch_request = httpx_mock.get_requests(method="PATCH")[0]
        assert marker.encode() in patch_request.content
        assert b"BLOCKING" in patch_request.content

    def test_marker_is_scoped_to_repo_and_pr(self, httpx_mock: HTTPXMock) -> None:
        """A comment tagged for a different repo/pr must not be matched."""
        other_marker = "<!-- argus-review: owner/repo#99 -->"
        httpx_mock.add_response(
            method="GET",
            url="https://api.github.com/repos/owner/repo/issues/42/comments?per_page=100&page=1",
            json=[{"id": 1, "body": f"{other_marker}\nsome other pr's review"}],
        )
        httpx_mock.add_response(
            method="POST",
            url="https://api.github.com/repos/owner/repo/issues/42/comments",
            json={"id": 100, "body": "new"},
        )
        client = GitHubClient(token="write-token")
        client._using_ro_fallback = False

        result = client.upsert_pr_comment("owner/repo", 42, "review body")
        assert result["id"] == 100

    def test_ro_fallback_403_raises_clear_error(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            method="GET",
            url="https://api.github.com/repos/owner/repo/issues/42/comments?per_page=100&page=1",
            json=[],
        )
        httpx_mock.add_response(
            method="POST",
            url="https://api.github.com/repos/owner/repo/issues/42/comments",
            status_code=403,
            json={"message": "Resource not accessible by personal access token"},
        )
        client = GitHubClient(token="ro-token")
        client._using_ro_fallback = True

        with pytest.raises(Exception) as exc_info:
            client.upsert_pr_comment("owner/repo", 42, "review body")
        assert "GITHUB_TOKEN_RO lacks write scope; set GITHUB_TOKEN" in str(exc_info.value)

    def test_paginates_past_first_page_to_find_marker(self, httpx_mock: HTTPXMock) -> None:
        """B4 regression: >100 comments must not create a duplicate comment."""
        marker = "<!-- argus-review: owner/repo#42 -->"
        page_one = [{"id": i, "body": "unrelated"} for i in range(100)]
        httpx_mock.add_response(
            method="GET",
            url="https://api.github.com/repos/owner/repo/issues/42/comments?per_page=100&page=1",
            json=page_one,
        )
        httpx_mock.add_response(
            method="GET",
            url="https://api.github.com/repos/owner/repo/issues/42/comments?per_page=100&page=2",
            json=[{"id": 200, "body": f"{marker}\nold review body"}],
        )
        httpx_mock.add_response(
            method="PATCH",
            url="https://api.github.com/repos/owner/repo/issues/comments/200",
            json={"id": 200, "body": "updated"},
        )
        client = GitHubClient(token="write-token")
        client._using_ro_fallback = False

        result = client.upsert_pr_comment("owner/repo", 42, "## Argus Review\nBLOCKING")

        assert result["id"] == 200
        # No POST (new comment) should have been issued.
        assert httpx_mock.get_requests(method="POST") == []

    @pytest.mark.parametrize(
        "repo",
        [
            "owner/repo/../secret-repo",
            "owner/../repo",
            "../repo",
            "owner/repo with space",
            "owner/repo-->",
        ],
    )
    def test_rejects_invalid_repo(self, repo: str) -> None:
        client = GitHubClient(token="write-token")
        with pytest.raises(ValueError, match="Invalid repo format"):
            client.upsert_pr_comment(repo, 42, "body")


class TestSetCommitStatus:
    def test_success_state_on_approve(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            method="POST",
            url="https://api.github.com/repos/owner/repo/statuses/abc123",
            json={"state": "success", "context": "argus/review"},
        )
        client = GitHubClient(token="write-token")
        client._using_ro_fallback = False

        result = client.set_commit_status("owner/repo", "abc123", "success", "Argus: APPROVE")

        assert result["state"] == "success"
        request = httpx_mock.get_requests(method="POST")[0]
        assert b'"context":"argus/review"' in request.content or b'"context": "argus/review"' in (
            request.content
        )

    def test_description_truncated_to_140_chars(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            method="POST",
            url="https://api.github.com/repos/owner/repo/statuses/abc123",
            json={"state": "failure"},
        )
        client = GitHubClient(token="write-token")
        client._using_ro_fallback = False
        long_description = "x" * 500

        client.set_commit_status("owner/repo", "abc123", "failure", long_description)

        import json as json_mod

        request = httpx_mock.get_requests(method="POST")[0]
        body = json_mod.loads(request.content)
        assert len(body["description"]) == 140

    def test_ro_fallback_403_raises_clear_error(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            method="POST",
            url="https://api.github.com/repos/owner/repo/statuses/abc123",
            status_code=403,
            json={"message": "Resource not accessible by personal access token"},
        )
        client = GitHubClient(token="ro-token")
        client._using_ro_fallback = True

        with pytest.raises(Exception) as exc_info:
            client.set_commit_status("owner/repo", "abc123", "failure", "Argus: BLOCKING")
        assert "GITHUB_TOKEN_RO lacks write scope; set GITHUB_TOKEN" in str(exc_info.value)

    @pytest.mark.parametrize(
        "sha",
        [
            "abc/../../../../../../user/keys",
            "abc123/../other",
            "not-hex!",
            "",
            "ab",
        ],
    )
    def test_rejects_invalid_sha(self, sha: str) -> None:
        client = GitHubClient(token="write-token")
        with pytest.raises(ValueError, match="Invalid sha format"):
            client.set_commit_status("owner/repo", sha, "success", "Argus: APPROVE")

    def test_rejects_invalid_repo(self) -> None:
        client = GitHubClient(token="write-token")
        with pytest.raises(ValueError, match="Invalid repo format"):
            client.set_commit_status("owner/repo/../secret-repo", "abc123", "success", "desc")
