"""Unit tests for GitHubClient.get_checks_signal.

New read method (not covered by tests/test_github_client_write.py, which
is scoped to the write-capable methods) — exercises the pending/failing/
passing/unknown status-mapping branches and GitHubAPIError handling.
"""

from __future__ import annotations

import pytest
from pytest_httpx import HTTPXMock

from argus.github_client import GitHubClient


@pytest.fixture(autouse=True)
def _clear_github_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN_RO", raising=False)


def _mock_check_runs(httpx_mock: HTTPXMock, runs: list[dict]) -> None:
    httpx_mock.add_response(
        method="GET",
        url="https://api.github.com/repos/owner/repo/commits/abc123/check-runs",
        json={"total_count": len(runs), "check_runs": runs},
    )


class TestGetChecksSignal:
    def test_no_checks_returns_unknown(self, httpx_mock: HTTPXMock) -> None:
        _mock_check_runs(httpx_mock, [])
        client = GitHubClient(token="ro-token")
        assert client.get_checks_signal("owner/repo", "abc123") == "unknown"

    def test_all_completed_success_returns_passing(self, httpx_mock: HTTPXMock) -> None:
        _mock_check_runs(
            httpx_mock,
            [
                {"status": "completed", "conclusion": "success"},
                {"status": "completed", "conclusion": "neutral"},
            ],
        )
        client = GitHubClient(token="ro-token")
        assert client.get_checks_signal("owner/repo", "abc123") == "passing"

    def test_any_in_progress_returns_pending(self, httpx_mock: HTTPXMock) -> None:
        _mock_check_runs(
            httpx_mock,
            [
                {"status": "completed", "conclusion": "success"},
                {"status": "in_progress", "conclusion": None},
            ],
        )
        client = GitHubClient(token="ro-token")
        assert client.get_checks_signal("owner/repo", "abc123") == "pending"

    def test_queued_returns_pending(self, httpx_mock: HTTPXMock) -> None:
        _mock_check_runs(httpx_mock, [{"status": "queued", "conclusion": None}])
        client = GitHubClient(token="ro-token")
        assert client.get_checks_signal("owner/repo", "abc123") == "pending"

    @pytest.mark.parametrize("conclusion", ["failure", "timed_out", "action_required", "cancelled"])
    def test_failing_conclusions_return_failing(
        self, httpx_mock: HTTPXMock, conclusion: str
    ) -> None:
        _mock_check_runs(
            httpx_mock,
            [
                {"status": "completed", "conclusion": "success"},
                {"status": "completed", "conclusion": conclusion},
            ],
        )
        client = GitHubClient(token="ro-token")
        assert client.get_checks_signal("owner/repo", "abc123") == "failing"

    def test_completed_skipped_only_returns_passing(self, httpx_mock: HTTPXMock) -> None:
        _mock_check_runs(httpx_mock, [{"status": "completed", "conclusion": "skipped"}])
        client = GitHubClient(token="ro-token")
        assert client.get_checks_signal("owner/repo", "abc123") == "passing"

    def test_api_error_returns_unknown(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            method="GET",
            url="https://api.github.com/repos/owner/repo/commits/abc123/check-runs",
            status_code=404,
        )
        client = GitHubClient(token="ro-token")
        assert client.get_checks_signal("owner/repo", "abc123") == "unknown"
