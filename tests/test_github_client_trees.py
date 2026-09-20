"""Unit tests for GitHubClient.get_tree_blob_shas and SHA extraction (TECH-6633)."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from argus.github_client import GitHubAPIError, GitHubClient


@pytest.fixture(autouse=True)
def _clear_github_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN_RO", raising=False)


class TestGetTreeBlobShas:
    """Tests for GitHubClient.get_tree_blob_shas."""

    def test_returns_blob_entries_and_filters_out_trees(self) -> None:
        client = GitHubClient(token="dummy")
        with patch.object(client, "_request") as mock_req:
            mock_req.return_value = {
                "tree": [
                    {"path": "file1.txt", "type": "blob", "sha": "sha1"},
                    {"path": "dir1", "type": "tree", "sha": "sha_tree"},
                    {"path": "file2.py", "type": "blob", "sha": "sha2"},
                ],
                "truncated": False,
            }
            blob_shas, is_truncated = client.get_tree_blob_shas("owner/repo", "ref123")

        assert blob_shas == {"file1.txt": "sha1", "file2.py": "sha2"}
        assert is_truncated is False

    def test_truncated_propagates_true_and_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = GitHubClient(token="dummy")
        with patch.object(client, "_request") as mock_req, caplog.at_level(logging.WARNING):
            mock_req.return_value = {
                "tree": [{"path": "file1.txt", "type": "blob", "sha": "sha1"}],
                "truncated": True,
            }
            blob_shas, is_truncated = client.get_tree_blob_shas("owner/repo", "ref123")

        assert is_truncated is True
        assert blob_shas == {"file1.txt": "sha1"}
        assert any(
            "truncated" in record.message and "ref123" in record.message
            for record in caplog.records
        )

    def test_entries_missing_path_or_sha_skipped_without_error(self) -> None:
        client = GitHubClient(token="dummy")
        with patch.object(client, "_request") as mock_req:
            mock_req.return_value = {
                "tree": [
                    {"path": "valid.py", "type": "blob", "sha": "sha_valid"},
                    {"path": "", "type": "blob", "sha": "sha_empty_path"},
                    {"type": "blob", "sha": "sha_missing_path"},
                    {"path": "empty_sha.py", "type": "blob", "sha": ""},
                    {"path": "missing_sha.py", "type": "blob"},
                    "not_a_dict",
                    None,
                ],
                "truncated": False,
            }
            blob_shas, is_truncated = client.get_tree_blob_shas("owner/repo", "ref123")

        assert blob_shas == {"valid.py": "sha_valid"}
        assert is_truncated is False

    def test_recursive_query_param_passed(self) -> None:
        client = GitHubClient(token="dummy")
        with patch.object(client, "_request") as mock_req:
            mock_req.return_value = {"tree": [], "truncated": False}
            client.get_tree_blob_shas("owner/repo", "commit123")

        mock_req.assert_called_once_with(
            "GET",
            "/repos/owner/repo/git/trees/commit123",
            params={"recursive": "1"},
        )

    def test_github_api_error_propagates_uncaught(self) -> None:
        client = GitHubClient(token="dummy")
        with patch.object(client, "_request") as mock_req:
            mock_req.side_effect = GitHubAPIError(status_code=404, message="Not Found")
            with pytest.raises(GitHubAPIError):
                client.get_tree_blob_shas("owner/repo", "missing_ref")


class TestCompareAndPrFilesExtractSha:
    """Tests that get_compare_files and get_pr_files extract sha and previous_filename."""

    def test_get_compare_files_includes_sha_and_previous_filename(self) -> None:
        client = GitHubClient(token="dummy")
        with patch.object(client, "_request") as mock_req:
            mock_req.return_value = {
                "files": [
                    {
                        "filename": "new_file.py",
                        "status": "renamed",
                        "patch": "@@ ...",
                        "previous_filename": "old_file.py",
                        "sha": "blob_sha_123",
                    }
                ]
            }
            files, is_truncated = client.get_compare_files("owner/repo", "base", "head")

        assert is_truncated is False
        assert len(files) == 1
        assert files[0]["filename"] == "new_file.py"
        assert files[0]["previous_filename"] == "old_file.py"
        assert files[0]["sha"] == "blob_sha_123"

    def test_get_pr_files_includes_sha_and_previous_filename(self) -> None:
        client = GitHubClient(token="dummy")
        with patch.object(client, "_paginate") as mock_paginate:
            mock_paginate.return_value = [
                {
                    "filename": "pr_new.py",
                    "status": "renamed",
                    "patch": "",
                    "previous_filename": "pr_old.py",
                    "sha": "pr_sha_456",
                }
            ]
            files, is_truncated = client.get_pr_files("owner/repo", 42)

        assert is_truncated is False
        assert len(files) == 1
        assert files[0]["filename"] == "pr_new.py"
        assert files[0]["previous_filename"] == "pr_old.py"
        assert files[0]["sha"] == "pr_sha_456"
