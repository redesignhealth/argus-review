"""Unit tests for _fetch_pr_diff_and_description in graph.py."""

from __future__ import annotations

import importlib
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from argus.models import ReviewRequest

_GH_CLIENT_CLASS = "argus.github_client.GitHubClient"


def _make_request(
    *,
    repo: str = "org/repo",
    pr_number: int = 0,
    sha: str | None = None,
    base_ref: str | None = None,
) -> ReviewRequest:
    """Build a ReviewRequest for testing."""
    return ReviewRequest(repo=repo, pr_number=pr_number, sha=sha, base_ref=base_ref)


def _reload_graph():
    """Force-reload graph module to pick up patches on lazy imports."""
    mod = "argus.graph"
    if mod in sys.modules:
        importlib.reload(sys.modules[mod])
    from argus.graph import _fetch_pr_diff_and_description

    return _fetch_pr_diff_and_description


class TestFetchPrDiffPRMode:
    """Tests for _fetch_pr_diff_and_description in PR-number mode."""

    @pytest.mark.asyncio
    async def test_pr_mode_calls_get_compare_diff(self) -> None:
        """PR mode uses get_compare_diff with base_branch and head_sha from PR metadata."""
        mock_gh = MagicMock()
        mock_gh.get_pull_request.return_value = {
            "base_branch": "main",
            "head_sha": "abc123def456",
            "body": "My PR description",
        }
        mock_gh.get_compare_diff.return_value = "diff --git a/f.py b/f.py\n+new"

        with patch(_GH_CLIENT_CLASS, return_value=mock_gh):
            fn = _reload_graph()
            diff, description, head_sha, base_branch = await fn(_make_request(pr_number=42))

        assert diff == "diff --git a/f.py b/f.py\n+new"
        assert description == "My PR description"
        assert head_sha == "abc123def456"
        assert base_branch == "main"
        mock_gh.get_compare_diff.assert_called_once_with(
            "org/repo", "main", "abc123def456", max_lines=5000
        )

    @pytest.mark.asyncio
    async def test_pr_mode_extracts_description(self) -> None:
        """PR mode returns the PR body as description."""
        mock_gh = MagicMock()
        mock_gh.get_pull_request.return_value = {
            "base_branch": "main",
            "head_sha": "abcdef1234567890",
            "body": "## Summary\nThis fixes a bug.",
        }
        mock_gh.get_compare_diff.return_value = "diff content"

        with patch(_GH_CLIENT_CLASS, return_value=mock_gh):
            fn = _reload_graph()
            _, description, _, _ = await fn(_make_request(pr_number=42))

        assert description == "## Summary\nThis fixes a bug."

    @pytest.mark.asyncio
    async def test_pr_mode_description_none_becomes_empty(self) -> None:
        """PR mode handles None body gracefully."""
        mock_gh = MagicMock()
        mock_gh.get_pull_request.return_value = {
            "base_branch": "main",
            "head_sha": "abcdef1234567890",
            "body": None,
        }
        mock_gh.get_compare_diff.return_value = "diff"

        with patch(_GH_CLIENT_CLASS, return_value=mock_gh):
            fn = _reload_graph()
            _, description, _, _ = await fn(_make_request(pr_number=42))

        assert description == ""

    @pytest.mark.asyncio
    async def test_pr_mode_rejects_dotdot_in_base_branch(self) -> None:
        """PR mode rejects '..' injection in base_branch."""
        mock_gh = MagicMock()
        mock_gh.get_pull_request.return_value = {
            "base_branch": "main/../evil",
            "head_sha": "abcdef1234567890",
            "body": "",
        }

        with patch(_GH_CLIENT_CLASS, return_value=mock_gh):
            fn = _reload_graph()
            with pytest.raises(ValueError, match="Invalid base_branch"):
                await fn(_make_request(pr_number=42))

    @pytest.mark.asyncio
    async def test_pr_mode_rejects_non_hex_head_sha(self) -> None:
        """PR mode rejects non-hex characters in head_sha."""
        mock_gh = MagicMock()
        mock_gh.get_pull_request.return_value = {
            "base_branch": "main",
            "head_sha": "not-a-valid-sha!",
            "body": "",
        }

        with patch(_GH_CLIENT_CLASS, return_value=mock_gh):
            fn = _reload_graph()
            with pytest.raises(ValueError, match="Invalid head_sha"):
                await fn(_make_request(pr_number=42))


class TestFetchPrDiffSHAMode:
    """Tests for _fetch_pr_diff_and_description in SHA mode."""

    @pytest.mark.asyncio
    async def test_sha_mode_calls_get_compare_diff(self) -> None:
        """SHA mode calls get_compare_diff with request.base_ref and request.sha."""
        mock_gh = MagicMock()
        mock_gh.get_compare_diff.return_value = "diff --git a/f.py b/f.py\n+sha mode"

        with patch(_GH_CLIENT_CLASS, return_value=mock_gh):
            fn = _reload_graph()
            diff, description, head_sha, base_branch = await fn(
                _make_request(sha="abc123def456", base_ref="main")
            )

        assert diff == "diff --git a/f.py b/f.py\n+sha mode"
        assert description == ""
        assert head_sha == "abc123def456"
        assert base_branch == "main"
        mock_gh.get_compare_diff.assert_called_once_with(
            "org/repo", "main", "abc123def456", max_lines=5000
        )

    @pytest.mark.asyncio
    async def test_sha_mode_validates_sha_hex(self) -> None:
        """SHA mode rejects non-hex sha."""
        mock_gh = MagicMock()

        with patch(_GH_CLIENT_CLASS, return_value=mock_gh):
            fn = _reload_graph()
            with pytest.raises(ValueError, match="Invalid sha"):
                await fn(_make_request(sha="not-hex!", base_ref="main"))

    @pytest.mark.asyncio
    async def test_sha_mode_validates_base_ref(self) -> None:
        """SHA mode rejects '..' in base_ref."""
        mock_gh = MagicMock()

        with patch(_GH_CLIENT_CLASS, return_value=mock_gh):
            fn = _reload_graph()
            with pytest.raises(ValueError, match="Invalid base_ref"):
                await fn(_make_request(sha="abc1234", base_ref="main/../evil"))


class TestFetchPrDiffErrorCases:
    """Tests for error cases in _fetch_pr_diff_and_description."""

    @pytest.mark.asyncio
    async def test_raises_when_neither_pr_nor_sha(self) -> None:
        """Raises ValueError when neither pr_number nor sha+base_ref provided."""
        mock_gh = MagicMock()

        with patch(_GH_CLIENT_CLASS, return_value=mock_gh):
            fn = _reload_graph()
            with pytest.raises(ValueError, match="must have pr_number or both sha and base_ref"):
                await fn(_make_request())


class TestNodeFetchDiffPreResolvedSha:
    """Tests for _node_fetch_diff using pre_resolved_head_sha from config."""

    @pytest.mark.asyncio
    async def test_node_fetch_diff_uses_pre_resolved_head_sha(self) -> None:
        """_node_fetch_diff passes pre_resolved_head_sha from config to _fetch_pr_diff_and_description."""
        from unittest.mock import AsyncMock, patch

        import importlib
        import sys

        mod = "argus.graph"
        if mod in sys.modules:
            importlib.reload(sys.modules[mod])
        from argus.graph import _node_fetch_diff

        pre_resolved_sha = "aabbccdd" * 5  # 40 hex chars

        config = {"configurable": {"head_sha": pre_resolved_sha}}
        state = {
            "request": _make_request(pr_number=42).model_dump(),
        }

        with (
            patch(
                "argus.graph._fetch_prior_review",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "argus.graph._fetch_dismissed_findings",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "argus.graph._fetch_pr_diff_and_description",
                new_callable=AsyncMock,
                return_value=("diff content", "description", pre_resolved_sha, "main"),
            ) as mock_fetch,
        ):
            await _node_fetch_diff(state, config)

        mock_fetch.assert_awaited_once()
        _, kwargs = mock_fetch.call_args
        assert kwargs.get("pre_resolved_head_sha") == pre_resolved_sha


class TestFetchFullPrChangedFiles:
    """Tests for _fetch_full_pr_changed_files full-PR path resolution and fallback."""

    @pytest.mark.asyncio
    async def test_round1_calls_compare_files_api(self) -> None:
        """On round 1, queries GitHub API to get the full file list across the PR (avoiding diff truncation)."""
        from argus.graph import _fetch_full_pr_changed_files

        mock_gh = MagicMock()
        mock_gh.get_compare_files.return_value = (
            [
                {"filename": ".argus/bench.toml", "status": "added"},
                {"filename": "src/main.py", "status": "added"},
            ],
            False,
        )
        diff = "diff --git a/src/main.py b/src/main.py\n+code\n"
        with patch(_GH_CLIENT_CLASS, return_value=mock_gh):
            files, unconfirmed = await _fetch_full_pr_changed_files(
                _make_request(pr_number=42),
                base_branch="main",
                head_sha="head1234",
                round_diff=diff,
                prior_sha=None,
            )

        assert [f["filename"] for f in files] == [".argus/bench.toml", "src/main.py"]
        assert unconfirmed is None
        mock_gh.get_compare_files.assert_called_once_with("org/repo", "main", "head1234")

    @pytest.mark.asyncio
    async def test_round2_calls_compare_files_api(self) -> None:
        """On round 2+ (prior_sha is set), calls gh.get_compare_files with base_branch and head_sha."""
        from argus.graph import _fetch_full_pr_changed_files

        mock_gh = MagicMock()
        mock_gh.get_compare_files.return_value = (
            [
                {"filename": ".argus/bench.toml", "status": "modified"},
                {"filename": "src/lib.py", "status": "modified"},
            ],
            False,
        )

        round_diff = "diff --git a/src/lib.py b/src/lib.py\n+lib\n"
        with patch(_GH_CLIENT_CLASS, return_value=mock_gh):
            files, unconfirmed = await _fetch_full_pr_changed_files(
                _make_request(pr_number=42),
                base_branch="main",
                head_sha="head1234",
                round_diff=round_diff,
                prior_sha="prior1234",
            )

        assert [f["filename"] for f in files] == [".argus/bench.toml", "src/lib.py"]
        assert unconfirmed is None
        mock_gh.get_compare_files.assert_called_once_with("org/repo", "main", "head1234")

    @pytest.mark.asyncio
    async def test_round2_fails_closed_on_api_error(self, caplog: pytest.LogCaptureFixture) -> None:
        """On API failure, fails CLOSED: logs a WARNING and returns an unconfirmed reason string."""
        import logging
        from argus.graph import _fetch_full_pr_changed_files

        mock_gh = MagicMock()
        mock_gh.get_compare_files.side_effect = RuntimeError("GitHub compare rate-limited")

        round_diff = "diff --git a/src/fallback.py b/src/fallback.py\n+fallback\n"
        with patch(_GH_CLIENT_CLASS, return_value=mock_gh), caplog.at_level(logging.WARNING):
            files, unconfirmed = await _fetch_full_pr_changed_files(
                _make_request(pr_number=42),
                base_branch="main",
                head_sha="head1234",
                round_diff=round_diff,
                prior_sha="prior1234",
            )

        assert [f["filename"] for f in files] == ["src/fallback.py"]
        assert unconfirmed is not None
        assert "GitHub API comparison failed" in unconfirmed
        assert any(
            "failing closed to prevent unverified bench configuration changes" in record.message
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_truncated_compare_list_fails_closed(self) -> None:
        """When compare API hits cap and cannot paginate, fails closed with unconfirmed reason."""
        from argus.graph import _fetch_full_pr_changed_files

        mock_gh = MagicMock()
        mock_gh.get_compare_files.return_value = (
            [{"filename": f"file_{i}.py", "status": "modified"} for i in range(300)],
            True,  # is_truncated
        )
        # pr_files also truncated
        mock_gh.get_pr_files.return_value = (
            [{"filename": f"file_{i}.py", "status": "modified"} for i in range(3000)],
            True,  # is_truncated
        )

        with patch(_GH_CLIENT_CLASS, return_value=mock_gh):
            files, unconfirmed = await _fetch_full_pr_changed_files(
                _make_request(pr_number=42),
                base_branch="main",
                head_sha="head1234",
                round_diff="diff",
                prior_sha=None,
            )

        assert len(files) == 3000
        assert unconfirmed is not None
        assert "exceeded maximum file limit" in unconfirmed

    @pytest.mark.asyncio
    async def test_previous_filename_and_sha_plumbed_through(self) -> None:
        """previous_filename and sha from get_compare_files are plumbed into parsed_files."""
        from argus.graph import _fetch_full_pr_changed_files

        mock_gh = MagicMock()
        mock_gh.get_compare_files.return_value = (
            [
                {
                    "filename": "new/path.py",
                    "previous_filename": "old/path.py",
                    "status": "renamed",
                    "patch": "",
                    "sha": "blob123",
                }
            ],
            False,
        )
        mock_gh.get_compare_metadata.return_value = MagicMock(merge_base_sha="mb123")
        mock_gh.get_tree_blob_shas.return_value = ({"old/path.py": "blob123"}, False)

        with patch(_GH_CLIENT_CLASS, return_value=mock_gh):
            files, unconfirmed = await _fetch_full_pr_changed_files(
                _make_request(pr_number=42),
                base_branch="main",
                head_sha="head1234",
                round_diff="diff",
                prior_sha=None,
            )

        assert len(files) == 1
        assert files[0]["filename"] == "new/path.py"
        assert files[0]["previous_filename"] == "old/path.py"
        assert files[0]["sha"] == "blob123"
        assert files[0]["content_identical_rename"] is True
        assert unconfirmed is None

    @pytest.mark.asyncio
    async def test_pr_files_pagination_preserves_rename_fields(self) -> None:
        """PR-files pagination rebuilds parsed_files with previous_filename, sha, and annotates renames."""
        from argus.graph import _fetch_full_pr_changed_files

        mock_gh = MagicMock()
        # compare files is truncated to trigger PR pagination
        mock_gh.get_compare_files.return_value = (
            [{"filename": f"file_{i}.py", "status": "modified"} for i in range(300)],
            True,
        )
        # get_pr_files returns paginated files including a pure rename
        mock_gh.get_pr_files.return_value = (
            [
                {
                    "filename": "new/renamed.py",
                    "previous_filename": "old/renamed.py",
                    "status": "renamed",
                    "patch": "",
                    "sha": "sha456",
                }
            ],
            False,
        )
        mock_gh.get_compare_metadata.return_value = MagicMock(merge_base_sha="mb456")
        mock_gh.get_tree_blob_shas.return_value = ({"old/renamed.py": "sha456"}, False)

        with patch(_GH_CLIENT_CLASS, return_value=mock_gh):
            files, unconfirmed = await _fetch_full_pr_changed_files(
                _make_request(pr_number=42),
                base_branch="main",
                head_sha="head1234",
                round_diff="diff",
                prior_sha=None,
            )

        assert unconfirmed is None
        assert len(files) == 1
        assert files[0]["filename"] == "new/renamed.py"
        assert files[0]["previous_filename"] == "old/renamed.py"
        assert files[0]["sha"] == "sha456"
        assert files[0]["content_identical_rename"] is True


class TestNodeFetchDiffBenchConfig:
    """Tests for _node_fetch_diff bench_config_changes state population."""

    @pytest.mark.asyncio
    async def test_node_fetch_diff_detects_bench_file_in_round1(self) -> None:
        from unittest.mock import AsyncMock, patch
        from argus.graph import _node_fetch_diff

        diff = "diff --git a/.argus/bench.toml b/.argus/bench.toml\n+model = 'gemini'\n"
        state = {"request": _make_request(pr_number=42).model_dump()}
        config: dict[str, Any] = {"configurable": {}}

        mock_gh = MagicMock()
        mock_gh.get_compare_files.return_value = (
            [{"filename": ".argus/bench.toml", "status": "added", "patch": "+model = 'gemini'\n"}],
            False,
        )

        with (
            patch("argus.graph._fetch_prior_review", new_callable=AsyncMock, return_value=None),
            patch("argus.graph._fetch_dismissed_findings", new_callable=AsyncMock, return_value=[]),
            patch(
                "argus.graph._fetch_pr_diff_and_description",
                new_callable=AsyncMock,
                return_value=(diff, "", "head1234", "main"),
            ),
            patch(_GH_CLIENT_CLASS, return_value=mock_gh),
        ):
            result = await _node_fetch_diff(state, config)

        assert result["bench_config_changes"] == [".argus/bench.toml"]
        assert result["bench_config_unconfirmed"] is None

    @pytest.mark.asyncio
    async def test_node_fetch_diff_clean_when_no_bench_config(self) -> None:
        from unittest.mock import AsyncMock, patch
        from argus.graph import _node_fetch_diff

        diff = "diff --git a/src/app.py b/src/app.py\n+print('hello')\n"
        state = {"request": _make_request(pr_number=42).model_dump()}
        config: dict[str, Any] = {"configurable": {}}

        mock_gh = MagicMock()
        mock_gh.get_compare_files.return_value = (
            [{"filename": "src/app.py", "status": "modified", "patch": "+print('hello')\n"}],
            False,
        )

        with (
            patch("argus.graph._fetch_prior_review", new_callable=AsyncMock, return_value=None),
            patch("argus.graph._fetch_dismissed_findings", new_callable=AsyncMock, return_value=[]),
            patch(
                "argus.graph._fetch_pr_diff_and_description",
                new_callable=AsyncMock,
                return_value=(diff, "", "head1234", "main"),
            ),
            patch(_GH_CLIENT_CLASS, return_value=mock_gh),
        ):
            result = await _node_fetch_diff(state, config)

        assert result["bench_config_changes"] == []
        assert result["bench_config_unconfirmed"] is None

    @pytest.mark.asyncio
    async def test_node_fetch_diff_excludes_deleted_bench_config(self) -> None:
        """A PR that deletes an existing .argus/bench.toml does not block."""
        from unittest.mock import AsyncMock, patch
        from argus.graph import _node_fetch_diff

        diff = "diff --git a/.argus/bench.toml b/.argus/bench.toml\ndeleted file mode 100644\n--- a/.argus/bench.toml\n+++ /dev/null\n"
        state = {"request": _make_request(pr_number=42).model_dump()}
        config: dict[str, Any] = {"configurable": {}}

        mock_gh = MagicMock()
        mock_gh.get_compare_files.return_value = (
            [{"filename": ".argus/bench.toml", "status": "removed"}],
            False,
        )

        with (
            patch("argus.graph._fetch_prior_review", new_callable=AsyncMock, return_value=None),
            patch("argus.graph._fetch_dismissed_findings", new_callable=AsyncMock, return_value=[]),
            patch(
                "argus.graph._fetch_pr_diff_and_description",
                new_callable=AsyncMock,
                return_value=(diff, "", "head1234", "main"),
            ),
            patch(_GH_CLIENT_CLASS, return_value=mock_gh),
        ):
            result = await _node_fetch_diff(state, config)

        assert result["bench_config_changes"] == []
        assert result["bench_config_unconfirmed"] is None
