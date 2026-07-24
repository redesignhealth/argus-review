"""Unit tests for _fetch_pr_diff_and_description in graph.py."""

from __future__ import annotations

import importlib
import sys
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
            diff, description, head_sha = await fn(_make_request(pr_number=42))

        assert diff == "diff --git a/f.py b/f.py\n+new"
        assert description == "My PR description"
        assert head_sha == "abc123def456"
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
            _, description, _ = await fn(_make_request(pr_number=42))

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
            _, description, _ = await fn(_make_request(pr_number=42))

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
            diff, description, head_sha = await fn(
                _make_request(sha="abc123def456", base_ref="main")
            )

        assert diff == "diff --git a/f.py b/f.py\n+sha mode"
        assert description == ""
        assert head_sha == "abc123def456"
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
                return_value=("diff content", "description", pre_resolved_sha),
            ) as mock_fetch,
        ):
            await _node_fetch_diff(state, config)

        mock_fetch.assert_awaited_once()
        _, kwargs = mock_fetch.call_args
        assert kwargs.get("pre_resolved_head_sha") == pre_resolved_sha
