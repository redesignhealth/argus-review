"""Unit tests for the deterministic catch-up-merge preflight gate.

Covers:
  (a) merge-only delta since prior APPROVE  → _node_preflight routes lite
  (b) new authored (non-merge) commit since prior → LLM path runs (not short-circuited)
  (c) round-1 (no prior review)             → gate is not invoked, LLM path runs
  (d) _is_catchup_merge_only helper: API error → returns False (safe degradation)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_GH_CLIENT_CLASS = "argus.github_client.GitHubClient"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prior_review_dict(
    prior_sha: str = "aaaa0000bbbb",
    prior_verdict: str = "APPROVE",
) -> dict:
    """Build a minimal prior_review state dict."""
    return {
        "review_id": "rev-1",
        "reviewed_sha": prior_sha,
        "findings": [],
        "notes_for_next_round": None,
        "round_number": 2,
        "prior_verdict": prior_verdict,
        "dismissed_findings": [],
    }


def _make_state(
    *,
    prior_review: dict | None = None,
    head_sha: str = "cccc1111dddd",
    repo: str = "org/repo",
    pr_number: int = 42,
) -> dict:
    """Build a minimal ReviewState dict for preflight testing."""
    return {
        "request": {"repo": repo, "pr_number": pr_number},
        "diff": "diff --git a/src/foo.py b/src/foo.py\n+x = 1",
        "description": "fix: add x",
        "head_sha": head_sha,
        "prior_review": prior_review or {},
    }


# ---------------------------------------------------------------------------
# Tests for _is_catchup_merge_only helper
# ---------------------------------------------------------------------------


class TestIsCatchupMergeOnly:
    """Unit tests for the _is_catchup_merge_only helper function."""

    def _get_fn(self):
        from argus.graph import _is_catchup_merge_only

        return _is_catchup_merge_only

    def test_all_merge_commits_returns_true(self) -> None:
        """When all commits in the window are merge commits, returns True."""
        mock_gh = MagicMock()
        mock_gh.get_compare_commits.return_value = [
            {"sha": "abc", "parent_count": 2},
            {"sha": "def", "parent_count": 2},
        ]
        with patch(_GH_CLIENT_CLASS, return_value=mock_gh):
            fn = self._get_fn()
            result = fn("org/repo", "aaaa0000", "cccc1111")
        assert result is True

    def test_authored_commit_present_returns_false(self) -> None:
        """When at least one non-merge (authored) commit is present, returns False."""
        mock_gh = MagicMock()
        mock_gh.get_compare_commits.return_value = [
            {"sha": "abc", "parent_count": 2},  # merge
            {"sha": "def", "parent_count": 1},  # authored
        ]
        with patch(_GH_CLIENT_CLASS, return_value=mock_gh):
            fn = self._get_fn()
            result = fn("org/repo", "aaaa0000", "cccc1111")
        assert result is False

    def test_empty_commits_returns_false(self) -> None:
        """Empty commit list (identical SHAs) is not a catch-up."""
        mock_gh = MagicMock()
        mock_gh.get_compare_commits.return_value = []
        with patch(_GH_CLIENT_CLASS, return_value=mock_gh):
            fn = self._get_fn()
            result = fn("org/repo", "aaaa0000", "cccc1111")
        assert result is False

    def test_api_error_returns_false(self) -> None:
        """API errors degrade safely — gate returns False (route to full)."""
        mock_gh = MagicMock()
        mock_gh.get_compare_commits.side_effect = Exception("network error")
        with patch(_GH_CLIENT_CLASS, return_value=mock_gh):
            fn = self._get_fn()
            result = fn("org/repo", "aaaa0000", "cccc1111")
        assert result is False

    def test_single_authored_commit_returns_false(self) -> None:
        """A single non-merge commit → not a catch-up."""
        mock_gh = MagicMock()
        mock_gh.get_compare_commits.return_value = [
            {"sha": "xyz", "parent_count": 1},
        ]
        with patch(_GH_CLIENT_CLASS, return_value=mock_gh):
            fn = self._get_fn()
            result = fn("org/repo", "aaaa0000", "cccc1111")
        assert result is False


# ---------------------------------------------------------------------------
# Tests for _node_preflight (integration of the gate into the node)
# ---------------------------------------------------------------------------


class TestNodePreflightCatchupGate:
    """Tests for catch-up-gate logic inside _node_preflight."""

    @pytest.mark.asyncio
    async def test_catchup_merge_routes_lite_without_llm(self) -> None:
        """(a) Merge-only delta → routes lite without calling run_preflight_check."""
        from argus.graph import _node_preflight

        state = _make_state(prior_review=_make_prior_review_dict())

        with (
            patch(
                "argus.graph._is_catchup_merge_only",
                return_value=True,
            ) as mock_gate,
            patch(
                "argus.graph.run_preflight_check",
                new_callable=AsyncMock,
            ) as mock_llm,
        ):
            result = await _node_preflight(state)

        # Gate should have been called with the right args
        mock_gate.assert_called_once_with(
            "org/repo",
            "aaaa0000bbbb",  # prior_sha from _make_prior_review_dict
            "cccc1111dddd",  # head_sha from _make_state
        )
        # LLM call must NOT have been made
        mock_llm.assert_not_called()
        # Result routes to lite
        assert result["is_lite"] is True
        assert result["preflight_result"]["route"] == "lite"
        assert "catch-up" in result["preflight_result"]["reason"]
        assert result["is_catchup_merge"] is True

    @pytest.mark.asyncio
    async def test_authored_commit_falls_through_to_llm(self) -> None:
        """(b) New authored commit present → gate returns False, LLM runs normally."""
        from argus.graph import PreflightResult, _node_preflight

        state = _make_state(prior_review=_make_prior_review_dict())

        llm_result = PreflightResult(route="full", reason="non-trivial change")

        with (
            patch(
                "argus.graph._is_catchup_merge_only",
                return_value=False,
            ),
            patch(
                "argus.graph.run_preflight_check",
                new_callable=AsyncMock,
                return_value=llm_result,
            ) as mock_llm,
        ):
            result = await _node_preflight(state)

        mock_llm.assert_called_once()
        assert result["is_lite"] is False
        assert result["preflight_result"]["route"] == "full"
        assert result["is_catchup_merge"] is False

    @pytest.mark.asyncio
    async def test_gate_raise_falls_through_to_llm(self) -> None:
        """Gate raising (e.g. thread-pool shutdown) must not crash the node — falls through to LLM."""
        from argus.graph import PreflightResult, _node_preflight

        state = _make_state(prior_review=_make_prior_review_dict())

        llm_result = PreflightResult(route="full", reason="non-trivial change")

        with (
            patch(
                "argus.graph._is_catchup_merge_only",
                side_effect=RuntimeError("cannot schedule new futures after shutdown"),
            ),
            patch(
                "argus.graph.run_preflight_check",
                new_callable=AsyncMock,
                return_value=llm_result,
            ) as mock_llm,
        ):
            result = await _node_preflight(state)

        # Node must not crash; it falls through to the normal preflight path.
        mock_llm.assert_called_once()
        assert result["is_lite"] is False
        assert result["preflight_result"]["route"] == "full"

    @pytest.mark.asyncio
    async def test_round_1_no_prior_skips_gate(self) -> None:
        """(c) Round 1 (no prior review) → gate is never called, LLM runs."""
        from argus.graph import PreflightResult, _node_preflight

        state = _make_state(prior_review={})

        llm_result = PreflightResult(route="lite", reason="trivial change")

        with (
            patch(
                "argus.graph._is_catchup_merge_only",
                return_value=True,
            ) as mock_gate,
            patch(
                "argus.graph.run_preflight_check",
                new_callable=AsyncMock,
                return_value=llm_result,
            ) as mock_llm,
        ):
            result = await _node_preflight(state)

        mock_gate.assert_not_called()
        mock_llm.assert_called_once()
        assert result["preflight_result"]["route"] == "lite"

    @pytest.mark.asyncio
    async def test_identical_shas_skips_gate(self) -> None:
        """(d-a) prior_sha == head_sha → gate is never called."""
        from argus.graph import PreflightResult, _node_preflight

        state = _make_state(
            prior_review=_make_prior_review_dict(prior_sha="same-sha"),
            head_sha="same-sha",
        )
        llm_result = PreflightResult(route="full", reason="re-run same sha")

        with (
            patch(
                "argus.graph._is_catchup_merge_only",
                return_value=True,
            ) as mock_gate,
            patch(
                "argus.graph.run_preflight_check",
                new_callable=AsyncMock,
                return_value=llm_result,
            ),
        ):
            await _node_preflight(state)

        mock_gate.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_pr_number_skips_gate(self) -> None:
        """(d-b) SHA-mode review (pr_number=0) → gate is never called."""
        from argus.graph import PreflightResult, _node_preflight

        # pr_number=0 is the SHA-mode sentinel (ReviewRequest default)
        state = _make_state(prior_review=_make_prior_review_dict(), pr_number=0)
        llm_result = PreflightResult(route="full", reason="sha-mode review")

        with (
            patch(
                "argus.graph._is_catchup_merge_only",
                return_value=True,
            ) as mock_gate,
            patch(
                "argus.graph.run_preflight_check",
                new_callable=AsyncMock,
                return_value=llm_result,
            ),
        ):
            await _node_preflight(state)

        mock_gate.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_prior_reviewed_sha_skips_gate(self) -> None:
        """(d-c) prior_reviewed_sha empty string → gate is never called."""
        from argus.graph import PreflightResult, _node_preflight

        state = _make_state(prior_review=_make_prior_review_dict(prior_sha=""))
        llm_result = PreflightResult(route="full", reason="no prior sha")

        with (
            patch(
                "argus.graph._is_catchup_merge_only",
                return_value=True,
            ) as mock_gate,
            patch(
                "argus.graph.run_preflight_check",
                new_callable=AsyncMock,
                return_value=llm_result,
            ),
        ):
            await _node_preflight(state)

        mock_gate.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_head_sha_in_state_skips_gate(self) -> None:
        """(d-d) head_sha absent from state → gate is never called."""
        from argus.graph import PreflightResult, _node_preflight

        state = _make_state(prior_review=_make_prior_review_dict())
        del state["head_sha"]
        llm_result = PreflightResult(route="full", reason="no head sha in state")

        with (
            patch(
                "argus.graph._is_catchup_merge_only",
                return_value=True,
            ) as mock_gate,
            patch(
                "argus.graph.run_preflight_check",
                new_callable=AsyncMock,
                return_value=llm_result,
            ),
        ):
            await _node_preflight(state)

        mock_gate.assert_not_called()
