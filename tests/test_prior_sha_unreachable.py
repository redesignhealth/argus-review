"""A prior review whose commit is not in this repo must not fail the round."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from argus.github_client import GitHubAPIError
from argus.graph import _node_fetch_diff
from argus.models import ReviewRequest

_GH_CLIENT_CLASS = "argus.github_client.GitHubClient"


@pytest.mark.asyncio
async def test_unreachable_prior_sha_falls_back_to_round_one() -> None:
    prior = MagicMock(reviewed_sha="a" * 40, findings=[], dismissed_findings=[])
    fetch = AsyncMock(
        side_effect=[
            GitHubAPIError(404, "Not Found"),
            ("diff", "desc", "b" * 40, "main", None),
        ]
    )
    gh = MagicMock()
    gh.get_compare_files.return_value = ([], False)
    state = {"request": ReviewRequest(repo="org/repo", pr_number=4).model_dump()}

    with (
        patch("argus.graph._fetch_prior_review", new_callable=AsyncMock, return_value=prior),
        patch("argus.graph._fetch_dismissed_findings", new_callable=AsyncMock, return_value=[]),
        patch("argus.graph._fetch_pr_diff_and_description", fetch),
        patch(_GH_CLIENT_CLASS, return_value=gh),
    ):
        result: dict[str, Any] = await _node_fetch_diff(state, {"configurable": {}})

    assert [c.kwargs["prior_sha"] for c in fetch.call_args_list] == ["a" * 40, None]
    assert result["prior_review"] == {}
    assert result["diff"] == "diff"


@pytest.mark.asyncio
async def test_404_without_prior_still_raises() -> None:
    fetch = AsyncMock(side_effect=GitHubAPIError(404, "Not Found"))
    state = {"request": ReviewRequest(repo="org/repo", pr_number=4).model_dump()}

    with (
        patch("argus.graph._fetch_prior_review", new_callable=AsyncMock, return_value=None),
        patch("argus.graph._fetch_dismissed_findings", new_callable=AsyncMock, return_value=[]),
        patch("argus.graph._fetch_pr_diff_and_description", fetch),
        pytest.raises(GitHubAPIError),
    ):
        await _node_fetch_diff(state, {"configurable": {}})
