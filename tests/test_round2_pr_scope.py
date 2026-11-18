"""Round 2+ reviews only the PR's own files, not what a catch-up merge brought from main."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from argus.graph import _node_fetch_diff
from argus.models import ReviewRequest

_GH_CLIENT_CLASS = "argus.github_client.GitHubClient"


def _file_diff(path: str) -> str:
    return f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-old\n+new\n"


def _prior() -> MagicMock:
    prior = MagicMock(reviewed_sha="a" * 40, findings=[], dismissed_findings=[])
    prior.model_dump.return_value = {"reviewed_sha": "a" * 40}
    return prior


async def _run(compare_files: list[dict[str, Any]], truncated: bool = False) -> dict[str, Any]:
    window = _file_diff("src/pr.py") + _file_diff("from_main/a.py") + _file_diff("from_main/b.py")
    gh = MagicMock()
    gh.get_compare_files.return_value = (compare_files, truncated)
    state = {"request": ReviewRequest(repo="org/repo", pr_number=126).model_dump()}
    with (
        patch("argus.graph._fetch_prior_review", new_callable=AsyncMock, return_value=_prior()),
        patch("argus.graph._fetch_dismissed_findings", new_callable=AsyncMock, return_value=[]),
        patch(
            "argus.graph._fetch_pr_diff_and_description",
            new_callable=AsyncMock,
            return_value=(window, "desc", "b" * 40, "main", "c" * 40),
        ),
        patch(_GH_CLIENT_CLASS, return_value=gh),
    ):
        return await _node_fetch_diff(state, {"configurable": {}})


@pytest.mark.asyncio
async def test_round2_diff_drops_files_the_pr_did_not_change() -> None:
    result = await _run([{"filename": "src/pr.py", "status": "modified", "patch": "+new\n"}])

    assert "src/pr.py" in result["diff"]
    assert "from_main" not in result["diff"]


@pytest.mark.asyncio
async def test_round2_diff_is_kept_whole_when_pr_file_list_is_truncated() -> None:
    files = [{"filename": "src/pr.py", "status": "modified", "patch": "+new\n"}]
    result = await _run(files, truncated=True)

    assert "from_main/a.py" in result["diff"]
