"""Who may dismiss a finding with a ``/dismiss`` PR comment (TECH-9842)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from argus.graph import _fetch_dismissed_findings


def _comment(user: str, association: str, body: str = "/dismiss B1 -- fine") -> dict[str, Any]:
    return {"id": 1, "user": user, "body": body, "author_association": association}


async def _run(comments: list[dict[str, Any]], *, pr_author: Any = "builder") -> list[Any]:
    gh = MagicMock()
    gh.list_issue_comments.return_value = comments
    if isinstance(pr_author, Exception):
        gh.get_pull_request.side_effect = pr_author
    else:
        gh.get_pull_request.return_value = {"user": pr_author}
    with patch("argus.github_client.GitHubClient", return_value=gh):
        return await _fetch_dismissed_findings("o/r", 7)


@pytest.mark.asyncio
async def test_comment_from_an_unprivileged_user_dismisses_nothing() -> None:
    for assoc in ("NONE", "CONTRIBUTOR", "FIRST_TIME_CONTRIBUTOR", "FIRST_TIMER"):
        assert await _run([_comment("stranger", assoc)]) == []


@pytest.mark.asyncio
async def test_pr_author_cannot_dismiss_even_with_write_access() -> None:
    assert await _run([_comment("Builder", "COLLABORATOR")]) == []
    assert await _run([_comment("builder", "OWNER")]) == []


@pytest.mark.asyncio
async def test_collaborator_who_is_not_the_author_can_dismiss() -> None:
    for assoc in ("OWNER", "MEMBER", "COLLABORATOR"):
        out = await _run([_comment("reviewer", assoc)])
        assert [d.dismissed_by for d in out] == ["reviewer"]


@pytest.mark.asyncio
async def test_only_the_trusted_comments_survive_a_mixed_thread() -> None:
    out = await _run(
        [
            _comment("stranger", "NONE"),
            _comment("builder", "COLLABORATOR"),
            _comment("reviewer", "MEMBER", "/dismiss B2 -- pre-existing"),
        ]
    )
    assert [d.dismissed_by for d in out] == ["reviewer"]


@pytest.mark.asyncio
async def test_fails_closed_when_the_pr_author_cannot_be_read() -> None:
    assert await _run([_comment("reviewer", "MEMBER")], pr_author=RuntimeError("boom")) == []


@pytest.mark.asyncio
async def test_a_comment_without_the_marker_is_not_a_dismissal() -> None:
    assert await _run([_comment("reviewer", "MEMBER", "looks good to me")]) == []
