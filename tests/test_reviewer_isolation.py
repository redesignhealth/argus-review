"""Reviewer sessions must not load settings files from the checkout they review.

The session's cwd is the PR's own worktree, so an unset ``setting_sources``
would let that checkout supply ``.claude/settings*.json`` (hooks, permissions,
env, model) to a session that holds review credentials, and would also load the
caller's personal ``~/.claude`` settings.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from argus import runners


class _CapturedOptions(Exception):
    """Abort the session right after options are built; nothing runs."""


@pytest.mark.asyncio
async def test_reviewer_session_passes_no_filesystem_setting_sources(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def _capture(**kwargs: Any) -> Any:
        captured.update(kwargs)
        raise _CapturedOptions

    monkeypatch.setattr(runners, "ClaudeAgentOptions", _capture)

    settings = SimpleNamespace(anthropic_credential=("ANTHROPIC_API_KEY", "test-key"))
    with pytest.raises(_CapturedOptions):
        await runners._run_claude_session(
            model="claude-default",
            system_prompt="review",
            user_message="review this",
            settings=settings,
            repo_root=str(tmp_path),
        )

    # `[]`, not None: None means "load user, project and local settings".
    assert captured["setting_sources"] == []
    # The session still declares its own tool surface, so isolation costs nothing.
    assert captured["strict_mcp_config"] is True
    assert set(captured["allowed_tools"]) >= {"Read", "Glob", "Grep"}
