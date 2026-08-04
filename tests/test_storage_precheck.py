"""Unit tests for argus.storage.precheck.

Covers the fail-safe "no Postgres configured" no-op path (both functions)
and the happy path against a mocked AsyncSession, following the mocking
pattern used by tests/test_graph_progress.py.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from argus.config import clear_cache
from argus.storage.precheck import CandidateFiring, log_candidate_firings, select_rule_statuses


def _no_db(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARGUS_DB_URL", raising=False)
    monkeypatch.delenv("SUPABASE_DB_URL", raising=False)
    clear_cache()


async def test_select_rule_statuses_empty_rule_ids_short_circuits() -> None:
    assert await select_rule_statuses([]) == {}


async def test_select_rule_statuses_no_db_configured_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_db(monkeypatch)
    assert await select_rule_statuses(["rule-a"]) == {}


async def test_log_candidate_firings_no_db_configured_is_silent_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_db(monkeypatch)
    # Must not raise even though no DB is configured.
    await log_candidate_firings(
        repo="o/r",
        pr_number=1,
        head_sha="a" * 40,
        firings=[CandidateFiring(rule_id="rule-a", finding={})],
    )


async def test_log_candidate_firings_empty_list_short_circuits() -> None:
    # Must not raise, and must not touch the DB at all (no session factory
    # patched here — a real attempt would raise ValueError: no DB URL).
    await log_candidate_firings(repo="o/r", pr_number=1, head_sha="a" * 40, firings=[])


def _mock_session_ctx(mock_session: AsyncMock) -> MagicMock:
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_session)
    ctx.__aexit__ = AsyncMock(return_value=None)
    return ctx


async def test_select_rule_statuses_maps_rows_to_dict() -> None:
    mock_session = AsyncMock()
    mock_session.execute.return_value = [
        SimpleNamespace(rule_id="rule-a", status="verified"),
        SimpleNamespace(rule_id="rule-b", status="suspended"),
    ]
    session_ctx = _mock_session_ctx(mock_session)

    with patch(
        "argus.storage.precheck.get_async_session_factory",
        return_value=lambda: session_ctx,
    ):
        result = await select_rule_statuses(["rule-a", "rule-b"])

    assert result == {"rule-a": "verified", "rule-b": "suspended"}


async def test_select_rule_statuses_db_error_returns_empty_not_raise() -> None:
    mock_session = AsyncMock()
    mock_session.execute.side_effect = RuntimeError("boom")
    session_ctx = _mock_session_ctx(mock_session)

    with patch(
        "argus.storage.precheck.get_async_session_factory",
        return_value=lambda: session_ctx,
    ):
        assert await select_rule_statuses(["rule-a"]) == {}


async def test_log_candidate_firings_ensures_rule_rows_then_inserts_and_commits() -> None:
    mock_session = AsyncMock()
    session_ctx = _mock_session_ctx(mock_session)

    with patch(
        "argus.storage.precheck.get_async_session_factory",
        return_value=lambda: session_ctx,
    ):
        await log_candidate_firings(
            repo="o/r",
            pr_number=7,
            head_sha="b" * 40,
            firings=[
                CandidateFiring(rule_id="rule-a", finding={"rule_id": "rule-a", "message": "m"}),
                CandidateFiring(rule_id="rule-b", finding={"rule_id": "rule-b", "message": "n"}),
            ],
        )

    # One batched ensure-rows call + one batched insert call, regardless of
    # how many firings were in the batch -- not one pair per firing.
    assert mock_session.execute.await_count == 2
    mock_session.commit.assert_awaited_once()


async def test_log_candidate_firings_db_error_is_swallowed() -> None:
    mock_session = AsyncMock()
    mock_session.execute.side_effect = RuntimeError("boom")
    session_ctx = _mock_session_ctx(mock_session)

    with patch(
        "argus.storage.precheck.get_async_session_factory",
        return_value=lambda: session_ctx,
    ):
        # Must not raise.
        await log_candidate_firings(
            repo="o/r",
            pr_number=1,
            head_sha="c" * 40,
            firings=[CandidateFiring(rule_id="rule-a", finding={})],
        )
