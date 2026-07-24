"""Tests for ``argus.storage.resolver``: mode resolution + adapter wiring.

``tests/conftest.py`` sets a default ``ARGUS_DB_URL`` so the many
pre-existing ``graph.py`` tests (predating the sqlite-default) keep
exercising the Postgres path without every one of them opting in
explicitly. Every test here that wants a *different* resolution therefore
starts by clearing the DB-url env vars (and the HTTP shim singleton) and
``clear_cache()``-ing settings.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from argus.config import clear_cache
from argus.storage.http import install_http_storage, reset_http_storage
from argus.storage.resolver import (
    HttpHistoryBackend,
    PostgresHistoryBackend,
    get_history_backend,
    resolve_history_backend_kind,
)
from argus.storage.sqlite import SqliteHistoryBackend

pytestmark = pytest.mark.asyncio


def _clear_db_url_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARGUS_DB_URL", raising=False)
    monkeypatch.delenv("SUPABASE_DB_URL", raising=False)
    clear_cache()


@pytest.fixture(autouse=True)
async def _reset_http_singleton():
    await reset_http_storage()
    yield
    await reset_http_storage()


# ---------------------------------------------------------------------------
# Mode-resolution matrix (env matrix -> backend kind)
# ---------------------------------------------------------------------------


async def test_no_env_vars_except_required_selects_sqlite(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bare-minimum-config case: neither a DB URL nor the HTTP storage
    pair is set -> sqlite is the resolved default (the OSS-first-run path)."""
    _clear_db_url_env(monkeypatch)

    assert resolve_history_backend_kind() == "sqlite"
    backend = get_history_backend()
    assert isinstance(backend, SqliteHistoryBackend)


async def test_db_url_selects_postgres(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARGUS_DB_URL", "postgresql://u:p@host/db")
    clear_cache()

    assert resolve_history_backend_kind() == "postgres"
    assert isinstance(get_history_backend(), PostgresHistoryBackend)


async def test_supabase_db_url_alias_selects_postgres(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_db_url_env(monkeypatch)
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://u:p@host/db")
    clear_cache()

    assert resolve_history_backend_kind() == "postgres"


async def test_http_pair_selects_http_when_no_db_url(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_db_url_env(monkeypatch)
    install_http_storage(
        read_url="https://example.test/{owner}/{repo}/{pr}",
        write_url="https://example.test/{owner}/{repo}/{pr}/rounds",
    )

    assert resolve_history_backend_kind() == "http"
    assert isinstance(get_history_backend(), HttpHistoryBackend)


async def test_db_url_wins_over_armed_http_shim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Postgres takes priority over an armed HTTP shim -- matches the
    documented resolution order ("ARGUS_DB_URL -> postgres; HTTP pair ->
    http; else sqlite")."""
    monkeypatch.setenv("ARGUS_DB_URL", "postgresql://u:p@host/db")
    clear_cache()
    install_http_storage(
        read_url="https://example.test/{owner}/{repo}/{pr}",
        write_url="https://example.test/{owner}/{repo}/{pr}/rounds",
    )

    assert resolve_history_backend_kind() == "postgres"


async def test_history_db_path_override_used_for_sqlite(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _clear_db_url_env(monkeypatch)
    custom_path = tmp_path / "custom-history.db"
    monkeypatch.setenv("ARGUS_HISTORY_DB_PATH", str(custom_path))
    clear_cache()

    backend = get_history_backend()
    assert isinstance(backend, SqliteHistoryBackend)
    assert backend._db_path == str(custom_path)


# ---------------------------------------------------------------------------
# HttpHistoryBackend construction guard
# ---------------------------------------------------------------------------


async def test_http_backend_raises_when_client_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mirrors the ``is_http_storage_enabled()`` True / client None logic
    bug guard that used to live inline in graph.py at every HTTP-mode call
    site -- now centralized in the adapter constructor."""
    _clear_db_url_env(monkeypatch)
    with (
        patch("argus.storage.resolver.is_http_storage_enabled", return_value=True),
        patch("argus.storage.resolver.get_http_storage", return_value=None),
    ):
        assert resolve_history_backend_kind() == "http"
        with pytest.raises(RuntimeError, match="HTTP storage enabled but client missing"):
            get_history_backend()


# ---------------------------------------------------------------------------
# HttpHistoryBackend: unsupported-operation gaps are documented, not silent
# ---------------------------------------------------------------------------


async def test_http_backend_select_recent_rounds_not_implemented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_db_url_env(monkeypatch)
    install_http_storage(
        read_url="https://example.test/{owner}/{repo}/{pr}",
        write_url="https://example.test/{owner}/{repo}/{pr}/rounds",
    )
    backend = get_history_backend()
    with pytest.raises(NotImplementedError):
        await backend.select_recent_rounds(repo="org/repo", pr_number=1, limit=10)


async def test_http_backend_select_status_by_flow_run_not_implemented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_db_url_env(monkeypatch)
    install_http_storage(
        read_url="https://example.test/{owner}/{repo}/{pr}",
        write_url="https://example.test/{owner}/{repo}/{pr}/rounds",
    )
    backend = get_history_backend()
    with pytest.raises(NotImplementedError):
        await backend.select_status_by_flow_run(flow_run_id="fr-1")


async def test_http_backend_select_recent_lite_rounds_is_documented_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_db_url_env(monkeypatch)
    install_http_storage(
        read_url="https://example.test/{owner}/{repo}/{pr}",
        write_url="https://example.test/{owner}/{repo}/{pr}/rounds",
    )
    backend = get_history_backend()
    assert await backend.select_recent_lite_rounds(repo="org/repo", pr_number=1) == []


async def test_http_backend_insert_agent_runs_is_documented_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_db_url_env(monkeypatch)
    install_http_storage(
        read_url="https://example.test/{owner}/{repo}/{pr}",
        write_url="https://example.test/{owner}/{repo}/{pr}/rounds",
    )
    backend = get_history_backend()
    # Must not raise -- the gap is a silent (but logged) drop, matching the
    # inline behavior at the finalize call site.
    await backend.insert_agent_runs(code_review_id=uuid4(), runs=[])


async def test_http_backend_select_latest_completed_round_maps_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one read HTTP genuinely supports: the wire response gets mapped
    into a ``CodeReviewRoundRow`` with ``prior_count`` populated from the
    returned round count."""
    _clear_db_url_env(monkeypatch)
    install_http_storage(
        read_url="https://example.test/{owner}/{repo}/{pr}",
        write_url="https://example.test/{owner}/{repo}/{pr}/rounds",
    )
    backend = get_history_backend()
    assert isinstance(backend, HttpHistoryBackend)

    from argus.storage.models import CodeReviewRoundRecord

    fake_record = CodeReviewRoundRecord(
        id=uuid4(),
        created_at="2026-05-19T12:00:00+00:00",
        repo="org/repo",
        pr_number=7,
        verdict="APPROVE",
        sha="abc123",
    )
    mock_client = MagicMock()
    mock_client.read_latest_completed_round = AsyncMock(return_value=(fake_record, 3))
    backend._client = mock_client

    row = await backend.select_latest_completed_round(repo="org/repo", pr_number=7)
    assert row is not None
    assert row.id == fake_record.id
    assert row.sha == "abc123"
    assert row.prior_count == 3


async def test_http_backend_select_latest_completed_round_none_when_no_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_db_url_env(monkeypatch)
    install_http_storage(
        read_url="https://example.test/{owner}/{repo}/{pr}",
        write_url="https://example.test/{owner}/{repo}/{pr}/rounds",
    )
    backend = get_history_backend()
    assert isinstance(backend, HttpHistoryBackend)
    mock_client = MagicMock()
    mock_client.read_latest_completed_round = AsyncMock(return_value=(None, 0))
    backend._client = mock_client

    row = await backend.select_latest_completed_round(repo="org/repo", pr_number=999)
    assert row is None
