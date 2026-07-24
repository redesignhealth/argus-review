"""Tests for checkpointer/history-backend mode resolution in ``graph.py``,
and graceful degradation when the history backend errors.

``tests/conftest.py`` sets a default ``ARGUS_DB_URL`` stub so the bulk of
the existing test suite keeps exercising the Postgres path. Tests here that
need a different resolution explicitly clear it (and the HTTP shim
singleton) first.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from argus.config import clear_cache
from argus.storage.http import install_http_storage, reset_http_storage

_GRAPH_MODULE = "argus.graph"


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
# build_pipeline(): checkpointer decoupled from history-backend mode --
# Postgres iff a DB URL is configured, else the LangGraph SQLite
# checkpointer, regardless of whether HTTP mode is armed.
# ---------------------------------------------------------------------------


class TestCheckpointerModeResolution:
    @pytest.mark.asyncio
    async def test_no_db_url_and_no_http_mode_uses_sqlite_checkpointer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The bare local-first case: no Postgres URL, HTTP shim not armed
        -> AsyncSqliteSaver (checkpointer selection is decoupled from
        history-backend mode, so this no longer requires HTTP mode to be
        armed)."""
        _clear_db_url_env(monkeypatch)

        from argus import graph as graph_module

        mock_saver = MagicMock()
        mock_saver.setup = AsyncMock()
        mock_saver_ctx = MagicMock()
        mock_saver_ctx.__aenter__ = AsyncMock(return_value=mock_saver)
        mock_saver_ctx.__aexit__ = AsyncMock(return_value=None)

        sentinel_graph = MagicMock(name="compiled_graph")
        mock_review_graph = MagicMock()
        mock_review_graph.copy.return_value = sentinel_graph

        with (
            patch(f"{_GRAPH_MODULE}._review_graph", mock_review_graph),
            patch(
                "langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver.from_conn_string",
                return_value=mock_saver_ctx,
            ),
        ):
            async with graph_module.build_pipeline() as graph:
                yielded = graph

        assert yielded is sentinel_graph
        mock_saver.setup.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_http_mode_armed_without_db_url_uses_sqlite_checkpointer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """HTTP history-backend mode still gets the SQLite checkpointer --
        same outcome as the no-http-mode case above, confirming the
        checkpointer decision no longer branches on HTTP mode at all."""
        _clear_db_url_env(monkeypatch)
        install_http_storage(
            read_url="https://example.test/{owner}/{repo}/{pr}",
            write_url="https://example.test/{owner}/{repo}/{pr}/rounds",
        )

        from argus import graph as graph_module

        mock_saver = MagicMock()
        mock_saver.setup = AsyncMock()
        mock_saver_ctx = MagicMock()
        mock_saver_ctx.__aenter__ = AsyncMock(return_value=mock_saver)
        mock_saver_ctx.__aexit__ = AsyncMock(return_value=None)

        sentinel_graph = MagicMock(name="compiled_graph")
        mock_review_graph = MagicMock()
        mock_review_graph.copy.return_value = sentinel_graph

        with (
            patch(f"{_GRAPH_MODULE}._review_graph", mock_review_graph),
            patch(
                "langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver.from_conn_string",
                return_value=mock_saver_ctx,
            ),
        ):
            async with graph_module.build_pipeline() as graph:
                yielded = graph

        assert yielded is sentinel_graph
        mock_saver.setup.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_db_url_uses_postgres_checkpointer_even_with_http_mode_armed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A configured DB URL always wins the checkpointer decision, even
        if HTTP history-backend mode happens to also be armed."""
        monkeypatch.setenv("ARGUS_DB_URL", "postgresql://user:pass@host:5432/db")
        clear_cache()
        install_http_storage(
            read_url="https://example.test/{owner}/{repo}/{pr}",
            write_url="https://example.test/{owner}/{repo}/{pr}/rounds",
        )

        from argus import graph as graph_module

        mock_pool = MagicMock()
        mock_pool.open = AsyncMock()
        mock_pool.close = AsyncMock()

        mock_saver = MagicMock()
        mock_saver.setup = AsyncMock()

        sentinel_graph = MagicMock(name="compiled_graph")
        mock_review_graph = MagicMock()
        mock_review_graph.copy.return_value = sentinel_graph

        original_flag = graph_module._checkpoint_tables_created
        try:
            with (
                patch(f"{_GRAPH_MODULE}._review_graph", mock_review_graph),
                patch("psycopg_pool.AsyncConnectionPool", return_value=mock_pool),
                patch(
                    "langgraph.checkpoint.postgres.aio.AsyncPostgresSaver",
                    return_value=mock_saver,
                ),
            ):
                graph_module._checkpoint_tables_created = False
                async with graph_module.build_pipeline() as graph:
                    yielded = graph
        finally:
            graph_module._checkpoint_tables_created = original_flag

        assert yielded is sentinel_graph
        mock_pool.open.assert_awaited_once()
        mock_saver.setup.assert_awaited_once()


# ---------------------------------------------------------------------------
# Graceful degradation: a history-backend read error is treated as round 1
# (warning logged, no crash) -- the same handling for every backend kind.
# ---------------------------------------------------------------------------


class TestFetchPriorReviewGracefulDegradation:
    @pytest.mark.asyncio
    async def test_backend_read_error_falls_back_to_round_one(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _clear_db_url_env(monkeypatch)

        from argus.graph import _fetch_prior_review

        mock_backend = AsyncMock()
        mock_backend.select_latest_completed_round.side_effect = RuntimeError("boom")

        with patch(f"{_GRAPH_MODULE}.get_history_backend", return_value=mock_backend):
            with caplog.at_level("WARNING"):
                result = await _fetch_prior_review("org/repo", 42)

        assert result is None
        assert any("treating as round 1" in rec.message for rec in caplog.records)

    @pytest.mark.asyncio
    async def test_http_storage_error_falls_back_to_round_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``HttpStorageError`` (a ``RuntimeError`` subclass) is caught by
        the same broad handler -- no special-casing needed."""
        _clear_db_url_env(monkeypatch)
        install_http_storage(
            read_url="https://example.test/{owner}/{repo}/{pr}",
            write_url="https://example.test/{owner}/{repo}/{pr}/rounds",
        )

        from argus.graph import _fetch_prior_review
        from argus.storage.http import HttpStorageError

        with patch(
            "argus.storage.resolver.HttpHistoryBackend.select_latest_completed_round",
            side_effect=HttpStorageError("network exploded"),
        ):
            result = await _fetch_prior_review("org/repo", 42)

        assert result is None

    @pytest.mark.asyncio
    async def test_module_not_found_error_still_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deployment-ordering bugs (a missing dependency) must still
        surface loudly rather than be swallowed as a generic storage
        failure."""
        _clear_db_url_env(monkeypatch)

        from argus.graph import _fetch_prior_review

        mock_backend = AsyncMock()
        mock_backend.select_latest_completed_round.side_effect = ModuleNotFoundError(
            "simulated missing module"
        )

        with (
            patch(f"{_GRAPH_MODULE}.get_history_backend", return_value=mock_backend),
            pytest.raises(ModuleNotFoundError),
        ):
            await _fetch_prior_review("org/repo", 42)
