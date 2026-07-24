"""Tests for argus.storage.session: URL normalization and engine construction."""

from __future__ import annotations

import pytest

from argus.storage.session import clear_engine_cache, ensure_asyncpg_url, get_async_engine


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("postgresql+asyncpg://u:p@host/db", "postgresql+asyncpg://u:p@host/db"),
        ("postgresql://u:p@host/db", "postgresql+asyncpg://u:p@host/db"),
        ("postgres://u:p@host/db", "postgresql+asyncpg://u:p@host/db"),
        ("sqlite:///local.db", "sqlite:///local.db"),
    ],
)
def test_ensure_asyncpg_url_normalizes_scheme(raw: str, expected: str) -> None:
    assert ensure_asyncpg_url(raw) == expected


def test_get_async_engine_raises_when_no_db_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARGUS_DB_URL", raising=False)
    monkeypatch.delenv("SUPABASE_DB_URL", raising=False)

    from argus.config import clear_cache

    clear_cache()
    clear_engine_cache()
    try:
        with pytest.raises(ValueError, match="ARGUS_DB_URL"):
            get_async_engine()
    finally:
        clear_cache()
        clear_engine_cache()


def test_get_async_engine_normalizes_postgres_scheme(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARGUS_DB_URL", "postgres://u:p@host/db")

    from argus.config import clear_cache

    clear_cache()
    clear_engine_cache()
    try:
        engine = get_async_engine()
        assert str(engine.url).startswith("postgresql+asyncpg://")
    finally:
        clear_cache()
        clear_engine_cache()
        engine.sync_engine.dispose()


def test_get_async_engine_is_cached_across_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """B1 regression: repeated calls must reuse one engine, not create a pool per call."""
    monkeypatch.setenv("ARGUS_DB_URL", "postgres://u:p@host/db")

    from argus.config import clear_cache

    clear_cache()
    clear_engine_cache()
    try:
        first = get_async_engine()
        second = get_async_engine()
        assert first is second
    finally:
        clear_cache()
        clear_engine_cache()
        first.sync_engine.dispose()
