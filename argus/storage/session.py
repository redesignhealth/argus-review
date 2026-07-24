"""Async SQLAlchemy session factory for the Postgres storage backend.

Replaces the platform's ``get_async_session_factory`` helper. Deliberately
does not import any ORM model package — vendored SQL in ``argus.storage.sql``
issues raw SQL against an existing schema (see ``schema/*.sql``), so there is
nothing for SQLAlchemy metadata to discover here.
"""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from argus.config import get_settings


def ensure_asyncpg_url(url: str) -> str:
    """Normalize a Postgres URL to the asyncpg driver scheme.

    ``ARGUS_DB_URL``/``SUPABASE_DB_URL`` may be set with the canonical
    ``postgresql://`` (or legacy ``postgres://``) scheme; SQLAlchemy's async
    engine factory requires the explicit ``postgresql+asyncpg://`` driver.
    """
    if url.startswith("postgresql+asyncpg://"):
        return url
    if url.startswith("postgresql://"):
        return "postgresql+asyncpg://" + url[len("postgresql://") :]
    if url.startswith("postgres://"):
        return "postgresql+asyncpg://" + url[len("postgres://") :]
    return url


@lru_cache(maxsize=1)
def get_async_engine() -> AsyncEngine:
    """Return the process-wide async SQLAlchemy engine for the configured Postgres URL.

    Cached for the lifetime of the process (one connection pool, not one per
    call) — call :func:`clear_engine_cache` to force recreation, e.g. in
    tests, or after mutating ``os.environ``.

    Raises:
        ValueError: If neither ``ARGUS_DB_URL`` nor ``SUPABASE_DB_URL`` is set.
    """
    settings = get_settings()
    db_url = settings.db_url
    if not db_url:
        raise ValueError(
            "No database URL configured. Set ARGUS_DB_URL (or the back-compat "
            "alias SUPABASE_DB_URL) to enable the Postgres storage backend."
        )
    return create_async_engine(ensure_asyncpg_url(db_url))


def clear_engine_cache() -> None:
    """Clear the cached async engine, forcing the next call to recreate it."""
    get_async_engine.cache_clear()


def get_async_session_factory() -> async_sessionmaker[AsyncSession]:
    """Create a SQLAlchemy session factory bound to the cached async engine."""
    engine = get_async_engine()
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )
