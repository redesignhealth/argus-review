"""Integration tests for review_patterns schema — COALESCE unique constraint.

Requires a real PostgreSQL instance via SUPABASE_DB_URL env var.
Exercises the uq_review_patterns_week_cat_dir unique index to verify
that (week_ending, category, COALESCE(directory, '')) correctly rejects
duplicates and handles NULL directory values.
"""

from __future__ import annotations

import datetime
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Skip entire module if no real DB available
pytestmark = pytest.mark.integration


@pytest.fixture
async def db_session() -> AsyncGenerator[async_sessionmaker[AsyncSession], None]:
    """Create a session factory connected to the real test database."""
    import os

    db_url = os.environ.get("SUPABASE_DB_URL")
    if not db_url:
        pytest.skip("SUPABASE_DB_URL not set — skipping integration test")

    # Skip if the URL looks like a stub (set by test helpers, not a real DB URL)
    if not (db_url.startswith("postgres://") or db_url.startswith("postgresql://")):
        pytest.skip("SUPABASE_DB_URL is not a valid postgres URL — skipping integration test")

    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    engine = create_async_engine(db_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


def _make_record(
    week_ending: datetime.date,
    category: str,
    directory: str | None = None,
) -> dict[str, Any]:
    return {
        "week_ending": week_ending,
        "category": category,
        "directory": directory,
        "occurrence_count": 5,
        "distinct_pr_count": 3,
        "avg_severity": "SUGGESTION",
        "sample_descriptions": ["test description"],
        "action_taken": "skipped",
        "action_detail": "test",
    }


@pytest.mark.asyncio
async def test_coalesce_unique_constraint_rejects_duplicate_null_directory(
    db_session: async_sessionmaker[AsyncSession],
) -> None:
    """Two rows with same (week_ending, category, NULL directory) should conflict."""
    # Use a unique week_ending to avoid colliding with real data
    test_week = datetime.date(2099, 1, 6)
    record = _make_record(test_week, f"test-{uuid.uuid4().hex[:8]}", None)

    async with db_session() as session:
        try:
            # Insert first row — should succeed
            await session.execute(
                text("""
                    INSERT INTO review_service.review_patterns
                        (week_ending, category, directory, occurrence_count,
                         distinct_pr_count, avg_severity, sample_descriptions,
                         action_taken, action_detail)
                    VALUES
                        (:week_ending, :category, :directory, :occurrence_count,
                         :distinct_pr_count, :avg_severity,
                         CAST(:sample_descriptions AS TEXT[]),
                         :action_taken, :action_detail)
                """),
                record,
            )

            # Insert duplicate — should trigger ON CONFLICT via upsert
            await session.execute(
                text("""
                    INSERT INTO review_service.review_patterns
                        (week_ending, category, directory, occurrence_count,
                         distinct_pr_count, avg_severity, sample_descriptions,
                         action_taken, action_detail)
                    VALUES
                        (:week_ending, :category, :directory, :occurrence_count,
                         :distinct_pr_count, :avg_severity,
                         CAST(:sample_descriptions AS TEXT[]),
                         :action_taken, :action_detail)
                    ON CONFLICT (week_ending, category, COALESCE(directory, ''))
                    DO UPDATE SET
                        occurrence_count = EXCLUDED.occurrence_count
                """),
                {**record, "occurrence_count": 10},
            )

            # Verify only one row exists with the updated count
            result = await session.execute(
                text("""
                    SELECT occurrence_count FROM review_service.review_patterns
                    WHERE week_ending = :week_ending AND category = :category
                    AND directory IS NULL
                """),
                {"week_ending": test_week, "category": record["category"]},
            )
            rows = result.fetchall()
            assert len(rows) == 1, f"Expected 1 row, got {len(rows)}"
            assert rows[0][0] == 10, f"Expected updated count 10, got {rows[0][0]}"
        finally:
            await session.rollback()


@pytest.mark.asyncio
async def test_coalesce_distinguishes_null_from_empty_directory(
    db_session: async_sessionmaker[AsyncSession],
) -> None:
    """NULL directory and '' directory should CONFLICT (COALESCE maps both to '')."""
    test_week = datetime.date(2099, 1, 13)
    category = f"test-{uuid.uuid4().hex[:8]}"
    record_null = _make_record(test_week, category, None)
    record_empty = _make_record(test_week, category, "")

    async with db_session() as session:
        try:
            await session.execute(
                text("""
                    INSERT INTO review_service.review_patterns
                        (week_ending, category, directory, occurrence_count,
                         distinct_pr_count, avg_severity, sample_descriptions,
                         action_taken, action_detail)
                    VALUES
                        (:week_ending, :category, :directory, :occurrence_count,
                         :distinct_pr_count, :avg_severity,
                         CAST(:sample_descriptions AS TEXT[]),
                         :action_taken, :action_detail)
                """),
                record_null,
            )

            # Empty string directory should conflict with NULL via COALESCE
            await session.execute(
                text("""
                    INSERT INTO review_service.review_patterns
                        (week_ending, category, directory, occurrence_count,
                         distinct_pr_count, avg_severity, sample_descriptions,
                         action_taken, action_detail)
                    VALUES
                        (:week_ending, :category, :directory, :occurrence_count,
                         :distinct_pr_count, :avg_severity,
                         CAST(:sample_descriptions AS TEXT[]),
                         :action_taken, :action_detail)
                    ON CONFLICT (week_ending, category, COALESCE(directory, ''))
                    DO UPDATE SET
                        occurrence_count = EXCLUDED.occurrence_count
                """),
                {**record_empty, "occurrence_count": 20},
            )

            # Should be exactly one row (upsert, not two inserts)
            result = await session.execute(
                text("""
                    SELECT count(*) FROM review_service.review_patterns
                    WHERE week_ending = :week_ending AND category = :category
                """),
                {"week_ending": test_week, "category": category},
            )
            count = result.scalar()
            assert count == 1, f"Expected 1 row (COALESCE merge), got {count}"
        finally:
            await session.rollback()
