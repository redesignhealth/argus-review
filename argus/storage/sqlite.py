"""SQLite-backed local round-history store — the third storage mode.

``argus/storage/sql.py`` is the Postgres-backed canonical writer; the HTTP
shim (``argus/storage/http.py``) is the writer for sandboxes that can't
reach Postgres directly. This module is the **local fallback** used when
neither ``ARGUS_DB_URL`` nor the HTTP storage pair is configured — the
default for self-hosted / open-source Argus runs (see ``docs/STORAGE.md``).
It implements the same seven logical operations as ``sql.py`` against a
single SQLite file, with the same Pydantic input/output shapes (imported
directly from ``sql.py`` rather than re-declared, so the two backends can
never drift on shape).

Async strategy — stdlib ``sqlite3`` + ``asyncio.to_thread``, not
``aiosqlite``:

- This module deliberately does not depend on ``argus.config`` or
  ``argus.storage.session``, and does not add a new third-party dependency
  (``aiosqlite``) — even though that would let a ``sqlite+aiosqlite://``
  SQLAlchemy engine mirror ``sql.py``'s ``AsyncSession`` shape more closely.
  This keeps the local fallback backend dependency-light and independent of
  the Postgres/SQLAlchemy stack the other backends use.
- The access pattern is single local developer machine, one process, rounds
  landing seconds-to-minutes apart — not a concurrency regime that benefits
  from a dedicated async driver. ``aiosqlite`` itself is implemented as a
  single background thread running the stdlib ``sqlite3`` module under an
  ``asyncio.Queue`` — i.e. exactly "stdlib sqlite3 on a thread", which is what
  this module does directly with ``asyncio.to_thread`` plus an ``asyncio.Lock``
  to serialize access to the one connection this backend instance owns.
- Every operation on a given backend instance is serialized through
  ``self._lock`` before being dispatched to a worker thread, so the
  ``check_same_thread=False`` connection is never touched concurrently even
  though different calls may land on different threadpool threads.

Schema translation from ``schema/*.sql`` (Postgres) to SQLite:

- ``JSONB`` → ``TEXT`` holding a ``json.dumps`` payload; decoded back to a
  ``dict`` on read (mirrors the JSONB-as-string coercion ``sql.py``'s Pydantic
  models already defend against for asyncpg).
- ``TEXT[]`` (``tool_names`` / ``files_explored`` on ``agent_runs``) → ``TEXT``
  holding a ``json.dumps`` array. Write-only in this module (no caller reads
  ``agent_runs`` back), so no decode path is needed yet.
- ``UUID`` primary keys → ``TEXT`` columns holding ``str(uuid4())``; SQLite has
  no native UUID type and storing the canonical string form keeps the values
  directly comparable/greppable in the file.
- ``TIMESTAMPTZ`` → ``TEXT`` holding a fixed-width UTC ISO-8601 string
  (``datetime.isoformat(timespec="microseconds")`` on an aware UTC datetime).
  ``timespec="microseconds"`` is load-bearing: plain ``.isoformat()`` drops the
  fractional part entirely when microseconds happen to be zero, which would
  silently break the lexical-sort-equals-chronological-sort invariant
  ``ORDER BY created_at DESC`` depends on.
- ``EXTRACT(EPOCH FROM (now() - created_at))`` (age_seconds in
  ``select_status_by_flow_run``) → computed in Python after fetching
  ``created_at``, rather than expressed in SQL — SQLite has no native
  timestamptz arithmetic and round-tripping through ``datetime`` is simpler
  and exactly as correct for a single-row read.
- The partial unique index ``... WHERE flow_run_id IS NOT NULL`` from
  migration ``011_add_review_progress_columns.sql`` is created verbatim as
  SQLite has supported partial indexes (and using them as an ``ON
  CONFLICT`` target) since 3.24 / 3.35 respectively; the bundled CPython
  3.12 ``sqlite3`` module links a recent-enough SQLite (>=3.43 as of this
  writing).
- Ties in ``ORDER BY created_at DESC`` (two rows landing in the same
  microsecond, common in fast test suites) are broken by SQLite's implicit
  ``rowid`` (``ORDER BY created_at DESC, rowid DESC``) so "latest" is always
  deterministic. Postgres relies on real wall-clock resolution plus
  ``gen_random_uuid()`` for tie-breaking in practice; this module makes the
  tie-break explicit since SQLite text-timestamp resolution can plausibly
  collide within a single test process.

Upsert semantics: this module uses SQLite's native
``INSERT ... ON CONFLICT(flow_run_id) WHERE flow_run_id IS NOT NULL DO
UPDATE ... RETURNING`` — the same conflict-target shape as the Postgres
statements in ``sql.py`` — rather than a manual
select-then-branch. Columns not present in the ``DO UPDATE SET`` list
(notably ``id`` and ``created_at``) are therefore left untouched on the
update path, exactly mirroring Postgres: the first ``INSERT`` for a given
``flow_run_id`` mints the id/created_at, and every subsequent upsert for
that same ``flow_run_id`` (running -> completed, or a retried completed
call) keeps them fixed while overwriting the mutable columns. The
``sha``/``base_ref`` ``COALESCE(excluded.x, code_reviews.x)`` clauses mirror
the documented Postgres invariant in ``sql.py`` ("sha is never cleared once
bound") byte-for-byte.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar
from uuid import UUID, uuid4

from argus.storage.sql import (
    AgentRunIn,
    CodeReviewRoundIn,
    CodeReviewRoundRow,
    CodeReviewStatusRow,
    UpsertReturnedNoRow,
)

__all__ = ["DEFAULT_DB_PATH", "SqliteHistoryBackend", "UpsertReturnedNoRow"]

DEFAULT_DB_PATH = Path.home() / ".local" / "share" / "argus" / "history.db"

_MEMORY_DB = ":memory:"

_T = TypeVar("_T")

_CODE_REVIEW_COLUMNS = (
    "id",
    "flow_run_id",
    "repo",
    "pr_number",
    "verdict",
    "risk_level",
    "blocking_count",
    "suggestion_count",
    "review_comment",
    "result_json",
    "cost_usd",
    "duration_seconds",
    "reviewer_version",
    "orchestrator_model",
    "subagent_model",
    "sha",
    "base_ref",
    "current_stage",
    "created_at",
)
_CODE_REVIEW_COLUMNS_SQL = ", ".join(_CODE_REVIEW_COLUMNS)

# =============================================================================
# DDL — idempotent bootstrap, mirrors schema/008 + schema/009 + schema/011 +
# schema/015 + schema/016 (review_patterns from schema/010 is out of scope:
# it belongs to the weekly feedback-loop flow, not the round-history path
# this module serves).
# =============================================================================

_BOOTSTRAP_DDL = """
CREATE TABLE IF NOT EXISTS code_reviews (
    id TEXT PRIMARY KEY,
    flow_run_id TEXT,
    repo TEXT NOT NULL,
    pr_number INTEGER NOT NULL,
    verdict TEXT,
    risk_level TEXT,
    blocking_count INTEGER,
    suggestion_count INTEGER,
    review_comment TEXT,
    result_json TEXT,
    cost_usd REAL,
    duration_seconds REAL,
    reviewer_version TEXT NOT NULL DEFAULT 'v3',
    orchestrator_model TEXT,
    subagent_model TEXT,
    sha TEXT,
    base_ref TEXT,
    current_stage TEXT NOT NULL DEFAULT 'completed',
    created_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_code_reviews_flow_run_id
    ON code_reviews (flow_run_id)
    WHERE flow_run_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_code_reviews_repo_pr
    ON code_reviews (repo, pr_number);

CREATE INDEX IF NOT EXISTS idx_code_reviews_created_at
    ON code_reviews (created_at);

CREATE TABLE IF NOT EXISTS agent_runs (
    id TEXT PRIMARY KEY,
    code_review_id TEXT NOT NULL
        REFERENCES code_reviews (id) ON DELETE CASCADE,
    agent_name TEXT NOT NULL,
    agent_type TEXT NOT NULL,
    model TEXT,
    cost_usd REAL DEFAULT 0,
    duration_seconds REAL DEFAULT 0,
    started_at TEXT,
    finished_at TEXT,
    tool_call_count INTEGER DEFAULT 0,
    tool_names TEXT,
    context7_call_count INTEGER DEFAULT 0,
    files_explored TEXT,
    finding_count INTEGER DEFAULT 0,
    result_text_length INTEGER DEFAULT 0,
    failure_reason TEXT
        CHECK (failure_reason IS NULL OR failure_reason IN ('timeout', 'worker_crashed')),
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_agent_runs_review
    ON agent_runs (code_review_id);
"""

# =============================================================================
# SQL constants
# =============================================================================

_SELECT_LATEST_COMPLETED_ROUND_SQL = f"""
    SELECT {_CODE_REVIEW_COLUMNS_SQL}
    FROM code_reviews
    WHERE repo = :repo AND pr_number = :pr_number AND verdict IS NOT NULL
    ORDER BY created_at DESC, rowid DESC
    LIMIT 1
"""

# Mirrors the correlated subquery in sql.py's _SELECT_LATEST_COMPLETED_ROUND_SQL:
# same ``verdict IS NOT NULL`` filter, no ``sha IS NOT NULL`` (see sql.py's
# module docstring on the COALESCE invariant this relies on).
_COUNT_COMPLETED_SQL = """
    SELECT COUNT(*) AS n FROM code_reviews
    WHERE repo = :repo AND pr_number = :pr_number AND verdict IS NOT NULL
"""

_SELECT_RECENT_ROUNDS_SQL = f"""
    SELECT {_CODE_REVIEW_COLUMNS_SQL}
    FROM code_reviews
    WHERE repo = :repo AND pr_number = :pr_number AND verdict IS NOT NULL
    ORDER BY created_at DESC, rowid DESC
    LIMIT :limit
"""

_SELECT_RECENT_LITE_ROUNDS_SQL = f"""
    SELECT {_CODE_REVIEW_COLUMNS_SQL}
    FROM code_reviews
    WHERE repo = :repo AND pr_number = :pr_number AND verdict IS NOT NULL
      AND reviewer_version = 'v3-lite'
    ORDER BY created_at DESC, rowid DESC
    LIMIT :limit
"""

_SELECT_STATUS_BY_FLOW_RUN_SQL = """
    SELECT result_json, current_stage, blocking_count, created_at
    FROM code_reviews
    WHERE flow_run_id = :flow_run_id
    ORDER BY created_at DESC, rowid DESC
    LIMIT 1
"""

_UPSERT_RUNNING_SQL = """
    INSERT INTO code_reviews (
        id, flow_run_id, repo, pr_number,
        current_stage, reviewer_version,
        sha, base_ref, created_at
    ) VALUES (
        :id, :flow_run_id, :repo, :pr_number,
        'running', 'v3',
        :sha, :base_ref, :created_at
    )
    ON CONFLICT (flow_run_id) WHERE flow_run_id IS NOT NULL
    DO UPDATE SET
        current_stage = 'running',
        verdict = NULL,
        result_json = NULL
"""

_UPSERT_COMPLETED_SQL = f"""
    INSERT INTO code_reviews ({_CODE_REVIEW_COLUMNS_SQL})
    VALUES (
        :id, :flow_run_id, :repo, :pr_number, :verdict, :risk_level,
        :blocking_count, :suggestion_count,
        :review_comment, :result_json, :cost_usd,
        :duration_seconds, :reviewer_version,
        :orchestrator_model, :subagent_model,
        :sha, :base_ref, :current_stage, :created_at
    )
    ON CONFLICT (flow_run_id) WHERE flow_run_id IS NOT NULL
    DO UPDATE SET
        verdict = excluded.verdict,
        risk_level = excluded.risk_level,
        blocking_count = excluded.blocking_count,
        suggestion_count = excluded.suggestion_count,
        review_comment = excluded.review_comment,
        result_json = excluded.result_json,
        cost_usd = excluded.cost_usd,
        duration_seconds = excluded.duration_seconds,
        orchestrator_model = excluded.orchestrator_model,
        subagent_model = excluded.subagent_model,
        -- Same "sha is never cleared once bound" invariant as sql.py's
        -- Postgres statement (see that module's docstring, deviation #1).
        sha = COALESCE(excluded.sha, code_reviews.sha),
        base_ref = COALESCE(excluded.base_ref, code_reviews.base_ref),
        current_stage = excluded.current_stage,
        reviewer_version = excluded.reviewer_version
    RETURNING {_CODE_REVIEW_COLUMNS_SQL}
"""

_INSERT_AGENT_RUN_SQL = """
    INSERT INTO agent_runs (
        id, code_review_id, agent_name, agent_type, model,
        cost_usd, duration_seconds, started_at, finished_at,
        tool_call_count, tool_names, context7_call_count,
        files_explored, finding_count, result_text_length, failure_reason, created_at
    ) VALUES (
        :id, :code_review_id, :agent_name, :agent_type, :model,
        :cost_usd, :duration_seconds, :started_at, :finished_at,
        :tool_call_count, :tool_names, :context7_call_count,
        :files_explored, :finding_count, :result_text_length, :failure_reason, :created_at
    )
"""


def _utcnow_iso() -> str:
    """Fixed-width UTC ISO-8601 timestamp; see module docstring on why
    ``timespec="microseconds"`` is required for correct lexical ordering."""
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _row_to_round(row: sqlite3.Row, *, prior_count: int | None = None) -> CodeReviewRoundRow:
    d: dict[str, Any] = dict(row)
    raw_json = d["result_json"]
    result_json = json.loads(raw_json) if raw_json is not None else None
    return CodeReviewRoundRow(
        id=UUID(d["id"]),
        flow_run_id=d["flow_run_id"],
        repo=d["repo"],
        pr_number=d["pr_number"],
        verdict=d["verdict"],
        risk_level=d["risk_level"],
        blocking_count=d["blocking_count"],
        suggestion_count=d["suggestion_count"],
        review_comment=d["review_comment"],
        result_json=result_json,
        cost_usd=d["cost_usd"],
        duration_seconds=d["duration_seconds"],
        reviewer_version=d["reviewer_version"],
        orchestrator_model=d["orchestrator_model"],
        subagent_model=d["subagent_model"],
        sha=d["sha"],
        base_ref=d["base_ref"],
        current_stage=d["current_stage"],
        created_at=datetime.fromisoformat(d["created_at"]),
        prior_count=prior_count,
    )


class SqliteHistoryBackend:
    """Local SQLite implementation of the seven ``sql.py`` operations.

    Owns exactly one ``sqlite3.Connection`` (created lazily, on first use)
    and serializes all access to it through ``self._lock`` — see the module
    docstring's "Async strategy" section. Callers get a plain instance
    (``SqliteHistoryBackend()`` for the default path, or
    ``SqliteHistoryBackend(db_path=...)`` to override) with no dependency on
    ``argus.config``.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path: str = str(db_path) if db_path is not None else str(DEFAULT_DB_PATH)
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()

    @property
    def db_path(self) -> str:
        """The resolved file path this backend reads/writes."""
        return self._db_path

    # -- connection management ------------------------------------------------

    def _open_and_bootstrap(self) -> sqlite3.Connection:
        if self._db_path != _MEMORY_DB:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        if self._db_path != _MEMORY_DB:
            # WAL is meaningless (and silently a no-op) for ``:memory:``
            # databases; only worth setting for file-backed ones, where it
            # also lets the "two backends, same file" case in
            # test_sqlite_backend.py read/write without lock contention.
            conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.executescript(_BOOTSTRAP_DDL)
        self._migrate_agent_runs_failure_reason(conn)
        conn.commit()
        return conn

    @staticmethod
    def _migrate_agent_runs_failure_reason(conn: sqlite3.Connection) -> None:
        """Self-healing column add for pre-existing local ``history.db`` files.

        ``CREATE TABLE IF NOT EXISTS`` in ``_BOOTSTRAP_DDL`` is a no-op once
        the table already exists, so a ``failure_reason`` column added to
        that DDL string would never actually reach an existing local file —
        the same class of gap Postgres's numbered ``schema/*.sql`` migrations
        exist to close, but this backend has no migration runner at all.
        Checking and adding the column at connect-time is the SQLite-shaped
        equivalent, scoped to this one column rather than a general
        migration framework this single-user local file doesn't need.
        """
        columns = {row[1] for row in conn.execute("PRAGMA table_info(agent_runs)")}
        if "failure_reason" not in columns:
            conn.execute(
                "ALTER TABLE agent_runs ADD COLUMN failure_reason TEXT "
                "CHECK (failure_reason IS NULL OR failure_reason IN ('timeout', 'worker_crashed'))"
            )

    async def _connection(self) -> sqlite3.Connection:
        # Must hold self._lock across the check-and-set: without it, two
        # concurrent coroutines can both observe self._conn is None, both
        # dispatch _open_and_bootstrap() to a worker thread, and the second
        # to finish overwrites self._conn, orphaning the first connection
        # (leaked file handle + WAL files).
        async with self._lock:
            if self._conn is None:
                self._conn = await asyncio.to_thread(self._open_and_bootstrap)
            return self._conn

    async def _run(self, fn: Callable[[sqlite3.Connection], _T]) -> _T:
        conn = await self._connection()
        async with self._lock:
            return await asyncio.to_thread(fn, conn)

    async def aclose(self) -> None:
        """Close the underlying connection, if one was opened. Safe to call
        multiple times (mirrors ``HttpStorageClient.aclose``)."""
        if self._conn is not None:
            conn = self._conn
            self._conn = None
            await asyncio.to_thread(conn.close)

    async def ping(self) -> None:
        """Open (or reuse) the connection, forcing the bootstrap DDL to run now.

        Exists so a caller can validate the configured ``db_path`` -- parent
        directory creatable, file writable, existing file not corrupt --
        before starting an expensive review, rather than discovering a
        permission or disk-full error only at the final ``upsert_completed_row``
        call after the whole pipeline has already run. Raises whatever
        ``sqlite3``/``OSError`` the underlying connect-and-bootstrap raises;
        callers should catch and wrap with a clearer message.
        """
        await self._connection()

    # -- reads -----------------------------------------------------------------

    async def select_latest_completed_round(
        self, *, repo: str, pr_number: int
    ) -> CodeReviewRoundRow | None:
        """Sqlite equivalent of ``sql.select_latest_completed_round``."""

        def _op(conn: sqlite3.Connection) -> CodeReviewRoundRow | None:
            row = conn.execute(
                _SELECT_LATEST_COMPLETED_ROUND_SQL,
                {"repo": repo, "pr_number": pr_number},
            ).fetchone()
            if row is None:
                return None
            count_row = conn.execute(
                _COUNT_COMPLETED_SQL, {"repo": repo, "pr_number": pr_number}
            ).fetchone()
            prior_count = int(count_row["n"]) if count_row is not None else 0
            return _row_to_round(row, prior_count=prior_count)

        return await self._run(_op)

    async def select_recent_rounds(
        self, *, repo: str, pr_number: int, limit: int
    ) -> list[CodeReviewRoundRow]:
        """Sqlite equivalent of ``sql.select_recent_rounds``."""

        def _op(conn: sqlite3.Connection) -> list[CodeReviewRoundRow]:
            rows = conn.execute(
                _SELECT_RECENT_ROUNDS_SQL,
                {"repo": repo, "pr_number": pr_number, "limit": limit},
            ).fetchall()
            return [_row_to_round(r) for r in rows]

        return await self._run(_op)

    async def select_recent_lite_rounds(
        self, *, repo: str, pr_number: int, limit: int = 200
    ) -> list[CodeReviewRoundRow]:
        """Sqlite equivalent of ``sql.select_recent_lite_rounds``."""

        def _op(conn: sqlite3.Connection) -> list[CodeReviewRoundRow]:
            rows = conn.execute(
                _SELECT_RECENT_LITE_ROUNDS_SQL,
                {"repo": repo, "pr_number": pr_number, "limit": limit},
            ).fetchall()
            return [_row_to_round(r) for r in rows]

        return await self._run(_op)

    async def select_status_by_flow_run(self, *, flow_run_id: str) -> CodeReviewStatusRow | None:
        """Sqlite equivalent of ``sql.select_status_by_flow_run``.

        ``age_seconds`` is computed in Python (see module docstring) rather
        than in SQL.
        """

        def _op(conn: sqlite3.Connection) -> CodeReviewStatusRow | None:
            row = conn.execute(
                _SELECT_STATUS_BY_FLOW_RUN_SQL, {"flow_run_id": flow_run_id}
            ).fetchone()
            if row is None:
                return None
            d: dict[str, Any] = dict(row)
            raw_json = d["result_json"]
            result_json = json.loads(raw_json) if raw_json is not None else None
            created_at = datetime.fromisoformat(d["created_at"])
            age_seconds = (datetime.now(UTC) - created_at).total_seconds()
            return CodeReviewStatusRow(
                result_json=result_json,
                current_stage=d["current_stage"],
                blocking_count=d["blocking_count"],
                age_seconds=age_seconds,
            )

        return await self._run(_op)

    # -- writes ------------------------------------------------------------

    async def upsert_running_row(
        self,
        *,
        flow_run_id: str,
        repo: str,
        pr_number: int,
        sha: str | None,
        base_ref: str | None,
    ) -> None:
        """Sqlite equivalent of ``sql.upsert_running_row``."""

        def _op(conn: sqlite3.Connection) -> None:
            conn.execute(
                _UPSERT_RUNNING_SQL,
                {
                    "id": str(uuid4()),
                    "flow_run_id": flow_run_id,
                    "repo": repo,
                    "pr_number": pr_number,
                    "sha": sha,
                    "base_ref": base_ref,
                    "created_at": _utcnow_iso(),
                },
            )
            conn.commit()

        await self._run(_op)

    async def upsert_completed_row(self, *, row: CodeReviewRoundIn) -> CodeReviewRoundRow:
        """Sqlite equivalent of ``sql.upsert_completed_row``.

        Uses SQLite's native ``INSERT ... ON CONFLICT ... DO UPDATE ...
        RETURNING`` (see module docstring "Upsert semantics") rather than a
        manual select-then-branch, so ``id``/``created_at`` are preserved
        across an update exactly the way Postgres preserves them (columns
        absent from ``DO UPDATE SET`` are untouched).
        """

        def _op(conn: sqlite3.Connection) -> CodeReviewRoundRow:
            params = {
                "id": str(uuid4()),
                "flow_run_id": row.flow_run_id,
                "repo": row.repo,
                "pr_number": row.pr_number,
                "verdict": row.verdict,
                "risk_level": row.risk_level,
                "blocking_count": row.blocking_count,
                "suggestion_count": row.suggestion_count,
                "review_comment": row.review_comment,
                "result_json": (
                    json.dumps(row.result_json) if row.result_json is not None else None
                ),
                "cost_usd": row.cost_usd,
                "duration_seconds": row.duration_seconds,
                "reviewer_version": row.reviewer_version,
                "orchestrator_model": row.orchestrator_model,
                "subagent_model": row.subagent_model,
                "sha": row.sha,
                "base_ref": row.base_ref,
                "current_stage": row.current_stage,
                "created_at": _utcnow_iso(),
            }
            cur = conn.execute(_UPSERT_COMPLETED_SQL, params)
            returned = cur.fetchone()
            conn.commit()
            if returned is None:
                # Practically unreachable given the DO UPDATE (no WHERE
                # clause narrowing it, unlike DO NOTHING) always matches on
                # conflict, and this backend serializes all writers through
                # self._lock — but sql.py exposes this exception for the
                # analogous Postgres race, so we defend the same contract
                # here (exercised in tests via a monkeypatched cursor).
                raise UpsertReturnedNoRow("upsert_completed_row: INSERT returned no row")
            return _row_to_round(returned)

        return await self._run(_op)

    async def insert_agent_runs(
        self, *, code_review_id: UUID | str, runs: list[AgentRunIn]
    ) -> None:
        """Sqlite equivalent of ``sql.insert_agent_runs``."""
        if not runs:
            return

        def _op(conn: sqlite3.Connection) -> None:
            payloads = [
                {
                    "id": str(uuid4()),
                    "code_review_id": str(code_review_id),
                    "agent_name": run.agent_name,
                    "agent_type": run.agent_type,
                    "model": run.model,
                    "cost_usd": run.cost_usd,
                    "duration_seconds": run.duration_seconds,
                    "started_at": (run.started_at.isoformat() if run.started_at else None),
                    "finished_at": (run.finished_at.isoformat() if run.finished_at else None),
                    "tool_call_count": run.tool_call_count,
                    "tool_names": json.dumps(run.tool_names),
                    "context7_call_count": run.context7_call_count,
                    "files_explored": json.dumps(run.files_explored),
                    "finding_count": run.finding_count,
                    "result_text_length": run.result_text_length,
                    "failure_reason": run.failure_reason,
                    "created_at": _utcnow_iso(),
                }
                for run in runs
            ]
            conn.executemany(_INSERT_AGENT_RUN_SQL, payloads)
            conn.commit()

        await self._run(_op)
