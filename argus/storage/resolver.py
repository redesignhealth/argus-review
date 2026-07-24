"""History-backend resolver — picks postgres | http | sqlite at call time.

``argus/graph.py`` needs one of three round-history stores depending on how
the process is configured:

1. **postgres** — ``ARGUS_DB_URL``/``SUPABASE_DB_URL`` is set. Canonical
   writer, wraps ``argus.storage.sql``'s functions with an owned
   ``AsyncSession`` per call (via ``argus.storage.session``).
2. **http** — no Postgres URL, but the HTTP storage shim is armed (see
   ``argus.storage.http.install_http_storage``, wired by the CLI's
   ``--storage-read-url``/``--storage-write-url`` flags). Used inside
   sandboxes that can only reach your configured HTTP backend over HTTPS.
3. **sqlite** — neither of the above. **The new default** for self-hosted /
   open-source Argus runs: a local ``argus.storage.sqlite.SqliteHistoryBackend``
   file at ``~/.local/share/argus/history.db`` (overridable via
   ``ARGUS_HISTORY_DB_PATH``).

All three backends implement :class:`HistoryBackend` — the same seven
logical operations ``argus/storage/sql.py`` defines — so ``graph.py``'s call
sites read a backend via :func:`get_history_backend` and call it uniformly,
rather than branching on ``is_http_storage_enabled()`` at every site.

**What this module intentionally does NOT unify:** the *finalize* write
(``upsert_completed_row`` + the follow-up ``insert_agent_runs``) has a
genuine behavioral difference between backends that predates this module and
must survive it — see the doc on :meth:`HttpHistoryBackend.upsert_completed_row`.
``graph.py`` keeps an explicit kind-based branch at that one call site;
:func:`resolve_history_backend_kind` is the single source of truth for which
branch to take.

The LangGraph *checkpointer* (Postgres vs SQLite via ``AsyncSqliteSaver``) is
a separate decision from history-backend selection and is NOT made here —
see ``graph.build_pipeline()``, which decouples it from HTTP mode entirely
(Postgres iff a DB URL is configured, else the LangGraph SQLite checkpointer,
regardless of history-backend kind).
"""

from __future__ import annotations

import logging
from typing import Literal, Protocol
from uuid import UUID

from argus.config import get_settings
from argus.storage.http import (
    HttpStorageError,
    get_http_storage,
    is_http_storage_enabled,
)
from argus.storage.models import CodeReviewRound
from argus.storage.session import get_async_session_factory
from argus.storage.sql import (
    AgentRunIn,
    CodeReviewRoundIn,
    CodeReviewRoundRow,
    CodeReviewStatusRow,
)
from argus.storage.sqlite import SqliteHistoryBackend

logger = logging.getLogger(__name__)

HistoryBackendKind = Literal["postgres", "http", "sqlite"]

__all__ = [
    "HistoryBackend",
    "HistoryBackendKind",
    "HttpHistoryBackend",
    "HttpStorageError",
    "PostgresHistoryBackend",
    "get_history_backend",
    "resolve_history_backend_kind",
]


class HistoryBackend(Protocol):
    """The seven logical round-history operations ``sql.py`` defines.

    Every method's signature matches ``argus.storage.sql``'s free functions
    (minus the ``session`` parameter, which each implementation owns
    internally) and ``argus.storage.sqlite.SqliteHistoryBackend``'s existing
    method surface exactly — that class already satisfies this Protocol
    without modification.
    """

    async def select_latest_completed_round(
        self, *, repo: str, pr_number: int
    ) -> CodeReviewRoundRow | None: ...

    async def select_recent_rounds(
        self, *, repo: str, pr_number: int, limit: int
    ) -> list[CodeReviewRoundRow]: ...

    async def select_recent_lite_rounds(
        self, *, repo: str, pr_number: int, limit: int = 200
    ) -> list[CodeReviewRoundRow]: ...

    async def select_status_by_flow_run(
        self, *, flow_run_id: str
    ) -> CodeReviewStatusRow | None: ...

    async def upsert_running_row(
        self,
        *,
        flow_run_id: str,
        repo: str,
        pr_number: int,
        sha: str | None,
        base_ref: str | None,
    ) -> None: ...

    async def upsert_completed_row(self, *, row: CodeReviewRoundIn) -> CodeReviewRoundRow: ...

    async def insert_agent_runs(
        self, *, code_review_id: UUID | str, runs: list[AgentRunIn]
    ) -> None: ...


# =============================================================================
# Postgres adapter
# =============================================================================


class PostgresHistoryBackend:
    """Wraps ``argus.storage.sql``'s functions with an owned session per call.

    Each method opens a fresh session via ``get_async_session_factory()``
    (module-level reference so tests can patch
    ``argus.storage.resolver.get_async_session_factory``), issues the
    corresponding ``sql.py`` call, and commits writes before returning —
    exactly the pattern ``graph.py`` used to inline at each call site.

    ``get_async_session_factory()`` is bound to ``get_async_engine()``, which
    is process-wide cached (``functools.lru_cache``, see
    ``argus.storage.session``) — so every ``PostgresHistoryBackend`` instance
    across every ``get_history_backend()`` call site shares one connection
    pool instead of minting a new asyncpg pool per method call.
    """

    async def select_latest_completed_round(
        self, *, repo: str, pr_number: int
    ) -> CodeReviewRoundRow | None:
        from argus.storage.sql import select_latest_completed_round

        session_factory = get_async_session_factory()
        async with session_factory() as session:
            return await select_latest_completed_round(session, repo=repo, pr_number=pr_number)

    async def select_recent_rounds(
        self, *, repo: str, pr_number: int, limit: int
    ) -> list[CodeReviewRoundRow]:
        from argus.storage.sql import select_recent_rounds

        session_factory = get_async_session_factory()
        async with session_factory() as session:
            return await select_recent_rounds(session, repo=repo, pr_number=pr_number, limit=limit)

    async def select_recent_lite_rounds(
        self, *, repo: str, pr_number: int, limit: int = 200
    ) -> list[CodeReviewRoundRow]:
        from argus.storage.sql import select_recent_lite_rounds

        session_factory = get_async_session_factory()
        async with session_factory() as session:
            return await select_recent_lite_rounds(
                session, repo=repo, pr_number=pr_number, limit=limit
            )

    async def select_status_by_flow_run(self, *, flow_run_id: str) -> CodeReviewStatusRow | None:
        from argus.storage.sql import select_status_by_flow_run

        session_factory = get_async_session_factory()
        async with session_factory() as session:
            return await select_status_by_flow_run(session, flow_run_id=flow_run_id)

    async def upsert_running_row(
        self,
        *,
        flow_run_id: str,
        repo: str,
        pr_number: int,
        sha: str | None,
        base_ref: str | None,
    ) -> None:
        from argus.storage.sql import upsert_running_row

        session_factory = get_async_session_factory()
        async with session_factory() as session:
            await upsert_running_row(
                session,
                flow_run_id=flow_run_id,
                repo=repo,
                pr_number=pr_number,
                sha=sha,
                base_ref=base_ref,
            )
            await session.commit()

    async def upsert_completed_row(self, *, row: CodeReviewRoundIn) -> CodeReviewRoundRow:
        from argus.storage.sql import upsert_completed_row

        session_factory = get_async_session_factory()
        async with session_factory() as session:
            persisted = await upsert_completed_row(session, row=row)
            await session.commit()
            return persisted

    async def insert_agent_runs(
        self, *, code_review_id: UUID | str, runs: list[AgentRunIn]
    ) -> None:
        from argus.storage.sql import insert_agent_runs

        session_factory = get_async_session_factory()
        async with session_factory() as session:
            await insert_agent_runs(session, code_review_id=code_review_id, runs=runs)
            await session.commit()


# =============================================================================
# HTTP adapter
# =============================================================================


class HttpHistoryBackend:
    """Wraps ``HttpStorageClient`` behind the seven-operation surface.

    Only two operations are reachable over HTTP today (see
    ``argus.storage.http``'s module docstring: ``read_latest_completed_round``
    + ``write_round``, the two endpoints the ``docs/STORAGE.md`` contract
    defines). This adapter maps those two onto
    ``select_latest_completed_round`` /
    ``upsert_completed_row``+``upsert_running_row`` and preserves the
    documented gaps for the rest **exactly as graph.py handled them inline
    before this module existed**:

    - ``select_recent_rounds`` / ``select_status_by_flow_run``: not part of
      the minimal HTTP contract; raise :class:`NotImplementedError` (neither
      is a call site graph.py exercises today).
    - ``select_recent_lite_rounds``: returns ``[]`` (documented no-op; the
      lite-round-history call site skips its section entirely on the HTTP
      path — see ``graph.py``'s explicit kind check there, not this method).
    - ``insert_agent_runs``: no-op with the same log line the finalize call
      site used to emit inline (``"agent_runs analytics skipped on HTTP
      path"``) — per-agent analytics are outside the minimal HTTP contract
      (see ``docs/STORAGE.md``).

    **Finalize is deliberately NOT swallowed here.** ``upsert_completed_row``
    lets any :class:`HttpStorageError` propagate. This mirrors the original
    inline comment in ``graph.py``: "the HTTP path has no such [independent
    id-tracking] fallback. Surface loudly." The Postgres/SQLite adapters'
    equivalent write is instead wrapped at the ``graph.py`` call site (kept
    there, not here, since it's the one call site with real behavioral
    divergence between backends).
    """

    def __init__(self) -> None:
        client = get_http_storage()
        if client is None:
            # Mirrors the narrowing-for-mypy guard graph.py used to run at
            # every HTTP-mode call site: is_http_storage_enabled() returning
            # True without an installed client is a logic bug that must
            # surface immediately, not deep inside a pipeline run.
            raise RuntimeError("HTTP storage enabled but client missing")
        self._client = client

    @staticmethod
    def _split_owner_repo(repo: str) -> tuple[str, str]:
        if "/" not in repo:
            raise ValueError(f"repo must be in owner/name format, got {repo!r}")
        owner, name = repo.split("/", 1)
        return owner, name

    async def select_latest_completed_round(
        self, *, repo: str, pr_number: int
    ) -> CodeReviewRoundRow | None:
        owner, name = self._split_owner_repo(repo)
        latest, count = await self._client.read_latest_completed_round(
            owner=owner, repo=name, pr=pr_number
        )
        if latest is None:
            return None
        return CodeReviewRoundRow(**latest.model_dump(), prior_count=count)

    async def select_recent_rounds(
        self, *, repo: str, pr_number: int, limit: int
    ) -> list[CodeReviewRoundRow]:
        raise NotImplementedError(
            "select_recent_rounds is not part of the minimal HTTP storage "
            "contract; not exercised by any current graph.py call site"
        )

    async def select_recent_lite_rounds(
        self, *, repo: str, pr_number: int, limit: int = 200
    ) -> list[CodeReviewRoundRow]:
        return []

    async def select_status_by_flow_run(self, *, flow_run_id: str) -> CodeReviewStatusRow | None:
        raise NotImplementedError(
            "select_status_by_flow_run is not part of the minimal HTTP storage "
            "contract; not exercised by any current graph.py call site"
        )

    async def upsert_running_row(
        self,
        *,
        flow_run_id: str,
        repo: str,
        pr_number: int,
        sha: str | None,
        base_ref: str | None,
    ) -> None:
        owner, name = self._split_owner_repo(repo)
        await self._client.write_round(
            owner=owner,
            repo=name,
            pr=pr_number,
            round_data=CodeReviewRound(
                flow_run_id=flow_run_id,
                repo=repo,
                pr_number=pr_number,
                current_stage="running",
                reviewer_version="v3",
                sha=sha,
                base_ref=base_ref,
            ),
        )

    async def upsert_completed_row(self, *, row: CodeReviewRoundIn) -> CodeReviewRoundRow:
        owner, name = self._split_owner_repo(row.repo)
        persisted = await self._client.write_round(
            owner=owner,
            repo=name,
            pr=row.pr_number,
            round_data=CodeReviewRound(**row.model_dump()),
        )
        return CodeReviewRoundRow(**persisted.model_dump(), prior_count=None)

    async def insert_agent_runs(
        self, *, code_review_id: UUID | str, runs: list[AgentRunIn]
    ) -> None:
        # Per-agent analytics are outside the minimal HTTP storage contract --
        # same documented data-loss gap the inline HTTP finalize path had.
        logger.info("agent_runs analytics skipped on HTTP path")


# =============================================================================
# Resolution
# =============================================================================


def resolve_history_backend_kind() -> HistoryBackendKind:
    """Return which history backend the current process configuration selects.

    Order: ``ARGUS_DB_URL``/``SUPABASE_DB_URL`` -> postgres; else an armed
    HTTP storage shim -> http; else -> sqlite (the new local-first default).
    """
    settings = get_settings()
    if settings.db_url:
        return "postgres"
    if is_http_storage_enabled():
        return "http"
    return "sqlite"


def get_history_backend() -> HistoryBackend:
    """Construct the ``HistoryBackend`` selected by the current configuration.

    Raises:
        RuntimeError: HTTP mode is armed (``is_http_storage_enabled()``) but
            no client is installed -- a logic bug that must surface
            immediately (see :class:`HttpHistoryBackend`'s constructor).
    """
    kind = resolve_history_backend_kind()
    if kind == "postgres":
        return PostgresHistoryBackend()
    if kind == "http":
        return HttpHistoryBackend()
    settings = get_settings()
    db_path = settings.ARGUS_HISTORY_DB_PATH
    return SqliteHistoryBackend(db_path=db_path) if db_path else SqliteHistoryBackend()
