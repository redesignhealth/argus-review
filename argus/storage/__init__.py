"""Round-history storage backends: Postgres, HTTP, and local SQLite.

Argus persists review-round results to whichever backend is configured
(see ``argus.storage.resolver``): Postgres by default when ``ARGUS_DB_URL``
is set, an HTTP shim (this package's ``http`` module) for callers that can't
reach Postgres directly — for example, Argus invocations inside a sandboxed
agent runtime where outbound egress is restricted to 443 — or local SQLite
as the zero-configuration fallback.

The HTTP shim is **opt-in via CLI flags / env vars** (see
``argus_review_local.py``). Default behavior is unchanged: Argus keeps doing
direct Postgres or local SQLite writes.
"""

from __future__ import annotations

from argus.storage.http import (
    HttpStorageClient,
    HttpStorageError,
    install_http_storage,
    is_http_storage_enabled,
)
from argus.storage.models import (
    CodeReviewRound,
    CodeReviewRoundRecord,
)
from argus.storage.resolver import (
    HistoryBackend,
    HistoryBackendConnectivityError,
    HistoryBackendKind,
    HttpHistoryBackend,
    PostgresHistoryBackend,
    get_history_backend,
    resolve_history_backend_kind,
    validate_history_backend_connectivity,
)
from argus.storage.sqlite import SqliteHistoryBackend

__all__ = [
    "CodeReviewRound",
    "CodeReviewRoundRecord",
    "HistoryBackend",
    "HistoryBackendConnectivityError",
    "HistoryBackendKind",
    "HttpHistoryBackend",
    "HttpStorageClient",
    "HttpStorageError",
    "PostgresHistoryBackend",
    "SqliteHistoryBackend",
    "get_history_backend",
    "install_http_storage",
    "is_http_storage_enabled",
    "resolve_history_backend_kind",
    "validate_history_backend_connectivity",
]
