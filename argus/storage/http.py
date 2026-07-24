"""HTTP storage shim for Argus runs that can't reach Postgres on 5432.

Talks to a backend service you run yourself, implementing the two-endpoint
contract documented in ``docs/STORAGE.md``. URL templates are substituted on
call so the same configured backend can target multiple ``(owner, repo, pr)``
tuples during a flow's lifetime.

URL template placeholders supported on both ``read_url`` and
``write_url``:

* ``{owner}``
* ``{repo}``
* ``{pr}``

Auth uses an ``X-API-Key`` header; the caller passes the raw secret value
via ``ARGUS_STORAGE_AUTH``, and the client wraps it in the right header. Your
backend is responsible for validating it however it sees fit.

**Scope:** this module is the **minimum viable** HTTP shim — just the
two operations (``read_latest_completed_round`` + ``write_round``) the
in-sandbox Argus path actually needs.

**Not covered by this shim:** per-agent ``agent_runs`` analytics inserts.
The in-sandbox path skips those (with a log line + TODO at the call site);
analytics from Argus runs that go through HTTP mode are lost. This is a
documented, accepted limitation of the minimal shim (see ``docs/STORAGE.md``).
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from argus.storage.models import (
    CodeReviewRound,
    CodeReviewRoundRecord,
    ListReviewRoundsResponse,
)

logger = logging.getLogger(__name__)


class HttpStorageError(RuntimeError):
    """Raised on any HTTP failure during a storage call."""


_DEFAULT_TIMEOUT_SECONDS = 30.0


def _substitute(template: str, *, owner: str, repo: str, pr: int) -> str:
    """Substitute ``{owner}``, ``{repo}``, ``{pr}`` into a URL template.

    ``str.format`` would also accept positional braces and choke on
    unrelated ``{...}`` segments, so explicit ``.replace`` for
    predictability.
    """
    return template.replace("{owner}", owner).replace("{repo}", repo).replace("{pr}", str(pr))


class HttpStorageClient:
    """HTTP-backed Argus storage shim.

    A single instance owns one :class:`httpx.AsyncClient`; the client
    is created lazily on first use and reused across calls. Callers
    can pass their own client via the constructor (handy for tests
    using ``pytest-httpx``).
    """

    def __init__(
        self,
        *,
        read_url: str,
        write_url: str,
        auth: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._read_url = read_url
        self._write_url = write_url
        self._auth = auth
        self._client = client
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Accept": "application/json"}
        if self._auth:
            headers["X-API-Key"] = self._auth
        return headers

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def read_latest_completed_round(
        self, *, owner: str, repo: str, pr: int
    ) -> tuple[CodeReviewRoundRecord | None, int]:
        """GET the most recent completed round + total completed count.

        Calls your configured backend's list endpoint, which is expected to
        return rounds ordered ``created_at DESC LIMIT 200``. The first
        element is the most recent — exactly what ``graph.py``'s
        prior-review fetch needs. Returns ``(None, 0)`` when no completed
        rounds exist (round-1 case); otherwise ``(latest_round,
        total_count)``.

        Why a tuple: graph.py uses the latest round for the Prior
        Feedback table AND the total count to compute the next round
        number. Returning both in one call avoids a second HTTP roundtrip.

        Cap note: ``total_count`` saturates at 200 (the backend's documented
        GET ``LIMIT``, see ``docs/STORAGE.md``). For PRs with deep history
        that bucks into that cap, the round number is approximate — but a
        single Argus convergence loop runs ~10 rounds; 200 covers any
        realistic re-dispatch history.
        """
        url = _substitute(self._read_url, owner=owner, repo=repo, pr=pr)
        client = self._get_client()
        try:
            response = await client.get(url, headers=self._headers())
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise HttpStorageError(f"GET {url} failed: {type(exc).__name__}: {exc}") from exc
        payload: dict[str, Any] = response.json()
        parsed = ListReviewRoundsResponse.model_validate(payload)
        if not parsed.rounds:
            return None, 0
        # Contract requires DESC ordering, so rounds[0] is the latest.
        return parsed.rounds[0], len(parsed.rounds)

    async def write_round(
        self,
        *,
        owner: str,
        repo: str,
        pr: int,
        round_data: CodeReviewRound,
    ) -> CodeReviewRoundRecord:
        """POST a new (completed or running) round to your configured backend."""
        if round_data.repo != f"{owner}/{repo}" or round_data.pr_number != pr:
            # Single source of truth: the path is what the backend
            # authoritatively writes against. Catch the mismatch
            # client-side rather than waiting for a 400 from the
            # server.
            raise HttpStorageError(
                f"round payload mismatches path "
                f"(payload repo={round_data.repo!r}/pr={round_data.pr_number} "
                f"vs path {owner}/{repo}/pr={pr})"
            )
        url = _substitute(self._write_url, owner=owner, repo=repo, pr=pr)
        client = self._get_client()
        # ``mode="json"`` serializes UUIDs / datetimes that may appear
        # nested in ``result_json``; a Postgres-backed implementation would
        # typically store it as JSONB, so any string-serializable value
        # is fine.
        body = round_data.model_dump(mode="json")
        try:
            response = await client.post(url, json=body, headers=self._headers())
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise HttpStorageError(f"POST {url} failed: {type(exc).__name__}: {exc}") from exc
        payload = response.json()
        return CodeReviewRoundRecord.model_validate(payload)

    async def aclose(self) -> None:
        """Close the underlying :class:`httpx.AsyncClient`, if any.

        Safe to call multiple times.
        """
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# Module-level singleton that ``graph.py`` checks via ``is_http_storage_enabled()``.
# Set by ``install_http_storage()`` from ``argus_review_local.py``'s CLI flag
# branch — keeps the flag-plumbing out of every code-path inside graph.py.
_active_client: HttpStorageClient | None = None


def install_http_storage(
    *,
    read_url: str,
    write_url: str,
    auth: str | None = None,
) -> HttpStorageClient:
    """Install a process-global HTTP storage client.

    Subsequent calls to ``is_http_storage_enabled()`` return ``True``
    and ``graph.py``'s two SQL sites route through the HTTP client
    instead of Postgres. The default (no install call) keeps graph.py
    on the inline-SQL path — operator-workstation runs are unaffected.
    """
    global _active_client
    if _active_client is not None:
        # Idempotent re-install with the same parameters is a no-op;
        # different parameters is an operator error we should surface.
        existing = _active_client
        if (
            existing._read_url == read_url
            and existing._write_url == write_url
            and existing._auth == auth
        ):
            return existing
        raise HttpStorageError(
            "HTTP storage already installed with different parameters; "
            "process-global state, only one install per run"
        )
    _active_client = HttpStorageClient(
        read_url=read_url,
        write_url=write_url,
        auth=auth,
    )
    logger.info("HTTP storage installed (read=%s write=%s)", read_url, write_url)
    return _active_client


def is_http_storage_enabled() -> bool:
    """Return ``True`` iff a previous ``install_http_storage()`` call
    armed the shim. Used by ``graph.py`` to branch."""
    return _active_client is not None


def get_http_storage() -> HttpStorageClient | None:
    """Return the active HTTP storage client, or ``None``."""
    return _active_client


async def reset_http_storage() -> None:
    """Test-only helper — clear the module-level singleton.

    Closes the underlying ``httpx.AsyncClient`` if one was created
    via lazy init, so pytest doesn't see leaked-resource warnings
    between tests. Async to mirror ``HttpStorageClient.aclose``;
    tests using this in sync contexts should ``await`` it inside
    a fixture or call ``asyncio.run(reset_http_storage())``.
    """
    global _active_client
    if _active_client is not None:
        await _active_client.aclose()
        _active_client = None
