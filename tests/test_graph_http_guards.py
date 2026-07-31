"""Tests for the input-validation guards added to ``run_review`` for the
HTTP storage path.

The guards fire before any pipeline work, so we don't need to mock the
whole LangGraph machinery — just patch ``is_http_storage_enabled`` and
``get_http_storage`` (consulted by ``argus.storage.resolver``, not
``graph.py`` directly) and assert the early raise.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from argus.models import ReviewRequest


async def test_run_review_raises_when_http_enabled_but_client_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``is_http_storage_enabled()`` returning True without an active
    client is a logic bug we want to surface immediately, not 30
    minutes into a pipeline run.
    """
    # Force history-backend resolution away from "postgres" (the
    # tests/conftest.py default) so resolve_history_backend_kind() actually
    # consults is_http_storage_enabled() below.
    monkeypatch.delenv("ARGUS_DB_URL", raising=False)
    monkeypatch.delenv("SUPABASE_DB_URL", raising=False)
    from argus.config import clear_cache

    clear_cache()

    with (
        patch("argus.storage.resolver.is_http_storage_enabled", return_value=True),
        patch("argus.storage.resolver.get_http_storage", return_value=None),
        # validate_history_backend_connectivity() currently no-ops for the
        # "http" backend kind without calling get_history_backend() at all,
        # so this passes without the patch today -- pinned explicitly so a
        # future refactor that starts calling it unconditionally doesn't
        # make this test hit the same "client missing" RuntimeError for
        # the wrong reason (the preflight check, not the guard under test).
        patch(
            "argus.graph.validate_history_backend_connectivity",
            new_callable=AsyncMock,
        ),
    ):
        from argus.graph import run_review

        with pytest.raises(RuntimeError, match="HTTP storage enabled but client missing"):
            await run_review(
                ReviewRequest(repo="org/repo", pr_number=42),
                flow_run_id="flow-1",
            )


# NOTE: a parallel test for the graph-level ``"/" not in request.repo``
# guard isn't included — ``ReviewRequest`` enforces ``owner/name`` shape
# upstream via its Pydantic regex, so a malformed string can't even
# reach ``run_review``. The graph-level guard is defense-in-depth in
# case ``request.repo`` ever gets mutated post-construction.
