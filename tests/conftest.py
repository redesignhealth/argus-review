"""Shared fixtures for Argus tests."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

# Every env var that could turn on LangSmith/LangChain tracing or point it
# at a real project. ``argus.graph.build_pipeline`` calls
# ``os.environ.setdefault(...)`` for a couple of these when
# ``Settings.LANGSMITH_API_KEY`` is truthy, so tests must never let that be
# ambiently true — otherwise the langsmith SDK's background flush thread
# posts real ``/runs/multipart`` requests to api.smith.langchain.com at
# session/interpreter teardown, which have no legitimate destination in CI
# and fail with 403s (and would post real data if they ever succeeded).
_TRACING_ENV_VARS = (
    "LANGSMITH_API_KEY",
    "LANGCHAIN_API_KEY",
    "LANGSMITH_PROJECT",
    "LANGCHAIN_PROJECT",
    "LANGSMITH_WORKSPACE_ID",
    "LANGSMITH_TRACING",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_ENDPOINT",
    "LANGSMITH_ENDPOINT",
)


@pytest.fixture(scope="session", autouse=True)
def _disable_live_langsmith_tracing() -> Iterator[None]:
    """Force LangSmith/LangChain tracing off for the whole test session.

    Tests must never perform live network calls to LangSmith. This removes
    any tracing credentials/flags that leaked in from the ambient
    environment (CI runner or developer shell) *before* any test runs, and
    pins the tracing flags to "false" so ``setdefault`` calls in
    application code (e.g. ``argus.graph.build_pipeline``) can't flip them
    back on mid-session. Restored at session end so this fixture has no
    effect outside the test run.
    """
    saved = {name: os.environ.pop(name, None) for name in _TRACING_ENV_VARS}
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"
    try:
        yield
    finally:
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
        os.environ.pop("LANGSMITH_TRACING", None)
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value


@pytest.fixture(autouse=True)
def _mock_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure tests never use real credentials.

    Uses the field names (uppercase) from ``argus.config.Settings``.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("GITHUB_TOKEN_RO", "test-github-token")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    # Default the history-backend/checkpointer resolution to "postgres" so
    # the many existing graph.py tests that mock a Postgres session factory
    # (predating the local-sqlite default) keep exercising that path
    # without every one of them needing to set this explicitly. Tests that
    # want to exercise the HTTP or local-sqlite resolution paths explicitly
    # ``monkeypatch.delenv("ARGUS_DB_URL", raising=False)`` (+ the
    # ``SUPABASE_DB_URL`` alias) and re-``clear_cache()``.
    #
    # Only set the stub when neither var is already present in the real
    # environment -- a developer running the postgres-backed
    # ``integration``-marked tests with a real ``ARGUS_DB_URL``/
    # ``SUPABASE_DB_URL`` exported must not have it clobbered by this stub.
    if not os.environ.get("ARGUS_DB_URL") and not os.environ.get("SUPABASE_DB_URL"):
        monkeypatch.setenv("ARGUS_DB_URL", "postgresql://test:test@localhost:5432/test")

    # Clear the settings cache so the env vars set above take effect.
    from argus.config import clear_cache
    from argus.storage.session import clear_engine_cache

    clear_cache()
    clear_engine_cache()
