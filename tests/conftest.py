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


@pytest.fixture(scope="session", autouse=True)
def _redirect_context_ledger_to_devnull() -> Iterator[None]:
    """Redirect the TECH-4734 phase 2 context-usage ledger to /dev/null for
    the whole test session.

    ``_append_context_ledger`` (argus/runners.py) is called unconditionally
    from every message in a real ``_run_claude_session`` message loop, not
    gated on tracing state (see that function's docstring) -- any test that
    drives such a loop would otherwise write real lines to
    ``/tmp/argus-context-ledger-<pytest-pid>.jsonl`` and leave them there
    permanently, the same class of test-pollution
    ``_disable_live_langsmith_tracing`` above exists to prevent for tracing.
    ``/dev/null`` accepts writes and discards them, so this is a true no-op
    rather than a redirected-but-still-accumulating tmp file.
    """
    saved = os.environ.get("ARGUS_CONTEXT_LEDGER_PATH")
    os.environ["ARGUS_CONTEXT_LEDGER_PATH"] = os.devnull
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("ARGUS_CONTEXT_LEDGER_PATH", None)
        else:
            os.environ["ARGUS_CONTEXT_LEDGER_PATH"] = saved


def _mock_settings_impl(node: pytest.Item, monkeypatch: pytest.MonkeyPatch) -> None:
    """The actual logic behind the ``_mock_settings`` fixture below.

    Pulled out as a plain function, taking the marker-bearing node directly
    rather than a full ``FixtureRequest``, specifically so
    ``tests/test_conftest.py`` can call it directly with a synthetic node in
    a unit test -- pytest raises if a ``@pytest.fixture``-wrapped function is
    called directly (it's meant to be requested as a parameter), and
    reaching into its ``__wrapped__`` attribute to bypass that would depend
    on an undocumented pytest implementation detail.

    Uses the field names (uppercase) from ``argus.config.Settings``.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    # Unconditional for every test EXCEPT ones explicitly opted out via the
    # dedicated `needs_real_github_token` marker -- deliberately narrower
    # than gating on the generic `integration` marker (an earlier version of
    # this fixture did that and broke tests/storage/test_backend_contract.py's
    # `[postgres]` param: that test is *also* integration-marked, doesn't
    # check GITHUB_TOKEN_RO itself, and still needs Settings() to construct
    # successfully, which requires this stub). Only
    # tests/test_precheck_shadow_integration.py carries the narrower marker
    # today, since it's the only test that needs to see whatever the
    # invoking shell actually exported (or nothing) rather than this stub.
    # Presence-in-environment gating (an even earlier version) isn't right
    # either: it broke isolation for the whole unit suite whenever a
    # developer happened to have a real token exported for unrelated reasons.
    if node.get_closest_marker("needs_real_github_token"):
        # Centralized here rather than left to each marked test's own
        # fixture: GITHUB_TOKEN_RO is a required (non-Optional) Settings
        # field, so a future needs_real_github_token test that doesn't
        # separately reimplement this skip would otherwise hit a confusing
        # pydantic ValidationError instead of a clean skip the moment
        # Settings() is constructed below. The marker alone now guarantees
        # the full contract.
        if not os.environ.get("GITHUB_TOKEN_RO"):
            pytest.skip("GITHUB_TOKEN_RO not set — skipping test marked needs_real_github_token")
    else:
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


@pytest.fixture(autouse=True)
def _mock_settings(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure tests never use real credentials. See ``_mock_settings_impl``."""
    _mock_settings_impl(request.node, monkeypatch)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Enforce that ``needs_real_github_token`` always implies ``integration``.

    Without this, a future test carrying only ``needs_real_github_token``
    (no ``integration``) would run in the *default* suite -- `_mock_settings`
    above would skip stubbing `GITHUB_TOKEN_RO` for it, so it'd see whatever
    real token happens to be in a developer's shell, contradicting both this
    fixture's own docstring and CONTRIBUTING.md's "no credentials required"
    claim. Making the pairing a collection-time error means a missing
    `integration` marker fails loudly and immediately, rather than only
    being caught by a reviewer noticing the omission.
    """
    offending = [
        item.nodeid
        for item in items
        if item.get_closest_marker("needs_real_github_token")
        and not item.get_closest_marker("integration")
    ]
    if offending:
        raise pytest.UsageError(
            "`needs_real_github_token` must always be paired with `integration` -- "
            "otherwise the test would run in the default suite against a real token "
            "from the developer's shell. Offending test(s):\n"
            + "\n".join(f"  - {nodeid}" for nodeid in offending)
        )
