"""Tests for the ARGUS_SPECIALIST_MODEL/ARGUS_FRONTIER_MODEL env-var override
in argus/llm/models.py.

CLAUDE_DEFAULT/CLAUDE_FRONTIER/CLAUDE_OPUS are module-level constants resolved
once at import time, so exercising the override requires reloading the module
under a patched environment rather than just setting the env var and reading
the already-imported constant.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from typing import Iterator

import pytest

import argus.llm.models as models

_OVERRIDE_ENV_VARS = ("ARGUS_SPECIALIST_MODEL", "ARGUS_FRONTIER_MODEL")


@pytest.fixture(autouse=True)
def _reload_after_test() -> Iterator[None]:
    """Restore the module to its pre-test state after each test, so an
    override set by one test can't leak into an unrelated test that imports
    the already-cached module.

    Snapshots and restores each override var's original present/absent
    state rather than unconditionally popping it: an unconditional pop would
    delete a real ARGUS_SPECIALIST_MODEL/ARGUS_FRONTIER_MODEL that existed in
    the ambient environment before the test session started, even after
    monkeypatch's own (earlier-running) teardown had already restored it.
    """
    saved = {var: os.environ.get(var) for var in _OVERRIDE_ENV_VARS}
    yield
    for var, value in saved.items():
        if value is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = value
    importlib.reload(models)


def test_claude_default_is_sonnet_4_6_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARGUS_SPECIALIST_MODEL", raising=False)
    importlib.reload(models)
    assert models.CLAUDE_DEFAULT == "claude-sonnet-4-6"


def test_specialist_override_only_affects_claude_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARGUS_SPECIALIST_MODEL", "claude-sonnet-5")
    monkeypatch.delenv("ARGUS_FRONTIER_MODEL", raising=False)
    importlib.reload(models)

    assert models.CLAUDE_DEFAULT == "claude-sonnet-5"
    assert models.CLAUDE_FRONTIER == models.ALIAS_MAP["claude-frontier"]
    assert models.CLAUDE_OPUS == models.ALIAS_MAP["claude-opus"]


def test_frontier_override_affects_both_frontier_and_opus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARGUS_FRONTIER_MODEL", "claude-opus-5")
    monkeypatch.delenv("ARGUS_SPECIALIST_MODEL", raising=False)
    importlib.reload(models)

    assert models.CLAUDE_FRONTIER == "claude-opus-5"
    assert models.CLAUDE_OPUS == "claude-opus-5"
    assert models.CLAUDE_DEFAULT == models.ALIAS_MAP["claude-default"]


def test_empty_string_env_var_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty-string env var (e.g. a shell export with no value) must not
    resolve to an empty model id -- that would silently break every LLM
    call site importing the constant."""
    monkeypatch.setenv("ARGUS_SPECIALIST_MODEL", "")
    importlib.reload(models)
    assert models.CLAUDE_DEFAULT == models.ALIAS_MAP["claude-default"]


def test_alias_key_env_var_resolves_through_alias_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """Passing a registry alias key (rather than a concrete model id) as the
    override value must resolve the same way resolve() would, matching the
    guarantee _env_override's docstring makes -- otherwise a caller who
    reaches for an alias like the CLI's own examples get a literal,
    unrecognized provider id sent straight to Anthropic/OpenAI."""
    monkeypatch.setenv("ARGUS_SPECIALIST_MODEL", "claude-opus")
    importlib.reload(models)
    assert models.CLAUDE_DEFAULT == models.ALIAS_MAP["claude-opus"]


def test_unrecognized_alias_like_value_warns_but_does_not_raise(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A typo'd alias key (not a recognized ALIAS_MAP entry, and not shaped
    like a real provider-prefixed model id either) must not raise -- a raise
    here would also reject legitimate future model ids this registry simply
    doesn't know about yet -- but should log a warning so the mistake
    surfaces closer to its source than an opaque provider-side error."""
    monkeypatch.setenv("ARGUS_SPECIALIST_MODEL", "cluade-default")
    with caplog.at_level("WARNING", logger="argus.llm.models"):
        importlib.reload(models)
    assert models.CLAUDE_DEFAULT == "cluade-default"
    assert any("cluade-default" in record.message for record in caplog.records)


def test_valid_looking_model_id_does_not_warn(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A real, provider-prefixed model id this registry has never seen
    before (e.g. a brand-new Anthropic release) must pass through silently
    -- it's not a typo, just a model not yet added to ALIAS_MAP."""
    monkeypatch.setenv("ARGUS_SPECIALIST_MODEL", "claude-sonnet-9-9")
    with caplog.at_level("WARNING", logger="argus.llm.models"):
        importlib.reload(models)
    assert models.CLAUDE_DEFAULT == "claude-sonnet-9-9"
    assert caplog.records == []


def test_cli_module_import_does_not_pull_in_llm_models() -> None:
    """Guards the import-order contract this override mechanism depends on:
    argus.cli must not import argus.llm.models (directly or transitively) at
    module-import time -- only later, inside _run_review's deferred imports,
    after the CLI flags have already been applied to os.environ. A future
    refactor that hoists e.g. `from argus.graph import run_review` to module
    scope would silently disable both --specialist-model and
    --frontier-model while leaving CI green, since the override env vars
    would then be set by _run_review AFTER argus.llm.models' constants were
    already resolved at process-startup import time -- exactly the failure
    mode a plain unit test on the already-imported module can't catch.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import argus.cli, sys; assert 'argus.llm.models' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
