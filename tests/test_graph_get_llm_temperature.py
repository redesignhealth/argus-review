"""Unit tests for _get_llm's temperature-stripping guard.

Whatever CLAUDE_FRONTIER / CLAUDE_DEFAULT currently resolve to (claude-fable-5
and claude-sonnet-4-6 as of writing) reject an explicit ``temperature`` kwarg
outright (``invalid_request_error: "temperature is deprecated for this
model"``), while claude-haiku-4-5 (CLAUDE_MINI) accepts it fine. ``_get_llm``
must therefore drop ``temperature`` from the kwargs it hands to
``init_chat_model`` for the former and pass it through for the latter.

Tests import CLAUDE_FRONTIER/CLAUDE_DEFAULT rather than hardcoding model
strings so they keep exercising the live alias instead of a stale literal
after a registry bump. ``argus/graph.py``'s ``_TEMPERATURE_UNSUPPORTED_MODELS``
is a union of ALIAS_MAP's fixed pins AND these constants, minus CLAUDE_MINI --
NOT derived from these constants alone -- specifically so that overriding
CLAUDE_DEFAULT via ARGUS_SPECIALIST_MODEL to a fixed-pin value (e.g.
claude-sonnet-5, the previous default) still strips temperature, while an
override that collides with CLAUDE_MINI's own value (e.g. claude-haiku-4-5)
never does. The two regression tests below verify _get_llm's consumption of
that formula's *outcome* under each override scenario via
``monkeypatch.setattr`` on ``_TEMPERATURE_UNSUPPORTED_MODELS`` directly,
rather than ``importlib.reload``-ing the full argus.graph/argus.llm.models
modules: a full reload re-executes argus.graph's module-level side effects
(compiled LangGraph StateGraph, the _REVIEWER_EXECUTOR singleton) well
beyond what these tests need, which an earlier version of this file did and
Argus's round-3 review flagged as fragile against future module-level state
additions. The override formula's actual resolution (env var -> CLAUDE_DEFAULT)
is independently covered by tests/test_llm_models_override.py, which reloads
only the lightweight argus.llm.models module.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from argus.graph import _TEMPERATURE_UNSUPPORTED_MODELS, _get_llm
from argus.llm.models import ALIAS_MAP, CLAUDE_DEFAULT, CLAUDE_FRONTIER, CLAUDE_MINI


def _patched_settings() -> MagicMock:
    settings = MagicMock()
    settings.anthropic_credential = ("x-api-key", "test-anthropic-key")
    return settings


def test_get_llm_strips_temperature_for_claude_default() -> None:
    with (
        patch("argus.graph.get_settings", return_value=_patched_settings()),
        patch("argus.graph.init_chat_model") as mock_init_chat_model,
    ):
        _get_llm(f"anthropic:{CLAUDE_DEFAULT}", temperature=0)

    mock_init_chat_model.assert_called_once()
    _, kwargs = mock_init_chat_model.call_args
    assert "temperature" not in kwargs


def test_get_llm_strips_temperature_for_claude_frontier() -> None:
    with (
        patch("argus.graph.get_settings", return_value=_patched_settings()),
        patch("argus.graph.init_chat_model") as mock_init_chat_model,
    ):
        _get_llm(f"anthropic:{CLAUDE_FRONTIER}", temperature=0.7)

    mock_init_chat_model.assert_called_once()
    _, kwargs = mock_init_chat_model.call_args
    assert "temperature" not in kwargs


def test_get_llm_passes_temperature_for_claude_haiku_4_5() -> None:
    with (
        patch("argus.graph.get_settings", return_value=_patched_settings()),
        patch("argus.graph.init_chat_model") as mock_init_chat_model,
    ):
        _get_llm("anthropic:claude-haiku-4-5", temperature=0)

    mock_init_chat_model.assert_called_once()
    _, kwargs = mock_init_chat_model.call_args
    assert kwargs.get("temperature") == 0


def test_get_llm_omits_temperature_when_none_regardless_of_model() -> None:
    with (
        patch("argus.graph.get_settings", return_value=_patched_settings()),
        patch("argus.graph.init_chat_model") as mock_init_chat_model,
    ):
        _get_llm("anthropic:claude-haiku-4-5", temperature=None)

    mock_init_chat_model.assert_called_once()
    _, kwargs = mock_init_chat_model.call_args
    assert "temperature" not in kwargs


def _temperature_unsupported_models_under_override(overridden_default: str) -> frozenset[str]:
    """Reproduce argus.graph's `_TEMPERATURE_UNSUPPORTED_MODELS` construction
    formula for a hypothetical ARGUS_SPECIALIST_MODEL override, without
    reloading argus.graph/argus.llm.models (see this module's docstring for
    why). Kept as a single source of truth for the formula so the two
    regression tests below can't silently drift from what graph.py actually
    computes -- if graph.py's formula ever changes, this helper (and thus
    both tests) must be updated to match, which is the point.
    """
    return frozenset(
        {
            ALIAS_MAP["claude-frontier"],
            ALIAS_MAP["claude-default"],
            CLAUDE_FRONTIER,
            overridden_default,
        }
    ) - {CLAUDE_MINI}


def test_helper_matches_real_formula_when_unoverridden() -> None:
    """The hand-copied helper above is only trustworthy if it's actually
    asserted equal to the real argus.graph._TEMPERATURE_UNSUPPORTED_MODELS
    at least once -- otherwise a future edit to the real formula could
    silently diverge from every regression test using the helper while they
    all stay green. Uses CLAUDE_DEFAULT (the live, unoverridden value) as
    the "overridden_default" argument, since in the no-override case
    CLAUDE_DEFAULT *is* what the real formula's own CLAUDE_DEFAULT term
    resolves to."""
    assert _temperature_unsupported_models_under_override(CLAUDE_DEFAULT) == (
        _TEMPERATURE_UNSUPPORTED_MODELS
    )


def test_temperature_guard_still_applies_for_a_never_seen_override_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genuinely novel model id (not a fixed ALIAS_MAP pin, not CLAUDE_MINI)
    must still land in the guard set via the CLAUDE_DEFAULT term -- this is
    the exact scenario the (now-corrected) misleading comment on
    _TEMPERATURE_UNSUPPORTED_MODELS used to claim was NOT covered."""
    overridden_set = _temperature_unsupported_models_under_override("claude-sonnet-9-9")
    assert "claude-sonnet-9-9" in overridden_set

    with (
        monkeypatch.context() as m,
        patch("argus.graph.get_settings", return_value=_patched_settings()),
        patch("argus.graph.init_chat_model") as mock_init_chat_model,
    ):
        m.setattr("argus.graph._TEMPERATURE_UNSUPPORTED_MODELS", overridden_set)
        from argus.graph import _get_llm as _get_llm_live

        _get_llm_live("anthropic:claude-sonnet-9-9", temperature=0)

    mock_init_chat_model.assert_called_once()
    _, kwargs = mock_init_chat_model.call_args
    assert "temperature" not in kwargs


def test_temperature_guard_still_applies_when_specialist_model_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for the round-2 Argus finding on this PR: an earlier
    version of _TEMPERATURE_UNSUPPORTED_MODELS was derived from ALIAS_MAP's
    fixed pins ALONE, not the (possibly overridden) CLAUDE_DEFAULT/
    CLAUDE_FRONTIER constants. That broke run_preflight_check, which calls
    _get_llm(f"anthropic:{CLAUDE_DEFAULT}", temperature=0): setting
    ARGUS_SPECIALIST_MODEL=claude-sonnet-5 (the previous default) resolved
    CLAUDE_DEFAULT to a model no longer in the guard set, forwarding
    temperature=0 straight through to a model verified to hard-reject it.
    """
    overridden_set = _temperature_unsupported_models_under_override("claude-sonnet-5")
    assert "claude-sonnet-5" in overridden_set

    with (
        monkeypatch.context() as m,
        patch("argus.graph.get_settings", return_value=_patched_settings()),
        patch("argus.graph.init_chat_model") as mock_init_chat_model,
    ):
        m.setattr("argus.graph._TEMPERATURE_UNSUPPORTED_MODELS", overridden_set)
        from argus.graph import _get_llm as _get_llm_live

        _get_llm_live("anthropic:claude-sonnet-5", temperature=0)

    mock_init_chat_model.assert_called_once()
    _, kwargs = mock_init_chat_model.call_args
    assert "temperature" not in kwargs


def test_temperature_guard_does_not_swallow_claude_mini_when_specialist_model_collides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for a round-3 Argus finding on this PR: the union fix
    above (test_temperature_guard_still_applies_when_specialist_model_overridden)
    was over-inclusive. Setting ARGUS_SPECIALIST_MODEL=claude-haiku-4-5 makes
    CLAUDE_DEFAULT resolve to the SAME value as CLAUDE_MINI, pulling
    CLAUDE_MINI's own resolved value into _TEMPERATURE_UNSUPPORTED_MODELS via
    the CLAUDE_DEFAULT term of the union. _match_dismissals
    (argus/graph.py) calls `_get_llm(f"anthropic:{CLAUDE_MINI}",
    temperature=0)` for deterministic dismissal-index matching -- that
    temperature=0 would then be silently stripped, breaking the exact
    invariant this set's own comment documents ("claude-haiku-4-5
    (CLAUDE_MINI) accepts it fine, so is deliberately not included here").
    The explicit `- {CLAUDE_MINI}` exclusion in the set's construction is
    what prevents this.
    """
    overridden_set = _temperature_unsupported_models_under_override(CLAUDE_MINI)
    assert CLAUDE_MINI not in overridden_set

    with (
        monkeypatch.context() as m,
        patch("argus.graph.get_settings", return_value=_patched_settings()),
        patch("argus.graph.init_chat_model") as mock_init_chat_model,
    ):
        m.setattr("argus.graph._TEMPERATURE_UNSUPPORTED_MODELS", overridden_set)
        from argus.graph import _get_llm as _get_llm_live

        _get_llm_live(f"anthropic:{CLAUDE_MINI}", temperature=0)

    mock_init_chat_model.assert_called_once()
    _, kwargs = mock_init_chat_model.call_args
    assert kwargs.get("temperature") == 0
