"""Tests for the pricing integration in argus.llm: get_token_cost + estimate_cost_usd.

The alias registry itself (resolve/infer_provider/build_chat_model) predates
this change and has no dedicated test file; these tests cover only the
cost-estimation surface.
"""

from __future__ import annotations

import pytest

from argus.llm.models import (
    ALIAS_MAP,
    CLAUDE_DEFAULT,
    EXPERIMENTAL_MODELS,
    GEMINI_FRONTIER,
    GPT_FRONTIER,
    GPT_MINI,
    estimate_cost_usd,
)
from argus.llm.pricing import get_token_cost


class TestPricingLookup:
    def test_every_alias_map_model_has_a_price_entry(self) -> None:
        """Every concrete model the registry currently resolves to must have
        a pricing entry in litellm so cost estimation never returns 0.0 for a
        legitimate registry member."""
        for alias, model in ALIAS_MAP.items():
            cost = get_token_cost(model)
            assert cost is not None, f"no pricing entry for {alias!r} -> {model!r}"

    def test_experimental_models_coverage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Any experimental alias must have pricing; verify with both real and synthetic entries."""
        for alias, model in EXPERIMENTAL_MODELS.items():
            cost = get_token_cost(model)
            assert cost is not None, f"no pricing entry for experimental {alias!r} -> {model!r}"

        monkeypatch.setattr(
            "argus.llm.models.EXPERIMENTAL_MODELS", {"exp-model": ALIAS_MAP["claude-default"]}
        )
        from argus.llm.models import EXPERIMENTAL_MODELS as exp

        for alias, model in exp.items():
            cost = get_token_cost(model)
            assert cost is not None, (
                f"no pricing entry for synthetic experimental {alias!r} -> {model!r}"
            )

    def test_claude_default_pricing_rates(self) -> None:
        """Regression guard: claude-sonnet-4-6 rates in litellm must match
        expected rates ($3/$15/$0.30 per Mtok)."""
        cost = get_token_cost(CLAUDE_DEFAULT)
        assert cost is not None
        assert cost.input_cost_per_token == pytest.approx(3e-6)
        assert cost.output_cost_per_token == pytest.approx(15e-6)
        assert cost.cache_read_cost_per_token == pytest.approx(0.3e-6)

    def test_gemini_pricing_is_real_not_a_placeholder(self) -> None:
        """Gemini has a real, reachable runner (argus.gemini_runner) -- its
        pricing must exist in litellm with cache-read cheaper than input."""
        cost = get_token_cost(GEMINI_FRONTIER)
        assert cost is not None
        assert cost.input_cost_per_token > 0
        assert cost.output_cost_per_token > 0
        assert cost.cache_read_cost_per_token > 0
        assert cost.cache_read_cost_per_token < cost.input_cost_per_token


class TestEstimateCostUsd:
    def test_computes_expected_cost_for_claude_default(self) -> None:
        cost = estimate_cost_usd(CLAUDE_DEFAULT, input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost == pytest.approx(3.00 + 15.00)

    def test_cached_input_tokens_billed_separately_at_cache_read_rate(self) -> None:
        """cached_input_tokens is additive (billed at the cache-read rate),
        not a subset subtracted from input_tokens."""
        cost = estimate_cost_usd(
            CLAUDE_DEFAULT,
            input_tokens=1_000_000,
            output_tokens=0,
            cached_input_tokens=1_000_000,
        )
        assert cost == pytest.approx(3.00 + 0.30)

    def test_cache_creation_tokens_billed(self) -> None:
        """cache_creation_tokens is billed at the cache-creation rate."""
        cost = estimate_cost_usd(
            ALIAS_MAP["claude-default"],
            input_tokens=0,
            output_tokens=0,
            cache_creation_tokens=1_000_000,
        )
        assert cost == pytest.approx(3.75)

    def test_zero_tokens_is_zero_cost(self) -> None:
        assert (
            estimate_cost_usd(ALIAS_MAP["claude-default"], input_tokens=0, output_tokens=0) == 0.0
        )

    def test_tier_relative_pricing_ordering(self) -> None:
        """Frontier tier input should exceed default, which exceeds mini."""
        cost_frontier = estimate_cost_usd(
            ALIAS_MAP["claude-frontier"], input_tokens=1_000_000, output_tokens=0
        )
        cost_default = estimate_cost_usd(
            ALIAS_MAP["claude-default"], input_tokens=1_000_000, output_tokens=0
        )
        cost_mini = estimate_cost_usd(
            ALIAS_MAP["claude-mini"], input_tokens=1_000_000, output_tokens=0
        )
        assert cost_frontier > cost_default > cost_mini > 0.0

    def test_unknown_model_returns_zero_cost_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING", logger="argus.llm.pricing"):
            cost = estimate_cost_usd("not-a-real-model", input_tokens=100, output_tokens=100)
        assert cost == 0.0
        assert "No litellm pricing entry" in caplog.text

    def test_gemini_model_estimates_real_nonzero_cost(self) -> None:
        """Gemini pricing is real in litellm, so token count must estimate
        a real, nonzero cost."""
        cost = estimate_cost_usd(GEMINI_FRONTIER, input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost > 0.0

    def test_openai_model_estimates_real_nonzero_cost(self) -> None:
        """OpenAI pricing is real in litellm, so token count must estimate
        a real, nonzero cost."""
        cost = estimate_cost_usd(GPT_FRONTIER, input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost > 0.0

    def test_gpt_frontier_pinned_to_approved_model(self) -> None:
        """Regression guard for a round-1 Argus BLOCKING finding on this
        PR: gpt-frontier previously resolved to gpt-5.5, which was not on
        the approved model list and may not have been shipped by OpenAI
        yet. The alias has since been legitimately re-bumped to gpt-5.6-sol
        (the entire gpt-5.6 family is now on the approved model list -- see
        pr-review-specialist-llm-patterns.md). Pin the alias to that
        approved value so a future accidental re-bump to an unapproved
        model string is caught here instead of at review time."""
        assert ALIAS_MAP["gpt-frontier"] == "gpt-5.6-sol"
        assert GPT_FRONTIER == "gpt-5.6-sol"

    def test_gpt_mini_pinned_to_approved_model(self) -> None:
        """Parallel regression guard for gpt-mini: it carries the exact
        same 'bump once shipped' comment pattern gpt-frontier did, and
        gpt-5.5-mini is equally unapproved. Pin it too so the same
        BLOCKING-finding class can't recur on this alias instead.

        Also exercises estimate_cost_usd(GPT_MINI, ...) directly (not just
        ALIAS_MAP/get_token_cost, covered elsewhere) so this test's home in
        TestEstimateCostUsd actually corresponds to the end-to-end cost
        pipeline it's implicitly claiming to cover for this alias."""
        assert ALIAS_MAP["gpt-mini"] == "gpt-5.4-mini"
        assert GPT_MINI == "gpt-5.4-mini"
        cost = estimate_cost_usd(GPT_MINI, input_tokens=1000, output_tokens=500)
        assert cost > 0.0


class TestResolveOverrides:
    def test_resolve_claude_default_honors_env_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import importlib
        import os
        import argus.llm.models as models

        orig = os.environ.get("ARGUS_SPECIALIST_MODEL")
        monkeypatch.setenv("ARGUS_SPECIALIST_MODEL", "claude-opus-5")
        importlib.reload(models)
        try:
            assert models.resolve("claude-default") == "claude-opus-5"
        finally:
            if orig is not None:
                monkeypatch.setenv("ARGUS_SPECIALIST_MODEL", orig)
            else:
                monkeypatch.delenv("ARGUS_SPECIALIST_MODEL", raising=False)
            importlib.reload(models)


class TestResolveExperimentalModels:
    """Regression guard: argus.bench's ``_VALID_MODEL_ALIASES`` accepts any
    ``EXPERIMENTAL_MODELS`` key as a valid bench ``model`` value, so
    ``resolve()`` must actually be able to turn one into a concrete model
    string too -- otherwise a bench config that validates cleanly at load
    time could still raise ``KeyError`` the first time that role runs.
    """

    def test_experimental_model_alias_resolves_to_its_concrete_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import argus.llm.models as models

        monkeypatch.setattr(models, "EXPERIMENTAL_MODELS", {"exp-model": "claude-opus-6-preview"})

        assert models.resolve("exp-model") == "claude-opus-6-preview"

    def test_experimental_model_alias_does_not_raise_keyerror(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The specific failure mode reported: an accepted (validated) bench
        config still blew up with KeyError the first time the role was
        actually used, because resolve() only ever consulted ALIAS_MAP."""
        import argus.llm.models as models

        monkeypatch.setattr(models, "EXPERIMENTAL_MODELS", {"exp-model": "gemini-4-preview"})

        try:
            result = models.resolve("exp-model")
        except KeyError:
            pytest.fail("resolve() raised KeyError for a declared EXPERIMENTAL_MODELS alias")
        assert result == "gemini-4-preview"

    def test_still_raises_keyerror_for_truly_unknown_alias(self) -> None:
        """Unregistered in both ALIAS_MAP and EXPERIMENTAL_MODELS -> still a
        loud failure, not a silent pass-through."""
        from argus.llm.models import resolve

        with pytest.raises(KeyError):
            resolve("not-a-real-alias-anywhere")
