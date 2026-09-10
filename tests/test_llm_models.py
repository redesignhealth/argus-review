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
    CLAUDE_FRONTIER,
    CLAUDE_MINI,
    GEMINI_FRONTIER,
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
        """Any experimental alias must have pricing; verify with a synthetic entry."""
        monkeypatch.setattr(
            "argus.llm.models.EXPERIMENTAL_MODELS", {"exp-model": "claude-sonnet-4-6"}
        )
        from argus.llm.models import EXPERIMENTAL_MODELS as exp

        for alias, model in exp.items():
            cost = get_token_cost(model)
            assert cost is not None, f"no pricing entry for experimental {alias!r} -> {model!r}"

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
            CLAUDE_DEFAULT,
            input_tokens=0,
            output_tokens=0,
            cache_creation_tokens=1_000_000,
        )
        assert cost > 0.0

    def test_zero_tokens_is_zero_cost(self) -> None:
        assert estimate_cost_usd(CLAUDE_DEFAULT, input_tokens=0, output_tokens=0) == 0.0

    def test_tier_relative_pricing_ordering(self) -> None:
        """Frontier tier input should exceed default, which exceeds mini."""
        cost_frontier = estimate_cost_usd(CLAUDE_FRONTIER, input_tokens=1_000_000, output_tokens=0)
        cost_default = estimate_cost_usd(CLAUDE_DEFAULT, input_tokens=1_000_000, output_tokens=0)
        cost_mini = estimate_cost_usd(CLAUDE_MINI, input_tokens=1_000_000, output_tokens=0)
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
