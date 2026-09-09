"""Tests for the pricing additions to argus.llm.models: _PRICES + estimate_cost_usd.

The alias registry itself (resolve/infer_provider/build_chat_model) predates
this change and has no dedicated test file; these tests cover only the new
cost-estimation surface.
"""

from __future__ import annotations

import pytest

from argus.llm.models import (
    CLAUDE_DEFAULT,
    CLAUDE_FRONTIER,
    CLAUDE_MINI,
    GEMINI_FRONTIER,
    GPT_FRONTIER,
    _PRICES,
    estimate_cost_usd,
)


class TestPricesTable:
    def test_every_alias_map_model_has_a_price_entry(self) -> None:
        """Every concrete model the registry currently resolves to must have
        a pricing entry -- including the not-yet-reachable Gemini/GPT
        placeholders -- so estimate_cost_usd never KeyErrors for a model
        that's otherwise a legitimate registry member."""
        from argus.llm.models import ALIAS_MAP

        for alias, model in ALIAS_MAP.items():
            assert model in _PRICES, f"no _PRICES entry for {alias!r} -> {model!r}"

    def test_claude_default_pricing_matches_graph_lite_review_constants(self) -> None:
        """Regression guard: argus/graph.py's lite-review cost aggregation
        hardcodes claude-sonnet-4-6 pricing as per-token constants
        (_SONNET_INPUT_COST=3e-6, _SONNET_OUTPUT_COST=15e-6,
        _SONNET_CACHE_READ_COST=0.3e-6). _PRICES must agree (in $/Mtok)."""
        input_cost, output_cost, cache_read_cost = _PRICES[CLAUDE_DEFAULT]
        assert input_cost == pytest.approx(3.00)
        assert output_cost == pytest.approx(15.00)
        assert cache_read_cost == pytest.approx(0.30)

    def test_gemini_pricing_is_real_not_a_placeholder(self) -> None:
        """Gemini has a real, reachable runner (argus.gemini_runner) -- its
        pricing must no longer be the Track-1 $0/$0/$0 placeholder, or
        every Gemini session's cost would silently under-report as $0."""
        input_cost, output_cost, cache_read_cost = _PRICES[GEMINI_FRONTIER]
        assert input_cost > 0
        assert output_cost > 0
        assert cache_read_cost > 0
        # The entire point of explicit caching is a materially cheaper
        # cache-read rate than the full input rate.
        assert cache_read_cost < input_cost

    def test_gpt_placeholder_is_still_zero(self) -> None:
        """No OpenAI Responses runner exists yet, so this remains a
        deliberate, documented $0 placeholder until one ships."""
        assert _PRICES[GPT_FRONTIER] == (0.0, 0.0, 0.0)


class TestEstimateCostUsd:
    def test_computes_expected_cost_for_claude_default(self) -> None:
        cost = estimate_cost_usd(CLAUDE_DEFAULT, input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost == pytest.approx(3.00 + 15.00)

    def test_cached_input_tokens_billed_separately_at_cache_read_rate(self) -> None:
        """cached_input_tokens is additive (billed at the cache-read rate),
        not a subset subtracted from input_tokens -- matches the convention
        already used by argus/graph.py's cost aggregation."""
        cost = estimate_cost_usd(
            CLAUDE_DEFAULT,
            input_tokens=1_000_000,
            output_tokens=0,
            cached_input_tokens=1_000_000,
        )
        assert cost == pytest.approx(3.00 + 0.30)

    def test_zero_tokens_is_zero_cost(self) -> None:
        assert estimate_cost_usd(CLAUDE_DEFAULT, input_tokens=0, output_tokens=0) == 0.0

    def test_frontier_model_uses_its_own_pricing(self) -> None:
        cost = estimate_cost_usd(CLAUDE_FRONTIER, input_tokens=1_000_000, output_tokens=0)
        assert cost == pytest.approx(15.00)

    def test_mini_model_uses_its_own_pricing(self) -> None:
        cost = estimate_cost_usd(CLAUDE_MINI, input_tokens=1_000_000, output_tokens=0)
        assert cost == pytest.approx(0.80)

    def test_unknown_model_raises_key_error_naming_model(self) -> None:
        with pytest.raises(KeyError, match="not-a-real-model"):
            estimate_cost_usd("not-a-real-model", input_tokens=100, output_tokens=100)

    def test_gemini_model_estimates_real_nonzero_cost(self) -> None:
        """Not a placeholder anymore -- Gemini pricing is real (see
        _PRICES), so a real token count must estimate a real, nonzero
        cost rather than silently reporting $0."""
        cost = estimate_cost_usd(GEMINI_FRONTIER, input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost > 0.0
