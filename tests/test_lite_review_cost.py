"""Tests for _estimate_lite_review_cost -- the lite-review path's only cost
computation, since it bypasses agent_runs tracking entirely.

Extracted from an inline block in run_review() specifically so this math is
independently testable (it previously had zero direct test coverage).
"""

from __future__ import annotations

from argus.graph import _estimate_lite_review_cost
from argus.models import TokenUsage


def test_zero_usage_is_zero_cost() -> None:
    assert _estimate_lite_review_cost(TokenUsage(), "claude-sonnet-5") == 0.0


def test_cost_scales_with_each_token_field_independently() -> None:
    base = _estimate_lite_review_cost(TokenUsage(input_tokens=1000), "claude-sonnet-5")
    assert base > 0

    more_output = _estimate_lite_review_cost(
        TokenUsage(input_tokens=1000, output_tokens=1000), "claude-sonnet-5"
    )
    assert more_output > base

    with_cache = _estimate_lite_review_cost(
        TokenUsage(input_tokens=1000, output_tokens=1000, cache_read_tokens=1000),
        "claude-sonnet-5",
    )
    assert with_cache > more_output


def test_cache_read_is_cheaper_than_fresh_input_at_equal_token_count() -> None:
    """A cache-read-heavy call must cost less than the same volume of fresh
    input tokens -- otherwise caching would show up as a cost regression.

    Implicitly depends on litellm's claude-sonnet-5 pricing row carrying a
    non-zero cache_read_input_token_cost; if a future litellm release drops
    or renames that key, get_token_cost degrades to 0.0 for cache reads and
    this assertion would need to be revisited (test_llm_pricing.py's alias
    coverage test would also start failing first, giving an earlier signal)."""
    fresh_input_cost = _estimate_lite_review_cost(
        TokenUsage(input_tokens=10_000), "claude-sonnet-5"
    )
    cache_read_cost = _estimate_lite_review_cost(
        TokenUsage(cache_read_tokens=10_000), "claude-sonnet-5"
    )
    assert cache_read_cost < fresh_input_cost


def test_unknown_model_returns_zero_not_raise() -> None:
    """Cost tracking is observability, not a correctness gate -- a model
    litellm doesn't know about must degrade to 0.0, never raise."""
    usage = TokenUsage(input_tokens=1000, output_tokens=1000)
    assert _estimate_lite_review_cost(usage, "definitely-not-a-real-model-xyz") == 0.0
