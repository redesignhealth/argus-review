"""Tests for argus.llm.pricing -- the litellm-backed per-token cost lookup.

Covers: known-model lookup shape, unknown-model graceful None (not a raise),
and that the live configured aliases (CLAUDE_DEFAULT/CLAUDE_FRONTIER/
CLAUDE_OPUS/CLAUDE_MINI) actually resolve against litellm's real data --
the "alias coverage" test that would have caught the model registry
silently outrunning what litellm's pricing table knows about.
"""

from __future__ import annotations

import pytest

from argus.llm.models import CLAUDE_DEFAULT, CLAUDE_FRONTIER, CLAUDE_MINI, CLAUDE_OPUS
from argus.llm.pricing import TokenCost, get_token_cost


@pytest.mark.parametrize("model", [CLAUDE_DEFAULT, CLAUDE_FRONTIER, CLAUDE_OPUS, CLAUDE_MINI])
def test_live_configured_aliases_have_pricing_entries(model: str) -> None:
    """The models argus/llm/models.py's ALIAS_MAP actually resolves to today
    must have a real litellm pricing entry, with sane per-token rates.
    Guards against the exact failure mode this module replaces: a model
    alias bumped forward with nothing checking whether its cost data came
    along too. Parametrized over the live aliases (not a hardcoded literal)
    so a future ALIAS_MAP bump can't silently keep testing the old model."""
    cost = get_token_cost(model)
    assert cost is not None, (
        f"{model!r} has no litellm pricing entry -- lite-review cost "
        "tracking would silently omit this model's cost."
    )
    assert isinstance(cost, TokenCost)
    assert cost.input_cost_per_token > 0
    assert cost.output_cost_per_token > 0
    # Output costs more than input for every commercial LLM pricing tier.
    assert cost.output_cost_per_token > cost.input_cost_per_token
    # Cache read must be cheaper than a fresh input token, or caching would
    # never be worth doing.
    assert cost.cache_read_cost_per_token < cost.input_cost_per_token


def test_unknown_model_returns_none_not_raise() -> None:
    """A model litellm has never heard of must degrade to 'cost unknown',
    not crash the caller -- cost tracking is observability, not a gate."""
    assert get_token_cost("definitely-not-a-real-model-xyz") is None
