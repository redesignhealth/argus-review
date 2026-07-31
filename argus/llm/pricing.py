"""Per-token model pricing, sourced from litellm's published cost table.

Single source of truth for the $/token rates Argus needs to approximate cost
outside the ``agent_runs`` tracking pipeline (currently just the lite-review
path in ``graph.py``, which bypasses per-agent cost tracking entirely).

Deliberately not a hand-maintained table: a prior version of this file
carried bare float constants for one specific model, copied forward from an
earlier model generation without updating the dollar amounts when the alias
moved on -- silently wrong for months with nothing to catch it (see
TECH-4674, the Argus-side half of TECH-4635's pricing-duplication cleanup).
litellm is already the pricing source ``claude-usage-tracker`` uses
internally at Redesign Health, so this keeps Argus's own approximation
consistent with that source rather than inventing a second one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import litellm

# Argus only ever reads the static litellm.model_cost dict below, never
# litellm's completion()/acompletion() call path -- so the usage-telemetry
# callbacks that fire on an actual LLM call never trigger regardless. This
# is a belt-and-suspenders guard against that assumption changing (e.g. a
# future call site importing this module and also using litellm for a real
# request), not a response to telemetry firing today.
litellm.telemetry = False

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TokenCost:
    """Per-token USD rates for one model, as published by litellm."""

    input_cost_per_token: float
    output_cost_per_token: float
    cache_read_cost_per_token: float
    cache_write_cost_per_token: float


def get_token_cost(model: str) -> TokenCost | None:
    """Look up per-token pricing for ``model`` from litellm's cost table.

    ``model`` must be a bare litellm model name (e.g. ``"claude-sonnet-5"``),
    not a LangChain provider-prefixed string (``"anthropic:claude-sonnet-5"``)
    -- the latter has no entry in litellm's table and just returns None with
    a warning, same as any other unknown model.

    Returns ``None`` (never raises) if litellm has no entry for the model --
    e.g. a brand-new model release litellm's bundled data hasn't caught up
    with yet. Callers must treat that as "cost unknown for this call," not
    a reason to fail the review; cost tracking is observability, not a
    correctness gate.
    """
    row = litellm.model_cost.get(model)
    if row is None:
        logger.warning(
            "No litellm pricing entry for model %r -- cost for this call will be omitted "
            "from the aggregate. If this model was just released, litellm's bundled cost "
            "table may not have caught up yet.",
            model,
        )
        return None
    return TokenCost(
        input_cost_per_token=row.get("input_cost_per_token", 0.0),
        output_cost_per_token=row.get("output_cost_per_token", 0.0),
        cache_read_cost_per_token=row.get("cache_read_input_token_cost", 0.0),
        cache_write_cost_per_token=row.get("cache_creation_input_token_cost", 0.0),
    )
