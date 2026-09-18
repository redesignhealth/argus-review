"""Per-stage LLM cost/duration tracking, one ``ContextVar`` ledger per review."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import LLMResult
from openai.types.responses.response_usage import ResponseUsage

from argus.llm.pricing import get_token_cost

logger = logging.getLogger(__name__)

_costs: ContextVar[dict[str, float] | None] = ContextVar("_stage_costs", default=None)
_seconds: ContextVar[dict[str, float] | None] = ContextVar("_stage_seconds", default=None)


@contextmanager
def stage_ledger() -> Iterator[None]:
    """Install a fresh per-stage cost/duration ledger for one review."""
    cost_token = _costs.set({})
    seconds_token = _seconds.set({})
    try:
        yield
    finally:
        _costs.reset(cost_token)
        _seconds.reset(seconds_token)


def record_stage_cost(stage: str, cost_usd: float, seconds: float = 0.0) -> None:
    """Add ``cost_usd``/``seconds`` to ``stage``'s running total; a no-op outside ``stage_ledger()``."""
    costs = _costs.get()
    if costs is not None:
        costs[stage] = costs.get(stage, 0.0) + cost_usd
    seconds_by_stage = _seconds.get()
    if seconds_by_stage is not None:
        seconds_by_stage[stage] = seconds_by_stage.get(stage, 0.0) + seconds


def stage_costs() -> dict[str, float]:
    return dict(_costs.get() or {})


def stage_seconds() -> dict[str, float]:
    return dict(_seconds.get() or {})


def _price_usage(model: str, usage: UsageMetadata) -> float:
    token_cost = get_token_cost(model.rsplit(":", 1)[-1])
    if token_cost is None:
        return 0.0
    input_details = usage.get("input_token_details") or {}
    cache_read = input_details.get("cache_read", 0)
    cache_creation = input_details.get("cache_creation", 0)
    # input_tokens is the TOTAL input count; cache_read/cache_creation are
    # subsets of it, not additions -- subtract before applying the input rate
    # so cached tokens aren't billed twice.
    fresh_input = max(0, usage["input_tokens"] - cache_read - cache_creation)
    return (
        fresh_input * token_cost.input_cost_per_token
        + usage["output_tokens"] * token_cost.output_cost_per_token
        + cache_read * token_cost.cache_read_cost_per_token
        + cache_creation * token_cost.cache_write_cost_per_token
    )


def price_openai_usage(model: str, usage: ResponseUsage) -> float:
    """Price an OpenAI Responses API ``usage`` object (``resp.usage``).

    Used for calls made through ``argus.openai_client`` rather than
    LangChain, which the callback handler above never sees.
    """
    token_cost = get_token_cost(model)
    if token_cost is None:
        return 0.0
    cached = usage.input_tokens_details.cached_tokens if usage.input_tokens_details else 0
    return float(
        (usage.input_tokens - cached) * token_cost.input_cost_per_token
        + usage.output_tokens * token_cost.output_cost_per_token
        + cached * token_cost.cache_read_cost_per_token
    )


class StageCostCallbackHandler(UsageMetadataCallbackHandler):
    """Prices each LLM call's usage_metadata into the active stage ledger."""

    def __init__(self, stage: str) -> None:
        super().__init__()
        self._stage = stage
        self._start = time.monotonic()

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        self._start = time.monotonic()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        super().on_llm_end(response, **kwargs)
        elapsed = time.monotonic() - self._start
        cost = sum(_price_usage(model, usage) for model, usage in self.usage_metadata.items())
        record_stage_cost(self._stage, cost, elapsed)
        self.usage_metadata.clear()
