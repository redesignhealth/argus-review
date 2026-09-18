"""Tests for argus.llm.usage -- the per-stage LLM cost/duration ledger.

Covers: pricing a real LLM callback payload against a known (monkeypatched)
get_token_cost result, an unknown model contributing 0.0 without raising,
and that costs accumulate correctly across stages and repeated calls.
"""

from __future__ import annotations
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from argus.llm.pricing import TokenCost
from argus.llm.usage import (
    StageCostCallbackHandler,
    record_stage_cost,
    stage_costs,
    stage_ledger,
    stage_seconds,
)
from argus.pipeline_models import AgentRunData, SystemReviewResult


def _llm_result(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    cache_read: int = 0,
) -> LLMResult:
    message = AIMessage(
        content="hi",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_token_details": {"cache_read": cache_read},
        },
        response_metadata={"model_name": model_name},
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def test_callback_prices_usage_against_known_pricing(monkeypatch) -> None:
    monkeypatch.setattr(
        "argus.llm.usage.get_token_cost",
        lambda model: TokenCost(
            input_cost_per_token=0.01,
            output_cost_per_token=0.02,
            cache_read_cost_per_token=0.001,
            cache_write_cost_per_token=0.002,
        ),
    )
    with stage_ledger():
        handler = StageCostCallbackHandler("planner")
        handler.on_llm_end(_llm_result("claude-sonnet-5", input_tokens=100, output_tokens=50))
        assert stage_costs()["planner"] == 100 * 0.01 + 50 * 0.02


def test_cache_read_tokens_are_not_billed_at_the_input_rate_too(monkeypatch) -> None:
    """input_tokens is the TOTAL count; cache_read is a subset of it, not an
    addition -- 1000 input tokens with 800 cache_read must price as 200 fresh
    + 800 cached, not 1000 fresh + 800 cached."""
    monkeypatch.setattr(
        "argus.llm.usage.get_token_cost",
        lambda model: TokenCost(
            input_cost_per_token=0.01,
            output_cost_per_token=0.02,
            cache_read_cost_per_token=0.001,
            cache_write_cost_per_token=0.002,
        ),
    )
    with stage_ledger():
        handler = StageCostCallbackHandler("planner")
        handler.on_llm_end(
            _llm_result("claude-sonnet-5", input_tokens=1000, output_tokens=0, cache_read=800)
        )
        assert stage_costs()["planner"] == pytest.approx(200 * 0.01 + 800 * 0.001)


def test_unknown_model_contributes_zero_not_raise(monkeypatch) -> None:
    monkeypatch.setattr("argus.llm.usage.get_token_cost", lambda model: None)
    with stage_ledger():
        handler = StageCostCallbackHandler("writer")
        handler.on_llm_end(
            _llm_result("definitely-not-a-real-model", input_tokens=100, output_tokens=50)
        )
        assert stage_costs()["writer"] == 0.0


def test_stages_accumulate_separately_and_repeated_calls_sum() -> None:
    with stage_ledger():
        record_stage_cost("planner", 1.0, seconds=2.0)
        record_stage_cost("writer", 5.0, seconds=1.0)
        record_stage_cost("planner", 0.5, seconds=1.0)

        assert stage_costs() == {"planner": 1.5, "writer": 5.0}
        assert stage_seconds() == {"planner": 3.0, "writer": 1.0}


def test_record_stage_cost_outside_ledger_is_a_noop() -> None:
    record_stage_cost("planner", 1.0)
    assert stage_costs() == {}


@pytest.mark.asyncio
async def test_reviewer_node_records_into_ledger_across_the_asyncio_task() -> None:
    """LangGraph fans reviewer nodes out as separate asyncio Tasks, so the
    ContextVar ledger must propagate into a spawned Task, not just direct
    calls in the same coroutine frame."""
    import asyncio

    from argus.graph import _node_run_reviewer

    result = SystemReviewResult(system_group="Auth", findings=[], files_explored=[], cost_usd=2.5)
    agent_run = AgentRunData(agent_name="system:Auth", agent_type="system", duration_seconds=12.0)
    inputs = {
        "reviewer_type": "system",
        "group": {
            "name": "Auth",
            "files": ["auth.py"],
            "conventions": "",
            "review_focus": "",
            "specialists_needed": [],
        },
        "specialist": "",
        "diff": "diff --git a/x b/x",
        "plan": {},
    }

    with stage_ledger():
        with patch(
            "argus.graph.review_system_group",
            new_callable=AsyncMock,
            return_value=(result, agent_run),
        ):
            await asyncio.create_task(_node_run_reviewer(inputs, {"configurable": {}}))

        assert stage_costs()["reviewer:system/Auth"] == 2.5
        assert stage_seconds()["reviewer:system/Auth"] == 12.0
