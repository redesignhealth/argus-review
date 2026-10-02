"""Native structured output binding for the non-streaming LLM stages.

TECH-7156: ``run_preflight_check`` and ``check_coverage`` must request
Anthropic native structured output explicitly via
``with_structured_output(..., method="json_schema")`` — without the explicit
method, langchain-anthropic defaults to tool-calling wire format. Neither
stage had a direct test of its LLM binding chain before (the existing preflight
tests in test_catchup_gate.py and coverage tests in
test_graph_timeout_surfacing.py patch these functions whole), so these are the
narrowest possible tests of that binding surface: each patches the
``init_chat_model`` chain one level deep and asserts the exact
``with_structured_output`` call.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from argus.pipeline_models import CoverageResult, ReviewPlan, SystemReviewResult


def _patched_settings() -> MagicMock:
    """A get_settings() stand-in carrying only what _get_llm reads."""
    s = MagicMock()
    s.anthropic_credential = ("ANTHROPIC_API_KEY", "test-key")
    return s


def _structured_llm(result: Any) -> MagicMock:
    """Mock matching the real chain:
    init_chat_model(...).with_structured_output(...).ainvoke(messages)."""
    structured = MagicMock()
    structured.ainvoke = AsyncMock(return_value=result)
    llm = MagicMock()
    llm.with_structured_output = MagicMock(return_value=structured)
    return llm


@pytest.mark.asyncio
async def test_run_preflight_check_uses_native_json_schema_structured_output() -> None:
    """run_preflight_check must pin with_structured_output to
    method="json_schema" (Anthropic native structured output), not the
    tool-calling default."""
    from argus.graph import PreflightResult, run_preflight_check

    llm = _structured_llm(PreflightResult(route="lite", reason="trivial change"))

    with (
        patch("argus.graph.get_settings", return_value=_patched_settings()),
        patch("argus.graph.init_chat_model", return_value=llm),
        patch("argus.graph.fetch_prompt", new=AsyncMock(return_value="preflight-prompt")),
    ):
        result = await run_preflight_check("diff --git a/x b/x\n+x = 1", None)

    llm.with_structured_output.assert_called_once_with(PreflightResult, method="json_schema")
    assert result.route == "lite"


@pytest.mark.asyncio
async def test_check_coverage_uses_native_json_schema_structured_output() -> None:
    """check_coverage's LLM triage call (the uncovered-files branch) must pin
    with_structured_output to method="json_schema" (Anthropic native
    structured output)."""
    from argus.graph import check_coverage

    # One manifest file explored, one uncovered — forces the LLM triage
    # branch instead of the mechanical all-covered early return.
    plan = ReviewPlan.model_validate(
        {
            "system_groups": [
                {
                    "name": "api",
                    "files": ["a.py", "b.py"],
                    "conventions": "",
                    "review_focus": "",
                    "specialists_needed": [],
                }
            ],
            "cross_cutting_concerns": [],
            "file_manifest": [
                {"path": "a.py", "change_type": "modified"},
                {"path": "b.py", "change_type": "modified"},
            ],
        }
    )
    findings = [
        SystemReviewResult(system_group="api", findings=[], files_explored=["a.py"], cost_usd=0.0)
    ]
    llm = _structured_llm(CoverageResult(is_covered=True, gaps=[]))

    with (
        patch("argus.graph.get_settings", return_value=_patched_settings()),
        patch("argus.graph.init_chat_model", return_value=llm),
        patch("argus.graph.fetch_prompt", new=AsyncMock(return_value="coverage-prompt")),
    ):
        result = await check_coverage(plan, findings)

    llm.with_structured_output.assert_called_once_with(CoverageResult, method="json_schema")
    assert result.is_covered is True
