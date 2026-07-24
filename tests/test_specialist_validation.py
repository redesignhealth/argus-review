"""Tests for specialist-name validation in the v3 review pipeline.

These guard against the regression that caused round-1 reviews to crash with
``ValueError: Unknown specialist: 'test-coverage'`` when the planner LLM
hallucinated a specialist name outside the registry.

The validation now happens in two layers:

1. **Pydantic schema** (``pipeline_models.SystemGroup.specialists_needed``)
   is typed ``list[SpecialistName]``. A planner output containing an unknown
   specialist fails ``model_validate`` with a clear ``ValidationError``.

2. **Dispatch boundary** (``graph._validate_specialist_name``) re-checks the
   value before calling ``review_specialist``. Defense in depth for any
   future caller that bypasses Pydantic validation via ``model_construct``.

The original ``ValueError`` in ``runners.run_specialist_reviewer`` is kept
as a third layer for tests that exercise that function directly.
"""

from __future__ import annotations

from typing import get_args

import pytest
from pydantic import ValidationError

from argus.graph import _SPECIALIST_ADAPTER, _validate_specialist_name
from argus.pipeline_models import (
    SpecialistName,
    SystemGroup,
)
from argus.runners import _SPECIALIST_PROMPT_MAP


# ---------------------------------------------------------------------------
# Single-source-of-truth check
# ---------------------------------------------------------------------------


def test_specialist_name_literal_matches_prompt_map() -> None:
    """The Literal type and the prompt map must list the same specialists.

    If they ever drift, the system review pipeline will fail at runtime.
    """
    literal_names = set(get_args(SpecialistName))
    prompt_map_names = set(_SPECIALIST_PROMPT_MAP.keys())
    assert literal_names == prompt_map_names, (
        "SpecialistName Literal and _SPECIALIST_PROMPT_MAP have diverged. "
        f"In Literal but not map: {literal_names - prompt_map_names}. "
        f"In map but not Literal: {prompt_map_names - literal_names}."
    )


def test_graph_adapter_validates_against_same_literal() -> None:
    """``graph._SPECIALIST_ADAPTER`` validates against the same Literal."""
    # Spot-check a few names — full coverage is via the parametrized tests below.
    assert _SPECIALIST_ADAPTER.validate_python("security") == "security"
    with pytest.raises(Exception):  # ValidationError
        _SPECIALIST_ADAPTER.validate_python("test-coverage")


# ---------------------------------------------------------------------------
# Pydantic-layer validation (the primary defense)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "specialist",
    sorted(get_args(SpecialistName)),
)
def test_system_group_accepts_every_registered_specialist(specialist: str) -> None:
    """Every name in the Literal is accepted by SystemGroup."""
    group = SystemGroup(
        name="g",
        files=["foo.py"],
        conventions="",
        review_focus="",
        specialists_needed=[specialist],
    )
    assert group.specialists_needed == [specialist]


@pytest.mark.parametrize(
    "bad_specialist",
    [
        "test-coverage",  # the regression that motivated this fix
        "test-adequacy",
        "performance",
        "senior-swe",  # legacy v1/v2 name
        "",
        "Security",  # case-sensitive
    ],
)
def test_system_group_rejects_unknown_specialist(bad_specialist: str) -> None:
    """Unknown specialist names fail at SystemGroup validation time."""
    with pytest.raises(ValidationError) as exc_info:
        SystemGroup(
            name="g",
            files=["foo.py"],
            conventions="",
            review_focus="",
            specialists_needed=[bad_specialist],
        )
    assert "specialists_needed" in str(exc_info.value)


def test_system_group_accepts_empty_specialists_list() -> None:
    """An empty ``specialists_needed`` is the common case for non-specialist groups."""
    group = SystemGroup(
        name="g",
        files=["README.md"],
        conventions="",
        review_focus="",
    )
    assert group.specialists_needed == []


# ---------------------------------------------------------------------------
# Dispatch-boundary validation (defense in depth)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("specialist", sorted(get_args(SpecialistName)))
def test_validate_specialist_name_accepts_registered(specialist: str) -> None:
    assert _validate_specialist_name(specialist) == specialist


@pytest.mark.parametrize(
    "bad_specialist",
    ["test-coverage", "test-adequacy", "", "unknown"],
)
def test_validate_specialist_name_rejects_unknown(bad_specialist: str) -> None:
    with pytest.raises(ValueError, match="Invalid specialist"):
        _validate_specialist_name(bad_specialist)
