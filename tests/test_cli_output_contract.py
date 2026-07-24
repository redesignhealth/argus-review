"""Output-contract freeze tests.

The ``argus-review-loop`` skill screen-parses two things from a CLI run:

1. The stderr/stdout summary block (``Round:``/``Verdict:``/``Risk:``/
   ``Findings:``/``Cost:``/``Elapsed:``), printed via
   ``argus.cli.render_review_output`` / ``render_summary_block``.
2. The ``-o`` markdown file and its sibling ``.json`` — the JSON is
   ``ReviewResponse.model_dump()``.

These tests build a realistic fixture ``ReviewResponse`` and assert both
byte-exact stdout formatting and the JSON schema shape. Do NOT "fix" a
failure here by updating the golden files unless the format change is
deliberate and the ``argus-review-loop`` skill is updated in lockstep.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from argus.cli import render_review_output, render_summary_block
from argus.models import (
    Finding,
    ReviewResponse,
    RiskLevel,
    Severity,
    SystemCoverage,
    TokenUsage,
    Verdict,
)

_GOLDEN_SCHEMA_PATH = Path(__file__).parent / "golden" / "review_response.schema.json"


def _fixture_response() -> ReviewResponse:
    """A realistic ReviewResponse: mixed findings, non-trivial usage/coverage."""
    return ReviewResponse(
        verdict=Verdict.BLOCKING,
        risk_level=RiskLevel.HIGH,
        findings=[
            Finding(
                severity=Severity.BLOCKING,
                category="security",
                file="argus/github_client.py",
                line=42,
                description="Token logged in plaintext.",
                suggestion="Redact the token before logging.",
            ),
            Finding(
                severity=Severity.SUGGESTION,
                category="code-correctness",
                file="argus/cli.py",
                line=10,
                description="Consider extracting this into a helper.",
                suggestion=None,
            ),
        ],
        coverage_map=[
            SystemCoverage(
                system="CLI",
                files_explored=["argus/cli.py"],
                checks_performed=["argparse wiring"],
            )
        ],
        review_comment="## Argus Review\n\n**Verdict:** BLOCKING\n\n- B1: token logged\n- S1: extract helper",
        review_round=1,
        usage=TokenUsage(
            input_tokens=1000,
            output_tokens=500,
            cache_read_tokens=200,
            cache_creation_tokens=50,
            cost_usd=1.2345,
        ),
        lite_mode=False,
    )


class TestRenderSummaryBlock:
    def test_matches_frozen_format_exactly(self) -> None:
        response = _fixture_response()
        block = render_summary_block(response, elapsed=125.4)

        expected = "\n".join(
            [
                "=" * 60,
                "Round:      1",
                "Verdict:    BLOCKING",
                "Risk:       HIGH",
                "Findings:   1 blocking, 1 suggestions",
                "Cost:       $1.23",
                "Elapsed:    125s",
                "=" * 60,
            ]
        )
        assert block == expected

    def test_lite_mode_round_label(self) -> None:
        response = _fixture_response()
        response.lite_mode = True
        response.review_round = 2
        block = render_summary_block(response, elapsed=10.0)
        assert "Round:      2 (Lite Mode)" in block


class TestRenderReviewOutput:
    def test_matches_pre_refactor_print_sequence_exactly(self) -> None:
        """Byte-exact match for the sequence of ``print()`` calls this replaced.

        This is deliberately written as the literal print-call sequence
        (not a call into render_summary_block) so a regression in either
        function shows up here independently.
        """
        response = _fixture_response()
        elapsed = 125.4

        blocking = sum(1 for f in response.findings if f.severity.value == "BLOCKING")
        suggestion = sum(1 for f in response.findings if f.severity.value == "SUGGESTION")
        round_label = (
            f"{response.review_round} (Lite Mode)"
            if response.lite_mode
            else str(response.review_round)
        )
        expected = (
            "\n\n"
            + response.review_comment
            + "\n"
            + "\n"
            + "=" * 60
            + "\n"
            + f"Round:      {round_label}\n"
            + f"Verdict:    {response.verdict.value}\n"
            + f"Risk:       {response.risk_level.value}\n"
            + f"Findings:   {blocking} blocking, {suggestion} suggestions\n"
            + f"Cost:       ${response.usage.cost_usd:.2f}\n"
            + f"Elapsed:    {elapsed:.0f}s\n"
            + "=" * 60
        )

        assert render_review_output(response, elapsed) == expected

    def test_stdout_capture_matches(self, capsys: "pytest.CaptureFixture[str]") -> None:
        """Drive the same print path ``_run_review`` uses and capture real stdout."""
        response = _fixture_response()
        print(render_review_output(response, elapsed=42.0))
        captured = capsys.readouterr()
        assert captured.out == render_review_output(response, elapsed=42.0) + "\n"
        assert "Round:      1" in captured.out
        assert "Verdict:    BLOCKING" in captured.out
        assert "Risk:       HIGH" in captured.out
        assert "Findings:   1 blocking, 1 suggestions" in captured.out
        assert "Cost:       $1.23" in captured.out
        assert "Elapsed:    42s" in captured.out


class TestOutputFileContract:
    def test_json_output_is_model_dump(self, tmp_path: Path) -> None:
        """``-o out.md`` writes a sibling ``out.json`` == ``response.model_dump()``."""
        response = _fixture_response()
        out_md = tmp_path / "review.md"
        out_md.write_text(response.review_comment, encoding="utf-8")
        out_json = out_md.with_suffix(".json")
        out_json.write_text(
            json.dumps(response.model_dump(), indent=2, default=str), encoding="utf-8"
        )

        loaded = json.loads(out_json.read_text(encoding="utf-8"))
        assert loaded == response.model_dump(mode="json")

    def test_json_schema_matches_golden_snapshot(self) -> None:
        """Freeze the ReviewResponse JSON schema (keys + types).

        Skill compatibility depends on the key set (findings[].severity,
        verdict, etc.) staying stable. If this fails, you have changed the
        public output contract of ReviewResponse.model_dump() — this breaks
        the argus-review-loop skill.
        """
        current = ReviewResponse.model_json_schema()
        golden = json.loads(_GOLDEN_SCHEMA_PATH.read_text(encoding="utf-8"))
        assert current == golden, (
            "ReviewResponse.model_json_schema() no longer matches "
            f"{_GOLDEN_SCHEMA_PATH} — this breaks the argus-review-loop skill. "
            "If this change is deliberate, update the golden file AND the skill "
            "in lockstep."
        )

    def test_golden_schema_has_documented_top_level_keys(self) -> None:
        """Sanity check the golden file itself still documents the keys the
        skill actually reads (finding IDs are synthesized by the skill from
        list order + severity, not a field on Finding)."""
        golden = json.loads(_GOLDEN_SCHEMA_PATH.read_text(encoding="utf-8"))
        top_level_props = set(golden["properties"].keys())
        assert {"verdict", "risk_level", "findings", "review_comment", "review_round"}.issubset(
            top_level_props
        )
        finding_props = set(golden["$defs"]["Finding"]["properties"].keys())
        assert {"severity", "category", "description"}.issubset(finding_props)
