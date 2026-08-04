"""Minimal parser for semgrep's native ``--sarif`` (SARIF 2.1.0) output.

Only extracts what the precheck engine needs (rule id, severity, file,
line, message) — this is not a general-purpose SARIF library. Semgrep
emits real SARIF 2.1.0 directly, so there's no custom normalization step
on our side: this matches the industry-standard interchange format
reviewdog/MegaLinter/DeepSource also converge on, buying free interop
(e.g. GitHub's Code Scanning tab) if we ever want it later.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SarifResult:
    """One SARIF result, flattened to the fields the precheck engine uses."""

    rule_id: str
    level: str  # SARIF: "error" | "warning" | "note"
    message: str
    file: str | None
    line: int | None

    def as_finding_dict(self) -> dict[str, Any]:
        """Shape matching ``argus.pipeline_models.RawFinding`` for writer context."""
        return {
            "file": self.file,
            "line": self.line,
            "description": self.message,
            "context": f"deterministic precheck rule: {self.rule_id}",
        }

    def as_storage_dict(self) -> dict[str, Any]:
        """Full result, stored verbatim in ``precheck_candidate_firings.finding``."""
        return {
            "rule_id": self.rule_id,
            "level": self.level,
            "message": self.message,
            "file": self.file,
            "line": self.line,
        }


def parse_semgrep_sarif(raw: bytes | str) -> list[SarifResult]:
    """Parse a SARIF 2.1.0 document into a flat list of results.

    Returns an empty list (rather than raising) on any malformed input —
    callers treat a parse failure the same as "no findings", matching the
    precheck engine's fail-open design: a precheck bug must never block a
    PR review.
    """
    if not raw:
        return []
    try:
        doc = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Could not parse semgrep SARIF output", exc_info=True)
        return []

    results: list[SarifResult] = []
    for run in doc.get("runs", []):
        for item in run.get("results", []):
            rule_id = item.get("ruleId")
            if not rule_id:
                continue
            message = item.get("message", {}).get("text", "")
            level = item.get("level", "warning")
            file: str | None = None
            line: int | None = None
            locations = item.get("locations") or []
            if locations:
                physical = locations[0].get("physicalLocation", {})
                file = physical.get("artifactLocation", {}).get("uri")
                region = physical.get("region", {})
                line = region.get("startLine")
            results.append(
                SarifResult(rule_id=rule_id, level=level, message=message, file=file, line=line)
            )
    return results
