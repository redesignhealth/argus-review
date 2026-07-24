"""Pure helper functions for the v3 review pipeline.

These have no LLM or SDK dependencies and can be tested in isolation.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from argus.pipeline_models import RawFinding, SystemReviewResult

logger = logging.getLogger(__name__)


def sanitize_file_paths(files: list[str], repo_root: str) -> list[str]:
    """Sanitize file paths to prevent path traversal attacks."""
    root = Path(repo_root).resolve()
    safe: list[str] = []
    for raw_path in files:
        cleaned = raw_path.lstrip("/")
        resolved = (root / cleaned).resolve()
        if not str(resolved).startswith(str(root) + "/") and resolved != root:
            logger.warning("Dropping path that escapes repo root: %r -> %s", raw_path, resolved)
            continue
        safe.append(str(resolved.relative_to(root)))
    return safe


def filter_diff_for_files(full_diff: str, files: list[str]) -> str:
    """Extract only the diff hunks for the specified files."""
    if not files or not full_diff:
        return ""

    file_set = set(files)
    sections = re.split(r"(?=^diff --git )", full_diff, flags=re.MULTILINE)

    kept: list[str] = []
    for section in sections:
        if not section.strip():
            continue
        match = re.match(r"diff --git a/(.+?) b/(.+?)(?:\n|$)", section)
        if match:
            a_path = match.group(1)
            b_path = match.group(2)
            if a_path in file_set or b_path in file_set:
                kept.append(section)

    return "".join(kept)


def parse_review_result(raw_text: str, group_name: str) -> SystemReviewResult:
    """Parse the agent's text output into a SystemReviewResult."""
    if not raw_text.strip():
        return SystemReviewResult(system_group=group_name, findings=[], files_explored=[])

    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_str = raw_text.strip()

    try:
        data = json.loads(json_str)
        findings: list[RawFinding] = []
        for f in data.get("findings", []):
            raw_line = f.get("line")
            if isinstance(raw_line, str):
                try:
                    raw_line = int(raw_line)
                except ValueError:
                    pass
            findings.append(
                RawFinding(
                    file=f.get("file"),
                    line=raw_line,
                    description=f.get("description", ""),
                    context=f.get("context"),
                )
            )
        files_explored = data.get("files_explored", [])
        return SystemReviewResult(
            system_group=data.get("system_group", group_name),
            findings=findings,
            files_explored=files_explored,
            cost_usd=0.0,
        )
    except (json.JSONDecodeError, TypeError, KeyError):
        logger.warning(
            "Could not parse JSON from %s reviewer output, using raw text as finding",
            group_name,
        )
        return SystemReviewResult(
            system_group=group_name,
            findings=[RawFinding(file=None, line=None, description=raw_text.strip(), context=None)],
            files_explored=[],
        )


def collect_reviewed_files(results: list[SystemReviewResult]) -> set[str]:
    """Build the set of all files mentioned across reviewer results."""
    reviewed: set[str] = set()
    for result in results:
        reviewed.update(result.files_explored)
        for finding in result.findings:
            if finding.file:
                reviewed.add(finding.file)
    return reviewed


def timed_out_reviewer_labels(results: list[SystemReviewResult]) -> list[str]:
    """Return the ``system_group`` labels of reviewers that hit the subprocess
    timeout, in the order they appear in ``results``.

    Used to distinguish "a reviewer was killed by the timeout" from "a
    reviewer ran to completion and genuinely found nothing" -- both currently
    collapse into a 0-finding SystemReviewResult, but only the former should
    be surfaced as degraded coverage.
    """
    return [result.system_group for result in results if result.timed_out]


def append_degraded_coverage_section(review_comment: str, timed_out_labels: list[str]) -> str:
    """Append a visible "Degraded coverage" section listing timed-out reviewers.

    No-op when ``timed_out_labels`` is empty. The markdown review body is not
    schema-frozen, so this is safe to append; it is purely additive to the
    rendered comment and does not change any structured field.
    """
    if not timed_out_labels:
        return review_comment
    bullets = "\n".join(f"- {label}" for label in timed_out_labels)
    section = (
        "\n\n---\n\n"
        "### ⚠ Degraded coverage\n\n"
        "The following reviewer session(s) were killed after exceeding the "
        f"subprocess timeout and reported 0 findings as a result, not because "
        "the area was clean. Treat these areas as **not reviewed** this round:\n\n"
        f"{bullets}\n"
    )
    return review_comment + section
