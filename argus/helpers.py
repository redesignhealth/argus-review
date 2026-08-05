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


def extract_changed_files(diff: str) -> list[str]:
    """Return the sorted, deduped set of repo-relative paths touched by a
    unified diff (both sides of a rename).

    Used to scope ``argus.precheck.engine.run_precheck`` to files this PR
    actually touched — a static-analysis scanner run against the whole
    worktree (as several precheck scanners now are, beyond the original
    small-footprint custom rules) would otherwise surface every pre-existing
    finding in the repo on every PR, not just what this PR introduced or
    changed, and ``run_precheck``'s own ``_MAX_RESULTS`` cap would then
    truncate that flood arbitrarily -- silently dropping real, in-scope
    findings alongside the noise.

    Same ``diff --git a/(\\S+) b/(\\S+)`` extraction already used ad hoc in
    ``graph._is_image_tag_bump_only``/``_is_high_blast_radius`` — centralized
    here rather than re-deriving the regex a third time.
    """
    if not diff:
        return []
    path_pairs: list[tuple[str, str]] = re.findall(
        r"^diff --git a/(\S+) b/(\S+)", diff, re.MULTILINE
    )
    files: set[str] = set()
    for a_path, b_path in path_pairs:
        files.add(a_path)
        files.add(b_path)
    return sorted(files)


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


def failed_reviewer_labels(results: list[SystemReviewResult]) -> list[tuple[str, str]]:
    """Return ``(system_group, failure_reason)`` pairs for reviewers that did
    not complete normally, in the order they appear in ``results``.

    Used to distinguish "a reviewer was killed/crashed" from "a reviewer ran
    to completion and genuinely found nothing" -- both currently collapse into
    a 0-finding SystemReviewResult, but only the former should be surfaced as
    degraded coverage. Covers both failure modes (timeout and worker crash),
    not just timeout -- a crashed worker's 0 findings is exactly as untrustworthy
    as a timed-out one's, and the previous timeout-only check silently treated
    a crash as a clean result.
    """
    return [
        (result.system_group, result.failure_reason)
        for result in results
        if result.failure_reason is not None
    ]


def append_degraded_coverage_section(
    review_comment: str, failed_labels: list[tuple[str, str]]
) -> str:
    """Append a visible "Degraded coverage" section listing failed reviewers.

    No-op when ``failed_labels`` is empty. The markdown review body is not
    schema-frozen, so this is safe to append; it is purely additive to the
    rendered comment and does not change any structured field.
    """
    if not failed_labels:
        return review_comment
    bullets = "\n".join(f"- {label} ({reason})" for label, reason in failed_labels)
    section = (
        "\n\n---\n\n"
        "### ⚠ Degraded coverage\n\n"
        "The following reviewer session(s) did not complete (timeout or worker "
        "crash) and reported 0 findings as a result, not because "
        "the area was clean. Treat these areas as **not reviewed** this round:\n\n"
        f"{bullets}\n"
    )
    return review_comment + section
