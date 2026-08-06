"""Pure helper functions for the v3 review pipeline.

These have no LLM or SDK dependencies and can be tested in isolation.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from argus.models import Finding, ReviewResponse, RiskLevel, Severity, Verdict
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


def build_degraded_coverage_labels(
    findings_models: list[SystemReviewResult], graph_result: dict[str, Any]
) -> list[tuple[str, str]]:
    """Combine failed LLM-reviewer sessions with failed precheck scanners
    into one ``(label, reason)`` list for :func:`append_degraded_coverage_section`.

    Pulled out of ``graph.run_review`` specifically so the exact state-key
    lookup (``graph_result.get("precheck_scanner_failures", [])``) is
    covered by a direct unit test -- that key is written on the producer
    side by ``graph._node_precheck_rules`` as a literal string
    (``update["precheck_scanner_failures"]``) and read here as another
    literal string; a typo on either end would otherwise pass the full
    test suite undetected, since nothing previously exercised this read
    path end to end (``run_review`` itself has no dedicated test harness).
    A typo in the producer's own literal is still only caught by
    ``tests/test_graph_precheck.py``'s existing assertion on that exact
    key -- this function closes the read side, not the write side.
    """
    failed_labels = failed_reviewer_labels(findings_models)
    failed_labels += [
        (f"precheck:{name}", "scanner did not complete this round")
        for name in graph_result.get("precheck_scanner_failures", [])
    ]
    return failed_labels


# Explicit ordering, not reliance on declaration order or enum identity:
# RiskLevel is a plain str Enum with no intrinsic ordering of its own, so
# this is the one place that ordering is defined and relied upon (by
# apply_precheck_scanner_failure_gate, to raise risk_level monotonically
# rather than overwrite it unconditionally).
_RISK_LEVEL_ORDER: dict[RiskLevel, int] = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.CRITICAL: 3,
}


def apply_precheck_scanner_failure_gate(
    response: ReviewResponse,
    precheck_scanner_failures: list[str],
    block_on_failure: bool,
) -> bool:
    """Force ``response``'s verdict to BLOCKING if a precheck scanner
    failed this round and the opt-in ``ARGUS_PRECHECK_BLOCK_ON_SCANNER_FAILURE``
    setting is on (see that setting's own docstring in ``argus/config.py``
    for the fail-open-vs-fail-closed tradeoff this exists for). Mutates
    ``response`` in place; returns whether it did anything, purely so the
    caller can decide whether to log.

    No-op in every other case: an empty ``precheck_scanner_failures``,
    ``block_on_failure`` false (the default), or a verdict that's already
    BLOCKING (nothing to strengthen). This only ever makes the verdict
    stricter, never looser -- it will never turn a BLOCKING verdict into
    APPROVE, and never fires at all when there's no scanner failure to
    react to regardless of the flag.

    Pulled out of ``graph.run_review`` for the same reason
    :func:`build_degraded_coverage_labels` was -- no dedicated test
    harness exists for that function as a whole, and this logic is
    directly unit-testable in isolation once separated from it.

    Mutates ``response.review_comment`` too, not just the structured
    ``verdict``/``risk_level``/``findings`` fields: the comment is what
    ``cli.py`` actually posts to the PR and persists to the DB, rendered
    *before* this gate ever runs, so leaving it untouched would let the
    human-visible comment still read "APPROVE" while the structured
    verdict says BLOCKING. Follows the same regex-rewrite pattern
    ``graph._node_validate_blockings`` already uses for the same reason
    (an LLM-authored comment's exact formatting varies -- bold wrapping,
    emoji, pipe-delimited risk -- so this can't be a plain string
    replace).

    ``risk_level`` is raised, never overwritten: ``RiskLevel`` has a real
    ordering (LOW < MEDIUM < HIGH < CRITICAL) and this gate must only ever
    strengthen an assessment, matching its own "stricter, never looser"
    contract -- an unconditional overwrite to HIGH would silently
    downgrade an existing CRITICAL.
    """
    if (
        not precheck_scanner_failures
        or response.verdict == Verdict.BLOCKING
        or not block_on_failure
    ):
        return False
    response.verdict = Verdict.BLOCKING
    if _RISK_LEVEL_ORDER[response.risk_level] < _RISK_LEVEL_ORDER[RiskLevel.HIGH]:
        response.risk_level = RiskLevel.HIGH
    note = (
        "Precheck scanner(s) "
        f"{', '.join(sorted(precheck_scanner_failures))} did not complete this "
        "round (crashed, timed out, or hit an execution error). "
        "ARGUS_PRECHECK_BLOCK_ON_SCANNER_FAILURE is set, so this round cannot be "
        "APPROVE without confirmed coverage from every configured scanner."
    )
    response.findings.append(
        Finding(
            severity=Severity.BLOCKING,
            category="deterministic-precheck",
            file=None,
            line=None,
            description=note,
            suggestion=(
                "Re-run once the underlying scanner failure is resolved -- see this "
                "round's logs (or the degraded-coverage section above) for which "
                "scanner(s) failed and why."
            ),
        )
    )
    # Same regex-rewrite pattern as _node_validate_blockings, and the same
    # reason: an LLM-authored comment's exact formatting varies (bold
    # wrapping, emoji, pipe-delimited risk), so this can't be a plain
    # string replace. count=1 -- there's exactly one verdict header line
    # to rewrite, and a broader replace risks touching a coincidental
    # "**Verdict**:"-shaped line quoted elsewhere in the comment body.
    # subn (not sub) so a header that doesn't match the expected shape --
    # e.g. future drift in the pr-review-writer/pr-review-lite prompts --
    # is a loud warning, not a silent no-op leaving the header still
    # reading the old verdict while the structured response and the
    # appended note below both say BLOCKING.
    response.review_comment, match_count = re.subn(
        r"\*\*Verdict\*\*:.*?(?=\n|$)",
        f"**Verdict**: 🚫 BLOCKING | **Risk**: {response.risk_level.value}",
        response.review_comment,
        count=1,
    )
    if match_count == 0:
        logger.warning(
            "apply_precheck_scanner_failure_gate: no '**Verdict**:'-shaped line found in "
            "review_comment to rewrite -- the rendered comment's header may still read the "
            "old verdict despite response.verdict now being BLOCKING"
        )
    response.review_comment += f"\n\n---\n\n### 🚫 Verdict forced to BLOCKING\n\n{note}\n"
    return True


def append_degraded_coverage_section(
    review_comment: str, failed_labels: list[tuple[str, str]]
) -> str:
    """Append a visible "Degraded coverage" section listing failed reviewers
    and/or deterministic precheck scanners.

    No-op when ``failed_labels`` is empty. The markdown review body is not
    schema-frozen, so this is safe to append; it is purely additive to the
    rendered comment and does not change any structured field. Deliberately
    worded to cover both LLM reviewer sessions (killed/timed out) and
    precheck scanners (crashed/timed out/produced unparseable output) --
    each entry's own ``reason`` string carries the specific detail, so the
    shared intro paragraph only needs to say what both classes have in
    common: something didn't complete, and the resulting silence isn't
    evidence the area was actually clean.
    """
    if not failed_labels:
        return review_comment
    bullets = "\n".join(f"- {label} ({reason})" for label, reason in failed_labels)
    section = (
        "\n\n---\n\n"
        "### ⚠ Degraded coverage\n\n"
        "The following did not complete this round and produced no findings "
        "as a result, not because the area was clean. Treat these areas as "
        "**not reviewed** this round:\n\n"
        f"{bullets}\n"
    )
    return review_comment + section
