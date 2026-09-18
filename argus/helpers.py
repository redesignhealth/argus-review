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

# Shared between the producer (build_degraded_coverage_labels) and consumer
# (build_degraded_coverage_findings) so the two can't drift apart.
_SCANNER_NOT_INSTALLED_REASON = "scanner not installed"


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
    labels: list[tuple[str, str]] = []
    for result in results:
        reason = result.failure_reason
        if reason is None and result.timed_out:
            reason = "timeout"
        if reason is not None:
            labels.append((result.system_group, reason))
    return labels


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

    ``precheck_missing_scanners`` (never-installed scanners) gets its own
    reason string, distinct from a crashed scanner's.
    """
    failed_labels = failed_reviewer_labels(findings_models)
    failed_labels += [
        (f"precheck:{name}", "scanner did not complete this round")
        for name in graph_result.get("precheck_scanner_failures", [])
    ]
    failed_labels += [
        (f"precheck:{name}", _SCANNER_NOT_INSTALLED_REASON)
        for name in graph_result.get("precheck_missing_scanners", [])
    ]
    return failed_labels


def reviewer_only_labels(failed_labels: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Filter a :func:`build_degraded_coverage_labels` list down to
    reviewer-session entries, dropping ``precheck:``-prefixed scanner
    failures.

    Pulled out of ``graph.run_review`` for the same reason
    :func:`build_degraded_coverage_labels` was -- so the precheck/reviewer
    split has a unit test that doesn't require mocking the full
    ``run_review`` pipeline.

    **Call this ONLY when the caller has confirmed
    ``apply_precheck_scanner_failure_gate`` already added its own
    BLOCKING/deterministic-precheck finding for the same precheck
    failures this round** (i.e. its return value was ``True`` --
    :func:`apply_precheck_gate_and_surface_degraded_coverage` calls the
    gate before this function specifically so that fact is known; that
    combined function, not ``graph.run_review`` directly, is what actually
    threads the gate's return value through today). That gate is a no-op
    whenever
    ``ARGUS_PRECHECK_BLOCK_ON_SCANNER_FAILURE`` is unset (the default) or
    the verdict is already BLOCKING for some other reason -- in either
    case it adds no finding at all, and unconditionally dropping precheck
    entries here would leave a crashed precheck scanner with ZERO
    structured findings, not merely a de-duplicated one. When the gate DID
    fire, passing an unfiltered ``failed_labels`` list straight to
    :func:`build_degraded_coverage_findings` would double-report the same
    scanner failure as a SUGGESTION/coverage-gap finding too -- that's the
    only case this filter exists to prevent.

    Relies on the ``precheck:`` prefix convention ``build_degraded_coverage_labels``
    itself establishes for scanner-failure labels -- not a dedicated
    tag/type field -- so a reviewer-session ``SystemGroup.name`` that
    happens to literally start with ``"precheck:"`` would be misclassified
    here. ``SystemGroup.name`` is planner/LLM-generated free text with no
    format constraint against this; accepted as a narrow, low-probability
    edge case rather than a dedicated-field redesign for this fix.
    """
    return [(label, reason) for label, reason in failed_labels if not label.startswith("precheck:")]


def build_degraded_coverage_findings(
    failed_labels: list[tuple[str, str]],
) -> list[Finding]:
    """Build SUGGESTION-level findings for reviewer sessions that failed to complete.

    Surfaces each timed-out or crashed reviewer session (and, per below,
    a crashed precheck scanner when appropriate) as an explicit
    coverage-gap finding so it appears in response.findings and the final
    review output, rather than only in telemetry.

    The production caller is :func:`coverage_gap_findings_for_round`,
    which passes the FULL combined ``build_degraded_coverage_labels``
    output -- including ``precheck:<name>`` entries -- on its own default
    (non-``gate_added_precheck_finding``) path. Passing precheck entries
    through to this function is intended behavior in that case, not an
    edge case: a failed precheck scanner is only ever additionally
    surfaced as its own BLOCKING/deterministic-precheck finding by
    ``apply_precheck_scanner_failure_gate`` when
    ``ARGUS_PRECHECK_BLOCK_ON_SCANNER_FAILURE`` is set AND that gate
    actually fired this round; in every other case (the default config),
    this function is the ONLY place a crashed precheck scanner is
    surfaced as a structured finding at all, so dropping precheck entries
    unconditionally here (a real round-3 BLOCKING bug on this file) would
    leave it with zero structured findings. Do not "fix" this function to
    unconditionally filter ``precheck:`` labels again -- that regresses
    the exact bug ``coverage_gap_findings_for_round`` exists to prevent.
    The ``is_precheck``-branch wording below (scanner-appropriate, not
    "Reviewer session") reflects that this is a first-class, expected
    input shape, not defense in depth for a caller mistake.
    """
    findings: list[Finding] = []
    for label, reason in failed_labels:
        is_precheck = label.startswith("precheck:")
        subject = "Precheck scanner" if is_precheck else "Reviewer session"
        is_missing = reason == _SCANNER_NOT_INSTALLED_REASON
        # Distinct wording from a crash/timeout: the reviewer was alive and
        # exploring, it just never reached finish_review.
        is_exhausted = reason == "turn_budget_exhausted"
        if is_missing:
            reason_desc = "is not installed"
        elif reason == "timeout":
            reason_desc = "timed out"
        elif is_exhausted:
            reason_desc = "ran out of turns before finishing"
        else:
            reason_desc = f"did not complete ({reason})"
        # Missing scanner: remediation is "install it", not "re-run".
        if is_missing:
            suggestion = (
                f"Install {label.removeprefix('precheck:')} to restore this scanner's coverage."
            )
        elif is_exhausted:
            suggestion = (
                f"Re-run the review or manually inspect '{label}' -- it explored this "
                "area but ran out of turns before reporting, so its files may be "
                "unreviewed."
            )
        else:
            suggestion = (
                f"Re-run the review or manually inspect the changes in '{label}' "
                f"to ensure potential issues were not missed due to {reason}."
            )
        findings.append(
            Finding(
                severity=Severity.SUGGESTION,
                category="coverage-gap",
                file=None,
                line=None,
                description=(
                    f"{subject} '{label}' {reason_desc} and produced no findings. "
                    "Coverage for this area is degraded."
                ),
                suggestion=suggestion,
            )
        )
    return findings


def coverage_gap_findings_for_round(
    failed_labels: list[tuple[str, str]], gate_added_precheck_finding: bool
) -> list[Finding]:
    """Build the structured coverage-gap findings for one review round,
    encoding the fix for a round-3 Argus BLOCKING finding on this PR:
    ``reviewer_only_labels`` must be applied ONLY when
    ``apply_precheck_scanner_failure_gate`` already added its own
    BLOCKING/deterministic-precheck finding for the same precheck
    failures this round (``gate_added_precheck_finding``).

    Unconditionally filtering ``precheck:``-prefixed entries (an earlier
    round's bug) silently dropped a crashed precheck scanner to ZERO
    structured findings whenever ``ARGUS_PRECHECK_BLOCK_ON_SCANNER_FAILURE``
    is unset (the default) -- the gate is a no-op in that case and adds no
    finding of its own, so nothing else would have surfaced the failure.
    When the gate DID fire, passing the precheck entries through
    unfiltered would double-report the same failure as both a
    SUGGESTION/coverage-gap finding and the gate's own BLOCKING finding.

    Pulled out into its own directly-testable function, mirroring
    :func:`build_degraded_coverage_labels`/:func:`reviewer_only_labels`'s
    own testability rationale, so this exact conditional -- the actual
    site of the round-3 bug -- has a unit test that doesn't require
    mocking the full ``run_review`` pipeline.
    """
    labels = reviewer_only_labels(failed_labels) if gate_added_precheck_finding else failed_labels
    return build_degraded_coverage_findings(labels)


def apply_precheck_gate_and_surface_degraded_coverage(
    response: ReviewResponse,
    findings_models: list[SystemReviewResult],
    graph_result: dict[str, Any],
    block_on_failure: bool,
) -> tuple[bool, list[tuple[str, str]]]:
    """Apply the precheck scanner-failure gate, THEN surface this round's
    degraded coverage (the markdown section and the structured
    coverage-gap findings) -- combined into one function specifically so
    this ordering can never be silently violated by a future refactor
    that pulls the two apart again. Also applies :func:`apply_reviewer_failure_gate`
    once the coverage-gap findings exist, since it needs to promote one of them.

    This ordering IS the round-3 BLOCKING fix: :func:`coverage_gap_findings_for_round`
    needs to know whether :func:`apply_precheck_scanner_failure_gate`
    already added its own finding THIS round, which is only knowable once
    the gate has actually run against this round's real
    ``precheck_scanner_failures``/``block_on_failure`` -- a hand-supplied
    boolean in a unit test can verify each half in isolation (see
    :func:`coverage_gap_findings_for_round`'s own tests) but cannot catch
    a refactor that reorders the two real call sites. Wrapping both in one
    function removes the reordering opportunity entirely: there is only
    one place left to call, and its own body fixes the order.

    Mutates ``response`` in place (verdict/risk_level/review_comment/
    findings), via the same effects
    ``apply_precheck_scanner_failure_gate``/``append_degraded_coverage_section``/
    ``coverage_gap_findings_for_round`` already document individually.
    Returns ``(gate_added_precheck_finding, failed_labels)`` so the caller
    can log accordingly -- this function does no logging itself, matching
    ``apply_precheck_scanner_failure_gate``'s own return-value-for-logging
    convention (logging lives in ``graph.run_review``, which has the
    logger and the rest of this round's context).

    Feeds the gate both crashed and never-installed scanner names combined --
    both mean "no confirmed coverage" for gating purposes; they only need to
    stay separate for wording (see :func:`build_degraded_coverage_labels`).
    """
    precheck_scanner_failures = graph_result.get("precheck_scanner_failures", [])
    precheck_missing_scanners = graph_result.get("precheck_missing_scanners", [])
    gate_added_precheck_finding = apply_precheck_scanner_failure_gate(
        response, precheck_scanner_failures + precheck_missing_scanners, block_on_failure
    )
    failed_labels = build_degraded_coverage_labels(findings_models, graph_result)
    if failed_labels:
        response.review_comment = append_degraded_coverage_section(
            response.review_comment, failed_labels
        )
        new_findings = coverage_gap_findings_for_round(failed_labels, gate_added_precheck_finding)
        response.findings.extend(new_findings)

        # reviewer failures are always the leading entries of failed_labels
        # (build_degraded_coverage_labels appends precheck entries after
        # them), and coverage_gap_findings_for_round preserves that relative
        # order -- so this slice is exactly the findings for failed reviewer
        # sessions, never a precheck scanner's.
        reviewer_failure_count = len(failed_reviewer_labels(findings_models))
        if reviewer_failure_count:
            # Verdict may already be BLOCKING (e.g. the precheck scanner-failure
            # gate above already forced it) -- that part of the gate is then
            # redundant, but the severity promotion and risk_level bump below
            # are not, so this must still run regardless of the current verdict.
            apply_reviewer_failure_gate(response, new_findings[:reviewer_failure_count])
    return gate_added_precheck_finding, failed_labels


def apply_reviewer_failure_gate(
    response: ReviewResponse, reviewer_coverage_gap_findings: list[Finding]
) -> None:
    """Force ``response.verdict`` to BLOCKING because a reviewer session
    failed to complete -- a crashed/timed-out/exhausted reviewer contributes
    zero findings, indistinguishable from a genuinely clean review.

    Promotes the given coverage-gap SUGGESTION finding(s) (already in
    ``response.findings``) to BLOCKING in place, and raises ``risk_level``
    to at least HIGH. Unconditional, unlike
    ``apply_precheck_scanner_failure_gate`` -- no opt-out flag.
    """
    for finding in reviewer_coverage_gap_findings:
        finding.severity = Severity.BLOCKING
    response.verdict = Verdict.BLOCKING
    if _RISK_LEVEL_ORDER[response.risk_level] < _RISK_LEVEL_ORDER[RiskLevel.HIGH]:
        response.risk_level = RiskLevel.HIGH
    response.review_comment, match_count = re.subn(
        r"\*\*Verdict\*\*:.*?(?=\n|$)",
        f"**Verdict**: 🚫 BLOCKING | **Risk**: {response.risk_level.value}",
        response.review_comment,
        count=1,
    )
    if match_count == 0:
        logger.warning(
            "apply_reviewer_failure_gate: no '**Verdict**:'-shaped line found in "
            "review_comment to rewrite -- the rendered comment's header may still read "
            "the old verdict despite response.verdict now being BLOCKING"
        )


def compute_persisted_finding_counts(findings: list[Finding]) -> tuple[int, int]:
    """Count BLOCKING/SUGGESTION findings for the persisted
    ``blocking_count``/``suggestion_count`` columns, excluding
    ``category == "coverage-gap"`` entries.

    Pulled out of ``graph.run_review`` for the same testability reason as
    :func:`build_degraded_coverage_labels`/:func:`reviewer_only_labels`.
    A coverage-gap finding (see :func:`build_degraded_coverage_findings`)
    is an infra-failure observability marker synthesized from a reviewer
    session that timed out or crashed -- not a real review finding, and
    not something a code change to the reviewed PR can ever resolve.
    Letting it inflate these two persisted counts would corrupt historical
    trend analysis with transient failures that have nothing to do with
    the PR's actual quality. It remains fully visible in
    ``response.findings``/``review_comment``; only these two aggregate
    counts exclude it.
    """
    blocking_count = sum(
        1 for f in findings if f.severity.value == "BLOCKING" and f.category != "coverage-gap"
    )
    suggestion_count = sum(
        1 for f in findings if f.severity.value == "SUGGESTION" and f.category != "coverage-gap"
    )
    return blocking_count, suggestion_count


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
    failed OR was never installed this round and the opt-in
    ``ARGUS_PRECHECK_BLOCK_ON_SCANNER_FAILURE`` setting is on (see that
    setting's own docstring in ``argus/config.py`` for the fail-open-vs-
    fail-closed tradeoff this exists for). ``precheck_scanner_failures``
    takes crashed and missing scanner names combined -- both count as "no
    confirmed coverage" for gating. Mutates ``response`` in place; returns
    whether it did anything, purely so the caller can decide whether to log.

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
        "round (crashed, timed out, hit an execution error, or is not installed). "
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
                "round's logs (or the degraded-coverage section below) for which "
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


def apply_bench_config_change_gate(
    response: ReviewResponse,
    bench_changes: list[str],
    unconfirmed_reason: str | None = None,
) -> bool:
    """Force ``response``'s verdict to BLOCKING if a bench configuration
    file or routing environment variable was modified, or if bench configuration
    detection completeness could not be verified (TECH-6282).

    No-op if ``bench_changes`` is empty and ``unconfirmed_reason`` is None.
    When either is present:
    - Unconditionally forces ``response.verdict = Verdict.BLOCKING``
    - Raises ``risk_level`` monotonically to at least ``HIGH`` (never downgrading an existing ``CRITICAL``)
    - Appends a structured ``Finding(severity=BLOCKING, category="argus-self-config", ...)``:
        - If ``bench_changes`` is present: names the matched files/lines, explaining that
          bench configuration changes require explicit human sign-off.
        - If ``unconfirmed_reason`` is present (and no direct changes detected): states that
          verification could not be completed and Argus fails closed.
    - Rewrites the ``**Verdict**:`` header line only when the verdict actually changed.
    - Appends a visible ``### 🚫 Verdict forced to BLOCKING`` markdown section to ``response.review_comment``.

    Mutates ``response`` in place; returns whether it did anything.
    """
    if not bench_changes and not unconfirmed_reason:
        return False

    verdict_changed = response.verdict != Verdict.BLOCKING
    response.verdict = Verdict.BLOCKING
    if _RISK_LEVEL_ORDER[response.risk_level] < _RISK_LEVEL_ORDER[RiskLevel.HIGH]:
        response.risk_level = RiskLevel.HIGH

    if bench_changes:
        paths_str = ", ".join(sorted(bench_changes))
        description = (
            f"PR touches Argus bench configuration or routing ({paths_str}). "
            "This file selects the LLM platform and model for every future reviewer "
            "run in this repo. Modifying review configuration requires explicit human sign-off."
        )
        suggestion = (
            "If this file was not intentionally part of this PR, remove it and re-run. "
            "If intentional, this cannot be resolved by editing code — it requires "
            "explicit human sign-off; escalate rather than dismiss."
        )

        file_target: str | None = None
        if len(bench_changes) == 1 and not bench_changes[0].startswith("added line"):
            file_target = bench_changes[0]

        response.findings.append(
            Finding(
                severity=Severity.BLOCKING,
                category="argus-self-config",
                file=file_target,
                line=None,
                description=description,
                suggestion=suggestion,
            )
        )

        note = (
            f"PR modifies Argus reviewer bench configuration or routing ({paths_str}). "
            "This file selects the LLM platform and model for every future reviewer run in this repo. "
            "Modifying review configuration requires explicit human sign-off.\n\n"
            f"**Suggestion**: {suggestion}"
        )
    else:
        # Completeness unconfirmed (API failure or file cap exceeded) — fail closed
        description = (
            f"Unable to verify whether this PR modifies Argus reviewer bench configuration ({unconfirmed_reason}). "
            "Because bench configuration controls LLM platform and model selection for every future reviewer "
            "run in this repo, Argus fails closed and requires confirmation before this PR can be approved."
        )
        suggestion = (
            "Re-run the review once GitHub API connectivity is restored or file limits are resolved, "
            "or obtain explicit human review and sign-off."
        )
        response.findings.append(
            Finding(
                severity=Severity.BLOCKING,
                category="argus-self-config",
                file=None,
                line=None,
                description=description,
                suggestion=suggestion,
            )
        )
        note = (
            f"Verification of Argus bench configuration could not be confirmed ({unconfirmed_reason}).\n\n"
            "Argus fails closed to ensure unverified bench configuration changes are not silently approved.\n\n"
            f"**Suggestion**: {suggestion}"
        )

    if verdict_changed:
        response.review_comment, match_count = re.subn(
            r"\*\*Verdict\*\*:.*?(?=\n|$)",
            f"**Verdict**: 🚫 BLOCKING | **Risk**: {response.risk_level.value}",
            response.review_comment,
            count=1,
        )
        if match_count == 0:
            logger.warning(
                "apply_bench_config_change_gate: no '**Verdict**:'-shaped line found in "
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
        f"{bullets}\n\n"
        "**Suggestion**: Re-run the review or manually inspect the unreviewed areas "
        "above to cover potential gaps.\n"
    )
    return review_comment + section
