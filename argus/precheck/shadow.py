"""Shadow-review harness: validate a candidate rule against a historical
PR corpus before it's ever allowed to fire live on a real review.

This is the OSS-side half of Phase 2's shadow-review step (see
docs/PRECHECKS.md's rule-status lifecycle): it generates evidence --
occurrence counts across a corpus of historical PRs -- and nothing more.
It never judges true/false positive itself. That judgment (a human
approving a candidate rule draft, and later the RH-internal async LLM
triage loop) happens on top of this harness's output, the same OSS/
RH-internal split as everything else in ``argus.precheck``: this module
ships the mechanism to *generate* the evidence, RH-internal infrastructure
owns the *decision* made from it.

Modeled on Google Tricorder's and Amazon CodeGuru's published methodology
of validating a new analyzer/rule against a historical corpus before it's
allowed to fire on live code, rather than trusting a hand-authored pattern
un-tested.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from argus.precheck.engine import run_semgrep_sarif, semgrep_available
from argus.precheck.sarif import SarifResult
from argus.repo_provision import provisioned_worktree

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorpusEntry:
    """One historical PR to validate a candidate rule against."""

    repo: str
    head_sha: str


@dataclass(frozen=True)
class ShadowReviewHit:
    """One rule match, attributed back to the corpus entry it came from."""

    corpus_entry: CorpusEntry
    result: SarifResult


@dataclass
class ShadowReviewResult:
    """Raw evidence for a candidate rule's behavior -- not a verdict.

    ``entries_scanned`` and ``entries_matched`` (vs. ``len(hits)``, which
    can exceed either since one entry can produce multiple hits) let a
    human or the RH-internal triage job compute a per-PR occurrence rate
    before deciding whether the rule is even worth the ongoing async
    triage loop. ``entries_failed`` is not a rule-precision signal --
    it's corpus/infra flakiness (bad SHA, transient clone error) and
    should be re-attempted or excluded from precision math, not counted
    as "rule found nothing here."
    """

    hits: list[ShadowReviewHit] = field(default_factory=list)
    entries_scanned: int = 0
    entries_matched: int = 0
    entries_failed: list[CorpusEntry] = field(default_factory=list)


async def run_shadow_review(
    *, rule_path: Path, corpus: list[CorpusEntry], github_token: str
) -> ShadowReviewResult:
    """Run one candidate rule file against a historical PR corpus.

    ``rule_path`` points directly at a single candidate rule (a draft not
    yet in any rules directory `argus.precheck.engine.resolve_rules_dir`
    would find, and deliberately not run through DB-status classification
    the way a live gate run is -- shadow review cares about raw occurrence
    counts, not the candidate/verified split that classification exists
    for).

    Each corpus entry is checked out into its own worktree (reusing the
    exact same provisioning path a live review uses, via
    ``repo_provision.provisioned_worktree``) and scanned in isolation. A
    failure on one entry (bad SHA, transient clone error, or semgrep
    itself failing to run against that entry) is recorded in
    ``entries_failed`` and skipped rather than aborting the whole run --
    a corpus of dozens-to-hundreds of historical PRs should tolerate a
    handful of unreachable entries. Failed entries are never counted
    toward ``entries_scanned``/``entries_matched``: see
    ``run_semgrep_sarif``'s docstring for why collapsing "semgrep didn't
    run" into "ran, found nothing" would corrupt the harness's whole
    purpose (a malformed candidate rule that fails to execute on every
    corpus entry must not look like strong zero-occurrence evidence).

    Raises:
        RuntimeError: If semgrep isn't installed at all -- a single clear
            failure up front, rather than every corpus entry independently
            landing in ``entries_failed`` after paying for a full clone
            each, indistinguishable from unrelated per-entry flakiness.
        FileNotFoundError: If ``rule_path`` doesn't exist -- checked before
            any corpus entry is cloned, for the same reason.
    """
    if not semgrep_available():
        raise RuntimeError(
            "semgrep not on PATH (argus[prechecks] extra not installed) — "
            "cannot run a shadow review"
        )
    if not rule_path.exists():
        raise FileNotFoundError(f"Shadow review rule path does not exist: {rule_path}")

    result = ShadowReviewResult()
    for entry in corpus:
        try:
            async with provisioned_worktree(
                repo=entry.repo, head_sha=entry.head_sha, token=github_token
            ) as worktree_path:
                sarif_results = await run_semgrep_sarif(worktree_path, rule_path)
        except Exception:  # noqa: BLE001 — one bad corpus entry must not abort the run
            logger.warning(
                "Shadow review: failed on %s@%s", entry.repo, entry.head_sha[:12], exc_info=True
            )
            result.entries_failed.append(entry)
            continue

        if sarif_results is None:
            # semgrep didn't run to completion on this entry (timeout, its
            # own execution error) -- infra flakiness, not "zero hits."
            logger.warning(
                "Shadow review: semgrep did not complete on %s@%s",
                entry.repo,
                entry.head_sha[:12],
            )
            result.entries_failed.append(entry)
            continue

        result.entries_scanned += 1
        if sarif_results:
            result.entries_matched += 1
            result.hits.extend(ShadowReviewHit(corpus_entry=entry, result=r) for r in sarif_results)

    return result
