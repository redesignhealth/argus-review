"""Deterministic-precheck rule execution: run semgrep against the worktree.

Fails open at every step (no semgrep installed, empty rules directory,
execution error, timeout, DB unreachable): returns an empty
:class:`PrecheckResult` rather than raising, so a precheck problem is a
silent no-op that never blocks or slows down a review. This mirrors the
existing philosophy at ``graph._node_preflight``'s own except-and-fall-
through for a failed preflight call.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
from dataclasses import dataclass, field, replace
from importlib import resources
from pathlib import Path

from argus.config import get_settings
from argus.precheck.sarif import SarifResult, parse_semgrep_sarif
from argus.storage.precheck import select_rule_statuses

logger = logging.getLogger(__name__)

_SEMGREP_TIMEOUT_S = 120

# Caps the aggregate size of candidate_findings only -- what reaches the
# LLM writer context. Deliberately NOT applied to verified_findings (see
# run_precheck's classify-before-cap comment below): a verified hit gates
# the PR regardless of count, so it must never be truncated away here.
# graph._node_precheck_fail applies its own separate, display-only cap to
# verified findings, for the different reason of bounding PR-comment size
# without ever affecting the gate decision itself. Per-message truncation
# (sarif._MAX_MESSAGE_LENGTH) bounds one finding's size; this bounds count.
_MAX_RESULTS = 50


@dataclass
class PrecheckResult:
    """Rule hits, split by the status ``argus.storage.precheck`` returned.

    ``candidate_findings`` are attached to pipeline state as non-blocking
    writer context; ``verified_findings`` short-circuit the PR before any
    LLM step runs. A rule with status 'suspended' is dropped entirely —
    neither list — since it's a known-bad rule that shouldn't influence a
    review at all until re-verified.
    """

    candidate_findings: list[SarifResult] = field(default_factory=list)
    verified_findings: list[SarifResult] = field(default_factory=list)


def resolve_rules_dir() -> Path | None:
    """Highest-priority-wins: ``ARGUS_RULES_DIR`` override, else the packaged
    (empty by default) rules directory. ``None`` if neither resolves to an
    existing directory.
    """
    override = get_settings().ARGUS_RULES_DIR
    if override:
        path = Path(override)
        if path.is_dir():
            return path
        logger.warning(
            "ARGUS_RULES_DIR=%r is not a directory — precheck rules disabled "
            "for this run (no fallback to packaged rules once an explicit "
            "override is set)",
            override,
        )
        return None
    try:
        # str(...) + is_dir() rather than importlib.resources.as_file(...):
        # as_file() is a context manager whose extracted path is only valid
        # inside the `with` block, but callers need this path well after
        # this function returns (semgrep runs later, against the path
        # returned here) — holding that context manager open across the
        # whole run_precheck call would be a much larger restructure for a
        # case (a zip-installed wheel) that pip/uv don't produce for pure-
        # Python packages by default. Logging on failure, below, at least
        # makes that theoretical gap observable instead of silent.
        packaged = resources.files("argus.precheck") / "rules"
        path = Path(str(packaged))
        if path.is_dir():
            return path
        logger.warning(
            "Packaged rules directory not found at %r (zip-installed package? "
            "see resolve_rules_dir's docstring) — precheck rules disabled",
            path,
        )
        return None
    except (ModuleNotFoundError, FileNotFoundError):
        logger.warning("Could not resolve argus.precheck package resources", exc_info=True)
        return None


def _has_rule_files(rules_dir: Path) -> bool:
    # Recursive (rglob), matching semgrep's own `--config <dir>` behavior,
    # which scans subdirectories -- a non-recursive check here would treat
    # a rules dir organized into subfolders as empty and skip a real run.
    return any(rules_dir.rglob("*.yml")) or any(rules_dir.rglob("*.yaml"))


def semgrep_available() -> bool:
    """True if the ``prechecks`` extra's semgrep binary is on PATH."""
    return shutil.which("semgrep") is not None


async def run_semgrep_sarif(worktree_path: str, config_path: Path) -> list[SarifResult] | None:
    """Run semgrep with ``config_path`` (a rules directory or a single rule
    file -- semgrep accepts either) against ``worktree_path``; return
    stripped, unclassified SARIF results.

    Shared by :func:`run_precheck` (live per-PR gate, classifies against
    ``argus.storage.precheck``'s DB-backed rule statuses) and
    :mod:`argus.precheck.shadow` (offline corpus validation of a candidate
    rule that has no DB row yet, and doesn't want one consulted -- shadow
    review cares about raw occurrence counts, not the live candidate/
    verified split).

    Returns ``None`` -- distinct from ``[]`` -- when semgrep did not
    actually run to completion (missing binary, timeout, semgrep's own
    execution error): ``[]`` means "ran successfully, zero hits," which is
    real evidence; ``None`` means "no evidence was produced at all." The
    two callers use that distinction differently: ``run_precheck`` treats
    both the same way (empty `PrecheckResult`, since either way there's
    nothing to gate or attach), but ``run_shadow_review`` must not count a
    `None` as "the rule matched nothing" -- an unvetted candidate rule
    that fails to execute on every corpus entry would otherwise produce a
    confident-looking zero-occurrence result that looks like strong
    evidence the rule is safe, when it never actually ran.
    """
    if not semgrep_available():
        logger.info("semgrep not on PATH (argus[prechecks] extra not installed) — skipping scan")
        return None

    # Verified empirically: semgrep namespaces each rule's reported ruleId
    # with the config path we hand it -- e.g. a config of "/tmp/x/rules"
    # turns rule id "foo" into ruleId "tmp.x.rules.foo" -- UNLESS that path
    # is just a bare name/"." relative to semgrep's own cwd, in which case
    # the id passes through unmodified. Since `select_rule_statuses` looks
    # up the raw rule_id a rule author wrote in `id:`, running with cwd set
    # to config_path's own directory (and passing a same-directory-relative
    # config arg) is required for DB status lookups to ever match -- passing
    # config_path directly, from an unrelated cwd, would silently namespace
    # every rule id and permanently misclassify verified rules as candidate.
    if config_path.is_dir():
        semgrep_cwd, config_arg = config_path, "."
    else:
        semgrep_cwd, config_arg = config_path.parent, config_path.name

    proc = await asyncio.create_subprocess_exec(
        "semgrep",
        "--config",
        config_arg,
        "--sarif",
        "--quiet",
        "--metrics=off",
        worktree_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=semgrep_cwd,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_SEMGREP_TIMEOUT_S)
    except (asyncio.TimeoutError, TimeoutError):
        proc.kill()
        with contextlib.suppress(Exception):
            await proc.communicate()
        logger.warning("semgrep timed out after %ds scanning %s", _SEMGREP_TIMEOUT_S, worktree_path)
        return None

    # Verified empirically (no --error flag passed above): semgrep exits 0
    # whether or not it produced findings, and only exits non-zero on its
    # own execution errors (e.g. exit 7 on an invalid rule file). So a
    # non-zero exit here is never "verified-rule hits got silently
    # dropped" -- it's semgrep itself failing to run, and None (rather
    # than the misleadingly-successful-looking []) is the correct signal.
    if proc.returncode != 0:
        logger.warning(
            "semgrep exited %d scanning %s.\nstderr: %s",
            proc.returncode,
            worktree_path,
            stderr.decode(errors="replace")[:500],
        )
        return None

    results = parse_semgrep_sarif(stdout)
    if not results:
        return []

    # semgrep's SARIF `artifactLocation.uri` is whatever path form we scan
    # with -- since we pass the absolute worktree_path as the scan target,
    # that's an absolute temp path (verified empirically), which would
    # otherwise leak into the fast-fail PR comment and writer context.
    # Strip it down to a path relative to the worktree, matching how every
    # other finding in this pipeline reports file paths. Two candidate
    # prefixes, not one: os.path.realpath's resolved form in addition to
    # the literal worktree_path, in case semgrep (or the OS, e.g. macOS's
    # /var -> /private/var symlink) reports a canonicalized path for
    # individual matched files even when the scan-root argument itself
    # wasn't resolved.
    # Longest-first: not just deduped, but ordered, so a prefix that happens
    # to be a strict prefix of the other (not the case today, but not an
    # invariant this code should silently depend on) always matches its
    # fullest form first rather than whichever the set happened to iterate.
    prefixes = sorted(
        {worktree_path.rstrip("/") + "/", os.path.realpath(worktree_path).rstrip("/") + "/"},
        key=len,
        reverse=True,
    )

    def _strip(r: SarifResult) -> SarifResult:
        if r.file is None:
            return r
        for prefix in prefixes:
            if r.file.startswith(prefix):
                return replace(r, file=r.file[len(prefix) :])
        return r

    return [_strip(r) for r in results]


async def run_precheck(worktree_path: str) -> PrecheckResult:
    """Run custom rules against ``worktree_path``; classify hits by DB status.

    The caller (``graph._node_precheck_rules``) owns logging candidate
    firings for later triage and short-circuiting the pipeline on verified
    findings — this function only runs semgrep and classifies the results.
    """
    # Redundant with run_semgrep_sarif's own check below -- kept here purely
    # as an optimization to skip resolving/scanning the rules directory
    # entirely when semgrep isn't installed, not for correctness (that
    # correctness now lives in run_semgrep_sarif itself, shared by every
    # caller including argus.precheck.shadow).
    if not semgrep_available():
        logger.info(
            "semgrep not on PATH (argus[prechecks] extra not installed) — skipping precheck"
        )
        return PrecheckResult()

    rules_dir = resolve_rules_dir()
    if rules_dir is None or not _has_rule_files(rules_dir):
        return PrecheckResult()

    # None (semgrep didn't run) and [] (ran, no hits) are both "nothing to
    # gate or attach" for the live pipeline -- see run_semgrep_sarif's
    # docstring for why run_shadow_review can't collapse these the same way.
    results = await run_semgrep_sarif(worktree_path, rules_dir)
    if not results:
        return PrecheckResult()

    rule_ids = sorted({r.rule_id for r in results})
    statuses = await select_rule_statuses(rule_ids)

    # Classify BEFORE capping: a verified hit must never be subject to the
    # aggregate-size cap below, whatever position it happens to land in
    # SARIF's scan-order-dependent results list. An earlier version capped
    # the raw `results` list first, which could silently drop a verified
    # (fast-fail) finding on a PR with many total hits -- exactly the
    # dropped-BLOCKING-finding bug this ordering exists to prevent.
    candidate: list[SarifResult] = []
    verified: list[SarifResult] = []
    for r in results:
        status = statuses.get(r.rule_id, "candidate")
        if status == "suspended":
            continue
        (verified if status == "verified" else candidate).append(r)

    # Per-message truncation (see sarif._MAX_MESSAGE_LENGTH) bounds one
    # finding's size but not how many reach the LLM writer context. Only
    # candidate findings are capped -- that's the list a size cap actually
    # protects (writer-context length); verified findings gate a PR
    # regardless of how many there are.
    if len(candidate) > _MAX_RESULTS:
        logger.warning(
            "semgrep precheck produced %d candidate results, capping to %d",
            len(candidate),
            _MAX_RESULTS,
        )
        candidate = candidate[:_MAX_RESULTS]

    return PrecheckResult(candidate_findings=candidate, verified_findings=verified)
