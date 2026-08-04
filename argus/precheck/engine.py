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

# Caps the aggregate size of what reaches the LLM writer context or a
# fast-fail PR comment -- per-message truncation (sarif._MAX_MESSAGE_LENGTH)
# bounds one finding, this bounds the count.
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


async def run_precheck(worktree_path: str) -> PrecheckResult:
    """Run custom rules against ``worktree_path``; classify hits by DB status.

    The caller (``graph._node_precheck_rules``) owns logging candidate
    firings for later triage and short-circuiting the pipeline on verified
    findings — this function only runs semgrep and classifies the results.
    """
    if not semgrep_available():
        logger.info(
            "semgrep not on PATH (argus[prechecks] extra not installed) — skipping precheck"
        )
        return PrecheckResult()

    rules_dir = resolve_rules_dir()
    if rules_dir is None or not _has_rule_files(rules_dir):
        return PrecheckResult()

    proc = await asyncio.create_subprocess_exec(
        "semgrep",
        "--config",
        str(rules_dir),
        "--sarif",
        "--quiet",
        "--metrics=off",
        worktree_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_SEMGREP_TIMEOUT_S)
    except (asyncio.TimeoutError, TimeoutError):
        proc.kill()
        with contextlib.suppress(Exception):
            await proc.communicate()
        logger.warning("semgrep precheck timed out after %ds", _SEMGREP_TIMEOUT_S)
        return PrecheckResult()

    # Verified empirically (no --error flag passed above): semgrep exits 0
    # whether or not it produced findings, and only exits non-zero on its
    # own execution errors (e.g. exit 7 on an invalid rule file). So a
    # non-zero exit here is never "verified-rule hits got silently
    # dropped" -- it's semgrep itself failing to run, and returning an
    # empty result is the correct fail-open response, not a lossy one.
    if proc.returncode != 0:
        logger.warning(
            "semgrep precheck exited %d — skipping precheck for this run.\nstderr: %s",
            proc.returncode,
            stderr.decode(errors="replace")[:500],
        )
        return PrecheckResult()

    results = parse_semgrep_sarif(stdout)
    if not results:
        return PrecheckResult()

    # Per-message truncation (see sarif._MAX_MESSAGE_LENGTH) bounds one
    # finding's size but not how many findings reach the LLM writer context
    # or a fast-fail PR comment. A rule matching a repeated pattern many
    # times over could otherwise still produce an oversized aggregate
    # payload; cap the count too.
    if len(results) > _MAX_RESULTS:
        logger.warning(
            "semgrep precheck produced %d results, capping to %d", len(results), _MAX_RESULTS
        )
        results = results[:_MAX_RESULTS]

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
    prefixes = {
        worktree_path.rstrip("/") + "/",
        os.path.realpath(worktree_path).rstrip("/") + "/",
    }

    def _strip(r: SarifResult) -> SarifResult:
        if r.file is None:
            return r
        for prefix in prefixes:
            if r.file.startswith(prefix):
                return replace(r, file=r.file[len(prefix) :])
        return r

    results = [_strip(r) for r in results]

    rule_ids = sorted({r.rule_id for r in results})
    statuses = await select_rule_statuses(rule_ids)

    candidate: list[SarifResult] = []
    verified: list[SarifResult] = []
    for r in results:
        status = statuses.get(r.rule_id, "candidate")
        if status == "suspended":
            continue
        (verified if status == "verified" else candidate).append(r)

    return PrecheckResult(candidate_findings=candidate, verified_findings=verified)
