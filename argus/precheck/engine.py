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

    ``worktree_path`` and ``config_path`` are both passed to semgrep exactly
    as given (no cwd juggling): correctness no longer depends on either
    being absolute or on this process's cwd.
    """
    if not semgrep_available():
        logger.info("semgrep not on PATH (argus[prechecks] extra not installed) — skipping scan")
        return None

    try:
        return await _run_semgrep_sarif_unguarded(worktree_path, config_path)
    except Exception:  # noqa: BLE001 -- enforce the documented never-raises contract
        logger.warning(
            "semgrep subprocess/SARIF-parsing failed unexpectedly scanning %s",
            worktree_path,
            exc_info=True,
        )
        return None


async def _kill_and_reap(proc: asyncio.subprocess.Process) -> None:
    """Best-effort kill + drain of a still-live subprocess.

    ``kill()`` itself can raise ``ProcessLookupError`` if the process already
    exited on its own between the caller noticing a problem and this call;
    the post-kill ``communicate()`` drain is unbounded (no timeout) since
    it's just reaping an already-killed/exiting process, not waiting on a
    live one, and any failure there is likewise not worth surfacing.
    """
    with contextlib.suppress(ProcessLookupError):
        proc.kill()
    with contextlib.suppress(Exception):
        await proc.communicate()


async def _run_semgrep_sarif_unguarded(
    worktree_path: str, config_path: Path
) -> list[SarifResult] | None:
    """The actual subprocess-and-parse work for :func:`run_semgrep_sarif`.

    Split out so the outer function can wrap this whole body in one
    try/except -- ``asyncio.create_subprocess_exec`` (e.g. a TOCTOU race if
    the semgrep binary vanishes between ``semgrep_available()`` and this
    call) and ``parse_semgrep_sarif`` (malformed SARIF) were previously
    unguarded despite the module's fail-open contract and this function's
    own docstring claiming it "never raises".
    """
    proc = await asyncio.create_subprocess_exec(
        "semgrep",
        "--config",
        str(config_path),
        # Verified empirically: semgrep's default --rewrite-rule-ids
        # namespaces a rule's reported ruleId with its path *relative to the
        # config root* -- e.g. a rule "foo" nested at "<config>/security/foo.yml"
        # is reported as "security.foo", not "foo", even for a directory
        # config passed as an absolute path with no cwd tricks. This bit for
        # bit defeats `select_rule_statuses`' DB lookup by rule_id for any
        # subdirectory-organized rule set (which `_has_rule_files`'s
        # recursive rglob explicitly anticipates and supports) -- a verified
        # rule under a subdirectory could never actually reach 'verified'
        # classification. --no-rewrite-rule-ids reports the bare `id:` value
        # unconditionally, for both flat and nested layouts, and for both a
        # directory and a single-file config -- no cwd manipulation needed
        # at all (an earlier version of this function juggled `cwd`/relative
        # `--config` args specifically to work around --rewrite-rule-ids for
        # the flat case only, which is why this flag is strictly better, not
        # just simpler).
        "--no-rewrite-rule-ids",
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
        await _kill_and_reap(proc)
        logger.warning("semgrep timed out after %ds scanning %s", _SEMGREP_TIMEOUT_S, worktree_path)
        return None
    except BaseException:
        # Guarantees the child is killed/reaped on any OTHER exception raised
        # while it's live too, not just the TimeoutError case above --
        # without this, an OSError/BrokenPipeError (or any other exception)
        # after spawn but before communicate() returns would propagate
        # straight out through this function's own outer try/except (see
        # run_semgrep_sarif) with the child left running, orphaned and
        # unreaped: a resource leak that's harder to notice than the crash
        # it replaced.
        await _kill_and_reap(proc)
        raise

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
    # with -- every current caller passes an absolute temp path as
    # worktree_path (verified empirically that semgrep echoes it back
    # as-given), which would otherwise leak into the fast-fail PR comment
    # and writer context. This is a caller contract, not something this
    # function itself enforces or normalizes (see this function's docstring);
    # a caller passing a relative path would see stripping silently no-op
    # rather than fail, since the prefixes below just wouldn't match.
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
