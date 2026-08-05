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
from collections.abc import Coroutine, Sequence
from dataclasses import dataclass, field, replace
from importlib import resources
from typing import Any
from pathlib import Path

from argus.config import get_settings
from argus.precheck.actions_scanner import run_zizmor_sarif, zizmor_available
from argus.precheck.js_scanner import eslint_available, run_eslint_sarif
from argus.precheck.migration_scanner import run_squawk_sarif, squawk_available
from argus.precheck.sarif import SarifResult, parse_semgrep_sarif
from argus.precheck.secrets_scanner import run_trivy_secrets_sarif, trivy_available
from argus.precheck.terraform_scanner import checkov_available, run_checkov_sarif
from argus.precheck.workflow_lint_scanner import (
    actionlint_available,
    run_actionlint_sarif,
)
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


def _find_duplicate_rule_ids(rules_dir: Path) -> dict[str, list[Path]]:
    """Load-time lint: which ``id:`` values appear in more than one rule file.

    Security-relevant, not just a lint nit: ``--no-rewrite-rule-ids`` (see
    ``run_semgrep_sarif``) means ``select_rule_statuses`` keys purely off the
    bare ``id:`` string in one flat table (see
    ``argus/precheck/rules/README.md``'s id-uniqueness note) -- two rule
    files sharing an ``id:`` are indistinguishable to that lookup, so a new
    rule accidentally reusing an already-``verified`` id would inherit
    fast-fail status immediately, skipping shadow-review/triage entirely.
    A rule file that fails to parse as YAML, or doesn't have the expected
    top-level ``rules: [...]`` shape, is skipped here rather than raising --
    this is advisory tooling on top of the fail-open precheck gate, not
    itself something that should ever block a review -- but logged at
    DEBUG (not silently), since a file this lint can't read is also a file
    it can't check for a collision, which is exactly the security-relevant
    condition this whole function exists to catch.

    Synchronous, deliberately: called via ``asyncio.to_thread`` from
    :func:`run_precheck` rather than made ``async`` itself, so it stays
    directly unit-testable without an event loop.

    Imports ``yaml`` lazily rather than at module level: ``pyyaml`` is
    declared in the ``prechecks`` extra, not core ``dependencies`` --
    unconditionally importing it at module level would make
    ``argus.precheck.engine`` itself fail to import without that extra
    installed, and ``graph._node_precheck_rules`` imports this module
    outside its own fail-open try block, so that ``ImportError`` would
    crash the whole review rather than no-op. (Latent today only because
    other dependencies happen to pull pyyaml in transitively -- not a
    contract this module should rely on.)
    """
    import yaml

    ids_to_files: dict[str, list[Path]] = {}
    for path in (*rules_dir.rglob("*.yml"), *rules_dir.rglob("*.yaml")):
        try:
            content = yaml.safe_load(path.read_text())
        except (yaml.YAMLError, OSError):
            # WARNING, not DEBUG: a file this lint can't read is a file it
            # can't check for a collision -- the same security-relevant
            # blind spot a detected collision itself warns about, so it
            # gets the same visibility.
            logger.warning("Could not parse %s while checking for duplicate rule ids", path)
            continue
        if not isinstance(content, dict):
            continue
        rules = content.get("rules", [])
        if not isinstance(rules, list):
            continue
        for rule in rules:
            if isinstance(rule, dict) and isinstance(rule.get("id"), str):
                ids_to_files.setdefault(rule["id"], []).append(path)
    return {rule_id: files for rule_id, files in ids_to_files.items() if len(files) > 1}


def semgrep_available() -> bool:
    """True if the ``prechecks`` extra's semgrep binary is on PATH."""
    return shutil.which("semgrep") is not None


async def run_semgrep_sarif(
    worktree_path: str, config_path: Path | Sequence[str | Path]
) -> list[SarifResult] | None:
    """Run semgrep against ``worktree_path``; return stripped, unclassified
    SARIF results.

    ``config_path`` accepts either a single rules directory/file (a bare
    ``Path`` -- every existing caller's shape, unchanged) or a sequence of
    config sources merged into one semgrep invocation via multiple
    ``--config`` flags (semgrep's own native way to combine sources; not a
    new subprocess per source). A sequence entry can be a local path or a
    semgrep registry pack id (e.g. ``"p/secrets"``) -- semgrep resolves the
    latter over the network, fetching once and caching locally. This is
    what lets :func:`run_precheck` combine a custom ``ARGUS_RULES_DIR`` with
    ``ARGUS_STOCK_SEMGREP_PACKS`` in one scan.

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

    ``worktree_path`` and every ``config_path`` entry are passed to semgrep
    exactly as given (no cwd juggling): correctness no longer depends on
    either being absolute or on this process's cwd.
    """
    if not semgrep_available():
        logger.info("semgrep not on PATH (argus[prechecks] extra not installed) — skipping scan")
        return None

    configs = [config_path] if isinstance(config_path, (str, Path)) else list(config_path)
    try:
        return await _run_semgrep_sarif_unguarded(worktree_path, configs)
    except Exception:  # noqa: BLE001 -- enforce the documented never-raises contract
        logger.warning(
            "semgrep subprocess/SARIF-parsing failed unexpectedly scanning %s",
            worktree_path,
            exc_info=True,
        )
        return None


_KILL_DRAIN_TIMEOUT_S = 5


async def _kill_and_reap(proc: asyncio.subprocess.Process) -> None:
    """Best-effort kill + bounded drain of a still-live subprocess.

    ``kill()`` itself can raise ``ProcessLookupError`` if the process already
    exited on its own between the caller noticing a problem and this call.

    The post-kill ``communicate()`` drain is bounded, NOT unbounded --
    verified empirically that semgrep spawns its own worker subprocesses
    (``semgrep-core``); ``proc.kill()`` only SIGKILLs the direct child, not
    the process group, so a grandchild that inherited the piped stdout/
    stderr file descriptors can keep them open after the direct child
    exits. An earlier version of this function assumed the drain would
    return quickly once killed and left it unbounded -- but this function
    runs on the per-PR review's critical path (``graph._node_precheck_rules``)
    with no LangGraph-level timeout/retry policy backstopping it, so an
    unbounded hang here would stall the whole review indefinitely instead
    of failing open, which is the opposite of this module's entire
    fail-open design intent. A timeout here also triggers an explicit
    transport-close attempt below, since the timeout alone only stops
    *this process* from hanging -- it doesn't release the pipe file
    descriptors on our side. Any exception from the drain (including the
    ``TimeoutError``/``asyncio.TimeoutError`` this bound itself can raise)
    is swallowed -- a cleanup step failing is not worth surfacing any more
    than any other cleanup failure here.

    Edge case, not handled: ``contextlib.suppress(Exception)`` doesn't
    suppress ``asyncio.CancelledError`` (a ``BaseException``, not an
    ``Exception``). If a *second* cancellation arrives while this drain is
    in flight, that new ``CancelledError`` propagates out of this function
    and replaces whatever exception the caller's own ``except BaseException:
    ... raise`` was in the middle of propagating -- silently discarding it.
    Low-likelihood (requires cancellation during cleanup of an already-
    cancelled/failed call), not handled specially here.
    """
    with contextlib.suppress(ProcessLookupError):
        proc.kill()
    try:
        await asyncio.wait_for(proc.communicate(), timeout=_KILL_DRAIN_TIMEOUT_S)
    except Exception:  # noqa: BLE001 -- see docstring: any cleanup failure is swallowed
        # A drain that reaches this point means a semgrep-core grandchild is
        # (or may be) still holding the pipe FDs open on the per-PR critical
        # path -- worth a log line before the swallowed cleanup attempt
        # below, matching the WARNING already logged for the sibling
        # timeout path in _run_semgrep_sarif_unguarded.
        logger.warning("Post-kill drain of semgrep subprocess failed or timed out", exc_info=True)
        # The timeout bound above stops *this process* from hanging on a
        # grandchild that's still holding the pipe FDs open -- it doesn't
        # release those FDs on our side. Explicitly closing the transport
        # does: it closes our end of the pipes regardless of whether a
        # semgrep-core grandchild still holds its own copy open, so this
        # process doesn't accumulate leaked FDs across many precheck runs
        # even though it can't force the grandchild to release its own.
        # `_transport` is a private asyncio.subprocess.Process attribute
        # (no public equivalent exists for this); best-effort, so any
        # failure here (including it simply not existing on some backend)
        # is swallowed like every other cleanup step in this function.
        with contextlib.suppress(Exception):
            transport = getattr(proc, "_transport", None)
            if transport is not None:
                transport.close()


async def _run_semgrep_sarif_unguarded(
    worktree_path: str, config_paths: list[str | Path]
) -> list[SarifResult] | None:
    """The actual subprocess-and-parse work for :func:`run_semgrep_sarif`.

    Split out so the outer function can wrap this whole body in one
    try/except -- ``asyncio.create_subprocess_exec`` (e.g. a TOCTOU race if
    the semgrep binary vanishes between ``semgrep_available()`` and this
    call) and ``parse_semgrep_sarif`` (malformed SARIF) were previously
    unguarded despite the module's fail-open contract and this function's
    own docstring claiming it "never raises".
    """
    # Converts a silent contract violation into a loud, immediately-
    # diagnosable one: a relative worktree_path would otherwise make the
    # SARIF path-stripping below silently no-op (the prefixes just wouldn't
    # match), leaking an absolute temp path into the fast-fail PR comment
    # and writer context with no error signal. Every current caller passes
    # an absolute path via repo_provision.py's tempfile.mkdtemp-based
    # worktree, so this is a caller-contract check, not a real-world case
    # this function needs to handle gracefully -- a raised ValueError here
    # is caught by run_semgrep_sarif's own outer except and fails open like
    # everything else in this module, it just also gets logged loudly via
    # exc_info=True there instead of silently misreporting paths. A bare
    # `assert` (an earlier version of this check used one) strips under
    # `python -O`/`PYTHONOPTIMIZE`, silently reverting to the leaky
    # behavior this check exists to prevent -- see the identical rationale
    # at `argus/storage/precheck.py`'s own length-mismatch guard.
    if not os.path.isabs(worktree_path):
        raise ValueError(f"worktree_path must be absolute, got {worktree_path!r}")

    config_args: list[str] = []
    for c in config_paths:
        config_args.extend(("--config", str(c)))

    proc = await asyncio.create_subprocess_exec(
        "semgrep",
        *config_args,
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
    # with -- since this function asserts worktree_path is absolute above,
    # that's always an absolute temp path here (verified empirically that
    # semgrep echoes it back as-given), which would otherwise leak into the
    # fast-fail PR comment and writer context.
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


async def _run_semgrep_precheck(worktree_path: str) -> list[SarifResult]:
    """The semgrep half of run_precheck: custom ``ARGUS_RULES_DIR`` rules
    plus any configured ``ARGUS_STOCK_SEMGREP_PACKS``, merged into one scan.

    Returns ``[]`` whenever semgrep isn't installed, has nothing to scan
    with (no custom rules dir AND no stock packs configured), or fails to
    run -- callers can't distinguish "no config sources" from "ran, no
    hits" here, but run_precheck doesn't need to: either way there's
    nothing to gate or attach from this half.
    """
    if not semgrep_available():
        logger.info("semgrep not on PATH (argus[prechecks] extra not installed) — skipping scan")
        return []

    config_sources: list[str | Path] = []

    rules_dir = resolve_rules_dir()
    if rules_dir is not None and _has_rule_files(rules_dir):
        # Advisory only -- logged loudly, never blocks the run itself. See
        # _find_duplicate_rule_ids' docstring for why a collision is
        # security-relevant (a new rule can silently inherit an existing
        # verified rule's fast-fail status), not merely a style nit.
        # to_thread: the function itself does blocking I/O (rglob,
        # read_text, yaml.safe_load) and stays synchronous/directly-
        # testable rather than becoming async -- see its own docstring.
        #
        # Narrowly scoped ImportError catch: pyyaml is genuinely optional
        # (only declared in the `prechecks` extra), and if it's actually
        # absent, only this advisory lint should be skipped -- not the
        # rest of run_precheck. Without this, an ImportError raised inside
        # _find_duplicate_rule_ids would propagate out through this
        # function's caller (graph._node_precheck_rules) to its own outer
        # except, which fail-opens the WHOLE precheck node including the
        # verified-rule fast-fail path -- a much bigger blast radius than
        # "the duplicate-id lint didn't run."
        try:
            duplicates = await asyncio.to_thread(_find_duplicate_rule_ids, rules_dir)
        except ImportError:
            logger.warning(
                "pyyaml not installed -- skipping the duplicate-rule-id lint "
                "(install the `prechecks` extra to enable it)"
            )
            duplicates = {}
        for rule_id, files in duplicates.items():
            logger.warning(
                "Duplicate precheck rule id %r found in multiple files: %s -- "
                "select_rule_statuses cannot distinguish these; a new rule "
                "reusing an existing verified rule's id would silently inherit "
                "fast-fail status",
                rule_id,
                ", ".join(str(f) for f in files),
            )
        config_sources.append(rules_dir)

    stock_packs = get_settings().ARGUS_STOCK_SEMGREP_PACKS
    if stock_packs:
        config_sources.extend(p.strip() for p in stock_packs.split(",") if p.strip())

    if not config_sources:
        return []

    # None (semgrep didn't run) and [] (ran, no hits) are both "nothing to
    # gate or attach" for the live pipeline -- see run_semgrep_sarif's
    # docstring for why run_shadow_review can't collapse these the same way.
    return await run_semgrep_sarif(worktree_path, config_sources) or []


async def run_precheck(
    worktree_path: str, *, changed_files: list[str] | None = None
) -> PrecheckResult:
    """Run stock + custom rules against ``worktree_path``; classify hits by
    DB status.

    Independent scanners feed the same classification pipeline below:
    semgrep (custom ``ARGUS_RULES_DIR`` rules and/or
    ``ARGUS_STOCK_SEMGREP_PACKS`` registry packs), zizmor (GitHub Actions
    security -- actions_scanner.py) and Trivy (secrets -- secrets_scanner.py,
    both whole-worktree/always-on), and squawk (Postgres migrations --
    migration_scanner.py), Checkov (Terraform IAM/privilege-escalation --
    terraform_scanner.py), actionlint (GitHub Actions syntax/shellcheck --
    workflow_lint_scanner.py), and eslint-plugin-security (JS/TS --
    js_scanner.py), all four of which only run when ``changed_files`` is
    given. None depend on the others being configured: an earlier version
    of this function returned early whenever no custom rules directory was
    configured, which would have made stock scanning permanently dead code
    in any deployment (like this
    project's own, as of this change) that hasn't wired up ARGUS_RULES_DIR
    yet.

    ``changed_files``, when given and non-empty, scopes findings to files
    this PR actually touched (see ``helpers.extract_changed_files``) --
    both scanners here scan the WHOLE worktree, not just the diff, so
    without this a repo with any pre-existing debt would surface it on
    every PR, and _MAX_RESULTS below would then truncate that flood
    arbitrarily, silently dropping real, in-scope findings alongside the
    noise. ``None`` (the default) applies no filtering -- used by callers
    that genuinely want whole-worktree results (this module's own tests; a
    future non-PR-context caller). An EMPTY list is treated as "nothing to
    scan for this round" (a no-op, same as if every scanner had been
    skipped) rather than as a real scope to filter against OR as license to
    run every scanner unscoped: extraction returning an empty list from a
    non-empty diff is, in practice, almost always a genuinely empty diff
    (e.g. a comment-triggered re-review with no new commits since the last
    round) rather than truncation -- GitHub compare-diff truncation keeps
    the diff's *leading* lines, which almost always still contain at least
    the first changed file's ``diff --git`` header, so truncation alone
    essentially never drives extraction all the way down to zero files. An
    earlier version of this fallback ran the whole-worktree scanners
    (semgrep/zizmor/Trivy) unscoped whenever ``changed_files == []``, which
    let their ``verified``-status findings reach the fast-fail gate
    (``graph._node_precheck_fail``) based on pre-existing debt in files
    this specific review round has no diff evidence relates to it at all --
    fail-CLOSED for verified rules, the one direction this module's stated
    fail-open philosophy forbids. No-op is the only response that's
    genuinely safe when there's no reliable scope to check against. A
    finding with no ``file`` at all is dropped when a real (non-empty)
    scope is active -- there's no way to confirm it's in scope, and this
    module's fail-open philosophy favors not blocking over false-blocking.

    Every scanner is a subprocess call with its own timeout (up to 120s
    each) -- run concurrently via ``asyncio.gather`` rather than
    sequentially awaited one after another, so this function's worst-case
    latency is roughly the slowest single scanner, not their sum. Safe to
    gather without ``return_exceptions=True``: every scanner wrapper here
    (``run_zizmor_sarif``, etc.) already guarantees it never raises (see
    each module's own docstring), so nothing here should ever propagate an
    exception through ``gather`` in practice.

    The caller (``graph._node_precheck_rules``) owns logging candidate
    firings for later triage and short-circuiting the pipeline on verified
    findings — this function only runs the scanners and classifies the
    results.
    """
    if changed_files is not None and not changed_files:
        # See this function's docstring: an empty (not None) changed_files
        # list means "no relevant scope for this round" in practice, almost
        # always a genuinely empty diff -- running scanners unscoped here
        # would let verified findings fast-fail the PR based on files this
        # round's diff never touched. No-op is the only safe response.
        logger.info(
            "changed_files was an empty list -- treating as no relevant "
            "scope for this review round and skipping precheck scanning "
            "entirely (see run_precheck's docstring)"
        )
        return PrecheckResult()

    scans: list[Coroutine[Any, Any, list[SarifResult] | None]] = [
        _run_semgrep_precheck(worktree_path)
    ]

    if zizmor_available():
        scans.append(run_zizmor_sarif(worktree_path))
    else:
        logger.info("zizmor not on PATH (argus[prechecks] extra not installed) — skipping scan")

    if trivy_available():
        scans.append(run_trivy_secrets_sarif(worktree_path))
    else:
        logger.info("trivy not on PATH (argus[prechecks] extra not installed) — skipping scan")

    # squawk, checkov, actionlint, and eslint all require an explicit
    # changed_files list -- see their own modules' docstrings for why
    # (none has a whole-worktree mode the way semgrep/zizmor/trivy do).
    if changed_files is not None:
        if squawk_available():
            scans.append(run_squawk_sarif(worktree_path, changed_files))
        else:
            logger.info(
                "squawk not on PATH (argus[prechecks] extra not installed) — skipping scan"
            )

        if checkov_available():
            scans.append(run_checkov_sarif(worktree_path, changed_files))
        else:
            logger.info(
                "checkov not on PATH (argus[prechecks] extra not installed) — skipping scan"
            )

        if actionlint_available():
            scans.append(run_actionlint_sarif(worktree_path, changed_files))
        else:
            logger.info(
                "actionlint not on PATH (argus[prechecks] extra not installed) — skipping scan"
            )

        if eslint_available():
            scans.append(run_eslint_sarif(worktree_path, changed_files))
        else:
            logger.info(
                "eslint security bundle not installed (see argus/precheck/eslint_bundle/"
                "README.md) — skipping scan"
            )

    scan_batches = await asyncio.gather(*scans)
    results: list[SarifResult] = [r for batch in scan_batches for r in (batch or [])]

    if changed_files is not None:
        # Guaranteed non-empty here -- the empty-list case returns early
        # above, before any scanner even runs.
        changed_set = set(changed_files)
        before = len(results)
        results = [r for r in results if r.file is not None and r.file in changed_set]
        if before != len(results):
            logger.info(
                "Diff-scoped precheck results: %d -> %d (kept findings in %d changed file(s))",
                before,
                len(results),
                len(changed_set),
            )

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
