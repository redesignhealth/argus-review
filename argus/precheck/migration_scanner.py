"""Deterministic-precheck rule execution: run squawk against changed SQL
migration files.

squawk (https://squawkhq.com) is a purpose-built linter for Postgres
migrations -- missing `CONCURRENTLY`/`IF NOT EXISTS`, unsafe column type
changes, constraints added without `NOT VALID`, missing lock/statement
timeouts, etc. A stock, externally-vetted scanner like zizmor, distinct
from custom/mined rules -- see docs/PRECHECKS.md's "stock rule sources"
section.

Unlike every other scanner in this package, this one does NOT scan the
whole worktree: it only ever runs when the caller (``run_precheck``)
already has an explicit ``changed_files`` list (i.e. never in the
whole-worktree/``changed_files=None`` mode other scanners also support --
see ``run_squawk_sarif``'s docstring for why). A migrations directory can
hold years of history; scanning all of it every PR would be slow and
would flood every PR with pre-existing findings regardless of the
post-merge diff-scoping ``run_precheck`` already applies.

squawk has no SARIF reporter (``plain``/``gcc``/``json`` only) -- this
module parses its ``--reporter json`` output directly into ``SarifResult``
rather than going through the shared SARIF parser like zizmor does.

**Not installable via this project's ``prechecks`` extra**: squawk is
distributed as an npm package (``squawk-cli``, a Rust binary), not on
PyPI -- ``pip``/``uv`` cannot install it, so it can't be declared as a
dependency the way semgrep/zizmor are. It must be installed separately
(``npm install -g squawk-cli``) wherever this package runs live (see
docs/PRECHECKS.md). Same ``squawk_available()``/graceful-skip fail-open
pattern as every other scanner here when it's absent -- this is an
operational/deployment gap, not a functional one.
"""

from __future__ import annotations

import json
import logging
import os
import shutil

from argus.precheck.sarif import SarifResult
from argus.precheck.scanner_utils import is_success_exit, run_scanner_subprocess

logger = logging.getLogger(__name__)

_SQUAWK_TIMEOUT_S = 60

# Style opinions, not safety/correctness issues -- noisy for a gate whose
# job is catching migration footguns, not enforcing a schema style guide.
# Verified against squawk's own default rule set; re-check on a version
# bump the same way this project re-verifies every other tool's CLI
# contract.
_EXCLUDED_RULES = ("prefer-text-field", "prefer-bigint-over-int")

# squawk uses exit 1 for BOTH "ran fine, found real issues" and "couldn't
# find any files to scan" (verified empirically: a nonexistent path also
# exits 1, with "Failed to find files..." on stderr, not stdout) -- the
# same exit code covers a genuine error and a successful-with-findings
# run, unlike every other tool this package wraps. Resolved by construction
# here, not by inspecting stderr: this module only ever invokes squawk
# with paths it has already confirmed exist on disk (see
# run_squawk_sarif), so a nonexistent-file error should never actually
# occur in practice; if it somehow does, treating exit 1 as "success" per
# is_success_exit below is an acceptable trade against never mis-treating
# a normal findings-only run as a scan failure.
_FINDINGS_EXIT_CODE = 1


def squawk_available() -> bool:
    """True if the ``prechecks`` extra's squawk binary is on PATH."""
    return shutil.which("squawk") is not None


async def run_squawk_sarif(
    worktree_path: str, changed_files: list[str]
) -> list[SarifResult] | None:
    """Run squawk against the ``.sql`` files in ``changed_files`` that
    still exist on disk; return unclassified results.

    Deliberately takes ``changed_files`` as a required, non-optional
    argument (unlike ``run_semgrep_sarif``/``run_zizmor_sarif``, which
    both support a whole-worktree mode) -- see this module's docstring.
    ``run_precheck`` skips calling this entirely when it has no
    ``changed_files`` list (i.e. its own ``changed_files=None`` mode).

    Returns ``[]`` without spawning a subprocess at all when none of
    ``changed_files`` are ``.sql`` files -- the common case for most
    PRs -- both because there's nothing to scan and because it sidesteps
    squawk's ambiguous "no files found" exit code entirely (see
    ``_FINDINGS_EXIT_CODE``'s docstring).

    Returns ``None`` -- distinct from ``[]`` -- when squawk did not
    actually run to completion (missing binary, timeout, execution
    error), matching every other scanner's contract in this package.
    """
    if not squawk_available():
        logger.info("squawk not on PATH (argus[prechecks] extra not installed) — skipping scan")
        return None

    sql_files = [
        f
        for f in changed_files
        if f.endswith(".sql") and os.path.isfile(os.path.join(worktree_path, f))
    ]
    if not sql_files:
        return []

    try:
        return await _run_squawk_sarif_unguarded(worktree_path, sql_files)
    except Exception:  # noqa: BLE001 -- enforce the documented never-raises contract
        logger.warning(
            "squawk subprocess/output-parsing failed unexpectedly scanning %s",
            worktree_path,
            exc_info=True,
        )
        return None


async def _run_squawk_sarif_unguarded(
    worktree_path: str, sql_files: list[str]
) -> list[SarifResult] | None:
    """The actual subprocess-and-parse work for :func:`run_squawk_sarif`.

    Invoked with ``cwd=worktree_path`` and relative file paths -- verified
    empirically that squawk then echoes back the same relative form in its
    JSON output, so no path-stripping is needed (matching zizmor's
    already-relative reporting, unlike semgrep's absolute-echo-then-strip
    dance).
    """
    if not os.path.isabs(worktree_path):
        raise ValueError(f"worktree_path must be absolute, got {worktree_path!r}")

    argv = ["squawk", "--reporter", "json"]
    for rule in _EXCLUDED_RULES:
        argv.extend(("--exclude", rule))
    # "--" guards against a dash-prefixed changed-file path being parsed as
    # a flag -- see scanner_utils.py's "Dash-prefixed-filename argument
    # injection" section for the full rationale (shared across all four
    # per-file scanners). Confirmed "--" preserves the exact relative path
    # in squawk's own JSON echo (its `file` field).
    argv.append("--")
    argv.extend(sql_files)

    outcome = await run_scanner_subprocess(argv, cwd=worktree_path, timeout=_SQUAWK_TIMEOUT_S)
    if outcome is None:
        return None
    stdout, stderr, returncode = outcome

    if not is_success_exit(returncode, findings_exit_code=_FINDINGS_EXIT_CODE):
        logger.warning(
            "squawk exited %d scanning %s.\nstderr: %s",
            returncode,
            worktree_path,
            stderr.decode(errors="replace")[:500],
        )
        return None

    try:
        raw_findings = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Could not parse squawk JSON output", exc_info=True)
        return []

    results = []
    for item in raw_findings:
        rule_name = item.get("rule_name")
        if not rule_name:
            continue
        # squawk's JSON `line` is 0-indexed (verified empirically against
        # its own `--reporter tty`, which reports the same finding
        # 1-indexed) -- re-verify on a squawk version bump.
        line = item.get("line")
        results.append(
            SarifResult(
                # Namespaced: squawk's own rule ids are bare, generic
                # hyphenated strings ("prefer-robust-stmts") with real
                # collision risk against other stock scanners' equally
                # generic ids (contrast zizmor/Checkov, whose own native
                # id conventions -- "zizmor/...", "CKV_AWS_..." -- are
                # already distinctive enough not to need this).
                rule_id=f"squawk/{rule_name}",
                level="warning",
                message=item.get("message", ""),
                file=item.get("file"),
                line=(line + 1) if isinstance(line, int) else None,
            )
        )
    return results
