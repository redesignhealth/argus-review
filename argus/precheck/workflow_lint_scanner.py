"""Deterministic-precheck rule execution: run actionlint against changed
GitHub Actions workflow files.

actionlint (https://github.com/rhysd/actionlint) is a syntax/correctness
linter for GitHub Actions workflows -- distinct from zizmor's security
focus (unpinned tags, script injection). actionlint catches things zizmor
structurally cannot: typo'd action inputs (validated against the actual
action's input schema), expression syntax errors (`=` vs `==`), and --
via its built-in shellcheck integration -- real shell-scripting bugs
inside `run:` blocks. A stock, externally-vetted scanner, distinct from
custom/mined rules -- see docs/PRECHECKS.md's "stock rule sources"
section.

Like squawk, and unlike zizmor/semgrep, this scanner takes an explicit
``changed_files`` list rather than scanning the whole worktree -- but for
a different reason than squawk's "years of migration history" one:
actionlint requires either running from inside a git-project root it can
detect (verified empirically: it errors with "no project was found" for
an arbitrary directory) or being given explicit file paths directly.
Passing the changed workflow files directly sidesteps that project-root
detection entirely and is naturally diff-scoped besides.

actionlint has no SARIF reporter -- this module parses its
``--format '{{json .}}'`` Go-template output directly into ``SarifResult``,
the same direct-JSON-parsing approach as squawk (see migration_scanner.py)
rather than zizmor's shared-SARIF-parser approach.

**Not installable via this project's ``prechecks`` extra**: like squawk,
actionlint is a standalone Go binary (distributed via ``go install``, a
package manager like Homebrew, or a binary download) -- not on PyPI, so
``pip``/``uv`` cannot install it. Its own built-in shellcheck integration
additionally requires the separate ``shellcheck`` binary on PATH to
produce shellcheck-kind findings at all (verified empirically: without
it, actionlint still runs and reports everything else, just silently
skips the shellcheck checks). Both must be installed separately wherever
this package runs live (see docs/PRECHECKS.md). Same graceful-skip
fail-open pattern as every other scanner here when either is absent.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil

from argus.precheck.sarif import SarifResult
from argus.precheck.scanner_utils import is_success_exit, run_scanner_subprocess

logger = logging.getLogger(__name__)

_ACTIONLINT_TIMEOUT_S = 60

# actionlint's own `kind` field is a coarse category ("action", "syntax",
# "expression", "shellcheck", ...), not a specific rule id -- every
# shellcheck finding (SC2034, SC2086, ...) shares the bare kind
# "shellcheck" with no finer-grained id anywhere except embedded in the
# free-text message. Extracted here so shellcheck findings can be
# individually classified (verified/suspended) rather than all sharing one
# bucket. Re-verify this message shape on an actionlint version bump.
_SHELLCHECK_CODE_RE = re.compile(r"\b(SC\d+)\b")

# actionlint exits 1 when it finds real issues (verified empirically) and
# a DISTINCT nonzero code (3, in the versions tested) for genuine errors
# (unreadable/nonexistent file) -- unlike squawk, where both cases share
# exit 1. This module still pre-filters to files that exist on disk (see
# run_actionlint_sarif) so the genuine-error path shouldn't occur in
# practice, but the two codes being distinct here means it wouldn't be
# silently misclassified as a successful scan even if it did.
_FINDINGS_EXIT_CODE = 1


def actionlint_available() -> bool:
    """True if the ``prechecks`` extra's actionlint binary is on PATH."""
    return shutil.which("actionlint") is not None


async def run_actionlint_sarif(
    worktree_path: str, changed_files: list[str]
) -> list[SarifResult] | None:
    """Run actionlint against the changed GitHub Actions workflow files in
    ``changed_files`` that still exist on disk; return unclassified
    results.

    Deliberately takes ``changed_files`` as a required argument -- see
    this module's docstring for why (project-root detection, not a
    flood-risk concern the way squawk's whole-worktree mode would be).
    ``run_precheck`` skips calling this entirely when it has no
    ``changed_files`` list.

    Returns ``[]`` without spawning a subprocess when none of
    ``changed_files`` are under ``.github/workflows/`` -- the common case
    for most PRs.

    Returns ``None`` -- distinct from ``[]`` -- when actionlint did not
    actually run to completion (missing binary, timeout, execution error),
    matching every other scanner's contract in this package.
    """
    if not actionlint_available():
        logger.info("actionlint not on PATH (argus[prechecks] extra not installed) — skipping scan")
        return None

    workflow_files = [
        f
        for f in changed_files
        if f.startswith(".github/workflows/")
        and f.endswith((".yml", ".yaml"))
        and os.path.isfile(os.path.join(worktree_path, f))
    ]
    if not workflow_files:
        return []

    try:
        return await _run_actionlint_sarif_unguarded(worktree_path, workflow_files)
    except Exception:  # noqa: BLE001 -- enforce the documented never-raises contract
        logger.warning(
            "actionlint subprocess/output-parsing failed unexpectedly scanning %s",
            worktree_path,
            exc_info=True,
        )
        return None


async def _run_actionlint_sarif_unguarded(
    worktree_path: str, workflow_files: list[str]
) -> list[SarifResult] | None:
    """The actual subprocess-and-parse work for :func:`run_actionlint_sarif`.

    Invoked with ``cwd=worktree_path`` and relative file paths, same as
    squawk -- verified empirically that actionlint then echoes back the
    same relative form in its JSON output, so no path-stripping is needed.
    """
    if not os.path.isabs(worktree_path):
        raise ValueError(f"worktree_path must be absolute, got {worktree_path!r}")

    # workflow_files are always prefixed with ".github/workflows/" (the scope
    # filter above), so none of these argv tokens can themselves start with
    # "-" even if the bare filename does -- unlike squawk/checkov/eslint,
    # this scanner isn't actually exposed to the dash-prefixed-filename
    # argument-injection risk those modules guard against. "--" added anyway
    # for defense in depth (verified empirically not to change actionlint's
    # behavior or its path echo).
    argv = ["actionlint", "--format", "{{json .}}", "--", *workflow_files]

    outcome = await run_scanner_subprocess(argv, cwd=worktree_path, timeout=_ACTIONLINT_TIMEOUT_S)
    if outcome is None:
        return None
    stdout, stderr, returncode = outcome

    if not is_success_exit(returncode, findings_exit_code=_FINDINGS_EXIT_CODE):
        logger.warning(
            "actionlint exited %d scanning %s.\nstderr: %s",
            returncode,
            worktree_path,
            stderr.decode(errors="replace")[:500],
        )
        return None

    try:
        raw_findings = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Could not parse actionlint JSON output", exc_info=True)
        return []

    results = []
    for item in raw_findings:
        kind = item.get("kind")
        message = item.get("message", "")
        if not kind:
            continue
        rule_id = f"actionlint/{kind}"
        if kind == "shellcheck":
            sc_match = _SHELLCHECK_CODE_RE.search(message)
            if sc_match:
                rule_id = f"actionlint/shellcheck/{sc_match.group(1)}"
        results.append(
            SarifResult(
                rule_id=rule_id,
                level="warning",
                message=message,
                file=item.get("filepath"),
                line=item.get("line"),
            )
        )
    return results
