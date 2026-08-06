"""Deterministic-precheck rule execution: run zizmor against the worktree.

zizmor (https://docs.zizmor.sh) is a purpose-built static analyzer for
GitHub Actions workflows/actions -- unpinned action tags, script-injection
via ``${{ }}`` interpolation, overly-broad permissions, etc. It's a stock,
externally-vetted scanner (not something this project authors or mines),
distinct from ``ARGUS_RULES_DIR``'s custom/mined rules and from semgrep
registry packs (``ARGUS_STOCK_SEMGREP_PACKS``) -- see docs/PRECHECKS.md's
"stock rule sources" section for how the three relate.

Mirrors ``argus.precheck.engine``'s semgrep runner's shape and fail-open
contract deliberately (same ``None`` vs ``[]`` distinction, same never-
raises guarantee) so ``run_precheck`` can treat both scanners identically.
Kept as its own module rather than folded into ``engine.py`` to avoid a
circular import: ``engine.run_precheck`` calls into this module, so this
module must not import back from ``engine``. Subprocess plumbing (spawn,
bounded wait, kill+reap) is shared with the other stock-scanner modules
via ``scanner_utils`` rather than ``engine._kill_and_reap`` -- that
function's extra complexity exists specifically to handle semgrep-core, a
grandchild process semgrep spawns that can hold pipe file descriptors open
after the direct child is killed; zizmor (and every other tool
``scanner_utils`` serves) is a single compiled binary with no documented
subprocess fan-out, so it doesn't need the same bounded-drain-plus-
transport-close dance.
"""

from __future__ import annotations

import logging
import os
import shutil

from argus.precheck.sarif import SarifResult, parse_semgrep_sarif
from argus.precheck.scanner_utils import is_success_exit, run_scanner_subprocess

logger = logging.getLogger(__name__)

_ZIZMOR_TIMEOUT_S = 60

# Empirically verified against zizmor 1.29.0 (see this repo's
# pyproject.toml `prechecks` extra for the pinned version range): a
# worktree with nothing zizmor can audit (no workflows/actions files, or
# all gitignored) exits 3 with this exact substring in stderr. This is
# the COMMON case (most repos/PRs never touch .github/workflows) and must
# be treated as "ran, zero findings" ([]), not a scan failure (None) --
# re-verify this string on a zizmor version bump, same as this project
# already re-verifies semgrep's own empirically-observed CLI contract.
_NO_INPUTS_MARKER = "no inputs collected"


def zizmor_available() -> bool:
    """True if the ``prechecks`` extra's zizmor binary is on PATH."""
    return shutil.which("zizmor") is not None


async def run_zizmor_sarif(worktree_path: str) -> list[SarifResult] | None:
    """Run zizmor against ``worktree_path``; return stripped, unclassified
    SARIF results.

    Scans the whole worktree root (not just ``.github/workflows``): zizmor
    auto-discovers workflows/actions itself (honoring ``.gitignore``, same
    as semgrep's own directory-config behavior) and -- verified
    empirically -- reports ``artifactLocation.uri`` already relative to
    whatever root it was pointed at, so scanning the worktree root directly
    yields paths already in the same worktree-relative form every other
    ``SarifResult`` in this pipeline uses, with no path-stripping needed
    (contrast ``engine.run_semgrep_sarif``, which strips an absolute path
    down to worktree-relative because semgrep echoes back the scan-root
    form verbatim).

    ``--offline``: zizmor's own default in the absence of a token, made
    explicit here rather than relied upon -- this scanner runs on the live
    per-PR gate path and must not depend on ``GITHUB_TOKEN_RO`` (or any
    token) to produce a result, and must not make outbound calls to a
    third-party repo's own GitHub API on this codebase's behalf.

    Returns ``None`` -- distinct from ``[]`` -- when zizmor did not
    actually run to completion (missing binary, timeout, execution error),
    matching ``run_semgrep_sarif``'s contract so ``run_precheck`` can treat
    both scanners' results identically.
    """
    if not zizmor_available():
        logger.info("zizmor not on PATH (argus[prechecks] extra not installed) — skipping scan")
        return None

    try:
        return await _run_zizmor_sarif_unguarded(worktree_path)
    except Exception:  # noqa: BLE001 -- enforce the documented never-raises contract
        logger.warning(
            "zizmor subprocess/SARIF-parsing failed unexpectedly scanning %s",
            worktree_path,
            exc_info=True,
        )
        return None


async def _run_zizmor_sarif_unguarded(worktree_path: str) -> list[SarifResult] | None:
    """The actual subprocess-and-parse work for :func:`run_zizmor_sarif`.

    Split out so the outer function can wrap this whole body in one
    try/except, matching ``engine._run_semgrep_sarif_unguarded``'s split.
    """
    if not os.path.isabs(worktree_path):
        raise ValueError(f"worktree_path must be absolute, got {worktree_path!r}")

    outcome = await run_scanner_subprocess(
        ["zizmor", "--offline", "--format", "sarif", "--no-progress", worktree_path],
        timeout=_ZIZMOR_TIMEOUT_S,
    )
    if outcome is None:
        return None
    stdout, stderr, returncode = outcome

    if not is_success_exit(returncode):
        stderr_text = stderr.decode(errors="replace")
        if _NO_INPUTS_MARKER in stderr_text:
            # The common case: nothing in this worktree for zizmor to
            # audit (no workflows/actions files). Genuine "ran, zero
            # findings" evidence, not a scan failure.
            return []
        logger.warning(
            "zizmor exited %d scanning %s.\nstderr: %s",
            returncode,
            worktree_path,
            stderr_text[:500],
        )
        return None

    return parse_semgrep_sarif(stdout)
