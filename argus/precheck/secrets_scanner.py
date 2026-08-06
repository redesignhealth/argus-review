"""Deterministic-precheck rule execution: run Trivy's secret scanner
against the worktree.

Trivy (https://trivy.dev, Aqua Security) is a broad scanner covering
secrets, IaC misconfiguration, vulnerabilities, and more -- used here for
secrets ONLY (``--scanners secret``). Two reasons the other scan types are
deliberately excluded:

- Misconfiguration: the specific check this integration was originally
  wanted for (generic Terraform IAM wildcard-resource detection) turned
  out to be a dead/deprecated rule in Trivy's own default bundle (empty
  rule body, silently skipped -- verified empirically). Checkov actually
  catches that gap (see terraform_scanner.py) and is scoped to it;
  running Trivy's misconfig scanner too would double-report several
  IAM/S3 conditions Checkov already covers, on the same lines.
- Vulnerabilities (SCA): out of scope for this integration for now.

Secrets specifically are a genuine improvement over semgrep's own
``p/secrets`` registry pack: verified empirically that the free/
unauthenticated tier of that pack only loads ~37 generic rules (a
``semgrep login`` session unlocks a much larger set this project doesn't
require), while Trivy ships dozens of built-in vendor-specific regexes
(AWS keys, GitHub tokens, Stripe keys, etc.) with no login or download
step. It has its own blind spot -- well-known placeholder/example keys
(e.g. AWS's own docs example key) are deliberately allowlisted to cut
false positives from docs/tests, verified empirically.

Unlike squawk/actionlint/checkov, this scanner does NOT require an
explicit ``changed_files`` list -- like zizmor, it scans the whole
worktree and relies on ``run_precheck``'s own post-merge diff-scoping
filter (see engine.py) rather than pre-filtering its own input, since
secrets scanning doesn't carry the same "years of history"/"broad default
catalog" flood risk squawk/Checkov do, and (verified empirically) Trivy
already reports paths relative to the scan root when given an absolute
worktree path directly -- no ``cwd``/relative-path trick needed, matching
zizmor's own reporting.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import replace

from argus.precheck.sarif import SarifResult, parse_semgrep_sarif
from argus.precheck.scanner_utils import is_success_exit, run_scanner_subprocess

logger = logging.getLogger(__name__)

_TRIVY_TIMEOUT_S = 120


def trivy_available() -> bool:
    """True if the ``prechecks`` extra's trivy binary is on PATH."""
    return shutil.which("trivy") is not None


async def run_trivy_secrets_sarif(worktree_path: str) -> list[SarifResult] | None:
    """Run Trivy's secret scanner against ``worktree_path``; return
    unclassified results.

    Returns ``None`` -- distinct from ``[]`` -- when Trivy did not
    actually run to completion (missing binary, timeout, execution
    error), matching every other scanner's contract in this package.
    """
    if not trivy_available():
        logger.info("trivy not on PATH (argus[prechecks] extra not installed) — skipping scan")
        return None

    try:
        return await _run_trivy_secrets_sarif_unguarded(worktree_path)
    except Exception:  # noqa: BLE001 -- enforce the documented never-raises contract
        logger.warning(
            "trivy subprocess/SARIF-parsing failed unexpectedly scanning %s",
            worktree_path,
            exc_info=True,
        )
        return None


async def _run_trivy_secrets_sarif_unguarded(worktree_path: str) -> list[SarifResult] | None:
    """The actual subprocess-and-parse work for :func:`run_trivy_secrets_sarif`."""
    if not os.path.isabs(worktree_path):
        raise ValueError(f"worktree_path must be absolute, got {worktree_path!r}")

    outcome = await run_scanner_subprocess(
        ["trivy", "fs", "--scanners", "secret", "--format", "sarif", "--quiet", worktree_path],
        timeout=_TRIVY_TIMEOUT_S,
    )
    if outcome is None:
        return None
    stdout, stderr, returncode = outcome

    if not is_success_exit(returncode):
        logger.warning(
            "trivy exited %d scanning %s.\nstderr: %s",
            returncode,
            worktree_path,
            stderr.decode(errors="replace")[:500],
        )
        return None

    results = parse_semgrep_sarif(stdout)
    # Namespaced: Trivy's own secret-rule ids are bare, generic strings
    # ("github-pat", "aws-access-key-id") with real collision risk against
    # other stock scanners' equally generic ids (same reasoning as
    # squawk's own namespacing -- see migration_scanner.py).
    return [replace(r, rule_id=f"trivy/{r.rule_id}") for r in results]
