"""Deterministic-precheck rule execution: run Checkov against changed
Terraform files.

Checkov (https://www.checkov.io) is Bridgecrew's IaC misconfiguration
scanner. Scoped here specifically to IAM wildcard-permission and
privilege-escalation checks (``_ALLOWED_CHECKS`` below) -- Checkov's full
default catalog is far broader than that (verified empirically: a plain,
otherwise-fine ``aws_s3_bucket`` resource with no other issues failed 7
unrelated best-practice checks -- versioning, logging, encryption,
lifecycle, replication, public-access-block -- under the default check
set). Running the full catalog would flood every Terraform PR with
opinionated-best-practice noise unrelated to the actual gap this
integration exists for; the ``-c`` allowlist keeps it scoped to that gap.
A stock, externally-vetted scanner, distinct from custom/mined rules --
see docs/PRECHECKS.md's "stock rule sources" section.

Trivy's own misconfiguration scanner deliberately does NOT also cover
Terraform here (see argus/precheck/secrets_scanner.py's module docstring)
to avoid the two tools double-reporting the same conditions on the same
lines.

Like squawk and actionlint, this scanner requires an explicit
``changed_files`` list -- Checkov's default catalog and Terraform's own
frequently-large infrastructure trees make whole-worktree scanning both
slow and flood-prone, the same class of concern as squawk's "years of
migration history."

Checkov's own exit-code convention is the same inversion as squawk's:
exit 0 only when clean, exit 1 when genuine findings exist (verified
empirically) -- NOT the semgrep/zizmor/Trivy convention of exit 0
regardless of findings.

Checkov emits real SARIF (unlike squawk/actionlint) but only to a FILE
(``--output-file-path``), never to stdout -- verified empirically: even
with ``--output sarif`` and no file-path override, it still writes
``results.sarif`` to disk and prints a separate compact terminal summary
to stdout regardless. This module reads the written file back rather than
parsing stdout, the one real deviation from every other scanner module's
shape here.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

from argus.precheck.sarif import SarifResult, parse_semgrep_sarif
from argus.precheck.scanner_utils import is_success_exit, run_scanner_subprocess

logger = logging.getLogger(__name__)

_CHECKOV_TIMEOUT_S = 120

# Scoped to the IAM wildcard/privilege-escalation gap this integration
# exists for (see module docstring) -- NOT Checkov's full default catalog.
# Verified empirically against a synthetic `Action="*"`/`Resource="*"`
# aws_iam_policy: all nine of these fired; a narrower prefix-wildcarded
# resource ARN (e.g. `db:rh-platform-*` on a single specific action) did
# NOT fire on any of them -- that narrower pattern is a policy judgment
# call specific to a naming convention, not something a generic tool
# should flag, and is out of scope here.
_ALLOWED_CHECKS = (
    "CKV_AWS_63",  # no "*" as a statement's actions
    "CKV_AWS_355",  # no "*" as resource for restrictable actions
    "CKV_AWS_62",  # no full "*-*" administrative privileges
    "CKV_AWS_286",  # no privilege escalation
    "CKV_AWS_287",  # no credentials exposure
    "CKV_AWS_288",  # no data exfiltration
    "CKV_AWS_289",  # no permissions management / resource exposure without constraints
    "CKV_AWS_290",  # no write access without constraints
    "CKV2_AWS_40",  # no full IAM privileges
)

_FINDINGS_EXIT_CODE = 1


def checkov_available() -> bool:
    """True if the ``prechecks`` extra's checkov binary is on PATH."""
    return shutil.which("checkov") is not None


async def run_checkov_sarif(
    worktree_path: str, changed_files: list[str]
) -> list[SarifResult] | None:
    """Run Checkov against the changed ``.tf``/``.tf.json`` files in
    ``changed_files`` that still exist on disk; return unclassified
    results.

    Deliberately takes ``changed_files`` as a required argument -- see
    this module's docstring for why. ``run_precheck`` skips calling this
    entirely when it has no ``changed_files`` list.

    Returns ``[]`` without spawning a subprocess when none of
    ``changed_files`` are Terraform files -- the common case for most PRs.

    Returns ``None`` -- distinct from ``[]`` -- when Checkov did not
    actually run to completion (missing binary, timeout, execution
    error), matching every other scanner's contract in this package.
    """
    if not checkov_available():
        logger.info("checkov not on PATH (argus[prechecks] extra not installed) — skipping scan")
        return None

    tf_files = [
        f
        for f in changed_files
        if f.endswith((".tf", ".tf.json"))
        and os.path.isfile(os.path.join(worktree_path, f))
    ]
    if not tf_files:
        return []

    try:
        return await _run_checkov_sarif_unguarded(worktree_path, tf_files)
    except Exception:  # noqa: BLE001 -- enforce the documented never-raises contract
        logger.warning(
            "checkov subprocess/output-parsing failed unexpectedly scanning %s",
            worktree_path,
            exc_info=True,
        )
        return None


async def _run_checkov_sarif_unguarded(
    worktree_path: str, tf_files: list[str]
) -> list[SarifResult] | None:
    """The actual subprocess-and-parse work for :func:`run_checkov_sarif`.

    Invoked with ``cwd=worktree_path`` and relative file paths, same as
    squawk/actionlint -- verified empirically that Checkov then echoes
    back the same relative form in its SARIF output.
    """
    if not os.path.isabs(worktree_path):
        raise ValueError(f"worktree_path must be absolute, got {worktree_path!r}")

    with tempfile.TemporaryDirectory() as output_dir:
        argv = [
            "checkov",
            "-f",
            *tf_files,
            "--framework",
            "terraform",
            "-c",
            ",".join(_ALLOWED_CHECKS),
            "--output",
            "sarif",
            "--output-file-path",
            output_dir,
            "--quiet",
            "--compact",
        ]

        outcome = await run_scanner_subprocess(argv, cwd=worktree_path, timeout=_CHECKOV_TIMEOUT_S)
        if outcome is None:
            return None
        _, stderr, returncode = outcome

        if not is_success_exit(returncode, findings_exit_code=_FINDINGS_EXIT_CODE):
            logger.warning(
                "checkov exited %d scanning %s.\nstderr: %s",
                returncode,
                worktree_path,
                stderr.decode(errors="replace")[:500],
            )
            return None

        sarif_path = Path(output_dir) / "results_sarif.sarif"
        if not sarif_path.is_file():
            logger.warning(
                "checkov exited %d but did not write %s scanning %s",
                returncode,
                sarif_path,
                worktree_path,
            )
            return None

        results = parse_semgrep_sarif(sarif_path.read_bytes())

    # Namespaced for consistency with squawk/actionlint even though
    # Checkov's own "CKV_AWS_..." ids are already fairly distinctive on
    # their own (unlike squawk's bare "prefer-robust-stmts"-style ids).
    return [replace(r, rule_id=f"checkov/{r.rule_id}") for r in results]
