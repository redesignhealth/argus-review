"""Deterministic-precheck rule execution: run eslint-plugin-security
against changed JS/TS files.

Unlike every other stock scanner in this package, this one uses a
**bundled** config + plugin (``argus/precheck/eslint_bundle/``) rather than
a standalone binary -- see that directory's README.md for why (scanning
with the *reviewed* repo's own eslint config would depend on that repo
already having a security plugin configured, the exact gap this
integration fills) and its one-time ``npm install`` setup step. A stock,
externally-vetted scanner, distinct from custom/mined rules -- see
docs/PRECHECKS.md's "stock rule sources" section.

Like squawk/actionlint/checkov, this scanner requires an explicit
``changed_files`` list rather than scanning the whole worktree -- eslint
has no natural "scan everything relevant" mode the way semgrep/zizmor do
without a project's own config telling it what to include/exclude, and a
JS/TS monorepo's ``node_modules``/build output make whole-tree scanning
both slow and noisy.

eslint's own JSON output always resolves to absolute file paths regardless
of cwd or how the path was given on the command line (verified
empirically -- unlike squawk/actionlint/Trivy, which all echo back
whatever relative form they were given) -- this module strips the
worktree-path prefix itself, the same class of adaptation
``engine.run_semgrep_sarif`` already does for semgrep.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from argus.precheck.sarif import SarifResult
from argus.precheck.scanner_utils import is_success_exit, run_scanner_subprocess

logger = logging.getLogger(__name__)

_ESLINT_TIMEOUT_S = 60

_BUNDLE_DIR = Path(__file__).parent / "eslint_bundle"
_ESLINT_BIN = _BUNDLE_DIR / "node_modules" / ".bin" / "eslint"
_CONFIG_PATH = _BUNDLE_DIR / "eslint.config.js"

_JS_EXTENSIONS = (".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx")

# eslint-plugin-security's own "recommended" config uses "warn" severity
# for every rule (verified empirically), not "error" -- which is what
# keeps eslint's own exit code at 0 regardless of findings (only
# errorCount, not warningCount, affects it), matching the semgrep/zizmor/
# Trivy convention rather than needing a findings_exit_code special case
# the way squawk/actionlint/checkov do. If eslint_bundle's config is ever
# changed to elevate any rule to "error", this module's exit-code handling
# would need is_success_exit(..., findings_exit_code=1) added -- re-verify
# together with any config change.


def eslint_available() -> bool:
    """True if the bundled eslint install (see eslint_bundle/README.md's
    one-time ``npm install`` setup step) is present.
    """
    return _ESLINT_BIN.is_file()


async def run_eslint_sarif(
    worktree_path: str, changed_files: list[str]
) -> list[SarifResult] | None:
    """Run the bundled eslint + eslint-plugin-security against the changed
    JS/TS files in ``changed_files`` that still exist on disk; return
    unclassified results.

    Deliberately takes ``changed_files`` as a required argument -- see
    this module's docstring for why. ``run_precheck`` skips calling this
    entirely when it has no ``changed_files`` list.

    Returns ``[]`` without spawning a subprocess when none of
    ``changed_files`` are JS/TS files -- the common case for a
    Python-primary repo, or any PR that doesn't touch JS/TS at all.

    Returns ``None`` -- distinct from ``[]`` -- when eslint did not
    actually run to completion (bundle not installed, timeout, execution
    error), matching every other scanner's contract in this package.
    """
    if not eslint_available():
        logger.info(
            "eslint security bundle not installed (see argus/precheck/eslint_bundle/"
            "README.md's one-time `npm install` step) — skipping scan"
        )
        return None

    js_files = [
        f
        for f in changed_files
        if f.endswith(_JS_EXTENSIONS) and os.path.isfile(os.path.join(worktree_path, f))
    ]
    if not js_files:
        return []

    try:
        return await _run_eslint_sarif_unguarded(worktree_path, js_files)
    except Exception:  # noqa: BLE001 -- enforce the documented never-raises contract
        logger.warning(
            "eslint subprocess/output-parsing failed unexpectedly scanning %s",
            worktree_path,
            exc_info=True,
        )
        return None


async def _run_eslint_sarif_unguarded(
    worktree_path: str, js_files: list[str]
) -> list[SarifResult] | None:
    """The actual subprocess-and-parse work for :func:`run_eslint_sarif`."""
    if not os.path.isabs(worktree_path):
        raise ValueError(f"worktree_path must be absolute, got {worktree_path!r}")

    argv = [
        str(_ESLINT_BIN),
        "--config",
        str(_CONFIG_PATH),
        "--format",
        "json",
        "--no-config-lookup",  # never merge the target repo's own eslint config
        # "--" guards against a dash-prefixed changed-file path being parsed
        # as a flag -- see scanner_utils.py's "Dash-prefixed-filename
        # argument injection" section for the full rationale (shared across
        # all four per-file scanners). Confirmed "--" preserves the exact
        # path in eslint's own JSON echo, so the prefix-stripping logic
        # below is unaffected.
        "--",
        *js_files,
    ]

    outcome = await run_scanner_subprocess(argv, cwd=worktree_path, timeout=_ESLINT_TIMEOUT_S)
    if outcome is None:
        return None
    stdout, stderr, returncode = outcome

    if not is_success_exit(returncode):
        logger.warning(
            "eslint exited %d scanning %s.\nstderr: %s",
            returncode,
            worktree_path,
            stderr.decode(errors="replace")[:500],
        )
        return None

    try:
        raw_files = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        # None, not [] -- this is exactly eslint's own dash-prefixed-
        # filename argv-injection failure mode (see scanner_utils.py's
        # "Dash-prefixed-filename argument injection" section): eslint
        # exits 0 but prints a plain-text error instead of JSON, so
        # is_success_exit already returned True above, and it's this
        # except clause's job to still distinguish "didn't really run"
        # from "ran clean, zero findings" -- [] would misreport the former
        # as the latter. NOT actually parallel to Checkov's parse path
        # (terraform_scanner.py's SARIF parsing goes through
        # sarif.parse_semgrep_sarif, which returns [] on a parse failure,
        # not None) -- eslint is simply the first scanner in this package
        # to make this specific distinction; squawk/actionlint's identical
        # except clauses still return [] here too (see migration_scanner.py
        # and workflow_lint_scanner.py), a known inconsistency, not a
        # package-wide convention yet.
        logger.warning("Could not parse eslint JSON output", exc_info=True)
        return None

    # Two candidate prefixes, not one: verified empirically (the same
    # macOS /tmp -> /private/tmp symlink-resolution case
    # engine.run_semgrep_sarif's own stripping already anticipates) that
    # eslint can report a realpath-resolved absolute path even when given
    # a non-canonicalized worktree_path. Longest-first so a prefix that
    # happens to be a strict prefix of the other always matches its
    # fullest form first.
    prefixes = sorted(
        {worktree_path.rstrip("/") + "/", os.path.realpath(worktree_path).rstrip("/") + "/"},
        key=len,
        reverse=True,
    )

    def _strip_prefix(file_path: str) -> str:
        for prefix in prefixes:
            if file_path.startswith(prefix):
                return file_path[len(prefix) :]
        return file_path

    results = []
    for file_result in raw_files:
        file_path = file_result.get("filePath", "")
        relative_path = _strip_prefix(file_path)
        for msg in file_result.get("messages", []):
            rule_id = msg.get("ruleId")
            if not rule_id:
                continue
            results.append(
                SarifResult(
                    # eslint-plugin-security's own rule ids are already
                    # namespaced by its plugin prefix ("security/detect-
                    # child-process") -- no additional prefixing needed,
                    # unlike squawk/Trivy's bare generic ids.
                    rule_id=rule_id,
                    level="warning" if msg.get("severity") == 1 else "error",
                    message=msg.get("message", ""),
                    file=relative_path,
                    line=msg.get("line"),
                )
            )
    return results
