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
import shutil
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from argus.config import get_settings
from argus.precheck.sarif import SarifResult, parse_semgrep_sarif
from argus.storage.precheck import select_rule_statuses

logger = logging.getLogger(__name__)

_SEMGREP_TIMEOUT_S = 120


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
        return path if path.is_dir() else None
    try:
        packaged = resources.files("argus.precheck") / "rules"
        path = Path(str(packaged))
        return path if path.is_dir() else None
    except (ModuleNotFoundError, FileNotFoundError):
        return None


def _has_rule_files(rules_dir: Path) -> bool:
    return any(rules_dir.glob("*.yml")) or any(rules_dir.glob("*.yaml"))


def semgrep_available() -> bool:
    """True if the ``prechecks`` extra's semgrep binary is on PATH."""
    return shutil.which("semgrep") is not None


async def run_precheck(worktree_path: str) -> PrecheckResult:
    """Run custom rules against ``worktree_path``; classify hits by DB status.

    The caller (``graph._node_precheck``) owns logging candidate firings
    for later triage and short-circuiting the pipeline on verified
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
