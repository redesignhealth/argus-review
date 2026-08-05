"""Shared subprocess-running plumbing for precheck scanner modules.

Every scanner module here (semgrep is the one exception -- see below)
follows the same shape: spawn a single subprocess, wait bounded, kill+reap
on timeout/exception, then distinguish a genuine execution failure from
"ran fine, exit code just means something different for this tool" (some
tools, like semgrep/zizmor/Trivy, exit 0 regardless of findings and use
SARIF content alone to carry the verdict; others, like Checkov/squawk,
exit 0 only when clean and a distinct nonzero code when genuine findings
exist -- both are "the scan ran successfully," not a failure). Consolidated
here once near-identical copies started accumulating across scanner
modules.

semgrep's own runner (``engine.py``) is deliberately NOT migrated to use
this: it carries extra semgrep-specific complexity (semgrep-core
grandchild processes that can hold pipe file descriptors open after the
direct child is killed) this shared helper does not, and migrating an
already-shipped, already-tested function here carries more regression
risk than benefit.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

logger = logging.getLogger(__name__)


def is_success_exit(returncode: int, *, findings_exit_code: int | None = None) -> bool:
    """True if ``returncode`` represents a scan that ran to completion
    (with or without findings), given this tool's own exit-code
    convention.

    Most scanners here (semgrep, zizmor, Trivy) exit 0 unconditionally and
    signal findings via SARIF content alone -- the default (leave
    ``findings_exit_code`` unset). Some (Checkov, squawk) use a distinct,
    non-zero exit code to mean "ran fine, found something" -- pass that
    code explicitly for those; only exit codes other than 0 and this one
    indicate a genuine scan failure.
    """
    if returncode == 0:
        return True
    return findings_exit_code is not None and returncode == findings_exit_code


async def kill_and_reap(proc: asyncio.subprocess.Process, *, timeout: float) -> None:
    """Best-effort kill + bounded drain of a still-live scanner subprocess.

    Deliberately simpler than ``engine._kill_and_reap``: every tool this
    helper serves is a single compiled/interpreted binary with no
    documented subprocess fan-out (unlike semgrep, which spawns
    semgrep-core workers that can outlive a killed direct child and hold
    pipe file descriptors open) -- an unbounded-drain-after-kill hang risk
    doesn't apply the same way here. Still bounded, out of caution; any
    exception during cleanup is swallowed, matching this module's overall
    fail-open philosophy (a cleanup failure is never worth surfacing).
    """
    with contextlib.suppress(ProcessLookupError):
        proc.kill()
    with contextlib.suppress(Exception):  # noqa: BLE001 -- cleanup failure is never worth raising
        await asyncio.wait_for(proc.communicate(), timeout=timeout)


async def run_scanner_subprocess(
    argv: list[str], *, timeout: float, cwd: str | None = None
) -> tuple[bytes, bytes, int] | None:
    """Spawn ``argv[0]`` with ``argv[1:]`` as arguments; return
    ``(stdout, stderr, returncode)``, or ``None`` if it didn't run to
    completion (failed to spawn, timed out, or any other exception during
    spawn/communicate -- the child is killed and reaped before returning).

    ``cwd``, when given, runs the subprocess from that directory -- needed
    by scanners (e.g. squawk) that echo back file paths exactly as given
    on the command line rather than resolving/stripping them themselves:
    invoking with ``cwd=worktree_path`` and relative paths keeps every
    scanner's reported paths in the same worktree-relative form, without
    each caller needing its own absolute-path-stripping logic.

    Callers still own: checking their own ``<tool>_available()`` before
    calling this (so a missing binary logs a clear, tool-specific message
    rather than a generic subprocess error), and interpreting the returned
    ``returncode`` via :func:`is_success_exit` (each tool's own
    success/failure convention differs).
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=cwd
        )
    except OSError:
        logger.warning("Failed to spawn %r", argv, exc_info=True)
        return None

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except (asyncio.TimeoutError, TimeoutError):
        await kill_and_reap(proc, timeout=timeout)
        logger.warning("%s timed out after %ds", argv[0], timeout)
        return None
    except BaseException:
        await kill_and_reap(proc, timeout=timeout)
        raise

    if proc.returncode is None:
        # Cannot happen in practice (communicate() only returns after the
        # process has exited and set this), but a bare `assert` here would
        # strip under `python -O`/PYTHONOPTIMIZE -- see this project's
        # other rejections of bare asserts (e.g. engine.py's absolute-path
        # check) for the same reasoning.
        raise RuntimeError(f"{argv[0]} subprocess has no returncode after communicate()")
    return stdout, stderr, proc.returncode
