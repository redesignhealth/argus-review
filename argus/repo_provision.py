"""SHA-pinned git worktree provisioning for Argus file exploration.

Creates a bare mirror of the target repo (cached at
``$ARGUS_REPO_CACHE/<owner>/<repo>.git`` or
``~/.cache/argus/<owner>/<repo>.git``) and checks out a specific
``head_sha`` into a temporary worktree.

The worktree path is passed to ``ClaudeSDKClient`` as its ``cwd`` so
all Read/Glob/Grep calls resolve against the exact commit under review,
regardless of what branch the ambient checkout is on or whether Argus
is installed as a wheel into a venv.

Usage::

    async with provisioned_worktree(repo="owner/repo", head_sha="abc123...") as worktree_path:
        # worktree_path is a str; ClaudeSDKClient cwd=worktree_path
        ...
    # worktree removed on exit; mirror kept for reuse

Security:
- ``repo`` must match ``<word>/<word>`` (dots/hyphens allowed).
- ``head_sha`` must be exactly 40 lowercase hex digits.
- Subprocess calls use argument lists (no shell string interpolation).
- ``GITHUB_TOKEN_RO`` is used for HTTPS clone; never logged.
- Token auth is written to a ``0600`` ``GIT_CONFIG_GLOBAL`` tempfile (never
  argv) and unlinked when the git subprocess returns. Caveat: a SIGKILL/OOM
  kill bypasses that cleanup, so on a shared host a stale ``argus-gitcfg-*``
  file (mode 0600, same UID) can linger. On ECS Fargate ``/tmp`` is per-task
  ephemeral, which bounds the exposure there.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import os
import re
import shutil
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

# Default per-command git timeout (seconds). Network clones get a longer
# budget; everything else (fetch, cat-file, rev-parse) is local-ish.
_GIT_DEFAULT_TIMEOUT = 60.0
_GIT_CLONE_TIMEOUT = 300.0

# Marker file written inside a mirror only after its clone fully completes and
# validates. Lets a racing peer distinguish a finished mirror from a partial
# or in-progress one (git creates the dir and HEAD before the clone finishes).
_CLONE_COMPLETE_SENTINEL = "argus-clone-complete"

# Strict 40-hex SHA validation — only full SHAs, never short refs,
# so we can use it as a literal git ref without ambiguity.
_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

# owner/repo — letters, digits, dots, hyphens, underscores; tightened to
# reject leading dots/hyphens in both segments (common path-traversal
# tricks; a leading dot in the name also yields invalid clone URLs).
_REPO_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]*/[A-Za-z0-9_][A-Za-z0-9._-]*$")

# Matches embedded GitHub tokens in URLs for redaction.
_TOKEN_RE = re.compile(r"x-access-token:[^@]+@")
# Matches HTTP Authorization: Basic headers (base64-encoded token) for redaction.
_AUTH_HEADER_RE = re.compile(r"Authorization: Basic \S+")


def _validate_repo(repo: str) -> None:
    """Raise ValueError if ``repo`` does not match ``owner/repo`` format."""
    if not _REPO_RE.match(repo):
        raise ValueError(f"Invalid repo format: {repo!r}")
    if ".." in repo:
        raise ValueError(f"Invalid repo format (path traversal): {repo!r}")


def _validate_sha(sha: str) -> None:
    """Raise ValueError if ``sha`` is not exactly 40 lowercase hex digits."""
    if not _FULL_SHA_RE.match(sha):
        raise ValueError(
            f"head_sha must be exactly 40 lowercase hex chars; got: {sha!r}. "
            "Pass the full SHA from the GitHub PR API, not a short ref."
        )


def _make_auth_header(token: str) -> str:
    """Return a Basic auth header value for use with git http.extraHeader.

    Encodes ``x-access-token:{token}`` as Base64 per RFC 7617.
    """
    return "Basic " + base64.b64encode(f"x-access-token:{token}".encode()).decode()


def _redact(s: str) -> str:
    """Strip any embedded token or auth header from a string before logging."""
    s = _TOKEN_RE.sub("x-access-token:***@", s)
    s = _AUTH_HEADER_RE.sub("Authorization: Basic ***", s)
    return s


def _mirror_path(repo: str) -> Path:
    """Return the stable path for the bare mirror of ``repo``.

    Respects ``$ARGUS_REPO_CACHE`` env var; falls back to
    ``~/.cache/argus/<owner>/<repo>.git``.
    """
    cache_base = os.environ.get("ARGUS_REPO_CACHE")
    if cache_base:
        base = Path(cache_base)
    else:
        base = Path.home() / ".cache" / "argus"
    owner, name = repo.split("/", 1)
    return base / owner / f"{name}.git"


async def _run_git(
    *args: str,
    check: bool = True,
    timeout: float = _GIT_DEFAULT_TIMEOUT,
    auth_token: str | None = None,
) -> tuple[asyncio.subprocess.Process, bytes, bytes]:
    """Run a git command, returning ``(process, stdout, stderr)``.

    All arguments are passed as a list (no shell interpolation).
    ``check=True`` raises ``RuntimeError`` on non-zero exit. Returning the
    captured streams lets callers that need output (e.g. rev-parse HEAD)
    reuse this helper instead of duplicating the kill/drain/timeout logic.

    ``timeout`` bounds ``communicate()`` so a hung network operation cannot
    block the event loop for the full flow-level timeout; the subprocess is
    killed and a ``RuntimeError`` raised on expiry.

    When ``auth_token`` is provided, the ``http.extraHeader`` Authorization
    header is supplied via a ``0600`` git config file referenced through
    ``GIT_CONFIG_GLOBAL`` (NOT via ``-c`` on the command line). This keeps the
    token out of ``/proc/<pid>/cmdline`` (and out of the process environment),
    where it would otherwise be readable by any same-user process for the
    lifetime of the subprocess. The tempfile is created inside the ``try`` so
    the ``finally`` always unlinks it, even if writing the header raises.
    """
    cmd = ["git", *args]
    tmp_cfg: str | None = None
    try:
        env: dict[str, str] | None = None
        if auth_token is not None:
            fd, tmp_cfg = tempfile.mkstemp(prefix="argus-gitcfg-")  # mode 0600
            try:
                header = _make_auth_header(auth_token)
                os.write(fd, f"[http]\n\textraHeader = Authorization: {header}\n".encode())
            finally:
                os.close(fd)
            # Override only the global config file; system config still applies.
            env = {**os.environ, "GIT_CONFIG_GLOBAL": tmp_cfg, "GIT_CONFIG_SYSTEM": os.devnull}

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError):
            proc.kill()
            with contextlib.suppress(Exception):
                await proc.communicate()
            raise RuntimeError(
                f"git command timed out after {timeout:.0f}s: {[_redact(a) for a in cmd]!r}"
            ) from None
    finally:
        if tmp_cfg is not None:
            with contextlib.suppress(OSError):
                os.unlink(tmp_cfg)

    if check and proc.returncode != 0:
        safe_cmd = [_redact(a) for a in cmd]
        safe_stderr = _redact(stderr.decode(errors="replace"))
        raise RuntimeError(
            f"git command failed (exit {proc.returncode}): {safe_cmd!r}\n"
            f"stderr: {safe_stderr[:500]}"
        )
    return proc, stdout, stderr


async def _validate_mirror(mirror: Path) -> bool:
    """Return True only if ``mirror`` is a fully-cloned, usable bare repo.

    Requires BOTH the clone-complete sentinel AND a valid git dir. ``git
    clone`` creates the destination dir and HEAD early, so ``rev-parse``
    alone passes on a partial or still-in-progress clone; the sentinel
    (written only after a clone fully completes) closes that gap.
    """
    if not await asyncio.to_thread((mirror / _CLONE_COMPLETE_SENTINEL).exists):
        return False
    try:
        await _run_git("-C", str(mirror), "rev-parse", "--git-dir")
    except RuntimeError:
        return False
    return True


async def _ensure_mirror(mirror: Path, repo: str, token: str) -> None:
    """Clone a bare mirror if absent; otherwise validate and reuse it.

    The mirror is created with ``--mirror`` so all refs are fetched. Auth is
    supplied via a 0600 ``GIT_CONFIG_GLOBAL`` file (see ``_run_git``), not on
    the command line, so the token never appears in argv or the remote URL.

    Handles the concurrent first-run clone race: two callers can both see the
    mirror absent and race ``git clone --mirror`` to the same path. If the
    clone fails but a racing peer has produced a fully-validated mirror (git
    dir + completion sentinel), treat it as success; otherwise re-raise so a
    partial/in-progress clone is never returned as ready.
    """
    if await asyncio.to_thread(mirror.exists):
        # Reuse if it's a real bare repo. (rev-parse alone is sufficient here:
        # a complete mirror from a prior run validates; the sentinel matters
        # only for distinguishing a racing peer's partial clone, below.)
        try:
            await _run_git("-C", str(mirror), "rev-parse", "--git-dir")
        except RuntimeError:
            logger.warning("Mirror at %s is not a valid bare repo; re-cloning", mirror)
            await asyncio.to_thread(shutil.rmtree, mirror, ignore_errors=True)
            if await asyncio.to_thread(mirror.exists):
                logger.warning(
                    "Could not fully remove invalid mirror at %s before re-clone", mirror
                )
            # fall through to clone
        else:
            logger.debug("Reusing existing mirror at %s", mirror)
            return

    await asyncio.to_thread(mirror.parent.mkdir, parents=True, exist_ok=True)
    clone_url = f"https://github.com/{repo}.git"
    logger.info("Cloning bare mirror for %s into %s", repo, mirror)
    try:
        await _run_git(
            "clone",
            "--mirror",
            clone_url,
            str(mirror),
            timeout=_GIT_CLONE_TIMEOUT,
            auth_token=token,
        )
    except RuntimeError:
        # A concurrent caller may have won the clone race. Trust the leftover
        # mirror only if it fully validates (git dir + completion sentinel);
        # a still-running or killed peer clone fails that check, so we
        # re-raise rather than return an unusable mirror.
        if await _validate_mirror(mirror):
            logger.info("Mirror already created by concurrent process at %s", mirror)
            return
        raise
    # Mark the clone complete so future runs and racing peers can trust it.
    await asyncio.to_thread((mirror / _CLONE_COMPLETE_SENTINEL).touch)


async def _fetch_sha_into_mirror(mirror: Path, sha: str, token: str) -> None:
    """Fetch the specific commit SHA into the mirror.

    ``git fetch origin <sha>`` is needed when the SHA is not yet in the
    mirror (e.g. very new commits or first run after a shallow clone).
    We fetch into the mirror's object store; no ref update required.

    Auth is supplied via a 0600 ``GIT_CONFIG_GLOBAL`` file (see ``_run_git``),
    not on the command line, so the token never appears in argv.
    """
    logger.debug("Fetching %s into mirror %s", sha[:12], mirror)
    # ``--update-head-ok`` suppresses the "refusing to fetch into current branch"
    # warning that can appear on bare repos; harmless but avoids noise.
    # Auth is supplied via a 0600 config file (see _run_git), not argv.
    await _run_git(
        "-C",
        str(mirror),
        "fetch",
        "origin",
        sha,
        "--update-head-ok",
        check=False,  # fetch exits non-zero when SHA is already present on some git versions
        auth_token=token,
    )
    # Confirm the object exists regardless of fetch exit code
    proc, _stdout, _stderr = await _run_git(
        "-C",
        str(mirror),
        "cat-file",
        "-e",
        sha,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"SHA {sha[:12]} not found in mirror {mirror} after fetch attempt. "
            "The commit may not be reachable from any ref (force-push removed it)."
        )


async def _add_worktree(mirror: Path, sha: str, worktree_dir: str) -> None:
    """Create a detached worktree at ``sha`` inside ``worktree_dir``."""
    logger.debug("Adding worktree at %s for SHA %s", worktree_dir, sha[:12])
    await _run_git(
        "-C",
        str(mirror),
        "worktree",
        "add",
        "--detach",
        worktree_dir,
        sha,
    )


async def _assert_worktree_sha(worktree_dir: str, expected_sha: str) -> None:
    """Fail closed if HEAD in the worktree does not match ``expected_sha``.

    Reuses ``_run_git`` (which owns the timeout/kill/drain logic) and reads
    the captured stdout it returns, rather than re-spawning the subprocess.
    """
    proc, stdout, stderr = await _run_git("-C", worktree_dir, "rev-parse", "HEAD", check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"rev-parse HEAD failed in worktree {worktree_dir}: "
            f"{stderr.decode(errors='replace')[:300]}"
        )
    actual_sha = stdout.decode().strip()
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"Worktree SHA mismatch: expected {expected_sha!r}, "
            f"got {actual_sha!r}. Refusing to proceed (fail closed)."
        )
    logger.info("Worktree SHA verified: %s at %s", expected_sha[:12], worktree_dir)


async def _remove_worktree(mirror: Path, worktree_dir: str) -> None:
    """Remove the worktree from the mirror's worktree list."""
    logger.debug("Removing worktree %s from mirror %s", worktree_dir, mirror)
    await _run_git(
        "-C",
        str(mirror),
        "worktree",
        "remove",
        "--force",
        worktree_dir,
        check=False,  # best-effort; tmpdir cleanup handles the files
    )
    # Prune stale entries in case the above partially failed
    await _run_git(
        "-C",
        str(mirror),
        "worktree",
        "prune",
        check=False,
    )


async def provision_worktree(*, repo: str, head_sha: str, token: str) -> str:
    """Provision a SHA-pinned worktree, returning its path.

    Caller is responsible for calling teardown_worktree() when done.
    Validates repo and head_sha, ensures mirror, fetches SHA, adds worktree,
    asserts SHA. Raises ValueError on invalid inputs, RuntimeError on failure.
    """
    _validate_repo(repo)
    _validate_sha(head_sha)
    mirror = _mirror_path(repo)
    await _ensure_mirror(mirror, repo, token)
    await _fetch_sha_into_mirror(mirror, head_sha, token)
    # Anchor the worktree inside a private mkdtemp parent (mode 0700, owned by
    # this process) and let ``git worktree add`` create the ``wt`` subdir. This
    # avoids the TOCTOU window of mkdtemp→rmdir→git-recreate, where another
    # process on a shared /tmp could plant a symlink at the freed path.
    tmp_parent = await asyncio.to_thread(tempfile.mkdtemp, prefix="argus-worktree-")
    worktree_path = os.path.join(tmp_parent, "wt")
    try:
        await _add_worktree(mirror, head_sha, worktree_path)
        await _assert_worktree_sha(worktree_path, head_sha)
    except Exception:
        await _remove_worktree(mirror, worktree_path)
        await asyncio.to_thread(shutil.rmtree, tmp_parent, ignore_errors=True)
        raise
    logger.info("Worktree provisioned: repo=%s sha=%s path=%s", repo, head_sha[:12], worktree_path)
    return worktree_path


async def teardown_worktree(repo: str, worktree_path: str) -> None:
    """Remove a worktree created by provision_worktree. Mirror is kept.

    Best-effort: logs but does not raise on cleanup failures.
    Raises ValueError immediately if ``worktree_path`` is not a path this
    module created (fails fast rather than rmtree-ing an unexpected directory).
    """
    # Guard against deleting an unexpected directory: only ever rmtree a path
    # we created here (basename "wt" anchored in an "argus-worktree-*" parent).
    # A malformed worktree_path must fail loudly, never silently widen the blast
    # radius of rmtree. Run this check BEFORE any git cleanup so a bad path is
    # rejected before _remove_worktree runs.
    cleanup_dir = os.path.dirname(worktree_path)
    if os.path.basename(worktree_path) != "wt" or "argus-worktree-" not in os.path.basename(
        cleanup_dir
    ):
        raise ValueError(f"Refusing to clean up unexpected worktree path: {worktree_path!r}")
    mirror = _mirror_path(repo)
    await _remove_worktree(mirror, worktree_path)
    try:
        if await asyncio.to_thread(os.path.exists, cleanup_dir):
            await asyncio.to_thread(shutil.rmtree, cleanup_dir, ignore_errors=True)
    except Exception:  # noqa: BLE001
        logger.warning("Could not clean up worktree directory %s", cleanup_dir, exc_info=True)
    logger.info("Worktree removed: %s", worktree_path)


@asynccontextmanager
async def provisioned_worktree(
    *,
    repo: str,
    head_sha: str,
    token: str,
) -> AsyncIterator[str]:
    """Async context manager: yield a SHA-pinned worktree path.

    Args:
        repo: GitHub repo in ``owner/name`` format.
        head_sha: Full 40-hex commit SHA to check out.
        token: ``GITHUB_TOKEN_RO`` — used for HTTPS clone auth; never logged.

    Yields:
        Absolute path (``str``) to the worktree root.

    Raises:
        ValueError: If ``repo`` or ``head_sha`` fail validation.
        RuntimeError: If SHA mismatch detected after worktree creation.

    Teardown:
        Removes the worktree on exit (``finally``); keeps the bare mirror.
    """
    tmp_dir: str | None = None
    try:
        tmp_dir = await provision_worktree(repo=repo, head_sha=head_sha, token=token)
        yield tmp_dir
    finally:
        if tmp_dir is not None:
            await teardown_worktree(repo, tmp_dir)
