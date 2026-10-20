"""Local bounded git object fallback for missing patch verification (TECH-7768).

When GitHub's compare or pulls files API omits diff patch content for a file
(e.g. large lockfiles like uv.lock or data snapshots), Argus cannot verify via
the API response alone whether the change introduced bench configuration routing
modifications (ARGUS_BENCH_FILE or ARGUS_NO_BENCH_OVERRIDES).

This module inspects the local bare repository mirror and verified worktree
to strictly verify the file's HEAD blob directly from git object storage.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Sequence

from argus.repo_provision import _mirror_path, _validate_repo

logger = logging.getLogger(__name__)

# Resource and validation limits
MAX_MISSING_FILES: Final[int] = 32
MAX_BLOB_SIZE: Final[int] = 8 * 1024 * 1024  # 8 MiB
MAX_TOTAL_SIZE: Final[int] = 32 * 1024 * 1024  # 32 MiB
MAX_RAW_DIFF_BYTES: Final[int] = 1024 * 1024  # 1 MiB
MAX_STDERR_BYTES: Final[int] = 4096  # 4 KiB
PER_COMMAND_TIMEOUT_SECONDS: Final[float] = 15.0
OVERALL_DEADLINE_SECONDS: Final[float] = 60.0
MAX_PATH_LENGTH: Final[int] = 4096
CHUNK_SIZE: Final[int] = 65536  # 64 KiB

_FULL_SHA_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}$")
_ROUTING_LINE_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:ARGUS_BENCH_FILE|ARGUS_NO_BENCH_OVERRIDES)\b"
)

_SUPPORTED_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "added",
        "modified",
        "changed",
        "renamed",
        "copied",
    }
)

VALID_FAIL_REASONS: Final[frozenset[str]] = frozenset(
    {
        "no_worktree",
        "head_untrusted",
        "base_unpinned",
        "mirror_invalid",
        "worktree_mismatch",
        "object_missing",
        "not_ancestor",
        "path_invalid",
        "status_unsupported",
        "status_mismatch",
        "non_regular_file",
        "blob_sha_mismatch",
        "oversize",
        "binary",
        "budget_exceeded",
        "too_many_files",
        "git_timeout",
        "git_error",
        "parse_error",
    }
)


@dataclass(frozen=True)
class LocalScanResult:
    """Result of local git object inspection for a single missing-patch file."""

    verified: bool
    routing_lines: tuple[str, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.verified:
            if self.reason is not None:
                raise ValueError(f"verified result cannot have failure reason: {self.reason!r}")
        else:
            if self.reason not in VALID_FAIL_REASONS:
                raise ValueError(f"invalid failure reason: {self.reason!r}")
            if self.routing_lines:
                raise ValueError("unverified result cannot have routing_lines")


class _OversizeError(Exception):
    pass


class _TimeoutError(Exception):
    pass


def _build_git_env(mirror: Path) -> dict[str, str]:
    """Construct an isolated environment for git subprocesses from scratch."""
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin"),
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_LITERAL_PATHSPECS": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
        "PAGER": "cat",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "HOME": str(mirror.parent.resolve()),
    }


async def _run_bounded_git(
    cmd: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    max_stdout: int,
    timeout: float,
) -> tuple[int, bytes]:
    """Execute a git command bounded by stdout capacity, timeout, and stderr cap.

    Enforces:
    - 64 KiB chunked stdout reading with before-extend capacity check.
    - Concurrent stderr drain capped at MAX_STDERR_BYTES.
    - Immediate process termination and reap on timeout, oversize, cancellation, or error.
    - Propagates asyncio.CancelledError cleanly.
    - Does NOT log stderr or raw contents.
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=env,
    )

    stdout_buf = bytearray()
    stderr_buf = bytearray()

    async def _read_stdout() -> None:
        assert proc.stdout is not None
        while True:
            chunk = await proc.stdout.read(CHUNK_SIZE)
            if not chunk:
                break
            if len(stdout_buf) + len(chunk) > max_stdout:
                raise _OversizeError()
            stdout_buf.extend(chunk)

    async def _drain_stderr() -> None:
        assert proc.stderr is not None
        while True:
            chunk = await proc.stderr.read(MAX_STDERR_BYTES)
            if not chunk:
                break
            if len(stderr_buf) < MAX_STDERR_BYTES:
                needed = MAX_STDERR_BYTES - len(stderr_buf)
                stderr_buf.extend(chunk[:needed])

    stdout_task = asyncio.create_task(_read_stdout())
    stderr_task = asyncio.create_task(_drain_stderr())

    try:
        await asyncio.wait_for(
            asyncio.gather(stdout_task, stderr_task, proc.wait()),
            timeout=timeout,
        )
    except (asyncio.TimeoutError, TimeoutError):
        proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()
        stdout_task.cancel()
        stderr_task.cancel()
        with contextlib.suppress(Exception):
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        raise _TimeoutError() from None
    except _OversizeError:
        proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()
        stdout_task.cancel()
        stderr_task.cancel()
        with contextlib.suppress(Exception):
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        raise
    except asyncio.CancelledError:
        proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()
        stdout_task.cancel()
        stderr_task.cancel()
        with contextlib.suppress(Exception):
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        raise
    except Exception:
        proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()
        stdout_task.cancel()
        stderr_task.cancel()
        with contextlib.suppress(Exception):
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        raise

    assert proc.returncode is not None
    return proc.returncode, bytes(stdout_buf)


def _validate_path(path: Any) -> bool:
    """Validate that path is non-empty, relative, NUL-free, and component-safe."""
    if not isinstance(path, str) or not path:
        return False
    if len(path) > MAX_PATH_LENGTH:
        return False
    if len(path.encode("utf-8", errors="replace")) > MAX_PATH_LENGTH:
        return False
    if path.startswith("/"):
        return False
    if "\x00" in path:
        return False
    parts = path.split("/")
    for part in parts:
        if part == "" or part in (".", "..") or part.lower() == ".git":
            return False
    return True


@dataclass(frozen=True)
class _DiffRawRecord:
    old_mode: str
    new_mode: str
    old_sha: str
    new_sha: str
    status: str


def _parse_diff_raw(raw_bytes: bytes, requested_paths: set[str]) -> dict[str, _DiffRawRecord]:
    """Strictly parse NUL-delimited git diff --raw -z output."""
    if not raw_bytes:
        return {}
    if not raw_bytes.endswith(b"\x00"):
        raise ValueError("truncated diff-raw output")

    tokens = raw_bytes[:-1].split(b"\x00")
    if len(tokens) % 2 != 0:
        raise ValueError("uneven diff-raw tokens")

    records: dict[str, _DiffRawRecord] = {}
    for i in range(0, len(tokens), 2):
        meta_bytes = tokens[i]
        path_bytes = tokens[i + 1]

        try:
            meta_str = meta_bytes.decode("ascii")
            path_str = path_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError("invalid encoding in diff-raw") from None

        if not meta_str.startswith(":"):
            raise ValueError(f"malformed diff-raw prefix: {meta_str!r}")

        parts = meta_str[1:].split(" ")
        if len(parts) != 5:
            raise ValueError(f"malformed diff-raw header: {meta_str!r}")

        old_mode, new_mode, old_sha, new_sha, status = parts
        if len(old_mode) != 6 or not old_mode.isdigit():
            raise ValueError(f"malformed old mode: {old_mode!r}")
        if len(new_mode) != 6 or not new_mode.isdigit():
            raise ValueError(f"malformed new mode: {new_mode!r}")
        if not _FULL_SHA_RE.fullmatch(old_sha):
            raise ValueError(f"malformed old sha: {old_sha!r}")
        if not _FULL_SHA_RE.fullmatch(new_sha):
            raise ValueError(f"malformed new sha: {new_sha!r}")
        if len(status) != 1 or not status.isalpha():
            raise ValueError(f"malformed status: {status!r}")

        if path_str not in requested_paths:
            raise ValueError(f"unrequested path in diff-raw: {path_str!r}")
        if path_str in records:
            raise ValueError(f"duplicate path in diff-raw: {path_str!r}")

        records[path_str] = _DiffRawRecord(
            old_mode=old_mode,
            new_mode=new_mode,
            old_sha=old_sha,
            new_sha=new_sha,
            status=status,
        )
    return records


async def verify_missing_patches_locally(
    *,
    repo: str,
    worktree_path: str | None,
    trusted_head_sha: str | None,
    head_sha: str,
    base_sha: str | None,
    entries: Sequence[dict[str, Any]],
) -> dict[str, LocalScanResult]:
    """Verify missing-patch files directly from local git mirror and worktree.

    Fails closed with specific fixed reason codes on any anomaly.
    Returns mapping from filename to LocalScanResult.
    """
    if not entries:
        return {}

    def _all_fail(reason: str) -> dict[str, LocalScanResult]:
        return {
            str(e.get("filename", "")): LocalScanResult(verified=False, reason=reason)
            for e in entries
        }

    # Cap check: maximum 32 missing files
    if len(entries) > MAX_MISSING_FILES:
        return _all_fail("too_many_files")

    deadline = time.monotonic() + OVERALL_DEADLINE_SECONDS

    # Validate head SHA and trusted head SHA
    if (
        not isinstance(head_sha, str)
        or not _FULL_SHA_RE.fullmatch(head_sha)
        or not isinstance(trusted_head_sha, str)
        or not _FULL_SHA_RE.fullmatch(trusted_head_sha)
        or trusted_head_sha != head_sha
    ):
        return _all_fail("head_untrusted")

    # Validate base SHA
    if (
        not isinstance(base_sha, str)
        or not _FULL_SHA_RE.fullmatch(base_sha)
        or base_sha == head_sha
    ):
        return _all_fail("base_unpinned")

    # Validate worktree path
    if not worktree_path or not isinstance(worktree_path, str) or not worktree_path.strip():
        return _all_fail("no_worktree")

    wt = Path(worktree_path)
    if not wt.is_absolute() or wt.is_symlink():
        return _all_fail("worktree_mismatch")
    if not wt.exists() or not wt.is_dir():
        return _all_fail("no_worktree")

    # Validate repository format and mirror path
    try:
        _validate_repo(repo)
        mirror = _mirror_path(repo)
    except Exception:
        return _all_fail("mirror_invalid")

    if not mirror.exists() or not mirror.is_dir() or mirror.is_symlink():
        return _all_fail("mirror_invalid")

    git_env = _build_git_env(mirror)

    # Verify mirror is a real bare git repository
    try:
        timeout = min(PER_COMMAND_TIMEOUT_SECONDS, max(0.01, deadline - time.monotonic()))
        cmd = [
            "git",
            "--no-replace-objects",
            "-c",
            "core.fsmonitor=false",
            "--git-dir",
            str(mirror),
            "rev-parse",
            "--is-bare-repository",
        ]
        ret, stdout = await _run_bounded_git(
            cmd, cwd=str(mirror), env=git_env, max_stdout=64, timeout=timeout
        )
        if ret != 0 or stdout.strip() != b"true":
            return _all_fail("mirror_invalid")
    except _TimeoutError:
        return _all_fail("git_timeout")
    except Exception:
        return _all_fail("mirror_invalid")

    # Verify worktree common-dir points to expected mirror
    try:
        timeout = min(PER_COMMAND_TIMEOUT_SECONDS, max(0.01, deadline - time.monotonic()))
        cmd = [
            "git",
            "--no-replace-objects",
            "-c",
            "core.fsmonitor=false",
            "-C",
            str(wt),
            "rev-parse",
            "--git-common-dir",
        ]
        ret, stdout = await _run_bounded_git(
            cmd, cwd=str(wt), env=git_env, max_stdout=4096, timeout=timeout
        )
        if ret != 0:
            return _all_fail("worktree_mismatch")

        raw_common = stdout.decode("utf-8", errors="replace").strip()
        common_path = Path(raw_common)
        if not common_path.is_absolute():
            common_path = (wt / common_path).resolve()
        else:
            common_path = common_path.resolve()

        if os.path.realpath(str(common_path)) != os.path.realpath(str(mirror)):
            return _all_fail("worktree_mismatch")
    except _TimeoutError:
        return _all_fail("git_timeout")
    except Exception:
        return _all_fail("worktree_mismatch")

    # Verify worktree HEAD matches head_sha
    try:
        timeout = min(PER_COMMAND_TIMEOUT_SECONDS, max(0.01, deadline - time.monotonic()))
        cmd = [
            "git",
            "--no-replace-objects",
            "-c",
            "core.fsmonitor=false",
            "-C",
            str(wt),
            "rev-parse",
            "HEAD",
        ]
        ret, stdout = await _run_bounded_git(
            cmd, cwd=str(wt), env=git_env, max_stdout=64, timeout=timeout
        )
        if ret != 0 or stdout.strip() != head_sha.encode("ascii"):
            return _all_fail("worktree_mismatch")
    except _TimeoutError:
        return _all_fail("git_timeout")
    except Exception:
        return _all_fail("worktree_mismatch")

    # Verify base and head commit objects exist in mirror
    for sha in (head_sha, base_sha):
        try:
            timeout = min(PER_COMMAND_TIMEOUT_SECONDS, max(0.01, deadline - time.monotonic()))
            cmd = [
                "git",
                "--no-replace-objects",
                "-c",
                "core.fsmonitor=false",
                "--git-dir",
                str(mirror),
                "cat-file",
                "-e",
                f"{sha}^{{commit}}",
            ]
            ret, _ = await _run_bounded_git(
                cmd, cwd=str(mirror), env=git_env, max_stdout=64, timeout=timeout
            )
            if ret != 0:
                return _all_fail("object_missing")
        except _TimeoutError:
            return _all_fail("git_timeout")
        except Exception:
            return _all_fail("object_missing")

    # Verify base is an ancestor of head
    try:
        timeout = min(PER_COMMAND_TIMEOUT_SECONDS, max(0.01, deadline - time.monotonic()))
        cmd = [
            "git",
            "--no-replace-objects",
            "-c",
            "core.fsmonitor=false",
            "--git-dir",
            str(mirror),
            "merge-base",
            "--is-ancestor",
            base_sha,
            head_sha,
        ]
        ret, _ = await _run_bounded_git(
            cmd, cwd=str(mirror), env=git_env, max_stdout=64, timeout=timeout
        )
        if ret != 0:
            return _all_fail("not_ancestor")
    except _TimeoutError:
        return _all_fail("git_timeout")
    except Exception:
        return _all_fail("not_ancestor")

    # Entry-level validation: path & status checks
    results: dict[str, LocalScanResult] = {}
    valid_entries: list[dict[str, Any]] = []
    paths_to_query: list[str] = []

    for entry in entries:
        raw_fn = entry.get("filename")
        if not _validate_path(raw_fn):
            results[str(raw_fn)] = LocalScanResult(verified=False, reason="path_invalid")
            continue
        fn: str = str(raw_fn)

        status = entry.get("status")
        if not isinstance(status, str) or status not in _SUPPORTED_STATUSES:
            results[fn] = LocalScanResult(verified=False, reason="status_unsupported")
            continue

        if status == "renamed":
            raw_prev_fn = entry.get("previous_filename")
            if not raw_prev_fn or not isinstance(raw_prev_fn, str) or raw_prev_fn == fn:
                results[fn] = LocalScanResult(verified=False, reason="status_unsupported")
                continue
            prev_fn: str = raw_prev_fn
            if not _validate_path(prev_fn):
                results[fn] = LocalScanResult(verified=False, reason="path_invalid")
                continue
            paths_to_query.append(prev_fn)

        paths_to_query.append(fn)
        valid_entries.append(entry)

    if not valid_entries:
        return results

    # Batched git diff --raw -z execution
    try:
        if time.monotonic() >= deadline:
            for e in valid_entries:
                results[e["filename"]] = LocalScanResult(verified=False, reason="git_timeout")
            return results

        timeout = min(PER_COMMAND_TIMEOUT_SECONDS, max(0.01, deadline - time.monotonic()))
        # Remove duplicates while preserving order
        unique_paths: list[str] = list(dict.fromkeys(paths_to_query))
        cmd = [
            "git",
            "--no-replace-objects",
            "-c",
            "core.fsmonitor=false",
            "--git-dir",
            str(mirror),
            "diff",
            "--raw",
            "-z",
            "--no-renames",
            "--no-abbrev",
            "--no-ext-diff",
            "--no-textconv",
            "--ignore-submodules=none",
            "--no-color",
            base_sha,
            head_sha,
            "--",
            *unique_paths,
        ]
        ret, diff_out = await _run_bounded_git(
            cmd, cwd=str(mirror), env=git_env, max_stdout=MAX_RAW_DIFF_BYTES, timeout=timeout
        )
        if ret != 0:
            for e in valid_entries:
                results[e["filename"]] = LocalScanResult(verified=False, reason="git_error")
            return results
    except _TimeoutError:
        for e in valid_entries:
            results[e["filename"]] = LocalScanResult(verified=False, reason="git_timeout")
        return results
    except _OversizeError:
        for e in valid_entries:
            results[e["filename"]] = LocalScanResult(verified=False, reason="oversize")
        return results
    except Exception:
        for e in valid_entries:
            results[e["filename"]] = LocalScanResult(verified=False, reason="git_error")
        return results

    # Parse diff records strictly
    try:
        records = _parse_diff_raw(diff_out, set(unique_paths))
    except Exception:
        for e in valid_entries:
            results[e["filename"]] = LocalScanResult(verified=False, reason="parse_error")
        return results

    # Inspect each valid entry against parsed records and inspect HEAD blob
    cumulative_size = 0

    for entry in valid_entries:
        fn = entry["filename"]
        status = entry["status"]

        if time.monotonic() >= deadline:
            results[fn] = LocalScanResult(verified=False, reason="git_timeout")
            continue

        if fn not in records:
            results[fn] = LocalScanResult(verified=False, reason="status_mismatch")
            continue

        rec = records[fn]

        # Mode check: regular files only (100644 or 100755)
        if rec.new_mode not in ("100644", "100755"):
            results[fn] = LocalScanResult(verified=False, reason="non_regular_file")
            continue

        # Status matching checks
        if status in ("added", "copied"):
            if rec.status != "A":
                results[fn] = LocalScanResult(verified=False, reason="status_mismatch")
                continue
        elif status in ("modified", "changed"):
            if rec.status != "M":
                results[fn] = LocalScanResult(verified=False, reason="status_mismatch")
                continue
            if rec.old_mode not in ("100644", "100755"):
                results[fn] = LocalScanResult(verified=False, reason="non_regular_file")
                continue
        elif status == "renamed":
            if rec.status != "A":
                results[fn] = LocalScanResult(verified=False, reason="status_mismatch")
                continue
            prev_fn = entry["previous_filename"]
            if prev_fn not in records:
                results[fn] = LocalScanResult(verified=False, reason="status_mismatch")
                continue
            prev_rec = records[prev_fn]
            if prev_rec.status != "D" or prev_rec.new_mode != "000000":
                results[fn] = LocalScanResult(verified=False, reason="status_mismatch")
                continue
            if prev_rec.old_mode not in ("100644", "100755"):
                results[fn] = LocalScanResult(verified=False, reason="non_regular_file")
                continue

        # Optional GitHub blob SHA check
        gh_sha = entry.get("sha")
        if gh_sha:
            if (
                not isinstance(gh_sha, str)
                or not _FULL_SHA_RE.fullmatch(gh_sha)
                or gh_sha.lower() != rec.new_sha.lower()
            ):
                results[fn] = LocalScanResult(verified=False, reason="blob_sha_mismatch")
                continue

        # Query blob size: git cat-file -s
        try:
            timeout = min(PER_COMMAND_TIMEOUT_SECONDS, max(0.01, deadline - time.monotonic()))
            cmd = [
                "git",
                "--no-replace-objects",
                "-c",
                "core.fsmonitor=false",
                "--git-dir",
                str(mirror),
                "cat-file",
                "-s",
                rec.new_sha,
            ]
            ret, size_out = await _run_bounded_git(
                cmd, cwd=str(mirror), env=git_env, max_stdout=64, timeout=timeout
            )
            if ret != 0:
                results[fn] = LocalScanResult(verified=False, reason="git_error")
                continue
            blob_size = int(size_out.decode("ascii").strip())
        except _TimeoutError:
            results[fn] = LocalScanResult(verified=False, reason="git_timeout")
            continue
        except Exception:
            results[fn] = LocalScanResult(verified=False, reason="git_error")
            continue

        if blob_size > MAX_BLOB_SIZE:
            results[fn] = LocalScanResult(verified=False, reason="oversize")
            continue

        if cumulative_size + blob_size > MAX_TOTAL_SIZE:
            results[fn] = LocalScanResult(verified=False, reason="budget_exceeded")
            continue

        cumulative_size += blob_size

        if blob_size == 0:
            results[fn] = LocalScanResult(verified=True, routing_lines=(), reason=None)
            continue

        # Fetch blob content: git cat-file blob
        try:
            if time.monotonic() >= deadline:
                results[fn] = LocalScanResult(verified=False, reason="git_timeout")
                continue

            timeout = min(PER_COMMAND_TIMEOUT_SECONDS, max(0.01, deadline - time.monotonic()))
            cmd = [
                "git",
                "--no-replace-objects",
                "-c",
                "core.fsmonitor=false",
                "--git-dir",
                str(mirror),
                "cat-file",
                "blob",
                rec.new_sha,
            ]
            ret, blob_bytes = await _run_bounded_git(
                cmd, cwd=str(mirror), env=git_env, max_stdout=blob_size, timeout=timeout
            )
            if ret != 0 or len(blob_bytes) != blob_size:
                results[fn] = LocalScanResult(verified=False, reason="git_error")
                continue
        except _TimeoutError:
            results[fn] = LocalScanResult(verified=False, reason="git_timeout")
            continue
        except _OversizeError:
            results[fn] = LocalScanResult(verified=False, reason="oversize")
            continue
        except Exception:
            results[fn] = LocalScanResult(verified=False, reason="git_error")
            continue

        # Check for binary content (including UTF-16 NUL bytes)
        if b"\x00" in blob_bytes:
            results[fn] = LocalScanResult(verified=False, reason="binary")
            continue

        # Decode and scan whole HEAD content for routing lines
        text = blob_bytes.decode("utf-8", errors="replace")
        matches: list[str] = []
        for line in text.splitlines():
            if _ROUTING_LINE_RE.search(line):
                cleaned = line.strip().replace("`", "")
                cleaned = "".join(ch for ch in cleaned if ch >= " " and ch != "\x7f")[:160]
                matches.append(cleaned)
                if len(matches) >= 20:
                    break

        if matches:
            results[fn] = LocalScanResult(verified=True, routing_lines=tuple(matches), reason=None)
        else:
            results[fn] = LocalScanResult(verified=True, routing_lines=(), reason=None)

    return results
