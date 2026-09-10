"""Platform-neutral leaf-reviewer tool implementations.

These are plain Python functions intended for reuse by ANY future
non-Claude runner (Gemini, OpenAI Responses, ...) -- deliberately NOT
coupled to a specific agent framework's ``@tool``/function-schema
decorator, since the actual agent-loop implementation that wraps them is
out of scope here (Track 3). A future runner adapter wraps
``read_file``/``glob_files``/``grep``/``report_finding``/``finish_review``
with whatever tool-declaration shape its own SDK expects.

Calling convention: all five functions read their sandboxed worktree
root and (for ``report_finding``/``finish_review``) their findings sink
from a pair of module-level ``contextvars.ContextVar``s, rather than
taking a ``root``/``session_id`` parameter directly. A caller opens one
:func:`review_session` per concurrent reviewer session (e.g. one per
``asyncio`` task in a 16-way fan-out):

    with review_tools.review_session(worktree_root) as findings:
        # ... drive an agent loop that calls read_file/glob_files/grep/
        # report_finding/finish_review as tool calls ...
        pass
    # `findings` still holds every reported finding here -- the context
    # manager only unbinds the ContextVars on exit, it doesn't clear the
    # list object itself.

Because ``contextvars`` context is copied (not shared) when an
``asyncio`` task is created, two concurrent reviewer sessions opened in
two different tasks never see each other's root or findings sink, even
though both run in the same process -- this is the "16 concurrent
reviewers don't cross-contaminate each other's findings" property.

Sandboxing: ``read_file``, ``glob_files``, and ``grep`` all resolve paths
against the active session's root using ``argus.helpers.sanitize_file_paths``
-- the same path-containment idiom ``argus.runners`` already uses to
sandbox LLM-supplied file paths -- rather than a new one invented here.
``glob_files`` and ``grep`` additionally re-resolve (``Path.resolve()``)
every individual candidate returned by ``Path.glob()`` and verify it is
still contained under the (already-resolved) root before treating it as a
match: ``sanitize_file_paths`` only validates the caller-supplied
``path``/``pattern`` argument itself, but a glob match can be a symlink
*inside* the sandboxed tree (a file symlink, or an intermediate path
segment that is a symlinked directory) whose real target lives *outside*
it -- see ``_is_within_root``.

Resource limits: ``grep`` caps the pattern length, the number of files
scanned, and the number of results returned, so a huge glob match set
can't blow up memory in a caller that can't bound its own input (an
LLM-supplied tool call). It additionally guards against a pathological/
catastrophic-backtracking regex with a wall-clock time budget
(``_GREP_TIMEOUT_SECONDS``), enforced via ``signal.SIGALRM`` when running
on the main thread -- confirmed empirically to interrupt a hung
``re.Pattern.search()`` call promptly, unlike a plain
``threading.Thread.join(timeout=...)`` watchdog, which CANNOT preempt it:
CPython's regex engine holds the GIL for the whole backtracking loop
without yielding, so a second thread waiting on ``join()`` never gets
scheduled until the pathological call itself returns. ``SIGALRM`` is
main-thread-only (a Python/POSIX constraint, not a design choice here),
so a caller invoking ``grep`` from a non-main thread gets no per-call
interrupt -- the pattern-length/files-scanned/file-size/results caps
above are what bound that case instead. ``read_file`` rejects a file
above a size cap outright and streams line-by-line up to a hard cap on
lines returned, rather than materializing the whole file in memory
before slicing it.
"""

from __future__ import annotations

import contextlib
import itertools
import re
import signal
import threading
from collections.abc import Iterator
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from argus.helpers import sanitize_file_paths

_ROOT_VAR: ContextVar[str | None] = ContextVar("argus_review_tools_root", default=None)
_FINDINGS_VAR: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "argus_review_tools_findings", default=None
)

_DEFAULT_READ_LIMIT = 2000
_VALID_GREP_MODES = frozenset({"files", "content"})

# read_file: avoid materializing an arbitrarily large file in memory before
# slicing it down to `limit` lines.
_MAX_READ_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MiB
_MAX_READ_LINES_HARD_CAP = 20_000  # absolute ceiling, even when limit=0

# glob_files: bound output size and file candidate enumeration to avoid memory spikes.
_MAX_GLOB_RESULTS = 500
_MAX_GLOB_CANDIDATES = 5000

# grep: bound pattern complexity, total work, and total output, since the
# pattern/glob ultimately come from an LLM tool call this code can't trust
# to be well-behaved.
_MAX_GREP_PATTERN_LENGTH = 500
_MAX_GREP_FILES_SCANNED = 5000
_MAX_GREP_FILE_SIZE_BYTES = 2 * 1024 * 1024  # 2 MiB; skip larger files outright
_MAX_GREP_RESULTS = 2000  # combined cap on file_hits/content_hits entries
_GREP_TIMEOUT_SECONDS = 10.0


class NoActiveReviewSessionError(RuntimeError):
    """Raised when a tool function is called outside of a `review_session`."""


@contextlib.contextmanager
def review_session(root: str) -> Iterator[list[dict[str, Any]]]:
    """Open a sandboxed tool session scoped to ``root`` for the current context.

    Yields the (initially empty) findings list that ``report_finding``
    appends to. The list is a plain object, not itself a ContextVar, so
    the caller can keep reading it after the ``with`` block exits (exit
    only resets the ContextVars binding root/sink to this context -- it
    does not clear the list's contents).
    """
    findings: list[dict[str, Any]] = []
    root_token = _ROOT_VAR.set(str(Path(root).resolve()))
    findings_token = _FINDINGS_VAR.set(findings)
    try:
        yield findings
    finally:
        _ROOT_VAR.reset(root_token)
        _FINDINGS_VAR.reset(findings_token)


def _require_root() -> str:
    root = _ROOT_VAR.get()
    if root is None:
        raise NoActiveReviewSessionError(
            "No active review_session. Call argus.review_tools.review_session(root) "
            "before using read_file/glob_files/grep."
        )
    return root


def _require_findings_sink() -> list[dict[str, Any]]:
    sink = _FINDINGS_VAR.get()
    if sink is None:
        raise NoActiveReviewSessionError(
            "No active review_session. Call argus.review_tools.review_session(root) "
            "before using report_finding/finish_review."
        )
    return sink


def _safe_path(path: str, root: str) -> Path:
    """Resolve ``path`` against ``root``, raising if it escapes the root.

    Reuses ``argus.helpers.sanitize_file_paths`` -- passing a single-item
    list and treating an empty result as rejection -- rather than
    reimplementing path-containment logic here.
    """
    safe = sanitize_file_paths([path], root)
    if not safe:
        raise ValueError(f"Path escapes the sandboxed worktree root: {path!r}")
    return Path(root).resolve() / safe[0]


def _reject_traversal_pattern(pattern: str, arg_name: str) -> None:
    """Reject a glob pattern that could escape the sandboxed root.

    ``Path.glob`` follows ``..`` segments and rejects absolute patterns
    outright on some Python versions but not consistently across all of
    them, so both are checked explicitly here rather than relied upon.
    """
    if Path(pattern).is_absolute():
        raise ValueError(f"{arg_name} must be a relative pattern, got absolute: {pattern!r}")
    if ".." in Path(pattern).parts:
        raise ValueError(f"{arg_name} must not contain '..': {pattern!r}")


def _is_within_root(candidate: Path, root: Path) -> bool:
    """True if ``candidate``'s REAL (symlink-resolved) path is contained in ``root``.

    ``root`` must already be a resolved, absolute path. Resolving
    ``candidate`` here -- rather than trusting the unresolved path shape
    ``Path.glob()`` returns -- is what closes the symlink-escape gap: a
    symlink *inside* the sandboxed tree that points *outside* it (a
    symlinked file, or a symlinked directory reached via an explicit path
    segment) is syntactically still "under root" before resolution, but
    ``Path.resolve()`` follows it to its real target first, so the
    containment check below correctly rejects it.
    """
    try:
        resolved = candidate.resolve()
    except OSError:
        return False
    return resolved == root or resolved.is_relative_to(root)


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


def read_file(path: str, offset: int = 0, limit: int = _DEFAULT_READ_LIMIT) -> str:
    """Read a file, sandboxed to the active session's root.

    Returns line-numbered content (``"<1-based line number>: <content>"``,
    one per line), starting at the 0-indexed ``offset`` and returning at
    most ``limit`` lines (``limit=0`` reads to EOF, capped at
    ``_MAX_READ_LINES_HARD_CAP``).

    Reads the file line-by-line rather than loading the whole thing into
    memory first: ``_safe_path`` (via ``sanitize_file_paths``) already
    fully resolves ``path`` (following any symlink) before the containment
    check, so a symlink escaping the sandboxed root is rejected there; the
    size cap below additionally guards against a single-huge-file (or
    single-huge-line) resource-exhaustion case that line-count limiting
    alone wouldn't catch.

    Raises:
        NoActiveReviewSessionError: if called outside a ``review_session``.
        ValueError: if ``path`` escapes the sandboxed root, or the file is
            larger than ``_MAX_READ_FILE_SIZE_BYTES``.
        FileNotFoundError: if the resolved path is not a file.
    """
    root = _require_root()
    resolved = _safe_path(path, root)
    if not resolved.is_file() or not _is_within_root(resolved, Path(root)):
        raise FileNotFoundError(f"No such file: {path!r} (resolved: {resolved})")

    try:
        size = resolved.stat().st_size
    except OSError as e:
        raise FileNotFoundError(f"No such file: {path!r} (resolved: {resolved})") from e
    if size > _MAX_READ_FILE_SIZE_BYTES:
        raise ValueError(
            f"{path!r} is {size} bytes, over the {_MAX_READ_FILE_SIZE_BYTES}-byte read cap; "
            "use grep to search it instead of reading it whole."
        )

    start = max(offset, 0)
    span = min(limit, _MAX_READ_LINES_HARD_CAP) if limit else _MAX_READ_LINES_HARD_CAP
    end = start + span

    numbered: list[str] = []
    with resolved.open(encoding="utf-8", errors="replace") as fh:
        for i, raw_line in enumerate(fh):
            if i < start:
                continue
            if i >= end:
                break
            numbered.append(f"{i + 1}: {raw_line.rstrip(chr(10))}")
    return "\n".join(numbered)


def glob_files(pattern: str) -> str:
    """List files under the active session's root matching ``pattern``.

    ``pattern`` is a ``pathlib``-style glob pattern (e.g. ``"**/*.py"``),
    relative to the sandboxed root. Returns matched paths (relative to
    root), one per line, sorted. A match whose real (symlink-resolved)
    path escapes the sandboxed root -- e.g. a symlinked file, or a path
    that traverses a symlinked directory -- is silently skipped rather
    than listed; see ``_is_within_root``.

    Output is capped at ``_MAX_GLOB_RESULTS`` matches to prevent unbounded
    memory growth from broad queries.

    Raises:
        NoActiveReviewSessionError: if called outside a ``review_session``.
        ValueError: if ``pattern`` is absolute or contains ``".."``.
    """
    root = _require_root()
    _reject_traversal_pattern(pattern, "pattern")

    root_path = Path(root)
    matches: list[str] = []
    truncated = False
    for candidate in itertools.islice(root_path.glob(pattern), _MAX_GLOB_CANDIDATES):
        if candidate.is_file() and _is_within_root(candidate, root_path):
            matches.append(str(candidate.relative_to(root_path)))
            if len(matches) >= _MAX_GLOB_RESULTS:
                truncated = True
                break

    matches.sort()
    result = "\n".join(matches)
    if truncated:
        result += f"\n... results capped at {_MAX_GLOB_RESULTS} matches"
    return result


@contextlib.contextmanager
def _grep_alarm(timeout_seconds: float) -> Iterator[None]:
    """Raise ``TimeoutError`` if the ``with`` block runs past ``timeout_seconds``.

    Uses ``signal.SIGALRM``/``setitimer`` rather than a thread-based
    watchdog: empirically, CPython's regex engine holds the GIL for its
    entire (possibly catastrophically-backtracking) C loop without
    yielding, so a second thread polling ``Thread.join(timeout=...)``
    never actually gets scheduled to notice the timeout until the
    pathological call returns on its own -- i.e. it doesn't work for this
    specific hazard. A real signal, by contrast, interrupts the C loop
    directly. Restores the previous ``SIGALRM`` handler on exit either way.
    """

    def _on_alarm(signum: int, frame: Any) -> None:
        raise TimeoutError(
            f"grep exceeded its {timeout_seconds}s time budget; the pattern "
            "may be pathologically slow to match -- narrow it and try again."
        )

    old_handler = signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    finally:
        try:
            signal.signal(signal.SIGALRM, old_handler)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)


def _grep_scan(
    regex: re.Pattern[str], root_path: Path, glob: str, mode: str
) -> tuple[list[str], list[str]]:
    """The actual glob-and-match loop.

    Enforces the total-files-scanned, per-file-size, and total-results
    caps, and skips any glob match whose real path escapes ``root_path``
    (symlink escape -- see ``_is_within_root``).
    """
    file_hits: list[str] = []
    content_hits: list[str] = []
    scanned = 0

    candidates = list(itertools.islice(root_path.glob(glob), _MAX_GREP_FILES_SCANNED * 2))
    candidates.sort()
    for candidate in candidates:
        if scanned >= _MAX_GREP_FILES_SCANNED:
            break
        if not candidate.is_file() or not _is_within_root(candidate, root_path):
            continue
        try:
            if candidate.stat().st_size > _MAX_GREP_FILE_SIZE_BYTES:
                continue
            text = candidate.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        scanned += 1

        rel = str(candidate.relative_to(root_path))
        matched = False
        for lineno, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                matched = True
                if mode == "content":
                    content_hits.append(f"{rel}:{lineno}: {line}")
                    if len(content_hits) >= _MAX_GREP_RESULTS:
                        return file_hits, content_hits
        if matched:
            file_hits.append(rel)
            if len(file_hits) >= _MAX_GREP_RESULTS:
                break

    return file_hits, content_hits


def grep(pattern: str, glob: str = "**/*", mode: str = "files") -> str:
    """Search for a regex ``pattern`` in files matching ``glob`` under root.

    ``mode="files"`` (default) returns matching file paths, one per line.
    ``mode="content"`` returns ``"<file>:<line>: <content>"`` for every
    matching line. A glob match whose real (symlink-resolved) path
    escapes the sandboxed root is silently skipped rather than read; see
    ``_is_within_root``.

    Bounded in several ways, since ``pattern``/``glob`` ultimately come
    from an LLM tool call this code can't trust to be well-behaved:
    ``pattern`` is capped at ``_MAX_GREP_PATTERN_LENGTH`` characters; at
    most ``_MAX_GREP_FILES_SCANNED`` files are scanned and files over
    ``_MAX_GREP_FILE_SIZE_BYTES`` are skipped; results are capped at
    ``_MAX_GREP_RESULTS`` entries; and, when called from the main thread,
    the scan runs under a ``_GREP_TIMEOUT_SECONDS`` ``SIGALRM`` wall-clock
    budget, so a catastrophic-backtracking regex raises instead of hanging
    the caller forever. See ``_grep_alarm`` for why this needs to be a
    real signal rather than a thread-based watchdog, and why it only
    applies on the main thread.

    Raises:
        NoActiveReviewSessionError: if called outside a ``review_session``.
        ValueError: if ``mode`` is unrecognized, ``glob`` is absolute or
            contains ``".."``, or ``pattern`` exceeds the length cap.
        re.error: if ``pattern`` is not a valid regular expression.
        TimeoutError: if the scan exceeds ``_GREP_TIMEOUT_SECONDS`` (only
            enforceable when called from the main thread; see above).
    """
    if mode not in _VALID_GREP_MODES:
        raise ValueError(f"Unknown grep mode {mode!r}; must be one of {sorted(_VALID_GREP_MODES)}")
    if len(pattern) > _MAX_GREP_PATTERN_LENGTH:
        raise ValueError(
            f"grep pattern is {len(pattern)} chars, over the "
            f"{_MAX_GREP_PATTERN_LENGTH}-char cap; use a shorter/simpler pattern"
        )

    root = _require_root()
    _reject_traversal_pattern(glob, "glob")

    root_path = Path(root)
    regex = re.compile(pattern)

    if hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread():
        with _grep_alarm(_GREP_TIMEOUT_SECONDS):
            file_hits, content_hits = _grep_scan(regex, root_path, glob, mode)
    else:
        # SIGALRM is POSIX-only and main-thread-only. Off the main thread
        # (or on a platform without it), we fall back to no hard wall-clock
        # interrupt -- the pattern-length/files-scanned/file-size/results
        # caps above are what bound this case instead.
        file_hits, content_hits = _grep_scan(regex, root_path, glob, mode)

    return "\n".join(content_hits if mode == "content" else file_hits)


def report_finding(
    file: str | None,
    line: int | str | None,
    description: str,
    context: str | None = None,
) -> str:
    """Record one review finding into the active session's findings sink.

    Raises:
        NoActiveReviewSessionError: if called outside a ``review_session``.
    """
    sink = _require_findings_sink()
    sink.append({"file": file, "line": line, "description": description, "context": context})
    return f"Recorded finding #{len(sink)}: {file}:{line} -- {description[:80]}"


def finish_review(files_explored: list[str]) -> str:
    """Signal that the reviewer is done exploring and reporting findings.

    Intended as the terminal tool call an agent loop watches for to stop
    iterating; the caller reads the accumulated findings back from the
    list yielded by ``review_session`` (this function does not clear or
    mutate the sink).

    Raises:
        NoActiveReviewSessionError: if called outside a ``review_session``.
    """
    sink = _require_findings_sink()
    return (
        f"Review finished: {len(sink)} finding(s) reported across "
        f"{len(files_explored)} file(s) explored."
    )
