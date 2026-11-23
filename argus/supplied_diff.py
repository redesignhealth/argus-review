"""Supplied-diff review kind (TECH-9826): a caller hands Argus the exact diff to review.

The document is an ``upstream-sync-resolution-diff v1``: the conflict resolutions and fix
commits of an upstream sync, not the whole sync. Pure helpers only (no network, no LLM).
The engine treats the document as the PR diff, never fetches a compare diff, and never
writes to GitHub for this kind.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Final, Literal

ReviewKind = Literal["pr", "supplied_diff"]

FORMAT_LINE: Final = "format: upstream-sync-resolution-diff v1"
MAX_DOC_CHARS: Final = 150_000  # same cap as scope.INPUT_CHARS
MAX_CHANGED_LINES: Final = 5_000

_HEX40 = re.compile(r"[0-9a-f]{40}")
_DIFF_GIT = re.compile(r"^diff --git a/(.+?) b/(.+?)$")

# Shown to every reviewer so it does not flag the absence of upstream context it was never given.
REVIEWER_NOTE: Final = (
    "You are reviewing a RESOLUTION DIFF only: the conflict resolutions and fix commits of an "
    "upstream sync, not the whole sync. Lines of the form `mechanical <rule> <path> <blob sha>` "
    "name files the generator settled mechanically; they are NOT reviewed and have no diff. "
    "The surrounding upstream context is deliberately absent, so do not flag missing upstream "
    "context, unchanged upstream code, or files that appear only as `mechanical` lines."
)


class SuppliedDiffError(ValueError):
    """The supplied diff document is malformed; ``str(exc)`` is safe to show to a caller."""


@dataclass(frozen=True)
class SuppliedDiff:
    """A validated supplied-diff document and what is derived from it."""

    text: str
    base_sha: str
    head_sha: str
    digest: str
    files: tuple[str, ...]
    changed_lines: int


def compute_digest(document: str) -> str:
    """``sha256:<hex>`` of the exact UTF-8 bytes of the document."""
    try:
        data = document.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise SuppliedDiffError(f"supplied diff is not valid UTF-8: {exc.reason}") from exc
    return "sha256:" + hashlib.sha256(data).hexdigest()


def count_changed_lines(document: str) -> int:
    """Lines starting with ``+`` or ``-``, excluding the ``---``/``+++`` file headers.

    Headers are only recognised between a ``diff --git`` line and its first ``@@`` hunk
    line; inside a hunk every ``+``/``-`` line is content, including a removed line whose
    text begins ``--``.
    """
    in_hunk = False
    count = 0
    for line in document.split("\n"):
        if line.startswith("diff --git "):
            in_hunk = False
        elif line.startswith("@@"):
            in_hunk = True
        elif not in_hunk and line.startswith(("--- ", "+++ ")):
            continue
        elif line.startswith(("+", "-")):
            count += 1
    return count


def parse_supplied_diff(document: str, *, expect_head_sha: str | None = None) -> SuppliedDiff:
    """Validate the header and derive digest, file list and changed-line count.

    Raises ``SuppliedDiffError`` for a malformed document, or when ``expect_head_sha`` is
    given and differs from the document's ``head:``. Size limits are NOT enforced here (the
    scope gate halts on them) so an oversized but well-formed document still gets a digest.
    """
    if not isinstance(document, str) or not document.strip():
        raise SuppliedDiffError("supplied diff is empty")
    lines = document.split("\n")
    if lines[0] != FORMAT_LINE:
        raise SuppliedDiffError(f"supplied diff must start with {FORMAT_LINE!r}")
    shas: dict[str, str] = {}
    for idx, key in ((1, "base"), (2, "head")):
        line = lines[idx] if len(lines) > idx else ""
        prefix = f"{key}: "
        value = line[len(prefix) :] if line.startswith(prefix) else ""
        if not _HEX40.fullmatch(value):
            raise SuppliedDiffError(
                f"supplied diff line {idx + 1} must be '{key}: <40 lowercase hex chars>'"
            )
        shas[key] = value
    if expect_head_sha is not None and expect_head_sha != shas["head"]:
        raise SuppliedDiffError(
            f"supplied diff head {shas['head'][:12]} does not match the review head "
            f"{expect_head_sha[:12]}"
        )
    files: list[str] = []
    for line in lines[3:]:
        m = _DIFF_GIT.match(line)
        if m and m.group(2) not in files:
            files.append(m.group(2))
    return SuppliedDiff(
        text=document,
        base_sha=shas["base"],
        head_sha=shas["head"],
        digest=compute_digest(document),
        files=tuple(files),
        changed_lines=count_changed_lines(document),
    )


def size_halt_reason(doc: SuppliedDiff) -> str | None:
    """Human-readable reason when the document exceeds a limit, else ``None``."""
    if len(doc.text) > MAX_DOC_CHARS:
        return (
            f"supplied diff is too large to review: {len(doc.text)} characters "
            f"(limit {MAX_DOC_CHARS})"
        )
    if doc.changed_lines > MAX_CHANGED_LINES:
        return (
            f"supplied diff is too large to review: {doc.changed_lines} changed lines "
            f"(limit {MAX_CHANGED_LINES})"
        )
    return None
