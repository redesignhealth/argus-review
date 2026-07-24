"""Prompt loading for the review pipeline: packaged files, with a search
chain of optional local override directories.

Prompt bodies ship as ``.md`` files under ``argus/prompts/`` and are
resolved in order (first directory containing ``<name>.md`` wins):

1. ``ARGUS_PROMPTS_DIR`` (``argus.config.Settings``) — an explicit override
   directory. Highest priority so a caller that sets it always gets exactly
   what it points at, regardless of the standard locations below.
2. ``./.argus/prompts/`` — a repo-local directory, resolved relative to the
   current working directory. Meant for prompts checked into a repo (team-
   shared) or gitignored (personal), without needing to export/set anything.
3. ``~/.config/argus/prompts/`` (respecting ``XDG_CONFIG_HOME`` if set) — a
   user-global directory that applies regardless of which repo Argus is
   run from.
4. The packaged files under ``argus.prompts``, resolved via
   ``importlib.resources``. These never change at runtime, so once loaded
   they're cached for the life of the process.

Setting ``ARGUS_NO_PROMPT_OVERRIDES`` truthy skips all of 1-3 and forces
packaged prompts only — for CI/official runs that must not pick up a
developer's local override by accident.

Files in an override directory are reloaded automatically when their mtime
changes, so edits take effect without a process restart. A name that
resolves to no location, or to an empty/whitespace-only file, raises
``ValueError`` naming the prompt and everywhere it was looked for.
"""

from __future__ import annotations

import logging
import os
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path

from argus.config import get_settings

logger = logging.getLogger(__name__)

_PACKAGE = "argus.prompts"

# The 20 prompt names actually referenced at runtime: 11 fetched via a
# literal `fetch_prompt("...")` call (see argus/graph.py, argus/runners.py)
# plus 9 specialist prompts reached indirectly through
# `runners._SPECIALIST_PROMPT_MAP`. Deliberately excludes
# `pr-review-feedback-proposal`: that name is only fetched by an offline
# weekly feedback-loop process (see `docs/CUSTOMIZING_PROMPTS.md`), which is
# not part of the per-PR pipeline this package runs.
REQUIRED_PROMPTS: frozenset[str] = frozenset(
    {
        # Static call sites (11)
        "pr-review-subagent",
        "pr-review-prior-art",
        "pr-review-cross-cutting",
        "pr-review-tests-and-docs",
        "pr-review-feedback-verifier",
        "pr-review-blocking-validator",
        "pr-review-planner",
        "pr-review-coverage-check",
        "pr-review-writer",
        "pr-review-preflight-router",
        "pr-review-lite",
        # Specialist prompts (9), reached via _SPECIALIST_PROMPT_MAP
        "pr-review-specialist-security",
        "pr-review-specialist-sql",
        "pr-review-specialist-infra",
        "pr-review-specialist-orchestration",
        "pr-review-specialist-frontend",
        "pr-review-specialist-slackbot",
        "pr-review-specialist-deployment",
        "pr-review-specialist-llm-patterns",
        "pr-review-specialist-observability",
    }
)

# cache key -> (content, mtime). Keys are namespaced ("override:<absolute-path>"
# / "packaged:<name>") so a name resolved from one source is never served
# from a stale cache entry populated by the other source (e.g. if
# ARGUS_PROMPTS_DIR is set/unset between calls in the same process, as
# tests do). The override key includes the full path, not just the prompt
# name, since multiple override directories are searched and the winning
# path for a given name can change between calls. `mtime` is the override
# file's mtime for override-dir hits, or `None` for packaged hits (packaged
# files are cached forever since they can't change during a process's
# lifetime).
_cache: dict[str, tuple[str, float | None]] = {}


def clear_cache() -> None:
    """Clear the in-memory prompt cache, forcing the next fetch to re-read from disk."""
    _cache.clear()


def known_packaged_prompts() -> frozenset[str]:
    """Return the prompt names available as packaged ``.md`` files.

    Recomputed on every call (a handful of directory-entry stats) rather
    than cached at import time, so it always reflects the on-disk/packaged
    state — useful for tests and for `validate_prompts` diagnostics.
    """
    package_dir = resources.files(_PACKAGE)
    return frozenset(
        entry.name[: -len(".md")]
        for entry in package_dir.iterdir()
        if entry.is_file() and entry.name.endswith(".md")
    )


# Module-level registry of packaged prompt names, derived from the packaged
# directory rather than hardcoded. `REQUIRED_PROMPTS` is the separate,
# hardcoded contract of names the runtime actually needs; the two are
# expected to be equal (see `validate_prompts` and
# `tests/test_prompts.py::test_bidirectional_completeness`).
KNOWN_PROMPTS: frozenset[str] = known_packaged_prompts()


def override_dirs() -> list[Path]:
    """Ordered list of directories to search for prompt overrides.

    Highest priority first, per the module docstring's resolution order.
    Empty when ``ARGUS_NO_PROMPT_OVERRIDES`` is set.
    """
    settings = get_settings()
    if settings.ARGUS_NO_PROMPT_OVERRIDES:
        return []

    dirs: list[Path] = []
    if settings.ARGUS_PROMPTS_DIR:
        dirs.append(Path(settings.ARGUS_PROMPTS_DIR))

    dirs.append(Path.cwd() / ".argus" / "prompts")

    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    config_home = Path(xdg_config_home) if xdg_config_home else Path.home() / ".config"
    dirs.append(config_home / "argus" / "prompts")

    return dirs


def resolve_override_path(name: str) -> Path | None:
    """Return the winning override path for ``name``, or ``None`` if no
    override directory has a matching file (packaged would be used)."""
    for override_dir in override_dirs():
        candidate = override_dir / f"{name}.md"
        if candidate.is_file():
            return candidate
    return None


async def fetch_prompt(name: str) -> str:
    """Fetch a prompt's content by name.

    See module docstring for resolution order. Async for call-site
    compatibility with the previous remote-service-backed implementation;
    the actual work here is synchronous file I/O. Delegates the override
    search itself to :func:`resolve_override_path` so there is exactly one
    place that walks the override chain.

    Raises:
        ValueError: if ``name`` doesn't resolve to a non-empty file in
            any override directory or the packaged prompts.
    """
    override_path = resolve_override_path(name)
    if override_path is not None:
        return _load_override(override_path)

    package_dir = resources.files(_PACKAGE)
    resource = package_dir / f"{name}.md"
    if resource.is_file():
        return _load_packaged(name, resource)

    searched = [str(d / f"{name}.md") for d in override_dirs()]
    searched.append(f"packaged:{_PACKAGE}/{name}.md")
    logger.error("Prompt %r not found. Searched: %s", name, ", ".join(searched))
    raise ValueError(f"Prompt '{name}' not found. Searched: {', '.join(searched)}")


def _load_override(path: Path) -> str:
    # Cache key includes the full path (not just the prompt name): with
    # multiple override directories now possible, the winning path for a
    # given name can change between calls (e.g. a higher-priority file is
    # deleted), and a name-only key would risk returning stale content from
    # a directory that's no longer the winner.
    cache_key = f"override:{path}"
    mtime = path.stat().st_mtime
    cached = _cache.get(cache_key)
    if cached is not None and cached[1] == mtime:
        return cached[0]

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        logger.error("Prompt override file at %s is empty", path)
        raise ValueError(f"Prompt '{path.stem}' at {path} is empty")

    _cache[cache_key] = (content, mtime)
    return content


def _load_packaged(name: str, resource: Traversable) -> str:
    cache_key = f"packaged:{name}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached[0]

    content = resource.read_text(encoding="utf-8").strip()
    if not content:
        logger.error("Prompt %r packaged file (%s) is empty", name, resource)
        raise ValueError(f"Prompt '{name}' packaged file ({resource}) is empty")

    _cache[cache_key] = (content, None)
    return content


async def validate_prompts() -> None:
    """Verify every runtime-required prompt resolves to non-empty content.

    Intended for callers to invoke once at startup (e.g. before the review
    graph runs) — this module only exposes the check itself; wiring it into
    the graph's startup path is out of scope here. Raises a single
    ``ValueError`` aggregating every prompt that failed to resolve, so a
    misconfiguration surfaces all missing prompts at once rather than one
    per run.
    """
    failures: list[str] = []
    for name in sorted(REQUIRED_PROMPTS):
        try:
            await fetch_prompt(name)
        except ValueError as exc:
            failures.append(str(exc))

    if failures:
        raise ValueError(
            "Prompt validation failed for "
            f"{len(failures)} required prompt(s):\n" + "\n".join(failures)
        )
