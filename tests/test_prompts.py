"""Tests for the packaged-prompts loader (``argus.prompts_runtime``).

Covers: every required prompt loads from the package, the packaged file
set and ``REQUIRED_PROMPTS`` are exactly the same set (an orphan file or a
missing file both fail), the override search chain (``ARGUS_PROMPTS_DIR``
> ``./.argus/prompts/`` > ``~/.config/argus/prompts/`` > packaged) resolves
in priority order with mtime-based reload, ``ARGUS_NO_PROMPT_OVERRIDES``
disables the whole chain, missing/empty content raise ``ValueError``, and
``clear_cache`` forces a re-read. A final grep-based test guards against
future drift between prompt names used in ``argus/*.py`` and
``REQUIRED_PROMPTS``.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Generator
from pathlib import Path

import pytest

from argus.config import clear_cache as clear_settings_cache
from argus.prompts_runtime import (
    KNOWN_PROMPTS,
    REQUIRED_PROMPTS,
    clear_cache,
    fetch_prompt,
    known_packaged_prompts,
    override_dirs,
    resolve_override_path,
    validate_prompts,
)

_ARGUS_PKG_DIR = Path(__file__).resolve().parent.parent / "argus"


@pytest.fixture(autouse=True)
def _clear_prompt_cache() -> Generator[None, None, None]:
    """Clear the prompt cache before and after each test."""
    clear_cache()
    yield
    clear_cache()


@pytest.fixture(autouse=True)
def _isolate_standard_override_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the two standard override locations at throwaway directories.

    Without this, a developer's real ``~/.config/argus/prompts/`` (or a
    ``.argus/prompts/`` left in whatever directory the test runner's cwd
    happens to be) could leak into test results. ``ARGUS_PROMPTS_DIR`` is
    left alone here — individual tests set it explicitly when they want to
    exercise that highest-priority override.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config-home"))
    monkeypatch.delenv("ARGUS_NO_PROMPT_OVERRIDES", raising=False)
    # Without this, a prior test's cached Settings (e.g. one that set
    # ARGUS_NO_PROMPT_OVERRIDES) can outlive monkeypatch's teardown of the
    # env var, silently skipping override dirs in a later test for the
    # wrong reason.
    clear_settings_cache()


@pytest.mark.parametrize("name", sorted(REQUIRED_PROMPTS))
@pytest.mark.asyncio
async def test_required_prompt_loads_from_package(name: str) -> None:
    content = await fetch_prompt(name)
    assert content.strip()


def test_bidirectional_completeness() -> None:
    """Packaged files and REQUIRED_PROMPTS must be exactly the same set.

    An orphan packaged file (present but not in REQUIRED_PROMPTS) or a
    missing packaged file (required but no file on disk) both fail.
    """
    assert known_packaged_prompts() == REQUIRED_PROMPTS
    assert KNOWN_PROMPTS == REQUIRED_PROMPTS


@pytest.mark.asyncio
async def test_validate_prompts_passes_for_packaged_set() -> None:
    await validate_prompts()


def test_no_packaged_prompt_endorses_superseded_structured_output_pattern() -> None:
    """D024 ("No `langchain` Package") supersedes init_chat_model().with_structured_output()
    in favor of LiteLLM's response_format json_schema. Every packaged prompt that mentions
    the superseded pattern must mark it as superseded/non-compliant, never recommend it
    unqualified -- catches the cross-cutting-vs-sibling-prompt contradiction Argus found
    when only pr-review-cross-cutting.md was updated for D024 and its siblings were not.
    """
    superseded = "init_chat_model().with_structured_output()"
    for path in sorted((_ARGUS_PKG_DIR / "prompts").glob("*.md")):
        content = path.read_text(encoding="utf-8")
        if superseded not in content:
            continue
        for line in content.splitlines():
            if superseded in line:
                assert "supersed" in line.lower() or "D024" in line, (
                    f"{path.name}: line mentions the superseded pattern without "
                    f"flagging it as non-compliant per D024: {line!r}"
                )


@pytest.mark.asyncio
async def test_fetch_prompt_missing_name_raises_with_name_in_message() -> None:
    with pytest.raises(ValueError, match="does-not-exist"):
        await fetch_prompt("does-not-exist")


@pytest.mark.asyncio
async def test_fetch_prompt_empty_override_file_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pr-review-lite.md").write_text("   \n\t\n")

    monkeypatch.setenv("ARGUS_PROMPTS_DIR", str(tmp_path))
    clear_settings_cache()

    with pytest.raises(ValueError, match="empty"):
        await fetch_prompt("pr-review-lite")


@pytest.mark.asyncio
async def test_override_dir_wins_for_edited_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override_content = "OVERRIDDEN CONTENT FOR PR-REVIEW-LITE"
    (tmp_path / "pr-review-lite.md").write_text(override_content)

    monkeypatch.setenv("ARGUS_PROMPTS_DIR", str(tmp_path))
    clear_settings_cache()

    result = await fetch_prompt("pr-review-lite")
    assert result == override_content


@pytest.mark.asyncio
async def test_override_dir_falls_back_to_packaged_for_unedited_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Override dir has only one of the required prompts overridden.
    (tmp_path / "pr-review-lite.md").write_text("OVERRIDDEN")

    monkeypatch.setenv("ARGUS_PROMPTS_DIR", str(tmp_path))
    clear_settings_cache()

    overridden = await fetch_prompt("pr-review-lite")
    packaged = await fetch_prompt("pr-review-planner")

    assert overridden == "OVERRIDDEN"
    assert packaged.strip()
    # Sanity: the fallback content actually came from the packaged file,
    # not from the (nonexistent) override.
    assert not (tmp_path / "pr-review-planner.md").exists()


@pytest.mark.asyncio
async def test_override_dir_reloads_on_mtime_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override_path = tmp_path / "pr-review-lite.md"
    override_path.write_text("version one")

    monkeypatch.setenv("ARGUS_PROMPTS_DIR", str(tmp_path))
    clear_settings_cache()

    first = await fetch_prompt("pr-review-lite")
    assert first == "version one"

    # Force the mtime forward explicitly so the test doesn't depend on
    # filesystem mtime resolution/timing.
    time.sleep(0.01)
    new_mtime = override_path.stat().st_mtime + 1
    override_path.write_text("version two")
    os.utime(override_path, (new_mtime, new_mtime))

    second = await fetch_prompt("pr-review-lite")
    assert second == "version two"


def test_override_dirs_order_with_no_explicit_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARGUS_PROMPTS_DIR", raising=False)
    clear_settings_cache()

    dirs = override_dirs()

    assert dirs == [
        Path.cwd() / ".argus" / "prompts",
        Path(os.environ["XDG_CONFIG_HOME"]) / "argus" / "prompts",
    ]


def test_override_dirs_order_with_explicit_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARGUS_PROMPTS_DIR", str(tmp_path))
    clear_settings_cache()

    dirs = override_dirs()

    assert dirs == [
        tmp_path,
        Path.cwd() / ".argus" / "prompts",
        Path(os.environ["XDG_CONFIG_HOME"]) / "argus" / "prompts",
    ]


def test_override_dirs_empty_when_overrides_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARGUS_PROMPTS_DIR", str(tmp_path))
    monkeypatch.setenv("ARGUS_NO_PROMPT_OVERRIDES", "1")
    clear_settings_cache()

    assert override_dirs() == []


@pytest.mark.asyncio
async def test_repo_local_dir_wins_over_packaged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARGUS_PROMPTS_DIR", raising=False)
    clear_settings_cache()

    repo_local = Path.cwd() / ".argus" / "prompts"
    repo_local.mkdir(parents=True)
    (repo_local / "pr-review-lite.md").write_text("REPO-LOCAL OVERRIDE")

    result = await fetch_prompt("pr-review-lite")
    assert result == "REPO-LOCAL OVERRIDE"


@pytest.mark.asyncio
async def test_user_global_dir_wins_over_packaged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARGUS_PROMPTS_DIR", raising=False)
    clear_settings_cache()

    user_global = Path(os.environ["XDG_CONFIG_HOME"]) / "argus" / "prompts"
    user_global.mkdir(parents=True)
    (user_global / "pr-review-lite.md").write_text("USER-GLOBAL OVERRIDE")

    result = await fetch_prompt("pr-review-lite")
    assert result == "USER-GLOBAL OVERRIDE"


@pytest.mark.asyncio
async def test_repo_local_dir_wins_over_user_global_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ARGUS_PROMPTS_DIR", raising=False)
    clear_settings_cache()

    repo_local = Path.cwd() / ".argus" / "prompts"
    repo_local.mkdir(parents=True)
    (repo_local / "pr-review-lite.md").write_text("REPO-LOCAL")

    user_global = Path(os.environ["XDG_CONFIG_HOME"]) / "argus" / "prompts"
    user_global.mkdir(parents=True)
    (user_global / "pr-review-lite.md").write_text("USER-GLOBAL")

    result = await fetch_prompt("pr-review-lite")
    assert result == "REPO-LOCAL"


@pytest.mark.asyncio
async def test_explicit_dir_wins_over_repo_local_and_user_global(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    explicit = tmp_path / "explicit-override"
    explicit.mkdir()
    (explicit / "pr-review-lite.md").write_text("EXPLICIT")
    monkeypatch.setenv("ARGUS_PROMPTS_DIR", str(explicit))
    clear_settings_cache()

    repo_local = Path.cwd() / ".argus" / "prompts"
    repo_local.mkdir(parents=True)
    (repo_local / "pr-review-lite.md").write_text("REPO-LOCAL")

    user_global = Path(os.environ["XDG_CONFIG_HOME"]) / "argus" / "prompts"
    user_global.mkdir(parents=True)
    (user_global / "pr-review-lite.md").write_text("USER-GLOBAL")

    result = await fetch_prompt("pr-review-lite")
    assert result == "EXPLICIT"


@pytest.mark.asyncio
async def test_no_prompt_overrides_ignores_every_override_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    explicit = tmp_path / "explicit-override"
    explicit.mkdir()
    (explicit / "pr-review-lite.md").write_text("EXPLICIT")
    monkeypatch.setenv("ARGUS_PROMPTS_DIR", str(explicit))
    monkeypatch.setenv("ARGUS_NO_PROMPT_OVERRIDES", "1")
    clear_settings_cache()

    repo_local = Path.cwd() / ".argus" / "prompts"
    repo_local.mkdir(parents=True)
    (repo_local / "pr-review-lite.md").write_text("REPO-LOCAL")

    packaged_content = (
        (_ARGUS_PKG_DIR / "prompts" / "pr-review-lite.md").read_text(encoding="utf-8").strip()
    )

    result = await fetch_prompt("pr-review-lite")
    assert result == packaged_content


def test_resolve_override_path_returns_none_when_no_override_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ARGUS_PROMPTS_DIR", raising=False)
    clear_settings_cache()

    assert resolve_override_path("pr-review-lite") is None


def test_resolve_override_path_returns_winning_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARGUS_PROMPTS_DIR", raising=False)
    clear_settings_cache()

    repo_local = Path.cwd() / ".argus" / "prompts"
    repo_local.mkdir(parents=True)
    override_path = repo_local / "pr-review-lite.md"
    override_path.write_text("REPO-LOCAL")

    assert resolve_override_path("pr-review-lite") == override_path


@pytest.mark.asyncio
async def test_clear_cache_forces_reread_of_packaged_prompt() -> None:
    first = await fetch_prompt("pr-review-lite")
    clear_cache()
    second = await fetch_prompt("pr-review-lite")
    assert first == second


def test_all_literal_fetch_prompt_calls_are_required() -> None:
    """Guard against drift: every literal name passed to fetch_prompt(...)
    in argus/*.py must be a member of REQUIRED_PROMPTS."""
    pattern = re.compile(r"""fetch_prompt\(\s*["'](pr-review-[a-z-]+)["']\s*\)""")
    found: set[str] = set()

    for path in _ARGUS_PKG_DIR.glob("*.py"):
        if path.name == "prompts_runtime.py":
            continue  # only illustrative examples in docstrings/comments here
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            found.add(match.group(1))

    assert found, "expected to find at least one literal fetch_prompt(...) call site"
    missing = found - REQUIRED_PROMPTS
    assert not missing, (
        f"fetch_prompt() called with name(s) not in REQUIRED_PROMPTS: {sorted(missing)}"
    )
