"""Tests for ``argus prompts list`` / ``argus prompts export``.

These commands are implemented directly against ``importlib.resources`` on
the packaged ``argus.prompts`` files, and directly against ``os.environ``
for override-directory resolution, rather than through
``argus.prompts_runtime``/``argus.config.Settings`` — see
``_prompts_list_override_dirs``'s docstring in ``argus/cli.py`` for why.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from argus.cli import (
    _cmd_prompts_export,
    _cmd_prompts_list,
    _packaged_prompt_names,
    _packaged_prompts_dir,
)


@pytest.fixture(autouse=True)
def _isolate_standard_override_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Similar to the fixture of the same name in ``tests/test_prompts.py``,
    minus its ``clear_settings_cache()`` call: this module tests functions
    that deliberately never go through ``argus.config.Settings`` (see the
    module docstring), so there's no settings cache to invalidate here."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config-home"))
    monkeypatch.delenv("ARGUS_NO_PROMPT_OVERRIDES", raising=False)


class TestPackagedPromptNames:
    def test_returns_nonempty_sorted_list_without_extension(self) -> None:
        names = _packaged_prompt_names()
        assert names
        assert names == sorted(names)
        assert all(not n.endswith(".md") for n in names)
        assert "pr-review-planner" in names

    def test_packaged_dir_exists_and_has_md_files(self) -> None:
        directory = _packaged_prompts_dir()
        assert directory.is_dir()
        assert list(directory.glob("*.md"))


class TestPromptsList:
    def test_all_packaged_by_default(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ARGUS_PROMPTS_DIR", raising=False)
        _cmd_prompts_list()
        out = capsys.readouterr().out
        lines = [line for line in out.splitlines() if line]
        assert len(lines) == len(_packaged_prompt_names())
        for line in lines:
            name, source = line.split("\t")
            assert source == "packaged"
            assert name in _packaged_prompt_names()

    def test_override_dir_reported_for_matching_file(
        self,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        one_name = _packaged_prompt_names()[0]
        (tmp_path / f"{one_name}.md").write_text("custom override", encoding="utf-8")
        monkeypatch.setenv("ARGUS_PROMPTS_DIR", str(tmp_path))

        _cmd_prompts_list()
        out = capsys.readouterr().out

        lines = {line.split("\t")[0]: line.split("\t")[1] for line in out.splitlines() if line}
        assert lines[one_name].startswith("override")
        # Every other prompt name (no matching override file) stays packaged.
        other_names = [n for n in _packaged_prompt_names() if n != one_name]
        assert lines[other_names[0]] == "packaged"

    def test_repo_local_dir_reported_for_matching_file(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ARGUS_PROMPTS_DIR", raising=False)
        one_name = _packaged_prompt_names()[0]
        repo_local = Path.cwd() / ".argus" / "prompts"
        repo_local.mkdir(parents=True)
        (repo_local / f"{one_name}.md").write_text("repo-local override", encoding="utf-8")

        _cmd_prompts_list()
        out = capsys.readouterr().out

        lines = {line.split("\t")[0]: line.split("\t")[1] for line in out.splitlines() if line}
        assert lines[one_name].startswith("override")

    def test_user_global_dir_reported_for_matching_file(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ARGUS_PROMPTS_DIR", raising=False)
        one_name = _packaged_prompt_names()[0]
        user_global = Path(os.environ["XDG_CONFIG_HOME"]) / "argus" / "prompts"
        user_global.mkdir(parents=True)
        (user_global / f"{one_name}.md").write_text("user-global override", encoding="utf-8")

        _cmd_prompts_list()
        out = capsys.readouterr().out

        lines = {line.split("\t")[0]: line.split("\t")[1] for line in out.splitlines() if line}
        assert lines[one_name].startswith("override")

    def test_no_prompt_overrides_forces_packaged(
        self,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        one_name = _packaged_prompt_names()[0]
        (tmp_path / f"{one_name}.md").write_text("custom override", encoding="utf-8")
        monkeypatch.setenv("ARGUS_PROMPTS_DIR", str(tmp_path))
        monkeypatch.setenv("ARGUS_NO_PROMPT_OVERRIDES", "1")

        _cmd_prompts_list()
        out = capsys.readouterr().out

        lines = {line.split("\t")[0]: line.split("\t")[1] for line in out.splitlines() if line}
        assert lines[one_name] == "packaged"


class TestPromptsExport:
    def test_export_copies_all_files_to_empty_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "out"
        _cmd_prompts_export(str(target), force=False)

        exported = sorted(p.name for p in target.glob("*.md"))
        packaged = sorted(p.name for p in _packaged_prompts_dir().glob("*.md"))
        assert exported == packaged
        assert len(exported) == len(_packaged_prompt_names())

    def test_export_content_is_byte_identical(self, tmp_path: Path) -> None:
        target = tmp_path / "out"
        _cmd_prompts_export(str(target), force=False)

        one_name = _packaged_prompt_names()[0]
        src = _packaged_prompts_dir() / f"{one_name}.md"
        dest = target / f"{one_name}.md"
        assert dest.read_bytes() == src.read_bytes()

    def test_refuses_nonempty_dir_without_force(self, tmp_path: Path) -> None:
        target = tmp_path / "out"
        target.mkdir()
        (target / "existing.txt").write_text("keep me", encoding="utf-8")

        with pytest.raises(SystemExit):
            _cmd_prompts_export(str(target), force=False)

        # Nothing was exported — refusal happens before any copy.
        assert not list(target.glob("*.md"))

    def test_force_overwrites_nonempty_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "out"
        target.mkdir()
        (target / "existing.txt").write_text("keep me", encoding="utf-8")

        _cmd_prompts_export(str(target), force=True)

        assert list(target.glob("*.md"))
        # Force only overwrites matching filenames; unrelated files survive.
        assert (target / "existing.txt").exists()

    def test_export_creates_missing_target_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "out"
        _cmd_prompts_export(str(target), force=False)
        assert target.is_dir()
        assert list(target.glob("*.md"))
