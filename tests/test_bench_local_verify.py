"""Tests for local bounded git object fallback for missing patch verification (TECH-7768)."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
from contextlib import ExitStack
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest_httpx import HTTPXMock

from argus.bench_local_verify import (
    MAX_MISSING_FILES,
    VALID_FAIL_REASONS,
    LocalScanResult,
    _OversizeError,
    _TimeoutError,
    _build_git_env,
    _parse_diff_raw,
    _run_bounded_git,
    _validate_path,
    verify_missing_patches_locally,
)
from argus.github_client import GitHubClient
from argus.graph import _node_fetch_diff
from argus.models import ReviewRequest


@pytest.fixture
def git_repo_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Set up a real bare mirror under ARGUS_REPO_CACHE and a detached worktree."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("ARGUS_REPO_CACHE", str(cache_dir))

    repo = "testorg/testrepo"
    mirror_dir = cache_dir / "testorg" / "testrepo.git"
    mirror_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["git", "init", "--bare", str(mirror_dir)])

    # Setup temporary source repo to create commits
    src_dir = tmp_path / "src"
    subprocess.check_call(["git", "init", str(src_dir)])
    subprocess.check_call(["git", "-C", str(src_dir), "config", "user.name", "Argus Tester"])
    subprocess.check_call(["git", "-C", str(src_dir), "config", "user.email", "argus@example.com"])

    # Base commit
    (src_dir / "README.md").write_text("# Test Repo\n", encoding="utf-8")
    (src_dir / "file1.txt").write_text("base line 1\nbase line 2\n", encoding="utf-8")
    subprocess.check_call(["git", "-C", str(src_dir), "add", "."])
    subprocess.check_call(["git", "-C", str(src_dir), "commit", "-m", "base commit"])
    base_sha = (
        subprocess.check_output(["git", "-C", str(src_dir), "rev-parse", "HEAD"]).decode().strip()
    )

    # Head commit
    (src_dir / "file1.txt").write_text("base line 1\nmodified line 2\n", encoding="utf-8")
    (src_dir / "clean_file.txt").write_text("clean content\n", encoding="utf-8")
    subprocess.check_call(["git", "-C", str(src_dir), "add", "."])
    subprocess.check_call(["git", "-C", str(src_dir), "commit", "-m", "head commit"])
    head_sha = (
        subprocess.check_output(["git", "-C", str(src_dir), "rev-parse", "HEAD"]).decode().strip()
    )

    # Push to bare mirror
    subprocess.check_call(
        ["git", "-C", str(src_dir), "push", str(mirror_dir), f"{head_sha}:refs/heads/main"]
    )

    # Add detached worktree from mirror
    worktree_dir = tmp_path / "worktree"
    subprocess.check_call(
        ["git", "-C", str(mirror_dir), "worktree", "add", "--detach", str(worktree_dir), head_sha]
    )

    return {
        "repo": repo,
        "src_dir": src_dir,
        "mirror_dir": mirror_dir,
        "worktree_dir": str(worktree_dir),
        "base_sha": base_sha,
        "head_sha": head_sha,
    }


class TestLocalScanResultDataclass:
    """Validation tests for the LocalScanResult dataclass."""

    def test_verified_clean(self) -> None:
        res = LocalScanResult(verified=True)
        assert res.verified is True
        assert res.routing_lines == ()
        assert res.reason is None

    def test_verified_with_routing_lines(self) -> None:
        res = LocalScanResult(verified=True, routing_lines=("ARGUS_BENCH_FILE = 'foo'",))
        assert res.verified is True
        assert len(res.routing_lines) == 1
        assert res.reason is None

    def test_verified_cannot_have_reason(self) -> None:
        with pytest.raises(ValueError, match="verified result cannot have failure reason"):
            LocalScanResult(verified=True, reason="binary")

    def test_unverified_requires_valid_reason(self) -> None:
        with pytest.raises(ValueError, match="invalid failure reason"):
            LocalScanResult(verified=False, reason="custom_unknown_code")

    def test_unverified_cannot_have_routing_lines(self) -> None:
        with pytest.raises(ValueError, match="unverified result cannot have routing_lines"):
            LocalScanResult(verified=False, routing_lines=("ARGUS_BENCH_FILE",), reason="binary")


class TestPathValidation:
    """Validation tests for file paths."""

    def test_valid_paths(self) -> None:
        assert _validate_path("file.txt")
        assert _validate_path("src/app.py")
        assert _validate_path("src/pkg/deep/file.rs")
        assert _validate_path("path with spaces/file.txt")
        assert _validate_path("path\nwith\nnewline/file.txt")
        assert _validate_path("unicode/🚀/file.txt")
        assert _validate_path("-leading-hyphen/file.txt")
        assert _validate_path("*glob?[abc]/file.txt")

    def test_invalid_paths(self) -> None:
        assert not _validate_path("")
        assert not _validate_path(None)
        assert not _validate_path(123)
        assert not _validate_path("/absolute/path")
        assert not _validate_path("has/\x00/nul")
        assert not _validate_path("empty//component")
        assert not _validate_path("./dot/path")
        assert not _validate_path("dot/./path")
        assert not _validate_path("../dotdot/path")
        assert not _validate_path("dotdot/../path")
        assert not _validate_path(".git/HEAD")
        assert not _validate_path(".GIT/config")
        assert not _validate_path("nested/.Git/file")
        assert not _validate_path("a" * 4097)


class TestDiffRawParsing:
    """Tests for strict parsing of git diff --raw -z output."""

    def test_parse_valid(self) -> None:
        raw = (
            b":100644 100644 "
            b"b6fc4c620b67d95f953a5c1c1230aaab5db5a1b0 "
            b"0005ded820bd8f447c101715e3ded330a16dfe29 M\x00file1.txt\x00"
        )
        records = _parse_diff_raw(raw, {"file1.txt"})
        assert "file1.txt" in records
        rec = records["file1.txt"]
        assert rec.old_mode == "100644"
        assert rec.new_mode == "100644"
        assert rec.old_sha == "b6fc4c620b67d95f953a5c1c1230aaab5db5a1b0"
        assert rec.new_sha == "0005ded820bd8f447c101715e3ded330a16dfe29"
        assert rec.status == "M"

    def test_parse_empty(self) -> None:
        assert _parse_diff_raw(b"", set()) == {}

    def test_parse_truncated(self) -> None:
        raw = b":100644 100644 "
        with pytest.raises(ValueError, match="truncated diff-raw output"):
            _parse_diff_raw(raw, {"file1.txt"})

    def test_parse_unrequested_path(self) -> None:
        raw = (
            b":100644 100644 "
            b"b6fc4c620b67d95f953a5c1c1230aaab5db5a1b0 "
            b"0005ded820bd8f447c101715e3ded330a16dfe29 M\x00other.txt\x00"
        )
        with pytest.raises(ValueError, match="unrequested path"):
            _parse_diff_raw(raw, {"file1.txt"})

    def test_parse_duplicate_path(self) -> None:
        raw = (
            b":100644 100644 "
            b"b6fc4c620b67d95f953a5c1c1230aaab5db5a1b0 "
            b"0005ded820bd8f447c101715e3ded330a16dfe29 M\x00file1.txt\x00"
            b":100644 100644 "
            b"b6fc4c620b67d95f953a5c1c1230aaab5db5a1b0 "
            b"0005ded820bd8f447c101715e3ded330a16dfe29 M\x00file1.txt\x00"
        )
        with pytest.raises(ValueError, match="duplicate path"):
            _parse_diff_raw(raw, {"file1.txt"})


class TestVerifyMissingPatchesLocally:
    """Functional tests for verify_missing_patches_locally."""

    @pytest.mark.asyncio
    async def test_pr100_clean_missing_files_verify_cleanly(
        self, git_repo_fixture: dict[str, Any]
    ) -> None:
        """PR 100 scenario: missing patch on clean files (e.g. uv.lock + data.json) verifies clean."""
        repo = git_repo_fixture["repo"]
        wt = git_repo_fixture["worktree_dir"]
        head_sha = git_repo_fixture["head_sha"]
        base_sha = git_repo_fixture["base_sha"]

        entries = [
            {"filename": "file1.txt", "status": "modified", "patch": None},
            {"filename": "clean_file.txt", "status": "added", "patch": ""},
        ]

        results = await verify_missing_patches_locally(
            repo=repo,
            worktree_path=wt,
            trusted_head_sha=head_sha,
            head_sha=head_sha,
            base_sha=base_sha,
            entries=entries,
        )

        assert results["file1.txt"].verified is True
        assert results["file1.txt"].routing_lines == ()
        assert results["file1.txt"].reason is None

        assert results["clean_file.txt"].verified is True
        assert results["clean_file.txt"].routing_lines == ()
        assert results["clean_file.txt"].reason is None

    @pytest.mark.asyncio
    async def test_routing_line_detected_in_head_blob(
        self, git_repo_fixture: dict[str, Any], tmp_path: Path
    ) -> None:
        """A missing-patch file introducing ARGUS_BENCH_FILE in HEAD blob is detected."""
        src_dir = git_repo_fixture["src_dir"]
        mirror_dir = git_repo_fixture["mirror_dir"]
        base_sha = git_repo_fixture["base_sha"]

        (src_dir / "uv.lock").write_text("ARGUS_BENCH_FILE = 'malicious.toml'\n", encoding="utf-8")
        subprocess.check_call(["git", "-C", str(src_dir), "add", "uv.lock"])
        subprocess.check_call(["git", "-C", str(src_dir), "commit", "-m", "add uv.lock"])
        head_sha = (
            subprocess.check_output(["git", "-C", str(src_dir), "rev-parse", "HEAD"])
            .decode()
            .strip()
        )
        subprocess.check_call(
            ["git", "-C", str(src_dir), "push", str(mirror_dir), f"{head_sha}:refs/heads/main"]
        )

        wt = tmp_path / "wt_routing"
        subprocess.check_call(
            ["git", "-C", str(mirror_dir), "worktree", "add", "--detach", str(wt), head_sha]
        )

        entries = [{"filename": "uv.lock", "status": "added", "patch": None}]
        results = await verify_missing_patches_locally(
            repo=git_repo_fixture["repo"],
            worktree_path=str(wt),
            trusted_head_sha=head_sha,
            head_sha=head_sha,
            base_sha=base_sha,
            entries=entries,
        )

        res = results["uv.lock"]
        assert res.verified is True
        assert len(res.routing_lines) == 1
        assert "ARGUS_BENCH_FILE = 'malicious.toml'" in res.routing_lines[0]
        assert res.reason is None

    @pytest.mark.asyncio
    async def test_preexisting_routing_line_blocks(
        self, git_repo_fixture: dict[str, Any], tmp_path: Path
    ) -> None:
        """Pre-existing routing lines in HEAD blob also block (conservative whole HEAD scan)."""
        src_dir = git_repo_fixture["src_dir"]
        mirror_dir = git_repo_fixture["mirror_dir"]

        (src_dir / "routing.txt").write_text("ARGUS_NO_BENCH_OVERRIDES=1\n", encoding="utf-8")
        subprocess.check_call(["git", "-C", str(src_dir), "add", "routing.txt"])
        subprocess.check_call(["git", "-C", str(src_dir), "commit", "-m", "preexisting"])
        base_sha = (
            subprocess.check_output(["git", "-C", str(src_dir), "rev-parse", "HEAD"])
            .decode()
            .strip()
        )

        (src_dir / "routing.txt").write_text(
            "ARGUS_NO_BENCH_OVERRIDES=1\nmodified\n", encoding="utf-8"
        )
        subprocess.check_call(["git", "-C", str(src_dir), "add", "routing.txt"])
        subprocess.check_call(["git", "-C", str(src_dir), "commit", "-m", "modify routing.txt"])
        head_sha = (
            subprocess.check_output(["git", "-C", str(src_dir), "rev-parse", "HEAD"])
            .decode()
            .strip()
        )
        subprocess.check_call(
            ["git", "-C", str(src_dir), "push", str(mirror_dir), f"{head_sha}:refs/heads/main"]
        )

        wt = tmp_path / "wt_preexisting"
        subprocess.check_call(
            ["git", "-C", str(mirror_dir), "worktree", "add", "--detach", str(wt), head_sha]
        )

        entries = [{"filename": "routing.txt", "status": "modified", "patch": None}]
        results = await verify_missing_patches_locally(
            repo=git_repo_fixture["repo"],
            worktree_path=str(wt),
            trusted_head_sha=head_sha,
            head_sha=head_sha,
            base_sha=base_sha,
            entries=entries,
        )

        res = results["routing.txt"]
        assert res.verified is True
        assert len(res.routing_lines) == 1
        assert "ARGUS_NO_BENCH_OVERRIDES=1" in res.routing_lines[0]

    @pytest.mark.asyncio
    async def test_plus_plus_routing_line_blocks(
        self, git_repo_fixture: dict[str, Any], tmp_path: Path
    ) -> None:
        """A line starting with '+++ARGUS_BENCH_FILE' in blob is detected (no +++ skip hole)."""
        src_dir = git_repo_fixture["src_dir"]
        mirror_dir = git_repo_fixture["mirror_dir"]
        base_sha = git_repo_fixture["base_sha"]

        (src_dir / "patch.diff").write_text("+++ARGUS_BENCH_FILE=override\n", encoding="utf-8")
        subprocess.check_call(["git", "-C", str(src_dir), "add", "patch.diff"])
        subprocess.check_call(["git", "-C", str(src_dir), "commit", "-m", "add patch.diff"])
        head_sha = (
            subprocess.check_output(["git", "-C", str(src_dir), "rev-parse", "HEAD"])
            .decode()
            .strip()
        )
        subprocess.check_call(
            ["git", "-C", str(src_dir), "push", str(mirror_dir), f"{head_sha}:refs/heads/main"]
        )

        wt = tmp_path / "wt_plusplus"
        subprocess.check_call(
            ["git", "-C", str(mirror_dir), "worktree", "add", "--detach", str(wt), head_sha]
        )

        entries = [{"filename": "patch.diff", "status": "added", "patch": None}]
        results = await verify_missing_patches_locally(
            repo=git_repo_fixture["repo"],
            worktree_path=str(wt),
            trusted_head_sha=head_sha,
            head_sha=head_sha,
            base_sha=base_sha,
            entries=entries,
        )

        res = results["patch.diff"]
        assert res.verified is True
        assert any("ARGUS_BENCH_FILE" in line for line in res.routing_lines)

    @pytest.mark.asyncio
    async def test_binary_and_utf16_content_fails_closed(
        self, git_repo_fixture: dict[str, Any], tmp_path: Path
    ) -> None:
        """Files containing NUL bytes (e.g. binary assets or UTF-16) fail closed with 'binary'."""
        src_dir = git_repo_fixture["src_dir"]
        mirror_dir = git_repo_fixture["mirror_dir"]
        base_sha = git_repo_fixture["base_sha"]

        (src_dir / "binary.bin").write_bytes(b"\x00\x01\x02\x03\x04")
        (src_dir / "utf16.txt").write_bytes("hello world".encode("utf-16"))
        subprocess.check_call(["git", "-C", str(src_dir), "add", "."])
        subprocess.check_call(["git", "-C", str(src_dir), "commit", "-m", "add binary files"])
        head_sha = (
            subprocess.check_output(["git", "-C", str(src_dir), "rev-parse", "HEAD"])
            .decode()
            .strip()
        )
        subprocess.check_call(
            ["git", "-C", str(src_dir), "push", str(mirror_dir), f"{head_sha}:refs/heads/main"]
        )

        wt = tmp_path / "wt_binary"
        subprocess.check_call(
            ["git", "-C", str(mirror_dir), "worktree", "add", "--detach", str(wt), head_sha]
        )

        entries = [
            {"filename": "binary.bin", "status": "added", "patch": None},
            {"filename": "utf16.txt", "status": "added", "patch": None},
        ]
        results = await verify_missing_patches_locally(
            repo=git_repo_fixture["repo"],
            worktree_path=str(wt),
            trusted_head_sha=head_sha,
            head_sha=head_sha,
            base_sha=base_sha,
            entries=entries,
        )

        assert results["binary.bin"].verified is False
        assert results["binary.bin"].reason == "binary"
        assert results["utf16.txt"].verified is False
        assert results["utf16.txt"].reason == "binary"

    @pytest.mark.asyncio
    async def test_empty_blob_is_clean(
        self, git_repo_fixture: dict[str, Any], tmp_path: Path
    ) -> None:
        """An empty file (size 0) verifies as clean."""
        src_dir = git_repo_fixture["src_dir"]
        mirror_dir = git_repo_fixture["mirror_dir"]
        base_sha = git_repo_fixture["base_sha"]

        (src_dir / "empty.txt").write_text("", encoding="utf-8")
        subprocess.check_call(["git", "-C", str(src_dir), "add", "empty.txt"])
        subprocess.check_call(["git", "-C", str(src_dir), "commit", "-m", "add empty file"])
        head_sha = (
            subprocess.check_output(["git", "-C", str(src_dir), "rev-parse", "HEAD"])
            .decode()
            .strip()
        )
        subprocess.check_call(
            ["git", "-C", str(src_dir), "push", str(mirror_dir), f"{head_sha}:refs/heads/main"]
        )

        wt = tmp_path / "wt_empty"
        subprocess.check_call(
            ["git", "-C", str(mirror_dir), "worktree", "add", "--detach", str(wt), head_sha]
        )

        entries = [{"filename": "empty.txt", "status": "added", "patch": None}]
        results = await verify_missing_patches_locally(
            repo=git_repo_fixture["repo"],
            worktree_path=str(wt),
            trusted_head_sha=head_sha,
            head_sha=head_sha,
            base_sha=base_sha,
            entries=entries,
        )

        assert results["empty.txt"].verified is True
        assert results["empty.txt"].routing_lines == ()
        assert results["empty.txt"].reason is None

    @pytest.mark.asyncio
    async def test_renames_handled(self, git_repo_fixture: dict[str, Any], tmp_path: Path) -> None:
        """Renamed file verifying deletion of old path and addition of new path."""
        src_dir = git_repo_fixture["src_dir"]
        mirror_dir = git_repo_fixture["mirror_dir"]
        base_sha = git_repo_fixture["base_sha"]

        subprocess.check_call(["git", "-C", str(src_dir), "mv", "file1.txt", "file1_renamed.txt"])
        subprocess.check_call(["git", "-C", str(src_dir), "commit", "-m", "rename file1"])
        head_sha = (
            subprocess.check_output(["git", "-C", str(src_dir), "rev-parse", "HEAD"])
            .decode()
            .strip()
        )
        subprocess.check_call(
            ["git", "-C", str(src_dir), "push", str(mirror_dir), f"{head_sha}:refs/heads/main"]
        )

        wt = tmp_path / "wt_rename"
        subprocess.check_call(
            ["git", "-C", str(mirror_dir), "worktree", "add", "--detach", str(wt), head_sha]
        )

        entries = [
            {
                "filename": "file1_renamed.txt",
                "previous_filename": "file1.txt",
                "status": "renamed",
                "patch": None,
            }
        ]
        results = await verify_missing_patches_locally(
            repo=git_repo_fixture["repo"],
            worktree_path=str(wt),
            trusted_head_sha=head_sha,
            head_sha=head_sha,
            base_sha=base_sha,
            entries=entries,
        )

        assert results["file1_renamed.txt"].verified is True
        assert results["file1_renamed.txt"].routing_lines == ()

    @pytest.mark.asyncio
    async def test_blob_sha_mismatch(self, git_repo_fixture: dict[str, Any]) -> None:
        """If GitHub provides a blob SHA that does not match the git object, fail with blob_sha_mismatch."""
        repo = git_repo_fixture["repo"]
        wt = git_repo_fixture["worktree_dir"]
        head_sha = git_repo_fixture["head_sha"]
        base_sha = git_repo_fixture["base_sha"]

        entries = [
            {
                "filename": "file1.txt",
                "status": "modified",
                "patch": None,
                "sha": "1" * 40,  # Deliberate mismatch
            }
        ]
        results = await verify_missing_patches_locally(
            repo=repo,
            worktree_path=wt,
            trusted_head_sha=head_sha,
            head_sha=head_sha,
            base_sha=base_sha,
            entries=entries,
        )

        assert results["file1.txt"].verified is False
        assert results["file1.txt"].reason == "blob_sha_mismatch"

    @pytest.mark.asyncio
    async def test_untrusted_head_sha(self, git_repo_fixture: dict[str, Any]) -> None:
        """Mismatched or unpinned head_sha fails with head_untrusted."""
        repo = git_repo_fixture["repo"]
        wt = git_repo_fixture["worktree_dir"]
        head_sha = git_repo_fixture["head_sha"]
        base_sha = git_repo_fixture["base_sha"]

        entries = [{"filename": "file1.txt", "status": "modified", "patch": None}]
        results = await verify_missing_patches_locally(
            repo=repo,
            worktree_path=wt,
            trusted_head_sha="0" * 40,
            head_sha=head_sha,
            base_sha=base_sha,
            entries=entries,
        )
        assert results["file1.txt"].reason == "head_untrusted"

    @pytest.mark.asyncio
    async def test_unpinned_base_sha(self, git_repo_fixture: dict[str, Any]) -> None:
        """Unpinned base_sha or base == head fails with base_unpinned."""
        repo = git_repo_fixture["repo"]
        wt = git_repo_fixture["worktree_dir"]
        head_sha = git_repo_fixture["head_sha"]

        entries = [{"filename": "file1.txt", "status": "modified", "patch": None}]
        results = await verify_missing_patches_locally(
            repo=repo,
            worktree_path=wt,
            trusted_head_sha=head_sha,
            head_sha=head_sha,
            base_sha=head_sha,  # base == head
            entries=entries,
        )
        assert results["file1.txt"].reason == "base_unpinned"

    @pytest.mark.asyncio
    async def test_no_worktree_and_worktree_mismatch(
        self, git_repo_fixture: dict[str, Any], tmp_path: Path
    ) -> None:
        """Worktree missing or HEAD mismatch fails with no_worktree / worktree_mismatch."""
        repo = git_repo_fixture["repo"]
        head_sha = git_repo_fixture["head_sha"]
        base_sha = git_repo_fixture["base_sha"]
        entries = [{"filename": "file1.txt", "status": "modified", "patch": None}]

        # None / empty worktree
        res_none = await verify_missing_patches_locally(
            repo=repo,
            worktree_path=None,
            trusted_head_sha=head_sha,
            head_sha=head_sha,
            base_sha=base_sha,
            entries=entries,
        )
        assert res_none["file1.txt"].reason == "no_worktree"

        # Nonexistent path
        res_nonexist = await verify_missing_patches_locally(
            repo=repo,
            worktree_path=str(tmp_path / "nonexistent"),
            trusted_head_sha=head_sha,
            head_sha=head_sha,
            base_sha=base_sha,
            entries=entries,
        )
        assert res_nonexist["file1.txt"].reason == "no_worktree"

        # Non-absolute path
        res_rel = await verify_missing_patches_locally(
            repo=repo,
            worktree_path="relative/path",
            trusted_head_sha=head_sha,
            head_sha=head_sha,
            base_sha=base_sha,
            entries=entries,
        )
        assert res_rel["file1.txt"].reason == "worktree_mismatch"

    @pytest.mark.asyncio
    async def test_not_ancestor(self, git_repo_fixture: dict[str, Any], tmp_path: Path) -> None:
        """When base is not an ancestor of head, fails with not_ancestor."""
        src_dir = git_repo_fixture["src_dir"]
        mirror_dir = git_repo_fixture["mirror_dir"]
        wt = git_repo_fixture["worktree_dir"]
        head_sha = git_repo_fixture["head_sha"]

        # Create an orphan commit in src_dir
        subprocess.check_call(
            ["git", "-C", str(src_dir), "checkout", "--orphan", "unrelated-branch"]
        )
        (src_dir / "unrelated.txt").write_text("unrelated\n", encoding="utf-8")
        subprocess.check_call(["git", "-C", str(src_dir), "add", "unrelated.txt"])
        subprocess.check_call(["git", "-C", str(src_dir), "commit", "-m", "unrelated"])
        unrelated_sha = (
            subprocess.check_output(["git", "-C", str(src_dir), "rev-parse", "HEAD"])
            .decode()
            .strip()
        )
        subprocess.check_call(
            [
                "git",
                "-C",
                str(src_dir),
                "push",
                str(mirror_dir),
                f"{unrelated_sha}:refs/heads/unrelated",
            ]
        )

        entries = [{"filename": "file1.txt", "status": "modified", "patch": None}]
        results = await verify_missing_patches_locally(
            repo=git_repo_fixture["repo"],
            worktree_path=wt,
            trusted_head_sha=head_sha,
            head_sha=head_sha,
            base_sha=unrelated_sha,
            entries=entries,
        )
        assert results["file1.txt"].reason == "not_ancestor"

    @pytest.mark.asyncio
    async def test_unknown_commit_object_missing(self, git_repo_fixture: dict[str, Any]) -> None:
        """If base or head object does not exist in mirror, fails with object_missing."""
        repo = git_repo_fixture["repo"]
        wt = git_repo_fixture["worktree_dir"]
        head_sha = git_repo_fixture["head_sha"]
        missing_base = "f" * 40

        entries = [{"filename": "file1.txt", "status": "modified", "patch": None}]
        results = await verify_missing_patches_locally(
            repo=repo,
            worktree_path=wt,
            trusted_head_sha=head_sha,
            head_sha=head_sha,
            base_sha=missing_base,
            entries=entries,
        )
        assert results["file1.txt"].reason == "object_missing"

    @pytest.mark.asyncio
    async def test_path_invalid_and_status_unsupported(
        self, git_repo_fixture: dict[str, Any]
    ) -> None:
        """Invalid paths and unsupported statuses return proper fixed codes."""
        repo = git_repo_fixture["repo"]
        wt = git_repo_fixture["worktree_dir"]
        head_sha = git_repo_fixture["head_sha"]
        base_sha = git_repo_fixture["base_sha"]

        entries = [
            {"filename": ".git/config", "status": "modified", "patch": None},
            {"filename": "normal.txt", "status": "deleted", "patch": None},
            {"filename": "renamed.txt", "status": "renamed", "patch": None},  # missing prev
        ]
        results = await verify_missing_patches_locally(
            repo=repo,
            worktree_path=wt,
            trusted_head_sha=head_sha,
            head_sha=head_sha,
            base_sha=base_sha,
            entries=entries,
        )
        assert results[".git/config"].reason == "path_invalid"
        assert results["normal.txt"].reason == "status_unsupported"
        assert results["renamed.txt"].reason == "status_unsupported"

    @pytest.mark.asyncio
    async def test_too_many_files_cap_32(self, git_repo_fixture: dict[str, Any]) -> None:
        """Passing more than 32 files fails all with too_many_files."""
        repo = git_repo_fixture["repo"]
        wt = git_repo_fixture["worktree_dir"]
        head_sha = git_repo_fixture["head_sha"]
        base_sha = git_repo_fixture["base_sha"]

        entries = [
            {"filename": f"file_{i}.txt", "status": "modified", "patch": None}
            for i in range(MAX_MISSING_FILES + 1)
        ]
        results = await verify_missing_patches_locally(
            repo=repo,
            worktree_path=wt,
            trusted_head_sha=head_sha,
            head_sha=head_sha,
            base_sha=base_sha,
            entries=entries,
        )
        assert len(results) == 33
        assert all(r.reason == "too_many_files" for r in results.values())

    @pytest.mark.asyncio
    async def test_replace_ref_cannot_suppress_routing(
        self, git_repo_fixture: dict[str, Any], tmp_path: Path
    ) -> None:
        """git replace refs cannot hide routing lines due to --no-replace-objects and GIT_NO_REPLACE_OBJECTS=1."""
        src_dir = git_repo_fixture["src_dir"]
        mirror_dir = git_repo_fixture["mirror_dir"]
        base_sha = git_repo_fixture["base_sha"]

        (src_dir / "target.py").write_text("ARGUS_BENCH_FILE = 'evil'\n", encoding="utf-8")
        subprocess.check_call(["git", "-C", str(src_dir), "add", "target.py"])
        subprocess.check_call(["git", "-C", str(src_dir), "commit", "-m", "add target.py"])
        head_sha = (
            subprocess.check_output(["git", "-C", str(src_dir), "rev-parse", "HEAD"])
            .decode()
            .strip()
        )
        subprocess.check_call(
            ["git", "-C", str(src_dir), "push", str(mirror_dir), f"{head_sha}:refs/heads/main"]
        )

        wt = tmp_path / "wt_replace"
        subprocess.check_call(
            ["git", "-C", str(mirror_dir), "worktree", "add", "--detach", str(wt), head_sha]
        )

        # Create a clean blob and a replace ref pointing to it in mirror
        clean_sha = (
            subprocess.check_output(
                ["git", "-C", str(mirror_dir), "hash-object", "-w", "--stdin"],
                input=b"# clean\n",
            )
            .decode()
            .strip()
        )
        bad_blob_sha = (
            subprocess.check_output(
                ["git", "-C", str(mirror_dir), "rev-parse", f"{head_sha}:target.py"]
            )
            .decode()
            .strip()
        )
        subprocess.check_call(["git", "-C", str(mirror_dir), "replace", bad_blob_sha, clean_sha])

        # Standard git cat-file would see the replaced object:
        replaced_content = subprocess.check_output(
            ["git", "-C", str(mirror_dir), "cat-file", "blob", bad_blob_sha]
        )
        assert b"# clean\n" in replaced_content

        # But verify_missing_patches_locally disables replace objects and detects routing!
        entries = [{"filename": "target.py", "status": "added", "patch": None}]
        results = await verify_missing_patches_locally(
            repo=git_repo_fixture["repo"],
            worktree_path=str(wt),
            trusted_head_sha=head_sha,
            head_sha=head_sha,
            base_sha=base_sha,
            entries=entries,
        )

        res = results["target.py"]
        assert res.verified is True
        assert len(res.routing_lines) == 1
        assert "ARGUS_BENCH_FILE = 'evil'" in res.routing_lines[0]

    @pytest.mark.asyncio
    async def test_literal_paths_spaces_newlines_unicode_hyphens_globs(
        self, git_repo_fixture: dict[str, Any], tmp_path: Path
    ) -> None:
        """Arbitrary valid filenames (spaces, newlines, unicode, hyphens, glob characters) are handled."""
        src_dir = git_repo_fixture["src_dir"]
        mirror_dir = git_repo_fixture["mirror_dir"]
        base_sha = git_repo_fixture["base_sha"]

        filenames = [
            "path with spaces.txt",
            "path\nwith\nnewlines.txt",
            "unicode_🚀_accentué.txt",
            "-leading-hyphen.txt",
            "glob_*_?[a-z].txt",
        ]

        for fn in filenames:
            (src_dir / fn).write_text("clean line\n", encoding="utf-8")
            subprocess.check_call(["git", "-C", str(src_dir), "add", "--", fn])

        subprocess.check_call(["git", "-C", str(src_dir), "commit", "-m", "add odd path files"])
        head_sha = (
            subprocess.check_output(["git", "-C", str(src_dir), "rev-parse", "HEAD"])
            .decode()
            .strip()
        )
        subprocess.check_call(
            ["git", "-C", str(src_dir), "push", str(mirror_dir), f"{head_sha}:refs/heads/main"]
        )

        wt = tmp_path / "wt_odd_paths"
        subprocess.check_call(
            ["git", "-C", str(mirror_dir), "worktree", "add", "--detach", str(wt), head_sha]
        )

        entries = [{"filename": fn, "status": "added", "patch": None} for fn in filenames]
        results = await verify_missing_patches_locally(
            repo=git_repo_fixture["repo"],
            worktree_path=str(wt),
            trusted_head_sha=head_sha,
            head_sha=head_sha,
            base_sha=base_sha,
            entries=entries,
        )

        for fn in filenames:
            assert results[fn].verified is True
            assert results[fn].routing_lines == ()

    @pytest.mark.asyncio
    async def test_control_character_cleaning_and_cap_160(
        self, git_repo_fixture: dict[str, Any], tmp_path: Path
    ) -> None:
        """Matched routing lines have control characters and backticks removed, and capped at 160."""
        src_dir = git_repo_fixture["src_dir"]
        mirror_dir = git_repo_fixture["mirror_dir"]
        base_sha = git_repo_fixture["base_sha"]

        long_line = (
            "`ARGUS_BENCH_FILE`\x07\x1b"
            + " = 'very_long_path_padding_to_exceed_160_characters_"
            + "x" * 200
            + "'\n"
        )
        (src_dir / "control.txt").write_text(long_line, encoding="utf-8")
        subprocess.check_call(["git", "-C", str(src_dir), "add", "control.txt"])
        subprocess.check_call(["git", "-C", str(src_dir), "commit", "-m", "add control.txt"])
        head_sha = (
            subprocess.check_output(["git", "-C", str(src_dir), "rev-parse", "HEAD"])
            .decode()
            .strip()
        )
        subprocess.check_call(
            ["git", "-C", str(src_dir), "push", str(mirror_dir), f"{head_sha}:refs/heads/main"]
        )

        wt = tmp_path / "wt_ctrl"
        subprocess.check_call(
            ["git", "-C", str(mirror_dir), "worktree", "add", "--detach", str(wt), head_sha]
        )

        entries = [{"filename": "control.txt", "status": "added", "patch": None}]
        results = await verify_missing_patches_locally(
            repo=git_repo_fixture["repo"],
            worktree_path=str(wt),
            trusted_head_sha=head_sha,
            head_sha=head_sha,
            base_sha=base_sha,
            entries=entries,
        )

        res = results["control.txt"]
        assert res.verified is True
        assert len(res.routing_lines) == 1
        line = res.routing_lines[0]
        assert len(line) <= 160
        assert "`" not in line
        assert "\x07" not in line
        assert "\x1b" not in line

    @pytest.mark.asyncio
    async def test_oversize_blob_fails_closed(self, git_repo_fixture: dict[str, Any]) -> None:
        """Blob size exceeding 8 MiB fails closed with oversize."""
        repo = git_repo_fixture["repo"]
        wt = git_repo_fixture["worktree_dir"]
        head_sha = git_repo_fixture["head_sha"]
        base_sha = git_repo_fixture["base_sha"]

        entries = [{"filename": "file1.txt", "status": "modified", "patch": None}]

        # Mock cat-file -s to report > 8 MiB (8 * 1024 * 1024 + 1 = 8388609)
        orig_run_bounded = _run_bounded_git

        async def _mock_run(cmd: list[str], **kwargs: Any) -> tuple[int, bytes]:
            if "cat-file" in cmd and "-s" in cmd:
                return 0, b"8388609\n"
            return await orig_run_bounded(cmd, **kwargs)

        with patch("argus.bench_local_verify._run_bounded_git", side_effect=_mock_run):
            results = await verify_missing_patches_locally(
                repo=repo,
                worktree_path=wt,
                trusted_head_sha=head_sha,
                head_sha=head_sha,
                base_sha=base_sha,
                entries=entries,
            )

        assert results["file1.txt"].verified is False
        assert results["file1.txt"].reason == "oversize"

    @pytest.mark.asyncio
    async def test_cumulative_budget_exceeded(self, git_repo_fixture: dict[str, Any]) -> None:
        """When cumulative blob size across batch exceeds 32 MiB, subsequent files fail with budget_exceeded."""
        repo = git_repo_fixture["repo"]
        wt = git_repo_fixture["worktree_dir"]
        head_sha = git_repo_fixture["head_sha"]
        base_sha = git_repo_fixture["base_sha"]

        entries = [
            {"filename": "file1.txt", "status": "modified", "patch": None},
            {"filename": "clean_file.txt", "status": "added", "patch": None},
        ]

        # First blob 7 MiB (7340032 <= 8 MiB), second blob 26 MiB -> cumulative 33 MiB > 32 MiB
        # But per-blob cap is 8 MiB, so second blob must also be <= 8 MiB if we test 5 blobs of 7 MiB!
        # Let's create entries for 6 files of 7 MiB each
        entries = [
            {"filename": f"file_{i}.txt", "status": "added", "patch": None} for i in range(6)
        ]

        orig_run_bounded = _run_bounded_git

        async def _mock_run(cmd: list[str], **kwargs: Any) -> tuple[int, bytes]:
            if "diff" in cmd and "--raw" in cmd:
                # Return fake diff-raw records for all 6 files
                records = b"".join(
                    f":000000 100644 {'0' * 40} {'1' * 40} A\x00file_{i}.txt\x00".encode()
                    for i in range(6)
                )
                return 0, records
            if "cat-file" in cmd and "-s" in cmd:
                return 0, b"7340032\n"  # 7 MiB
            if "cat-file" in cmd and "blob" in cmd:
                return 0, b"# clean\n" * (7340032 // 8)
            return await orig_run_bounded(cmd, **kwargs)

        with patch("argus.bench_local_verify._run_bounded_git", side_effect=_mock_run):
            results = await verify_missing_patches_locally(
                repo=repo,
                worktree_path=wt,
                trusted_head_sha=head_sha,
                head_sha=head_sha,
                base_sha=base_sha,
                entries=entries,
            )

        # 4 files * 7 MiB = 28 MiB (verified)
        for i in range(4):
            assert results[f"file_{i}.txt"].verified is True
        # 5th file: 28 + 7 = 35 MiB > 32 MiB -> budget_exceeded
        assert results["file_4.txt"].verified is False
        assert results["file_4.txt"].reason == "budget_exceeded"
        assert results["file_5.txt"].verified is False
        assert results["file_5.txt"].reason == "budget_exceeded"

    @pytest.mark.asyncio
    async def test_non_regular_file_mode(self, git_repo_fixture: dict[str, Any]) -> None:
        """Symlinks (120000) or submodules (160000) in diff-raw output fail with non_regular_file."""
        repo = git_repo_fixture["repo"]
        wt = git_repo_fixture["worktree_dir"]
        head_sha = git_repo_fixture["head_sha"]
        base_sha = git_repo_fixture["base_sha"]

        entries = [{"filename": "clean_file.txt", "status": "added", "patch": None}]

        orig_run_bounded = _run_bounded_git

        async def _mock_run(cmd: list[str], **kwargs: Any) -> tuple[int, bytes]:
            if "diff" in cmd and "--raw" in cmd:
                # Return mode 120000 (symlink)
                fake_raw = (
                    b":000000 120000 0000000000000000000000000000000000000000 "
                    b"1234567890abcdef1234567890abcdef12345678 A\x00clean_file.txt\x00"
                )
                return 0, fake_raw
            return await orig_run_bounded(cmd, **kwargs)

        with patch("argus.bench_local_verify._run_bounded_git", side_effect=_mock_run):
            results = await verify_missing_patches_locally(
                repo=repo,
                worktree_path=wt,
                trusted_head_sha=head_sha,
                head_sha=head_sha,
                base_sha=base_sha,
                entries=entries,
            )

        assert results["clean_file.txt"].verified is False
        assert results["clean_file.txt"].reason == "non_regular_file"

    @pytest.mark.asyncio
    async def test_status_mismatch_wrong_status(self, git_repo_fixture: dict[str, Any]) -> None:
        """If GitHub claims modified but git diff-raw shows added, fail with status_mismatch."""
        repo = git_repo_fixture["repo"]
        wt = git_repo_fixture["worktree_dir"]
        head_sha = git_repo_fixture["head_sha"]
        base_sha = git_repo_fixture["base_sha"]

        # clean_file.txt was added in HEAD, but entry claims modified
        entries = [{"filename": "clean_file.txt", "status": "modified", "patch": None}]
        results = await verify_missing_patches_locally(
            repo=repo,
            worktree_path=wt,
            trusted_head_sha=head_sha,
            head_sha=head_sha,
            base_sha=base_sha,
            entries=entries,
        )

        assert results["clean_file.txt"].verified is False
        assert results["clean_file.txt"].reason == "status_mismatch"


class TestGraphFetchDiffIntegration:
    """Integration tests for _node_fetch_diff calling verify_missing_patches_locally."""

    @pytest.mark.asyncio
    async def test_clean_missing_patch_unconfirmed_cleared(
        self, git_repo_fixture: dict[str, Any]
    ) -> None:
        """When local verification succeeds cleanly, bench_config_unconfirmed remains None."""
        repo = git_repo_fixture["repo"]
        wt = git_repo_fixture["worktree_dir"]
        head_sha = git_repo_fixture["head_sha"]
        base_sha = git_repo_fixture["base_sha"]

        req = ReviewRequest(repo=repo, pr_number=100, sha=None, base_ref=None)
        diff = "diff --git a/file1.txt b/file1.txt\n"

        mock_gh = MagicMock()
        mock_gh.get_compare_files.return_value = (
            [
                {"filename": "file1.txt", "status": "modified", "patch": None},
                {"filename": "clean_file.txt", "status": "added", "patch": None},
            ],
            False,
        )

        state = {"request": req.model_dump()}
        config: dict[str, Any] = {
            "configurable": {
                "worktree_path": wt,
                "head_sha": head_sha,
            }
        }

        with (
            patch("argus.graph._fetch_prior_review", new_callable=AsyncMock, return_value=None),
            patch("argus.graph._fetch_dismissed_findings", new_callable=AsyncMock, return_value=[]),
            patch(
                "argus.graph._fetch_pr_diff_and_description",
                new_callable=AsyncMock,
                return_value=(diff, "desc", head_sha, "main", base_sha),
            ),
            patch("argus.github_client.GitHubClient", return_value=mock_gh),
        ):
            node_result = await _node_fetch_diff(cast(Any, state), cast(Any, config))

        assert node_result["bench_config_unconfirmed"] is None
        assert node_result["bench_config_changes"] == []

    @pytest.mark.asyncio
    async def test_verified_routing_adds_present_in_verified_head_content(
        self, git_repo_fixture: dict[str, Any], tmp_path: Path
    ) -> None:
        """When local verification detects routing lines, it records them with 'present in verified head content'."""
        src_dir = git_repo_fixture["src_dir"]
        mirror_dir = git_repo_fixture["mirror_dir"]
        base_sha = git_repo_fixture["base_sha"]

        (src_dir / "uv.lock").write_text("ARGUS_BENCH_FILE = 'override'\n", encoding="utf-8")
        subprocess.check_call(["git", "-C", str(src_dir), "add", "uv.lock"])
        subprocess.check_call(["git", "-C", str(src_dir), "commit", "-m", "add uv.lock"])
        head_sha = (
            subprocess.check_output(["git", "-C", str(src_dir), "rev-parse", "HEAD"])
            .decode()
            .strip()
        )
        subprocess.check_call(
            ["git", "-C", str(src_dir), "push", str(mirror_dir), f"{head_sha}:refs/heads/main"]
        )

        wt = tmp_path / "wt_graph_routing"
        subprocess.check_call(
            ["git", "-C", str(mirror_dir), "worktree", "add", "--detach", str(wt), head_sha]
        )

        req = ReviewRequest(repo=git_repo_fixture["repo"], pr_number=100, sha=None, base_ref=None)
        diff = "diff --git a/uv.lock b/uv.lock\n"

        mock_gh = MagicMock()
        mock_gh.get_compare_files.return_value = (
            [{"filename": "uv.lock", "status": "added", "patch": None}],
            False,
        )

        state = {"request": req.model_dump()}
        config: dict[str, Any] = {
            "configurable": {
                "worktree_path": str(wt),
                "head_sha": head_sha,
            }
        }

        with (
            patch("argus.graph._fetch_prior_review", new_callable=AsyncMock, return_value=None),
            patch("argus.graph._fetch_dismissed_findings", new_callable=AsyncMock, return_value=[]),
            patch(
                "argus.graph._fetch_pr_diff_and_description",
                new_callable=AsyncMock,
                return_value=(diff, "desc", head_sha, "main", base_sha),
            ),
            patch("argus.github_client.GitHubClient", return_value=mock_gh),
        ):
            node_result = await _node_fetch_diff(cast(Any, state), cast(Any, config))

        assert node_result["bench_config_unconfirmed"] is None
        assert any(
            "present in verified head content modifying bench routing in uv.lock" in c
            for c in node_result["bench_config_changes"]
        )

    @pytest.mark.asyncio
    async def test_head_gitattributes_cannot_hide_routing_lines(
        self, git_repo_fixture: dict[str, Any], tmp_path: Path
    ) -> None:
        """Even if repo declares file generated in .gitattributes, local fallback detects routing lines."""
        src_dir = git_repo_fixture["src_dir"]
        mirror_dir = git_repo_fixture["mirror_dir"]
        base_sha = git_repo_fixture["base_sha"]

        (src_dir / ".gitattributes").write_text("generated.json -diff\n", encoding="utf-8")
        (src_dir / "generated.json").write_text("ARGUS_BENCH_FILE = 'pwn'\n", encoding="utf-8")
        subprocess.check_call(["git", "-C", str(src_dir), "add", "."])
        subprocess.check_call(
            ["git", "-C", str(src_dir), "commit", "-m", "add generated with routing"]
        )
        head_sha = (
            subprocess.check_output(["git", "-C", str(src_dir), "rev-parse", "HEAD"])
            .decode()
            .strip()
        )
        subprocess.check_call(
            ["git", "-C", str(src_dir), "push", str(mirror_dir), f"{head_sha}:refs/heads/main"]
        )

        wt = tmp_path / "wt_attr_routing"
        subprocess.check_call(
            ["git", "-C", str(mirror_dir), "worktree", "add", "--detach", str(wt), head_sha]
        )

        req = ReviewRequest(repo=git_repo_fixture["repo"], pr_number=100, sha=None, base_ref=None)
        diff = "diff --git a/generated.json b/generated.json\n"

        mock_gh = MagicMock()
        mock_gh.get_compare_files.return_value = (
            [{"filename": "generated.json", "status": "added", "patch": None}],
            False,
        )

        state = {"request": req.model_dump()}
        config: dict[str, Any] = {
            "configurable": {
                "worktree_path": str(wt),
                "head_sha": head_sha,
            }
        }

        with (
            patch("argus.graph._fetch_prior_review", new_callable=AsyncMock, return_value=None),
            patch("argus.graph._fetch_dismissed_findings", new_callable=AsyncMock, return_value=[]),
            patch(
                "argus.graph._fetch_pr_diff_and_description",
                new_callable=AsyncMock,
                return_value=(diff, "desc", head_sha, "main", base_sha),
            ),
            patch("argus.github_client.GitHubClient", return_value=mock_gh),
        ):
            node_result = await _node_fetch_diff(cast(Any, state), cast(Any, config))

        # It must block with routing line detected, NOT be silently exempted by .gitattributes!
        assert any(
            "present in verified head content modifying bench routing in generated.json" in c
            for c in node_result["bench_config_changes"]
        )

    @pytest.mark.asyncio
    async def test_prior_api_truncation_uncertainty_not_cleared(
        self, git_repo_fixture: dict[str, Any]
    ) -> None:
        """If GitHub API file comparison was truncated (e.g. 3000 cap), fallback is not called and uncertainty stays."""
        repo = git_repo_fixture["repo"]
        wt = git_repo_fixture["worktree_dir"]
        head_sha = git_repo_fixture["head_sha"]
        base_sha = git_repo_fixture["base_sha"]

        req = ReviewRequest(repo=repo, pr_number=100, sha=None, base_ref=None)
        diff = "diff --git a/file1.txt b/file1.txt\n"

        mock_gh = MagicMock()
        mock_gh.get_compare_files.return_value = (
            [{"filename": "file1.txt", "status": "modified", "patch": None}],
            True,  # Truncated!
        )
        mock_gh.get_pr_files.return_value = (
            [{"filename": "file1.txt", "status": "modified", "patch": None}],
            True,  # Still truncated!
        )

        state = {"request": req.model_dump()}
        config: dict[str, Any] = {
            "configurable": {
                "worktree_path": wt,
                "head_sha": head_sha,
            }
        }

        with (
            patch("argus.graph._fetch_prior_review", new_callable=AsyncMock, return_value=None),
            patch("argus.graph._fetch_dismissed_findings", new_callable=AsyncMock, return_value=[]),
            patch(
                "argus.graph._fetch_pr_diff_and_description",
                new_callable=AsyncMock,
                return_value=(diff, "desc", head_sha, "main", base_sha),
            ),
            patch("argus.github_client.GitHubClient", return_value=mock_gh),
            patch(
                "argus.bench_local_verify.verify_missing_patches_locally",
                new_callable=AsyncMock,
            ) as mock_local_verify,
        ):
            node_result = await _node_fetch_diff(cast(Any, state), cast(Any, config))

        mock_local_verify.assert_not_called()
        assert node_result["bench_config_unconfirmed"] is not None
        assert "exceeded maximum file limit" in node_result["bench_config_unconfirmed"]

    @pytest.mark.asyncio
    async def test_round2_passes_scope_base_sha_not_prior_sha(
        self, git_repo_fixture: dict[str, Any]
    ) -> None:
        """On round 2+, _node_fetch_diff passes scope_base_sha, NOT prior_sha, to local verify."""
        repo = git_repo_fixture["repo"]
        wt = git_repo_fixture["worktree_dir"]
        head_sha = git_repo_fixture["head_sha"]
        base_sha = git_repo_fixture["base_sha"]
        prior_sha = "1" * 40

        mock_prior = MagicMock()
        mock_prior.findings = []
        mock_prior.dismissed_findings = []
        mock_prior.model_dump.return_value = {"reviewed_sha": prior_sha}
        mock_prior.review_id = "rev-123"

        req = ReviewRequest(repo=repo, pr_number=100, sha=None, base_ref=None)
        diff = "diff --git a/file1.txt b/file1.txt\n"

        mock_gh = MagicMock()
        mock_gh.get_compare_files.return_value = (
            [{"filename": "file1.txt", "status": "modified", "patch": None}],
            False,
        )

        state = {"request": req.model_dump()}
        config: dict[str, Any] = {
            "configurable": {
                "worktree_path": wt,
                "head_sha": head_sha,
            }
        }

        with (
            patch(
                "argus.graph._fetch_prior_review", new_callable=AsyncMock, return_value=mock_prior
            ),
            patch("argus.graph._fetch_dismissed_findings", new_callable=AsyncMock, return_value=[]),
            patch(
                "argus.graph._fetch_pr_diff_and_description",
                new_callable=AsyncMock,
                return_value=(diff, "desc", head_sha, "main", base_sha),
            ),
            patch("argus.github_client.GitHubClient", return_value=mock_gh),
            patch(
                "argus.bench_local_verify.verify_missing_patches_locally",
                new_callable=AsyncMock,
                return_value={"file1.txt": LocalScanResult(verified=True)},
            ) as mock_local_verify,
        ):
            await _node_fetch_diff(cast(Any, state), cast(Any, config))

        mock_local_verify.assert_called_once()
        kwargs = mock_local_verify.call_args.kwargs
        assert kwargs["base_sha"] == base_sha
        assert kwargs["base_sha"] != prior_sha

    @pytest.mark.asyncio
    async def test_oversize_legacy_generated_exemption_in_graph(
        self, git_repo_fixture: dict[str, Any]
    ) -> None:
        """When fallback fails (e.g. oversize), .gitattributes generated check exempts it as last resort."""
        repo = git_repo_fixture["repo"]
        wt = git_repo_fixture["worktree_dir"]
        head_sha = git_repo_fixture["head_sha"]
        base_sha = git_repo_fixture["base_sha"]

        req = ReviewRequest(repo=repo, pr_number=100, sha=None, base_ref=None)
        diff = "diff --git a/migrations/snapshot.json b/migrations/snapshot.json\n"

        mock_gh = MagicMock()
        mock_gh.get_compare_files.return_value = (
            [{"filename": "migrations/snapshot.json", "status": "added", "patch": None}],
            False,
        )

        state = {"request": req.model_dump()}
        config: dict[str, Any] = {
            "configurable": {
                "worktree_path": wt,
                "head_sha": head_sha,
            }
        }

        with (
            patch("argus.graph._fetch_prior_review", new_callable=AsyncMock, return_value=None),
            patch("argus.graph._fetch_dismissed_findings", new_callable=AsyncMock, return_value=[]),
            patch(
                "argus.graph._fetch_pr_diff_and_description",
                new_callable=AsyncMock,
                return_value=(diff, "desc", head_sha, "main", base_sha),
            ),
            patch("argus.github_client.GitHubClient", return_value=mock_gh),
            patch(
                "argus.bench_local_verify.verify_missing_patches_locally",
                new_callable=AsyncMock,
                return_value={
                    "migrations/snapshot.json": LocalScanResult(verified=False, reason="oversize")
                },
            ),
            patch(
                "argus.graph._is_generated_nodiff_path", new_callable=AsyncMock, return_value=True
            ),
        ):
            node_result = await _node_fetch_diff(cast(Any, state), cast(Any, config))

        # Because _is_generated_nodiff_path returned True, unconfirmed_reason is NOT set
        assert node_result["bench_config_unconfirmed"] is None
        assert node_result["bench_config_changes"] == []


class TestBoundedSubprocessAndEnv:
    """Tests for _run_bounded_git process lifecycle, limits, and clean env."""

    @pytest.mark.asyncio
    async def test_bounded_subprocess_timeout_reaps_child(self, tmp_path: Path) -> None:
        """When a git subprocess exceeds per-command timeout, it is killed, reaped, and raises _TimeoutError."""
        cmd = ["sleep", "10"]
        with pytest.raises(_TimeoutError):
            await _run_bounded_git(
                cmd,
                cwd=str(tmp_path),
                env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                max_stdout=1024,
                timeout=0.1,
            )

    @pytest.mark.asyncio
    async def test_bounded_subprocess_oversize_stdout_reaps_child(self, tmp_path: Path) -> None:
        """When stdout exceeds max_stdout, process is killed, reaped, and raises _OversizeError."""
        cmd = ["python3", "-c", "import sys; sys.stdout.buffer.write(b'x' * 1000)"]
        with pytest.raises(_OversizeError):
            await _run_bounded_git(
                cmd,
                cwd=str(tmp_path),
                env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                max_stdout=50,
                timeout=5.0,
            )

    @pytest.mark.asyncio
    async def test_bounded_subprocess_cancel_reaps_child(self, tmp_path: Path) -> None:
        """When the task running _run_bounded_git is cancelled, it is killed, reaped, and CancelledError propagates."""
        task = asyncio.create_task(
            _run_bounded_git(
                ["sleep", "10"],
                cwd=str(tmp_path),
                env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                max_stdout=1024,
                timeout=5.0,
            )
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    def test_git_env_no_inherited_git_vars(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_build_git_env builds clean env from scratch without arbitrary GIT_* variables."""
        monkeypatch.setenv("GIT_DIR", "/some/bad/dir")
        monkeypatch.setenv("GIT_WORK_TREE", "/some/bad/wt")
        monkeypatch.setenv("GIT_REPLACE_REF_BASE", "refs/replace")

        mirror = tmp_path / "repo.git"
        env = _build_git_env(mirror)

        assert "GIT_DIR" not in env
        assert "GIT_WORK_TREE" not in env
        assert "GIT_REPLACE_REF_BASE" not in env

        assert env["LC_ALL"] == "C"
        assert env["GIT_CONFIG_NOSYSTEM"] == "1"
        assert env["GIT_CONFIG_GLOBAL"] == os.devnull
        assert env["GIT_NO_REPLACE_OBJECTS"] == "1"
        assert env["GIT_LITERAL_PATHSPECS"] == "1"
        assert env["GIT_TERMINAL_PROMPT"] == "0"
        assert env["GIT_OPTIONAL_LOCKS"] == "0"
        assert env["PAGER"] == "cat"
        assert env["GIT_ATTR_NOSYSTEM"] == "1"
        assert env["GIT_NO_LAZY_FETCH"] == "1"
        assert env["HOME"] == str(mirror.parent.resolve())


# ---------------------------------------------------------------------------
# Independent security-coverage audit additions (TECH-7768).
#
# These cases augment -- not duplicate -- the coverage above: mirror/worktree
# authentication before any blob is trusted, overall-budget fail-closed
# behavior, stderr flood and child-reap handling, untrusted-worktree-content
# isolation, non-UTF-8 git paths, literal glob component names, the
# github_client entry-field passthrough both APIs feed the fallback with, and
# graph-level wiring of the fallback's trusted inputs.
# ---------------------------------------------------------------------------


def _commit_and_advance(
    src_dir: Path, mirror_dir: Path, files: dict[str, bytes], message: str
) -> str:
    """Write ``files`` into the source repo, commit, push to the mirror, return HEAD sha."""
    # GIT_LITERAL_PATHSPECS keeps ``git add`` literal for adversarial names --
    # a file literally named "*" or "**" must not glob the whole tree here.
    add_env = dict(os.environ, GIT_LITERAL_PATHSPECS="1")
    for name, content in files.items():
        (src_dir / name).write_bytes(content)
        subprocess.check_call(["git", "-C", str(src_dir), "add", "--", name], env=add_env)
    subprocess.check_call(["git", "-C", str(src_dir), "commit", "-m", message])
    sha = subprocess.check_output(["git", "-C", str(src_dir), "rev-parse", "HEAD"]).decode().strip()
    subprocess.check_call(
        ["git", "-C", str(src_dir), "push", str(mirror_dir), f"{sha}:refs/heads/main"]
    )
    return sha


def _worktree_at(mirror_dir: Path, tmp_path: Path, name: str, sha: str) -> str:
    """Add a detached worktree at ``sha`` under ``tmp_path`` and return its path."""
    wt = tmp_path / name
    subprocess.check_call(
        ["git", "-C", str(mirror_dir), "worktree", "add", "--detach", str(wt), sha]
    )
    return str(wt)


async def _run_fetch_diff_node(
    mock_gh: MagicMock,
    *,
    diff: str,
    head_sha: str,
    base_sha: str,
    prior: Any = None,
    worktree_path: str | None = None,
    config_head_sha: str | None = None,
    local_verify_mock: AsyncMock | None = None,
) -> dict[str, Any]:
    """Invoke _node_fetch_diff with the standard network patches applied."""
    req = ReviewRequest(repo="testorg/testrepo", pr_number=100, sha=None, base_ref=None)
    state = {"request": req.model_dump()}
    config: dict[str, Any] = {
        "configurable": {
            "worktree_path": worktree_path,
            "head_sha": config_head_sha,
        }
    }
    with ExitStack() as stack:
        stack.enter_context(
            patch("argus.graph._fetch_prior_review", new_callable=AsyncMock, return_value=prior)
        )
        stack.enter_context(
            patch("argus.graph._fetch_dismissed_findings", new_callable=AsyncMock, return_value=[])
        )
        stack.enter_context(
            patch(
                "argus.graph._fetch_pr_diff_and_description",
                new_callable=AsyncMock,
                return_value=(diff, "desc", head_sha, "main", base_sha),
            )
        )
        stack.enter_context(patch("argus.github_client.GitHubClient", return_value=mock_gh))
        if local_verify_mock is not None:
            stack.enter_context(
                patch("argus.bench_local_verify.verify_missing_patches_locally", local_verify_mock)
            )
        return await _node_fetch_diff(cast(Any, state), cast(Any, config))


class TestDiffRawParsingMalformed:
    """Strict rejection of malformed ``git diff --raw -z`` metadata (fail closed)."""

    @staticmethod
    def _record(
        *,
        old_mode: str = "100644",
        new_mode: str = "100644",
        old_sha: str = "0" * 40,
        new_sha: str = "1" * 40,
        status: str = "M",
        path: bytes = b"file.txt",
    ) -> bytes:
        meta = f":{old_mode} {new_mode} {old_sha} {new_sha} {status}".encode("ascii")
        return meta + b"\x00" + path + b"\x00"

    def test_missing_colon_prefix_rejected(self) -> None:
        with pytest.raises(ValueError, match="malformed diff-raw prefix"):
            _parse_diff_raw(self._record()[1:], {"file.txt"})

    def test_wrong_field_count_rejected(self) -> None:
        raw = b":100644 100644 " + b"0" * 40 + b" M\x00file.txt\x00"
        with pytest.raises(ValueError, match="malformed diff-raw header"):
            _parse_diff_raw(raw, {"file.txt"})

    def test_malformed_modes_rejected(self) -> None:
        with pytest.raises(ValueError, match="malformed old mode"):
            _parse_diff_raw(self._record(old_mode="10064a"), {"file.txt"})
        with pytest.raises(ValueError, match="malformed new mode"):
            _parse_diff_raw(self._record(new_mode="10064"), {"file.txt"})

    def test_malformed_shas_rejected(self) -> None:
        with pytest.raises(ValueError, match="malformed old sha"):
            _parse_diff_raw(self._record(old_sha="A" * 40), {"file.txt"})
        with pytest.raises(ValueError, match="malformed new sha"):
            _parse_diff_raw(self._record(new_sha="1" * 39), {"file.txt"})

    def test_rename_score_and_non_alpha_status_rejected(self) -> None:
        # Scores (e.g. R100) only appear with rename detection, which the
        # verification command disables via --no-renames; anything other than
        # a single alpha char must be treated as hostile/truncated output.
        with pytest.raises(ValueError, match="malformed status"):
            _parse_diff_raw(self._record(status="R100"), {"file.txt"})
        with pytest.raises(ValueError, match="malformed status"):
            _parse_diff_raw(self._record(status="7"), {"file.txt"})

    def test_non_utf8_path_bytes_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid encoding in diff-raw"):
            _parse_diff_raw(self._record(path=b"bad\x80\xff.bin"), {"file.txt"})

    def test_uneven_token_count_rejected(self) -> None:
        raw = self._record() + b":orphan-meta\x00"
        with pytest.raises(ValueError, match="uneven diff-raw tokens"):
            _parse_diff_raw(raw, {"file.txt"})


class TestShaFormatValidation:
    """head/base must be exactly 40 lowercase hex characters."""

    @pytest.mark.asyncio
    async def test_uppercase_short_and_missing_shas_fail_closed(
        self, git_repo_fixture: dict[str, Any]
    ) -> None:
        entries = [{"filename": "file1.txt", "status": "modified", "patch": None}]
        wt = git_repo_fixture["worktree_dir"]
        base_sha = git_repo_fixture["base_sha"]
        common: dict[str, Any] = dict(
            repo=git_repo_fixture["repo"],
            worktree_path=wt,
            entries=entries,
        )

        res_upper = await verify_missing_patches_locally(
            trusted_head_sha="A" * 40, head_sha="A" * 40, base_sha=base_sha, **common
        )
        assert res_upper["file1.txt"].reason == "head_untrusted"

        res_short = await verify_missing_patches_locally(
            trusted_head_sha="a" * 39, head_sha="a" * 39, base_sha=base_sha, **common
        )
        assert res_short["file1.txt"].reason == "head_untrusted"

        res_upper_base = await verify_missing_patches_locally(
            trusted_head_sha="a" * 40, head_sha="a" * 40, base_sha="B" * 40, **common
        )
        assert res_upper_base["file1.txt"].reason == "base_unpinned"

        res_none_base = await verify_missing_patches_locally(
            trusted_head_sha="a" * 40, head_sha="a" * 40, base_sha=None, **common
        )
        assert res_none_base["file1.txt"].reason == "base_unpinned"


class TestMirrorAndWorktreeIntegrity:
    """The mirror/worktree pair must be authenticated before any blob is trusted."""

    @pytest.mark.asyncio
    async def test_invalid_repo_format_fails_mirror_invalid(
        self, git_repo_fixture: dict[str, Any]
    ) -> None:
        entries = [{"filename": "file1.txt", "status": "modified", "patch": None}]
        for bad_repo in ("no-slash", "../evil", "a/b/c", "/abs/repo", ""):
            results = await verify_missing_patches_locally(
                repo=bad_repo,
                worktree_path=git_repo_fixture["worktree_dir"],
                trusted_head_sha=git_repo_fixture["head_sha"],
                head_sha=git_repo_fixture["head_sha"],
                base_sha=git_repo_fixture["base_sha"],
                entries=entries,
            )
            assert results["file1.txt"].reason == "mirror_invalid", bad_repo

    @pytest.mark.asyncio
    async def test_missing_mirror_directory_fails_mirror_invalid(
        self, git_repo_fixture: dict[str, Any]
    ) -> None:
        entries = [{"filename": "file1.txt", "status": "modified", "patch": None}]
        results = await verify_missing_patches_locally(
            repo="ghostorg/ghostrepo",  # valid format, never provisioned
            worktree_path=git_repo_fixture["worktree_dir"],
            trusted_head_sha=git_repo_fixture["head_sha"],
            head_sha=git_repo_fixture["head_sha"],
            base_sha=git_repo_fixture["base_sha"],
            entries=entries,
        )
        assert results["file1.txt"].reason == "mirror_invalid"

    @pytest.mark.asyncio
    async def test_non_bare_mirror_git_dir_rejected(
        self, git_repo_fixture: dict[str, Any], tmp_path: Path
    ) -> None:
        """A valid but non-bare git dir at the mirror path must be rejected."""
        src_nb = tmp_path / "nonbare_src"
        subprocess.check_call(["git", "init", str(src_nb)])
        cache_root = git_repo_fixture["mirror_dir"].parent.parent
        mirror_nb = cache_root / "otherorg" / "nonbare.git"
        mirror_nb.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_nb / ".git", mirror_nb)  # core.bare=false git dir

        entries = [{"filename": "file1.txt", "status": "modified", "patch": None}]
        results = await verify_missing_patches_locally(
            repo="otherorg/nonbare",
            worktree_path=git_repo_fixture["worktree_dir"],
            trusted_head_sha=git_repo_fixture["head_sha"],
            head_sha=git_repo_fixture["head_sha"],
            base_sha=git_repo_fixture["base_sha"],
            entries=entries,
        )
        assert results["file1.txt"].reason == "mirror_invalid"

    @pytest.mark.asyncio
    async def test_symlinked_mirror_rejected(self, git_repo_fixture: dict[str, Any]) -> None:
        cache_root = git_repo_fixture["mirror_dir"].parent.parent
        real = cache_root / "symorg" / "realrepo.git"
        real.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["git", "init", "--bare", str(real)])
        link = cache_root / "symorg" / "symrepo.git"
        os.symlink(str(real), str(link))

        entries = [{"filename": "file1.txt", "status": "modified", "patch": None}]
        results = await verify_missing_patches_locally(
            repo="symorg/symrepo",
            worktree_path=git_repo_fixture["worktree_dir"],
            trusted_head_sha=git_repo_fixture["head_sha"],
            head_sha=git_repo_fixture["head_sha"],
            base_sha=git_repo_fixture["base_sha"],
            entries=entries,
        )
        assert results["file1.txt"].reason == "mirror_invalid"

    @pytest.mark.asyncio
    async def test_worktree_from_foreign_mirror_rejected(
        self, git_repo_fixture: dict[str, Any], tmp_path: Path
    ) -> None:
        """A worktree attached to a different mirror is not the provisioned one."""
        src_dir = git_repo_fixture["src_dir"]
        head_sha = git_repo_fixture["head_sha"]
        mirror2 = tmp_path / "foreign.git"
        subprocess.check_call(["git", "init", "--bare", str(mirror2)])
        subprocess.check_call(
            ["git", "-C", str(src_dir), "push", str(mirror2), f"{head_sha}:refs/heads/main"]
        )
        wt2 = tmp_path / "wt_foreign"
        subprocess.check_call(
            ["git", "-C", str(mirror2), "worktree", "add", "--detach", str(wt2), head_sha]
        )

        entries = [{"filename": "file1.txt", "status": "modified", "patch": None}]
        results = await verify_missing_patches_locally(
            repo=git_repo_fixture["repo"],
            worktree_path=str(wt2),
            trusted_head_sha=head_sha,
            head_sha=head_sha,
            base_sha=git_repo_fixture["base_sha"],
            entries=entries,
        )
        assert results["file1.txt"].reason == "worktree_mismatch"

    @pytest.mark.asyncio
    async def test_worktree_checked_out_at_wrong_commit_rejected(
        self, git_repo_fixture: dict[str, Any], tmp_path: Path
    ) -> None:
        """A worktree parked on the base commit is not at the trusted head."""
        wt_base = _worktree_at(
            git_repo_fixture["mirror_dir"], tmp_path, "wt_at_base", git_repo_fixture["base_sha"]
        )
        entries = [{"filename": "file1.txt", "status": "modified", "patch": None}]
        results = await verify_missing_patches_locally(
            repo=git_repo_fixture["repo"],
            worktree_path=wt_base,
            trusted_head_sha=git_repo_fixture["head_sha"],
            head_sha=git_repo_fixture["head_sha"],
            base_sha=git_repo_fixture["base_sha"],
            entries=entries,
        )
        assert results["file1.txt"].reason == "worktree_mismatch"

    @pytest.mark.asyncio
    async def test_symlinked_worktree_rejected(
        self, git_repo_fixture: dict[str, Any], tmp_path: Path
    ) -> None:
        link = tmp_path / "wt_link"
        os.symlink(git_repo_fixture["worktree_dir"], str(link))
        entries = [{"filename": "file1.txt", "status": "modified", "patch": None}]
        results = await verify_missing_patches_locally(
            repo=git_repo_fixture["repo"],
            worktree_path=str(link),
            trusted_head_sha=git_repo_fixture["head_sha"],
            head_sha=git_repo_fixture["head_sha"],
            base_sha=git_repo_fixture["base_sha"],
            entries=entries,
        )
        assert results["file1.txt"].reason == "worktree_mismatch"


class TestOverallDeadlineBudget:
    """The overall 60s budget must fail every remaining file closed with git_timeout."""

    @pytest.mark.asyncio
    async def test_exhausted_overall_budget_fails_all_entries_closed(
        self,
        git_repo_fixture: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Shrink both budgets far below real git spawn latency: whichever guard
        # trips first (a per-command timeout or a stage deadline check), every
        # entry must end up unverified with the fixed git_timeout reason.
        monkeypatch.setattr("argus.bench_local_verify.OVERALL_DEADLINE_SECONDS", 0.02)
        monkeypatch.setattr("argus.bench_local_verify.PER_COMMAND_TIMEOUT_SECONDS", 0.01)
        entries = [
            {"filename": "file1.txt", "status": "modified", "patch": None},
            {"filename": "clean_file.txt", "status": "added", "patch": None},
        ]
        results = await verify_missing_patches_locally(
            repo=git_repo_fixture["repo"],
            worktree_path=git_repo_fixture["worktree_dir"],
            trusted_head_sha=git_repo_fixture["head_sha"],
            head_sha=git_repo_fixture["head_sha"],
            base_sha=git_repo_fixture["base_sha"],
            entries=entries,
        )
        assert set(results) == {"file1.txt", "clean_file.txt"}
        for res in results.values():
            assert res.verified is False
            assert res.reason == "git_timeout"
            assert res.reason in VALID_FAIL_REASONS


class TestRunBoundedGitStderrFloodAndReap:
    """stderr is drained (capped), never returned, and never blocks kill/reap."""

    _FLOOD = (
        "import sys, time\n"
        "for _ in range(256):\n"
        "    sys.stderr.buffer.write(b'E' * 65536)\n"
        "sys.stderr.buffer.flush()\n"
        "time.sleep(30)\n"
    )

    @pytest.mark.asyncio
    async def test_sixteen_mib_stderr_drains_without_deadlock(self, tmp_path: Path) -> None:
        code = (
            "import sys\n"
            "sys.stderr.buffer.write(b'E' * (16 * 1024 * 1024))\n"
            "sys.stderr.buffer.flush()\n"
            "print('ok')\n"
        )
        ret, out = await _run_bounded_git(
            ["python3", "-c", code],
            cwd=str(tmp_path),
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            max_stdout=64,
            timeout=10.0,
        )
        assert ret == 0
        assert out == b"ok\n"

    @pytest.mark.asyncio
    async def test_timeout_during_stderr_flood_kills_and_reaps(self, tmp_path: Path) -> None:
        with pytest.raises(_TimeoutError):
            await _run_bounded_git(
                ["python3", "-c", self._FLOOD],
                cwd=str(tmp_path),
                env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                max_stdout=64,
                timeout=0.25,
            )

    @pytest.mark.asyncio
    async def test_cancel_during_stderr_flood_kills_and_reaps(self, tmp_path: Path) -> None:
        task = asyncio.create_task(
            _run_bounded_git(
                ["python3", "-c", self._FLOOD],
                cwd=str(tmp_path),
                env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                max_stdout=64,
                timeout=10.0,
            )
        )
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_nonzero_exit_stderr_is_discarded_not_returned(self, tmp_path: Path) -> None:
        code = "import sys\nsys.stderr.buffer.write(b'secret stderr detail\\n')\nsys.exit(3)\n"
        ret, out = await _run_bounded_git(
            ["python3", "-c", code],
            cwd=str(tmp_path),
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            max_stdout=64,
            timeout=5.0,
        )
        assert ret == 3
        assert out == b""


class TestUntrustedWorktreeContentNeverRead:
    """Verification reads mirror blobs only; dirty worktree files are never trusted."""

    @pytest.mark.asyncio
    async def test_dirty_worktree_cannot_fabricate_or_hide_routing(
        self, git_repo_fixture: dict[str, Any], tmp_path: Path
    ) -> None:
        src_dir = git_repo_fixture["src_dir"]
        mirror_dir = git_repo_fixture["mirror_dir"]
        head2 = _commit_and_advance(
            src_dir, mirror_dir, {"routing.txt": b"ARGUS_BENCH_FILE='real'\n"}, "add routing"
        )
        wt = _worktree_at(mirror_dir, tmp_path, "wt_dirty", head2)

        # Fabricate routing in a file whose committed blob is clean, and scrub
        # a file whose committed blob carries a routing line. Neither may
        # influence the result: only mirror blobs at the verified HEAD count.
        (Path(wt) / "file1.txt").write_text("ARGUS_BENCH_FILE='fabricated'\n", encoding="utf-8")
        (Path(wt) / "routing.txt").write_text("scrubbed on disk\n", encoding="utf-8")

        entries = [
            {"filename": "file1.txt", "status": "modified", "patch": None},
            {"filename": "routing.txt", "status": "added", "patch": None},
        ]
        results = await verify_missing_patches_locally(
            repo=git_repo_fixture["repo"],
            worktree_path=wt,
            trusted_head_sha=head2,
            head_sha=head2,
            base_sha=git_repo_fixture["base_sha"],
            entries=entries,
        )
        assert results["file1.txt"].verified is True
        assert results["file1.txt"].routing_lines == ()  # fabricated line ignored
        assert results["routing.txt"].verified is True
        assert any("ARGUS_BENCH_FILE='real'" in ln for ln in results["routing.txt"].routing_lines)


class TestNonUtf8GitPathFailsClosed:
    """A git path that cannot be decoded as UTF-8 must fail closed, never verify."""

    @pytest.mark.asyncio
    async def test_non_utf8_git_path_fails_closed(
        self, git_repo_fixture: dict[str, Any], tmp_path: Path
    ) -> None:
        mirror_dir = git_repo_fixture["mirror_dir"]
        base_sha = git_repo_fixture["base_sha"]
        head_sha = git_repo_fixture["head_sha"]
        # os.fsdecode(b"bad\x80\xff.bin") -- the lone-surrogate str round-trips
        # to the raw bytes through argv, but the strict UTF-8 parse of the
        # diff-raw output must reject it.
        fn = "bad\udc80\udcff.bin"
        raw_name = fn.encode("utf-8", "surrogateescape")

        # Build a commit whose tree contains the non-UTF-8 path purely via
        # object plumbing: at least one supported platform (macOS) refuses to
        # materialize invalid-UTF-8 names on disk, so no normal checkout (or
        # plain `git worktree add`) can create it -- git object storage itself
        # is byte-transparent, and that is exactly the hostile shape a mirror
        # could contain.
        ident_env = dict(
            os.environ,
            GIT_AUTHOR_NAME="Argus Tester",
            GIT_AUTHOR_EMAIL="argus@example.com",
            GIT_COMMITTER_NAME="Argus Tester",
            GIT_COMMITTER_EMAIL="argus@example.com",
        )
        blob_sha = (
            subprocess.check_output(
                ["git", "-C", str(mirror_dir), "hash-object", "-w", "--stdin"],
                input=b"clean content\n",
            )
            .decode()
            .strip()
        )
        tree_sha = (
            subprocess.check_output(
                ["git", "-C", str(mirror_dir), "mktree"],
                input=b"100644 blob " + blob_sha.encode() + b"\t" + raw_name + b"\n",
            )
            .decode()
            .strip()
        )
        head2 = (
            subprocess.check_output(
                ["git", "-C", str(mirror_dir), "commit-tree", tree_sha, "-p", head_sha],
                input=b"add non-utf8 name\n",
                env=ident_env,
            )
            .decode()
            .strip()
        )
        # --no-checkout: the worktree HEAD points at the commit without ever
        # materializing the undecodable filename on the filesystem.
        wt = tmp_path / "wt_nonutf8"
        subprocess.check_call(
            [
                "git",
                "-C",
                str(mirror_dir),
                "worktree",
                "add",
                "--detach",
                "--no-checkout",
                str(wt),
                head2,
            ]
        )

        entries = [{"filename": fn, "status": "added", "patch": None}]
        results = await verify_missing_patches_locally(
            repo=git_repo_fixture["repo"],
            worktree_path=str(wt),
            trusted_head_sha=head2,
            head_sha=head2,
            base_sha=base_sha,
            entries=entries,
        )
        res = results[fn]
        assert res.verified is False
        assert res.reason == "parse_error"
        assert res.reason in VALID_FAIL_REASONS


class TestRoutingMatchCap:
    """Routing-line matches are capped at 20 per file."""

    @pytest.mark.asyncio
    async def test_more_than_20_matches_capped_at_20(
        self, git_repo_fixture: dict[str, Any], tmp_path: Path
    ) -> None:
        src_dir = git_repo_fixture["src_dir"]
        mirror_dir = git_repo_fixture["mirror_dir"]
        content = b"".join(f"ARGUS_BENCH_FILE='cap{i}'\n".encode() for i in range(30))
        head2 = _commit_and_advance(src_dir, mirror_dir, {"many.txt": content}, "add many matches")
        wt = _worktree_at(mirror_dir, tmp_path, "wt_many", head2)

        entries = [{"filename": "many.txt", "status": "added", "patch": None}]
        results = await verify_missing_patches_locally(
            repo=git_repo_fixture["repo"],
            worktree_path=wt,
            trusted_head_sha=head2,
            head_sha=head2,
            base_sha=git_repo_fixture["base_sha"],
            entries=entries,
        )
        assert results["many.txt"].verified is True
        assert len(results["many.txt"].routing_lines) == 20
        assert results["many.txt"].routing_lines[0] == "ARGUS_BENCH_FILE='cap0'"
        assert results["many.txt"].routing_lines[19] == "ARGUS_BENCH_FILE='cap19'"


class TestLiteralGlobComponentNames:
    """Paths whose components are literally '**' or '*' stay literal, never globbed."""

    @pytest.mark.asyncio
    async def test_double_star_and_star_filenames_literal(
        self, git_repo_fixture: dict[str, Any], tmp_path: Path
    ) -> None:
        src_dir = git_repo_fixture["src_dir"]
        mirror_dir = git_repo_fixture["mirror_dir"]
        head2 = _commit_and_advance(
            src_dir, mirror_dir, {"**": b"clean\n", "*": b"clean\n"}, "add star files"
        )
        wt = _worktree_at(mirror_dir, tmp_path, "wt_stars", head2)

        entries = [
            {"filename": "**", "status": "added", "patch": None},
            {"filename": "*", "status": "added", "patch": None},
        ]
        results = await verify_missing_patches_locally(
            repo=git_repo_fixture["repo"],
            worktree_path=wt,
            trusted_head_sha=head2,
            head_sha=head2,
            base_sha=git_repo_fixture["base_sha"],
            entries=entries,
        )
        assert results["**"].verified is True
        assert results["**"].routing_lines == ()
        assert results["*"].verified is True
        assert results["*"].routing_lines == ()


class TestNonRegularModeCrossCheck:
    """Submodule gitlinks (mode 160000) fail closed exactly like symlinks (120000)."""

    @pytest.mark.asyncio
    async def test_submodule_gitlink_mode_rejected(self, git_repo_fixture: dict[str, Any]) -> None:
        entries = [{"filename": "clean_file.txt", "status": "added", "patch": None}]
        orig_run_bounded = _run_bounded_git

        async def _mock_run(cmd: list[str], **kwargs: Any) -> tuple[int, bytes]:
            if "diff" in cmd and "--raw" in cmd:
                fake_raw = (
                    b":000000 160000 0000000000000000000000000000000000000000 "
                    b"1234567890abcdef1234567890abcdef12345678 A\x00clean_file.txt\x00"
                )
                return 0, fake_raw
            return await orig_run_bounded(cmd, **kwargs)

        with patch("argus.bench_local_verify._run_bounded_git", side_effect=_mock_run):
            results = await verify_missing_patches_locally(
                repo=git_repo_fixture["repo"],
                worktree_path=git_repo_fixture["worktree_dir"],
                trusted_head_sha=git_repo_fixture["head_sha"],
                head_sha=git_repo_fixture["head_sha"],
                base_sha=git_repo_fixture["base_sha"],
                entries=entries,
            )
        assert results["clean_file.txt"].verified is False
        assert results["clean_file.txt"].reason == "non_regular_file"


class TestSubprocessEnvIsolationEndToEnd:
    """Hostile GIT_* env vars in the ambient environment must not reach git."""

    @pytest.mark.asyncio
    async def test_poisoned_git_env_vars_cannot_hijack_verification(
        self,
        git_repo_fixture: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # If GIT_DIR leaked into the verification subprocesses, the worktree
        # common-dir check would resolve to the decoy and fail closed.
        decoy = tmp_path / "decoy.git"
        subprocess.check_call(["git", "init", "--bare", str(decoy)])
        monkeypatch.setenv("GIT_DIR", str(decoy))
        monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path))
        monkeypatch.setenv("GIT_EXTERNAL_DIFF", "/usr/bin/false")

        entries = [
            {"filename": "file1.txt", "status": "modified", "patch": None},
            {"filename": "clean_file.txt", "status": "added", "patch": None},
        ]
        results = await verify_missing_patches_locally(
            repo=git_repo_fixture["repo"],
            worktree_path=git_repo_fixture["worktree_dir"],
            trusted_head_sha=git_repo_fixture["head_sha"],
            head_sha=git_repo_fixture["head_sha"],
            base_sha=git_repo_fixture["base_sha"],
            entries=entries,
        )
        assert results["file1.txt"].verified is True
        assert results["clean_file.txt"].verified is True


class TestNoSensitiveContentLogged:
    """Failures must log only fixed metadata -- never blob content or git stderr."""

    @pytest.mark.asyncio
    async def test_failures_never_log_blob_content_or_git_stderr(
        self,
        git_repo_fixture: dict[str, Any],
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        marker = "SECRET_BLOB_MARKER_7f3a91"
        src_dir = git_repo_fixture["src_dir"]
        mirror_dir = git_repo_fixture["mirror_dir"]
        head2 = _commit_and_advance(
            src_dir, mirror_dir, {"secret.bin": marker.encode() + b"\x00\x01\x02"}, "add secret"
        )
        wt = _worktree_at(mirror_dir, tmp_path, "wt_secret", head2)

        entries = [{"filename": "secret.bin", "status": "added", "patch": None}]
        caplog.set_level(logging.DEBUG, logger="argus")
        results = await verify_missing_patches_locally(
            repo=git_repo_fixture["repo"],
            worktree_path=wt,
            trusted_head_sha=head2,
            head_sha=head2,
            base_sha=git_repo_fixture["base_sha"],
            entries=entries,
        )
        assert results["secret.bin"].reason == "binary"

        # The object-missing path makes git emit "fatal: ..." on stderr.
        results2 = await verify_missing_patches_locally(
            repo=git_repo_fixture["repo"],
            worktree_path=wt,
            trusted_head_sha=head2,
            head_sha=head2,
            base_sha="f" * 40,
            entries=entries,
        )
        assert results2["secret.bin"].reason == "object_missing"

        for rec in caplog.records:
            msg = rec.getMessage()
            assert marker not in msg
            assert "fatal" not in msg.lower()
            assert ":100644" not in msg  # raw diff-raw metadata
            assert "\x00" not in msg  # raw blob bytes / NUL-delimited output


class TestGraphFallbackSecurity:
    """Graph-level wiring of the TECH-7768 fallback's trusted inputs."""

    @pytest.mark.asyncio
    async def test_graph_realistic_large_lockfile_pair_clean_not_flagged(
        self, git_repo_fixture: dict[str, Any], tmp_path: Path
    ) -> None:
        """PR-100 shape: two omitted patches incl. a ~1.5 MiB uv.lock, both clean."""
        src_dir = git_repo_fixture["src_dir"]
        mirror_dir = git_repo_fixture["mirror_dir"]
        base_sha = git_repo_fixture["base_sha"]

        parts: list[str] = ["# generated lockfile snapshot\n"]
        total = len(parts[0])
        i = 0
        while total < 1_500_000:
            entry = (
                f'\n[[package]]\nname = "pkg-{i}"\nversion = "0.{i}.9"\n'
                f'source = "registry+https://example.com/crates-index"\nchecksum = "{i:064x}"\n'
            )
            parts.append(entry)
            total += len(entry)
            i += 1
        rows = ",".join(f'{{"id": {j}, "label": "row-{j}"}}' for j in range(6000))
        head2 = _commit_and_advance(
            src_dir,
            mirror_dir,
            {"uv.lock": "".join(parts).encode("utf-8"), "data.json": ("[" + rows + "]").encode()},
            "add large lockfile and data snapshot",
        )
        uv_blob_sha = (
            subprocess.check_output(["git", "-C", str(mirror_dir), "rev-parse", f"{head2}:uv.lock"])
            .decode()
            .strip()
        )
        wt = _worktree_at(mirror_dir, tmp_path, "wt_large", head2)

        mock_gh = MagicMock()
        mock_gh.get_compare_files.return_value = (
            [
                {"filename": "uv.lock", "status": "added", "patch": None, "sha": uv_blob_sha},
                {"filename": "data.json", "status": "added", "patch": None},
            ],
            False,
        )
        node_result = await _run_fetch_diff_node(
            mock_gh,
            diff="diff --git a/uv.lock b/uv.lock\n",
            head_sha=head2,
            base_sha=base_sha,
            worktree_path=wt,
            config_head_sha=head2,
        )
        assert node_result["bench_config_unconfirmed"] is None
        assert node_result["bench_config_changes"] == []

    @pytest.mark.asyncio
    async def test_graph_trusted_head_mismatch_fails_closed(
        self, git_repo_fixture: dict[str, Any]
    ) -> None:
        """A trusted (config) head that differs from the fetched head stays unconfirmed."""
        head_sha = git_repo_fixture["head_sha"]
        base_sha = git_repo_fixture["base_sha"]

        mock_gh = MagicMock()
        mock_gh.get_compare_files.return_value = (
            [{"filename": "file1.txt", "status": "modified", "patch": None}],
            False,
        )
        node_result = await _run_fetch_diff_node(
            mock_gh,
            diff="diff --git a/file1.txt b/file1.txt\n",
            head_sha=head_sha,
            base_sha=base_sha,
            worktree_path=git_repo_fixture["worktree_dir"],
            config_head_sha=base_sha,  # trusted head != fetched head
        )
        assert node_result["bench_config_changes"] == []
        assert node_result["bench_config_unconfirmed"] is not None
        assert "could not be verified" in node_result["bench_config_unconfirmed"]

    @pytest.mark.asyncio
    async def test_graph_bench_config_path_missing_patch_flagged_without_local_verify(
        self, git_repo_fixture: dict[str, Any]
    ) -> None:
        """A bench config file with an omitted patch is flagged by path alone."""
        head_sha = git_repo_fixture["head_sha"]
        base_sha = git_repo_fixture["base_sha"]

        mock_gh = MagicMock()
        mock_gh.get_compare_files.return_value = (
            [{"filename": ".argus/bench.toml", "status": "added", "patch": None}],
            False,
        )
        local_verify_mock = AsyncMock()
        node_result = await _run_fetch_diff_node(
            mock_gh,
            diff="diff --git a/.argus/bench.toml b/.argus/bench.toml\n",
            head_sha=head_sha,
            base_sha=base_sha,
            worktree_path=git_repo_fixture["worktree_dir"],
            config_head_sha=head_sha,
            local_verify_mock=local_verify_mock,
        )
        local_verify_mock.assert_not_called()
        assert node_result["bench_config_changes"] == [".argus/bench.toml"]
        assert node_result["bench_config_unconfirmed"] is None

    @pytest.mark.asyncio
    async def test_graph_round2_routing_from_round1_flagged_via_full_scope_base(
        self, git_repo_fixture: dict[str, Any], tmp_path: Path
    ) -> None:
        """A routing line committed before prior_sha is still flagged: the fallback
        must use the whole-PR scope base, and the blob scan covers the whole HEAD."""
        src_dir = git_repo_fixture["src_dir"]
        mirror_dir = git_repo_fixture["mirror_dir"]
        base_sha = git_repo_fixture["base_sha"]

        r1 = _commit_and_advance(
            src_dir, mirror_dir, {"uv.lock": b"ARGUS_BENCH_FILE='override'\n"}, "round1 add uv.lock"
        )
        # Round 2 does not touch uv.lock at all -- with the correct whole-PR
        # scope base the added-in-round-1 file still shows up as an A record.
        r2 = _commit_and_advance(
            src_dir, mirror_dir, {"other.txt": b"clean\n"}, "round2 touch other"
        )
        wt = _worktree_at(mirror_dir, tmp_path, "wt_round2", r2)

        mock_prior = MagicMock()
        mock_prior.reviewed_sha = r1
        mock_prior.findings = []
        mock_prior.dismissed_findings = []
        mock_prior.review_id = "rev-round2"
        mock_prior.model_dump.return_value = {"reviewed_sha": r1}

        mock_gh = MagicMock()
        mock_gh.get_compare_files.return_value = (
            [{"filename": "uv.lock", "status": "added", "patch": None}],
            False,
        )
        node_result = await _run_fetch_diff_node(
            mock_gh,
            diff="diff --git a/other.txt b/other.txt\n",
            head_sha=r2,
            base_sha=base_sha,
            prior=mock_prior,
            worktree_path=wt,
            config_head_sha=r2,
        )
        assert node_result["bench_config_unconfirmed"] is None
        assert any(
            "present in verified head content modifying bench routing in uv.lock" in c
            for c in node_result["bench_config_changes"]
        )

    @pytest.mark.asyncio
    async def test_graph_local_verify_binary_failure_non_generated_keeps_unconfirmed(
        self, git_repo_fixture: dict[str, Any], tmp_path: Path
    ) -> None:
        """A fallback failure on a non-generated file keeps the fail-closed verdict."""
        src_dir = git_repo_fixture["src_dir"]
        mirror_dir = git_repo_fixture["mirror_dir"]
        base_sha = git_repo_fixture["base_sha"]

        head2 = _commit_and_advance(
            src_dir, mirror_dir, {"blob.bin": b"\x00\x01\x02binary"}, "add binary blob"
        )
        wt = _worktree_at(mirror_dir, tmp_path, "wt_binary_graph", head2)

        mock_gh = MagicMock()
        mock_gh.get_compare_files.return_value = (
            [{"filename": "blob.bin", "status": "added", "patch": None}],
            False,
        )
        node_result = await _run_fetch_diff_node(
            mock_gh,
            diff="diff --git a/blob.bin b/blob.bin\n",
            head_sha=head2,
            base_sha=base_sha,
            worktree_path=wt,
            config_head_sha=head2,
        )
        assert node_result["bench_config_changes"] == []
        assert node_result["bench_config_unconfirmed"] is not None
        assert "could not be verified" in node_result["bench_config_unconfirmed"]

    @pytest.mark.asyncio
    async def test_graph_renamed_entry_with_previous_filename_verifies(
        self, git_repo_fixture: dict[str, Any], tmp_path: Path
    ) -> None:
        """The rename's previous_filename survives both APIs' passthrough into the fallback."""
        src_dir = git_repo_fixture["src_dir"]
        mirror_dir = git_repo_fixture["mirror_dir"]
        base_sha = git_repo_fixture["base_sha"]

        subprocess.check_call(["git", "-C", str(src_dir), "mv", "file1.txt", "file1_renamed.txt"])
        subprocess.check_call(["git", "-C", str(src_dir), "commit", "-m", "rename file1"])
        head2 = (
            subprocess.check_output(["git", "-C", str(src_dir), "rev-parse", "HEAD"])
            .decode()
            .strip()
        )
        subprocess.check_call(
            ["git", "-C", str(src_dir), "push", str(mirror_dir), f"{head2}:refs/heads/main"]
        )
        wt = _worktree_at(mirror_dir, tmp_path, "wt_rename_graph", head2)

        mock_gh = MagicMock()
        mock_gh.get_compare_files.return_value = (
            [
                {
                    "filename": "file1_renamed.txt",
                    "status": "renamed",
                    "previous_filename": "file1.txt",
                    "patch": None,
                }
            ],
            False,
        )
        node_result = await _run_fetch_diff_node(
            mock_gh,
            diff="diff --git a/file1.txt b/file1_renamed.txt\n",
            head_sha=head2,
            base_sha=base_sha,
            worktree_path=wt,
            config_head_sha=head2,
        )
        assert node_result["bench_config_unconfirmed"] is None
        assert node_result["bench_config_changes"] == []


class TestGithubClientEntryFieldPassthrough:
    """Both file-listing APIs must preserve rename + blob-sha fields (fallback inputs)."""

    def test_get_compare_files_preserves_previous_filename_and_sha(
        self, httpx_mock: HTTPXMock
    ) -> None:
        httpx_mock.add_response(
            method="GET",
            url="https://api.github.com/repos/org/repo/compare/base...headsha",
            json={
                "files": [
                    {
                        "filename": "new.txt",
                        "status": "renamed",
                        "patch": None,
                        "previous_filename": "old.txt",
                        "sha": "c" * 40,
                    },
                    {"filename": "plain.txt", "status": "modified", "patch": "@@ -1 +1 @@"},
                ]
            },
        )
        client = GitHubClient(token="ro-token")
        files, truncated = client.get_compare_files("org/repo", "base", "headsha")
        assert truncated is False
        assert files[0]["previous_filename"] == "old.txt"
        assert files[0]["sha"] == "c" * 40
        assert files[1]["previous_filename"] == ""
        assert files[1]["sha"] == ""

    def test_get_pr_files_preserves_previous_filename_and_sha(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            method="GET",
            url="https://api.github.com/repos/org/repo/pulls/100/files?per_page=100&page=1",
            json=[
                {
                    "filename": "new.txt",
                    "status": "renamed",
                    "patch": None,
                    "previous_filename": "old.txt",
                    "sha": "c" * 40,
                },
                {"filename": "plain.txt", "status": "modified", "patch": "@@ -1 +1 @@"},
            ],
        )
        client = GitHubClient(token="ro-token")
        files, truncated = client.get_pr_files("org/repo", 100)
        assert truncated is False
        assert files[0]["previous_filename"] == "old.txt"
        assert files[0]["sha"] == "c" * 40
        assert files[1]["previous_filename"] == ""
        assert files[1]["sha"] == ""
