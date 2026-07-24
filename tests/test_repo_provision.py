"""Unit tests for argus.repo_provision.

Tests validation helpers, mirror-path resolution, subprocess helpers, and the
provisioned_worktree context manager — all mocked to avoid network/git access.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from argus.repo_provision import (
    _CLONE_COMPLETE_SENTINEL,
    _assert_worktree_sha,
    _ensure_mirror,
    _fetch_sha_into_mirror,
    _make_auth_header,
    _mirror_path,
    _run_git,
    _validate_repo,
    _validate_sha,
    provisioned_worktree,
)


# ---------------------------------------------------------------------------
# _validate_repo
# ---------------------------------------------------------------------------


class TestValidateRepo:
    def test_valid_simple(self) -> None:
        _validate_repo("owner/repo")  # no exception

    def test_valid_with_dots_and_dashes(self) -> None:
        _validate_repo("my-org/my.repo-name")

    def test_valid_underscores(self) -> None:
        _validate_repo("my_org/my_repo")

    def test_missing_slash(self) -> None:
        with pytest.raises(ValueError, match="Invalid repo format"):
            _validate_repo("ownerrepo")

    def test_empty(self) -> None:
        with pytest.raises(ValueError, match="Invalid repo format"):
            _validate_repo("")

    def test_path_traversal(self) -> None:
        with pytest.raises(ValueError, match="Invalid repo format"):
            _validate_repo("../evil/../../etc")

    def test_path_traversal_dotdot_in_name(self) -> None:
        """owner/.. must be rejected even though '..' matches [A-Za-z0-9._-]* after the dot check."""
        with pytest.raises(ValueError, match="Invalid repo format"):
            _validate_repo("owner/..")

    def test_leading_dot_in_name_rejected(self) -> None:
        """A leading dot in the name segment (e.g. owner/.) yields an invalid clone URL."""
        with pytest.raises(ValueError, match="Invalid repo format"):
            _validate_repo("owner/.")
        with pytest.raises(ValueError, match="Invalid repo format"):
            _validate_repo("owner/.hidden")

    def test_shell_injection(self) -> None:
        with pytest.raises(ValueError, match="Invalid repo format"):
            _validate_repo("owner/repo; rm -rf /")


# ---------------------------------------------------------------------------
# _validate_sha
# ---------------------------------------------------------------------------


class TestValidateSha:
    def test_valid_full_sha(self) -> None:
        _validate_sha("a" * 40)  # no exception

    def test_valid_mixed_hex(self) -> None:
        _validate_sha("0123456789abcdef" * 2 + "01234567")

    def test_short_sha_rejected(self) -> None:
        with pytest.raises(ValueError, match="exactly 40"):
            _validate_sha("abc1234")

    def test_uppercase_rejected(self) -> None:
        # Full SHA but uppercase — must be lowercase per spec
        with pytest.raises(ValueError, match="exactly 40"):
            _validate_sha("A" * 40)

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="exactly 40"):
            _validate_sha("")

    def test_non_hex_rejected(self) -> None:
        with pytest.raises(ValueError, match="exactly 40"):
            _validate_sha("g" * 40)

    def test_41_chars_rejected(self) -> None:
        with pytest.raises(ValueError, match="exactly 40"):
            _validate_sha("a" * 41)


# ---------------------------------------------------------------------------
# _mirror_path
# ---------------------------------------------------------------------------


class TestMirrorPath:
    def test_default_uses_home_cache(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            # Remove ARGUS_REPO_CACHE if present
            os.environ.pop("ARGUS_REPO_CACHE", None)
            result = _mirror_path("myorg/myrepo")
        expected = Path.home() / ".cache" / "argus" / "myorg" / "myrepo.git"
        assert result == expected

    def test_env_override(self, tmp_path: Path) -> None:
        with patch.dict(os.environ, {"ARGUS_REPO_CACHE": str(tmp_path)}):
            result = _mirror_path("myorg/myrepo")
        assert result == tmp_path / "myorg" / "myrepo.git"

    def test_owner_and_name_separated(self) -> None:
        result = _mirror_path("redesign-health/argus")
        assert result.name == "argus.git"
        assert result.parent.name == "redesign-health"


# ---------------------------------------------------------------------------
# _ensure_mirror — mirror reuse vs. clone
# ---------------------------------------------------------------------------


class TestEnsureMirror:
    @pytest.mark.asyncio
    async def test_skips_clone_when_mirror_exists(self, tmp_path: Path) -> None:
        """If mirror directory already exists and is valid, only rev-parse is called (no clone)."""
        mirror = tmp_path / "owner" / "repo.git"
        mirror.mkdir(parents=True)

        with patch("argus.repo_provision._run_git", new_callable=AsyncMock) as mock_git:
            mock_git.return_value = MagicMock()
            await _ensure_mirror(mirror, "owner/repo", "test-token")

        # Only the validation call (rev-parse --git-dir) should have been made — not clone
        mock_git.assert_awaited_once()
        call_args = mock_git.call_args[0]
        assert "rev-parse" in call_args
        assert "--git-dir" in call_args
        assert "clone" not in call_args

    @pytest.mark.asyncio
    async def test_invalid_mirror_triggers_reclone(self, tmp_path: Path) -> None:
        """If the mirror directory exists but is not a valid bare repo (rev-parse fails),
        _ensure_mirror removes it and falls through to clone. If removal is incomplete,
        a warning is logged but the function continues (best-effort)."""
        mirror = tmp_path / "owner" / "corrupt.git"
        mirror.mkdir(parents=True)

        async def _fake_git(*args: str, **kwargs: object) -> tuple:
            if "rev-parse" in args:
                raise RuntimeError("not a git repository")
            if "clone" in args:
                # Simulate successful clone creating the mirror dir + sentinel
                mirror.mkdir(parents=True, exist_ok=True)
                (mirror / _CLONE_COMPLETE_SENTINEL).touch()
            return (MagicMock(returncode=0), b"", b"")

        with patch("argus.repo_provision._run_git", side_effect=_fake_git) as mock_git:
            await _ensure_mirror(mirror, "owner/corrupt", "test-token")

        # Clone must have been called after the invalid-mirror detection
        call_args_list = [c[0] for c in mock_git.call_args_list]
        assert any("clone" in args for args in call_args_list)
        assert (mirror / _CLONE_COMPLETE_SENTINEL).exists()

    @pytest.mark.asyncio
    async def test_partial_removal_warning_and_clone_attempted(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When rmtree fails to fully remove the invalid mirror (dir still exists
        after rmtree), a warning is logged and the clone is still attempted.

        Drives the path:
          mirror exists -> rev-parse raises -> rmtree (no-op) -> dir still exists
          -> warning logged -> fall through to clone.
        """
        mirror = tmp_path / "owner" / "sticky.git"
        mirror.mkdir(parents=True)

        clone_called: list[bool] = []

        async def _fake_git(*args: str, **kwargs: object) -> tuple:
            if "rev-parse" in args:
                raise RuntimeError("not a git repository")
            if "clone" in args:
                clone_called.append(True)
            return (MagicMock(returncode=0), b"", b"")

        # Patch shutil.rmtree to a no-op so the dir remains after the invalid-mirror removal.
        with (
            patch("argus.repo_provision._run_git", side_effect=_fake_git),
            patch("argus.repo_provision.shutil.rmtree"),
            caplog.at_level(logging.WARNING, logger="argus.repo_provision"),
        ):
            await _ensure_mirror(mirror, "owner/sticky", "test-token")

        # Warning about incomplete removal must have been emitted.
        messages = [r.getMessage() for r in caplog.records]
        assert any("Could not fully remove invalid mirror" in m for m in messages)
        # Clone must still have been attempted despite the leftover dir.
        assert clone_called, "clone was not attempted after partial removal"

    @pytest.mark.asyncio
    async def test_clones_when_mirror_absent(self, tmp_path: Path) -> None:
        """If mirror does not exist, clone --mirror runs and a completion
        sentinel is written. The dead remote set-url scrub is gone."""
        mirror = tmp_path / "owner" / "newrepo.git"

        async def _fake_git(*args: str, **kwargs: object) -> tuple:
            if "clone" in args:
                mirror.mkdir(parents=True, exist_ok=True)
            return (MagicMock(returncode=0), b"", b"")

        with patch("argus.repo_provision._run_git", side_effect=_fake_git) as mock_git:
            await _ensure_mirror(mirror, "owner/newrepo", "secret-token")

        # Exactly one git call: clone --mirror (the set-url scrub was removed).
        assert mock_git.await_count == 1
        first_args = mock_git.call_args_list[0][0]
        assert "clone" in first_args
        assert "--mirror" in first_args
        assert any("github.com" in arg for arg in first_args)
        # Completion sentinel written so future runs / racing peers trust it.
        assert (mirror / _CLONE_COMPLETE_SENTINEL).exists()


# ---------------------------------------------------------------------------
# provisioned_worktree — happy path and fail-closed assertion
# ---------------------------------------------------------------------------


_VALID_SHA = "a" * 40
_VALID_REPO = "myorg/myrepo"


class TestProvisionedWorktree:
    @pytest.mark.asyncio
    async def test_happy_path_yields_path_and_tears_down(self, tmp_path: Path) -> None:
        """Context manager yields the "<mkdtemp>/wt" path and cleans up on exit."""
        # mkdtemp returns the anchoring parent dir; the worktree lives at
        # <parent>/wt (git creates the subdir; no mkdtemp+rmdir TOCTOU window).
        parent_dir = tmp_path / "argus-worktree-xyz"
        parent_dir.mkdir()
        parent_dir_str = str(parent_dir)
        expected_path = os.path.join(parent_dir_str, "wt")

        with (
            patch(
                "argus.repo_provision._ensure_mirror",
                new_callable=AsyncMock,
            ),
            patch(
                "argus.repo_provision._fetch_sha_into_mirror",
                new_callable=AsyncMock,
            ),
            patch(
                "argus.repo_provision._add_worktree",
                new_callable=AsyncMock,
                side_effect=lambda mirror, sha, path: Path(path).mkdir(parents=True, exist_ok=True),
            ),
            patch(
                "argus.repo_provision._assert_worktree_sha",
                new_callable=AsyncMock,
            ),
            patch(
                "argus.repo_provision._remove_worktree",
                new_callable=AsyncMock,
            ) as mock_remove,
            patch(
                "argus.repo_provision.tempfile.mkdtemp",
                return_value=parent_dir_str,
            ),
        ):
            captured: list[str] = []
            async with provisioned_worktree(
                repo=_VALID_REPO,
                head_sha=_VALID_SHA,
                token="test-token",
            ) as path:
                captured.append(path)

        assert captured == [expected_path]
        mock_remove.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fail_closed_on_sha_mismatch(self, tmp_path: Path) -> None:
        """If _assert_worktree_sha raises, the context manager propagates and tears down."""
        worktree_dir = tmp_path / "wt-mismatch"
        worktree_dir.mkdir()
        worktree_dir_str = str(worktree_dir)

        with (
            patch(
                "argus.repo_provision._ensure_mirror",
                new_callable=AsyncMock,
            ),
            patch(
                "argus.repo_provision._fetch_sha_into_mirror",
                new_callable=AsyncMock,
            ),
            patch(
                "argus.repo_provision._add_worktree",
                new_callable=AsyncMock,
            ),
            patch(
                "argus.repo_provision._assert_worktree_sha",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Worktree SHA mismatch"),
            ),
            patch(
                "argus.repo_provision._remove_worktree",
                new_callable=AsyncMock,
            ) as mock_remove,
            patch(
                "argus.repo_provision.tempfile.mkdtemp",
                return_value=worktree_dir_str,
            ),
        ):
            with pytest.raises(RuntimeError, match="SHA mismatch"):
                async with provisioned_worktree(
                    repo=_VALID_REPO,
                    head_sha=_VALID_SHA,
                    token="test-token",
                ):
                    pass  # should not reach here
        # RuntimeError propagates; that is the fail-closed behaviour. Cleanup
        # of the orphaned worktree is the critical invariant: assert it ran.
        mock_remove.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_teardown_runs_on_body_exception(self, tmp_path: Path) -> None:
        """Teardown runs even if the body of the async with block raises."""
        worktree_dir = tmp_path / "argus-worktree-body-exc"
        worktree_dir.mkdir()
        worktree_dir_str = str(worktree_dir)

        with (
            patch(
                "argus.repo_provision._ensure_mirror",
                new_callable=AsyncMock,
            ),
            patch(
                "argus.repo_provision._fetch_sha_into_mirror",
                new_callable=AsyncMock,
            ),
            patch(
                "argus.repo_provision._add_worktree",
                new_callable=AsyncMock,
                side_effect=lambda mirror, sha, path: Path(path).mkdir(parents=True, exist_ok=True),
            ),
            patch(
                "argus.repo_provision._assert_worktree_sha",
                new_callable=AsyncMock,
            ),
            patch(
                "argus.repo_provision._remove_worktree",
                new_callable=AsyncMock,
            ) as mock_remove,
            patch(
                "argus.repo_provision.tempfile.mkdtemp",
                return_value=worktree_dir_str,
            ),
        ):
            with pytest.raises(ValueError, match="body error"):
                async with provisioned_worktree(
                    repo=_VALID_REPO,
                    head_sha=_VALID_SHA,
                    token="test-token",
                ):
                    raise ValueError("body error")

        mock_remove.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_validation_rejects_bad_repo(self) -> None:
        """Raises ValueError before any git operations if repo is invalid."""
        with patch("argus.repo_provision._ensure_mirror", new_callable=AsyncMock) as mock_ensure:
            with pytest.raises(ValueError, match="Invalid repo format"):
                async with provisioned_worktree(
                    repo="not-a-valid-repo",
                    head_sha=_VALID_SHA,
                    token="test-token",
                ):
                    pass

        mock_ensure.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_validation_rejects_bad_sha(self) -> None:
        """Raises ValueError before any git operations if sha is invalid."""
        with patch("argus.repo_provision._ensure_mirror", new_callable=AsyncMock) as mock_ensure:
            with pytest.raises(ValueError, match="exactly 40"):
                async with provisioned_worktree(
                    repo=_VALID_REPO,
                    head_sha="shortsha",
                    token="test-token",
                ):
                    pass

        mock_ensure.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_mirror_kept_on_teardown(self, tmp_path: Path) -> None:
        """_remove_worktree is called (worktree removed) but mirror dir is not deleted."""
        mirror_kept: list[bool] = []
        fake_mirror = Path("/tmp/fake-mirror.git")
        worktree_dir = tmp_path / "argus-worktree-mirror-kept"
        worktree_dir.mkdir()
        worktree_dir_str = str(worktree_dir)

        with (
            patch(
                "argus.repo_provision._ensure_mirror",
                new_callable=AsyncMock,
            ),
            patch(
                "argus.repo_provision._fetch_sha_into_mirror",
                new_callable=AsyncMock,
            ),
            patch(
                "argus.repo_provision._add_worktree",
                new_callable=AsyncMock,
                side_effect=lambda mirror, sha, path: Path(path).mkdir(parents=True, exist_ok=True),
            ),
            patch(
                "argus.repo_provision._assert_worktree_sha",
                new_callable=AsyncMock,
            ),
            patch(
                "argus.repo_provision._remove_worktree",
                new_callable=AsyncMock,
                side_effect=lambda mirror, wt: mirror_kept.append(True),
            ),
            patch(
                "argus.repo_provision._mirror_path",
                return_value=fake_mirror,
            ),
            patch(
                "argus.repo_provision.tempfile.mkdtemp",
                return_value=worktree_dir_str,
            ),
        ):
            async with provisioned_worktree(
                repo=_VALID_REPO,
                head_sha=_VALID_SHA,
                token="test-token",
            ):
                pass

        # _remove_worktree was called (worktree removed) but mirror was passed — not rmtree'd
        assert mirror_kept == [True]


# ---------------------------------------------------------------------------
# provision_worktree / teardown_worktree — public API
# ---------------------------------------------------------------------------

from argus.repo_provision import provision_worktree, teardown_worktree  # noqa: E402


class TestProvisionWorktreePublicAPI:
    @pytest.mark.asyncio
    async def test_provision_returns_path(self, tmp_path: Path) -> None:
        """provision_worktree() returns the "<mkdtemp>/wt" worktree path string."""
        parent_dir = tmp_path / "argus-worktree-pub"
        parent_dir.mkdir()
        parent_dir_str = str(parent_dir)
        expected_path = os.path.join(parent_dir_str, "wt")

        with (
            patch("argus.repo_provision._ensure_mirror", new_callable=AsyncMock),
            patch("argus.repo_provision._fetch_sha_into_mirror", new_callable=AsyncMock),
            patch(
                "argus.repo_provision._add_worktree",
                new_callable=AsyncMock,
                side_effect=lambda mirror, sha, path: Path(path).mkdir(parents=True, exist_ok=True),
            ),
            patch("argus.repo_provision._assert_worktree_sha", new_callable=AsyncMock),
            patch("argus.repo_provision.tempfile.mkdtemp", return_value=parent_dir_str),
        ):
            result = await provision_worktree(
                repo=_VALID_REPO, head_sha=_VALID_SHA, token="test-token"
            )

        assert result == expected_path

    @pytest.mark.asyncio
    async def test_provision_rejects_invalid_repo(self) -> None:
        """provision_worktree() raises ValueError for invalid repo without calling git."""
        with patch("argus.repo_provision._ensure_mirror", new_callable=AsyncMock) as mock_ensure:
            with pytest.raises(ValueError, match="Invalid repo format"):
                await provision_worktree(repo="bad-repo", head_sha=_VALID_SHA, token="test-token")
        mock_ensure.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_provision_rejects_invalid_sha(self) -> None:
        """provision_worktree() raises ValueError for short SHA without calling git."""
        with patch("argus.repo_provision._ensure_mirror", new_callable=AsyncMock) as mock_ensure:
            with pytest.raises(ValueError, match="exactly 40"):
                await provision_worktree(repo=_VALID_REPO, head_sha="badf00d", token="test-token")
        mock_ensure.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_provision_fails_closed_on_error(self, tmp_path: Path) -> None:
        """provision_worktree() propagates RuntimeError — no fallback path returned."""
        worktree_dir_str = str(tmp_path / "wt-fail")

        with (
            patch(
                "argus.repo_provision._ensure_mirror",
                new_callable=AsyncMock,
                side_effect=RuntimeError("clone failed"),
            ),
            patch("argus.repo_provision.tempfile.mkdtemp", return_value=worktree_dir_str),
        ):
            with pytest.raises(RuntimeError, match="clone failed"):
                await provision_worktree(repo=_VALID_REPO, head_sha=_VALID_SHA, token="test-token")

    @pytest.mark.asyncio
    async def test_teardown_calls_remove_worktree(self, tmp_path: Path) -> None:
        """teardown_worktree() calls _remove_worktree and rmtree's the parent
        argus-worktree-* dir only — never the grandparent."""
        parent = tmp_path / "argus-worktree-abc"
        wt = parent / "wt"
        wt.mkdir(parents=True)

        with (
            patch("argus.repo_provision._remove_worktree", new_callable=AsyncMock) as mock_remove,
        ):
            await teardown_worktree(_VALID_REPO, str(wt))

        mock_remove.assert_awaited_once()
        # The anchoring parent (and the worktree within it) is gone...
        assert not parent.exists()
        # ...but the grandparent (here pytest's tmp_path) is untouched.
        assert tmp_path.exists()

    @pytest.mark.asyncio
    async def test_teardown_refuses_unexpected_path(self, tmp_path: Path) -> None:
        """teardown_worktree() raises before calling _remove_worktree when the
        path is not one this module created (guards against a malformed
        worktree_path widening the blast radius of rmtree).
        The guard now runs BEFORE any git cleanup, so _remove_worktree must
        NOT be called when the path is rejected."""
        stray = tmp_path / "not-a-worktree"
        stray.mkdir()

        with patch("argus.repo_provision._remove_worktree", new_callable=AsyncMock) as mock_remove:
            with pytest.raises(ValueError, match="unexpected worktree path"):
                await teardown_worktree(_VALID_REPO, str(stray))

        # Guard fires first: git cleanup must not have run.
        mock_remove.assert_not_awaited()
        # The directory must NOT have been deleted.
        assert stray.exists()


# ---------------------------------------------------------------------------
# _ensure_mirror — concurrent clone race condition
# ---------------------------------------------------------------------------


class TestEnsureMirrorRaceCondition:
    @pytest.mark.asyncio
    async def test_race_condition_handled_gracefully(self, tmp_path: Path) -> None:
        """If clone fails but mirror now exists, treat as success (concurrent winner)."""
        mirror = tmp_path / "owner" / "race.git"

        async def _fake_git(*args: str, **kwargs: object) -> tuple:
            # Clone raises because a concurrent process won; that peer FINISHED
            # (mirror dir + completion sentinel present). rev-parse validation
            # then succeeds, so _validate_mirror returns True.
            if "clone" in args:
                mirror.mkdir(parents=True, exist_ok=True)
                (mirror / _CLONE_COMPLETE_SENTINEL).touch()
                raise RuntimeError("destination path already exists")
            return (MagicMock(returncode=0), b"", b"")

        with patch("argus.repo_provision._run_git", side_effect=_fake_git):
            # Should NOT raise: concurrent process "won" the clone race and the
            # mirror validates as a complete bare repo (sentinel + git dir).
            await _ensure_mirror(mirror, "owner/race", "test-token")

    @pytest.mark.asyncio
    async def test_race_condition_reraises_if_mirror_still_absent(self, tmp_path: Path) -> None:
        """If clone fails and mirror does NOT appear, re-raise the original error."""
        mirror = tmp_path / "owner" / "no-race.git"

        async def _fake_clone(*args: str, **kwargs: object) -> MagicMock:
            # Clone fails and mirror is still absent (genuine failure)
            raise RuntimeError("authentication failed")

        with patch("argus.repo_provision._run_git", side_effect=_fake_clone):
            with pytest.raises(RuntimeError, match="authentication failed"):
                await _ensure_mirror(mirror, "owner/no-race", "test-token")

    @pytest.mark.asyncio
    async def test_race_reraises_when_mirror_incomplete(self, tmp_path: Path) -> None:
        """If clone fails and the leftover dir lacks the completion sentinel
        (a partial or still-in-progress peer clone), re-raise rather than
        returning an unusable mirror."""
        mirror = tmp_path / "owner" / "partial.git"

        async def _fake_git(*args: str, **kwargs: object) -> tuple:
            if "clone" in args:
                # git creates the destination dir early, before the clone
                # finishes — but no completion sentinel is written.
                mirror.mkdir(parents=True, exist_ok=True)
                raise RuntimeError("destination path already exists")
            return (MagicMock(returncode=0), b"", b"")

        with patch("argus.repo_provision._run_git", side_effect=_fake_git):
            # _validate_mirror short-circuits False on the missing sentinel, so
            # the original clone error propagates.
            with pytest.raises(RuntimeError, match="destination path already exists"):
                await _ensure_mirror(mirror, "owner/partial", "test-token")


# ---------------------------------------------------------------------------
# _fetch_sha_into_mirror: fetch exit-code quirk handling
# ---------------------------------------------------------------------------


class TestFetchShaIntoMirror:
    @pytest.mark.asyncio
    async def test_succeeds_when_object_present_despite_nonzero_fetch(self, tmp_path: Path) -> None:
        """fetch may exit non-zero when the SHA is already present; cat-file -e
        exiting 0 confirms the object exists, so the function should succeed."""
        # fetch ignores its return; cat-file is unpacked as (proc, stdout, stderr).
        fetch_ret = (MagicMock(returncode=1), b"", b"")  # already present on some git versions
        catfile_ret = (MagicMock(returncode=0), b"", b"")  # object IS in the store
        with patch(
            "argus.repo_provision._run_git",
            new_callable=AsyncMock,
            side_effect=[fetch_ret, catfile_ret],
        ) as mock_git:
            await _fetch_sha_into_mirror(tmp_path / "m.git", "a" * 40, "tok")
        assert mock_git.await_count == 2

    @pytest.mark.asyncio
    async def test_raises_when_object_missing(self, tmp_path: Path) -> None:
        """If cat-file -e exits non-zero the object is absent; raise RuntimeError."""
        fetch_ret = (MagicMock(returncode=0), b"", b"")
        catfile_ret = (MagicMock(returncode=1), b"", b"")  # object NOT found
        with patch(
            "argus.repo_provision._run_git",
            new_callable=AsyncMock,
            side_effect=[fetch_ret, catfile_ret],
        ):
            with pytest.raises(RuntimeError, match="not found in mirror"):
                await _fetch_sha_into_mirror(tmp_path / "m.git", "a" * 40, "tok")


# ---------------------------------------------------------------------------
# _run_git — credential redaction
# ---------------------------------------------------------------------------


class TestRunGitRedaction:
    @pytest.mark.asyncio
    async def test_auth_header_not_leaked_in_error(self) -> None:
        """Authorization: Basic header token is redacted from RuntimeError on git failure."""
        fake_proc = AsyncMock()
        fake_proc.returncode = 1
        fake_proc.communicate = AsyncMock(
            return_value=(
                b"",
                b"some error about http.extraHeader=Authorization: Basic abc123token",
            )
        )

        with patch("asyncio.create_subprocess_exec", return_value=fake_proc):
            with pytest.raises(RuntimeError) as exc_info:
                await _run_git(
                    "-c",
                    "http.extraHeader=Authorization: Basic abc123token",
                    "fetch",
                    "origin",
                    "somesha",
                )

        error_str = str(exc_info.value)
        assert "abc123token" not in error_str
        assert "Authorization: Basic ***" in error_str


# ---------------------------------------------------------------------------
# _run_git: token kept out of argv
# ---------------------------------------------------------------------------


class TestRunGitAuthToken:
    @pytest.mark.asyncio
    async def test_token_supplied_via_config_file_not_argv(self) -> None:
        """When auth_token is given, the token must not appear in the process
        args; auth is supplied through a GIT_CONFIG_GLOBAL file instead."""
        secret = "supersecrettoken123"
        fake_proc = AsyncMock()
        fake_proc.returncode = 0
        fake_proc.communicate = AsyncMock(return_value=(b"", b""))

        captured: dict[str, object] = {}

        async def _fake_exec(*args: str, **kwargs: object) -> AsyncMock:
            captured["args"] = args
            env = kwargs.get("env")
            captured["env"] = env
            # File exists at exec time (unlinked in _run_git's finally afterward).
            if isinstance(env, dict) and "GIT_CONFIG_GLOBAL" in env:
                with open(env["GIT_CONFIG_GLOBAL"]) as fh:
                    captured["cfg"] = fh.read()
            return fake_proc

        with patch("asyncio.create_subprocess_exec", side_effect=_fake_exec):
            await _run_git(
                "clone",
                "--mirror",
                "https://github.com/o/r.git",
                "/tmp/x",  # nosec B108 - test path, never created
                auth_token=secret,
            )

        # Raw token never appears in argv.
        captured_args = captured["args"]
        assert isinstance(captured_args, tuple)
        assert all(secret not in str(a) for a in captured_args)
        # Auth routed through a per-call global config file.
        env = captured["env"]
        assert isinstance(env, dict)
        assert "GIT_CONFIG_GLOBAL" in env
        # The config file carries the Basic auth header for this token.
        assert _make_auth_header(secret) in str(captured["cfg"])


# ---------------------------------------------------------------------------
# _run_git — timeout path (kill + token-file cleanup)
# ---------------------------------------------------------------------------


class TestRunGitTimeout:
    @pytest.mark.asyncio
    async def test_timeout_kills_process_and_unlinks_token_file(self) -> None:
        """On timeout, the subprocess is killed and the 0600 token config file
        created for this call is still unlinked by the finally block."""
        import tempfile as _tempfile

        fake_proc = MagicMock()
        fake_proc.kill = MagicMock()
        fake_proc.communicate = AsyncMock(return_value=(b"", b""))

        recorded: dict[str, str] = {}
        real_mkstemp = _tempfile.mkstemp

        def _spy_mkstemp(*a: Any, **k: Any) -> tuple[int, str]:
            fd, path = real_mkstemp(*a, **k)
            recorded["path"] = path
            return fd, path

        async def _fake_wait_for(awaitable: object, timeout: float) -> object:
            # Close the un-awaited communicate() coroutine to avoid warnings.
            close = getattr(awaitable, "close", None)
            if close:
                close()
            raise asyncio.TimeoutError()

        with (
            patch("asyncio.create_subprocess_exec", return_value=fake_proc),
            patch("argus.repo_provision.tempfile.mkstemp", side_effect=_spy_mkstemp),
            patch("asyncio.wait_for", side_effect=_fake_wait_for),
        ):
            with pytest.raises(RuntimeError, match="timed out"):
                await _run_git("clone", "x", "y", auth_token="sometoken")

        fake_proc.kill.assert_called_once()
        assert "path" in recorded
        assert not os.path.exists(recorded["path"])  # token file cleaned up


# ---------------------------------------------------------------------------
# _assert_worktree_sha — verdict logic over _run_git output
# ---------------------------------------------------------------------------


class TestAssertWorktreeSha:
    @pytest.mark.asyncio
    async def test_mismatch_raises(self) -> None:
        with patch(
            "argus.repo_provision._run_git",
            new_callable=AsyncMock,
            return_value=(MagicMock(returncode=0), b"b" * 40 + b"\n", b""),
        ):
            with pytest.raises(RuntimeError, match="SHA mismatch"):
                await _assert_worktree_sha("/wt", "a" * 40)

    @pytest.mark.asyncio
    async def test_revparse_failure_raises(self) -> None:
        with patch(
            "argus.repo_provision._run_git",
            new_callable=AsyncMock,
            return_value=(MagicMock(returncode=1), b"", b"fatal: not a git repository"),
        ):
            with pytest.raises(RuntimeError, match="rev-parse HEAD failed"):
                await _assert_worktree_sha("/wt", "a" * 40)

    @pytest.mark.asyncio
    async def test_match_passes(self) -> None:
        sha = "a" * 40
        with patch(
            "argus.repo_provision._run_git",
            new_callable=AsyncMock,
            return_value=(MagicMock(returncode=0), (sha + "\n").encode(), b""),
        ):
            await _assert_worktree_sha("/wt", sha)  # no raise
