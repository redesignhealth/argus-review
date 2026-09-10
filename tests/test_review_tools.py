"""Tests for argus.review_tools: platform-neutral reviewer tool functions.

Covers: sandboxing (path-escape rejection for read_file/glob_files/grep),
symlink-escape rejection for the same three tools (a symlinked file or
directory inside the sandboxed root pointing outside it), resource limits
on grep (pattern length cap, catastrophic-backtracking timeout) and
read_file (oversized-file cap), the review_session context-manager
lifecycle (no-session errors, findings sink readable after the `with`
block exits), and concurrent-session isolation via contextvars -- the
specific property this design exists to guarantee (16 concurrent reviewer
sessions must never cross-contaminate each other's findings).
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from argus import review_tools


@pytest.fixture
def worktree(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("line one\nline two\nline three\n")
    (tmp_path / "src" / "util.py").write_text("def helper():\n    return 42\n")
    (tmp_path / "README.md").write_text("# hello\n")
    outside = tmp_path.parent / "outside-secret.txt"
    outside.write_text("should never be readable")
    return tmp_path


# ---------------------------------------------------------------------------
# No active session
# ---------------------------------------------------------------------------


class TestNoActiveSession:
    def test_read_file_without_session_raises(self) -> None:
        with pytest.raises(review_tools.NoActiveReviewSessionError):
            review_tools.read_file("src/app.py")

    def test_glob_files_without_session_raises(self) -> None:
        with pytest.raises(review_tools.NoActiveReviewSessionError):
            review_tools.glob_files("**/*.py")

    def test_grep_without_session_raises(self) -> None:
        with pytest.raises(review_tools.NoActiveReviewSessionError):
            review_tools.grep("helper")

    def test_report_finding_without_session_raises(self) -> None:
        with pytest.raises(review_tools.NoActiveReviewSessionError):
            review_tools.report_finding("src/app.py", 1, "desc")

    def test_finish_review_without_session_raises(self) -> None:
        with pytest.raises(review_tools.NoActiveReviewSessionError):
            review_tools.finish_review([])


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


class TestReadFile:
    def test_reads_line_numbered_content(self, worktree) -> None:
        with review_tools.review_session(str(worktree)):
            result = review_tools.read_file("src/app.py")
        assert result == "1: line one\n2: line two\n3: line three"

    def test_respects_offset_and_limit(self, worktree) -> None:
        with review_tools.review_session(str(worktree)):
            result = review_tools.read_file("src/app.py", offset=1, limit=1)
        assert result == "2: line two"

    def test_missing_file_raises_file_not_found(self, worktree) -> None:
        with review_tools.review_session(str(worktree)):
            with pytest.raises(FileNotFoundError):
                review_tools.read_file("src/does-not-exist.py")

    def test_rejects_path_escaping_root(self, worktree) -> None:
        with review_tools.review_session(str(worktree)):
            with pytest.raises(ValueError, match="escapes"):
                review_tools.read_file("../outside-secret.txt")

    def test_absolute_path_is_reinterpreted_as_root_relative_not_escaped(self, worktree) -> None:
        """Matches argus.helpers.sanitize_file_paths's existing behavior: a
        leading '/' is stripped rather than treated as an escape attempt, so
        an absolute-looking path is reinterpreted relative to root -- it does
        NOT leak the real absolute file outside the sandbox. Since no such
        relative path exists under root, this 404s rather than reading the
        real outside file.
        """
        outside = worktree.parent / "outside-secret.txt"
        with review_tools.review_session(str(worktree)):
            with pytest.raises(FileNotFoundError):
                review_tools.read_file(str(outside))


# ---------------------------------------------------------------------------
# glob_files
# ---------------------------------------------------------------------------


class TestGlobFiles:
    def test_matches_relative_to_root(self, worktree) -> None:
        with review_tools.review_session(str(worktree)):
            result = review_tools.glob_files("src/*.py")
        assert result == "src/app.py\nsrc/util.py"

    def test_rejects_dotdot_pattern(self, worktree) -> None:
        with review_tools.review_session(str(worktree)):
            with pytest.raises(ValueError, match="\\.\\."):
                review_tools.glob_files("../*")

    def test_rejects_absolute_pattern(self, worktree) -> None:
        with review_tools.review_session(str(worktree)):
            with pytest.raises(ValueError, match="absolute"):
                review_tools.glob_files("/etc/*")

    def test_glob_files_caps_results_and_truncates(
        self, worktree, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(review_tools, "_MAX_GLOB_RESULTS", 1)
        with review_tools.review_session(str(worktree)):
            result = review_tools.glob_files("src/*.py")
        assert "src/app.py" in result or "src/util.py" in result
        assert "capped at 1" in result


# ---------------------------------------------------------------------------
# grep
# ---------------------------------------------------------------------------


class TestGrep:
    def test_files_mode_returns_matching_paths(self, worktree) -> None:
        with review_tools.review_session(str(worktree)):
            result = review_tools.grep("helper")
        assert result == "src/util.py"

    def test_content_mode_returns_file_line_content(self, worktree) -> None:
        with review_tools.review_session(str(worktree)):
            result = review_tools.grep("helper", mode="content")
        assert result == "src/util.py:1: def helper():"

    def test_no_matches_returns_empty_string(self, worktree) -> None:
        with review_tools.review_session(str(worktree)):
            result = review_tools.grep("this-pattern-does-not-appear-anywhere")
        assert result == ""

    def test_unknown_mode_raises(self, worktree) -> None:
        with review_tools.review_session(str(worktree)):
            with pytest.raises(ValueError, match="mode"):
                review_tools.grep("helper", mode="bogus")

    def test_rejects_dotdot_glob(self, worktree) -> None:
        with review_tools.review_session(str(worktree)):
            with pytest.raises(ValueError, match="\\.\\."):
                review_tools.grep("helper", glob="../**/*")


# ---------------------------------------------------------------------------
# Symlink-escape rejection (High: symlink sandbox bypass)
# ---------------------------------------------------------------------------


class TestSymlinkEscape:
    """A symlink *inside* the sandboxed worktree that points *outside* it
    must never be listed (glob_files), read (grep/read_file), or leak
    content -- for both a symlinked file and a symlinked directory."""

    def test_read_file_rejects_symlinked_file_pointing_outside_root(
        self, worktree, tmp_path
    ) -> None:
        outside = tmp_path.parent / "outside-secret-for-read.txt"
        outside.write_text("do-not-read-me\n")
        link = worktree / "readlink.txt"
        link.symlink_to(outside)

        with review_tools.review_session(str(worktree)):
            with pytest.raises(ValueError, match="escapes"):
                review_tools.read_file("readlink.txt")

    def test_glob_files_skips_symlinked_file_pointing_outside_root(
        self, worktree, tmp_path
    ) -> None:
        outside = tmp_path.parent / "outside-symlink-target.txt"
        outside.write_text("secret\n")
        link = worktree / "escape.txt"
        link.symlink_to(outside)

        with review_tools.review_session(str(worktree)):
            result = review_tools.glob_files("*.txt")

        assert "escape.txt" not in result

    def test_glob_files_skips_symlinked_directory_pointing_outside_root(
        self, worktree, tmp_path
    ) -> None:
        outside_dir = tmp_path.parent / "outside-dir"
        outside_dir.mkdir()
        (outside_dir / "secret.py").write_text("SECRET = 1\n")
        link_dir = worktree / "escape_dir"
        link_dir.symlink_to(outside_dir)

        with review_tools.review_session(str(worktree)):
            result = review_tools.glob_files("escape_dir/*.py")

        assert result == ""

    def test_grep_does_not_read_symlinked_file_pointing_outside_root(
        self, worktree, tmp_path
    ) -> None:
        outside = tmp_path.parent / "outside-secret-for-grep.txt"
        outside.write_text("TOP-SECRET-CONTENT\n")
        link = worktree / "leak.txt"
        link.symlink_to(outside)

        with review_tools.review_session(str(worktree)):
            result = review_tools.grep("TOP-SECRET-CONTENT")

        assert result == ""

    def test_grep_does_not_read_through_symlinked_directory(self, worktree, tmp_path) -> None:
        outside_dir = tmp_path.parent / "outside-dir-for-grep"
        outside_dir.mkdir()
        (outside_dir / "secret.py").write_text("TOP-SECRET-DIR-CONTENT\n")
        link_dir = worktree / "escape_dir_grep"
        link_dir.symlink_to(outside_dir)

        with review_tools.review_session(str(worktree)):
            result = review_tools.grep("TOP-SECRET-DIR-CONTENT", glob="escape_dir_grep/*")

        assert result == ""


# ---------------------------------------------------------------------------
# Resource-exhaustion limits (Medium)
# ---------------------------------------------------------------------------


class TestReadFileResourceLimits:
    def test_oversized_file_raises_value_error(self, worktree, monkeypatch) -> None:
        monkeypatch.setattr(review_tools, "_MAX_READ_FILE_SIZE_BYTES", 10)
        big_file = worktree / "big.txt"
        big_file.write_text("x" * 1000)

        with review_tools.review_session(str(worktree)):
            with pytest.raises(ValueError, match="cap"):
                review_tools.read_file("big.txt")

    def test_file_under_cap_still_reads_fine(self, worktree, monkeypatch) -> None:
        monkeypatch.setattr(review_tools, "_MAX_READ_FILE_SIZE_BYTES", 10_000)
        with review_tools.review_session(str(worktree)):
            result = review_tools.read_file("src/app.py")
        assert result == "1: line one\n2: line two\n3: line three"


class TestGrepResourceLimits:
    def test_pattern_too_long_raises_value_error(self, worktree) -> None:
        with review_tools.review_session(str(worktree)):
            with pytest.raises(ValueError, match="chars"):
                review_tools.grep("a" * (review_tools._MAX_GREP_PATTERN_LENGTH + 1))

    def test_catastrophic_backtracking_regex_times_out(self, worktree, monkeypatch) -> None:
        """(a+)+$ against a run of a's with no trailing match is a classic
        catastrophic-backtracking pattern -- exponential in the length of
        the run (uncontested, ~5s+ for just 26 a's). With the timeout
        budget cranked down, grep must raise TimeoutError promptly instead
        of hanging for however long the pathological match would actually
        take -- this is the SIGALRM path (see _grep_alarm), not a
        thread-based watchdog, specifically because a thread-based
        watchdog CANNOT preempt this (CPython's regex engine holds the GIL
        for the whole backtracking loop)."""
        monkeypatch.setattr(review_tools, "_GREP_TIMEOUT_SECONDS", 0.2)
        adversarial = worktree / "adversarial.txt"
        adversarial.write_text("a" * 30 + "!\n")

        with review_tools.review_session(str(worktree)):
            start = time.monotonic()
            with pytest.raises(TimeoutError):
                review_tools.grep(r"(a+)+$", glob="adversarial.txt", mode="content")
            elapsed = time.monotonic() - start

        # Generous upper bound: proves the alarm actually interrupted the
        # match rather than merely happening to finish first.
        assert elapsed < 3.0

    def test_grep_from_non_main_thread_still_works_without_a_hard_timeout(self, worktree) -> None:
        """SIGALRM is main-thread-only, so grep() called off the main
        thread can't get a per-call hard interrupt (see module docstring)
        -- but it must still work correctly for a well-behaved pattern."""
        result_holder: dict[str, str] = {}
        error_holder: dict[str, BaseException] = {}

        def _worker() -> None:
            try:
                with review_tools.review_session(str(worktree)):
                    result_holder["result"] = review_tools.grep("helper")
            except BaseException as exc:  # noqa: BLE001 - surfaced via assertion below
                error_holder["error"] = exc

        thread = threading.Thread(target=_worker)
        thread.start()
        thread.join(timeout=5)

        assert not thread.is_alive()
        assert "error" not in error_holder, error_holder.get("error")
        assert result_holder["result"] == "src/util.py"


# ---------------------------------------------------------------------------
# report_finding / finish_review / review_session lifecycle
# ---------------------------------------------------------------------------


class TestFindingsSink:
    def test_report_finding_appends_to_sink(self, worktree) -> None:
        with review_tools.review_session(str(worktree)) as findings:
            review_tools.report_finding("src/app.py", 2, "off-by-one", context="loop bound")
            review_tools.report_finding("src/util.py", None, "unused import")

        assert findings == [
            {"file": "src/app.py", "line": 2, "description": "off-by-one", "context": "loop bound"},
            {"file": "src/util.py", "line": None, "description": "unused import", "context": None},
        ]

    def test_findings_list_readable_after_session_exits(self, worktree) -> None:
        with review_tools.review_session(str(worktree)) as findings:
            review_tools.report_finding("src/app.py", 1, "issue")
        # `with` block has exited (ContextVars reset), but the yielded list
        # object itself still holds everything appended during the session.
        assert len(findings) == 1

    def test_finish_review_does_not_clear_sink(self, worktree) -> None:
        with review_tools.review_session(str(worktree)) as findings:
            review_tools.report_finding("src/app.py", 1, "issue")
            summary = review_tools.finish_review(files_explored=["src/app.py"])
            assert "1 finding" in summary
            assert "1 file" in summary
        assert len(findings) == 1

    def test_nested_sessions_do_not_leak_after_inner_exits(self, worktree) -> None:
        with review_tools.review_session(str(worktree)) as outer_findings:
            review_tools.report_finding("outer.py", 1, "outer finding")
            with review_tools.review_session(str(worktree)) as inner_findings:
                review_tools.report_finding("inner.py", 1, "inner finding")
            # Back in the outer context: report_finding must go to the
            # outer sink again, not the (now-exited) inner one.
            review_tools.report_finding("outer.py", 2, "outer finding 2")

        assert len(outer_findings) == 2
        assert len(inner_findings) == 1


# ---------------------------------------------------------------------------
# Concurrency isolation: the core property this design exists to guarantee
# ---------------------------------------------------------------------------


class TestConcurrentSessionIsolation:
    @pytest.mark.asyncio
    async def test_concurrent_asyncio_tasks_do_not_cross_contaminate_findings(
        self, worktree
    ) -> None:
        """Simulates the 16-concurrent-reviewers scenario: each asyncio Task
        opens its own review_session and reports a different number of
        findings. Because asyncio copies the contextvars Context at task
        creation, no task's report_finding calls should ever land in
        another task's sink.
        """

        async def _run_session(session_id: int, finding_count: int) -> list[dict]:
            with review_tools.review_session(str(worktree)) as findings:
                for i in range(finding_count):
                    # Yield control between calls so tasks genuinely interleave.
                    await asyncio.sleep(0)
                    review_tools.report_finding(
                        f"session-{session_id}.py", i, f"finding from session {session_id}"
                    )
                await asyncio.sleep(0)
            return findings

        results = await asyncio.gather(
            *(_run_session(session_id, finding_count=session_id + 1) for session_id in range(16))
        )

        for session_id, findings in enumerate(results):
            assert len(findings) == session_id + 1
            for finding in findings:
                assert finding["file"] == f"session-{session_id}.py"
                assert f"session {session_id}" in finding["description"]

    def test_read_file_root_isolated_across_threads(self, tmp_path) -> None:
        """A ContextVar set in one thread's default context must not leak
        into a fresh thread that never entered a review_session."""
        import threading

        root_a = tmp_path / "a"
        root_a.mkdir()
        (root_a / "f.py").write_text("in a\n")

        with review_tools.review_session(str(root_a)):
            result_holder: dict[str, object] = {}

            def _other_thread() -> None:
                try:
                    review_tools.read_file("f.py")
                except review_tools.NoActiveReviewSessionError as e:
                    result_holder["error"] = e

            t = threading.Thread(target=_other_thread)
            t.start()
            t.join()

        assert isinstance(result_holder.get("error"), review_tools.NoActiveReviewSessionError)
