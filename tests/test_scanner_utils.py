"""Unit tests for argus.precheck.scanner_utils."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch


from argus.precheck.scanner_utils import is_success_exit, kill_and_reap, run_scanner_subprocess


class TestIsSuccessExit:
    def test_zero_is_always_success(self) -> None:
        assert is_success_exit(0) is True
        assert is_success_exit(0, findings_exit_code=1) is True

    def test_nonzero_without_findings_exit_code_is_failure(self) -> None:
        assert is_success_exit(1) is False
        assert is_success_exit(3) is False

    def test_matching_findings_exit_code_is_success(self) -> None:
        assert is_success_exit(1, findings_exit_code=1) is True

    def test_other_nonzero_with_findings_exit_code_set_is_still_failure(self) -> None:
        assert is_success_exit(2, findings_exit_code=1) is False


class TestRunScannerSubprocess:
    def _mock_proc(self, stdout: bytes, stderr: bytes = b"", returncode: int = 0) -> AsyncMock:
        proc = AsyncMock()
        proc.communicate.return_value = (stdout, stderr)
        proc.returncode = returncode
        return proc

    async def test_returns_stdout_stderr_returncode_on_success(self) -> None:
        proc = self._mock_proc(b"out", b"err", returncode=0)
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            result = await run_scanner_subprocess(["true"], timeout=5)
        assert result == (b"out", b"err", 0)

    async def test_returns_none_when_spawn_raises(self) -> None:
        with patch("asyncio.create_subprocess_exec", side_effect=OSError("no such binary")):
            result = await run_scanner_subprocess(["nonexistent-tool"], timeout=5)
        assert result is None

    async def test_returns_none_on_timeout(self) -> None:
        proc = AsyncMock()

        async def _never_returns(*args: object, **kwargs: object) -> tuple[bytes, bytes]:
            import asyncio

            await asyncio.sleep(10)
            return (b"", b"")

        proc.communicate.side_effect = _never_returns
        proc.kill = lambda: None

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            result = await run_scanner_subprocess(["slow-tool"], timeout=0.01)
        assert result is None

    async def test_kill_and_reap_suppresses_process_lookup_error(self) -> None:
        proc = AsyncMock()
        proc.kill = lambda: (_ for _ in ()).throw(ProcessLookupError())
        proc.communicate.return_value = (b"", b"")
        await kill_and_reap(proc, timeout=1)  # must not raise

    async def test_kill_and_reap_suppresses_drain_failure(self) -> None:
        proc = AsyncMock()
        proc.kill = lambda: None
        proc.communicate.side_effect = RuntimeError("drain failed")
        await kill_and_reap(proc, timeout=1)  # must not raise
