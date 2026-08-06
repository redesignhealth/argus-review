"""Unit tests for argus.precheck.secrets_scanner."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from argus.precheck.secrets_scanner import run_trivy_secrets_sarif, trivy_available


def _sarif_bytes(*rule_ids: str) -> bytes:
    return json.dumps(
        {
            "runs": [
                {
                    "results": [
                        {
                            "ruleId": rule_id,
                            "level": "error",
                            "message": {"text": f"hit for {rule_id}"},
                            "locations": [
                                {
                                    "physicalLocation": {
                                        "artifactLocation": {"uri": "config.py"},
                                        "region": {"startLine": 1},
                                    }
                                }
                            ],
                        }
                        for rule_id in rule_ids
                    ]
                }
            ]
        }
    ).encode()


def _mock_subprocess(stdout: bytes, stderr: bytes = b"", returncode: int = 0) -> AsyncMock:
    proc = AsyncMock()
    proc.communicate.return_value = (stdout, stderr)
    proc.returncode = returncode
    return proc


def test_trivy_available_reflects_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("argus.precheck.secrets_scanner.shutil.which", lambda _: None)
    assert trivy_available() is False
    monkeypatch.setattr("argus.precheck.secrets_scanner.shutil.which", lambda _: "/usr/bin/trivy")
    assert trivy_available() is True


async def test_run_trivy_secrets_sarif_returns_none_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("argus.precheck.secrets_scanner.trivy_available", lambda: False)
    result = await run_trivy_secrets_sarif("/tmp/worktree")
    assert result is None


async def test_run_trivy_secrets_sarif_parses_and_namespaces_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("argus.precheck.secrets_scanner.trivy_available", lambda: True)
    proc = _mock_subprocess(_sarif_bytes("github-pat"))

    with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
        result = await run_trivy_secrets_sarif("/tmp/worktree")

    assert result is not None
    assert result[0].rule_id == "trivy/github-pat"
    assert result[0].file == "config.py"

    args = mock_exec.call_args.args
    assert args[0] == "trivy"
    assert "secret" in args
    assert "/tmp/worktree" in args
    assert "misconfig" not in args
    assert "vuln" not in args


async def test_run_trivy_secrets_sarif_clean_scan_returns_empty_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("argus.precheck.secrets_scanner.trivy_available", lambda: True)
    proc = _mock_subprocess(_sarif_bytes())

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        result = await run_trivy_secrets_sarif("/tmp/worktree")

    assert result == []


async def test_run_trivy_secrets_sarif_returns_none_on_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("argus.precheck.secrets_scanner.trivy_available", lambda: True)
    proc = _mock_subprocess(b"", stderr=b"fatal error", returncode=1)

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        result = await run_trivy_secrets_sarif("/tmp/worktree")

    assert result is None


async def test_run_trivy_secrets_sarif_returns_none_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("argus.precheck.secrets_scanner.trivy_available", lambda: True)
    monkeypatch.setattr("argus.precheck.secrets_scanner._TRIVY_TIMEOUT_S", 0.01)

    proc = AsyncMock()

    async def _never_returns(*args: object, **kwargs: object) -> tuple[bytes, bytes]:
        import asyncio

        await asyncio.sleep(10)
        return (b"", b"")

    proc.communicate.side_effect = _never_returns
    proc.kill = lambda: None

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        result = await run_trivy_secrets_sarif("/tmp/worktree")

    assert result is None


async def test_run_trivy_secrets_sarif_returns_none_for_relative_worktree_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("argus.precheck.secrets_scanner.trivy_available", lambda: True)
    result = await run_trivy_secrets_sarif("relative/worktree")
    assert result is None
