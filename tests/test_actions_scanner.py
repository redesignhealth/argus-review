"""Unit tests for argus.precheck.actions_scanner."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from argus.precheck.actions_scanner import (
    run_zizmor_sarif,
    zizmor_available,
)


def _sarif_bytes(*rule_ids: str) -> bytes:
    doc = {
        "runs": [
            {
                "results": [
                    {
                        "ruleId": rule_id,
                        "level": "warning",
                        "message": {"text": f"hit for {rule_id}"},
                        "locations": [
                            {
                                "physicalLocation": {
                                    "artifactLocation": {"uri": ".github/workflows/ci.yml"},
                                    "region": {"startLine": 10},
                                }
                            }
                        ],
                    }
                    for rule_id in rule_ids
                ]
            }
        ]
    }
    return json.dumps(doc).encode()


def _mock_subprocess(stdout: bytes, stderr: bytes = b"", returncode: int = 0) -> AsyncMock:
    proc = AsyncMock()
    proc.communicate.return_value = (stdout, stderr)
    proc.returncode = returncode
    return proc


def test_zizmor_available_reflects_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("argus.precheck.actions_scanner.shutil.which", lambda _: None)
    assert zizmor_available() is False
    monkeypatch.setattr("argus.precheck.actions_scanner.shutil.which", lambda _: "/usr/bin/zizmor")
    assert zizmor_available() is True


async def test_run_zizmor_sarif_returns_none_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("argus.precheck.actions_scanner.zizmor_available", lambda: False)
    result = await run_zizmor_sarif("/tmp/worktree")
    assert result is None


async def test_run_zizmor_sarif_parses_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("argus.precheck.actions_scanner.zizmor_available", lambda: True)
    proc = _mock_subprocess(_sarif_bytes("unpinned-uses", "template-injection"))

    with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
        result = await run_zizmor_sarif("/tmp/worktree")

    mock_exec.assert_awaited_once()
    args = mock_exec.call_args.args
    assert args[0] == "zizmor"
    assert "--offline" in args
    assert "sarif" in args
    assert args[-1] == "/tmp/worktree"
    assert result is not None
    assert [r.rule_id for r in result] == ["unpinned-uses", "template-injection"]
    assert result[0].file == ".github/workflows/ci.yml"


async def test_run_zizmor_sarif_returns_empty_list_for_genuine_no_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("argus.precheck.actions_scanner.zizmor_available", lambda: True)
    proc = _mock_subprocess(_sarif_bytes())

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        result = await run_zizmor_sarif("/tmp/worktree")

    assert result == []


async def test_run_zizmor_sarif_treats_no_inputs_collected_as_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The common case: a worktree with nothing for zizmor to audit exits
    non-zero with this specific message -- must be [] (ran, no findings),
    not None (scan failure), or every PR in a repo without GitHub Actions
    workflows would log a spurious scanner-failure warning.
    """
    monkeypatch.setattr("argus.precheck.actions_scanner.zizmor_available", lambda: True)
    proc = _mock_subprocess(b"", stderr=b"error: no inputs collected\n", returncode=3)

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        result = await run_zizmor_sarif("/tmp/worktree")

    assert result == []


async def test_run_zizmor_sarif_returns_none_on_other_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("argus.precheck.actions_scanner.zizmor_available", lambda: True)
    proc = _mock_subprocess(b"", stderr=b"error: something else broke\n", returncode=1)

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        result = await run_zizmor_sarif("/tmp/worktree")

    assert result is None


async def test_run_zizmor_sarif_returns_none_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("argus.precheck.actions_scanner.zizmor_available", lambda: True)
    monkeypatch.setattr("argus.precheck.actions_scanner._ZIZMOR_TIMEOUT_S", 0.01)

    proc = AsyncMock()

    async def _never_returns(*args: object, **kwargs: object) -> tuple[bytes, bytes]:
        import asyncio

        await asyncio.sleep(10)
        return (b"", b"")

    proc.communicate.side_effect = _never_returns
    proc.kill = lambda: None

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        result = await run_zizmor_sarif("/tmp/worktree")

    assert result is None


async def test_run_zizmor_sarif_returns_none_for_relative_worktree_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("argus.precheck.actions_scanner.zizmor_available", lambda: True)
    result = await run_zizmor_sarif("relative/worktree")
    assert result is None


async def test_run_zizmor_sarif_returns_none_when_subprocess_creation_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("argus.precheck.actions_scanner.zizmor_available", lambda: True)

    with patch("asyncio.create_subprocess_exec", side_effect=OSError("boom")):
        result = await run_zizmor_sarif("/tmp/worktree")

    assert result is None
