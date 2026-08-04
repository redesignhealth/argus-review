"""Unit tests for argus.precheck.engine."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from argus.precheck.engine import (
    PrecheckResult,
    resolve_rules_dir,
    run_precheck,
    semgrep_available,
)


def test_resolve_rules_dir_prefers_override(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARGUS_RULES_DIR", str(tmp_path))
    from argus.config import clear_cache

    clear_cache()
    assert resolve_rules_dir() == tmp_path


def test_resolve_rules_dir_nonexistent_override_returns_none(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARGUS_RULES_DIR", str(tmp_path / "does-not-exist"))
    from argus.config import clear_cache

    clear_cache()
    assert resolve_rules_dir() is None


def test_resolve_rules_dir_falls_back_to_packaged_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARGUS_RULES_DIR", raising=False)
    from argus.config import clear_cache

    clear_cache()
    packaged = resolve_rules_dir()
    assert packaged is not None
    assert packaged.name == "rules"


def test_semgrep_available_reflects_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("argus.precheck.engine.shutil.which", lambda _: None)
    assert semgrep_available() is False
    monkeypatch.setattr("argus.precheck.engine.shutil.which", lambda _: "/usr/bin/semgrep")
    assert semgrep_available() is True


async def test_run_precheck_noop_when_semgrep_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("argus.precheck.engine.semgrep_available", lambda: False)
    result = await run_precheck("/tmp/worktree")
    assert result == PrecheckResult()


async def test_run_precheck_noop_when_no_rule_files(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("argus.precheck.engine.semgrep_available", lambda: True)
    monkeypatch.setattr("argus.precheck.engine.resolve_rules_dir", lambda: tmp_path)
    result = await run_precheck("/tmp/worktree")
    assert result == PrecheckResult()


def _sarif_bytes(*rule_ids: str) -> bytes:
    doc = {
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
                                    "artifactLocation": {"uri": "a.py"},
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
    return json.dumps(doc).encode()


def _mock_subprocess(stdout: bytes, returncode: int = 0) -> AsyncMock:
    proc = AsyncMock()
    proc.communicate.return_value = (stdout, b"")
    proc.returncode = returncode
    return proc


async def test_run_precheck_classifies_by_status(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "rule.yml").write_text("rules: []\n")
    monkeypatch.setattr("argus.precheck.engine.semgrep_available", lambda: True)
    monkeypatch.setattr("argus.precheck.engine.resolve_rules_dir", lambda: tmp_path)

    proc = _mock_subprocess(_sarif_bytes("verified-rule", "candidate-rule", "suspended-rule"))

    with (
        patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec,
        patch(
            "argus.precheck.engine.select_rule_statuses",
            new=AsyncMock(
                return_value={"verified-rule": "verified", "suspended-rule": "suspended"}
            ),
        ),
    ):
        result = await run_precheck("/tmp/worktree")

    mock_exec.assert_awaited_once()
    assert [f.rule_id for f in result.verified_findings] == ["verified-rule"]
    assert [f.rule_id for f in result.candidate_findings] == ["candidate-rule"]


async def test_run_precheck_unknown_rule_id_defaults_to_candidate(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "rule.yml").write_text("rules: []\n")
    monkeypatch.setattr("argus.precheck.engine.semgrep_available", lambda: True)
    monkeypatch.setattr("argus.precheck.engine.resolve_rules_dir", lambda: tmp_path)

    proc = _mock_subprocess(_sarif_bytes("brand-new-rule"))

    with (
        patch("asyncio.create_subprocess_exec", return_value=proc),
        patch("argus.precheck.engine.select_rule_statuses", new=AsyncMock(return_value={})),
    ):
        result = await run_precheck("/tmp/worktree")

    assert result.verified_findings == []
    assert [f.rule_id for f in result.candidate_findings] == ["brand-new-rule"]


async def test_run_precheck_nonzero_exit_returns_empty(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "rule.yml").write_text("rules: []\n")
    monkeypatch.setattr("argus.precheck.engine.semgrep_available", lambda: True)
    monkeypatch.setattr("argus.precheck.engine.resolve_rules_dir", lambda: tmp_path)

    proc = _mock_subprocess(b"", returncode=2)

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        result = await run_precheck("/tmp/worktree")

    assert result == PrecheckResult()


async def test_run_precheck_timeout_returns_empty(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "rule.yml").write_text("rules: []\n")
    monkeypatch.setattr("argus.precheck.engine.semgrep_available", lambda: True)
    monkeypatch.setattr("argus.precheck.engine.resolve_rules_dir", lambda: tmp_path)
    monkeypatch.setattr("argus.precheck.engine._SEMGREP_TIMEOUT_S", 0.01)

    proc = AsyncMock()

    async def _never_returns(*args: object, **kwargs: object) -> tuple[bytes, bytes]:
        import asyncio

        await asyncio.sleep(10)
        return (b"", b"")

    proc.communicate.side_effect = _never_returns
    proc.kill = lambda: None

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        result = await run_precheck("/tmp/worktree")

    assert result == PrecheckResult()
