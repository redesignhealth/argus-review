"""Unit tests for argus.precheck.engine."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from argus.precheck.engine import (
    PrecheckResult,
    _has_rule_files,
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


def test_has_rule_files_finds_nested_rules(tmp_path) -> None:
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "nested.yml").write_text("rules: []\n")
    assert _has_rule_files(tmp_path) is True


def test_has_rule_files_false_for_empty_dir(tmp_path) -> None:
    assert _has_rule_files(tmp_path) is False


def _sarif_bytes_with_file(rule_id: str, file: str) -> bytes:
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
                                    "artifactLocation": {"uri": file},
                                    "region": {"startLine": 1},
                                }
                            }
                        ],
                    }
                ]
            }
        ]
    }
    return json.dumps(doc).encode()


async def test_run_precheck_strips_absolute_worktree_prefix(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "rule.yml").write_text("rules: []\n")
    monkeypatch.setattr("argus.precheck.engine.semgrep_available", lambda: True)
    monkeypatch.setattr("argus.precheck.engine.resolve_rules_dir", lambda: tmp_path)

    worktree = "/tmp/some-worktree"
    proc = _mock_subprocess(_sarif_bytes_with_file("r1", f"{worktree}/src/app.py"))

    with (
        patch("asyncio.create_subprocess_exec", return_value=proc),
        patch("argus.precheck.engine.select_rule_statuses", new=AsyncMock(return_value={})),
    ):
        result = await run_precheck(worktree)

    assert result.candidate_findings[0].file == "src/app.py"


async def test_run_precheck_leaves_non_worktree_paths_untouched(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "rule.yml").write_text("rules: []\n")
    monkeypatch.setattr("argus.precheck.engine.semgrep_available", lambda: True)
    monkeypatch.setattr("argus.precheck.engine.resolve_rules_dir", lambda: tmp_path)

    proc = _mock_subprocess(_sarif_bytes_with_file("r1", "src/app.py"))

    with (
        patch("asyncio.create_subprocess_exec", return_value=proc),
        patch("argus.precheck.engine.select_rule_statuses", new=AsyncMock(return_value={})),
    ):
        result = await run_precheck("/tmp/some-worktree")

    assert result.candidate_findings[0].file == "src/app.py"


async def test_run_precheck_caps_result_count(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from argus.precheck.engine import _MAX_RESULTS

    (tmp_path / "rule.yml").write_text("rules: []\n")
    monkeypatch.setattr("argus.precheck.engine.semgrep_available", lambda: True)
    monkeypatch.setattr("argus.precheck.engine.resolve_rules_dir", lambda: tmp_path)

    many_rule_ids = [f"r{i}" for i in range(_MAX_RESULTS + 10)]
    proc = _mock_subprocess(_sarif_bytes(*many_rule_ids))

    with (
        patch("asyncio.create_subprocess_exec", return_value=proc),
        patch("argus.precheck.engine.select_rule_statuses", new=AsyncMock(return_value={})),
    ):
        result = await run_precheck("/tmp/worktree")

    assert len(result.candidate_findings) == _MAX_RESULTS


async def test_run_precheck_verified_finding_survives_the_cap(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression test: an earlier version capped the raw SARIF results list
    # *before* classifying by rule status, so a verified (fast-fail) hit
    # landing past the cap boundary in SARIF's scan-order-dependent results
    # list would be silently dropped -- letting a PR that should have been
    # blocked continue through the full LLM pipeline instead. The cap must
    # only ever apply to candidate_findings.
    from argus.precheck.engine import _MAX_RESULTS

    (tmp_path / "rule.yml").write_text("rules: []\n")
    monkeypatch.setattr("argus.precheck.engine.semgrep_available", lambda: True)
    monkeypatch.setattr("argus.precheck.engine.resolve_rules_dir", lambda: tmp_path)

    # One verified rule ID placed LAST, after far more than _MAX_RESULTS
    # candidate hits -- exactly the position a naive pre-classification cap
    # would drop.
    many_candidate_ids = [f"candidate-{i}" for i in range(_MAX_RESULTS + 10)]
    all_rule_ids = [*many_candidate_ids, "verified-rule"]
    proc = _mock_subprocess(_sarif_bytes(*all_rule_ids))

    with (
        patch("asyncio.create_subprocess_exec", return_value=proc),
        patch(
            "argus.precheck.engine.select_rule_statuses",
            new=AsyncMock(return_value={"verified-rule": "verified"}),
        ),
    ):
        result = await run_precheck("/tmp/worktree")

    assert [f.rule_id for f in result.verified_findings] == ["verified-rule"]
    assert len(result.candidate_findings) == _MAX_RESULTS


async def test_run_precheck_strips_realpath_resolved_prefix(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Distinct from test_run_precheck_strips_absolute_worktree_prefix: this
    # exercises the os.path.realpath() branch specifically, simulating a
    # symlink-resolved scan (e.g. macOS's /var -> /private/var) where the
    # SARIF file path matches the *resolved* form, not the literal
    # worktree_path passed to semgrep.
    (tmp_path / "rule.yml").write_text("rules: []\n")
    monkeypatch.setattr("argus.precheck.engine.semgrep_available", lambda: True)
    monkeypatch.setattr("argus.precheck.engine.resolve_rules_dir", lambda: tmp_path)

    worktree = "/var/folders/worktree"
    resolved = "/private/var/folders/worktree"
    monkeypatch.setattr(
        "argus.precheck.engine.os.path.realpath",
        lambda p: resolved if p == worktree else p,
    )

    proc = _mock_subprocess(_sarif_bytes_with_file("r1", f"{resolved}/src/app.py"))

    with (
        patch("asyncio.create_subprocess_exec", return_value=proc),
        patch("argus.precheck.engine.select_rule_statuses", new=AsyncMock(return_value={})),
    ):
        result = await run_precheck(worktree)

    assert result.candidate_findings[0].file == "src/app.py"
