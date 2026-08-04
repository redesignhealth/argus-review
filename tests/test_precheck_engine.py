"""Unit tests for argus.precheck.engine."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from argus.precheck.engine import (
    PrecheckResult,
    _find_duplicate_rule_ids,
    _has_rule_files,
    _kill_and_reap,
    resolve_rules_dir,
    run_precheck,
    run_semgrep_sarif,
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
    # _kill_and_reap's transport-close path (see engine.py) does
    # `getattr(proc, "_transport", None)` -- on a bare AsyncMock, that
    # auto-vivifies as another AsyncMock, whose .close() call returns an
    # unawaited coroutine (a real transport's close() is synchronous).
    # None matches every real Process that never needed this cleanup path
    # exercised, avoiding a spurious "coroutine was never awaited" warning
    # on every test that happens to hit a _kill_and_reap call.
    proc._transport = None
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
    proc._transport = None  # see _mock_subprocess's comment on this

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


# ---------------------------------------------------------------------------
# run_semgrep_sarif: direct tests (used by both run_precheck and
# argus.precheck.shadow, which needs to tell "didn't run" (None) apart
# from "ran, no hits" ([]) -- see its own docstring).
# ---------------------------------------------------------------------------


async def test_run_semgrep_sarif_returns_none_when_semgrep_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("argus.precheck.engine.semgrep_available", lambda: False)
    result = await run_semgrep_sarif("/tmp/worktree", Path("/tmp/rules"))
    assert result is None


async def test_run_semgrep_sarif_returns_none_on_timeout(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("argus.precheck.engine.semgrep_available", lambda: True)
    monkeypatch.setattr("argus.precheck.engine._SEMGREP_TIMEOUT_S", 0.01)

    proc = AsyncMock()
    proc._transport = None  # see _mock_subprocess's comment on this
    call_count = {"n": 0}

    async def _communicate(*args: object, **kwargs: object) -> tuple[bytes, bytes]:
        import asyncio

        # Only the FIRST call needs to actually sleep past the patched
        # timeout -- that's the one asyncio.wait_for wraps, and it needs a
        # real slow awaitable to genuinely trigger TimeoutError. The SECOND
        # call is the production code's post-kill drain in _kill_and_reap,
        # which IS bounded (by _KILL_DRAIN_TIMEOUT_S, separately from this
        # test's patched _SEMGREP_TIMEOUT_S) -- but an unconditionally
        # sleeping mock would still block this test for the real 10s on
        # that call too if it also slept, instead of the intended ~0.01s.
        call_count["n"] += 1
        if call_count["n"] == 1:
            await asyncio.sleep(10)
        return (b"", b"")

    proc.communicate.side_effect = _communicate
    proc.kill = lambda: None

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        result = await run_semgrep_sarif("/tmp/worktree", tmp_path)

    assert result is None


async def test_run_semgrep_sarif_returns_none_on_nonzero_exit(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("argus.precheck.engine.semgrep_available", lambda: True)
    proc = _mock_subprocess(b"", returncode=2)

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        result = await run_semgrep_sarif("/tmp/worktree", tmp_path)

    assert result is None


async def test_run_semgrep_sarif_returns_empty_list_for_genuine_no_hits(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("argus.precheck.engine.semgrep_available", lambda: True)
    proc = _mock_subprocess(_sarif_bytes())  # no rule_ids -> empty results list

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        result = await run_semgrep_sarif("/tmp/worktree", tmp_path)

    assert result == []


async def test_run_semgrep_sarif_accepts_a_single_rule_file_as_config_path(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("argus.precheck.engine.semgrep_available", lambda: True)
    rule_file = tmp_path / "one-rule.yml"
    rule_file.write_text("rules: []\n")
    proc = _mock_subprocess(_sarif_bytes("r1"))

    with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
        result = await run_semgrep_sarif("/tmp/worktree", rule_file)

    assert result is not None
    assert [r.rule_id for r in result] == ["r1"]
    # config_path is passed through exactly as given -- no cwd juggling, no
    # relative-ification. --no-rewrite-rule-ids is what keeps ruleId bare
    # (see run_semgrep_sarif's docstring for why an earlier version instead
    # manipulated cwd/relative-config for this, and why that was insufficient
    # for nested rule directories).
    assert mock_exec.await_args is not None
    assert str(rule_file) in mock_exec.await_args.args
    assert "--no-rewrite-rule-ids" in mock_exec.await_args.args
    assert "cwd" not in mock_exec.await_args.kwargs


async def test_run_semgrep_sarif_directory_config_passed_as_is(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("argus.precheck.engine.semgrep_available", lambda: True)
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    proc = _mock_subprocess(_sarif_bytes("r1"))

    with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
        result = await run_semgrep_sarif("/tmp/worktree", rules_dir)

    assert result is not None
    assert [r.rule_id for r in result] == ["r1"]
    assert mock_exec.await_args is not None
    assert str(rules_dir) in mock_exec.await_args.args
    assert "--no-rewrite-rule-ids" in mock_exec.await_args.args
    assert "cwd" not in mock_exec.await_args.kwargs


async def test_run_semgrep_sarif_returns_none_when_subprocess_creation_raises(
    tmp_path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Enforces the documented "never raises" contract shadow.py's own
    # narrower try/except now depends on -- a TOCTOU race (semgrep vanishes
    # from PATH between semgrep_available() and this call) or any other
    # create_subprocess_exec failure must fail open, not propagate.
    monkeypatch.setattr("argus.precheck.engine.semgrep_available", lambda: True)

    with (
        patch("asyncio.create_subprocess_exec", side_effect=OSError("boom")),
        caplog.at_level("WARNING", logger="argus.precheck.engine"),
    ):
        result = await run_semgrep_sarif("/tmp/worktree", tmp_path)

    assert result is None
    assert any("failed unexpectedly" in record.message for record in caplog.records)


async def test_run_semgrep_sarif_returns_none_when_sarif_parsing_raises(
    tmp_path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr("argus.precheck.engine.semgrep_available", lambda: True)
    proc = _mock_subprocess(b"not valid json")

    with (
        patch("asyncio.create_subprocess_exec", return_value=proc),
        patch(
            "argus.precheck.engine.parse_semgrep_sarif", side_effect=ValueError("malformed SARIF")
        ),
        caplog.at_level("WARNING", logger="argus.precheck.engine"),
    ):
        result = await run_semgrep_sarif("/tmp/worktree", tmp_path)

    assert result is None
    assert any("failed unexpectedly" in record.message for record in caplog.records)


async def test_run_semgrep_sarif_kills_and_reaps_on_non_timeout_exception(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A non-TimeoutError exception raised while communicate() is in flight
    # (e.g. OSError/BrokenPipeError) must still kill/reap the child before
    # propagating out of _run_semgrep_sarif_unguarded -- otherwise the
    # process is orphaned rather than merely "the scan failed."
    monkeypatch.setattr("argus.precheck.engine.semgrep_available", lambda: True)
    proc = AsyncMock()
    proc._transport = None  # see _mock_subprocess's comment on this
    proc.communicate.side_effect = [OSError("pipe broke"), (b"", b"")]
    proc.kill = MagicMock()

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        result = await run_semgrep_sarif("/tmp/worktree", tmp_path)

    assert result is None
    proc.kill.assert_called_once()
    assert proc.communicate.await_count == 2  # the failing call, then the post-kill drain


async def test_run_semgrep_sarif_suppresses_process_lookup_error_on_kill(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Covers _kill_and_reap's other suppress branch: the child can already
    # have exited on its own (race) by the time kill() is called.
    monkeypatch.setattr("argus.precheck.engine.semgrep_available", lambda: True)
    proc = AsyncMock()
    proc._transport = None  # see _mock_subprocess's comment on this
    proc.communicate.side_effect = [OSError("pipe broke"), (b"", b"")]
    proc.kill = MagicMock(side_effect=ProcessLookupError)

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        result = await run_semgrep_sarif("/tmp/worktree", tmp_path)

    assert result is None
    proc.kill.assert_called_once()
    assert proc.communicate.await_count == 2


# ---------------------------------------------------------------------------
# _find_duplicate_rule_ids
# ---------------------------------------------------------------------------


def test_find_duplicate_rule_ids_returns_empty_for_unique_ids(tmp_path) -> None:
    (tmp_path / "a.yml").write_text("rules:\n  - id: rule-a\n")
    (tmp_path / "b.yml").write_text("rules:\n  - id: rule-b\n")

    assert _find_duplicate_rule_ids(tmp_path) == {}


def test_find_duplicate_rule_ids_detects_collision_across_files(tmp_path) -> None:
    (tmp_path / "a.yml").write_text("rules:\n  - id: shared-id\n")
    sub = tmp_path / "security"
    sub.mkdir()
    (sub / "b.yml").write_text("rules:\n  - id: shared-id\n")

    duplicates = _find_duplicate_rule_ids(tmp_path)

    assert set(duplicates.keys()) == {"shared-id"}
    assert set(duplicates["shared-id"]) == {tmp_path / "a.yml", sub / "b.yml"}


def test_find_duplicate_rule_ids_detects_collision_within_one_file(tmp_path) -> None:
    (tmp_path / "a.yml").write_text("rules:\n  - id: shared-id\n  - id: shared-id\n")

    duplicates = _find_duplicate_rule_ids(tmp_path)

    assert list(duplicates.keys()) == ["shared-id"]
    assert duplicates["shared-id"] == [tmp_path / "a.yml", tmp_path / "a.yml"]


def test_find_duplicate_rule_ids_skips_unparseable_files(tmp_path) -> None:
    (tmp_path / "bad.yml").write_text(": not: valid: yaml: [")
    (tmp_path / "good.yml").write_text("rules:\n  - id: fine\n")

    assert _find_duplicate_rule_ids(tmp_path) == {}


def test_find_duplicate_rule_ids_skips_non_dict_rule_entries(tmp_path) -> None:
    (tmp_path / "a.yml").write_text("rules:\n  - id: fine\n  - just-a-string\n")

    assert _find_duplicate_rule_ids(tmp_path) == {}


def test_find_duplicate_rule_ids_skips_non_list_rules_value(tmp_path) -> None:
    # A `rules:` key that isn't a list (e.g. a typo'd dict/string) must be
    # skipped, not raise or misbehave.
    (tmp_path / "a.yml").write_text("rules: not-a-list\n")

    assert _find_duplicate_rule_ids(tmp_path) == {}


def test_find_duplicate_rule_ids_detects_collision_across_yml_and_yaml_extensions(
    tmp_path,
) -> None:
    (tmp_path / "a.yml").write_text("rules:\n  - id: shared-id\n")
    (tmp_path / "b.yaml").write_text("rules:\n  - id: shared-id\n")

    duplicates = _find_duplicate_rule_ids(tmp_path)

    assert set(duplicates.keys()) == {"shared-id"}
    assert set(duplicates["shared-id"]) == {tmp_path / "a.yml", tmp_path / "b.yaml"}


async def test_run_semgrep_sarif_returns_none_for_relative_worktree_path(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The isabs check (a raised ValueError, not a bare assert -- see the
    # comment at its call site for why) must fail open via
    # run_semgrep_sarif's own outer except, not propagate.
    monkeypatch.setattr("argus.precheck.engine.semgrep_available", lambda: True)

    result = await run_semgrep_sarif("relative/worktree", tmp_path)

    assert result is None


async def test_kill_and_reap_closes_transport_when_drain_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("argus.precheck.engine._KILL_DRAIN_TIMEOUT_S", 0.01)
    proc = AsyncMock()
    proc.kill = MagicMock()

    async def _hangs_forever(*args: object, **kwargs: object) -> tuple[bytes, bytes]:
        await asyncio.sleep(10)
        return (b"", b"")

    proc.communicate.side_effect = _hangs_forever
    transport = MagicMock()
    proc._transport = transport

    await _kill_and_reap(proc)

    proc.kill.assert_called_once()
    transport.close.assert_called_once()


async def test_run_precheck_logs_warning_on_duplicate_rule_ids(
    tmp_path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr("argus.precheck.engine.semgrep_available", lambda: True)
    monkeypatch.setattr("argus.precheck.engine.resolve_rules_dir", lambda: tmp_path)
    (tmp_path / "a.yml").write_text("rules:\n  - id: shared-id\n")
    (tmp_path / "b.yml").write_text("rules:\n  - id: shared-id\n")
    proc = _mock_subprocess(_sarif_bytes())

    with (
        patch("asyncio.create_subprocess_exec", return_value=proc),
        caplog.at_level("WARNING", logger="argus.precheck.engine"),
    ):
        await run_precheck("/tmp/worktree")

    assert any("Duplicate precheck rule id" in record.message for record in caplog.records)
