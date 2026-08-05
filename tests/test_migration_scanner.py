"""Unit tests for argus.precheck.migration_scanner."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from argus.precheck.migration_scanner import (
    run_squawk_sarif,
    squawk_available,
)


def _squawk_json(*findings: dict) -> bytes:
    return json.dumps(list(findings)).encode()


def _mock_subprocess(stdout: bytes, stderr: bytes = b"", returncode: int = 0) -> AsyncMock:
    proc = AsyncMock()
    proc.communicate.return_value = (stdout, stderr)
    proc.returncode = returncode
    return proc


def test_squawk_available_reflects_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("argus.precheck.migration_scanner.shutil.which", lambda _: None)
    assert squawk_available() is False
    monkeypatch.setattr(
        "argus.precheck.migration_scanner.shutil.which", lambda _: "/usr/bin/squawk"
    )
    assert squawk_available() is True


async def test_run_squawk_sarif_returns_none_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("argus.precheck.migration_scanner.squawk_available", lambda: False)
    result = await run_squawk_sarif("/tmp/worktree", changed_files=["a.sql"])
    assert result is None


async def test_run_squawk_sarif_skips_without_spawning_when_no_sql_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("argus.precheck.migration_scanner.squawk_available", lambda: True)
    with patch("asyncio.create_subprocess_exec") as mock_exec:
        result = await run_squawk_sarif("/tmp/worktree", changed_files=["a.py", "b.md"])
    assert result == []
    mock_exec.assert_not_called()


async def test_run_squawk_sarif_skips_sql_files_that_dont_exist_on_disk(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("argus.precheck.migration_scanner.squawk_available", lambda: True)
    with patch("asyncio.create_subprocess_exec") as mock_exec:
        result = await run_squawk_sarif(str(tmp_path), changed_files=["does_not_exist.sql"])
    assert result == []
    mock_exec.assert_not_called()


async def test_run_squawk_sarif_parses_hits_and_corrects_line_index(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "001_migration.sql").write_text("CREATE INDEX idx ON t (c);\n")
    monkeypatch.setattr("argus.precheck.migration_scanner.squawk_available", lambda: True)

    proc = _mock_subprocess(
        _squawk_json(
            {
                "file": "001_migration.sql",
                "line": 0,  # squawk's JSON reporter is 0-indexed
                "rule_name": "require-concurrent-index-creation",
                "message": "blocking writes",
            }
        ),
        returncode=1,  # squawk exits 1 when findings exist
    )

    with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
        result = await run_squawk_sarif(str(tmp_path), changed_files=["001_migration.sql"])

    assert result is not None
    assert len(result) == 1
    assert result[0].rule_id == "squawk/require-concurrent-index-creation"
    assert result[0].file == "001_migration.sql"
    assert result[0].line == 1  # corrected from 0-indexed

    # Invoked from worktree_path as cwd with the relative path, not an
    # absolute one -- this is what keeps the reported path already
    # worktree-relative with no stripping needed.
    call_kwargs = mock_exec.call_args.kwargs
    assert call_kwargs["cwd"] == str(tmp_path)
    assert "001_migration.sql" in mock_exec.call_args.args


async def test_run_squawk_sarif_uses_dash_dash_before_files(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed file whose repo-root-relative path starts with "-" (e.g. a
    new top-level "-x.sql") would otherwise be parsed by squawk's clap-based
    CLI as an unknown flag, causing the whole invocation to error (verified
    empirically: exit 2, not _FINDINGS_EXIT_CODE) and silently fail open
    for every OTHER file in the same batch too. "--" before the file list
    is what prevents that -- regression test for its presence.
    """
    (tmp_path / "-x.sql").write_text("select 1;\n")
    monkeypatch.setattr("argus.precheck.migration_scanner.squawk_available", lambda: True)
    proc = _mock_subprocess(
        _squawk_json(
            {
                "file": "-x.sql",
                "line": 0,
                "column": 0,
                "level": "Warning",
                "message": "Missing `IF NOT EXISTS`",
                "help": None,
                "rule_name": "prefer-robust-stmts",
                "column_end": 1,
                "line_end": 0,
            }
        ),
        returncode=1,
    )

    with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
        result = await run_squawk_sarif(str(tmp_path), changed_files=["-x.sql"])

    args = mock_exec.call_args.args
    assert "--" in args
    dash_dash_index = args.index("--")
    assert args[dash_dash_index + 1 :] == ("-x.sql",)

    # Pin the argv change and engine.py's changed_set match together: the
    # echoed `file` must be the exact bare path, not "./-x.sql" or similar,
    # or run_precheck's diff-scoping filter would silently drop this finding.
    assert result is not None
    assert len(result) == 1
    assert result[0].file == "-x.sql"


async def test_run_squawk_sarif_excludes_noisy_style_rules(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a.sql").write_text("select 1;\n")
    monkeypatch.setattr("argus.precheck.migration_scanner.squawk_available", lambda: True)
    proc = _mock_subprocess(_squawk_json(), returncode=0)

    with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
        await run_squawk_sarif(str(tmp_path), changed_files=["a.sql"])

    args = mock_exec.call_args.args
    assert "prefer-text-field" in args
    assert "prefer-bigint-over-int" in args
    exclude_positions = [i for i, a in enumerate(args) if a == "--exclude"]
    excluded_values = {args[i + 1] for i in exclude_positions}
    assert excluded_values == {"prefer-text-field", "prefer-bigint-over-int"}


async def test_run_squawk_sarif_clean_scan_returns_empty_list(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a.sql").write_text("select 1;\n")
    monkeypatch.setattr("argus.precheck.migration_scanner.squawk_available", lambda: True)
    proc = _mock_subprocess(_squawk_json(), returncode=0)

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        result = await run_squawk_sarif(str(tmp_path), changed_files=["a.sql"])

    assert result == []


async def test_run_squawk_sarif_returns_none_on_genuine_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a.sql").write_text("select 1;\n")
    monkeypatch.setattr("argus.precheck.migration_scanner.squawk_available", lambda: True)
    proc = _mock_subprocess(b"", stderr=b"panic!", returncode=101)

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        result = await run_squawk_sarif(str(tmp_path), changed_files=["a.sql"])

    assert result is None


async def test_run_squawk_sarif_returns_none_on_timeout(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a.sql").write_text("select 1;\n")
    monkeypatch.setattr("argus.precheck.migration_scanner.squawk_available", lambda: True)
    monkeypatch.setattr("argus.precheck.migration_scanner._SQUAWK_TIMEOUT_S", 0.01)

    proc = AsyncMock()

    async def _never_returns(*args: object, **kwargs: object) -> tuple[bytes, bytes]:
        import asyncio

        await asyncio.sleep(10)
        return (b"", b"")

    proc.communicate.side_effect = _never_returns
    proc.kill = lambda: None

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        result = await run_squawk_sarif(str(tmp_path), changed_files=["a.sql"])

    assert result is None


async def test_run_squawk_sarif_returns_none_for_relative_worktree_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("argus.precheck.migration_scanner.squawk_available", lambda: True)
    monkeypatch.setattr("os.path.isfile", lambda _: True)
    result = await run_squawk_sarif("relative/worktree", changed_files=["a.sql"])
    assert result is None
