"""Unit tests for argus.precheck.js_scanner."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from argus.precheck.js_scanner import eslint_available, run_eslint_sarif


def _eslint_json(*file_results: dict) -> bytes:
    return json.dumps(list(file_results)).encode()


def _mock_subprocess(stdout: bytes, stderr: bytes = b"", returncode: int = 0) -> AsyncMock:
    proc = AsyncMock()
    proc.communicate.return_value = (stdout, stderr)
    proc.returncode = returncode
    return proc


def test_eslint_available_reflects_bundle_presence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pathlib.Path.is_file", lambda self: False)
    assert eslint_available() is False
    monkeypatch.setattr("pathlib.Path.is_file", lambda self: True)
    assert eslint_available() is True


async def test_run_eslint_sarif_returns_none_when_bundle_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("argus.precheck.js_scanner.eslint_available", lambda: False)
    result = await run_eslint_sarif("/tmp/worktree", changed_files=["a.js"])
    assert result is None


async def test_run_eslint_sarif_skips_without_spawning_when_no_js_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("argus.precheck.js_scanner.eslint_available", lambda: True)
    with patch("asyncio.create_subprocess_exec") as mock_exec:
        result = await run_eslint_sarif("/tmp/worktree", changed_files=["a.py", "b.md"])
    assert result == []
    mock_exec.assert_not_called()


async def test_run_eslint_sarif_skips_files_that_dont_exist_on_disk(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("argus.precheck.js_scanner.eslint_available", lambda: True)
    with patch("asyncio.create_subprocess_exec") as mock_exec:
        result = await run_eslint_sarif(str(tmp_path), changed_files=["does_not_exist.js"])
    assert result == []
    mock_exec.assert_not_called()


async def test_run_eslint_sarif_parses_hits_and_strips_absolute_path(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "bad.js").write_text("exec(userInput);\n")
    monkeypatch.setattr("argus.precheck.js_scanner.eslint_available", lambda: True)

    proc = _mock_subprocess(
        _eslint_json(
            {
                "filePath": f"{tmp_path}/bad.js",
                "messages": [
                    {
                        "ruleId": "security/detect-child-process",
                        "severity": 1,
                        "message": "Found child_process.exec()",
                        "line": 1,
                    }
                ],
            }
        )
    )

    with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
        result = await run_eslint_sarif(str(tmp_path), changed_files=["bad.js"])

    assert result is not None
    assert len(result) == 1
    assert result[0].rule_id == "security/detect-child-process"
    assert result[0].file == "bad.js"
    assert result[0].line == 1
    assert result[0].level == "warning"

    call = mock_exec.call_args
    assert call.kwargs["cwd"] == str(tmp_path)
    assert "bad.js" in call.args
    assert "--no-config-lookup" in call.args


async def test_run_eslint_sarif_uses_dash_dash_before_files(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed file whose repo-root-relative path starts with "-" (e.g. a
    new top-level "-x.js") would otherwise be parsed by eslint's yargs-based
    CLI as an unknown flag (verified empirically: "Invalid option '-e'"),
    silently producing no output for the WHOLE batch, not just the
    offending file. "--" before the file list is what prevents that --
    regression test for its presence.
    """
    (tmp_path / "-x.js").write_text("exec(userInput);\n")
    monkeypatch.setattr("argus.precheck.js_scanner.eslint_available", lambda: True)
    proc = _mock_subprocess(
        _eslint_json(
            {
                "filePath": f"{tmp_path}/-x.js",
                "messages": [
                    {
                        "ruleId": "security/detect-child-process",
                        "severity": 1,
                        "message": "Found child_process.exec()",
                        "line": 1,
                    }
                ],
            }
        )
    )

    with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
        result = await run_eslint_sarif(str(tmp_path), changed_files=["-x.js"])

    args = mock_exec.call_args.args
    assert "--" in args
    dash_dash_index = args.index("--")
    assert args[dash_dash_index + 1 :] == ("-x.js",)

    # Pin the argv change and engine.py's changed_set match together: the
    # stripped `file` must be the exact bare path, or run_precheck's
    # diff-scoping filter would silently drop this finding.
    assert result is not None
    assert len(result) == 1
    assert result[0].file == "-x.js"


async def test_run_eslint_sarif_maps_severity_2_to_error(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "bad.js").write_text("exec(userInput);\n")
    monkeypatch.setattr("argus.precheck.js_scanner.eslint_available", lambda: True)

    proc = _mock_subprocess(
        _eslint_json(
            {
                "filePath": f"{tmp_path}/bad.js",
                "messages": [
                    {"ruleId": "security/detect-eval-with-expression", "severity": 2, "message": "m", "line": 2}
                ],
            }
        )
    )

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        result = await run_eslint_sarif(str(tmp_path), changed_files=["bad.js"])

    assert result is not None
    assert result[0].level == "error"


async def test_run_eslint_sarif_clean_scan_returns_empty_list(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "clean.js").write_text("console.log('hi');\n")
    monkeypatch.setattr("argus.precheck.js_scanner.eslint_available", lambda: True)
    proc = _mock_subprocess(_eslint_json({"filePath": f"{tmp_path}/clean.js", "messages": []}))

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        result = await run_eslint_sarif(str(tmp_path), changed_files=["clean.js"])

    assert result == []


async def test_run_eslint_sarif_returns_none_on_genuine_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "bad.js").write_text("x\n")
    monkeypatch.setattr("argus.precheck.js_scanner.eslint_available", lambda: True)
    proc = _mock_subprocess(b"", stderr=b"config error", returncode=2)

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        result = await run_eslint_sarif(str(tmp_path), changed_files=["bad.js"])

    assert result is None


async def test_run_eslint_sarif_returns_none_on_timeout(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "bad.js").write_text("x\n")
    monkeypatch.setattr("argus.precheck.js_scanner.eslint_available", lambda: True)
    monkeypatch.setattr("argus.precheck.js_scanner._ESLINT_TIMEOUT_S", 0.01)

    proc = AsyncMock()

    async def _never_returns(*args: object, **kwargs: object) -> tuple[bytes, bytes]:
        import asyncio

        await asyncio.sleep(10)
        return (b"", b"")

    proc.communicate.side_effect = _never_returns
    proc.kill = lambda: None

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        result = await run_eslint_sarif(str(tmp_path), changed_files=["bad.js"])

    assert result is None


async def test_run_eslint_sarif_returns_none_for_relative_worktree_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("argus.precheck.js_scanner.eslint_available", lambda: True)
    monkeypatch.setattr("os.path.isfile", lambda _: True)
    result = await run_eslint_sarif("relative/worktree", changed_files=["a.js"])
    assert result is None
