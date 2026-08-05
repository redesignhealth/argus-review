"""Unit tests for argus.precheck.workflow_lint_scanner."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from argus.precheck.workflow_lint_scanner import (
    actionlint_available,
    run_actionlint_sarif,
)


def _actionlint_json(*findings: dict) -> bytes:
    return json.dumps(list(findings)).encode()


def _mock_subprocess(stdout: bytes, stderr: bytes = b"", returncode: int = 0) -> AsyncMock:
    proc = AsyncMock()
    proc.communicate.return_value = (stdout, stderr)
    proc.returncode = returncode
    return proc


def test_actionlint_available_reflects_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("argus.precheck.workflow_lint_scanner.shutil.which", lambda _: None)
    assert actionlint_available() is False
    monkeypatch.setattr(
        "argus.precheck.workflow_lint_scanner.shutil.which", lambda _: "/usr/bin/actionlint"
    )
    assert actionlint_available() is True


async def test_run_actionlint_sarif_returns_none_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("argus.precheck.workflow_lint_scanner.actionlint_available", lambda: False)
    result = await run_actionlint_sarif("/tmp/worktree", changed_files=[".github/workflows/ci.yml"])
    assert result is None


async def test_run_actionlint_sarif_skips_without_spawning_when_no_workflow_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("argus.precheck.workflow_lint_scanner.actionlint_available", lambda: True)
    with patch("asyncio.create_subprocess_exec") as mock_exec:
        result = await run_actionlint_sarif("/tmp/worktree", changed_files=["src/main.py"])
    assert result == []
    mock_exec.assert_not_called()


async def test_run_actionlint_sarif_ignores_non_workflow_yaml(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A .yml file outside .github/workflows/ (e.g. a Docker Compose file)
    is not something actionlint should ever be pointed at.
    """
    (tmp_path / "docker-compose.yml").write_text("services: {}\n")
    monkeypatch.setattr("argus.precheck.workflow_lint_scanner.actionlint_available", lambda: True)
    with patch("asyncio.create_subprocess_exec") as mock_exec:
        result = await run_actionlint_sarif(str(tmp_path), changed_files=["docker-compose.yml"])
    assert result == []
    mock_exec.assert_not_called()


async def test_run_actionlint_sarif_skips_files_that_dont_exist_on_disk(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("argus.precheck.workflow_lint_scanner.actionlint_available", lambda: True)
    with patch("asyncio.create_subprocess_exec") as mock_exec:
        result = await run_actionlint_sarif(
            str(tmp_path), changed_files=[".github/workflows/does_not_exist.yml"]
        )
    assert result == []
    mock_exec.assert_not_called()


async def test_run_actionlint_sarif_parses_hits(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "ci.yml").write_text("name: ci\non: push\njobs: {}\n")
    monkeypatch.setattr("argus.precheck.workflow_lint_scanner.actionlint_available", lambda: True)

    proc = _mock_subprocess(
        _actionlint_json(
            {
                "message": 'input "nodee-version" is not defined',
                "filepath": ".github/workflows/ci.yml",
                "line": 9,
                "kind": "action",
            }
        ),
        returncode=1,
    )

    with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
        result = await run_actionlint_sarif(
            str(tmp_path), changed_files=[".github/workflows/ci.yml"]
        )

    assert result is not None
    assert len(result) == 1
    assert result[0].rule_id == "actionlint/action"
    assert result[0].file == ".github/workflows/ci.yml"
    assert result[0].line == 9

    call_kwargs = mock_exec.call_args.kwargs
    assert call_kwargs["cwd"] == str(tmp_path)
    assert ".github/workflows/ci.yml" in mock_exec.call_args.args


async def test_run_actionlint_sarif_uses_dash_dash_before_files(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mirrors the equivalent regression test for squawk/eslint/checkov, for
    consistency -- though actionlint's own files are always prefixed with
    ".github/workflows/" (the scope filter above), so no argv token here can
    itself start with "-" even if the bare filename does. "--" is added
    anyway for defense in depth; verified empirically it doesn't change
    actionlint's behavior or path echo.
    """
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "-x.yml").write_text("name: ci\non: push\njobs: {}\n")
    monkeypatch.setattr("argus.precheck.workflow_lint_scanner.actionlint_available", lambda: True)
    proc = _mock_subprocess(_actionlint_json())

    with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
        await run_actionlint_sarif(str(tmp_path), changed_files=[".github/workflows/-x.yml"])

    args = mock_exec.call_args.args
    assert "--" in args
    dash_dash_index = args.index("--")
    assert args[dash_dash_index + 1 :] == (".github/workflows/-x.yml",)


async def test_run_actionlint_sarif_extracts_shellcheck_code(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "ci.yml").write_text("name: ci\non: push\njobs: {}\n")
    monkeypatch.setattr("argus.precheck.workflow_lint_scanner.actionlint_available", lambda: True)

    proc = _mock_subprocess(
        _actionlint_json(
            {
                "message": "shellcheck reported issue in this script: SC2034:warning:1:1 FOO appears unused.",
                "filepath": ".github/workflows/ci.yml",
                "line": 10,
                "kind": "shellcheck",
            }
        ),
        returncode=1,
    )

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        result = await run_actionlint_sarif(
            str(tmp_path), changed_files=[".github/workflows/ci.yml"]
        )

    assert result is not None
    assert result[0].rule_id == "actionlint/shellcheck/SC2034"


async def test_run_actionlint_sarif_clean_scan_returns_empty_list(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "ci.yml").write_text("name: ci\non: push\njobs: {}\n")
    monkeypatch.setattr("argus.precheck.workflow_lint_scanner.actionlint_available", lambda: True)
    proc = _mock_subprocess(_actionlint_json(), returncode=0)

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        result = await run_actionlint_sarif(
            str(tmp_path), changed_files=[".github/workflows/ci.yml"]
        )

    assert result == []


async def test_run_actionlint_sarif_returns_none_on_genuine_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "ci.yml").write_text("name: ci\non: push\njobs: {}\n")
    monkeypatch.setattr("argus.precheck.workflow_lint_scanner.actionlint_available", lambda: True)
    proc = _mock_subprocess(b"", stderr=b"could not read file", returncode=3)

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        result = await run_actionlint_sarif(
            str(tmp_path), changed_files=[".github/workflows/ci.yml"]
        )

    assert result is None


async def test_run_actionlint_sarif_returns_none_on_timeout(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "ci.yml").write_text("name: ci\non: push\njobs: {}\n")
    monkeypatch.setattr("argus.precheck.workflow_lint_scanner.actionlint_available", lambda: True)
    monkeypatch.setattr("argus.precheck.workflow_lint_scanner._ACTIONLINT_TIMEOUT_S", 0.01)

    proc = AsyncMock()

    async def _never_returns(*args: object, **kwargs: object) -> tuple[bytes, bytes]:
        import asyncio

        await asyncio.sleep(10)
        return (b"", b"")

    proc.communicate.side_effect = _never_returns
    proc.kill = lambda: None

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        result = await run_actionlint_sarif(
            str(tmp_path), changed_files=[".github/workflows/ci.yml"]
        )

    assert result is None


async def test_run_actionlint_sarif_returns_none_for_relative_worktree_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("argus.precheck.workflow_lint_scanner.actionlint_available", lambda: True)
    monkeypatch.setattr("os.path.isfile", lambda _: True)
    result = await run_actionlint_sarif(
        "relative/worktree", changed_files=[".github/workflows/ci.yml"]
    )
    assert result is None
