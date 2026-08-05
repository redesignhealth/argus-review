"""Unit tests for argus.precheck.terraform_scanner.

Checkov writes SARIF to a file (``--output-file-path``), never stdout --
these tests simulate that real side effect via the mocked subprocess's
``side_effect``, inspecting the argv it was called with to find where to
write the fake SARIF content, rather than mocking file I/O directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from argus.precheck.terraform_scanner import (
    checkov_available,
    run_checkov_sarif,
)


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
                                        "artifactLocation": {"uri": "iam.tf"},
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


def _mock_exec_writing_sarif(sarif_bytes: bytes | None, returncode: int = 0):
    async def _side_effect(*args: object, **kwargs: object):
        if sarif_bytes is not None:
            argv = args
            idx = argv.index("--output-file-path")
            output_dir = argv[idx + 1]
            assert isinstance(output_dir, str)
            (Path(output_dir) / "results_sarif.sarif").write_bytes(sarif_bytes)
        proc = AsyncMock()
        proc.communicate.return_value = (b"", b"")
        proc.returncode = returncode
        return proc

    return _side_effect


def test_checkov_available_reflects_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("argus.precheck.terraform_scanner.shutil.which", lambda _: None)
    assert checkov_available() is False
    monkeypatch.setattr(
        "argus.precheck.terraform_scanner.shutil.which", lambda _: "/usr/bin/checkov"
    )
    assert checkov_available() is True


async def test_run_checkov_sarif_returns_none_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("argus.precheck.terraform_scanner.checkov_available", lambda: False)
    result = await run_checkov_sarif("/tmp/worktree", changed_files=["a.tf"])
    assert result is None


async def test_run_checkov_sarif_skips_without_spawning_when_no_tf_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("argus.precheck.terraform_scanner.checkov_available", lambda: True)
    with patch("asyncio.create_subprocess_exec") as mock_exec:
        result = await run_checkov_sarif("/tmp/worktree", changed_files=["a.py"])
    assert result == []
    mock_exec.assert_not_called()


async def test_run_checkov_sarif_skips_files_that_dont_exist_on_disk(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("argus.precheck.terraform_scanner.checkov_available", lambda: True)
    with patch("asyncio.create_subprocess_exec") as mock_exec:
        result = await run_checkov_sarif(str(tmp_path), changed_files=["does_not_exist.tf"])
    assert result == []
    mock_exec.assert_not_called()


async def test_run_checkov_sarif_parses_hits_and_namespaces_rule_id(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "iam.tf").write_text('resource "aws_iam_policy" "bad" {}\n')
    monkeypatch.setattr("argus.precheck.terraform_scanner.checkov_available", lambda: True)

    with patch(
        "asyncio.create_subprocess_exec",
        side_effect=_mock_exec_writing_sarif(_sarif_bytes("CKV_AWS_63"), returncode=1),
    ) as mock_exec:
        result = await run_checkov_sarif(str(tmp_path), changed_files=["iam.tf"])

    assert result is not None
    assert len(result) == 1
    assert result[0].rule_id == "checkov/CKV_AWS_63"
    assert result[0].file == "iam.tf"
    assert result[0].line == 1

    call = mock_exec.call_args
    assert call.kwargs["cwd"] == str(tmp_path)
    assert "--file=iam.tf" in call.args
    assert "-c" in call.args


async def test_run_checkov_sarif_uses_file_equals_form_not_bare_dash_f(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed file whose repo-root-relative path starts with "-" (e.g. a
    new top-level "-x.tf") would otherwise make Checkov's argparse `-f`
    (nargs='+') stop consuming values at that token and error out
    ("expected at least one argument") -- verified empirically that a bare
    "--" separator does NOT fix this for argparse's nargs='+' the way it
    does for squawk/eslint. "--file=<path>" (one flag per file, "=" form)
    is the verified-safe alternative: argparse treats everything after "="
    as one opaque token. Regression test for that argv shape.
    """
    (tmp_path / "-x.tf").write_text('resource "aws_iam_policy" "bad" {}\n')
    monkeypatch.setattr("argus.precheck.terraform_scanner.checkov_available", lambda: True)

    sarif_bytes = json.dumps(
        {
            "runs": [
                {
                    "results": [
                        {
                            "ruleId": "CKV_AWS_63",
                            "level": "error",
                            "message": {"text": "hit for CKV_AWS_63"},
                            "locations": [
                                {
                                    "physicalLocation": {
                                        "artifactLocation": {"uri": "-x.tf"},
                                        "region": {"startLine": 1},
                                    }
                                }
                            ],
                        }
                    ]
                }
            ]
        }
    ).encode()

    with patch(
        "asyncio.create_subprocess_exec",
        side_effect=_mock_exec_writing_sarif(sarif_bytes, returncode=1),
    ) as mock_exec:
        result = await run_checkov_sarif(str(tmp_path), changed_files=["-x.tf"])

    args = mock_exec.call_args.args
    assert "--file=-x.tf" in args
    assert "-f" not in args

    # Pin the argv change and engine.py's changed_set match together: the
    # echoed `file` must be the exact bare path, or run_precheck's
    # diff-scoping filter would silently drop this finding.
    assert result is not None
    assert len(result) == 1
    assert result[0].file == "-x.tf"


async def test_run_checkov_sarif_clean_scan_returns_empty_list(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "iam.tf").write_text('resource "aws_iam_policy" "ok" {}\n')
    monkeypatch.setattr("argus.precheck.terraform_scanner.checkov_available", lambda: True)

    with patch(
        "asyncio.create_subprocess_exec",
        side_effect=_mock_exec_writing_sarif(_sarif_bytes(), returncode=0),
    ):
        result = await run_checkov_sarif(str(tmp_path), changed_files=["iam.tf"])

    assert result == []


async def test_run_checkov_sarif_returns_none_on_genuine_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "iam.tf").write_text('resource "aws_iam_policy" "bad" {}\n')
    monkeypatch.setattr("argus.precheck.terraform_scanner.checkov_available", lambda: True)

    with patch(
        "asyncio.create_subprocess_exec",
        side_effect=_mock_exec_writing_sarif(None, returncode=2),
    ):
        result = await run_checkov_sarif(str(tmp_path), changed_files=["iam.tf"])

    assert result is None


async def test_run_checkov_sarif_returns_none_when_sarif_file_missing(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A success exit code (0 or 1) with no results_sarif.sarif actually
    written is treated as a failure, not a silent empty result.
    """
    (tmp_path / "iam.tf").write_text('resource "aws_iam_policy" "bad" {}\n')
    monkeypatch.setattr("argus.precheck.terraform_scanner.checkov_available", lambda: True)

    with patch(
        "asyncio.create_subprocess_exec",
        side_effect=_mock_exec_writing_sarif(None, returncode=1),
    ):
        result = await run_checkov_sarif(str(tmp_path), changed_files=["iam.tf"])

    assert result is None


async def test_run_checkov_sarif_returns_none_on_timeout(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "iam.tf").write_text('resource "aws_iam_policy" "bad" {}\n')
    monkeypatch.setattr("argus.precheck.terraform_scanner.checkov_available", lambda: True)
    monkeypatch.setattr("argus.precheck.terraform_scanner._CHECKOV_TIMEOUT_S", 0.01)

    proc = AsyncMock()

    async def _never_returns(*args: object, **kwargs: object) -> tuple[bytes, bytes]:
        import asyncio

        await asyncio.sleep(10)
        return (b"", b"")

    proc.communicate.side_effect = _never_returns
    proc.kill = lambda: None

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        result = await run_checkov_sarif(str(tmp_path), changed_files=["iam.tf"])

    assert result is None


async def test_run_checkov_sarif_returns_none_for_relative_worktree_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("argus.precheck.terraform_scanner.checkov_available", lambda: True)
    monkeypatch.setattr("os.path.isfile", lambda _: True)
    result = await run_checkov_sarif("relative/worktree", changed_files=["a.tf"])
    assert result is None
