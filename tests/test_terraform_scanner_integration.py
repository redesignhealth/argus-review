"""Integration tests for argus.precheck.terraform_scanner -- real checkov subprocess.

Unlike tests/test_terraform_scanner.py (which mocks asyncio.create_subprocess_exec
entirely), these exercise the real subprocess/SARIF-parsing path end to end
against an actual `checkov` binary. No network access needed -- Checkov's
Terraform static analysis runs entirely locally against a scratch directory.
Skipped, not failed, when checkov isn't on PATH, matching this repo's
existing integration-test convention (see test_precheck_engine_integration.py).
"""

from __future__ import annotations

import shutil

import pytest

from argus.precheck.scanner_utils import run_scanner_subprocess
from argus.precheck.terraform_scanner import _FINDINGS_EXIT_CODE, run_checkov_sarif

pytestmark = pytest.mark.integration

if not shutil.which("checkov"):
    pytest.skip(
        "checkov not on PATH (install the `prechecks` extra) -- skipping integration test",
        allow_module_level=True,
    )


_WILDCARD_IAM_POLICY_TF = """\
resource "aws_iam_policy" "bad" {
  name = "bad-policy"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "*"
      Resource = "*"
    }]
  })
}
"""


async def test_real_checkov_finds_wildcard_iam_policy(tmp_path) -> None:
    (tmp_path / "bad.tf").write_text(_WILDCARD_IAM_POLICY_TF)

    result = await run_checkov_sarif(str(tmp_path), changed_files=["bad.tf"])

    assert result is not None
    assert len(result) > 0
    assert all(f.rule_id.startswith("checkov/CKV") for f in result)
    assert all(f.file == "bad.tf" for f in result)


async def test_real_checkov_dash_prefixed_filename_is_scanned(tmp_path) -> None:
    # Regression test for the argv-injection fix: a changed file whose
    # repo-root-relative path starts with "-" must still be scanned, not
    # silently dropped by a Checkov argparse error. See scanner_utils.py's
    # "Dash-prefixed-filename argument injection" section.
    (tmp_path / "-x.tf").write_text(_WILDCARD_IAM_POLICY_TF)

    result = await run_checkov_sarif(str(tmp_path), changed_files=["-x.tf"])

    assert result is not None
    assert len(result) > 0
    assert all(f.file == "-x.tf" for f in result)


async def test_real_checkov_bare_dash_dash_does_not_fix_dash_prefixed_filename(
    tmp_path,
) -> None:
    # Documents WHY terraform_scanner.py uses "--file=<path>" instead of a
    # bare "--" separator (unlike squawk/eslint): Checkov's argparse
    # `-f/--file` uses nargs='+', which a bare "--" does not neutralize the
    # same way it does for clap (squawk)/yargs (eslint) parsers. This test
    # exercises the REJECTED shape directly (not through our own module,
    # which never emits it) to pin the empirical claim in terraform_scanner.py's
    # comments against a real binary, not just a mock.
    (tmp_path / "-x.tf").write_text(_WILDCARD_IAM_POLICY_TF)

    outcome = await run_scanner_subprocess(
        ["checkov", "-f", "--", "-x.tf", "--framework", "terraform", "-c", "CKV_AWS_63"],
        cwd=str(tmp_path),
        timeout=30,
    )

    # Checks rejection semantics (a genuine argparse error, distinct from
    # both "clean scan" (0) and "findings exist" (_FINDINGS_EXIT_CODE)) via
    # a content check that's more version-stable than an exact phrase/code,
    # while still verifying this is specifically an ARGUMENT-parsing
    # rejection (not e.g. a crash or an unrelated CLI error) -- those
    # details aren't guaranteed stable across the pinned
    # `checkov>=3.3.0,<4.0.0` range's patch/minor releases, only that this
    # argv shape keeps being rejected as invalid.
    assert outcome is not None
    _, stderr, returncode = outcome
    assert returncode not in (0, _FINDINGS_EXIT_CODE)
    lowered_stderr = stderr.lower()
    assert b"argument" in lowered_stderr or b"usage:" in lowered_stderr


async def test_real_checkov_scans_all_files_with_repeated_file_flag(tmp_path) -> None:
    # Regression test for the "does --file= accumulate or does each repeat
    # overwrite the last?" concern: both files must appear in the results,
    # not just the last one in the list.
    (tmp_path / "bad.tf").write_text(_WILDCARD_IAM_POLICY_TF)
    (tmp_path / "bad2.tf").write_text(_WILDCARD_IAM_POLICY_TF.replace("bad-policy", "bad-policy-2"))

    result = await run_checkov_sarif(str(tmp_path), changed_files=["bad.tf", "bad2.tf"])

    assert result is not None
    files_seen = {f.file for f in result}
    assert files_seen == {"bad.tf", "bad2.tf"}
