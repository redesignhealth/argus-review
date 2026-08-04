"""Integration tests for argus.precheck.engine — real semgrep subprocess.

Unlike tests/test_precheck_engine.py (which mocks asyncio.create_subprocess_exec
entirely), these exercise the real subprocess/SARIF-parsing/path-stripping
path end to end: an actual `semgrep` binary scanning an actual directory.
No network access needed (semgrep runs entirely locally against a scratch
directory) -- only a real semgrep install, which the `prechecks` extra
provides. Skipped, not failed, when semgrep isn't on PATH, matching this
repo's existing integration-test convention (see
tests/test_review_patterns_integration.py's SUPABASE_DB_URL skip).
"""

from __future__ import annotations

import shutil

import pytest

from argus.precheck.engine import run_semgrep_sarif

pytestmark = pytest.mark.integration

if not shutil.which("semgrep"):
    pytest.skip(
        "semgrep not on PATH (install the `prechecks` extra) — skipping integration test",
        allow_module_level=True,
    )


_RULE_YAML = """\
rules:
  - id: integration-test-bare-except
    languages: [python]
    severity: WARNING
    message: "Integration-test rule: bare except Exception."
    pattern: |
      try:
          ...
      except Exception:
          ...
"""


def _write_target(tmp_path, content: str) -> None:
    (tmp_path / "target.py").write_text(content)


async def test_real_semgrep_finds_a_real_match(tmp_path) -> None:
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    (rules_dir / "rule.yml").write_text(_RULE_YAML)

    worktree = tmp_path / "worktree"
    worktree.mkdir()
    _write_target(
        worktree,
        "def f():\n    try:\n        risky()\n    except Exception:\n        pass\n",
    )

    result = await run_semgrep_sarif(str(worktree), rules_dir)

    assert result is not None
    assert len(result) == 1
    assert result[0].rule_id == "integration-test-bare-except"
    # Path stripping: file should be relative to the worktree, not an
    # absolute temp path.
    assert result[0].file == "target.py"


async def test_real_semgrep_no_match_returns_empty_list(tmp_path) -> None:
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    (rules_dir / "rule.yml").write_text(_RULE_YAML)

    worktree = tmp_path / "worktree"
    worktree.mkdir()
    _write_target(worktree, "def f():\n    return 1 + 1\n")

    result = await run_semgrep_sarif(str(worktree), rules_dir)

    assert result == []


async def test_real_semgrep_invalid_rule_returns_none(tmp_path) -> None:
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    (rules_dir / "bad.yml").write_text("rules:\n  - id: broken\n    this: is not valid\n")

    worktree = tmp_path / "worktree"
    worktree.mkdir()
    _write_target(worktree, "x = 1\n")

    result = await run_semgrep_sarif(str(worktree), rules_dir)

    # semgrep exits non-zero on an invalid rule config -- verified in this
    # repo's own design discussion (empirically confirmed the exit-code
    # contract this whole module depends on); None, not [], is the correct
    # "didn't run" signal.
    assert result is None
