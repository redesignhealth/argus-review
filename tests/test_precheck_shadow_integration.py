"""Integration tests for argus.precheck.shadow — real clone + real semgrep.

Unlike tests/test_precheck_shadow.py (which mocks provisioned_worktree and
run_semgrep_sarif entirely), this exercises the real corpus path end to
end: an actual git clone via repo_provision.provisioned_worktree, against
a real (tiny, stable, public) GitHub repo, scanned by a real semgrep
subprocess. Requires GITHUB_TOKEN_RO and network access — skipped, not
failed, when the token isn't set, matching this repo's existing
integration-test convention (see test_review_patterns_integration.py's
SUPABASE_DB_URL skip).

octocat/Hello-World is used as the fixture repo deliberately: it's
GitHub's own canonical, effectively-immutable demo repository (a single
README, unchanged for years), which is why the exact SHA and file content
below are safe to hardcode rather than resolved dynamically.
"""

from __future__ import annotations

import os
import shutil

import pytest

from argus.precheck.shadow import CorpusEntry, run_shadow_review

pytestmark = pytest.mark.integration

if not shutil.which("semgrep"):
    pytest.skip(
        "semgrep not on PATH (install the `prechecks` extra) — skipping integration test",
        allow_module_level=True,
    )

_HELLO_WORLD_SHA = "7fd1a60b01f91b314f59955a4e4d4e80d8edf11d"

_MATCHING_RULE_YAML = """\
rules:
  - id: integration-test-finds-hello
    languages: [generic]
    severity: WARNING
    message: "Integration-test rule: literal 'Hello' text."
    patterns:
      - pattern-regex: "Hello"
"""

_NONMATCHING_RULE_YAML = """\
rules:
  - id: integration-test-finds-nothing
    languages: [generic]
    severity: WARNING
    message: "Integration-test rule: text that does not appear in the fixture repo."
    patterns:
      - pattern-regex: "this-string-does-not-appear-anywhere-in-hello-world"
"""


@pytest.fixture
def github_token() -> str:
    token = os.environ.get("GITHUB_TOKEN_RO")
    if not token:
        pytest.skip("GITHUB_TOKEN_RO not set — skipping integration test")
    return token


async def test_real_corpus_scan_finds_a_real_hit(tmp_path, github_token: str) -> None:
    rule_path = tmp_path / "rule.yml"
    rule_path.write_text(_MATCHING_RULE_YAML)

    result = await run_shadow_review(
        rule_path=rule_path,
        corpus=[CorpusEntry(repo="octocat/Hello-World", head_sha=_HELLO_WORLD_SHA)],
        github_token=github_token,
    )

    assert result.entries_failed == []
    assert result.entries_scanned == 1
    assert result.entries_matched == 1
    assert len(result.hits) == 1
    assert result.hits[0].result.rule_id == "integration-test-finds-hello"


async def test_real_corpus_scan_with_no_hits(tmp_path, github_token: str) -> None:
    rule_path = tmp_path / "rule.yml"
    rule_path.write_text(_NONMATCHING_RULE_YAML)

    result = await run_shadow_review(
        rule_path=rule_path,
        corpus=[CorpusEntry(repo="octocat/Hello-World", head_sha=_HELLO_WORLD_SHA)],
        github_token=github_token,
    )

    assert result.entries_failed == []
    assert result.entries_scanned == 1
    assert result.entries_matched == 0
    assert result.hits == []


async def test_real_corpus_scan_bad_sha_lands_in_entries_failed(
    tmp_path, github_token: str
) -> None:
    rule_path = tmp_path / "rule.yml"
    rule_path.write_text(_MATCHING_RULE_YAML)

    bad_entry = CorpusEntry(repo="octocat/Hello-World", head_sha="f" * 40)
    result = await run_shadow_review(
        rule_path=rule_path, corpus=[bad_entry], github_token=github_token
    )

    assert result.entries_failed == [bad_entry]
    assert result.entries_scanned == 0
    assert result.hits == []
