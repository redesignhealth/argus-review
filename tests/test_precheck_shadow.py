"""Unit tests for argus.precheck.shadow."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from argus.precheck.shadow import CorpusEntry, run_shadow_review
from argus.precheck.sarif import SarifResult

_PROVISIONED_WORKTREE = "argus.precheck.shadow.provisioned_worktree"
_RUN_SEMGREP_SARIF = "argus.precheck.shadow.run_semgrep_sarif"
_SEMGREP_AVAILABLE = "argus.precheck.shadow.semgrep_available"


def _mock_worktree_ctx(path: str = "/tmp/fake-worktree") -> MagicMock:
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=path)
    ctx.__aexit__ = AsyncMock(return_value=None)
    return ctx


@pytest.fixture(autouse=True)
def _semgrep_available_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # Every test in this file exercises the corpus loop, not the upfront
    # semgrep-availability guard -- default it to available so each test
    # only needs to override when specifically testing that guard.
    monkeypatch.setattr(_SEMGREP_AVAILABLE, lambda: True)


async def test_semgrep_unavailable_raises_before_touching_the_corpus(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_SEMGREP_AVAILABLE, lambda: False)
    rule_path = tmp_path / "rule.yml"
    rule_path.write_text("rules: []\n")

    with patch(_PROVISIONED_WORKTREE) as mock_provision:
        with pytest.raises(RuntimeError, match="semgrep"):
            await run_shadow_review(
                rule_path=rule_path,
                corpus=[CorpusEntry(repo="org/a", head_sha="a" * 40)],
                github_token="t",
            )
    mock_provision.assert_not_called()


async def test_missing_rule_path_raises_before_touching_the_corpus(tmp_path) -> None:
    missing = tmp_path / "does-not-exist.yml"

    with patch(_PROVISIONED_WORKTREE) as mock_provision:
        with pytest.raises(FileNotFoundError):
            await run_shadow_review(
                rule_path=missing,
                corpus=[CorpusEntry(repo="org/a", head_sha="a" * 40)],
                github_token="t",
            )
    mock_provision.assert_not_called()


async def test_empty_corpus_returns_empty_result(tmp_path) -> None:
    rule_path = tmp_path / "rule.yml"
    rule_path.write_text("rules: []\n")

    result = await run_shadow_review(rule_path=rule_path, corpus=[], github_token="t")
    assert result.hits == []
    assert result.entries_scanned == 0
    assert result.entries_matched == 0
    assert result.entries_failed == []


async def test_scans_every_entry_and_records_hits(tmp_path) -> None:
    rule_path = tmp_path / "rule.yml"
    rule_path.write_text("rules: []\n")
    corpus = [
        CorpusEntry(repo="org/a", head_sha="a" * 40),
        CorpusEntry(repo="org/b", head_sha="b" * 40),
    ]
    hit = SarifResult(rule_id="r1", level="warning", message="m", file="x.py", line=1)

    with (
        patch(_PROVISIONED_WORKTREE, return_value=_mock_worktree_ctx()),
        patch(_RUN_SEMGREP_SARIF, new=AsyncMock(side_effect=[[hit], []])),
    ):
        result = await run_shadow_review(rule_path=rule_path, corpus=corpus, github_token="t")

    assert result.entries_scanned == 2
    assert result.entries_matched == 1
    assert len(result.hits) == 1
    assert result.hits[0].corpus_entry == corpus[0]
    assert result.hits[0].result == hit
    assert result.entries_failed == []


async def test_entry_with_no_hits_is_scanned_but_not_matched(tmp_path) -> None:
    rule_path = tmp_path / "rule.yml"
    rule_path.write_text("rules: []\n")
    corpus = [CorpusEntry(repo="org/a", head_sha="a" * 40)]

    with (
        patch(_PROVISIONED_WORKTREE, return_value=_mock_worktree_ctx()),
        patch(_RUN_SEMGREP_SARIF, new=AsyncMock(return_value=[])),
    ):
        result = await run_shadow_review(rule_path=rule_path, corpus=corpus, github_token="t")

    assert result.entries_scanned == 1
    assert result.entries_matched == 0
    assert result.hits == []
    assert result.entries_failed == []


async def test_semgrep_did_not_run_on_entry_is_failed_not_scanned(tmp_path) -> None:
    # Regression test: run_semgrep_sarif returning None ("didn't run" --
    # timeout, missing binary, execution error) must land in
    # entries_failed, NOT be treated as "ran, found nothing." Collapsing
    # these would let a malformed candidate rule that never executes on
    # any corpus entry produce a confident-looking zero-occurrence result.
    rule_path = tmp_path / "rule.yml"
    rule_path.write_text("rules: []\n")
    corpus = [CorpusEntry(repo="org/a", head_sha="a" * 40)]

    with (
        patch(_PROVISIONED_WORKTREE, return_value=_mock_worktree_ctx()),
        patch(_RUN_SEMGREP_SARIF, new=AsyncMock(return_value=None)),
    ):
        result = await run_shadow_review(rule_path=rule_path, corpus=corpus, github_token="t")

    assert result.entries_scanned == 0
    assert result.entries_matched == 0
    assert result.hits == []
    assert result.entries_failed == corpus


async def test_failed_entry_is_recorded_not_fatal_to_the_run(tmp_path) -> None:
    rule_path = tmp_path / "rule.yml"
    rule_path.write_text("rules: []\n")
    corpus = [
        CorpusEntry(repo="org/bad", head_sha="a" * 40),
        CorpusEntry(repo="org/good", head_sha="b" * 40),
    ]
    hit = SarifResult(rule_id="r1", level="warning", message="m", file="x.py", line=1)

    good_ctx = _mock_worktree_ctx()
    bad_ctx = MagicMock()
    bad_ctx.__aenter__ = AsyncMock(side_effect=RuntimeError("clone failed"))
    bad_ctx.__aexit__ = AsyncMock(return_value=None)

    with (
        patch(_PROVISIONED_WORKTREE, side_effect=[bad_ctx, good_ctx]),
        patch(_RUN_SEMGREP_SARIF, new=AsyncMock(return_value=[hit])),
    ):
        result = await run_shadow_review(rule_path=rule_path, corpus=corpus, github_token="t")

    assert result.entries_failed == [corpus[0]]
    assert result.entries_scanned == 1
    assert result.entries_matched == 1
    assert len(result.hits) == 1
    assert result.hits[0].corpus_entry == corpus[1]


async def test_run_semgrep_sarif_called_with_rule_path_directly(tmp_path) -> None:
    rule_path = tmp_path / "candidate-rule.yml"
    rule_path.write_text("rules: []\n")
    corpus = [CorpusEntry(repo="org/a", head_sha="a" * 40)]

    with (
        patch(_PROVISIONED_WORKTREE, return_value=_mock_worktree_ctx("/tmp/wt")),
        patch(_RUN_SEMGREP_SARIF, new=AsyncMock(return_value=[])) as mock_run,
    ):
        await run_shadow_review(rule_path=rule_path, corpus=corpus, github_token="t")

    mock_run.assert_awaited_once_with("/tmp/wt", rule_path)
