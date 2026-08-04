"""Unit tests for argus.precheck.shadow."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from argus.precheck.shadow import CorpusEntry, run_shadow_review
from argus.precheck.sarif import SarifResult

_PROVISIONED_WORKTREE = "argus.precheck.shadow.provisioned_worktree"
_RUN_SEMGREP_SARIF = "argus.precheck.shadow.run_semgrep_sarif"


def _mock_worktree_ctx(path: str = "/tmp/fake-worktree") -> MagicMock:
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=path)
    ctx.__aexit__ = AsyncMock(return_value=None)
    return ctx


async def test_empty_corpus_returns_empty_result() -> None:
    result = await run_shadow_review(rule_path=Path("/tmp/rule.yml"), corpus=[], github_token="t")
    assert result.hits == []
    assert result.entries_scanned == 0
    assert result.entries_matched == 0
    assert result.entries_failed == []


async def test_scans_every_entry_and_records_hits() -> None:
    corpus = [
        CorpusEntry(repo="org/a", head_sha="a" * 40),
        CorpusEntry(repo="org/b", head_sha="b" * 40),
    ]
    hit = SarifResult(rule_id="r1", level="warning", message="m", file="x.py", line=1)

    with (
        patch(_PROVISIONED_WORKTREE, return_value=_mock_worktree_ctx()),
        patch(_RUN_SEMGREP_SARIF, new=AsyncMock(side_effect=[[hit], []])),
    ):
        result = await run_shadow_review(
            rule_path=Path("/tmp/rule.yml"), corpus=corpus, github_token="t"
        )

    assert result.entries_scanned == 2
    assert result.entries_matched == 1
    assert len(result.hits) == 1
    assert result.hits[0].corpus_entry == corpus[0]
    assert result.hits[0].result == hit
    assert result.entries_failed == []


async def test_entry_with_no_hits_is_scanned_but_not_matched() -> None:
    corpus = [CorpusEntry(repo="org/a", head_sha="a" * 40)]

    with (
        patch(_PROVISIONED_WORKTREE, return_value=_mock_worktree_ctx()),
        patch(_RUN_SEMGREP_SARIF, new=AsyncMock(return_value=[])),
    ):
        result = await run_shadow_review(
            rule_path=Path("/tmp/rule.yml"), corpus=corpus, github_token="t"
        )

    assert result.entries_scanned == 1
    assert result.entries_matched == 0
    assert result.hits == []


async def test_failed_entry_is_recorded_not_fatal_to_the_run() -> None:
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
        result = await run_shadow_review(
            rule_path=Path("/tmp/rule.yml"), corpus=corpus, github_token="t"
        )

    assert result.entries_failed == [corpus[0]]
    assert result.entries_scanned == 1
    assert result.entries_matched == 1
    assert len(result.hits) == 1
    assert result.hits[0].corpus_entry == corpus[1]


async def test_run_semgrep_sarif_called_with_rule_path_directly() -> None:
    corpus = [CorpusEntry(repo="org/a", head_sha="a" * 40)]
    rule_path = Path("/tmp/candidate-rule.yml")

    with (
        patch(_PROVISIONED_WORKTREE, return_value=_mock_worktree_ctx("/tmp/wt")),
        patch(_RUN_SEMGREP_SARIF, new=AsyncMock(return_value=[])) as mock_run,
    ):
        await run_shadow_review(rule_path=rule_path, corpus=corpus, github_token="t")

    mock_run.assert_awaited_once_with("/tmp/wt", rule_path)
