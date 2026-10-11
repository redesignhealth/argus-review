"""Scope gate node, graph wiring, run_review persistence, storage reads, and CLI flags."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

import argus.graph as graph_mod
from argus.graph import (
    _PIPELINE_ONLY_RESPONSE_FIELDS,
    _edge_scope_decision,
    _fetch_scope_manifest,
    _node_scope_gate,
    run_review,
)
from argus.models import ReviewRequest, ReviewResponse, RiskLevel, ScopeBehavior, Verdict
from argus.scope import ScopeAssessment, ScopeUnverified

HEAD = "a" * 40
BASE = "b" * 40
FILES = [
    {"filename": "a.py", "status": "modified", "patch": "@@ x"},
    {"filename": "b.py", "status": "added", "patch": "@@ y"},
]


def _state(**over: Any) -> Any:
    state = {
        "request": ReviewRequest(repo="org/repo", pr_number=7).model_dump(),
        "head_sha": HEAD,
        "base_branch": "main",
        "description": "desc",
        "diff": "round-diff-only",
        "prior_review": {},
    }
    state.update(over)
    return state


def _assessment(*groups: list[str], indeterminate: bool = False) -> ScopeAssessment:
    return ScopeAssessment(
        indeterminate=indeterminate,
        behaviors=[ScopeBehavior(title=f"b{i}", files=g) for i, g in enumerate(groups)],
        rationale="why",
    )


def _gate(assessment: Any = None, files: Any = FILES, fetch_error: Exception | None = None) -> Any:
    fetch = MagicMock(return_value=(BASE, files), side_effect=fetch_error)
    assess = (
        AsyncMock(side_effect=assessment)
        if isinstance(assessment, Exception)
        else AsyncMock(return_value=assessment)
    )
    return (
        patch.object(graph_mod, "_fetch_scope_manifest", fetch),
        patch.object(graph_mod, "_assess_scope", assess),
        fetch,
        assess,
    )


async def _run_gate(state: Any, config: Any = None, **kw: Any) -> tuple[dict[str, Any], Any, Any]:
    p1, p2, fetch, assess = _gate(**kw)
    with p1, p2:
        out = await _node_scope_gate(state, config or {"configurable": {}})
    return out, fetch, assess


class TestScopeGateNode:
    async def test_single_behavior_sets_pinned_receipt(self) -> None:
        out, fetch, assess = await _run_gate(_state(), assessment=_assessment(["a.py", "b.py"]))
        assert "response" not in out
        assert out["scope"]["head_sha"] == HEAD
        assert out["scope"]["base_sha"] == BASE
        assert len(out["scope"]["behaviors"]) == 1
        assert out["scope"]["override_reason"] is None
        fetch.assert_called_once_with("org/repo", "main", HEAD)
        assert _edge_scope_decision(_state(**out)) == "early_verifier"

    async def test_scope_input_is_full_manifest_not_round_diff(self) -> None:
        """Round 2+ (incremental diff in state) still scopes the full pinned PR."""
        prior = {"reviewed_sha": "c" * 40, "round_number": 2}
        out, _, assess = await _run_gate(
            _state(prior_review=prior), assessment=_assessment(["a.py", "b.py"])
        )
        payload = assess.call_args.args[0]
        assert "a.py" in payload and "b.py" in payload
        assert "round-diff-only" not in payload
        assert out["scope"]["head_sha"] == HEAD

    async def test_empty_pr_skips_model_and_yields_empty_receipt(self) -> None:
        out, _, assess = await _run_gate(_state(), assessment=None, files=[])
        assess.assert_not_called()
        assert out["scope"]["behaviors"] == []

    async def test_multi_behavior_halts_scope_split(self) -> None:
        out, _, _ = await _run_gate(_state(), assessment=_assessment(["a.py"], ["b.py"]))
        resp = out["response"]
        assert resp["halt"]["code"] == "scope_split"
        assert resp["halt"]["head_sha"] == HEAD
        assert [b["title"] for b in resp["halt"]["split_suggestions"]] == ["b0", "b1"]
        assert resp["verdict"] == "BLOCKING" and resp["risk_level"] == "HIGH"
        assert resp["findings"] == []
        assert "scope" not in out
        assert _edge_scope_decision(_state(**out)) == "__end__"

    async def test_split_halt_preserves_round_numbering(self) -> None:
        prior = {
            "review_id": "rev-1",
            "round_number": 3,
            "reviewed_sha": "c" * 40,
            "prior_verdict": "BLOCKING",
        }
        out, _, _ = await _run_gate(
            _state(prior_review=prior), assessment=_assessment(["a.py"], ["b.py"])
        )
        assert out["response"]["review_round"] == 3
        assert out["response"]["prior_review_id"] == "rev-1"

    async def test_override_allows_split_and_records_reason(self) -> None:
        config = {
            "configurable": {"review_override_sha": HEAD, "review_override_reason": "human ok"}
        }
        out, _, _ = await _run_gate(_state(), config, assessment=_assessment(["a.py"], ["b.py"]))
        assert "response" not in out
        assert len(out["scope"]["behaviors"]) == 2
        assert out["scope"]["override_reason"] == "human ok"

    async def test_override_pinned_to_stale_sha_does_not_apply(self) -> None:
        config = {
            "configurable": {"review_override_sha": "d" * 40, "review_override_reason": "stale"}
        }
        out, _, _ = await _run_gate(_state(), config, assessment=_assessment(["a.py"], ["b.py"]))
        assert out["response"]["halt"]["code"] == "scope_split"

    @pytest.mark.parametrize(
        "assessment",
        [
            _assessment(["a.py"]),  # b.py unassigned
            _assessment(["a.py", "b.py", "ghost.py"]),
            _assessment(["a.py", "b.py"], ["b.py"]),
            _assessment(indeterminate=True),
            ScopeAssessment(indeterminate=False, behaviors=[], rationale="r"),
        ],
    )
    async def test_invalid_manifest_halts_unverified_even_with_override(
        self, assessment: ScopeAssessment
    ) -> None:
        config = {"configurable": {"review_override_sha": HEAD, "review_override_reason": "x"}}
        out, _, _ = await _run_gate(_state(), config, assessment=assessment)
        assert out["response"]["halt"]["code"] == "scope_unverified"
        assert "scope" not in out

    async def test_scope_unverified_from_fetch_halts(self) -> None:
        out, _, assess = await _run_gate(
            _state(), assessment=None, fetch_error=ScopeUnverified("300-file cap")
        )
        assert out["response"]["halt"]["code"] == "scope_unverified"
        assert "300-file cap" in out["response"]["halt"]["detail"]
        assess.assert_not_called()

    async def test_unexpected_error_halts_without_leaking_message(self) -> None:
        out, _, _ = await _run_gate(_state(), assessment=RuntimeError("secret-token-xyz"))
        halt = out["response"]["halt"]
        assert halt["code"] == "scope_unverified"
        assert "RuntimeError" in halt["detail"]
        assert "secret-token-xyz" not in halt["detail"]


class TestFetchScopeManifest:
    def _gh(self, merge_base: str = BASE, truncated: bool = False) -> MagicMock:
        gh = MagicMock()
        gh.get_compare_metadata.return_value = {"merge_base_sha": merge_base, "status": "ahead"}
        gh.get_compare_files.return_value = (FILES, truncated)
        return gh

    def test_pins_merge_base_then_lists_files_from_it(self) -> None:
        gh = self._gh()
        with patch("argus.github_client.GitHubClient", return_value=gh):
            base, files = _fetch_scope_manifest("org/repo", "main", HEAD)
        assert (base, files) == (BASE, FILES)
        gh.get_compare_metadata.assert_called_once_with("org/repo", "main", HEAD)
        gh.get_compare_files.assert_called_once_with("org/repo", BASE, HEAD)
        gh.get_pr_files.assert_not_called()

    @pytest.mark.parametrize("bad", ["", "abc", "B" * 40, "g" * 40, "a" * 39])
    def test_invalid_merge_base_is_unverified(self, bad: str) -> None:
        with (
            patch("argus.github_client.GitHubClient", return_value=self._gh(bad)),
            pytest.raises(ScopeUnverified),
        ):
            _fetch_scope_manifest("org/repo", "main", HEAD)

    def test_300_file_truncation_halts_without_pr_files_fallback(self) -> None:
        gh = self._gh(truncated=True)
        with (
            patch("argus.github_client.GitHubClient", return_value=gh),
            pytest.raises(ScopeUnverified, match="300"),
        ):
            _fetch_scope_manifest("org/repo", "main", HEAD)
        gh.get_pr_files.assert_not_called()


class TestGraphWiring:
    async def test_halt_at_gate_stops_before_verifier_preflight_and_lite(self) -> None:
        """Real compiled graph: a split halt ends the run; nothing downstream executes."""
        boom = AsyncMock(side_effect=AssertionError("must not run after a halt"))
        fetched = _state()
        p1, p2, _, _ = _gate(assessment=_assessment(["a.py"], ["b.py"]))
        with (
            p1,
            p2,
            patch.object(graph_mod, "_node_fetch_diff", AsyncMock(return_value=fetched)),
            patch.object(graph_mod, "_node_precheck_checks", AsyncMock(return_value={})),
            patch.object(graph_mod, "_node_precheck_rules", AsyncMock(return_value={})),
            patch.object(graph_mod, "_node_early_verifier", boom),
            patch.object(graph_mod, "_node_preflight", boom),
            patch.object(graph_mod, "_node_lite_review", boom),
            patch.object(graph_mod, "_node_plan", boom),
        ):
            compiled = graph_mod._build_review_graph().compile()
            result = await compiled.ainvoke({"request": fetched["request"]})
        assert result["response"]["halt"]["code"] == "scope_split"
        boom.assert_not_called()

    async def _run_plan_graph(self, n_groups: int, override: bool) -> tuple[dict[str, Any], int]:
        """Real compiled graph through scope_gate -> plan -> fan-out, stubbing only I/O nodes."""
        from argus.pipeline_models import FileEntry, ReviewPlan, SystemGroup

        plan = ReviewPlan(
            system_groups=[
                SystemGroup(name=f"g{i}", files=[f"f{i}"], conventions="", review_focus="x")
                for i in range(n_groups)
            ],
            cross_cutting_concerns=[],
            file_manifest=[FileEntry(path="f0", change_type="modified")],
        )
        reviewer = AsyncMock(return_value={"findings": []})
        approve = _approve()
        covered = {"coverage": {"is_covered": True, "gaps": []}}
        p1, p2, _, _ = _gate(assessment=_assessment(["a.py", "b.py"]))
        with (
            p1,
            p2,
            patch.object(graph_mod, "_node_fetch_diff", AsyncMock(return_value=_state())),
            patch.object(graph_mod, "_node_precheck_checks", AsyncMock(return_value={})),
            patch.object(graph_mod, "_node_precheck_rules", AsyncMock(return_value={})),
            patch.object(graph_mod, "_node_early_verifier", AsyncMock(return_value={})),
            patch.object(
                graph_mod,
                "_node_preflight",
                AsyncMock(return_value={"is_lite": False, "preflight_result": {}}),
            ),
            patch.object(graph_mod, "plan_review", AsyncMock(return_value=plan)),
            patch.object(graph_mod, "_node_run_reviewer", reviewer),
            patch.object(graph_mod, "_node_collect_findings", AsyncMock(return_value={})),
            patch.object(graph_mod, "_node_check_coverage", AsyncMock(return_value=covered)),
            patch.object(
                graph_mod, "_node_write_review", AsyncMock(return_value={"response": approve})
            ),
            patch.object(graph_mod, "_node_validate_blockings", AsyncMock(return_value={})),
        ):
            compiled = graph_mod._build_review_graph().compile()
            config: Any = {
                "configurable": {
                    "review_override_sha": HEAD if override else None,
                    "review_override_reason": "human ok" if override else None,
                },
                "max_concurrency": 16,
            }
            result = await compiled.ainvoke({"request": _state()["request"]}, config=config)
        return result, reviewer.await_count

    async def test_real_graph_over_budget_halts_with_zero_reviewers(self) -> None:
        result, reviewers = await self._run_plan_graph(15, override=False)  # 17 > 16
        assert result["response"]["halt"]["code"] == "dispatch_budget_exceeded"
        assert reviewers == 0

    async def test_real_graph_override_reaches_plan_node_and_dispatches_all(self) -> None:
        result, reviewers = await self._run_plan_graph(20, override=True)  # 22 > 16, <= 50
        assert reviewers == 22
        assert "halt" not in result["response"] or result["response"]["halt"] is None
        assert result["scope"]["override_reason"] == "human ok"

    async def test_verified_precheck_failure_still_short_circuits_before_gate(self) -> None:
        p1, p2, fetch, assess = _gate(assessment=_assessment(["a.py", "b.py"]))
        fast_fail = {"precheck_fast_fail": [{"rule_id": "r", "message": "m", "file": "a.py"}]}
        with (
            p1,
            p2,
            patch.object(graph_mod, "_node_fetch_diff", AsyncMock(return_value=_state())),
            patch.object(graph_mod, "_node_precheck_checks", AsyncMock(return_value={})),
            patch.object(graph_mod, "_node_precheck_rules", AsyncMock(return_value=fast_fail)),
        ):
            compiled = graph_mod._build_review_graph().compile()
            result = await compiled.ainvoke({"request": _state()["request"]})
        assert result["response"]["findings"][0]["category"] == "deterministic-precheck"
        fetch.assert_not_called()
        assess.assert_not_called()

    def test_scope_and_halt_are_never_llm_generated(self) -> None:
        assert {"scope", "halt"} <= _PIPELINE_ONLY_RESPONSE_FIELDS
        from argus.llm.output_models import pydantic_to_response_format

        fmt = pydantic_to_response_format(
            ReviewResponse, "review_response", exclude=_PIPELINE_ONLY_RESPONSE_FIELDS
        )
        props = fmt["schema"]["properties"]
        assert "scope" not in props and "halt" not in props

    def test_scope_prompt_is_packaged_and_required(self) -> None:
        from argus.prompts_runtime import REQUIRED_PROMPTS, known_packaged_prompts

        assert "pr-review-scope" in REQUIRED_PROMPTS
        assert "pr-review-scope" in known_packaged_prompts()
        text = (Path(graph_mod.__file__).parent / "prompts" / "pr-review-scope.md").read_text()
        assert "data" in text and "indeterminate" in text


# ---------------------------------------------------------------------------
# run_review: persistence of halts, receipt requirement, override pin
# ---------------------------------------------------------------------------


def _scope_dict(head: str = HEAD, n: int = 1, override: str | None = None) -> dict[str, Any]:
    return {
        "base_sha": BASE,
        "head_sha": head,
        "behaviors": [{"title": f"t{i}", "files": [f"f{i}"], "depends_on": []} for i in range(n)],
        "rationale": "r",
        "override_reason": override,
    }


async def _run(
    graph_result: dict[str, Any], **kwargs: Any
) -> tuple[ReviewResponse, MagicMock, MagicMock]:
    backend = MagicMock()
    backend.upsert_completed_row = AsyncMock(return_value=SimpleNamespace(id=None))
    backend.insert_agent_runs = AsyncMock()
    graph = AsyncMock()
    graph.ainvoke.return_value = graph_result
    wt = MagicMock()
    wt.__aenter__ = AsyncMock(return_value="/tmp/wt")
    wt.__aexit__ = AsyncMock(return_value=None)
    gh = MagicMock()
    gh.get_pull_request.return_value = {"head_sha": HEAD}
    with (
        patch.object(graph_mod, "build_pipeline") as build,
        patch.object(graph_mod, "validate_history_backend_connectivity", AsyncMock()),
        patch.object(graph_mod, "get_history_backend", return_value=backend),
        patch.object(graph_mod, "resolve_history_backend_kind", return_value="sqlite"),
        patch("argus.github_client.GitHubClient", return_value=gh),
        patch.object(graph_mod, "provisioned_worktree", return_value=wt),
    ):
        build.return_value.__aenter__ = AsyncMock(return_value=graph)
        build.return_value.__aexit__ = AsyncMock(return_value=None)
        resp = await run_review(ReviewRequest(repo="org/repo", pr_number=7), **kwargs)
    return resp, backend, graph


def _approve(**over: Any) -> dict[str, Any]:
    return ReviewResponse(
        verdict=Verdict.APPROVE, risk_level=RiskLevel.LOW, review_comment="ok", **over
    ).model_dump()


class TestRunReviewHaltAndReceipt:
    async def test_halt_persists_null_verdict_completed_and_skips_gates(self) -> None:
        from argus.scope import halt_response

        halted = halt_response("scope_split", "two behaviors", HEAD).model_dump()
        resp, backend, _ = await _run(
            {
                "response": halted,
                "head_sha": HEAD,
                "findings": [],
                "bench_config_changes": [".argus/bench.toml"],
                "precheck_scanner_failures": ["semgrep"],
            }
        )
        assert resp.halt is not None and resp.halt.code == "scope_split"
        assert resp.verdict is Verdict.BLOCKING and resp.risk_level is RiskLevel.HIGH
        assert resp.findings == []  # no dummy findings from the post-graph gates
        row = backend.upsert_completed_row.call_args.kwargs["row"]
        assert row.verdict is None
        assert row.current_stage == "completed"
        assert row.blocking_count == 0
        assert row.result_json["halt"]["code"] == "scope_split"
        assert row.sha == HEAD

    async def test_approve_without_receipt_is_refused(self) -> None:
        with pytest.raises(RuntimeError, match="scope receipt"):
            await _run({"response": _approve(), "head_sha": HEAD, "findings": []})

    async def test_approve_with_receipt_for_other_head_is_refused(self) -> None:
        with pytest.raises(RuntimeError, match="scope receipt"):
            await _run(
                {
                    "response": _approve(),
                    "head_sha": HEAD,
                    "findings": [],
                    "scope": _scope_dict(head="c" * 40),
                }
            )

    async def test_approve_with_multi_behavior_receipt_needs_override(self) -> None:
        with pytest.raises(RuntimeError, match="override"):
            await _run(
                {
                    "response": _approve(),
                    "head_sha": HEAD,
                    "findings": [],
                    "scope": _scope_dict(n=2),
                }
            )

    async def test_approve_with_valid_receipt_attaches_it_and_persists_verdict(self) -> None:
        resp, backend, _ = await _run(
            {"response": _approve(), "head_sha": HEAD, "findings": [], "scope": _scope_dict()}
        )
        assert resp.scope is not None and resp.scope.head_sha == HEAD
        row = backend.upsert_completed_row.call_args.kwargs["row"]
        assert row.verdict == "APPROVE"
        assert row.result_json["scope"]["head_sha"] == HEAD

    async def test_override_reaches_graph_config_and_multi_behavior_approves(self) -> None:
        resp, _, graph = await _run(
            {
                "response": _approve(),
                "head_sha": HEAD,
                "findings": [],
                "scope": _scope_dict(n=2, override="human ok"),
            },
            review_override_sha=HEAD,
            review_override_reason=" human ok ",
        )
        conf = graph.ainvoke.call_args.kwargs["config"]["configurable"]
        assert conf["review_override_sha"] == HEAD
        assert conf["review_override_reason"] == "human ok"
        assert resp.scope is not None and resp.scope.override_reason == "human ok"

    async def test_stale_override_sha_fails_before_any_graph_run(self) -> None:
        with pytest.raises(ValueError, match="stale"):
            await _run({}, review_override_sha="d" * 40, review_override_reason="r")

    async def test_override_pair_is_required(self) -> None:
        with pytest.raises(ValueError, match="together"):
            await _run({}, review_override_sha=HEAD)

    def test_override_is_not_on_the_serialized_request(self) -> None:
        assert not any("override" in f for f in ReviewRequest.model_fields)


# ---------------------------------------------------------------------------
# Storage: a NULL-verdict completed round never counts as the latest round
# ---------------------------------------------------------------------------


async def test_sqlite_null_verdict_round_after_completed_keeps_prior_sha_and_count(
    tmp_path: Path,
) -> None:
    from argus.storage.sql import CodeReviewRoundIn
    from argus.storage.sqlite import SqliteHistoryBackend

    backend = SqliteHistoryBackend(db_path=tmp_path / "h.db")
    try:
        await backend.upsert_completed_row(
            row=CodeReviewRoundIn(
                repo="org/repo",
                pr_number=7,
                verdict="BLOCKING",
                sha="1" * 40,
                current_stage="completed",
            )
        )
        await backend.upsert_completed_row(
            row=CodeReviewRoundIn(
                repo="org/repo",
                pr_number=7,
                verdict=None,
                sha="2" * 40,
                current_stage="completed",
                result_json={"halt": {"code": "scope_split"}},
            )
        )
        latest = await backend.select_latest_completed_round(repo="org/repo", pr_number=7)
    finally:
        await backend.aclose()
    assert latest is not None
    assert latest.sha == "1" * 40
    assert latest.prior_count == 1


def test_postgres_reads_already_filter_null_verdict_without_schema_change() -> None:
    from argus.storage import sql

    for query in (sql._SELECT_LATEST_COMPLETED_ROUND_SQL, sql._SELECT_RECENT_ROUNDS_SQL):
        assert "verdict IS NOT NULL" in query
    assert "verdict IS NOT NULL) AS prior_count" in sql._SELECT_LATEST_COMPLETED_ROUND_SQL


async def test_http_wire_null_verdict_round_is_skipped_for_latest_and_count(
    httpx_mock: Any,
) -> None:
    from datetime import UTC, datetime
    from uuid import uuid4

    from argus.storage import HttpStorageClient

    def row(verdict: str | None, sha: str) -> dict[str, Any]:
        return {
            "id": str(uuid4()),
            "created_at": datetime(2026, 5, 19, tzinfo=UTC).isoformat(),
            "repo": "acme/r",
            "pr_number": 5,
            "verdict": verdict,
            "sha": sha,
            "current_stage": "completed",
            "result_json": {"halt": {"code": "scope_split"}} if verdict is None else {},
        }

    url = "https://fake.test/{owner}/{repo}/{pr}"
    httpx_mock.add_response(
        method="GET",
        url="https://fake.test/acme/r/5",
        json={"rounds": [row(None, "2" * 40), row("BLOCKING", "1" * 40), row("APPROVE", "0" * 40)]},
    )
    client = HttpStorageClient(read_url=url, write_url=url + "/rounds")
    try:
        latest, count = await client.read_latest_completed_round(owner="acme", repo="r", pr=5)
    finally:
        await client.aclose()
    assert latest is not None and latest.sha == "1" * 40
    assert count == 2


async def test_http_all_null_rounds_behave_like_round_one(httpx_mock: Any) -> None:
    from datetime import UTC, datetime
    from uuid import uuid4

    from argus.storage import HttpStorageClient

    httpx_mock.add_response(
        method="GET",
        url="https://fake.test/acme/r/5",
        json={
            "rounds": [
                {
                    "id": str(uuid4()),
                    "created_at": datetime(2026, 5, 19, tzinfo=UTC).isoformat(),
                    "repo": "acme/r",
                    "pr_number": 5,
                    "verdict": None,
                }
            ]
        },
    )
    url = "https://fake.test/{owner}/{repo}/{pr}"
    client = HttpStorageClient(read_url=url, write_url=url + "/rounds")
    try:
        assert await client.read_latest_completed_round(owner="acme", repo="r", pr=5) == (None, 0)
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# Config + CLI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("raw", "ok"), [("3", True), ("50", True), ("2", False), ("51", False)])
def test_dispatch_budget_bounds(monkeypatch: pytest.MonkeyPatch, raw: str, ok: bool) -> None:
    from argus.config import Settings

    monkeypatch.setenv("ARGUS_REVIEWER_DISPATCH_BUDGET", raw)
    if ok:
        assert Settings().ARGUS_REVIEWER_DISPATCH_BUDGET == int(raw)
    else:
        with pytest.raises(ValidationError):
            Settings()


def test_dispatch_budget_default_is_16(monkeypatch: pytest.MonkeyPatch) -> None:
    from argus.config import Settings

    monkeypatch.delenv("ARGUS_REVIEWER_DISPATCH_BUDGET", raising=False)
    assert Settings().ARGUS_REVIEWER_DISPATCH_BUDGET == 16


def _parse(*argv: str) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    from argus.cli import _build_parser

    parser = _build_parser()
    return parser, parser.parse_args(["review", "org/repo", "--pr", "1", *argv])


class TestCliOverrideFlags:
    def test_valid_pair_parses_and_validates(self) -> None:
        from argus.cli import _validate_review_args

        parser, args = _parse("--review-override-sha", HEAD, "--review-override-reason", "ok")
        _validate_review_args(parser, args)
        assert (args.review_override_sha, args.review_override_reason) == (HEAD, "ok")

    @pytest.mark.parametrize(
        "argv",
        [
            ["--review-override-sha", HEAD],
            ["--review-override-reason", "ok"],
            ["--review-override-sha", "A" * 40, "--review-override-reason", "ok"],
            ["--review-override-sha", "a" * 7, "--review-override-reason", "ok"],
            ["--review-override-sha", HEAD, "--review-override-reason", "  "],
            ["--review-override-sha", HEAD, "--review-override-reason", "x" * 501],
        ],
    )
    def test_invalid_pairs_exit(self, argv: list[str]) -> None:
        from argus.cli import _validate_review_args

        parser, args = _parse(*argv)
        with pytest.raises(SystemExit):
            _validate_review_args(parser, args)

    def test_no_flags_means_no_override(self) -> None:
        _, args = _parse()
        assert args.review_override_sha is None and args.review_override_reason is None


class TestCliCommitStatusOnHalt:
    def _args(self, **over: Any) -> argparse.Namespace:
        base = {"post": False, "commit_status": True, "pr": 9, "sha": None}
        base.update(over)
        return argparse.Namespace(**base)

    def test_halt_status_targets_halt_head_not_live_head(self) -> None:
        from argus.cli import _post_review
        from argus.scope import halt_response

        client = MagicMock()
        client.get_pull_request.return_value = {"head_sha": "f" * 40}
        with patch("argus.github_client.GitHubClient.for_writes", return_value=client):
            _post_review("org/repo", self._args(), halt_response("scope_split", "d", HEAD))
        client.get_pull_request.assert_not_called()
        sha, state, desc = client.set_commit_status.call_args.args[1:4]
        assert (sha, state) == (HEAD, "failure")
        assert "scope_split" in desc

    def test_non_halt_status_unchanged(self) -> None:
        from argus.cli import _post_review

        client = MagicMock()
        client.get_pull_request.return_value = {"head_sha": "f" * 40}
        resp = ReviewResponse(verdict=Verdict.APPROVE, risk_level=RiskLevel.LOW, review_comment="x")
        with patch("argus.github_client.GitHubClient.for_writes", return_value=client):
            _post_review("org/repo", self._args(), resp)
        assert client.set_commit_status.call_args.args[1:3] == ("f" * 40, "success")
