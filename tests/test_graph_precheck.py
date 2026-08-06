"""Unit tests for the deterministic-precheck graph node/edge wiring.

Covers:
  - _edge_precheck_decision routes to precheck_fail iff a verified hit exists
  - _node_precheck_fail synthesizes a BLOCKING response with zero LLM calls
  - _node_precheck_checks: checks-signal read, retry-then-unknown on
    repeated failure
  - _node_precheck_rules: worktree-absent short-circuit, candidate findings
    attached to state, verified findings trigger precheck_fast_fail, and a
    run_precheck exception is swallowed
  - _node_write_review attaches precheck_findings as extra writer context
    without polluting state["findings"] (collect_findings' accounting)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from argus.graph import (
    _edge_precheck_decision,
    _node_precheck_checks,
    _node_precheck_fail,
    _node_precheck_rules,
    _node_write_review,
)
from argus.models import ReviewResponse, RiskLevel, Verdict

_GH_CLIENT_CLASS = "argus.github_client.GitHubClient"


def _make_state(*, precheck_fast_fail: list | None = None, **overrides: object) -> dict:
    state = {
        "request": {"repo": "org/repo", "pr_number": 42},
        "diff": "diff --git a/a.py b/a.py\n+x = 1",
        "description": "fix: add x",
        "head_sha": "a" * 40,
        "prior_review": {},
    }
    if precheck_fast_fail is not None:
        state["precheck_fast_fail"] = precheck_fast_fail
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# _edge_precheck_decision
# ---------------------------------------------------------------------------


def test_edge_routes_to_early_verifier_when_no_fast_fail() -> None:
    assert _edge_precheck_decision(_make_state()) == "early_verifier"


def test_edge_routes_to_precheck_fail_when_verified_hit_present() -> None:
    state = _make_state(precheck_fast_fail=[{"rule_id": "r1", "message": "m"}])
    assert _edge_precheck_decision(state) == "precheck_fail"


def test_edge_ignores_empty_fast_fail_list() -> None:
    state = _make_state(precheck_fast_fail=[])
    assert _edge_precheck_decision(state) == "early_verifier"


# ---------------------------------------------------------------------------
# _node_precheck_fail
# ---------------------------------------------------------------------------


async def test_precheck_fail_synthesizes_blocking_response() -> None:
    state = _make_state(
        precheck_fast_fail=[
            {"rule_id": "no-hardcoded-secret", "message": "secret found", "file": "a.py", "line": 3}
        ]
    )
    result = await _node_precheck_fail(state)
    response = result["response"]
    assert response["verdict"] == Verdict.BLOCKING.value
    assert response["findings"][0]["severity"] == "BLOCKING"
    assert response["findings"][0]["category"] == "deterministic-precheck"
    assert "no-hardcoded-secret" in response["review_comment"]
    assert "fast-fail" in response["preflight_reason"]


async def test_precheck_fail_preserves_round_number_from_prior_review() -> None:
    state = _make_state(
        precheck_fast_fail=[{"rule_id": "r1", "message": "m", "file": None, "line": None}],
        prior_review={
            "review_id": "rev-1",
            "reviewed_sha": "b" * 40,
            "findings": [],
            "notes_for_next_round": None,
            "round_number": 3,
            "prior_verdict": "BLOCKING",
            "dismissed_findings": [],
        },
    )
    result = await _node_precheck_fail(state)
    assert result["response"]["review_round"] == 3
    assert result["response"]["prior_review_id"] == "rev-1"


async def test_precheck_fail_caps_displayed_hits_without_dropping_the_gate() -> None:
    # Regression test: an earlier version had no bound at all on how many
    # verified hits get rendered into the comment body, risking exceeding
    # GitHub's per-comment size limit for a broad rule matching many
    # locations -- the exact scenario this gate exists to catch. The cap
    # must be display-only: the verdict/findings-list length is capped,
    # but nothing here should ever make the gate decision itself go away.
    from argus.graph import _MAX_DISPLAYED_FAST_FAIL_HITS

    many_hits = [
        {"rule_id": "r1", "message": f"hit {i}", "file": f"f{i}.py", "line": i}
        for i in range(_MAX_DISPLAYED_FAST_FAIL_HITS + 10)
    ]
    state = _make_state(precheck_fast_fail=many_hits)

    result = await _node_precheck_fail(state)
    response = result["response"]

    assert response["verdict"] == Verdict.BLOCKING.value
    assert len(response["findings"]) == _MAX_DISPLAYED_FAST_FAIL_HITS
    assert "10 more hit(s)" in response["review_comment"]


# ---------------------------------------------------------------------------
# _node_precheck_checks
# ---------------------------------------------------------------------------


def _mock_gh_client(signal: str = "passing") -> AsyncMock:
    client = AsyncMock()
    client.get_checks_signal = lambda *a, **k: signal  # sync method on the real client
    return client


async def test_precheck_checks_reads_signal() -> None:
    with patch(_GH_CLIENT_CLASS, return_value=_mock_gh_client("failing")):
        result = await _node_precheck_checks(_make_state())
    assert result == {"checks_signal": "failing"}


async def test_precheck_checks_retries_once_before_falling_back_to_unknown() -> None:
    calls = {"n": 0}

    def _flaky(*_args: object, **_kwargs: object) -> str:
        calls["n"] += 1
        raise RuntimeError("boom")

    client = AsyncMock()
    client.get_checks_signal = _flaky
    with patch(_GH_CLIENT_CLASS, return_value=client):
        result = await _node_precheck_checks(_make_state())
    assert result == {"checks_signal": "unknown"}
    assert calls["n"] == 2  # _CHECKS_SIGNAL_ATTEMPTS


async def test_precheck_checks_succeeds_on_second_attempt() -> None:
    calls = {"n": 0}

    def _flaky_then_ok(*_args: object, **_kwargs: object) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return "passing"

    client = AsyncMock()
    client.get_checks_signal = _flaky_then_ok
    with patch(_GH_CLIENT_CLASS, return_value=client):
        result = await _node_precheck_checks(_make_state())
    assert result == {"checks_signal": "passing"}


async def test_precheck_checks_client_construction_failure_falls_back_to_unknown() -> None:
    # Regression test: GitHubClient(...) raises ValueError on a missing/
    # invalid token *before* any network call, i.e. before get_checks_signal
    # is ever reached. This must still fall back to "unknown", not crash the
    # node -- round 2 of this PR's own review caught exactly this bug
    # (construction had been moved outside the retry/fail-open boundary).
    with patch(_GH_CLIENT_CLASS, side_effect=ValueError("no token")):
        result = await _node_precheck_checks(_make_state())
    assert result == {"checks_signal": "unknown"}


# ---------------------------------------------------------------------------
# _node_precheck_rules
# ---------------------------------------------------------------------------


async def test_precheck_rules_no_worktree_is_noop() -> None:
    result = await _node_precheck_rules(_make_state(), {"configurable": {}})
    assert result == {}


async def test_precheck_rules_candidate_findings_attached_and_logged() -> None:
    from argus.precheck.engine import PrecheckResult
    from argus.precheck.sarif import SarifResult

    candidate_hit = SarifResult(rule_id="r1", level="warning", message="m", file="a.py", line=1)

    with (
        patch(
            "argus.precheck.engine.run_precheck",
            new=AsyncMock(return_value=PrecheckResult(candidate_findings=[candidate_hit])),
        ),
        patch("argus.storage.precheck.log_candidate_firings", new=AsyncMock()) as mock_log,
    ):
        result = await _node_precheck_rules(
            _make_state(), {"configurable": {"worktree_path": "/tmp/wt"}}
        )

    assert result["precheck_findings"] == [candidate_hit.as_finding_dict()]
    assert "precheck_fast_fail" not in result
    mock_log.assert_awaited_once()
    # Batched: one call carrying the whole list, not one call per finding.
    assert mock_log.await_args is not None
    assert len(mock_log.await_args.kwargs["firings"]) == 1


async def test_precheck_rules_passes_changed_files_derived_from_diff() -> None:
    """run_precheck must be diff-scoped on the live per-PR gate path -- see
    argus.precheck.engine.run_precheck's own docstring for why (whole-
    worktree scanners would otherwise flood every PR with pre-existing
    findings, which _MAX_RESULTS would then truncate arbitrarily).
    """
    from argus.precheck.engine import PrecheckResult

    with (
        patch(
            "argus.precheck.engine.run_precheck",
            new=AsyncMock(return_value=PrecheckResult()),
        ) as mock_run_precheck,
        patch("argus.storage.precheck.log_candidate_firings", new=AsyncMock()),
    ):
        await _node_precheck_rules(_make_state(), {"configurable": {"worktree_path": "/tmp/wt"}})

    mock_run_precheck.assert_awaited_once_with("/tmp/wt", changed_files=["a.py"])


async def test_precheck_rules_attaches_failed_scanner_names() -> None:
    """A scanner returning None (see PrecheckResult.failed_scanners) must
    surface into state as precheck_scanner_failures -- consumed by
    run_review's degraded-coverage section so a crashed/timed-out scanner
    is visible in the review comment, not silently indistinguishable from
    "ran clean". This module stays fail-open regardless: it must not
    produce precheck_fast_fail/precheck_findings just because a scanner
    failed.
    """
    from argus.precheck.engine import PrecheckResult

    with (
        patch(
            "argus.precheck.engine.run_precheck",
            new=AsyncMock(return_value=PrecheckResult(failed_scanners=["zizmor", "trivy"])),
        ),
        patch("argus.storage.precheck.log_candidate_firings", new=AsyncMock()) as mock_log,
    ):
        result = await _node_precheck_rules(
            _make_state(), {"configurable": {"worktree_path": "/tmp/wt"}}
        )

    assert result["precheck_scanner_failures"] == ["zizmor", "trivy"]
    assert "precheck_findings" not in result
    assert "precheck_fast_fail" not in result
    mock_log.assert_not_awaited()


async def test_precheck_rules_verified_findings_set_fast_fail() -> None:
    from argus.precheck.engine import PrecheckResult
    from argus.precheck.sarif import SarifResult

    verified_hit = SarifResult(rule_id="r2", level="error", message="m", file="b.py", line=2)

    with (
        patch(
            "argus.precheck.engine.run_precheck",
            new=AsyncMock(return_value=PrecheckResult(verified_findings=[verified_hit])),
        ),
        patch("argus.storage.precheck.log_candidate_firings", new=AsyncMock()) as mock_log,
    ):
        result = await _node_precheck_rules(
            _make_state(), {"configurable": {"worktree_path": "/tmp/wt"}}
        )

    assert result["precheck_fast_fail"] == [verified_hit.as_storage_dict()]
    # No candidate findings in this case — nothing to log.
    mock_log.assert_not_awaited()


async def test_precheck_rules_fast_fail_survives_candidate_logging_failure() -> None:
    # Regression test: an earlier version computed precheck_fast_fail only
    # after candidate-firing logging, inside the same try block -- a
    # log_candidate_firings failure there discarded an already-computed
    # verified-rule gate decision instead of just failing to log telemetry.
    # A logging failure must never suppress a gate decision.
    from argus.precheck.engine import PrecheckResult
    from argus.precheck.sarif import SarifResult

    verified_hit = SarifResult(rule_id="r2", level="error", message="m", file="b.py", line=2)
    candidate_hit = SarifResult(rule_id="r1", level="warning", message="m", file="a.py", line=1)

    with (
        patch(
            "argus.precheck.engine.run_precheck",
            new=AsyncMock(
                return_value=PrecheckResult(
                    candidate_findings=[candidate_hit], verified_findings=[verified_hit]
                )
            ),
        ),
        patch(
            "argus.storage.precheck.log_candidate_firings",
            new=AsyncMock(side_effect=RuntimeError("db boom")),
        ),
    ):
        result = await _node_precheck_rules(
            _make_state(), {"configurable": {"worktree_path": "/tmp/wt"}}
        )

    assert result["precheck_fast_fail"] == [verified_hit.as_storage_dict()]
    assert result["precheck_findings"] == [candidate_hit.as_finding_dict()]


async def test_precheck_rules_engine_exception_is_swallowed() -> None:
    with patch(
        "argus.precheck.engine.run_precheck",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ):
        result = await _node_precheck_rules(
            _make_state(), {"configurable": {"worktree_path": "/tmp/wt"}}
        )

    assert result == {}


async def test_precheck_rules_malformed_state_propagates_not_swallowed() -> None:
    # A missing state["head_sha"] is a graph-wiring bug, not a precheck-
    # engine hiccup -- only run_precheck's own try/except is fail-open;
    # state access happens outside it and must still raise.
    state = _make_state()
    del state["head_sha"]

    with pytest.raises(KeyError):
        await _node_precheck_rules(state, {"configurable": {"worktree_path": "/tmp/wt"}})


async def test_precheck_rules_malformed_request_propagates_not_swallowed() -> None:
    # Same boundary as the head_sha case above, for state["request"]:
    # ReviewRequest.model_validate happens outside every try in this node,
    # not inside the narrow candidate-logging try, so a malformed request
    # must still raise rather than being swallowed as a logging failure.
    state = _make_state(request={"not": "a valid request"})

    with pytest.raises(ValidationError):
        await _node_precheck_rules(state, {"configurable": {"worktree_path": "/tmp/wt"}})


# ---------------------------------------------------------------------------
# _node_write_review: precheck_findings attached without polluting
# state["findings"] (collect_findings' expected-vs-succeeded accounting
# already ran by the time this node executes, so it must not see an extra,
# unplanned reviewer entry retroactively).
# ---------------------------------------------------------------------------


async def test_write_review_appends_precheck_findings_as_extra_context() -> None:
    plan: dict[str, list[object]] = {
        "system_groups": [],
        "cross_cutting_concerns": [],
        "file_manifest": [],
    }
    state = _make_state(
        plan=plan,
        findings=[],
        precheck_findings=[
            {"file": "a.py", "line": 1, "description": "hit", "context": "rule: r1"}
        ],
    )

    fake_response = ReviewResponse(
        verdict=Verdict.APPROVE,
        risk_level=RiskLevel.LOW,
        review_comment="looks fine",
    )

    with patch("argus.graph.write_review", new=AsyncMock(return_value=fake_response)) as mock_write:
        await _node_write_review(state)

    assert mock_write.await_args is not None
    findings_arg = mock_write.await_args.args[0]
    assert len(findings_arg) == 1
    assert findings_arg[0].system_group == "deterministic-precheck"
    assert findings_arg[0].findings[0].description == "hit"


async def test_write_review_no_precheck_findings_leaves_findings_untouched() -> None:
    plan: dict[str, list[object]] = {
        "system_groups": [],
        "cross_cutting_concerns": [],
        "file_manifest": [],
    }
    state = _make_state(plan=plan, findings=[])

    fake_response = ReviewResponse(
        verdict=Verdict.APPROVE, risk_level=RiskLevel.LOW, review_comment="looks fine"
    )

    with patch("argus.graph.write_review", new=AsyncMock(return_value=fake_response)) as mock_write:
        await _node_write_review(state)

    assert mock_write.await_args is not None
    assert mock_write.await_args.args[0] == []


# ---------------------------------------------------------------------------
# Structural: the compiled StateGraph actually has the fan-out/fan-in
# topology this file's tests otherwise only exercise node-function-by-
# node-function. LangGraph can silently mis-wire an edge (wrong source,
# unreachable node) with no test noticing until runtime.
# ---------------------------------------------------------------------------


def test_graph_wires_precheck_fan_out_and_fan_in() -> None:
    from argus.graph import _build_review_graph

    compiled = _build_review_graph().compile()
    graph = compiled.get_graph()
    nodes = set(graph.nodes.keys())
    edges = {(e.source, e.target) for e in graph.edges}

    for name in ("precheck_checks", "precheck_rules", "precheck_join", "precheck_fail"):
        assert name in nodes

    # fetch_diff fans out to both precheck nodes ...
    assert ("fetch_diff", "precheck_checks") in edges
    assert ("fetch_diff", "precheck_rules") in edges
    # ... which fan back in at precheck_join ...
    assert ("precheck_checks", "precheck_join") in edges
    assert ("precheck_rules", "precheck_join") in edges
    # ... which conditionally routes to the fast-fail terminal node or
    # continues into the normal round-1+ path.
    assert ("precheck_join", "precheck_fail") in edges
    assert ("precheck_join", "early_verifier") in edges
    assert ("precheck_fail", "__end__") in edges
