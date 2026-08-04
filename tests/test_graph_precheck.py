"""Unit tests for the deterministic-precheck graph node/edge wiring.

Covers:
  - _edge_precheck_decision routes to precheck_fail iff a verified hit exists
  - _node_precheck_fail synthesizes a BLOCKING response with zero LLM calls
  - _node_precheck: checks-signal read, worktree-absent short-circuit,
    candidate findings attached to state, verified findings trigger
    precheck_fast_fail, and a run_precheck exception is swallowed
  - _node_write_review attaches precheck_findings as extra writer context
    without polluting state["findings"] (collect_findings' accounting)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch


from argus.graph import (
    _edge_precheck_decision,
    _node_precheck,
    _node_precheck_fail,
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


# ---------------------------------------------------------------------------
# _node_precheck
# ---------------------------------------------------------------------------


def _mock_gh_client(signal: str = "passing") -> AsyncMock:
    client = AsyncMock()
    client.get_checks_signal = lambda *a, **k: signal  # sync method on the real client
    return client


async def test_precheck_no_worktree_still_reads_checks_signal() -> None:
    with patch(_GH_CLIENT_CLASS, return_value=_mock_gh_client("failing")):
        result = await _node_precheck(_make_state(), {"configurable": {}})
    assert result == {"checks_signal": "failing"}


async def test_precheck_checks_api_error_falls_back_to_unknown() -> None:
    def _raise(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("boom")

    client = AsyncMock()
    client.get_checks_signal = _raise
    with patch(_GH_CLIENT_CLASS, return_value=client):
        result = await _node_precheck(_make_state(), {"configurable": {}})
    assert result == {"checks_signal": "unknown"}


async def test_precheck_candidate_findings_attached_and_logged() -> None:
    from argus.precheck.engine import PrecheckResult
    from argus.precheck.sarif import SarifResult

    candidate_hit = SarifResult(rule_id="r1", level="warning", message="m", file="a.py", line=1)

    with (
        patch(_GH_CLIENT_CLASS, return_value=_mock_gh_client("passing")),
        patch(
            "argus.precheck.engine.run_precheck",
            new=AsyncMock(return_value=PrecheckResult(candidate_findings=[candidate_hit])),
        ),
        patch("argus.storage.precheck.log_candidate_firing", new=AsyncMock()) as mock_log,
    ):
        result = await _node_precheck(_make_state(), {"configurable": {"worktree_path": "/tmp/wt"}})

    assert result["checks_signal"] == "passing"
    assert result["precheck_findings"] == [candidate_hit.as_finding_dict()]
    assert "precheck_fast_fail" not in result
    mock_log.assert_awaited_once()


async def test_precheck_verified_findings_set_fast_fail() -> None:
    from argus.precheck.engine import PrecheckResult
    from argus.precheck.sarif import SarifResult

    verified_hit = SarifResult(rule_id="r2", level="error", message="m", file="b.py", line=2)

    with (
        patch(_GH_CLIENT_CLASS, return_value=_mock_gh_client("unknown")),
        patch(
            "argus.precheck.engine.run_precheck",
            new=AsyncMock(return_value=PrecheckResult(verified_findings=[verified_hit])),
        ),
        patch("argus.storage.precheck.log_candidate_firing", new=AsyncMock()),
    ):
        result = await _node_precheck(_make_state(), {"configurable": {"worktree_path": "/tmp/wt"}})

    assert result["precheck_fast_fail"] == [verified_hit.as_storage_dict()]


async def test_precheck_engine_exception_is_swallowed() -> None:
    with (
        patch(_GH_CLIENT_CLASS, return_value=_mock_gh_client("passing")),
        patch(
            "argus.precheck.engine.run_precheck",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ),
    ):
        result = await _node_precheck(_make_state(), {"configurable": {"worktree_path": "/tmp/wt"}})

    assert result == {"checks_signal": "passing"}


# ---------------------------------------------------------------------------
# _node_write_review: precheck_findings attached without polluting
# state["findings"] (collect_findings' expected-vs-succeeded accounting
# already ran by the time this node executes, so it must not see an extra,
# unplanned reviewer entry retroactively).
# ---------------------------------------------------------------------------


async def test_write_review_appends_precheck_findings_as_extra_context() -> None:
    plan = {"system_groups": [], "cross_cutting_concerns": [], "file_manifest": []}
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

    findings_arg = mock_write.await_args.args[0]
    assert len(findings_arg) == 1
    assert findings_arg[0].system_group == "deterministic-precheck"
    assert findings_arg[0].findings[0].description == "hit"


async def test_write_review_no_precheck_findings_leaves_findings_untouched() -> None:
    plan = {"system_groups": [], "cross_cutting_concerns": [], "file_manifest": []}
    state = _make_state(plan=plan, findings=[])

    fake_response = ReviewResponse(
        verdict=Verdict.APPROVE, risk_level=RiskLevel.LOW, review_comment="looks fine"
    )

    with patch("argus.graph.write_review", new=AsyncMock(return_value=fake_response)) as mock_write:
        await _node_write_review(state)

    assert mock_write.await_args.args[0] == []
