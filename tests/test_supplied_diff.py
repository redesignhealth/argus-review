"""Supplied-diff review kind (TECH-9826)."""

from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from argus import cli
from argus.graph import _node_fetch_diff, _node_scope_gate, run_review
from argus.models import (
    ReviewRequest,
    ReviewResponse,
    RiskLevel,
    Verdict,
)
from argus.supplied_diff import (
    MAX_CHANGED_LINES,
    MAX_DOC_CHARS,
    SuppliedDiffError,
    compute_digest,
    count_changed_lines,
    parse_supplied_diff,
)

BASE = "b" * 40
HEAD = "a" * 40
_GRAPH = "argus.graph"


def make_doc(body: str | None = None, *, head: str = HEAD) -> str:
    if body is None:
        body = (
            f"== commit {HEAD} merge upstream {'c' * 40} ==\n"
            f"mechanical take-theirs docs/x.md {'d' * 40}\n"
            "diff --git a/app/x.py b/app/x.py\n"
            "--- a/app/x.py\n"
            "+++ b/app/x.py\n"
            "@@ -1,2 +1,2 @@\n"
            "-old\n"
            "+new\n"
            f"== commit {HEAD} own ==\n"
        )
    return f"format: upstream-sync-resolution-diff v1\nbase: {BASE}\nhead: {head}\n{body}"


def _request(doc: str, **kw: Any) -> ReviewRequest:
    return ReviewRequest(repo="o/r", supplied_diff=doc, **kw)


def _gh_that_must_not_be_called() -> MagicMock:
    return MagicMock(side_effect=AssertionError("supplied diff must not touch GitHub"))


# ---------------------------------------------------------------------------
# Document parsing, digest, size
# ---------------------------------------------------------------------------


def test_digest_is_sha256_of_the_exact_utf8_bytes() -> None:
    doc = make_doc("diff --git a/é.py b/é.py\n+ünï\n")
    expected = "sha256:" + hashlib.sha256(doc.encode("utf-8")).hexdigest()
    assert compute_digest(doc) == expected
    assert parse_supplied_diff(doc).digest == expected
    assert parse_supplied_diff(doc).digest == compute_digest(doc)  # stable
    assert compute_digest(doc + "\n") != expected  # exact bytes, no normalisation


def test_parse_extracts_shas_files_and_changed_lines() -> None:
    parsed = parse_supplied_diff(make_doc())
    assert (parsed.base_sha, parsed.head_sha) == (BASE, HEAD)
    assert parsed.files == ("app/x.py",)
    assert parsed.changed_lines == 2  # headers excluded


def test_removed_line_starting_with_dashes_counts_as_content() -> None:
    doc = "diff --git a/f b/f\n--- a/f\n+++ b/f\n@@ -1 +1 @@\n--- sql comment\n+ok\n"
    assert count_changed_lines(doc) == 2
    # a removed "-- x" directly followed by an added "++ y" inside a hunk is still content
    doc = "diff --git a/f b/f\n--- a/f\n+++ b/f\n@@ -1 +1 @@\n--- x\n+++ y\n"
    assert count_changed_lines(doc) == 2


def test_count_changed_lines_across_files_and_commits() -> None:
    doc = make_doc(
        "== commit " + "1" * 40 + " own ==\n"
        "diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -1 +1 @@\n-x\n+y\n"
        "diff --git a/b b/b\nnew file mode 100644\n--- /dev/null\n+++ b/b\n@@ -0,0 +1,2 @@\n+1\n+2\n"
        "== commit " + "2" * 40 + " own ==\n"
        f"mechanical take-theirs docs/x.md {'d' * 40}\n"
        "diff --git a/c b/c\n--- a/c\n+++ b/c\n@@ -1 +0,0 @@\n-gone\n"
    )
    assert count_changed_lines(doc) == 5
    assert parse_supplied_diff(doc).files == ("a", "b", "c")


def test_size_halt_reason_directly() -> None:
    from argus.supplied_diff import size_halt_reason

    assert size_halt_reason(parse_supplied_diff(make_doc())) is None
    assert "characters" in (size_halt_reason(parse_supplied_diff(make_doc("x" * 150_001))) or "")


def test_supplied_diff_rejects_dismissals() -> None:
    with pytest.raises(ValueError, match="dismissals"):
        ReviewRequest(repo="o/r", supplied_diff=make_doc(), dismissals=["B1 -- x"])


@pytest.mark.asyncio
async def test_precheck_checks_skips_github_for_supplied_diff() -> None:
    from argus.graph import _node_precheck_checks

    req = _request(make_doc())
    with patch("argus.github_client.GitHubClient", _gh_that_must_not_be_called()) as gh:
        out = await _node_precheck_checks({"request": req.model_dump(), "head_sha": HEAD})
    gh.assert_not_called()
    assert out == {"checks_signal": "unknown"}


def test_supplied_diff_rejects_a_pr_number() -> None:
    with pytest.raises(ValueError, match="pr_number"):
        ReviewRequest(repo="o/r", pr_number=3, supplied_diff=make_doc())


@pytest.mark.parametrize(
    "doc",
    [
        "",
        f"format: upstream-sync-resolution-diff v2\nbase: {BASE}\nhead: {HEAD}\n",
        f"base: {BASE}\nhead: {HEAD}\n",
        f"format: upstream-sync-resolution-diff v1\nbase: {BASE[:39]}\nhead: {HEAD}\n",
        f"format: upstream-sync-resolution-diff v1\nbase: {BASE}\nhead: {HEAD.upper()}\n",
        f"format: upstream-sync-resolution-diff v1\nhead: {HEAD}\nbase: {BASE}\n",
        f"format: upstream-sync-resolution-diff v1\nbase: {BASE}\n",
    ],
)
def test_malformed_documents_are_rejected(doc: str) -> None:
    with pytest.raises(SuppliedDiffError):
        parse_supplied_diff(doc)


def test_head_must_equal_the_review_head() -> None:
    parse_supplied_diff(make_doc(), expect_head_sha=HEAD)
    with pytest.raises(SuppliedDiffError, match="does not match"):
        parse_supplied_diff(make_doc(), expect_head_sha="e" * 40)


def test_request_kind_is_implied_by_the_document() -> None:
    assert ReviewRequest(repo="o/r").review_kind == "pr"
    assert _request(make_doc()).review_kind == "supplied_diff"
    with pytest.raises(ValueError, match="requires supplied_diff"):
        ReviewRequest(repo="o/r", review_kind="supplied_diff")


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


async def _fetch(req: ReviewRequest, **conf: Any) -> dict[str, Any]:
    return await _node_fetch_diff({"request": req.model_dump()}, {"configurable": conf})


@pytest.mark.asyncio
async def test_fetch_uses_the_document_and_never_touches_github_or_dismissals() -> None:
    doc = make_doc()
    # A trusted collaborator /dismiss comment exists on the PR, and the caller also supplies a
    # dismissal; neither may be read or applied.
    req = _request(doc)
    with (
        patch("argus.github_client.GitHubClient", _gh_that_must_not_be_called()) as gh,
        patch(f"{_GRAPH}._fetch_dismissed_findings", new_callable=AsyncMock) as dismissed,
        patch(f"{_GRAPH}._fetch_prior_review", new_callable=AsyncMock) as prior,
        patch(f"{_GRAPH}._fetch_pr_diff_and_description", new_callable=AsyncMock) as fetch_pr,
        patch(f"{_GRAPH}._fetch_full_pr_changed_files", new_callable=AsyncMock) as changed,
    ):
        out = await _fetch(req)
    gh.assert_not_called()
    for fn in (dismissed, prior, fetch_pr, changed):
        fn.assert_not_called()
    assert out["diff"] == doc
    assert out["head_sha"] == HEAD
    assert out["scope_base_sha"] == BASE
    assert out["prior_review"] == {}
    assert "resolution diff" in out["description"].lower()
    assert "mechanical" in out["description"]
    assert out["bench_config_unconfirmed"] is None


@pytest.mark.asyncio
async def test_fetch_rejects_a_head_that_differs_from_the_review_head() -> None:
    with pytest.raises(SuppliedDiffError):
        await _fetch(_request(make_doc(), sha="e" * 40))
    with pytest.raises(SuppliedDiffError):
        await _fetch(_request(make_doc()), head_sha="e" * 40)


@pytest.mark.asyncio
async def test_fetch_flags_bench_config_changes_from_the_document() -> None:
    doc = make_doc("diff --git a/.argus/bench.toml b/.argus/bench.toml\n+x = 1\n")
    out = await _fetch(_request(doc))
    assert ".argus/bench.toml" in out["bench_config_changes"]


async def _scope(doc: str) -> dict[str, Any]:
    req = _request(doc)
    state: dict[str, Any] = {"request": req.model_dump(), "head_sha": HEAD, "diff": doc}
    with (
        patch("argus.github_client.GitHubClient", _gh_that_must_not_be_called()),
        patch(f"{_GRAPH}._assess_scope", new_callable=AsyncMock) as assess,
    ):
        out = await _node_scope_gate(state, {"configurable": {}})
    assess.assert_not_called()  # the one-behaviour rule is not applied
    return out


@pytest.mark.asyncio
async def test_scope_gate_passes_without_the_one_behaviour_rule() -> None:
    out = await _scope(make_doc())
    assert "response" not in out
    assert out["scope"]["head_sha"] == HEAD and out["scope"]["base_sha"] == BASE
    assert out["scope"]["behaviors"] == []


@pytest.mark.asyncio
async def test_oversize_by_characters_halts_with_a_clear_reason() -> None:
    doc = make_doc("x" * (MAX_DOC_CHARS + 1))
    out = await _scope(doc)
    halt = out["response"]["halt"]
    assert out["response"]["verdict"] == "BLOCKING"
    assert halt["code"] == "scope_unverified"
    assert f"{len(doc)} characters" in halt["detail"] and str(MAX_DOC_CHARS) in halt["detail"]


@pytest.mark.asyncio
async def test_oversize_by_changed_lines_halts_with_a_clear_reason() -> None:
    body = "diff --git a/f b/f\n--- a/f\n+++ b/f\n" + "+a\n" * (MAX_CHANGED_LINES + 1)
    doc = make_doc(body)
    assert len(doc) < MAX_DOC_CHARS
    out = await _scope(doc)
    detail = out["response"]["halt"]["detail"]
    assert f"{MAX_CHANGED_LINES + 1} changed lines" in detail


@pytest.mark.asyncio
async def test_exactly_at_the_limits_is_not_halted() -> None:
    body = "diff --git a/f b/f\n--- a/f\n+++ b/f\n" + "+a\n" * MAX_CHANGED_LINES
    out = await _scope(make_doc(body))
    assert "response" not in out


@pytest.mark.asyncio
async def test_ordinary_pr_scope_gate_still_runs_the_model() -> None:
    req = ReviewRequest(repo="o/r", pr_number=1)
    state: dict[str, Any] = {
        "request": req.model_dump(),
        "head_sha": HEAD,
        "diff": "",
        "scope_base_sha": BASE,
    }
    with patch(f"{_GRAPH}._fetch_scope_manifest", return_value=[]) as manifest:
        out = await _node_scope_gate(state, {"configurable": {}})
    manifest.assert_called_once()
    assert out["scope"]["behaviors"] == []


# ---------------------------------------------------------------------------
# run_review: result fields, storage row, no GitHub access
# ---------------------------------------------------------------------------


def _graph_returning(response: ReviewResponse) -> AsyncMock:
    graph = AsyncMock()
    graph.ainvoke.return_value = {
        "response": response.model_dump(),
        "findings": [],
        "head_sha": HEAD,
        "scope": {"base_sha": BASE, "head_sha": HEAD, "behaviors": [], "rationale": "r"},
    }
    return graph


async def _run(request: ReviewRequest) -> tuple[ReviewResponse, AsyncMock, MagicMock]:
    response = ReviewResponse(
        verdict=Verdict.APPROVE, risk_level=RiskLevel.LOW, review_comment="ok"
    )
    graph = _graph_returning(response)
    backend = MagicMock()
    backend.upsert_completed_row = AsyncMock(return_value=MagicMock(id=None))
    worktree = MagicMock()
    worktree.__aenter__ = AsyncMock(return_value="/tmp/wt")
    worktree.__aexit__ = AsyncMock(return_value=None)
    with (
        patch(f"{_GRAPH}.build_pipeline") as build,
        patch(f"{_GRAPH}.validate_history_backend_connectivity", new_callable=AsyncMock),
        patch(f"{_GRAPH}.get_history_backend", return_value=backend),
        patch(f"{_GRAPH}.resolve_history_backend_kind", return_value="sqlite"),
        patch(f"{_GRAPH}.provisioned_worktree", return_value=worktree) as prov,
        patch("argus.github_client.GitHubClient", _gh_that_must_not_be_called()) as gh,
    ):
        build.return_value.__aenter__ = AsyncMock(return_value=graph)
        build.return_value.__aexit__ = AsyncMock(return_value=None)
        out = await run_review(request)
    gh.assert_not_called()
    assert prov.call_args.kwargs["head_sha"] == HEAD
    return out, graph, backend


@pytest.mark.asyncio
async def test_run_review_records_kind_and_digest_on_result_and_row() -> None:
    doc = make_doc()
    out, graph, backend = await _run(_request(doc))
    digest = compute_digest(doc)
    assert (out.review_kind, out.diff_digest) == ("supplied_diff", digest)
    row = backend.upsert_completed_row.call_args.kwargs["row"]
    assert (row.review_kind, row.diff_digest) == ("supplied_diff", digest)
    assert row.result_json["diff_digest"] == digest
    assert row.base_ref == BASE and row.sha == HEAD
    conf = graph.ainvoke.call_args.kwargs["config"]["configurable"]
    assert conf["review_kind"] == "supplied_diff"


@pytest.mark.asyncio
async def test_run_review_rejects_a_malformed_document_before_any_work() -> None:
    with patch(f"{_GRAPH}.build_pipeline") as build, pytest.raises(SuppliedDiffError):
        await run_review(_request("not a contract document"))
    build.assert_not_called()


@pytest.mark.asyncio
async def test_run_review_rejects_a_head_mismatch() -> None:
    with pytest.raises(SuppliedDiffError, match="does not match"):
        await run_review(_request(make_doc(), sha="e" * 40))


@pytest.mark.asyncio
async def test_ordinary_pr_keeps_default_kind_and_no_digest() -> None:
    response = ReviewResponse(
        verdict=Verdict.APPROVE, risk_level=RiskLevel.LOW, review_comment="ok"
    )
    graph = _graph_returning(response)
    backend = MagicMock()
    backend.upsert_completed_row = AsyncMock(return_value=MagicMock(id=None))
    worktree = MagicMock()
    worktree.__aenter__ = AsyncMock(return_value="/tmp/wt")
    worktree.__aexit__ = AsyncMock(return_value=None)
    gh = MagicMock()
    gh.get_pull_request.return_value = {"head_sha": HEAD}
    with (
        patch(f"{_GRAPH}.build_pipeline") as build,
        patch(f"{_GRAPH}.validate_history_backend_connectivity", new_callable=AsyncMock),
        patch(f"{_GRAPH}.get_history_backend", return_value=backend),
        patch(f"{_GRAPH}.resolve_history_backend_kind", return_value="sqlite"),
        patch(f"{_GRAPH}.provisioned_worktree", return_value=worktree),
        patch("argus.github_client.GitHubClient", return_value=gh),
    ):
        build.return_value.__aenter__ = AsyncMock(return_value=graph)
        build.return_value.__aexit__ = AsyncMock(return_value=None)
        out = await run_review(ReviewRequest(repo="o/r", pr_number=3))
    assert (out.review_kind, out.diff_digest) == ("pr", None)
    row = backend.upsert_completed_row.call_args.kwargs["row"]
    assert (row.review_kind, row.diff_digest) == ("pr", None)
    assert graph.ainvoke.call_args.kwargs["config"]["configurable"]["review_kind"] == "pr"


# ---------------------------------------------------------------------------
# Reviewer prompt and CLI
# ---------------------------------------------------------------------------


def test_reviewer_prompt_carries_the_resolution_note_only_for_this_kind() -> None:
    from argus.runners import _CAUSAL_SCOPE_DIRECTIVE, _causal_scope_directive, review_kind_var

    assert _causal_scope_directive() == _CAUSAL_SCOPE_DIRECTIVE
    token = review_kind_var.set("supplied_diff")
    try:
        text = _causal_scope_directive()
    finally:
        review_kind_var.reset(token)
    assert text.startswith(_CAUSAL_SCOPE_DIRECTIVE)
    assert "RESOLUTION DIFF" in text and "mechanical <rule> <path> <blob sha>" in text
    assert "do not flag missing upstream" in text


def _parse(*argv: str) -> Any:
    parser = cli._build_parser()
    args = parser.parse_args(["review", "o/r", *argv])
    return parser, args


def test_cli_rejects_posting_with_a_diff_file() -> None:
    for flag in ("--post", "--commit-status"):
        parser, args = _parse("--diff-file", "x.diff", flag)
        with pytest.raises(SystemExit):
            cli._validate_review_args(parser, args)


def test_cli_kind_and_diff_file_must_agree() -> None:
    parser, args = _parse("--review-kind", "supplied_diff")
    with pytest.raises(SystemExit):
        cli._validate_review_args(parser, args)
    parser, args = _parse("--review-kind", "pr", "--diff-file", "x.diff")
    with pytest.raises(SystemExit):
        cli._validate_review_args(parser, args)
    for extra in (["--pr", "1"], ["--dismiss", "B1 -- x"]):
        parser, args = _parse("--diff-file", "x.diff", *extra)
        with pytest.raises(SystemExit):
            cli._validate_review_args(parser, args)
    parser, args = _parse("--diff-file", "x.diff")
    cli._validate_review_args(parser, args)  # diff-file alone implies the kind
