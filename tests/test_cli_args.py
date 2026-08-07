"""Tests for CLI argument parsing: positional repo vs --repo, --post/--sha
interaction, and the ``argus prompts`` subcommands.

Argus is invoked as ``argus review owner/repo --pr N`` (positional, per the
published docs) or ``argus review --repo owner/repo --pr N`` (the legacy
flag form the argus-review-loop skill still uses). Both must resolve to an
identical ``ReviewRequest.repo``.
"""

from __future__ import annotations

import argparse
import os

import pytest

from argus.cli import _build_parser, _package_version, _resolve_repo, _validate_review_args
from argus.models import ReviewRequest


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = _build_parser()
    return parser.parse_args(argv)


class TestRepoPositionalVsFlag:
    def test_positional_repo_form(self) -> None:
        args = _parse(["review", "owner/repo", "--pr", "42"])
        parser = _build_parser()
        assert _resolve_repo(parser, args) == "owner/repo"

    def test_flag_repo_form_unchanged(self) -> None:
        args = _parse(["review", "--repo", "owner/repo", "--pr", "42"])
        parser = _build_parser()
        assert _resolve_repo(parser, args) == "owner/repo"

    def test_both_forms_produce_identical_resolved_repo(self) -> None:
        parser = _build_parser()
        positional = _parse(["review", "owner/repo", "--pr", "42"])
        flag = _parse(["review", "--repo", "owner/repo", "--pr", "42"])
        assert _resolve_repo(parser, positional) == _resolve_repo(parser, flag)

    def test_both_positional_and_flag_is_an_error(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["review", "owner/repo", "--repo", "other/repo", "--pr", "1"])

    def test_neither_positional_nor_flag_is_an_error(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["review", "--pr", "1"])
        with pytest.raises(SystemExit):
            _resolve_repo(parser, args)

    def test_both_forms_build_identical_review_request(self) -> None:
        """Acceptance criterion: --repo path is unchanged end-to-end — both
        invocation forms must produce an identical ReviewRequest."""
        parser = _build_parser()
        positional = _parse(["review", "owner/repo", "--pr", "42", "--dismiss", "B1 -- ok"])
        flag = _parse(["review", "--repo", "owner/repo", "--pr", "42", "--dismiss", "B1 -- ok"])

        def _to_request(args: argparse.Namespace) -> ReviewRequest:
            return ReviewRequest(
                repo=_resolve_repo(parser, args),
                pr_number=args.pr,
                sha=args.sha,
                base_ref=args.base_ref,
                dismissals=args.dismiss,
            )

        assert _to_request(positional) == _to_request(flag)

    def test_flat_legacy_invocation_still_works(self) -> None:
        """``argus --repo owner/repo --pr N`` (no ``review`` subcommand) is
        rewritten to ``review --repo owner/repo --pr N`` by ``main()``; here
        we exercise the parser directly with the rewritten form."""
        parser = _build_parser()
        args = parser.parse_args(["review", "--repo", "owner/repo", "--pr", "42"])
        assert _resolve_repo(parser, args) == "owner/repo"


class TestPostShaValidation:
    def test_post_with_sha_and_no_pr_errors(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            ["review", "owner/repo", "--sha", "abc123", "--post"],
        )
        with pytest.raises(SystemExit):
            _validate_review_args(parser, args)

    def test_post_with_pr_is_fine(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["review", "owner/repo", "--pr", "42", "--post"])
        _validate_review_args(parser, args)  # must not raise

    def test_post_defaults_off(self) -> None:
        args = _parse(["review", "owner/repo", "--pr", "42"])
        assert args.post is False
        assert args.commit_status is False

    def test_commit_status_flag_parses(self) -> None:
        args = _parse(["review", "owner/repo", "--pr", "42", "--commit-status"])
        assert args.commit_status is True


class TestNoPromptOverridesFlag:
    def test_defaults_off(self) -> None:
        args = _parse(["review", "owner/repo", "--pr", "42"])
        assert args.no_prompt_overrides is False

    def test_flag_parses(self) -> None:
        args = _parse(["review", "owner/repo", "--pr", "42", "--no-prompt-overrides"])
        assert args.no_prompt_overrides is True


class TestVersionFlag:
    def test_package_version_returns_a_nonempty_string(self) -> None:
        version = _package_version()
        assert isinstance(version, str)
        assert version
        # Guards against this test passing for the wrong reason: in a
        # properly installed dev checkout, dist-info always resolves, so
        # this should never silently be exercising the "unknown" fallback
        # path instead (that path has its own dedicated test below).
        assert version != "unknown"

    def test_package_version_falls_back_to_unknown_when_not_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import importlib.metadata

        def _raise(_name: str) -> str:
            raise importlib.metadata.PackageNotFoundError

        monkeypatch.setattr(importlib.metadata, "version", _raise)
        assert _package_version() == "unknown"

    def test_version_flag_prints_and_exits_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert out.startswith("argus ")

    def test_main_routes_bare_version_flag_to_top_level_parser(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``--version`` must not be rewritten into ``review --version`` by
        main()'s flat-legacy-invocation argv rewrite (see main()'s
        docstring comment on that rewrite) -- the review subparser has no
        --version flag of its own."""
        from argus.cli import main

        monkeypatch.setattr("sys.argv", ["argus", "--version"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert out.startswith("argus ")


class TestModelOverrideFlags:
    """--specialist-model/--frontier-model argparse-level behavior.

    See TestApplyModelOverrideFlag below for the env-var-setting behavior
    these parsed values feed into via ``_run_review`` ->
    ``_apply_model_override_flag``.
    """

    def test_flags_default_to_unset_sentinel_when_omitted(self) -> None:
        """Omitting the flag must be distinguishable from explicitly passing
        an empty string -- see _MODEL_OVERRIDE_UNSET's module docstring for
        why a plain None/env-resolved default can't tell these apart."""
        from argus.cli import _MODEL_OVERRIDE_UNSET

        args = _parse(["review", "owner/repo", "--pr", "1"])
        assert args.specialist_model is _MODEL_OVERRIDE_UNSET
        assert args.frontier_model is _MODEL_OVERRIDE_UNSET

    def test_flags_override_explicitly(self) -> None:
        args = _parse(
            [
                "review",
                "owner/repo",
                "--pr",
                "1",
                "--specialist-model",
                "claude-sonnet-4-6",
                "--frontier-model",
                "claude-opus-5",
            ]
        )
        assert args.specialist_model == "claude-sonnet-4-6"
        assert args.frontier_model == "claude-opus-5"

    def test_flags_accept_explicit_empty_string(self) -> None:
        """An explicit empty string (to clear an already-set env var, see
        TestApplyModelOverrideFlag) must parse distinctly from the omitted
        case, not collapse to the same sentinel."""
        args = _parse(
            ["review", "owner/repo", "--pr", "1", "--specialist-model", "", "--frontier-model", ""]
        )
        assert args.specialist_model == ""
        assert args.frontier_model == ""


class TestApplyModelOverrideFlag:
    """``_apply_model_override_flag`` -- the env-var-setting behavior that
    ``_run_review`` applies to a parsed --specialist-model/--frontier-model
    value before ``argus.llm.models`` is ever imported. Covers the actual
    runtime behavior added to cli.py, not just argparse-level parsing."""

    def test_unset_sentinel_leaves_env_var_untouched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.cli import _MODEL_OVERRIDE_UNSET, _apply_model_override_flag

        monkeypatch.setenv("ARGUS_SPECIALIST_MODEL", "claude-sonnet-5")
        _apply_model_override_flag("ARGUS_SPECIALIST_MODEL", _MODEL_OVERRIDE_UNSET)
        assert os.environ["ARGUS_SPECIALIST_MODEL"] == "claude-sonnet-5"

    def test_unset_sentinel_leaves_absent_env_var_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from argus.cli import _MODEL_OVERRIDE_UNSET, _apply_model_override_flag

        monkeypatch.delenv("ARGUS_SPECIALIST_MODEL", raising=False)
        _apply_model_override_flag("ARGUS_SPECIALIST_MODEL", _MODEL_OVERRIDE_UNSET)
        assert "ARGUS_SPECIALIST_MODEL" not in os.environ

    def test_explicit_value_sets_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_apply_model_override_flag writes directly to os.environ, not
        through monkeypatch, so monkeypatch.delenv registers no teardown
        for a variable that was already absent -- seed a placeholder first
        (monkeypatch.setenv) so monkeypatch always has a prior value to
        restore, regardless of the ambient environment's starting state,
        preventing this test's ARGUS_FRONTIER_MODEL write from leaking into
        later tests."""
        from argus.cli import _apply_model_override_flag

        monkeypatch.setenv("ARGUS_FRONTIER_MODEL", "placeholder-pre-test-value")
        _apply_model_override_flag("ARGUS_FRONTIER_MODEL", "claude-opus-5")
        assert os.environ["ARGUS_FRONTIER_MODEL"] == "claude-opus-5"

    def test_explicit_empty_string_clears_already_set_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from argus.cli import _apply_model_override_flag

        monkeypatch.setenv("ARGUS_SPECIALIST_MODEL", "claude-sonnet-5")
        _apply_model_override_flag("ARGUS_SPECIALIST_MODEL", "")
        assert "ARGUS_SPECIALIST_MODEL" not in os.environ

    def test_non_string_value_raises_type_error(self) -> None:
        """Must be a real raise, not a bare assert -- assert is stripped
        under -O/PYTHONOPTIMIZE, which would degrade this into an opaque
        TypeError from os.environ.__setitem__ instead of a clear message
        naming the offending env var and value."""
        from argus.cli import _apply_model_override_flag

        with pytest.raises(TypeError, match="ARGUS_SPECIALIST_MODEL"):
            _apply_model_override_flag("ARGUS_SPECIALIST_MODEL", 123)


class TestPromptsSubcommand:
    def test_prompts_list_parses(self) -> None:
        args = _parse(["prompts", "list"])
        assert args.command == "prompts"
        assert args.prompts_command == "list"

    def test_prompts_export_parses_with_dir_and_force(self) -> None:
        args = _parse(["prompts", "export", "/tmp/out", "--force"])
        assert args.command == "prompts"
        assert args.prompts_command == "export"
        assert args.dir == "/tmp/out"
        assert args.force is True

    def test_prompts_export_force_defaults_off(self) -> None:
        args = _parse(["prompts", "export", "/tmp/out"])
        assert args.force is False
