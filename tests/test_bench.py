"""Tests for argus.bench: the leaf-reviewer platform/model config.

Covers: the override chain (packaged default < user-global < repo-local <
ARGUS_BENCH_FILE) resolves in priority order with sparse-overlay merge
semantics, ``ARGUS_NO_BENCH_OVERRIDES`` disables the whole chain, loud
validation errors for unknown top-level/table keys, unknown platform/
caching values, and a dangling ``prompt_name``, unknown-role resolution
errors, the ``claude-sdk``/``gemini``/``openai-responses`` platform
runner registrations, platform inference for sparse model-only overrides,
and verification of packaged default resolutions (Gemini for bulk reviewers,
Claude for individual roles).
"""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from argus import bench
from argus import runners as runners_module
from argus.config import clear_cache as clear_settings_cache
from argus.llm.models import CLAUDE_DEFAULT, resolve as resolve_alias


@pytest.fixture(autouse=True)
def _clear_bench_cache() -> Generator[None, None, None]:
    bench.clear_cache()
    yield
    bench.clear_cache()


@pytest.fixture(autouse=True)
def _isolate_standard_override_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the two standard override locations at throwaway directories.

    Mirrors ``tests/test_prompts.py``'s fixture of the same purpose: without
    this, a developer's real ``~/.config/argus/bench.toml`` (or a
    ``.argus/bench.toml`` left in whatever directory the test runner's cwd
    happens to be) could leak into test results.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config-home"))
    monkeypatch.delenv("ARGUS_NO_BENCH_OVERRIDES", raising=False)
    monkeypatch.delenv("ARGUS_BENCH_FILE", raising=False)
    clear_settings_cache()
    bench.clear_cache()


def _write_toml(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


# ---------------------------------------------------------------------------
# Packaged default resolutions
# ---------------------------------------------------------------------------


class TestPackagedDefaultResolutions:
    """The packaged default bench (argus/bench_default.toml):
    - [bulk_reviewer] defaults to gemini (gemini-mini, auto caching)
    - [roles.*] (cross-cutting, blocking-validator, feedback-verifier) default to claude-sdk
    """

    def test_system_generalist_matches_packaged_default(self) -> None:
        entry = bench.resolve("system-generalist")
        assert entry.platform == "gemini"
        assert entry.model == "gemini-mini"
        assert entry.caching == "auto"
        assert resolve_alias(entry.model) == "gemini-3.8-flash"
        assert entry.prompt_name == "pr-review-subagent"

    def test_tests_and_docs_matches_packaged_default(self) -> None:
        entry = bench.resolve("tests-and-docs")
        assert entry.platform == "gemini"
        assert entry.model == "gemini-mini"
        assert entry.caching == "auto"
        assert resolve_alias(entry.model) == "gemini-3.8-flash"
        assert entry.prompt_name == "pr-review-tests-and-docs"

    @pytest.mark.parametrize(
        "specialist",
        [
            "security",
            "sql",
            "infra",
            "orchestration",
            "frontend",
            "slackbot",
            "deployment",
            "llm-patterns",
            "observability",
        ],
    )
    def test_every_specialist_matches_packaged_default(self, specialist: str) -> None:
        entry = bench.resolve(f"specialist-{specialist}")
        assert entry.platform == "gemini"
        assert entry.model == "gemini-mini"
        assert entry.caching == "auto"
        assert resolve_alias(entry.model) == "gemini-3.8-flash"
        assert entry.prompt_name == f"pr-review-specialist-{specialist}"

    def test_cross_cutting_matches_packaged_default(self) -> None:
        entry = bench.resolve("cross-cutting")
        assert entry.platform == "claude-sdk"
        assert resolve_alias(entry.model) == runners_module._CROSS_CUTTING_MODEL
        assert entry.prompt_name == "pr-review-cross-cutting"

    def test_blocking_validator_matches_packaged_default(self) -> None:
        entry = bench.resolve("blocking-validator")
        assert entry.platform == "claude-sdk"
        assert resolve_alias(entry.model) == runners_module._SYSTEM_REVIEWER_MODEL
        assert entry.prompt_name == "pr-review-blocking-validator"

    def test_feedback_verifier_matches_packaged_default(self) -> None:
        entry = bench.resolve("feedback-verifier")
        assert entry.platform == "claude-sdk"
        assert resolve_alias(entry.model) == runners_module._SYSTEM_REVIEWER_MODEL
        assert entry.prompt_name == "pr-review-feedback-verifier"


# ---------------------------------------------------------------------------
# Override chain precedence + sparse-overlay merge
# ---------------------------------------------------------------------------


class TestOverrideChain:
    def test_no_overrides_uses_packaged_default(self) -> None:
        entry = bench.resolve("system-generalist")
        assert entry.model == "gemini-mini"
        assert entry.platform == "gemini"

    def test_user_global_sparse_overlay_overrides_only_specified_key(self, tmp_path: Path) -> None:
        user_global = Path(os.environ["XDG_CONFIG_HOME"]) / "argus" / "bench.toml"
        _write_toml(user_global, '[bulk_reviewer]\nmodel = "claude-mini"\n')
        bench.clear_cache()

        entry = bench.resolve("system-generalist")
        assert entry.model == "claude-mini"
        assert entry.platform == "claude-sdk"  # platform inferred from model family
        assert entry.caching == "auto"  # untouched key still falls through

    def test_user_global_sparse_model_override_infers_platform_openai(self, tmp_path: Path) -> None:
        user_global = Path(os.environ["XDG_CONFIG_HOME"]) / "argus" / "bench.toml"
        _write_toml(user_global, '[bulk_reviewer]\nmodel = "gpt-mini"\n')
        bench.clear_cache()

        entry = bench.resolve("system-generalist")
        assert entry.model == "gpt-mini"
        assert entry.platform == "openai-responses"
        assert entry.caching == "auto"

    def test_role_sparse_model_override_infers_platform(self, tmp_path: Path) -> None:
        repo_local = Path.cwd() / ".argus" / "bench.toml"
        _write_toml(repo_local, '[roles.cross-cutting]\nmodel = "gemini-frontier"\n')
        bench.clear_cache()

        entry = bench.resolve("cross-cutting")
        assert entry.model == "gemini-frontier"
        assert entry.platform == "gemini"

    def test_explicit_platform_model_mismatch_raises(self, tmp_path: Path) -> None:
        repo_local = Path.cwd() / ".argus" / "bench.toml"
        _write_toml(
            repo_local,
            '[bulk_reviewer]\nplatform = "gemini"\nmodel = "claude-mini"\n',
        )
        bench.clear_cache()
        with pytest.raises(
            ValueError, match="model 'claude-mini' is not compatible with platform 'gemini'"
        ):
            bench.load_bench()

    def test_specialist_model_forces_claude_sdk_for_bulk_reviewer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import importlib
        from argus.llm import models as models_module

        monkeypatch.setenv("ARGUS_SPECIALIST_MODEL", "claude-haiku-4-5")
        importlib.reload(models_module)
        clear_settings_cache()
        bench.clear_cache()

        try:
            entry = bench.resolve("system-generalist")
            assert entry.platform == "claude-sdk"
            assert entry.model == "claude-default"
            assert models_module.resolve(entry.model) == "claude-haiku-4-5"
        finally:
            monkeypatch.delenv("ARGUS_SPECIALIST_MODEL", raising=False)
            importlib.reload(models_module)
            clear_settings_cache()
            bench.clear_cache()

    def test_specialist_model_cleared_with_empty_string_preserves_gemini(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ARGUS_SPECIALIST_MODEL", "")
        clear_settings_cache()
        bench.clear_cache()

        entry = bench.resolve("system-generalist")
        assert entry.platform == "gemini"
        assert entry.model == "gemini-mini"

    def test_repo_local_overlay_wins_over_user_global(self) -> None:
        user_global = Path(os.environ["XDG_CONFIG_HOME"]) / "argus" / "bench.toml"
        _write_toml(user_global, '[bulk_reviewer]\nmodel = "claude-mini"\n')

        repo_local = Path.cwd() / ".argus" / "bench.toml"
        _write_toml(repo_local, '[bulk_reviewer]\nmodel = "claude-frontier"\n')
        bench.clear_cache()

        entry = bench.resolve("system-generalist")
        assert entry.model == "claude-frontier"

    def test_explicit_bench_file_wins_over_repo_local_and_user_global(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        user_global = Path(os.environ["XDG_CONFIG_HOME"]) / "argus" / "bench.toml"
        _write_toml(user_global, '[bulk_reviewer]\nmodel = "claude-mini"\n')

        repo_local = Path.cwd() / ".argus" / "bench.toml"
        _write_toml(repo_local, '[bulk_reviewer]\nmodel = "claude-frontier"\n')

        explicit = tmp_path / "explicit-bench.toml"
        _write_toml(explicit, '[bulk_reviewer]\nmodel = "claude-default"\n')
        monkeypatch.setenv("ARGUS_BENCH_FILE", str(explicit))
        clear_settings_cache()
        bench.clear_cache()

        entry = bench.resolve("system-generalist")
        assert entry.model == "claude-default"

    def test_explicit_bench_file_still_falls_through_for_unspecified_keys(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A two-line ARGUS_BENCH_FILE only overrides what it specifies --
        it does not replace the whole packaged document."""
        explicit = tmp_path / "explicit-bench.toml"
        _write_toml(explicit, '[roles.cross-cutting]\nmodel = "claude-mini"\n')
        monkeypatch.setenv("ARGUS_BENCH_FILE", str(explicit))
        clear_settings_cache()
        bench.clear_cache()

        cross_cutting = bench.resolve("cross-cutting")
        assert cross_cutting.model == "claude-mini"
        assert cross_cutting.prompt_name == "pr-review-cross-cutting"  # fell through

        # An entirely untouched bulk-bucket role still resolves to the
        # packaged default -- proves this is a sparse merge, not a full
        # document replacement.
        system_generalist = bench.resolve("system-generalist")
        assert system_generalist.model == "gemini-mini"
        assert system_generalist.platform == "gemini"

    def test_argus_bench_file_missing_path_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ARGUS_BENCH_FILE", str(tmp_path / "does-not-exist.toml"))
        clear_settings_cache()
        bench.clear_cache()

        with pytest.raises(ValueError, match="does-not-exist.toml"):
            bench.load_bench()

    def test_no_bench_overrides_ignores_every_layer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        user_global = Path(os.environ["XDG_CONFIG_HOME"]) / "argus" / "bench.toml"
        _write_toml(user_global, '[bulk_reviewer]\nmodel = "claude-mini"\n')

        repo_local = Path.cwd() / ".argus" / "bench.toml"
        _write_toml(repo_local, '[bulk_reviewer]\nmodel = "claude-frontier"\n')

        explicit = tmp_path / "explicit-bench.toml"
        _write_toml(explicit, '[bulk_reviewer]\nmodel = "claude-mini"\n')
        monkeypatch.setenv("ARGUS_BENCH_FILE", str(explicit))
        monkeypatch.setenv("ARGUS_NO_BENCH_OVERRIDES", "1")
        clear_settings_cache()
        bench.clear_cache()

        entry = bench.resolve("system-generalist")
        assert entry.model == "gemini-mini"  # packaged default, untouched
        assert entry.platform == "gemini"

    def test_load_bench_is_cached_until_clear_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        first = bench.load_bench()
        second = bench.load_bench()
        assert first is second

        bench.clear_cache()
        third = bench.load_bench()
        assert third is not first
        assert third == first  # same content, different object


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_unknown_top_level_key_raises(self) -> None:
        repo_local = Path.cwd() / ".argus" / "bench.toml"
        _write_toml(repo_local, '[typo_section]\nfoo = "bar"\n')
        bench.clear_cache()

        with pytest.raises(ValueError, match="typo_section"):
            bench.load_bench()

    def test_unknown_key_in_bulk_reviewer_raises(self) -> None:
        repo_local = Path.cwd() / ".argus" / "bench.toml"
        _write_toml(repo_local, '[bulk_reviewer]\nprompt_name = "pr-review-subagent"\n')
        bench.clear_cache()

        with pytest.raises(ValueError, match="prompt_name"):
            bench.load_bench()

    def test_unknown_key_in_role_table_raises(self) -> None:
        repo_local = Path.cwd() / ".argus" / "bench.toml"
        _write_toml(repo_local, '[roles.cross-cutting]\nbogus_key = "x"\n')
        bench.clear_cache()

        with pytest.raises(ValueError, match="bogus_key"):
            bench.load_bench()

    def test_unknown_platform_value_raises(self) -> None:
        repo_local = Path.cwd() / ".argus" / "bench.toml"
        _write_toml(repo_local, '[bulk_reviewer]\nplatform = "not-a-real-platform"\n')
        bench.clear_cache()

        with pytest.raises(ValueError, match="not-a-real-platform"):
            bench.load_bench()

    def test_unknown_caching_value_raises(self) -> None:
        repo_local = Path.cwd() / ".argus" / "bench.toml"
        _write_toml(repo_local, '[bulk_reviewer]\ncaching = "sometimes"\n')
        bench.clear_cache()

        with pytest.raises(ValueError, match="sometimes"):
            bench.load_bench()

    def test_dangling_prompt_name_raises(self) -> None:
        repo_local = Path.cwd() / ".argus" / "bench.toml"
        _write_toml(repo_local, '[roles.cross-cutting]\nprompt_name = "pr-review-does-not-exist"\n')
        bench.clear_cache()

        with pytest.raises(ValueError, match="pr-review-does-not-exist"):
            bench.load_bench()

    def test_new_role_missing_required_key_raises(self) -> None:
        repo_local = Path.cwd() / ".argus" / "bench.toml"
        _write_toml(repo_local, '[roles.brand-new-role]\nplatform = "claude-sdk"\n')
        bench.clear_cache()

        with pytest.raises(ValueError, match="brand-new-role"):
            bench.load_bench()

    def test_non_string_platform_raises_value_error_not_type_error(self) -> None:
        """A TOML array/table value for `platform` must raise the promised
        load-time ValueError, not an uncaught TypeError from `not in
        <frozenset>` on an unhashable value."""
        repo_local = Path.cwd() / ".argus" / "bench.toml"
        _write_toml(repo_local, '[bulk_reviewer]\nplatform = ["claude-sdk"]\n')
        bench.clear_cache()

        with pytest.raises(ValueError):
            bench.load_bench()

    def test_non_string_caching_raises_value_error_not_type_error(self) -> None:
        repo_local = Path.cwd() / ".argus" / "bench.toml"
        _write_toml(repo_local, '[bulk_reviewer]\ncaching = ["auto"]\n')
        bench.clear_cache()

        with pytest.raises(ValueError):
            bench.load_bench()

    def test_non_string_model_raises_value_error_not_type_error(self) -> None:
        repo_local = Path.cwd() / ".argus" / "bench.toml"
        _write_toml(repo_local, "[bulk_reviewer]\nmodel = [1, 2, 3]\n")
        bench.clear_cache()

        with pytest.raises(ValueError):
            bench.load_bench()

    def test_empty_model_string_raises(self) -> None:
        repo_local = Path.cwd() / ".argus" / "bench.toml"
        _write_toml(repo_local, '[bulk_reviewer]\nmodel = ""\n')
        bench.clear_cache()

        with pytest.raises(ValueError, match="non-empty"):
            bench.load_bench()

    def test_unknown_model_alias_raises(self) -> None:
        repo_local = Path.cwd() / ".argus" / "bench.toml"
        _write_toml(repo_local, '[bulk_reviewer]\nmodel = "not-a-real-model-alias"\n')
        bench.clear_cache()

        with pytest.raises(ValueError, match="not-a-real-model-alias"):
            bench.load_bench()

    def test_unknown_model_alias_in_role_table_raises(self) -> None:
        repo_local = Path.cwd() / ".argus" / "bench.toml"
        _write_toml(repo_local, '[roles.cross-cutting]\nmodel = "not-a-real-model-alias"\n')
        bench.clear_cache()

        with pytest.raises(ValueError, match="not-a-real-model-alias"):
            bench.load_bench()


# ---------------------------------------------------------------------------
# EXPERIMENTAL_MODELS end-to-end: accepted by validation AND resolvable
# ---------------------------------------------------------------------------


class TestExperimentalModelAliasEndToEnd:
    """Regression guard: ``_VALID_MODEL_ALIASES`` accepts any
    ``EXPERIMENTAL_MODELS`` key as a valid ``model`` value at load time --
    a bench config using one must therefore ALSO actually resolve to a
    concrete model string at runtime (via ``argus.llm.models.resolve``),
    not raise ``KeyError`` the first time that role is used. Previously
    ``resolve()`` only ever consulted ``ALIAS_MAP``, so an accepted config
    could still crash later.
    """

    def test_experimental_model_alias_validates_and_resolves(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import argus.llm.models as models

        monkeypatch.setattr(
            models, "EXPERIMENTAL_MODELS", {"claude-preview-eval": "claude-opus-6-preview"}
        )
        # _VALID_MODEL_ALIASES is a frozenset computed once at argus.bench's
        # import time from the (at-the-time) EXPERIMENTAL_MODELS dict, so
        # simulate a real, checked-in EXPERIMENTAL_MODELS entry by
        # recomputing it the same way bench.py itself does.
        monkeypatch.setattr(
            bench,
            "_VALID_MODEL_ALIASES",
            frozenset(models.ALIAS_MAP) | frozenset(models.EXPERIMENTAL_MODELS),
        )

        repo_local = Path.cwd() / ".argus" / "bench.toml"
        _write_toml(repo_local, '[roles.cross-cutting]\nmodel = "claude-preview-eval"\n')
        bench.clear_cache()

        # Load-time validation must accept it.
        entry = bench.resolve("cross-cutting")
        assert entry.model == "claude-preview-eval"

        # The actual regression: turning the validated alias into a
        # concrete model string must NOT raise KeyError.
        assert resolve_alias(entry.model) == "claude-opus-6-preview"


# ---------------------------------------------------------------------------
# resolve(): unknown role
# ---------------------------------------------------------------------------


class TestResolveUnknownRole:
    def test_unknown_role_raises_with_known_roles_listed(self) -> None:
        with pytest.raises(ValueError, match="nonexistent-role") as exc_info:
            bench.resolve("nonexistent-role")
        message = str(exc_info.value)
        assert "system-generalist" in message
        assert "cross-cutting" in message


# ---------------------------------------------------------------------------
# resolve(): wiring-status transparency warning
# ---------------------------------------------------------------------------


class TestResolveWiringStatusWarning:
    """A caller resolving a role no runner actually consults yet must be
    warned loudly -- overriding it in bench.toml would otherwise silently
    do nothing (see argus.bench's 'Wiring status' docstring section)."""

    @pytest.mark.parametrize(
        "role",
        [
            "system-generalist",
            "tests-and-docs",
            "specialist-security",
            "cross-cutting",
            "blocking-validator",
            "feedback-verifier",
        ],
    )
    def test_wired_role_does_not_warn(self, role: str, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING", logger="argus.bench"):
            bench.resolve(role)
        assert role not in caplog.text

    def test_unwired_role_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with patch("argus.bench._WIRED_ROLES", frozenset({"system-generalist"})):
            with caplog.at_level("WARNING", logger="argus.bench"):
                bench.resolve("cross-cutting")
        assert any("cross-cutting" in record.message for record in caplog.records)
        assert any("no effect" in record.message.lower() for record in caplog.records)

    def test_unwired_role_warns_only_once(self, caplog: pytest.LogCaptureFixture) -> None:
        with patch("argus.bench._WIRED_ROLES", frozenset({"system-generalist"})):
            with caplog.at_level("WARNING", logger="argus.bench"):
                bench.resolve("cross-cutting")
                bench.resolve("cross-cutting")
        matching = [r for r in caplog.records if "cross-cutting" in r.message]
        assert len(matching) == 1

    def test_wired_roles_covers_all_declared_roles(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Sanity guard: every declared role is now wired."""
        monkeypatch.setenv("ARGUS_NO_BENCH_OVERRIDES", "1")
        bench.clear_cache()
        raw = bench.load_bench()
        all_roles = set(bench.BULK_ROLE_PROMPTS) | set(raw.get("roles", {}))
        assert bench._WIRED_ROLES == all_roles

    def test_load_bench_warns_on_unwired_overlay_role(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        overlay = tmp_path / ".argus" / "bench.toml"
        _write_toml(
            overlay,
            """
            [roles.cross-cutting]
            platform = "claude-sdk"
            model = "claude-opus"
            prompt_name = "pr-review-cross-cutting"
            """,
        )
        monkeypatch.delenv("ARGUS_NO_BENCH_OVERRIDES", raising=False)
        bench.clear_cache()
        with patch("argus.bench._WIRED_ROLES", frozenset({"system-generalist"})):
            with caplog.at_level("WARNING", logger="argus.bench"):
                bench.load_bench()
        assert "cross-cutting" in caplog.text


# ---------------------------------------------------------------------------
# Platform runners
# ---------------------------------------------------------------------------


class TestPlatformRunners:
    def test_platform_runners_has_all_three_platforms(self) -> None:
        assert set(bench.PLATFORM_RUNNERS) == {"claude-sdk", "gemini", "openai-responses"}

    @pytest.mark.asyncio
    async def test_claude_sdk_runner_delegates_to_run_session_isolated(self) -> None:
        entry = bench.BenchEntry(
            role="system-generalist",
            platform="claude-sdk",
            model="claude-default",
            prompt_name="pr-review-subagent",
        )
        runner = bench.runner_for(entry)
        assert runner is bench.PLATFORM_RUNNERS["claude-sdk"]

        with patch(
            "argus.runners._run_session_isolated",
            new_callable=AsyncMock,
            return_value="session-result-sentinel",
        ) as mock_isolated:
            result = await runner(
                entry=entry,
                system_prompt="sys",
                user_message="msg",
                anthropic_api_key="key",
                anthropic_auth_token=None,
                context7_key=None,
                context7_library_id=None,
                context7_base_url=None,
                timeout_s=300,
                cwd="/tmp/repo",
                label="system:test",
                is_system_reviewer_role=True,
            )

        assert result == "session-result-sentinel"
        mock_isolated.assert_called_once_with(
            model=CLAUDE_DEFAULT,
            system_prompt="sys",
            user_message="msg",
            anthropic_api_key="key",
            anthropic_auth_token=None,
            context7_key=None,
            context7_library_id=None,
            context7_base_url=None,
            timeout_s=300,
            cwd="/tmp/repo",
            label="system:test",
            is_system_reviewer_role=True,
        )

    @pytest.mark.asyncio
    async def test_gemini_runner_delegates_to_run_session_gemini(self) -> None:
        """The ``gemini`` platform has a real runner (Track 3) -- this adapter
        must translate the shared RunnerFn call-site kwargs (anthropic_*,
        context7_*, timeout_s, cwd) into argus.gemini_runner.run_session_gemini's
        own signature (settings, repo_root), mirroring _claude_sdk_runner's
        adapter role exactly."""
        entry = bench.BenchEntry(
            role="cross-cutting",
            platform="gemini",
            model="gemini-frontier",
            prompt_name="pr-review-cross-cutting",
        )
        runner = bench.runner_for(entry)
        assert runner is bench.PLATFORM_RUNNERS["gemini"]

        with patch(
            "argus.gemini_runner.run_session_gemini",
            new_callable=AsyncMock,
            return_value="session-result-sentinel",
        ) as mock_run_session_gemini:
            result = await runner(
                entry=entry,
                system_prompt="sys",
                user_message="msg",
                anthropic_api_key=None,
                anthropic_auth_token=None,
                context7_key=None,
                context7_library_id=None,
                timeout_s=300,
                cwd="/tmp/repo",
                label="cross-cutting",
            )

        assert result == "session-result-sentinel"
        mock_run_session_gemini.assert_called_once()
        call_kwargs = mock_run_session_gemini.call_args.kwargs
        assert call_kwargs["entry"] is entry
        assert call_kwargs["system_prompt"] == "sys"
        assert call_kwargs["user_message"] == "msg"
        assert call_kwargs["label"] == "cross-cutting"
        assert call_kwargs["repo_root"] == "/tmp/repo"
        assert call_kwargs["timeout_s"] == 300
        assert "settings" in call_kwargs  # get_settings() result, not asserted further here

    @pytest.mark.asyncio
    async def test_gemini_runner_threads_caller_supplied_timeout_s_through(self) -> None:
        """`_gemini_runner` must not silently discard its own `timeout_s`
        parameter -- `run_session_gemini` re-deriving a global settings-based
        timeout instead would diverge from whatever the caller (e.g.
        run_system_reviewer, which resolves timeout_s from its own settings
        object) actually asked for."""
        entry = bench.BenchEntry(
            role="cross-cutting",
            platform="gemini",
            model="gemini-frontier",
            prompt_name="pr-review-cross-cutting",
        )
        runner = bench.runner_for(entry)

        with patch(
            "argus.gemini_runner.run_session_gemini",
            new_callable=AsyncMock,
            return_value="session-result-sentinel",
        ) as mock_run_session_gemini:
            await runner(
                entry=entry,
                system_prompt="sys",
                user_message="msg",
                anthropic_api_key=None,
                anthropic_auth_token=None,
                context7_key=None,
                context7_library_id=None,
                timeout_s=42,
                cwd="/tmp/repo",
                label="cross-cutting",
            )

        call_kwargs = mock_run_session_gemini.call_args.kwargs
        assert call_kwargs["timeout_s"] == 42

    @pytest.mark.asyncio
    async def test_openai_responses_runner_delegates_to_run_session_openai(self) -> None:
        entry = bench.BenchEntry(
            role="cross-cutting",
            platform="openai-responses",
            model="gpt-mini",
            prompt_name="pr-review-cross-cutting",
        )
        runner = bench.runner_for(entry)

        with patch(
            "argus.openai_runner.run_session_openai",
            new_callable=AsyncMock,
            return_value="openai-session-result-sentinel",
        ) as mock_run_session_openai:
            result = await runner(
                entry=entry,
                system_prompt="sys",
                user_message="msg",
                anthropic_api_key=None,
                anthropic_auth_token=None,
                context7_key=None,
                context7_library_id=None,
                timeout_s=300,
                cwd="/tmp/repo",
                label="cross-cutting",
            )

        assert result == "openai-session-result-sentinel"
        mock_run_session_openai.assert_called_once()
        call_kwargs = mock_run_session_openai.call_args.kwargs
        assert call_kwargs["entry"] is entry
        assert call_kwargs["system_prompt"] == "sys"
        assert call_kwargs["user_message"] == "msg"
        assert call_kwargs["label"] == "cross-cutting"
        assert call_kwargs["repo_root"] == "/tmp/repo"
        assert call_kwargs["timeout_s"] == 300
        assert "settings" in call_kwargs

    @pytest.mark.asyncio
    async def test_openai_runner_threads_caller_supplied_timeout_s_through(self) -> None:
        entry = bench.BenchEntry(
            role="cross-cutting",
            platform="openai-responses",
            model="gpt-mini",
            prompt_name="pr-review-cross-cutting",
        )
        runner = bench.runner_for(entry)

        with patch(
            "argus.openai_runner.run_session_openai",
            new_callable=AsyncMock,
            return_value="sentinel",
        ) as mock_run_session_openai:
            await runner(
                entry=entry,
                system_prompt="sys",
                user_message="msg",
                anthropic_api_key=None,
                anthropic_auth_token=None,
                context7_key=None,
                context7_library_id=None,
                timeout_s=42,
                cwd="/tmp/repo",
                label="cross-cutting",
            )

        call_kwargs = mock_run_session_openai.call_args.kwargs
        assert call_kwargs["timeout_s"] == 42

    def test_runner_for_unregistered_platform_raises(self) -> None:
        entry = bench.BenchEntry(
            role="x",
            platform="claude-sdk",
            model="claude-default",
            prompt_name="pr-review-subagent",
        )
        # Simulate an entry with a platform that somehow isn't registered.
        object.__setattr__(entry, "platform", "made-up-platform")
        with pytest.raises(ValueError, match="made-up-platform"):
            bench.runner_for(entry)


# ---------------------------------------------------------------------------
# run_system_reviewer: end-to-end wiring through bench
# ---------------------------------------------------------------------------


class TestRunSystemReviewerBenchWiring:
    @pytest.mark.asyncio
    async def test_run_system_reviewer_routes_through_bench_to_gemini(self) -> None:
        """With the packaged default bench, run_system_reviewer routes through
        bench to run_session_gemini with platform="gemini" and model="gemini-mini".
        """
        from datetime import datetime, timezone
        from unittest.mock import MagicMock

        from argus.pipeline_models import SystemGroup

        group = SystemGroup(
            name="backend",
            files=["src/app.py"],
            conventions="",
            review_focus="",
        )
        mock_settings = MagicMock(CONTEXT7_API_KEY=None)

        fake_session = MagicMock(
            result_text='{"system_group": "backend", "findings": [], "files_explored": []}',
            failure_reason=None,
            timed_out=False,
            cost_usd=0.0,
            tool_call_count=0,
            tool_names=[],
            context7_call_count=0,
            model="gemini-3.8-flash",
            result_text_length=0,
            duration_seconds=1.0,
            started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            finished_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )

        with (
            patch(
                "argus.runners.fetch_prompt",
                new_callable=AsyncMock,
                return_value="base prompt",
            ),
            patch(
                "argus.gemini_runner.run_session_gemini",
                new_callable=AsyncMock,
                return_value=fake_session,
            ) as mock_gemini,
        ):
            await runners_module.run_system_reviewer(
                group=group,
                diff_text="diff --git a/src/app.py b/src/app.py\n@@ -1 +1 @@\n-a\n+b\n",
                settings=mock_settings,
            )

        mock_gemini.assert_called_once()
        assert mock_gemini.call_args.kwargs["entry"].platform == "gemini"
        assert mock_gemini.call_args.kwargs["entry"].model == "gemini-mini"

    @pytest.mark.asyncio
    async def test_run_system_reviewer_with_claude_override_routes_to_claude(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When bench is overridden to claude-sdk, run_system_reviewer routes
        to _run_session_isolated.
        """
        from datetime import datetime, timezone
        from unittest.mock import MagicMock

        from argus.pipeline_models import SystemGroup

        explicit = tmp_path / "explicit-bench.toml"
        _write_toml(
            explicit,
            '[bulk_reviewer]\nplatform = "claude-sdk"\nmodel = "claude-default"\n',
        )
        monkeypatch.setenv("ARGUS_BENCH_FILE", str(explicit))
        clear_settings_cache()
        bench.clear_cache()

        group = SystemGroup(
            name="backend",
            files=["src/app.py"],
            conventions="",
            review_focus="",
        )
        mock_settings = MagicMock(CONTEXT7_API_KEY=None)

        fake_session = MagicMock()
        fake_session.result_text = ""
        fake_session.failure_reason = None
        fake_session.timed_out = False
        fake_session.cost_usd = 0.0
        fake_session.tool_call_count = 0
        fake_session.tool_names = []
        fake_session.context7_call_count = 0
        fake_session.model = runners_module._SYSTEM_REVIEWER_MODEL
        fake_session.result_text_length = 0
        fake_session.duration_seconds = 1.0
        fake_session.started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        fake_session.finished_at = datetime(2026, 1, 1, tzinfo=timezone.utc)

        with (
            patch(
                "argus.runners.fetch_prompt",
                new_callable=AsyncMock,
                return_value="base prompt",
            ),
            patch(
                "argus.runners._run_session_isolated",
                new_callable=AsyncMock,
                return_value=fake_session,
            ) as mock_isolated,
        ):
            await runners_module.run_system_reviewer(
                group=group,
                diff_text="diff --git a/src/app.py b/src/app.py\n@@ -1 +1 @@\n-a\n+b\n",
                settings=mock_settings,
            )

        mock_isolated.assert_called_once()
        assert mock_isolated.call_args.kwargs["model"] == runners_module._SYSTEM_REVIEWER_MODEL

    @pytest.mark.asyncio
    async def test_sparse_model_only_override_executes_on_inferred_claude_platform(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end regression: a sparse override specifying only `model = 'claude-mini'`
        must infer platform='claude-sdk' and execute via _run_session_isolated, NOT crash
        by routing to Gemini."""
        from datetime import datetime, timezone
        from unittest.mock import MagicMock

        from argus.pipeline_models import SystemGroup

        explicit = tmp_path / "explicit-bench.toml"
        _write_toml(explicit, '[bulk_reviewer]\nmodel = "claude-mini"\n')
        monkeypatch.setenv("ARGUS_BENCH_FILE", str(explicit))
        clear_settings_cache()
        bench.clear_cache()

        group = SystemGroup(
            name="backend",
            files=["src/app.py"],
            conventions="",
            review_focus="",
        )
        mock_settings = MagicMock(CONTEXT7_API_KEY=None)

        fake_session = MagicMock()
        fake_session.result_text = ""
        fake_session.failure_reason = None
        fake_session.timed_out = False
        fake_session.cost_usd = 0.0
        fake_session.tool_call_count = 0
        fake_session.tool_names = []
        fake_session.context7_call_count = 0
        fake_session.model = "claude-haiku-4-5"
        fake_session.result_text_length = 0
        fake_session.duration_seconds = 1.0
        fake_session.started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        fake_session.finished_at = datetime(2026, 1, 1, tzinfo=timezone.utc)

        def _boom(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("run_session_gemini must not be called for claude model")

        with (
            patch(
                "argus.runners.fetch_prompt",
                new_callable=AsyncMock,
                return_value="base prompt",
            ),
            patch(
                "argus.runners._run_session_isolated",
                new_callable=AsyncMock,
                return_value=fake_session,
            ) as mock_isolated,
            patch("argus.gemini_runner.run_session_gemini", side_effect=_boom),
        ):
            await runners_module.run_system_reviewer(
                group=group,
                diff_text="diff --git a/src/app.py b/src/app.py\n@@ -1 +1 @@\n-a\n+b\n",
                settings=mock_settings,
            )

        mock_isolated.assert_called_once()
        assert mock_isolated.call_args.kwargs["model"] == "claude-haiku-4-5"

    @pytest.mark.asyncio
    async def test_sparse_model_only_override_executes_on_inferred_openai_platform(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end regression: a sparse override specifying only `model = 'gpt-mini'`
        must infer platform='openai-responses' and execute via run_session_openai."""
        from datetime import datetime, timezone
        from unittest.mock import MagicMock

        from argus.pipeline_models import SystemGroup

        explicit = tmp_path / "explicit-bench.toml"
        _write_toml(explicit, '[bulk_reviewer]\nmodel = "gpt-mini"\n')
        monkeypatch.setenv("ARGUS_BENCH_FILE", str(explicit))
        clear_settings_cache()
        bench.clear_cache()

        group = SystemGroup(
            name="backend",
            files=["src/app.py"],
            conventions="",
            review_focus="",
        )
        mock_settings = MagicMock(CONTEXT7_API_KEY=None)

        fake_session = MagicMock()
        fake_session.result_text = ""
        fake_session.failure_reason = None
        fake_session.timed_out = False
        fake_session.cost_usd = 0.0
        fake_session.tool_call_count = 0
        fake_session.tool_names = []
        fake_session.context7_call_count = 0
        fake_session.model = "gpt-5.6-luna"
        fake_session.result_text_length = 0
        fake_session.duration_seconds = 1.0
        fake_session.started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        fake_session.finished_at = datetime(2026, 1, 1, tzinfo=timezone.utc)

        def _boom(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("run_session_gemini must not be called for gpt model")

        with (
            patch(
                "argus.runners.fetch_prompt",
                new_callable=AsyncMock,
                return_value="base prompt",
            ),
            patch(
                "argus.openai_runner.run_session_openai",
                new_callable=AsyncMock,
                return_value=fake_session,
            ) as mock_openai,
            patch("argus.gemini_runner.run_session_gemini", side_effect=_boom),
        ):
            await runners_module.run_system_reviewer(
                group=group,
                diff_text="diff --git a/src/app.py b/src/app.py\n@@ -1 +1 @@\n-a\n+b\n",
                settings=mock_settings,
            )

        mock_openai.assert_called_once()
        assert mock_openai.call_args.kwargs["entry"].platform == "openai-responses"
        assert mock_openai.call_args.kwargs["entry"].model == "gpt-mini"

    @pytest.mark.asyncio
    async def test_specialist_model_flag_executes_on_claude(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Setting ARGUS_SPECIALIST_MODEL forces bulk reviewers to claude-sdk with claude-default."""
        import importlib
        from datetime import datetime, timezone
        from unittest.mock import MagicMock

        from argus.llm import models as models_module
        from argus.pipeline_models import SystemGroup

        monkeypatch.setenv("ARGUS_SPECIALIST_MODEL", "claude-haiku-4-5")
        importlib.reload(models_module)
        clear_settings_cache()
        bench.clear_cache()

        group = SystemGroup(
            name="backend",
            files=["src/app.py"],
            conventions="",
            review_focus="",
        )
        mock_settings = MagicMock(CONTEXT7_API_KEY=None)

        fake_session = MagicMock()
        fake_session.result_text = ""
        fake_session.failure_reason = None
        fake_session.timed_out = False
        fake_session.cost_usd = 0.0
        fake_session.tool_call_count = 0
        fake_session.tool_names = []
        fake_session.context7_call_count = 0
        fake_session.model = "claude-haiku-4-5"
        fake_session.result_text_length = 0
        fake_session.duration_seconds = 1.0
        fake_session.started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        fake_session.finished_at = datetime(2026, 1, 1, tzinfo=timezone.utc)

        def _boom(*_args: object, **_kwargs: object) -> None:
            raise AssertionError(
                "run_session_gemini must not be called when specialist_model is set"
            )

        try:
            with (
                patch(
                    "argus.runners.fetch_prompt",
                    new_callable=AsyncMock,
                    return_value="base prompt",
                ),
                patch(
                    "argus.runners._run_session_isolated",
                    new_callable=AsyncMock,
                    return_value=fake_session,
                ) as mock_isolated,
                patch("argus.gemini_runner.run_session_gemini", side_effect=_boom),
            ):
                await runners_module.run_system_reviewer(
                    group=group,
                    diff_text="diff --git a/src/app.py b/src/app.py\n@@ -1 +1 @@\n-a\n+b\n",
                    settings=mock_settings,
                )

            mock_isolated.assert_called_once()
            assert mock_isolated.call_args.kwargs["model"] == "claude-haiku-4-5"
        finally:
            monkeypatch.delenv("ARGUS_SPECIALIST_MODEL", raising=False)
            importlib.reload(models_module)
            clear_settings_cache()
            bench.clear_cache()

    @pytest.mark.asyncio
    async def test_run_specialist_reviewer_routes_through_bench_to_same_model(self) -> None:
        from datetime import datetime, timezone
        from unittest.mock import MagicMock

        from argus.pipeline_models import SystemGroup

        group = SystemGroup(
            name="backend",
            files=["src/app.py"],
            conventions="",
            review_focus="",
        )
        mock_settings = MagicMock(CONTEXT7_API_KEY=None)

        fake_session = MagicMock(
            result_text='{"system_group": "backend::security", "findings": [], "files_explored": []}',
            failure_reason=None,
            timed_out=False,
            cost_usd=0.0,
            tool_call_count=0,
            tool_names=[],
            context7_call_count=0,
            model="gemini-3.8-flash",
            result_text_length=0,
            duration_seconds=1.0,
            started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            finished_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )

        with (
            patch(
                "argus.runners.fetch_prompt",
                new_callable=AsyncMock,
                return_value="base prompt",
            ),
            patch(
                "argus.gemini_runner.run_session_gemini",
                new_callable=AsyncMock,
                return_value=fake_session,
            ) as mock_gemini,
        ):
            await runners_module.run_specialist_reviewer(
                specialist="security",
                group=group,
                diff_text="diff --git a/src/app.py b/src/app.py\n@@ -1 +1 @@\n-a\n+b\n",
                settings=mock_settings,
            )

        mock_gemini.assert_called_once()
        assert mock_gemini.call_args.kwargs["entry"].platform == "gemini"
        assert mock_gemini.call_args.kwargs["entry"].model == "gemini-mini"

    @pytest.mark.asyncio
    async def test_run_cross_cutting_reviewer_routes_through_bench_to_same_model(self) -> None:
        from datetime import datetime, timezone
        from unittest.mock import MagicMock

        from argus.pipeline_models import FileEntry, ReviewPlan, SystemGroup

        plan = ReviewPlan(
            system_groups=[
                SystemGroup(name="b", files=["src/app.py"], conventions="", review_focus="")
            ],
            file_manifest=[FileEntry(path="src/app.py", change_type="modified")],
            cross_cutting_concerns=[],
        )
        mock_settings = MagicMock(CONTEXT7_API_KEY=None)

        fake_session = MagicMock(
            result_text="",
            failure_reason=None,
            timed_out=False,
            cost_usd=0.0,
            tool_call_count=0,
            tool_names=[],
            context7_call_count=0,
            model=runners_module._CROSS_CUTTING_MODEL,
            result_text_length=0,
            duration_seconds=1.0,
            started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            finished_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )

        with (
            patch(
                "argus.runners.fetch_prompt",
                new_callable=AsyncMock,
                return_value="base prompt",
            ),
            patch(
                "argus.runners._run_session_isolated",
                new_callable=AsyncMock,
                return_value=fake_session,
            ) as mock_isolated,
        ):
            await runners_module.run_cross_cutting_reviewer(
                plan=plan,
                diff_text="diff --git a/src/app.py b/src/app.py\n@@ -1 +1 @@\n-a\n+b\n",
                settings=mock_settings,
            )

        mock_isolated.assert_called_once()
        assert mock_isolated.call_args.kwargs["model"] == runners_module._CROSS_CUTTING_MODEL
        assert mock_isolated.call_args.kwargs["is_system_reviewer_role"] is False

    @pytest.mark.asyncio
    async def test_run_tests_and_docs_reviewer_routes_through_bench(self) -> None:
        from datetime import datetime, timezone
        from unittest.mock import MagicMock

        from argus.pipeline_models import FileEntry, ReviewPlan

        plan = ReviewPlan(
            system_groups=[],
            file_manifest=[FileEntry(path="src/app.py", change_type="modified")],
            cross_cutting_concerns=[],
        )
        mock_settings = MagicMock(CONTEXT7_API_KEY=None)

        fake_session = MagicMock(
            result_text='{"system_group": "tests-and-docs", "findings": [], "files_explored": []}',
            failure_reason=None,
            timed_out=False,
            cost_usd=0.0,
            tool_call_count=0,
            tool_names=[],
            context7_call_count=0,
            model="gemini-3.8-flash",
            result_text_length=0,
            duration_seconds=1.0,
            started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            finished_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )

        with (
            patch(
                "argus.runners.fetch_prompt",
                new_callable=AsyncMock,
                return_value="base prompt",
            ),
            patch(
                "argus.gemini_runner.run_session_gemini",
                new_callable=AsyncMock,
                return_value=fake_session,
            ) as mock_gemini,
        ):
            await runners_module.run_tests_and_docs_reviewer(
                plan=plan,
                diff_text="diff --git a/src/app.py b/src/app.py\n@@ -1 +1 @@\n-a\n+b\n",
                settings=mock_settings,
            )

        mock_gemini.assert_called_once()
        assert mock_gemini.call_args.kwargs["entry"].platform == "gemini"
        assert mock_gemini.call_args.kwargs["entry"].model == "gemini-mini"

    @pytest.mark.asyncio
    async def test_run_blocking_validator_routes_through_bench(self) -> None:
        from datetime import datetime, timezone
        from unittest.mock import MagicMock

        mock_settings = MagicMock(CONTEXT7_API_KEY=None)

        fake_session = MagicMock(
            result_text='{"items": []}',
            failure_reason=None,
            timed_out=False,
            cost_usd=0.0,
            tool_call_count=0,
            tool_names=[],
            context7_call_count=0,
            model=runners_module._SYSTEM_REVIEWER_MODEL,
            result_text_length=0,
            duration_seconds=1.0,
            started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            finished_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )

        with (
            patch(
                "argus.runners.fetch_prompt",
                new_callable=AsyncMock,
                return_value="base prompt",
            ),
            patch(
                "argus.runners._run_session_isolated",
                new_callable=AsyncMock,
                return_value=fake_session,
            ) as mock_isolated,
        ):
            await runners_module.run_blocking_validator(
                blocking_findings=[{"file": "a.py", "line": 1, "description": "d"}],
                diff_text="diff --git a/a.py b/a.py\n@@ -1 +1 @@\n-a\n+b\n",
                settings=mock_settings,
            )

        mock_isolated.assert_called_once()
        assert mock_isolated.call_args.kwargs["model"] == runners_module._SYSTEM_REVIEWER_MODEL
        assert mock_isolated.call_args.kwargs["is_system_reviewer_role"] is True

    @pytest.mark.asyncio
    async def test_run_feedback_verifier_routes_through_bench(self) -> None:
        from datetime import datetime, timezone
        from unittest.mock import MagicMock

        from argus.pipeline_models import PriorFinding, PriorReviewContext

        prior_context = PriorReviewContext(
            review_id="00000000-0000-0000-0000-000000000000",
            reviewed_sha="a" * 40,
            findings=[PriorFinding(severity="BLOCKING", description="d")],
        )
        mock_settings = MagicMock(CONTEXT7_API_KEY=None)

        fake_session = MagicMock(
            result_text='{"items": []}',
            failure_reason=None,
            timed_out=False,
            cost_usd=0.0,
            tool_call_count=0,
            tool_names=[],
            context7_call_count=0,
            model=runners_module._SYSTEM_REVIEWER_MODEL,
            result_text_length=0,
            duration_seconds=1.0,
            started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            finished_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )

        with (
            patch(
                "argus.runners.fetch_prompt",
                new_callable=AsyncMock,
                return_value="base prompt",
            ),
            patch(
                "argus.runners._run_session_isolated",
                new_callable=AsyncMock,
                return_value=fake_session,
            ) as mock_isolated,
        ):
            await runners_module.run_feedback_verifier(
                prior_context=prior_context,
                diff_text="diff --git a/a.py b/a.py\n@@ -1 +1 @@\n-a\n+b\n",
                settings=mock_settings,
            )

        mock_isolated.assert_called_once()
        assert mock_isolated.call_args.kwargs["model"] == runners_module._SYSTEM_REVIEWER_MODEL
        assert mock_isolated.call_args.kwargs["is_system_reviewer_role"] is True


# ---------------------------------------------------------------------------
# Claude-configured roles: must never reach Gemini runner
# ---------------------------------------------------------------------------


class TestClaudeRolesNeverTouchGeminiRunner:
    """Roles configured with platform = "claude-sdk" (such as the default
    cross-cutting, blocking-validator, and feedback-verifier roles, or a bulk
    reviewer overridden to claude-sdk) must never invoke argus.gemini_runner.
    Proven here by making run_session_gemini itself explode if reached, then
    running the bench path and confirming it completes normally anyway.
    """

    @pytest.mark.asyncio
    async def test_cross_cutting_default_bench_never_invokes_gemini_runner(self) -> None:
        from datetime import datetime, timezone
        from unittest.mock import MagicMock

        from argus.pipeline_models import FileEntry, ReviewPlan

        # Sanity precondition: cross-cutting really defaults to claude-sdk.
        assert bench.resolve("cross-cutting").platform == "claude-sdk"

        plan = ReviewPlan(
            system_groups=[],
            file_manifest=[FileEntry(path="src/app.py", change_type="modified")],
            cross_cutting_concerns=[],
        )
        mock_settings = MagicMock(CONTEXT7_API_KEY=None)

        fake_session = MagicMock()
        fake_session.result_text = ""
        fake_session.failure_reason = None
        fake_session.timed_out = False
        fake_session.cost_usd = 0.0
        fake_session.tool_call_count = 0
        fake_session.tool_names = []
        fake_session.context7_call_count = 0
        fake_session.model = runners_module._CROSS_CUTTING_MODEL
        fake_session.duration_seconds = 1.0
        fake_session.started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        fake_session.finished_at = datetime(2026, 1, 1, tzinfo=timezone.utc)

        def _boom(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("run_session_gemini must never be invoked for cross-cutting")

        with (
            patch(
                "argus.runners.fetch_prompt",
                new_callable=AsyncMock,
                return_value="base prompt",
            ),
            patch(
                "argus.runners._run_session_isolated",
                new_callable=AsyncMock,
                return_value=fake_session,
            ),
            patch("argus.gemini_runner.run_session_gemini", side_effect=_boom),
        ):
            await runners_module.run_cross_cutting_reviewer(
                plan=plan,
                diff_text="diff --git a/src/app.py b/src/app.py\n@@ -1 +1 @@\n-a\n+b\n",
                settings=mock_settings,
            )

    @pytest.mark.asyncio
    async def test_claude_override_never_invokes_gemini_runner(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from datetime import datetime, timezone
        from unittest.mock import MagicMock

        from argus.pipeline_models import SystemGroup

        explicit = tmp_path / "explicit-bench.toml"
        _write_toml(
            explicit,
            '[bulk_reviewer]\nplatform = "claude-sdk"\nmodel = "claude-default"\n',
        )
        monkeypatch.setenv("ARGUS_BENCH_FILE", str(explicit))
        clear_settings_cache()
        bench.clear_cache()

        assert bench.resolve("system-generalist").platform == "claude-sdk"

        group = SystemGroup(
            name="backend",
            files=["src/app.py"],
            conventions="",
            review_focus="",
        )
        mock_settings = MagicMock(CONTEXT7_API_KEY=None)

        fake_session = MagicMock()
        fake_session.result_text = ""
        fake_session.failure_reason = None
        fake_session.timed_out = False
        fake_session.cost_usd = 0.0
        fake_session.tool_call_count = 0
        fake_session.tool_names = []
        fake_session.context7_call_count = 0
        fake_session.model = runners_module._SYSTEM_REVIEWER_MODEL
        fake_session.duration_seconds = 1.0
        fake_session.started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        fake_session.finished_at = datetime(2026, 1, 1, tzinfo=timezone.utc)

        def _boom(*_args: object, **_kwargs: object) -> None:
            raise AssertionError(
                "run_session_gemini must never be invoked when bulk_reviewer is overridden to claude-sdk"
            )

        with (
            patch(
                "argus.runners.fetch_prompt",
                new_callable=AsyncMock,
                return_value="base prompt",
            ),
            patch(
                "argus.runners._run_session_isolated",
                new_callable=AsyncMock,
                return_value=fake_session,
            ),
            patch("argus.gemini_runner.run_session_gemini", side_effect=_boom),
        ):
            await runners_module.run_system_reviewer(
                group=group,
                diff_text="diff --git a/src/app.py b/src/app.py\n@@ -1 +1 @@\n-a\n+b\n",
                settings=mock_settings,
            )
