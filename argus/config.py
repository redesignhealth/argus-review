"""Configuration for the standalone Argus reviewer.

A slim ``pydantic-settings`` implementation. All configuration comes from
environment variables (or a local ``.env`` file loaded via
``argus.dotenv_utils``) — no AWS, no SSM.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

# Single source of truth for the reviewer-subprocess wall-clock budget.
# ``argus.runners._SUBPROCESS_TIMEOUT_S`` (the fallback used when no Settings
# instance is available, mostly tests) imports this same constant rather than
# hardcoding its own copy, so the two can never drift out of sync.
DEFAULT_ARGUS_SESSION_TIMEOUT_S = 600


class Settings(BaseSettings):
    """Application settings for the Argus reviewer.

    Required:
        ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN): Claude Agent SDK +
            LangChain calls. Exactly the same dual-credential convention
            Anthropic's own SDK and the Claude Code CLI support natively:
            ``ANTHROPIC_API_KEY`` is sent as ``x-api-key`` (a real Anthropic
            API key); ``ANTHROPIC_AUTH_TOKEN`` is sent as
            ``Authorization: Bearer`` (the standard mechanism for routing
            through a corporate LLM gateway/proxy, where the credential
            isn't a real Anthropic key). Provide at least one; if both are
            set, ``ANTHROPIC_API_KEY`` wins.
        GITHUB_TOKEN_RO: PR diff fetch + repo clone (read-only PAT).
        OPENAI_API_KEY: Plan-extraction fallback path.

    Optional:
        ARGUS_DB_URL / SUPABASE_DB_URL: Postgres history + checkpoints.
            ``SUPABASE_DB_URL`` is a back-compat alias; if both are set,
            ``ARGUS_DB_URL`` wins.
        ARGUS_STORAGE_READ_URL / ARGUS_STORAGE_WRITE_URL / ARGUS_STORAGE_AUTH:
            HTTP storage-shim mode.
        ARGUS_SQLITE_CHECKPOINT_PATH: Pin the LangGraph SQLite checkpoint file.
        ARGUS_HISTORY_DB_PATH: Override the local SQLite round-history file
            (``argus.storage.sqlite.SqliteHistoryBackend``). Only consulted
            when neither a Postgres URL nor the HTTP storage pair is
            configured. Defaults to ``~/.local/share/argus/history.db``.
        ARGUS_PROMPTS_DIR: Highest-priority prompt override directory. See
            ``argus.prompts_runtime`` for the full override search chain.
        ARGUS_NO_PROMPT_OVERRIDES: Set truthy to ignore every override
            directory (including ``ARGUS_PROMPTS_DIR``) and force packaged
            prompts only. For CI/official runs that must not pick up a
            developer's local override by accident.
        ARGUS_RULES_DIR: Override directory for custom deterministic-precheck
            rules (semgrep YAML). Same "explicit override always wins" idea
            as ``ARGUS_PROMPTS_DIR``, but simpler: a single directory whose
            ``*.yml``/``*.yaml`` files replace the packaged (empty-by-default)
            rules directory wholesale, rather than a multi-location search
            chain merged by filename -- rules aren't looked up by a fixed
            name the way prompts are, so there's nothing to merge. Only
            consulted when the ``prechecks`` extra is installed; see
            ``argus.precheck``.
        ARGUS_STOCK_SEMGREP_PACKS: Comma-separated semgrep registry pack IDs
            (e.g. ``"p/secrets"``) to run alongside (or instead of) a custom
            ``ARGUS_RULES_DIR`` — unlike a local rules directory, each pack
            is fetched over the network by semgrep itself on first use
            (cached locally after). Unset by default: this is an opt-in
            addition of vetted, community-maintained rules, not a silent
            default, since it adds a network dependency to the live gate.
            See ``docs/PRECHECKS.md``'s "stock rule sources" section.
        ARGUS_PRECHECK_BLOCK_ON_SCANNER_FAILURE: Set truthy to force the
            verdict to BLOCKING whenever a deterministic precheck scanner
            crashed/timed out/errored this round (``PrecheckResult.
            failed_scanners`` non-empty) instead of the default fail-open
            behavior (surface it in the review comment's degraded-coverage
            section — see ``argus.helpers.build_degraded_coverage_labels``
            — but let the review's own verdict stand on its own merits).
            False by default: every other part of this module's design is
            deliberately fail-open (a broken scanner should never be the
            reason a review can't complete), and this flag exists for
            callers who've decided the opposite tradeoff is worth it for
            their repo -- e.g. one whose CI treats a specific scanner as a
            hard release gate, where "the gate silently didn't run" is a
            worse failure mode than "the review is blocked until it's
            fixed." Deliberately does NOT downgrade an existing APPROVE
            with no failures, and never *un-blocks* a review that would
            have been BLOCKING anyway -- it only ever makes the verdict
            stricter, never looser.
        ARGUS_REPO_CACHE: Mirror-clone cache directory.
        ARGUS_SPECIALIST_MODEL: Override the model used by the system
            reviewer, specialist reviewers, the writer, and the lite-review
            path (``argus.llm.models.CLAUDE_DEFAULT``, default
            ``claude-sonnet-4-6``). Set via ``--specialist-model``; read
            directly from ``os.environ`` by ``argus.llm.models`` at import
            time (module-level constant resolution, not a per-request
            Settings lookup), so this field is never actually read off a
            ``Settings`` instance anywhere in the codebase -- unlike
            ``ARGUS_NO_PROMPT_OVERRIDES`` above, which genuinely is
            (``prompts_runtime.override_dirs`` reads
            ``settings.ARGUS_NO_PROMPT_OVERRIDES``). It's declared here
            purely for documentation/discoverability, same as any other
            env var this class's docstring covers.
        ARGUS_FRONTIER_MODEL: Override the model used by the planner,
            coverage check, and cross-cutting reviewer
            (``argus.llm.models.CLAUDE_FRONTIER`` / ``CLAUDE_OPUS``). Set via
            ``--frontier-model``; same read pattern (and same
            documentation-only Settings field) as ``ARGUS_SPECIALIST_MODEL``
            above.
        LANGSMITH_API_KEY / LANGSMITH_PROJECT: Optional tracing.
        CONTEXT7_API_KEY / ARGUS_CONTEXT7_LIBRARY_ID: Context7 docs MCP.
        ARGUS_CONTEXT7_BASE_URL: Override for Context7's MCP endpoint
            (defaults to the real ``https://mcp.context7.com/mcp`` when
            unset). Needed by callers that proxy the Context7 key through an
            intermediary (e.g. the argus-review-loop skill's rh-mcp
            credential proxy, TECH-4736) rather than passing the real key
            directly -- pointing at the real host with a proxy-issued
            credential the real host doesn't recognize would fail every
            Context7 call.
        ARGUS_SESSION_TIMEOUT: Wall-clock seconds a reviewer subprocess is
            allowed to run before it is killed and reported as a failure.
            Defaults to 600 (10 minutes) — first raised from 300 to 420 after
            production logs showed legitimate (non-runaway) specialist
            reviewers finishing as late as 294s, right at the old timeout's
            edge; raised again to 600 to match rh-data-platform's
            production-proven value ahead of this package taking over as the
            actual production reviewer (rh-data-platform's review_service is
            being retired in its favor).
    """

    ANTHROPIC_API_KEY: str | None = None
    ANTHROPIC_AUTH_TOKEN: str | None = None
    GITHUB_TOKEN_RO: str
    OPENAI_API_KEY: str

    ARGUS_DB_URL: str | None = None
    SUPABASE_DB_URL: str | None = None

    ARGUS_STORAGE_READ_URL: str | None = None
    ARGUS_STORAGE_WRITE_URL: str | None = None
    ARGUS_STORAGE_AUTH: str | None = None

    ARGUS_SQLITE_CHECKPOINT_PATH: str | None = None
    ARGUS_HISTORY_DB_PATH: str | None = None
    ARGUS_PROMPTS_DIR: str | None = None
    ARGUS_NO_PROMPT_OVERRIDES: bool = False
    ARGUS_RULES_DIR: str | None = None
    ARGUS_STOCK_SEMGREP_PACKS: str | None = None
    ARGUS_PRECHECK_BLOCK_ON_SCANNER_FAILURE: bool = False
    ARGUS_REPO_CACHE: str | None = None
    ARGUS_SPECIALIST_MODEL: str | None = None
    ARGUS_FRONTIER_MODEL: str | None = None

    LANGSMITH_API_KEY: str | None = None
    LANGSMITH_PROJECT: str | None = None

    CONTEXT7_API_KEY: str | None = None
    ARGUS_CONTEXT7_LIBRARY_ID: str | None = None
    ARGUS_CONTEXT7_BASE_URL: str | None = None

    ARGUS_SESSION_TIMEOUT: int = DEFAULT_ARGUS_SESSION_TIMEOUT_S

    model_config = SettingsConfigDict(
        case_sensitive=True,
        extra="ignore",
        env_file=(".env",),
        env_file_encoding="utf-8",
    )

    @property
    def db_url(self) -> str | None:
        """Resolved Postgres URL: ``ARGUS_DB_URL`` wins over ``SUPABASE_DB_URL``."""
        return self.ARGUS_DB_URL or self.SUPABASE_DB_URL

    @property
    def anthropic_credential(self) -> tuple[str, str]:
        """The configured Anthropic credential as ``(env_var_name, value)``.

        Callers that need to hand this credential to something that itself
        reads an env var (the spawned ``claude`` CLI subprocess) must set
        the SAME variable name the caller configured — forcing everything
        to ``ANTHROPIC_API_KEY`` would send a proxy/gateway bearer token as
        an ``x-api-key``, which not every gateway accepts. ``ANTHROPIC_API_KEY``
        wins when both are set, matching the Anthropic SDK's own precedence.

        Raises ``ValueError`` if neither is set -- ``Settings`` construction
        deliberately allows this (enforcement is ``cli._check_settings``'s
        job, for a clean CLI error instead of a raw pydantic traceback), so
        this property is the backstop for any caller that reaches here
        without having gone through that check first.
        """
        if self.ANTHROPIC_API_KEY:
            return ("ANTHROPIC_API_KEY", self.ANTHROPIC_API_KEY)
        if self.ANTHROPIC_AUTH_TOKEN:
            return ("ANTHROPIC_AUTH_TOKEN", self.ANTHROPIC_AUTH_TOKEN)
        raise ValueError("One of ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN is required")


@lru_cache(maxsize=1)
def _cached_settings() -> Settings:
    return Settings()


def get_settings() -> Settings:
    """Get the process-wide Settings singleton.

    Cached for the lifetime of the process; call :func:`clear_cache` to
    force a reload (e.g. in tests, or after mutating ``os.environ``).
    """
    return _cached_settings()


def clear_cache() -> None:
    """Clear the cached Settings singleton, forcing the next call to reload."""
    _cached_settings.cache_clear()
