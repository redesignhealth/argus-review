"""Configuration for the standalone Argus reviewer.

A slim ``pydantic-settings`` implementation. All configuration comes from
environment variables (or a local ``.env`` file loaded via
``argus.dotenv_utils``) — no AWS, no SSM.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings for the Argus reviewer.

    Required:
        ANTHROPIC_API_KEY: Claude Agent SDK + LangChain calls.
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
        ARGUS_REPO_CACHE: Mirror-clone cache directory.
        LANGSMITH_API_KEY / LANGSMITH_PROJECT: Optional tracing.
        CONTEXT7_API_KEY / ARGUS_CONTEXT7_LIBRARY_ID: Context7 docs MCP.
        ARGUS_SESSION_TIMEOUT: Wall-clock seconds a reviewer subprocess is
            allowed to run before it is killed and reported as TIMED_OUT.
            Defaults to 300 (5 minutes).
    """

    ANTHROPIC_API_KEY: str
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
    ARGUS_REPO_CACHE: str | None = None

    LANGSMITH_API_KEY: str | None = None
    LANGSMITH_PROJECT: str | None = None

    CONTEXT7_API_KEY: str | None = None
    ARGUS_CONTEXT7_LIBRARY_ID: str | None = None

    ARGUS_SESSION_TIMEOUT: int = 300

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
