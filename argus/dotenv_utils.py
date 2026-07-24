"""Helpers for loading local dotenv files safely and consistently.

Why this exists:
- Many entrypoints (scripts, listeners, local servers) need a `.env.local`
  to be loaded *before* importing settings/clients so credentials are present.
- Hardcoding relative paths like `Path(__file__).parent.parent...` is brittle in
  a monorepo; this helper finds the nearest env file by walking ancestors.
"""

from __future__ import annotations

import os
from pathlib import Path


def _preferred_env_filename() -> str:
    """Pick the default env filename based on ENVIRONMENT."""
    env = (os.environ.get("ENVIRONMENT") or "").lower()
    if env == "production":
        return ".env.production"
    if env == "test":
        return ".env.test"
    return ".env.local"


def find_env_file(*, start: Path | None = None, env_filename: str | None = None) -> Path | None:
    """Find an env file by walking from `start` up through parent directories.

    Resolution order:
    1) If ENV_FILE is set, try that path first (absolute or relative to CWD).
    2) Otherwise search for `env_filename` (or environment-derived default)
       in `start` and its ancestors.
    """
    env_file_override = os.environ.get("ENV_FILE")
    if env_file_override:
        override_path = Path(env_file_override).expanduser()
        if not override_path.is_absolute():
            override_path = (Path.cwd() / override_path).resolve()
        return override_path if override_path.exists() else None

    filename = env_filename or _preferred_env_filename()

    start_path = start.resolve() if start is not None else Path.cwd().resolve()
    start_dir = start_path if start_path.is_dir() else start_path.parent

    for candidate_dir in (start_dir, *start_dir.parents):
        candidate = candidate_dir / filename
        if candidate.exists():
            return candidate

    return None


def load_dotenv_early(
    *,
    start: Path | None = None,
    env_filename: str | None = None,
    override: bool = False,
    skip_in_test: bool = False,
) -> Path | None:
    """Load dotenv file early (before importing settings/clients).

    This is a no-op when:
    - running in AWS Lambda (secrets are expected to come from SSM/env), or
    - running in pytest and `skip_in_test=True`, or
    - python-dotenv isn't installed, or
    - no env file can be found.
    """
    # Never load local dotenv inside Lambda.
    if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        return None

    if skip_in_test and (
        os.environ.get("ENVIRONMENT") == "test" or os.getenv("PYTEST_CURRENT_TEST")
    ):
        return None

    try:
        from dotenv import load_dotenv
    except ImportError:
        return None

    env_path = find_env_file(start=start, env_filename=env_filename)
    if env_path is None:
        return None

    load_dotenv(env_path, override=override)
    return env_path
