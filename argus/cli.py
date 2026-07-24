#!/usr/bin/env python3
"""Local runner for the Argus v3 review agent.

Configuration is env-var / ``.env`` only — no AWS, no SSM. If required
secrets are missing after the load, the runner exits with a clear error.

Usage:
    # Review a PR (positional repo, or --repo — mutually exclusive, not both)
    uv run argus review owner/repo --pr 3124
    uv run argus review --repo owner/repo --pr 3124

    # Review a specific SHA against main
    uv run argus review owner/repo --sha abc123 --base-ref main

    # Write review comment to a file instead of just printing
    uv run argus review owner/repo --pr 3124 -o review.md

    # Also post/update the review as a PR comment, and set a commit status
    uv run argus review owner/repo --pr 3124 --post --commit-status

    # Dismiss a finding before running the next round
    uv run argus review owner/repo --pr 3124 --dismiss "B2 -- pre-existing, not from this PR"

    # List / export the packaged prompts for customization
    uv run argus prompts list
    uv run argus prompts export ./my-prompts

    # Also runnable as a flat legacy invocation (no subcommand) or via -m:
    uv run python -m argus.cli --repo owner/repo --pr 3124

Environment variables (HTTP-mode opt-in):
    ARGUS_STORAGE_READ_URL / ARGUS_STORAGE_WRITE_URL / ARGUS_STORAGE_AUTH
        Defaults for the corresponding ``--storage-*`` flags. Both
        READ_URL and WRITE_URL must be set together (AUTH is optional)
        to arm the HTTP shim and let the runner work inside sandboxes
        that can't reach Postgres on 5432.

    ARGUS_SQLITE_CHECKPOINT_PATH
        Applies whenever no Postgres URL is configured (the SQLite
        checkpointer is the default; ``ARGUS_DB_URL`` opts into
        Postgres). When unset (default), the runner writes the
        LangGraph checkpoint to ``/tmp/argus-checkpoint-<PID>.db`` and
        unlinks it on exit. Set to a fixed path to preserve the
        checkpoint across runs (e.g. for post-mortem inspection of a
        failed pipeline). The runner will not delete an operator-pinned
        path.

No storage env vars are required: with neither ``ARGUS_DB_URL`` nor the
HTTP-shim URLs set, round history and checkpoints default to local SQLite.
Required secrets are only ANTHROPIC_API_KEY, GITHUB_TOKEN_RO,
and OPENAI_API_KEY.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argus.config import Settings
    from argus.models import ReviewRequest, ReviewResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("argus_review_local")

# Hard wall-clock backstop: if the entire review has not completed within this
# many seconds, the watchdog thread force-exits the process via os._exit. This
# is the only reliable cap locally, because a deadlocked event loop (e.g. the
# macOS child-watcher misreaping subprocess grandchildren) cannot be interrupted
# by asyncio-level timeouts. 60 minutes is well above a healthy review (8-20 min).
_WATCHDOG_TIMEOUT_S = 3600


def _sweep_stale_argus_tempdirs() -> None:
    """Best-effort removal of Argus temp artifacts the forced exit would orphan."""
    import glob
    import shutil
    import tempfile

    tmp = tempfile.gettempdir()
    for path in glob.glob(os.path.join(tmp, "argus-worktree-*")):
        with contextlib.suppress(Exception):
            shutil.rmtree(path, ignore_errors=True)
    for path in glob.glob(os.path.join(tmp, "argus-gitcfg-*")):
        with contextlib.suppress(Exception):
            os.unlink(path)


def _start_watchdog(repo: str, pr: object) -> threading.Thread:
    """Start the hard wall-clock backstop as a daemon thread.

    If the review has not finished within ``_WATCHDOG_TIMEOUT_S``, the thread
    sweeps stale Argus temp artifacts (best-effort) and force-exits via
    ``os._exit`` - the only reliable cap when a deadlocked event loop cannot be
    interrupted by asyncio-level timeouts. ``repo`` / ``pr`` are captured so the
    log line names WHAT timed out.
    """

    def _watchdog() -> None:
        time.sleep(_WATCHDOG_TIMEOUT_S)
        logger.critical(
            "Watchdog: review of %s PR #%s exceeded hard timeout of %ds; force-exiting",
            repo,
            pr,
            _WATCHDOG_TIMEOUT_S,
        )
        try:
            _sweep_stale_argus_tempdirs()
        except Exception:
            pass
        os._exit(1)

    thread = threading.Thread(target=_watchdog, daemon=True, name="argus-watchdog")
    thread.start()
    return thread


def _load_settings() -> "Settings":
    """Load settings from .env / shell.

    Clears the cached Settings singleton first so a fresh process env
    (e.g. a `.env` loaded moments ago) is picked up.
    """
    from argus.dotenv_utils import load_dotenv_early

    # Pin to '.env' explicitly — the docs reference '.env', and the default
    # in load_dotenv_early would otherwise pick '.env.local' (or
    # '.env.production') based on ENVIRONMENT, leading to silent no-op loads.
    env_path = load_dotenv_early(start=Path.cwd(), env_filename=".env")
    if env_path:
        logger.info("Loaded local env vars from %s", env_path)

    from argus.config import clear_cache, get_settings

    # Clear the settings cache so it picks up any just-loaded .env values.
    clear_cache()

    settings = get_settings()

    # Propagate any loaded keys back into os.environ so child code that reads
    # them directly picks up the right values.
    for attr in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GITHUB_TOKEN_RO",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
        "CONTEXT7_API_KEY",
    ):
        val = getattr(settings, attr, None)
        if val:
            os.environ[attr] = val
    if settings.db_url:
        os.environ.setdefault("SUPABASE_DB_URL", settings.db_url)

    return settings


def _check_settings(settings: "Settings") -> None:
    """Validate that critical secrets were loaded, fail loudly otherwise.

    Only the three API credentials are required: ``ANTHROPIC_API_KEY``
    (Agent SDK + LangChain), ``GITHUB_TOKEN_RO`` (diff fetch + clone), and
    ``OPENAI_API_KEY`` (plan-extraction path). No storage configuration is
    required — with neither ``ARGUS_DB_URL`` nor the HTTP-shim URLs set,
    history and checkpoints default to local SQLite.

    Args:
        settings: Resolved settings object.
    """
    missing = []
    if not settings.ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")
    if not settings.GITHUB_TOKEN_RO:
        missing.append("GITHUB_TOKEN_RO")
    if not settings.OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if missing:
        logger.error(
            "Missing required secrets: %s. Add them to your shell or .env file.",
            ", ".join(missing),
        )
        sys.exit(1)


async def run(request: "ReviewRequest") -> "ReviewResponse":
    """Run the review pipeline (storage backend resolved automatically:
    Postgres if ``ARGUS_DB_URL`` is set, HTTP shim if the storage URLs are
    set, else local SQLite)."""
    from argus.graph import run_review

    return await run_review(request, flow_run_id=None)


def _add_review_args(parser: argparse.ArgumentParser) -> None:
    repo_group = parser.add_mutually_exclusive_group()
    repo_group.add_argument(
        "repo_positional",
        nargs="?",
        default=None,
        metavar="repo",
        help="GitHub repo (owner/repo), positional form: 'argus review owner/repo --pr N'",
    )
    repo_group.add_argument(
        "--repo",
        dest="repo_flag",
        default=None,
        help="GitHub repo (owner/repo). Mutually exclusive with the positional form.",
    )
    parser.add_argument("--pr", type=int, default=0, help="PR number to review")
    parser.add_argument("--sha", default=None, help="Specific commit SHA to review")
    parser.add_argument("--base-ref", default=None, help="Base ref for SHA mode (e.g. 'main')")
    parser.add_argument(
        "--dismiss",
        action="append",
        default=[],
        help='Dismiss a prior finding: --dismiss "B2 -- pre-existing, not from this PR"',
    )
    parser.add_argument(
        "--storage-read-url",
        default=os.environ.get("ARGUS_STORAGE_READ_URL"),
        help=(
            "URL template for prior-rounds GET. When set (along "
            "with --storage-write-url), Argus's storage I/O routes through "
            "your own HTTP backend over HTTPS instead of direct Postgres — for "
            "in-sandbox Argus runs that can't reach Postgres on port 5432. "
            "Template supports {owner}, {repo}, {pr}."
        ),
    )
    parser.add_argument(
        "--storage-write-url",
        default=os.environ.get("ARGUS_STORAGE_WRITE_URL"),
        help="URL template for new-round POST. See --storage-read-url.",
    )
    parser.add_argument(
        "--storage-auth",
        default=os.environ.get("ARGUS_STORAGE_AUTH"),
        help=(
            "API key value for the ``X-API-Key`` header on storage requests "
            "your HTTP backend can validate. Optional."
        ),
    )
    parser.add_argument("-o", "--output", default=None, help="Write review markdown to file")
    parser.add_argument(
        "--post",
        action="store_true",
        help=(
            "Upsert the finished review as a PR comment (updates Argus's prior "
            "comment on re-run instead of stacking a new one). Requires --pr. "
            "Uses GITHUB_TOKEN if set, else falls back to GITHUB_TOKEN_RO "
            "(which may lack write scope)."
        ),
    )
    parser.add_argument(
        "--commit-status",
        action="store_true",
        dest="commit_status",
        help=(
            "Set a commit status on the head SHA (context 'argus/review', "
            "success iff verdict is APPROVE). Same token rules as --post."
        ),
    )
    parser.add_argument(
        "--no-prompt-overrides",
        action="store_true",
        dest="no_prompt_overrides",
        help=(
            "Ignore every prompt override directory (ARGUS_PROMPTS_DIR, "
            "./.argus/prompts/, ~/.config/argus/prompts/) and force the "
            "packaged prompts only. For CI/official runs that must not pick "
            "up a developer's local override by accident."
        ),
    )


def _package_version() -> str:
    """Resolve the installed ``argus-code-review`` package version.

    Falls back to ``"unknown"`` rather than raising — ``--version`` should
    never be the thing that crashes, even in an unusual install (e.g. a
    partially-built editable checkout with no dist-info yet). Called
    eagerly from ``_build_parser()`` (the ``version=`` kwarg is an f-string,
    evaluated at ``add_argument()`` time), so this dist-info lookup runs on
    every ``argus`` invocation, not only when ``--version`` is passed.
    """
    import importlib.metadata

    try:
        return importlib.metadata.version("argus-code-review")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="argus",
        description="Run the Argus v3 review agent locally (no external orchestrator/CI needed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_package_version()}",
    )
    subparsers = parser.add_subparsers(dest="command")
    review_parser = subparsers.add_parser(
        "review",
        help="Run a PR or SHA review",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_review_args(review_parser)

    prompts_parser = subparsers.add_parser(
        "prompts",
        help="Inspect or export the packaged prompt files",
    )
    prompts_subparsers = prompts_parser.add_subparsers(dest="prompts_command")
    prompts_subparsers.add_parser(
        "list",
        help="List prompt names and whether each resolves from the packaged "
        "files or an ARGUS_PROMPTS_DIR override",
    )
    export_parser = prompts_subparsers.add_parser(
        "export",
        help="Copy the packaged prompt files into a directory for customization",
    )
    export_parser.add_argument("dir", help="Target directory")
    export_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite files if the target directory is non-empty",
    )

    return parser


def _resolve_repo(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str:
    """Resolve the target repo from the positional or --repo form.

    The mutually-exclusive group in ``_add_review_args`` already rejects
    supplying both; this only needs to reject supplying neither.
    """
    repo: str | None = args.repo_positional or args.repo_flag
    if not repo:
        parser.error(
            "Must provide repo as a positional argument (argus review owner/repo) or via --repo"
        )
    assert repo is not None  # parser.error() above always raises SystemExit
    return repo


def _validate_review_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Cross-flag validation that argparse's own declarative options can't express."""
    if args.post and args.sha and not args.pr:
        parser.error("--post requires --pr (a PR to comment on); --sha alone has nowhere to post")


def _check_prerequisites() -> None:
    """Verify runtime prerequisites are on PATH, before any network call.

    ``git`` is needed for repo provisioning and the ``claude`` CLI is spawned
    as a subprocess by the Agent SDK — both must be resolvable on PATH before
    Argus burns any API calls. Missing settings are checked separately by
    ``_check_settings`` (which also fails loudly, after settings load).
    """
    import shutil

    missing = [name for name in ("git", "claude") if shutil.which(name) is None]
    if not missing:
        return
    for name in missing:
        if name == "git":
            logger.error(
                "`git` was not found on PATH. Install git "
                "(https://git-scm.com/downloads) and ensure it's on PATH."
            )
        else:
            logger.error(
                "`claude` CLI was not found on PATH. Install the Claude Code "
                "CLI (https://docs.claude.com/en/docs/claude-code) — the Agent "
                "SDK spawns it as a subprocess to run the review."
            )
    sys.exit(1)


def render_summary_block(response: "ReviewResponse", elapsed: float) -> str:
    """Render the frozen stderr summary block (Round/Verdict/Risk/Findings/Cost/Elapsed).

    This is a **frozen public contract**: the
    ``argus-review-loop`` skill screen-parses this exact block. Do not change
    the field order, labels, or spacing without updating the skill in lockstep.
    """
    blocking = sum(1 for f in response.findings if f.severity.value == "BLOCKING")
    suggestion = sum(1 for f in response.findings if f.severity.value == "SUGGESTION")
    round_label = (
        f"{response.review_round} (Lite Mode)" if response.lite_mode else str(response.review_round)
    )
    lines = [
        "=" * 60,
        f"Round:      {round_label}",
        f"Verdict:    {response.verdict.value}",
        f"Risk:       {response.risk_level.value}",
        f"Findings:   {blocking} blocking, {suggestion} suggestions",
        f"Cost:       ${response.usage.cost_usd:.2f}",
        f"Elapsed:    {elapsed:.0f}s",
        "=" * 60,
    ]
    return "\n".join(lines)


def render_review_output(response: "ReviewResponse", elapsed: float) -> str:
    """Render the full stdout block: review comment + summary.

    Byte-identical to the pre-refactor sequence of ``print()`` calls (see the
    output-contract freeze note above); kept as a pure function so it can be
    golden-file tested without driving the whole pipeline.
    """
    return "\n\n" + response.review_comment + "\n\n" + render_summary_block(response, elapsed)


def _post_review(repo: str, args: argparse.Namespace, response: "ReviewResponse") -> None:
    """Handle ``--post`` / ``--commit-status``: upsert PR comment and/or commit status."""
    from argus.github_client import GitHubClient, GitHubClientError

    try:
        client = GitHubClient.for_writes()
    except GitHubClientError as exc:
        logger.error("Cannot post to GitHub: %s", exc)
        sys.exit(1)

    if args.post:
        if not args.pr:
            logger.error("--post requires --pr; nothing to comment on")
            sys.exit(1)
        try:
            client.upsert_pr_comment(repo, args.pr, response.review_comment)
        except Exception as exc:  # noqa: BLE001 — surface any GitHub API error clearly
            logger.error("Failed to post PR comment: %s", exc)
            sys.exit(1)

    if args.commit_status:
        sha = args.sha
        if not sha and args.pr:
            pr = client.get_pull_request(repo, args.pr)
            sha = pr["head_sha"]
        if not sha:
            logger.error("--commit-status could not resolve a head SHA (need --pr or --sha)")
            sys.exit(1)
        state = "success" if response.verdict.value == "APPROVE" else "failure"
        description = f"Argus: {response.verdict.value} ({response.risk_level.value} risk)"
        try:
            client.set_commit_status(repo, sha, state, description)
        except Exception as exc:  # noqa: BLE001 — surface any GitHub API error clearly
            logger.error("Failed to set commit status: %s", exc)
            sys.exit(1)


def _run_review(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    repo = _resolve_repo(parser, args)

    http_mode = bool(args.storage_read_url and args.storage_write_url)
    if bool(args.storage_read_url) != bool(args.storage_write_url):
        sys.exit(
            "--storage-read-url and --storage-write-url must be set together "
            "(or neither; storage then uses ARGUS_DB_URL if set, else local SQLite)"
        )
    if http_mode:
        from argus.storage.http import install_http_storage

        install_http_storage(
            read_url=args.storage_read_url,
            write_url=args.storage_write_url,
            auth=args.storage_auth,
        )

    if not args.pr and not args.sha:
        sys.exit("Must provide either --pr or --sha")
    if args.pr and args.sha:
        sys.exit("Provide either --pr or --sha, not both")

    if args.no_prompt_overrides:
        os.environ["ARGUS_NO_PROMPT_OVERRIDES"] = "1"

    _check_prerequisites()

    try:
        settings = _load_settings()
    except Exception as exc:  # pydantic ValidationError for missing required vars
        logger.error("Failed to load settings: %s", exc)
        sys.exit(1)
    _check_settings(settings)

    from argus.models import ReviewRequest

    request = ReviewRequest(
        repo=repo,
        pr_number=args.pr,
        sha=args.sha,
        base_ref=args.base_ref,
        dismissals=args.dismiss,
    )

    target = f"SHA {args.sha[:12]}" if args.sha else f"PR #{args.pr}"
    logger.info("Starting local review: %s %s", repo, target)

    _start_watchdog(repo, args.pr)

    start = time.monotonic()
    response = asyncio.run(run(request))

    elapsed = time.monotonic() - start

    print(render_review_output(response, elapsed))

    # Write to file if requested
    if args.output:
        Path(args.output).write_text(response.review_comment, encoding="utf-8")
        json_path = str(Path(args.output).with_suffix(".json"))
        Path(json_path).write_text(
            json.dumps(response.model_dump(), indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("Written to %s and %s", args.output, json_path)

    if args.post or args.commit_status:
        _post_review(repo, args, response)


def _packaged_prompts_dir() -> Path:
    """Directory containing the packaged (verbatim) prompt ``.md`` files."""
    import importlib.resources

    return Path(str(importlib.resources.files("argus").joinpath("prompts")))


def _packaged_prompt_names() -> list[str]:
    return sorted(p.stem for p in _packaged_prompts_dir().glob("*.md"))


# The exact case-insensitive string set pydantic-settings v2 accepts for a
# ``bool`` field (verified against ``ARGUS_NO_PROMPT_OVERRIDES: bool`` in
# argus.config.Settings). Kept in sync here so `argus prompts list`/`export`
# (deliberately Settings-free, see below) treats the env var identically to
# `argus review`, which resolves it through Settings.
_PYDANTIC_TRUTHY_STRINGS = frozenset({"1", "true", "t", "yes", "y", "on"})


def _is_truthy_env_value(value: str) -> bool:
    return value.strip().lower() in _PYDANTIC_TRUTHY_STRINGS


def _prompts_list_override_dirs() -> list[Path]:
    """Same three-directory search chain as
    ``argus.prompts_runtime.override_dirs``, computed directly from
    ``os.environ`` rather than ``argus.config.Settings``.

    ``argus prompts list``/``export`` are deliberately implemented without
    going through ``Settings`` (see ``tests/test_cli_prompts.py``'s module
    docstring) so they work before the three required API keys are set —
    a first-time user should be able to inspect the packaged prompts
    without any credentials configured yet.
    """
    if _is_truthy_env_value(os.environ.get("ARGUS_NO_PROMPT_OVERRIDES", "")):
        return []

    dirs: list[Path] = []
    explicit = os.environ.get("ARGUS_PROMPTS_DIR")
    if explicit:
        dirs.append(Path(explicit))

    dirs.append(Path.cwd() / ".argus" / "prompts")

    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    config_home = Path(xdg_config_home) if xdg_config_home else Path.home() / ".config"
    dirs.append(config_home / "argus" / "prompts")

    return dirs


def _cmd_prompts_list() -> None:
    """``argus prompts list``: print each packaged prompt name and its source."""
    override_dirs = _prompts_list_override_dirs()
    for name in _packaged_prompt_names():
        source = "packaged"
        for override_dir in override_dirs:
            candidate = override_dir / f"{name}.md"
            if candidate.is_file():
                source = f"override ({candidate})"
                break
        print(f"{name}\t{source}")


def _cmd_prompts_export(target_dir: str, force: bool) -> None:
    """``argus prompts export <dir>``: copy packaged prompt files into ``target_dir``."""
    import shutil as shutil_mod

    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    if any(target.iterdir()) and not force:
        sys.exit(
            f"{target} is not empty. Use --force to export anyway "
            "(existing files with matching names will be overwritten)."
        )

    packaged = _packaged_prompts_dir()
    count = 0
    for src in sorted(packaged.glob("*.md")):
        shutil_mod.copyfile(src, target / src.name)
        count += 1
    logger.info("Exported %d prompt files to %s", count, target)
    print(f"Exported {count} prompt files to {target}")


def _run_prompts(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.prompts_command == "list":
        _cmd_prompts_list()
    elif args.prompts_command == "export":
        _cmd_prompts_export(args.dir, args.force)
    else:
        parser.error("Usage: argus prompts {list,export}")


def main() -> None:
    # Load .env BEFORE building the argparse parser so the
    # ``default=os.environ.get(...)`` lookups for ARGUS_STORAGE_* below
    # see values supplied via .env, not just the shell env. The full
    # settings load still happens inside ``_load_settings`` after args
    # are parsed — this is just to make .env-only env vars visible to
    # argparse defaults.
    from argus.dotenv_utils import load_dotenv_early

    load_dotenv_early(start=Path.cwd(), env_filename=".env")

    parser = _build_parser()

    # Accept both the "argus review ..." subcommand form and the flat
    # legacy form ("argus --repo ..." / "python -m argus.cli --repo ...")
    # without a subcommand — but leave a bare "-h"/"--help"/"--version"
    # (and the "prompts" subcommand) alone so they route correctly.
    argv = sys.argv[1:]
    if argv and argv[0] not in ("review", "prompts", "-h", "--help", "--version"):
        argv = ["review", *argv]

    args = parser.parse_args(argv)

    if args.command == "prompts":
        _run_prompts(parser, args)
        return

    if args.command != "review":
        parser.print_help()
        sys.exit(1)

    _validate_review_args(parser, args)
    _run_review(parser, args)


if __name__ == "__main__":
    main()
