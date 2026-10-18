"""Guard: the SDK-bundled Claude Code CLI must be new enough for claude-opus-5-5.

Anthropic's API rejects claude-opus-5-5 from Claude Code older than 2.1.280
(HTTP 400, ``claude_code_version_too_old``). The Claude Agent SDK bundles the
CLI, so the SDK pin determines it. See TECH-7686.
"""

import re
import subprocess
from pathlib import Path

import claude_agent_sdk

MIN_BUNDLED_CLI = (2, 1, 280)


def test_bundled_claude_cli_supports_opus_5_5() -> None:
    cli = Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude"
    assert cli.exists(), f"bundled Claude Code CLI missing at {cli}"
    out = subprocess.run(
        [str(cli), "--version"], capture_output=True, text=True, timeout=60, check=True
    ).stdout
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", out)
    assert match, f"could not parse CLI version from {out!r}"
    version = tuple(int(g) for g in match.groups())
    assert version >= MIN_BUNDLED_CLI, (
        f"bundled Claude Code {version} is older than {MIN_BUNDLED_CLI}; "
        "claude-opus-5-5 requests will be rejected (TECH-7686)"
    )
