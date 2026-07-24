# Contributing to Argus

Thanks for your interest in contributing. Argus is a self-orchestrated PR
review agent (LangGraph + Claude Agent SDK); this doc covers local dev setup
and what we expect from a PR.

Participation in this project is governed by our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Dev setup

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/redesignhealth/argus-review.git
cd argus-review
uv sync --all-extras
```

This creates a `.venv` and installs Argus (editable) plus the `dev` extra
(pytest, mypy, ruff). No credentials are required to run the test suite —
`tests/conftest.py` stubs `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` /
`GITHUB_TOKEN_RO` for you.

## Running checks locally

```bash
uv run pytest              # full test suite (includes the packaging test,
                            # which shells out to `uv build` — expect it to
                            # take a few seconds longer than the rest)
uv run ruff check .
uv run ruff format --check .
uv run mypy .               # strict mode — zero errors required
```

All four must pass before a PR is mergeable; CI runs the same commands (see
`.github/workflows/ci.yml`) on Ubuntu and macOS.

### Type safety

MyPy runs in **strict mode** (`[tool.mypy] strict = true` in
`pyproject.toml`). Every function needs full parameter and return type
annotations. Test helpers get some slack (see the `tests.*` mypy override in
`pyproject.toml`), but `argus/` code does not.

### Formatting

Ruff format is the source of truth for style; don't hand-format around it.
If you're using Claude Code, the repo's `.claude/settings.json` PostToolUse
hook runs `ruff format` automatically after edits.

## PR expectations

- Keep PRs scoped to one change. Don't mix a refactor with a feature.
- Add or update tests for behavior you change — this repo does not accept
  untested logic changes to `argus/`.
- If you're changing a prompt under `argus/prompts/`, read
  [`docs/CUSTOMIZING_PROMPTS.md`](docs/CUSTOMIZING_PROMPTS.md) first: some
  sections are transferable review methodology, others are
  Redesign-Health-specific convention — know which one you're touching.
- Reference the issue you're fixing, if one exists.
- CI (lint + guard + test, on Ubuntu and macOS) must be green.

## Reporting bugs / requesting features

Open a GitHub issue. For security vulnerabilities, see
[`SECURITY.md`](SECURITY.md) instead — do not open a public issue.

## Releasing

Maintainers: see [`docs/RELEASING.md`](docs/RELEASING.md) for the tag →
CI → verify release process. Most contributors won't need this.
