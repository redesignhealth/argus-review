# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Deterministic, non-LLM precheck gate running before the review pipeline
  spends any LLM tokens: a routing signal from the target repo's own CI
  status (always on, no extra dependency), and custom semgrep rules
  against the worktree that can fast-fail a PR (once verified) or attach
  non-blocking writer context (while candidate) -- this half is gated
  behind the new `prechecks` extra
  (`pip install "argus-code-review[prechecks]"`) and is a complete no-op
  without it installed. See `docs/PRECHECKS.md`.
- `gitleaks` and `actionlint` CI jobs for this repo's own source.
- Seven additional stock rule sources for the precheck gate, feeding the
  same candidate/verified pipeline as custom `ARGUS_RULES_DIR` rules and
  independent of whether one is configured: semgrep registry packs
  (`ARGUS_STOCK_SEMGREP_PACKS`, e.g. `p/secrets`); `zizmor` and
  `actionlint` for GitHub Actions (security, and syntax/shellcheck,
  respectively); Trivy for secrets; squawk for Postgres migration safety;
  Checkov for Terraform IAM/privilege-escalation; and a bundled
  `eslint-plugin-security` for JS/TS. See `docs/PRECHECKS.md`'s "Stock
  rule sources" section for exact versions, install instructions (several
  are standalone binaries or npm packages `pip` cannot install), and
  scope/overlap decisions (e.g. why Trivy's own misconfiguration scanner
  is deliberately unused in favor of Checkov).
- `run_precheck` now accepts a `changed_files` list to scope findings to
  what the PR actually touched, and runs every scanner concurrently via
  `asyncio.gather` rather than sequentially -- both prerequisites for
  adding the additional scanners above without flooding every PR with
  pre-existing findings or stacking up each scanner's own timeout.

## [0.1.3] - 2026-08-03

### Added

- Per-message and per-tool-result context-usage logging (TECH-4734),
  making an autocompact-thrashing pattern in long reviewer sessions
  (context repeatedly refilling to the limit within a few turns,
  starving the reviewer of any real progress) directly visible in logs
  for the first time instead of surfacing only as an empty/garbled
  finding.
- `ARGUS_CONTEXT7_BASE_URL` override (TECH-4736), so a caller proxying
  `CONTEXT7_API_KEY` through an intermediary (e.g. the
  `argus-review-loop` skill's rh-mcp credential proxy) can point
  Context7 MCP calls at the proxy instead of the real host.

## [0.1.2] - 2026-07-31

### Fixed

- `ANTHROPIC_AUTH_TOKEN` (the standard bearer/gateway-proxy credential
  convention) is now supported alongside `ANTHROPIC_API_KEY`. Previously
  `Settings` required `ANTHROPIC_API_KEY` unconditionally, so a caller
  routing through a corporate LLM gateway/proxy with only
  `ANTHROPIC_AUTH_TOKEN` set got a hard startup failure with no
  workaround.

## [0.1.1] - 2026-07-24

### Fixed

- README quickstart command: `uvx argus-code-review ...` doesn't work
  as documented — `uvx` only infers the console script name from the
  package name when they match, and here they don't (package
  `argus-code-review`, script `argus`). Corrected to
  `uvx --from argus-code-review argus ...`, confirmed working
  end-to-end against a real PR.

## [0.1.0] - 2026-07-24

### Added

- Initial public release of Argus: a self-orchestrated PR review agent
  using LangGraph + the Claude Agent SDK.
- Three storage backends for round history and pipeline checkpoints:
  local SQLite (zero-config default), Postgres, and an HTTP shim for
  sandboxed environments.
- A prompt override search chain (`ARGUS_PROMPTS_DIR` →
  `./.argus/prompts/` → `~/.config/argus/prompts/` → packaged prompts),
  plus `ARGUS_NO_PROMPT_OVERRIDES`/`--no-prompt-overrides` to force the
  packaged set.
- `argus --version`, `argus prompts list`, and `argus prompts export`.

[Unreleased]: https://github.com/redesignhealth/argus-review/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/redesignhealth/argus-review/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/redesignhealth/argus-review/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/redesignhealth/argus-review/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/redesignhealth/argus-review/commits/v0.1.0
