# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2] - 2026-07-31

### Added

- `argus/llm/pricing.py`: litellm-backed per-token cost lookup, replacing
  the previous hardcoded lite-review pricing table.
- `claude-opus` model tier (`CLAUDE_OPUS`). The cross-cutting reviewer now
  runs on it instead of the frontier tier — evals showed no measurable
  quality gain from frontier on that stage, at roughly double the
  per-token cost.
- Failure-reason tracking distinguishes `timeout` from `worker_crashed`
  for reviewer sessions, instead of collapsing both into a single
  boolean, so degraded-coverage reporting can say which actually
  happened.

### Fixed

- `ANTHROPIC_AUTH_TOKEN` (the standard bearer/gateway-proxy credential
  convention) is now supported alongside `ANTHROPIC_API_KEY`. Previously
  `Settings` required `ANTHROPIC_API_KEY` unconditionally, so a caller
  routing through a corporate LLM gateway/proxy with only
  `ANTHROPIC_AUTH_TOKEN` set got a hard startup failure with no
  workaround.
- Review prompts now steer reviewers away from full-file `Read` on large
  files, to avoid blowing context on files that don't need to be read in
  full.
- A bad history-backend DB path/URL now fails fast instead of surfacing a
  confusing downstream error.
- `temperature` is no longer sent to `claude-sonnet-5`/`claude-fable-5`,
  which reject the parameter.
- Closed a few architecture-mandate false-reject routes in the
  cross-cutting/writer/blocking-validator prompts, and corrected the
  Architecture Compliance rule.

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

[Unreleased]: https://github.com/redesignhealth/argus-review/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/redesignhealth/argus-review/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/redesignhealth/argus-review/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/redesignhealth/argus-review/commits/v0.1.0
