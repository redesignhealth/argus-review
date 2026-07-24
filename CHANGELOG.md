# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/redesignhealth/argus-review/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/redesignhealth/argus-review/commits/v0.1.0
