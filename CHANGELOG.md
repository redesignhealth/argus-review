# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.2] - 2026-09-11

### Added

- `docs/BENCH.md`, documenting the reviewer bench's config-driven platform,
  model, and caching selection for leaf-reviewer roles, including the override
  chain, valid values, platform credentials, and a worked Gemini configuration.
  README and `docs/BUILDING_A_REVIEW_LOOP.md` now point to the new guide.

### Fixed

- Updated the `gemini-mini` alias from the superseded `gemini-3-flash-preview`
  model to the stable `gemini-3.8-flash` release. The approved-model policy
  table's Google default now matches, and a regression test pins the alias to
  the approved value.

## [0.2.1] - 2026-09-10

### Added

- `GOOGLE_BASE_URL` proxy support for the Gemini runner (`argus/gemini_runner.py`),
  mirroring `OPENAI_BASE_URL`: closes a gap where the Gemini platform could only
  be used with a raw, unproxied `GOOGLE_API_KEY` talking directly to
  `generativelanguage.googleapis.com`, unlike the OpenAI (`OPENAI_BASE_URL`) and
  Claude (`ANTHROPIC_BASE_URL`) paths that the `argus-review-loop` skill relies
  on to route through a short-lived, PR-scoped credential proxy instead of a
  standing API key.

## [0.2.0] - 2026-09-10

### Added

- Config-driven reviewer bench system (`argus/bench.py` and `argus/bench_default.toml`):
  allows configuring the execution platform (`claude-sdk`, `gemini`, `openai`)
  and model per reviewer role via TOML, with a sparse-overlay hierarchy
  (packaged default -> user-global `~/.config/argus/bench.toml` -> repo-local
  `.argus/bench.toml` -> `ARGUS_BENCH_FILE`), plus `--bench-file` and
  `--no-bench-overrides`/`ARGUS_NO_BENCH_OVERRIDES` controls.
- Real Gemini leaf-reviewer platform (`argus/gemini_runner.py`) backed by the
  `google-genai` SDK with explicit context caching (`argus/gemini_cache.py`)
  for cost and latency optimization; available via the optional `[gemini]` extra.
- Real OpenAI Responses-API leaf-reviewer platform (`argus/openai_runner.py`):
  native multi-turn tool execution loop using the OpenAI Responses API, with
  support for `OPENAI_BASE_URL` and credential proxying.
- Shared reviewer tools (`argus/review_tools.py`) providing decoupled, sandboxed
  tool execution (read_file, edit_file, grep, bash, webfetch) for non-Claude-SDK
  runners.
- Loud timeout and crash surfacing (TECH-6146): reviewer subprocess crashes and
  timeouts that were previously swallowed into silent 0-finding results now record
  `failure_reason`, surface as structured coverage-gap findings in review comments,
  and trigger a guard preventing all-failed reviewer rounds from issuing a clean
  verdict on zero actual coverage.
- Approved OpenAI model family updates: `gpt-frontier` default updated to
  `gpt-5.6-sol` and `gpt-mini` to `gpt-5.6-luna`.

### Changed

- All leaf-reviewer roles are now wired through the bench system:
  `system-generalist`, `specialist-infra-security`, `specialist-api-patterns`,
  `specialist-testing-observability`, and `cross-cutting-synthesizer` now resolve
  their platform and model dynamically via `bench.resolve()`.
- Default reviewer session timeout increased from 10 minutes (600s) to 15 minutes
  (900s) across all three runner platforms (`DEFAULT_ARGUS_SESSION_TIMEOUT_S`).

### Fixed

- Preserved partial turn usage, tool call counts, and cost metrics when an OpenAI
  or Gemini reviewer session crashes or times out mid-execution.
- Degraded-response detection for OpenAI Responses API (incomplete token budget,
  refusals, or error statuses now properly fail closed with a worker-crashed reason).
- Resolved false-positive double-reporting of precheck scanner failures in
  coverage-gap finding lists.

## [0.1.5] - 2026-08-07

### Added

- `--specialist-model`/`ARGUS_SPECIALIST_MODEL` and `--frontier-model`/`ARGUS_FRONTIER_MODEL`
  CLI flags/env vars to override the reviewer models per run.
  `--specialist-model` overrides the system reviewer, specialist reviewers,
  writer, and lite-review path; `--frontier-model` overrides both the
  planner/coverage tier and the cross-cutting reviewer together. Pass an
  empty string to clear an already-set override for one run.

- LangSmith span enrichment across the reviewer subprocess boundary
  (TECH-4734 phase 2): when `LANGSMITH_API_KEY` is set, the ambient
  LangGraph run tree is captured in the parent process and forwarded to
  each spawned reviewer subprocess via `langsmith_extra`, so subagent
  sessions nest under the graph-level trace instead of starting orphaned
  root traces.
- A per-session local context-usage ledger: sizes/counts (never
  prompt/response content) for every message a reviewer subprocess sends,
  written unconditionally to `/tmp/argus-context-ledger-<pid>.jsonl`
  regardless of whether LangSmith tracing is enabled. Auto-removed on
  subprocess exit, or by the parent process if the subprocess is killed
  after a timeout or crash. Override the path with the new
  `ARGUS_CONTEXT_LEDGER_PATH` env var (persists the file instead of
  auto-removing it).

### Changed

- The default `claude-default` model pin (`CLAUDE_DEFAULT`) moved from
  `claude-sonnet-5` to `claude-sonnet-4-6`.

## [0.1.4] - 2026-08-06

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
- `PrecheckResult.failed_scanners`: a scanner that crashes/times out is
  now surfaced in the review comment's degraded-coverage section, not
  just a backend log line, so a genuine coverage gap is distinguishable
  from "ran clean, found nothing." Purely observability by default.
- `ARGUS_PRECHECK_BLOCK_ON_SCANNER_FAILURE` (opt-in, off by default):
  forces the verdict to `BLOCKING` when a precheck scanner failed this
  round, for repos that have decided a silently-skipped deterministic
  gate is a worse outcome than a blocked review. See `docs/PRECHECKS.md`'s
  "Opting into fail-closed on scanner failure" section.

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

[Unreleased]: https://github.com/redesignhealth/argus-review/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/redesignhealth/argus-review/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/redesignhealth/argus-review/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/redesignhealth/argus-review/compare/v0.1.5...v0.2.0
[0.1.5]: https://github.com/redesignhealth/argus-review/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/redesignhealth/argus-review/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/redesignhealth/argus-review/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/redesignhealth/argus-review/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/redesignhealth/argus-review/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/redesignhealth/argus-review/commits/v0.1.0
