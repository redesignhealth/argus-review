# Customizing the review prompts

Argus ships with the prompts Redesign Health actually runs in production,
byte-for-byte. This is a deliberate choice, not an oversight: a "neutral"
prompt set that hedges every judgment call would be worse for everyone, and
would also be dishonest about what's actually been tested at volume. What
you get instead is **a worked example of a fully tuned deployment** — a real
answer to "what does a mature agentic code-review harness's prompt graph
look like after many iterations," which you can then adapt.

That means some of what's in these prompts will not apply to your repo. Read
this doc before you start editing.

## Anatomy: prompt → pipeline step

Each pipeline stage that calls an LLM reads its instructions from a single
named prompt (adapted from the pipeline description in
`docs/ARCHITECTURE.md`):

| Prompt name | Pipeline step | What it does |
|---|---|---|
| `pr-review-planner` | Plan | Reads the diff + PR description, groups changed files into system groups, assigns specialists per group, flags cross-cutting concerns. Also carries the shared review-perspectives rubric (Senior SWE, Security, Test Coverage, Performance, SQL & Database, Architecture, Self-Orchestration, Stub Completeness, Silent Fallbacks vs. Loud Failures, Re-Invention / Prior Art, Cross-File Integration, Documentation Compliance) — see the note below the table. |
| `pr-review-subagent` | System reviewer (per group, parallel) | General-purpose reviewer instructions for one system group: correctness, security basics, tests, SQL |
| `pr-review-specialist-security` | Specialist (parallel, when triggered) | Deep pass on injection, auth bypass, secrets exposure, session management |
| `pr-review-specialist-sql` | Specialist (parallel, when triggered) | Deep pass on migration safety, query patterns, ORM alignment, batch operations |
| `pr-review-specialist-infra` | Specialist (parallel, when triggered) | Deep pass on Terraform/IAM, deployment configuration, CI/CD workflows, secret paths |
| `pr-review-cross-cutting` | Cross-cutting reviewer (parallel with all of the above) | Multi-file data flow, deployment ordering, session/connection lifecycle, IAM tracing, contract verification, architecture-doc compliance, model-registry compliance |
| `pr-review-tests-and-docs` | Parallel with the above | Dedicated test-coverage and documentation-compliance pass across the whole diff |
| `pr-review-prior-art` | Injected into system/specialist/cross-cutting reviewers | Checks for re-invented internal utilities and re-implemented public libraries |
| `pr-review-coverage-check` | Coverage check | Verifies every changed file was examined by at least one reviewer; returns gaps if not |
| `pr-review-writer` | Writer | Consolidates all findings, assigns severity, produces verdict + risk level + the formatted review comment. Carries the same shared review-perspectives rubric as `pr-review-planner` (see the note below the table). |
| `pr-review-feedback-verifier` | Round 2+, before planning | Checks each prior-round finding against the new diff: resolved, regressed, or still open |
| `pr-review-blocking-validator` | After the writer | Re-confirms each BLOCKING finding against the actual diff before it reaches the engineer — the system's main anti-hallucination check |
| `pr-review-preflight-router` | Before planning | Decides lite vs. full review path for the incoming diff |
| `pr-review-lite` | Lite path only | Single-pass, no-tools review for small/low-risk diffs; skipped entirely on the full path |

`pr-review-feedback-proposal` doesn't appear in the table above and isn't
part of the per-PR pipeline this package runs — it belongs to Redesign
Health's internal offline weekly prompt-tuning process, which isn't
included in this repo (see `docs/STORAGE.md` for the related
`review_patterns` table, which is also out of scope). It's not fetched by
`argus review`, doesn't ship as a packaged prompt, and `argus prompts
export` won't produce a file for it.

**A shared rubric, not a shared file.** `pr-review-planner` and
`pr-review-writer` both need the same review-perspectives rubric (Senior
SWE, Security, Test Coverage, Performance, SQL & Database, Architecture,
Self-Orchestration, Stub Completeness, Silent Fallbacks vs. Loud Failures,
Re-Invention / Prior Art, Cross-File Integration, Documentation
Compliance). In this package that content is duplicated inline in both
files rather than injected from a shared fragment — there's no separate
`pr-review-criteria` prompt to edit. If you change the rubric, edit both
`pr-review-planner.md` and `pr-review-writer.md` to keep them in sync.

Run `argus prompts list` to see the exact set packaged with your installed
version, and `argus prompts export ./my-prompts` to get real files on disk
you can diff against this table and edit directly.

## What's transferable methodology vs. what's Redesign Health convention

Every one of the prompts above mixes two kinds of content. Telling them
apart is the point of this doc.

**Transferable methodology** — the parts we'd expect to be useful in any
repo, because they encode a general theory of what agentic code review
should catch, not a house style:

- **Silent-fallback detection** (`pr-review-planner`, `pr-review-writer`, `pr-review-cross-cutting`):
  the BLOCKING-severity distinction between a fallback path that silently
  swallows a failure and a loud, explicit failure that surfaces it. This is
  a real, repo-agnostic bug class in agent-generated code.
- **Stub completeness** (`pr-review-planner`, `pr-review-writer`, `pr-review-cross-cutting`):
  catching functions that claim to perform an action (write to cache,
  commit a transaction, call an API) but don't actually do it. Also
  general — it's about verifying an implementation does what its name
  promises.
- **Loud-failure philosophy**: prefer explicit exceptions over caught,
  logged, and ignored errors. A general engineering stance, not RH-specific.
- **Coverage mechanics** (`pr-review-coverage-check`): the idea of
  cross-checking the reviewed-files set against the changed-files manifest,
  and dispatching targeted re-review for any gap, is architecture-level, not
  content-level — it works with any file manifest.
- **Severity rubric** (`pr-review-writer`, `pr-review-blocking-validator`):
  the BLOCKING vs. suggestion vs. nit distinction, and the practice of a
  dedicated validation pass that re-confirms each BLOCKING against the real
  diff before it ships, generalize past any specific stack.
- **Re-invention / prior-art detection as a category** (`pr-review-prior-art`,
  `pr-review-cross-cutting`): checking whether a PR reimplements something
  that already exists is a useful check in any codebase with shared
  libraries — only the specific library names are RH-specific (see below).

**RH-specific convention** — the parts that are correct for Redesign
Health's stack and should be replaced with your own team's equivalents,
not just deleted:

- **`rh-lib` prior-art checks**: `pr-review-planner`, `pr-review-writer`, and
  `pr-review-subagent` explicitly flag `[REINVENTS rh-lib]` and
  `[EXTRACT TO rh-lib]` findings, and `pr-review-specialist-sql` calls out
  "New models must use shared Base class from rh-lib." Replace `rh-lib`
  with the name of your own internal shared package (or delete the check
  if you don't have one).
- **`.cursorrules` compliance**: several prompts instruct the reviewer to
  "Read the .cursorrules file for the directory being modified" and flag
  violations. If your repo doesn't use `.cursorrules` (or uses `CLAUDE.md`,
  a style guide, or nothing), swap in whatever convention file your repo
  actually has, or remove the instruction.
- **SSM conventions**: `pr-review-specialist-infra`, `pr-review-cross-cutting`,
  and `pr-review-tests-and-docs` all check that new secrets are "documented
  in SSM_DEPLOYMENT.md" and follow the `/general/{env}/` vs. `/{project}/{env}/`
  path convention. This is Redesign Health's AWS SSM Parameter Store
  layout specifically — replace with your own secrets-management
  convention (Vault, Doppler, plain env vars, whatever you use), or drop it.
- **House stack references**: the "Self-Orchestration / Framework
  Re-Implementation" section in `pr-review-planner` and `pr-review-writer` flags
  reimplementing Prefect, LangGraph, FastAPI, SQLAlchemy, or Slack Bolt
  functionality instead of using the framework natively; `pr-review-writer`
  and the model-registry check in `pr-review-cross-cutting` reference
  `rh_lib.llm.models.ALIAS_MAP` specifically. Swap the framework/library
  names for whatever your stack actually is.

**How to use this taxonomy in practice**: keep the methodology sections
as-is (they're doing real work), and for each convention section, either
(a) replace the RH-specific names/paths with your own equivalents, or
(b) delete the section if the underlying practice doesn't apply to your
repo. Don't just delete convention sections wholesale without checking
whether an equivalent check would still be valuable — most of them encode a
real category of bug (undocumented secrets, unreviewed shared-code
duplication) even if the specific tool names don't transfer.

## Recipe: exporting, editing, and using your own prompts

Every prompt name resolves through a search chain — the first directory
below that contains a matching `<name>.md` wins, and any name none of them
have falls back to the packaged file:

1. `ARGUS_PROMPTS_DIR` — an explicit override directory, if set.
2. `./.argus/prompts/` — resolved relative to wherever `argus` is run from.
   Good for a repo-specific override: gitignore it for a personal
   experiment, or commit it if your whole team should share the override.
3. `~/.config/argus/prompts/` (respecting `XDG_CONFIG_HOME` if you set it) —
   applies no matter which repo you're in. Good for a standing personal
   customization you want everywhere.
4. The packaged prompts.

The simplest setup — a standing personal override, no path to remember or
export on every run:

```bash
# 1. Export the packaged prompts straight to the user-global location
argus prompts export ~/.config/argus/prompts

# 2. Edit whichever files need to change for your repo
$EDITOR ~/.config/argus/prompts/pr-review-cross-cutting.md

# 3. Just run Argus — no ARGUS_PROMPTS_DIR needed, it's already found
argus review owner/repo --pr 123
```

Or, for a one-off / repo-specific override without touching your global
config:

```bash
argus prompts export ./my-prompts
$EDITOR ./my-prompts/pr-review-cross-cutting.md
export ARGUS_PROMPTS_DIR=./my-prompts   # highest priority, this run only
argus review owner/repo --pr 123
```

Whichever directory you use, prompts you don't override still resolve from
the packaged set — you can override just `pr-review-cross-cutting.md` and
leave everything else on the shipped defaults. `argus prompts list` shows
exactly which source (and, for an override, which path) each prompt name
currently resolves to.

If you need to guarantee pristine packaged prompts regardless of what's on
disk or in the environment — a CI job verifying the shipped defaults,
for example — set `ARGUS_NO_PROMPT_OVERRIDES=1` (or pass `--no-prompt-overrides`
to `argus review`) to skip the entire override chain.

## Evaluating a prompt change

Prompt edits are easy to get subtly wrong — a change that reads as an
improvement can quietly widen or narrow what a reviewer flags in ways that
aren't obvious from the prompt text alone. The cheapest way to check:

1. Pick a PR you've already reviewed with the stock prompts (ideally one
   with a known, understood set of findings — including any you think
   were false positives or misses).
2. Re-run Argus against that same PR/SHA with `ARGUS_PROMPTS_DIR` pointed
   at your edited prompt.
3. Diff the two rounds' findings: what showed up that didn't before, what
   disappeared, and whether severity assignments shifted.
4. If you're iterating on a convention check specifically (e.g., swapping
   in your own shared-library name), verify the new check actually fires
   on a PR that should trigger it and stays quiet on one that shouldn't —
   the same "positive and negative example" sanity check you'd run on any
   other prompt.

Because round history is keyed on `(repo, pr_number)`, re-running against
the same PR after a prompt change will be treated as a new round for that
PR and will pull in prior-round context (see `docs/STORAGE.md`) — if you
want a clean, independent comparison instead, review a fresh SHA with
`--sha`/`--base-ref`, or clear that PR's history in your storage backend
between runs.
