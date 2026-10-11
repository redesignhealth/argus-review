# Argus v3: Self-Orchestrated Parallel Architecture

This document describes the pipeline design behind Argus's third iteration
(v3), written when the project moved off an earlier architecture. It's
included here because the "why," not just the "what," is useful context for
anyone customizing the prompts or the pipeline itself.

Prompts are packaged files, optionally overridden via a search chain
(`ARGUS_PROMPTS_DIR`, then `./.argus/prompts/`, then
`~/.config/argus/prompts/` — see `docs/CUSTOMIZING_PROMPTS.md`), and there
is no orchestrator dependency
— the CLI runs the pipeline directly. `flow_run_id` in the storage schema is
`NULL` for CLI-driven runs; it exists to support callers that run Argus
under an external orchestrator (e.g. Prefect) and populate it themselves.

## Problem

An earlier version of this engine used a Claude Agent SDK orchestrator that
dispatched subagents via the SDK's `Agent` tool. Each `Agent` call blocked
until the subagent returned. A PR with 5 system groups took roughly
5 × subagent time instead of roughly 1×.

Testing confirmed this was a fundamental SDK limitation, not a prompting
issue: named agents, explicit parallel-execution instructions, and various
prompt-engineering approaches all still resulted in sequential execution.

## Solution

Replace the Agent SDK orchestrator with a self-orchestrated pipeline using
LangGraph for step coordination and true parallel fan-out (via LangGraph's
`Send` API / `asyncio.gather`) for subagent execution.

## Architecture

### Full review path

```
         ┌──────────────────┐
         │  fetch_diff      │ GitHub API + prior-round lookup from storage
         └────────┬─────────┘
                  │
         ┌────────┴────────┐
         ▼                 ▼
┌──────────────────┐ ┌──────────────────┐
│ precheck_checks  │ │ precheck_rules   │  Deterministic, non-LLM (parallel):
│ GitHub Checks API│ │ semgrep vs.      │  CI-status signal + custom-rule scan.
│ status (signal)  │ │ worktree         │  See docs/PRECHECKS.md.
└────────┬─────────┘ └────────┬─────────┘
         └────────┬───────────┘
         ┌────────▼─────────┐        ┌───────────────────┐
         │  precheck_join   │──FAIL──▶│  precheck_fail    │ Verified-rule hit:
         │  (fan-in)        │        │  BLOCKING, no LLM  │ synthesized response,
         └────────┬─────────┘        └───────────────────┘ routes straight to END.
                  │ (no verified hit)
         ┌────────▼─────────┐        ┌───────────────────┐
         │  scope_gate      │──HALT──▶│  END (BLOCKING)   │ scope_split /
         │  full PR, pinned │        │  halt, no findings │ scope_unverified
         └────────┬─────────┘        └───────────────────┘
                  │ one behavior
         ┌────────▼─────────┐
         │  early_verifier  │ Sonnet agent — round 2+ only, no-op on round 1
         │  (sequential)    │ Verifies prior BLOCKINGs before routing decision
         └────────┬─────────┘
                  │
         ┌────────▼─────────┐
         │  preflight       │ Sonnet structured call — lite vs full decision
         │                  │ prompt: pr-review-preflight-router
         └────────┬─────────┘
                  │
         ┌────────▼─────────┐        ┌──────────────────────┐
         │ blast-radius gate│──FULL──▶│  Planner (Opus)      │
         │ + open BLOCKING  │        │  prompt: pr-review-plan│
         │   gate           │        └──────────┬───────────┘
         └────────┬─────────┘                   │
                  │                   system_groups[], cross_cutting_concerns
                LITE                             │
                  │              ┌───────────────┼───────────────┐
                  ▼              ▼               ▼               ▼
         ┌────────────────┐ ┌──────────┐ ┌──────────┐ ┌─────────────────┐
         │  lite_review   │ │reviewer  │ │reviewer  │ │ Cross-Cutting   │
         │  Sonnet call   │ │(1) Sonnet│ │(N) Sonnet│ │ Opus agent      │
         │  no tools      │ │Read/Glob │ │Read/Glob │ │ Read/Glob/Grep  │
         └────────┬───────┘ └────┬─────┘ └────┬─────┘ └────────┬────────┘
                  │              └─────────────┴────────────────┘
                  │                            │ parallel fan-out
                  │                   all findings collected
                  │                            │
                  │                   ┌────────▼────────┐
                  │                   │ Coverage Check  │ Sonnet, ~3s
                  │                   └────────┬────────┘
                  │                            │
                  │                   ┌────────▼────────┐
                  │                   │ Writer          │ Sonnet
                  │                   │ prompt: pr-review-writer
                  │                   └────────┬────────┘
                  │                            │
                  │                   ┌────────▼────────┐
                  │                   │GPT-family model │ Schema extraction
                  │                   └────────┬────────┘
                  │                            │
                  │                   ┌────────▼────────┐
                  │                   │validate_blockings│ Sonnet agent
                  │                   └────────┬────────┘
                  │                            │
                  └────────────────────────────┘
                                    END
```

### Scope gate and workload halts

`scope_gate` runs after the deterministic precheck (a verified-rule hit
still short-circuits first) and before `early_verifier` and `preflight`. It
runs on every round, including incremental and lite rounds, and always over
the full PR, not the round diff.

1. It pins the base: `get_compare_metadata(base, head)` returns the merge
   base, which must be 40 lowercase hex characters. `get_compare_files(merge_base, head)`
   then lists the files. A listing of 300 files or more halts; the gate never
   falls back to the unpinned PR files endpoint. `fetch_diff` pins the base
   once; the round-1 (or SHA-mode) diff, the bench guard, and this gate all use
   that same SHA. Round 2+ diffs keep the `prior_sha` and rebase-clamp base.
2. A Sonnet call (native JSON schema, prompt `pr-review-scope`) partitions
   the files into independently mergeable behaviors. The prompt treats the
   description and patches as data. Each patch is capped at 6000 characters
   and the whole input at 150000 characters; a missing patch is marked as
   changed, never as no change. Model output is capped at 4096 tokens and 12
   behaviors.
3. Code validates the partition against the manifest. Overlapping, unknown,
   or unassigned files, empty scope for a non-empty PR, duplicate titles, and
   self or cyclic `depends_on` are `scope_unverified`. So is an `indeterminate`
   answer, an oversized input, and any gate error (the error message is not
   copied into the halt).
4. More than one behavior is `scope_split`, with up to 8 suggested slices.

The planner node counts reviewers (`sum(1 + specialists)` per group plus 2
always-on) before the fan-out edge emits any `Send`. A count over
`ARGUS_REVIEWER_DISPATCH_BUDGET` halts with `dispatch_budget_exceeded`.
A count over 50 always halts. The check-coverage node halts with
`gap_fill_cap_exceeded` when more than 50 gap-fill reviewers are needed
(the limit of 50 applies to each fan-out separately). Nothing truncates.
Every halt routes to `END`.

A human override (`--review-override-sha`, `--review-override-reason`) can
pass `scope_split` and a budget overrun of up to 50. It is passed in the
graph config, not the serialized request. A successful run always ends with
a `scope` receipt whose `head_sha` is the reviewed head and that has at most
one behavior (or an `override_reason`); `run_review` raises instead of
returning an `APPROVE` without one.

### Lite path

Triggered when the preflight LLM routes `"lite"` **and** neither hard gate
below fires.

Hard gates that force full review regardless of the LLM's own decision:

1. **Blast-radius floor**: any changed file matching a shared-library
   directory, `infrastructure/`, `/migrations/`, `/alembic/`,
   `/.github/workflows/`, `*.tf`, `Dockerfile`, or `serverless.yml` → full.
2. **Open BLOCKING gate**: prior-round verification shows any UNRESOLVED or
   REGRESSED BLOCKING → full.

Lite path: a single Sonnet call, no tools, no agent sessions — roughly
10-15 seconds and a fraction of a cent.

## Steps in detail

### Step 1: Planner

**Model**: a large-context, high-reasoning Claude model (Anthropic native
structured output via `output_config.format` with the `ReviewPlan` JSON
schema; an OpenAI GPT-family model is used for JSON repair when needed).
**Prompt**: `pr-review-planner`.

The planner streams the native structured-output JSON as text and collects
the complete response before strict Pydantic validation as `ReviewPlan`. If
the JSON is malformed, the pipeline sends the raw JSON text to an OpenAI
GPT-family model for schema-constrained repair, preserving the plan content
while fixing its encoding. If the stream is truncated, the planner detects
the incomplete JSON and retries the transient planner operation rather than
attempting repair on partial output.

**Why the larger model (not the smaller one used for reviewers)**: planner
quality directly gates every downstream reviewer — a mis-grouped file
manifest silently drops findings with no recovery path. The larger model's
context and file-classification accuracy measurably reduce planner-caused
coverage gaps in practice. The incremental cost (on the order of $0.20 per
round) is small relative to the cost of a missed blocker cascading into a
second round. The round-trip stays well under a few seconds.

**Input**: PR diff, PR description, relevant repo-convention content.

**Output** (structured):
```python
class ReviewPlan(BaseModel):
    system_groups: list[SystemGroup]
    cross_cutting_concerns: list[str]
    file_manifest: list[FileEntry]

class SystemGroup(BaseModel):
    name: str                    # e.g. "backend API endpoints"
    files: list[str]             # file paths assigned to this group
    conventions: str             # relevant repo-convention excerpt
    review_focus: str            # specific things to check
```

### Step 2: System reviewers (parallel)

**Model**: a mid-tier Claude model (agent session with tools).
**Prompt**: `pr-review-subagent`.
**Tools**: Read, Glob, Grep.
**Implementation**: each reviewer is a standalone Claude Agent SDK client
session. All are launched via parallel fan-out.
**Input**: the `SystemGroup` assignment from the planner.
**Output**: raw findings (file, line, description, context). No severity
assignment — that's deferred to the writer.

**Why standalone sessions**: avoids the Agent-tool blocking problem
described above. Each session runs independently.

### Step 3: Cross-cutting reviewer (parallel with Step 2)

**Model**: the larger/higher-reasoning Claude model (agent session with
tools).
**Prompt**: `pr-review-cross-cutting`.
**Tools**: Read, Glob, Grep.
**Input**: the full file manifest + `cross_cutting_concerns` from the
planner.
**Output**: findings focused on:
- Conditional execution path tracing across files
- Frontend/backend contract verification
- Deployment ordering (migration before code)
- Backward compatibility of changed read paths
- Session/connection lifecycle across `await` boundaries

**Why the larger model**: these are the issues the mid-tier model
consistently misses in gap analysis (documented at roughly 7 out of 40
findings in the internal gap-analysis exercise this design responded to).
The larger model is better at multi-file reasoning.

**Why parallel with Step 2**: it reads different aspects of the same
files. There is no dependency on system-reviewer output.

### Step 4a: Coverage check

**Model**: mid-tier Claude model (structured output, no tools).
**Prompt**: `pr-review-coverage-check`.
**Input**: `file_manifest` from the planner + all findings from Steps 2+3.
**Output**:
```python
class CoverageResult(BaseModel):
    is_covered: bool
    gaps: list[CoverageGap]  # empty if covered

class CoverageGap(BaseModel):
    files: list[str]
    reason: str  # why this wasn't covered
```

If gaps exist, the pipeline dispatches 1-2 targeted reviewers to fill them,
then proceeds. More than 50 gaps halts the review (`gap_fill_cap_exceeded`)
instead of dropping gaps.

### Step 4b: Review writer

**Model**: mid-tier Claude model (structured output, no tools).
**Prompt**: `pr-review-writer`.
**Input**: all findings + the coverage summary.
**Output**: scored findings with severity, verdict, risk level, and a
formatted review comment.

### Post-processing: structured extraction

Always runs. Takes the writer's text output and extracts it into the
`ReviewResponse` Pydantic schema using a GPT-family model. Handles type
mismatches, missing fields, and formatting issues that a single
constrained-generation pass on a schema this complex tends to produce on
large PRs.

## Model budget (approximate, from the original design)

| Step | Model tier | Estimated cost | Time |
|------|-------|---------------|------|
| Planner | Large/high-reasoning (streamed native structured output) | $0.25 | 5s |
| Planner fallback (rare) | GPT-family, small | ~$0.01 | +2s |
| System reviewers (3 avg) | 3× mid-tier | $0.60 | 40-60s (parallel) |
| Cross-cutting | Large/high-reasoning | $1.50 | 60s (parallel with above) |
| Coverage check | Mid-tier | $0.03 | 3s |
| Writer | Mid-tier | $0.20 | 15s |
| Structured extraction | GPT-family, small | $0.01 | 10s |
| **Total** | | **~$2.60** | **~1.5-2 min** |

These are the figures from the original three-architecture comparison
(v1/v2/v3); see the
[agentic code review harness write-up](https://www.redesignhealth.com/content/agentic-code-review-harness)
for later, larger-sample production numbers, which run higher on both cost
and time as the corpus of PRs reviewed skews toward larger, more complex
changes.

## Specialists (optional co-reviewers)

The planner can flag system groups that need specialist co-review. When
flagged, a specialist runs in parallel alongside the general system
reviewer for that group.

| Specialist | Prompt | When dispatched | Focus |
|---|---|---|---|
| Security | `pr-review-specialist-security` | Auth, endpoints, user input, secrets, IAM/OAuth scopes | Injection, auth bypass, secrets exposure, session management |
| SQL/Database | `pr-review-specialist-sql` | Migrations, SQL functions, ORM models, queries | Volatility, batch ops, migration ordering, N+1 |
| Infrastructure | `pr-review-specialist-infra` | Terraform, IAM, container orchestration, secret stores | Permission scopes, resource references, secret paths |
| Orchestration | (bundled into system/cross-cutting review in this build) | Workflow engines, async patterns | Framework-native solutions, parallel dispatch, retries, async correctness |
| Frontend | (bundled into system/cross-cutting review in this build) | React/TypeScript | Components, hooks, data fetching, API contracts |
| Deployment | (bundled into system/cross-cutting review in this build) | Dockerfiles, serverless, CI workflows | Deploy ordering, env config, image safety |
| LLM Patterns | (bundled into system/cross-cutting review in this build) | LLM SDK usage, prompts | Structured output, cost awareness |
| Observability | (bundled into system/cross-cutting review in this build) | Logging, tracing | Structured logging, secret leakage, feature monitoring |

The planner's output includes `specialists_needed: list[str]` per
`SystemGroup`. A group can have zero, one, or multiple specialists.
Additionally, all system reviewers and the cross-cutting reviewer receive
the `pr-review-prior-art` prompt, which checks for re-invented internal
utilities, framework-native solutions, and better libraries — see
`docs/CUSTOMIZING_PROMPTS.md` for which parts of that check are generic
methodology versus Redesign-Health-specific convention.

## Multi-round iterative review

A full review from scratch on every trigger is the round-1 behavior.
Round 2+ is an incremental review: the planner receives prior findings from
storage and tells reviewers to:

1. Check each open finding against the new diff — mark as **resolved**,
   **regressed**, or **still open**.
2. Review only the new code introduced by the fixes, not the entire PR
   again.
3. Verify that coverage gaps from the prior round were addressed.

### Data flow

```
Round 1 result (storage) → notes_for_next_round
    ↓
Round 2 planner receives: prior findings + coverage map + gap analysis
    ↓
Planner tells reviewers: "verify these were fixed, only review new changes"
    ↓
Writer: "auditor of round 1 + reviewer of fixes"
```

### Why this matters

- **Cost**: round 2 reviews run substantially cheaper — fewer reviewers,
  focused scope.
- **Speed**: round 2 reviews run substantially faster — less diff to
  process.
- **Quality**: verified fixes are more reliable than an independent
  re-review's fresh opinion, which can drift or fail to check whether the
  original issue was actually addressed.

## Re-invention detection

The cross-cutting reviewer checks for:

- **Internal re-invention**: code duplicating existing shared-library
  utilities.
- **External re-invention**: code re-implementing well-known public
  libraries.

The writer classifies material re-inventions as BLOCKING — the position is
that the PR should use the existing tool, not duplicate it.

## Risks (as identified in the original design)

1. **Planner quality**: if the planner misassigns files to groups,
   reviewers miss things. Mitigated by the coverage-check step.
2. **Cross-cutting reviewer cost**: the larger model on every review adds
   meaningfully to per-review cost. Could be made optional for small/
   low-risk PRs.
3. **Prompt drift**: more prompts to maintain than the prior architecture
   had — more surface area for inconsistency.
4. **Parallel resource pressure**: N concurrent agent sessions running at
   once. May need memory/CPU tuning depending on your execution
   environment.

## Success criteria (as originally set)

- Review time under 3 minutes for typical PRs (under 20 files).
- Blocking-detection rate at or above the prior architecture's rate.
- Cross-file path-tracing issues caught (the class of misses the prior
  architecture's gap analysis identified).
- Cost per review at or below roughly $3.00 average.
- Zero structured-extraction failures.

## Lite path: deliberately deferred design gaps

These are intentional decisions, not oversights, carried over from the
original design.

### 1. The blocking-validator step is bypassed on lite-emitted BLOCKINGs

**Decision**: lite reviews skip the post-writer validator agent.

**Rationale**: the validator is an Agent SDK session that reads real files
to confirm BLOCKING claims. Running it defeats the purpose of lite mode
(fast, cheap, no tools). The lite reviewer prompt is scoped to obvious
runtime bugs where false positives are rare; the validator would add
latency and cost that dominates the lite review's own cost.

**Risk accepted**: a hallucinated BLOCKING on a lite review is not
re-checked by the validator. Treat a lite BLOCKING as a strong suggestion
and verify manually.

**Removal condition**: if the false-positive rate on lite BLOCKINGs
exceeds the full-path rate (documented at roughly 15% in the original
design), wire lite through the blocking-validator step as a conditional
edge when the review has any blocking findings.

### 2. Lite reviews are a weaker fit for any feedback/pattern-mining loop you build

**Not part of this toolkit**: an aggregation job that mines a corpus of
recurring findings across rounds for prompt tuning (discussed conceptually
in the white paper) is not something this repo runs — Argus persists round
history, but nothing here reads back across rounds to mine patterns. If
you build that kind of loop yourself, this is the design note to have in
hand first.

**Rationale**: that kind of feedback loop is designed to improve the full
multi-agent pipeline's specialist prompts, planner routing, and
coverage-check logic. Lite reviews exercise none of those surfaces — a
pattern-mining job that includes lite (`v3-lite`) rounds alongside
full-path (`v3`) rounds would dilute its signal with single-pass reviews
that have no specialist or cross-cutting component. Scope any such job to
full-path reviews only.

**If you also want lite-specific feedback**: build a separate loop scoped
to what lite mode actually exercises (e.g., tuning the preflight-router
prompt from real routing decisions), rather than folding lite rounds into
a full-path corpus.
