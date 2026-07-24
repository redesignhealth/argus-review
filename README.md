# Argus Review

**Argus is an open-source code review tool built by
[Redesign Health](https://www.redesignhealth.com).**

It reviews pull requests by decomposing each change into the flows it
participates in, dispatching repository-aware specialist agents in parallel
against those flows, validating findings before surfacing them, and
persisting every round so later rounds can check what was resolved,
regressed, or still open instead of re-reviewing from scratch. The
architecture borrows from the same ideas now showing up in the broader
agentic-harness conversation: stochastic and deterministic hooks composed
deliberately, just-in-time context retrieval, structured outputs, and
explicit coverage accounting.

For the design principles and production results behind Argus, see the
[agentic code review harness write-up](https://www.redesignhealth.com/content/agentic-code-review-harness)
on our site.

The prompts Argus ships with are opinionated — they encode one team's
production coding standards, not a neutral default. Expect to adapt them
to your own codebase and conventions (see
[Adapting Argus to your team](#adapting-argus-to-your-team)) rather than
running them unmodified.

> ### Cost Responsibility
>
> Argus makes API calls to third-party services (Anthropic, OpenAI, and
> optionally GitHub) using your API keys. You are solely responsible for
> all token consumption and associated costs incurred by running Argus.
> Costs vary significantly depending on PR size, repository complexity,
> and configuration — in production use, a single review may cost anywhere
> from approximately $2 to $42 or more in API fees. Redesign Health does
> not monitor, limit, or reimburse these costs. By using Argus, you
> acknowledge that you bear full financial responsibility for any and all
> API charges incurred through your credentials.
>
> ### No Warranty; Limitation of Liability
>
> Argus is provided "AS IS" and "AS AVAILABLE," without warranty of any
> kind, express or implied, including but not limited to the warranties of
> merchantability, fitness for a particular purpose, accuracy, or
> non-infringement. Redesign Health Inc. makes no representation or
> warranty that Argus will be error-free, uninterrupted, secure, or free of
> bugs or defects.
>
> To the fullest extent permitted by applicable law, in no event shall
> Redesign Health Inc., its affiliates, officers, directors, employees, or
> agents be liable for any indirect, incidental, special, consequential, or
> punitive damages, or any loss of profits, data, use, or goodwill, arising
> out of or in connection with your use of or inability to use Argus,
> regardless of the theory of liability (contract, tort, strict liability,
> or otherwise), even if advised of the possibility of such damages.
>
> Your use of Argus is at your sole risk. This tool performs automated code
> review and may produce inaccurate, incomplete, or misleading findings. It
> is not a substitute for human code review, security auditing, or
> professional judgment. You are solely responsible for any decisions made
> or actions taken based on Argus's output.
>
> For the full license terms, see the [Apache License 2.0](LICENSE)
> included in the repository.

- [Cost Responsibility](#cost-responsibility)
- [No Warranty; Limitation of Liability](#no-warranty-limitation-of-liability)
- [How it works](#how-it-works)
- [Getting started](#getting-started)
- [Using Argus](#using-argus)
- [Managing secrets](#managing-secrets)
- [Connecting a database](#connecting-a-database)
- [Adapting Argus to your team](#adapting-argus-to-your-team)
- [Measured results](#measured-results)
- [Docs](#docs)

## How it works

A pull request goes in. Argus's planner groups the changed files into
"system groups" (roughly: one per feature area touched), assigns specialist
co-reviewers where the file patterns warrant it (security, SQL, infra, ...),
and flags anything that needs a cross-file pass. Each group and specialist
then runs as its own Claude Agent SDK session — with `Read`/`Glob`/`Grep`
tools against a SHA-pinned local git worktree, not a shared context window
— in parallel. A coverage check verifies every changed file got reviewed by
someone (gaps get a targeted follow-up pass), and a final writer step
consolidates every finding into one review with a verdict and severity
tags.

```
fetch_diff → plan → [system reviewers ‖ specialists ‖ cross-cutting] (parallel)
           → check_coverage → (fill_gaps if needed) → write_review → persist
```

Full design rationale is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Getting started

**Prerequisites:**

- `git`
- Python 3.12+
- The [`claude` CLI](https://docs.claude.com/en/docs/claude-code) (Node.js) —
  the Claude Agent SDK spawns it as a subprocess for every reviewer session
- Three API keys — see [Managing secrets](#managing-secrets)

**Run:**

```bash
export ANTHROPIC_API_KEY=...
export GITHUB_TOKEN_RO=...
export OPENAI_API_KEY=...

uvx --from argus-code-review argus review owner/repo --pr 123
```

Prefer a `.env` file over exporting in your shell? Copy
[`.env.example`](.env.example) to `.env`, fill in the three keys above, and
Argus picks it up automatically (see [Managing secrets](#managing-secrets)).

That's it for a first run. With no other configuration, Argus stores round
history and pipeline checkpoints in a local SQLite database at
`~/.local/share/argus/history.db` — nothing else to set up.

`uvx --from argus-code-review argus ...` downloads the `argus-code-review`
package into a throwaway environment and runs its `argus` console script —
the `--from` is required because the installable package name
(`argus-code-review`) and the console script it provides (`argus`) differ,
and `uvx` only infers the script name from the package name when they
match. If you'll be running Argus repeatedly, install it once instead of
re-resolving it every invocation:

```bash
uv tool install argus-code-review
argus review owner/repo --pr 123
```

## Using Argus

```bash
# Review an open PR
argus review owner/repo --pr 123

# Review a specific commit against a base ref (no open PR required)
argus review owner/repo --sha abc123 --base-ref main

# Write markdown + JSON output
argus review owner/repo --pr 123 -o review.md

# Publish the review as a PR comment (upserts on re-run) and set a commit status
argus review owner/repo --pr 123 --post --commit-status

# Dismiss a finding before the next round
argus review owner/repo --pr 123 --dismiss "B2 -- pre-existing, not from this PR"

# Inspect and export the packaged prompts
argus prompts list
argus prompts export ./my-prompts
```

**What you get back:**

- A formatted review comment (Markdown) with a verdict (`APPROVE` /
  `BLOCKING`), a risk level, and findings tagged `B1`, `B2`, ... (blocking)
  or `S1`, `S2`, ... (suggestion).
- `-o review.md` writes that Markdown to a file, plus a sibling `review.json`
  with the full structured response.
- Re-running against the same PR (same `owner/repo` + `--pr`) picks up the
  prior round from storage automatically: round 2+ verifies whether
  previously reported findings were resolved, regressed, or are still open,
  scoped to just the commits since the last review — not a full re-review.

## Managing secrets

Argus reads configuration from environment variables, or from a `.env` file
in the working directory (add it to `.gitignore` — never commit it).
[`.env.example`](.env.example) lists every variable Argus reads, required
and optional, with comments — copy it to `.env` and fill in what you need.

Three keys are required for a first run:

| Env var | Used for |
|---|---|
| `ANTHROPIC_API_KEY` | The Claude Agent SDK reviewer sessions and the planner/coverage/writer calls |
| `GITHUB_TOKEN_RO` | Fetching the PR diff and cloning the repo into the review worktree |
| `OPENAI_API_KEY` | Structured extraction of the final review (see the note in [Getting started](#getting-started)) |

`GITHUB_TOKEN_RO` needs read-only access: *Contents: read* + *Pull
requests: read* on a fine-grained PAT, or `repo` read on a classic one.

How you *get* these values into the environment is up to your setup —
export them in your shell, load them from a secrets manager in CI, or use
a `.env` file locally. Argus itself has no opinion and no cloud dependency
for secrets; it just reads `os.environ`.

The full configuration surface (storage URLs, prompt overrides, tracing,
etc.) is documented where each feature is introduced below, and summarized
in [`docs/STORAGE.md`](docs/STORAGE.md) for storage-specific variables.

## Connecting a database

Argus needs somewhere to keep **round history** (so re-reviewing a PR knows
what was already found) and **pipeline checkpoints** (so a crash mid-run
can resume). Three modes are supported, resolved automatically from which
environment variables are set — there's no `--backend` flag.

### Recipe: local, zero-config (default)

Set nothing. Both round history and checkpoints land in a local SQLite file
at `~/.local/share/argus/history.db` (created automatically on first run).
Good for solo use and for trying Argus out — this is what the quickstart
above uses.

### Recipe: shared Postgres, e.g. Supabase

Use this to share round history across a team or CI. Any Postgres works;
Supabase is a common zero-ops choice.

1. Create a Supabase project (or use an existing one), then grab its
   **pooler** connection string (Project Settings → Database → Connection
   string → "Transaction" pooling mode, port `6543`) — the pooler is what
   you want for a short-lived CLI process like Argus, not the direct
   connection on port `5432`.
2. Apply the schema, in order:
   ```bash
   psql "$SUPABASE_DB_URL" -f schema/008_add_code_reviews.sql
   psql "$SUPABASE_DB_URL" -f schema/009_add_reviewer_version.sql
   psql "$SUPABASE_DB_URL" -f schema/010_add_review_patterns.sql
   psql "$SUPABASE_DB_URL" -f schema/011_add_review_progress_columns.sql
   psql "$SUPABASE_DB_URL" -f schema/015_create_agent_runs.sql
   ```
   The gaps in that numbering (no `001`-`007` or `012`-`014`) aren't missing
   files — these five are self-contained extracts from a longer internal
   migration series, and `008` creates the schema from scratch, so nothing
   earlier is required. Apply exactly the five files above, in order, and
   you have the complete schema.
3. Set the connection string and run Argus as usual — that one variable
   switches both round history and checkpointing to Postgres:
   ```bash
   export SUPABASE_DB_URL="postgresql://postgres.xxxx:pass@aws-0-region.pooler.supabase.com:6543/postgres"
   argus review owner/repo --pr 123
   ```
   (`SUPABASE_DB_URL` is accepted as an alias for `ARGUS_DB_URL` — if you're
   pointing at a non-Supabase Postgres instance, use `ARGUS_DB_URL` instead;
   they're equivalent, and if both are set `ARGUS_DB_URL` wins.)

### Recipe: HTTP storage backend

For environments that can reach HTTPS but not a raw Postgres port (e.g. a
sandboxed agent runtime with egress restricted to port 443), or for teams
that don't want to expose their database directly to wherever Argus runs:
route storage through your own HTTP API with `ARGUS_STORAGE_READ_URL` /
`ARGUS_STORAGE_WRITE_URL` (optionally `ARGUS_STORAGE_AUTH` for a bearer
token your service checks). The full two-endpoint contract — everything
you need to implement a compatible backend service — and mode-resolution
details are in [`docs/STORAGE.md`](docs/STORAGE.md).

## Adapting Argus to your team

Argus ships with a full set of production review prompts — 20 files, one
per reviewer role — rather than a thin default. Adapting them to your team
is the main lever for making Argus *yours*.

### Editing what Argus looks for

```bash
argus prompts export ~/.config/argus/prompts   # copy the packaged prompts out
$EDITOR ~/.config/argus/prompts/pr-review-subagent.md
argus review owner/repo --pr 123               # no env var needed — it's already found
```

That's the standing-customization path: an explicit `ARGUS_PROMPTS_DIR` (if
you need one), then a repo-local `./.argus/prompts/`, then
`~/.config/argus/prompts/`, are all checked automatically, highest priority
first — see
[`docs/CUSTOMIZING_PROMPTS.md`](docs/CUSTOMIZING_PROMPTS.md#recipe-exporting-editing-and-using-your-own-prompts)
for the full resolution chain and a `--no-prompt-overrides` escape hatch.

Any file you don't touch keeps loading from the package, so adapt
incrementally. Each prompt mixes two kinds of content — pull them apart as
you edit:

- **Transferable methodology** — silent-fallback detection, stub
  completeness, loud-failure-over-silent-failure philosophy, coverage
  mechanics, the severity rubric. This is *why* the reviewer is effective;
  keep it unless you have a specific reason not to.
- **House conventions from wherever the prompts were tuned** — references to
  specific shared libraries, a particular lint-config or `.cursorrules`-like
  compliance check, a specific secrets-manager's path conventions, a
  specific orchestration/LLM framework stack. This is the part that should
  become *your* team's equivalent — your shared libraries, your lint rules,
  your infra stack, the failure modes your codebase actually has.

Start with `pr-review-subagent.md` (the core reviewer prompt) since it has
the most coding-standards content per line. Leave the pipeline-mechanics
prompts (planner, coverage-check, writer) alone until you have evidence
they need to change — they shape *how* the pipeline runs, not *what* it
looks for. To evaluate any change, re-run the same PR before and after and
diff the findings. Full prompt-by-prompt anatomy:
[`docs/CUSTOMIZING_PROMPTS.md`](docs/CUSTOMIZING_PROMPTS.md).

### Adding a new specialist reviewer

Nine specialists ship today (security, SQL, infra, orchestration, frontend,
Slackbot integrations, deployment, LLM patterns, observability), assigned
by the planner based on which files a group touches. Editing an *existing*
specialist's prompt is just editing a file, as above.

Adding an **entirely new** specialist currently requires a small code
change, not just a new prompt file: the specialist name has to be added to
the `SpecialistName` type and the prompt-name mapping in `argus/runners.py`
(a couple of lines each), so the planner's structured output can validate
against it. This is a known rough edge — making the specialist roster
fully prompt-file-driven (drop a file in a `specialists/` folder, get a new
specialist, no code change) is tracked as a planned improvement. Check the
repo's issue tracker for its status before assuming you need to fork.

## Measured results

Argus's pipeline has gone through three iterations (v1/v2/v3 in its
internal history; v3 is what ships here). Measured across 22 real PRs:

| Metric | v1 (round 1) | v1 (all rounds) | v2 | **v3 (this engine)** |
|---|---|---|---|---|
| Avg blocking findings / PR | 1.6 | 4.1 | 2.7 | **5.0** |
| Single pass? | Yes | No (2-4 rounds) | Yes | **Yes** |
| Tracked cost / review | ~$0.75 | ~$2.25 | $2.93 | **$0.30*** |
| Time / review | ~5 min | ~20 min | 8.5 min | **9.6 min** |
| APPROVE rate | N/A | N/A | 0% | **4.5% (1/22)** |

\* This early tracked-cost figure covered only the planner + writer LLM
calls, not the Agent SDK subagent sessions that do most of the work — true
cost per review in that same 22-PR set is closer to $2-4, and larger
production samples put it at $6-42 depending on PR size. Full methodology:
see the [write-up](https://www.redesignhealth.com/content/agentic-code-review-harness).

## Docs

- [`docs/CUSTOMIZING_PROMPTS.md`](docs/CUSTOMIZING_PROMPTS.md) — full
  prompt-by-prompt anatomy for adapting Argus to your team.
- [`docs/STORAGE.md`](docs/STORAGE.md) — storage modes, schema, and the HTTP
  contract for self-hosters.
- [`docs/BUILDING_A_REVIEW_LOOP.md`](docs/BUILDING_A_REVIEW_LOOP.md) — how to
  wrap Argus in a review → fix → re-review automation loop gated on CI.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — the pipeline design and
  why it looks the way it does.
- [`docs/RELEASING.md`](docs/RELEASING.md) — how releases are cut, for
  contributors.
- [Agentic code review harness write-up](https://www.redesignhealth.com/content/agentic-code-review-harness) —
  design principles, production results, and a worked example of a full
  review round.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup and PR expectations,
[SECURITY.md](SECURITY.md) to report a vulnerability,
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community standards, and
[CHANGELOG.md](CHANGELOG.md) for a history of notable changes.

## License

[Apache License 2.0](LICENSE).

---

Built by [Redesign Health](https://www.redesignhealth.com). Named by Nathan
Marinoff, after [Argus](https://en.wikipedia.org/wiki/Argus_Panoptes), the
many-eyed giant of Greek mythology.
