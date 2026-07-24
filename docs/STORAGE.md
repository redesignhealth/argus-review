# Storage

Argus persists two things across a run:

1. **Round history** — the per-round result (verdict, findings, cost,
   timing, full structured output) for a given `(repo, pr_number)`, so
   round 2+ can verify whether prior findings were resolved instead of
   re-reviewing the whole PR from scratch.
2. **Pipeline checkpoints** — LangGraph's own state, so a crashed or
   interrupted run can, in principle, resume rather than losing all
   progress.

Three backends are supported for round history; two for checkpoints. Which
one is active is resolved entirely from environment variables — there is no
separate `--backend` flag to set.

## Mode resolution

**Round history:**

1. `ARGUS_DB_URL` (or its alias `SUPABASE_DB_URL`) is set → **Postgres**.
2. Else, both `ARGUS_STORAGE_READ_URL` and `ARGUS_STORAGE_WRITE_URL` are
   set → **HTTP**. Setting only one of the pair is a startup error — they
   are validated together, not independently.
3. Else → **local SQLite** (the default; no configuration required).

**Checkpoints** (decoupled from the history mode choice):

1. `ARGUS_DB_URL` is set → `AsyncPostgresSaver`.
2. Else → `AsyncSqliteSaver`, writing to `ARGUS_SQLITE_CHECKPOINT_PATH` if
   set, otherwise a per-run temp file that is cleaned up on exit.

This means HTTP-mode round history and SQLite checkpoints is a normal,
supported combination — HTTP mode exists specifically for environments
(such as a sandboxed agent runtime) that can reach an HTTPS endpoint but
cannot open a direct Postgres connection on port 5432.

## Mode 1: Postgres

Round history lives in two tables under the `review_service` schema:
`code_reviews` (one row per round) and `agent_runs` (one row per reviewer
session within a round — system reviewer, specialist, cross-cutting, etc.,
with token cost, duration, tool-call count, and files explored).

The canonical schema is in `schema/*.sql` in this repo. Apply them in order:

- `schema/008_add_code_reviews.sql` — creates the `review_service` schema
  and the `code_reviews` table (verdict, risk_level, blocking_count,
  suggestion_count, review_comment, result_json JSONB, cost_usd,
  duration_seconds, model info, sha/base_ref, agent_trace JSONB,
  created_at), plus indexes on `(repo, pr_number)`, `flow_run_id`,
  `created_at`, and `sha`.
- `schema/009_add_reviewer_version.sql` — adds `reviewer_version` (`v3` /
  `v3-lite`) so multiple engine generations can coexist in one table.
- `schema/010_add_review_patterns.sql` — a separate `review_patterns` table
  for weekly aggregate pattern analysis. This supports a weekly feedback
  loop that isn't part of this toolkit (see the
  [write-up](https://www.redesignhealth.com/content/agentic-code-review-harness)
  for the concept); the table is not read or written by the core Argus
  pipeline itself, so self-hosters can skip it if you don't intend to
  build a similar aggregation job.
- `schema/011_add_review_progress_columns.sql` — adds `current_stage`
  (`pending` → `running` → `planning` → `reviewing` → `writing` →
  `completed`) and makes `verdict`/`review_comment` nullable so a
  "running" row can be inserted before the pipeline finishes. Also adds
  the partial unique index on `flow_run_id` that the upsert logic depends
  on. **Must be applied before deploying code that does the initial
  running-state INSERT** — it's a hard ordering dependency, not just a
  convenience migration.
- `schema/015_create_agent_runs.sql` — creates `agent_runs`, one row per
  sub-agent per review round, foreign-keyed to `code_reviews.id` with
  `ON DELETE CASCADE`.

The numbering (008-011, 015) is not a typo or a sign of missing
prerequisites — these files are extracted from a longer internal migration
sequence, and `008_add_code_reviews.sql` creates the `review_service` schema
from scratch, so nothing before it in that original sequence is needed
here. Apply the five files above, in order, and you have the complete
Postgres schema this pipeline needs.

The write path is idempotent on `flow_run_id` via the partial unique index
(`WHERE flow_run_id IS NOT NULL`): a "running" row is upserted at the start
of a review and finalized in place at the end, rather than inserting two
separate rows. In-sandbox runs driven without an external orchestrator have
no `flow_run_id` and simply insert without hitting the conflict path.

One documented, intentional deviation from a naive "just overwrite the row"
upsert: the finalize step uses
`sha = COALESCE(EXCLUDED.sha, code_reviews.sha)` (and the same for
`base_ref`) instead of an unconditional overwrite, so a caller that finalizes
with `sha=None` (the no-orchestrator / agent-storage path) never clobbers a
previously-bound `sha`. The invariant this preserves — "`sha` is never
cleared once bound" — is why the round-history read queries don't need to
separately filter on `sha IS NOT NULL`.

## Mode 2: HTTP

For runtimes that can't open a direct Postgres connection (for example, a
sandboxed agent execution environment with egress restricted to port 443),
round history goes over two plain HTTP calls to a backend service you run
yourself. This is a **minimum viable contract** — exactly the two
operations the pipeline needs, nothing more.

**GET — read the latest completed round**

- URL: your configured `ARGUS_STORAGE_READ_URL`, with `{owner}`, `{repo}`,
  and `{pr}` substituted (plain string replacement, not `str.format` — so
  unrelated `{...}` segments in the URL are left alone).
- Headers: `Accept: application/json`; `X-API-Key: <ARGUS_STORAGE_AUTH>` if
  that env var is set.
- Response body: `{"rounds": [ <round>, <round>, ... ]}`, ordered newest
  first (`created_at DESC`), capped at 200 rounds. Argus reads `rounds[0]`
  as the latest completed round and uses `len(rounds)` as the round-number
  hint (approximate once a PR's history exceeds the 200-row cap — in
  practice a single convergence loop runs on the order of 10 rounds, so this
  is not a real-world limitation).
- An empty `rounds` list means "no completed rounds yet" (round 1 case).

**POST — write a new round**

- URL: your configured `ARGUS_STORAGE_WRITE_URL`, same substitution rules.
- Headers: same as above, plus `Content-Type: application/json` implied by
  the JSON body.
- Body: the round payload, JSON-serialized (UUIDs and datetimes as
  strings). Matches the same column set as the Postgres `code_reviews` row:
  `flow_run_id`, `repo`, `pr_number`, `verdict`, `risk_level`,
  `blocking_count`, `suggestion_count`, `review_comment`, `result_json`,
  `cost_usd`, `duration_seconds`, `reviewer_version`, `orchestrator_model`,
  `subagent_model`, `sha`, `base_ref`, `current_stage`.
- The client validates that the payload's `repo`/`pr_number` match the URL
  path before sending — a mismatch is rejected client-side rather than
  waiting for your server to 400 on it.
- Response: the persisted round, echoing the same shape back plus a
  server-assigned `id` (UUID) and `created_at` (timestamp).

**The documented `agent_runs` skip.** This HTTP shim covers round history
only. It does **not** have an equivalent for the Postgres `agent_runs`
table (per-agent cost/duration/tool-call analytics). Runs that go through
HTTP-mode storage skip that insert entirely — analytics for those runs are
lost, not degraded. This is a scoped, accepted limitation of the minimal
shim, not a bug; if you're implementing your own backend and want
per-agent analytics, you'll need to add that endpoint yourself (there's no
existing contract to match against because Argus's own HTTP path doesn't
call it).

If you want to implement a compatible backend service, these two endpoints
(and the `CodeReviewRound` / `CodeReviewRoundRecord` / `ListReviewRoundsResponse`
shapes in `argus/storage/models.py`) are the entire contract — nothing else
in the pipeline needs to know your backend exists.

## Mode 3: Local SQLite (default)

When neither Postgres nor HTTP is configured, Argus falls back to a local
SQLite database at `~/.local/share/argus/history.db` for round history, and
a SQLite file for LangGraph checkpoints (a per-run temp file by default, or
`ARGUS_SQLITE_CHECKPOINT_PATH` if you want it pinned to survive across
runs — for example, to inspect a failed pipeline's mid-run state).

This is the zero-configuration path: `argus review owner/repo --pr 123`
with only `ANTHROPIC_API_KEY`, `GITHUB_TOKEN_RO`, and `OPENAI_API_KEY` set
works end to end, including round-2 verification of round-1 findings,
without touching Postgres or an HTTP backend at all.

`argus/storage/sqlite.py` implements this. It runs the same seven logical
operations as the Postgres path (`select_latest_completed_round`,
`select_recent_rounds`, `select_recent_lite_rounds`, `upsert_running_row`,
`upsert_completed_row`, `insert_agent_runs`, `select_status_by_flow_run`)
against a single SQLite file, bootstrapped on first use from DDL that
mirrors `schema/008` + `schema/009` + `schema/011` + `schema/015` (the
`review_patterns` table from `schema/010` is out of scope — see the note
above). JSONB columns become `TEXT` holding JSON-encoded strings, `TEXT[]`
array columns likewise, and the Postgres partial unique index
(`WHERE flow_run_id IS NOT NULL`) is created verbatim — SQLite has
supported partial indexes (and using them as an `ON CONFLICT` target)
since 3.24 / 3.35 respectively. Reads and writes go through stdlib
`sqlite3` on a single connection, serialized with an `asyncio.Lock` and
dispatched via `asyncio.to_thread` — no extra dependency, and the
single-developer-machine access pattern this mode serves doesn't need a
dedicated async driver.

## Choosing a mode

- **Running Argus by hand or in CI, no team-shared history needed?** Do
  nothing — local SQLite just works.
- **Want durable, queryable round history shared across a team, or you're
  already running Postgres?** Set `ARGUS_DB_URL` and apply `schema/*.sql`.
- **Running inside a sandbox that can only reach the network over HTTPS on
  443?** Stand up a small backend implementing the two-endpoint contract
  above and set `ARGUS_STORAGE_READ_URL` / `ARGUS_STORAGE_WRITE_URL` (and
  `ARGUS_STORAGE_AUTH` if you want it authenticated).
