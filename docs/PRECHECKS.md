# Deterministic prechecks

A non-LLM gate that runs before Argus's LLM pipeline spends any tokens.
Two distinct things, both fail-open (a precheck problem never blocks or
slows a review) but different in how "optional" applies to each:

1. **CI status as a routing signal.** Always on, no extra to install —
   this is a signal that can be ignored, but the read itself isn't gated
   behind anything. `argus.graph.run_preflight_check`
   reads the target repo's own GitHub Checks status for the PR's head
   commit and passes it to the lite-vs-full routing decision as one more
   input — never a hard gate. Argus deliberately does **not** re-run the
   target repo's own linters/tests (ruff, eslint, mypy, gitleaks, whatever
   CI that repo already has): a repo worth reviewing almost always already
   runs those on every PR, and re-executing them inside Argus would just
   duplicate a gate that repo already has, for real engineering cost and
   near-zero marginal catch rate. It's also unsafe to treat as an
   authoritative gate on its own — a PR can edit its own workflow file to
   force its checks green, fork PRs from first-time contributors may have
   no checks running at all, and Argus typically starts review while
   checks are still `queued`/`in_progress` — hence "signal", not "gate".

2. **Custom rules** — `argus.precheck` — semgrep rules specific to patterns
   *Argus itself* has previously flagged in review, which is the one thing
   no repo's own generic CI could ever know to check for. This is what the
   rest of this doc covers.

## Enabling it

```bash
pip install "argus-code-review[prechecks]"   # installs semgrep
```

Without the extra installed, `argus.precheck.engine.run_precheck` no-ops —
the pipeline behaves exactly as if this feature didn't exist.

## Rule files

Semgrep YAML rules go in a directory resolved the same "explicit override
wins" way `ARGUS_PROMPTS_DIR` resolves prompts, but simpler — a single
directory whose `*.yml`/`*.yaml` files are used wholesale (rules aren't
looked up by a fixed name, so there's nothing to merge across locations):

1. `ARGUS_RULES_DIR`, if set.
2. The packaged `argus/precheck/rules/` directory — **empty by default**.
   See `argus/precheck/rules/README.md` for an example rule.

**Footgun:** these two steps are not a fallback chain past step 1 — a
*set-but-invalid* `ARGUS_RULES_DIR` (pointing at a path that isn't a
directory) returns `None` outright and disables the gate for that run
entirely; it does not fall through to the packaged rules. Explicitly
setting the override always wins, even when it's wrong.

## Rule status lifecycle

A rule's `id:` field is the key used to look up its status in
`review_service.precheck_rules` (`schema/017_add_precheck_rules.sql`). The
database is the single source of truth for status; rule files only supply
pattern content:

| Status | Behavior |
|---|---|
| *(no row yet)* | Treated as `candidate` — the safe default for a brand-new rule. |
| `candidate` | Findings attach to pipeline state as non-blocking context for the writer. Every firing is queued (`review_service.precheck_candidate_firings`) for later, out-of-band triage. |
| `verified` | Findings fast-fail the PR before any LLM step runs — zero LLM spend. |
| `suspended` | Findings are dropped entirely. |

**What graduates a rule from `candidate` to `verified` (or flips it to
`suspended`) is RH-internal infrastructure, not part of this package** —
a narrower split than `schema/010_add_review_patterns.sql`'s weekly job,
not the same one: the read/write path itself (`select_rule_statuses`,
`log_candidate_firings`) ships and runs in-package every round; only the
status-transition logic and the rule-mining job are external (see
[`docs/STORAGE.md`](STORAGE.md) for the exact boundary). In short: an
out-of-band job clusters
recurring Argus findings into candidate rule drafts (human-approved before
they ship), and a separate async job judges each candidate firing's
true/false-positive rate over time, graduating or suspending rules based on
a measured precision bound rather than a human re-reviewing every hit.

Nothing in this package ever touches `updated_at` on `precheck_rules`
after row creation (the ensure-row insert is `ON CONFLICT DO NOTHING`) —
that column is owned entirely by the out-of-band triage job described
above, for the future implementer of that job to update on every status
transition.

### Shadow-review harness

Before a candidate rule draft is allowed to fire on any live PR at all,
`argus.precheck.shadow.run_shadow_review` can validate it against a
corpus of historical PRs — entirely offline and out-of-band, not a node
in the live per-PR graph (`argus/graph.py` has no reference to it). It
checks out each corpus entry into its own worktree (reusing
`repo_provision.provisioned_worktree`), runs the rule via the same
`argus.precheck.engine.run_semgrep_sarif` the live gate uses, and returns
raw occurrence evidence:

- `entries_scanned` — corpus entries semgrep actually completed a scan on.
- `entries_matched` — of those, how many had at least one hit.
- `hits` — every individual match, attributed back to its corpus entry.
- `entries_failed` — entries where semgrep didn't run to completion at
  all (missing binary, timeout, its own execution error) or the corpus
  entry itself couldn't be provisioned (bad SHA, clone error). This is
  corpus/infra flakiness, not rule-precision evidence — it's deliberately
  excluded from `entries_scanned`/`entries_matched` so a malformed
  candidate rule that fails to execute on every entry can't produce a
  confident-looking zero-occurrence result that reads as "this rule is
  safe" when it never actually ran.

It does **not** judge true/false positive itself, and does not consult
`precheck_rules`' DB status at all (a draft has no row yet, and shadow
review isn't the live gate). That judgment — human approval of the
draft, and later the RH-internal async triage loop — is a separate,
external step built on top of this harness's output. Same
OSS-ships-the-mechanism / RH-internal-owns-the-decision split as
everything else here.

Reuses `repo_provision`'s bare-mirror cache the same way the live gate
does, which is a deliberate tradeoff worth knowing about here: the live
gate only ever touches one repo per review, but a shadow-review corpus
spanning many distinct repos will accumulate one full bare mirror per
distinct repo with no automatic cleanup. Fine for occasional use; worth
revisiting with explicit cache management if shadow review becomes a
routine, large-multi-repo workflow.

## Pipeline placement

```
fetch_diff -> (precheck_checks, precheck_rules) -> precheck_join
    -> (precheck_fail | early_verifier) -> preflight -> ...
```

`precheck_checks` (GitHub Checks API) and `precheck_rules` (semgrep vs.
the worktree) run in parallel — independent work, no reason to sequence
them — and fan in at `precheck_join` before the routing decision.
`precheck_rules` runs against the same worktree already checked out for
the review (no extra provisioning). See `argus.graph._node_precheck_checks`
and `_node_precheck_rules`'s docstrings for the full node contracts.
