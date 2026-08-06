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
pip install "argus-code-review[prechecks]"   # installs semgrep, zizmor, checkov
```

Without the extra installed, `argus.precheck.engine.run_precheck` no-ops —
the pipeline behaves exactly as if this feature didn't exist. Several stock
scanners (below) are standalone binaries or npm packages that `pip` cannot
install — each needs a separate one-time setup step wherever this package
runs live, and each degrades gracefully (skips its own scan, logs once) if
its step hasn't been done.

## Diff scoping

`run_precheck(worktree_path, changed_files=...)` scopes findings to files
the PR actually touched (`changed_files`, derived from the PR diff via
`helpers.extract_changed_files` — see `graph._node_precheck_rules`).
Several scanners below (Trivy, zizmor) scan the *whole* worktree, not just
the diff; without this scoping a repo with any pre-existing debt would
surface it on every PR, and `_MAX_RESULTS` would then truncate that flood
arbitrarily — silently dropping real, in-scope findings alongside the
noise. Others (squawk, Checkov, actionlint, eslint) additionally require
an explicit `changed_files` list to run *at all* (see each one's own
module docstring for why — a large migrations/Terraform history, or a
project-root requirement, make scanning the whole worktree impractical
regardless of post-hoc filtering).

`changed_files` has three distinct states, not two:

- **`None`** — no scoping requested; every scanner that can run
  whole-worktree does (semgrep/zizmor/Trivy), and the changed-files-only
  scanners (squawk/Checkov/actionlint/eslint) don't run at all. Used by
  this module's own tests and any future non-PR-context caller; the live
  per-PR gate never passes this.
- **A non-empty list** — the normal live-PR case: whole-worktree scanners'
  results are filtered down to files in the list, and the changed-files-only
  scanners run scoped to it.
- **`[]` (empty, not `None`)** — a full no-op: *no* scanner runs at all,
  including the always-on whole-worktree ones. This is the live per-PR
  gate's state whenever `helpers.extract_changed_files` finds nothing to
  scope to — in practice almost always a genuinely empty diff (e.g. a
  comment-triggered re-review with no new commits since the last round),
  though a diff whose only changed file(s) have a space in their path can
  also trigger it (the extraction regex can't match those). Logged at
  `WARNING` specifically because it silently disables the whole gate for
  the round.

## Scanner failure observability

This module stays fail-open by design (see the module docstring): a scanner
that crashes, times out, or produces unparseable output never blocks a
review, and its contribution to that round is silently treated as if it had
run clean with zero findings. That's the right call for a gate that must
never be the thing standing between a PR and its review — but "silently"
means a genuine coverage gap (the scanner never actually ran) was, until
recently, indistinguishable from "the scanner ran and found nothing."

`run_precheck` now names every scanner that returned `None` (a real
failure, not a clean `[]`) in `PrecheckResult.failed_scanners` — logged
once, loudly, inside `run_precheck` itself, and a second time via
`graph._node_precheck_rules` → `run_review`'s existing degraded-coverage
section (the same mechanism already used for a killed/timed-out LLM
reviewer session), so a scanner failure is visible in the rendered review
comment, not just a backend log line. Still purely observability: nothing
about the review's verdict, gating, or blocking behavior changes based on
this.

This does NOT close every gap, only the ones each scanner's own wrapper
already reports as `None`. semgrep is not tracked here at all —
`_run_semgrep_precheck` already collapses its own `None` into `[]` before
this aggregation ever sees it (see that function's docstring for why); its
own failures are only visible via the plain log line semgrep's own runner
already emits. Less obviously, squawk/actionlint's JSON-parse-failure
except clauses and Checkov's SARIF-parsing path all still return `[]`, not
`None`, on a genuinely malformed-but-successfully-exited run — only THEIR
OWN execution-error/exit-code/timeout failures are caught by this
observability layer today; a parse failure specifically is still silently
indistinguishable from "ran clean" for those three.

### Opting into fail-closed on scanner failure

Set `ARGUS_PRECHECK_BLOCK_ON_SCANNER_FAILURE=true` to make a scanner
failure force the verdict to `BLOCKING`, instead of the default (surface
it in the review comment's degraded-coverage section, but let the
review's own verdict stand on its own merits). Off by default: every
other part of this module is deliberately fail-open (see the module
docstring) — a broken scanner should never, by default, be the reason a
review can't complete. This flag exists for repos that have decided the
opposite tradeoff is worth it — e.g. one whose own CI treats a specific
scanner as a hard release gate, where "the gate silently didn't run" is a
worse outcome than "the review is blocked until it's fixed."

The gate only ever makes the verdict *stricter*: it never turns an
already-`BLOCKING` verdict back to `APPROVE`, and never fires at all when
there's no scanner failure regardless of the flag. It's also scoped to
precheck scanner failures specifically — a killed/timed-out LLM reviewer
session (a different, pre-existing failure class this flag was never
meant to cover) doesn't trigger it.

## Stock rule sources vs. custom/mined rules

`run_precheck` runs several independent scanners and merges their findings
into the same candidate/verified classification pipeline below. None
depends on the others being configured — a deployment that has never set
`ARGUS_RULES_DIR` still gets full benefit from every stock source:

1. **Custom or mined rules** (`ARGUS_RULES_DIR`, see "Rule files" below) —
   patterns *this specific codebase's own review history* has flagged
   repeatedly. The one thing no stock tool could know to check for.
2. **Semgrep registry packs** (`ARGUS_STOCK_SEMGREP_PACKS`, unset/off by
   default) — a comma-separated list of semgrep registry pack ids (e.g.
   `p/secrets`) merged into the same semgrep invocation as your custom
   rules via multiple `--config` flags. **Two caveats before enabling
   this:** (a) unlike a local rules directory, semgrep fetches each pack
   over the network on first use (cached afterward) — a real, if usually
   small, added latency/network dependency on the live per-PR gate; (b)
   the free/unauthenticated tier of a registry pack is noticeably smaller
   than what's actually in the registry (`p/secrets` loads ~37 generic
   rules without a `semgrep login` session vs. a much larger authenticated
   set) — verified empirically, not a guess, while building this feature.
   `semgrep login` support for a richer pack isn't wired up here; start
   with the free tier and revisit if it's not enough.
3. **zizmor** ([docs.zizmor.sh](https://docs.zizmor.sh), `pip install
   zizmor`, part of the `prechecks` extra) — always on when installed, no
   opt-in setting: a purpose-built, externally-maintained GitHub Actions
   security scanner (unpinned action tags, `${{ }}` script-injection,
   credential-persistence via artifacts, etc.). Runs fully offline
   (`--offline`, explicit — never depends on a GitHub token or makes
   outbound calls on the target repo's behalf) against the whole worktree;
   auto-discovers workflow/action files and no-ops cleanly (not an error)
   on a worktree with nothing to audit, which is the common case for most
   PRs. See `argus/precheck/actions_scanner.py`.
4. **Trivy secrets** (`trivy` binary — **not on PyPI**, install via
   `brew install trivy` or a
   [binary download](https://github.com/aquasecurity/trivy/releases);
   verified against 0.73.0) — always on when installed. Scoped to
   `--scanners secret` only: Trivy's misconfiguration scanner is
   deliberately NOT used (the specific check this integration originally
   wanted — generic Terraform IAM wildcard detection — turned out to be a
   dead/deprecated rule in Trivy's own default bundle, verified
   empirically; running it anyway would double-report several
   IAM/S3 conditions Checkov (below) already covers, on the same lines).
   Ships dozens of built-in vendor-specific secret regexes (AWS keys,
   GitHub tokens, Stripe keys) with no login/download step — a genuine
   improvement over semgrep's own thin `p/secrets` free tier. Has its own
   blind spot: well-known example/placeholder keys (e.g. AWS's own docs
   example key) are deliberately allowlisted to cut false positives from
   docs/tests. See `argus/precheck/secrets_scanner.py`.
5. **squawk** (Postgres migration linting — `squawk` binary, **npm-only**,
   install via `npm install -g squawk-cli`; verified against 2.61.0) —
   requires `changed_files`; only scans `.sql` files that were actually
   changed. Catches missing `CONCURRENTLY`/`IF NOT EXISTS`, unsafe column
   type changes, constraints added without `NOT VALID`, missing lock/
   statement timeouts. No SARIF reporter — this module parses squawk's
   `--reporter json` output directly (whose line numbers are 0-indexed,
   verified empirically against squawk's own TTY reporter — corrected to
   1-indexed before reaching the shared `SarifResult` shape). See
   `argus/precheck/migration_scanner.py`.
6. **Checkov** (Terraform IAM/privilege-escalation — `pip install checkov`,
   part of the `prechecks` extra; verified against 3.3.9) — requires
   `changed_files`; only scans `.tf`/`.tf.json` files that were actually
   changed, and only Checkov's own IAM-wildcard/privilege-escalation
   checks (`terraform_scanner._ALLOWED_CHECKS`) — Checkov's *default*
   catalog is far broader (verified empirically: a plain, otherwise-fine
   `aws_s3_bucket` resource failed 7 unrelated best-practice checks under
   the defaults), which would flood every Terraform PR with opinionated
   noise unrelated to the actual gap this integration exists for.
   **Exit-code convention is inverted** relative to every other scanner
   here: Checkov exits 0 only when clean and 1 when genuine findings exist
   (verified empirically) — the same inversion squawk has, handled by a
   shared `is_success_exit(..., findings_exit_code=1)` helper
   (`scanner_utils.py`), not a Checkov-specific special case. Also unlike
   every other scanner, Checkov writes SARIF only to a file
   (`--output-file-path`), never stdout — this module reads that file back
   after the subprocess completes. See `argus/precheck/terraform_scanner.py`.
7. **actionlint** (GitHub Actions syntax/shellcheck — **not on PyPI**,
   install via `brew install actionlint shellcheck`
   or a [binary download](https://github.com/rhysd/actionlint/releases);
   verified against 1.7.12, shellcheck 0.11.0) — requires `changed_files`;
   only scans changed `.github/workflows/*.yml`/`.yaml` files, invoked
   with explicit file paths (not directory auto-discovery, which requires
   a git-project root actionlint couldn't detect in an arbitrary worktree
   — verified empirically). Genuinely different from zizmor's security
   focus: catches typo'd action inputs (validated against the action's
   real input schema), expression syntax errors (`=` vs `==`), and — via
   its own built-in shellcheck integration, which degrades gracefully
   (verified empirically) if `shellcheck` isn't separately installed —
   real shell-scripting bugs inside `run:` blocks. See
   `argus/precheck/workflow_lint_scanner.py`.
8. **eslint-plugin-security** (JS/TS — **bundled**, not pip/npm-global;
   see `argus/precheck/eslint_bundle/README.md`'s one-time `npm install`
   setup step) — requires `changed_files`; only scans changed JS/TS files.
   Uses a dedicated bundled eslint config (`eslint_bundle/eslint.config.js`,
   invoked with `--no-config-lookup` so the target repo's own eslint setup
   is never merged in) applying only `eslint-plugin-security`'s recommended
   rules — independent of whether the reviewed repo has any JS/TS security
   linting configured at all, which is the actual gap this fills. See
   `argus/precheck/js_scanner.py`.

Every rule id from every source goes through the *exact same* status
lookup as a custom rule — a stock scanner's finding starts as `candidate`
(non-blocking, logged for triage) exactly like a brand-new custom rule
would, per the table below. Nothing from any of these three sources ever
fast-fails a PR on day one.

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

**Operator note:** two earlier versions of `run_semgrep_sarif` both caused
semgrep to namespace-prefix a rule's reported `ruleId`, defeating
`select_rule_statuses`'s DB lookup by the bare `rule_id`: the original
version passed semgrep's `--config` an absolute directory path from an
unrelated `cwd` (e.g. `foo` became `tmp.x.rules.foo` for *every* rule,
flat or nested), and the intermediate cwd-juggling fix (`cwd=config_path`,
`--config "."`) narrowed this but still namespaced any rule organized into
a subdirectory (e.g. `rules/security/foo.yml`'s `foo` became
`security.foo`). Both meant no rule under either affected layout could
ever functionally reach `verified` status. `log_candidate_firings`
auto-creates a `precheck_rules` row (via its `ON CONFLICT DO NOTHING`
ensure-row insert) under whatever `rule_id` a firing reports on *every*
candidate firing, not only on rules a human happened to manually triage —
so any repo that ran this feature before the fix likely already has rows
keyed by one of these namespaced ids. Those rows are orphaned going
forward: the bare id the fixed code now reports is a different row that
starts back at `candidate`. Given `verified` was unreachable anyway, this
is a one-time acceptable reset rather than a migration to write; find any
such rows with:

```sql
SELECT rule_id FROM review_service.precheck_rules WHERE rule_id LIKE '%.%';
```

**This is a starting point for manual inspection, not a to-do list:**
semgrep registry-style rules commonly use dotted ids by convention (e.g.
`python.lang.security.audit.something`), so a match here isn't
automatically a namespaced-by-the-bug id -- cross-reference each match's
`rule_id` against the actual `id:` values in your rule files first.

Even once you've identified a genuinely orphaned row, **don't bother
deleting it, and this doc previously recommended doing so incorrectly:**
`service_role` (the role this package's own code runs as) is granted only
`SELECT, INSERT, UPDATE` on both tables (see
`schema/017_add_precheck_rules.sql`'s GRANT statements), not `DELETE` --
though a table-owner or superuser connection (e.g. via the Supabase SQL
editor) bypasses grants entirely, so the grant itself isn't a hard blocker
for an operator with that level of access. The real obstacle either way is
`precheck_candidate_firings.rule_id`'s foreign key, which has no
`ON DELETE` clause: any orphaned row created through this package's own
normal write path (`log_candidate_firings`) has at least one dependent
firing row and can't be deleted without deleting those first -- though a
row seeded some other way (e.g. by a future out-of-band triage job) isn't
guaranteed to have one. Leaving an orphaned row in place is harmless
regardless -- it's an inert `candidate`-status row no rule file's current
`id:` ever matches again, so nothing fires under it -- and is the
recommended outcome rather than a cleanup task.

**Correction:** an earlier revision of this note called any dotted
`rule_id` match "safe to delete." That was wrong for two reasons: the
registry-style-id false-positive risk above, and because a row created
through this package's own write path can't be deleted without deleting
its dependent firing row(s) first -- an ordinary `DELETE ... WHERE
rule_id = ...` against `precheck_rules` alone would have failed outright
on the FK, not silently succeeded. If you already attempted this with a
connection that also had `DELETE` on `precheck_candidate_firings` (broader
than `service_role`'s own grant) and deleted the dependent rows first, a
legitimate registry-style rule's row could genuinely have been removed --
re-verify your `precheck_rules` table against your rule files' actual
`id:` values regardless.

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
- `entries_failed` — entries where semgrep didn't run to completion on
  that entry (timeout, its own execution error) or the corpus entry
  itself couldn't be provisioned (bad SHA, clone error). This is
  corpus/infra flakiness, not rule-precision evidence — it's deliberately
  excluded from `entries_scanned`/`entries_matched` so a malformed
  candidate rule that fails to execute on every entry can't produce a
  confident-looking zero-occurrence result that reads as "this rule is
  safe" when it never actually ran. A missing semgrep binary is *not* one
  of these per-entry cases: `run_shadow_review` checks for it once before
  touching the corpus at all and raises immediately, since that's a
  systemic setup problem, not something worth diagnosing per entry after
  paying for a full clone each.

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

`tests/test_precheck_shadow_integration.py` exercises this harness for
real (a real clone, real semgrep) rather than mocking `provisioned_worktree`
the way `tests/test_precheck_shadow.py` does. It's `pytest.mark.integration`
plus a dedicated `pytest.mark.needs_real_github_token` marker, and is
deliberately **not** wired into CI (see `CONTRIBUTING.md`) since it needs a
real `GITHUB_TOKEN_RO`, not the fixed stub `tests/conftest.py` sets for
every other test. Run it locally with
`GITHUB_TOKEN_RO=$(gh auth token) uv run pytest -m integration tests/test_precheck_shadow_integration.py`.

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
