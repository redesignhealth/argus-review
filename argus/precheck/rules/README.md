# Deterministic-precheck rules

Empty by default. Semgrep rule files (`*.yml`/`*.yaml`) placed here — or in
the directory pointed at by `ARGUS_RULES_DIR`, which takes priority over
this one — are picked up by `argus.precheck.engine.run_precheck` and run
against the PR worktree before the LLM pipeline runs.

**Before authoring a custom rule here, check whether a stock source
already covers it** — semgrep registry packs (`ARGUS_STOCK_SEMGREP_PACKS`),
zizmor and actionlint (GitHub Actions), Trivy (secrets), squawk (Postgres
migrations), Checkov (Terraform IAM), and eslint-plugin-security (JS/TS)
all run independently of this directory and may already catch what you're
about to hand-write, especially for well-known idioms (unpinned
dependencies, hardcoded secrets, GitHub Actions script injection, unsafe
migrations). This directory's real value is patterns specific to *this
codebase's own* recurring mistakes, not things a general-purpose,
externally-maintained tool already solves. See
`docs/PRECHECKS.md`'s "Stock rule sources vs. custom/mined rules" section.

## Status lifecycle

A rule's `id:` field is the key `argus.storage.precheck` uses to look up its
status in the `review_service.precheck_rules` table (schema/017). The
database is the single source of truth for status — this directory only
supplies pattern content.

**`id:` must be globally unique across every rule file you add, not just
unique within a directory.** `select_rule_statuses` keys purely off the bare
id string in one flat table — semgrep's own directory-based namespacing is
deliberately disabled (`--no-rewrite-rule-ids`, so nested rule directories
work correctly; see `argus/precheck/engine.py`), which also means two rules
in different subdirectories that happen to share an `id:` are
indistinguishable to the status lookup. A new rule that accidentally reuses
an already-`verified` rule's id would inherit `verified` (fast-fail) status
immediately, skipping the shadow-review/triage process entirely.
`argus.precheck.engine.run_precheck` runs a load-time lint on every
invocation (`_find_duplicate_rule_ids`) that logs a loud `WARNING` listing
every colliding `id:` and the files it appears in -- advisory only (it
never blocks a review, matching this whole module's fail-open design), so
normal code review before merging a rule file change is still the actual
enforcement mechanism; the lint is a backstop for catching a collision a
reviewer missed, not a substitute for review.

Status:

- **No row in the database** (the common case for a brand-new rule): treated
  as `candidate`. Findings are attached to pipeline state as non-blocking
  context for the writer — never gate a PR.
- **`verified`**: findings from this rule fast-fail the PR before any LLM
  step runs.
- **`suspended`**: findings are dropped entirely.

Rules graduate from `candidate` to `verified` (or get `suspended`) through
an out-of-band triage process that is not part of this package — same split
as `schema/010_add_review_patterns.sql`'s weekly job. See `docs/STORAGE.md`.

## Example rule

Not shipped here (this directory ships empty) — for illustration only:

```yaml
rules:
  - id: no-hardcoded-admin-password
    languages: [python]
    severity: ERROR
    message: >
      Hardcoded credential literal assigned to a variable that looks like
      an admin/root password. Use a secrets manager or environment
      variable instead.
    patterns:
      - pattern: $VAR = "..."
      - metavariable-regex:
          metavariable: $VAR
          regex: (?i)(admin|root).*(password|passwd|pwd)
```
