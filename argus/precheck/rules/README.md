# Deterministic-precheck rules

Empty by default. Semgrep rule files (`*.yml`/`*.yaml`) placed here — or in
the directory pointed at by `ARGUS_RULES_DIR`, which takes priority over
this one — are picked up by `argus.precheck.engine.run_precheck` and run
against the PR worktree before the LLM pipeline runs.

## Status lifecycle

A rule's `id:` field is the key `argus.storage.precheck` uses to look up its
status in the `review_service.precheck_rules` table (schema/017). The
database is the single source of truth for status — this directory only
supplies pattern content:

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
