-- Deterministic-precheck rule lifecycle + candidate-finding queue.
--
-- Supports a gate that runs custom, non-LLM static-analysis rules (semgrep)
-- against the PR worktree before the LLM pipeline spends any tokens. Rules
-- start in 'candidate' status (findings attached to pipeline state as
-- non-blocking context only) and graduate to 'verified' (findings fast-fail
-- the PR before the LLM steps run) once an out-of-band triage process
-- confirms low false-positive rate. A rule can be 'suspended' if it starts
-- producing bad judgments after graduating.
--
-- Source-of-truth split: this table is authoritative for rule STATUS.
-- Rule pattern content lives in files (semgrep YAML), keyed by each file's
-- own `id:` field, which is the same string as this table's rule_id. A rule
-- file with no matching row here has no status yet and is treated as
-- 'candidate' by the precheck engine -- i.e. it can fire and attach
-- findings, but it can never block a PR until a row exists here with
-- status='verified'. This avoids needing to keep a status field embedded
-- in the rule file itself in sync with this table.
--
-- The out-of-band job that judges candidate firings and flips status here
-- is not part of this toolkit -- a narrower split than schema/010's
-- review_patterns, not the same one: review_patterns is read/written only
-- by its external job (the core pipeline never touches it), whereas the
-- core pipeline reads and writes THESE two tables directly every round --
-- only the status-transition logic and the rule-mining job are external.
-- See docs/STORAGE.md for the exact boundary.

CREATE TABLE IF NOT EXISTS review_service.precheck_rules (
    rule_id TEXT PRIMARY KEY,        -- matches the rule file's own `id:` field
    status TEXT NOT NULL DEFAULT 'candidate',
    occurrence_count INTEGER NOT NULL DEFAULT 0,
    agreement_count INTEGER NOT NULL DEFAULT 0,
    disagreement_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    CONSTRAINT chk_precheck_rules_status CHECK (status IN ('candidate', 'verified', 'suspended'))
);

-- Queue of individual candidate-rule firings awaiting async triage judgment.
-- Written synchronously (cheap insert) by the in-pipeline precheck node;
-- read and updated only by the out-of-band triage job, never by the live
-- per-PR pipeline -- keeps the per-PR path fast regardless of triage
-- backlog size.
CREATE TABLE IF NOT EXISTS review_service.precheck_candidate_firings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rule_id TEXT NOT NULL REFERENCES review_service.precheck_rules(rule_id),
    repo TEXT NOT NULL,
    pr_number INTEGER NOT NULL,
    head_sha TEXT NOT NULL,
    finding JSONB NOT NULL,          -- normalized SARIF result for this firing
    judgment TEXT,                   -- NULL until triaged: 'agree' or 'disagree'
    judged_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now(),
    CONSTRAINT chk_precheck_firings_judgment CHECK (judgment IS NULL OR judgment IN ('agree', 'disagree'))
);

-- Triage job scans unjudged firings in creation order.
CREATE INDEX IF NOT EXISTS idx_precheck_firings_unjudged
    ON review_service.precheck_candidate_firings (created_at)
    WHERE judgment IS NULL;

-- Precheck node looks up a rule's status by id on every PR; keep it cheap
-- even though rule_id is already the primary key (belt-and-suspenders for
-- the common "status = 'verified'" filtered scan some callers may prefer
-- over a per-id lookup, e.g. loading the full verified allowlist at once).
-- Not queried by anything in this PR (select_rule_statuses looks up by
-- rule_id, already PK-covered) -- exists for the out-of-scope triage job's
-- status-scan queries (e.g. "load every verified rule"), same category of
-- forward-looking index as review_patterns' own out-of-scope-job support.
CREATE INDEX IF NOT EXISTS idx_precheck_rules_status
    ON review_service.precheck_rules (status);

-- Both tables get UPDATE even though nothing in this repo issues one yet:
-- precheck_rules' status/counters and precheck_candidate_firings' judgment/
-- judged_at are both written by the future out-of-band triage job running
-- as this same role, not by anything here -- granted preemptively so that
-- job doesn't hit permission-denied on day one.
GRANT SELECT, INSERT, UPDATE ON review_service.precheck_rules TO service_role;
GRANT SELECT, INSERT, UPDATE ON review_service.precheck_candidate_firings TO service_role;
