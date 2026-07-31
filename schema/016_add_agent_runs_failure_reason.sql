-- Add failure_reason to review_service.agent_runs.
--
-- SessionResult/AgentRunData already carry failure_reason ("timeout" or
-- "worker_crashed") in-memory, surfaced in the review comment's degraded-
-- coverage banner -- but AgentRunIn (argus/storage/sql.py) never persisted
-- it, so the analytics table has no durable way to distinguish a timed-out
-- or crashed reviewer from one that ran fine.
--
-- Nullable, no default beyond NULL: NULL means the run completed normally,
-- matching the in-memory contract (None = success, non-None string = which
-- failure mode).
--
-- IMPORTANT -- deploy ordering: apply to every environment BEFORE the code
-- that constructs AgentRunIn(failure_reason=...) goes live.

ALTER TABLE review_service.agent_runs
    ADD COLUMN IF NOT EXISTS failure_reason TEXT;

-- Added as a separate idempotent block, not inline on ADD COLUMN: PostgreSQL
-- treats the whole ADD COLUMN IF NOT EXISTS statement as a no-op once the
-- column exists, which would silently skip an inline CONSTRAINT clause on
-- any re-run after the column has already landed in an environment -- same
-- DO/EXCEPTION pattern this file's own chk_agent_type constraint uses.
-- NOT VALID + a separate VALIDATE CONSTRAINT (rather than a single validated
-- ADD CONSTRAINT) mirrors the two-phase pattern schema/015's chk_agent_type
-- uses: it takes only a brief ACCESS EXCLUSIVE lock to add the constraint,
-- then validates existing rows in a second pass that only needs a lighter
-- lock -- avoiding holding the heavier lock for the full table scan.
DO $$ BEGIN
    ALTER TABLE review_service.agent_runs
        ADD CONSTRAINT agent_runs_failure_reason_check
        CHECK (failure_reason IS NULL OR failure_reason IN ('timeout', 'worker_crashed'))
        NOT VALID;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

ALTER TABLE review_service.agent_runs VALIDATE CONSTRAINT agent_runs_failure_reason_check;
