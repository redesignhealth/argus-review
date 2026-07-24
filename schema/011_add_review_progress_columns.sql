-- Add current_stage column and nullable verdict for progress tracking.
-- In-flight progress (stage, reviewers_completed) is derived at query time
-- from LangGraph's public checkpoint schema — not from custom columns.
--
-- DEPLOY ORDER: This migration MUST be applied before deploying the code
-- that does the initial INSERT with verdict=NULL.
--
-- IDEMPOTENT: Safe to re-run. All operations use IF NOT EXISTS / IF EXISTS
-- guards or are naturally idempotent (DROP NOT NULL, DELETE with WHERE).

BEGIN;

-- Allow initial INSERT before pipeline completes (verdict/comment unknown yet)
ALTER TABLE review_service.code_reviews
    ALTER COLUMN verdict DROP NOT NULL,
    ALTER COLUMN review_comment DROP NOT NULL;

-- Pipeline stage written by graph.py at start ('running') and end ('completed')
ALTER TABLE review_service.code_reviews
    ADD COLUMN IF NOT EXISTS current_stage TEXT NOT NULL DEFAULT 'pending';

-- Restrict current_stage to known values (idempotent via DO block)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'ck_code_reviews_current_stage_valid'
            AND conrelid = 'review_service.code_reviews'::regclass
    ) THEN
        ALTER TABLE review_service.code_reviews
            ADD CONSTRAINT ck_code_reviews_current_stage_valid
            CHECK (current_stage IN ('pending', 'running', 'planning', 'reviewing', 'writing', 'completed'));
    END IF;
END $$;

-- Deduplicate existing rows before adding unique constraint.
-- Keeps the most recent row per flow_run_id.
DELETE FROM review_service.code_reviews a
USING review_service.code_reviews b
WHERE a.flow_run_id = b.flow_run_id
  AND a.flow_run_id IS NOT NULL
  AND a.created_at < b.created_at;

-- Ensure one progress row per flow run (enables safe upsert on retry).
-- Supersedes the non-unique idx_code_reviews_flow_run from migration 008.
CREATE UNIQUE INDEX IF NOT EXISTS uq_code_reviews_flow_run_id
    ON review_service.code_reviews (flow_run_id)
    WHERE flow_run_id IS NOT NULL;

DROP INDEX IF EXISTS review_service.idx_code_reviews_flow_run;

COMMIT;
