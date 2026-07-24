-- Migration: 015_create_agent_runs
-- Creates the agent_runs table for per-agent execution data in Argus reviews.
-- One row per sub-agent per review round.

CREATE SCHEMA IF NOT EXISTS review_service;

CREATE TABLE IF NOT EXISTS review_service.agent_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    code_review_id UUID NOT NULL
        REFERENCES review_service.code_reviews(id) ON DELETE CASCADE,
    agent_name TEXT NOT NULL,
    agent_type TEXT NOT NULL
        CHECK (agent_type IN ('system', 'specialist', 'cross_cutting', 'tests_and_docs', 'feedback_verifier', 'blocking_validator')),
    model TEXT,
    cost_usd NUMERIC(12, 8) DEFAULT 0,
    duration_seconds NUMERIC DEFAULT 0,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    tool_call_count INTEGER DEFAULT 0,
    tool_names TEXT[],
    context7_call_count INTEGER DEFAULT 0,
    files_explored TEXT[],
    finding_count INTEGER DEFAULT 0,
    result_text_length INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_agent_runs_review
    ON review_service.agent_runs(code_review_id);

CREATE INDEX IF NOT EXISTS idx_agent_runs_review_type
    ON review_service.agent_runs(code_review_id, agent_type);

-- Idempotent schema convergence (handles environments where 015 was already applied
-- with different column types or missing constraints)
DROP INDEX IF EXISTS review_service.idx_agent_runs_agent_type;

ALTER TABLE review_service.agent_runs
    ALTER COLUMN cost_usd TYPE NUMERIC(12, 8) USING cost_usd::NUMERIC(12, 8),
    ALTER COLUMN duration_seconds TYPE NUMERIC USING duration_seconds::NUMERIC;

DO $$ BEGIN
    ALTER TABLE review_service.agent_runs
        DROP CONSTRAINT IF EXISTS agent_runs_code_review_id_fkey;
    ALTER TABLE review_service.agent_runs
        ADD CONSTRAINT agent_runs_code_review_id_fkey
        FOREIGN KEY (code_review_id) REFERENCES review_service.code_reviews(id) ON DELETE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    ALTER TABLE review_service.agent_runs
        ADD CONSTRAINT chk_agent_type
        CHECK (agent_type IN ('system', 'specialist', 'cross_cutting', 'tests_and_docs', 'feedback_verifier', 'blocking_validator'))
        NOT VALID;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

ALTER TABLE review_service.agent_runs VALIDATE CONSTRAINT chk_agent_type;

GRANT SELECT, INSERT, UPDATE ON review_service.agent_runs TO service_role;
