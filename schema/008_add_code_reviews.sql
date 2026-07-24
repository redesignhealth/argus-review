-- Code review tracking table for the v2 orchestrated review service.
-- Stores every review run with verdict, findings, cost, timing, and full result.
-- Schema: review_service (platform-level tooling, not a product feature)

CREATE SCHEMA IF NOT EXISTS review_service;

CREATE TABLE IF NOT EXISTS review_service.code_reviews (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    flow_run_id TEXT,
    repo TEXT NOT NULL,
    pr_number INTEGER NOT NULL,
    verdict TEXT NOT NULL,
    risk_level TEXT,
    blocking_count INTEGER DEFAULT 0,
    suggestion_count INTEGER DEFAULT 0,
    review_comment TEXT NOT NULL,
    result_json JSONB,
    cost_usd NUMERIC,
    duration_seconds NUMERIC,
    model TEXT,
    orchestrator_model TEXT,
    subagent_model TEXT,
    orchestrator_effort TEXT,
    sha TEXT,
    base_ref TEXT,
    agent_trace JSONB,
    comparison_tag TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Index for looking up reviews by PR
CREATE INDEX IF NOT EXISTS idx_code_reviews_repo_pr ON review_service.code_reviews (repo, pr_number);

-- Index for status endpoint polling by flow run ID
CREATE INDEX IF NOT EXISTS idx_code_reviews_flow_run ON review_service.code_reviews (flow_run_id);

-- Index for cost/usage analytics
CREATE INDEX IF NOT EXISTS idx_code_reviews_created_at ON review_service.code_reviews (created_at);

-- Index for SHA-based comparison queries
CREATE INDEX IF NOT EXISTS idx_code_reviews_sha ON review_service.code_reviews (sha);
