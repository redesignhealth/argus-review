-- Pattern analysis table for the review feedback loop.
-- Stores weekly aggregated patterns detected from code review findings.
-- Used by an offline weekly job to track recurring issues and prompt tuning actions.
-- This table (and the weekly job that populates it) is not part of this
-- toolkit; see docs/STORAGE.md for what actually ships here.

CREATE TABLE IF NOT EXISTS review_service.review_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    week_ending DATE NOT NULL,
    category TEXT NOT NULL,
    directory TEXT,                  -- e.g. "services/api"
    occurrence_count INTEGER NOT NULL DEFAULT 0,
    distinct_pr_count INTEGER NOT NULL DEFAULT 0,
    avg_severity TEXT,               -- "BLOCKING" or "SUGGESTION"
    sample_descriptions TEXT[],      -- Top N representative finding descriptions
    action_taken TEXT,               -- "prompt_update", "pattern_created", "skipped"
    action_detail TEXT,              -- Human-readable description of what was changed
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    CONSTRAINT chk_review_patterns_action CHECK (action_taken IN ('prompt_update', 'pattern_created', 'skipped'))
);

-- Index for querying patterns by week
CREATE INDEX IF NOT EXISTS idx_review_patterns_week_ending
    ON review_service.review_patterns (week_ending);

-- Index for querying patterns by category + directory
CREATE INDEX IF NOT EXISTS idx_review_patterns_category_dir
    ON review_service.review_patterns (category, directory);

-- Uniqueness constraint: one record per (week_ending, category, directory)
-- Enables idempotent upserts on retry.
CREATE UNIQUE INDEX IF NOT EXISTS uq_review_patterns_week_cat_dir
    ON review_service.review_patterns (week_ending, category, COALESCE(directory, ''));
