-- Add reviewer_version column to track which review engine (v2 or v3) produced the result.

ALTER TABLE review_service.code_reviews
    ADD COLUMN IF NOT EXISTS reviewer_version TEXT;
