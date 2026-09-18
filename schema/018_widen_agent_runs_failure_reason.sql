-- Widen review_service.agent_runs.failure_reason to a third value.
--
-- schema/016_add_agent_runs_failure_reason.sql constrained this column to
-- 'timeout'/'worker_crashed'/NULL. Neither value fires when a reviewer
-- session (Gemini, OpenAI, or the Claude Agent SDK) runs out of its
-- tool-calling turn budget without ever calling finish_review -- that case
-- was previously left as failure_reason=NULL, indistinguishable from a
-- session that completed normally and genuinely found nothing, silently
-- inflating the review's reported success/coverage.
--
-- 'turn_budget_exhausted' closes that gap: set by argus/gemini_runner.py,
-- argus/openai_runner.py, and argus/runners.py's Claude Agent SDK path
-- (on ResultMessage.subtype == "error_max_turns") when the turn budget
-- runs out before finish_review is called.
--
-- IMPORTANT -- deploy ordering: apply to every environment BEFORE the code
-- that constructs AgentRunIn(failure_reason="turn_budget_exhausted") goes
-- live.

ALTER TABLE review_service.agent_runs
    DROP CONSTRAINT IF EXISTS agent_runs_failure_reason_check;

-- Same NOT VALID + separate VALIDATE two-phase pattern schema/016 uses (and
-- schema/015's chk_agent_type before it): a brief ACCESS EXCLUSIVE lock to
-- add the constraint, then a lighter-locked full validation pass, rather
-- than holding the heavier lock for the whole table scan.
DO $$ BEGIN
    ALTER TABLE review_service.agent_runs
        ADD CONSTRAINT agent_runs_failure_reason_check
        CHECK (failure_reason IS NULL OR failure_reason IN
            ('timeout', 'worker_crashed', 'turn_budget_exhausted'))
        NOT VALID;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

ALTER TABLE review_service.agent_runs VALIDATE CONSTRAINT agent_runs_failure_reason_check;
