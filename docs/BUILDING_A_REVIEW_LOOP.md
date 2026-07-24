# Building a review loop around Argus

Argus itself does one thing: given a repo and a PR (or a SHA + base ref),
produce a structured review. Running it once and reading the output is
useful on its own, but the highest-leverage way to use it is as the review
step inside an automated loop: **review → fix the findings → re-review →
repeat until approved**, gated on CI.

This doc describes the pattern Redesign Health uses internally — as a
Claude Code [skill](https://docs.claude.com/en/docs/claude-code/skills) that
drives that loop end to end on a PR. The skill itself isn't part of this
repo (it's internal tooling, not a redistributable artifact), but the
pattern is straightforward to rebuild for your own team, in whatever
automation surface you use — a Claude Code skill, a GitHub Actions job, a
cron script, or a Slack-triggered bot.

## The pieces

A review loop built on Argus needs five things:

1. **A pinned install.** Reproducibility matters here — you don't want a
   convergence loop's behavior to shift mid-run because `pip` resolved a
   newer Argus release. Install into an isolated environment and pin the
   version:
   ```bash
   uv venv --python 3.12 ~/.argus/venv
   uv pip install --python ~/.argus/venv/bin/python argus-code-review==0.1.0
   ```
   Bump the pin deliberately when you want the update, rather than
   floating on `latest`.

2. **Secrets resolution.** Argus needs `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
   and `GITHUB_TOKEN_RO` at minimum (see [`.env.example`](../.env.example)
   and the [README](../README.md#managing-secrets)). Whatever automates the
   loop should resolve these itself rather than assuming the invoking
   environment already has them exported — the pattern we use:
   - If the variable is already set in the environment, use it as-is.
   - Otherwise, fetch it from wherever your team keeps shared secrets (a
     secrets manager, a vault, a parameter store) — this is the one part
     of the loop that's genuinely team-specific, since it depends on your
     existing secrets infrastructure.
   - If neither works, fail with a clear message naming exactly which
     variable is missing and how to set it — never fail silently or fall
     back to a degraded mode that produces a misleading review.

3. **The convergence loop itself.** Pseudocode for the core cycle:
   ```
   round = 1
   while round <= MAX_ROUNDS:
       result = argus review OWNER/REPO --pr PR_NUMBER -o review.json
       if result.verdict == APPROVE:
           break
       if round == MAX_ROUNDS:
           escalate_to_human(result)
           break
       fix_the_findings(result.findings)   # e.g. dispatch a coding agent
       commit_and_push()
       round += 1
   ```
   A few things matter for this to behave well in practice:
   - **Re-running against the same `owner/repo` + `--pr` is intentional,
     not accidental.** Argus picks up the prior round from storage
     automatically and verifies whether previously reported findings were
     resolved, regressed, or are still open — see [Using
     Argus](../README.md#using-argus) — so round 2+ is a targeted
     re-check, not a full re-review.
   - **Cap the rounds.** An LLM-driven fix step can loop on a finding it
     keeps re-introducing. A hard round limit with a human-escalation path
     is the backstop.
   - **Let the fix step dismiss a finding it disagrees with** via
     `argus review ... --dismiss "<rationale>"` rather than looping forever
     trying to satisfy a false positive — this feeds into the same
     dismiss-tracking Argus already does for PR comments prefixed
     `/dismiss` (see the README's "Using Argus" section).

4. **A shared storage backend, so a CI gate can see the verdict.** If your
   loop runs somewhere other than the CI system that ultimately merges the
   PR (e.g. a local agent session, or a separate bot), point Argus at
   shared storage — Postgres or the HTTP backend, see [Connecting a
   database](../README.md#connecting-a-database) — so the verdict is
   durable and query-able. A CI job (running in the target repo, gating the
   merge) then reads the latest round for that `(repo, pr_number)` and
   blocks the merge unless the verdict is `APPROVE`. This is what turns
   Argus from "a review I can run" into "a gate nothing can bypass" — the
   loop and the gate share one source of truth instead of trusting the
   loop's own exit code.

5. **Prompt overrides, if your team's conventions diverge from the
   packaged prompts.** See [`docs/CUSTOMIZING_PROMPTS.md`](CUSTOMIZING_PROMPTS.md)
   for the full picture, but for a review loop specifically: point
   `ARGUS_PROMPTS_DIR` at a directory your automation controls (not a
   developer's local override — this is the loop's own working directory),
   so the loop's prompts stay independent of whatever the operator running
   the CLI by hand has set locally.

## Putting it together as a Claude Code skill

If you're building this as a Claude Code
[skill](https://docs.claude.com/en/docs/claude-code/skills) (a markdown file
with instructions an agent follows, rather than a fixed script), the same
five pieces map onto skill sections:

- A **setup** section that runs the pinned install if it's not already
  present, and resolves secrets before doing anything else — fail fast with
  a clear message if a secret can't be resolved.
- A **run** step that shells out to `argus review ...` and parses the JSON
  output (`result.verdict`, `result.findings`) to decide what happens next.
- A **fix** step where the agent itself is the "fix the findings" part of
  the loop — read each finding, patch the code, commit.
- A **loop** with an explicit round counter and a hard cap, exactly as
  above — an agent-driven loop needs this guardrail even more than a
  scripted one, since the agent can otherwise "helpfully" keep trying past
  where a script would have stopped.
- A **final report** back to whoever invoked the skill: verdict, round
  count, findings dismissed and why, and a link to the PR.

Writing the skill this way keeps Argus itself unopinionated about
orchestration — it stays a single CLI invocation with a structured output
contract — while your team's loop, secrets handling, and merge gate stay in
your own automation, versioned and iterated on independently of Argus
releases.
