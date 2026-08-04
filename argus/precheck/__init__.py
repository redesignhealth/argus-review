"""Deterministic, non-LLM precheck gate.

Runs custom static-analysis rules (semgrep) against the PR worktree before
the LLM pipeline spends any tokens. Distinct from -- and does not replace --
generic off-the-shelf linting (ruff, eslint, gitleaks, etc.), which stays
the target repo's own CI's job; see ``argus.graph.run_preflight_check`` for
how that CI status is read as a routing signal instead.

Requires the ``prechecks`` extra (``pip install argus-code-review[prechecks]``,
which installs semgrep). Entirely optional: with the extra not installed,
or no rule files present, ``run_precheck`` is a no-op and the pipeline
behaves exactly as it did before this package existed.
"""
