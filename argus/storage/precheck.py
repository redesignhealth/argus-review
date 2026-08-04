"""Postgres access for the deterministic-precheck rule lifecycle.

Unlike ``argus.storage.resolver``'s three-backend round-history abstraction
(postgres / http / sqlite), this is Postgres-only: rule status and
graduation (schema/017_add_precheck_rules.sql) is RH-internal
infrastructure that only makes sense with a shared database an out-of-band
triage job also reads and writes -- there is no HTTP-shim or local-sqlite
equivalent. When no ``ARGUS_DB_URL``/``SUPABASE_DB_URL`` is configured,
every function here no-ops safely: ``select_rule_statuses`` returns an empty
mapping (every rule is treated as its safe default, 'candidate' -- see
``argus.precheck.engine``) and ``log_candidate_firing`` is a silent no-op.
A precheck DB hiccup must never fail or block a PR review.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import text

from argus.config import get_settings
from argus.storage.session import get_async_session_factory

logger = logging.getLogger(__name__)

_SELECT_STATUSES_SQL = """
    SELECT rule_id, status
    FROM review_service.precheck_rules
    WHERE rule_id = ANY(:rule_ids)
"""

_ENSURE_RULE_ROW_SQL = """
    INSERT INTO review_service.precheck_rules (rule_id)
    VALUES (:rule_id)
    ON CONFLICT (rule_id) DO NOTHING
"""

_INSERT_FIRING_SQL = """
    INSERT INTO review_service.precheck_candidate_firings
        (rule_id, repo, pr_number, head_sha, finding)
    VALUES (:rule_id, :repo, :pr_number, :head_sha, CAST(:finding AS JSONB))
"""


async def select_rule_statuses(rule_ids: list[str]) -> dict[str, str]:
    """Return ``{rule_id: status}`` for every rule that already has a row.

    A rule_id absent from the returned mapping has no row yet -- callers
    must treat that as 'candidate' (see the module docstring's fail-safe
    default), not as an error.
    """
    if not rule_ids or not get_settings().db_url:
        return {}
    try:
        session_factory = get_async_session_factory()
        async with session_factory() as session:
            result = await session.execute(text(_SELECT_STATUSES_SQL), {"rule_ids": rule_ids})
            return {row.rule_id: row.status for row in result}
    except Exception:  # noqa: BLE001 — a precheck DB hiccup must never block a PR
        logger.warning("Failed to read precheck rule statuses", exc_info=True)
        return {}


async def log_candidate_firing(
    *, rule_id: str, repo: str, pr_number: int, head_sha: str, finding: dict[str, Any]
) -> None:
    """Queue a candidate-rule firing for later, out-of-band triage.

    Best-effort: swallows and logs every failure rather than raising, since
    this is called from the synchronous per-PR pipeline path and must never
    slow down or fail a review over a logging write. Auto-creates the
    ``precheck_rules`` row (default status 'candidate') on first firing of a
    rule that hasn't been seen before, so the firing insert's foreign key
    always has somewhere to point.
    """
    if not get_settings().db_url:
        return
    try:
        session_factory = get_async_session_factory()
        async with session_factory() as session:
            await session.execute(text(_ENSURE_RULE_ROW_SQL), {"rule_id": rule_id})
            await session.execute(
                text(_INSERT_FIRING_SQL),
                {
                    "rule_id": rule_id,
                    "repo": repo,
                    "pr_number": pr_number,
                    "head_sha": head_sha,
                    "finding": json.dumps(finding),
                },
            )
            await session.commit()
    except Exception:  # noqa: BLE001 — see docstring
        logger.warning("Failed to log candidate-rule firing for %s", rule_id, exc_info=True)
