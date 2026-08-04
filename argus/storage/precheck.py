"""Postgres access for the deterministic-precheck rule lifecycle.

Unlike ``argus.storage.resolver``'s three-backend round-history abstraction
(postgres / http / sqlite), this is Postgres-only: rule status and
graduation (schema/017_add_precheck_rules.sql) is RH-internal
infrastructure that only makes sense with a shared database an out-of-band
triage job also reads and writes -- there is no HTTP-shim or local-sqlite
equivalent. When no ``ARGUS_DB_URL``/``SUPABASE_DB_URL`` is configured,
every function here no-ops safely: ``select_rule_statuses`` returns an empty
mapping (every rule is treated as its safe default, 'candidate' -- see
``argus.precheck.engine``) and ``log_candidate_firings`` is a silent no-op.
A precheck DB hiccup must never fail or block a PR review.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy import ARRAY, Integer, TEXT, bindparam, text

from argus.config import get_settings
from argus.storage.session import get_async_session_factory

logger = logging.getLogger(__name__)

_SELECT_STATUSES_SQL = """
    SELECT rule_id, status
    FROM review_service.precheck_rules
    WHERE rule_id = ANY(:rule_ids)
"""

_ENSURE_RULE_ROWS_SQL = """
    INSERT INTO review_service.precheck_rules (rule_id)
    SELECT unnest(:rule_ids)
    ON CONFLICT (rule_id) DO NOTHING
"""

_INSERT_FIRINGS_SQL = """
    INSERT INTO review_service.precheck_candidate_firings
        (rule_id, repo, pr_number, head_sha, finding)
    SELECT rule_id, repo, pr_number, head_sha, finding::jsonb
    FROM unnest(:rule_ids, :repos, :pr_numbers, :head_shas, :findings)
        AS t(rule_id, repo, pr_number, head_sha, finding)
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
            stmt = text(_SELECT_STATUSES_SQL).bindparams(bindparam("rule_ids", type_=ARRAY(TEXT)))
            result = await session.execute(stmt, {"rule_ids": rule_ids})
            return {row.rule_id: row.status for row in result}
    except Exception:  # noqa: BLE001 — a precheck DB hiccup must never block a PR
        logger.warning("Failed to read precheck rule statuses", exc_info=True)
        return {}


@dataclass(frozen=True)
class CandidateFiring:
    """One candidate-rule hit to queue for later, out-of-band triage."""

    rule_id: str
    finding: dict[str, Any]


async def log_candidate_firings(
    *, repo: str, pr_number: int, head_sha: str, firings: list[CandidateFiring]
) -> None:
    """Queue candidate-rule firings for later, out-of-band triage.

    Batched into a single multi-row INSERT (via ``unnest``) rather than one
    round-trip + commit per firing: this runs on the synchronous per-PR
    pipeline path before any LLM step, so its cost is fully on the critical
    path regardless of how many candidate rules fired this round.
    Best-effort: swallows and logs every failure rather than raising. Auto-
    creates each ``precheck_rules`` row (default status 'candidate') for any
    rule_id not seen before, so the firing inserts' foreign key always has
    somewhere to point -- deduped across the batch via a single ``unnest``
    into the ``ON CONFLICT DO NOTHING`` upsert, same reasoning as the firings
    insert itself.

    Trade-off, deliberate: batching removes the per-firing fault isolation
    a loop-of-inserts would have had -- one malformed row (e.g. a `finding`
    dict that fails the ``finding::jsonb`` cast) now drops the whole
    round's candidate firings in the same all-or-nothing transaction,
    rather than just that one row. Consistent with this module's fail-open
    design (a firing-logging problem must never surface to the PR review),
    just at coarser granularity than before.
    """
    if not firings or not get_settings().db_url:
        return
    try:
        session_factory = get_async_session_factory()
        async with session_factory() as session:
            rule_ids = [f.rule_id for f in firings]
            repos = [repo] * len(firings)
            pr_numbers = [pr_number] * len(firings)
            head_shas = [head_sha] * len(firings)
            findings_json = [json.dumps(f.finding) for f in firings]
            # unnest() zips positionally and pads short arrays with NULL on
            # a length mismatch rather than erroring -- a silent, hard-to-
            # notice data-corruption mode. Not guarded by an explicit check:
            # all five lists are provably the same length here (each is
            # `len(firings)` elements, built from `firings`/`repo`/
            # `pr_number`/`head_sha` directly above, not from independent
            # sources), and any check would sit inside this same try/except
            # and be swallowed identically to the corruption it would
            # guard against -- it would document the invariant, not enforce
            # it any more than this comment already does. Keep the five
            # lists built this way (from `firings` and the single scalar
            # args) if this function is ever refactored.

            ensure_stmt = text(_ENSURE_RULE_ROWS_SQL).bindparams(
                bindparam("rule_ids", type_=ARRAY(TEXT))
            )
            await session.execute(ensure_stmt, {"rule_ids": rule_ids})

            insert_stmt = text(_INSERT_FIRINGS_SQL).bindparams(
                bindparam("rule_ids", type_=ARRAY(TEXT)),
                bindparam("repos", type_=ARRAY(TEXT)),
                bindparam("pr_numbers", type_=ARRAY(Integer)),
                bindparam("head_shas", type_=ARRAY(TEXT)),
                bindparam("findings", type_=ARRAY(TEXT)),
            )
            await session.execute(
                insert_stmt,
                {
                    "rule_ids": rule_ids,
                    "repos": repos,
                    "pr_numbers": pr_numbers,
                    "head_shas": head_shas,
                    "findings": findings_json,
                },
            )
            await session.commit()
    except Exception:  # noqa: BLE001 — see docstring
        logger.warning(
            "Failed to log %d candidate-rule firing(s) for %s#%s",
            len(firings),
            repo,
            pr_number,
            exc_info=True,
        )
