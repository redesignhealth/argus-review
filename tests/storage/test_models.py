"""Tests for storage Pydantic model policy boundaries.

Pins the forbid-vs-ignore policy and the read-only JSONB string
decoder so an accidental revert (e.g. moving ``_decode_jsonb_string``
back onto the write model) doesn't silently break the HTTP path.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from argus.storage.models import (
    CodeReviewRound,
    CodeReviewRoundRecord,
    ListReviewRoundsResponse,
)


def _record_payload(**overrides):
    base = {
        "id": str(uuid4()),
        "created_at": datetime(2026, 5, 19, 12, tzinfo=UTC).isoformat(),
        "repo": "org/repo",
        "pr_number": 42,
        "verdict": "APPROVE",
        "risk_level": "LOW",
        "result_json": {"findings": []},
        "sha": "abc1234",
        "current_stage": "completed",
    }
    base.update(overrides)
    return base


class TestExtraPolicy:
    def test_write_model_rejects_unknown_field(self) -> None:
        with pytest.raises(ValidationError):
            CodeReviewRound(
                repo="o/r",
                pr_number=1,
                bogus_column="oops",  # type: ignore[call-arg]
            )

    def test_read_model_accepts_unknown_field(self) -> None:
        """Forward-compat: the backend adding a column shouldn't 5xx older clients."""
        payload = _record_payload(server_only_future_column="ok")
        record = CodeReviewRoundRecord.model_validate(payload)
        assert record.verdict == "APPROVE"

    def test_list_response_accepts_unknown_field(self) -> None:
        record = _record_payload()
        parsed = ListReviewRoundsResponse.model_validate(
            {"rounds": [record], "total_for_pagination": 99}
        )
        assert len(parsed.rounds) == 1


class TestJsonbDecode:
    def test_record_coerces_json_string_to_dict(self) -> None:
        """asyncpg sometimes returns JSONB as a raw string — accept it."""
        payload = _record_payload(result_json='{"findings": [{"severity": "BLOCKING"}]}')
        record = CodeReviewRoundRecord.model_validate(payload)
        assert isinstance(record.result_json, dict)
        assert record.result_json["findings"][0]["severity"] == "BLOCKING"

    def test_record_passes_dict_through(self) -> None:
        record = CodeReviewRoundRecord.model_validate(_record_payload())
        assert record.result_json == {"findings": []}

    def test_record_rejects_invalid_json_string(self) -> None:
        with pytest.raises(ValidationError):
            CodeReviewRoundRecord.model_validate(_record_payload(result_json="not json"))

    def test_write_model_rejects_string_result_json(self) -> None:
        """Write side must NOT silently coerce — that's a caller bug
        the validator on the read model would mask.
        """
        with pytest.raises(ValidationError):
            CodeReviewRound(
                repo="o/r",
                pr_number=1,
                result_json='{"findings": []}',
            )
