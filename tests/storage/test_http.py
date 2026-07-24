"""Tests for the HTTP storage shim.

Covers the minimum surface ``graph.py`` actually uses:
``read_latest_completed_round`` + ``write_round``. Plus the
process-global ``install_http_storage()`` / ``is_http_storage_enabled()``
pattern that lets the CLI flags arm the shim without graph.py needing
to know.

Tests use ``pytest-httpx`` to mock the backend side; no real network.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pytest_httpx import HTTPXMock

from argus.storage import (
    CodeReviewRound,
    HttpStorageClient,
    HttpStorageError,
)
from argus.storage.http import (
    get_http_storage,
    install_http_storage,
    is_http_storage_enabled,
    reset_http_storage,
)


@pytest.fixture(autouse=True)
async def _reset_singleton():
    """Each test starts with no HTTP storage installed.

    ``@pytest.fixture`` (not ``@pytest_asyncio.fixture``) is correct here:
    ``pyproject.toml`` sets ``asyncio_mode = "auto"``,
    under which pytest-asyncio auto-detects and schedules ``async def``
    fixtures as coroutines. The explicit ``@pytest.mark.asyncio`` on
    test functions is redundant but harmless.
    """
    await reset_http_storage()
    yield
    await reset_http_storage()


_READ_URL = (
    "https://fake-storage-backend.test/api/v1/code-review/storage/reviews/{owner}/{repo}/{pr}"
)
_WRITE_URL = "https://fake-storage-backend.test/api/v1/code-review/storage/reviews/{owner}/{repo}/{pr}/rounds"


def _row_payload(**overrides):
    """Build a CodeReviewRoundRecord-shaped dict for mock responses."""
    base = {
        "id": str(uuid4()),
        "created_at": datetime(2026, 5, 19, 12, tzinfo=UTC).isoformat(),
        "flow_run_id": None,
        "repo": "acme/example-repo",
        "pr_number": 42,
        "verdict": "APPROVE",
        "risk_level": "LOW",
        "blocking_count": 0,
        "suggestion_count": 1,
        "review_comment": "lgtm",
        "result_json": {"findings": []},
        "cost_usd": 0.02,
        "duration_seconds": 12.5,
        "reviewer_version": "v3",
        "orchestrator_model": "claude-default",
        "subagent_model": "claude-default",
        "sha": "abc1234",
        "base_ref": "main",
        "current_stage": "completed",
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_read_latest_completed_round_happy(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="GET",
        url="https://fake-storage-backend.test/api/v1/code-review/storage/reviews/acme/example-repo/42",
        json={"rounds": [_row_payload(verdict="APPROVE")]},
    )
    client = HttpStorageClient(read_url=_READ_URL, write_url=_WRITE_URL, auth="secret-key-value")
    try:
        latest, count = await client.read_latest_completed_round(
            owner="acme", repo="example-repo", pr=42
        )
    finally:
        await client.aclose()

    assert latest is not None
    assert latest.verdict == "APPROVE"
    assert count == 1
    request = httpx_mock.get_request()
    assert request is not None
    # The HTTP storage contract expects X-API-Key, not Authorization.
    assert request.headers.get("X-API-Key") == "secret-key-value"
    assert "Authorization" not in request.headers


@pytest.mark.asyncio
async def test_read_returns_none_for_empty(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="GET",
        url="https://fake-storage-backend.test/api/v1/code-review/storage/reviews/acme/example-repo/999",
        json={"rounds": []},
    )
    client = HttpStorageClient(read_url=_READ_URL, write_url=_WRITE_URL)
    try:
        latest, count = await client.read_latest_completed_round(
            owner="acme", repo="example-repo", pr=999
        )
    finally:
        await client.aclose()
    assert latest is None
    assert count == 0


@pytest.mark.asyncio
async def test_read_wraps_http_error(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="GET",
        url="https://fake-storage-backend.test/api/v1/code-review/storage/reviews/acme/example-repo/42",
        status_code=503,
    )
    client = HttpStorageClient(read_url=_READ_URL, write_url=_WRITE_URL)
    try:
        with pytest.raises(HttpStorageError):
            await client.read_latest_completed_round(owner="acme", repo="example-repo", pr=42)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_write_round_happy(httpx_mock: HTTPXMock) -> None:
    written_id = uuid4()
    httpx_mock.add_response(
        method="POST",
        url="https://fake-storage-backend.test/api/v1/code-review/storage/reviews/acme/example-repo/42/rounds",
        status_code=201,
        json=_row_payload(id=str(written_id)),
    )
    client = HttpStorageClient(read_url=_READ_URL, write_url=_WRITE_URL, auth="ApiKey k")
    round_data = CodeReviewRound(
        repo="acme/example-repo",
        pr_number=42,
        verdict="APPROVE",
        risk_level="LOW",
        blocking_count=0,
        suggestion_count=1,
    )
    try:
        result = await client.write_round(
            owner="acme", repo="example-repo", pr=42, round_data=round_data
        )
    finally:
        await client.aclose()
    assert str(result.id) == str(written_id)


@pytest.mark.asyncio
async def test_write_round_rejects_path_body_mismatch() -> None:
    client = HttpStorageClient(read_url=_READ_URL, write_url=_WRITE_URL)
    try:
        with pytest.raises(HttpStorageError, match="mismatches path"):
            await client.write_round(
                owner="acme",
                repo="example-repo",
                pr=42,
                round_data=CodeReviewRound(
                    repo="evil/elsewhere",
                    pr_number=42,
                ),
            )
    finally:
        await client.aclose()


def test_install_singleton_pattern() -> None:
    assert is_http_storage_enabled() is False
    assert get_http_storage() is None

    install_http_storage(read_url=_READ_URL, write_url=_WRITE_URL, auth="ApiKey k")
    assert is_http_storage_enabled() is True
    assert get_http_storage() is not None


def test_install_is_idempotent_with_same_params() -> None:
    a = install_http_storage(read_url=_READ_URL, write_url=_WRITE_URL, auth="ApiKey k")
    b = install_http_storage(read_url=_READ_URL, write_url=_WRITE_URL, auth="ApiKey k")
    assert a is b


@pytest.mark.asyncio
async def test_read_returns_count_for_multi_round(httpx_mock: HTTPXMock) -> None:
    """A PR on its Nth review should produce ``count == N`` so the caller
    can compute ``round_number = count + 1`` — guards the regression that
    motivated lifting ``round_number`` out of the HTTP-mode hardcode.
    """
    httpx_mock.add_response(
        method="GET",
        url="https://fake-storage-backend.test/api/v1/code-review/storage/reviews/acme/example-repo/42",
        json={
            "rounds": [
                _row_payload(sha="r3"),
                _row_payload(sha="r2"),
                _row_payload(sha="r1"),
            ]
        },
    )
    client = HttpStorageClient(read_url=_READ_URL, write_url=_WRITE_URL)
    try:
        latest, count = await client.read_latest_completed_round(
            owner="acme", repo="example-repo", pr=42
        )
    finally:
        await client.aclose()
    assert latest is not None
    assert latest.sha == "r3"
    assert count == 3


@pytest.mark.asyncio
async def test_read_omits_api_key_when_no_auth(httpx_mock: HTTPXMock) -> None:
    """``auth=None`` must not send an ``X-API-Key`` header at all
    (vs sending an empty/None value, which would 401 on most gateways).
    """
    httpx_mock.add_response(
        method="GET",
        url="https://fake-storage-backend.test/api/v1/code-review/storage/reviews/acme/example-repo/42",
        json={"rounds": []},
    )
    client = HttpStorageClient(read_url=_READ_URL, write_url=_WRITE_URL, auth=None)
    try:
        await client.read_latest_completed_round(owner="acme", repo="example-repo", pr=42)
    finally:
        await client.aclose()
    request = httpx_mock.get_request()
    assert request is not None
    assert "X-API-Key" not in request.headers


def test_install_raises_on_conflicting_reinstall() -> None:
    install_http_storage(read_url=_READ_URL, write_url=_WRITE_URL, auth="ApiKey k")
    with pytest.raises(HttpStorageError, match="different parameters"):
        install_http_storage(read_url=_READ_URL, write_url=_WRITE_URL, auth="ApiKey DIFFERENT")
