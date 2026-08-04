"""Direct unit tests for tests/conftest.py's marker-enforcement and
credential-stubbing logic.

Both were previously exercised only indirectly, via the CI-excluded
integration tier (a test carrying `needs_real_github_token` skipping
correctly, or `pytest_collection_modifyitems` never actually firing because
every existing marker pairing happens to be correct). These call the
functions directly with synthetic items/requests so the branch logic itself
has coverage that runs in the default suite.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from tests.conftest import _mock_settings, pytest_collection_modifyitems

# `.__wrapped__` unwraps pytest's fixture decorator to get the plain
# function -- pytest raises if a `@pytest.fixture` is called directly (it's
# meant to be requested as a parameter, not invoked), so these tests exercise
# the raw logic instead of going through the fixture machinery. `getattr`,
# not a bare attribute access: mypy strict mode doesn't know pytest's
# `FixtureFunctionDefinition` wrapper carries `__wrapped__` (it does, at
# runtime, same as any `functools.wraps`-style decorator).
_mock_settings_raw = getattr(_mock_settings, "__wrapped__")


def _fake_item(nodeid: str, markers: set[str]) -> MagicMock:
    item = MagicMock()
    item.nodeid = nodeid
    item.get_closest_marker.side_effect = lambda name: object() if name in markers else None
    return item


def test_collection_hook_passes_when_markers_paired_correctly() -> None:
    items = [
        _fake_item("test_a", {"integration", "needs_real_github_token"}),
        _fake_item("test_b", {"integration"}),
        _fake_item("test_c", set()),
    ]
    pytest_collection_modifyitems(items)  # must not raise


def test_collection_hook_raises_on_unpaired_marker() -> None:
    items = [_fake_item("test_a", {"needs_real_github_token"})]
    with pytest.raises(pytest.UsageError, match="test_a"):
        pytest_collection_modifyitems(items)


def test_collection_hook_aggregates_every_offender_in_one_error() -> None:
    items = [
        _fake_item("test_a", {"needs_real_github_token"}),
        _fake_item("test_b", {"needs_real_github_token"}),
        _fake_item("test_c", {"integration", "needs_real_github_token"}),
    ]
    with pytest.raises(pytest.UsageError) as exc_info:
        pytest_collection_modifyitems(items)
    assert "test_a" in str(exc_info.value)
    assert "test_b" in str(exc_info.value)
    assert "test_c" not in str(exc_info.value)


def test_mock_settings_stubs_github_token_ro_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    request = _fake_item("some::test", set())
    _mock_settings_raw(request, monkeypatch)
    assert os.environ["GITHUB_TOKEN_RO"] == "test-github-token"


def test_mock_settings_skips_when_marker_present_and_token_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN_RO", raising=False)
    request = _fake_item("some::test", {"needs_real_github_token"})
    with pytest.raises(pytest.skip.Exception):
        _mock_settings_raw(request, monkeypatch)


def test_mock_settings_does_not_stub_when_marker_present_and_token_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN_RO", "a-real-token")
    request = _fake_item("some::test", {"needs_real_github_token"})
    _mock_settings_raw(request, monkeypatch)
    assert os.environ["GITHUB_TOKEN_RO"] == "a-real-token"
