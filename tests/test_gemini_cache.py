"""Tests for argus.gemini_cache: Gemini explicit-cache lifecycle bookkeeping.

Covers: get_or_create with no create_fn raises NotImplementedError when a
fresh cache is actually needed, an unexpired record is reused without
calling create_fn again, an expired record triggers recreation, the
cache key changes when either system_prompt or tool_schema changes (the
specific property this design exists to guarantee -- see module
docstring), CacheRecord JSON round-tripping, the configured
ARGUS_GEMINI_CACHE_TTL setting is honored as the default TTL, and the
per-key file lock + atomic write prevent a race between concurrent
callers creating the same cache key.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from argus.gemini_cache import CacheRecord, GeminiCacheKeeper


def _make_keeper(tmp_path: Path, create_fn=None, now_fn=None) -> GeminiCacheKeeper:
    return GeminiCacheKeeper(
        cache_dir=tmp_path / "gemini-cache", create_fn=create_fn, now_fn=now_fn
    )


class TestNoCreateFn:
    def test_get_or_create_without_create_fn_raises_when_cache_needed(self, tmp_path: Path) -> None:
        keeper = _make_keeper(tmp_path)
        with pytest.raises(NotImplementedError, match="create_fn"):
            keeper.get_or_create(
                role="cross-cutting",
                model="gemini-3.1-pro-preview",
                system_prompt="You are a reviewer.",
                tool_schema={"tools": []},
            )


class TestUnexpiredReusesExisting:
    def test_second_call_with_same_inputs_does_not_recreate(self, tmp_path: Path) -> None:
        calls: list[dict] = []

        def _create_fn(**kwargs):
            calls.append(kwargs)
            return f"cache-{len(calls)}"

        keeper = _make_keeper(tmp_path, create_fn=_create_fn)

        first = keeper.get_or_create(
            role="cross-cutting",
            model="gemini-3.1-pro-preview",
            system_prompt="You are a reviewer.",
            tool_schema={"tools": []},
            ttl_seconds=3600,
        )
        second = keeper.get_or_create(
            role="cross-cutting",
            model="gemini-3.1-pro-preview",
            system_prompt="You are a reviewer.",
            tool_schema={"tools": []},
            ttl_seconds=3600,
        )

        assert len(calls) == 1
        assert first.cache_name == "cache-1"
        assert second.cache_name == "cache-1"
        assert second == first

    def test_reuse_persists_across_keeper_instances(self, tmp_path: Path) -> None:
        """The record is file-backed, so a fresh GeminiCacheKeeper instance
        pointed at the same directory reuses it too, not just the same
        Python object."""
        cache_dir = tmp_path / "gemini-cache"

        def _create_fn(**kwargs):
            return "cache-persisted"

        keeper_one = GeminiCacheKeeper(cache_dir=cache_dir, create_fn=_create_fn)
        keeper_one.get_or_create(
            role="cross-cutting",
            model="gemini-3.1-pro-preview",
            system_prompt="prompt",
            tool_schema={},
        )

        keeper_two = GeminiCacheKeeper(cache_dir=cache_dir, create_fn=None)
        record = keeper_two.get_or_create(
            role="cross-cutting",
            model="gemini-3.1-pro-preview",
            system_prompt="prompt",
            tool_schema={},
        )
        assert record.cache_name == "cache-persisted"


class TestExpiryTriggersRecreate:
    def test_expired_record_is_recreated(self, tmp_path: Path) -> None:
        calls: list[dict] = []
        current_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

        def _create_fn(**kwargs):
            calls.append(kwargs)
            return f"cache-{len(calls)}"

        def _now_fn():
            return current_time

        keeper = _make_keeper(tmp_path, create_fn=_create_fn, now_fn=_now_fn)

        first = keeper.get_or_create(
            role="cross-cutting",
            model="gemini-3.1-pro-preview",
            system_prompt="prompt",
            tool_schema={},
            ttl_seconds=60,
        )
        assert first.cache_name == "cache-1"

        # Advance the injected clock past expiry.
        current_time = current_time + timedelta(seconds=120)

        second = keeper.get_or_create(
            role="cross-cutting",
            model="gemini-3.1-pro-preview",
            system_prompt="prompt",
            tool_schema={},
            ttl_seconds=60,
        )

        assert len(calls) == 2
        assert second.cache_name == "cache-2"
        assert second.expires_at > first.expires_at

    def test_not_yet_expired_is_not_recreated(self, tmp_path: Path) -> None:
        calls: list[dict] = []
        current_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

        def _create_fn(**kwargs):
            calls.append(kwargs)
            return f"cache-{len(calls)}"

        def _now_fn():
            return current_time

        keeper = _make_keeper(tmp_path, create_fn=_create_fn, now_fn=_now_fn)

        keeper.get_or_create(
            role="cross-cutting",
            model="gemini-3.1-pro-preview",
            system_prompt="prompt",
            tool_schema={},
            ttl_seconds=3600,
        )
        current_time = current_time + timedelta(seconds=10)  # well within ttl
        keeper.get_or_create(
            role="cross-cutting",
            model="gemini-3.1-pro-preview",
            system_prompt="prompt",
            tool_schema={},
            ttl_seconds=3600,
        )

        assert len(calls) == 1


class TestCacheKeyChangesForceRecreation:
    """The one correctness property this design exists to guarantee: a
    prompt or tool-schema edit must never silently serve a stale cache."""

    def test_changing_system_prompt_changes_key_and_forces_recreation(self, tmp_path: Path) -> None:
        calls: list[dict] = []

        def _create_fn(**kwargs):
            calls.append(kwargs)
            return f"cache-{len(calls)}"

        keeper = _make_keeper(tmp_path, create_fn=_create_fn)

        first = keeper.get_or_create(
            role="cross-cutting",
            model="gemini-3.1-pro-preview",
            system_prompt="You are reviewer version A.",
            tool_schema={"tools": ["read_file"]},
        )
        second = keeper.get_or_create(
            role="cross-cutting",
            model="gemini-3.1-pro-preview",
            system_prompt="You are reviewer version B.",  # edited prompt
            tool_schema={"tools": ["read_file"]},
        )

        assert len(calls) == 2  # a new cache entry was created, not reused
        assert first.key != second.key
        assert first.cache_name != second.cache_name

    def test_changing_tool_schema_changes_key_and_forces_recreation(self, tmp_path: Path) -> None:
        calls: list[dict] = []

        def _create_fn(**kwargs):
            calls.append(kwargs)
            return f"cache-{len(calls)}"

        keeper = _make_keeper(tmp_path, create_fn=_create_fn)

        first = keeper.get_or_create(
            role="cross-cutting",
            model="gemini-3.1-pro-preview",
            system_prompt="prompt",
            tool_schema={"tools": ["read_file"]},
        )
        second = keeper.get_or_create(
            role="cross-cutting",
            model="gemini-3.1-pro-preview",
            system_prompt="prompt",
            tool_schema={"tools": ["read_file", "grep"]},  # schema changed
        )

        assert len(calls) == 2
        assert first.key != second.key
        assert first.cache_name != second.cache_name

    def test_key_is_stable_regardless_of_dict_key_ordering(self, tmp_path: Path) -> None:
        """tool_schema is serialized with sort_keys=True, so equivalent
        dicts with different insertion order must produce the SAME key."""
        keeper = _make_keeper(tmp_path)

        key_a = keeper._compute_key("role", "model", "prompt", {"b": 1, "a": 2})
        key_b = keeper._compute_key("role", "model", "prompt", {"a": 2, "b": 1})
        assert key_a == key_b

    def test_changing_role_or_model_also_changes_key(self, tmp_path: Path) -> None:
        keeper = _make_keeper(tmp_path)
        base = keeper._compute_key("cross-cutting", "gemini-3.1-pro-preview", "prompt", {})
        diff_role = keeper._compute_key(
            "blocking-validator", "gemini-3.1-pro-preview", "prompt", {}
        )
        diff_model = keeper._compute_key("cross-cutting", "gemini-3-flash-preview", "prompt", {})
        assert base != diff_role
        assert base != diff_model


class TestCacheRecordJsonRoundTrip:
    def test_to_json_from_json_round_trips(self) -> None:
        record = CacheRecord(
            key="abc123",
            cache_name="cachedContents/xyz",
            role="cross-cutting",
            model="gemini-3.1-pro-preview",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            expires_at=datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
        )
        restored = CacheRecord.from_json(record.to_json())
        assert restored == record

    def test_is_expired(self) -> None:
        record = CacheRecord(
            key="abc123",
            cache_name="cachedContents/xyz",
            role="cross-cutting",
            model="gemini-3.1-pro-preview",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            expires_at=datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
        )
        assert not record.is_expired(datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc))
        assert record.is_expired(datetime(2026, 1, 1, 2, 0, tzinfo=timezone.utc))


class TestConfiguredTtlIsHonored:
    """ARGUS_GEMINI_CACHE_TTL must actually reach get_or_create's default,
    not be shadowed by a hardcoded literal."""

    def test_get_or_create_uses_configured_ttl_when_not_overridden(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from argus.config import clear_cache as clear_settings_cache

        monkeypatch.setenv("ARGUS_GEMINI_CACHE_TTL", "120")
        clear_settings_cache()
        try:
            keeper = GeminiCacheKeeper(
                cache_dir=tmp_path / "gemini-cache", create_fn=lambda **kwargs: "cache-1"
            )
            record = keeper.get_or_create(
                role="cross-cutting",
                model="gemini-3.1-pro-preview",
                system_prompt="prompt",
                tool_schema={},
                # ttl_seconds deliberately omitted -- must fall back to the
                # configured setting, not a hardcoded 3600.
            )
            assert (record.expires_at - record.created_at) == timedelta(seconds=120)
        finally:
            clear_settings_cache()

    def test_explicit_ttl_seconds_still_overrides_configured_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from argus.config import clear_cache as clear_settings_cache

        monkeypatch.setenv("ARGUS_GEMINI_CACHE_TTL", "120")
        clear_settings_cache()
        try:
            keeper = GeminiCacheKeeper(
                cache_dir=tmp_path / "gemini-cache", create_fn=lambda **kwargs: "cache-1"
            )
            record = keeper.get_or_create(
                role="cross-cutting",
                model="gemini-3.1-pro-preview",
                system_prompt="prompt",
                tool_schema={},
                ttl_seconds=30,
            )
            assert (record.expires_at - record.created_at) == timedelta(seconds=30)
        finally:
            clear_settings_cache()

    def test_constructor_default_ttl_seconds_overrides_configured_setting(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A caller can also pin the default at construction time, without
        touching the environment -- useful for tests and non-default
        callers alike."""
        from argus.config import clear_cache as clear_settings_cache

        monkeypatch.setenv("ARGUS_GEMINI_CACHE_TTL", "120")
        clear_settings_cache()
        try:
            keeper = GeminiCacheKeeper(
                cache_dir=tmp_path / "gemini-cache",
                create_fn=lambda **kwargs: "cache-1",
                default_ttl_seconds=45,
            )
            record = keeper.get_or_create(
                role="cross-cutting",
                model="gemini-3.1-pro-preview",
                system_prompt="prompt",
                tool_schema={},
            )
            assert (record.expires_at - record.created_at) == timedelta(seconds=45)
        finally:
            clear_settings_cache()


class TestConcurrencyRace:
    """The read-check-create-write sequence must be safe under concurrent
    callers racing on the identical cache key: only one should ever call
    create_fn, and no reader should observe a corrupt record file."""

    def test_concurrent_get_or_create_calls_only_create_once(self, tmp_path: Path) -> None:
        calls: list[dict] = []
        create_started = threading.Event()
        release_create = threading.Event()

        def _create_fn(**kwargs):
            calls.append(kwargs)
            create_started.set()
            # Simulate a slow upstream call, holding the per-key lock the
            # whole time -- every other racing thread must block here,
            # not sneak in and also call create_fn.
            release_create.wait(timeout=5)
            return "cache-single"

        keeper = GeminiCacheKeeper(cache_dir=tmp_path / "gemini-cache", create_fn=_create_fn)

        results: list[CacheRecord] = []
        errors: list[BaseException] = []
        results_lock = threading.Lock()

        def _worker() -> None:
            try:
                record = keeper.get_or_create(
                    role="cross-cutting",
                    model="gemini-3.1-pro-preview",
                    system_prompt="prompt",
                    tool_schema={},
                )
                with results_lock:
                    results.append(record)
            except BaseException as exc:  # noqa: BLE001 - surfaced via assertion below
                with results_lock:
                    errors.append(exc)

        threads = [threading.Thread(target=_worker) for _ in range(5)]
        for t in threads:
            t.start()

        assert create_started.wait(timeout=5), "create_fn was never called"
        release_create.set()
        for t in threads:
            t.join(timeout=5)
            assert not t.is_alive()

        assert errors == []
        assert len(calls) == 1, "create_fn must be called exactly once across all racers"
        assert len(results) == 5
        assert all(r.cache_name == "cache-single" for r in results)

    def test_write_record_leaves_no_leftover_temp_files(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "gemini-cache"
        keeper = GeminiCacheKeeper(cache_dir=cache_dir, create_fn=lambda **kwargs: "cache-x")
        keeper.get_or_create(
            role="cross-cutting",
            model="gemini-3.1-pro-preview",
            system_prompt="prompt",
            tool_schema={},
        )

        names = {p.name for p in cache_dir.iterdir()}
        assert not any(n.endswith(".tmp") for n in names)
        assert any(n.endswith(".json") for n in names)


class TestDefaultCacheDir:
    def test_default_cache_dir_respects_argus_gemini_cache_dir_setting(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from argus.config import clear_cache as clear_settings_cache
        from argus.gemini_cache import _default_cache_dir

        override = tmp_path / "custom-gemini-cache"
        monkeypatch.setenv("ARGUS_GEMINI_CACHE_DIR", str(override))
        clear_settings_cache()
        try:
            assert _default_cache_dir() == override
        finally:
            clear_settings_cache()

    def test_default_cache_dir_falls_back_to_local_share(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from argus.config import clear_cache as clear_settings_cache
        from argus.gemini_cache import _DEFAULT_CACHE_DIR, _default_cache_dir

        monkeypatch.delenv("ARGUS_GEMINI_CACHE_DIR", raising=False)
        clear_settings_cache()
        try:
            assert _default_cache_dir() == _DEFAULT_CACHE_DIR
        finally:
            clear_settings_cache()


class TestInvalidate:
    """A cache the API has silently forgotten about (TTL drift, manual
    deletion) must be recoverable: the caller invalidates the local record
    so the NEXT get_or_create() call creates a fresh one, rather than
    handing out the same stale cache_name forever."""

    def test_invalidate_forces_recreation_on_next_get_or_create(self, tmp_path: Path) -> None:
        calls: list[dict] = []

        def _create_fn(**kwargs: object) -> str:
            calls.append(kwargs)
            return f"cachedContents/gen-{len(calls)}"

        keeper = _make_keeper(tmp_path, create_fn=_create_fn)
        first = keeper.get_or_create(
            role="cross-cutting",
            model="gemini-3.1-pro-preview",
            system_prompt="You are a reviewer.",
            tool_schema={"tools": []},
        )
        assert first.cache_name == "cachedContents/gen-1"
        assert len(calls) == 1

        # Still well within TTL -- without invalidate(), this would reuse
        # the existing record and NOT call create_fn again.
        keeper.invalidate(
            role="cross-cutting",
            model="gemini-3.1-pro-preview",
            system_prompt="You are a reviewer.",
            tool_schema={"tools": []},
        )

        second = keeper.get_or_create(
            role="cross-cutting",
            model="gemini-3.1-pro-preview",
            system_prompt="You are a reviewer.",
            tool_schema={"tools": []},
        )
        assert second.cache_name == "cachedContents/gen-2"
        assert len(calls) == 2

    def test_invalidate_on_a_key_with_no_record_is_a_no_op(self, tmp_path: Path) -> None:
        keeper = _make_keeper(tmp_path)
        # Must not raise even though nothing has ever been created for
        # this key.
        keeper.invalidate(
            role="never-created",
            model="gemini-3.1-pro-preview",
            system_prompt="sys",
            tool_schema={"tools": []},
        )

    def test_locked_raises_runtime_error_when_fcntl_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import argus.gemini_cache as gc

        monkeypatch.setattr(gc, "fcntl", None)
        keeper = _make_keeper(tmp_path)
        with pytest.raises(RuntimeError, match="fcntl"):
            with keeper._locked("some-key"):
                pass
