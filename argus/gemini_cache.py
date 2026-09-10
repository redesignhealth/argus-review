"""File-backed lifecycle bookkeeping for Gemini explicit context caches.

There is no real Gemini SDK call available yet (Track 3, blocked on a
separate repo), so this module's job is deliberately narrow: compute a
cache key from the inputs that make a cache valid or stale, read/write a
JSON lifecycle record for that key, decide whether an existing record is
still usable, and -- only when a fresh one is actually needed -- call an
injected ``create_fn`` callback to obtain a new ``cache_name``. The real
``client.caches.create(...)`` call is Track 3's job to supply as that
callback; this module is fully unit-testable without any Gemini
credentials or network access.

The one correctness property this design exists to guarantee: changing
the reviewer's ``system_prompt`` or ``tool_schema`` changes the cache
key, forcing recreation instead of silently serving a stale cache after
a prompt file was edited. See ``GeminiCacheKeeper._compute_key``.

Concurrency: the read-check-create-write sequence in ``get_or_create`` is
guarded by a per-key ``fcntl.flock`` (POSIX advisory file lock) so two
concurrent callers racing on the identical cache key can't both decide
"no valid record exists" and both call ``create_fn`` (a duplicate,
wasted upstream cache creation) or interleave writes to the same record
file. The record file itself is written atomically (temp file in the
same directory, then ``os.replace()`` onto the final path), so a reader
can never observe a partially-written JSON file either.
"""

from __future__ import annotations

import contextlib

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore[assignment]
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from argus.config import get_settings

# Matches the ``~/.local/share/argus/...`` convention already used by
# ``argus.storage.sqlite.DEFAULT_DB_PATH`` for this project's local
# application-data files (no XDG_DATA_HOME indirection there either).
_DEFAULT_CACHE_DIR = Path.home() / ".local" / "share" / "argus" / "gemini-cache"

CreateFn = Callable[..., str]


@dataclass(frozen=True)
class CacheRecord:
    """One Gemini explicit-cache lifecycle record."""

    key: str
    cache_name: str | None
    role: str
    model: str
    created_at: datetime
    expires_at: datetime

    def is_expired(self, now: datetime | None = None) -> bool:
        now = now if now is not None else datetime.now(timezone.utc)
        return now >= self.expires_at

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "cache_name": self.cache_name,
            "role": self.role,
            "model": self.model,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> CacheRecord:
        return cls(
            key=data["key"],
            cache_name=data["cache_name"],
            role=data["role"],
            model=data["model"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]),
        )


def _default_cache_dir() -> Path:
    settings = get_settings()
    override = getattr(settings, "ARGUS_GEMINI_CACHE_DIR", None)
    if override:
        return Path(override)
    return _DEFAULT_CACHE_DIR


class GeminiCacheKeeper:
    """Lifecycle bookkeeping for Gemini explicit context-cache records.

    Args:
        cache_dir: Directory to store one ``<key>.json`` record file per
            cache key. Defaults to ``ARGUS_GEMINI_CACHE_DIR`` (via
            ``argus.config.get_settings()``) or
            ``~/.local/share/argus/gemini-cache/`` when unset. Tests
            should always pass an explicit ``tmp_path``-backed directory
            to avoid touching a real home directory.
        create_fn: Callback invoked (as
            ``create_fn(role=..., model=..., system_prompt=..., tool_schema=...,
            ttl_seconds=...)``) to actually create a new upstream cache and
            return its ``cache_name``, only when :meth:`get_or_create`
            determines a fresh cache is needed (no record, or the
            existing one expired/changed). Left ``None`` here -- the real
            ``client.caches.create(...)`` call is Track 3's job to inject.
            Calling :meth:`get_or_create` when a fresh cache is needed but
            no ``create_fn`` was configured raises ``NotImplementedError``.
        now_fn: Injectable clock, defaulting to
            ``lambda: datetime.now(timezone.utc)``. Tests use this to
            control expiry without sleeping.
        default_ttl_seconds: Default TTL (seconds) for a fresh cache
            record when :meth:`get_or_create`'s own ``ttl_seconds``
            argument is left ``None``. Defaults to the configured
            ``ARGUS_GEMINI_CACHE_TTL`` setting (via
            ``argus.config.get_settings()``) when unset here.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        create_fn: CreateFn | None = None,
        now_fn: Callable[[], datetime] | None = None,
        default_ttl_seconds: int | None = None,
    ) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
        self._create_fn = create_fn
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self._default_ttl_seconds = (
            default_ttl_seconds
            if default_ttl_seconds is not None
            else get_settings().ARGUS_GEMINI_CACHE_TTL
        )

    @staticmethod
    def _compute_key(role: str, model: str, system_prompt: str, tool_schema: Any) -> str:
        """Hash the inputs that make a cache entry valid or stale.

        ``tool_schema`` is serialized with ``sort_keys=True`` so dict key
        ordering never changes the key spuriously; changing either
        ``system_prompt`` or the schema's actual content DOES change the
        key, which is the specific property this design exists to
        guarantee (see module docstring).
        """
        schema_json = json.dumps(tool_schema, sort_keys=True, default=str)
        raw = f"{len(role)}:{role}|{len(model)}:{model}|{len(system_prompt)}:{system_prompt}|{len(schema_json)}:{schema_json}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _record_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def _lock_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.lock"

    @contextlib.contextmanager
    def _locked(self, key: str) -> Iterator[None]:
        """Hold a per-key exclusive OS advisory lock for the duration of the
        ``with`` block.

        Guards the read-check-create-write sequence in :meth:`get_or_create`
        against two concurrent callers racing on the SAME key: without
        this, both could read "no valid record", both decide a fresh cache
        is needed, and both call ``create_fn`` (a duplicate, wasted
        upstream cache creation). ``fcntl.flock`` locks are associated with
        the open file description, so this blocks correctly across both
        threads and processes sharing this cache directory -- not just
        within one Python process.
        """
        if fcntl is None:
            raise RuntimeError("gemini cache locking requires a POSIX platform with fcntl support")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._lock_path(key), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _read_record(self, key: str) -> CacheRecord | None:
        path = self._record_path(key)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return CacheRecord.from_json(data)
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def _write_record(self, record: CacheRecord) -> None:
        """Write ``record`` atomically: a reader must never observe a
        partially-written JSON file.

        Writes to a throwaway temp file in the SAME directory as the final
        path (so the later ``os.replace()`` is a same-filesystem rename,
        which POSIX guarantees is atomic), then swaps it into place.
        """
        path = self._record_path(record.key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{record.key}.", suffix=".tmp")
        opened = False
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                opened = True
                f.write(json.dumps(record.to_json(), indent=2))
            os.replace(tmp_name, path)
        except BaseException:
            if not opened:
                with contextlib.suppress(OSError):
                    os.close(fd)
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise

    def get_or_create(
        self,
        role: str,
        model: str,
        system_prompt: str,
        tool_schema: Any,
        ttl_seconds: int | None = None,
    ) -> CacheRecord:
        """Return a valid :class:`CacheRecord`, creating one if needed.

        A record is reused as-is when one exists for the computed key and
        it has not yet expired. Otherwise (no record, or expired) a fresh
        one is created via ``create_fn`` and persisted, replacing any
        stale record for the same key. The whole read-check-create-write
        sequence runs under a per-key lock (see :meth:`_locked`) so
        concurrent callers racing on the same key can't both trigger a
        duplicate upstream creation.

        Args:
            ttl_seconds: TTL for a freshly-created record. Defaults to
                this instance's configured ``default_ttl_seconds`` (see
                ``__init__``, itself defaulting to the
                ``ARGUS_GEMINI_CACHE_TTL`` setting) when left ``None``.

        Raises:
            NotImplementedError: a fresh cache is needed but this
                instance was constructed without a ``create_fn``.
        """
        key = self._compute_key(role, model, system_prompt, tool_schema)
        effective_ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl_seconds

        with self._locked(key):
            now = self._now_fn()

            existing = self._read_record(key)
            if existing is not None and not existing.is_expired(now):
                return existing

            if self._create_fn is None:
                raise NotImplementedError(
                    "GeminiCacheKeeper.get_or_create needs to create a new cache entry "
                    f"(key={key}) but no create_fn was configured. Inject the real "
                    "client.caches.create(...) callback (Track 3) to enable this."
                )

            cache_name = self._create_fn(
                role=role,
                model=model,
                system_prompt=system_prompt,
                tool_schema=tool_schema,
                ttl_seconds=effective_ttl,
            )
            record = CacheRecord(
                key=key,
                cache_name=cache_name,
                role=role,
                model=model,
                created_at=now,
                expires_at=now + timedelta(seconds=effective_ttl),
            )
            self._write_record(record)
            return record

    def invalidate(
        self,
        role: str,
        model: str,
        system_prompt: str,
        tool_schema: Any,
    ) -> None:
        """Remove any locally-recorded record for this key, forcing the next
        :meth:`get_or_create` call to create a fresh upstream cache.

        Used when a caller discovers -- typically because a
        ``generate_content`` call using a ``cached_content=`` reference this
        keeper believed was still unexpired actually failed upstream (TTL
        drift, manual deletion, etc.) -- that the locally-recorded record no
        longer reflects reality. A no-op (not an error) if no record exists
        for this key. Runs under the same per-key lock as
        :meth:`get_or_create` so it can't race a concurrent create for the
        identical key.
        """
        key = self._compute_key(role, model, system_prompt, tool_schema)
        with self._locked(key):
            with contextlib.suppress(FileNotFoundError):
                self._record_path(key).unlink()
