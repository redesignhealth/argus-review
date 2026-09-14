"""Bench configuration: which platform/model a leaf reviewer runs on.

A "bench" is a lightweight, human-edited TOML config that decides which
LLM *platform* (Claude Agent SDK and Gemini have packaged defaults;
OpenAI Responses has a real runner too, see ``argus.openai_runner``;
opt-in) and *model* each leaf reviewer in the review pipeline runs on.
This is deliberately NOT a dynamic/adaptive routing system -- it is a static,
PR-reviewed config with a human-editable override chain, in the same
spirit as ``argus.prompts_runtime``'s prompt override chain.

Two kinds of config unit, not a per-role table:

1. ``[bulk_reviewer]`` -- ONE shared platform+model+caching setting that
   applies to every generalist-shaped leaf reviewer as a single unit:
   ``run_system_reviewer`` (per system group, including gap-fill
   reviewers, which reuse the same code path), every named specialist,
   and the tests-and-docs reviewer. Each of these still resolves its OWN
   existing prompt file exactly as before (see ``BULK_ROLE_PROMPTS``) --
   only platform/model/caching is shared.
2. ``[roles.<name>]`` -- an independent platform+model+prompt_name+caching
   config for everything else: ``cross-cutting``, ``blocking-validator``,
   ``feedback-verifier``. Unrelated to the bulk bucket and to each other.

Override chain (mirrors ``argus.prompts_runtime`` exactly, including its
opt-out convention), lowest to highest priority:

1. Packaged ``argus/bench_default.toml`` -- the base. ``[bulk_reviewer]``
   defaults to ``platform="gemini"`` (``model="gemini-mini"``,
   ``caching="auto"``), while individual roles (``cross-cutting``,
   ``blocking-validator``, ``feedback-verifier``) default to ``claude-sdk``.
2. ``~/.config/argus/bench.toml`` (respecting ``XDG_CONFIG_HOME``) -- a
   user-global sparse overlay: only the keys it specifies are overridden;
   everything else falls through to the layer below.
3. ``./.argus/bench.toml`` -- a repo-local sparse overlay, same semantics.
4. ``ARGUS_BENCH_FILE`` env var -- an explicit override file, sparse-merged
   on top of everything below it, so it "wins outright" for any key it
   specifies. Raises if the path doesn't exist (the caller explicitly
   asked for it, so a typo should fail loudly, not silently fall through).

Setting ``ARGUS_NO_BENCH_OVERRIDES`` truthy skips layers 2-4 and forces
the packaged default only -- for CI/official runs that must not pick up a
developer's local override by accident.

Validation happens once, at load time, not mid-review: unknown top-level
or per-table TOML keys, an unknown ``platform`` value, an unknown
``caching`` value, and a ``prompt_name`` that doesn't correspond to an
actual packaged file under ``argus/prompts/`` all raise a loud
``ValueError`` naming the offending key/file.

There is deliberately NO safety floor / minimum-tier guardrail here: the
bench is PR-reviewed for its defaults, and a caller who overrides it at
runtime is trusted to know what they're doing.

## Wiring status

All leaf-reviewer roles defined in this module are wired to their respective
runner functions in ``argus.runners`` via ``bench.resolve(role)`` and
``bench.runner_for(entry)(...)``:
- Bulk-reviewer roles (via ``[bulk_reviewer]``):
  - ``"system-generalist"``: ``run_system_reviewer`` (including gap-fill reviewers)
  - ``"tests-and-docs"``: ``run_tests_and_docs_reviewer``
  - ``"specialist-<name>"``: ``run_specialist_reviewer`` for each specialist
- Individual roles (via ``[roles.<name>]``):
  - ``"cross-cutting"``: ``run_cross_cutting_reviewer``
  - ``"blocking-validator"``: ``run_blocking_validator``
  - ``"feedback-verifier"``: ``run_feedback_verifier``

``resolve()`` logs a warning whenever a caller resolves a role outside
``_WIRED_ROLES``, ensuring any future or unmapped roles that are not wired
fail loudly rather than silently having no effect.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Final, Literal, get_args

from pydantic import TypeAdapter, ValidationError

from argus.llm import models as model_aliases
from argus.pipeline_models import SpecialistName
from argus.prompts_runtime import known_packaged_prompts

logger = logging.getLogger(__name__)

_PACKAGE = "argus"
_DEFAULT_RESOURCE = "bench_default.toml"

Platform = Literal["claude-sdk", "gemini", "openai-responses"]

_VALID_PLATFORMS: Final[frozenset[str]] = frozenset(get_args(Platform))
_VALID_CACHING: Final[frozenset[str]] = frozenset({"auto", "off", "on"})
_VALID_MODEL_ALIASES: Final[frozenset[str]] = frozenset(model_aliases.ALIAS_MAP) | frozenset(
    model_aliases.EXPERIMENTAL_MODELS
)

_TOP_LEVEL_KEYS: Final[frozenset[str]] = frozenset({"bulk_reviewer", "roles"})
_BULK_KEYS: Final[frozenset[str]] = frozenset({"platform", "model", "caching"})
_ROLE_KEYS: Final[frozenset[str]] = frozenset({"platform", "model", "prompt_name", "caching"})

_MODEL_FAMILY_TO_PLATFORM: Final[dict[str, Platform]] = {
    "claude": "claude-sdk",
    "gemini": "gemini",
    "gpt": "openai-responses",
}


def infer_platform_for_model(model: str) -> Platform | None:
    """Infer the execution platform from a model alias or model name.

    Returns the Platform matching the model family prefix ('claude-*' -> 'claude-sdk',
    'gemini-*' -> 'gemini', 'gpt-*' -> 'openai-responses'), or None if unknown.
    """
    family = model.lower().split("-", 1)[0]
    return _MODEL_FAMILY_TO_PLATFORM.get(family)


# Roles a runner function actually resolves via bench.resolve(). Every bulk
# role and every individual role is wired to its corresponding runner function.
_WIRED_ROLES: Final[frozenset[str]] = frozenset(
    {
        "system-generalist",
        "tests-and-docs",
        *(f"specialist-{name}" for name in get_args(SpecialistName)),
        "cross-cutting",
        "blocking-validator",
        "feedback-verifier",
    }
)

# The bulk-bucket role registry: role name -> the prompt file that role
# already resolves for itself (unchanged by the bench). Names mirror
# `runners._SPECIALIST_PROMPT_MAP` (derived from `SpecialistName` here to
# avoid a second hand-maintained copy drifting from it) plus the two
# non-specialist bulk-shaped reviewers. "system-generalist" also covers
# gap-fill reviewers, which dispatch through the exact same
# `run_system_reviewer` code path as the initial fan-out (see
# `graph._edge_fan_out_gap_fills`).
BULK_ROLE_PROMPTS: Final[dict[str, str]] = {
    "system-generalist": "pr-review-subagent",
    "tests-and-docs": "pr-review-tests-and-docs",
    **{f"specialist-{name}": f"pr-review-specialist-{name}" for name in get_args(SpecialistName)},
}


@dataclass(frozen=True)
class BenchEntry:
    """A single resolved role's platform+model+prompt+caching config."""

    role: str
    platform: Platform
    model: str
    prompt_name: str
    caching: str = "auto"


RunnerFn = Callable[..., Awaitable[Any]]


# ---------------------------------------------------------------------------
# Loading + merging
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` onto ``base``, overlay wins per-key.

    A nested dict value is merged recursively (sparse overlay: only the
    keys actually present in ``overlay`` are changed); any other value
    type replaces the base value outright.
    """
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_toml_bytes(data: bytes) -> dict[str, Any]:
    import tomllib

    return tomllib.loads(data.decode("utf-8"))


def _load_packaged_default() -> dict[str, Any]:
    resource = resources.files(_PACKAGE) / _DEFAULT_RESOURCE
    return _load_toml_bytes(resource.read_bytes())


def _load_toml_file(path: Path) -> dict[str, Any]:
    return _load_toml_bytes(path.read_bytes())


def _overlay_layers(settings_or_bench_file: Any = None) -> list[Path]:
    """Ordered overlay file paths, lowest to highest priority.

    Only existing files are included for the two standard locations
    (silently skipped when absent, like ``prompts_runtime.override_dirs``).
    ``ARGUS_BENCH_FILE`` is different: it names an explicit file the
    caller asked for, so a missing path raises instead of being skipped.
    """
    layers: list[Path] = []

    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    config_home = Path(xdg_config_home) if xdg_config_home else Path.home() / ".config"
    user_global = config_home / "argus" / "bench.toml"
    if user_global.is_file():
        layers.append(user_global)

    repo_local = Path.cwd() / ".argus" / "bench.toml"
    if repo_local.is_file():
        layers.append(repo_local)

    bench_file: str | None = None
    if isinstance(settings_or_bench_file, (str, Path)):
        bench_file = str(settings_or_bench_file)
    elif settings_or_bench_file is not None and hasattr(settings_or_bench_file, "ARGUS_BENCH_FILE"):
        val = getattr(settings_or_bench_file, "ARGUS_BENCH_FILE", None)
        if isinstance(val, (str, Path)) and str(val):
            bench_file = str(val)

    if bench_file:
        bench_path = Path(bench_file)
        if not bench_path.is_file():
            raise ValueError(f"ARGUS_BENCH_FILE={bench_file!r} does not exist or is not a file")
        layers.append(bench_path)

    return layers


def _infer_platforms_for_overlay(overlay: dict[str, Any]) -> dict[str, Any]:
    """Infer the compatible platform for any table that specifies `model` but omits `platform`.

    Prevents sparse model-only overrides (such as `model = "claude-mini"`) from inheriting
    an incompatible platform (like the default `platform = "gemini"`) from a lower layer.
    """
    result = dict(overlay)
    if "bulk_reviewer" in result and isinstance(result["bulk_reviewer"], dict):
        bulk = dict(result["bulk_reviewer"])
        if "model" in bulk and isinstance(bulk["model"], str) and "platform" not in bulk:
            inferred = infer_platform_for_model(bulk["model"])
            if inferred is not None:
                bulk["platform"] = inferred
        result["bulk_reviewer"] = bulk
    if "roles" in result and isinstance(result["roles"], dict):
        roles = dict(result["roles"])
        for role_name, role_table in roles.items():
            if isinstance(role_table, dict):
                r = dict(role_table)
                if "model" in r and isinstance(r["model"], str) and "platform" not in r:
                    inferred = infer_platform_for_model(r["model"])
                    if inferred is not None:
                        r["platform"] = inferred
                roles[role_name] = r
        result["roles"] = roles
    return result


_WARNED_ROLES: set[str] = set()
_WARNED_ROLES_LOCK: threading.Lock = threading.Lock()

_BOOL_ADAPTER: Final[TypeAdapter[bool]] = TypeAdapter(bool)


def _parse_bool(val: Any, var_name: str = "ARGUS_NO_BENCH_OVERRIDES") -> bool:
    """Parse a boolean value matching Pydantic's coercion rules.

    When val is None (i.e. env var unset), returns False default.
    Otherwise delegates directly to Pydantic's TypeAdapter(bool)
    and converts ValidationError to ValueError with a clear message.
    """
    if val is None:
        return False
    from unittest.mock import NonCallableMock

    if isinstance(val, NonCallableMock):
        return False
    try:
        return _BOOL_ADAPTER.validate_python(val)
    except ValidationError as exc:
        raise ValueError(
            f"Invalid boolean value for {var_name}: {val!r}. "
            f"Expected a valid boolean (e.g. true/false, yes/no, 1/0, on/off)."
        ) from exc


def _extract_bench_settings(settings: Any) -> tuple[bool, str | None, str | None]:
    """Extract (no_bench_overrides, bench_file, specialist_model) from a settings-like object.

    When settings is provided, its fields are authoritative and never fall back to
    ambient os.environ (preserving dependency injection). For unit test mocks
    (MagicMock), unconfigured attributes that were never set on the mock fall back to
    os.environ so ambient test fixtures can supply values.
    """
    from unittest.mock import NonCallableMock

    if isinstance(settings, NonCallableMock):
        if "ARGUS_NO_BENCH_OVERRIDES" in settings.__dict__:
            no_overrides = _parse_bool(
                settings.ARGUS_NO_BENCH_OVERRIDES, "ARGUS_NO_BENCH_OVERRIDES"
            )
        else:
            no_overrides = _parse_bool(
                os.environ.get("ARGUS_NO_BENCH_OVERRIDES"), "ARGUS_NO_BENCH_OVERRIDES"
            )

        if "ARGUS_BENCH_FILE" in settings.__dict__:
            bf = settings.ARGUS_BENCH_FILE
            bench_file = str(bf) if isinstance(bf, (str, Path)) and str(bf) else None
        else:
            bench_file = os.environ.get("ARGUS_BENCH_FILE") or None

        if "ARGUS_SPECIALIST_MODEL" in settings.__dict__:
            sm = settings.ARGUS_SPECIALIST_MODEL
            specialist_model = str(sm) if isinstance(sm, str) and sm else None
        else:
            specialist_model = os.environ.get("ARGUS_SPECIALIST_MODEL") or None

        return (no_overrides, bench_file, specialist_model)

    # Real Settings or dataclass: strictly authoritative, NEVER fall back to os.environ
    no_overrides = _parse_bool(
        getattr(settings, "ARGUS_NO_BENCH_OVERRIDES", False), "ARGUS_NO_BENCH_OVERRIDES"
    )
    bf = getattr(settings, "ARGUS_BENCH_FILE", None)
    bench_file = str(bf) if isinstance(bf, (str, Path)) and str(bf) else None
    sm = getattr(settings, "ARGUS_SPECIALIST_MODEL", None)
    specialist_model = str(sm) if isinstance(sm, str) and sm else None
    return (no_overrides, bench_file, specialist_model)


def _resolve_bench_settings(settings: Any = None) -> tuple[bool, str | None, str | None]:
    """Resolve bench-routing settings from an injected object or ambient environment."""
    if settings is not None:
        return _extract_bench_settings(settings)

    # Read os.environ directly from the ambient environment without
    # invoking get_settings() or any dotenv loaders, ensuring bench loading
    # never requires credentials, never mutates os.environ, and never loads
    # untrusted repo-local .env files into the process.
    no_overrides = _parse_bool(
        os.environ.get("ARGUS_NO_BENCH_OVERRIDES"), "ARGUS_NO_BENCH_OVERRIDES"
    )
    bench_file = os.environ.get("ARGUS_BENCH_FILE") or None
    specialist_model = os.environ.get("ARGUS_SPECIALIST_MODEL") or None
    return (no_overrides, bench_file, specialist_model)


@lru_cache(maxsize=16)
def _load_bench_cached(
    no_bench_overrides: bool,
    bench_file: str | None,
    specialist_model: str | None,
) -> dict[str, Any]:
    merged = _load_packaged_default()

    if not no_bench_overrides:
        for layer_path in _overlay_layers(bench_file):
            overlay = _load_toml_file(layer_path)
            roles = overlay.get("roles", {})
            if isinstance(roles, dict):
                for role in roles:
                    _warn_if_not_wired(role)
            prepared = _infer_platforms_for_overlay(overlay)
            merged = _deep_merge(merged, prepared)

    # --specialist-model / ARGUS_SPECIALIST_MODEL is a CLI/env knob that
    # overrides the system reviewer and specialist reviewers. When set,
    # force bulk_reviewer to claude-sdk with claude-default so the override
    # controls the bulk reviewer roles as documented.
    if specialist_model:
        merged["bulk_reviewer"]["platform"] = "claude-sdk"
        merged["bulk_reviewer"]["model"] = "claude-default"

    _validate_raw_bench(merged)
    return merged


def load_bench(settings: Any = None) -> dict[str, Any]:
    """Load, merge, and validate the effective bench config.

    Cached for the life of the process; call :func:`clear_cache` to force
    a reload (e.g. in tests, or after mutating ``os.environ``).
    """
    no_overrides, bench_file, specialist_model = _resolve_bench_settings(settings)
    return _load_bench_cached(no_overrides, bench_file, specialist_model)


def clear_cache() -> None:
    """Clear the cached bench config, forcing the next call to reload."""
    with _WARNED_ROLES_LOCK:
        _load_bench_cached.cache_clear()
        _WARNED_ROLES.clear()


load_bench.cache_clear = _load_bench_cached.cache_clear  # type: ignore[attr-defined]
load_bench.cache_info = _load_bench_cached.cache_info  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_platform(value: Any, where: str) -> None:
    # isinstance check FIRST: `value not in <frozenset>` raises TypeError
    # (not the promised load-time ValueError) for an unhashable value like
    # a TOML array (`platform = ["a", "b"]`) or table.
    if not isinstance(value, str) or value not in _VALID_PLATFORMS:
        raise ValueError(
            f"{where}: platform must be one of {sorted(_VALID_PLATFORMS)}, got {value!r}"
        )


def _validate_caching(value: Any, where: str) -> None:
    if not isinstance(value, str) or value not in _VALID_CACHING:
        raise ValueError(f"{where}: caching must be one of {sorted(_VALID_CACHING)}, got {value!r}")


def _validate_model(value: Any, where: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where}: model must be a non-empty string, got {value!r}")
    if value not in _VALID_MODEL_ALIASES:
        raise ValueError(
            f"{where}: model {value!r} is not a known model alias; "
            f"must be one of {sorted(_VALID_MODEL_ALIASES)}"
        )


def _validate_prompt_name(name: Any, where: str) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError(f"{where}: prompt_name must be a non-empty string, got {name!r}")
    known = known_packaged_prompts()
    if name not in known:
        raise ValueError(
            f"{where}: prompt_name {name!r} does not correspond to a packaged prompt file "
            f"(argus/prompts/{name}.md not found)"
        )


def _validate_model_platform_compatibility(platform: str, model: str, where: str) -> None:
    expected_platform = infer_platform_for_model(model)
    if expected_platform is not None and platform != expected_platform:
        raise ValueError(
            f"{where}: model {model!r} is not compatible with platform {platform!r} "
            f"(expected platform={expected_platform!r}). "
            f"Set platform = {expected_platform!r} or select a model compatible with {platform!r}."
        )


def _validate_raw_bench(raw: dict[str, Any]) -> None:
    """Validate the fully-merged bench config, raising loudly on any problem.

    Runs once at load time (see ``load_bench``) so a misconfiguration
    (typo'd key, unknown platform, dangling prompt_name) is caught before
    any review runs, never mid-review.
    """
    unknown_top = set(raw) - _TOP_LEVEL_KEYS
    if unknown_top:
        raise ValueError(
            f"Unknown bench config key(s): {sorted(unknown_top)}. "
            f"Allowed top-level keys: {sorted(_TOP_LEVEL_KEYS)}"
        )

    if "bulk_reviewer" not in raw:
        raise ValueError("Bench config missing required [bulk_reviewer] table")
    bulk = raw["bulk_reviewer"]
    if not isinstance(bulk, dict):
        raise ValueError("[bulk_reviewer] must be a table")
    unknown_bulk = set(bulk) - _BULK_KEYS
    if unknown_bulk:
        raise ValueError(
            f"Unknown key(s) in [bulk_reviewer]: {sorted(unknown_bulk)}. "
            f"Allowed: {sorted(_BULK_KEYS)}"
        )
    for required in ("platform", "model"):
        if required not in bulk:
            raise ValueError(f"[bulk_reviewer] missing required key {required!r}")
    _validate_platform(bulk["platform"], "[bulk_reviewer]")
    _validate_model(bulk["model"], "[bulk_reviewer]")
    _validate_caching(bulk.get("caching", "auto"), "[bulk_reviewer]")
    _validate_model_platform_compatibility(bulk["platform"], bulk["model"], "[bulk_reviewer]")

    roles = raw.get("roles", {})
    if not isinstance(roles, dict):
        raise ValueError("[roles] must be a table of tables")
    for name, role_table in roles.items():
        where = f"[roles.{name}]"
        if not isinstance(role_table, dict):
            raise ValueError(f"{where} must be a table")
        unknown_role_keys = set(role_table) - _ROLE_KEYS
        if unknown_role_keys:
            raise ValueError(
                f"Unknown key(s) in {where}: {sorted(unknown_role_keys)}. "
                f"Allowed: {sorted(_ROLE_KEYS)}"
            )
        for required in ("platform", "model", "prompt_name"):
            if required not in role_table:
                raise ValueError(f"{where} missing required key {required!r}")
        _validate_platform(role_table["platform"], where)
        _validate_model(role_table["model"], where)
        _validate_caching(role_table.get("caching", "auto"), where)
        _validate_prompt_name(role_table["prompt_name"], where)
        _validate_model_platform_compatibility(role_table["platform"], role_table["model"], where)

    # The bulk bucket's own known prompt names are hardcoded (not
    # TOML-supplied), but validate them too: a packaged/override prompt
    # file getting renamed or deleted should fail loudly here, not with a
    # confusing "prompt not found" error deep inside a reviewer session.
    for role, prompt_name in BULK_ROLE_PROMPTS.items():
        _validate_prompt_name(prompt_name, f"[bulk_reviewer] (role={role!r})")


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _warn_if_not_wired(role: str) -> None:
    """Log a loud warning when configuring or resolving a role no runner consults yet.

    See this module's "Wiring status" docstring section and
    ``_WIRED_ROLES``: a caller CAN resolve (and override, via bench.toml)
    any role in ``BULK_ROLE_PROMPTS``/``[roles.*]``, but only the roles in
    ``_WIRED_ROLES`` are ever actually looked up by a runner function --
    overriding anything else silently has no effect on a real review run
    unless this warning is here to say otherwise.
    """
    with _WARNED_ROLES_LOCK:
        if role not in _WIRED_ROLES and role not in _WARNED_ROLES:
            _WARNED_ROLES.add(role)
            should_warn = True
        else:
            should_warn = False

    if should_warn:
        logger.warning(
            "bench: role %r is not yet wired to any runner function, "
            "so configuring it in bench.toml has NO effect on an actual "
            "review run. See argus.bench's module docstring ('Wiring status') "
            "and _WIRED_ROLES for the roles that are currently wired.",
            role,
        )


def resolve(role: str, settings: Any = None) -> BenchEntry:
    """Resolve ``role`` to its effective :class:`BenchEntry`.

    Bulk-bucket roles (see ``BULK_ROLE_PROMPTS``) route through
    ``[bulk_reviewer]`` plus that role's own known prompt name. Every
    other role routes through ``[roles.<role>]``. Raises ``ValueError``
    naming the role and the known roles if ``role`` matches neither.

    Resolving succeeds for every declared role regardless of wiring
    status, but logs a warning (see ``_warn_if_not_wired``) when ``role``
    is not in ``_WIRED_ROLES`` -- see this module's "Wiring status"
    docstring section for why most roles currently have no runner
    consulting them.
    """
    raw = load_bench(settings=settings)

    if role in BULK_ROLE_PROMPTS:
        bulk = raw["bulk_reviewer"]
        _warn_if_not_wired(role)
        return BenchEntry(
            role=role,
            platform=bulk["platform"],
            model=bulk["model"],
            prompt_name=BULK_ROLE_PROMPTS[role],
            caching=bulk.get("caching", "auto"),
        )

    roles = raw.get("roles", {})
    if role in roles:
        role_table = roles[role]
        _warn_if_not_wired(role)
        return BenchEntry(
            role=role,
            platform=role_table["platform"],
            model=role_table["model"],
            prompt_name=role_table["prompt_name"],
            caching=role_table.get("caching", "auto"),
        )

    known = sorted(set(BULK_ROLE_PROMPTS) | set(roles))
    raise ValueError(f"Unknown bench role {role!r}. Known roles: {known}")


# ---------------------------------------------------------------------------
# Platform runners
# ---------------------------------------------------------------------------


async def _claude_sdk_runner(
    *,
    entry: BenchEntry,
    system_prompt: str,
    user_message: str,
    anthropic_api_key: str | None = None,
    anthropic_auth_token: str | None = None,
    context7_key: str | None = None,
    context7_library_id: str | None = None,
    context7_base_url: str | None = None,
    timeout_s: int,
    cwd: str,
    label: str,
    is_system_reviewer_role: bool = False,
) -> Any:
    """Thin adapter for the ``claude-sdk`` platform.

    Reproduces today's exact existing call to
    ``argus.runners._run_session_isolated`` with no behavior change --
    the model alias in ``entry.model`` is resolved to a concrete model
    string via ``argus.llm.models.resolve`` first, exactly as the
    pre-bench call sites did by importing the resolved constant directly.

    ``context7_base_url`` and ``is_system_reviewer_role`` are forwarded
    straight through to ``_run_session_isolated`` unchanged -- see that
    function (and ``_run_claude_session``) for what each does; this
    adapter has no opinion on either, it just passes along whatever the
    shared ``RunnerFn`` call site supplies.

    Imports ``argus.runners`` and ``argus.llm.models`` lazily (function
    body, not module level) to avoid a circular import: ``argus.runners``
    imports this module to call ``resolve``/``runner_for``.
    """
    from argus.llm.models import resolve as resolve_model_alias
    from argus.runners import _run_session_isolated

    if entry.caching != "auto":
        logger.warning(
            "bench: caching=%r is set for role %r but claude-sdk runner does not "
            "use explicit caching; setting ignored",
            entry.caching,
            entry.role,
        )

    model = resolve_model_alias(entry.model)

    return await _run_session_isolated(
        model=model,
        system_prompt=system_prompt,
        user_message=user_message,
        anthropic_api_key=anthropic_api_key,
        anthropic_auth_token=anthropic_auth_token,
        context7_key=context7_key,
        context7_library_id=context7_library_id,
        context7_base_url=context7_base_url,
        timeout_s=timeout_s,
        cwd=cwd,
        label=label,
        is_system_reviewer_role=is_system_reviewer_role,
    )


async def _gemini_runner(
    *,
    entry: BenchEntry,
    system_prompt: str,
    user_message: str,
    anthropic_api_key: str | None = None,
    anthropic_auth_token: str | None = None,
    context7_key: str | None = None,
    context7_library_id: str | None = None,
    context7_base_url: str | None = None,
    timeout_s: int,
    cwd: str,
    label: str,
    is_system_reviewer_role: bool = False,
) -> Any:
    """Thin adapter for the ``gemini`` platform.

    Mirrors ``_claude_sdk_runner``'s adapter role exactly: this function's
    signature is the ``RunnerFn`` contract every call site in
    ``PLATFORM_RUNNERS`` dispatches through (see ``runner_for``/
    ``run_system_reviewer``'s call site, which passes exactly these
    kwargs and nothing else -- no ``settings`` object, no ``repo_root``).
    ``argus.gemini_runner.run_session_gemini`` -- the actual tool-calling
    session implementation -- has a different, purpose-built signature
    (``settings: Any`` instead of individual credential kwargs, ``repo_root``
    instead of ``cwd``), so this adapter translates between the two, the
    same way ``_claude_sdk_runner`` translates into
    ``argus.runners._run_session_isolated``'s own signature.

    The Anthropic/Context7/``is_system_reviewer_role`` kwargs above are
    accepted (the shared ``RunnerFn`` call site always passes them) but
    unused here -- this platform has no Context7 MCP integration in scope
    (see ``argus.gemini_runner``'s module docstring), needs no Anthropic
    credential, and has no analog of the Claude-SDK-specific 1M-context
    beta ``is_system_reviewer_role`` gates.

    Imports ``argus.config`` and ``argus.gemini_runner`` lazily (function
    body, not module level) for the same reason ``_claude_sdk_runner``
    imports lazily: avoids a needless import of the Gemini runner module
    for every caller of this module, even ones that never touch the
    ``gemini`` platform.
    """
    from argus.config import get_settings
    from argus.gemini_runner import run_session_gemini

    return await run_session_gemini(
        entry=entry,
        system_prompt=system_prompt,
        user_message=user_message,
        settings=get_settings(),
        label=label,
        repo_root=cwd,
        # Must win over run_session_gemini's own settings-based fallback --
        # the shared RunnerFn contract's caller (run_system_reviewer et al.)
        # already resolves this from the SAME settings object, so silently
        # discarding it and letting run_session_gemini re-derive its own
        # would only coincidentally agree, and diverges the moment a caller
        # passes a different settings object or an explicit override.
        timeout_s=timeout_s,
    )


async def _openai_runner(
    *,
    entry: BenchEntry,
    system_prompt: str,
    user_message: str,
    anthropic_api_key: str | None = None,
    anthropic_auth_token: str | None = None,
    context7_key: str | None = None,
    context7_library_id: str | None = None,
    context7_base_url: str | None = None,
    timeout_s: int,
    cwd: str,
    label: str,
    is_system_reviewer_role: bool = False,
) -> Any:
    """Thin adapter for the ``openai-responses`` platform.

    Mirrors ``_gemini_runner``'s adapter role exactly: translates from the
    shared ``RunnerFn`` signature into ``run_session_openai``'s signature.
    """
    from argus.config import get_settings
    from argus.openai_runner import run_session_openai

    return await run_session_openai(
        entry=entry,
        system_prompt=system_prompt,
        user_message=user_message,
        settings=get_settings(),
        label=label,
        repo_root=cwd,
        timeout_s=timeout_s,
    )


async def _unimplemented_runner(*, entry: BenchEntry, **_kwargs: Any) -> Any:
    """Stub for platforms wired into the enum but not yet implemented."""
    raise NotImplementedError(
        f"Bench platform {entry.platform!r} has no runner implementation yet "
        f"(role={entry.role!r}, model={entry.model!r})."
    )


PLATFORM_RUNNERS: Final[dict[str, RunnerFn]] = {
    "claude-sdk": _claude_sdk_runner,
    "gemini": _gemini_runner,
    "openai-responses": _openai_runner,
}


def runner_for(entry: BenchEntry) -> RunnerFn:
    """Return the runner function registered for ``entry.platform``."""
    try:
        return PLATFORM_RUNNERS[entry.platform]
    except KeyError as e:
        raise ValueError(f"No runner registered for platform {entry.platform!r}") from e
