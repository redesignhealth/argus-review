"""Central model alias registry.

Single source of truth for which concrete LLM each logical alias resolves to.
To upgrade a model family across the codebase (e.g. GPT-5.4 -> GPT-5.5), edit
``ALIAS_MAP`` below and nothing else.

Call sites should import the resolved constants (``GPT_MINI``,
``CLAUDE_FRONTIER``, etc.) rather than hardcoding model strings like
``"gpt-5.5-mini"``.

Only aliases with a real call site are registered in ``ALIAS_MAP`` -- this is
a standalone, isolated package, not the larger monorepo it was extracted
from, so there's no value in carrying registry entries with no real caller.
``gemini-frontier``/``gemini-mini`` ARE real call sites (Track 3's
``argus.gemini_runner``, dispatched via ``argus.bench``'s ``"gemini"``
platform, resolves ``entry.model`` through this same ``ALIAS_MAP``) and stay
registered; ``gpt-frontier``/``gpt-nano`` have no runner yet (``argus.bench``'s
``"openai-responses"`` platform is ``_unimplemented_runner``) and are NOT
registered here -- their concrete model ids are kept as bare, non-alias
constants below purely so ``_PRICES``/``estimate_cost_usd`` has a complete,
future-proof cost table. Add an alias back to ``ALIAS_MAP`` if a future call
site actually needs it.

Tier semantics:
    *frontier* -- best reasoning available in the family; slow / expensive.
    *opus*     -- next tier down from frontier -- strong reasoning at roughly
                  half frontier's per-token cost. Introduced for call sites
                  (e.g. the cross-cutting reviewer) where evals showed no
                  measurable quality gain from frontier, so the cost isn't
                  justified.
    *default*  -- workhorse balance of cost and capability.
    *mini*     -- fast and cheap; suitable for high-volume, low-stakes calls.

Runtime overrides:
    ``ARGUS_SPECIALIST_MODEL`` (``--specialist-model``) overrides
    ``CLAUDE_DEFAULT`` -- the model used by the system reviewer, specialist
    reviewers, the writer, and the lite-review path.

    ``ARGUS_FRONTIER_MODEL`` (``--frontier-model``) overrides both
    ``CLAUDE_FRONTIER`` (planner, coverage) and ``CLAUDE_OPUS``
    (cross-cutting) -- there is a single frontier-tier knob at the CLI, even
    though the two aliases keep independent defaults so cross-cutting still
    runs on the cheaper Opus tier when no override is given.

    Both env vars must be set before this module is first imported --
    ``cli.py`` sets them from the CLI flags ahead of any deferred import of
    ``argus.graph``/``argus.runners``, the same ordering
    ``ARGUS_NO_PROMPT_OVERRIDES`` relies on.
"""

import logging
import os
from typing import Any, Final

logger = logging.getLogger(__name__)

ALIAS_MAP: Final[dict[str, str]] = {
    # OpenAI -- gpt-5 family
    "gpt-mini": "gpt-5.4-mini",  # bump to gpt-5.5-mini once OpenAI ships it
    # Anthropic
    "claude-frontier": "claude-fable-5",
    "claude-opus": "claude-opus-5",
    "claude-default": "claude-sonnet-4-6",
    "claude-mini": "claude-haiku-4-5",
    # Google -- gemini-3 family. Real call site: argus.gemini_runner
    # (Track 3), dispatched via argus.bench's "gemini" platform.
    # NOTE: gemini-3-pro-preview was deprecated by Google (404 as of 2026-06).
    # gemini-3.1-pro-preview is the current working successor.
    # TODO: upgrade to stable gemini-3.1-pro (non-preview) when GA;
    # re-eval by 2026-09-01.
    "gemini-frontier": "gemini-3.1-pro-preview",
    "gemini-mini": "gemini-3-flash-preview",
}

# Short-lived pins for evals / preview models. Anything in here is a known
# escape hatch from the alias system -- add a comment with the tracking
# issue/PR and expected removal date.
EXPERIMENTAL_MODELS: Final[dict[str, str]] = {}

_PROVIDER_BY_FAMILY: Final[dict[str, str]] = {
    "gpt": "openai",
    "claude": "anthropic",
    # init_chat_model expects "google_genai", not "google".
    "gemini": "google_genai",
}


def infer_provider(alias_or_model: str) -> str:
    """Return the LangChain provider name for an alias or concrete model id.

    Works for both registry aliases (``claude-default``, ``gpt-mini``)
    and concrete model strings (``claude-sonnet-4-6``, ``gpt-5.5-mini``) by
    matching on the family prefix before the first ``-``.

    Raises ``ValueError`` for unknown families so unsupported providers fail
    loudly rather than silently routing to a wrong client.

    Defined ahead of ``_env_override`` (and the module-level constants that
    call it) in this file specifically so ``_env_override`` can call it at
    import time -- moving it below those constants, as it once was, would
    make that call a ``NameError``.
    """
    family = alias_or_model.lower().split("-", 1)[0]
    try:
        return _PROVIDER_BY_FAMILY[family]
    except KeyError as e:
        raise ValueError(
            f"Unsupported model family '{family}' (got {alias_or_model!r}); "
            f"must be one of {sorted(_PROVIDER_BY_FAMILY)}",
        ) from e


def _env_override(env_var: str, default: str) -> str:
    """Return the env var's value if set and non-empty, else ``default``.

    Resolves the raw value through ``ALIAS_MAP`` first (``ALIAS_MAP.get(value,
    value)``), so passing a registry alias (e.g. ``claude-default``) behaves
    the same as passing the concrete model id it resolves to -- otherwise a
    caller who reaches for an alias key instead of a real model id gets that
    literal string sent straight to the provider as an unrecognized model.
    This is deliberately MORE permissive than ``resolve()``, not equivalent
    to it: ``resolve()`` raises ``KeyError`` on an unrecognized alias, so a
    typo fails loudly at the call site; ``ALIAS_MAP.get(value, value)``
    instead falls through and returns the raw string unchanged, deferring
    any failure to whatever the provider API does with an unrecognized
    model id. That's intentional here -- an override value is expected to
    often be a real (non-registry) model id, not always an alias.

    When the value falls through unchanged (not a recognized alias), this
    logs a warning -- not a raise -- if it also doesn't look like a valid
    provider-prefixed model id (``infer_provider`` can't map its family
    prefix to a known provider). A raise would reject legitimate model ids
    this registry simply doesn't know about yet (providers ship new models
    on their own schedule); a silent pass-through would leave a plain typo
    (e.g. ``cluade-default``) to surface only much later as an opaque
    provider-side "model not found" error. The warning is a middle ground:
    it doesn't block anything, but gives a caller a signal closer to the
    actual mistake.
    """
    value = os.environ.get(env_var)
    if not value:
        return default
    resolved = ALIAS_MAP.get(value, value)
    if resolved == value and value not in ALIAS_MAP:
        try:
            infer_provider(resolved)
        except ValueError:
            logger.warning(
                "%s=%r is not a recognized ALIAS_MAP key and does not look like a "
                "valid provider-prefixed model id (e.g. claude-*, gpt-*, gemini-*) "
                "-- this override may fail at the provider.",
                env_var,
                value,
            )
    return resolved


GPT_MINI: Final[str] = ALIAS_MAP["gpt-mini"]
CLAUDE_FRONTIER: Final[str] = _env_override("ARGUS_FRONTIER_MODEL", ALIAS_MAP["claude-frontier"])
CLAUDE_OPUS: Final[str] = _env_override("ARGUS_FRONTIER_MODEL", ALIAS_MAP["claude-opus"])
CLAUDE_DEFAULT: Final[str] = _env_override("ARGUS_SPECIALIST_MODEL", ALIAS_MAP["claude-default"])
CLAUDE_MINI: Final[str] = ALIAS_MAP["claude-mini"]
GEMINI_FRONTIER: Final[str] = ALIAS_MAP["gemini-frontier"]
GEMINI_MINI: Final[str] = ALIAS_MAP["gemini-mini"]

# Not registered in ALIAS_MAP (see module docstring) -- no OpenAI Responses
# runner exists yet to actually call these, but the concrete ids are kept as
# bare constants so the pricing table below stays complete and forward-
# compatible with the day a real runner ships.
GPT_FRONTIER: Final[str] = "gpt-5.5"
GPT_NANO: Final[str] = "gpt-5-nano"


def resolve(alias: str) -> str:
    """Resolve a model alias to its concrete model name.

    Raises ``KeyError`` if the alias is not registered, so typos fail loudly
    rather than silently routing to a wrong model.
    """
    return ALIAS_MAP[alias]


# Per-Mtok USD pricing: model name -> (input_cost, output_cost, cache_read_cost).
# Deliberately NOT litellm-backed -- this is a lean, dependency-free OSS CLI,
# so pricing is a small hand-maintained table instead of a third-party
# pricing-data dependency.
#
# Anthropic entries mirror the per-token constants already used for
# lite-review cost aggregation in argus/graph.py (_SONNET_INPUT_COST etc.,
# $3/$15/$0.30 per Mtok for claude-sonnet-4-6) -- kept in sync manually,
# there is no single source of truth spanning both modules yet. CLAUDE_MINI
# (Haiku) follows Anthropic's published Sonnet-relative pricing ratio
# (~0.27x Sonnet input/output). CLAUDE_FRONTIER (claude-fable-5) keeps the
# same ~5x-Sonnet ratio the pre-fable frontier alias (an actual Opus model)
# used. CLAUDE_OPUS is new (main introduced a separate "opus" tier distinct
# from "frontier") and is priced at roughly half of CLAUDE_FRONTIER's rate,
# matching this module's own "opus... roughly half frontier's per-token
# cost" tier semantics above. None of these Anthropic estimates beyond
# CLAUDE_DEFAULT are independently exercised by any existing
# cost-aggregation call site today, only estimate_cost_usd below.
#
# Gemini entries are REAL: argus.gemini_runner is a real, reachable runner
# (see argus.bench's "gemini" platform / PLATFORM_RUNNERS), so a $0/$0/$0
# placeholder here would silently under-report every Gemini session's cost.
# Modeled on Google's publicly published Gemini 2.5 Pro/Flash per-Mtok rates
# (<=200k-token context tier; cached-content input priced at roughly a
# quarter of the full input rate, matching Google's own published ratio for
# that tier).
# TODO(gemini-pricing): these are plausible-but-NOT-independently-verified
# placeholders -- confirm against Google's CURRENT published rates for the
# specific gemini-3.1-pro-preview/gemini-3-flash-preview models this
# registry actually resolves to (see ALIAS_MAP above). Preview-tier pricing
# in particular is prone to changing without notice; re-check before
# relying on this for real billing reconciliation.
#
# GPT rows are still placeholders: no OpenAI Responses runner exists yet
# (argus.bench's "openai-responses" platform is _unimplemented_runner), so
# these models are not reachable by any runner. Update with real per-Mtok
# rates once a runner ships.
_PRICES: Final[dict[str, tuple[float, float, float]]] = {
    CLAUDE_DEFAULT: (3.00, 15.00, 0.30),
    CLAUDE_FRONTIER: (15.00, 75.00, 1.50),
    CLAUDE_OPUS: (7.50, 37.50, 0.75),
    CLAUDE_MINI: (0.80, 4.00, 0.08),
    GEMINI_FRONTIER: (1.25, 10.00, 0.31),  # TODO(gemini-pricing): confirm, see comment above
    GEMINI_MINI: (0.30, 2.50, 0.075),  # TODO(gemini-pricing): confirm, see comment above
    GPT_FRONTIER: (0.0, 0.0, 0.0),  # placeholder -- no OpenAI Responses runner yet
    GPT_MINI: (0.0, 0.0, 0.0),  # placeholder -- no OpenAI Responses runner yet
    GPT_NANO: (0.0, 0.0, 0.0),  # placeholder -- no OpenAI Responses runner yet
}


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float:
    """Estimate USD cost for one LLM call from token counts.

    ``cached_input_tokens`` is a SEPARATE token count billed at the
    cache-read rate, not a subset of ``input_tokens`` to subtract out --
    matches the convention already used for cost aggregation in
    ``argus/graph.py``'s lite-review path (and how Anthropic's own usage
    object reports ``input_tokens`` and ``cache_read_input_tokens`` as
    distinct counters, not one nested in the other).

    ``model`` must be a concrete model name with a ``_PRICES`` entry
    (e.g. ``CLAUDE_DEFAULT``, not the alias key ``"claude-default"``).

    Raises ``KeyError`` if ``model`` has no pricing entry -- fails loudly
    rather than silently reporting a $0.00 cost for an unrecognized model.
    """
    try:
        input_cost, output_cost, cache_read_cost = _PRICES[model]
    except KeyError:
        raise KeyError(
            f"No pricing entry for model {model!r}; add one to _PRICES in argus/llm/models.py"
        ) from None

    return (
        input_tokens / 1_000_000 * input_cost
        + cached_input_tokens / 1_000_000 * cache_read_cost
        + output_tokens / 1_000_000 * output_cost
    )


def build_chat_model(alias: str) -> Any:
    """Build a LangChain chat model from a registry alias key.

    Combines ``resolve`` + ``infer_provider`` + ``init_chat_model`` into the
    single call sites typically need. Use this from agent hosts, structured-
    output pipelines, anywhere you'd otherwise re-derive the
    ``provider:model`` string by hand.

    Raises ``KeyError`` on unknown alias (delegated to ``resolve``);
    ``ValueError`` on unknown provider family (delegated to ``infer_provider``).
    """
    from langchain.chat_models import init_chat_model

    return init_chat_model(f"{infer_provider(alias)}:{resolve(alias)}")
