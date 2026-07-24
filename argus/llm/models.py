"""Central model alias registry.

Single source of truth for which concrete LLM each logical alias resolves to.
To upgrade a model family across the codebase (e.g. GPT-5.4 -> GPT-5.5), edit
``ALIAS_MAP`` below and nothing else.

Call sites should import the resolved constants (``GPT_FRONTIER``, ``GPT_MINI``,
``CLAUDE_FRONTIER``, etc.) rather than hardcoding model strings like
``"gpt-5.5-mini"``.

Tier semantics:
    *frontier* -- best reasoning available in the family; slow / expensive.
    *default*  -- workhorse balance of cost and capability.
    *mini*     -- fast and cheap; suitable for high-volume, low-stakes calls.
    *nano*     -- smallest / cheapest tier (OpenAI only).

Validation: a live integration test (``tests/integration/test_alias_live.py``)
hits each provider with the resolved name and confirms the model is recognized.
Run it after editing ``ALIAS_MAP``.
"""

from typing import Any, Final

ALIAS_MAP: Final[dict[str, str]] = {
    # OpenAI -- gpt-5 family
    "gpt-frontier": "gpt-5.5",
    "gpt-mini": "gpt-5.4-mini",  # bump to gpt-5.5-mini once OpenAI ships it
    "gpt-nano": "gpt-5-nano",
    # Anthropic -- claude-4 family
    "claude-frontier": "claude-opus-4-7",
    "claude-default": "claude-sonnet-4-6",
    "claude-mini": "claude-haiku-4-5",
    # Google -- gemini-3 family
    # NOTE: gemini-3-pro-preview was deprecated by Google (404 as of 2026-06).
    # gemini-3.1-pro-preview is the current working successor; confirmed live
    # in prod via slides-to-text (EXPERIMENTAL_MODELS). The integration test
    # will catch if these names stop resolving.
    # TODO: upgrade to stable gemini-3.1-pro (non-preview) when GA;
    # re-eval by 2026-09-01.
    "gemini-frontier": "gemini-3.1-pro-preview",
    "gemini-mini": "gemini-3-flash-preview",
}

EMBEDDING_SMALL: Final[str] = "text-embedding-3-small"
EMBEDDING_LARGE: Final[str] = "text-embedding-3-large"

# Native output dim for each embedding model. Keep in lockstep with the
# pgvector column declarations in any model that uses them
# (e.g. ``Concept.embedding``). text-embedding-3-large supports OpenAI's
# matryoshka truncation via the ``dimensions=`` parameter, so callers can
# choose a smaller output if storage / index limits demand it; the
# defaults below are the **native** dims you get without truncation.
EMBEDDING_SMALL_DIM: Final[int] = 1536
EMBEDDING_LARGE_DIM: Final[int] = 3072

# Short-lived pins for evals / preview models. Anything in here is a known
# escape hatch from the alias system -- add a comment with the ticket and
# expected removal date. The check_model_strings lint allows these names.
EXPERIMENTAL_MODELS: Final[dict[str, str]] = {
    # claude-fable-5 confirmed in Anthropic usage API as of 2026-06-30.
    # Pricing: $10/$50 input/output per M tokens (same as claude-fable-5 GA tier).
    # TODO: promote to ALIAS_MAP["claude-frontier"] once team adopts for inference.
    "claude-fable-5": "claude-fable-5",
}

GPT_FRONTIER: Final[str] = ALIAS_MAP["gpt-frontier"]
GPT_MINI: Final[str] = ALIAS_MAP["gpt-mini"]
GPT_NANO: Final[str] = ALIAS_MAP["gpt-nano"]
CLAUDE_FRONTIER: Final[str] = ALIAS_MAP["claude-frontier"]
CLAUDE_DEFAULT: Final[str] = ALIAS_MAP["claude-default"]
CLAUDE_MINI: Final[str] = ALIAS_MAP["claude-mini"]
GEMINI_FRONTIER: Final[str] = ALIAS_MAP["gemini-frontier"]
GEMINI_MINI: Final[str] = ALIAS_MAP["gemini-mini"]


def resolve(alias: str) -> str:
    """Resolve a model alias to its concrete model name.

    Raises ``KeyError`` if the alias is not registered, so typos fail loudly
    rather than silently routing to a wrong model.
    """
    return ALIAS_MAP[alias]


_PROVIDER_BY_FAMILY: Final[dict[str, str]] = {
    "gpt": "openai",
    "claude": "anthropic",
    # init_chat_model expects "google_genai", not "google".
    "gemini": "google_genai",
}


def infer_provider(alias_or_model: str) -> str:
    """Return the LangChain provider name for an alias or concrete model id.

    Works for both registry aliases (``claude-default``, ``gpt-frontier``)
    and concrete model strings (``claude-sonnet-4-6``, ``gpt-5.5-mini``) by
    matching on the family prefix before the first ``-``.

    Raises ``ValueError`` for unknown families so unsupported providers fail
    loudly rather than silently routing to a wrong client.
    """
    family = alias_or_model.lower().split("-", 1)[0]
    try:
        return _PROVIDER_BY_FAMILY[family]
    except KeyError as e:
        raise ValueError(
            f"Unsupported model family '{family}' (got {alias_or_model!r}); "
            f"must be one of {sorted(_PROVIDER_BY_FAMILY)}",
        ) from e


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
