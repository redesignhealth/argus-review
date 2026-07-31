"""Central model alias registry.

Single source of truth for which concrete LLM each logical alias resolves to.
To upgrade a model family across the codebase (e.g. GPT-5.4 -> GPT-5.5), edit
``ALIAS_MAP`` below and nothing else.

Call sites should import the resolved constants (``GPT_MINI``,
``CLAUDE_FRONTIER``, etc.) rather than hardcoding model strings like
``"gpt-5.5-mini"``.

Only aliases this package actually calls are registered here -- this is a
standalone, isolated package, not the larger monorepo it was extracted from,
so there's no value in carrying registry entries (gpt-frontier, gpt-nano, the
Gemini family) with no real call site. Add an alias back if a future call
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
"""

from typing import Any, Final

ALIAS_MAP: Final[dict[str, str]] = {
    # OpenAI -- gpt-5 family
    "gpt-mini": "gpt-5.4-mini",  # bump to gpt-5.5-mini once OpenAI ships it
    # Anthropic
    "claude-frontier": "claude-fable-5",
    "claude-opus": "claude-opus-5",
    "claude-default": "claude-sonnet-5",
    "claude-mini": "claude-haiku-4-5",
}

# Short-lived pins for evals / preview models. Anything in here is a known
# escape hatch from the alias system -- add a comment with the tracking
# issue/PR and expected removal date.
EXPERIMENTAL_MODELS: Final[dict[str, str]] = {}

GPT_MINI: Final[str] = ALIAS_MAP["gpt-mini"]
CLAUDE_FRONTIER: Final[str] = ALIAS_MAP["claude-frontier"]
CLAUDE_OPUS: Final[str] = ALIAS_MAP["claude-opus"]
CLAUDE_DEFAULT: Final[str] = ALIAS_MAP["claude-default"]
CLAUDE_MINI: Final[str] = ALIAS_MAP["claude-mini"]


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

    Works for both registry aliases (``claude-default``, ``gpt-mini``)
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
