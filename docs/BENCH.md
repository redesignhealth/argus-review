# Reviewer bench

The reviewer bench is Argus's static, config-driven assignment of an LLM
platform, model, and caching mode to each leaf reviewer. It exists so a team
can make deliberate, reviewable choices about where reviewer work runs—for
example, moving the reviewer fan-out to Gemini without changing the review
graph or prompt files.

The bench is **not** dynamic or adaptive routing. Configuration is human-edited
and should be committed and reviewed like any other operational configuration.
The packaged default preserves the pre-bench Claude behavior. The bench applies
to the leaf reviewers; planner, writer, coverage, and lite-review paths remain
bench-independent.

## Configuration units

The TOML configuration has two kinds of unit. It is not a per-role table for
every reviewer:

- `[bulk_reviewer]` is one shared platform/model/caching setting for all
  generalist-shaped reviewers: each system-group reviewer (including gap-fill
  reviewers), every named specialist, and the tests-and-docs reviewer. These
  reviewers continue to use their own existing packaged prompt files; only
  platform, model, and caching are shared.
- `[roles.<name>]` configures an individual role independently. The currently
  wired individual roles are `cross-cutting`, `blocking-validator`, and
  `feedback-verifier`. These entries also specify `prompt_name`.

All of the leaf roles currently defined by the bench are wired to their runner
functions. An override for an unknown or future role is validated and emits a
warning that the role is not wired, so it has no effect on an actual review run.

## Override chain

Argus starts with the packaged `argus/bench_default.toml` and sparse-merges the
following layers in this order, from lowest to highest priority:

1. Packaged `argus/bench_default.toml`.
2. `~/.config/argus/bench.toml`, respecting `$XDG_CONFIG_HOME` when it is set.
3. `./.argus/bench.toml`, resolved as `Path.cwd() / ".argus" / "bench.toml"`.
   This means it is relative to the working directory of the `argus` process
   when it runs, not necessarily the repository root.
4. The file named by `ARGUS_BENCH_FILE`, an explicit override file merged on
   top of all the others. If this path does not exist or is not a file, Argus
   raises an error instead of silently ignoring the request.

The standard user and working-directory files are optional and skipped when
absent. Every overlay is **sparse**: include only the keys you want to change;
omitted keys fall through to the layer below.

Set `ARGUS_NO_BENCH_OVERRIDES` to a truthy value to skip layers 2–4 and force
the packaged default only. This is useful for CI or official runs that must
not accidentally inherit a developer's local configuration.

## Values and credentials

Valid `platform` values are exactly:

- `"claude-sdk"`
- `"gemini"`
- `"openai-responses"`

Valid `caching` values are `"auto"` (the default), `"on"`, and `"off"`. However, how caching is handled depends on the platform runner:

- **`gemini`**: The Gemini runner actively consumes and respects `caching`. `"auto"` and `"on"` enable explicit context caching (with graceful degradation if prompt and toolset size fall below Gemini's minimum cache threshold), while `"off"` skips cache creation and usage entirely.
- **`claude-sdk`**: Does not use explicit caching. If `caching` is explicitly set to anything other than `"auto"` (e.g. `"on"` or `"off"`), the runner logs a warning and ignores the setting.
- **`openai-responses`**: Does not inspect `entry.caching`. OpenAI's Responses API handles prompt caching automatically on the server side (for prefixes >= 1024 tokens) without manual cache control.

### Platform credentials

Because the bench configures leaf reviewers only, full reviews still require baseline Anthropic, OpenAI, and GitHub credentials for bench-independent pipeline stages (including repository cloning, diff fetching, the planner, coverage, the writer, and the lite-review path) regardless of which platform leaf reviewers use. See [Managing secrets in the README](../README.md#managing-secrets) for the complete baseline credential list.

Leaf reviewers additionally require credentials for their configured platform:

| Platform | Required environment | Optional environment |
| --- | --- | --- |
| `claude-sdk` | `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` | — |
| `openai-responses` | `OPENAI_API_KEY` | `OPENAI_BASE_URL` |
| `gemini` | `GOOGLE_API_KEY` | `GOOGLE_BASE_URL` |

The Gemini runner raises `ValueError("GOOGLE_API_KEY is required")` when its
credential is missing. The base URL variables are useful when routing a
provider through a compatible proxy or gateway.

Bench configuration is validated once when it is loaded, before review work
starts. Unknown top-level or table keys, invalid platforms or caching values,
missing required keys, unknown model aliases, and invalid packaged prompt names
fail loudly with `ValueError`. There is no minimum-tier safety floor: the
bench's defaults are expected to be PR-reviewed, and runtime overrides are
treated as an intentional operator choice.

## Example: use Gemini for generalists and specialists

Because `[bulk_reviewer]` is the shared unit for the system-generalist,
specialist, and tests-and-docs reviewers, one sparse overlay can move both the
generalist and specialist reviewers to Gemini:

```toml
[bulk_reviewer]
platform = "gemini"
model    = "gemini-mini"
caching  = "auto"
```

At the time of writing, `gemini-mini` resolves through
`argus/llm/models.py`'s `ALIAS_MAP` to `gemini-3.8-flash`. The alias is
preferable to hard-coding the provider model ID: the alias can be updated by a
reviewed Argus change while this configuration remains stable.

Place this snippet in `.argus/bench.toml` for the current process working
directory, in the XDG user config for a user-wide setting, or in an explicit
file selected with `ARGUS_BENCH_FILE`. Ensure `GOOGLE_API_KEY` is available to
the process before running the review.

For an individual role, use a sparse `[roles.<name>]` overlay instead. Because
the overlay layers are merged recursively onto `argus/bench_default.toml`, an
existing packaged role (such as `cross-cutting`, `blocking-validator`, or
`feedback-verifier`) only requires the specific keys you want to change—for
example, specifying only `model = "claude-frontier"` or `platform = "openai-responses"`.
All omitted keys fall through and inherit from the default layer. Defining all
required keys (`platform`, `model`, and `prompt_name`; with `caching` optional
and defaulting to `"auto"`) is only necessary when configuring a brand-new role
that does not exist in the packaged default.
