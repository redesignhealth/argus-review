# Releasing Argus

This is the maintainer runbook for cutting a release. Most contributors
don't need this — see [`CONTRIBUTING.md`](../CONTRIBUTING.md) instead.

## Versioning

The version is **single-sourced from the git tag** via
[`hatch-vcs`](https://github.com/ofek/hatch-vcs) (`[tool.hatch.version]` in
`pyproject.toml`, `source = "vcs"`). There is no hand-maintained version
literal to bump — `pyproject.toml`'s `[project]` table declares
`dynamic = ["version"]` and hatchling asks hatch-vcs to derive it from git at
build time.

- A build made straight from a tag `v0.1.0` resolves to version `0.1.0`.
- A build made from an untagged commit resolves to a PEP 440 dev version
  like `0.1.dev5+gabc1234.d20260706` (commits-since-tag + short SHA + date).
  This is normal for CI runs on `main` between releases and for local dev
  builds — don't chase it, tag a release when you actually want a clean
  version.

Tag format: `vMAJOR.MINOR.PATCH` for a stable release, `vMAJOR.MINOR.PATCHrcN`
for a release candidate (e.g. `v0.1.0rc1`). The word `rc` in the tag name is
what the release workflow uses to decide `prerelease: true` on the GitHub
release — keep that substring exact.

## Cutting a release

1. Make sure `main` is green (CI passing) at the commit you want to release.
2. Move [`CHANGELOG.md`](../CHANGELOG.md)'s `[Unreleased]` entries under a
   new `## [X.Y.Z] - YYYY-MM-DD` heading (leave `[Unreleased]` in place,
   empty, for the next round of changes). Add the compare-link footer for
   the version you're about to tag, e.g.
   `[X.Y.Z]: https://github.com/redesignhealth/argus-review/compare/vPREV...vX.Y.Z`
   (or `.../commits/vX.Y.Z` for the very first tag, which has no previous
   tag to diff against). Commit and push this to `main` before tagging:
   ```bash
   git add CHANGELOG.md
   git commit -m "chore: update CHANGELOG for vX.Y.Z"
   git push origin main
   ```
3. Tag the resulting commit and push the tag:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```
4. This triggers `.github/workflows/release.yml`, which:
   - re-runs the full test suite against the tagged commit,
   - builds the sdist + wheel with `uv build`,
   - generates `dist/SHA256SUMS`,
   - creates a GitHub release with the sdist, wheel, and checksums attached
     (marked prerelease automatically if the tag contains `rc`),
   - auto-generates release notes from merged PRs since the last tag.
5. Watch the **Actions** tab for the `Release` workflow to go green.

## Verifying a release

After the workflow completes:

```bash
# Download and verify checksums
gh release download v0.1.0 --repo redesignhealth/argus-review -D /tmp/argus-release
cd /tmp/argus-release
shasum -a 256 -c SHA256SUMS

# Sanity-install and check the console script resolves
uv venv /tmp/argus-check
uv pip install --python /tmp/argus-check argus_code_review-0.1.0-py3-none-any.whl
/tmp/argus-check/bin/argus --help
/tmp/argus-check/bin/argus --version   # confirms 0.1.0, matching the tag
```

Confirm the GitHub release page lists both `argus_code_review-*.whl` and
`argus_code_review-*.tar.gz` plus `SHA256SUMS`.

## PyPI / TestPyPI publishing (currently inert)

`release.yml` includes `publish-testpypi` (rc tags) and `publish-pypi`
(stable tags) jobs using
[PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/) (OIDC —
no stored API tokens). Both are gated behind the repository variable
`vars.PUBLISH_PYPI == 'true'` and are **skipped** until that's set, because
whether Argus publishes to PyPI at all (vs. GitHub-release-wheels-only, the
current distribution mechanism the `argus-review-loop` skill's installer
expects) is an open decision — see the project's issue tracker for status.

To enable PyPI publishing once that decision is made:

1. Register the project on PyPI (and TestPyPI, if using rc verification) and
   configure a trusted publisher pointing at this repo +
   `.github/workflows/release.yml` + the `pypi` (and `testpypi`) GitHub
   Environments.
2. Create the `pypi` and `testpypi` GitHub Environments in repo settings
   (used for environment-scoped OIDC + optional required reviewers).
3. Set the repository variable `PUBLISH_PYPI=true` (Settings → Secrets and
   variables → Actions → Variables).
4. Cut a release as above; the publish jobs will now run instead of being
   skipped.

## Packaging notes

- **Prompt files (`argus/prompts/*.md`) ship in the wheel.** They live
  inside the `argus` package directory, and hatchling's default wheel
  packaging (`[tool.hatch.build.targets.wheel] packages = ["argus"]`) copies
  the whole package tree, so no extra `include`/`force-include` config is
  needed. `tests/test_packaging.py` builds the wheel and asserts all prompt
  files are present, so a future packaging config change that silently
  drops them will fail CI.
- **`schema/*.sql` (Postgres DDL for the postgres storage backend) does
  NOT ship in the wheel** — it's outside the `argus` package, so hatchling
  never includes it there. It *does* ship in the sdist (see
  `[tool.hatch.build.targets.sdist] include` in `pyproject.toml`) and lives
  in the repo for self-hosters to apply directly (see `docs/STORAGE.md`).
  This is intentional: it's schema for an external database, not a runtime
  resource `argus` imports, so it doesn't belong in the installable
  library artifact.

## License

`LICENSE` is Apache-2.0, copyright "Redesign Health" 2026. Confirmed final.
