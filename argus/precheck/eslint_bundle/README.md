# eslint security bundle

Self-contained `eslint` + `eslint-plugin-security` install, invoked directly
by `argus.precheck.js_scanner` -- independent of whatever eslint setup (or
lack thereof) the repo being reviewed has. Mirrors how `argus/precheck/rules/`
ships as data alongside this package, not as installable Python code.

## Setup (one-time, wherever this package runs live)

```bash
cd argus/precheck/eslint_bundle
npm install
```

`node_modules/` is deliberately **not** committed or shipped as part of the
Python package (a full JS toolchain, including eslint's own transitive
dependencies, is large and has its own cross-platform binary concerns --
not something a Python wheel should vendor). `argus.precheck.js_scanner`'s
`eslint_available()` checks for `node_modules/.bin/eslint` here and
gracefully skips the scan (same fail-open pattern as every other scanner
in this package) if this setup step hasn't been run.

## Why bundled instead of using the target repo's own eslint

Scanning with the *reviewed* repo's own eslint config would depend on that
repo having `eslint-plugin-security` (or an equivalent) configured at all --
the exact gap this integration exists to fill (see docs/PRECHECKS.md's
"stock rule sources" section). This config (`eslint.config.js`) applies only
`eslint-plugin-security`'s own recommended rules, nothing else.
