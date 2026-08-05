// Bundled, minimal eslint flat config -- applies only eslint-plugin-security's
// own recommended rules (all "warn" severity, not "error": verified
// empirically that this keeps eslint's own exit code at 0 regardless of
// findings, matching semgrep/zizmor/Trivy's convention rather than needing
// a findings_exit_code special case). Not the target repo's own eslint
// config -- this file is invoked directly via its own path, independent of
// whatever (if anything) the reviewed repo has configured.
const security = require("eslint-plugin-security");

module.exports = [
  {
    plugins: { security },
    rules: { ...security.configs.recommended.rules },
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "module",
    },
  },
];
