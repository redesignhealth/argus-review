# Security Policy

## Reporting a Vulnerability

Please report security vulnerabilities using
[GitHub's private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability):

1. Go to the repository's **Security** tab.
2. Click **Report a vulnerability** under "Advisories".
3. Fill in as much detail as you can: affected version, reproduction steps,
   and impact.

This opens a private channel with the maintainers — please do **not** open
a public GitHub issue for a security report.

## What's in scope

Argus orchestrates LLM calls (Claude Agent SDK, LangChain/LangGraph) against
a target repo's PR diff, and can optionally write back to GitHub (PR
comments, commit statuses) and to a storage backend (SQLite/Postgres/HTTP).
Vulnerability classes we care about most:

- Prompt injection via PR/diff content that escalates into unintended tool
  use (e.g. arbitrary file writes, shell execution, exfiltration of
  credentials) beyond the documented review/write-back surface.
- Credential handling: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
  `GITHUB_TOKEN_RO`, storage backend URLs/auth — any path that could leak
  these into logs, prompts, or the review output itself.
- Storage backend injection (SQL injection against the Postgres backend,
  SSRF via the HTTP storage mode URL templates).

## Supported versions

Security fixes are made against the latest released minor version. Given
this project is pre-1.0, we do not maintain older release branches.

## Response

We aim to acknowledge reports within a few business days and to keep the
reporter updated as we investigate and (if applicable) prepare a fix and
coordinated disclosure.
