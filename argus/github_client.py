"""GitHub REST API client for lightweight metadata fetching.

This client fetches PR and commit metadata (titles, messages, file paths)
for context. Full diffs and detailed queries should use MCP tools to avoid
blowing up LLM context windows.

Requirements:
- GITHUB_TOKEN_RO environment variable or passed to constructor

Usage:
    from argus.github_client import GitHubClient

    client = GitHubClient()
    prs = client.list_pull_requests("owner/repo", days_back=14)
    commits = client.list_commits("owner/repo", days_back=14)
"""

from __future__ import annotations

import logging
import os
import re
from datetime import UTC, datetime, timedelta
from typing import Any, TypedDict, cast

import httpx

from argus.config import get_settings

logger = logging.getLogger(__name__)

# Hidden marker embedded in Argus's own PR comments so re-runs can find and
# update the prior comment (upsert) instead of stacking new ones. Scoped to
# repo/pr so a shared marker string can't collide across repos.
_ARGUS_COMMENT_MARKER_TMPL = "<!-- argus-review: {repo}#{pr} -->"

# owner/repo — same shape as argus.repo_provision._REPO_RE: letters, digits,
# dots, hyphens, underscores; no leading dot/hyphen in either segment (a
# common path-traversal trick, and a leading dot also yields an invalid
# clone URL). Applied to every write-capable API path below so a malformed
# `repo` can't escape `/repos/{repo}/...` into an unrelated endpoint (e.g.
# `owner/repo/../../user/keys`) or break the HTML-comment marker via `-->`.
_WRITE_REPO_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]*/[A-Za-z0-9_][A-Za-z0-9._-]*$")

# Commit SHA for write-capable endpoints (commit statuses): hex digits only,
# 4-40 chars (git's minimum unambiguous abbreviation through a full SHA).
# Rejects `/`, `..`, and any other character that could be used to escape
# the URL path.
_WRITE_SHA_RE = re.compile(r"^[0-9a-f]{4,40}$")


def _validate_repo_for_write(repo: str) -> None:
    """Raise ValueError if ``repo`` isn't a safe ``owner/repo`` string.

    Guards every write-capable GitHub API call (and the HTML comment marker
    embedded in PR comments) against path traversal / injection via an
    attacker- or misconfigured-controlled ``repo`` value.
    """
    if not _WRITE_REPO_RE.match(repo) or ".." in repo:
        raise ValueError(f"Invalid repo format: {repo!r}. Expected 'owner/repo'.")


def _validate_sha_for_write(sha: str) -> None:
    """Raise ValueError if ``sha`` isn't a safe hex commit SHA (7-40 chars)."""
    if not _WRITE_SHA_RE.match(sha):
        raise ValueError(f"Invalid sha format: {sha!r}. Expected 7-40 lowercase hex characters.")


def _resolve_write_token() -> tuple[str, bool]:
    """Resolve a token for write operations (PR comments, commit statuses).

    Resolution order: ``GITHUB_TOKEN`` (expected to be write-capable), then
    ``GITHUB_TOKEN_RO`` (read-only PAT; may lack write scope — callers using
    the fallback should surface a clear error on a 403 rather than a raw
    GitHub API error).

    Returns:
        ``(token, is_ro_fallback)``.

    Raises:
        GitHubClientError: If neither variable is set.
    """
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token, False

    # Read the env var directly rather than going through ``get_settings()``:
    # the full Settings model requires ANTHROPIC_API_KEY/OPENAI_API_KEY too,
    # which would raise a ValidationError unrelated to GitHub write access
    # in contexts (e.g. tests, ``prompts`` subcommands) that never load
    # those secrets.
    ro_token = os.environ.get("GITHUB_TOKEN_RO")
    if not ro_token:
        raise GitHubClientError(
            "No GitHub token available for write operations. Set GITHUB_TOKEN "
            "to a write-capable token."
        )
    return ro_token, True


class GitHubClientError(Exception):
    """Base exception for GitHub client errors."""

    pass


class GitHubAPIError(GitHubClientError):
    """Raised when GitHub API returns an error."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"GitHub API error ({status_code}): {message}")


class CompareMetadata(TypedDict):
    """Typed return shape for :meth:`GitHubClient.get_compare_metadata`.

    Mirrors the subset of fields we read from GitHub's ``/compare`` response.
    """

    status: str  # "identical" | "ahead" | "behind" | "diverged" (or "" on malformed)
    merge_base_sha: str  # "" when the response omits ``merge_base_commit``
    ahead_by: int
    behind_by: int


class CommitParentInfo(TypedDict):
    """Typed return shape for each entry in :meth:`GitHubClient.get_compare_commits`.

    Contains only the fields needed for merge-commit detection.
    """

    sha: str
    parent_count: (
        int  # 0 = root commit / missing parents (non-merge); 1 = normal commit; >1 = merge commit
    )


class GitHubClient:
    """Client for GitHub REST API (lightweight metadata only).

    This client fetches PR and commit metadata for LLM context.
    For full diffs or detailed queries, use GitHub MCP tools.

    Attributes:
        token: GitHub personal access token or app token.
        base_url: GitHub API base URL.
    """

    DEFAULT_BASE_URL = "https://api.github.com"
    DEFAULT_TIMEOUT = 30.0

    # Set by ``for_writes()`` when the write-capable ``GITHUB_TOKEN`` env var
    # is absent and the client fell back to ``GITHUB_TOKEN_RO``. Lets the
    # write helpers below surface a clearer error on a 403 than a raw
    # GitHubAPIError would.
    _using_ro_fallback: bool = False

    def __init__(
        self,
        token: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ):
        """Initialize GitHub client.

        Args:
            token: GitHub token. If not provided, reads from Settings.
            base_url: Override the default API URL.
            timeout: HTTP request timeout in seconds.

        Raises:
            ValueError: If no token is provided or found.
        """
        # Token from constructor, Settings, or error
        if token is None:
            settings = get_settings()
            token = settings.GITHUB_TOKEN_RO

        if not token:
            raise ValueError(
                "GitHub token is required. Pass token to constructor "
                "or configure GITHUB_TOKEN_RO in Settings."
            )

        self.token = token

        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.timeout = timeout if timeout is not None else self.DEFAULT_TIMEOUT
        self._headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Make an authenticated request to GitHub API.

        Args:
            method: HTTP method.
            endpoint: API endpoint (e.g., "/repos/owner/repo/pulls").
            params: Query parameters.

        Returns:
            JSON response data.

        Raises:
            GitHubAPIError: If the API returns an error.
        """
        url = f"{self.base_url}{endpoint}"
        with httpx.Client(timeout=self.timeout) as client:
            response = client.request(
                method,
                url,
                headers=self._headers,
                params=params,
            )
            if not response.is_success:
                raise GitHubAPIError(response.status_code, response.text)
            return response.json()

    def _paginate(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        max_items: int = 100,
    ) -> list[dict[str, Any]]:
        """Paginate through GitHub API results.

        Args:
            endpoint: API endpoint.
            params: Query parameters.
            max_items: Maximum items to fetch.

        Returns:
            List of all items.
        """
        params = params or {}
        params["per_page"] = min(100, max_items)
        page = 1
        all_items: list[dict[str, Any]] = []

        while len(all_items) < max_items:
            params["page"] = page
            items = self._request("GET", endpoint, params)

            if not items:
                break

            all_items.extend(items)
            if len(items) < params["per_page"]:
                break

            page += 1

        return all_items[:max_items]

    # =========================================================================
    # Pull Request Methods
    # =========================================================================

    def list_pull_requests(
        self,
        repo: str,
        state: str = "all",
        days_back: int = 14,
        max_items: int = 50,
    ) -> list[dict[str, Any]]:
        """List pull requests with lightweight metadata.

        Args:
            repo: Repository in "owner/repo" format.
            state: PR state - "open", "closed", or "all".
            days_back: Only include PRs updated in last N days.
            max_items: Maximum PRs to return.

        Returns:
            List of PR metadata dicts with keys:
            - number, title, state, user, created_at, updated_at, merged_at
            - head_branch, base_branch
            - draft, mergeable_state
            - files_changed (list of file paths, fetched separately)
        """
        cutoff = datetime.now(UTC) - timedelta(days=days_back)
        params = {
            "state": state,
            "sort": "updated",
            "direction": "desc",
        }

        prs = self._paginate(f"/repos/{repo}/pulls", params, max_items * 2)

        # Filter by date and extract lightweight metadata
        results = []
        for pr in prs:
            updated_at = datetime.fromisoformat(pr["updated_at"].replace("Z", "+00:00"))
            if updated_at < cutoff:
                continue

            # Get list of changed file paths (lightweight - just paths)
            files = self._get_pr_file_paths(repo, pr["number"])

            results.append(
                {
                    "number": pr["number"],
                    "title": pr["title"],
                    "state": pr["state"],
                    "draft": pr.get("draft", False),
                    "user": pr["user"]["login"],
                    "head_branch": pr["head"]["ref"],
                    "base_branch": pr["base"]["ref"],
                    "created_at": pr["created_at"],
                    "updated_at": pr["updated_at"],
                    "merged_at": pr.get("merged_at"),
                    "files_changed": files,
                    "url": pr["html_url"],
                }
            )

            if len(results) >= max_items:
                break

        logger.info(
            "Fetched %d PRs from %s (last %d days)",
            len(results),
            repo,
            days_back,
        )
        return results

    def _get_pr_file_paths(self, repo: str, pr_number: int) -> list[str]:
        """Get list of file paths changed in a PR (no diff content).

        Args:
            repo: Repository in "owner/repo" format.
            pr_number: Pull request number.

        Returns:
            List of file paths changed.
        """
        try:
            files = self._request("GET", f"/repos/{repo}/pulls/{pr_number}/files")
            return [f["filename"] for f in files]
        except GitHubAPIError as e:
            logger.warning("Failed to get files for PR #%d: %s", pr_number, e)
            return []

    def get_pull_request(self, repo: str, pr_number: int) -> dict[str, Any]:
        """Get a single pull request with metadata.

        Args:
            repo: Repository in "owner/repo" format.
            pr_number: Pull request number.

        Returns:
            PR metadata dict.
        """
        pr = self._request("GET", f"/repos/{repo}/pulls/{pr_number}")
        files = self._get_pr_file_paths(repo, pr_number)

        return {
            "number": pr["number"],
            "title": pr["title"],
            "body": pr.get("body", ""),
            "state": pr["state"],
            "draft": pr.get("draft", False),
            "user": pr["user"]["login"],
            "head_branch": pr["head"]["ref"],
            "head_sha": pr["head"]["sha"],
            "base_branch": pr["base"]["ref"],
            "created_at": pr["created_at"],
            "updated_at": pr["updated_at"],
            "merged_at": pr.get("merged_at"),
            "files_changed": files,
            "url": pr["html_url"],
            "additions": pr.get("additions", 0),
            "deletions": pr.get("deletions", 0),
            "changed_files": pr.get("changed_files", 0),
        }

    def _fetch_diff(self, url: str, max_lines: int) -> str:
        """Fetch a unified diff from the given URL with truncation.

        Args:
            url: Full GitHub API URL that returns a diff.
            max_lines: Maximum lines of diff to return (truncates if exceeded).

        Returns:
            Unified diff as a string.
        """
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(
                url,
                headers={
                    **self._headers,
                    "Accept": "application/vnd.github.diff",
                },
            )
            if not response.is_success:
                raise GitHubAPIError(response.status_code, response.text)

            diff = response.text
            lines = diff.split("\n")

            if len(lines) > max_lines:
                truncated = lines[:max_lines]
                truncated.append(f"... (truncated, {len(lines) - max_lines} more lines)")
                return "\n".join(truncated)

            return diff

    def get_pr_diff(
        self,
        repo: str,
        pr_number: int,
        max_lines: int = 500,
    ) -> str:
        """Get the unified diff for a pull request.

        Args:
            repo: Repository in "owner/repo" format.
            pr_number: Pull request number.
            max_lines: Maximum lines of diff to return (truncates if exceeded).

        Returns:
            Unified diff as a string.
        """
        url = f"{self.base_url}/repos/{repo}/pulls/{pr_number}"
        return self._fetch_diff(url, max_lines)

    def get_compare_diff(
        self,
        repo: str,
        base: str,
        head: str,
        max_lines: int = 500,
    ) -> str:
        """Get the unified diff between two refs via the compare endpoint.

        Uses GET /repos/{repo}/compare/{base}...{head} which returns the true
        net diff — what would change if head merged into base right now. This
        avoids stale changes that the PR diff endpoint can include when the
        base branch has moved ahead.

        Args:
            repo: Repository in "owner/repo" format.
            base: Base ref (branch name or SHA).
            head: Head ref (branch name or SHA).
            max_lines: Maximum lines of diff to return (truncates if exceeded).

        Returns:
            Unified diff as a string.
        """
        url = f"{self.base_url}/repos/{repo}/compare/{base}...{head}"
        return self._fetch_diff(url, max_lines)

    def get_compare_metadata(
        self,
        repo: str,
        base: str,
        head: str,
    ) -> CompareMetadata:
        """Get compare metadata (status, merge_base, ahead/behind counts).

        Calls the same ``/compare/{base}...{head}`` endpoint as ``get_compare_diff``
        but requests the JSON payload instead of the diff body. Useful for
        detecting rebases: if ``status`` is ``"diverged"``, ``base`` is no longer
        an ancestor of ``head``.

        Args:
            repo: Repository in "owner/repo" format.
            base: Base ref (branch name or SHA).
            head: Head ref (branch name or SHA).

        Returns:
            :class:`CompareMetadata` with ``status`` (``"identical"``,
            ``"ahead"``, ``"behind"``, or ``"diverged"``), ``merge_base_sha``,
            ``ahead_by``, and ``behind_by``.
        """
        data = self._request("GET", f"/repos/{repo}/compare/{base}...{head}")
        merge_base = (data.get("merge_base_commit") or {}).get("sha") or ""
        return CompareMetadata(
            status=data.get("status", ""),
            merge_base_sha=merge_base,
            ahead_by=data.get("ahead_by", 0),
            behind_by=data.get("behind_by", 0),
        )

    def get_compare_commits(
        self,
        repo: str,
        base: str,
        head: str,
    ) -> list[CommitParentInfo]:
        """Return the commits in ``base..head`` with their parent counts.

        Calls the same ``/compare/{base}...{head}`` JSON endpoint as
        :meth:`get_compare_metadata` but extracts the ``commits`` array
        (capped to the first 250 GitHub returns without pagination).
        Each entry carries only the SHA and parent count; the rest of the
        commit payload is discarded to keep the return value small.

        Used by the round-2 catch-up-merge gate to decide whether every
        commit since the prior review is a merge commit (parent_count > 1).

        Args:
            repo: Repository in "owner/repo" format.
            base: Base ref (branch name or SHA).
            head: Head ref (branch name or SHA).

        Returns:
            List of :class:`CommitParentInfo` dicts, one per commit in
            ``base..head``, in the order GitHub returns them (oldest first).
            Empty list when ``base`` and ``head`` are identical.
        """
        data = self._request("GET", f"/repos/{repo}/compare/{base}...{head}")
        total_commits: int = data.get("total_commits") or 0
        commits: list[CommitParentInfo] = []
        for c in data.get("commits") or []:
            sha = (c.get("sha") or "").strip()
            if not sha:
                logger.warning(
                    "get_compare_commits: skipping commit with missing sha in %s %s..%s",
                    repo,
                    base,
                    head,
                )
                continue
            parents = c.get("parents") or []
            commits.append(CommitParentInfo(sha=sha, parent_count=len(parents)))
        # GitHub caps the commits array at 250. If total_commits exceeds the
        # returned count, the list is truncated and the gate result would be
        # unreliable — return empty so the caller degrades safely to False.
        if total_commits > len(commits):
            logger.warning(
                "get_compare_commits: result truncated (%d returned, %d total) "
                "for %s %s..%s — returning empty to force safe-degrade",
                len(commits),
                total_commits,
                repo,
                base,
                head,
            )
            return []
        return commits

    def get_pr_files_with_diff(
        self,
        repo: str,
        pr_number: int,
        max_files: int = 20,
    ) -> list[dict[str, Any]]:
        """Get files changed in a PR with their patches.

        Args:
            repo: Repository in "owner/repo" format.
            pr_number: Pull request number.
            max_files: Maximum number of files to return.

        Returns:
            List of file objects with filename, status, additions, deletions, patch.
        """
        files = self._request("GET", f"/repos/{repo}/pulls/{pr_number}/files")

        results = []
        for f in files[:max_files]:
            results.append(
                {
                    "filename": f["filename"],
                    "status": f["status"],  # added, removed, modified, renamed
                    "additions": f.get("additions", 0),
                    "deletions": f.get("deletions", 0),
                    "patch": f.get("patch", ""),  # The actual diff for this file
                }
            )

        if len(files) > max_files:
            logger.info(
                "Truncated PR files from %d to %d for PR #%d",
                len(files),
                max_files,
                pr_number,
            )

        return results

    def get_pr_reviews(
        self,
        repo: str,
        pr_number: int,
    ) -> list[dict[str, Any]]:
        """Get reviews for a pull request.

        Args:
            repo: Repository in "owner/repo" format.
            pr_number: Pull request number.

        Returns:
            List of review objects with user, state, body, submitted_at.
        """
        reviews = self._request("GET", f"/repos/{repo}/pulls/{pr_number}/reviews")

        results = []
        for review in reviews:
            results.append(
                {
                    "id": review["id"],
                    "user": review["user"]["login"],
                    "state": review["state"],  # APPROVED, CHANGES_REQUESTED, COMMENTED, etc.
                    "body": review.get("body", ""),
                    "submitted_at": review.get("submitted_at"),
                }
            )

        logger.info(
            "Fetched %d reviews for PR #%d in %s",
            len(results),
            pr_number,
            repo,
        )
        return results

    def get_pr_review_comments(
        self,
        repo: str,
        pr_number: int,
    ) -> list[dict[str, Any]]:
        """Get review comments (inline comments) for a pull request.

        Args:
            repo: Repository in "owner/repo" format.
            pr_number: Pull request number.

        Returns:
            List of comment objects with user, body, path, line, created_at.
        """
        comments = self._request("GET", f"/repos/{repo}/pulls/{pr_number}/comments")

        results = []
        for comment in comments:
            results.append(
                {
                    "id": comment["id"],
                    "user": comment["user"]["login"],
                    "body": comment.get("body", ""),
                    "path": comment.get("path", ""),
                    "line": comment.get("line"),
                    "original_line": comment.get("original_line"),
                    "created_at": comment.get("created_at"),
                    "updated_at": comment.get("updated_at"),
                }
            )

        logger.info(
            "Fetched %d review comments for PR #%d in %s",
            len(results),
            pr_number,
            repo,
        )
        return results

    def list_issue_comments(
        self,
        repo: str,
        issue_number: int,
    ) -> list[dict[str, Any]]:
        """Get issue/PR conversation comments (not inline review comments).

        Args:
            repo: Repository in "owner/repo" format.
            issue_number: Issue or pull request number.

        Returns:
            List of comment objects with id, user, body, created_at.
        """
        comments = self._request("GET", f"/repos/{repo}/issues/{issue_number}/comments")

        results = []
        for comment in comments:
            results.append(
                {
                    "id": comment["id"],
                    "user": comment["user"]["login"],
                    "body": comment.get("body", ""),
                    "created_at": comment.get("created_at"),
                }
            )

        logger.info(
            "Fetched %d issue comments for #%d in %s",
            len(results),
            issue_number,
            repo,
        )
        return results

    # =========================================================================
    # Commit Methods
    # =========================================================================

    def list_commits(
        self,
        repo: str,
        branch: str = "main",
        days_back: int = 14,
        max_items: int = 100,
    ) -> list[dict[str, Any]]:
        """List commits with lightweight metadata.

        Args:
            repo: Repository in "owner/repo" format.
            branch: Branch to list commits from.
            days_back: Only include commits from last N days.
            max_items: Maximum commits to return.

        Returns:
            List of commit metadata dicts with keys:
            - sha, message (first line only), author, date, url
        """
        since = (datetime.now(UTC) - timedelta(days=days_back)).isoformat()
        params = {
            "sha": branch,
            "since": since,
        }

        commits = self._paginate(f"/repos/{repo}/commits", params, max_items)

        results = []
        for commit in commits:
            # Extract first line of commit message
            full_message = commit["commit"]["message"]
            first_line = full_message.split("\n")[0]

            results.append(
                {
                    "sha": commit["sha"][:7],  # Short SHA
                    "full_sha": commit["sha"],
                    "message": first_line,
                    "author": commit["commit"]["author"]["name"],
                    "author_username": commit["author"]["login"] if commit.get("author") else None,
                    "date": commit["commit"]["author"]["date"],
                    "url": commit["html_url"],
                }
            )

        logger.info(
            "Fetched %d commits from %s/%s (last %d days)",
            len(results),
            repo,
            branch,
            days_back,
        )
        return results

    # =========================================================================
    # Branch Methods
    # =========================================================================

    def list_branches(
        self,
        repo: str,
        pattern: str | None = None,
        max_items: int = 100,
    ) -> list[dict[str, Any]]:
        """List branches, optionally filtered by pattern.

        Args:
            repo: Repository in "owner/repo" format.
            pattern: Optional pattern to filter branches (case-insensitive contains).
            max_items: Maximum branches to return.

        Returns:
            List of branch metadata dicts with keys:
            - name, sha, protected
        """
        branches = self._paginate(f"/repos/{repo}/branches", max_items=max_items)

        results = []
        for branch in branches:
            name = branch["name"]

            # Filter by pattern if provided
            if pattern and pattern.lower() not in name.lower():
                continue

            results.append(
                {
                    "name": name,
                    "sha": branch["commit"]["sha"][:7],
                    "protected": branch.get("protected", False),
                }
            )

        if pattern:
            logger.info(
                "Found %d branches matching '%s' in %s",
                len(results),
                pattern,
                repo,
            )
        else:
            logger.info("Fetched %d branches from %s", len(results), repo)

        return results

    def find_branches_for_ticket(
        self,
        repo: str,
        ticket_id: str,
    ) -> list[dict[str, Any]]:
        """Find branches that contain a ticket ID.

        Args:
            repo: Repository in "owner/repo" format.
            ticket_id: Ticket identifier (e.g., "PROJ-490").

        Returns:
            List of matching branch metadata dicts.
        """
        return self.list_branches(repo, pattern=ticket_id)

    # =========================================================================
    # Write-back (CLI ``--post`` / ``--commit-status``)
    # =========================================================================

    @classmethod
    def for_writes(
        cls,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> "GitHubClient":
        """Construct a client using a write-capable token.

        Resolves ``GITHUB_TOKEN`` first, falling back to ``GITHUB_TOKEN_RO``
        (which may lack write scope — a 403 from a write call made on the
        fallback token raises a clearer error via :meth:`_write_request`).

        Raises:
            GitHubClientError: If neither token is available.
        """
        token, is_fallback = _resolve_write_token()
        client = cls(token=token, base_url=base_url, timeout=timeout)
        client._using_ro_fallback = is_fallback
        return client

    def _write_request(self, method: str, endpoint: str, json_body: dict[str, Any]) -> Any:
        """Make an authenticated write (POST/PATCH) request with a JSON body.

        Additive sibling to ``_request`` (which only supports query params).
        On a 403 while using the ``GITHUB_TOKEN_RO`` fallback, raises a
        GitHubAPIError with an actionable message instead of the raw GitHub
        response body.

        Args:
            method: HTTP method (``"POST"`` or ``"PATCH"``).
            endpoint: API endpoint (e.g. "/repos/owner/repo/issues/1/comments").
            json_body: JSON-serializable request body.

        Returns:
            JSON response data.

        Raises:
            GitHubAPIError: If the API returns an error.
        """
        url = f"{self.base_url}{endpoint}"
        with httpx.Client(timeout=self.timeout) as client:
            response = client.request(method, url, headers=self._headers, json=json_body)
            if response.status_code == 403 and self._using_ro_fallback:
                raise GitHubAPIError(
                    403,
                    "GITHUB_TOKEN_RO lacks write scope; set GITHUB_TOKEN to a "
                    "write-capable token to use --post/--commit-status.",
                )
            if not response.is_success:
                raise GitHubAPIError(response.status_code, response.text)
            return response.json()

    def find_argus_comment(self, repo: str, pr_number: int) -> dict[str, Any] | None:
        """Find Argus's prior comment on a PR by its hidden marker, if any.

        Paginates through *all* issue comments (not just the first 100) so a
        PR with a long comment history still finds the marker; otherwise
        ``upsert_pr_comment`` would create a duplicate comment instead of
        updating the existing one.

        Args:
            repo: Repository in "owner/repo" format.
            pr_number: Pull request number.

        Returns:
            The comment dict (as returned by the GitHub issue-comments
            endpoint) if a prior Argus comment exists, else ``None``.
        """
        _validate_repo_for_write(repo)
        marker = _ARGUS_COMMENT_MARKER_TMPL.format(repo=repo, pr=pr_number)
        per_page = 100
        page = 1
        while True:
            items = self._request(
                "GET",
                f"/repos/{repo}/issues/{pr_number}/comments",
                {"per_page": per_page, "page": page},
            )
            if not items:
                return None
            for comment in items:
                if marker in (comment.get("body") or ""):
                    return cast("dict[str, Any]", comment)
            if len(items) < per_page:
                return None
            page += 1

    def upsert_pr_comment(self, repo: str, pr_number: int, body: str) -> dict[str, Any]:
        """Create or update Argus's review comment on a PR.

        Finds a prior Argus comment (by hidden ``<!-- argus-review: ... -->``
        marker scoped to this repo/pr) and PATCHes it if found, else POSTs a
        new issue comment. This gives idempotent re-runs: a re-review updates
        the existing comment rather than stacking a new one each round.

        Args:
            repo: Repository in "owner/repo" format.
            pr_number: Pull request number.
            body: Review markdown body (the marker is prepended automatically).

        Returns:
            The created/updated comment dict.

        Raises:
            ValueError: If ``repo`` isn't a safe ``owner/repo`` string.
            GitHubAPIError: If the API returns an error (including a clearer
                403 message when falling back to a read-only token).
        """
        _validate_repo_for_write(repo)
        marker = _ARGUS_COMMENT_MARKER_TMPL.format(repo=repo, pr=pr_number)
        tagged_body = f"{marker}\n{body}"
        existing = self.find_argus_comment(repo, pr_number)
        if existing is not None:
            comment = cast(
                "dict[str, Any]",
                self._write_request(
                    "PATCH",
                    f"/repos/{repo}/issues/comments/{existing['id']}",
                    {"body": tagged_body},
                ),
            )
            logger.info(
                "Updated existing Argus comment %s on %s PR #%d", existing["id"], repo, pr_number
            )
            return comment
        comment = cast(
            "dict[str, Any]",
            self._write_request(
                "POST", f"/repos/{repo}/issues/{pr_number}/comments", {"body": tagged_body}
            ),
        )
        logger.info("Created new Argus comment on %s PR #%d", repo, pr_number)
        return comment

    def set_commit_status(
        self,
        repo: str,
        sha: str,
        state: str,
        description: str,
        context: str = "argus/review",
        target_url: str | None = None,
    ) -> dict[str, Any]:
        """Set a commit status on ``sha``.

        Args:
            repo: Repository in "owner/repo" format.
            sha: Full commit SHA to set the status on.
            state: One of ``"success"``, ``"failure"``, ``"error"``, ``"pending"``.
            description: Short human-readable description (GitHub truncates
                to 140 chars; this method truncates first to avoid a 422).
            context: Status context/label. Defaults to ``"argus/review"``.
            target_url: Optional URL the status links to.

        Returns:
            The created commit-status dict.

        Raises:
            ValueError: If ``repo`` or ``sha`` isn't a safe value.
            GitHubAPIError: If the API returns an error (including a clearer
                403 message when falling back to a read-only token).
        """
        _validate_repo_for_write(repo)
        _validate_sha_for_write(sha)
        payload: dict[str, Any] = {
            "state": state,
            "context": context,
            "description": description[:140],
        }
        if target_url:
            payload["target_url"] = target_url
        status = cast(
            "dict[str, Any]", self._write_request("POST", f"/repos/{repo}/statuses/{sha}", payload)
        )
        logger.info("Set commit status %r (%s) on %s@%s", context, state, repo, sha[:12])
        return status
