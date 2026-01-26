# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - GITHUB API CLIENT
# =============================================================================
"""
GitHub API Client

Low-level client for GitHub REST API interactions.
Handles authentication, rate limiting, and error handling.

Features:
    - Token and GitHub App authentication
    - Automatic rate limit handling
    - Retry logic with exponential backoff
    - Request/response logging

Usage:
    client = GitHubClient(token="ghp_xxx", repo="owner/repo")
    issue = client.get_issue(123)
    client.add_comment(123, "Hello from AI Agent!")
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


logger = logging.getLogger(__name__)


# =============================================================================
# EXCEPTIONS
# =============================================================================

class GitHubAPIError(Exception):
    """Base exception for GitHub API errors."""

    def __init__(self, message: str, status_code: int = None, response: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response or {}

    def __str__(self):
        if self.status_code:
            return f"[{self.status_code}] {super().__str__()}"
        return super().__str__()


class RateLimitError(GitHubAPIError):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str, reset_time: int = None):
        super().__init__(message, status_code=403)
        self.reset_time = reset_time


class NotFoundError(GitHubAPIError):
    """Raised when resource is not found."""

    def __init__(self, message: str):
        super().__init__(message, status_code=404)


class AuthenticationError(GitHubAPIError):
    """Raised when authentication fails."""

    def __init__(self, message: str):
        super().__init__(message, status_code=401)


class ValidationError(GitHubAPIError):
    """Raised when request validation fails."""

    def __init__(self, message: str, errors: list = None):
        super().__init__(message, status_code=422)
        self.errors = errors or []


# =============================================================================
# GITHUB CLIENT CLASS
# =============================================================================

class GitHubClient:
    """
    Low-level GitHub API client.

    Attributes:
        token: GitHub API token
        repo: Repository in owner/repo format
        base_url: GitHub API base URL
        session: HTTP session for requests

    Configuration:
        - Supports GitHub.com and GitHub Enterprise
        - Handles rate limiting automatically
        - Implements retry with backoff
    """

    DEFAULT_BASE_URL = "https://api.github.com"
    DEFAULT_TIMEOUT = 30
    DEFAULT_RETRY_COUNT = 3
    DEFAULT_BACKOFF_FACTOR = 0.5
    RATE_LIMIT_THRESHOLD = 10  # Wait when remaining requests below this

    def __init__(
        self,
        token: str = None,
        repo: str = None,
        base_url: str = None,
        timeout: int = None,
        retry_count: int = None,
        backoff_factor: float = None,
    ):
        """
        Initialize GitHub client.

        Args:
            token: GitHub API token (default: from GITHUB_TOKEN env)
            repo: Repository name (default: from GITHUB_REPO env)
            base_url: API base URL (default: github.com)
            timeout: Request timeout in seconds
            retry_count: Number of retries on failure
            backoff_factor: Backoff multiplier for retries

        Raises:
            ValueError: If token or repo not provided
        """
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.repo = repo or os.environ.get("GITHUB_REPO")
        self.base_url = (base_url or os.environ.get("GITHUB_API_URL") or
                         self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self.retry_count = retry_count if retry_count is not None else self.DEFAULT_RETRY_COUNT
        self.backoff_factor = backoff_factor or self.DEFAULT_BACKOFF_FACTOR

        if not self.token:
            raise ValueError(
                "GitHub token required. Set GITHUB_TOKEN environment variable "
                "or pass token parameter."
            )

        if not self.repo:
            raise ValueError(
                "GitHub repository required. Set GITHUB_REPO environment variable "
                "or pass repo parameter (format: owner/repo)."
            )

        # Validate repo format
        if "/" not in self.repo:
            raise ValueError(
                f"Invalid repository format: {self.repo}. "
                "Expected format: owner/repo"
            )

        self._session = self._create_session()
        self._rate_limit_remaining = None
        self._rate_limit_reset = None

        logger.info(f"GitHubClient initialized for {self.repo}")

    def _create_session(self) -> requests.Session:
        """Create configured HTTP session with retry logic."""
        session = requests.Session()

        # Set authentication headers
        session.headers.update({
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "AI-Agent-Development-System/1.0",
        })

        # Configure retry strategy
        retry_strategy = Retry(
            total=self.retry_count,
            backoff_factor=self.backoff_factor,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        return session

    # =========================================================================
    # ISSUE OPERATIONS
    # =========================================================================

    def get_issue(self, issue_number: int) -> dict:
        """
        Get issue by number.

        Args:
            issue_number: Issue number

        Returns:
            Issue data dictionary with:
            - number, title, body, state
            - labels, assignees
            - created_at, updated_at

        Raises:
            NotFoundError: If issue doesn't exist
            GitHubAPIError: If request fails
        """
        endpoint = f"/repos/{self.repo}/issues/{issue_number}"
        return self._request("GET", endpoint)

    def create_issue(
        self,
        title: str,
        body: str = None,
        labels: List[str] = None,
        assignees: List[str] = None,
        milestone: int = None,
    ) -> dict:
        """
        Create a new issue.

        Args:
            title: Issue title
            body: Issue body (markdown)
            labels: List of label names
            assignees: List of GitHub usernames
            milestone: Milestone number

        Returns:
            Created issue data
        """
        endpoint = f"/repos/{self.repo}/issues"
        data = {"title": title}

        if body is not None:
            data["body"] = body
        if labels:
            data["labels"] = labels
        if assignees:
            data["assignees"] = assignees
        if milestone is not None:
            data["milestone"] = milestone

        return self._request("POST", endpoint, data=data)

    def update_issue(
        self,
        issue_number: int,
        title: str = None,
        body: str = None,
        state: str = None,
        labels: List[str] = None,
        assignees: List[str] = None,
        milestone: int = None,
    ) -> dict:
        """
        Update an existing issue.

        Args:
            issue_number: Issue to update
            title: New title (optional)
            body: New body (optional)
            state: New state: "open" or "closed"
            labels: New labels (replaces existing)
            assignees: New assignees (replaces existing)
            milestone: New milestone number

        Returns:
            Updated issue data
        """
        endpoint = f"/repos/{self.repo}/issues/{issue_number}"
        data = {}

        if title is not None:
            data["title"] = title
        if body is not None:
            data["body"] = body
        if state is not None:
            if state not in ("open", "closed"):
                raise ValueError(f"Invalid state: {state}. Must be 'open' or 'closed'")
            data["state"] = state
        if labels is not None:
            data["labels"] = labels
        if assignees is not None:
            data["assignees"] = assignees
        if milestone is not None:
            data["milestone"] = milestone

        if not data:
            raise ValueError("At least one field must be provided for update")

        return self._request("PATCH", endpoint, data=data)

    def close_issue(self, issue_number: int) -> dict:
        """Close an issue."""
        return self.update_issue(issue_number, state="closed")

    def reopen_issue(self, issue_number: int) -> dict:
        """Reopen a closed issue."""
        return self.update_issue(issue_number, state="open")

    def list_issues(
        self,
        state: str = "open",
        labels: List[str] = None,
        assignee: str = None,
        creator: str = None,
        sort: str = "created",
        direction: str = "desc",
        since: str = None,
        per_page: int = 30,
        page: int = 1,
    ) -> List[dict]:
        """
        List issues matching criteria.

        Args:
            state: "open", "closed", or "all"
            labels: Filter by labels (comma-separated string or list)
            assignee: Filter by assignee username
            creator: Filter by creator username
            sort: "created", "updated", or "comments"
            direction: "asc" or "desc"
            since: Only issues updated after this ISO 8601 timestamp
            per_page: Results per page (max 100)
            page: Page number

        Returns:
            List of issue dictionaries
        """
        endpoint = f"/repos/{self.repo}/issues"
        params = {
            "state": state,
            "sort": sort,
            "direction": direction,
            "per_page": min(per_page, 100),
            "page": page,
        }

        if labels:
            params["labels"] = ",".join(labels) if isinstance(labels, list) else labels
        if assignee:
            params["assignee"] = assignee
        if creator:
            params["creator"] = creator
        if since:
            params["since"] = since

        return self._request("GET", endpoint, params=params)

    def list_all_issues(
        self,
        state: str = "open",
        labels: List[str] = None,
        **kwargs
    ) -> List[dict]:
        """
        List all issues matching criteria (handles pagination).

        Returns:
            Complete list of all matching issues
        """
        all_issues = []
        page = 1
        per_page = 100

        while True:
            issues = self.list_issues(
                state=state,
                labels=labels,
                per_page=per_page,
                page=page,
                **kwargs
            )

            if not issues:
                break

            all_issues.extend(issues)

            if len(issues) < per_page:
                break

            page += 1

        return all_issues

    # =========================================================================
    # LABEL OPERATIONS
    # =========================================================================

    def add_labels(self, issue_number: int, labels: List[str]) -> List[dict]:
        """
        Add labels to an issue.

        Args:
            issue_number: Issue to label
            labels: List of label names to add

        Returns:
            List of all labels on the issue
        """
        endpoint = f"/repos/{self.repo}/issues/{issue_number}/labels"
        return self._request("POST", endpoint, data={"labels": labels})

    def remove_label(self, issue_number: int, label: str) -> bool:
        """
        Remove a label from an issue.

        Args:
            issue_number: Issue number
            label: Label name to remove

        Returns:
            True if removed successfully
        """
        endpoint = f"/repos/{self.repo}/issues/{issue_number}/labels/{label}"
        try:
            self._request("DELETE", endpoint)
            return True
        except NotFoundError:
            # Label wasn't on issue, that's fine
            return False

    def set_labels(self, issue_number: int, labels: List[str]) -> List[dict]:
        """
        Replace all labels on an issue.

        Args:
            issue_number: Issue number
            labels: New list of labels (replaces existing)

        Returns:
            List of labels on the issue
        """
        endpoint = f"/repos/{self.repo}/issues/{issue_number}/labels"
        return self._request("PUT", endpoint, data={"labels": labels})

    def get_labels(self, issue_number: int) -> List[dict]:
        """Get all labels on an issue."""
        endpoint = f"/repos/{self.repo}/issues/{issue_number}/labels"
        return self._request("GET", endpoint)

    def create_label(
        self,
        name: str,
        color: str,
        description: str = None,
    ) -> dict:
        """
        Create a repository label.

        Args:
            name: Label name
            color: Color hex code (without #)
            description: Label description

        Returns:
            Created label data
        """
        endpoint = f"/repos/{self.repo}/labels"
        data = {
            "name": name,
            "color": color.lstrip("#"),
        }
        if description:
            data["description"] = description

        return self._request("POST", endpoint, data=data)

    def get_or_create_label(
        self,
        name: str,
        color: str = "ededed",
        description: str = None,
    ) -> dict:
        """Get existing label or create if it doesn't exist."""
        endpoint = f"/repos/{self.repo}/labels/{name}"
        try:
            return self._request("GET", endpoint)
        except NotFoundError:
            return self.create_label(name, color, description)

    def list_repo_labels(self, per_page: int = 100) -> List[dict]:
        """List all labels in the repository."""
        endpoint = f"/repos/{self.repo}/labels"
        return self._request("GET", endpoint, params={"per_page": per_page})

    # =========================================================================
    # COMMENT OPERATIONS
    # =========================================================================

    def add_comment(self, issue_number: int, body: str) -> dict:
        """
        Add a comment to an issue.

        Args:
            issue_number: Issue to comment on
            body: Comment body (markdown)

        Returns:
            Created comment data
        """
        endpoint = f"/repos/{self.repo}/issues/{issue_number}/comments"
        return self._request("POST", endpoint, data={"body": body})

    def update_comment(self, comment_id: int, body: str) -> dict:
        """
        Update an existing comment.

        Args:
            comment_id: Comment ID to update
            body: New comment body

        Returns:
            Updated comment data
        """
        endpoint = f"/repos/{self.repo}/issues/comments/{comment_id}"
        return self._request("PATCH", endpoint, data={"body": body})

    def delete_comment(self, comment_id: int) -> bool:
        """
        Delete a comment.

        Args:
            comment_id: Comment ID to delete

        Returns:
            True if deleted successfully
        """
        endpoint = f"/repos/{self.repo}/issues/comments/{comment_id}"
        self._request("DELETE", endpoint)
        return True

    def list_comments(
        self,
        issue_number: int,
        since: str = None,
        per_page: int = 30,
        page: int = 1,
    ) -> List[dict]:
        """
        Get all comments on an issue.

        Args:
            issue_number: Issue number
            since: Only comments updated after this ISO 8601 timestamp
            per_page: Results per page
            page: Page number

        Returns:
            List of comment dictionaries
        """
        endpoint = f"/repos/{self.repo}/issues/{issue_number}/comments"
        params = {"per_page": per_page, "page": page}
        if since:
            params["since"] = since

        return self._request("GET", endpoint, params=params)

    # =========================================================================
    # REPOSITORY OPERATIONS
    # =========================================================================

    def get_repository(self) -> dict:
        """Get repository information."""
        endpoint = f"/repos/{self.repo}"
        return self._request("GET", endpoint)

    def get_contents(self, path: str, ref: str = None) -> dict:
        """
        Get contents of a file or directory.

        Args:
            path: Path to file/directory
            ref: Git reference (branch, tag, commit)

        Returns:
            File or directory contents
        """
        endpoint = f"/repos/{self.repo}/contents/{path}"
        params = {}
        if ref:
            params["ref"] = ref

        return self._request("GET", endpoint, params=params if params else None)

    # =========================================================================
    # RATE LIMIT HANDLING
    # =========================================================================

    def get_rate_limit(self) -> dict:
        """
        Get current rate limit status.

        Returns:
            {
                "limit": 5000,
                "remaining": 4999,
                "reset": 1234567890,
                "used": 1
            }
        """
        endpoint = "/rate_limit"
        response = self._request("GET", endpoint, check_rate_limit=False)
        return response.get("resources", {}).get("core", response.get("rate", {}))

    def _update_rate_limit(self, response: requests.Response):
        """Update rate limit info from response headers."""
        try:
            self._rate_limit_remaining = int(
                response.headers.get("X-RateLimit-Remaining", 0)
            )
            self._rate_limit_reset = int(
                response.headers.get("X-RateLimit-Reset", 0)
            )
        except (ValueError, TypeError):
            pass

    def _check_rate_limit(self):
        """
        Check rate limit and wait if necessary.

        If remaining requests are low, waits until reset.
        """
        if self._rate_limit_remaining is None:
            return

        if self._rate_limit_remaining <= self.RATE_LIMIT_THRESHOLD:
            if self._rate_limit_reset:
                wait_time = max(0, self._rate_limit_reset - time.time()) + 1
                if wait_time > 0 and wait_time < 3600:  # Don't wait more than 1 hour
                    logger.warning(
                        f"Rate limit low ({self._rate_limit_remaining} remaining). "
                        f"Waiting {wait_time:.0f} seconds..."
                    )
                    time.sleep(wait_time)

    # =========================================================================
    # HTTP METHODS
    # =========================================================================

    def _request(
        self,
        method: str,
        endpoint: str,
        data: dict = None,
        params: dict = None,
        check_rate_limit: bool = True,
    ) -> Any:
        """
        Make authenticated API request.

        Handles:
        - Authentication headers
        - Rate limit checking
        - Error responses
        - Retry logic
        """
        if check_rate_limit:
            self._check_rate_limit()

        url = urljoin(self.base_url, endpoint)

        logger.debug(f"GitHub API: {method} {endpoint}")

        try:
            response = self._session.request(
                method=method,
                url=url,
                json=data,
                params=params,
                timeout=self.timeout,
            )

            self._update_rate_limit(response)

            # Handle errors
            if response.status_code >= 400:
                self._handle_error(response)

            # Return empty dict for 204 No Content
            if response.status_code == 204:
                return {}

            return response.json()

        except requests.exceptions.Timeout:
            raise GitHubAPIError(f"Request timed out: {method} {endpoint}")
        except requests.exceptions.ConnectionError as e:
            raise GitHubAPIError(f"Connection error: {e}")
        except requests.exceptions.RequestException as e:
            raise GitHubAPIError(f"Request failed: {e}")

    def _handle_error(self, response: requests.Response) -> None:
        """
        Handle error response from API.

        Raises appropriate exception based on status code.
        """
        status_code = response.status_code

        try:
            error_data = response.json()
            message = error_data.get("message", response.text)
            errors = error_data.get("errors", [])
        except Exception:
            message = response.text
            errors = []

        logger.error(f"GitHub API error [{status_code}]: {message}")

        if status_code == 401:
            raise AuthenticationError(
                "Authentication failed. Check your GitHub token."
            )

        if status_code == 403:
            if "rate limit" in message.lower():
                raise RateLimitError(
                    message,
                    reset_time=self._rate_limit_reset
                )
            raise GitHubAPIError(message, status_code, error_data)

        if status_code == 404:
            raise NotFoundError(f"Resource not found: {message}")

        if status_code == 422:
            raise ValidationError(message, errors)

        raise GitHubAPIError(message, status_code)

    # =========================================================================
    # CONTEXT MANAGER
    # =========================================================================

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close session."""
        self.close()

    def close(self):
        """Close the HTTP session."""
        if self._session:
            self._session.close()
            logger.debug("GitHubClient session closed")