# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - GITHUB INTEGRATION PACKAGE
# =============================================================================
"""
GitHub Integration Package

This package provides GitHub API integration for the orchestrator.
It handles all communication with GitHub for issue management.

Components:
    - GitHubClient: Low-level API client
    - IssueManager: High-level issue operations
    - WebhookHandler: Process incoming webhooks

Features:
    - Issue CRUD operations
    - Label management
    - Comment posting
    - Webhook processing
    - Rate limit handling

Authentication:
    Supports both Personal Access Tokens and GitHub Apps.

Usage:
    from orchestrator.github import GitHubClient, GitHubAPIError

    client = GitHubClient(token="ghp_xxx", repo="owner/repo")
    issue = client.get_issue(123)
"""

from orchestrator.github.client import (
    GitHubClient,
    GitHubAPIError,
    RateLimitError,
    NotFoundError,
    AuthenticationError,
    ValidationError,
)

from orchestrator.github.issue_manager import (
    IssueManager,
    create_issue_manager,
    # Enums
    WorkflowLabel,
    IssueType,
    # Templates
    COMMENT_TEMPLATES,
    # Constants
    ALL_STATE_LABELS,
    LABEL_COLORS,
    # Exceptions
    IssueManagerError,
    IssueTransitionError,
    IssueNotFoundError,
)

__all__ = [
    # Client
    "GitHubClient",
    # Client Exceptions
    "GitHubAPIError",
    "RateLimitError",
    "NotFoundError",
    "AuthenticationError",
    "ValidationError",
    # Issue Manager
    "IssueManager",
    "create_issue_manager",
    # Issue Enums
    "WorkflowLabel",
    "IssueType",
    # Issue Templates
    "COMMENT_TEMPLATES",
    # Issue Constants
    "ALL_STATE_LABELS",
    "LABEL_COLORS",
    # Issue Exceptions
    "IssueManagerError",
    "IssueTransitionError",
    "IssueNotFoundError",
]