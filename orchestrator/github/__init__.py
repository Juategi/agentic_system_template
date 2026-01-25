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
"""

__all__ = [
    "GitHubClient",
    "IssueManager",
    "WebhookHandler",
]
