# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - ISSUE MANAGER
# =============================================================================
"""
Issue Manager

High-level interface for managing GitHub Issues in the workflow context.
Provides workflow-aware operations on top of the raw API client.

Responsibilities:
    - Translate workflow states to GitHub labels
    - Query issues by workflow state
    - Manage issue lifecycle
    - Handle issue comments for agent communication
    - Create and manage subtasks
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import (
    Dict,
    List,
    Any,
    Optional,
    Set,
    TYPE_CHECKING,
)

from orchestrator.github.client import GitHubClient, GitHubAPIError

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================


class WorkflowLabel(Enum):
    """Workflow state labels for GitHub issues."""
    READY = "READY"
    IN_PROGRESS = "IN_PROGRESS"
    PLANNING = "PLANNING"
    DEVELOPMENT = "DEVELOPMENT"
    QA = "QA"
    QA_FAILED = "QA_FAILED"
    REVIEW = "REVIEW"
    DOCUMENTATION = "DOCUMENTATION"
    BLOCKED = "BLOCKED"
    DONE = "DONE"
    AWAIT_SUBTASKS = "AWAIT_SUBTASKS"


class IssueType(Enum):
    """Issue type labels."""
    FEATURE = "feature"
    TASK = "task"
    BUG = "bug"
    SUBTASK = "subtask"


# All workflow state labels (used for cleanup when transitioning)
ALL_STATE_LABELS = {label.value for label in WorkflowLabel}

# Label colors for auto-creation
LABEL_COLORS = {
    "READY": "0e8a16",
    "IN_PROGRESS": "fbca04",
    "PLANNING": "c5def5",
    "DEVELOPMENT": "1d76db",
    "QA": "5319e7",
    "QA_FAILED": "e11d48",
    "REVIEW": "f97316",
    "DOCUMENTATION": "0d9488",
    "BLOCKED": "b60205",
    "DONE": "0e8a16",
    "AWAIT_SUBTASKS": "d4c5f9",
    "feature": "a2eeef",
    "task": "d4c5f9",
    "bug": "d73a4a",
    "subtask": "e8d5b7",
}


# =============================================================================
# EXCEPTIONS
# =============================================================================


class IssueManagerError(Exception):
    """Base exception for issue manager errors."""
    pass


class IssueTransitionError(IssueManagerError):
    """Error during state transition."""
    pass


class IssueNotFoundError(IssueManagerError):
    """Issue not found."""
    pass


# =============================================================================
# COMMENT TEMPLATES
# =============================================================================


COMMENT_TEMPLATES = {
    "agent_start": (
        "## Agent Started\n\n"
        "**Agent:** {agent_type}\n"
        "**Iteration:** {iteration}/{max_iterations}\n"
        "**Started:** {timestamp}\n"
    ),
    "agent_complete": (
        "## Agent Completed\n\n"
        "**Agent:** {agent_type}\n"
        "**Status:** {status}\n"
        "**Duration:** {duration}\n\n"
        "### Summary\n{summary}\n"
    ),
    "qa_passed": (
        "## QA Passed\n\n"
        "All acceptance criteria have been met.\n\n"
        "### Results\n{details}\n"
    ),
    "qa_failed": (
        "## QA Failed (Iteration {iteration}/{max_iterations})\n\n"
        "### Feedback\n{feedback}\n\n"
        "### Failed Criteria\n{failed_criteria}\n"
    ),
    "blocked": (
        "## Issue Blocked\n\n"
        "This issue requires human intervention.\n\n"
        "**Reason:** {reason}\n\n"
        "**Iterations completed:** {iterations}\n"
    ),
    "done": (
        "## Issue Completed\n\n"
        "{summary}\n\n"
        "**Total iterations:** {iterations}\n"
        "**Completed:** {timestamp}\n"
    ),
    "subtask_created": (
        "### Subtask Created\n\n"
        "Created subtask #{child_number}: {child_title}\n"
    ),
    "planning_complete": (
        "## Planning Complete\n\n"
        "Feature decomposed into {count} subtasks:\n\n"
        "{subtask_list}\n"
    ),
    "review_approved": (
        "## Review Approved\n\n"
        "Code review passed.\n\n"
        "### Notes\n{notes}\n"
    ),
    "review_changes_requested": (
        "## Review: Changes Requested\n\n"
        "### Required Changes\n{changes}\n"
    ),
}


# =============================================================================
# ISSUE MANAGER CLASS
# =============================================================================


class IssueManager:
    """
    High-level issue management for the workflow.

    This class provides workflow-aware operations on GitHub issues,
    translating between internal workflow states and GitHub labels.

    Attributes:
        client: GitHubClient instance
        config: Configuration dict
    """

    def __init__(self, client: GitHubClient, config: Dict[str, Any]):
        """
        Initialize issue manager.

        Args:
            client: Configured GitHubClient
            config: Configuration from github.yaml
        """
        self.client = client
        self.config = config

        # Configurable label names (allows customization)
        self.label_map = config.get("labels", {})
        for label in WorkflowLabel:
            if label.value not in self.label_map:
                self.label_map[label.value] = label.value

        # Max iterations from config
        self.max_iterations = config.get("max_iterations", 5)

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    async def ensure_labels_exist(self) -> None:
        """
        Ensure all workflow labels exist in the repository.

        Creates any missing labels with appropriate colors.
        """
        logger.info("Ensuring workflow labels exist in repository")

        all_labels = {**LABEL_COLORS}

        for label_name, color in all_labels.items():
            try:
                await asyncio.to_thread(
                    self.client.get_or_create_label,
                    label_name,
                    color,
                )
            except GitHubAPIError as e:
                logger.warning(f"Failed to ensure label '{label_name}': {e}")

    # =========================================================================
    # ISSUE QUERIES
    # =========================================================================

    async def get_issue(self, issue_number: int) -> Dict[str, Any]:
        """
        Get issue details.

        Args:
            issue_number: GitHub issue number

        Returns:
            Issue data dict

        Raises:
            IssueNotFoundError: If issue not found
        """
        try:
            return await asyncio.to_thread(
                self.client.get_issue, issue_number
            )
        except GitHubAPIError as e:
            if e.status_code == 404:
                raise IssueNotFoundError(f"Issue #{issue_number} not found") from e
            raise IssueManagerError(f"Failed to get issue: {e}") from e

    async def get_ready_issues(self) -> List[Dict[str, Any]]:
        """
        Get issues ready for processing.

        Returns issues with READY label, sorted by priority and age.
        Excludes issues with BLOCKED label.

        Returns:
            List of issue data dicts
        """
        ready_label = self.label_map.get("READY", "READY")

        issues = await asyncio.to_thread(
            self.client.list_issues,
            labels=[ready_label],
            state="open",
            sort="created",
            direction="asc",
        )

        # Filter out blocked issues
        blocked_label = self.label_map.get("BLOCKED", "BLOCKED")
        issues = [
            issue for issue in issues
            if not self._has_label(issue, blocked_label)
        ]

        return issues

    async def get_in_progress_issues(self) -> List[Dict[str, Any]]:
        """
        Get issues currently being processed.

        Used for recovery after orchestrator restart.

        Returns:
            List of in-progress issue data dicts
        """
        in_progress_label = self.label_map.get("IN_PROGRESS", "IN_PROGRESS")

        return await asyncio.to_thread(
            self.client.list_issues,
            labels=[in_progress_label],
            state="open",
        )

    async def get_blocked_issues(self) -> List[Dict[str, Any]]:
        """
        Get issues that are blocked and need human attention.

        Returns:
            List of blocked issue data dicts
        """
        blocked_label = self.label_map.get("BLOCKED", "BLOCKED")

        return await asyncio.to_thread(
            self.client.list_issues,
            labels=[blocked_label],
            state="open",
        )

    async def get_issues_by_state(self, state: str) -> List[Dict[str, Any]]:
        """
        Get issues in a specific workflow state.

        Args:
            state: Workflow state (READY, IN_PROGRESS, QA, etc.)

        Returns:
            List of issues in that state
        """
        label = self.label_map.get(state, state)

        return await asyncio.to_thread(
            self.client.list_issues,
            labels=[label],
            state="open",
        )

    async def get_issue_state(self, issue_number: int) -> Optional[str]:
        """
        Determine the current workflow state of an issue from its labels.

        Args:
            issue_number: GitHub issue number

        Returns:
            Current workflow state string, or None if no state label found
        """
        issue = await self.get_issue(issue_number)
        labels = {l.get("name", "") for l in issue.get("labels", [])}

        # Check state labels in priority order
        state_priority = [
            WorkflowLabel.BLOCKED,
            WorkflowLabel.DONE,
            WorkflowLabel.QA_FAILED,
            WorkflowLabel.QA,
            WorkflowLabel.REVIEW,
            WorkflowLabel.DOCUMENTATION,
            WorkflowLabel.DEVELOPMENT,
            WorkflowLabel.PLANNING,
            WorkflowLabel.AWAIT_SUBTASKS,
            WorkflowLabel.IN_PROGRESS,
            WorkflowLabel.READY,
        ]

        for state in state_priority:
            mapped_label = self.label_map.get(state.value, state.value)
            if mapped_label in labels:
                return state.value

        return None

    # =========================================================================
    # STATE TRANSITIONS
    # =========================================================================

    async def transition_to_ready(self, issue_number: int) -> None:
        """Mark issue as ready for processing."""
        await self._transition(
            issue_number,
            WorkflowLabel.READY,
            remove_states=True,
        )
        logger.info(f"Issue #{issue_number} transitioned to READY")

    async def transition_to_in_progress(
        self,
        issue_number: int,
        agent_type: str,
    ) -> None:
        """
        Mark issue as being worked on.

        Adds IN_PROGRESS label and agent assignment comment.
        """
        await self._transition(
            issue_number,
            WorkflowLabel.IN_PROGRESS,
            additional_labels=[agent_type.upper()],
            remove_states=True,
            keep_labels=["IN_PROGRESS"],
        )

        # Post start comment
        await self.post_agent_start(issue_number, agent_type)
        logger.info(f"Issue #{issue_number} transitioned to IN_PROGRESS ({agent_type})")

    async def transition_to_planning(self, issue_number: int) -> None:
        """Mark issue as in planning phase."""
        await self._transition(
            issue_number,
            WorkflowLabel.PLANNING,
            additional_labels=["IN_PROGRESS"],
            remove_states=True,
            keep_labels=["IN_PROGRESS"],
        )

    async def transition_to_development(self, issue_number: int) -> None:
        """Mark issue as in development."""
        await self._transition(
            issue_number,
            WorkflowLabel.DEVELOPMENT,
            additional_labels=["IN_PROGRESS"],
            remove_states=True,
            keep_labels=["IN_PROGRESS"],
        )

    async def transition_to_qa(self, issue_number: int) -> None:
        """Mark issue as ready for QA validation."""
        await self._transition(
            issue_number,
            WorkflowLabel.QA,
            remove_states=True,
            keep_labels=["IN_PROGRESS"],
        )
        logger.info(f"Issue #{issue_number} transitioned to QA")

    async def transition_to_qa_failed(
        self,
        issue_number: int,
        feedback: str,
        iteration: int,
    ) -> None:
        """
        Mark issue as failed QA.

        Adds failure comment with feedback and iteration count.
        """
        await self._transition(
            issue_number,
            WorkflowLabel.QA_FAILED,
            remove_states=True,
            keep_labels=["IN_PROGRESS"],
        )

        # Post QA failure comment
        comment = COMMENT_TEMPLATES["qa_failed"].format(
            iteration=iteration,
            max_iterations=self.max_iterations,
            feedback=feedback,
            failed_criteria="See feedback above.",
        )
        await self._add_comment(issue_number, comment)

        logger.info(f"Issue #{issue_number} transitioned to QA_FAILED (iteration {iteration})")

    async def transition_to_review(self, issue_number: int) -> None:
        """Mark issue as ready for review."""
        await self._transition(
            issue_number,
            WorkflowLabel.REVIEW,
            remove_states=True,
            keep_labels=["IN_PROGRESS"],
        )
        logger.info(f"Issue #{issue_number} transitioned to REVIEW")

    async def transition_to_documentation(self, issue_number: int) -> None:
        """Mark issue as in documentation phase."""
        await self._transition(
            issue_number,
            WorkflowLabel.DOCUMENTATION,
            remove_states=True,
            keep_labels=["IN_PROGRESS"],
        )

    async def transition_to_blocked(
        self,
        issue_number: int,
        reason: str,
        iterations: int = 0,
    ) -> None:
        """
        Mark issue as blocked.

        Adds BLOCKED label and posts explanation comment.
        """
        await self._transition(
            issue_number,
            WorkflowLabel.BLOCKED,
            remove_states=True,
        )

        comment = COMMENT_TEMPLATES["blocked"].format(
            reason=reason,
            iterations=iterations,
        )
        await self._add_comment(issue_number, comment)

        logger.warning(f"Issue #{issue_number} transitioned to BLOCKED: {reason}")

    async def transition_to_done(
        self,
        issue_number: int,
        summary: str = "",
        iterations: int = 0,
    ) -> None:
        """
        Mark issue as completed.

        Adds DONE label, posts summary comment, closes issue.
        """
        await self._transition(
            issue_number,
            WorkflowLabel.DONE,
            remove_states=True,
        )

        # Post completion comment
        comment = COMMENT_TEMPLATES["done"].format(
            summary=summary or "Task completed successfully.",
            iterations=iterations,
            timestamp=datetime.utcnow().isoformat(),
        )
        await self._add_comment(issue_number, comment)

        # Close the issue
        await asyncio.to_thread(
            self.client.close_issue, issue_number
        )

        logger.info(f"Issue #{issue_number} transitioned to DONE")

    # =========================================================================
    # AGENT COMMUNICATION
    # =========================================================================

    async def post_agent_start(
        self,
        issue_number: int,
        agent_type: str,
        iteration: int = 1,
    ) -> None:
        """
        Post comment when agent starts working.

        Args:
            issue_number: Issue number
            agent_type: Type of agent starting
            iteration: Current iteration number
        """
        comment = COMMENT_TEMPLATES["agent_start"].format(
            agent_type=agent_type.capitalize(),
            iteration=iteration,
            max_iterations=self.max_iterations,
            timestamp=datetime.utcnow().isoformat(),
        )
        await self._add_comment(issue_number, comment)

    async def post_agent_complete(
        self,
        issue_number: int,
        agent_type: str,
        result: Dict[str, Any],
    ) -> None:
        """
        Post comment when agent completes.

        Args:
            issue_number: Issue number
            agent_type: Type of agent that completed
            result: Agent result data
        """
        status = "Success" if result.get("success", False) else "Failed"
        summary = result.get("summary", "No summary provided.")
        duration = result.get("duration", "Unknown")

        comment = COMMENT_TEMPLATES["agent_complete"].format(
            agent_type=agent_type.capitalize(),
            status=status,
            duration=duration,
            summary=summary,
        )
        await self._add_comment(issue_number, comment)

    async def post_qa_result(
        self,
        issue_number: int,
        passed: bool,
        details: Dict[str, Any],
    ) -> None:
        """
        Post QA results comment.

        Args:
            issue_number: Issue number
            passed: Whether QA passed
            details: QA result details
        """
        if passed:
            details_text = self._format_qa_details(details)
            comment = COMMENT_TEMPLATES["qa_passed"].format(
                details=details_text,
            )
        else:
            feedback = details.get("feedback", "No feedback provided.")
            criteria = details.get("failed_criteria", [])
            criteria_text = "\n".join(f"- {c}" for c in criteria) if criteria else "None specified."

            comment = COMMENT_TEMPLATES["qa_failed"].format(
                iteration=details.get("iteration", "?"),
                max_iterations=self.max_iterations,
                feedback=feedback,
                failed_criteria=criteria_text,
            )

        await self._add_comment(issue_number, comment)

    async def post_review_result(
        self,
        issue_number: int,
        approved: bool,
        notes: str = "",
    ) -> None:
        """
        Post review result comment.

        Args:
            issue_number: Issue number
            approved: Whether review was approved
            notes: Review notes or required changes
        """
        if approved:
            comment = COMMENT_TEMPLATES["review_approved"].format(
                notes=notes or "No additional notes.",
            )
        else:
            comment = COMMENT_TEMPLATES["review_changes_requested"].format(
                changes=notes or "No specific changes listed.",
            )

        await self._add_comment(issue_number, comment)

    async def post_planning_complete(
        self,
        issue_number: int,
        subtasks: List[Dict[str, Any]],
    ) -> None:
        """
        Post planning completion comment with subtask list.

        Args:
            issue_number: Issue number
            subtasks: List of created subtask data
        """
        subtask_list = "\n".join(
            f"- #{s.get('number', '?')}: {s.get('title', 'Untitled')}"
            for s in subtasks
        )

        comment = COMMENT_TEMPLATES["planning_complete"].format(
            count=len(subtasks),
            subtask_list=subtask_list or "No subtasks created.",
        )
        await self._add_comment(issue_number, comment)

    # =========================================================================
    # ISSUE CREATION
    # =========================================================================

    async def create_subtask(
        self,
        parent_number: int,
        title: str,
        body: str,
        labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Create a subtask linked to parent issue.

        Adds subtask label and parent reference in body.

        Args:
            parent_number: Parent issue number
            title: Subtask title
            body: Subtask body/description
            labels: Additional labels

        Returns:
            Created issue data dict
        """
        # Build body with parent reference
        full_body = (
            f"Parent: #{parent_number}\n\n"
            f"---\n\n"
            f"{body}"
        )

        # Build labels
        subtask_labels = ["subtask", "READY"]
        if labels:
            subtask_labels.extend(labels)

        # Create the issue
        child_issue = await asyncio.to_thread(
            self.client.create_issue,
            title=title,
            body=full_body,
            labels=subtask_labels,
        )

        child_number = child_issue["number"]

        # Post comment on parent
        comment = COMMENT_TEMPLATES["subtask_created"].format(
            child_number=child_number,
            child_title=title,
        )
        await self._add_comment(parent_number, comment)

        logger.info(f"Created subtask #{child_number} for parent #{parent_number}")
        return child_issue

    async def create_issue(
        self,
        title: str,
        body: str,
        labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new issue.

        Args:
            title: Issue title
            body: Issue body
            labels: Issue labels

        Returns:
            Created issue data dict
        """
        return await asyncio.to_thread(
            self.client.create_issue,
            title=title,
            body=body,
            labels=labels or [],
        )

    # =========================================================================
    # ISSUE HISTORY
    # =========================================================================

    async def get_comments(
        self,
        issue_number: int,
        since: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get comments on an issue.

        Args:
            issue_number: Issue number
            since: Only get comments after this ISO timestamp

        Returns:
            List of comment data dicts
        """
        return await asyncio.to_thread(
            self.client.list_comments,
            issue_number,
            since=since,
        )

    async def get_agent_comments(
        self,
        issue_number: int,
    ) -> List[Dict[str, Any]]:
        """
        Get only agent-posted comments on an issue.

        Filters comments by looking for agent comment markers.

        Args:
            issue_number: Issue number

        Returns:
            List of agent comment data dicts
        """
        comments = await self.get_comments(issue_number)

        # Filter for agent comments (those starting with ## Agent or ## QA)
        agent_markers = ["## Agent", "## QA", "## Issue", "## Review", "## Planning"]
        return [
            c for c in comments
            if any(c.get("body", "").startswith(m) for m in agent_markers)
        ]

    async def get_qa_feedback(
        self,
        issue_number: int,
    ) -> Optional[str]:
        """
        Get the most recent QA feedback for an issue.

        Args:
            issue_number: Issue number

        Returns:
            QA feedback string or None
        """
        comments = await self.get_comments(issue_number)

        # Find the most recent QA failure comment
        for comment in reversed(comments):
            body = comment.get("body", "")
            if body.startswith("## QA Failed"):
                return body

        return None

    # =========================================================================
    # BULK OPERATIONS
    # =========================================================================

    async def recover_orphaned_issues(self) -> List[int]:
        """
        Find and reset issues that were IN_PROGRESS but have no active agent.

        Used after orchestrator restart to recover from crashes.

        Returns:
            List of recovered issue numbers
        """
        in_progress = await self.get_in_progress_issues()
        recovered = []

        for issue in in_progress:
            issue_number = issue["number"]

            # Check if issue is actually stuck (no recent agent comments)
            comments = await self.get_comments(issue_number)

            if comments:
                last_comment = comments[-1]
                last_time = last_comment.get("created_at", "")
                if last_time:
                    try:
                        last_dt = datetime.fromisoformat(
                            last_time.replace("Z", "+00:00")
                        )
                        age = datetime.now(last_dt.tzinfo) - last_dt
                        # If last comment is older than 2 hours, consider orphaned
                        if age.total_seconds() < 7200:
                            continue
                    except (ValueError, TypeError):
                        pass

            # Reset to READY
            await self.transition_to_ready(issue_number)
            recovered.append(issue_number)
            logger.warning(f"Recovered orphaned issue #{issue_number}")

        return recovered

    async def get_workflow_summary(self) -> Dict[str, int]:
        """
        Get a summary of issues by workflow state.

        Returns:
            Dict mapping state names to issue counts
        """
        summary = {}

        for state in WorkflowLabel:
            issues = await self.get_issues_by_state(state.value)
            summary[state.value] = len(issues)

        return summary

    # =========================================================================
    # PRIVATE HELPERS
    # =========================================================================

    async def _transition(
        self,
        issue_number: int,
        to_state: WorkflowLabel,
        additional_labels: Optional[List[str]] = None,
        remove_states: bool = True,
        keep_labels: Optional[List[str]] = None,
    ) -> None:
        """
        Perform a state transition on an issue.

        Args:
            issue_number: Issue number
            to_state: Target workflow state
            additional_labels: Extra labels to add
            remove_states: Whether to remove other state labels
            keep_labels: State labels to keep even when removing
        """
        keep_set = set(keep_labels or [])

        if remove_states:
            # Remove all state labels except those in keep_set
            issue = await self.get_issue(issue_number)
            current_labels = {l.get("name", "") for l in issue.get("labels", [])}

            for label in current_labels:
                if label in ALL_STATE_LABELS and label not in keep_set:
                    try:
                        await asyncio.to_thread(
                            self.client.remove_label, issue_number, label
                        )
                    except GitHubAPIError:
                        pass  # Label might already be removed

        # Add new state label
        target_label = self.label_map.get(to_state.value, to_state.value)
        labels_to_add = [target_label]

        if additional_labels:
            labels_to_add.extend(additional_labels)

        try:
            await asyncio.to_thread(
                self.client.add_labels, issue_number, labels_to_add
            )
        except GitHubAPIError as e:
            raise IssueTransitionError(
                f"Failed to add labels for transition to {to_state.value}: {e}"
            ) from e

    async def _add_comment(self, issue_number: int, body: str) -> Dict[str, Any]:
        """Add a comment to an issue."""
        try:
            return await asyncio.to_thread(
                self.client.add_comment, issue_number, body
            )
        except GitHubAPIError as e:
            logger.warning(f"Failed to add comment to issue #{issue_number}: {e}")
            return {}

    def _has_label(self, issue: Dict[str, Any], label_name: str) -> bool:
        """Check if an issue has a specific label."""
        return any(
            l.get("name", "") == label_name
            for l in issue.get("labels", [])
        )

    def _format_qa_details(self, details: Dict[str, Any]) -> str:
        """Format QA details into a readable string."""
        lines = []

        criteria = details.get("criteria", [])
        for criterion in criteria:
            name = criterion.get("name", "Unknown")
            passed = criterion.get("passed", False)
            icon = "PASS" if passed else "FAIL"
            lines.append(f"- [{icon}] {name}")

        if not lines:
            return "No detailed criteria available."

        return "\n".join(lines)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_issue_manager(
    client: GitHubClient,
    config: Optional[Dict[str, Any]] = None,
) -> IssueManager:
    """
    Create an issue manager instance.

    Args:
        client: Configured GitHubClient
        config: Optional configuration

    Returns:
        Configured IssueManager instance
    """
    return IssueManager(client, config or {})


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main class
    "IssueManager",
    "create_issue_manager",
    # Enums
    "WorkflowLabel",
    "IssueType",
    # Templates
    "COMMENT_TEMPLATES",
    # Constants
    "ALL_STATE_LABELS",
    "LABEL_COLORS",
    # Exceptions
    "IssueManagerError",
    "IssueTransitionError",
    "IssueNotFoundError",
]