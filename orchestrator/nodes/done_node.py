# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - DONE NODE
# =============================================================================
"""
Done Node Implementation

Terminal success state. Closes the GitHub issue,
posts a completion summary, and records final state.

Workflow Position:
    DOCUMENTATION --> DONE --> [END]
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

from orchestrator.engine.state_manager import IssueState
from orchestrator.nodes._base import (
    NodeContext,
    update_labels,
    add_history_entry,
)

logger = logging.getLogger(__name__)


# =============================================================================
# NODE IMPLEMENTATION
# =============================================================================


async def done_node(
    state: Dict[str, Any],
    ctx: NodeContext,
) -> Dict[str, Any]:
    """
    Mark issue as completed.

    Adds DONE label, posts summary, and closes the issue.

    Args:
        state: Current workflow state
        ctx: Node execution context

    Returns:
        Final state
    """
    issue_number = state["issue_number"]
    logger.info(f"Done node: issue #{issue_number}")

    state["issue_state"] = IssueState.DONE.value

    try:
        # Update GitHub labels
        await update_labels(
            ctx, issue_number,
            add=["DONE"],
            remove=["DOCUMENTATION", "IN_PROGRESS"],
        )

        # Build completion summary
        iterations = state.get("iteration_count", 0)
        issue_type = state.get("issue_type", "unknown")
        metadata = state.get("metadata", {})
        modified_files = metadata.get("modified_files", [])
        child_issues = state.get("child_issues", [])

        summary_parts = [f"**Type:** {issue_type}"]
        summary_parts.append(f"**Iterations:** {iterations}")

        if modified_files:
            file_list = ", ".join(
                f.get("path", f) if isinstance(f, dict) else str(f)
                for f in modified_files[:10]
            )
            summary_parts.append(f"**Files modified:** {file_list}")

        if child_issues:
            summary_parts.append(f"**Subtasks completed:** {len(child_issues)}")

        summary = "\n".join(f"- {p}" for p in summary_parts)

        # Post completion and close issue
        await ctx.issue_manager.transition_to_done(
            issue_number,
            summary=summary,
            iterations=iterations,
        )

        add_history_entry(state, "done", "completed", {
            "iterations": iterations,
        })

        # Notify parent issue if this is a subtask
        parent = state.get("parent_issue")
        if parent:
            from orchestrator.nodes._base import post_comment
            await post_comment(
                ctx, parent,
                f"### Subtask Completed\n\nSubtask #{issue_number} has been completed."
            )

        logger.info(f"Issue #{issue_number} completed successfully")

    except Exception as e:
        logger.error(f"Done node failed for issue #{issue_number}: {e}")

    return state


__all__ = ["done_node"]