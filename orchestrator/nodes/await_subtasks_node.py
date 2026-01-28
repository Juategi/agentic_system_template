# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - AWAIT SUBTASKS NODE
# =============================================================================
"""
Await Subtasks Node Implementation

Checks whether all child issues of a feature have completed.
If complete, proceeds to documentation. If not, exits to wait.

Workflow Position:
    PLANNING --> AWAIT_SUBTASKS --(complete)--> DOCUMENTATION
                               --(waiting)--> [END] (poll later)
                               --(error)--> BLOCKED
"""

from __future__ import annotations

import logging
from typing import Dict, Any

from orchestrator.engine.state_manager import IssueState
from orchestrator.nodes._base import (
    NodeContext,
    update_labels,
    post_comment,
    add_history_entry,
)

logger = logging.getLogger(__name__)


# =============================================================================
# NODE IMPLEMENTATION
# =============================================================================


async def await_subtasks_node(
    state: Dict[str, Any],
    ctx: NodeContext,
) -> Dict[str, Any]:
    """
    Check if all subtasks are complete.

    Queries the state of each child issue and determines
    whether the parent feature can proceed to documentation.

    Args:
        state: Current workflow state
        ctx: Node execution context

    Returns:
        Updated state with subtasks_complete flag
    """
    issue_number = state["issue_number"]
    child_issues = state.get("child_issues", [])
    logger.info(f"Await subtasks node: issue #{issue_number} ({len(child_issues)} children)")

    state["issue_state"] = IssueState.AWAIT_SUBTASKS.value

    try:
        # Update GitHub
        await update_labels(
            ctx, issue_number,
            add=["AWAIT_SUBTASKS"],
            remove=["PLANNING"],
        )

        if not child_issues:
            # No subtasks - go directly to documentation
            metadata = state.get("metadata", {})
            metadata["subtasks_complete"] = True
            state["metadata"] = metadata

            add_history_entry(state, "await_subtasks", "no_subtasks", {})
            return state

        # Check each child issue
        completed = []
        pending = []
        blocked = []

        for child_number in child_issues:
            child_state = await ctx.state_manager.get_state(child_number)

            if child_state is None:
                # No state yet - check GitHub labels
                issue_state = await ctx.issue_manager.get_issue_state(child_number)
                if issue_state == IssueState.DONE.value:
                    completed.append(child_number)
                elif issue_state == IssueState.BLOCKED.value:
                    blocked.append(child_number)
                else:
                    pending.append(child_number)
            elif child_state.issue_state == IssueState.DONE.value:
                completed.append(child_number)
            elif child_state.issue_state == IssueState.BLOCKED.value:
                blocked.append(child_number)
            else:
                pending.append(child_number)

        all_complete = len(completed) == len(child_issues)
        has_blocked = len(blocked) > 0

        metadata = state.get("metadata", {})
        metadata["subtasks_complete"] = all_complete
        metadata["subtasks_status"] = {
            "total": len(child_issues),
            "completed": len(completed),
            "pending": len(pending),
            "blocked": len(blocked),
        }
        state["metadata"] = metadata

        if has_blocked and not pending:
            # All non-blocked tasks are done, but some are blocked
            state["error_message"] = (
                f"Subtasks blocked: {blocked}. "
                f"Completed: {len(completed)}/{len(child_issues)}"
            )

        # Post status update
        status_text = (
            f"**Subtask Status:**\n"
            f"- Completed: {len(completed)}/{len(child_issues)}\n"
            f"- Pending: {len(pending)}\n"
            f"- Blocked: {len(blocked)}\n"
        )

        if all_complete:
            await post_comment(
                ctx, issue_number,
                f"## All Subtasks Complete\n\n{status_text}\n"
                f"Proceeding to documentation."
            )
        else:
            logger.info(
                f"Issue #{issue_number}: {len(completed)}/{len(child_issues)} "
                f"subtasks complete, waiting..."
            )

        add_history_entry(state, "await_subtasks", "status_check", {
            "completed": len(completed),
            "pending": len(pending),
            "blocked": len(blocked),
            "all_complete": all_complete,
        })

    except Exception as e:
        logger.error(f"Await subtasks failed for issue #{issue_number}: {e}")
        state["error_message"] = f"Subtask check failed: {e}"
        metadata = state.get("metadata", {})
        metadata["subtasks_complete"] = False
        state["metadata"] = metadata

    return state


# =============================================================================
# ROUTER
# =============================================================================


def await_subtasks_router(state: Dict[str, Any]) -> str:
    """
    Route after checking subtasks.

    Returns:
        "complete" - all done, proceed
        "waiting" - still waiting, exit workflow
        "error" - error occurred
    """
    if state.get("error_message"):
        return "error"

    complete = state.get("metadata", {}).get("subtasks_complete", False)
    return "complete" if complete else "waiting"


__all__ = ["await_subtasks_node", "await_subtasks_router"]