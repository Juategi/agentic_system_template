# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - REVIEW NODE
# =============================================================================
"""
Review Node Implementation

Launches the Reviewer Agent for code quality review after QA passes.

Workflow Position:
    QA --(pass)--> REVIEW --(approved)--> DOCUMENTATION
                         --(changes_requested)--> QA
                         --(error)--> BLOCKED
"""

from __future__ import annotations

import logging
from typing import Dict, Any

from orchestrator.engine.state_manager import IssueState
from orchestrator.scheduler.agent_launcher import AgentType
from orchestrator.nodes._base import (
    NodeContext,
    launch_agent_and_wait,
    update_labels,
    post_comment,
    add_history_entry,
    NODE_CONFIGS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# NODE CONFIGURATION
# =============================================================================

REVIEW_CONFIG = {
    "agent_type": "reviewer",
    "timeout_seconds": 600,
}


# =============================================================================
# NODE IMPLEMENTATION
# =============================================================================


async def review_node(
    state: Dict[str, Any],
    ctx: NodeContext,
) -> Dict[str, Any]:
    """
    Code review via Reviewer Agent.

    Checks code quality, style, security, and best practices.

    Args:
        state: Current workflow state
        ctx: Node execution context

    Returns:
        Updated state with review result
    """
    issue_number = state["issue_number"]
    logger.info(f"Review node: issue #{issue_number}")

    state["issue_state"] = IssueState.REVIEW.value
    state["current_agent"] = AgentType.REVIEWER.value

    try:
        # Update GitHub
        await update_labels(
            ctx, issue_number,
            add=["REVIEW"],
            remove=["QA"],
        )

        await post_comment(
            ctx, issue_number,
            "## Review Started\n\nReviewer Agent is checking code quality."
        )

        # Prepare context
        metadata = state.get("metadata", {})
        agent_context = {
            "title": state.get("issue_title", ""),
            "body": state.get("issue_body", ""),
            "modified_files": metadata.get("modified_files", []),
            "branch_name": metadata.get("branch_name", ""),
            "qa_output": state.get("last_agent_output", {}),
        }

        # Launch reviewer agent
        timeout = REVIEW_CONFIG["timeout_seconds"]
        result = await launch_agent_and_wait(
            ctx, AgentType.REVIEWER, issue_number, agent_context, timeout
        )

        state["last_agent_output"] = result.output
        state["current_agent"] = None

        # Determine review decision
        if result.success and result.output:
            decision = result.output.get("decision", "CHANGES_REQUESTED")
            notes = result.output.get("notes", "")
        else:
            decision = "CHANGES_REQUESTED"
            notes = "Review agent failed to complete."

        state["review_result"] = decision
        metadata["review_result"] = decision
        state["metadata"] = metadata

        # Post review result
        await ctx.issue_manager.post_review_result(
            issue_number,
            approved=(decision == "APPROVED"),
            notes=notes,
        )

        add_history_entry(state, "review", "review_complete", {
            "decision": decision,
        })

        logger.info(f"Review result for issue #{issue_number}: {decision}")

    except Exception as e:
        logger.error(f"Review failed for issue #{issue_number}: {e}")
        state["error_message"] = f"Review failed: {e}"
        state["review_result"] = "error"
        state["current_agent"] = None

    return state


# =============================================================================
# ROUTER
# =============================================================================


def review_router(state: Dict[str, Any]) -> str:
    """
    Route after review.

    Returns:
        "approved", "changes_requested", or "error"
    """
    result = state.get("review_result", "")

    if result == "APPROVED":
        return "approved"
    elif result == "CHANGES_REQUESTED":
        return "changes_requested"
    else:
        return "error"


__all__ = [
    "review_node",
    "review_router",
    "REVIEW_CONFIG",
]