# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - QA FAILED NODE
# =============================================================================
"""
QA Failed Node Implementation

Handles QA failure by incrementing the iteration counter
and preparing feedback for the next development cycle.

Workflow Position:
    QA --(fail)--> QA_FAILED --> DEVELOPMENT (retry)
                             --> BLOCKED (max iterations)
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


async def qa_failed_node(
    state: Dict[str, Any],
    ctx: NodeContext,
) -> Dict[str, Any]:
    """
    Handle QA failure.

    Increments iteration counter and prepares context for retry.
    This is the ONLY node that increments iteration_count.

    Args:
        state: Current workflow state
        ctx: Node execution context

    Returns:
        Updated state with incremented iteration
    """
    issue_number = state["issue_number"]
    logger.info(f"QA failed node: issue #{issue_number}")

    state["issue_state"] = IssueState.QA_FAILED.value

    # Increment iteration counter
    iteration = state.get("iteration_count", 0) + 1
    state["iteration_count"] = iteration
    max_iter = state.get("max_iterations", ctx.max_iterations)

    # Update GitHub
    await update_labels(
        ctx, issue_number,
        add=["QA_FAILED"],
        remove=["QA"],
    )

    # Extract feedback from QA output
    qa_output = state.get("last_agent_output", {})
    feedback = qa_output.get("feedback", "No specific feedback provided.")
    issues_found = qa_output.get("issues_found", [])

    # Format issues list
    issues_text = ""
    if issues_found:
        issues_text = "\n### Issues Found\n"
        for issue in issues_found:
            if isinstance(issue, dict):
                issues_text += (
                    f"- **{issue.get('issue', 'Unknown')}**"
                    f" ({issue.get('location', 'unknown location')})\n"
                    f"  Suggestion: {issue.get('suggestion', 'N/A')}\n"
                )
            else:
                issues_text += f"- {issue}\n"

    # Post failure comment
    comment = (
        f"## QA Failed (Iteration {iteration}/{max_iter})\n\n"
        f"### Feedback\n{feedback}\n"
        f"{issues_text}\n"
    )

    if iteration >= max_iter:
        comment += (
            f"\n**Max iterations reached.** "
            f"This issue will be marked as BLOCKED.\n"
        )

    await post_comment(ctx, issue_number, comment)

    add_history_entry(state, "qa_failed", "iteration_incremented", {
        "iteration": iteration,
        "max_iterations": max_iter,
        "will_retry": iteration < max_iter,
    })

    logger.info(
        f"Issue #{issue_number} QA failed: iteration {iteration}/{max_iter}"
    )

    return state


__all__ = ["qa_failed_node"]