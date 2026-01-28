# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - TRIAGE NODE
# =============================================================================
"""
Triage Node Implementation

Entry point for new issues. Analyzes issue content to determine
type (feature, task, bug) and validity using the LLM.

Workflow Position:
    [New Issue] --> TRIAGE --> PLANNING (feature)
                           --> DEVELOPMENT (task/bug)
                           --> BLOCKED (invalid)
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
# NODE CONFIGURATION
# =============================================================================

TRIAGE_CONFIG = {
    "timeout_seconds": 120,
    "valid_types": {"feature", "task", "bug"},
}


# =============================================================================
# NODE IMPLEMENTATION
# =============================================================================


async def triage_node(
    state: Dict[str, Any],
    ctx: NodeContext,
) -> Dict[str, Any]:
    """
    Triage an issue to determine its type and validity.

    Uses the LLM to analyze the issue title, body, and labels
    to classify it as feature, task, bug, or invalid.

    Args:
        state: Current workflow state
        ctx: Node execution context

    Returns:
        Updated state with issue_type set
    """
    issue_number = state["issue_number"]
    logger.info(f"Triage node: issue #{issue_number}")

    state["issue_state"] = IssueState.TRIAGE.value

    try:
        # Update GitHub labels
        await update_labels(ctx, issue_number, add=["TRIAGE"])

        # Use LLM to classify the issue
        triage_result = await ctx.langchain_engine.triage_issue(
            issue_number=issue_number,
            title=state.get("issue_title", ""),
            body=state.get("issue_body", ""),
            labels=state.get("issue_labels", []),
        )

        # Extract and validate type
        issue_type = triage_result.get("issue_type", "task")
        if issue_type not in TRIAGE_CONFIG["valid_types"]:
            issue_type = "invalid"

        state["issue_type"] = issue_type

        # Store triage metadata
        metadata = state.get("metadata", {})
        metadata["triage_result"] = triage_result
        state["metadata"] = metadata

        # Update labels with type
        await update_labels(
            ctx, issue_number,
            add=[issue_type.upper()],
            remove=["TRIAGE"],
        )

        # Post triage comment
        await post_comment(
            ctx, issue_number,
            f"## Triage Complete\n\n"
            f"**Type:** {issue_type}\n"
            f"**Reasoning:** {triage_result.get('reasoning', 'N/A')}\n"
        )

        add_history_entry(state, "triage", "triage_complete", {
            "issue_type": issue_type,
        })

        logger.info(f"Issue #{issue_number} triaged as: {issue_type}")

    except Exception as e:
        logger.error(f"Triage failed for issue #{issue_number}: {e}")
        state["issue_type"] = "invalid"
        state["error_message"] = f"Triage failed: {e}"

    return state


# =============================================================================
# ROUTER
# =============================================================================


def triage_router(state: Dict[str, Any]) -> str:
    """
    Route after triage based on issue type.

    Returns:
        "feature", "task", "bug", or "invalid"
    """
    return state.get("issue_type", "invalid")


__all__ = ["triage_node", "triage_router", "TRIAGE_CONFIG"]