# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - BLOCKED NODE
# =============================================================================
"""
Blocked Node Implementation

Terminal blocked state. Marks the issue as requiring human
intervention. Posts blocking reason and details.

Workflow Position:
    Any node --(error/max_iterations)--> BLOCKED --> [END]
"""

from __future__ import annotations

import logging
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


async def blocked_node(
    state: Dict[str, Any],
    ctx: NodeContext,
) -> Dict[str, Any]:
    """
    Mark issue as blocked requiring human intervention.

    Args:
        state: Current workflow state
        ctx: Node execution context

    Returns:
        Final blocked state
    """
    issue_number = state["issue_number"]
    error_msg = state.get("error_message", "Unknown reason")
    iterations = state.get("iteration_count", 0)
    logger.warning(f"Blocked node: issue #{issue_number} - {error_msg}")

    state["issue_state"] = IssueState.BLOCKED.value

    try:
        # Transition via issue manager (handles labels + comment + notification)
        await ctx.issue_manager.transition_to_blocked(
            issue_number,
            reason=error_msg,
            iterations=iterations,
        )

        add_history_entry(state, "blocked", "blocked", {
            "reason": error_msg,
            "iterations": iterations,
        })

        logger.warning(
            f"Issue #{issue_number} blocked after {iterations} iterations: {error_msg}"
        )

    except Exception as e:
        logger.error(f"Blocked node failed for issue #{issue_number}: {e}")

    return state


__all__ = ["blocked_node"]