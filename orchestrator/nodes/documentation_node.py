# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - DOCUMENTATION NODE
# =============================================================================
"""
Documentation Node Implementation

Updates project memory files after successful implementation.
Launches the Doc Agent to update ARCHITECTURE.md, feature memory, etc.

Workflow Position:
    REVIEW --(approved)--> DOCUMENTATION --> DONE
    AWAIT_SUBTASKS --(complete)--> DOCUMENTATION --> DONE
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
    load_project_context,
    NODE_CONFIGS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# NODE CONFIGURATION
# =============================================================================

DOCUMENTATION_CONFIG = {
    "agent_type": "doc",
    "timeout_seconds": 300,
}


# =============================================================================
# NODE IMPLEMENTATION
# =============================================================================


async def documentation_node(
    state: Dict[str, Any],
    ctx: NodeContext,
) -> Dict[str, Any]:
    """
    Update project memory and documentation.

    Launches the Doc Agent to update memory files
    based on what was implemented.

    Args:
        state: Current workflow state
        ctx: Node execution context

    Returns:
        Updated state
    """
    issue_number = state["issue_number"]
    logger.info(f"Documentation node: issue #{issue_number}")

    state["issue_state"] = IssueState.DOCUMENTATION.value
    state["current_agent"] = AgentType.DOC.value

    try:
        # Update GitHub
        await update_labels(
            ctx, issue_number,
            add=["DOCUMENTATION"],
            remove=["REVIEW", "AWAIT_SUBTASKS"],
        )

        await post_comment(
            ctx, issue_number,
            "## Documentation Started\n\nDoc Agent is updating project memory."
        )

        # Load current project context
        project_context = load_project_context(ctx.memory_path)

        # Prepare context
        metadata = state.get("metadata", {})
        agent_context = {
            "title": state.get("issue_title", ""),
            "body": state.get("issue_body", ""),
            "issue_type": state.get("issue_type", ""),
            "modified_files": metadata.get("modified_files", []),
            "history": state.get("history", []),
            "current_memory": project_context,
            "memory_path": ctx.memory_path,
        }

        # Launch doc agent
        timeout = DOCUMENTATION_CONFIG["timeout_seconds"]
        result = await launch_agent_and_wait(
            ctx, AgentType.DOC, issue_number, agent_context, timeout
        )

        state["last_agent_output"] = result.output
        state["current_agent"] = None

        # Track what was updated
        if result.success and result.output:
            metadata["docs_updated"] = result.output.get("updated_files", [])
            state["metadata"] = metadata

        add_history_entry(state, "documentation", "documentation_complete", {
            "success": result.success,
            "files_updated": len(result.output.get("updated_files", []))
                            if result.output else 0,
        })

    except Exception as e:
        logger.error(f"Documentation failed for issue #{issue_number}: {e}")
        state["error_message"] = f"Documentation failed: {e}"
        state["current_agent"] = None

    return state


__all__ = ["documentation_node", "DOCUMENTATION_CONFIG"]