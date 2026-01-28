# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - DEVELOPMENT NODE
# =============================================================================
"""
Development Node Implementation

Launches the Developer Agent to implement code changes for a task.
Handles both first attempts and retries with QA feedback.

Workflow Position:
    TRIAGE --(is_task/bug)--> DEVELOPMENT --> QA
    QA_FAILED --------------> DEVELOPMENT --> QA
"""

from __future__ import annotations

import logging
from typing import Dict, List, Any, Optional, Tuple

from orchestrator.engine.state_manager import IssueState
from orchestrator.scheduler.agent_launcher import AgentType
from orchestrator.nodes._base import (
    NodeContext,
    launch_agent_and_wait,
    update_labels,
    post_comment,
    add_history_entry,
    extract_acceptance_criteria,
    load_project_context,
    NODE_CONFIGS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# NODE CONFIGURATION
# =============================================================================

DEVELOPMENT_CONFIG = {
    "agent_type": "developer",
    "timeout_seconds": 1800,
    "branch_prefix": "agent/",
    "run_linter": True,
}


# =============================================================================
# NODE IMPLEMENTATION
# =============================================================================


async def development_node(
    state: Dict[str, Any],
    ctx: NodeContext,
) -> Dict[str, Any]:
    """
    Execute development for a task issue.

    Launches the Developer Agent to write code. On retries,
    includes QA feedback from the previous iteration.

    Args:
        state: Current workflow state
        ctx: Node execution context

    Returns:
        Updated state with agent output and modified files
    """
    issue_number = state["issue_number"]
    iteration = state.get("iteration_count", 0)
    logger.info(f"Development node: issue #{issue_number} (iteration {iteration})")

    state["issue_state"] = IssueState.DEVELOPMENT.value
    state["current_agent"] = AgentType.DEVELOPER.value

    try:
        # Update GitHub
        await update_labels(
            ctx, issue_number,
            add=["DEVELOPMENT", "IN_PROGRESS"],
            remove=["TRIAGE", "READY", "QA_FAILED"],
        )

        # Post start comment
        await ctx.issue_manager.post_agent_start(
            issue_number, "developer", iteration=iteration + 1
        )

        # Load project context
        project_context = load_project_context(ctx.memory_path)

        # Extract acceptance criteria
        criteria = extract_acceptance_criteria(state.get("issue_body", ""))

        # Prepare agent context
        agent_context = {
            "title": state.get("issue_title", ""),
            "body": state.get("issue_body", ""),
            "labels": state.get("issue_labels", []),
            "acceptance_criteria": criteria,
            "iteration": iteration,
            "project_context": project_context,
            "config": {
                "create_branch": True,
                "branch_prefix": DEVELOPMENT_CONFIG["branch_prefix"],
                "run_linter": DEVELOPMENT_CONFIG["run_linter"],
            },
        }

        # Add QA feedback if this is a retry
        if iteration > 0 and state.get("last_agent_output"):
            qa_output = state["last_agent_output"]
            agent_context["qa_feedback"] = {
                "result": qa_output.get("qa_result", "FAIL"),
                "feedback": qa_output.get("feedback", ""),
                "issues_found": qa_output.get("issues_found", []),
                "suggested_fixes": qa_output.get("suggested_fixes", []),
            }

        # Also check for QA feedback in GitHub comments
        qa_feedback = await ctx.issue_manager.get_qa_feedback(issue_number)
        if qa_feedback:
            agent_context["qa_feedback_comment"] = qa_feedback

        # Launch developer agent
        timeout = DEVELOPMENT_CONFIG["timeout_seconds"]
        result = await launch_agent_and_wait(
            ctx, AgentType.DEVELOPER, issue_number, agent_context, timeout
        )

        state["last_agent_output"] = result.output
        state["current_agent"] = None

        # Extract results
        if result.success and result.output:
            metadata = state.get("metadata", {})
            metadata["modified_files"] = result.output.get("modified_files", [])
            metadata["branch_name"] = result.output.get("branch_name", "")
            metadata["commit_sha"] = result.output.get("commit_sha", "")
            state["metadata"] = metadata

        # Post completion comment
        await ctx.issue_manager.post_agent_complete(
            issue_number, "developer",
            {
                "success": result.success,
                "summary": result.output.get("implementation_notes", "")
                           if result.output else "No output",
                "duration": result.output.get("duration", "unknown")
                           if result.output else "unknown",
            }
        )

        add_history_entry(state, "development", "development_complete", {
            "success": result.success,
            "iteration": iteration,
            "files_modified": len(result.output.get("modified_files", []))
                             if result.output else 0,
        })

    except Exception as e:
        logger.error(f"Development failed for issue #{issue_number}: {e}")
        state["error_message"] = f"Development failed: {e}"
        state["current_agent"] = None

    return state


# =============================================================================
# HELPERS
# =============================================================================


def validate_developer_output(
    output: Dict[str, Any],
) -> Tuple[bool, str]:
    """Validate Developer Agent output structure."""
    if not output:
        return False, "Empty output"

    if output.get("status") != "success":
        return False, f"Status: {output.get('status', 'unknown')}"

    modified = output.get("modified_files", [])
    if not modified:
        return False, "No modified files reported"

    return True, ""


__all__ = [
    "development_node",
    "validate_developer_output",
    "DEVELOPMENT_CONFIG",
]