# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - PLANNING NODE
# =============================================================================
"""
Planning Node Implementation

Decomposes feature issues into smaller tasks by launching
the Planner Agent. Creates sub-issues in GitHub.

Workflow Position:
    TRIAGE --(is_feature)--> PLANNING --> AWAIT_SUBTASKS
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

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

PLANNING_CONFIG = {
    "agent_type": "planner",
    "timeout_seconds": 600,
    "max_subtasks": 10,
    "min_task_granularity_hours": 2,
    "max_task_granularity_hours": 8,
    "required_context_files": ["PROJECT.md", "ARCHITECTURE.md"],
}


# =============================================================================
# NODE IMPLEMENTATION
# =============================================================================


async def planning_node(
    state: Dict[str, Any],
    ctx: NodeContext,
) -> Dict[str, Any]:
    """
    Decompose a feature issue into subtasks.

    Launches the Planner Agent to analyze the feature and
    create sub-issues in GitHub.

    Args:
        state: Current workflow state
        ctx: Node execution context

    Returns:
        Updated state with child_issues populated
    """
    issue_number = state["issue_number"]
    logger.info(f"Planning node: issue #{issue_number}")

    state["issue_state"] = IssueState.PLANNING.value
    state["current_agent"] = AgentType.PLANNER.value

    try:
        # Update GitHub
        await update_labels(
            ctx, issue_number,
            add=["PLANNING", "IN_PROGRESS"],
            remove=["TRIAGE", "READY"],
        )

        await post_comment(
            ctx, issue_number,
            "## Planning Started\n\nPlanner Agent is analyzing this feature for decomposition."
        )

        # Load project context
        project_context = load_project_context(ctx.memory_path)

        # Prepare agent input
        agent_context = {
            "title": state.get("issue_title", ""),
            "body": state.get("issue_body", ""),
            "labels": state.get("issue_labels", []),
            "project_context": project_context,
            "config": {
                "max_subtasks": PLANNING_CONFIG["max_subtasks"],
                "min_task_hours": PLANNING_CONFIG["min_task_granularity_hours"],
                "max_task_hours": PLANNING_CONFIG["max_task_granularity_hours"],
            },
        }

        # Launch planner agent
        timeout = PLANNING_CONFIG["timeout_seconds"]
        result = await launch_agent_and_wait(
            ctx, AgentType.PLANNER, issue_number, agent_context, timeout
        )

        state["last_agent_output"] = result.output
        state["current_agent"] = None

        # Process subtasks
        child_issues = []
        if result.success and result.output:
            subtasks = result.output.get("subtasks", [])

            for subtask in subtasks:
                child = await ctx.issue_manager.create_subtask(
                    parent_number=issue_number,
                    title=subtask.get("title", "Untitled subtask"),
                    body=subtask.get("body", ""),
                    labels=subtask.get("labels"),
                )
                child_issues.append(child["number"])

            state["child_issues"] = child_issues

            # Create feature memory file
            _create_feature_memory(
                ctx.memory_path, issue_number, state, subtasks
            )

            # Post planning complete comment
            await ctx.issue_manager.post_planning_complete(
                issue_number,
                [{"number": n, "title": s.get("title", "")}
                 for n, s in zip(child_issues, subtasks)]
            )

        if not child_issues:
            state["error_message"] = "Planning produced no subtasks"
            logger.warning(f"No subtasks created for issue #{issue_number}")

        add_history_entry(state, "planning", "planning_complete", {
            "child_issues": child_issues,
            "subtask_count": len(child_issues),
        })

    except Exception as e:
        logger.error(f"Planning failed for issue #{issue_number}: {e}")
        state["error_message"] = f"Planning failed: {e}"
        state["current_agent"] = None

    return state


# =============================================================================
# HELPERS
# =============================================================================


def _create_feature_memory(
    memory_path: str,
    issue_number: int,
    state: Dict[str, Any],
    subtasks: List[Dict[str, Any]],
) -> Optional[str]:
    """Create the feature memory markdown file."""
    try:
        features_dir = Path(memory_path) / "features"
        features_dir.mkdir(parents=True, exist_ok=True)

        filepath = features_dir / f"feature-{issue_number}.md"

        subtask_rows = "\n".join(
            f"| #{s.get('number', '?')} | {s.get('title', '')} | READY |"
            for s in subtasks
        )

        content = (
            f"# Feature: {state.get('issue_title', 'Untitled')}\n\n"
            f"## Metadata\n"
            f"- Issue: #{issue_number}\n"
            f"- Status: PLANNING_COMPLETE\n"
            f"- Created: {datetime.utcnow().isoformat()}\n"
            f"- Subtasks: {len(subtasks)}\n\n"
            f"## Description\n"
            f"{state.get('issue_body', 'No description')}\n\n"
            f"## Subtasks\n"
            f"| Issue | Title | Status |\n"
            f"|-------|-------|--------|\n"
            f"{subtask_rows}\n\n"
            f"## History\n"
            f"| Date | Event | Details |\n"
            f"|------|-------|---------|\n"
            f"| {datetime.utcnow().isoformat()} | Planning Complete | {len(subtasks)} subtasks created |\n"
        )

        filepath.write_text(content, encoding="utf-8")
        logger.info(f"Created feature memory: {filepath}")
        return str(filepath)

    except Exception as e:
        logger.warning(f"Failed to create feature memory: {e}")
        return None


def validate_planner_output(output: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate Planner Agent output structure."""
    if not output:
        return False, "Empty output"

    if output.get("status") != "success":
        return False, f"Status: {output.get('status', 'unknown')}"

    subtasks = output.get("subtasks", [])
    if not subtasks:
        return False, "No subtasks in output"

    for i, task in enumerate(subtasks):
        if not task.get("title"):
            return False, f"Subtask {i} missing title"

    return True, ""


__all__ = [
    "planning_node",
    "validate_planner_output",
    "PLANNING_CONFIG",
]