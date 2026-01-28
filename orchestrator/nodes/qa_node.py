# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - QA NODE
# =============================================================================
"""
QA (Quality Assurance) Node Implementation

Validates that the implementation meets acceptance criteria by
launching the QA Agent. Determines pass/fail outcome.

Workflow Position:
    DEVELOPMENT --> QA --(pass)--> REVIEW
                      --(fail)--> QA_FAILED

NO TASK IS COMPLETE WITHOUT PASSING QA.
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
    NODE_CONFIGS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# NODE CONFIGURATION
# =============================================================================

QA_CONFIG = {
    "agent_type": "qa",
    "timeout_seconds": 600,
    "require_all_tests_pass": True,
    "require_no_linter_errors": True,
    "allow_warnings": True,
}


# =============================================================================
# NODE IMPLEMENTATION
# =============================================================================


async def qa_node(
    state: Dict[str, Any],
    ctx: NodeContext,
) -> Dict[str, Any]:
    """
    Validate implementation via QA Agent.

    Launches the QA Agent to run tests, check acceptance criteria,
    and validate code quality. Sets qa_result to PASS or FAIL.

    Args:
        state: Current workflow state
        ctx: Node execution context

    Returns:
        Updated state with QA results
    """
    issue_number = state["issue_number"]
    logger.info(f"QA node: issue #{issue_number}")

    state["issue_state"] = IssueState.QA.value
    state["current_agent"] = AgentType.QA.value

    try:
        # Update GitHub
        await update_labels(
            ctx, issue_number,
            add=["QA"],
            remove=["DEVELOPMENT", "REVIEW"],
        )

        await post_comment(
            ctx, issue_number,
            "## QA Started\n\nQA Agent is validating the implementation."
        )

        # Extract acceptance criteria
        criteria = extract_acceptance_criteria(state.get("issue_body", ""))

        # Prepare agent context
        metadata = state.get("metadata", {})
        agent_context = {
            "title": state.get("issue_title", ""),
            "body": state.get("issue_body", ""),
            "acceptance_criteria": criteria,
            "modified_files": metadata.get("modified_files", []),
            "developer_output": state.get("last_agent_output", {}),
            "branch_name": metadata.get("branch_name", ""),
            "config": {
                "require_all_tests_pass": QA_CONFIG["require_all_tests_pass"],
                "require_no_linter_errors": QA_CONFIG["require_no_linter_errors"],
                "test_timeout_seconds": 300,
            },
        }

        # Launch QA agent
        timeout = QA_CONFIG["timeout_seconds"]
        result = await launch_agent_and_wait(
            ctx, AgentType.QA, issue_number, agent_context, timeout
        )

        state["last_agent_output"] = result.output
        state["current_agent"] = None

        # Determine QA result
        qa_passed = False
        if result.success and result.output:
            qa_passed = result.output.get("passed", False)
            if not qa_passed:
                # Also check qa_result field
                qa_passed = result.output.get("qa_result") == "PASS"

        state["qa_result"] = "PASS" if qa_passed else "FAIL"
        metadata["qa_result"] = state["qa_result"]
        state["metadata"] = metadata

        # Post QA result
        await ctx.issue_manager.post_qa_result(
            issue_number,
            passed=qa_passed,
            details={
                "criteria": result.output.get("acceptance_checklist", [])
                           if result.output else [],
                "feedback": result.output.get("feedback", "")
                           if result.output else "",
                "failed_criteria": result.output.get("failed_criteria", [])
                                  if result.output else [],
                "iteration": state.get("iteration_count", 0),
            }
        )

        add_history_entry(state, "qa", "qa_complete", {
            "qa_result": state["qa_result"],
        })

        logger.info(f"QA result for issue #{issue_number}: {state['qa_result']}")

    except Exception as e:
        logger.error(f"QA failed for issue #{issue_number}: {e}")
        state["error_message"] = f"QA failed: {e}"
        state["qa_result"] = "FAIL"
        state["current_agent"] = None

    return state


# =============================================================================
# ROUTER
# =============================================================================


def qa_router(state: Dict[str, Any], max_iterations: int = 5) -> str:
    """
    Route after QA based on result and iteration count.

    Returns:
        "pass" - QA passed, go to review
        "fail_retriable" - QA failed, can retry
        "fail_blocked" - QA failed, max iterations reached
    """
    if state.get("qa_result") == "PASS":
        return "pass"

    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", max_iterations)

    if iteration < max_iter:
        return "fail_retriable"
    else:
        return "fail_blocked"


# =============================================================================
# HELPERS
# =============================================================================


def format_qa_comment(
    qa_result: str,
    checklist: List[Dict[str, Any]],
    test_results: Dict[str, Any],
    feedback: Optional[str] = None,
) -> str:
    """Format QA results as a GitHub comment."""
    result_icon = "PASS" if qa_result == "PASS" else "FAIL"

    # Format checklist
    checklist_lines = []
    for item in checklist:
        icon = "PASS" if item.get("result") == "PASS" else "FAIL"
        checklist_lines.append(f"- [{icon}] {item.get('criterion', 'Unknown')}")
    checklist_text = "\n".join(checklist_lines) or "No criteria evaluated."

    # Format test results
    test_lines = []
    for test_type, results in test_results.items():
        passed = results.get("passed", 0)
        failed = results.get("failed", 0)
        test_lines.append(f"- **{test_type}**: {passed} passed, {failed} failed")
    test_text = "\n".join(test_lines) or "No tests executed."

    comment = (
        f"## QA Validation: {result_icon}\n\n"
        f"### Acceptance Criteria\n{checklist_text}\n\n"
        f"### Test Results\n{test_text}\n"
    )

    if feedback:
        comment += f"\n### Feedback\n{feedback}\n"

    return comment


__all__ = [
    "qa_node",
    "qa_router",
    "format_qa_comment",
    "QA_CONFIG",
]