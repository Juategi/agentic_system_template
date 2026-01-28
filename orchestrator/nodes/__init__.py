# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - LANGGRAPH NODES PACKAGE
# =============================================================================
"""
LangGraph Nodes Package

This package contains all node implementations for the LangGraph workflow.
Each node represents a step in the issue processing pipeline.

Node Contract:
    Each node function must:
    1. Accept (state: Dict, ctx: NodeContext) as arguments
    2. Return updated state Dict
    3. Handle errors internally (don't raise)
    4. Log all significant actions
    5. Update issue state in GitHub

Usage:
    from orchestrator.nodes import (
        NodeContext,
        triage_node,
        development_node,
        qa_node,
    )

    ctx = NodeContext(
        github_client=client,
        issue_manager=manager,
        langchain_engine=engine,
        state_manager=state_mgr,
        agent_launcher=launcher,
        config=config,
    )

    state = await triage_node(state, ctx)
"""

# Base module
from orchestrator.nodes._base import (
    NodeContext,
    NODE_CONFIGS,
    launch_agent_and_wait,
    update_labels,
    post_comment,
    add_history_entry,
    extract_acceptance_criteria,
    load_memory_file,
    load_project_context,
)

# Node implementations
from orchestrator.nodes.triage_node import triage_node, triage_router
from orchestrator.nodes.planning_node import planning_node, validate_planner_output
from orchestrator.nodes.await_subtasks_node import await_subtasks_node, await_subtasks_router
from orchestrator.nodes.development_node import development_node, validate_developer_output
from orchestrator.nodes.qa_node import qa_node, qa_router, format_qa_comment
from orchestrator.nodes.qa_failed_node import qa_failed_node
from orchestrator.nodes.review_node import review_node, review_router
from orchestrator.nodes.documentation_node import documentation_node
from orchestrator.nodes.done_node import done_node
from orchestrator.nodes.blocked_node import blocked_node

# Node configurations
from orchestrator.nodes.triage_node import TRIAGE_CONFIG
from orchestrator.nodes.planning_node import PLANNING_CONFIG
from orchestrator.nodes.development_node import DEVELOPMENT_CONFIG
from orchestrator.nodes.qa_node import QA_CONFIG
from orchestrator.nodes.review_node import REVIEW_CONFIG
from orchestrator.nodes.documentation_node import DOCUMENTATION_CONFIG


__all__ = [
    # Base
    "NodeContext",
    "NODE_CONFIGS",
    "launch_agent_and_wait",
    "update_labels",
    "post_comment",
    "add_history_entry",
    "extract_acceptance_criteria",
    "load_memory_file",
    "load_project_context",
    # Nodes
    "triage_node",
    "planning_node",
    "await_subtasks_node",
    "development_node",
    "qa_node",
    "qa_failed_node",
    "review_node",
    "documentation_node",
    "done_node",
    "blocked_node",
    # Routers
    "triage_router",
    "await_subtasks_router",
    "qa_router",
    "review_router",
    # Validators
    "validate_planner_output",
    "validate_developer_output",
    # Helpers
    "format_qa_comment",
    # Configs
    "TRIAGE_CONFIG",
    "PLANNING_CONFIG",
    "DEVELOPMENT_CONFIG",
    "QA_CONFIG",
    "REVIEW_CONFIG",
    "DOCUMENTATION_CONFIG",
]