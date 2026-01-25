# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - LANGGRAPH NODES PACKAGE
# =============================================================================
"""
LangGraph Nodes Package

This package contains all node implementations for the LangGraph workflow.
Each node represents a step in the issue processing pipeline.

Nodes:
    - triage_node: Entry point, determines issue type
    - planning_node: Decomposes features into tasks
    - development_node: Implements code changes
    - qa_node: Validates implementation
    - qa_failed_node: Handles QA failures
    - review_node: Reviews code quality
    - documentation_node: Updates project memory
    - done_node: Marks issue as complete
    - blocked_node: Marks issue as blocked

Node Contract:
    Each node function must:
    1. Accept WorkflowState as input
    2. Return updated WorkflowState
    3. Handle errors internally (don't raise)
    4. Log all significant actions
    5. Update issue state in GitHub

Usage:
    from orchestrator.nodes import (
        triage_node,
        development_node,
        qa_node,
        # ... etc
    )

    # Add to LangGraph
    graph.add_node("triage", triage_node)
    graph.add_node("development", development_node)
"""

__all__ = [
    "triage_node",
    "planning_node",
    "development_node",
    "qa_node",
    "qa_failed_node",
    "review_node",
    "documentation_node",
    "done_node",
    "blocked_node",
]
