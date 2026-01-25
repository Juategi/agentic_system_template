# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - ORCHESTRATOR ENGINE PACKAGE
# =============================================================================
"""
Orchestrator Engine Package

This package contains the core engine components for the orchestrator:

1. langchain_setup: LangChain configuration and LLM initialization
2. langgraph_workflow: LangGraph state machine definition
3. state_manager: State persistence and recovery

The engine is the heart of the orchestrator, managing:
- LLM interactions for decision making
- State machine transitions
- Persistent state across restarts

Components:
    LangChainEngine: Manages LLM clients and tools
    WorkflowEngine: Manages the LangGraph state machine
    StateManager: Handles state persistence

Usage:
    from orchestrator.engine import (
        LangChainEngine,
        WorkflowEngine,
        StateManager,
    )

    # Initialize components
    llm_engine = LangChainEngine(config)
    state_manager = StateManager(config)
    workflow = WorkflowEngine(llm_engine, state_manager)

    # Run workflow for an issue
    result = await workflow.run(issue_number=123)
"""

__all__ = [
    "LangChainEngine",
    "WorkflowEngine",
    "StateManager",
    "WorkflowState",
]
