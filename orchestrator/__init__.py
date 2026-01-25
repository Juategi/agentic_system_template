# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - ORCHESTRATOR PACKAGE
# =============================================================================
"""
Orchestrator Package

This package contains the central orchestration system for the AI Agent
Development System. The orchestrator is responsible for:

1. Monitoring GitHub Issues for new tasks
2. Managing the state machine (LangGraph) for issue processing
3. Launching agent containers for each workflow step
4. Persisting state for recovery and audit
5. Coordinating the development loop (Dev → QA → Review → Done)

Package Structure:
    - main.py: Entry point and main loop
    - engine/: Core orchestration engine
        - langchain_setup.py: LangChain configuration
        - langgraph_workflow.py: LangGraph state machine
        - state_manager.py: State persistence
    - nodes/: LangGraph node implementations
    - github/: GitHub API integration
    - scheduler/: Agent scheduling and execution

Usage:
    The orchestrator is designed to run continuously (24/7) and can be
    started via Docker or directly:

    ```python
    from orchestrator import Orchestrator

    orchestrator = Orchestrator()
    orchestrator.run()
    ```

Environment Variables Required:
    - GITHUB_TOKEN: GitHub API token
    - GITHUB_REPO: Target repository (owner/repo)
    - LLM_PROVIDER: LLM provider (anthropic, openai, etc.)
    - ANTHROPIC_API_KEY or OPENAI_API_KEY: LLM API key

For detailed configuration, see config/orchestrator.yaml
"""

__version__ = "1.0.0"
__author__ = "AI Agent Development System"

# Public API exports
# These will be implemented in the respective modules
__all__ = [
    "Orchestrator",
    "OrchestratorConfig",
    "WorkflowState",
    "AgentLauncher",
]
