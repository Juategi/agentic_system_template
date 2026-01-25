# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - AGENTS PACKAGE
# =============================================================================
"""
Agents Package

This package contains the agent implementations that perform actual work
in the AI development system. Each agent type is specialized for a
specific task in the development workflow.

Agent Types:
    - Planner: Decomposes features into tasks
    - Developer: Implements code changes
    - QA: Validates implementations
    - Reviewer: Reviews code quality
    - Doc: Updates documentation and memory

Key Design Principles:
    1. SINGLE IMAGE: All agents share one Docker image
    2. ENVIRONMENT-DRIVEN: Behavior determined by env vars
    3. EPHEMERAL: Agents run, complete, and terminate
    4. STATELESS: All state via mounted volumes
    5. AUDITABLE: All actions are logged

Package Structure:
    agents/
    ├── __init__.py          # This file
    ├── base/                 # Shared infrastructure
    │   ├── agent_interface.py   # Common interface
    │   ├── context_loader.py    # Context loading
    │   └── output_handler.py    # Result handling
    ├── planner/              # Planner agent
    ├── developer/            # Developer agent
    ├── qa/                   # QA agent
    ├── reviewer/             # Reviewer agent
    └── doc/                  # Documentation agent

Agent Lifecycle:
    1. Container starts with environment variables
    2. Agent loads context from mounted volumes
    3. Agent performs its specialized task
    4. Agent writes output to output volume
    5. Container exits with status code

Usage (inside container):
    The entrypoint script reads AGENT_TYPE and runs:

    ```python
    from agents import get_agent

    agent = get_agent(os.environ["AGENT_TYPE"])
    result = agent.run()
    sys.exit(0 if result.success else 1)
    ```
"""

__version__ = "1.0.0"

__all__ = [
    "AgentInterface",
    "AgentResult",
    "get_agent",
    "PlannerAgent",
    "DeveloperAgent",
    "QAAgent",
    "ReviewerAgent",
    "DocAgent",
]

# =============================================================================
# AGENT FACTORY
# =============================================================================
"""
def get_agent(agent_type: str) -> AgentInterface:
    '''
    Factory function to get appropriate agent instance.

    Args:
        agent_type: Type of agent to create
            - "planner": PlannerAgent
            - "developer": DeveloperAgent
            - "qa": QAAgent
            - "reviewer": ReviewerAgent
            - "doc": DocAgent

    Returns:
        AgentInterface implementation

    Raises:
        ValueError: If agent_type is unknown

    Usage:
        agent = get_agent("developer")
        result = agent.run()
    '''
    agents = {
        "planner": PlannerAgent,
        "developer": DeveloperAgent,
        "qa": QAAgent,
        "reviewer": ReviewerAgent,
        "doc": DocAgent,
    }

    if agent_type not in agents:
        raise ValueError(f"Unknown agent type: {agent_type}")

    return agents[agent_type]()
"""
