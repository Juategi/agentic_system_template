# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - PLANNER AGENT PACKAGE
# =============================================================================
"""
Planner Agent Package

The Planner Agent is responsible for decomposing complex features into
smaller, actionable tasks. It is the first agent in the workflow for
feature-type issues.

Responsibilities:
    1. Analyze feature requirements
    2. Identify logical sub-tasks
    3. Create GitHub Issues for each sub-task
    4. Establish task dependencies
    5. Create feature memory file

Input:
    - Feature issue (title, body, acceptance criteria)
    - Project context (memory files)
    - Configuration (max tasks, granularity)

Output:
    - List of created sub-issues
    - Feature memory file path
    - Decomposition summary

Example Workflow:
    Feature: "User Authentication System"

    Planner creates:
    1. [Task] Implement user registration endpoint
    2. [Task] Implement login endpoint
    3. [Task] Add password hashing
    4. [Task] Create authentication middleware
    5. [Task] Add session management

    With dependencies:
    - Task 3 must complete before Task 2
    - Task 4 depends on Task 2 and Task 3

Usage:
    from agents.planner import PlannerAgent

    agent = PlannerAgent()
    result = agent.run()  # Reads context from environment

    # Or with custom context
    from agents.base import AgentContext
    context = AgentContext(issue_number=123, project_id="my-project")
    result = agent.execute(context)
"""

from .planner_agent import PlannerAgent, DECOMPOSITION_SCHEMA

__all__ = ["PlannerAgent", "DECOMPOSITION_SCHEMA"]
__version__ = "0.1.0"
