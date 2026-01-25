# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - DEVELOPER AGENT PACKAGE
# =============================================================================
"""
Developer Agent Package

The Developer Agent implements code changes to fulfill task requirements.
It is the primary code-writing agent in the system.

Responsibilities:
    1. Understand task requirements from issue
    2. Analyze existing codebase
    3. Write implementation code
    4. Add or update tests
    5. Ensure code follows conventions
    6. Document changes

Input:
    - Task issue (requirements, acceptance criteria)
    - Project context (conventions, architecture)
    - Previous QA feedback (if retry)
    - Related code context

Output:
    - Modified files list
    - Implementation notes
    - Suggested commit message
    - Tests added (if any)

Iteration Support:
    This agent handles both first attempts and retries:
    - First attempt: Fresh implementation from requirements
    - Retry: Incorporates QA feedback to fix issues

Code Quality:
    The agent follows project conventions from CONVENTIONS.md
    and architectural patterns from ARCHITECTURE.md.
"""

from agents.developer.developer_agent import DeveloperAgent

__all__ = ["DeveloperAgent"]
