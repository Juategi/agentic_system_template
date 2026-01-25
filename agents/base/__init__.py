# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - AGENT BASE PACKAGE
# =============================================================================
"""
Agent Base Package

This package contains shared infrastructure used by all agent types.
It provides:
1. Common interface that all agents implement
2. Context loading utilities
3. Output handling and formatting
4. GitHub integration helpers
5. LLM interaction utilities

All agents extend the base interface and use these shared utilities
to ensure consistent behavior across agent types.

Components:
    - AgentInterface: Abstract base class for all agents
    - AgentResult: Standard result structure
    - ContextLoader: Loads memory, repo, and issue context
    - OutputHandler: Writes structured output
    - GitHubHelper: Simplified GitHub operations
    - LLMClient: Configured LLM client

Design Goals:
    - Minimize code duplication across agents
    - Ensure consistent input/output formats
    - Provide robust error handling
    - Enable easy testing via dependency injection
"""

__all__ = [
    "AgentInterface",
    "AgentResult",
    "AgentContext",
    "ContextLoader",
    "OutputHandler",
    "GitHubHelper",
    "LLMClient",
]
