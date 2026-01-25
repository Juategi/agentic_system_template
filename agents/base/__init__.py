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
    - AgentContext: Context data for agents
    - AgentStatus: Execution status enum
    - ContextLoader: Loads memory, repo, and issue context
    - OutputHandler: Writes structured output
    - ResultFormatter: Formats results for different outputs
    - GitHubHelper: Simplified GitHub operations
    - LLMClient: Configured LLM client

Design Goals:
    - Minimize code duplication across agents
    - Ensure consistent input/output formats
    - Provide robust error handling
    - Enable easy testing via dependency injection

Usage:
    from agents.base import AgentInterface, AgentResult, AgentContext

    class MyAgent(AgentInterface):
        def get_agent_type(self) -> str:
            return "my_agent"

        def validate_context(self, context: AgentContext) -> bool:
            return True

        def execute(self, context: AgentContext) -> AgentResult:
            # Agent logic here
            return AgentResult(
                status=AgentStatus.SUCCESS,
                output={"result": "done"},
                message="Task completed successfully"
            )
"""

# Agent interface and data structures
from .agent_interface import (
    AgentInterface,
    AgentResult,
    AgentContext,
    AgentStatus,
)

# Context loading utilities
from .context_loader import (
    ContextLoader,
    GitHubHelper,
    load_markdown_file,
    parse_memory_file,
    find_feature_memory_file,
    extract_acceptance_criteria,
)

# Output handling
from .output_handler import (
    OutputHandler,
    ResultFormatter,
    validate_output,
    PLANNER_OUTPUT_SCHEMA,
    DEVELOPER_OUTPUT_SCHEMA,
    QA_OUTPUT_SCHEMA,
    REVIEWER_OUTPUT_SCHEMA,
    DOC_OUTPUT_SCHEMA,
)

# LLM client
from .llm_client import (
    LLMClient,
    LLMResponse,
    LLMMessage,
    create_llm_client,
    estimate_tokens,
)


__all__ = [
    # Core interfaces
    "AgentInterface",
    "AgentResult",
    "AgentContext",
    "AgentStatus",

    # Context loading
    "ContextLoader",
    "GitHubHelper",
    "load_markdown_file",
    "parse_memory_file",
    "find_feature_memory_file",
    "extract_acceptance_criteria",

    # Output handling
    "OutputHandler",
    "ResultFormatter",
    "validate_output",
    "PLANNER_OUTPUT_SCHEMA",
    "DEVELOPER_OUTPUT_SCHEMA",
    "QA_OUTPUT_SCHEMA",
    "REVIEWER_OUTPUT_SCHEMA",
    "DOC_OUTPUT_SCHEMA",

    # LLM client
    "LLMClient",
    "LLMResponse",
    "LLMMessage",
    "create_llm_client",
    "estimate_tokens",
]


# Version info
__version__ = "0.1.0"
