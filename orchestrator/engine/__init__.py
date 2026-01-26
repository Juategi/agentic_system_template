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
        WorkflowState,
        IssueState,
    )

    # Initialize components
    llm_engine = LangChainEngine(config)
    state_manager = await StateManager.create(config)
    workflow = WorkflowEngine(llm_engine, state_manager)

    # Run workflow for an issue
    result = await workflow.run(issue_number=123)
"""

from orchestrator.engine.state_manager import (
    # Main classes
    StateManager,
    StateManagerInterface,
    FileStateManager,
    RedisStateManager,
    PostgreSQLStateManager,
    # Data structures
    WorkflowState,
    TransitionRecord,
    IssueState,
    # Exceptions
    StateManagerError,
    StateNotFoundError,
    StateConcurrencyError,
    StateBackendError,
    StateValidationError,
    # Utilities
    recover_orphaned_states,
    cleanup_completed_states,
)

from orchestrator.engine.langchain_setup import (
    # Main class
    LangChainEngine,
    create_langchain_engine,
    # Configuration
    LLMConfig,
    LLMProvider,
    # Data structures
    LLMResponse,
    ToolCall,
    ToolResult,
    LLMMetrics,
    # Tools
    ToolRegistry,
    TOOL_SCHEMAS,
    # Prompts
    TRIAGE_PROMPT,
    TRANSITION_PROMPT,
    PLANNING_PROMPT,
    # Clients
    LLMClientInterface,
    AnthropicClient,
    OpenAIClient,
    AzureOpenAIClient,
    OllamaClient,
    # Exceptions
    LangChainError,
    LLMProviderError,
    ToolExecutionError,
    ConfigurationError,
)

__all__ = [
    # State Manager
    "StateManager",
    "StateManagerInterface",
    "FileStateManager",
    "RedisStateManager",
    "PostgreSQLStateManager",
    # State Data structures
    "WorkflowState",
    "TransitionRecord",
    "IssueState",
    # State Exceptions
    "StateManagerError",
    "StateNotFoundError",
    "StateConcurrencyError",
    "StateBackendError",
    "StateValidationError",
    # State Utilities
    "recover_orphaned_states",
    "cleanup_completed_states",
    # LangChain Engine
    "LangChainEngine",
    "create_langchain_engine",
    # LLM Configuration
    "LLMConfig",
    "LLMProvider",
    # LLM Data structures
    "LLMResponse",
    "ToolCall",
    "ToolResult",
    "LLMMetrics",
    # Tools
    "ToolRegistry",
    "TOOL_SCHEMAS",
    # Prompts
    "TRIAGE_PROMPT",
    "TRANSITION_PROMPT",
    "PLANNING_PROMPT",
    # LLM Clients
    "LLMClientInterface",
    "AnthropicClient",
    "OpenAIClient",
    "AzureOpenAIClient",
    "OllamaClient",
    # LangChain Exceptions
    "LangChainError",
    "LLMProviderError",
    "ToolExecutionError",
    "ConfigurationError",
    # To be implemented
    "WorkflowEngine",
]