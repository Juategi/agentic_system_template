# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - AGENT INTERFACE
# =============================================================================
"""
Agent Interface Module

This module defines the common interface that all agents must implement.
It provides:
1. Abstract base class with required methods
2. Standard result structure
3. Context data structure
4. Common lifecycle methods

All agent types (Planner, Developer, QA, Reviewer, Doc) must extend
AgentInterface and implement its abstract methods.

Agent Lifecycle:
    ┌─────────────────────────────────────────────────────────────┐
    │                      AGENT LIFECYCLE                         │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  ┌──────────┐    ┌──────────┐    ┌──────────┐             │
    │  │   INIT   │───▶│   RUN    │───▶│ CLEANUP  │             │
    │  └──────────┘    └──────────┘    └──────────┘             │
    │       │               │               │                    │
    │       ▼               ▼               ▼                    │
    │  Load config    Execute task    Write output              │
    │  Load context   Use LLM         Update GitHub             │
    │  Validate env   Make changes    Log results               │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

Environment Variables:
    Required for all agents:
    - AGENT_TYPE: Type of agent (planner, developer, qa, reviewer, doc)
    - PROJECT_ID: Project identifier
    - ISSUE_NUMBER: GitHub issue to work on
    - MEMORY_PATH: Path to memory volume
    - REPO_PATH: Path to repository volume
    - OUTPUT_PATH: Path to output volume
    - GITHUB_TOKEN: GitHub API token
    - GITHUB_REPO: Repository (owner/repo)
    - LLM_PROVIDER: LLM provider (anthropic, openai)
    - ANTHROPIC_API_KEY or OPENAI_API_KEY: LLM API key

    Optional:
    - MAX_ITERATIONS: Max iteration count (for context)
    - ITERATION: Current iteration number
    - LOG_LEVEL: Logging verbosity
"""

# =============================================================================
# DATA STRUCTURES
# =============================================================================
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


class AgentStatus(Enum):
    '''Possible agent execution statuses.'''
    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class AgentResult:
    '''
    Standard result structure returned by all agents.

    Attributes:
        status: Execution status (success, failure, error)
        output: Primary output data (type depends on agent)
        message: Human-readable summary message
        details: Additional details dictionary
        errors: List of error messages if any
        metrics: Execution metrics (duration, tokens, etc.)
        timestamp: Completion timestamp

    All agents return this structure to ensure consistent
    handling by the orchestrator.
    '''
    status: AgentStatus
    output: Dict[str, Any]
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def success(self) -> bool:
        '''Check if execution was successful.'''
        return self.status == AgentStatus.SUCCESS

    def to_dict(self) -> dict:
        '''Convert to dictionary for JSON serialization.'''
        return {
            "status": self.status.value,
            "output": self.output,
            "message": self.message,
            "details": self.details,
            "errors": self.errors,
            "metrics": self.metrics,
            "timestamp": self.timestamp
        }


@dataclass
class AgentContext:
    '''
    Context data provided to agents.

    Attributes:
        issue_number: GitHub issue being worked on
        project_id: Project identifier
        iteration: Current iteration number (for dev loop)
        max_iterations: Maximum iterations allowed

        issue_data: Issue title, body, labels, etc.
        memory: Project memory contents
        repository: Repository metadata

        input_data: Additional input from orchestrator
        config: Agent-specific configuration

    The ContextLoader populates this structure from
    environment and mounted volumes.
    '''
    # Identifiers
    issue_number: int
    project_id: str
    iteration: int = 0
    max_iterations: int = 5

    # GitHub Issue data
    issue_data: Dict[str, Any] = field(default_factory=dict)

    # Project memory
    memory: Dict[str, str] = field(default_factory=dict)

    # Repository info
    repository: Dict[str, Any] = field(default_factory=dict)

    # Orchestrator input
    input_data: Dict[str, Any] = field(default_factory=dict)

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
"""

# =============================================================================
# AGENT INTERFACE
# =============================================================================
"""
class AgentInterface(ABC):
    '''
    Abstract base class for all agent implementations.

    All agent types must extend this class and implement
    the abstract methods. The base class provides:
    - Common initialization logic
    - Context loading
    - Output handling
    - Error handling
    - Logging setup

    Subclasses implement:
    - execute(): Core agent logic
    - validate_context(): Context validation
    - get_agent_type(): Return agent type string

    Usage:
        class DeveloperAgent(AgentInterface):
            def get_agent_type(self) -> str:
                return "developer"

            def validate_context(self, context: AgentContext) -> bool:
                # Validate developer-specific requirements
                return True

            def execute(self, context: AgentContext) -> AgentResult:
                # Implement development logic
                return AgentResult(...)
    '''

    def __init__(self):
        '''
        Initialize the agent.

        Initialization steps:
        1. Set up logging
        2. Load configuration
        3. Initialize LLM client
        4. Initialize GitHub helper
        5. Prepare output handler
        '''
        self.logger = None  # Configured logger
        self.llm = None     # LLM client
        self.github = None  # GitHub helper
        self.output = None  # Output handler
        self.config = {}    # Agent configuration

    def run(self) -> AgentResult:
        '''
        Main entry point for agent execution.

        This method orchestrates the agent lifecycle:
        1. Load context from environment and volumes
        2. Validate context is sufficient
        3. Execute agent-specific logic
        4. Handle errors and write output
        5. Return result

        Returns:
            AgentResult with execution status and output

        This method should NOT be overridden. Override
        execute() instead for agent-specific logic.
        '''
        try:
            # Load context
            context = self._load_context()

            # Validate context
            if not self.validate_context(context):
                return AgentResult(
                    status=AgentStatus.ERROR,
                    output={},
                    message="Context validation failed",
                    errors=["Required context is missing or invalid"]
                )

            # Execute agent logic
            result = self.execute(context)

            # Write output
            self._write_output(result)

            # Update GitHub (if applicable)
            self._update_github(context, result)

            return result

        except Exception as e:
            # Handle unexpected errors
            error_result = AgentResult(
                status=AgentStatus.ERROR,
                output={},
                message=f"Agent execution failed: {str(e)}",
                errors=[str(e)]
            )
            self._write_output(error_result)
            return error_result

    @abstractmethod
    def get_agent_type(self) -> str:
        '''
        Return the agent type identifier.

        Returns:
            Agent type string (planner, developer, qa, reviewer, doc)
        '''
        pass

    @abstractmethod
    def validate_context(self, context: AgentContext) -> bool:
        '''
        Validate that context is sufficient for execution.

        Args:
            context: Loaded agent context

        Returns:
            True if context is valid, False otherwise

        Override to add agent-specific validation.
        Common validations:
        - Required memory files exist
        - Issue has required fields
        - Configuration is complete
        '''
        pass

    @abstractmethod
    def execute(self, context: AgentContext) -> AgentResult:
        '''
        Execute the agent's primary task.

        Args:
            context: Validated agent context

        Returns:
            AgentResult with execution outcome

        This is where agent-specific logic lives:
        - Planner: Decompose features
        - Developer: Write code
        - QA: Validate implementation
        - Reviewer: Review code
        - Doc: Update documentation
        '''
        pass

    def _load_context(self) -> AgentContext:
        '''
        Load context from environment and volumes.

        Uses ContextLoader to:
        1. Read environment variables
        2. Load memory files
        3. Fetch issue data
        4. Load orchestrator input

        Returns:
            Populated AgentContext
        '''
        pass

    def _write_output(self, result: AgentResult):
        '''
        Write result to output volume.

        Uses OutputHandler to:
        1. Format result as JSON
        2. Write to output file
        3. Log summary
        '''
        pass

    def _update_github(self, context: AgentContext, result: AgentResult):
        '''
        Update GitHub issue with result.

        Posts comment with:
        - Agent type and status
        - Summary of actions taken
        - Errors if any
        '''
        pass
'''
"""

# =============================================================================
# IMPLEMENTATION NOTES
# =============================================================================
"""
Implementation Notes:

1. AGENT CONTRACT
   Every agent must:
   - Implement the three abstract methods
   - Return AgentResult from execute()
   - Handle errors gracefully (don't crash)
   - Write meaningful output

2. CONTEXT ISOLATION
   Agents should only access:
   - Mounted volumes (/memory, /repo, /output)
   - Environment variables
   - External APIs (GitHub, LLM)

   Agents should NOT:
   - Access host filesystem
   - Communicate with other containers
   - Maintain persistent state

3. IDEMPOTENCY
   Agents may be run multiple times for the same issue.
   Implementations should:
   - Check existing state before acting
   - Not duplicate work
   - Handle partial completion gracefully

4. TESTING
   Agents should be testable by:
   - Mocking the LLM client
   - Providing test context
   - Verifying output structure

   Example:
       def test_developer_agent():
           agent = DeveloperAgent()
           context = create_test_context()
           result = agent.execute(context)
           assert result.status == AgentStatus.SUCCESS
"""
