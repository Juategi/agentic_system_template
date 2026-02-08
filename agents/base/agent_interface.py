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
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import logging
import os
import traceback

from monitoring import setup_logging as monitoring_setup_logging, AuditLogger


# =============================================================================
# ENUMS AND DATA STRUCTURES
# =============================================================================

class AgentStatus(Enum):
    """Possible agent execution statuses."""
    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class AgentResult:
    """
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
    """
    status: AgentStatus
    output: Dict[str, Any]
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def success(self) -> bool:
        """Check if execution was successful."""
        return self.status == AgentStatus.SUCCESS

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status.value,
            "output": self.output,
            "message": self.message,
            "details": self.details,
            "errors": self.errors,
            "metrics": self.metrics,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentResult":
        """Create AgentResult from dictionary."""
        return cls(
            status=AgentStatus(data.get("status", "error")),
            output=data.get("output", {}),
            message=data.get("message", ""),
            details=data.get("details", {}),
            errors=data.get("errors", []),
            metrics=data.get("metrics", {}),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat())
        )

    @classmethod
    def error(cls, message: str, errors: List[str] = None) -> "AgentResult":
        """Create an error result."""
        return cls(
            status=AgentStatus.ERROR,
            output={},
            message=message,
            errors=errors or [message]
        )

    @classmethod
    def failure(cls, message: str, output: Dict[str, Any] = None, details: Dict[str, Any] = None) -> "AgentResult":
        """Create a failure result."""
        return cls(
            status=AgentStatus.FAILURE,
            output=output or {},
            message=message,
            details=details or {}
        )


@dataclass
class AgentContext:
    """
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
    """
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

    # Images from issue (mockups, screenshots, diagrams)
    images: List[Any] = field(default_factory=list)

    @property
    def has_images(self) -> bool:
        """Check if images are available for this context."""
        return len(self.images) > 0

    @property
    def issue_title(self) -> str:
        """Get issue title."""
        return self.issue_data.get("title", "")

    @property
    def issue_body(self) -> str:
        """Get issue body."""
        return self.issue_data.get("body", "")

    @property
    def issue_labels(self) -> List[str]:
        """Get issue labels."""
        return self.issue_data.get("labels", [])

    def get_memory_file(self, name: str) -> str:
        """Get content of a memory file."""
        return self.memory.get(name, "")

    def has_memory_file(self, name: str) -> bool:
        """Check if memory file exists."""
        return name in self.memory and len(self.memory[name]) > 0


# =============================================================================
# AGENT INTERFACE
# =============================================================================

class AgentInterface(ABC):
    """
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
    """

    def __init__(self):
        """
        Initialize the agent.

        Initialization steps:
        1. Set up logging
        2. Load configuration
        3. Initialize LLM client (lazy)
        4. Initialize GitHub helper (lazy)
        5. Prepare output handler (lazy)
        """
        self.logger = self._setup_logging()
        self.config: Dict[str, Any] = {}
        self._llm = None
        self._github = None
        self._output_handler = None
        self._context_loader = None
        self._start_time: Optional[datetime] = None
        self._metrics: Dict[str, Any] = {
            "llm_calls": 0,
            "tokens_input": 0,
            "tokens_output": 0,
            "files_read": 0,
            "files_modified": 0
        }

        # Audit logger for permanent agent execution trail
        log_dir = os.environ.get("LOG_PATH", "./logs")
        self.audit = AuditLogger(output_path=f"{log_dir}/audit.jsonl")

    def _setup_logging(self) -> logging.Logger:
        """Set up agent-specific logger using the monitoring module."""
        log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
        log_fmt = os.environ.get("LOG_FORMAT", "json")
        log_dir = os.environ.get("LOG_PATH", "./logs")
        agent_type = self.get_agent_type()

        # Use monitoring module: writes to logs/agent-{type}.log with rotation
        monitoring_setup_logging(
            level=log_level,
            fmt=log_fmt,
            log_file=f"{log_dir}/agent-{agent_type}.log",
            mask_sensitive=True,
            max_bytes=50 * 1024 * 1024,  # 50 MB per agent log
            backup_count=3,
        )

        return logging.getLogger(f"agent.{agent_type}")

    @property
    def llm(self):
        """Lazy-load LLM client."""
        if self._llm is None:
            from .llm_client import LLMClient
            self._llm = LLMClient()
        return self._llm

    @property
    def github(self):
        """Lazy-load GitHub helper."""
        if self._github is None:
            from .context_loader import GitHubHelper
            self._github = GitHubHelper()
        return self._github

    @property
    def output_handler(self):
        """Lazy-load output handler."""
        if self._output_handler is None:
            from .output_handler import OutputHandler
            self._output_handler = OutputHandler()
        return self._output_handler

    @property
    def context_loader(self):
        """Lazy-load context loader."""
        if self._context_loader is None:
            from .context_loader import ContextLoader
            self._context_loader = ContextLoader()
        return self._context_loader

    def run(self) -> AgentResult:
        """
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
        """
        self._start_time = datetime.utcnow()
        self.logger.info(f"Starting {self.get_agent_type()} agent")

        try:
            # Load context
            self.logger.debug("Loading context...")
            context = self._load_context()
            self.logger.info(f"Context loaded for issue #{context.issue_number}")

            # Validate context
            self.logger.debug("Validating context...")
            if not self.validate_context(context):
                result = AgentResult.error(
                    "Context validation failed",
                    ["Required context is missing or invalid"]
                )
                self._finalize_result(result)
                self._write_output(result)
                return result

            # Execute agent logic
            self.logger.info("Executing agent logic...")
            result = self.execute(context)

            # Finalize metrics
            self._finalize_result(result)

            # Write output
            self._write_output(result)

            # Update GitHub (if applicable)
            self._update_github(context, result)

            # Audit: log agent execution
            duration = (datetime.utcnow() - self._start_time).total_seconds()
            self.audit.log_agent_execution(
                agent_type=self.get_agent_type(),
                issue_number=context.issue_number,
                result=result.status.value,
                duration=duration,
                output_summary=result.message,
            )

            self.logger.info(f"Agent completed with status: {result.status.value}")
            return result

        except Exception as e:
            self.logger.error(f"Agent execution failed: {str(e)}")
            self.logger.error(traceback.format_exc())

            # Audit: log error
            duration = (datetime.utcnow() - self._start_time).total_seconds() if self._start_time else 0
            self.audit.log_error(
                component=f"agent.{self.get_agent_type()}",
                error_type=type(e).__name__,
                message=str(e),
            )

            error_result = AgentResult.error(
                f"Agent execution failed: {str(e)}",
                [str(e), traceback.format_exc()]
            )
            self._finalize_result(error_result)
            self._write_output(error_result)
            return error_result

    def _finalize_result(self, result: AgentResult):
        """Add execution metrics to result."""
        if self._start_time:
            duration = (datetime.utcnow() - self._start_time).total_seconds()
            self._metrics["duration_seconds"] = duration

        result.metrics.update(self._metrics)

    @abstractmethod
    def get_agent_type(self) -> str:
        """
        Return the agent type identifier.

        Returns:
            Agent type string (planner, developer, qa, reviewer, doc)
        """
        pass

    @abstractmethod
    def validate_context(self, context: AgentContext) -> bool:
        """
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
        """
        pass

    @abstractmethod
    def execute(self, context: AgentContext) -> AgentResult:
        """
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
        """
        pass

    def _load_context(self) -> AgentContext:
        """
        Load context from environment and volumes.

        Uses ContextLoader to:
        1. Read environment variables
        2. Load memory files
        3. Fetch issue data
        4. Load orchestrator input

        Returns:
            Populated AgentContext
        """
        return self.context_loader.load()

    def _write_output(self, result: AgentResult):
        """
        Write result to output volume.

        Uses OutputHandler to:
        1. Format result as JSON
        2. Write to output file
        3. Log summary
        """
        try:
            self.output_handler.write_result(result)
            self.logger.debug("Result written to output volume")
        except Exception as e:
            self.logger.error(f"Failed to write output: {e}")

    def _update_github(self, context: AgentContext, result: AgentResult):
        """
        Update GitHub issue with result.

        Posts comment with:
        - Agent type and status
        - Summary of actions taken
        - Errors if any
        """
        try:
            from .output_handler import ResultFormatter
            comment_body = ResultFormatter.to_markdown(result, self.get_agent_type())
            self.github.add_comment(context.issue_number, comment_body)
            self.logger.debug(f"Posted comment to issue #{context.issue_number}")
        except Exception as e:
            self.logger.warning(f"Failed to update GitHub: {e}")

    def track_llm_call(self, tokens_input: int = 0, tokens_output: int = 0):
        """Track LLM API call metrics."""
        self._metrics["llm_calls"] += 1
        self._metrics["tokens_input"] += tokens_input
        self._metrics["tokens_output"] += tokens_output

    def track_file_read(self, count: int = 1):
        """Track file read operations."""
        self._metrics["files_read"] += count

    def track_file_modified(self, count: int = 1):
        """Track file modification operations."""
        self._metrics["files_modified"] += count
