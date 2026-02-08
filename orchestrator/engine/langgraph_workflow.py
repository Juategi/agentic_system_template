# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - LANGGRAPH WORKFLOW DEFINITION
# =============================================================================
"""
LangGraph Workflow Module

This module defines the state machine for the AI agent development workflow
using LangGraph. The workflow manages the lifecycle of GitHub issues from
creation to completion.

State Machine Overview:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        LANGGRAPH WORKFLOW                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │    ┌──────────┐                                                         │
    │    │  TRIAGE  │ ─────────────────┬─────────────────┐                   │
    │    └────┬─────┘                  │                 │                   │
    │         │                        │                 │                   │
    │         │ (is_feature)           │ (is_task)       │ (invalid)         │
    │         ▼                        ▼                 ▼                   │
    │    ┌──────────┐            ┌──────────┐      ┌──────────┐             │
    │    │ PLANNING │            │   DEV    │      │ BLOCKED  │             │
    │    └────┬─────┘            └────┬─────┘      └──────────┘             │
    │         │                       │                                      │
    │         │                       │                                      │
    │         ▼                       ▼                                      │
    │    ┌──────────┐            ┌──────────┐                               │
    │    │  AWAIT   │            │    QA    │◀─────────────────┐            │
    │    │ SUBTASKS │            └────┬─────┘                  │            │
    │    └────┬─────┘                 │                        │            │
    │         │                       ├──── (pass) ────┐       │            │
    │         │                       │                │       │            │
    │         │                       │ (fail)         │       │            │
    │         │                       ▼                │       │            │
    │         │                  ┌──────────┐          │       │            │
    │         │                  │ QA_FAIL  │          │       │            │
    │         │                  └────┬─────┘          │       │            │
    │         │                       │                │       │            │
    │         │          ┌────────────┼────────────┐   │       │            │
    │         │          │            │            │   │       │            │
    │         │    (can retry)   (max iters)       │   │       │            │
    │         │          │            │            │   │       │            │
    │         │          │            ▼            │   │       │            │
    │         │          │       ┌──────────┐      │   │       │            │
    │         │          │       │ BLOCKED  │      │   │       │            │
    │         │          │       └──────────┘      │   │       │            │
    │         │          │                         │   │       │            │
    │         │          └────────── DEV ◀─────────┘   │       │            │
    │         │                                        │       │            │
    │         │                                        ▼       │            │
    │         │                                   ┌──────────┐ │            │
    │         │                                   │  REVIEW  │─┘            │
    │         │                                   └────┬─────┘ (changes)    │
    │         │                                        │                    │
    │         │                            (approved)  │                    │
    │         │                                        ▼                    │
    │         │                                   ┌──────────┐              │
    │         └──────────────────────────────────▶│   DOC    │              │
    │                                             └────┬─────┘              │
    │                                                  │                    │
    │                                                  ▼                    │
    │                                             ┌──────────┐              │
    │                                             │   DONE   │              │
    │                                             └──────────┘              │
    │                                                                       │
    └───────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import (
    Dict,
    List,
    Any,
    Optional,
    Callable,
    Awaitable,
    TypedDict,
    Literal,
    Union,
    Tuple,
    TYPE_CHECKING,
)
import json
import traceback

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.checkpoint.base import BaseCheckpointSaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = "__end__"
    CompiledStateGraph = None
    BaseCheckpointSaver = None

# Local imports
from orchestrator.engine.state_manager import (
    StateManager,
    WorkflowState as PersistedState,
    IssueState,
    TransitionRecord,
    StateNotFoundError,
)
from orchestrator.engine.langchain_setup import (
    LangChainEngine,
    LLMResponse,
    ToolCall,
)
from orchestrator.scheduler.agent_launcher import (
    AgentLauncher,
    AgentType,
    AgentResult,
    ContainerStatus,
    ContainerTimeoutError,
)

if TYPE_CHECKING:
    from orchestrator.github.client import GitHubClient


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# EXCEPTIONS
# =============================================================================


class WorkflowError(Exception):
    """Base exception for workflow errors."""
    pass


class WorkflowNodeError(WorkflowError):
    """Error during node execution."""
    pass


class WorkflowTransitionError(WorkflowError):
    """Error during state transition."""
    pass


class WorkflowTimeoutError(WorkflowError):
    """Timeout waiting for agent or operation."""
    pass


class WorkflowConfigError(WorkflowError):
    """Configuration error in workflow."""
    pass


# =============================================================================
# WORKFLOW STATE TYPED DICT
# =============================================================================


class GraphState(TypedDict, total=False):
    """
    State object passed through the LangGraph workflow.

    This TypedDict defines the schema for state that flows through
    the graph. Each node can read and modify this state.
    """
    # Core identifiers
    issue_number: int
    issue_state: str
    issue_type: str

    # Issue details
    issue_title: str
    issue_body: str
    issue_labels: List[str]

    # Agent tracking
    current_agent: Optional[str]
    iteration_count: int
    max_iterations: int

    # Results and outputs
    last_agent_output: Optional[Dict[str, Any]]
    qa_result: Optional[str]
    review_result: Optional[str]

    # Error handling
    error_message: Optional[str]

    # Hierarchical issues
    parent_issue: Optional[int]
    child_issues: List[int]

    # Metadata and history
    history: List[Dict[str, Any]]
    metadata: Dict[str, Any]

    # Timestamps
    created_at: str
    updated_at: str


# =============================================================================
# NODE RESULT TYPES
# =============================================================================


@dataclass
class NodeResult:
    """Result from a node execution."""
    success: bool
    next_state: Optional[str] = None
    error_message: Optional[str] = None
    agent_output: Optional[Dict[str, Any]] = None
    metadata_updates: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# WORKFLOW ENGINE CLASS
# =============================================================================


class WorkflowEngine:
    """
    Manages the LangGraph state machine for issue processing.

    This class:
    1. Constructs the LangGraph workflow
    2. Executes workflow for issues
    3. Handles state persistence
    4. Manages transitions and routing

    Attributes:
        langchain_engine: LangChain engine for LLM calls
        state_manager: State persistence manager
        agent_launcher: Agent container launcher
        github_client: GitHub API client
        graph: Compiled LangGraph workflow
        config: Workflow configuration
    """

    def __init__(
        self,
        langchain_engine: LangChainEngine,
        state_manager: StateManager,
        agent_launcher: AgentLauncher,
        github_client: 'GitHubClient',
        config: Dict[str, Any],
    ):
        """
        Initialize the workflow engine.

        Args:
            langchain_engine: LangChain engine instance
            state_manager: State persistence manager
            agent_launcher: Agent container launcher
            github_client: GitHub API client
            config: Workflow configuration
        """
        self.langchain_engine = langchain_engine
        self.state_manager = state_manager
        self.agent_launcher = agent_launcher
        self.github_client = github_client
        self.config = config

        # Configuration values
        self.max_iterations = config.get("max_iterations", 5)
        self.agent_timeout = config.get("agent_timeout", 3600)  # 1 hour default
        self.node_timeout = config.get("node_timeout", 7200)  # 2 hours default

        # Build the workflow graph
        self.graph: Optional[CompiledStateGraph] = None
        self._node_handlers: Dict[str, Callable] = {}

        # Initialize
        self._setup_node_handlers()
        if LANGGRAPH_AVAILABLE:
            self.graph = self._build_graph()
        else:
            logger.warning("LangGraph not available, using fallback execution")

    def _setup_node_handlers(self) -> None:
        """Setup node handler functions."""
        self._node_handlers = {
            "triage": self._triage_node,
            "planning": self._planning_node,
            "await_subtasks": self._await_subtasks_node,
            "development": self._development_node,
            "qa": self._qa_node,
            "qa_failed": self._qa_failed_node,
            "review": self._review_node,
            "documentation": self._documentation_node,
            "done": self._done_node,
            "blocked": self._blocked_node,
        }

    def _build_graph(self) -> CompiledStateGraph:
        """
        Build the LangGraph workflow definition.

        Returns:
            Compiled LangGraph workflow
        """
        if not LANGGRAPH_AVAILABLE:
            raise WorkflowConfigError("LangGraph is not installed")

        # Create state graph
        graph = StateGraph(GraphState)

        # Add all nodes
        graph.add_node("triage", self._triage_node)
        graph.add_node("planning", self._planning_node)
        graph.add_node("await_subtasks", self._await_subtasks_node)
        graph.add_node("development", self._development_node)
        graph.add_node("qa", self._qa_node)
        graph.add_node("qa_failed", self._qa_failed_node)
        graph.add_node("review", self._review_node)
        graph.add_node("documentation", self._documentation_node)
        graph.add_node("done", self._done_node)
        graph.add_node("blocked", self._blocked_node)

        # Set entry point
        graph.set_entry_point("triage")

        # Add conditional edges from triage
        graph.add_conditional_edges(
            "triage",
            self._triage_router,
            {
                "feature": "planning",
                "task": "development",
                "bug": "development",
                "invalid": "blocked",
            }
        )

        # Planning -> Await Subtasks
        graph.add_edge("planning", "await_subtasks")

        # Await Subtasks -> Documentation (when all subtasks complete)
        graph.add_conditional_edges(
            "await_subtasks",
            self._await_subtasks_router,
            {
                "complete": "documentation",
                "waiting": END,  # Exit and wait for subtask completion
                "error": "blocked",
            }
        )

        # Development -> QA
        graph.add_edge("development", "qa")

        # QA conditional edges
        graph.add_conditional_edges(
            "qa",
            self._qa_router,
            {
                "pass": "review",
                "fail_retriable": "qa_failed",
                "fail_blocked": "blocked",
            }
        )

        # QA Failed -> Development (retry)
        graph.add_edge("qa_failed", "development")

        # Review conditional edges
        graph.add_conditional_edges(
            "review",
            self._review_router,
            {
                "approved": "documentation",
                "changes_requested": "qa",  # Back to QA after changes
                "error": "blocked",
            }
        )

        # Documentation -> Done
        graph.add_edge("documentation", "done")

        # Terminal nodes
        graph.add_edge("done", END)
        graph.add_edge("blocked", END)

        # Compile the graph
        return graph.compile()

    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================

    async def run(self, issue_number: int) -> GraphState:
        """
        Execute the workflow for an issue.

        Args:
            issue_number: GitHub issue number

        Returns:
            Final workflow state
        """
        logger.info(f"Starting workflow for issue #{issue_number}")

        # Initialize or load state
        state = await self._initialize_state(issue_number)

        try:
            if self.graph is not None:
                # Run with LangGraph
                result = await self._run_with_langgraph(state)
            else:
                # Fallback execution without LangGraph
                result = await self._run_fallback(state)

            return result

        except Exception as e:
            logger.error(f"Workflow error for issue #{issue_number}: {e}")
            state["error_message"] = str(e)
            state["issue_state"] = IssueState.BLOCKED.value
            await self._persist_state(issue_number, state)
            raise WorkflowError(f"Workflow failed: {e}") from e

    async def resume(self, issue_number: int) -> GraphState:
        """
        Resume workflow from persisted state.

        Args:
            issue_number: Issue to resume

        Returns:
            Workflow state after resumption
        """
        logger.info(f"Resuming workflow for issue #{issue_number}")

        # Load persisted state
        persisted = await self.state_manager.get_state(issue_number)
        if persisted is None:
            raise StateNotFoundError(f"No state found for issue #{issue_number}")

        # Convert to graph state
        state = self._persisted_to_graph_state(persisted)

        # Continue execution from current state
        if self.graph is not None:
            return await self._run_with_langgraph(state)
        else:
            return await self._run_fallback(state)

    async def get_state(self, issue_number: int) -> Optional[GraphState]:
        """
        Get current state for an issue.

        Args:
            issue_number: GitHub issue number

        Returns:
            Current graph state or None
        """
        persisted = await self.state_manager.get_state(issue_number)
        if persisted is None:
            return None
        return self._persisted_to_graph_state(persisted)

    async def process_subtask_completion(
        self,
        parent_issue: int,
        child_issue: int,
    ) -> None:
        """
        Handle completion of a subtask.

        Called when a child issue reaches DONE state.

        Args:
            parent_issue: Parent issue number
            child_issue: Completed child issue number
        """
        logger.info(
            f"Processing subtask #{child_issue} completion for parent #{parent_issue}"
        )

        # Update parent state
        parent_state = await self.get_state(parent_issue)
        if parent_state is None:
            logger.warning(f"Parent issue #{parent_issue} not found")
            return

        # Check if all subtasks are complete
        if parent_state.get("issue_state") == IssueState.AWAIT_SUBTASKS.value:
            await self.resume(parent_issue)

    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================

    async def _initialize_state(self, issue_number: int) -> GraphState:
        """Initialize state for a new issue."""
        # Check for existing state
        existing = await self.state_manager.get_state(issue_number)
        if existing is not None:
            return self._persisted_to_graph_state(existing)

        # Fetch issue details from GitHub
        issue = self.github_client.get_issue(issue_number)

        # Create new state
        now = datetime.utcnow().isoformat()
        state: GraphState = {
            "issue_number": issue_number,
            "issue_state": IssueState.TRIAGE.value,
            "issue_type": "",
            "issue_title": issue.get("title", ""),
            "issue_body": issue.get("body", ""),
            "issue_labels": [l.get("name", "") for l in issue.get("labels", [])],
            "current_agent": None,
            "iteration_count": 0,
            "max_iterations": self.max_iterations,
            "last_agent_output": None,
            "qa_result": None,
            "review_result": None,
            "error_message": None,
            "parent_issue": None,
            "child_issues": [],
            "history": [],
            "metadata": {},
            "created_at": now,
            "updated_at": now,
        }

        # Persist initial state
        await self._persist_state(issue_number, state)

        return state

    def _persisted_to_graph_state(self, persisted: PersistedState) -> GraphState:
        """Convert persisted state to graph state."""
        return GraphState(
            issue_number=persisted.issue_number,
            issue_state=persisted.issue_state,
            issue_type=persisted.issue_type,
            issue_title=persisted.metadata.get("title", ""),
            issue_body=persisted.metadata.get("body", ""),
            issue_labels=persisted.metadata.get("labels", []),
            current_agent=persisted.current_agent,
            iteration_count=persisted.iteration_count,
            max_iterations=persisted.max_iterations,
            last_agent_output=persisted.last_agent_output,
            qa_result=persisted.metadata.get("qa_result"),
            review_result=persisted.metadata.get("review_result"),
            error_message=persisted.error_message,
            parent_issue=persisted.parent_issue,
            child_issues=persisted.child_issues,
            history=[asdict(h) if hasattr(h, '__dataclass_fields__') else h
                     for h in persisted.history],
            metadata=persisted.metadata,
            created_at=persisted.created_at,
            updated_at=persisted.updated_at,
        )

    def _graph_to_persisted_state(self, state: GraphState) -> PersistedState:
        """Convert graph state to persisted state."""
        metadata = state.get("metadata", {}).copy()
        metadata["title"] = state.get("issue_title", "")
        metadata["body"] = state.get("issue_body", "")
        metadata["labels"] = state.get("issue_labels", [])
        metadata["qa_result"] = state.get("qa_result")
        metadata["review_result"] = state.get("review_result")

        return PersistedState(
            issue_number=state["issue_number"],
            issue_state=state.get("issue_state", IssueState.TRIAGE.value),
            issue_type=state.get("issue_type", ""),
            current_agent=state.get("current_agent"),
            iteration_count=state.get("iteration_count", 0),
            max_iterations=state.get("max_iterations", self.max_iterations),
            last_agent_output=state.get("last_agent_output"),
            error_message=state.get("error_message"),
            parent_issue=state.get("parent_issue"),
            child_issues=state.get("child_issues", []),
            history=state.get("history", []),
            metadata=metadata,
            created_at=state.get("created_at", datetime.utcnow().isoformat()),
            updated_at=datetime.utcnow().isoformat(),
        )

    async def _persist_state(self, issue_number: int, state: GraphState) -> None:
        """Persist the current state."""
        state["updated_at"] = datetime.utcnow().isoformat()
        persisted = self._graph_to_persisted_state(state)
        await self.state_manager.save_state(issue_number, persisted)

    async def _record_transition(
        self,
        state: GraphState,
        from_state: str,
        to_state: str,
        trigger: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> GraphState:
        """Record a state transition in history."""
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "from_state": from_state,
            "to_state": to_state,
            "trigger": trigger,
            "agent_type": state.get("current_agent"),
            "details": details or {},
        }

        history = list(state.get("history", []))
        history.append(record)
        state["history"] = history

        logger.info(
            f"Issue #{state['issue_number']}: {from_state} -> {to_state} ({trigger})"
        )

        return state

    # =========================================================================
    # EXECUTION METHODS
    # =========================================================================

    async def _run_with_langgraph(self, state: GraphState) -> GraphState:
        """Execute workflow using LangGraph."""
        if self.graph is None:
            raise WorkflowConfigError("LangGraph not initialized")

        # Determine entry node based on current state
        entry = self._get_entry_node(state)

        # Run the graph
        config = {"recursion_limit": 50}

        async for event in self.graph.astream(state, config=config):
            # Process events (for logging/monitoring)
            for node_name, node_state in event.items():
                if node_name != "__end__":
                    logger.debug(f"Node {node_name} completed")
                    state = node_state
                    await self._persist_state(state["issue_number"], state)

        return state

    async def _run_fallback(self, state: GraphState) -> GraphState:
        """Execute workflow without LangGraph (fallback mode)."""
        current_node = self._get_entry_node(state)

        while current_node != "__end__":
            logger.info(f"Executing node: {current_node}")

            # Get handler
            handler = self._node_handlers.get(current_node)
            if handler is None:
                raise WorkflowNodeError(f"Unknown node: {current_node}")

            # Execute node
            try:
                state = await asyncio.wait_for(
                    handler(state),
                    timeout=self.node_timeout
                )
            except asyncio.TimeoutError:
                state["error_message"] = f"Timeout in node: {current_node}"
                state["issue_state"] = IssueState.BLOCKED.value
                await self._persist_state(state["issue_number"], state)
                break

            # Persist state
            await self._persist_state(state["issue_number"], state)

            # Determine next node
            current_node = self._get_next_node(current_node, state)

        return state

    def _get_entry_node(self, state: GraphState) -> str:
        """Get the entry node based on current state."""
        current = state.get("issue_state", IssueState.TRIAGE.value)

        state_to_node = {
            IssueState.TRIAGE.value: "triage",
            IssueState.PLANNING.value: "planning",
            IssueState.AWAIT_SUBTASKS.value: "await_subtasks",
            IssueState.DEVELOPMENT.value: "development",
            IssueState.QA.value: "qa",
            IssueState.QA_FAILED.value: "qa_failed",
            IssueState.REVIEW.value: "review",
            IssueState.DOCUMENTATION.value: "documentation",
            IssueState.DONE.value: "done",
            IssueState.BLOCKED.value: "blocked",
        }

        return state_to_node.get(current, "triage")

    def _get_next_node(self, current_node: str, state: GraphState) -> str:
        """Get the next node based on current node and state."""
        if current_node == "triage":
            return self._triage_router(state)
        elif current_node == "planning":
            return "await_subtasks"
        elif current_node == "await_subtasks":
            result = self._await_subtasks_router(state)
            if result == "complete":
                return "documentation"
            elif result == "waiting":
                return "__end__"
            else:
                return "blocked"
        elif current_node == "development":
            return "qa"
        elif current_node == "qa":
            result = self._qa_router(state)
            if result == "pass":
                return "review"
            elif result == "fail_retriable":
                return "qa_failed"
            else:
                return "blocked"
        elif current_node == "qa_failed":
            return "development"
        elif current_node == "review":
            result = self._review_router(state)
            if result == "approved":
                return "documentation"
            elif result == "changes_requested":
                return "qa"
            else:
                return "blocked"
        elif current_node == "documentation":
            return "done"
        elif current_node in ("done", "blocked"):
            return "__end__"
        else:
            return "__end__"

    # =========================================================================
    # NODE IMPLEMENTATIONS
    # =========================================================================

    async def _triage_node(self, state: GraphState) -> GraphState:
        """
        Triage node - Entry point for new issues.

        Analyzes the issue to determine its type and validity.
        """
        issue_number = state["issue_number"]
        logger.info(f"Triage node for issue #{issue_number}")

        old_state = state.get("issue_state", IssueState.TRIAGE.value)
        state["issue_state"] = IssueState.TRIAGE.value

        try:
            # Use LLM to triage the issue
            triage_result = await self.langchain_engine.triage_issue(
                issue_number=issue_number,
                title=state.get("issue_title", ""),
                body=state.get("issue_body", ""),
                labels=state.get("issue_labels", []),
            )

            # Extract issue type from triage
            issue_type = triage_result.get("issue_type", "task")

            # Validate issue type
            valid_types = {"feature", "task", "bug", "invalid"}
            if issue_type not in valid_types:
                issue_type = "invalid"

            state["issue_type"] = issue_type
            state["metadata"] = state.get("metadata", {})
            state["metadata"]["triage_result"] = triage_result

            # Update GitHub labels
            await self._update_github_labels(
                issue_number,
                add_labels=["TRIAGE", issue_type.upper()],
            )

            # Record transition
            state = await self._record_transition(
                state, old_state, IssueState.TRIAGE.value,
                "triage_complete",
                {"issue_type": issue_type}
            )

        except Exception as e:
            logger.error(f"Triage failed for issue #{issue_number}: {e}")
            state["issue_type"] = "invalid"
            state["error_message"] = f"Triage failed: {e}"

        return state

    async def _planning_node(self, state: GraphState) -> GraphState:
        """
        Planning node - Decompose features into tasks.

        Launches the Planner agent to break down the feature.
        """
        issue_number = state["issue_number"]
        logger.info(f"Planning node for issue #{issue_number}")

        old_state = state.get("issue_state")
        state["issue_state"] = IssueState.PLANNING.value
        state["current_agent"] = AgentType.PLANNER.value

        try:
            # Update GitHub
            await self._update_github_labels(
                issue_number,
                add_labels=["PLANNING", "IN_PROGRESS"],
                remove_labels=["TRIAGE", "READY"],
            )

            # Launch planner agent
            result = await self._launch_agent(
                AgentType.PLANNER,
                issue_number,
                context={
                    "title": state.get("issue_title", ""),
                    "body": state.get("issue_body", ""),
                    "labels": state.get("issue_labels", []),
                }
            )

            state["last_agent_output"] = result.output
            state["current_agent"] = None

            # Process created subtasks
            if result.success and result.output:
                subtasks = result.output.get("subtasks", [])
                child_issues = []

                for subtask in subtasks:
                    # Create child issue via GitHub client
                    child_issue = self.github_client.create_issue(
                        title=subtask.get("title", ""),
                        body=subtask.get("body", ""),
                        labels=["task", "READY"],
                    )
                    child_issues.append(child_issue["number"])

                state["child_issues"] = child_issues

            # Record transition
            state = await self._record_transition(
                state, old_state, IssueState.PLANNING.value,
                "planning_complete",
                {"child_issues": state.get("child_issues", [])}
            )

        except Exception as e:
            logger.error(f"Planning failed for issue #{issue_number}: {e}")
            state["error_message"] = f"Planning failed: {e}"
            state["current_agent"] = None

        return state

    async def _await_subtasks_node(self, state: GraphState) -> GraphState:
        """
        Await Subtasks node - Wait for all child issues to complete.
        """
        issue_number = state["issue_number"]
        logger.info(f"Await subtasks node for issue #{issue_number}")

        old_state = state.get("issue_state")
        state["issue_state"] = IssueState.AWAIT_SUBTASKS.value

        try:
            # Update GitHub
            await self._update_github_labels(
                issue_number,
                add_labels=["AWAIT_SUBTASKS"],
                remove_labels=["PLANNING"],
            )

            # Check status of all child issues
            child_issues = state.get("child_issues", [])
            all_complete = True

            for child_number in child_issues:
                child_state = await self.state_manager.get_state(child_number)
                if child_state is None or child_state.issue_state != IssueState.DONE.value:
                    all_complete = False
                    break

            state["metadata"] = state.get("metadata", {})
            state["metadata"]["subtasks_complete"] = all_complete

            # Record transition
            state = await self._record_transition(
                state, old_state, IssueState.AWAIT_SUBTASKS.value,
                "await_subtasks_check",
                {"all_complete": all_complete}
            )

        except Exception as e:
            logger.error(f"Await subtasks failed for issue #{issue_number}: {e}")
            state["error_message"] = f"Await subtasks check failed: {e}"

        return state

    async def _development_node(self, state: GraphState) -> GraphState:
        """
        Development node - Implement the task.

        Launches the Developer agent to write code.
        """
        issue_number = state["issue_number"]
        logger.info(f"Development node for issue #{issue_number}")

        old_state = state.get("issue_state")
        state["issue_state"] = IssueState.DEVELOPMENT.value
        state["current_agent"] = AgentType.DEVELOPER.value

        try:
            # Update GitHub
            await self._update_github_labels(
                issue_number,
                add_labels=["DEVELOPMENT", "IN_PROGRESS"],
                remove_labels=["TRIAGE", "READY", "QA_FAILED"],
            )

            # Gather context including QA feedback if available
            context = {
                "title": state.get("issue_title", ""),
                "body": state.get("issue_body", ""),
                "labels": state.get("issue_labels", []),
                "iteration": state.get("iteration_count", 0),
            }

            # Add QA feedback if this is a retry
            if state.get("iteration_count", 0) > 0:
                qa_output = state.get("last_agent_output") or {}
                context["qa_feedback"] = qa_output.get("feedback", {})

            # Launch developer agent
            result = await self._launch_agent(
                AgentType.DEVELOPER,
                issue_number,
                context=context
            )

            state["last_agent_output"] = result.output
            state["current_agent"] = None

            if result.success and result.output:
                state["metadata"] = state.get("metadata", {})
                state["metadata"]["modified_files"] = result.output.get("modified_files", [])

            # Record transition
            state = await self._record_transition(
                state, old_state, IssueState.DEVELOPMENT.value,
                "development_complete",
                {"success": result.success}
            )

        except Exception as e:
            logger.error(f"Development failed for issue #{issue_number}: {e}")
            state["error_message"] = f"Development failed: {e}"
            state["current_agent"] = None

        return state

    async def _qa_node(self, state: GraphState) -> GraphState:
        """
        QA node - Validate implementation.

        Launches the QA agent to test the changes.
        """
        issue_number = state["issue_number"]
        logger.info(f"QA node for issue #{issue_number}")

        old_state = state.get("issue_state")
        state["issue_state"] = IssueState.QA.value
        state["current_agent"] = AgentType.QA.value

        try:
            # Update GitHub
            await self._update_github_labels(
                issue_number,
                add_labels=["QA"],
                remove_labels=["DEVELOPMENT", "REVIEW"],
            )

            # Gather context
            context = {
                "title": state.get("issue_title", ""),
                "body": state.get("issue_body", ""),
                "modified_files": state.get("metadata", {}).get("modified_files", []),
                "developer_output": state.get("last_agent_output") or {},
            }

            # Launch QA agent
            result = await self._launch_agent(
                AgentType.QA,
                issue_number,
                context=context
            )

            state["last_agent_output"] = result.output
            state["current_agent"] = None

            # Determine QA result
            qa_passed = result.success and (result.output or {}).get("passed", False)
            state["qa_result"] = "PASS" if qa_passed else "FAIL"
            state["metadata"] = state.get("metadata", {})
            state["metadata"]["qa_result"] = state["qa_result"]

            # Record transition
            state = await self._record_transition(
                state, old_state, IssueState.QA.value,
                "qa_complete",
                {"qa_result": state["qa_result"]}
            )

        except Exception as e:
            logger.error(f"QA failed for issue #{issue_number}: {e}")
            state["error_message"] = f"QA failed: {e}"
            state["qa_result"] = "FAIL"
            state["current_agent"] = None

        return state

    async def _qa_failed_node(self, state: GraphState) -> GraphState:
        """
        QA Failed node - Handle failed validation.

        Increments iteration counter and prepares for retry.
        """
        issue_number = state["issue_number"]
        logger.info(f"QA failed node for issue #{issue_number}")

        old_state = state.get("issue_state")
        state["issue_state"] = IssueState.QA_FAILED.value

        # Increment iteration counter
        iteration = state.get("iteration_count", 0) + 1
        state["iteration_count"] = iteration

        # Update GitHub
        await self._update_github_labels(
            issue_number,
            add_labels=["QA_FAILED"],
            remove_labels=["QA"],
        )

        # Add comment with QA feedback
        qa_output = state.get("last_agent_output") or {}
        feedback = qa_output.get("feedback", "No feedback provided")

        self.github_client.add_comment(
            issue_number,
            f"## QA Failed (Iteration {iteration}/{state.get('max_iterations', self.max_iterations)})\n\n"
            f"**Feedback:**\n{feedback}\n\n"
            f"Retrying development..."
        )

        # Record transition
        state = await self._record_transition(
            state, old_state, IssueState.QA_FAILED.value,
            "qa_failed",
            {"iteration": iteration}
        )

        return state

    async def _review_node(self, state: GraphState) -> GraphState:
        """
        Review node - Code quality review.

        Launches the Reviewer agent to check code quality.
        """
        issue_number = state["issue_number"]
        logger.info(f"Review node for issue #{issue_number}")

        old_state = state.get("issue_state")
        state["issue_state"] = IssueState.REVIEW.value
        state["current_agent"] = AgentType.REVIEWER.value

        try:
            # Update GitHub
            await self._update_github_labels(
                issue_number,
                add_labels=["REVIEW"],
                remove_labels=["QA"],
            )

            # Gather context
            context = {
                "title": state.get("issue_title", ""),
                "body": state.get("issue_body", ""),
                "modified_files": state.get("metadata", {}).get("modified_files", []),
                "qa_output": state.get("last_agent_output") or {},
            }

            # Launch reviewer agent
            result = await self._launch_agent(
                AgentType.REVIEWER,
                issue_number,
                context=context
            )

            state["last_agent_output"] = result.output
            state["current_agent"] = None

            # Determine review result
            if result.success and result.output:
                review_decision = result.output.get("decision", "CHANGES_REQUESTED")
                state["review_result"] = review_decision
            else:
                state["review_result"] = "CHANGES_REQUESTED"

            state["metadata"] = state.get("metadata", {})
            state["metadata"]["review_result"] = state["review_result"]

            # Record transition
            state = await self._record_transition(
                state, old_state, IssueState.REVIEW.value,
                "review_complete",
                {"review_result": state["review_result"]}
            )

        except Exception as e:
            logger.error(f"Review failed for issue #{issue_number}: {e}")
            state["error_message"] = f"Review failed: {e}"
            state["review_result"] = "error"
            state["current_agent"] = None

        return state

    async def _documentation_node(self, state: GraphState) -> GraphState:
        """
        Documentation node - Update project memory.

        Launches the Doc agent to update documentation.
        """
        issue_number = state["issue_number"]
        logger.info(f"Documentation node for issue #{issue_number}")

        old_state = state.get("issue_state")
        state["issue_state"] = IssueState.DOCUMENTATION.value
        state["current_agent"] = AgentType.DOC.value

        try:
            # Update GitHub
            await self._update_github_labels(
                issue_number,
                add_labels=["DOCUMENTATION"],
                remove_labels=["REVIEW", "AWAIT_SUBTASKS"],
            )

            # Gather context
            context = {
                "title": state.get("issue_title", ""),
                "body": state.get("issue_body", ""),
                "modified_files": state.get("metadata", {}).get("modified_files", []),
                "history": state.get("history", []),
            }

            # Launch doc agent
            result = await self._launch_agent(
                AgentType.DOC,
                issue_number,
                context=context
            )

            state["last_agent_output"] = result.output
            state["current_agent"] = None

            # Record transition
            state = await self._record_transition(
                state, old_state, IssueState.DOCUMENTATION.value,
                "documentation_complete",
                {"success": result.success}
            )

        except Exception as e:
            logger.error(f"Documentation failed for issue #{issue_number}: {e}")
            state["error_message"] = f"Documentation failed: {e}"
            state["current_agent"] = None

        return state

    async def _done_node(self, state: GraphState) -> GraphState:
        """
        Done node - Terminal success state.

        Closes the issue and records completion.
        """
        issue_number = state["issue_number"]
        logger.info(f"Done node for issue #{issue_number}")

        old_state = state.get("issue_state")
        state["issue_state"] = IssueState.DONE.value

        try:
            # Update GitHub labels and close issue
            await self._update_github_labels(
                issue_number,
                add_labels=["DONE"],
                remove_labels=["DOCUMENTATION", "IN_PROGRESS"],
            )

            # Add completion comment
            iterations = state.get("iteration_count", 0)
            self.github_client.add_comment(
                issue_number,
                f"## Issue Completed\n\n"
                f"This issue has been successfully completed.\n\n"
                f"- **Iterations:** {iterations}\n"
                f"- **Type:** {state.get('issue_type', 'unknown')}\n"
            )

            # Close the issue
            self.github_client.update_issue(issue_number, state="closed")

            # Record transition
            state = await self._record_transition(
                state, old_state, IssueState.DONE.value,
                "completed",
                {}
            )

        except Exception as e:
            logger.error(f"Done node failed for issue #{issue_number}: {e}")

        return state

    async def _blocked_node(self, state: GraphState) -> GraphState:
        """
        Blocked node - Terminal blocked state.

        Marks the issue as blocked and notifies.
        """
        issue_number = state["issue_number"]
        logger.info(f"Blocked node for issue #{issue_number}")

        old_state = state.get("issue_state")
        state["issue_state"] = IssueState.BLOCKED.value

        try:
            # Update GitHub
            await self._update_github_labels(
                issue_number,
                add_labels=["BLOCKED"],
                remove_labels=["IN_PROGRESS", "DEVELOPMENT", "QA", "REVIEW"],
            )

            # Add blocking comment
            error_msg = state.get("error_message", "Unknown reason")
            self.github_client.add_comment(
                issue_number,
                f"## Issue Blocked\n\n"
                f"This issue requires human intervention.\n\n"
                f"**Reason:** {error_msg}\n\n"
                f"**Iterations:** {state.get('iteration_count', 0)}/{state.get('max_iterations', self.max_iterations)}\n"
            )

            # Record transition
            state = await self._record_transition(
                state, old_state, IssueState.BLOCKED.value,
                "blocked",
                {"error_message": error_msg}
            )

        except Exception as e:
            logger.error(f"Blocked node failed for issue #{issue_number}: {e}")

        return state

    # =========================================================================
    # ROUTER IMPLEMENTATIONS
    # =========================================================================

    def _triage_router(self, state: GraphState) -> str:
        """
        Determine path after triage.

        Returns:
            - "feature" if issue needs decomposition
            - "task" if issue is ready for development
            - "bug" if issue is a bug fix
            - "invalid" if issue is malformed
        """
        issue_type = state.get("issue_type", "")

        if issue_type == "feature":
            return "feature"
        elif issue_type == "task":
            return "task"
        elif issue_type == "bug":
            return "bug"
        else:
            return "invalid"

    def _await_subtasks_router(self, state: GraphState) -> str:
        """
        Determine path after await subtasks.

        Returns:
            - "complete" if all subtasks are done
            - "waiting" if still waiting for subtasks
            - "error" if there's an error
        """
        if state.get("error_message"):
            return "error"

        subtasks_complete = state.get("metadata", {}).get("subtasks_complete", False)

        if subtasks_complete:
            return "complete"
        else:
            return "waiting"

    def _qa_router(self, state: GraphState) -> str:
        """
        Determine path after QA.

        Returns:
            - "pass" if QA passed
            - "fail_retriable" if failed but can retry
            - "fail_blocked" if max iterations reached
        """
        qa_result = state.get("qa_result", "FAIL")

        if qa_result == "PASS":
            return "pass"

        iteration = state.get("iteration_count", 0)
        max_iter = state.get("max_iterations", self.max_iterations)

        if iteration < max_iter:
            return "fail_retriable"
        else:
            return "fail_blocked"

    def _review_router(self, state: GraphState) -> str:
        """
        Determine path after review.

        Returns:
            - "approved" if review passed
            - "changes_requested" if changes needed
            - "error" if review failed
        """
        review_result = state.get("review_result", "")

        if review_result == "APPROVED":
            return "approved"
        elif review_result == "CHANGES_REQUESTED":
            return "changes_requested"
        else:
            return "error"

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    async def _launch_agent(
        self,
        agent_type: AgentType,
        issue_number: int,
        context: Dict[str, Any],
    ) -> AgentResult:
        """Launch an agent and wait for completion."""
        logger.info(f"Launching {agent_type.value} agent for issue #{issue_number}")

        try:
            # Launch the container
            container_id = await self.agent_launcher.launch_agent(
                agent_type=agent_type.value,
                issue_number=issue_number,
                context=context,
            )

            # Wait for completion
            result = await self.agent_launcher.wait_for_completion(
                container_id,
                timeout=self.agent_timeout,
            )

            return result

        except ContainerTimeoutError:
            logger.error(f"Agent timeout for issue #{issue_number}")
            return AgentResult(
                status=ContainerStatus.TIMEOUT,
                output={"error": "Agent timeout"},
                exit_code=-1,
                logs="",
            )
        except Exception as e:
            logger.error(f"Agent launch failed: {e}")
            return AgentResult(
                status=ContainerStatus.FAILED,
                output={"error": str(e)},
                exit_code=-1,
                logs=traceback.format_exc(),
            )

    async def _update_github_labels(
        self,
        issue_number: int,
        add_labels: Optional[List[str]] = None,
        remove_labels: Optional[List[str]] = None,
    ) -> None:
        """Update GitHub issue labels."""
        try:
            if add_labels:
                self.github_client.add_labels(issue_number, add_labels)
            if remove_labels:
                for label in remove_labels:
                    self.github_client.remove_label(issue_number, label)
        except Exception as e:
            logger.warning(f"Failed to update labels for issue #{issue_number}: {e}")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_workflow_engine(
    langchain_engine: LangChainEngine,
    state_manager: StateManager,
    agent_launcher: AgentLauncher,
    github_client: 'GitHubClient',
    config: Optional[Dict[str, Any]] = None,
) -> WorkflowEngine:
    """
    Create a workflow engine instance.

    Args:
        langchain_engine: LangChain engine for LLM calls
        state_manager: State persistence manager
        agent_launcher: Agent container launcher
        github_client: GitHub API client
        config: Optional workflow configuration

    Returns:
        Configured WorkflowEngine instance
    """
    config = config or {}

    return WorkflowEngine(
        langchain_engine=langchain_engine,
        state_manager=state_manager,
        agent_launcher=agent_launcher,
        github_client=github_client,
        config=config,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main class
    "WorkflowEngine",
    "create_workflow_engine",
    # State types
    "GraphState",
    "NodeResult",
    # Exceptions
    "WorkflowError",
    "WorkflowNodeError",
    "WorkflowTransitionError",
    "WorkflowTimeoutError",
    "WorkflowConfigError",
    # Constants
    "LANGGRAPH_AVAILABLE",
]