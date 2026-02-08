# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - NODE BASE MODULE
# =============================================================================
"""
Shared types, context, and utilities for all workflow nodes.

Every node in the workflow shares:
- NodeContext: Access to services (GitHub, agents, state, LLM)
- GraphState: The state dict flowing through the graph
- Helper functions for common operations
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, TYPE_CHECKING

from orchestrator.engine.state_manager import IssueState
from orchestrator.scheduler.agent_launcher import AgentType, AgentResult, ContainerStatus

if TYPE_CHECKING:
    from orchestrator.github.client import GitHubClient
    from orchestrator.github.issue_manager import IssueManager
    from orchestrator.engine.langchain_setup import LangChainEngine
    from orchestrator.engine.state_manager import StateManager
    from orchestrator.engine.langgraph_workflow import GraphState
    from orchestrator.scheduler.agent_launcher import AgentLauncher

logger = logging.getLogger(__name__)


# =============================================================================
# NODE CONTEXT
# =============================================================================


@dataclass
class NodeContext:
    """
    Shared context passed to every node.

    Provides access to all orchestrator services.
    """
    github_client: 'GitHubClient'
    issue_manager: 'IssueManager'
    langchain_engine: 'LangChainEngine'
    state_manager: 'StateManager'
    agent_launcher: 'AgentLauncher'
    config: Dict[str, Any] = field(default_factory=dict)

    @property
    def max_iterations(self) -> int:
        return self.config.get("max_iterations", 5)

    @property
    def agent_timeout(self) -> int:
        return self.config.get("agent_timeout", 3600)

    @property
    def memory_path(self) -> str:
        return self.config.get("memory_path", "./memory")

    @property
    def repo_path(self) -> str:
        return self.config.get("repo_path", "./repo")


# =============================================================================
# NODE CONFIGURATION DEFAULTS
# =============================================================================


NODE_CONFIGS = {
    "triage": {
        "timeout_seconds": 120,
    },
    "planning": {
        "timeout_seconds": 600,
        "max_subtasks": 10,
    },
    "development": {
        "timeout_seconds": 1800,
        "branch_prefix": "agent/",
    },
    "qa": {
        "timeout_seconds": 600,
        "require_all_tests_pass": True,
        "require_no_linter_errors": True,
    },
    "review": {
        "timeout_seconds": 600,
    },
    "documentation": {
        "timeout_seconds": 300,
    },
}


# =============================================================================
# SHARED HELPERS
# =============================================================================


async def launch_agent_and_wait(
    ctx: NodeContext,
    agent_type: AgentType,
    issue_number: int,
    context: Dict[str, Any],
    timeout: Optional[int] = None,
) -> AgentResult:
    """
    Launch an agent container and wait for completion.

    Args:
        ctx: Node context
        agent_type: Type of agent to launch
        issue_number: Issue being processed
        context: Agent input context
        timeout: Override timeout (uses config default if None)

    Returns:
        AgentResult with success status and output
    """
    timeout = timeout or ctx.agent_timeout

    logger.info(f"Launching {agent_type.value} agent for issue #{issue_number}")

    try:
        container_id = await ctx.agent_launcher.launch_agent(
            agent_type=agent_type,
            issue_number=issue_number,
            context=context,
        )

        result = await ctx.agent_launcher.wait_for_completion(
            container_id,
            timeout=timeout,
        )

        logger.info(
            f"Agent {agent_type.value} completed for issue #{issue_number}: "
            f"success={result.success}"
        )
        return result

    except Exception as e:
        logger.error(f"Agent {agent_type.value} failed for issue #{issue_number}: {e}")
        return AgentResult(
            status=ContainerStatus.FAILED,
            output={"error": str(e)},
            exit_code=-1,
            logs=str(e),
        )


async def update_labels(
    ctx: NodeContext,
    issue_number: int,
    add: Optional[List[str]] = None,
    remove: Optional[List[str]] = None,
) -> None:
    """Update GitHub issue labels safely."""
    try:
        if remove:
            for label in remove:
                try:
                    await asyncio.to_thread(
                        ctx.github_client.remove_label, issue_number, label
                    )
                except Exception:
                    pass
        if add:
            await asyncio.to_thread(
                ctx.github_client.add_labels, issue_number, add
            )
    except Exception as e:
        logger.warning(f"Failed to update labels for #{issue_number}: {e}")


async def post_comment(
    ctx: NodeContext,
    issue_number: int,
    body: str,
) -> None:
    """Post a comment to GitHub issue."""
    try:
        await asyncio.to_thread(
            ctx.github_client.add_comment, issue_number, body
        )
    except Exception as e:
        logger.warning(f"Failed to post comment on #{issue_number}: {e}")


def add_history_entry(
    state: Dict[str, Any],
    node: str,
    action: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Add an entry to the state history."""
    history = list(state.get("history", []))
    history.append({
        "timestamp": datetime.utcnow().isoformat(),
        "node": node,
        "action": action,
        "agent_type": state.get("current_agent"),
        "details": details or {},
    })
    state["history"] = history


def extract_acceptance_criteria(body: str) -> List[str]:
    """
    Extract acceptance criteria from an issue body.

    Looks for:
    - ## Acceptance Criteria
    - - [ ] criterion
    - - criterion
    """
    criteria = []

    # Find acceptance criteria section
    pattern = r"(?:##\s*Acceptance\s+Criteria|##\s*Criteria)\s*\n(.*?)(?:\n##|\Z)"
    match = re.search(pattern, body, re.IGNORECASE | re.DOTALL)

    if match:
        section = match.group(1)
        # Extract list items
        for line in section.strip().split("\n"):
            line = line.strip()
            # Match "- [ ] text", "- [x] text", "- text", "* text"
            item_match = re.match(r"^[-*]\s*(?:\[.\]\s*)?(.+)$", line)
            if item_match:
                criteria.append(item_match.group(1).strip())

    return criteria


def load_memory_file(memory_path: str, filename: str) -> Optional[str]:
    """Load a memory file if it exists."""
    filepath = Path(memory_path) / filename
    if filepath.exists():
        return filepath.read_text(encoding="utf-8")
    return None


def load_project_context(memory_path: str) -> Dict[str, str]:
    """Load all project context files from memory."""
    context = {}

    files = {
        "project": "PROJECT.md",
        "architecture": "ARCHITECTURE.md",
        "conventions": "CONVENTIONS.md",
        "constraints": "CONSTRAINTS.md",
    }

    for key, filename in files.items():
        content = load_memory_file(memory_path, filename)
        if content:
            context[key] = content

    return context


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "NodeContext",
    "NODE_CONFIGS",
    "launch_agent_and_wait",
    "update_labels",
    "post_comment",
    "add_history_entry",
    "extract_acceptance_criteria",
    "load_memory_file",
    "load_project_context",
]