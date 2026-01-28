# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - ORCHESTRATOR MAIN ENTRY POINT
# =============================================================================
"""
Orchestrator Main Module

This is the entry point for the orchestrator service. It initializes all
components and runs the main processing loop.

The orchestrator operates in two modes:
1. Polling Mode: Periodically checks GitHub for new issues
2. Webhook Mode: Receives real-time events from GitHub

Usage:
    python -m orchestrator.main
    python -m orchestrator.main --config config/orchestrator.yaml
    python -m orchestrator.main --debug
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import yaml

# Local imports
from orchestrator.github.client import GitHubClient
from orchestrator.github.issue_manager import IssueManager, create_issue_manager
from orchestrator.engine.state_manager import StateManager, IssueState
from orchestrator.engine.langchain_setup import LangChainEngine, create_langchain_engine
from orchestrator.engine.langgraph_workflow import (
    WorkflowEngine,
    create_workflow_engine,
    GraphState,
    WorkflowError,
)
from orchestrator.scheduler.agent_launcher import AgentLauncher
from orchestrator.scheduler.queue_manager import (
    QueueManager,
    create_queue_manager,
    QueueStatus,
)

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file and environment variables.

    Environment variables override YAML values.

    Args:
        config_path: Path to orchestrator.yaml

    Returns:
        Merged configuration dictionary
    """
    config = {}

    # Load YAML config if exists
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file) as f:
            config = yaml.safe_load(f) or {}
        logger.info(f"Loaded config from {config_path}")
    else:
        logger.warning(f"Config file not found: {config_path}, using defaults")

    # Environment variable overrides
    env_mappings = {
        # GitHub
        "GITHUB_TOKEN": ("github", "token"),
        "GITHUB_REPO": ("github", "repo"),
        "GITHUB_APP_ID": ("github", "app_id"),
        "GITHUB_APP_KEY": ("github", "app_private_key"),
        # LLM
        "LLM_PROVIDER": ("llm", "provider"),
        "LLM_MODEL": ("llm", "model"),
        "LLM_API_KEY": ("llm", "api_key"),
        "ANTHROPIC_API_KEY": ("llm", "api_key"),
        "OPENAI_API_KEY": ("llm", "openai_api_key"),
        "OLLAMA_BASE_URL": ("llm", "ollama_base_url"),
        # State Manager
        "STATE_BACKEND": ("state", "backend"),
        "REDIS_URL": ("state", "redis_url"),
        "DATABASE_URL": ("state", "database_url"),
        "STATE_DIR": ("state", "state_dir"),
        # Docker / Agent
        "DOCKER_IMAGE": ("agent", "docker_image"),
        "MAX_CONCURRENT_AGENTS": ("agent", "max_concurrent"),
        "AGENT_TIMEOUT": ("agent", "timeout"),
        # Orchestrator
        "POLL_INTERVAL": ("orchestrator", "poll_interval"),
        "MAX_ITERATIONS": ("orchestrator", "max_iterations"),
        "MODE": ("orchestrator", "mode"),
        # Project
        "PROJECT_ID": ("project", "id"),
        "MEMORY_PATH": ("project", "memory_path"),
        "REPO_PATH": ("project", "repo_path"),
    }

    for env_var, (section, key) in env_mappings.items():
        value = os.environ.get(env_var)
        if value is not None:
            if section not in config:
                config[section] = {}
            # Convert numeric strings
            if value.isdigit():
                value = int(value)
            config[section][key] = value

    # Apply defaults
    defaults = {
        "github": {
            "token": "",
            "repo": "",
        },
        "llm": {
            "provider": "anthropic",
            "model": "claude-3-5-sonnet-20241022",
            "temperature": 0.0,
            "max_tokens": 4096,
        },
        "state": {
            "backend": "file",
            "state_dir": "./state",
        },
        "agent": {
            "docker_image": "ai-agent:latest",
            "max_concurrent": 3,
            "timeout": 3600,
        },
        "orchestrator": {
            "poll_interval": 60,
            "max_iterations": 5,
            "mode": "polling",
        },
        "queue": {
            "backend": "memory",
            "max_concurrent": 5,
            "ready_label": "READY",
        },
        "project": {
            "id": "default",
            "memory_path": "./memory",
            "repo_path": "./repo",
        },
    }

    for section, section_defaults in defaults.items():
        if section not in config:
            config[section] = {}
        for key, default_value in section_defaults.items():
            if key not in config[section]:
                config[section][key] = default_value

    return config


def setup_logging(debug: bool = False) -> None:
    """Configure structured logging."""
    level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("docker").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


# =============================================================================
# ORCHESTRATOR CLASS
# =============================================================================


class Orchestrator:
    """
    Main orchestrator class that coordinates the AI agent development workflow.

    This class is responsible for:
    1. Initializing all system components
    2. Managing the main processing loop
    3. Coordinating between GitHub, LangGraph, and agent containers
    4. Handling state persistence and recovery
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the orchestrator with configuration.

        Args:
            config: Merged configuration dictionary
        """
        self.config = config
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._tasks: List[asyncio.Task] = []

        # Components (initialized in setup())
        self.github_client: Optional[GitHubClient] = None
        self.issue_manager: Optional[IssueManager] = None
        self.state_manager: Optional[StateManager] = None
        self.langchain_engine: Optional[LangChainEngine] = None
        self.workflow_engine: Optional[WorkflowEngine] = None
        self.agent_launcher: Optional[AgentLauncher] = None
        self.queue_manager: Optional[QueueManager] = None

    async def setup(self) -> None:
        """
        Initialize all orchestrator components.

        This must be called before run().
        """
        logger.info("Initializing orchestrator components...")

        # 1. GitHub Client
        github_config = self.config.get("github", {})
        self.github_client = GitHubClient(
            token=github_config.get("token", ""),
            repo=github_config.get("repo", ""),
        )
        logger.info("GitHub client initialized")

        # 2. Issue Manager
        self.issue_manager = create_issue_manager(
            self.github_client,
            config=github_config,
        )

        # Ensure workflow labels exist
        try:
            await self.issue_manager.ensure_labels_exist()
        except Exception as e:
            logger.warning(f"Could not ensure labels: {e}")

        logger.info("Issue manager initialized")

        # 3. State Manager
        state_config = self.config.get("state", {})
        self.state_manager = await StateManager.create(state_config)
        logger.info("State manager initialized")

        # 4. LangChain Engine
        llm_config = self.config.get("llm", {})
        self.langchain_engine = create_langchain_engine(llm_config)
        logger.info("LangChain engine initialized")

        # 5. Agent Launcher
        agent_config = self.config.get("agent", {})
        self.agent_launcher = AgentLauncher(config=agent_config)
        logger.info("Agent launcher initialized")

        # 6. Workflow Engine
        workflow_config = {
            "max_iterations": self.config.get("orchestrator", {}).get("max_iterations", 5),
            "agent_timeout": agent_config.get("timeout", 3600),
        }
        self.workflow_engine = create_workflow_engine(
            langchain_engine=self.langchain_engine,
            state_manager=self.state_manager,
            agent_launcher=self.agent_launcher,
            github_client=self.github_client,
            config=workflow_config,
        )
        logger.info("Workflow engine initialized")

        # 7. Queue Manager
        queue_config = self.config.get("queue", {})
        self.queue_manager = await create_queue_manager(
            github_client=self.github_client,
            config=queue_config,
        )
        logger.info("Queue manager initialized")

        logger.info("All orchestrator components initialized successfully")

    async def run(self) -> None:
        """
        Main orchestrator loop.

        Runs continuously until stop() is called.
        """
        self._running = True
        mode = self.config.get("orchestrator", {}).get("mode", "polling")
        poll_interval = self.config.get("orchestrator", {}).get("poll_interval", 60)

        logger.info(f"Starting orchestrator in {mode} mode")

        # Recover any in-progress issues from previous run
        await self._recover_state()

        try:
            while self._running:
                try:
                    # Refresh the queue from GitHub
                    added = await self.queue_manager.refresh_queue()
                    if added > 0:
                        logger.info(f"Added {added} new issues to queue")

                    # Process available issues
                    await self._process_queue()

                    # Log status
                    stats = await self.queue_manager.get_stats()
                    logger.debug(
                        f"Queue: {stats.pending_items} pending, "
                        f"{stats.processing_items} processing, "
                        f"{stats.completed_items} completed"
                    )

                except WorkflowError as e:
                    logger.error(f"Workflow error: {e}")
                except Exception as e:
                    logger.error(f"Error in main loop: {e}", exc_info=True)

                # Wait for next poll or shutdown
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=poll_interval
                    )
                    # If we get here, shutdown was requested
                    break
                except asyncio.TimeoutError:
                    # Normal timeout, continue loop
                    continue

        finally:
            await self._cleanup()

        logger.info("Orchestrator stopped")

    async def process_issue(self, issue_number: int) -> Dict[str, Any]:
        """
        Process a single GitHub issue through the workflow.

        Args:
            issue_number: The GitHub issue number to process

        Returns:
            Dict with result, state, and output
        """
        logger.info(f"Processing issue #{issue_number}")

        try:
            # Run the workflow
            final_state = await self.workflow_engine.run(issue_number)

            # Determine result
            issue_state = final_state.get("issue_state", "")
            if issue_state == IssueState.DONE.value:
                result = "completed"
            elif issue_state == IssueState.BLOCKED.value:
                result = "blocked"
            else:
                result = "in_progress"

            # Update queue status
            if result == "completed":
                await self.queue_manager.mark_complete(issue_number)
            elif result == "blocked":
                error = final_state.get("error_message", "Unknown")
                await self.queue_manager.mark_blocked(issue_number, error)

            return {
                "result": result,
                "state": issue_state,
                "output": final_state.get("last_agent_output"),
                "iterations": final_state.get("iteration_count", 0),
            }

        except WorkflowError as e:
            logger.error(f"Workflow error for issue #{issue_number}: {e}")
            await self.queue_manager.mark_failed(issue_number)
            return {
                "result": "error",
                "state": "ERROR",
                "output": {"error": str(e)},
            }

    async def stop(self) -> None:
        """Gracefully stop the orchestrator."""
        logger.info("Stopping orchestrator...")
        self._running = False
        self._shutdown_event.set()

        # Cancel any running tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    async def _process_queue(self) -> None:
        """Process all available issues from the queue."""
        max_concurrent = self.config.get("agent", {}).get("max_concurrent", 3)

        while True:
            # Check if we can process more
            current = self.queue_manager.processing_count
            if current >= max_concurrent:
                logger.debug(f"Max concurrent reached ({current}/{max_concurrent})")
                break

            # Get next issue
            issue_data = await self.queue_manager.get_next()
            if issue_data is None:
                break  # No more issues

            issue_number = issue_data.get("number")
            if issue_number is None:
                continue

            # Process issue in background
            task = asyncio.create_task(
                self._process_issue_task(issue_number),
                name=f"issue-{issue_number}",
            )
            self._tasks.append(task)

            # Clean up completed tasks
            self._tasks = [t for t in self._tasks if not t.done()]

    async def _process_issue_task(self, issue_number: int) -> None:
        """Background task to process a single issue."""
        try:
            result = await self.process_issue(issue_number)
            logger.info(
                f"Issue #{issue_number} processing result: {result.get('result')}"
            )
        except Exception as e:
            logger.error(f"Failed to process issue #{issue_number}: {e}")
            try:
                await self.queue_manager.mark_failed(issue_number)
            except Exception:
                pass

    async def _recover_state(self) -> None:
        """Recover state from previous run."""
        logger.info("Recovering state from previous run...")

        try:
            # Find orphaned in-progress issues
            recovered = await self.issue_manager.recover_orphaned_issues()
            if recovered:
                logger.info(f"Recovered {len(recovered)} orphaned issues: {recovered}")
            else:
                logger.info("No orphaned issues found")

        except Exception as e:
            logger.error(f"State recovery failed: {e}")

    async def _cleanup(self) -> None:
        """Cleanup resources on shutdown."""
        logger.info("Cleaning up resources...")

        try:
            if self.queue_manager:
                await self.queue_manager.close()
        except Exception as e:
            logger.warning(f"Queue manager cleanup failed: {e}")

        try:
            if self.state_manager:
                await self.state_manager.close()
        except Exception as e:
            logger.warning(f"State manager cleanup failed: {e}")

        logger.info("Cleanup complete")

    # =========================================================================
    # STATUS AND MONITORING
    # =========================================================================

    async def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status."""
        queue_stats = await self.queue_manager.get_stats() if self.queue_manager else None
        workflow_summary = (
            await self.issue_manager.get_workflow_summary()
            if self.issue_manager else None
        )

        return {
            "running": self._running,
            "active_tasks": len([t for t in self._tasks if not t.done()]),
            "queue": {
                "pending": queue_stats.pending_items if queue_stats else 0,
                "processing": queue_stats.processing_items if queue_stats else 0,
                "completed": queue_stats.completed_items if queue_stats else 0,
                "failed": queue_stats.failed_items if queue_stats else 0,
            },
            "workflow": workflow_summary or {},
            "config": {
                "mode": self.config.get("orchestrator", {}).get("mode"),
                "max_concurrent": self.config.get("agent", {}).get("max_concurrent"),
                "max_iterations": self.config.get("orchestrator", {}).get("max_iterations"),
            },
        }


# =============================================================================
# CLI ARGUMENT PARSING
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI Agent Development System - Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        default="config/orchestrator.yaml",
        help="Path to configuration file (default: config/orchestrator.yaml)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without making real changes",
    )

    return parser.parse_args()


# =============================================================================
# SIGNAL HANDLING
# =============================================================================


def setup_signal_handlers(orchestrator: Orchestrator, loop: asyncio.AbstractEventLoop) -> None:
    """Setup signal handlers for graceful shutdown."""

    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, initiating shutdown...")
        loop.create_task(orchestrator.stop())

    # Only set signal handlers if running on Unix-like systems
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


async def async_main(config_path: str, debug: bool = False) -> None:
    """Async entry point for the orchestrator."""
    # Load configuration
    config = load_config(config_path)

    # Create and setup orchestrator
    orchestrator = Orchestrator(config)

    # Setup signal handlers
    loop = asyncio.get_running_loop()
    setup_signal_handlers(orchestrator, loop)

    try:
        # Initialize components
        await orchestrator.setup()

        # Run the main loop
        await orchestrator.run()

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        await orchestrator.stop()
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        await orchestrator.stop()
        raise


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    setup_logging(debug=args.debug)

    logger.info("=" * 60)
    logger.info("AI Agent Development System - Orchestrator")
    logger.info("=" * 60)

    # Run the async main
    try:
        asyncio.run(async_main(args.config, debug=args.debug))
    except KeyboardInterrupt:
        logger.info("Orchestrator stopped by user")
    except Exception as e:
        logger.critical(f"Orchestrator failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()