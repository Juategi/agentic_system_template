# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - AGENT LAUNCHER
# =============================================================================
"""
Agent Launcher

Manages Docker container lifecycle for agent execution.
Launches containers, monitors their progress, and collects results.

Responsibilities:
    - Build container configuration
    - Launch agent containers
    - Monitor container status
    - Collect container output
    - Clean up completed containers

Container Configuration:
    All agents share the same Docker image (ai-agent).
    Behavior is controlled by environment variables:
    - AGENT_TYPE: planner, developer, qa, reviewer, doc
    - ISSUE_NUMBER: GitHub issue to work on
    - PROJECT_ID: Project identifier
    - MAX_ITERATIONS: Iteration limit
"""

import os
import json
import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import docker
from docker.errors import DockerException, NotFound, APIError


logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class AgentType(Enum):
    """Available agent types."""
    PLANNER = "planner"
    DEVELOPER = "developer"
    QA = "qa"
    REVIEWER = "reviewer"
    DOC = "doc"


class ContainerStatus(Enum):
    """Container execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    NOT_FOUND = "not_found"


@dataclass
class AgentResult:
    """Result from agent execution."""
    status: ContainerStatus
    exit_code: Optional[int] = None
    output: Optional[Dict[str, Any]] = None
    logs: str = ""
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    duration_seconds: float = 0.0

    @property
    def success(self) -> bool:
        """Check if agent completed successfully."""
        return self.status == ContainerStatus.COMPLETED and self.exit_code == 0


@dataclass
class ContainerInfo:
    """Information about a running container."""
    container_id: str
    agent_type: str
    issue_number: int
    started_at: datetime
    status: ContainerStatus = ContainerStatus.RUNNING

    @property
    def running_seconds(self) -> float:
        """Get how long the container has been running."""
        return (datetime.utcnow() - self.started_at).total_seconds()


# =============================================================================
# EXCEPTIONS
# =============================================================================

class AgentLauncherError(Exception):
    """Base exception for agent launcher errors."""
    pass


class ContainerLaunchError(AgentLauncherError):
    """Failed to launch container."""
    pass


class ContainerTimeoutError(AgentLauncherError):
    """Container execution timed out."""
    pass


class ImageNotFoundError(AgentLauncherError):
    """Docker image not found."""
    pass


# =============================================================================
# AGENT LAUNCHER CLASS
# =============================================================================

class AgentLauncher:
    """
    Manages agent container lifecycle.

    Attributes:
        docker_client: Docker API client
        image_name: Agent Docker image name
        network: Docker network for containers
        config: Launcher configuration

    Configuration:
        - docker_agent_image: Image name (default: ai-agent:latest)
        - docker_network: Network name
        - cpu_limit: CPU limit per container
        - memory_limit: Memory limit per container
        - timeout: Execution timeout
    """

    DEFAULT_CONFIG = {
        "docker_image": "ai-agent:latest",
        "docker_network": "ai-agent-network",
        "cpu_limit": 2.0,
        "memory_limit": "4g",
        "default_timeout": 1800,  # 30 minutes
        "memory_path": "/memory",
        "repo_path": "/repo",
        "output_path": "/output",
        "input_path": "/input",
        "auto_remove": False,  # Keep for debugging
        "pull_image": False,  # Don't auto-pull
    }

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize agent launcher.

        Args:
            config: Configuration dictionary

        Initialization:
        1. Connect to Docker daemon
        2. Verify image exists
        3. Ensure network exists
        """
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self._docker: Optional[docker.DockerClient] = None
        self._container_pool = ContainerPool(
            max_concurrent=self.config.get("max_concurrent", 3)
        )

        # Paths for volume mounts (host paths for Docker API)
        self._host_memory_path = self.config.get("host_memory_path", "./memory")
        self._host_repo_path = self.config.get("host_repo_path", "./repo")
        self._host_output_path = self.config.get("host_output_path", "./output")

        # Local paths inside orchestrator container (for creating dirs/files)
        self._local_output_path = self.config.get("local_output_path", "/output")

        logger.info("AgentLauncher initialized")

    @property
    def docker(self) -> docker.DockerClient:
        """Lazy-load Docker client."""
        if self._docker is None:
            try:
                self._docker = docker.from_env()
                self._docker.ping()
                logger.info("Connected to Docker daemon")
            except DockerException as e:
                raise AgentLauncherError(f"Failed to connect to Docker: {e}")
        return self._docker

    def verify_setup(self) -> Dict[str, Any]:
        """
        Verify Docker setup is correct.

        Returns:
            Dict with verification results
        """
        results = {
            "docker_connected": False,
            "image_exists": False,
            "network_exists": False,
            "errors": [],
        }

        # Check Docker connection
        try:
            self.docker.ping()
            results["docker_connected"] = True
        except Exception as e:
            results["errors"].append(f"Docker connection failed: {e}")
            return results

        # Check image exists
        image_name = self.config["docker_image"]
        try:
            self.docker.images.get(image_name)
            results["image_exists"] = True
        except NotFound:
            results["errors"].append(f"Image not found: {image_name}")
            if self.config.get("pull_image"):
                try:
                    logger.info(f"Pulling image {image_name}...")
                    self.docker.images.pull(image_name)
                    results["image_exists"] = True
                except Exception as e:
                    results["errors"].append(f"Failed to pull image: {e}")

        # Check/create network
        network_name = self.config["docker_network"]
        try:
            self.docker.networks.get(network_name)
            results["network_exists"] = True
        except NotFound:
            try:
                self.docker.networks.create(network_name, driver="bridge")
                results["network_exists"] = True
                logger.info(f"Created network: {network_name}")
            except Exception as e:
                results["errors"].append(f"Failed to create network: {e}")

        return results

    async def launch_agent(
        self,
        agent_type: str,
        issue_number: int,
        context: Dict[str, Any] = None,
        timeout: int = None,
    ) -> str:
        """
        Launch an agent container.

        Args:
            agent_type: Type of agent (planner, developer, etc.)
            issue_number: GitHub issue to work on
            context: Additional context to pass to agent
            timeout: Execution timeout in seconds

        Returns:
            Container ID

        Steps:
        1. Build environment variables
        2. Write input file (if context provided)
        3. Create container with volumes
        4. Start container
        5. Return container ID for monitoring
        """
        # Validate agent type
        try:
            AgentType(agent_type)
        except ValueError:
            raise ContainerLaunchError(
                f"Invalid agent type: {agent_type}. "
                f"Valid types: {[t.value for t in AgentType]}"
            )

        # Check pool capacity
        if not self._container_pool.can_launch():
            raise ContainerLaunchError(
                f"Maximum concurrent containers reached "
                f"({self._container_pool.max_concurrent})"
            )

        # Build configuration
        environment = self._build_environment(agent_type, issue_number, timeout)
        volumes = self._build_volumes(issue_number)

        # Write input context if provided
        if context:
            input_file = self._write_input_file(issue_number, context)
            environment["INPUT_FILE"] = f"{self.config['input_path']}/input.json"

        # Create container
        container_name = f"agent-{agent_type}-{issue_number}-{int(datetime.utcnow().timestamp())}"

        try:
            container = self.docker.containers.create(
                image=self.config["docker_image"],
                name=container_name,
                environment=environment,
                volumes=volumes,
                network=self.config["docker_network"],
                cpu_period=100000,
                cpu_quota=int(self.config["cpu_limit"] * 100000),
                mem_limit=self.config["memory_limit"],
                detach=True,
                auto_remove=self.config["auto_remove"],
            )

            # Start container
            container.start()

            # Register in pool
            info = ContainerInfo(
                container_id=container.id,
                agent_type=agent_type,
                issue_number=issue_number,
                started_at=datetime.utcnow(),
            )
            self._container_pool.register(container.id, info)

            logger.info(
                f"Launched {agent_type} agent for issue #{issue_number} "
                f"(container: {container.short_id})"
            )

            return container.id

        except APIError as e:
            raise ContainerLaunchError(f"Failed to create container: {e}")
        except Exception as e:
            raise ContainerLaunchError(f"Unexpected error launching agent: {e}")

    async def wait_for_completion(
        self,
        container_id: str,
        timeout: int = None,
        poll_interval: float = 2.0,
    ) -> AgentResult:
        """
        Wait for agent container to complete.

        Args:
            container_id: Container to monitor
            timeout: Maximum wait time in seconds
            poll_interval: Seconds between status checks

        Returns:
            AgentResult with status, output, and logs
        """
        timeout = timeout or self.config["default_timeout"]
        start_time = datetime.utcnow()

        try:
            container = self.docker.containers.get(container_id)
        except NotFound:
            return AgentResult(
                status=ContainerStatus.NOT_FOUND,
                error_message=f"Container not found: {container_id}",
            )

        # Poll until completion or timeout
        while True:
            container.reload()
            status = container.status

            if status == "exited":
                # Container finished
                exit_code = container.attrs["State"]["ExitCode"]
                finished_at = datetime.utcnow()

                result = AgentResult(
                    status=ContainerStatus.COMPLETED if exit_code == 0 else ContainerStatus.FAILED,
                    exit_code=exit_code,
                    started_at=start_time,
                    finished_at=finished_at,
                    duration_seconds=(finished_at - start_time).total_seconds(),
                )

                # Collect logs
                result.logs = self._collect_logs(container)

                # Parse output
                result.output = self._parse_agent_output(container_id)

                if exit_code != 0:
                    result.error_message = f"Agent exited with code {exit_code}"

                # Cleanup
                self._container_pool.unregister(container_id)

                logger.info(
                    f"Container {container.short_id} completed with exit code {exit_code}"
                )

                return result

            # Check timeout
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            if elapsed >= timeout:
                logger.warning(f"Container {container.short_id} timed out after {timeout}s")

                # Stop the container
                await self.stop_agent(container_id)

                return AgentResult(
                    status=ContainerStatus.TIMEOUT,
                    started_at=start_time,
                    finished_at=datetime.utcnow(),
                    duration_seconds=elapsed,
                    error_message=f"Execution timed out after {timeout} seconds",
                    logs=self._collect_logs(container),
                )

            # Wait before next poll
            await asyncio.sleep(poll_interval)

    async def get_agent_status(self, container_id: str) -> Dict[str, Any]:
        """
        Get current status of agent container.

        Returns:
            {
                "status": "running" | "exited" | "not_found",
                "exit_code": None | int,
                "started_at": "...",
                "running_for": seconds
            }
        """
        try:
            container = self.docker.containers.get(container_id)
            container.reload()

            state = container.attrs["State"]
            started_at = state.get("StartedAt", "")

            return {
                "status": container.status,
                "exit_code": state.get("ExitCode") if container.status == "exited" else None,
                "started_at": started_at,
                "running_for": self._container_pool.get_running_time(container_id),
            }

        except NotFound:
            return {
                "status": "not_found",
                "exit_code": None,
                "started_at": None,
                "running_for": None,
            }

    async def stop_agent(self, container_id: str, timeout: int = 10):
        """
        Stop a running agent container.

        Args:
            container_id: Container to stop
            timeout: Grace period before kill
        """
        try:
            container = self.docker.containers.get(container_id)
            container.stop(timeout=timeout)
            self._container_pool.unregister(container_id)
            logger.info(f"Stopped container {container.short_id}")
        except NotFound:
            logger.warning(f"Container {container_id} not found for stopping")
        except Exception as e:
            logger.error(f"Error stopping container {container_id}: {e}")

    async def cleanup_container(self, container_id: str):
        """
        Remove a container and clean up resources.

        Called after collecting results from completed agent.
        """
        try:
            container = self.docker.containers.get(container_id)
            container.remove(force=True)
            self._container_pool.unregister(container_id)
            logger.info(f"Removed container {container.short_id}")

            # Clean up input/output files
            self._cleanup_files(container_id)

        except NotFound:
            pass  # Already removed
        except Exception as e:
            logger.error(f"Error removing container {container_id}: {e}")

    def _build_environment(
        self,
        agent_type: str,
        issue_number: int,
        timeout: int = None,
    ) -> Dict[str, str]:
        """Build environment variables for agent container."""
        env = {
            "AGENT_TYPE": agent_type,
            "ISSUE_NUMBER": str(issue_number),
            "PROJECT_ID": os.environ.get("PROJECT_ID", "default"),
            "MEMORY_PATH": self.config["memory_path"],
            "REPO_PATH": self.config["repo_path"],
            "OUTPUT_PATH": self.config["output_path"],
            "MAX_ITERATIONS": os.environ.get("AGENT_MAX_ITERATIONS", "5"),
        }

        # Pass through LLM configuration
        for key in ["LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
            if os.environ.get(key):
                env[key] = os.environ[key]

        # Pass through GitHub configuration
        for key in ["GITHUB_TOKEN", "GITHUB_REPO"]:
            if os.environ.get(key):
                env[key] = os.environ[key]

        if timeout:
            env["TIMEOUT_SECONDS"] = str(timeout)

        return env

    def _build_volumes(self, issue_number: int) -> Dict[str, Dict[str, str]]:
        """
        Build volume configuration for agent container.

        Mounts:
        - /memory: Project memory (read)
        - /repo: Code repository (read/write)
        - /output: Agent output (write)
        - /input: Orchestrator input (read)
        """
        # Create issue-specific output directory inside orchestrator container
        # (the /output mount is shared with the host via docker-compose)
        local_output = Path(self._local_output_path) / str(issue_number)
        local_output.mkdir(parents=True, exist_ok=True)
        (local_output / "logs").mkdir(parents=True, exist_ok=True)
        (local_output / "artifacts").mkdir(parents=True, exist_ok=True)

        local_input = local_output / "input"
        local_input.mkdir(parents=True, exist_ok=True)

        # Use HOST paths for Docker volume mounts (agent containers are siblings)
        host_output = Path(self._host_output_path) / str(issue_number)
        host_input = host_output / "input"

        return {
            str(Path(self._host_memory_path).absolute()): {
                "bind": self.config["memory_path"],
                "mode": "ro",  # Read-only for safety
            },
            str(Path(self._host_repo_path).absolute()): {
                "bind": self.config["repo_path"],
                "mode": "rw",
            },
            str(host_output.absolute()): {
                "bind": self.config["output_path"],
                "mode": "rw",
            },
            str(host_input.absolute()): {
                "bind": self.config["input_path"],
                "mode": "ro",
            },
        }

    def _write_input_file(
        self,
        issue_number: int,
        context: Dict[str, Any],
    ) -> Path:
        """Write input context for agent."""
        input_dir = Path(self._local_output_path) / str(issue_number) / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        input_file = input_dir / "input.json"
        input_file.write_text(json.dumps(context, indent=2), encoding="utf-8")

        logger.debug(f"Wrote input file: {input_file}")
        return input_file

    def _parse_agent_output(self, container_id: str) -> Optional[Dict[str, Any]]:
        """
        Parse output from completed agent.

        Reads result.json from output volume.
        """
        # Get issue number from pool
        info = self._container_pool.get_info(container_id)
        if not info:
            return None

        output_file = (
            Path(self._local_output_path) /
            str(info.issue_number) /
            "result.json"
        )

        if output_file.exists():
            try:
                return json.loads(output_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse agent output: {e}")
                return None

        logger.warning(f"Output file not found: {output_file}")
        return None

    def _collect_logs(self, container) -> str:
        """Collect container logs for debugging."""
        try:
            logs = container.logs(stdout=True, stderr=True, timestamps=True)
            return logs.decode("utf-8", errors="replace")
        except Exception as e:
            logger.error(f"Failed to collect logs: {e}")
            return ""

    def _cleanup_files(self, container_id: str):
        """Clean up input/output files for a container."""
        info = self._container_pool.get_info(container_id)
        if not info:
            return

        # Optionally clean up files - keeping them for now for debugging
        # issue_dir = Path(self._host_output_path) / str(info.issue_number)
        # shutil.rmtree(issue_dir, ignore_errors=True)

    def list_running_agents(self) -> List[ContainerInfo]:
        """List all currently running agent containers."""
        return list(self._container_pool.running.values())

    def get_pool_status(self) -> Dict[str, Any]:
        """Get status of the container pool."""
        return {
            "running_count": len(self._container_pool.running),
            "max_concurrent": self._container_pool.max_concurrent,
            "can_launch": self._container_pool.can_launch(),
            "containers": [
                {
                    "container_id": info.container_id[:12],
                    "agent_type": info.agent_type,
                    "issue_number": info.issue_number,
                    "running_seconds": info.running_seconds,
                }
                for info in self._container_pool.running.values()
            ],
        }

    def close(self):
        """Close Docker client connection."""
        if self._docker:
            self._docker.close()
            self._docker = None
            logger.debug("AgentLauncher closed")


# =============================================================================
# CONTAINER POOL
# =============================================================================

class ContainerPool:
    """
    Manages pool of running agent containers.

    Tracks:
    - Currently running containers
    - Container to issue mapping
    - Resource utilization

    Enforces:
    - Maximum concurrent containers
    - Resource limits
    """

    def __init__(self, max_concurrent: int = 3):
        self.max_concurrent = max_concurrent
        self.running: Dict[str, ContainerInfo] = {}
        self._issue_to_container: Dict[int, str] = {}

    def can_launch(self) -> bool:
        """Check if another container can be launched."""
        return len(self.running) < self.max_concurrent

    def register(self, container_id: str, info: ContainerInfo):
        """Register a newly launched container."""
        self.running[container_id] = info
        self._issue_to_container[info.issue_number] = container_id
        logger.debug(f"Registered container {container_id[:12]} in pool")

    def unregister(self, container_id: str):
        """Remove a completed container from pool."""
        if container_id in self.running:
            info = self.running.pop(container_id)
            self._issue_to_container.pop(info.issue_number, None)
            logger.debug(f"Unregistered container {container_id[:12]} from pool")

    def get_by_issue(self, issue_number: int) -> Optional[str]:
        """Get container ID for an issue."""
        return self._issue_to_container.get(issue_number)

    def get_info(self, container_id: str) -> Optional[ContainerInfo]:
        """Get info for a container."""
        return self.running.get(container_id)

    def get_running_time(self, container_id: str) -> Optional[float]:
        """Get how long a container has been running."""
        info = self.running.get(container_id)
        return info.running_seconds if info else None

    def is_issue_running(self, issue_number: int) -> bool:
        """Check if an issue already has a running container."""
        return issue_number in self._issue_to_container