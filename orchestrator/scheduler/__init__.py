# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - SCHEDULER PACKAGE
# =============================================================================
"""
Scheduler Package

This package handles agent scheduling and container management.

Components:
    - QueueManager: Manages the issue processing queue
    - AgentLauncher: Launches and monitors agent containers

The scheduler ensures:
    - Concurrency limits are respected
    - Agents are launched with correct configuration
    - Container lifecycle is properly managed
    - Results are collected from completed agents

Usage:
    from orchestrator.scheduler import AgentLauncher, AgentType

    launcher = AgentLauncher(config={
        "docker_image": "ai-agent:latest",
        "max_concurrent": 3,
    })

    container_id = await launcher.launch_agent(
        agent_type="developer",
        issue_number=123,
        context={"qa_feedback": {...}},
    )

    result = await launcher.wait_for_completion(container_id)
    if result.success:
        print(result.output)
"""

from orchestrator.scheduler.agent_launcher import (
    # Main class
    AgentLauncher,
    ContainerPool,
    # Data structures
    AgentType,
    ContainerStatus,
    AgentResult,
    ContainerInfo,
    # Exceptions
    AgentLauncherError,
    ContainerLaunchError,
    ContainerTimeoutError,
    ImageNotFoundError,
)

__all__ = [
    # Main classes
    "AgentLauncher",
    "ContainerPool",
    # Enums
    "AgentType",
    "ContainerStatus",
    # Data classes
    "AgentResult",
    "ContainerInfo",
    # Exceptions
    "AgentLauncherError",
    "ContainerLaunchError",
    "ContainerTimeoutError",
    "ImageNotFoundError",
    # To be implemented
    "QueueManager",
]