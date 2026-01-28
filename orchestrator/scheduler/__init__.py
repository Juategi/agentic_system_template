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

from orchestrator.scheduler.queue_manager import (
    # Main class
    QueueManager,
    create_queue_manager,
    # Backend interface
    QueueBackendInterface,
    MemoryQueueBackend,
    RedisQueueBackend,
    # Data structures
    QueueItem,
    QueueStats,
    RateLimitConfig,
    # Enums
    Priority,
    QueueStatus,
    # Rate limiter
    RateLimiter,
    # Exceptions
    QueueError,
    QueueFullError,
    RateLimitError,
    DependencyError,
    # Constants
    PRIORITY_LABELS,
    REDIS_AVAILABLE,
)

__all__ = [
    # Agent Launcher
    "AgentLauncher",
    "ContainerPool",
    # Agent Enums
    "AgentType",
    "ContainerStatus",
    # Agent Data classes
    "AgentResult",
    "ContainerInfo",
    # Agent Exceptions
    "AgentLauncherError",
    "ContainerLaunchError",
    "ContainerTimeoutError",
    "ImageNotFoundError",
    # Queue Manager
    "QueueManager",
    "create_queue_manager",
    # Queue Backends
    "QueueBackendInterface",
    "MemoryQueueBackend",
    "RedisQueueBackend",
    # Queue Data structures
    "QueueItem",
    "QueueStats",
    "RateLimitConfig",
    # Queue Enums
    "Priority",
    "QueueStatus",
    # Rate Limiter
    "RateLimiter",
    # Queue Exceptions
    "QueueError",
    "QueueFullError",
    "RateLimitError",
    "DependencyError",
    # Queue Constants
    "PRIORITY_LABELS",
    "REDIS_AVAILABLE",
]