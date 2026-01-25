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
"""

__all__ = [
    "QueueManager",
    "AgentLauncher",
]
