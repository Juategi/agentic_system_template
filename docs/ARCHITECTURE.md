# Architecture Guide

## System Overview

The AI Agent Development System is a template infrastructure for autonomous AI-driven software development. It enables AI agents to develop software projects with minimal human intervention.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AI SOFTWARE FACTORY                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                           GITHUB                                     │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │   │
│  │  │   Issues    │  │   Labels    │  │  Comments   │                 │   │
│  │  │  (Backlog)  │  │  (States)   │  │ (Feedback)  │                 │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                 │   │
│  └─────────┼────────────────┼────────────────┼──────────────────────────┘   │
│            │                │                │                              │
│            └────────────────┼────────────────┘                              │
│                             │                                               │
│                             ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     ORCHESTRATOR (24/7)                              │   │
│  │                                                                      │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │                    LANGGRAPH STATE MACHINE                    │   │   │
│  │  │                                                               │   │   │
│  │  │   TRIAGE → PLANNING → DEVELOPMENT → QA → REVIEW → DOC → DONE │   │   │
│  │  │                           ↑          ↓                        │   │   │
│  │  │                           └── QA_FAILED                       │   │   │
│  │  │                                  ↓                            │   │   │
│  │  │                              BLOCKED                          │   │   │
│  │  │                                                               │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  │                                                                      │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │   │
│  │  │  LangChain  │  │   GitHub    │  │    State    │                 │   │
│  │  │   Engine    │  │   Client    │  │   Manager   │                 │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                 │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                             │                                               │
│                             │ Launches                                      │
│                             ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    AGENT CONTAINERS (Ephemeral)                      │   │
│  │                                                                      │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │ PLANNER  │  │DEVELOPER │  │    QA    │  │ REVIEWER │  │ DOC  │  │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──┬───┘  │   │
│  │       │             │             │             │            │       │   │
│  │       └─────────────┴─────────────┴─────────────┴────────────┘       │   │
│  │                             │                                        │   │
│  │                    ┌────────┴────────┐                              │   │
│  │                    │ SHARED DOCKER   │                              │   │
│  │                    │ IMAGE (ai-agent)│                              │   │
│  │                    └─────────────────┘                              │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                             │                                               │
│            ┌────────────────┼────────────────┐                             │
│            ▼                ▼                ▼                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                        │
│  │   MEMORY    │  │ REPOSITORY  │  │  METRICS    │                        │
│  │ (Markdown)  │  │   (Code)    │  │  & LOGS     │                        │
│  └─────────────┘  └─────────────┘  └─────────────┘                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. GitHub (Task Management)

GitHub Issues serve as the central task management system:
- **Issues** = Task backlog
- **Labels** = Workflow states
- **Comments** = Agent communication

### 2. Orchestrator

The orchestrator runs 24/7 and coordinates all activity:

- **LangChain Engine**: LLM interactions and tool management
- **LangGraph Workflow**: State machine for issue lifecycle
- **GitHub Client**: API integration for issue operations
- **State Manager**: Persistent state storage
- **Agent Launcher**: Container lifecycle management

### 3. Agent Containers

All agents share a single Docker image differentiated by environment variables:

| Agent | Responsibility |
|-------|----------------|
| Planner | Decompose features into tasks |
| Developer | Implement code changes |
| QA | Validate implementations |
| Reviewer | Review code quality |
| Doc | Update documentation |

### 4. Persistent Storage

- **Memory (Markdown)**: Project context and feature files
- **Repository**: Code being developed
- **State**: Workflow state persistence

## Workflow States

```
READY → IN_PROGRESS → QA → REVIEW → DONE
              ↑         ↓
              └── QA_FAILED
                      ↓
                  BLOCKED (max iterations)
```

## Data Flow

### Issue Processing Flow

1. **Issue Created**: Human creates GitHub Issue with feature/task label
2. **Triage**: Orchestrator detects new issue, determines type
3. **Planning** (features): Planner agent decomposes into tasks
4. **Development**: Developer agent implements code
5. **QA**: QA agent validates implementation
6. **Iteration**: If QA fails, return to development
7. **Review**: Reviewer agent checks quality
8. **Documentation**: Doc agent updates memory
9. **Completion**: Issue marked DONE

### Agent Execution Flow

1. Orchestrator determines next action
2. Prepares context (memory, input data)
3. Launches agent container
4. Agent reads context from volumes
5. Agent performs task using LLM
6. Agent writes output to volume
7. Container exits
8. Orchestrator processes results
9. State transitions accordingly

## Key Design Decisions

### Single Docker Image

All agents use one image to:
- Simplify maintenance
- Ensure consistency
- Reduce build times
- Enable easy updates

### Ephemeral Agents

Agents are stateless containers that:
- Start, execute, and terminate
- Read all context from volumes
- Write all output to volumes
- Don't maintain persistent connections

### Memory in Repository

Project memory lives in Markdown files:
- Version controlled with code
- Human-readable and editable
- Easy for agents to parse
- Portable across systems

### GitHub as Single Source of Truth

GitHub Issues provide:
- Central task tracking
- Human visibility
- Built-in notifications
- Integration capabilities

## Scaling Considerations

### Multi-Project

Each project is completely independent:
- Own repository
- Own orchestrator instance
- Own memory
- No cross-project dependencies

### Horizontal Scaling

For high volume:
- Run multiple orchestrator instances
- Use Redis for state coordination
- Implement work distribution

### Cloud Deployment

See [Deployment Guide](DEPLOYMENT.md) for:
- Kubernetes configurations
- Cloud provider options
- Auto-scaling strategies
