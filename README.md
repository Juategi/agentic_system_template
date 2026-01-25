# AI Agent Development System - Template Repository

## Overview

This repository is a **template infrastructure** for building autonomous AI-driven software development systems. It is NOT a final product but a reusable foundation that enables AI agents to develop software projects autonomously.

The system allows you to:
- Define complex software features
- Manage them through GitHub Issues
- Automatically decompose features into smaller tasks
- Execute specialized AI agents autonomously
- Iterate automatically until tasks pass QA
- Maintain persistent project memory
- Replicate in minutes for new projects
- Run locally or in the cloud
- Scale to multiple independent projects

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AI SOFTWARE FACTORY                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌─────────────────────────────────────────────────┐   │
│  │   GitHub    │────▶│              ORCHESTRATOR (24/7)                │   │
│  │   Issues    │◀────│         LangChain + LangGraph Engine            │   │
│  │  (Backlog)  │     │                                                 │   │
│  └─────────────┘     │  ┌─────────────────────────────────────────┐   │   │
│                      │  │           STATE MACHINE                  │   │   │
│                      │  │  PLANNING → DEV → QA → REVIEW → DONE    │   │   │
│                      │  └─────────────────────────────────────────┘   │   │
│                      └──────────────────┬──────────────────────────────┘   │
│                                         │                                   │
│                    ┌────────────────────┼────────────────────┐             │
│                    ▼                    ▼                    ▼             │
│            ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│            │   PLANNER    │    │  DEVELOPER   │    │      QA      │       │
│            │    Agent     │    │    Agent     │    │    Agent     │       │
│            └──────────────┘    └──────────────┘    └──────────────┘       │
│            ┌──────────────┐    ┌──────────────┐                           │
│            │   REVIEWER   │    │     DOC      │                           │
│            │    Agent     │    │    Agent     │                           │
│            └──────────────┘    └──────────────┘                           │
│                    │                    │                    │             │
│                    └────────────────────┼────────────────────┘             │
│                                         ▼                                   │
│                              ┌─────────────────────┐                       │
│                              │   SHARED DOCKER     │                       │
│                              │   IMAGE (ai-agent)  │                       │
│                              └─────────────────────┘                       │
│                                         │                                   │
│                    ┌────────────────────┼────────────────────┐             │
│                    ▼                    ▼                    ▼             │
│            ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│            │    MEMORY    │    │  REPOSITORY  │    │   METRICS    │       │
│            │  (Markdown)  │    │    (Code)    │    │   & LOGS     │       │
│            └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Repository Structure

```
agentic_system_template/
├── .env.template                    # Environment variables template
├── .gitignore                       # Git ignore rules
├── CLAUDE.md                        # Instructions for AI assistants
├── README.md                        # This file
├── docker-compose.yml               # Docker orchestration
├── Makefile                         # Common commands
│
├── config/                          # System configuration
│   ├── agents.yaml                  # Agent definitions and behaviors
│   ├── orchestrator.yaml            # Orchestrator settings
│   ├── github.yaml                  # GitHub integration settings
│   └── monitoring.yaml              # Metrics and logging config
│
├── orchestrator/                    # Central orchestration system
│   ├── __init__.py
│   ├── main.py                      # Entry point
│   ├── engine/                      # Core engine
│   │   ├── __init__.py
│   │   ├── langchain_setup.py       # LangChain configuration
│   │   ├── langgraph_workflow.py    # LangGraph state machine
│   │   └── state_manager.py         # Persistent state management
│   ├── nodes/                       # LangGraph nodes
│   │   ├── __init__.py
│   │   ├── planning_node.py
│   │   ├── development_node.py
│   │   ├── qa_node.py
│   │   ├── review_node.py
│   │   └── documentation_node.py
│   ├── github/                      # GitHub integration
│   │   ├── __init__.py
│   │   ├── client.py                # GitHub API client
│   │   ├── webhook_handler.py       # Webhook processing
│   │   └── issue_manager.py         # Issue operations
│   └── scheduler/                   # Task scheduling
│       ├── __init__.py
│       ├── queue_manager.py
│       └── agent_launcher.py
│
├── agents/                          # Agent definitions
│   ├── __init__.py
│   ├── base/                        # Shared agent infrastructure
│   │   ├── __init__.py
│   │   ├── agent_interface.py       # Common interface
│   │   ├── context_loader.py        # Context loading
│   │   └── output_handler.py        # Result handling
│   ├── planner/                     # Planner agent
│   │   ├── __init__.py
│   │   ├── planner_agent.py
│   │   └── prompts/
│   ├── developer/                   # Developer agent
│   │   ├── __init__.py
│   │   ├── developer_agent.py
│   │   └── prompts/
│   ├── qa/                          # QA agent
│   │   ├── __init__.py
│   │   ├── qa_agent.py
│   │   └── prompts/
│   ├── reviewer/                    # Reviewer agent
│   │   ├── __init__.py
│   │   ├── reviewer_agent.py
│   │   └── prompts/
│   └── doc/                         # Documentation agent
│       ├── __init__.py
│       ├── doc_agent.py
│       └── prompts/
│
├── memory/                          # Persistent project memory
│   ├── PROJECT.md                   # Global project context
│   ├── ARCHITECTURE.md              # Technical architecture
│   ├── CONVENTIONS.md               # Coding conventions
│   ├── CONSTRAINTS.md               # Technical constraints
│   └── features/                    # Feature-specific memory
│       └── _TEMPLATE.md             # Feature file template
│
├── docker/                          # Docker infrastructure
│   ├── Dockerfile.agent             # Unified agent image
│   ├── Dockerfile.orchestrator      # Orchestrator image
│   └── entrypoint.sh                # Agent entrypoint
│
├── scripts/                         # Utility scripts
│   ├── init_project.sh              # Initialize new project
│   ├── setup_github.sh              # Configure GitHub integration
│   ├── start_orchestrator.sh        # Start orchestrator
│   └── health_check.sh              # System health check
│
├── monitoring/                      # Monitoring infrastructure
│   ├── __init__.py
│   ├── metrics.py                   # Metrics collection
│   ├── logger.py                    # Structured logging
│   └── dashboard/                   # Optional dashboard
│
├── docs/                            # Extended documentation
│   ├── SETUP.md                     # Setup guide
│   ├── ARCHITECTURE.md              # Detailed architecture
│   ├── AGENTS.md                    # Agent specifications
│   ├── ORCHESTRATOR.md              # Orchestrator details
│   ├── GITHUB_INTEGRATION.md        # GitHub setup
│   ├── MEMORY_SYSTEM.md             # Memory system guide
│   ├── DEPLOYMENT.md                # Deployment guide
│   └── TROUBLESHOOTING.md           # Common issues
│
└── tests/                           # System tests
    ├── __init__.py
    ├── test_orchestrator.py
    ├── test_agents.py
    └── test_github_integration.py
```

## Quick Start

### 1. Clone and Initialize

```bash
# Clone this template
git clone https://github.com/your-org/agentic_system_template.git my-project

# Navigate to project
cd my-project

# Run initialization script
./scripts/init_project.sh
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.template .env

# Edit with your values
# - GITHUB_TOKEN
# - GITHUB_REPO
# - LLM_API_KEY
# - Other settings
```

### 3. Start the System

```bash
# Start orchestrator (development)
docker-compose up orchestrator

# Or start full system
docker-compose up
```

### 4. Create Your First Feature

Create a GitHub Issue with label `feature` and the system will automatically begin processing it.

## Core Concepts

### GitHub Issues as Task System

GitHub Issues serve as the **single source of truth** for all tasks:

| Label | Description |
|-------|-------------|
| `READY` | Task is ready for agent assignment |
| `IN_PROGRESS` | Agent is currently working on task |
| `QA` | Task is undergoing quality assurance |
| `QA_FAILED` | QA failed, returning to development |
| `REVIEW` | Task is being reviewed |
| `BLOCKED` | Task requires human intervention |
| `DONE` | Task completed successfully |

### Agents

All agents share a single Docker image and differentiate by environment variables:

| Agent | Role |
|-------|------|
| **Planner** | Decomposes features into tasks, creates issues |
| **Developer** | Implements code changes for issues |
| **QA** | Validates acceptance criteria, runs tests |
| **Reviewer** | Reviews code quality and alignment |
| **Doc** | Updates memory and documentation |

### Orchestrator

The orchestrator runs 24/7 using LangChain + LangGraph:

- **LangChain**: LLM interaction, prompts, tools
- **LangGraph**: State machine, workflow control, transitions

### Memory System

Persistent memory lives in Markdown files:

- `memory/PROJECT.md` - Global project context
- `memory/features/*.md` - Feature-specific context

## Multi-Project Support

Each project from this template is completely independent:
- Own repository
- Own GitHub Issues
- Own memory
- Own orchestrator instance
- No cross-project dependencies

## Deployment Options

### Local Development
```bash
docker-compose up
```

### Cloud Deployment
See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for:
- AWS/GCP/Azure configurations
- Kubernetes manifests
- Serverless options

## Documentation

- [Setup Guide](docs/SETUP.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Agent Specifications](docs/AGENTS.md)
- [Orchestrator Details](docs/ORCHESTRATOR.md)
- [GitHub Integration](docs/GITHUB_INTEGRATION.md)
- [Memory System](docs/MEMORY_SYSTEM.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## License

MIT License - See LICENSE file for details.

---

**This template is the foundation for an AI Software Factory.**
