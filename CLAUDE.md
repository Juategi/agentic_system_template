# CLAUDE.md - AI Assistant Instructions

## Project Overview

This is an **AI Agent Development System Template** - a reusable infrastructure for autonomous AI-driven software development. This repository is NOT a final product but a foundation that enables AI agents to develop software projects autonomously.

## System Purpose

The system enables:
1. Complex software features defined via GitHub Issues
2. Automatic decomposition of features into smaller tasks
3. Autonomous execution by specialized AI agents
4. Automatic iteration until QA passes
5. Persistent project memory in Markdown
6. Replicability for new projects
7. Local or cloud execution
8. Multi-project scaling

## Architecture Summary

```
GitHub Issues (Backlog) ←→ Orchestrator (LangChain/LangGraph) → Agents (Docker)
                                    ↓
                              Memory (Markdown)
```

## Key Components

### 1. Orchestrator (`/orchestrator`)
- Runs 24/7 continuously
- Uses LangChain for LLM interactions
- Uses LangGraph as state machine
- Manages issue states and agent execution
- Controls iteration loops

### 2. Agents (`/agents`)
- All share single Docker image (`ai-agent`)
- Differentiate by `AGENT_TYPE` environment variable
- Types: Planner, Developer, QA, Reviewer, Doc
- Ephemeral: execute task and terminate

### 3. Memory (`/memory`)
- PROJECT.md: Global context
- ARCHITECTURE.md: Technical design
- CONVENTIONS.md: Coding standards
- CONSTRAINTS.md: Technical limits
- features/*.md: Feature-specific context

### 4. GitHub Integration (`/orchestrator/github`)
- Issues = Task backlog
- Labels = Task states
- Comments = Agent communication
- Webhooks/Polling for real-time updates

## GitHub Issue Labels

| Label | Meaning |
|-------|---------|
| `READY` | Available for assignment |
| `IN_PROGRESS` | Being worked on |
| `QA` | Under quality validation |
| `QA_FAILED` | Failed QA, needs rework |
| `REVIEW` | Under code review |
| `BLOCKED` | Needs human intervention |
| `DONE` | Successfully completed |

## State Machine Flow

```
[New Issue] → PLANNING → DEVELOPMENT → QA
                              ↑          ↓
                              └── QA_FAILED
                                         ↓
                                    (max iterations?)
                                         ↓
                                      BLOCKED

              QA_PASSED → REVIEW → DOCUMENTATION → DONE
```

## Development Guidelines

### When Modifying This Template

1. **Maintain Modularity**: Each component should be independent
2. **Environment-Driven**: All behavior via environment variables
3. **No Project-Specific Logic**: Keep template generic
4. **Document Everything**: Other LLMs must understand the system
5. **Audit Trail**: All decisions must be traceable

### File Naming Conventions

- Python: `snake_case.py`
- Config: `lowercase.yaml`
- Documentation: `UPPERCASE.md`
- Features: `feature-slug.md`

### Configuration Files

- `/config/agents.yaml`: Agent definitions
- `/config/orchestrator.yaml`: Orchestrator settings
- `/config/github.yaml`: GitHub integration
- `/config/monitoring.yaml`: Metrics and logging

## Agent Development

### Adding a New Agent Type

1. Create folder in `/agents/{agent_name}/`
2. Implement agent class extending base interface
3. Add prompts in `/agents/{agent_name}/prompts/`
4. Register in `/config/agents.yaml`
5. Add node in `/orchestrator/nodes/`
6. Update LangGraph workflow

### Agent Contract

Every agent must:
- Accept task via environment variables
- Read context from mounted volumes
- Write results to mounted volumes
- Update GitHub Issue status
- Exit cleanly after completion

## Memory System

### Feature Memory Template

```markdown
# Feature: [TITLE]

## Metadata
- Issue: #[NUMBER]
- Status: [STATE]
- Created: [DATE]
- Updated: [DATE]

## Description
[Feature description]

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2

## Technical Approach
[Implementation strategy]

## Dependencies
- [List of dependencies]

## History
| Date | Agent | Action | Result |
|------|-------|--------|--------|

## QA Feedback
[QA results and feedback]
```

## Docker Infrastructure

### Single Image Philosophy

All agents use ONE Docker image. Behavior is controlled by:

```bash
AGENT_TYPE=developer|planner|qa|reviewer|doc
PROJECT_ID=unique-project-id
ISSUE_NUMBER=123
MEMORY_PATH=/memory
REPO_PATH=/repo
MAX_ITERATIONS=3
```

### Volume Mounts

| Mount | Purpose |
|-------|---------|
| `/memory` | Project memory (read/write) |
| `/repo` | Code repository (read/write) |
| `/output` | Agent outputs |

## Important Constraints

1. **No Task Complete Without QA**: Every task must pass QA
2. **Max Iterations**: Tasks have iteration limits
3. **Blocked = Human Needed**: System cannot unblock itself
4. **Ephemeral Agents**: No persistent agent state
5. **Memory in Repo**: All memory is version-controlled

## Debugging

### Orchestrator Logs
```bash
docker-compose logs -f orchestrator
```

### Agent Execution
```bash
docker-compose logs -f agent-runner
```

### Memory State
Check `/memory/` files for current state.

## Common Tasks

### Initialize New Project
```bash
./scripts/init_project.sh
```

### Start Development
```bash
docker-compose up
```

### Check Health
```bash
./scripts/health_check.sh
```

## External Dependencies

- GitHub API
- LLM Provider (OpenAI/Anthropic/etc.)
- Docker runtime
- Optional: Redis for state, PostgreSQL for metrics

## Security Notes

- Store secrets in `.env` (never commit)
- Use GitHub App for API access
- Rotate tokens regularly
- Limit agent permissions

---

**This template enables autonomous AI software development. Maintain its generality and clarity.**
