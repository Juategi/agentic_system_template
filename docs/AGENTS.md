# Agent Specifications

## Overview

The AI Agent Development System uses five specialized agents that share a single Docker image. Each agent has a specific role in the development workflow.

## Agent Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SHARED AGENT IMAGE                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              BASE AGENT INFRASTRUCTURE               │   │
│  │  - Context Loading                                   │   │
│  │  - LLM Client                                        │   │
│  │  - GitHub Helper                                     │   │
│  │  - Output Handler                                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│         ┌─────────────────┼─────────────────┐              │
│         │                 │                 │              │
│         ▼                 ▼                 ▼              │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐         │
│  │  PLANNER  │    │ DEVELOPER │    │    QA     │         │
│  └───────────┘    └───────────┘    └───────────┘         │
│         │                 │                 │              │
│         ▼                 ▼                 ▼              │
│  ┌───────────┐    ┌───────────┐                           │
│  │ REVIEWER  │    │    DOC    │                           │
│  └───────────┘    └───────────┘                           │
│                                                             │
│  AGENT_TYPE env var selects which agent runs               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Planner Agent

### Purpose
Decomposes complex features into smaller, actionable tasks.

### When Used
- New feature issues are created
- Large tasks need breakdown

### Input
- Feature issue (title, body, acceptance criteria)
- Project context (PROJECT.md, ARCHITECTURE.md)
- Constraints (CONSTRAINTS.md)

### Output
- Created sub-task issues
- Feature memory file
- Decomposition summary

### Behavior
1. Analyze feature requirements
2. Identify logical sub-components
3. Create GitHub Issues for each task
4. Establish dependencies between tasks
5. Create feature memory file

### Configuration
```yaml
planner:
  max_sub_issues: 10
  min_granularity_hours: 2
  max_granularity_hours: 8
  create_dependencies: true
```

## Developer Agent

### Purpose
Implements code changes to fulfill task requirements.

### When Used
- Task enters DEVELOPMENT state
- QA fails and iteration returns to development

### Input
- Task issue (requirements, criteria)
- Project conventions (CONVENTIONS.md)
- QA feedback (if retry)
- Related code context

### Output
- Modified files list
- Implementation notes
- Commit message
- Tests added (if any)

### Behavior
1. Parse task requirements
2. Analyze related codebase
3. Plan implementation approach
4. Generate code changes
5. Verify changes compile
6. Create git branch/commit

### Configuration
```yaml
developer:
  max_files_modified: 20
  max_lines_per_file: 500
  create_branch: true
  run_linter: true
```

## QA Agent

### Purpose
Validates that implementations meet acceptance criteria.

### When Used
- Task enters QA state after development

### Input
- Task issue (acceptance criteria)
- Modified files from developer
- Test commands from config

### Output
- QA result (PASS/FAIL)
- Acceptance checklist
- Test results
- Feedback (if failed)
- Suggested fixes (if failed)

### Behavior
1. Extract acceptance criteria
2. Run automated tests
3. Verify each criterion
4. Compile results
5. Generate actionable feedback

### Configuration
```yaml
qa:
  require_all_tests_pass: true
  require_no_linter_errors: true
  test_timeout_seconds: 300
```

### Critical Role
**NO TASK COMPLETES WITHOUT QA PASSING**

## Reviewer Agent

### Purpose
Reviews code quality and alignment with standards.

### When Used
- Task enters REVIEW state after QA passes

### Input
- Modified files
- Conventions (CONVENTIONS.md)
- Architecture (ARCHITECTURE.md)

### Output
- Review result (APPROVED/CHANGES_REQUESTED)
- Quality score
- Comments and suggestions

### Behavior
1. Review code quality
2. Check convention adherence
3. Verify architecture alignment
4. Calculate quality score
5. Generate improvement suggestions

### Configuration
```yaml
reviewer:
  approval_threshold: 70
  criteria_weights:
    code_quality: 0.3
    convention_adherence: 0.2
    architecture_alignment: 0.2
    maintainability: 0.15
    documentation: 0.15
```

## Doc Agent

### Purpose
Maintains project memory and documentation.

### When Used
- Task enters DOCUMENTATION state after review

### Input
- Task issue
- Implementation notes
- QA feedback
- Review comments

### Output
- Updated memory files
- Changelog entry
- Documentation notes

### Behavior
1. Update feature memory file
2. Record implementation decisions
3. Capture lessons learned
4. Generate changelog entry

### Configuration
```yaml
doc:
  document_decisions: true
  document_alternatives: true
  create_changelog_entry: true
```

## Agent Lifecycle

```
┌──────────┐
│   INIT   │ Load environment, config, context
└────┬─────┘
     │
     ▼
┌──────────┐
│ VALIDATE │ Check required inputs exist
└────┬─────┘
     │
     ▼
┌──────────┐
│ EXECUTE  │ Run agent-specific logic
└────┬─────┘
     │
     ▼
┌──────────┐
│  OUTPUT  │ Write results to volume
└────┬─────┘
     │
     ▼
┌──────────┐
│   EXIT   │ Container terminates
└──────────┘
```

## Environment Variables

All agents receive:

| Variable | Description |
|----------|-------------|
| AGENT_TYPE | planner/developer/qa/reviewer/doc |
| PROJECT_ID | Project identifier |
| ISSUE_NUMBER | Issue being processed |
| MEMORY_PATH | Path to memory volume |
| REPO_PATH | Path to repository volume |
| OUTPUT_PATH | Path to output volume |
| GITHUB_TOKEN | GitHub API token |
| GITHUB_REPO | Repository (owner/repo) |
| ITERATION | Current iteration number |
| MAX_ITERATIONS | Maximum allowed iterations |

## Volume Mounts

| Mount | Purpose | Access |
|-------|---------|--------|
| /memory | Project memory files | Read |
| /repo | Code repository | Read/Write |
| /output | Agent output | Write |
| /input | Orchestrator input | Read |

## Adding New Agents

1. Create agent package in `agents/{agent_name}/`
2. Implement agent class extending `AgentInterface`
3. Add prompts in `agents/{agent_name}/prompts/`
4. Register in `config/agents.yaml`
5. Add node in `orchestrator/nodes/`
6. Update LangGraph workflow
