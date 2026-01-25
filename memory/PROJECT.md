# Project Memory - Global Context

## Project Overview

<!--
This file contains the global context for the project.
It is read by all agents to understand the project's purpose,
rules, and current state.

Update this file when:
- Project scope changes
- New constraints are added
- Major decisions are made
-->

### Project Name
[PROJECT_NAME]

### Project Description
[Brief description of what this project does]

### Project Goals
1. [Primary goal]
2. [Secondary goal]
3. [Additional goals]

## Current Status

### Phase
[Planning | Development | Testing | Production]

### Active Features
| Feature | Issue | Status | Priority |
|---------|-------|--------|----------|
| [Feature name] | #[number] | [status] | [priority] |

### Blocked Items
| Item | Issue | Blocked By | Since |
|------|-------|------------|-------|
| [None currently] | | | |

## Project Rules

### Development Rules
1. All code changes must be associated with a GitHub Issue
2. No task is complete without passing QA
3. Follow coding conventions in CONVENTIONS.md
4. Respect architectural patterns in ARCHITECTURE.md

### Quality Rules
1. All new code must have appropriate test coverage
2. No linter errors in committed code
3. Code must pass review before completion
4. Documentation must be updated with changes

### Process Rules
1. Features are decomposed into tasks before development
2. Tasks iterate through Dev→QA until passing
3. Maximum [N] iterations before requiring human intervention
4. All decisions must be documented in memory

## Technology Stack

### Primary Languages
- [Language 1]
- [Language 2]

### Frameworks
- [Framework 1]
- [Framework 2]

### Infrastructure
- [Infrastructure component 1]
- [Infrastructure component 2]

## Key Decisions

<!-- Record important decisions here with context -->

| Date | Decision | Rationale | Issue |
|------|----------|-----------|-------|
| [Date] | [Decision made] | [Why this decision] | #[issue] |

## External Dependencies

### APIs
- [API 1]: [Purpose]
- [API 2]: [Purpose]

### Services
- [Service 1]: [Purpose]

## Contacts

### Human Oversight
- Primary: [Name/Role]
- Backup: [Name/Role]

## Notes

<!--
Add any important notes that don't fit elsewhere.
Agents can reference this section for context.
-->

---

*Last updated: [TIMESTAMP]*
*Updated by: [AGENT_TYPE or HUMAN]*
