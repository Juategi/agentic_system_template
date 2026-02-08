# Technical Constraints

## Overview

<!--
This file documents technical constraints and limitations that agents
and developers must be aware of. These constraints affect what can
and cannot be implemented.
-->

## Infrastructure Constraints

### Hosting Environment
- **Platform**: [Cloud provider / On-premise]
- **Region**: [Geographic restrictions if any]
- **Compute**: [Resource limits]
- **Storage**: [Storage limits and types]

### Resource Limits
| Resource | Limit | Notes |
|----------|-------|-------|
| Memory | [X GB] | Per container/instance |
| CPU | [X cores] | Shared/dedicated |
| Storage | [X GB] | Persistent storage |
| Bandwidth | [X GB/month] | Network transfer |

## Technology Constraints

### Required Technologies
These must be used (non-negotiable):
- [Technology 1]: [Reason]
- [Technology 2]: [Reason]

### Prohibited Technologies
These must NOT be used:
- [Technology 1]: [Reason - security/license/compatibility]
- [Technology 2]: [Reason]

### Version Constraints
| Technology | Required Version | Reason |
|------------|------------------|--------|
| [Python] | [>=3.9] | [Compatibility] |
| [Node.js] | [>=18] | [LTS requirement] |
| [Database] | [Version] | [Feature requirement] |

## Security Constraints

### Data Handling
- **PII**: [How to handle personally identifiable information]
- **Secrets**: Must use environment variables or secret manager
- **Encryption**: [Requirements for data at rest/in transit]

### Authentication/Authorization
- [Authentication requirements]
- [Authorization model constraints]

### Compliance
- [Relevant compliance requirements: GDPR, HIPAA, SOC2, etc.]

## Performance Constraints

### Response Time
| Operation | Maximum Time | Notes |
|-----------|-------------|-------|
| API response | [200ms] | P95 latency |
| Page load | [3s] | Initial load |
| Database query | [100ms] | Single query |

### Throughput
- Minimum: [X requests/second]
- Peak capacity: [Y requests/second]

### Availability
- Target uptime: [99.9%]
- Maintenance windows: [Scheduled times]

## Integration Constraints

### External APIs
| API | Rate Limit | Auth Method | Notes |
|-----|------------|-------------|-------|
| [API 1] | [X/min] | [OAuth] | [Constraints] |
| [API 2] | [Y/hour] | [API Key] | [Constraints] |

### Internal Services
- [Service dependencies and their constraints]

## Development Constraints

### Code Constraints
- Maximum file size: [X lines recommended]
- Maximum function length: [Y lines recommended]
- Maximum cyclomatic complexity: [Z]
- Test coverage minimum: [X%]

### Deployment Constraints
- Deployment frequency: [How often]
- Deployment windows: [When allowed]
- Rollback requirements: [Time to rollback]

## Budget Constraints

### Cost Limits
| Resource | Monthly Budget | Notes |
|----------|---------------|-------|
| Compute | [$X] | Cloud instances |
| Storage | [$Y] | Databases, files |
| API calls | [$Z] | Third-party APIs |
| LLM usage | [$W] | Token costs |

### Cost Optimization Rules
- [Rules for managing costs]
- [When to alert on cost overruns]

## Timeline Constraints

### Deadlines
| Milestone | Date | Notes |
|-----------|------|-------|
| [Milestone 1] | [Date] | [Hard/soft deadline] |
| [Milestone 2] | [Date] | [Dependencies] |

### Time Zones
- Primary: [Timezone]
- Support hours: [Hours in timezone]

## Compatibility Constraints

### Browser Support
| Browser | Minimum Version | Notes |
|---------|-----------------|-------|
| Chrome | [Version] | Desktop/Mobile |
| Firefox | [Version] | Desktop |
| Safari | [Version] | Desktop/iOS |
| Edge | [Version] | Desktop |

### Device Support
- Desktop: [Requirements]
- Mobile: [Requirements]
- Tablet: [Requirements]

## Legal Constraints

### Licensing
- Project license: [License type]
- Dependency restrictions: [GPL-compatible only, etc.]

### Data Residency
- [Where data can/cannot be stored]
- [Data transfer restrictions]

## Agent-Specific Constraints

### What Agents CAN Do
- Create and modify files in /repo
- Read from /memory
- Write to /output
- Call configured APIs
- Run specified test commands

### What Agents CANNOT Do
- Delete files (without explicit instruction)
- Access network outside allowed hosts
- Modify system files
- Install new packages without approval
- Push to remote repository directly

### Iteration Limits
- Maximum QA iterations: [5]
- Maximum planning depth: [3 levels]
- Task blocked after: [N failures]

---

*Last updated: [TIMESTAMP]*
*Review these constraints before implementing any feature*
