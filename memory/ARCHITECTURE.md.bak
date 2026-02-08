# Architecture Memory

## System Architecture Overview

<!--
This file documents the technical architecture of the project.
Agents reference this to understand:
- System structure and components
- How components interact
- Architectural patterns to follow
- Technical decisions and their rationale
-->

## High-Level Architecture

```
[Add your system architecture diagram here using ASCII art or describe it]

Example:
┌─────────────────────────────────────────────────────────────┐
│                      CLIENT LAYER                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Web App   │  │ Mobile App  │  │    CLI      │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         └────────────────┼────────────────┘                │
└──────────────────────────┼──────────────────────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────────┐
│                      API LAYER                               │
│                    ┌─────┴─────┐                            │
│                    │  API GW   │                            │
│                    └─────┬─────┘                            │
└──────────────────────────┼──────────────────────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────────┐
│                    SERVICE LAYER                             │
│  ┌─────────────┐  ┌─────┴─────┐  ┌─────────────┐          │
│  │  Service A  │  │ Service B │  │  Service C  │          │
│  └──────┬──────┘  └─────┬─────┘  └──────┬──────┘          │
└─────────┼───────────────┼───────────────┼───────────────────┘
          │               │               │
┌─────────┼───────────────┼───────────────┼───────────────────┐
│         └───────────────┼───────────────┘                   │
│                    DATA LAYER                                │
│                    ┌─────┴─────┐                            │
│                    │  Database │                            │
│                    └───────────┘                            │
└─────────────────────────────────────────────────────────────┘
```

## Component Descriptions

### [Component 1 Name]
- **Purpose**: [What this component does]
- **Location**: [Where in the codebase]
- **Dependencies**: [What it depends on]
- **Interfaces**: [APIs it exposes]

### [Component 2 Name]
- **Purpose**: [Description]
- **Location**: [Path]
- **Dependencies**: [List]
- **Interfaces**: [List]

## Directory Structure

```
project/
├── src/                    # Source code
│   ├── api/                # API layer
│   ├── services/           # Business logic
│   ├── models/             # Data models
│   └── utils/              # Utilities
├── tests/                  # Test files
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── config/                 # Configuration
└── docs/                   # Documentation
```

## Architectural Patterns

### Pattern 1: [Pattern Name]
- **Where used**: [Components]
- **Implementation**: [How it's implemented]
- **Example**: [Reference to code]

### Pattern 2: [Pattern Name]
- **Where used**: [Components]
- **Implementation**: [How it's implemented]

## Data Flow

### Request Flow
1. [Step 1]: Client sends request to API Gateway
2. [Step 2]: Gateway routes to appropriate service
3. [Step 3]: Service processes request
4. [Step 4]: Response returns through same path

### Data Models

#### [Model 1 Name]
```
{
    "id": "string",
    "field1": "type",
    "field2": "type",
    "created_at": "datetime"
}
```

## Integration Points

### External APIs
| API | Purpose | Auth Method | Rate Limits |
|-----|---------|-------------|-------------|
| [API Name] | [Purpose] | [OAuth/Key] | [Limits] |

### Internal Services
| Service | Protocol | Port | Purpose |
|---------|----------|------|---------|
| [Service] | [HTTP/gRPC] | [Port] | [Purpose] |

## Security Architecture

### Authentication
- [Description of auth mechanism]

### Authorization
- [Description of authorization model]

### Data Protection
- [Encryption, data handling policies]

## Performance Considerations

### Caching Strategy
- [Where caching is used]
- [Cache invalidation approach]

### Scaling Approach
- [How the system scales]
- [Bottlenecks to be aware of]

## Architectural Decisions Record (ADR)

### ADR-001: [Decision Title]
- **Date**: [Date]
- **Status**: [Accepted/Deprecated/Superseded]
- **Context**: [Why this decision was needed]
- **Decision**: [What was decided]
- **Consequences**: [Impact of this decision]

## Diagrams

### Sequence Diagram: [Flow Name]
```
Client -> API: Request
API -> Service: Process
Service -> DB: Query
DB -> Service: Result
Service -> API: Response
API -> Client: Result
```

## Technical Debt

| Item | Description | Priority | Issue |
|------|-------------|----------|-------|
| [Item] | [Description] | [High/Med/Low] | #[number] |

---

*Last updated: [TIMESTAMP]*
*Maintained by: Agents and Human Architects*
