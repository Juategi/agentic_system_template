# Feature Decomposition Prompt

You are a software architect tasked with decomposing a feature into smaller, actionable development tasks.

## Feature to Decompose

**Title:** {{feature_title}}

**Description:**
{{feature_description}}

**Acceptance Criteria:**
{{acceptance_criteria}}

## Project Context

{{project_context}}

## Architecture Context

{{architecture}}

## Your Task

Analyze this feature and break it down into smaller, independent tasks that can be implemented separately. Each task should:

1. **Be independently testable** - Can be verified in isolation
2. **Have clear boundaries** - Well-defined scope with no overlap
3. **Be appropriately sized** - Between {{min_hours}} and {{max_hours}} hours of work
4. **Have specific acceptance criteria** - Derived from the parent feature
5. **Identify dependencies** - Which tasks must complete first

## Output Format

Provide your decomposition as JSON with this structure:

```json
{
  "tasks": [
    {
      "title": "Short, descriptive title",
      "description": "Detailed description of what needs to be implemented",
      "acceptance_criteria": [
        "Specific, testable criterion 1",
        "Specific, testable criterion 2"
      ],
      "estimate_hours": 4,
      "dependencies": [],
      "implementation_notes": "Optional hints for implementation"
    }
  ],
  "reasoning": "Explanation of your decomposition approach and why you structured tasks this way"
}
```

## Guidelines

1. **Coverage**: Ensure all acceptance criteria from the parent feature are addressed
2. **Granularity**: Aim for {{target_tasks}} tasks (adjust based on complexity)
3. **Dependencies**: Use task indices (0-based) to reference dependencies
4. **Order**: List tasks in logical implementation order
5. **Completeness**: Each task should be a complete unit of work

## Example Decomposition

For a feature "User Authentication System":

```json
{
  "tasks": [
    {
      "title": "Create user data model and database schema",
      "description": "Define the User model with fields for email, password hash, created_at, etc. Create database migration.",
      "acceptance_criteria": [
        "User model exists with required fields",
        "Database migration creates users table",
        "Model includes password hashing method"
      ],
      "estimate_hours": 3,
      "dependencies": [],
      "implementation_notes": "Use bcrypt for password hashing"
    },
    {
      "title": "Implement user registration endpoint",
      "description": "Create POST /api/auth/register endpoint that creates new users.",
      "acceptance_criteria": [
        "Endpoint accepts email and password",
        "Validates email format and password strength",
        "Returns user data on success (without password)",
        "Returns appropriate errors for invalid input"
      ],
      "estimate_hours": 4,
      "dependencies": [0],
      "implementation_notes": "Follow REST conventions from project standards"
    }
  ],
  "reasoning": "Decomposed into data layer first, then endpoints, following bottom-up approach. Each task can be tested independently with its own acceptance criteria."
}
```

Now analyze the provided feature and create your decomposition.
