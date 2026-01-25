# Implementation Planning Prompt

You are a senior software developer creating an implementation plan for a task.

## Task to Implement

**Title:** {{task_title}}

**Description:**
{{task_description}}

**Acceptance Criteria:**
{{acceptance_criteria}}

**Implementation Hints:**
{{implementation_hints}}

{{#if qa_feedback}}
## Previous QA Feedback (Iteration {{iteration}})
This is a retry. Address these issues:
{{qa_feedback}}
{{/if}}

## Project Context

### Conventions
{{conventions}}

### Architecture
{{architecture}}

## Relevant Code

{{relevant_code}}

## Your Task

Create a detailed implementation plan. Consider:

1. **Which files need to be created or modified** - Be specific about file paths
2. **The order of changes** - Dependencies between files
3. **What tests should be added** - Ensure testability
4. **Potential risks or challenges** - What could go wrong

## Output Format

Return your plan as JSON with this structure:

```json
{
  "approach": "High-level description of implementation approach",
  "files": [
    {
      "path": "path/to/file.py",
      "action": "create" or "modify",
      "description": "What changes are needed",
      "dependencies": ["other/file.py"]
    }
  ],
  "tests": ["Description of tests to add"],
  "risks": ["Potential risks or challenges"]
}
```

## Guidelines

1. **Minimize Changes**: Only modify what's necessary
2. **Follow Patterns**: Match existing code patterns
3. **Be Specific**: Provide exact file paths
4. **Consider Dependencies**: Order files by dependency
5. **Include Tests**: Always plan for testing

## Example Plan

For a task "Add user validation to registration":

```json
{
  "approach": "Add input validation layer before user creation, with custom validators and error messages.",
  "files": [
    {
      "path": "src/validators/user_validator.py",
      "action": "create",
      "description": "Create validator class with email, password, and username validation rules",
      "dependencies": []
    },
    {
      "path": "src/services/user_service.py",
      "action": "modify",
      "description": "Integrate validator in create_user method, handle validation errors",
      "dependencies": ["src/validators/user_validator.py"]
    },
    {
      "path": "tests/test_user_validator.py",
      "action": "create",
      "description": "Add unit tests for all validation rules",
      "dependencies": ["src/validators/user_validator.py"]
    }
  ],
  "tests": [
    "Test valid email formats are accepted",
    "Test invalid emails are rejected",
    "Test password strength requirements",
    "Test username format and length limits"
  ],
  "risks": [
    "Existing user creation might bypass validation if called directly",
    "Need to ensure error messages don't leak sensitive info"
  ]
}
```

Now create your implementation plan.
