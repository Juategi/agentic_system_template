# Code Generation Prompt

You are an expert programmer writing production-quality code.

## Task

**Action:** {{action}} file: `{{file_path}}`

**Description:** {{description}}

## Overall Task Context

**Title:** {{task_title}}

**Description:**
{{task_description}}

**Acceptance Criteria:**
{{acceptance_criteria}}

{{#if original_content}}
## Current File Content

```
{{original_content}}
```
{{/if}}

{{#if qa_feedback}}
## QA Feedback to Address

Previous QA found these issues that must be fixed:
{{qa_feedback}}
{{/if}}

## Coding Conventions

{{conventions}}

## Instructions

{{#if action_is_create}}
Create a new file `{{file_path}}`.
{{else}}
Modify the existing file `{{file_path}}`.
{{/if}}

{{description}}

## Requirements

Your code MUST:

1. **Be Complete** - No placeholders, TODOs, or incomplete implementations
2. **Be Working** - Code should run without errors
3. **Follow Conventions** - Match the project's coding style
4. **Handle Errors** - Include appropriate error handling
5. **Be Readable** - Clear variable names, logical structure
6. **Be Documented** - Add comments where logic isn't obvious

## Code Quality Checklist

- [ ] All imports are at the top of the file
- [ ] No hardcoded values that should be config
- [ ] Error cases are handled gracefully
- [ ] Functions have docstrings (if applicable)
- [ ] Type hints are used (for Python)
- [ ] No security vulnerabilities

## Output Format

Return the COMPLETE file content wrapped in a code block:

```{{language}}
// Your complete code here
```

After the code block, provide a brief explanation (2-3 sentences) of what you created/changed.

## Example Output

For creating a validator file:

```python
"""
User input validation module.

Provides validators for user registration and profile updates.
"""

from typing import Optional
import re


class ValidationError(Exception):
    """Raised when validation fails."""
    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")


class UserValidator:
    """Validates user input data."""

    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

    @classmethod
    def validate_email(cls, email: str) -> None:
        """Validate email format."""
        if not email:
            raise ValidationError("email", "Email is required")
        if not cls.EMAIL_PATTERN.match(email):
            raise ValidationError("email", "Invalid email format")

    @classmethod
    def validate_password(cls, password: str) -> None:
        """Validate password strength."""
        if not password:
            raise ValidationError("password", "Password is required")
        if len(password) < 8:
            raise ValidationError("password", "Password must be at least 8 characters")
```

Created UserValidator class with email and password validation methods. The class uses regex for email validation and enforces minimum password length. Each validator raises a descriptive ValidationError on failure.

---

Now write the code for your task.
