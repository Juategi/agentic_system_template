# Code Review Request

Review the following file for quality, conventions, and best practices.

## File Information

**Path:** {{file_path}}
**Language:** {{language}}

## File Content

```
{{file_content}}
```

## Project Conventions

{{conventions}}

## Architecture Guidelines

{{architecture}}

## Review Instructions

Perform a thorough code review analyzing the following aspects:

### 1. Code Quality (code_quality)
- Is the code clean and readable?
- Are variable/function names descriptive?
- Is the logic clear and well-structured?
- Are there any obvious bugs or issues?

### 2. Convention Adherence (conventions)
- Does the code follow project conventions?
- Is the style consistent with the codebase?
- Are naming conventions followed?
- Is the file structure appropriate?

### 3. Architecture Alignment (architecture)
- Does the code fit the system architecture?
- Are the right patterns being used?
- Is there proper separation of concerns?
- Are dependencies appropriate?

### 4. Maintainability (maintainability)
- Is the code easy to understand?
- Would it be easy to modify in the future?
- Is there appropriate abstraction?
- Is complexity well-managed?

### 5. Documentation (documentation)
- Are public APIs documented?
- Are complex sections commented?
- Are comments accurate and helpful?
- Is there any misleading documentation?

### 6. Security (security) - if applicable
- Are inputs validated?
- Are there potential injection vulnerabilities?
- Is sensitive data handled properly?

### 7. Performance (performance) - if applicable
- Are there obvious performance issues?
- Are resources used efficiently?
- Are there unnecessary operations?

## Output Format

Return your review as JSON:

```json
{
  "scores": {
    "code_quality": 85,
    "conventions": 90,
    "architecture": 80,
    "maintainability": 85,
    "documentation": 75,
    "security": 90,
    "performance": 85
  },
  "comments": [
    {
      "line": 42,
      "line_end": 45,
      "category": "code_quality",
      "severity": "major",
      "message": "This function has too many responsibilities",
      "suggestion": "Split into separate functions for validation and processing"
    },
    {
      "line": 10,
      "category": "documentation",
      "severity": "minor",
      "message": "Missing docstring for public function",
      "suggestion": "Add docstring with parameters and return value documentation"
    },
    {
      "line": 78,
      "category": "code_quality",
      "severity": "praise",
      "message": "Good use of early returns to reduce nesting"
    }
  ],
  "summary": "Overall good code quality. Main concerns are around function complexity in the data processing section and missing documentation for public APIs."
}
```

## Severity Levels

- **critical**: Must fix - security vulnerabilities, data corruption, crashes, breaking bugs
- **major**: Should fix - significant bugs, major style violations, performance problems
- **minor**: Nice to fix - minor style issues, small improvements, nitpicks
- **suggestion**: Optional - alternative approaches, potential enhancements
- **praise**: Positive feedback - highlight good practices (include if {{include_praise}})

## Guidelines

1. **Be Constructive**: Provide actionable feedback
2. **Be Specific**: Reference exact lines and suggest fixes
3. **Be Fair**: Acknowledge good code, not just problems
4. **Prioritize**: Focus on important issues first
5. **Context Matters**: Consider the file's purpose and constraints

---

Now review the code and provide your assessment.
