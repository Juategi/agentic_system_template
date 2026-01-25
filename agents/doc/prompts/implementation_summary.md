# Implementation Summary Generator

Generate a concise technical summary of what was implemented.

## Issue Information

**Title:** {{issue_title}}

**Description:**
{{issue_body}}

## Files Modified

{{files_modified}}

## Your Task

Write a clear, technical summary (2-4 sentences) of what was implemented.

### Focus On

1. **What Changed**: The main functionality added or modified
2. **How It Works**: Brief technical approach
3. **Key Components**: Main files or modules involved
4. **Notable Decisions**: Any significant technical choices

### Guidelines

- Be concise but informative
- Use technical terms appropriately
- Focus on the "what" and "why", not implementation details
- Write for a developer audience

### Examples

**Good Summary:**
"Implemented user authentication with JWT tokens. Added login/logout endpoints in auth_controller.py and token validation middleware. User sessions are stored in Redis with 24-hour expiration."

**Too Vague:**
"Added authentication feature."

**Too Detailed:**
"Created a function called validate_token on line 45 that takes a string parameter..."

## Output

Return only the summary text. No headers, bullets, or formatting.
Just 2-4 sentences of plain text.
