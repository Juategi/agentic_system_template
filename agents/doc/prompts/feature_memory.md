# Feature Memory Generator

Generate a comprehensive feature memory document for the project knowledge base.

## Issue Information

**Number:** {{issue_number}}
**Title:** {{issue_title}}

**Description:**
{{issue_body}}

## Implementation Details

**Files Modified:**
{{files_modified}}

**Implementation Notes:**
{{implementation_notes}}

**QA History:**
{{qa_history}}

**Decisions Made:**
{{decisions}}

## Your Task

Generate a feature memory document that captures:

1. **Technical Approach**
   - How was this feature implemented?
   - What patterns or architecture were used?

2. **Key Decisions**
   - What technical decisions were made?
   - What alternatives were considered?
   - Why was this approach chosen?

3. **File Changes**
   - Which files were modified/created?
   - What is each file's role?

4. **Lessons Learned**
   - Any challenges encountered?
   - What would be done differently?

5. **Future Considerations**
   - Any technical debt introduced?
   - Potential improvements for later?

## Output Format

Return the content in this markdown structure:

```markdown
## Technical Approach

[Describe the implementation approach]

## Key Decisions

### Decision 1: [Title]
- **Choice:** [What was decided]
- **Alternatives:** [What else was considered]
- **Rationale:** [Why this choice]

## Implementation Details

[Detailed implementation notes]

## Lessons Learned

- [Lesson 1]
- [Lesson 2]

## Future Considerations

- [Consideration 1]
- [Consideration 2]
```

## Guidelines

- Be specific and technical
- Document the "why" behind decisions
- Include context for future developers
- Note any trade-offs made
- Keep it concise but comprehensive
