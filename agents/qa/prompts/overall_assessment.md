# Overall QA Assessment

You are providing a final QA assessment after all individual criteria have been verified.

## Task Information

**Title:** {{task_title}}

**Description:**
{{task_description}}

## Verification Results

### Acceptance Criteria Results
{{criteria_summary}}

### Test Results
{{test_summary}}

## Modified Files

{{modified_files_list}}

## Your Task

Provide an overall assessment of the implementation quality.

### Assessment Areas

1. **Requirements Coverage**
   - Are all acceptance criteria addressed?
   - Are there gaps in the implementation?

2. **Code Quality**
   - Does the code follow project conventions?
   - Is it maintainable and readable?

3. **Test Coverage**
   - Are there adequate tests?
   - Do tests cover edge cases?

4. **Integration**
   - Does it integrate well with existing code?
   - Are there potential conflicts?

5. **Risk Assessment**
   - What could go wrong in production?
   - Are there any red flags?

## Output Format

```json
{
  "overall_verdict": "PASS" | "FAIL" | "CONDITIONAL_PASS",
  "confidence": "high" | "medium" | "low",
  "summary": "Brief summary of the overall assessment",
  "strengths": [
    "Well-structured validation logic",
    "Good error messages"
  ],
  "concerns": [
    "Missing edge case for empty input",
    "No integration tests"
  ],
  "recommendations": [
    "Add test for edge case X before merging",
    "Consider refactoring Y in future iteration"
  ],
  "ready_for_merge": true | false,
  "blocking_issues": ["List any issues that must be fixed before merge"]
}
```

### Verdict Meanings

- **PASS**: Ready for merge, all criteria met, good quality
- **CONDITIONAL_PASS**: Minor issues that don't block but should be tracked
- **FAIL**: Significant issues that must be addressed

### Confidence Levels

- **high**: Clear evidence for the verdict
- **medium**: Some uncertainty but reasonable conclusion
- **low**: Limited evidence, need more review

## Guidelines

1. **Be Balanced**
   - Acknowledge what works well
   - Be clear about what needs improvement

2. **Prioritize Feedback**
   - Blocking issues first
   - Nice-to-haves last

3. **Consider the Big Picture**
   - Is this a net improvement?
   - Does it introduce technical debt?

4. **Be Actionable**
   - Every concern should have a path to resolution
   - Recommendations should be specific

---

Now provide your overall assessment.
