# Overall Code Review Assessment

Provide a final assessment after reviewing all files in a change set.

## Change Set Information

**Issue/PR:** {{issue_title}}
**Files Changed:** {{files_count}}
**Total Lines Changed:** {{lines_changed}}

## Individual File Reviews

{{file_reviews_summary}}

## Scores Breakdown

{{scores_breakdown}}

## Your Task

Synthesize the individual file reviews into an overall assessment.

### Consider

1. **Consistency Across Files**
   - Do the files follow consistent patterns?
   - Is there code duplication that should be extracted?

2. **Integration Quality**
   - Do the changes work well together?
   - Are there missing integration points?

3. **Change Scope**
   - Is the change focused and appropriate?
   - Are there unrelated changes mixed in?

4. **Risk Assessment**
   - What could go wrong with these changes?
   - Are there areas that need extra testing?

5. **Overall Impact**
   - Does this improve the codebase?
   - Is technical debt being added or reduced?

## Output Format

```json
{
  "decision": "APPROVED" | "CHANGES_REQUESTED" | "NEEDS_DISCUSSION",
  "confidence": "high" | "medium" | "low",
  "overall_assessment": "Brief overall assessment of the changes",
  "strengths": [
    "Well-structured code with clear separation of concerns",
    "Good test coverage for new functionality"
  ],
  "concerns": [
    "Missing error handling in edge cases",
    "Documentation needs updating"
  ],
  "blocking_issues": [
    "Critical security issue in authentication flow"
  ],
  "recommendations": [
    "Add unit tests for the new validation logic",
    "Consider extracting common patterns to a shared utility"
  ],
  "summary": "Markdown-formatted summary for the PR/issue comment"
}
```

## Decision Criteria

### APPROVED
- No critical or major issues
- Score above threshold (typically 70+)
- Changes improve or maintain code quality

### CHANGES_REQUESTED
- Has critical or major issues that must be fixed
- Score below threshold
- Significant quality concerns

### NEEDS_DISCUSSION
- Borderline score
- Architectural concerns that need team input
- Trade-offs that require discussion

---

Now provide your overall assessment.
