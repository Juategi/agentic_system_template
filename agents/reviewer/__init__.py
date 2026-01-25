# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - REVIEWER AGENT PACKAGE
# =============================================================================
"""
Reviewer Agent Package

The Reviewer Agent performs code review focusing on quality, consistency,
and alignment with project standards.

Responsibilities:
    1. Review code quality and style
    2. Check adherence to conventions
    3. Verify architectural alignment
    4. Assess maintainability
    5. Provide improvement suggestions

Input:
    - Modified files from developer
    - Project conventions
    - Architecture documentation
    - Previous review comments (if any)

Output:
    - Review result (APPROVED/CHANGES_REQUESTED)
    - Quality score
    - Inline comments
    - Improvement suggestions
"""

from agents.reviewer.reviewer_agent import ReviewerAgent

__all__ = ["ReviewerAgent"]
