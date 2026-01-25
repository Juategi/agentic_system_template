# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - REVIEWER AGENT PACKAGE
# =============================================================================
"""
Reviewer Agent Package

The Reviewer Agent performs code review focusing on quality, consistency,
and alignment with project standards.

Unlike QA (which validates functionality), the Reviewer focuses on:
- Code craftsmanship and readability
- Convention adherence
- Architectural alignment
- Maintainability
- Documentation quality

Responsibilities:
    1. Review code quality and style
    2. Check adherence to conventions
    3. Verify architectural alignment
    4. Assess maintainability
    5. Provide improvement suggestions

Input:
    - Modified files from developer
    - Project conventions (CONVENTIONS.md)
    - Architecture documentation (ARCHITECTURE.md)
    - Previous review comments (if any)

Output:
    - Review decision (APPROVED/CHANGES_REQUESTED/NEEDS_DISCUSSION)
    - Quality score (0-100)
    - Inline comments with severity
    - Improvement suggestions

Usage:
    from agents.reviewer import ReviewerAgent, ReviewDecision

    agent = ReviewerAgent()
    result = agent.run()

    # Check decision
    if result.output["review_result"] == "APPROVED":
        print("Ready to merge!")
    else:
        for comment in result.output["file_reviews"][0]["comments"]:
            print(f"{comment['severity']}: {comment['message']}")

    # With custom config
    agent = ReviewerAgent(config={
        "approval_threshold": 80,
        "block_on_major": True,
        "include_praise": True
    })
"""

from .reviewer_agent import (
    ReviewerAgent,
    ReviewDecision,
    CommentSeverity,
    ReviewCategory,
    ReviewComment,
    FileReview,
    ReviewReport,
    FILE_REVIEW_SCHEMA,
)

__all__ = [
    # Main agent
    "ReviewerAgent",
    # Enums
    "ReviewDecision",
    "CommentSeverity",
    "ReviewCategory",
    # Data classes
    "ReviewComment",
    "FileReview",
    "ReviewReport",
    # Schemas
    "FILE_REVIEW_SCHEMA",
]

__version__ = "0.1.0"
