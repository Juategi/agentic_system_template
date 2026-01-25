# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - QA AGENT PACKAGE
# =============================================================================
"""
QA (Quality Assurance) Agent Package

The QA Agent validates that implementations meet acceptance criteria.
It is the gatekeeper that ensures no task completes without proper validation.

Responsibilities:
    1. Parse acceptance criteria from task
    2. Run automated tests
    3. Verify each criterion is met
    4. Provide detailed feedback on failures
    5. Suggest fixes for issues found

Input:
    - Task issue (acceptance criteria)
    - Modified files from developer
    - Test commands from project config
    - Implementation notes

Output:
    - QA result (PASS/FAIL)
    - Acceptance criteria checklist
    - Test results
    - Detailed feedback (if failed)
    - Suggested fixes (if failed)

Critical Role:
    NO TASK IS COMPLETE WITHOUT QA PASSING.
    This agent is the quality gate of the entire system.
"""

from agents.qa.qa_agent import QAAgent

__all__ = ["QAAgent"]
