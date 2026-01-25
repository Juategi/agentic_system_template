# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - DOCUMENTATION AGENT PACKAGE
# =============================================================================
"""
Documentation Agent Package

The Doc Agent maintains project memory and documentation.
It ensures the knowledge base stays current as the project evolves.

Responsibilities:
    1. Update feature memory files
    2. Record implementation decisions
    3. Capture QA feedback
    4. Maintain historical records
    5. Generate changelog entries
"""

from agents.doc.doc_agent import DocAgent

__all__ = ["DocAgent"]
