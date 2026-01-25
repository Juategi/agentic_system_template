# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - DOCUMENTATION AGENT PACKAGE
# =============================================================================
"""
Documentation Agent Package

The Doc Agent maintains project memory and documentation.
It ensures the knowledge base stays current as the project evolves.

This agent runs after successful task completion to record:
- What was implemented
- How it was implemented
- Decisions made and alternatives considered
- QA iteration history
- Lessons learned

Responsibilities:
    1. Update feature memory files
    2. Record implementation decisions
    3. Capture QA feedback history
    4. Maintain historical records
    5. Generate changelog entries

Memory Structure:
    memory/
    ├── PROJECT.md           # Global project context
    ├── ARCHITECTURE.md      # Technical architecture
    ├── CONVENTIONS.md       # Coding standards
    ├── CHANGELOG.md         # Project changelog
    └── features/            # Feature-specific memory
        └── issue-123-*.md   # Per-issue documentation

Usage:
    from agents.doc import DocAgent, FeatureMemory

    agent = DocAgent()
    result = agent.run()

    # Check what was updated
    for file in result.output["updated_files"]:
        print(f"Updated: {file}")

    # With custom config
    agent = DocAgent(config={
        "memory_path": "docs/memory",
        "create_changelog_entry": True
    })
"""

from .doc_agent import (
    DocAgent,
    DocumentationType,
    HistoryEntryType,
    HistoryEntry,
    FeatureMemory,
    ChangelogEntry,
    DocumentationUpdate,
)

__all__ = [
    # Main agent
    "DocAgent",
    # Enums
    "DocumentationType",
    "HistoryEntryType",
    # Data classes
    "HistoryEntry",
    "FeatureMemory",
    "ChangelogEntry",
    "DocumentationUpdate",
]

__version__ = "0.1.0"
