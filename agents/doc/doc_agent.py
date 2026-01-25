# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - DOCUMENTATION AGENT IMPLEMENTATION
# =============================================================================
"""
Documentation Agent Implementation

This agent maintains project memory and documentation.
It runs after successful task completion to record what was done.

Documentation Tasks:
    1. Update feature memory with implementation details
    2. Record decisions and alternatives considered
    3. Capture lessons learned from QA iterations
    4. Update changelog
    5. Maintain audit trail

Memory System:
    The agent manages markdown files in the /memory directory:
    - PROJECT.md: Global project context
    - features/*.md: Feature-specific memory
    - CHANGELOG.md: Project changelog
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.base import (
    AgentContext,
    AgentInterface,
    AgentResult,
    AgentStatus,
    LLMClient,
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


class DocumentationType(Enum):
    """Types of documentation updates."""
    FEATURE_MEMORY = "feature_memory"
    CHANGELOG = "changelog"
    ARCHITECTURE = "architecture"
    CONVENTIONS = "conventions"
    PROJECT = "project"


class HistoryEntryType(Enum):
    """Types of history entries."""
    CREATED = "created"
    PLANNING = "planning"
    DEVELOPMENT = "development"
    QA_PASSED = "qa_passed"
    QA_FAILED = "qa_failed"
    REVIEW = "review"
    COMPLETED = "completed"
    BLOCKED = "blocked"


@dataclass
class HistoryEntry:
    """A single entry in the feature history."""
    timestamp: str
    entry_type: HistoryEntryType
    agent: str
    action: str
    result: str
    details: Optional[str] = None

    def to_markdown_row(self) -> str:
        """Convert to markdown table row."""
        return f"| {self.timestamp} | {self.agent} | {self.action} | {self.result} |"


@dataclass
class FeatureMemory:
    """Structure for feature memory file."""
    issue_number: int
    title: str
    status: str
    created_at: str
    updated_at: str
    description: str
    acceptance_criteria: List[str]
    technical_approach: str
    implementation_summary: str
    files_modified: List[str]
    decisions: List[Dict[str, str]]
    dependencies: List[str]
    history: List[HistoryEntry]
    qa_feedback: List[str]
    notes: str

    def to_markdown(self) -> str:
        """Convert to markdown format."""
        lines = [
            f"# Feature: {self.title}",
            "",
            "## Metadata",
            f"- **Issue:** #{self.issue_number}",
            f"- **Status:** {self.status}",
            f"- **Created:** {self.created_at}",
            f"- **Updated:** {self.updated_at}",
            "",
            "## Description",
            self.description,
            "",
            "## Acceptance Criteria",
        ]

        for criterion in self.acceptance_criteria:
            lines.append(f"- [ ] {criterion}")

        lines.extend([
            "",
            "## Technical Approach",
            self.technical_approach,
            "",
            "## Implementation Summary",
            self.implementation_summary,
            "",
            "## Files Modified",
        ])

        for file_path in self.files_modified:
            lines.append(f"- `{file_path}`")

        if self.decisions:
            lines.extend(["", "## Decisions"])
            for decision in self.decisions:
                lines.append(f"### {decision.get('title', 'Decision')}")
                lines.append(f"**Choice:** {decision.get('choice', 'N/A')}")
                if decision.get('alternatives'):
                    lines.append(f"**Alternatives:** {decision.get('alternatives')}")
                if decision.get('rationale'):
                    lines.append(f"**Rationale:** {decision.get('rationale')}")
                lines.append("")

        if self.dependencies:
            lines.extend(["", "## Dependencies"])
            for dep in self.dependencies:
                lines.append(f"- {dep}")

        lines.extend([
            "",
            "## History",
            "| Date | Agent | Action | Result |",
            "|------|-------|--------|--------|",
        ])

        for entry in self.history:
            lines.append(entry.to_markdown_row())

        if self.qa_feedback:
            lines.extend(["", "## QA Feedback"])
            for feedback in self.qa_feedback:
                lines.append(f"- {feedback}")

        if self.notes:
            lines.extend([
                "",
                "## Notes",
                self.notes,
            ])

        return "\n".join(lines)


@dataclass
class ChangelogEntry:
    """A changelog entry."""
    version: Optional[str]
    date: str
    issue_number: int
    title: str
    category: str  # Added, Changed, Fixed, Removed, Security
    summary: str
    details: List[str]

    def to_markdown(self) -> str:
        """Convert to markdown format."""
        lines = [
            f"### [{self.category}] {self.title} (#{self.issue_number})",
            "",
            self.summary,
            "",
        ]

        if self.details:
            for detail in self.details:
                lines.append(f"- {detail}")

        return "\n".join(lines)


@dataclass
class DocumentationUpdate:
    """Represents a documentation update."""
    doc_type: DocumentationType
    file_path: str
    content: str
    action: str  # create, update, append

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.doc_type.value,
            "file": self.file_path,
            "action": self.action,
        }


# =============================================================================
# DOC AGENT CLASS
# =============================================================================


class DocAgent(AgentInterface):
    """
    Agent that updates documentation and project memory.

    This agent runs after successful task completion to:
    - Update feature memory files with implementation details
    - Record decisions and alternatives
    - Capture QA iteration history
    - Generate changelog entries
    - Maintain audit trail

    Configuration:
        document_decisions: Record implementation decisions (default: True)
        document_alternatives: Record alternatives considered (default: True)
        create_changelog_entry: Generate changelog entries (default: True)
        update_feature_memory: Update feature memory files (default: True)
        memory_path: Path to memory directory (default: "memory")
    """

    DEFAULT_CONFIG = {
        "document_decisions": True,
        "document_alternatives": True,
        "create_changelog_entry": True,
        "update_feature_memory": True,
        "memory_path": "memory",
        "features_subdir": "features",
        "temperature": 0.3,
    }

    # Category keywords for changelog
    CATEGORY_KEYWORDS = {
        "Added": ["add", "new", "create", "implement", "introduce"],
        "Changed": ["change", "update", "modify", "refactor", "improve", "enhance"],
        "Fixed": ["fix", "bug", "issue", "resolve", "repair", "correct"],
        "Removed": ["remove", "delete", "deprecate", "drop"],
        "Security": ["security", "vulnerability", "auth", "permission", "encrypt"],
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Doc agent with configuration."""
        super().__init__(config)
        self._llm: Optional[LLMClient] = None
        self._prompts: Dict[str, str] = {}

        # Merge default config with provided config
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}

    @property
    def llm(self) -> LLMClient:
        """Lazy-load LLM client."""
        if self._llm is None:
            self._llm = LLMClient(
                temperature=self.config["temperature"]
            )
        return self._llm

    def get_agent_type(self) -> str:
        """Return agent type identifier."""
        return "doc"

    def validate_context(self, context: AgentContext) -> bool:
        """
        Validate context has required information.

        Required:
            - Issue number for tracking
        """
        if not context.issue_data:
            self.logger.error("No issue data provided")
            return False

        if not context.issue_data.get("number"):
            self.logger.error("Issue number is required")
            return False

        return True

    def execute(self, context: AgentContext) -> AgentResult:
        """
        Update project documentation.

        Workflow:
            1. Gather implementation details from context
            2. Load or create feature memory
            3. Update feature memory with new information
            4. Generate changelog entry
            5. Write all documentation updates
            6. Return summary of changes

        Args:
            context: Agent execution context

        Returns:
            AgentResult with documentation updates
        """
        issue_number = context.issue_data.get("number")
        self.logger.info(f"Starting documentation update for issue #{issue_number}")

        try:
            documentation_updates: List[DocumentationUpdate] = []

            # =================================================================
            # STEP 1: Gather Implementation Details
            # =================================================================
            self.logger.info("Step 1: Gathering implementation details")

            impl_details = self._gather_implementation_details(context)

            # =================================================================
            # STEP 2: Update Feature Memory
            # =================================================================
            if self.config["update_feature_memory"]:
                self.logger.info("Step 2: Updating feature memory")

                feature_update = self._update_feature_memory(
                    context=context,
                    impl_details=impl_details
                )
                if feature_update:
                    documentation_updates.append(feature_update)

            # =================================================================
            # STEP 3: Generate Changelog Entry
            # =================================================================
            if self.config["create_changelog_entry"]:
                self.logger.info("Step 3: Generating changelog entry")

                changelog_update = self._generate_changelog_entry(
                    context=context,
                    impl_details=impl_details
                )
                if changelog_update:
                    documentation_updates.append(changelog_update)

            # =================================================================
            # STEP 4: Write Documentation Updates
            # =================================================================
            self.logger.info("Step 4: Writing documentation updates")

            written_files = self._write_documentation(
                updates=documentation_updates,
                repo_path=context.repo_path
            )

            # =================================================================
            # STEP 5: Generate Summary
            # =================================================================
            self.logger.info("Step 5: Generating summary")

            summary = self._generate_summary(
                context=context,
                updates=documentation_updates,
                written_files=written_files
            )

            # =================================================================
            # STEP 6: Return Results
            # =================================================================
            return AgentResult(
                status=AgentStatus.SUCCESS,
                output={
                    "updated_files": written_files,
                    "documentation_updates": [u.to_dict() for u in documentation_updates],
                    "implementation_summary": impl_details.get("summary", ""),
                    "changelog_entry": next(
                        (u.content for u in documentation_updates
                         if u.doc_type == DocumentationType.CHANGELOG),
                        None
                    ),
                    "summary": summary
                },
                message=f"Documentation updated: {len(written_files)} files",
                metrics={
                    "files_updated": len(written_files),
                    "feature_memory_updated": any(
                        u.doc_type == DocumentationType.FEATURE_MEMORY
                        for u in documentation_updates
                    ),
                    "changelog_updated": any(
                        u.doc_type == DocumentationType.CHANGELOG
                        for u in documentation_updates
                    ),
                }
            )

        except Exception as e:
            self.logger.error(f"Documentation update failed: {e}", exc_info=True)
            return AgentResult(
                status=AgentStatus.FAILED,
                output={"error": str(e)},
                message=f"Documentation error: {str(e)}"
            )

    # =========================================================================
    # IMPLEMENTATION DETAILS GATHERING
    # =========================================================================

    def _gather_implementation_details(self, context: AgentContext) -> Dict[str, Any]:
        """
        Gather implementation details from context.

        Collects:
            - Modified files
            - Implementation notes
            - QA results
            - Decisions made
        """
        input_data = context.input_data or {}

        details = {
            "summary": input_data.get("implementation_notes", ""),
            "files_modified": input_data.get("modified_files", []),
            "decisions": input_data.get("decisions", []),
            "qa_results": input_data.get("qa_results", []),
            "qa_iterations": input_data.get("qa_iterations", 0),
            "commit_message": input_data.get("commit_message", ""),
            "branch_name": input_data.get("branch_name", ""),
        }

        # If no summary provided, generate one
        if not details["summary"] and details["files_modified"]:
            details["summary"] = self._generate_implementation_summary(context, details)

        return details

    def _generate_implementation_summary(
        self,
        context: AgentContext,
        details: Dict[str, Any]
    ) -> str:
        """Generate implementation summary using LLM."""
        prompt = self._load_prompt("implementation_summary")

        prompt = prompt.replace("{{issue_title}}", context.issue_data.get("title", ""))
        prompt = prompt.replace("{{issue_body}}", context.issue_data.get("body", "")[:1000])
        prompt = prompt.replace("{{files_modified}}", "\n".join(
            f"- {f}" for f in details["files_modified"][:20]
        ))

        try:
            response = self.llm.generate(
                prompt=prompt,
                system="You are a technical writer summarizing code implementations."
            )
            return response.strip()
        except Exception as e:
            self.logger.warning(f"Could not generate summary: {e}")
            return f"Implemented changes for: {context.issue_data.get('title', 'Unknown')}"

    # =========================================================================
    # FEATURE MEMORY
    # =========================================================================

    def _update_feature_memory(
        self,
        context: AgentContext,
        impl_details: Dict[str, Any]
    ) -> Optional[DocumentationUpdate]:
        """
        Update or create feature memory file.

        The feature memory captures everything about a feature's implementation.
        """
        issue_number = context.issue_data.get("number")
        issue_title = context.issue_data.get("title", "Unknown Feature")

        # Determine file path
        memory_path = self.config["memory_path"]
        features_subdir = self.config["features_subdir"]
        feature_slug = self._slugify(issue_title)
        file_path = f"{memory_path}/{features_subdir}/issue-{issue_number}-{feature_slug}.md"

        # Load existing memory or create new
        existing_memory = self._load_feature_memory(file_path, context.repo_path)

        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        if existing_memory:
            # Update existing memory
            memory = existing_memory
            memory.updated_at = now
            memory.status = "COMPLETED"
            memory.implementation_summary = impl_details.get("summary", memory.implementation_summary)

            # Add new files
            new_files = impl_details.get("files_modified", [])
            for f in new_files:
                if f not in memory.files_modified:
                    memory.files_modified.append(f)

            # Add new decisions
            for decision in impl_details.get("decisions", []):
                memory.decisions.append(decision)

            # Add history entry
            memory.history.append(HistoryEntry(
                timestamp=now,
                entry_type=HistoryEntryType.COMPLETED,
                agent="doc",
                action="Documentation updated",
                result="Success"
            ))

            # Add QA feedback
            for qa_result in impl_details.get("qa_results", []):
                if isinstance(qa_result, dict):
                    memory.qa_feedback.append(qa_result.get("feedback", str(qa_result)))
                else:
                    memory.qa_feedback.append(str(qa_result))

        else:
            # Create new memory
            memory = FeatureMemory(
                issue_number=issue_number,
                title=issue_title,
                status="COMPLETED",
                created_at=now,
                updated_at=now,
                description=context.issue_data.get("body", "")[:500],
                acceptance_criteria=self._extract_acceptance_criteria(
                    context.issue_data.get("body", "")
                ),
                technical_approach=impl_details.get("summary", ""),
                implementation_summary=impl_details.get("summary", ""),
                files_modified=impl_details.get("files_modified", []),
                decisions=impl_details.get("decisions", []),
                dependencies=[],
                history=[
                    HistoryEntry(
                        timestamp=now,
                        entry_type=HistoryEntryType.COMPLETED,
                        agent="doc",
                        action="Feature completed",
                        result="Success"
                    )
                ],
                qa_feedback=[],
                notes=""
            )

        content = memory.to_markdown()

        return DocumentationUpdate(
            doc_type=DocumentationType.FEATURE_MEMORY,
            file_path=file_path,
            content=content,
            action="update" if existing_memory else "create"
        )

    def _load_feature_memory(
        self,
        file_path: str,
        repo_path: Optional[str]
    ) -> Optional[FeatureMemory]:
        """Load existing feature memory file."""
        base_path = Path(repo_path) if repo_path else Path.cwd()
        full_path = base_path / file_path

        if not full_path.exists():
            return None

        try:
            content = full_path.read_text(encoding="utf-8")
            return self._parse_feature_memory(content)
        except Exception as e:
            self.logger.warning(f"Could not load feature memory: {e}")
            return None

    def _parse_feature_memory(self, content: str) -> Optional[FeatureMemory]:
        """Parse feature memory from markdown content."""
        # This is a simplified parser - in production you'd want more robust parsing
        try:
            # Extract basic fields using regex
            title_match = re.search(r'^# Feature: (.+)$', content, re.MULTILINE)
            issue_match = re.search(r'\*\*Issue:\*\* #(\d+)', content)
            status_match = re.search(r'\*\*Status:\*\* (.+)$', content, re.MULTILINE)
            created_match = re.search(r'\*\*Created:\*\* (.+)$', content, re.MULTILINE)
            updated_match = re.search(r'\*\*Updated:\*\* (.+)$', content, re.MULTILINE)

            # Extract sections
            description = self._extract_section(content, "Description")
            technical_approach = self._extract_section(content, "Technical Approach")
            implementation_summary = self._extract_section(content, "Implementation Summary")

            # Extract files
            files_section = self._extract_section(content, "Files Modified")
            files = re.findall(r'`([^`]+)`', files_section) if files_section else []

            return FeatureMemory(
                issue_number=int(issue_match.group(1)) if issue_match else 0,
                title=title_match.group(1) if title_match else "Unknown",
                status=status_match.group(1) if status_match else "UNKNOWN",
                created_at=created_match.group(1) if created_match else "",
                updated_at=updated_match.group(1) if updated_match else "",
                description=description or "",
                acceptance_criteria=[],
                technical_approach=technical_approach or "",
                implementation_summary=implementation_summary or "",
                files_modified=files,
                decisions=[],
                dependencies=[],
                history=[],
                qa_feedback=[],
                notes=""
            )
        except Exception as e:
            self.logger.warning(f"Error parsing feature memory: {e}")
            return None

    def _extract_section(self, content: str, section_name: str) -> Optional[str]:
        """Extract content of a markdown section."""
        pattern = rf'^## {section_name}\s*\n(.*?)(?=^## |\Z)'
        match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
        return match.group(1).strip() if match else None

    def _extract_acceptance_criteria(self, body: str) -> List[str]:
        """Extract acceptance criteria from issue body."""
        criteria = []

        # Look for checkbox items
        checkbox_pattern = r'-\s*\[[\sx]\]\s*(.+?)(?=\n|$)'
        for match in re.finditer(checkbox_pattern, body):
            criteria.append(match.group(1).strip())

        return criteria

    # =========================================================================
    # CHANGELOG
    # =========================================================================

    def _generate_changelog_entry(
        self,
        context: AgentContext,
        impl_details: Dict[str, Any]
    ) -> Optional[DocumentationUpdate]:
        """Generate changelog entry for the completed work."""
        issue_number = context.issue_data.get("number")
        issue_title = context.issue_data.get("title", "Unknown")

        # Determine category
        category = self._determine_changelog_category(issue_title, impl_details)

        # Create entry
        entry = ChangelogEntry(
            version=None,  # Version determined by release process
            date=datetime.now().strftime("%Y-%m-%d"),
            issue_number=issue_number,
            title=issue_title,
            category=category,
            summary=impl_details.get("summary", f"Completed: {issue_title}"),
            details=[f"`{f}`" for f in impl_details.get("files_modified", [])[:5]]
        )

        # Format for appending to changelog
        changelog_content = entry.to_markdown()

        return DocumentationUpdate(
            doc_type=DocumentationType.CHANGELOG,
            file_path=f"{self.config['memory_path']}/CHANGELOG.md",
            content=changelog_content,
            action="append"
        )

    def _determine_changelog_category(
        self,
        title: str,
        impl_details: Dict[str, Any]
    ) -> str:
        """Determine changelog category based on title and details."""
        title_lower = title.lower()

        for category, keywords in self.CATEGORY_KEYWORDS.items():
            if any(kw in title_lower for kw in keywords):
                return category

        # Default to "Changed" if no match
        return "Changed"

    # =========================================================================
    # WRITING DOCUMENTATION
    # =========================================================================

    def _write_documentation(
        self,
        updates: List[DocumentationUpdate],
        repo_path: Optional[str]
    ) -> List[str]:
        """Write all documentation updates to disk."""
        written_files = []
        base_path = Path(repo_path) if repo_path else Path.cwd()

        for update in updates:
            full_path = base_path / update.file_path

            try:
                # Ensure directory exists
                full_path.parent.mkdir(parents=True, exist_ok=True)

                if update.action == "append":
                    # Append to existing file
                    existing_content = ""
                    if full_path.exists():
                        existing_content = full_path.read_text(encoding="utf-8")

                    # Add separator if file has content
                    if existing_content.strip():
                        new_content = existing_content.rstrip() + "\n\n---\n\n" + update.content
                    else:
                        new_content = f"# Changelog\n\n{update.content}"

                    full_path.write_text(new_content, encoding="utf-8")
                else:
                    # Create or overwrite
                    full_path.write_text(update.content, encoding="utf-8")

                written_files.append(update.file_path)
                self.logger.info(f"  Written: {update.file_path}")

            except Exception as e:
                self.logger.error(f"Failed to write {update.file_path}: {e}")

        return written_files

    # =========================================================================
    # SUMMARY GENERATION
    # =========================================================================

    def _generate_summary(
        self,
        context: AgentContext,
        updates: List[DocumentationUpdate],
        written_files: List[str]
    ) -> str:
        """Generate human-readable summary of documentation updates."""
        lines = [
            f"## Documentation Updated for Issue #{context.issue_data.get('number')}",
            "",
            f"**{context.issue_data.get('title', 'Unknown')}**",
            "",
            "### Files Updated",
        ]

        for file_path in written_files:
            update = next((u for u in updates if u.file_path == file_path), None)
            action = update.action if update else "updated"
            lines.append(f"- `{file_path}` ({action})")

        lines.extend([
            "",
            "### Updates Made",
        ])

        for update in updates:
            doc_type = update.doc_type.value.replace("_", " ").title()
            lines.append(f"- {doc_type}: {update.action}")

        return "\n".join(lines)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _slugify(self, text: str) -> str:
        """Convert text to URL-friendly slug."""
        # Remove special characters
        slug = re.sub(r'[^\w\s-]', '', text.lower())
        # Replace spaces with hyphens
        slug = re.sub(r'[\s_]+', '-', slug)
        # Remove multiple hyphens
        slug = re.sub(r'-+', '-', slug)
        # Trim length
        return slug[:50].strip('-')

    def _load_prompt(self, name: str) -> str:
        """Load a prompt template from the prompts directory."""
        if name in self._prompts:
            return self._prompts[name]

        prompt_path = Path(__file__).parent / "prompts" / f"{name}.md"

        if prompt_path.exists():
            self._prompts[name] = prompt_path.read_text(encoding="utf-8")
            return self._prompts[name]

        # Return default prompts
        defaults = {
            "implementation_summary": self._get_default_summary_prompt(),
            "feature_memory": self._get_default_feature_memory_prompt(),
        }

        return defaults.get(name, f"# {name}\n\nNo prompt template found.")

    def _get_default_summary_prompt(self) -> str:
        """Default prompt for generating implementation summary."""
        return """# Implementation Summary Generator

Generate a concise summary of the implementation.

## Issue Information

**Title:** {{issue_title}}

**Description:**
{{issue_body}}

## Files Modified

{{files_modified}}

## Your Task

Write a 2-3 sentence summary of what was implemented.

Focus on:
- What was added or changed
- The main technical approach
- Any notable decisions

Keep it concise and technical.

## Output

Return only the summary text, no formatting or headers.
"""

    def _get_default_feature_memory_prompt(self) -> str:
        """Default prompt for feature memory generation."""
        return """# Feature Memory Generator

Generate a comprehensive feature memory document.

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

## Your Task

Generate a feature memory document capturing:
1. Technical approach taken
2. Key decisions made
3. Files modified and why
4. Any lessons learned

## Output Format

Return the content in markdown format following the feature memory template.
"""


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "DocAgent",
    "DocumentationType",
    "HistoryEntryType",
    "HistoryEntry",
    "FeatureMemory",
    "ChangelogEntry",
    "DocumentationUpdate",
]
