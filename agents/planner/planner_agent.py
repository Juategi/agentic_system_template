# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - PLANNER AGENT IMPLEMENTATION
# =============================================================================
"""
Planner Agent Implementation

This agent analyzes feature issues and decomposes them into smaller tasks.
It uses an LLM to understand requirements and create a logical breakdown.

Decomposition Strategy:
    1. Parse feature description and acceptance criteria
    2. Identify distinct functional areas
    3. Break into tasks of appropriate granularity (2-8 hours)
    4. Identify dependencies between tasks
    5. Create GitHub issues for each task
    6. Link tasks to parent feature
    7. Create feature memory file
"""

import os
import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from agents.base import (
    AgentInterface,
    AgentResult,
    AgentContext,
    AgentStatus,
    LLMMessage,
)


logger = logging.getLogger(__name__)


# =============================================================================
# DECOMPOSITION SCHEMA
# =============================================================================

DECOMPOSITION_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "acceptance_criteria": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "estimate_hours": {"type": "number"},
                    "dependencies": {
                        "type": "array",
                        "items": {"type": "integer"}
                    },
                    "implementation_notes": {"type": "string"}
                },
                "required": ["title", "description", "acceptance_criteria", "estimate_hours"]
            }
        },
        "reasoning": {"type": "string"}
    },
    "required": ["tasks", "reasoning"]
}


# =============================================================================
# PLANNER AGENT CLASS
# =============================================================================

class PlannerAgent(AgentInterface):
    """
    Agent that decomposes features into tasks.

    This agent:
    1. Reads feature issue and project context
    2. Uses LLM to analyze and decompose
    3. Creates sub-task issues in GitHub
    4. Creates feature memory file
    5. Returns decomposition results

    Configuration:
        max_sub_issues: Maximum tasks to create (default: 10)
        min_granularity_hours: Minimum task size (default: 2)
        max_granularity_hours: Maximum task size (default: 8)
        create_dependencies: Whether to link dependencies (default: true)
        target_tasks: Target number of tasks (default: 5)
    """

    # Default configuration
    DEFAULT_CONFIG = {
        "max_sub_issues": 10,
        "min_granularity_hours": 2,
        "max_granularity_hours": 8,
        "create_dependencies": True,
        "target_tasks": 5
    }

    def __init__(self):
        """Initialize the Planner agent."""
        super().__init__()
        self._prompt_template: Optional[str] = None

    def get_agent_type(self) -> str:
        """Return agent type identifier."""
        return "planner"

    def validate_context(self, context: AgentContext) -> bool:
        """
        Validate context for planning.

        Requirements:
        - Issue must exist and be accessible
        - Issue should have a body with requirements
        - Project memory should be available (warning if not)
        """
        # Check issue data exists
        if not context.issue_data:
            self.logger.error("No issue data in context")
            return False

        # Check issue has body (requirements)
        if not context.issue_data.get("body"):
            self.logger.error("Issue has no body/description")
            return False

        # Check project context exists (warning only)
        if not context.has_memory_file("PROJECT.md"):
            self.logger.warning("PROJECT.md not found, proceeding with limited context")

        return True

    def execute(self, context: AgentContext) -> AgentResult:
        """
        Execute feature decomposition.

        Steps:
        1. Extract feature information
        2. Prepare LLM prompt
        3. Get decomposition from LLM
        4. Validate decomposition
        5. Create GitHub issues
        6. Create feature memory file
        7. Return results
        """
        try:
            # Get configuration
            config = {**self.DEFAULT_CONFIG, **context.config}

            # Step 1: Extract feature information
            self.logger.info("Extracting feature information...")
            feature_info = self._extract_feature_info(context)

            # Step 2: Prepare LLM prompt
            self.logger.info("Preparing decomposition prompt...")
            prompt = self._prepare_prompt(feature_info, context, config)

            # Step 3: Get decomposition from LLM
            self.logger.info("Getting decomposition from LLM...")
            decomposition = self._get_decomposition(prompt)
            self.track_llm_call(
                tokens_input=decomposition.get("_tokens_input", 0),
                tokens_output=decomposition.get("_tokens_output", 0)
            )

            # Step 4: Validate decomposition
            self.logger.info("Validating decomposition...")
            validation_result = self._validate_decomposition(decomposition, config)
            if not validation_result["valid"]:
                return AgentResult.failure(
                    f"Decomposition validation failed: {validation_result['errors']}",
                    output={"decomposition": decomposition},
                    details={"validation_errors": validation_result["errors"]}
                )

            # Step 5: Create GitHub issues
            self.logger.info("Creating GitHub issues...")
            created_issues = self._create_issues(
                decomposition["tasks"],
                context.issue_number,
                config
            )

            # Step 6: Create feature memory file
            self.logger.info("Creating feature memory file...")
            memory_path = self._create_feature_memory(
                feature_info,
                created_issues,
                decomposition,
                context
            )

            # Step 7: Return results
            return AgentResult(
                status=AgentStatus.SUCCESS,
                output={
                    "created_issues": created_issues,
                    "feature_memory_file": memory_path,
                    "decomposition_summary": decomposition.get("reasoning", "")
                },
                message=f"Decomposed feature into {len(created_issues)} tasks",
                details={
                    "feature_title": feature_info["title"],
                    "total_estimate_hours": sum(
                        issue.get("estimate_hours", 0) for issue in created_issues
                    )
                }
            )

        except Exception as e:
            self.logger.error(f"Decomposition failed: {e}")
            return AgentResult.error(
                f"Feature decomposition failed: {str(e)}",
                errors=[str(e)]
            )

    # =========================================================================
    # FEATURE EXTRACTION
    # =========================================================================

    def _extract_feature_info(self, context: AgentContext) -> Dict[str, Any]:
        """
        Extract feature information from issue.

        Returns:
            Dictionary with title, description, acceptance_criteria, constraints
        """
        issue = context.issue_data
        body = issue.get("body", "")

        return {
            "title": issue.get("title", "Untitled Feature"),
            "description": self._extract_description(body),
            "acceptance_criteria": self._extract_criteria(body),
            "constraints": self._extract_constraints(body),
            "issue_number": context.issue_number,
            "labels": issue.get("labels", [])
        }

    def _extract_description(self, body: str) -> str:
        """Extract main description from issue body."""
        if not body:
            return ""

        # Try to find description section
        patterns = [
            r"## Description\s*\n(.*?)(?=\n##|\Z)",
            r"## Overview\s*\n(.*?)(?=\n##|\Z)",
            r"## Summary\s*\n(.*?)(?=\n##|\Z)"
        ]

        for pattern in patterns:
            match = re.search(pattern, body, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # If no section found, take first paragraph
        paragraphs = body.split("\n\n")
        if paragraphs:
            # Skip if it's a heading
            first = paragraphs[0].strip()
            if not first.startswith("#"):
                return first

        return body[:1000]  # Fallback: first 1000 chars

    def _extract_criteria(self, body: str) -> List[str]:
        """Extract acceptance criteria as list."""
        if not body:
            return []

        criteria = []

        # Try to find acceptance criteria section
        patterns = [
            r"## Acceptance Criteria\s*\n(.*?)(?=\n##|\Z)",
            r"## Requirements\s*\n(.*?)(?=\n##|\Z)",
            r"## Criteria\s*\n(.*?)(?=\n##|\Z)"
        ]

        section_text = None
        for pattern in patterns:
            match = re.search(pattern, body, re.DOTALL | re.IGNORECASE)
            if match:
                section_text = match.group(1)
                break

        if section_text:
            # Extract bullet points
            for line in section_text.split("\n"):
                line = line.strip()
                if line.startswith(("-", "*", "+")):
                    text = line.lstrip("-*+ ").strip()
                    # Remove checkbox if present
                    if text.startswith("[ ]") or text.startswith("[x]"):
                        text = text[3:].strip()
                    if text:
                        criteria.append(text)

        return criteria

    def _extract_constraints(self, body: str) -> List[str]:
        """Extract technical constraints from issue body."""
        if not body:
            return []

        constraints = []

        # Look for constraints section
        patterns = [
            r"## Constraints\s*\n(.*?)(?=\n##|\Z)",
            r"## Technical Constraints\s*\n(.*?)(?=\n##|\Z)",
            r"## Limitations\s*\n(.*?)(?=\n##|\Z)"
        ]

        for pattern in patterns:
            match = re.search(pattern, body, re.DOTALL | re.IGNORECASE)
            if match:
                section_text = match.group(1)
                for line in section_text.split("\n"):
                    line = line.strip()
                    if line.startswith(("-", "*", "+")):
                        constraints.append(line.lstrip("-*+ ").strip())
                break

        return constraints

    # =========================================================================
    # PROMPT PREPARATION
    # =========================================================================

    def _prepare_prompt(
        self,
        feature_info: Dict[str, Any],
        context: AgentContext,
        config: Dict[str, Any]
    ) -> str:
        """Prepare the decomposition prompt."""
        template = self._load_prompt_template()

        # Format acceptance criteria
        criteria_text = "\n".join(
            f"- {c}" for c in feature_info["acceptance_criteria"]
        ) or "No specific criteria provided"

        # Get project context
        project_context = context.get_memory_file("PROJECT.md")
        if not project_context:
            project_context = "No project context available"

        architecture = context.get_memory_file("ARCHITECTURE.md")
        if not architecture:
            architecture = "No architecture documentation available"

        # Fill template
        prompt = template.replace("{{feature_title}}", feature_info["title"])
        prompt = prompt.replace("{{feature_description}}", feature_info["description"])
        prompt = prompt.replace("{{acceptance_criteria}}", criteria_text)
        prompt = prompt.replace("{{project_context}}", project_context[:3000])  # Limit size
        prompt = prompt.replace("{{architecture}}", architecture[:2000])
        prompt = prompt.replace("{{min_hours}}", str(config["min_granularity_hours"]))
        prompt = prompt.replace("{{max_hours}}", str(config["max_granularity_hours"]))
        prompt = prompt.replace("{{target_tasks}}", str(config["target_tasks"]))

        return prompt

    def _load_prompt_template(self) -> str:
        """Load the decomposition prompt template."""
        if self._prompt_template:
            return self._prompt_template

        # Try to load from file
        prompt_path = Path(__file__).parent / "prompts" / "decomposition.md"

        if prompt_path.exists():
            with open(prompt_path, "r", encoding="utf-8") as f:
                self._prompt_template = f.read()
        else:
            # Fallback inline template
            self._prompt_template = self._get_fallback_prompt()

        return self._prompt_template

    def _get_fallback_prompt(self) -> str:
        """Return fallback prompt if file not found."""
        return """
You are a software architect. Decompose this feature into smaller tasks.

## Feature
**Title:** {{feature_title}}
**Description:** {{feature_description}}

## Acceptance Criteria
{{acceptance_criteria}}

## Output Format
Return JSON:
```json
{
  "tasks": [
    {
      "title": "Task title",
      "description": "What to implement",
      "acceptance_criteria": ["Criterion 1", "Criterion 2"],
      "estimate_hours": 4,
      "dependencies": [],
      "implementation_notes": "Hints"
    }
  ],
  "reasoning": "Why this breakdown"
}
```

Create {{target_tasks}} tasks, each {{min_hours}}-{{max_hours}} hours.
"""

    # =========================================================================
    # LLM INTERACTION
    # =========================================================================

    def _get_decomposition(self, prompt: str) -> Dict[str, Any]:
        """Get decomposition from LLM."""
        system_prompt = """You are an expert software architect specializing in breaking down
complex features into manageable development tasks. You always respond with valid JSON
that follows the requested schema exactly. Your decompositions are practical, well-structured,
and enable parallel development where possible."""

        response = self.llm.complete(
            prompt=prompt,
            system=system_prompt,
            max_tokens=4096,
            temperature=0.3  # Lower temperature for more consistent output
        )

        # Parse JSON from response
        content = response.content

        # Try to extract JSON from markdown code block if present
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
        if json_match:
            content = json_match.group(1)

        try:
            decomposition = json.loads(content)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response as JSON: {e}")
            self.logger.debug(f"Response content: {content[:500]}")
            raise ValueError(f"Invalid JSON in LLM response: {e}")

        # Add token metrics
        decomposition["_tokens_input"] = response.tokens_input
        decomposition["_tokens_output"] = response.tokens_output

        return decomposition

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def _validate_decomposition(
        self,
        decomposition: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate the decomposition output.

        Checks:
        - Has required fields
        - Task count within limits
        - Each task has required fields
        - Dependencies are valid
        - No circular dependencies
        """
        errors = []

        # Check required fields
        if "tasks" not in decomposition:
            errors.append("Missing 'tasks' field")
            return {"valid": False, "errors": errors}

        tasks = decomposition["tasks"]

        # Check task count
        if len(tasks) == 0:
            errors.append("No tasks generated")
        elif len(tasks) > config["max_sub_issues"]:
            errors.append(f"Too many tasks: {len(tasks)} > {config['max_sub_issues']}")

        # Validate each task
        for i, task in enumerate(tasks):
            if not task.get("title"):
                errors.append(f"Task {i}: missing title")
            if not task.get("description"):
                errors.append(f"Task {i}: missing description")
            if not task.get("acceptance_criteria"):
                errors.append(f"Task {i}: missing acceptance_criteria")

            # Check estimate
            estimate = task.get("estimate_hours", 0)
            if estimate < config["min_granularity_hours"]:
                errors.append(
                    f"Task {i}: estimate too small ({estimate}h < {config['min_granularity_hours']}h)"
                )
            if estimate > config["max_granularity_hours"]:
                errors.append(
                    f"Task {i}: estimate too large ({estimate}h > {config['max_granularity_hours']}h)"
                )

            # Validate dependencies reference valid tasks
            deps = task.get("dependencies", [])
            for dep in deps:
                if not isinstance(dep, int) or dep < 0 or dep >= len(tasks):
                    errors.append(f"Task {i}: invalid dependency {dep}")
                if dep == i:
                    errors.append(f"Task {i}: self-dependency")

        # Check for circular dependencies
        if not self._validate_dependencies(tasks):
            errors.append("Circular dependencies detected")

        return {
            "valid": len(errors) == 0,
            "errors": errors
        }

    def _validate_dependencies(self, tasks: List[Dict]) -> bool:
        """
        Validate dependency graph has no cycles using topological sort.

        Returns True if valid (no cycles), False if cycles exist.
        """
        n = len(tasks)
        in_degree = [0] * n
        adj = [[] for _ in range(n)]

        # Build adjacency list
        for i, task in enumerate(tasks):
            for dep in task.get("dependencies", []):
                if 0 <= dep < n:
                    adj[dep].append(i)
                    in_degree[i] += 1

        # Kahn's algorithm
        queue = [i for i in range(n) if in_degree[i] == 0]
        processed = 0

        while queue:
            node = queue.pop(0)
            processed += 1
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return processed == n  # All nodes processed = no cycle

    # =========================================================================
    # GITHUB ISSUE CREATION
    # =========================================================================

    def _create_issues(
        self,
        tasks: List[Dict],
        parent_issue: int,
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create GitHub issues for each task."""
        created_issues = []
        task_to_issue: Dict[int, int] = {}  # Map task index to issue number

        for i, task in enumerate(tasks):
            # Format issue body
            body = self._format_task_body(task, parent_issue, task_to_issue)

            # Determine labels
            labels = ["task", "READY"]
            if task.get("dependencies"):
                # Tasks with dependencies start as pending
                labels = ["task", "pending-dependency"]

            try:
                # Create issue
                issue_number = self.github.create_issue(
                    title=f"[Task] {task['title']}",
                    body=body,
                    labels=labels
                )

                task_to_issue[i] = issue_number

                created_issues.append({
                    "number": issue_number,
                    "title": task["title"],
                    "estimate_hours": task.get("estimate_hours", 0),
                    "dependencies": [
                        task_to_issue.get(d) for d in task.get("dependencies", [])
                        if d in task_to_issue
                    ]
                })

                self.logger.info(f"Created issue #{issue_number}: {task['title']}")

            except Exception as e:
                self.logger.error(f"Failed to create issue for task {i}: {e}")
                # Continue with other tasks

        # Add summary comment to parent issue
        if created_issues:
            self._add_summary_comment(parent_issue, created_issues)

        return created_issues

    def _format_task_body(
        self,
        task: Dict,
        parent_issue: int,
        task_to_issue: Dict[int, int]
    ) -> str:
        """Format task as GitHub issue body."""
        lines = [
            "## Description",
            "",
            task.get("description", "No description provided"),
            "",
            "## Acceptance Criteria",
            ""
        ]

        for criterion in task.get("acceptance_criteria", []):
            lines.append(f"- [ ] {criterion}")

        lines.extend([
            "",
            "## Parent Feature",
            "",
            f"Part of #{parent_issue}",
            ""
        ])

        # Add dependencies if any
        deps = task.get("dependencies", [])
        if deps:
            lines.extend([
                "## Dependencies",
                "",
                "This task depends on:",
                ""
            ])
            for dep_idx in deps:
                if dep_idx in task_to_issue:
                    lines.append(f"- #{task_to_issue[dep_idx]}")
                else:
                    lines.append(f"- Task {dep_idx + 1} (pending creation)")
            lines.append("")

        # Add estimate
        estimate = task.get("estimate_hours", 0)
        if estimate:
            lines.extend([
                "## Estimate",
                "",
                f"~{estimate} hours",
                ""
            ])

        # Add implementation notes if present
        notes = task.get("implementation_notes", "")
        if notes:
            lines.extend([
                "## Implementation Notes",
                "",
                notes,
                ""
            ])

        return "\n".join(lines)

    def _add_summary_comment(
        self,
        parent_issue: int,
        created_issues: List[Dict]
    ):
        """Add summary comment to parent feature issue."""
        lines = [
            "## Planning Complete",
            "",
            f"This feature has been decomposed into {len(created_issues)} tasks:",
            ""
        ]

        total_hours = 0
        for issue in created_issues:
            hours = issue.get("estimate_hours", 0)
            total_hours += hours
            lines.append(f"- #{issue['number']}: {issue['title']} (~{hours}h)")

        lines.extend([
            "",
            f"**Total estimate:** ~{total_hours} hours",
            "",
            "---",
            "*Generated by Planner Agent*"
        ])

        try:
            self.github.add_comment(parent_issue, "\n".join(lines))
        except Exception as e:
            self.logger.warning(f"Failed to add summary comment: {e}")

    # =========================================================================
    # FEATURE MEMORY
    # =========================================================================

    def _create_feature_memory(
        self,
        feature_info: Dict[str, Any],
        created_issues: List[Dict],
        decomposition: Dict[str, Any],
        context: AgentContext
    ) -> str:
        """Create feature memory markdown file."""
        timestamp = datetime.utcnow().isoformat()

        lines = [
            f"# Feature: {feature_info['title']}",
            "",
            "## Metadata",
            "",
            f"- **Issue:** #{feature_info['issue_number']}",
            f"- **Status:** PLANNING_COMPLETE",
            f"- **Created:** {timestamp}",
            f"- **Tasks:** {len(created_issues)}",
            ""
        ]

        # Description
        lines.extend([
            "## Description",
            "",
            feature_info.get("description", "No description"),
            ""
        ])

        # Acceptance Criteria
        lines.append("## Acceptance Criteria")
        lines.append("")
        for criterion in feature_info.get("acceptance_criteria", []):
            lines.append(f"- [ ] {criterion}")
        lines.append("")

        # Tasks table
        lines.extend([
            "## Tasks",
            "",
            "| # | Title | Status | Estimate |",
            "|---|-------|--------|----------|"
        ])

        for issue in created_issues:
            lines.append(
                f"| #{issue['number']} | {issue['title']} | READY | {issue.get('estimate_hours', '?')}h |"
            )
        lines.append("")

        # Dependencies
        has_deps = any(issue.get("dependencies") for issue in created_issues)
        if has_deps:
            lines.extend([
                "## Dependencies",
                "",
                "```",
            ])
            for issue in created_issues:
                deps = issue.get("dependencies", [])
                if deps:
                    dep_str = ", ".join(f"#{d}" for d in deps)
                    lines.append(f"#{issue['number']} depends on: {dep_str}")
            lines.extend(["```", ""])

        # Planning Notes
        reasoning = decomposition.get("reasoning", "")
        if reasoning:
            lines.extend([
                "## Planning Notes",
                "",
                reasoning,
                ""
            ])

        # History
        lines.extend([
            "## History",
            "",
            "| Date | Event | Details |",
            "|------|-------|---------|",
            f"| {timestamp[:10]} | Planned | Created {len(created_issues)} tasks |",
            ""
        ])

        content = "\n".join(lines)

        # Write to memory path
        memory_path = f"features/feature-{feature_info['issue_number']}.md"
        full_path = os.path.join(
            os.environ.get("MEMORY_PATH", "/memory"),
            memory_path
        )

        # Ensure directory exists
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        try:
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            self.logger.info(f"Created feature memory: {memory_path}")
        except Exception as e:
            self.logger.error(f"Failed to write feature memory: {e}")

        return memory_path
