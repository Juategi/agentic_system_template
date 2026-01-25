# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - DEVELOPER AGENT IMPLEMENTATION
# =============================================================================
"""
Developer Agent Implementation

This agent writes code to implement task requirements. It uses an LLM
to understand requirements and generate appropriate code changes.

Implementation Strategy:
    1. Parse task requirements and acceptance criteria
    2. Analyze related existing code
    3. Plan implementation approach
    4. Generate code changes
    5. Verify changes compile/parse
    6. Write changes to repository

LLM Interaction:
    The agent uses a multi-step approach:
    1. Understanding: Analyze requirements and codebase
    2. Planning: Design implementation approach
    3. Coding: Generate actual code
    4. Review: Self-review for obvious issues

File Operations:
    - Reads existing files for context
    - Creates new files as needed
    - Modifies existing files
    - Does NOT delete files (safety measure)
"""

import os
import re
import ast
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from agents.base import (
    AgentInterface,
    AgentResult,
    AgentContext,
    AgentStatus,
)


logger = logging.getLogger(__name__)


# =============================================================================
# SCHEMAS
# =============================================================================

IMPLEMENTATION_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "approach": {"type": "string"},
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "action": {"type": "string", "enum": ["create", "modify"]},
                    "description": {"type": "string"},
                    "dependencies": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["path", "action", "description"]
            }
        },
        "tests": {
            "type": "array",
            "items": {"type": "string"}
        },
        "risks": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["approach", "files"]
}

CODE_CHANGE_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "action": {"type": "string"},
        "content": {"type": "string"},
        "explanation": {"type": "string"}
    },
    "required": ["path", "action", "content"]
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TaskInfo:
    """Parsed task information from issue."""
    title: str
    description: str
    acceptance_criteria: List[str]
    implementation_hints: List[str]
    mentioned_files: List[str]
    parent_feature: Optional[int] = None


@dataclass
class FileChange:
    """Represents a change to a file."""
    path: str
    action: str  # "create" or "modify"
    content: str
    original_content: Optional[str] = None
    explanation: str = ""

    @property
    def lines_changed(self) -> int:
        """Estimate lines changed."""
        if self.action == "create":
            return self.content.count('\n') + 1
        elif self.original_content:
            # Simple diff count
            original_lines = set(self.original_content.splitlines())
            new_lines = set(self.content.splitlines())
            return len(original_lines.symmetric_difference(new_lines))
        return self.content.count('\n') + 1


@dataclass
class ImplementationPlan:
    """Plan for implementing the task."""
    approach: str
    files: List[Dict[str, Any]]
    tests: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)


# =============================================================================
# DEVELOPER AGENT CLASS
# =============================================================================

class DeveloperAgent(AgentInterface):
    """
    Agent that implements code changes.

    Attributes:
        llm: LLM client for code generation
        github: GitHub API helper
        repo_path: Path to repository volume
        config: Agent configuration

    Configuration:
        max_files_modified: Maximum files to change (default: 20)
        max_lines_per_file: Maximum lines per file (default: 500)
        create_branch: Whether to create git branch (default: true)
        run_linter: Whether to run linter after changes (default: true)
    """

    DEFAULT_CONFIG = {
        "max_files_modified": 20,
        "max_lines_per_file": 500,
        "create_branch": True,
        "run_linter": True,
        "verify_syntax": True,
        "max_retries_per_file": 2,
        "temperature": 0.3,  # Lower temperature for code generation
    }

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Developer Agent."""
        super().__init__()
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self._repo_path: Optional[str] = None
        self._changes: List[FileChange] = []

    def get_agent_type(self) -> str:
        """Return agent type identifier."""
        return "developer"

    @property
    def repo_path(self) -> str:
        """Get repository path from environment."""
        if not self._repo_path:
            self._repo_path = os.environ.get("REPO_PATH", "/repo")
        return self._repo_path

    def validate_context(self, context: AgentContext) -> bool:
        """
        Validate context for development.

        Requirements:
        - Task issue exists with requirements
        - Repository is accessible
        - Conventions are available (warning if not)
        """
        if not context.issue_data:
            self.logger.error("No issue data in context")
            return False

        if not context.issue_data.get("body"):
            self.logger.error("Issue has no body/description")
            return False

        # Check repository access
        if not os.path.isdir(self.repo_path):
            self.logger.error(f"Repository path not accessible: {self.repo_path}")
            return False

        # Warning if no conventions (but don't fail)
        if "CONVENTIONS.md" not in context.memory:
            self.logger.warning("No CONVENTIONS.md found - code style may be inconsistent")

        return True

    def execute(self, context: AgentContext) -> AgentResult:
        """
        Execute code implementation.

        Steps:
        1. Parse task requirements
        2. Load QA feedback if retry
        3. Analyze relevant codebase
        4. Plan implementation
        5. Generate code changes
        6. Apply changes to files
        7. Verify changes (syntax check)
        8. Create git branch/commit
        9. Return results
        """
        try:
            # =========================================================
            # STEP 1: Parse Task Requirements
            # =========================================================
            self.logger.info("Parsing task requirements...")
            task_info = self._parse_task_requirements(context)

            # =========================================================
            # STEP 2: Load QA Feedback (if retry)
            # =========================================================
            qa_feedback = None
            if context.iteration > 0:
                self.logger.info(f"Loading QA feedback (iteration {context.iteration})...")
                qa_feedback = self._load_qa_feedback(context)

            # =========================================================
            # STEP 3: Analyze Relevant Codebase
            # =========================================================
            self.logger.info("Analyzing relevant codebase...")
            relevant_code = self._find_relevant_code(task_info, context)

            # =========================================================
            # STEP 4: Plan Implementation
            # =========================================================
            self.logger.info("Planning implementation...")
            plan = self._create_implementation_plan(
                task_info,
                relevant_code,
                context,
                qa_feedback
            )

            # Validate plan limits
            if len(plan.files) > self.config["max_files_modified"]:
                return AgentResult(
                    status=AgentStatus.FAILURE,
                    output={"error": "Too many files to modify"},
                    message=f"Plan requires {len(plan.files)} files, max is {self.config['max_files_modified']}"
                )

            # =========================================================
            # STEP 5: Generate Code Changes
            # =========================================================
            self.logger.info("Generating code changes...")
            changes = self._generate_code_changes(plan, task_info, context, qa_feedback)

            # =========================================================
            # STEP 6: Apply Changes to Files
            # =========================================================
            self.logger.info("Applying changes to files...")
            applied_files = self._apply_changes(changes)

            # =========================================================
            # STEP 7: Verify Changes
            # =========================================================
            if self.config["verify_syntax"]:
                self.logger.info("Verifying changes...")
                verification = self._verify_changes(applied_files)

                if not verification["success"]:
                    self.logger.warning(f"Verification issues: {verification['errors']}")
                    # Attempt to fix syntax errors
                    fixed = self._attempt_fix_errors(verification["errors"], changes)
                    if not fixed:
                        return AgentResult(
                            status=AgentStatus.FAILURE,
                            output={
                                "error": "Code verification failed",
                                "verification_errors": verification["errors"]
                            },
                            message="Generated code has syntax errors that could not be fixed"
                        )

            # =========================================================
            # STEP 8: Git Operations
            # =========================================================
            git_info = {}
            if self.config["create_branch"]:
                self.logger.info("Handling git operations...")
                git_info = self._handle_git_operations(context.issue_number, applied_files, task_info)

            # =========================================================
            # STEP 9: Return Results
            # =========================================================
            commit_message = self._generate_commit_message(task_info, changes)
            tests_added = self._identify_tests(applied_files)

            modified_files_output = [
                {
                    "path": f.path,
                    "action": f.action,
                    "lines_changed": f.lines_changed,
                    "explanation": f.explanation
                }
                for f in applied_files
            ]

            return AgentResult(
                status=AgentStatus.SUCCESS,
                output={
                    "modified_files": modified_files_output,
                    "commit_message": commit_message,
                    "branch_name": git_info.get("branch"),
                    "implementation_notes": plan.approach,
                    "tests_added": tests_added,
                    "plan": {
                        "approach": plan.approach,
                        "risks": plan.risks
                    }
                },
                message=f"Implemented task: {len(applied_files)} files modified",
                metrics={
                    "files_modified": len(applied_files),
                    "files_created": sum(1 for f in applied_files if f.action == "create"),
                    "lines_changed": sum(f.lines_changed for f in applied_files),
                    "tests_added": len(tests_added)
                }
            )

        except Exception as e:
            self.logger.exception(f"Development failed: {e}")
            return AgentResult(
                status=AgentStatus.FAILURE,
                output={"error": str(e)},
                message=f"Development failed: {e}"
            )

    # =========================================================================
    # TASK PARSING
    # =========================================================================

    def _parse_task_requirements(self, context: AgentContext) -> TaskInfo:
        """Extract structured task information from issue."""
        issue = context.issue_data
        body = issue.get("body", "")

        # Extract acceptance criteria
        criteria = self._extract_acceptance_criteria(body)

        # Extract implementation hints
        hints = self._extract_implementation_hints(body)

        # Extract mentioned files
        mentioned_files = self._extract_mentioned_files(body)

        # Extract parent feature reference
        parent_feature = self._extract_parent_feature(body, issue.get("labels", []))

        return TaskInfo(
            title=issue.get("title", ""),
            description=self._extract_description(body),
            acceptance_criteria=criteria,
            implementation_hints=hints,
            mentioned_files=mentioned_files,
            parent_feature=parent_feature
        )

    def _extract_description(self, body: str) -> str:
        """Extract main description from issue body."""
        # Remove sections to get main description
        lines = []
        in_section = False

        for line in body.split('\n'):
            # Check for section headers
            if re.match(r'^#+\s+(Acceptance|Implementation|Technical|Dependencies)', line, re.IGNORECASE):
                in_section = True
            elif re.match(r'^#+\s+', line):
                in_section = False

            if not in_section:
                lines.append(line)

        return '\n'.join(lines).strip()

    def _extract_acceptance_criteria(self, body: str) -> List[str]:
        """Extract acceptance criteria from issue body."""
        criteria = []

        # Look for acceptance criteria section
        pattern = r'(?:##?\s*Acceptance\s*Criteria|##?\s*Requirements)\s*\n(.*?)(?=\n##|\Z)'
        match = re.search(pattern, body, re.IGNORECASE | re.DOTALL)

        if match:
            section = match.group(1)
            # Extract bullet points or checkboxes
            for line in section.split('\n'):
                line = line.strip()
                if re.match(r'^[-*]\s+\[.\]|^[-*]\s+|^\d+\.\s+', line):
                    # Clean up the criterion
                    criterion = re.sub(r'^[-*]\s*\[.\]\s*|^[-*]\s*|^\d+\.\s*', '', line)
                    if criterion:
                        criteria.append(criterion)

        return criteria

    def _extract_implementation_hints(self, body: str) -> List[str]:
        """Extract implementation hints from issue body."""
        hints = []

        # Look for implementation notes section
        pattern = r'(?:##?\s*Implementation\s*(?:Notes|Hints)?|##?\s*Technical\s*Notes?)\s*\n(.*?)(?=\n##|\Z)'
        match = re.search(pattern, body, re.IGNORECASE | re.DOTALL)

        if match:
            section = match.group(1)
            for line in section.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    hints.append(line)

        return hints

    def _extract_mentioned_files(self, body: str) -> List[str]:
        """Extract file paths mentioned in issue body."""
        # Match common file patterns
        patterns = [
            r'`([^`]+\.[a-zA-Z]+)`',  # Backticked file names
            r'(?:^|\s)([\w/\\.-]+\.(?:py|js|ts|java|go|rs|rb|php|css|html|json|yaml|yml|md))\b',  # File extensions
        ]

        files = set()
        for pattern in patterns:
            matches = re.findall(pattern, body)
            files.update(matches)

        return list(files)

    def _extract_parent_feature(self, body: str, labels: List[Dict]) -> Optional[int]:
        """Extract parent feature issue number."""
        # Check labels for parent reference
        for label in labels:
            label_name = label.get("name", "") if isinstance(label, dict) else str(label)
            match = re.match(r'feature[:-](\d+)', label_name, re.IGNORECASE)
            if match:
                return int(match.group(1))

        # Check body for parent reference
        pattern = r'(?:Parent|Feature)\s*(?:Issue)?[:#]\s*#?(\d+)'
        match = re.search(pattern, body, re.IGNORECASE)
        if match:
            return int(match.group(1))

        return None

    # =========================================================================
    # QA FEEDBACK HANDLING
    # =========================================================================

    def _load_qa_feedback(self, context: AgentContext) -> Optional[Dict[str, Any]]:
        """Load QA feedback from previous iteration."""
        if not context.input_data:
            return None

        return context.input_data.get("qa_feedback", {})

    # =========================================================================
    # CODE ANALYSIS
    # =========================================================================

    def _find_relevant_code(
        self,
        task_info: TaskInfo,
        context: AgentContext
    ) -> Dict[str, str]:
        """
        Find and load relevant code files.

        Uses LLM to identify which files are relevant based on task.
        """
        relevant = {}

        # Start with explicitly mentioned files
        for file_path in task_info.mentioned_files:
            full_path = os.path.join(self.repo_path, file_path)
            if os.path.isfile(full_path):
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        relevant[file_path] = f.read()
                except Exception as e:
                    self.logger.warning(f"Could not read {file_path}: {e}")

        # Use LLM to identify additional relevant files
        file_list = self._get_repository_files()

        if file_list:
            prompt = self._format_file_identification_prompt(task_info, file_list)
            response = self.llm.complete(
                prompt=prompt,
                system="You are a code analyst. Identify files relevant to the given task.",
                max_tokens=1000,
                temperature=0.2
            )

            # Parse LLM response for file paths
            additional_files = self._parse_file_list(response.content, file_list)

            for file_path in additional_files[:10]:  # Limit to 10 additional files
                if file_path not in relevant:
                    full_path = os.path.join(self.repo_path, file_path)
                    if os.path.isfile(full_path):
                        try:
                            with open(full_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                # Limit file size
                                if len(content) < 50000:
                                    relevant[file_path] = content
                        except Exception as e:
                            self.logger.warning(f"Could not read {file_path}: {e}")

        return relevant

    def _get_repository_files(self) -> List[str]:
        """Get list of files in repository."""
        files = []
        exclude_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'dist', 'build'}
        exclude_extensions = {'.pyc', '.pyo', '.exe', '.dll', '.so', '.o', '.a', '.lib'}

        for root, dirs, filenames in os.walk(self.repo_path):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]

            for filename in filenames:
                # Skip hidden files and excluded extensions
                if filename.startswith('.'):
                    continue
                if any(filename.endswith(ext) for ext in exclude_extensions):
                    continue

                rel_path = os.path.relpath(os.path.join(root, filename), self.repo_path)
                files.append(rel_path.replace('\\', '/'))

        return files

    def _format_file_identification_prompt(self, task_info: TaskInfo, file_list: List[str]) -> str:
        """Format prompt for file identification."""
        return f"""Given this task, identify which files from the repository are likely relevant.

## Task
**Title:** {task_info.title}
**Description:** {task_info.description}

## Acceptance Criteria
{chr(10).join(f'- {c}' for c in task_info.acceptance_criteria)}

## Repository Files
{chr(10).join(file_list[:200])}  # Limit to first 200 files

## Instructions
List the file paths that are most likely relevant to implementing this task.
Include files that:
1. Would need to be modified
2. Contain related functionality to understand
3. Define interfaces or types that will be used
4. Contain tests that should be updated

Return ONLY a list of file paths, one per line. No explanations."""

    def _parse_file_list(self, response: str, valid_files: List[str]) -> List[str]:
        """Parse LLM response for valid file paths."""
        files = []
        valid_set = set(valid_files)

        for line in response.strip().split('\n'):
            line = line.strip()
            # Remove common prefixes
            line = re.sub(r'^[-*\d.)\s]+', '', line)
            line = line.strip('`')

            if line in valid_set:
                files.append(line)

        return files

    # =========================================================================
    # IMPLEMENTATION PLANNING
    # =========================================================================

    def _create_implementation_plan(
        self,
        task_info: TaskInfo,
        relevant_code: Dict[str, str],
        context: AgentContext,
        qa_feedback: Optional[Dict[str, Any]] = None
    ) -> ImplementationPlan:
        """Create a plan for implementing the task."""
        # Load prompt template
        prompt = self._format_planning_prompt(task_info, relevant_code, context, qa_feedback)

        response = self.llm.complete(
            prompt=prompt,
            system=self._get_planning_system_prompt(),
            max_tokens=2000,
            temperature=0.3
        )

        # Parse JSON response
        plan_data = self._extract_json(response.content)

        if not plan_data:
            # Fallback: create basic plan
            self.logger.warning("Could not parse LLM plan, using fallback")
            plan_data = {
                "approach": "Direct implementation based on requirements",
                "files": [{"path": f, "action": "modify", "description": "Update"}
                         for f in relevant_code.keys()][:5]
            }

        return ImplementationPlan(
            approach=plan_data.get("approach", ""),
            files=plan_data.get("files", []),
            tests=plan_data.get("tests", []),
            risks=plan_data.get("risks", [])
        )

    def _get_planning_system_prompt(self) -> str:
        """Get system prompt for planning."""
        return """You are a senior software developer creating an implementation plan.

Your plan should be:
1. Specific and actionable
2. Following existing code patterns
3. Minimizing changes needed
4. Including appropriate tests

Return your plan as JSON with this structure:
{
  "approach": "High-level description of implementation approach",
  "files": [
    {
      "path": "path/to/file.py",
      "action": "create" or "modify",
      "description": "What changes are needed",
      "dependencies": ["other/file.py"]
    }
  ],
  "tests": ["Description of tests to add"],
  "risks": ["Potential risks or challenges"]
}"""

    def _format_planning_prompt(
        self,
        task_info: TaskInfo,
        relevant_code: Dict[str, str],
        context: AgentContext,
        qa_feedback: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format the planning prompt."""
        # Build context sections
        code_context = ""
        for path, content in list(relevant_code.items())[:5]:  # Limit files
            code_context += f"\n### {path}\n```\n{content[:3000]}\n```\n"

        conventions = context.memory.get("CONVENTIONS.md", "No conventions file found.")
        architecture = context.memory.get("ARCHITECTURE.md", "No architecture file found.")

        qa_section = ""
        if qa_feedback:
            qa_section = f"""
## Previous QA Feedback (Iteration {context.iteration})
This is a retry. Address these issues:
- Issues Found: {qa_feedback.get('issues', [])}
- Suggestions: {qa_feedback.get('suggestions', [])}
"""

        return f"""## Task to Implement

**Title:** {task_info.title}

**Description:**
{task_info.description}

**Acceptance Criteria:**
{chr(10).join(f'- [ ] {c}' for c in task_info.acceptance_criteria)}

**Implementation Hints:**
{chr(10).join(f'- {h}' for h in task_info.implementation_hints)}
{qa_section}
## Project Conventions
{conventions[:2000]}

## Architecture
{architecture[:2000]}

## Relevant Code
{code_context}

## Instructions
Create an implementation plan for this task. Consider:
1. Which files need to be created or modified
2. The order of changes (dependencies)
3. What tests should be added
4. Potential risks or challenges

Return your plan as JSON."""

    # =========================================================================
    # CODE GENERATION
    # =========================================================================

    def _generate_code_changes(
        self,
        plan: ImplementationPlan,
        task_info: TaskInfo,
        context: AgentContext,
        qa_feedback: Optional[Dict[str, Any]] = None
    ) -> List[FileChange]:
        """Generate code changes for each file in the plan."""
        changes = []

        for file_plan in plan.files:
            file_path = file_plan["path"]
            action = file_plan["action"]

            self.logger.info(f"Generating {action} for {file_path}")

            # Read existing content if modifying
            original_content = None
            if action == "modify":
                full_path = os.path.join(self.repo_path, file_path)
                if os.path.isfile(full_path):
                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            original_content = f.read()
                    except Exception as e:
                        self.logger.warning(f"Could not read {file_path}: {e}")

            # Generate code
            new_content, explanation = self._generate_file_content(
                file_path=file_path,
                action=action,
                description=file_plan.get("description", ""),
                original_content=original_content,
                task_info=task_info,
                context=context,
                qa_feedback=qa_feedback
            )

            if new_content:
                changes.append(FileChange(
                    path=file_path,
                    action=action,
                    content=new_content,
                    original_content=original_content,
                    explanation=explanation
                ))

        return changes

    def _generate_file_content(
        self,
        file_path: str,
        action: str,
        description: str,
        original_content: Optional[str],
        task_info: TaskInfo,
        context: AgentContext,
        qa_feedback: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        """Generate content for a single file."""
        prompt = self._format_coding_prompt(
            file_path=file_path,
            action=action,
            description=description,
            original_content=original_content,
            task_info=task_info,
            context=context,
            qa_feedback=qa_feedback
        )

        response = self.llm.complete(
            prompt=prompt,
            system=self._get_coding_system_prompt(file_path),
            max_tokens=4000,
            temperature=self.config["temperature"]
        )

        # Extract code from response
        content, explanation = self._extract_code_and_explanation(response.content, file_path)

        return content, explanation

    def _get_coding_system_prompt(self, file_path: str) -> str:
        """Get system prompt for code generation."""
        ext = os.path.splitext(file_path)[1].lower()

        language_hints = {
            ".py": "Python with type hints, following PEP 8",
            ".js": "Modern JavaScript (ES6+)",
            ".ts": "TypeScript with proper types",
            ".java": "Java following standard conventions",
            ".go": "Go following standard conventions",
            ".rs": "Rust following standard conventions",
        }

        lang_hint = language_hints.get(ext, "the appropriate language")

        return f"""You are an expert programmer writing production-quality code in {lang_hint}.

Your code must:
1. Be complete and working - no placeholders or TODOs
2. Follow existing code patterns and conventions
3. Include appropriate error handling
4. Be well-documented with clear comments where needed
5. Be efficient and readable

IMPORTANT: Return the COMPLETE file content, not just the changes.
Wrap your code in a code block with the appropriate language tag.
After the code block, provide a brief explanation of what was changed/created."""

    def _format_coding_prompt(
        self,
        file_path: str,
        action: str,
        description: str,
        original_content: Optional[str],
        task_info: TaskInfo,
        context: AgentContext,
        qa_feedback: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format the coding prompt."""
        original_section = ""
        if original_content:
            original_section = f"""
## Current File Content
```
{original_content}
```
"""

        qa_section = ""
        if qa_feedback:
            qa_section = f"""
## QA Feedback to Address
{json.dumps(qa_feedback, indent=2)}
"""

        conventions = context.memory.get("CONVENTIONS.md", "")

        return f"""## Task
{action.upper()} file: {file_path}

**Description:** {description}

## Overall Task Context
**Title:** {task_info.title}
**Description:** {task_info.description}

**Acceptance Criteria:**
{chr(10).join(f'- {c}' for c in task_info.acceptance_criteria)}
{original_section}{qa_section}
## Coding Conventions
{conventions[:1500]}

## Instructions
{"Create a new" if action == "create" else "Modify the existing"} file `{file_path}`.

{f"The file should: {description}" if description else ""}

Return the COMPLETE file content in a code block.
Then explain what you created/changed."""

    def _extract_code_and_explanation(self, response: str, file_path: str) -> Tuple[str, str]:
        """Extract code and explanation from LLM response."""
        # Find code block
        ext = os.path.splitext(file_path)[1].lower()
        lang_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
        }
        expected_lang = lang_map.get(ext, "")

        # Try to match code block with language
        pattern = rf'```(?:{expected_lang}|{ext[1:]}|)\n(.*?)```'
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)

        if not match:
            # Try generic code block
            match = re.search(r'```\w*\n(.*?)```', response, re.DOTALL)

        if match:
            code = match.group(1).strip()
            # Get explanation after code block
            explanation_start = match.end()
            explanation = response[explanation_start:].strip()
            # Limit explanation length
            explanation = explanation[:500] if explanation else ""
            return code, explanation

        # No code block found - try to extract code heuristically
        lines = response.split('\n')
        code_lines = []
        explanation_lines = []
        in_code = False

        for line in lines:
            if line.strip().startswith(('import ', 'from ', 'def ', 'class ', 'function ', 'const ', 'let ', 'var ')):
                in_code = True
            if in_code:
                code_lines.append(line)
            else:
                explanation_lines.append(line)

        if code_lines:
            return '\n'.join(code_lines), '\n'.join(explanation_lines)[:500]

        return response, ""

    # =========================================================================
    # FILE OPERATIONS
    # =========================================================================

    def _apply_changes(self, changes: List[FileChange]) -> List[FileChange]:
        """Apply changes to the repository."""
        applied = []

        for change in changes:
            full_path = os.path.join(self.repo_path, change.path)

            try:
                # Create directory if needed
                os.makedirs(os.path.dirname(full_path), exist_ok=True)

                # Write file
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(change.content)

                self.logger.info(f"Applied {change.action}: {change.path}")
                applied.append(change)

            except Exception as e:
                self.logger.error(f"Failed to apply change to {change.path}: {e}")

        return applied

    # =========================================================================
    # VERIFICATION
    # =========================================================================

    def _verify_changes(self, changes: List[FileChange]) -> Dict[str, Any]:
        """Verify that changes are syntactically valid."""
        errors = []

        for change in changes:
            ext = os.path.splitext(change.path)[1].lower()

            if ext == ".py":
                # Python syntax check
                try:
                    ast.parse(change.content)
                except SyntaxError as e:
                    errors.append({
                        "file": change.path,
                        "error": str(e),
                        "line": e.lineno
                    })

            elif ext in (".json",):
                # JSON syntax check
                try:
                    json.loads(change.content)
                except json.JSONDecodeError as e:
                    errors.append({
                        "file": change.path,
                        "error": str(e),
                        "line": e.lineno
                    })

            # Add more language checks as needed

        return {
            "success": len(errors) == 0,
            "errors": errors
        }

    def _attempt_fix_errors(self, errors: List[Dict], changes: List[FileChange]) -> bool:
        """Attempt to fix syntax errors."""
        # For now, just log - could implement auto-fix with LLM
        for error in errors:
            self.logger.warning(f"Syntax error in {error['file']} line {error.get('line')}: {error['error']}")

        # Could implement retry with LLM to fix errors
        return False

    # =========================================================================
    # GIT OPERATIONS
    # =========================================================================

    def _handle_git_operations(
        self,
        issue_number: int,
        changes: List[FileChange],
        task_info: TaskInfo
    ) -> Dict[str, Any]:
        """Handle git branch creation and commit."""
        result = {}

        try:
            # Create branch name
            branch_name = f"agent/issue-{issue_number}"
            result["branch"] = branch_name

            # Check if git repo
            git_dir = os.path.join(self.repo_path, ".git")
            if not os.path.isdir(git_dir):
                self.logger.warning("Not a git repository, skipping git operations")
                return result

            # Run git commands
            def run_git(*args):
                cmd = ["git", "-C", self.repo_path] + list(args)
                return subprocess.run(cmd, capture_output=True, text=True)

            # Create and checkout branch
            run_git("checkout", "-b", branch_name)

            # Stage changed files
            for change in changes:
                run_git("add", change.path)

            # Create commit
            commit_message = self._generate_commit_message(task_info, changes)
            run_git("commit", "-m", commit_message)

            result["commit_created"] = True
            result["commit_message"] = commit_message

        except Exception as e:
            self.logger.error(f"Git operations failed: {e}")
            result["git_error"] = str(e)

        return result

    def _generate_commit_message(self, task_info: TaskInfo, changes: List[FileChange]) -> str:
        """Generate a descriptive commit message."""
        # Create summary line
        summary = task_info.title
        if len(summary) > 50:
            summary = summary[:47] + "..."

        # Create body
        body_lines = [
            "",
            task_info.description[:200] if task_info.description else "",
            "",
            "Changes:",
        ]

        for change in changes[:10]:
            body_lines.append(f"  - {change.action}: {change.path}")

        if len(changes) > 10:
            body_lines.append(f"  - ... and {len(changes) - 10} more files")

        return f"{summary}\n" + "\n".join(body_lines)

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def _identify_tests(self, changes: List[FileChange]) -> List[str]:
        """Identify test files in changes."""
        tests = []

        for change in changes:
            path = change.path.lower()
            if 'test' in path or path.endswith('_test.py') or path.endswith('.test.js'):
                tests.append(change.path)

        return tests

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from text, handling markdown code blocks."""
        # Try to find JSON in code block
        match = re.search(r'```(?:json)?\n(.*?)```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to parse entire text as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in text
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None
