# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - QA AGENT IMPLEMENTATION
# =============================================================================
"""
QA Agent Implementation

This agent validates implementations against acceptance criteria and tests.
It provides the quality gate that ensures all work meets standards.

Validation Strategy:
    1. Parse and structure acceptance criteria
    2. Run automated tests
    3. Use LLM to verify criteria against code
    4. Compile comprehensive results
    5. Generate actionable feedback

Test Execution:
    The agent runs tests in an isolated environment:
    - Unit tests (pytest, jest, etc.)
    - Linter checks (ruff, eslint, etc.)
    - Type checks (mypy, tsc, etc.)
"""

import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
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


class QAVerdict(Enum):
    """Result of QA validation."""
    PASS = "PASS"
    FAIL = "FAIL"
    PARTIAL = "PARTIAL"
    SKIPPED = "SKIPPED"


class CriterionResult(Enum):
    """Result for a single criterion."""
    PASS = "PASS"
    FAIL = "FAIL"
    UNCLEAR = "UNCLEAR"


@dataclass
class AcceptanceCriterion:
    """A single acceptance criterion to verify."""
    id: int
    text: str
    category: str = "functional"  # functional, performance, security, etc.


@dataclass
class CriterionVerification:
    """Result of verifying one criterion."""
    criterion: AcceptanceCriterion
    result: CriterionResult
    evidence: str
    reason: Optional[str] = None
    checked_files: List[str] = field(default_factory=list)


@dataclass
class TestResult:
    """Result of running a test command."""
    name: str
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    passed: bool

    @property
    def output(self) -> str:
        """Combined output."""
        return f"{self.stdout}\n{self.stderr}".strip()


@dataclass
class FixSuggestion:
    """A suggested fix for a QA issue."""
    issue: str
    location: Optional[str]
    suggestion: str
    priority: str  # high, medium, low
    related_criterion: Optional[int] = None


@dataclass
class QAReport:
    """Complete QA validation report."""
    verdict: QAVerdict
    criteria_results: List[CriterionVerification]
    test_results: List[TestResult]
    feedback: str
    suggestions: List[FixSuggestion]
    summary: str

    @property
    def criteria_passed(self) -> int:
        return sum(1 for c in self.criteria_results if c.result == CriterionResult.PASS)

    @property
    def criteria_total(self) -> int:
        return len(self.criteria_results)

    @property
    def tests_passed(self) -> int:
        return sum(1 for t in self.test_results if t.passed)

    @property
    def tests_total(self) -> int:
        return len(self.test_results)


# =============================================================================
# JSON SCHEMAS FOR LLM RESPONSES
# =============================================================================

CRITERION_VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "result": {
            "type": "string",
            "enum": ["PASS", "FAIL", "UNCLEAR"],
            "description": "Whether the criterion is met"
        },
        "evidence": {
            "type": "string",
            "description": "Evidence from code that supports the verdict"
        },
        "reason": {
            "type": "string",
            "description": "Explanation of why the criterion passed or failed"
        },
        "files_checked": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Files examined to verify this criterion"
        }
    },
    "required": ["result", "evidence", "reason"]
}

FIX_SUGGESTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "issue": {"type": "string"},
                    "location": {"type": "string"},
                    "suggestion": {"type": "string"},
                    "priority": {"type": "string", "enum": ["high", "medium", "low"]}
                },
                "required": ["issue", "suggestion", "priority"]
            }
        }
    },
    "required": ["suggestions"]
}


# =============================================================================
# QA AGENT CLASS
# =============================================================================


class QAAgent(AgentInterface):
    """
    Agent that validates implementations against acceptance criteria.

    This is the quality gate of the system - no task completes without
    passing QA validation.

    Attributes:
        llm: LLM client for criterion verification
        config: Agent configuration

    Configuration:
        require_all_tests_pass: Fail if any test fails (default: true)
        require_no_linter_errors: Fail on linter errors (default: true)
        test_timeout_seconds: Test execution timeout (default: 300)
        allow_partial_pass: Allow passing with some unclear criteria (default: false)
    """

    DEFAULT_CONFIG = {
        "require_all_tests_pass": True,
        "require_no_linter_errors": True,
        "test_timeout_seconds": 300,
        "allow_partial_pass": False,
        "run_unit_tests": True,
        "run_linter": True,
        "run_type_checker": True,
        "temperature": 0.2,  # Low temp for consistent verification
        "max_test_output_lines": 200,
    }

    # Default test commands by language
    DEFAULT_TEST_COMMANDS = {
        "python": {
            "unit": "pytest tests/ -v --tb=short",
            "lint": "ruff check .",
            "type": "mypy . --ignore-missing-imports",
        },
        "javascript": {
            "unit": "npm test",
            "lint": "npm run lint",
            "type": "npm run type-check",
        },
        "typescript": {
            "unit": "npm test",
            "lint": "npm run lint",
            "type": "tsc --noEmit",
        },
        "dart": {
            "unit": "dart test",
            "lint": "dart analyze",
            "format": "dart format --set-exit-if-changed .",
        },
        "flutter": {
            "unit": "flutter test",
            "lint": "flutter analyze",
            "format": "dart format --set-exit-if-changed .",
            "build": "flutter build apk --debug",
        },
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize QA agent with configuration."""
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
        return "qa"

    def validate_context(self, context: AgentContext) -> bool:
        """
        Validate context has required information.

        Required:
            - Issue data with body/acceptance criteria

        Optional but expected:
            - Modified files list
            - Test commands
        """
        if not context.issue_data:
            self.logger.error("No issue data provided")
            return False

        if not context.issue_data.get("body"):
            self.logger.error("Issue has no body with acceptance criteria")
            return False

        if not context.input_data.get("modified_files"):
            self.logger.warning("No modified files specified - will check all relevant files")

        return True

    def execute(self, context: AgentContext) -> AgentResult:
        """
        Execute QA validation.

        Workflow:
            1. Extract acceptance criteria from issue
            2. Identify and configure test commands
            3. Run automated tests
            4. Verify each criterion with LLM
            5. Compile results
            6. Generate feedback if failed
            7. Return comprehensive QA report

        Args:
            context: Agent execution context

        Returns:
            AgentResult with QA report
        """
        self.logger.info(f"Starting QA validation for issue #{context.issue_data.get('number', '?')}")

        try:
            # =================================================================
            # STEP 1: Extract Acceptance Criteria
            # =================================================================
            self.logger.info("Step 1: Extracting acceptance criteria")

            criteria = self._extract_criteria(context.issue_data.get("body", ""))

            if not criteria:
                self.logger.warning("No acceptance criteria found in issue")
                # Try to infer from title/description
                criteria = [AcceptanceCriterion(
                    id=1,
                    text=f"Implementation meets requirements: {context.issue_data.get('title', 'Task completed')}",
                    category="functional"
                )]

            self.logger.info(f"Found {len(criteria)} acceptance criteria")

            # =================================================================
            # STEP 2: Identify Test Commands
            # =================================================================
            self.logger.info("Step 2: Identifying test commands")

            test_commands = self._get_test_commands(context)

            self.logger.info(f"Will run {len(test_commands)} test commands")

            # =================================================================
            # STEP 3: Run Automated Tests
            # =================================================================
            self.logger.info("Step 3: Running automated tests")

            test_results = self._run_tests(test_commands, context.repo_path)

            tests_passed = all(r.passed for r in test_results) if test_results else True
            self.logger.info(f"Tests: {sum(1 for r in test_results if r.passed)}/{len(test_results)} passed")

            # =================================================================
            # STEP 4: Get Modified Files Content
            # =================================================================
            self.logger.info("Step 4: Loading modified files for verification")

            modified_files = context.input_data.get("modified_files", [])
            file_contents = self._load_file_contents(modified_files, context.repo_path)

            # =================================================================
            # STEP 5: Verify Each Criterion with LLM
            # =================================================================
            self.logger.info("Step 5: Verifying acceptance criteria")

            criteria_results = []
            for criterion in criteria:
                self.logger.info(f"  Verifying criterion {criterion.id}: {criterion.text[:50]}...")

                verification = self._verify_criterion(
                    criterion=criterion,
                    file_contents=file_contents,
                    test_results=test_results,
                    context=context
                )
                criteria_results.append(verification)

            # =================================================================
            # STEP 6: Compile Results
            # =================================================================
            self.logger.info("Step 6: Compiling QA results")

            criteria_passed = all(
                v.result == CriterionResult.PASS
                for v in criteria_results
            )

            criteria_with_unclear = all(
                v.result in [CriterionResult.PASS, CriterionResult.UNCLEAR]
                for v in criteria_results
            )

            # Determine verdict
            if tests_passed and criteria_passed:
                verdict = QAVerdict.PASS
            elif tests_passed and criteria_with_unclear and self.config["allow_partial_pass"]:
                verdict = QAVerdict.PARTIAL
            else:
                verdict = QAVerdict.FAIL

            # =================================================================
            # STEP 7: Generate Feedback
            # =================================================================
            self.logger.info("Step 7: Generating feedback")

            feedback = ""
            suggestions: List[FixSuggestion] = []

            if verdict in [QAVerdict.FAIL, QAVerdict.PARTIAL]:
                feedback = self._generate_feedback(criteria_results, test_results)
                suggestions = self._generate_fix_suggestions(
                    criteria_results=criteria_results,
                    test_results=test_results,
                    file_contents=file_contents,
                    context=context
                )

            # =================================================================
            # STEP 8: Create Report
            # =================================================================
            summary = self._generate_summary(verdict, criteria_results, test_results)

            report = QAReport(
                verdict=verdict,
                criteria_results=criteria_results,
                test_results=test_results,
                feedback=feedback,
                suggestions=suggestions,
                summary=summary
            )

            # =================================================================
            # STEP 9: Return Results
            # =================================================================
            self.logger.info(f"QA Validation complete: {verdict.value}")

            return AgentResult(
                status=AgentStatus.SUCCESS,
                output={
                    "qa_result": verdict.value,
                    "acceptance_checklist": [
                        {
                            "id": v.criterion.id,
                            "text": v.criterion.text,
                            "result": v.result.value,
                            "evidence": v.evidence,
                            "reason": v.reason
                        }
                        for v in criteria_results
                    ],
                    "test_results": [
                        {
                            "name": t.name,
                            "command": t.command,
                            "passed": t.passed,
                            "exit_code": t.exit_code,
                            "duration": t.duration_seconds,
                            "output": self._truncate_output(t.output)
                        }
                        for t in test_results
                    ],
                    "feedback": feedback if feedback else None,
                    "suggested_fixes": [
                        {
                            "issue": s.issue,
                            "location": s.location,
                            "suggestion": s.suggestion,
                            "priority": s.priority
                        }
                        for s in suggestions
                    ] if suggestions else None,
                    "summary": summary
                },
                message=f"QA {verdict.value}: {report.criteria_passed}/{report.criteria_total} criteria, {report.tests_passed}/{report.tests_total} tests",
                metrics={
                    "criteria_total": report.criteria_total,
                    "criteria_passed": report.criteria_passed,
                    "tests_total": report.tests_total,
                    "tests_passed": report.tests_passed,
                    "suggestions_count": len(suggestions)
                }
            )

        except Exception as e:
            self.logger.error(f"QA validation failed with error: {e}", exc_info=True)
            return AgentResult(
                status=AgentStatus.FAILED,
                output={"error": str(e)},
                message=f"QA validation error: {str(e)}"
            )

    # =========================================================================
    # CRITERIA EXTRACTION
    # =========================================================================

    def _extract_criteria(self, issue_body: str) -> List[AcceptanceCriterion]:
        """
        Extract acceptance criteria from issue body.

        Looks for:
            - Checkbox lists under "Acceptance Criteria" header
            - Numbered lists under similar headers
            - Bullet points after acceptance criteria mention
        """
        criteria = []
        criterion_id = 1

        # Pattern 1: Checkbox format: - [ ] or - [x]
        checkbox_pattern = r'-\s*\[[\sx]\]\s*(.+?)(?=\n|$)'

        # Pattern 2: Numbered list: 1. or 1)
        numbered_pattern = r'^\s*\d+[.)]\s*(.+?)(?=\n|$)'

        # Pattern 3: Bullet points: - or *
        bullet_pattern = r'^[-*]\s+(.+?)(?=\n|$)'

        # Try to find acceptance criteria section
        sections = re.split(r'\n##?\s*', issue_body, flags=re.IGNORECASE)

        ac_section = None
        for section in sections:
            lower_section = section.lower()
            if any(term in lower_section for term in ['acceptance criteria', 'acceptance', 'criteria', 'requirements']):
                ac_section = section
                break

        # If no explicit section, use whole body
        text_to_parse = ac_section if ac_section else issue_body

        # Extract from checkboxes first (most explicit)
        for match in re.finditer(checkbox_pattern, text_to_parse, re.MULTILINE):
            text = match.group(1).strip()
            if text and len(text) > 3:  # Avoid empty or very short criteria
                criteria.append(AcceptanceCriterion(
                    id=criterion_id,
                    text=text,
                    category=self._categorize_criterion(text)
                ))
                criterion_id += 1

        # If no checkboxes, try numbered list
        if not criteria:
            for match in re.finditer(numbered_pattern, text_to_parse, re.MULTILINE):
                text = match.group(1).strip()
                if text and len(text) > 3:
                    criteria.append(AcceptanceCriterion(
                        id=criterion_id,
                        text=text,
                        category=self._categorize_criterion(text)
                    ))
                    criterion_id += 1

        # If still nothing, try bullet points
        if not criteria:
            for match in re.finditer(bullet_pattern, text_to_parse, re.MULTILINE):
                text = match.group(1).strip()
                # Filter out likely non-criteria items
                if text and len(text) > 10 and not text.lower().startswith(('note:', 'see', 'ref')):
                    criteria.append(AcceptanceCriterion(
                        id=criterion_id,
                        text=text,
                        category=self._categorize_criterion(text)
                    ))
                    criterion_id += 1

        return criteria

    def _categorize_criterion(self, text: str) -> str:
        """Categorize criterion based on keywords."""
        lower_text = text.lower()

        if any(word in lower_text for word in ['performance', 'fast', 'latency', 'response time', 'load']):
            return "performance"
        elif any(word in lower_text for word in ['security', 'auth', 'permission', 'encrypt', 'sanitize', 'xss', 'injection']):
            return "security"
        elif any(word in lower_text for word in ['test', 'coverage', 'spec']):
            return "testing"
        elif any(word in lower_text for word in ['document', 'readme', 'comment', 'docstring']):
            return "documentation"
        elif any(word in lower_text for word in ['error', 'exception', 'handle', 'graceful', 'recover']):
            return "error_handling"
        else:
            return "functional"

    # =========================================================================
    # TEST EXECUTION
    # =========================================================================

    def _get_test_commands(self, context: AgentContext) -> Dict[str, str]:
        """
        Determine which test commands to run.

        Sources (in priority order):
            1. Input data (explicit commands)
            2. Project conventions
            3. Default commands by detected language
        """
        commands = {}

        # Check input data for explicit commands
        if context.input_data.get("test_commands"):
            return context.input_data["test_commands"]

        # Check conventions for test commands
        if context.conventions:
            # Look for test command patterns in conventions
            conventions_lower = context.conventions.lower()

            if "pytest" in conventions_lower:
                commands["unit"] = "pytest tests/ -v --tb=short"
            if "ruff" in conventions_lower:
                commands["lint"] = "ruff check ."
            if "mypy" in conventions_lower:
                commands["type"] = "mypy . --ignore-missing-imports"

        # If no commands found, use defaults based on project files
        if not commands:
            repo_path = Path(context.repo_path) if context.repo_path else Path.cwd()

            # Detect language from project files
            if (repo_path / "pyproject.toml").exists() or (repo_path / "requirements.txt").exists():
                commands = self.DEFAULT_TEST_COMMANDS["python"].copy()
            elif (repo_path / "pubspec.yaml").exists():
                # Dart/Flutter project - check if it's Flutter
                pubspec_path = repo_path / "pubspec.yaml"
                try:
                    pubspec_content = pubspec_path.read_text(encoding="utf-8")
                    if "flutter:" in pubspec_content or "sdk: flutter" in pubspec_content:
                        commands = self.DEFAULT_TEST_COMMANDS["flutter"].copy()
                    else:
                        commands = self.DEFAULT_TEST_COMMANDS["dart"].copy()
                except Exception:
                    commands = self.DEFAULT_TEST_COMMANDS["dart"].copy()
            elif (repo_path / "package.json").exists():
                if (repo_path / "tsconfig.json").exists():
                    commands = self.DEFAULT_TEST_COMMANDS["typescript"].copy()
                else:
                    commands = self.DEFAULT_TEST_COMMANDS["javascript"].copy()

        # Filter based on config
        if not self.config.get("run_unit_tests"):
            commands.pop("unit", None)
        if not self.config.get("run_linter"):
            commands.pop("lint", None)
        if not self.config.get("run_type_checker"):
            commands.pop("type", None)

        return commands

    def _run_tests(self, commands: Dict[str, str], repo_path: Optional[str]) -> List[TestResult]:
        """
        Execute test commands and capture results.

        Args:
            commands: Dict of test name -> command string
            repo_path: Directory to run tests in

        Returns:
            List of TestResult objects
        """
        results = []
        cwd = repo_path or os.getcwd()
        timeout = self.config["test_timeout_seconds"]

        for name, command in commands.items():
            self.logger.info(f"  Running {name}: {command}")
            start_time = time.time()

            try:
                process = subprocess.run(
                    command,
                    shell=True,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )

                duration = time.time() - start_time

                results.append(TestResult(
                    name=name,
                    command=command,
                    exit_code=process.returncode,
                    stdout=process.stdout,
                    stderr=process.stderr,
                    duration_seconds=duration,
                    passed=process.returncode == 0
                ))

                status = "✓" if process.returncode == 0 else "✗"
                self.logger.info(f"    {status} {name} completed in {duration:.2f}s")

            except subprocess.TimeoutExpired:
                duration = time.time() - start_time
                results.append(TestResult(
                    name=name,
                    command=command,
                    exit_code=-1,
                    stdout="",
                    stderr=f"Test timed out after {timeout} seconds",
                    duration_seconds=duration,
                    passed=False
                ))
                self.logger.warning(f"    ⏱ {name} timed out after {timeout}s")

            except Exception as e:
                duration = time.time() - start_time
                results.append(TestResult(
                    name=name,
                    command=command,
                    exit_code=-1,
                    stdout="",
                    stderr=f"Error running test: {str(e)}",
                    duration_seconds=duration,
                    passed=False
                ))
                self.logger.error(f"    ✗ {name} error: {e}")

        return results

    # =========================================================================
    # CRITERION VERIFICATION
    # =========================================================================

    def _load_file_contents(self, file_paths: List[str], repo_path: Optional[str]) -> Dict[str, str]:
        """Load content of modified files."""
        contents = {}
        base_path = Path(repo_path) if repo_path else Path.cwd()

        for file_path in file_paths:
            full_path = base_path / file_path
            try:
                if full_path.exists() and full_path.is_file():
                    contents[file_path] = full_path.read_text(encoding="utf-8")
                    self.logger.debug(f"Loaded {file_path} ({len(contents[file_path])} chars)")
            except Exception as e:
                self.logger.warning(f"Could not read {file_path}: {e}")

        return contents

    def _verify_criterion(
        self,
        criterion: AcceptanceCriterion,
        file_contents: Dict[str, str],
        test_results: List[TestResult],
        context: AgentContext
    ) -> CriterionVerification:
        """
        Verify a single acceptance criterion using LLM.

        The LLM examines the code changes and test results to determine
        if the criterion has been met.
        """
        # Load verification prompt
        prompt = self._load_prompt("verification")

        # Format file contents for prompt
        files_text = self._format_files_for_prompt(file_contents)

        # Format test results
        tests_text = self._format_tests_for_prompt(test_results)

        # Build full prompt
        full_prompt = prompt.replace("{{criterion_id}}", str(criterion.id))
        full_prompt = full_prompt.replace("{{criterion_text}}", criterion.text)
        full_prompt = full_prompt.replace("{{criterion_category}}", criterion.category)
        full_prompt = full_prompt.replace("{{file_contents}}", files_text)
        full_prompt = full_prompt.replace("{{test_results}}", tests_text)
        full_prompt = full_prompt.replace("{{task_title}}", context.issue_data.get("title", ""))
        full_prompt = full_prompt.replace("{{task_description}}", context.issue_data.get("body", "")[:1000])

        # Get LLM verification
        try:
            response = self.llm.generate(
                prompt=full_prompt,
                system="You are a QA engineer verifying code implementations. Be thorough but fair."
            )

            verification_data = self._extract_json(response)

            result_str = verification_data.get("result", "UNCLEAR").upper()
            result = CriterionResult[result_str] if result_str in CriterionResult.__members__ else CriterionResult.UNCLEAR

            return CriterionVerification(
                criterion=criterion,
                result=result,
                evidence=verification_data.get("evidence", "No evidence provided"),
                reason=verification_data.get("reason"),
                checked_files=verification_data.get("files_checked", [])
            )

        except Exception as e:
            self.logger.error(f"Error verifying criterion {criterion.id}: {e}")
            return CriterionVerification(
                criterion=criterion,
                result=CriterionResult.UNCLEAR,
                evidence=f"Verification error: {str(e)}",
                reason="Could not complete verification due to error"
            )

    def _format_files_for_prompt(self, file_contents: Dict[str, str], max_lines: int = 200) -> str:
        """Format file contents for inclusion in prompt."""
        if not file_contents:
            return "No modified files available."

        parts = []
        for path, content in file_contents.items():
            lines = content.split('\n')
            if len(lines) > max_lines:
                truncated = '\n'.join(lines[:max_lines])
                parts.append(f"### {path}\n```\n{truncated}\n... (truncated, {len(lines) - max_lines} more lines)\n```")
            else:
                parts.append(f"### {path}\n```\n{content}\n```")

        return '\n\n'.join(parts)

    def _format_tests_for_prompt(self, test_results: List[TestResult]) -> str:
        """Format test results for inclusion in prompt."""
        if not test_results:
            return "No test results available."

        parts = []
        for test in test_results:
            status = "✓ PASSED" if test.passed else "✗ FAILED"
            output = self._truncate_output(test.output, max_lines=50)
            parts.append(f"### {test.name}: {status}\nCommand: `{test.command}`\n```\n{output}\n```")

        return '\n\n'.join(parts)

    # =========================================================================
    # FEEDBACK GENERATION
    # =========================================================================

    def _generate_feedback(
        self,
        criteria_results: List[CriterionVerification],
        test_results: List[TestResult]
    ) -> str:
        """
        Generate human-readable feedback for failures.

        Creates detailed feedback that helps the developer agent
        understand what needs to be fixed.
        """
        lines = ["## QA Validation Failed\n"]

        # Failed criteria
        failed_criteria = [c for c in criteria_results if c.result == CriterionResult.FAIL]
        unclear_criteria = [c for c in criteria_results if c.result == CriterionResult.UNCLEAR]

        if failed_criteria:
            lines.append("### Failed Acceptance Criteria\n")
            for c in failed_criteria:
                lines.append(f"**{c.criterion.id}. ❌ {c.criterion.text}**")
                lines.append(f"   - Reason: {c.reason or 'Not specified'}")
                lines.append(f"   - Evidence: {c.evidence}")
                lines.append("")

        if unclear_criteria:
            lines.append("### Unclear Criteria (Need Review)\n")
            for c in unclear_criteria:
                lines.append(f"**{c.criterion.id}. ❓ {c.criterion.text}**")
                lines.append(f"   - Reason: {c.reason or 'Could not verify'}")
                lines.append("")

        # Failed tests
        failed_tests = [t for t in test_results if not t.passed]

        if failed_tests:
            lines.append("### Test Failures\n")
            for t in failed_tests:
                lines.append(f"**{t.name}**: `{t.command}`")
                lines.append("```")
                lines.append(self._truncate_output(t.output, max_lines=30))
                lines.append("```")
                lines.append("")

        # Summary
        passed_criteria = sum(1 for c in criteria_results if c.result == CriterionResult.PASS)
        passed_tests = sum(1 for t in test_results if t.passed)

        lines.append("### Summary\n")
        lines.append(f"- Criteria: {passed_criteria}/{len(criteria_results)} passed")
        lines.append(f"- Tests: {passed_tests}/{len(test_results)} passed")
        lines.append("")
        lines.append("Please address the issues above and resubmit for QA.")

        return '\n'.join(lines)

    def _generate_fix_suggestions(
        self,
        criteria_results: List[CriterionVerification],
        test_results: List[TestResult],
        file_contents: Dict[str, str],
        context: AgentContext
    ) -> List[FixSuggestion]:
        """
        Generate specific fix suggestions using LLM.

        Analyzes failures and suggests concrete fixes.
        """
        # Collect failures
        failures = []

        for c in criteria_results:
            if c.result == CriterionResult.FAIL:
                failures.append({
                    "type": "criterion",
                    "id": c.criterion.id,
                    "description": c.criterion.text,
                    "reason": c.reason,
                    "evidence": c.evidence
                })

        for t in test_results:
            if not t.passed:
                failures.append({
                    "type": "test",
                    "name": t.name,
                    "command": t.command,
                    "output": self._truncate_output(t.output, max_lines=50)
                })

        if not failures:
            return []

        # Load prompt and generate suggestions
        prompt = self._load_prompt("fix_suggestions")

        prompt = prompt.replace("{{failures}}", json.dumps(failures, indent=2))
        prompt = prompt.replace("{{file_contents}}", self._format_files_for_prompt(file_contents, max_lines=100))

        try:
            response = self.llm.generate(
                prompt=prompt,
                system="You are a senior developer providing actionable fix suggestions."
            )

            data = self._extract_json(response)
            suggestions_data = data.get("suggestions", [])

            suggestions = []
            for s in suggestions_data:
                suggestions.append(FixSuggestion(
                    issue=s.get("issue", "Unknown issue"),
                    location=s.get("location"),
                    suggestion=s.get("suggestion", "No suggestion provided"),
                    priority=s.get("priority", "medium"),
                    related_criterion=s.get("criterion_id")
                ))

            return suggestions

        except Exception as e:
            self.logger.error(f"Error generating fix suggestions: {e}")
            return []

    def _generate_summary(
        self,
        verdict: QAVerdict,
        criteria_results: List[CriterionVerification],
        test_results: List[TestResult]
    ) -> str:
        """Generate a brief summary of QA results."""
        passed_criteria = sum(1 for c in criteria_results if c.result == CriterionResult.PASS)
        failed_criteria = sum(1 for c in criteria_results if c.result == CriterionResult.FAIL)
        passed_tests = sum(1 for t in test_results if t.passed)
        failed_tests = sum(1 for t in test_results if not t.passed)

        if verdict == QAVerdict.PASS:
            return f"All {len(criteria_results)} acceptance criteria met. All {len(test_results)} tests passed."
        elif verdict == QAVerdict.PARTIAL:
            return f"Partial pass: {passed_criteria}/{len(criteria_results)} criteria verified, some unclear. {passed_tests}/{len(test_results)} tests passed."
        else:
            issues = []
            if failed_criteria > 0:
                issues.append(f"{failed_criteria} criteria failed")
            if failed_tests > 0:
                issues.append(f"{failed_tests} tests failed")
            return f"QA Failed: {', '.join(issues)}. See detailed feedback for fixes needed."

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _load_prompt(self, name: str) -> str:
        """Load a prompt template from the prompts directory."""
        if name in self._prompts:
            return self._prompts[name]

        # Try to load from file
        prompt_path = Path(__file__).parent / "prompts" / f"{name}.md"

        if prompt_path.exists():
            self._prompts[name] = prompt_path.read_text(encoding="utf-8")
            return self._prompts[name]

        # Return default prompts if file not found
        defaults = {
            "verification": self._get_default_verification_prompt(),
            "fix_suggestions": self._get_default_fix_suggestions_prompt()
        }

        return defaults.get(name, f"# {name}\n\nNo prompt template found.")

    def _get_default_verification_prompt(self) -> str:
        """Default prompt for criterion verification."""
        return """# Acceptance Criterion Verification

You are verifying whether an acceptance criterion has been met by the implementation.

## Criterion to Verify

**ID:** {{criterion_id}}
**Category:** {{criterion_category}}
**Criterion:** {{criterion_text}}

## Task Context

**Title:** {{task_title}}

**Description:**
{{task_description}}

## Modified Files

{{file_contents}}

## Test Results

{{test_results}}

## Your Task

Analyze the code and test results to determine if this criterion is satisfied.

Consider:
1. Does the code implement the required functionality?
2. Is there evidence in the tests that this works?
3. Are there any edge cases not handled?
4. Does the implementation match the criterion exactly?

## Output Format

Return JSON:

```json
{
  "result": "PASS" | "FAIL" | "UNCLEAR",
  "evidence": "Specific code or test output that supports your verdict",
  "reason": "Explanation of why the criterion passed or failed",
  "files_checked": ["list of files examined"]
}
```

Be strict but fair. If the criterion is clearly met, return PASS.
If there are clear deficiencies, return FAIL.
If you cannot determine from the available evidence, return UNCLEAR.
"""

    def _get_default_fix_suggestions_prompt(self) -> str:
        """Default prompt for generating fix suggestions."""
        return """# Fix Suggestions Generator

Generate specific, actionable fix suggestions for the following QA failures.

## Failures to Address

{{failures}}

## Code Context

{{file_contents}}

## Your Task

For each failure, provide a concrete suggestion on how to fix it.

## Output Format

Return JSON:

```json
{
  "suggestions": [
    {
      "issue": "Description of the problem",
      "location": "file.py:line or general location",
      "suggestion": "Specific fix to implement",
      "priority": "high" | "medium" | "low",
      "criterion_id": 1 (if related to a criterion)
    }
  ]
}
```

Guidelines:
- Be specific about what code changes are needed
- Provide file and line references when possible
- Prioritize critical issues (high) over minor ones (low)
- Each suggestion should be actionable
"""

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response, handling code blocks."""
        # Try to find JSON in code block
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))

        # Try to find raw JSON object
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))

        # If no JSON found, try parsing entire text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    def _truncate_output(self, text: str, max_lines: int = None) -> str:
        """Truncate output to configured max lines."""
        max_lines = max_lines or self.config["max_test_output_lines"]
        lines = text.split('\n')

        if len(lines) <= max_lines:
            return text

        truncated = lines[:max_lines]
        truncated.append(f"\n... ({len(lines) - max_lines} more lines truncated)")
        return '\n'.join(truncated)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "QAAgent",
    "QAVerdict",
    "CriterionResult",
    "AcceptanceCriterion",
    "CriterionVerification",
    "TestResult",
    "FixSuggestion",
    "QAReport",
    "CRITERION_VERIFICATION_SCHEMA",
    "FIX_SUGGESTIONS_SCHEMA",
]
