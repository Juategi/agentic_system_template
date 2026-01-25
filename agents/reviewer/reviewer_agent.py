# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - REVIEWER AGENT IMPLEMENTATION
# =============================================================================
"""
Reviewer Agent Implementation

This agent performs code review to ensure quality and consistency.
It acts as a second pair of eyes before code is considered complete.

Review Criteria:
    1. Code Quality - Clean, readable, well-structured
    2. Convention Adherence - Follows project standards
    3. Architecture Alignment - Fits system design
    4. Maintainability - Easy to understand and modify
    5. Documentation - Appropriate comments and docs

Unlike QA (which checks functionality), Reviewer focuses on code craftsmanship.
"""

import json
import logging
import re
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


class ReviewDecision(Enum):
    """Final review decision."""
    APPROVED = "APPROVED"
    CHANGES_REQUESTED = "CHANGES_REQUESTED"
    NEEDS_DISCUSSION = "NEEDS_DISCUSSION"


class CommentSeverity(Enum):
    """Severity level for review comments."""
    CRITICAL = "critical"      # Must fix before merge
    MAJOR = "major"            # Should fix, blocks approval
    MINOR = "minor"            # Nice to fix, doesn't block
    SUGGESTION = "suggestion"  # Optional improvement
    PRAISE = "praise"          # Positive feedback


class ReviewCategory(Enum):
    """Categories of review feedback."""
    CODE_QUALITY = "code_quality"
    CONVENTIONS = "conventions"
    ARCHITECTURE = "architecture"
    MAINTAINABILITY = "maintainability"
    DOCUMENTATION = "documentation"
    SECURITY = "security"
    PERFORMANCE = "performance"


@dataclass
class ReviewComment:
    """A single review comment on code."""
    file_path: str
    line_number: Optional[int]
    line_end: Optional[int]
    category: ReviewCategory
    severity: CommentSeverity
    message: str
    suggestion: Optional[str] = None
    code_snippet: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file_path,
            "line": self.line_number,
            "line_end": self.line_end,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "suggestion": self.suggestion,
            "code_snippet": self.code_snippet,
        }


@dataclass
class FileReview:
    """Review results for a single file."""
    file_path: str
    comments: List[ReviewComment]
    scores: Dict[str, int]  # category -> score (0-100)
    summary: str

    @property
    def overall_score(self) -> float:
        if not self.scores:
            return 100.0
        return sum(self.scores.values()) / len(self.scores)

    @property
    def has_blocking_issues(self) -> bool:
        return any(
            c.severity in [CommentSeverity.CRITICAL, CommentSeverity.MAJOR]
            for c in self.comments
        )


@dataclass
class ReviewReport:
    """Complete code review report."""
    decision: ReviewDecision
    file_reviews: List[FileReview]
    overall_score: float
    scores_breakdown: Dict[str, float]
    summary: str
    blocking_issues: List[ReviewComment]
    improvement_areas: List[str]

    @property
    def total_comments(self) -> int:
        return sum(len(fr.comments) for fr in self.file_reviews)

    @property
    def critical_count(self) -> int:
        return sum(
            1 for fr in self.file_reviews
            for c in fr.comments
            if c.severity == CommentSeverity.CRITICAL
        )

    @property
    def major_count(self) -> int:
        return sum(
            1 for fr in self.file_reviews
            for c in fr.comments
            if c.severity == CommentSeverity.MAJOR
        )


# =============================================================================
# JSON SCHEMAS
# =============================================================================

FILE_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "object",
            "properties": {
                "code_quality": {"type": "integer", "minimum": 0, "maximum": 100},
                "conventions": {"type": "integer", "minimum": 0, "maximum": 100},
                "architecture": {"type": "integer", "minimum": 0, "maximum": 100},
                "maintainability": {"type": "integer", "minimum": 0, "maximum": 100},
                "documentation": {"type": "integer", "minimum": 0, "maximum": 100},
            }
        },
        "comments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "line": {"type": "integer"},
                    "line_end": {"type": "integer"},
                    "category": {"type": "string"},
                    "severity": {"type": "string"},
                    "message": {"type": "string"},
                    "suggestion": {"type": "string"}
                },
                "required": ["category", "severity", "message"]
            }
        },
        "summary": {"type": "string"}
    },
    "required": ["scores", "comments", "summary"]
}


# =============================================================================
# REVIEWER AGENT CLASS
# =============================================================================


class ReviewerAgent(AgentInterface):
    """
    Agent that performs code review.

    This agent reviews code changes for quality, consistency, and adherence
    to project standards. It provides detailed feedback and a final decision.

    Configuration:
        criteria_weights: Weight for each review aspect (default: equal)
        approval_threshold: Minimum score to approve (default: 70)
        block_on_critical: Block approval on critical issues (default: True)
        block_on_major: Block approval on major issues (default: True)
        max_comments_per_file: Limit comments to avoid noise (default: 20)
    """

    DEFAULT_CONFIG = {
        "criteria_weights": {
            "code_quality": 0.25,
            "conventions": 0.20,
            "architecture": 0.20,
            "maintainability": 0.20,
            "documentation": 0.15,
        },
        "approval_threshold": 70,
        "block_on_critical": True,
        "block_on_major": True,
        "max_comments_per_file": 20,
        "temperature": 0.3,
        "include_praise": True,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Reviewer agent with configuration."""
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
        return "reviewer"

    def validate_context(self, context: AgentContext) -> bool:
        """
        Validate context has required information.

        Required:
            - Modified files list with content or paths
        """
        if not context.input_data.get("modified_files"):
            self.logger.error("No modified files provided for review")
            return False

        return True

    def execute(self, context: AgentContext) -> AgentResult:
        """
        Execute code review.

        Workflow:
            1. Load modified files and their content
            2. Load conventions and architecture docs
            3. Review each file individually
            4. Calculate overall scores
            5. Determine approval status
            6. Compile review report

        Args:
            context: Agent execution context

        Returns:
            AgentResult with review report
        """
        self.logger.info("Starting code review")

        try:
            # =================================================================
            # STEP 1: Load Modified Files
            # =================================================================
            self.logger.info("Step 1: Loading modified files")

            modified_files = context.input_data.get("modified_files", [])
            file_contents = self._load_files(modified_files, context.repo_path)

            if not file_contents:
                self.logger.warning("No file contents could be loaded")
                return AgentResult(
                    status=AgentStatus.SUCCESS,
                    output={
                        "review_result": "APPROVED",
                        "message": "No files to review"
                    },
                    message="No files to review"
                )

            self.logger.info(f"Loaded {len(file_contents)} files for review")

            # =================================================================
            # STEP 2: Load Project Standards
            # =================================================================
            self.logger.info("Step 2: Loading project standards")

            conventions = context.conventions or self._get_default_conventions()
            architecture = context.architecture or ""

            # =================================================================
            # STEP 3: Review Each File
            # =================================================================
            self.logger.info("Step 3: Reviewing files")

            file_reviews = []
            for file_path, content in file_contents.items():
                self.logger.info(f"  Reviewing {file_path}...")

                review = self._review_file(
                    file_path=file_path,
                    content=content,
                    conventions=conventions,
                    architecture=architecture,
                    context=context
                )
                file_reviews.append(review)

            # =================================================================
            # STEP 4: Calculate Overall Scores
            # =================================================================
            self.logger.info("Step 4: Calculating scores")

            scores_breakdown = self._calculate_overall_scores(file_reviews)
            overall_score = self._calculate_weighted_score(scores_breakdown)

            self.logger.info(f"Overall score: {overall_score:.1f}/100")

            # =================================================================
            # STEP 5: Determine Approval Status
            # =================================================================
            self.logger.info("Step 5: Determining approval status")

            blocking_issues = self._get_blocking_issues(file_reviews)
            decision = self._determine_decision(
                overall_score=overall_score,
                blocking_issues=blocking_issues
            )

            self.logger.info(f"Decision: {decision.value}")

            # =================================================================
            # STEP 6: Compile Report
            # =================================================================
            self.logger.info("Step 6: Compiling review report")

            improvement_areas = self._identify_improvement_areas(file_reviews, scores_breakdown)
            summary = self._generate_summary(decision, overall_score, file_reviews, blocking_issues)

            report = ReviewReport(
                decision=decision,
                file_reviews=file_reviews,
                overall_score=overall_score,
                scores_breakdown=scores_breakdown,
                summary=summary,
                blocking_issues=blocking_issues,
                improvement_areas=improvement_areas
            )

            # =================================================================
            # STEP 7: Return Results
            # =================================================================
            return AgentResult(
                status=AgentStatus.SUCCESS,
                output={
                    "review_result": decision.value,
                    "quality_score": round(overall_score, 1),
                    "scores_breakdown": {k: round(v, 1) for k, v in scores_breakdown.items()},
                    "file_reviews": [
                        {
                            "file": fr.file_path,
                            "score": round(fr.overall_score, 1),
                            "comments_count": len(fr.comments),
                            "summary": fr.summary,
                            "comments": [c.to_dict() for c in fr.comments]
                        }
                        for fr in file_reviews
                    ],
                    "blocking_issues": [bi.to_dict() for bi in blocking_issues],
                    "improvement_areas": improvement_areas,
                    "summary": summary
                },
                message=f"Review {decision.value}: {overall_score:.1f}/100 ({report.total_comments} comments)",
                metrics={
                    "files_reviewed": len(file_reviews),
                    "total_comments": report.total_comments,
                    "critical_issues": report.critical_count,
                    "major_issues": report.major_count,
                    "overall_score": overall_score
                }
            )

        except Exception as e:
            self.logger.error(f"Code review failed with error: {e}", exc_info=True)
            return AgentResult(
                status=AgentStatus.FAILED,
                output={"error": str(e)},
                message=f"Review error: {str(e)}"
            )

    # =========================================================================
    # FILE LOADING
    # =========================================================================

    def _load_files(
        self,
        modified_files: List[Any],
        repo_path: Optional[str]
    ) -> Dict[str, str]:
        """
        Load content of modified files.

        Handles both:
            - List of file paths (strings)
            - List of dicts with 'path' and optionally 'content'
        """
        contents = {}
        base_path = Path(repo_path) if repo_path else Path.cwd()

        for item in modified_files:
            # Handle both string paths and dict format
            if isinstance(item, str):
                file_path = item
                file_content = None
            elif isinstance(item, dict):
                file_path = item.get("path", "")
                file_content = item.get("content")
            else:
                continue

            if not file_path:
                continue

            # If content provided, use it
            if file_content:
                contents[file_path] = file_content
                continue

            # Otherwise load from disk
            full_path = base_path / file_path
            try:
                if full_path.exists() and full_path.is_file():
                    # Skip binary files
                    if self._is_binary_file(full_path):
                        self.logger.debug(f"Skipping binary file: {file_path}")
                        continue

                    contents[file_path] = full_path.read_text(encoding="utf-8")
            except Exception as e:
                self.logger.warning(f"Could not read {file_path}: {e}")

        return contents

    def _is_binary_file(self, path: Path) -> bool:
        """Check if file is binary."""
        binary_extensions = {
            '.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg',
            '.pdf', '.zip', '.tar', '.gz',
            '.exe', '.dll', '.so', '.dylib',
            '.pyc', '.pyo', '.class',
            '.woff', '.woff2', '.ttf', '.eot',
        }
        return path.suffix.lower() in binary_extensions

    # =========================================================================
    # FILE REVIEW
    # =========================================================================

    def _review_file(
        self,
        file_path: str,
        content: str,
        conventions: str,
        architecture: str,
        context: AgentContext
    ) -> FileReview:
        """
        Review a single file using LLM.

        Returns detailed review with scores and comments.
        """
        # Load review prompt
        prompt = self._load_prompt("file_review")

        # Determine file language
        language = self._detect_language(file_path)

        # Truncate content if too long
        max_lines = 500
        lines = content.split('\n')
        if len(lines) > max_lines:
            content = '\n'.join(lines[:max_lines])
            content += f"\n\n... (truncated, {len(lines) - max_lines} more lines)"

        # Build prompt
        full_prompt = prompt.replace("{{file_path}}", file_path)
        full_prompt = full_prompt.replace("{{language}}", language)
        full_prompt = full_prompt.replace("{{file_content}}", content)
        full_prompt = full_prompt.replace("{{conventions}}", conventions[:2000] if conventions else "No conventions provided")
        full_prompt = full_prompt.replace("{{architecture}}", architecture[:1000] if architecture else "No architecture docs provided")
        full_prompt = full_prompt.replace("{{include_praise}}", str(self.config["include_praise"]).lower())

        # Get LLM review
        try:
            response = self.llm.generate(
                prompt=full_prompt,
                system="You are an expert code reviewer. Be thorough, constructive, and specific."
            )

            review_data = self._extract_json(response)

            # Parse scores
            scores = {}
            raw_scores = review_data.get("scores", {})
            for category in ReviewCategory:
                key = category.value
                scores[key] = raw_scores.get(key, 80)  # Default to 80 if not provided

            # Parse comments
            comments = []
            raw_comments = review_data.get("comments", [])[:self.config["max_comments_per_file"]]

            for rc in raw_comments:
                try:
                    severity_str = rc.get("severity", "minor").lower()
                    category_str = rc.get("category", "code_quality").lower()

                    severity = CommentSeverity(severity_str) if severity_str in [s.value for s in CommentSeverity] else CommentSeverity.MINOR
                    category = ReviewCategory(category_str) if category_str in [c.value for c in ReviewCategory] else ReviewCategory.CODE_QUALITY

                    comments.append(ReviewComment(
                        file_path=file_path,
                        line_number=rc.get("line"),
                        line_end=rc.get("line_end"),
                        category=category,
                        severity=severity,
                        message=rc.get("message", ""),
                        suggestion=rc.get("suggestion"),
                        code_snippet=rc.get("code_snippet")
                    ))
                except Exception as e:
                    self.logger.warning(f"Error parsing comment: {e}")

            return FileReview(
                file_path=file_path,
                comments=comments,
                scores=scores,
                summary=review_data.get("summary", "Review completed.")
            )

        except Exception as e:
            self.logger.error(f"Error reviewing file {file_path}: {e}")
            return FileReview(
                file_path=file_path,
                comments=[],
                scores={cat.value: 70 for cat in ReviewCategory},
                summary=f"Review error: {str(e)}"
            )

    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        ext_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.tsx': 'TypeScript/React',
            '.jsx': 'JavaScript/React',
            '.java': 'Java',
            '.go': 'Go',
            '.rs': 'Rust',
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.cs': 'C#',
            '.cpp': 'C++',
            '.c': 'C',
            '.swift': 'Swift',
            '.kt': 'Kotlin',
            '.scala': 'Scala',
            '.md': 'Markdown',
            '.yaml': 'YAML',
            '.yml': 'YAML',
            '.json': 'JSON',
            '.html': 'HTML',
            '.css': 'CSS',
            '.scss': 'SCSS',
            '.sql': 'SQL',
            '.sh': 'Shell',
            '.bash': 'Bash',
        }
        ext = Path(file_path).suffix.lower()
        return ext_map.get(ext, 'Unknown')

    # =========================================================================
    # SCORE CALCULATION
    # =========================================================================

    def _calculate_overall_scores(self, file_reviews: List[FileReview]) -> Dict[str, float]:
        """Calculate overall scores across all files."""
        if not file_reviews:
            return {cat.value: 100.0 for cat in ReviewCategory}

        category_totals: Dict[str, List[float]] = {cat.value: [] for cat in ReviewCategory}

        for fr in file_reviews:
            for category, score in fr.scores.items():
                if category in category_totals:
                    category_totals[category].append(score)

        return {
            category: sum(scores) / len(scores) if scores else 100.0
            for category, scores in category_totals.items()
        }

    def _calculate_weighted_score(self, scores_breakdown: Dict[str, float]) -> float:
        """Calculate weighted overall score."""
        weights = self.config["criteria_weights"]
        total_weight = sum(weights.values())

        weighted_sum = sum(
            scores_breakdown.get(category, 80) * weight
            for category, weight in weights.items()
        )

        return weighted_sum / total_weight if total_weight > 0 else 80.0

    # =========================================================================
    # DECISION MAKING
    # =========================================================================

    def _get_blocking_issues(self, file_reviews: List[FileReview]) -> List[ReviewComment]:
        """Collect all blocking issues from reviews."""
        blocking = []

        for fr in file_reviews:
            for comment in fr.comments:
                if comment.severity == CommentSeverity.CRITICAL:
                    if self.config["block_on_critical"]:
                        blocking.append(comment)
                elif comment.severity == CommentSeverity.MAJOR:
                    if self.config["block_on_major"]:
                        blocking.append(comment)

        return blocking

    def _determine_decision(
        self,
        overall_score: float,
        blocking_issues: List[ReviewComment]
    ) -> ReviewDecision:
        """Determine final review decision."""
        # Critical issues always block
        has_critical = any(
            bi.severity == CommentSeverity.CRITICAL
            for bi in blocking_issues
        )
        if has_critical:
            return ReviewDecision.CHANGES_REQUESTED

        # Major issues block if configured
        has_major = any(
            bi.severity == CommentSeverity.MAJOR
            for bi in blocking_issues
        )
        if has_major and self.config["block_on_major"]:
            return ReviewDecision.CHANGES_REQUESTED

        # Score threshold
        if overall_score < self.config["approval_threshold"]:
            return ReviewDecision.CHANGES_REQUESTED

        # Edge case: score is borderline
        if overall_score < self.config["approval_threshold"] + 10:
            # If there are any major issues, request discussion
            if has_major:
                return ReviewDecision.NEEDS_DISCUSSION

        return ReviewDecision.APPROVED

    # =========================================================================
    # REPORT GENERATION
    # =========================================================================

    def _identify_improvement_areas(
        self,
        file_reviews: List[FileReview],
        scores_breakdown: Dict[str, float]
    ) -> List[str]:
        """Identify areas that need improvement."""
        areas = []
        threshold = 75  # Below this needs improvement

        for category, score in scores_breakdown.items():
            if score < threshold:
                readable_category = category.replace('_', ' ').title()
                areas.append(f"{readable_category}: {score:.1f}/100 - needs attention")

        # Add specific feedback from comments
        category_issues: Dict[str, int] = {}
        for fr in file_reviews:
            for comment in fr.comments:
                if comment.severity in [CommentSeverity.CRITICAL, CommentSeverity.MAJOR]:
                    cat = comment.category.value
                    category_issues[cat] = category_issues.get(cat, 0) + 1

        for cat, count in sorted(category_issues.items(), key=lambda x: -x[1])[:3]:
            readable = cat.replace('_', ' ').title()
            areas.append(f"{readable}: {count} significant issues found")

        return areas[:5]  # Limit to top 5

    def _generate_summary(
        self,
        decision: ReviewDecision,
        overall_score: float,
        file_reviews: List[FileReview],
        blocking_issues: List[ReviewComment]
    ) -> str:
        """Generate human-readable review summary."""
        lines = []

        # Decision header
        if decision == ReviewDecision.APPROVED:
            lines.append(f"## Code Review: APPROVED ({overall_score:.1f}/100)")
            lines.append("")
            lines.append("The code meets quality standards and is ready for merge.")
        elif decision == ReviewDecision.CHANGES_REQUESTED:
            lines.append(f"## Code Review: CHANGES REQUESTED ({overall_score:.1f}/100)")
            lines.append("")
            lines.append("Please address the following issues before merging:")
        else:
            lines.append(f"## Code Review: NEEDS DISCUSSION ({overall_score:.1f}/100)")
            lines.append("")
            lines.append("Some aspects need clarification or discussion:")

        # Blocking issues
        if blocking_issues:
            lines.append("")
            lines.append("### Blocking Issues")
            for issue in blocking_issues[:5]:
                severity = "CRITICAL" if issue.severity == CommentSeverity.CRITICAL else "MAJOR"
                location = f"{issue.file_path}"
                if issue.line_number:
                    location += f":{issue.line_number}"
                lines.append(f"- **[{severity}]** {location}: {issue.message}")

        # File summaries
        lines.append("")
        lines.append("### Files Reviewed")
        for fr in file_reviews:
            status = "✓" if not fr.has_blocking_issues else "✗"
            lines.append(f"- {status} `{fr.file_path}` ({fr.overall_score:.0f}/100) - {len(fr.comments)} comments")

        # Statistics
        total_comments = sum(len(fr.comments) for fr in file_reviews)
        critical_count = sum(1 for fr in file_reviews for c in fr.comments if c.severity == CommentSeverity.CRITICAL)
        major_count = sum(1 for fr in file_reviews for c in fr.comments if c.severity == CommentSeverity.MAJOR)

        lines.append("")
        lines.append("### Summary")
        lines.append(f"- Files reviewed: {len(file_reviews)}")
        lines.append(f"- Total comments: {total_comments}")
        if critical_count > 0:
            lines.append(f"- Critical issues: {critical_count}")
        if major_count > 0:
            lines.append(f"- Major issues: {major_count}")

        return '\n'.join(lines)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

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
            "file_review": self._get_default_file_review_prompt(),
        }

        return defaults.get(name, f"# {name}\n\nNo prompt template found.")

    def _get_default_file_review_prompt(self) -> str:
        """Default prompt for file review."""
        return """# Code Review Request

Review the following file for quality, conventions, and best practices.

## File Information

**Path:** {{file_path}}
**Language:** {{language}}

## File Content

```
{{file_content}}
```

## Project Conventions

{{conventions}}

## Architecture Guidelines

{{architecture}}

## Review Instructions

Analyze the code and provide:

1. **Scores** (0-100) for each category:
   - code_quality: Is the code clean, readable, well-structured?
   - conventions: Does it follow project conventions?
   - architecture: Does it fit the system architecture?
   - maintainability: Is it easy to understand and modify?
   - documentation: Are comments and docs appropriate?

2. **Comments** on specific issues or good practices:
   - Specify line numbers when possible
   - Use appropriate severity: critical, major, minor, suggestion, praise
   - Provide actionable suggestions

3. **Summary** of the overall review

{{#if include_praise}}
Include positive feedback (praise) for good code practices.
{{/if}}

## Output Format

Return JSON:

```json
{
  "scores": {
    "code_quality": 85,
    "conventions": 90,
    "architecture": 80,
    "maintainability": 85,
    "documentation": 75
  },
  "comments": [
    {
      "line": 42,
      "line_end": 45,
      "category": "code_quality",
      "severity": "major",
      "message": "This function is too complex (cyclomatic complexity > 10)",
      "suggestion": "Consider breaking this into smaller functions"
    },
    {
      "line": 10,
      "category": "documentation",
      "severity": "minor",
      "message": "Missing docstring for public function",
      "suggestion": "Add a docstring explaining the function's purpose and parameters"
    }
  ],
  "summary": "Overall good code quality with some areas for improvement in documentation and function complexity."
}
```

## Severity Guidelines

- **critical**: Security vulnerabilities, data corruption risks, crashes
- **major**: Bugs, significant style violations, performance issues
- **minor**: Minor style issues, small improvements
- **suggestion**: Optional enhancements, alternative approaches
- **praise**: Good practices worth highlighting

Now review the code.
"""

    def _get_default_conventions(self) -> str:
        """Return default coding conventions if none provided."""
        return """
## Default Coding Conventions

1. **Naming**
   - Use descriptive names
   - Follow language conventions (snake_case for Python, camelCase for JS)

2. **Functions**
   - Keep functions small and focused
   - Limit parameters (prefer 3 or fewer)
   - Single responsibility principle

3. **Comments**
   - Comment why, not what
   - Keep comments up to date
   - Use docstrings for public APIs

4. **Error Handling**
   - Handle errors explicitly
   - Don't swallow exceptions silently
   - Provide meaningful error messages

5. **Code Organization**
   - Group related code together
   - Use consistent file structure
   - Avoid deep nesting
"""

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response."""
        # Try to find JSON in code block
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))

        # Try to find raw JSON object
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"scores": {}, "comments": [], "summary": "Could not parse review"}


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "ReviewerAgent",
    "ReviewDecision",
    "CommentSeverity",
    "ReviewCategory",
    "ReviewComment",
    "FileReview",
    "ReviewReport",
    "FILE_REVIEW_SCHEMA",
]
