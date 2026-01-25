# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - OUTPUT HANDLER
# =============================================================================
"""
Output Handler Module

This module handles writing agent outputs in a consistent format.
All agents use this handler to:
1. Write results to output volume
2. Format output for orchestrator consumption
3. Log execution metrics
4. Handle errors gracefully

Output Structure:
    /output/
    ├── result.json          # Main result file (read by orchestrator)
    ├── logs/
    │   └── execution.log    # Detailed execution log
    ├── artifacts/           # Agent-specific artifacts
    │   ├── changes.patch    # (Developer) Code changes
    │   ├── test_results.xml # (QA) Test results
    │   └── review.md        # (Reviewer) Review notes
    └── metrics.json         # Execution metrics
"""

import os
import json
import tempfile
import shutil
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Union
from pathlib import Path

from .agent_interface import AgentResult, AgentStatus


logger = logging.getLogger(__name__)


# =============================================================================
# OUTPUT HANDLER CLASS
# =============================================================================

class OutputHandler:
    """
    Handles writing agent outputs to the output volume.

    This class ensures:
    1. Consistent output format across agents
    2. Proper file permissions and encoding
    3. Atomic writes (temp file + rename)
    4. Error handling for write failures

    Attributes:
        output_path: Base path for output files
        agent_type: Type of agent producing output
        issue_number: Issue being processed

    Usage:
        handler = OutputHandler(
            output_path="/output",
            agent_type="developer",
            issue_number=123
        )

        # Write main result
        handler.write_result(agent_result)

        # Write artifacts
        handler.write_artifact("changes.patch", patch_content)

        # Write metrics
        handler.write_metrics(metrics_dict)
    """

    def __init__(
        self,
        output_path: str = None,
        agent_type: str = None,
        issue_number: int = None
    ):
        """
        Initialize the output handler.

        Args:
            output_path: Path to output volume (default: from OUTPUT_PATH env)
            agent_type: Agent type (default: from AGENT_TYPE env)
            issue_number: Issue number (default: from ISSUE_NUMBER env)

        Creates output directory structure if not exists.
        """
        self.output_path = output_path or os.environ.get("OUTPUT_PATH", "/output")
        self.agent_type = agent_type or os.environ.get("AGENT_TYPE", "unknown")
        self.issue_number = issue_number or int(os.environ.get("ISSUE_NUMBER", "0"))

        # Ensure directories exist
        self._ensure_directories()

    def _ensure_directories(self):
        """Create output directory structure if needed."""
        dirs = [
            self.output_path,
            os.path.join(self.output_path, "logs"),
            os.path.join(self.output_path, "artifacts")
        ]

        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

    def write_result(self, result: AgentResult) -> bool:
        """
        Write the main result file.

        Args:
            result: AgentResult to write

        Returns:
            True if successful, False otherwise

        Writes to: {output_path}/result.json

        The orchestrator reads this file to determine
        agent completion status and get output data.
        """
        try:
            result_path = os.path.join(self.output_path, "result.json")

            # Build full result structure
            full_result = {
                "status": result.status.value,
                "agent_type": self.agent_type,
                "issue_number": self.issue_number,
                "timestamp": result.timestamp,
                "output": result.output,
                "message": result.message,
                "details": result.details,
                "errors": result.errors,
                "metrics": result.metrics
            }

            # Write atomically
            content = json.dumps(full_result, indent=2, default=str)
            self._atomic_write(result_path, content)

            logger.info(f"Result written to {result_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to write result: {e}")
            return False

    def write_artifact(self, name: str, content: Union[str, bytes]) -> bool:
        """
        Write an artifact file.

        Args:
            name: Artifact filename
            content: Artifact content (str or bytes)

        Returns:
            True if successful, False otherwise

        Writes to: {output_path}/artifacts/{name}
        """
        try:
            artifact_path = os.path.join(self.output_path, "artifacts", name)

            # Ensure parent directory exists
            os.makedirs(os.path.dirname(artifact_path), exist_ok=True)

            if isinstance(content, bytes):
                with open(artifact_path, "wb") as f:
                    f.write(content)
            else:
                self._atomic_write(artifact_path, content)

            logger.debug(f"Artifact written: {name}")
            return True

        except Exception as e:
            logger.error(f"Failed to write artifact {name}: {e}")
            return False

    def write_metrics(self, metrics: Dict[str, Any]) -> bool:
        """
        Write execution metrics.

        Args:
            metrics: Metrics dictionary

        Returns:
            True if successful

        Writes to: {output_path}/metrics.json
        """
        try:
            metrics_path = os.path.join(self.output_path, "metrics.json")

            # Add metadata
            full_metrics = {
                "agent_type": self.agent_type,
                "issue_number": self.issue_number,
                "timestamp": datetime.utcnow().isoformat(),
                **metrics
            }

            content = json.dumps(full_metrics, indent=2, default=str)
            self._atomic_write(metrics_path, content)

            logger.debug("Metrics written")
            return True

        except Exception as e:
            logger.error(f"Failed to write metrics: {e}")
            return False

    def write_log(self, message: str, level: str = "INFO"):
        """
        Write to execution log.

        Args:
            message: Log message
            level: Log level (DEBUG, INFO, WARNING, ERROR)

        Appends to: {output_path}/logs/execution.log
        """
        try:
            log_path = os.path.join(self.output_path, "logs", "execution.log")
            timestamp = datetime.utcnow().isoformat()
            log_line = f"{timestamp} [{level}] {message}\n"

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(log_line)

        except Exception as e:
            logger.error(f"Failed to write to execution log: {e}")

    def _atomic_write(self, path: str, content: str):
        """
        Write file atomically using temp file + rename.

        Ensures partial writes don't corrupt output.
        """
        # Get directory for temp file
        dir_path = os.path.dirname(path)

        # Write to temp file
        fd, temp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)

            # Atomic rename
            shutil.move(temp_path, path)

        except Exception:
            # Clean up temp file on failure
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def read_previous_result(self) -> Optional[AgentResult]:
        """
        Read previous result if it exists.

        Useful for checking previous state on retry.

        Returns:
            AgentResult or None
        """
        result_path = os.path.join(self.output_path, "result.json")

        if not os.path.exists(result_path):
            return None

        try:
            with open(result_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return AgentResult.from_dict(data)

        except Exception as e:
            logger.warning(f"Failed to read previous result: {e}")
            return None


# =============================================================================
# OUTPUT FORMATTERS
# =============================================================================

class ResultFormatter:
    """
    Formats AgentResult for different output types.

    Supports:
    - JSON (default, for orchestrator)
    - Markdown (for GitHub comments)
    - Plain text (for logs)
    """

    # Status emoji mapping
    STATUS_EMOJI = {
        AgentStatus.SUCCESS: "✅",
        AgentStatus.FAILURE: "❌",
        AgentStatus.ERROR: "🔴",
        AgentStatus.TIMEOUT: "⏰"
    }

    @staticmethod
    def to_json(result: AgentResult) -> str:
        """Format result as JSON string."""
        return json.dumps(result.to_dict(), indent=2, default=str)

    @staticmethod
    def to_markdown(result: AgentResult, agent_type: str = "Agent") -> str:
        """
        Format result as Markdown for GitHub comment.

        Args:
            result: AgentResult to format
            agent_type: Type of agent for header

        Returns:
            Formatted markdown string
        """
        emoji = ResultFormatter.STATUS_EMOJI.get(result.status, "❓")
        status_text = result.status.value.upper()

        lines = [
            f"## {agent_type.title()} Agent Report",
            "",
            f"**Status:** {emoji} {status_text}",
            "",
            f"**Summary:** {result.message}",
            ""
        ]

        # Add details if present
        if result.details:
            lines.append("### Details")
            lines.append("")
            for key, value in result.details.items():
                if isinstance(value, list):
                    lines.append(f"**{key}:**")
                    for item in value[:10]:  # Limit to 10 items
                        lines.append(f"- {item}")
                elif isinstance(value, dict):
                    lines.append(f"**{key}:** `{json.dumps(value)}`")
                else:
                    lines.append(f"**{key}:** {value}")
            lines.append("")

        # Add output highlights
        if result.output:
            lines.append("### Output")
            lines.append("")
            lines.append("```json")
            # Truncate large output
            output_str = json.dumps(result.output, indent=2, default=str)
            if len(output_str) > 2000:
                output_str = output_str[:2000] + "\n... (truncated)"
            lines.append(output_str)
            lines.append("```")
            lines.append("")

        # Add errors if any
        if result.errors:
            lines.append("### Errors")
            lines.append("")
            for error in result.errors[:5]:  # Limit to 5 errors
                # Truncate long error messages
                error_text = str(error)
                if len(error_text) > 500:
                    error_text = error_text[:500] + "..."
                lines.append(f"- {error_text}")
            lines.append("")

        # Add metrics
        if result.metrics:
            lines.append("### Metrics")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for key, value in result.metrics.items():
                formatted_value = ResultFormatter._format_metric_value(key, value)
                lines.append(f"| {key} | {formatted_value} |")
            lines.append("")

        # Add timestamp
        lines.append(f"---")
        lines.append(f"*Generated at {result.timestamp}*")

        return "\n".join(lines)

    @staticmethod
    def to_text(result: AgentResult) -> str:
        """Format result as plain text for logs."""
        lines = [
            f"=== Agent Result ===",
            f"Status: {result.status.value}",
            f"Message: {result.message}",
            f"Timestamp: {result.timestamp}"
        ]

        if result.errors:
            lines.append(f"Errors: {len(result.errors)}")
            for error in result.errors[:3]:
                lines.append(f"  - {error[:100]}")

        if result.metrics:
            lines.append("Metrics:")
            for key, value in result.metrics.items():
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)

    @staticmethod
    def _format_metric_value(key: str, value: Any) -> str:
        """Format a metric value for display."""
        if key == "duration_seconds":
            return f"{value:.2f}s"
        elif "tokens" in key.lower():
            return f"{value:,}"
        elif isinstance(value, float):
            return f"{value:.2f}"
        elif isinstance(value, int):
            return f"{value:,}"
        else:
            return str(value)


# =============================================================================
# AGENT-SPECIFIC OUTPUT SCHEMAS
# =============================================================================

# These schemas document the expected output structure for each agent type.
# They can be used for validation or documentation purposes.

PLANNER_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "created_issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "number": {"type": "integer"},
                    "title": {"type": "string"},
                    "type": {"type": "string"},
                    "estimate_hours": {"type": "number"}
                }
            }
        },
        "feature_memory_file": {"type": "string"},
        "decomposition_summary": {"type": "string"}
    }
}

DEVELOPER_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "modified_files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "action": {"type": "string"},
                    "lines_changed": {"type": "integer"}
                }
            }
        },
        "commit_message": {"type": "string"},
        "branch_name": {"type": "string"},
        "implementation_notes": {"type": "string"},
        "tests_added": {"type": "array", "items": {"type": "string"}}
    }
}

QA_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "qa_result": {"type": "string", "enum": ["PASS", "FAIL"]},
        "acceptance_checklist": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "criterion": {"type": "string"},
                    "result": {"type": "string"},
                    "evidence": {"type": "string"}
                }
            }
        },
        "test_results": {"type": "object"},
        "feedback": {"type": "string"},
        "suggested_fixes": {"type": "array", "items": {"type": "string"}}
    }
}

REVIEWER_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "review_result": {"type": "string", "enum": ["APPROVED", "CHANGES_REQUESTED"]},
        "quality_score": {"type": "integer", "minimum": 0, "maximum": 100},
        "comments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "line": {"type": "integer"},
                    "comment": {"type": "string"}
                }
            }
        },
        "improvement_areas": {"type": "array", "items": {"type": "string"}}
    }
}

DOC_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "updated_files": {"type": "array", "items": {"type": "string"}},
        "changelog_entry": {"type": "string"},
        "documentation_notes": {"type": "string"}
    }
}


def validate_output(output: Dict[str, Any], agent_type: str) -> bool:
    """
    Validate agent output against schema.

    Args:
        output: Output dictionary to validate
        agent_type: Agent type for schema selection

    Returns:
        True if valid
    """
    schemas = {
        "planner": PLANNER_OUTPUT_SCHEMA,
        "developer": DEVELOPER_OUTPUT_SCHEMA,
        "qa": QA_OUTPUT_SCHEMA,
        "reviewer": REVIEWER_OUTPUT_SCHEMA,
        "doc": DOC_OUTPUT_SCHEMA
    }

    schema = schemas.get(agent_type)
    if not schema:
        return True  # No validation for unknown types

    # Basic validation - check required properties exist
    properties = schema.get("properties", {})
    for prop in properties:
        if prop not in output:
            logger.warning(f"Missing expected output property: {prop}")

    return True
