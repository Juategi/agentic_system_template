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

Result File Format:
    {
        "status": "success|failure|error",
        "agent_type": "developer",
        "issue_number": 123,
        "timestamp": "2024-01-01T00:00:00Z",
        "output": {
            // Agent-specific output
        },
        "message": "Human-readable summary",
        "errors": [],
        "metrics": {
            "duration_seconds": 120,
            "llm_calls": 5,
            "tokens_used": 10000
        }
    }
"""

# =============================================================================
# OUTPUT HANDLER CLASS
# =============================================================================
"""
class OutputHandler:
    '''
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
    '''

    def __init__(
        self,
        output_path: str = None,
        agent_type: str = None,
        issue_number: int = None
    ):
        '''
        Initialize the output handler.

        Args:
            output_path: Path to output volume (default: from OUTPUT_PATH env)
            agent_type: Agent type (default: from AGENT_TYPE env)
            issue_number: Issue number (default: from ISSUE_NUMBER env)

        Creates output directory structure if not exists.
        '''
        pass

    def write_result(self, result: AgentResult) -> bool:
        '''
        Write the main result file.

        Args:
            result: AgentResult to write

        Returns:
            True if successful, False otherwise

        Writes to: {output_path}/result.json

        The orchestrator reads this file to determine
        agent completion status and get output data.
        '''
        pass

    def write_artifact(self, name: str, content: str | bytes) -> bool:
        '''
        Write an artifact file.

        Args:
            name: Artifact filename
            content: Artifact content (str or bytes)

        Returns:
            True if successful, False otherwise

        Writes to: {output_path}/artifacts/{name}

        Common artifacts:
        - Developer: changes.patch, diff.txt
        - QA: test_results.xml, coverage.html
        - Reviewer: review.md, suggestions.json
        '''
        pass

    def write_metrics(self, metrics: dict) -> bool:
        '''
        Write execution metrics.

        Args:
            metrics: Metrics dictionary

        Returns:
            True if successful

        Writes to: {output_path}/metrics.json

        Standard metrics:
        - duration_seconds: Total execution time
        - llm_calls: Number of LLM API calls
        - tokens_input: Input tokens used
        - tokens_output: Output tokens used
        - files_read: Number of files read
        - files_modified: Number of files modified
        '''
        pass

    def write_log(self, message: str, level: str = "INFO"):
        '''
        Write to execution log.

        Args:
            message: Log message
            level: Log level (DEBUG, INFO, WARNING, ERROR)

        Appends to: {output_path}/logs/execution.log
        '''
        pass

    def _ensure_directories(self):
        '''Create output directory structure if needed.'''
        pass

    def _atomic_write(self, path: str, content: str):
        '''
        Write file atomically using temp file + rename.

        Ensures partial writes don't corrupt output.
        '''
        pass
'''

# =============================================================================
# OUTPUT FORMATTERS
# =============================================================================
'''
class ResultFormatter:
    '''
    Formats AgentResult for different output types.

    Supports:
    - JSON (default, for orchestrator)
    - Markdown (for GitHub comments)
    - Plain text (for logs)
    '''

    @staticmethod
    def to_json(result: AgentResult) -> str:
        '''Format result as JSON string.'''
        pass

    @staticmethod
    def to_markdown(result: AgentResult) -> str:
        '''
        Format result as Markdown for GitHub comment.

        Returns formatted string like:

        ## Agent Execution Complete

        **Status:** ✅ Success

        **Summary:**
        {message}

        **Details:**
        {formatted details}

        **Metrics:**
        - Duration: {duration}s
        - LLM Calls: {calls}
        '''
        pass

    @staticmethod
    def to_text(result: AgentResult) -> str:
        '''Format result as plain text for logs.'''
        pass
'''

# =============================================================================
# AGENT-SPECIFIC OUTPUT SCHEMAS
# =============================================================================
'''
# Each agent type has specific output fields

PLANNER_OUTPUT_SCHEMA = {
    "created_issues": [
        {
            "number": 124,
            "title": "Task title",
            "type": "task",
            "estimate_hours": 4
        }
    ],
    "feature_memory_file": "features/feature-123.md",
    "decomposition_summary": "Created 5 tasks from feature"
}

DEVELOPER_OUTPUT_SCHEMA = {
    "modified_files": [
        {"path": "src/file.py", "action": "modified", "lines_changed": 50}
    ],
    "commit_message": "Implement feature X",
    "branch_name": "agent/issue-123",
    "implementation_notes": "Description of implementation approach",
    "tests_added": ["test_feature_x"]
}

QA_OUTPUT_SCHEMA = {
    "qa_result": "PASS|FAIL",
    "acceptance_checklist": [
        {"id": 1, "criterion": "...", "result": "PASS|FAIL", "evidence": "..."}
    ],
    "test_results": {
        "unit": {"passed": 10, "failed": 0, "output": "..."},
        "lint": {"errors": 0, "warnings": 2}
    },
    "feedback": "Detailed feedback if failed",
    "suggested_fixes": ["Fix suggestion 1", "Fix suggestion 2"]
}

REVIEWER_OUTPUT_SCHEMA = {
    "review_result": "APPROVED|CHANGES_REQUESTED",
    "quality_score": 85,
    "comments": [
        {"file": "src/file.py", "line": 10, "comment": "..."}
    ],
    "improvement_areas": ["Consider refactoring X"]
}

DOC_OUTPUT_SCHEMA = {
    "updated_files": ["memory/features/feature-123.md"],
    "changelog_entry": "Added feature X",
    "documentation_notes": "Updated memory with implementation details"
}
'''
"""

# =============================================================================
# IMPLEMENTATION NOTES
# =============================================================================
"""
Implementation Notes:

1. ATOMIC WRITES
   Always use atomic writes for result.json:
   - Write to temp file
   - Rename to final path
   This prevents partial reads by orchestrator.

2. ENCODING
   All text files use UTF-8 encoding.
   Binary artifacts (images, etc.) written as-is.

3. FILE PERMISSIONS
   Output files should be readable by orchestrator.
   Default permissions: 644 for files, 755 for dirs.

4. ERROR HANDLING
   If write fails:
   - Log error
   - Attempt to write error result
   - Return False to caller

5. METRICS COLLECTION
   Agents should track and report:
   - Execution duration
   - LLM token usage
   - Files processed
   These enable cost and performance monitoring.
"""
