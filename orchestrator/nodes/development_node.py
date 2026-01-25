# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - DEVELOPMENT NODE
# =============================================================================
"""
Development Node Implementation

This node handles the implementation phase where the Developer Agent
writes code to fulfill the requirements specified in a GitHub Issue.

Workflow Position:
    TRIAGE ──(is_task)──▶ DEVELOPMENT ──▶ QA
                               ▲
    QA_FAILED ─────────────────┘
    REVIEW ──(changes_requested)──┘

Responsibilities:
    1. Prepare development context
    2. Include previous QA feedback (if retry)
    3. Launch Developer Agent container
    4. Wait for agent completion
    5. Capture modified files
    6. Update workflow state

Input State Requirements:
    - issue_number: The task issue to implement
    - issue_type: Must be "task" or "bug"
    - (Optional) metadata.qa_feedback: Feedback from failed QA

Output State Changes:
    - last_agent_output: Developer agent results
    - metadata.modified_files: List of files changed
    - metadata.branch_name: Git branch created (if any)
    - iteration_count: Unchanged (incremented in qa_failed_node)

The Dev→QA Loop:
    This node may be entered multiple times for the same issue
    as part of the iterative development loop. Each iteration
    receives feedback from the previous QA cycle.
"""

# =============================================================================
# NODE IMPLEMENTATION
# =============================================================================
"""
async def development_node(state: WorkflowState, context: NodeContext) -> WorkflowState:
    '''
    Execute the development node logic.

    Args:
        state: Current workflow state
        context: Node execution context

    Returns:
        Updated workflow state

    Execution Flow:
    1. Update issue label to IN_PROGRESS
    2. Determine if this is first attempt or retry
    3. Fetch issue details and requirements
    4. Load relevant memory and context
    5. Include QA feedback if retry
    6. Launch Developer Agent
    7. Wait for completion
    8. Parse and validate output
    9. Update state with results
    '''

    # -------------------------------------------------------------------------
    # STEP 1: Update Issue Status
    # -------------------------------------------------------------------------
    '''
    Update GitHub issue:
    - Add label: IN_PROGRESS
    - Remove labels: QA_FAILED, READY
    - Add comment: "Developer Agent starting work (iteration X/Y)"
    '''

    # -------------------------------------------------------------------------
    # STEP 2: Determine Iteration Context
    # -------------------------------------------------------------------------
    '''
    Check state.iteration_count:
    - If 0: This is the first attempt
    - If > 0: This is a retry after QA failure

    For retries:
    - Load previous QA feedback from state.metadata.qa_feedback
    - Include in agent context
    - Log iteration number
    '''

    # -------------------------------------------------------------------------
    # STEP 3: Fetch Issue Details
    # -------------------------------------------------------------------------
    '''
    Get from GitHub:
    - Issue title and body
    - Acceptance criteria
    - Implementation notes (if any)
    - Related issues/dependencies

    Parse issue body to extract structured information.
    '''

    # -------------------------------------------------------------------------
    # STEP 4: Load Context
    # -------------------------------------------------------------------------
    '''
    Load relevant files:

    Memory:
    - PROJECT.md: Project rules
    - CONVENTIONS.md: Coding standards
    - Feature memory file (if subtask)

    Codebase:
    - Related existing code
    - Test files
    - Configuration files

    The context helps the agent understand:
    - Where to make changes
    - What patterns to follow
    - What already exists
    '''

    # -------------------------------------------------------------------------
    # STEP 5: Prepare Agent Input
    # -------------------------------------------------------------------------
    '''
    Create input package for Developer Agent:
    {
        "issue_number": 123,
        "iteration": 1,
        "task": {
            "title": "...",
            "description": "...",
            "acceptance_criteria": [...],
            "implementation_notes": "..."
        },
        "context": {
            "project_rules": "...",
            "conventions": "...",
            "related_code": {...}
        },
        "feedback": {  // Only on retry
            "qa_result": "FAIL",
            "issues_found": [...],
            "suggestions": [...]
        },
        "config": {
            "create_branch": true,
            "branch_prefix": "agent/",
            "run_linter": true
        }
    }
    '''

    # -------------------------------------------------------------------------
    # STEP 6: Launch Developer Agent
    # -------------------------------------------------------------------------
    '''
    Start container with:
    - AGENT_TYPE=developer
    - ISSUE_NUMBER={issue_number}
    - ITERATION={iteration_count}
    - MAX_ITERATIONS={max_iterations}

    Volumes:
    - /memory: Project memory (read)
    - /repo: Code repository (read/write)
    - /output: Agent output (write)
    '''

    # -------------------------------------------------------------------------
    # STEP 7: Wait for Completion
    # -------------------------------------------------------------------------
    '''
    Monitor container:
    - Poll for completion
    - Capture stdout/stderr
    - Handle timeout (30 min default)

    On timeout:
    - Kill container
    - Log error
    - Don't retry (will go to QA which will fail)
    '''

    # -------------------------------------------------------------------------
    # STEP 8: Parse Agent Output
    # -------------------------------------------------------------------------
    '''
    Expected output structure:
    {
        "status": "success",
        "modified_files": [
            {"path": "src/feature.py", "action": "created"},
            {"path": "tests/test_feature.py", "action": "created"}
        ],
        "commit_message": "Implement feature X for issue #123",
        "branch_name": "agent/issue-123",
        "implementation_notes": "...",
        "tests_added": ["test_feature_basic", "test_feature_edge_case"]
    }

    Validate:
    - Status is success
    - Modified files list is present
    - Files actually exist in repo
    '''

    # -------------------------------------------------------------------------
    # STEP 9: Update State
    # -------------------------------------------------------------------------
    '''
    Update state with:
    - last_agent_output: Agent output
    - metadata.modified_files: List of paths
    - metadata.branch_name: Branch name (if created)
    - metadata.commit_sha: Commit hash (if committed)
    - history: Add development record

    State transitions to QA node next.
    '''

    pass  # Implementation placeholder
'''

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
'''
def extract_acceptance_criteria(issue_body: str) -> list[str]:
    '''
    Extract acceptance criteria from issue body.

    Looks for sections like:
    ## Acceptance Criteria
    - [ ] Criterion 1
    - [ ] Criterion 2

    Returns list of criteria strings.
    '''
    pass


def prepare_related_code_context(
    repo_path: str,
    issue_body: str,
    memory: dict
) -> dict:
    '''
    Identify and load related code for context.

    Uses:
    - Mentions in issue body
    - Feature memory references
    - Import analysis

    Returns dict of file_path -> content.
    '''
    pass


def validate_developer_output(
    output: dict,
    repo_path: str
) -> tuple[bool, str]:
    '''
    Validate Developer Agent output.

    Checks:
    - Required fields present
    - Modified files exist
    - No obvious errors

    Returns (is_valid, error_message).
    '''
    pass
'''

# =============================================================================
# NODE CONFIGURATION
# =============================================================================
'''
DEVELOPMENT_NODE_CONFIG = {
    "agent_type": "developer",
    "timeout_seconds": 1800,  # 30 minutes
    "retry_on_timeout": False,
    "github_labels": {
        "add": ["IN_PROGRESS"],
        "remove": ["QA_FAILED", "READY"]
    },
    "comment_template": "🤖 **[AI Agent]** Developer Agent starting work (iteration {iteration}/{max_iterations})"
}
'''
"""

# =============================================================================
# IMPLEMENTATION NOTES
# =============================================================================
"""
Implementation Notes:

1. ITERATION HANDLING
   The development node handles both first attempts and retries:
   - First attempt: No prior context, fresh implementation
   - Retry: Includes QA feedback to guide fixes

   The iteration count is NOT incremented here (done in qa_failed_node).

2. GIT WORKFLOW
   The Developer Agent can:
   - Create a new branch (agent/{issue-number})
   - Make commits on that branch
   - NOT push to remote (orchestrator handles this)

   Branch/commit handling ensures:
   - Clean separation of changes
   - Easy rollback if needed
   - Clear audit trail

3. CONTEXT QUALITY
   The quality of agent output depends on context quality.
   Ensure:
   - Relevant code is included
   - Memory files are up to date
   - Previous feedback is clearly formatted

4. TIMEOUT HANDLING
   Development may take time for complex tasks.
   On timeout:
   - Container is killed
   - Partial work may be lost
   - Consider increasing timeout for complex issues
"""
