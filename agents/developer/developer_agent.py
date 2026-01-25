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

# =============================================================================
# DEVELOPER AGENT CLASS
# =============================================================================
"""
class DeveloperAgent(AgentInterface):
    '''
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
    '''

    def get_agent_type(self) -> str:
        return "developer"

    def validate_context(self, context: AgentContext) -> bool:
        '''
        Validate context for development.

        Requirements:
        - Task issue exists with requirements
        - Repository is accessible
        - Conventions are available (warning if not)
        '''
        if not context.issue_data:
            return False
        if not context.issue_data.get("body"):
            return False
        return True

    def execute(self, context: AgentContext) -> AgentResult:
        '''
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
        '''

        # =====================================================================
        # STEP 1: Parse Task Requirements
        # =====================================================================
        '''
        Extract from issue:
        - What needs to be implemented
        - Acceptance criteria
        - Implementation hints
        - Related files mentioned

        task_info = {
            "title": context.issue_data["title"],
            "description": self._extract_description(context.issue_data["body"]),
            "acceptance_criteria": self._extract_criteria(context.issue_data["body"]),
            "hints": self._extract_hints(context.issue_data["body"])
        }
        '''

        # =====================================================================
        # STEP 2: Load QA Feedback (if retry)
        # =====================================================================
        '''
        If iteration > 0, load previous feedback:

        if context.iteration > 0:
            qa_feedback = context.input_data.get("qa_feedback", {})
            # Feedback includes:
            # - What failed
            # - Specific issues found
            # - Suggested fixes
        '''

        # =====================================================================
        # STEP 3: Analyze Relevant Codebase
        # =====================================================================
        '''
        Identify and load relevant code:
        - Files mentioned in task
        - Files with related functionality
        - Test files for reference
        - Configuration files

        Use LLM to identify relevant files from task description.

        relevant_code = self._find_relevant_code(
            task_info,
            context.repository
        )
        '''

        # =====================================================================
        # STEP 4: Plan Implementation
        # =====================================================================
        '''
        Use LLM to create implementation plan:

        plan = self.llm.generate(
            prompt=self._format_planning_prompt(
                task_info,
                relevant_code,
                context.memory.get("CONVENTIONS.md", "")
            )
        )

        Plan includes:
        - Files to create/modify
        - High-level approach
        - Potential challenges
        '''

        # =====================================================================
        # STEP 5: Generate Code Changes
        # =====================================================================
        '''
        For each file in plan, generate changes:

        changes = []
        for file_plan in plan["files"]:
            if file_plan["action"] == "create":
                content = self._generate_new_file(file_plan, context)
            else:
                content = self._generate_modifications(file_plan, context)

            changes.append({
                "path": file_plan["path"],
                "action": file_plan["action"],
                "content": content
            })
        '''

        # =====================================================================
        # STEP 6: Apply Changes to Files
        # =====================================================================
        '''
        Write changes to repository volume:

        modified_files = []
        for change in changes:
            full_path = os.path.join(self.repo_path, change["path"])

            if change["action"] == "create":
                self._create_file(full_path, change["content"])
            else:
                self._modify_file(full_path, change["content"])

            modified_files.append({
                "path": change["path"],
                "action": change["action"],
                "lines_changed": self._count_changes(change)
            })
        '''

        # =====================================================================
        # STEP 7: Verify Changes
        # =====================================================================
        '''
        Basic verification of changes:
        - Syntax check (parse files)
        - Linter check (if configured)
        - Import check (no missing imports)

        verification = self._verify_changes(modified_files)
        if not verification["success"]:
            # Attempt to fix issues
            pass
        '''

        # =====================================================================
        # STEP 8: Git Operations
        # =====================================================================
        '''
        If configured, handle git:
        - Create branch: agent/issue-{number}
        - Stage changes
        - Create commit (but don't push)

        git_info = self._handle_git(
            context.issue_number,
            modified_files
        )
        '''

        # =====================================================================
        # STEP 9: Return Results
        # =====================================================================
        '''
        return AgentResult(
            status=AgentStatus.SUCCESS,
            output={
                "modified_files": modified_files,
                "commit_message": self._generate_commit_message(task_info, changes),
                "branch_name": git_info.get("branch"),
                "implementation_notes": plan.get("notes", ""),
                "tests_added": self._identify_tests(modified_files)
            },
            message=f"Implemented task: {len(modified_files)} files modified",
            metrics={
                "files_modified": len(modified_files),
                "lines_added": sum(f.get("lines_changed", 0) for f in modified_files),
                "llm_calls": llm_call_count,
                "tokens_used": total_tokens
            }
        )
        '''

        pass  # Implementation placeholder
'''
"""

# =============================================================================
# IMPLEMENTATION NOTES
# =============================================================================
"""
Key Implementation Considerations:

1. CODE QUALITY
   - Follow project conventions from CONVENTIONS.md
   - Match existing code style
   - Add appropriate comments
   - Handle errors properly

2. TESTING
   - Add tests for new functionality when appropriate
   - Update existing tests if behavior changes
   - Ensure tests are runnable

3. SAFETY
   - Never delete files without explicit instruction
   - Create backups before major changes
   - Limit scope of changes

4. ITERATION HANDLING
   - On retry, focus on fixing specific issues
   - Don't rewrite entire implementation
   - Address QA feedback directly

5. GIT WORKFLOW
   - Create descriptive branch names
   - Write meaningful commit messages
   - Don't push (orchestrator handles this)
"""
