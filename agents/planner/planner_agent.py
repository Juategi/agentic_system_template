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

LLM Interaction:
    The agent sends the feature context to the LLM with a structured
    prompt asking for:
    - Task breakdown with titles and descriptions
    - Acceptance criteria for each task
    - Dependency relationships
    - Implementation notes

Quality Considerations:
    - Tasks should be independently testable
    - Tasks should have clear boundaries
    - Dependencies should form a valid DAG
    - Total tasks should cover all acceptance criteria
"""

# =============================================================================
# PLANNER AGENT CLASS
# =============================================================================
"""
class PlannerAgent(AgentInterface):
    '''
    Agent that decomposes features into tasks.

    This agent:
    1. Reads feature issue and project context
    2. Uses LLM to analyze and decompose
    3. Creates sub-task issues in GitHub
    4. Creates feature memory file
    5. Returns decomposition results

    Attributes:
        llm: LLM client for analysis
        github: GitHub API helper
        config: Agent configuration

    Configuration:
        max_sub_issues: Maximum tasks to create (default: 10)
        min_granularity_hours: Minimum task size (default: 2)
        max_granularity_hours: Maximum task size (default: 8)
        create_dependencies: Whether to link dependencies (default: true)
    '''

    def get_agent_type(self) -> str:
        return "planner"

    def validate_context(self, context: AgentContext) -> bool:
        '''
        Validate context for planning.

        Requirements:
        - Issue must exist and be accessible
        - Issue should be type "feature"
        - Project memory should be available
        '''
        # Check issue data exists
        if not context.issue_data:
            self.logger.error("No issue data in context")
            return False

        # Check issue has body (requirements)
        if not context.issue_data.get("body"):
            self.logger.error("Issue has no body/description")
            return False

        # Check project context exists
        if "PROJECT.md" not in context.memory:
            self.logger.warning("PROJECT.md not found, proceeding with limited context")

        return True

    def execute(self, context: AgentContext) -> AgentResult:
        '''
        Execute feature decomposition.

        Steps:
        1. Extract feature information
        2. Prepare LLM prompt
        3. Get decomposition from LLM
        4. Validate decomposition
        5. Create GitHub issues
        6. Create feature memory file
        7. Return results
        '''

        # =====================================================================
        # STEP 1: Extract Feature Information
        # =====================================================================
        '''
        Parse the feature issue to extract:
        - Title and description
        - Acceptance criteria
        - Technical constraints
        - Any mentioned dependencies

        feature_info = {
            "title": context.issue_data["title"],
            "description": self._extract_description(context.issue_data["body"]),
            "acceptance_criteria": self._extract_criteria(context.issue_data["body"]),
            "constraints": self._extract_constraints(context.issue_data["body"])
        }
        '''

        # =====================================================================
        # STEP 2: Prepare LLM Prompt
        # =====================================================================
        '''
        Load prompt template and fill with context:

        prompt = self._load_prompt("decomposition.md")
        filled_prompt = prompt.format(
            feature_title=feature_info["title"],
            feature_description=feature_info["description"],
            acceptance_criteria=feature_info["acceptance_criteria"],
            project_context=context.memory.get("PROJECT.md", ""),
            architecture=context.memory.get("ARCHITECTURE.md", ""),
            config=json.dumps(self.config)
        )
        '''

        # =====================================================================
        # STEP 3: Get Decomposition from LLM
        # =====================================================================
        '''
        Call LLM with structured output request:

        response = self.llm.generate(
            prompt=filled_prompt,
            response_format={
                "type": "json",
                "schema": DECOMPOSITION_SCHEMA
            }
        )

        Expected response:
        {
            "tasks": [
                {
                    "title": "Implement user registration",
                    "description": "...",
                    "acceptance_criteria": ["..."],
                    "estimate_hours": 4,
                    "dependencies": []
                },
                ...
            ],
            "reasoning": "Explanation of decomposition approach"
        }
        '''

        # =====================================================================
        # STEP 4: Validate Decomposition
        # =====================================================================
        '''
        Validate the LLM output:
        - Number of tasks within limits
        - Each task has required fields
        - Dependencies reference valid tasks
        - No circular dependencies
        - Estimates within configured range

        If validation fails, may retry with feedback.
        '''

        # =====================================================================
        # STEP 5: Create GitHub Issues
        # =====================================================================
        '''
        For each task in decomposition:

        created_issues = []
        for task in decomposition["tasks"]:
            issue_body = self._format_task_body(
                task,
                parent_issue=context.issue_number
            )

            issue_number = self.github.create_issue(
                title=f"[Task] {task['title']}",
                body=issue_body,
                labels=["task", "READY", "subtask"]
            )

            # Link to parent
            self.github.add_comment(
                context.issue_number,
                f"Created subtask: #{issue_number}"
            )

            created_issues.append({
                "number": issue_number,
                "title": task["title"],
                "estimate_hours": task["estimate_hours"]
            })
        '''

        # =====================================================================
        # STEP 6: Create Feature Memory File
        # =====================================================================
        '''
        Create markdown file documenting the feature:

        memory_content = self._generate_feature_memory(
            feature_info=feature_info,
            tasks=created_issues,
            decomposition=decomposition
        )

        memory_path = f"features/feature-{context.issue_number}.md"
        self._write_memory_file(memory_path, memory_content)
        '''

        # =====================================================================
        # STEP 7: Return Results
        # =====================================================================
        '''
        return AgentResult(
            status=AgentStatus.SUCCESS,
            output={
                "created_issues": created_issues,
                "feature_memory_file": memory_path,
                "decomposition_summary": decomposition["reasoning"]
            },
            message=f"Decomposed feature into {len(created_issues)} tasks",
            metrics={
                "tasks_created": len(created_issues),
                "llm_calls": 1,
                "tokens_used": response.usage.total_tokens
            }
        )
        '''

        pass  # Implementation placeholder
'''

# =============================================================================
# HELPER METHODS
# =============================================================================
'''
    def _extract_description(self, body: str) -> str:
        """Extract main description from issue body."""
        pass

    def _extract_criteria(self, body: str) -> list:
        """Extract acceptance criteria as list."""
        pass

    def _format_task_body(self, task: dict, parent_issue: int) -> str:
        """
        Format task as GitHub issue body.

        Template:
        ## Description
        {task description}

        ## Acceptance Criteria
        - [ ] {criterion 1}
        - [ ] {criterion 2}

        ## Parent Feature
        Part of #{parent_issue}

        ## Estimate
        ~{hours} hours
        """
        pass

    def _generate_feature_memory(
        self,
        feature_info: dict,
        tasks: list,
        decomposition: dict
    ) -> str:
        """
        Generate feature memory markdown content.

        Template:
        # Feature: {title}

        ## Metadata
        - Issue: #{number}
        - Status: PLANNING_COMPLETE
        - Created: {timestamp}
        - Tasks: {count}

        ## Description
        {description}

        ## Acceptance Criteria
        {criteria}

        ## Tasks
        | # | Title | Status | Estimate |
        |---|-------|--------|----------|
        | 124 | Task 1 | READY | 4h |

        ## Dependencies
        {dependency graph}

        ## Planning Notes
        {decomposition reasoning}

        ## History
        | Date | Event | Details |
        |------|-------|---------|
        | {date} | Planned | Created {n} tasks |
        """
        pass

    def _validate_dependencies(self, tasks: list) -> bool:
        """
        Validate dependency graph has no cycles.

        Uses topological sort to detect cycles.
        """
        pass
'''
"""

# =============================================================================
# DECOMPOSITION SCHEMA
# =============================================================================
"""
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
"""
