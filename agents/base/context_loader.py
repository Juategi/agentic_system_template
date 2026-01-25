# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - CONTEXT LOADER
# =============================================================================
"""
Context Loader Module

This module provides utilities for loading context data that agents need
to perform their tasks. It reads from:
1. Environment variables
2. Mounted volume files (memory, input)
3. GitHub API (issue details)

The ContextLoader creates an AgentContext object that agents use
throughout their execution.
"""

import os
import json
import glob
import re
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .agent_interface import AgentContext


logger = logging.getLogger(__name__)


# =============================================================================
# CONTEXT LOADER CLASS
# =============================================================================

class ContextLoader:
    """
    Loads and assembles context for agent execution.

    This class handles:
    1. Reading environment variables
    2. Loading memory files from volume
    3. Fetching issue data from GitHub
    4. Loading orchestrator input

    Attributes:
        memory_path: Path to memory volume
        repo_path: Path to repository volume
        input_path: Path to input file
        github_client: GitHub API client

    Usage:
        loader = ContextLoader()
        context = loader.load()
        # context is now AgentContext with all data
    """

    # Required environment variables
    REQUIRED_ENV_VARS = [
        "PROJECT_ID",
        "ISSUE_NUMBER",
        "GITHUB_TOKEN",
        "GITHUB_REPO"
    ]

    # Optional environment variables with defaults
    OPTIONAL_ENV_VARS = {
        "AGENT_TYPE": "developer",
        "ITERATION": "0",
        "MAX_ITERATIONS": "5",
        "LOG_LEVEL": "INFO",
        "MEMORY_PATH": "/memory",
        "REPO_PATH": "/repo",
        "INPUT_PATH": "/input/input.json",
        "OUTPUT_PATH": "/output"
    }

    # Memory files to load
    MEMORY_FILES = [
        "PROJECT.md",
        "ARCHITECTURE.md",
        "CONVENTIONS.md",
        "CONSTRAINTS.md"
    ]

    def __init__(
        self,
        memory_path: str = None,
        repo_path: str = None,
        input_path: str = None
    ):
        """
        Initialize the context loader.

        Args:
            memory_path: Path to memory volume (default: from env)
            repo_path: Path to repo volume (default: from env)
            input_path: Path to input file (default: /input/input.json)

        Reads default paths from environment:
        - MEMORY_PATH
        - REPO_PATH
        - INPUT_PATH
        """
        self.memory_path = memory_path or os.environ.get("MEMORY_PATH", "/memory")
        self.repo_path = repo_path or os.environ.get("REPO_PATH", "/repo")
        self.input_path = input_path or os.environ.get("INPUT_PATH", "/input/input.json")

        self._github_helper: Optional[GitHubHelper] = None
        self._env_cache: Optional[Dict[str, str]] = None

    def load(self) -> AgentContext:
        """
        Load complete context for agent.

        Returns:
            Populated AgentContext

        Loading sequence:
        1. Load environment variables
        2. Load memory files
        3. Fetch issue from GitHub
        4. Load orchestrator input
        5. Assemble context object
        """
        # 1. Load environment
        env = self._load_environment()
        issue_number = int(env["ISSUE_NUMBER"])
        project_id = env["PROJECT_ID"]

        # 2. Load memory files
        memory = self._load_memory()

        # 3. Load feature-specific memory
        feature_memory = self._load_feature_memory(issue_number)
        if feature_memory:
            memory[f"feature-{issue_number}"] = feature_memory

        # 4. Fetch issue from GitHub
        issue_data = self._fetch_issue(issue_number)

        # 5. Load orchestrator input
        input_data = self._load_orchestrator_input()

        # 6. Load repository info
        repository = self._load_repository_info()

        # 7. Assemble context
        return AgentContext(
            issue_number=issue_number,
            project_id=project_id,
            iteration=int(env.get("ITERATION", "0")),
            max_iterations=int(env.get("MAX_ITERATIONS", "5")),
            issue_data=issue_data,
            memory=memory,
            repository=repository,
            input_data=input_data,
            config=self._load_agent_config(env.get("AGENT_TYPE", "developer"))
        )

    def _load_environment(self) -> Dict[str, str]:
        """
        Load required environment variables.

        Returns:
            Dictionary of environment values

        Raises:
            ValueError: If required variable is missing
        """
        if self._env_cache is not None:
            return self._env_cache

        env = {}

        # Check required variables
        missing = []
        for var in self.REQUIRED_ENV_VARS:
            value = os.environ.get(var)
            if value is None:
                missing.append(var)
            else:
                env[var] = value

        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

        # Load optional variables with defaults
        for var, default in self.OPTIONAL_ENV_VARS.items():
            env[var] = os.environ.get(var, default)

        self._env_cache = env
        return env

    def _load_memory(self) -> Dict[str, str]:
        """
        Load project memory files.

        Returns:
            Dictionary mapping filename to content
        """
        memory = {}

        for filename in self.MEMORY_FILES:
            filepath = os.path.join(self.memory_path, filename)
            content = load_markdown_file(filepath)
            if content:
                memory[filename] = content
                logger.debug(f"Loaded memory file: {filename}")
            else:
                logger.debug(f"Memory file not found or empty: {filename}")

        return memory

    def _load_feature_memory(self, issue_number: int) -> str:
        """
        Load feature memory for specific issue.

        Returns:
            Feature memory content or empty string
        """
        features_path = os.path.join(self.memory_path, "features")

        # Try multiple naming conventions
        candidates = [
            f"feature-{issue_number}.md",
            f"issue-{issue_number}.md",
            f"{issue_number}.md"
        ]

        for candidate in candidates:
            filepath = os.path.join(features_path, candidate)
            content = load_markdown_file(filepath)
            if content:
                logger.debug(f"Loaded feature memory: {candidate}")
                return content

        # Search for files mentioning the issue number
        result = find_feature_memory_file(features_path, issue_number)
        if result:
            return load_markdown_file(result)

        return ""

    def _fetch_issue(self, issue_number: int) -> Dict[str, Any]:
        """
        Fetch issue details from GitHub API.

        Returns dictionary with issue data.
        """
        github = self._get_github_helper()
        try:
            return github.get_issue(issue_number)
        except Exception as e:
            logger.error(f"Failed to fetch issue #{issue_number}: {e}")
            return {
                "number": issue_number,
                "title": "",
                "body": "",
                "labels": [],
                "state": "unknown",
                "error": str(e)
            }

    def _load_orchestrator_input(self) -> Dict[str, Any]:
        """
        Load input file from orchestrator.

        Returns:
            Parsed input dictionary or empty dict
        """
        if not os.path.exists(self.input_path):
            logger.debug(f"No orchestrator input file at {self.input_path}")
            return {}

        try:
            with open(self.input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.debug("Loaded orchestrator input")
                return data
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in orchestrator input: {e}")
            return {}
        except Exception as e:
            logger.error(f"Failed to load orchestrator input: {e}")
            return {}

    def _load_repository_info(self) -> Dict[str, Any]:
        """Load basic repository information."""
        repo_info = {
            "path": self.repo_path,
            "exists": os.path.isdir(self.repo_path)
        }

        # Check for common project files
        common_files = [
            "package.json",
            "pyproject.toml",
            "setup.py",
            "Cargo.toml",
            "go.mod",
            "pom.xml"
        ]

        for filename in common_files:
            filepath = os.path.join(self.repo_path, filename)
            if os.path.exists(filepath):
                repo_info["project_file"] = filename
                break

        return repo_info

    def _load_agent_config(self, agent_type: str) -> Dict[str, Any]:
        """Load agent-specific configuration."""
        config_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "config", "agents.yaml"
        )

        if not os.path.exists(config_path):
            return {}

        try:
            import yaml
            with open(config_path, "r") as f:
                all_config = yaml.safe_load(f)

            agents = all_config.get("agents", {})
            return agents.get(agent_type, {})
        except Exception as e:
            logger.warning(f"Failed to load agent config: {e}")
            return {}

    def _get_github_helper(self) -> "GitHubHelper":
        """Get or create GitHub helper instance."""
        if self._github_helper is None:
            self._github_helper = GitHubHelper()
        return self._github_helper


# =============================================================================
# MEMORY FILE HELPERS
# =============================================================================

def load_markdown_file(path: str, max_size: int = 1_000_000) -> str:
    """
    Load a markdown file, handling common issues.

    Args:
        path: Path to markdown file
        max_size: Maximum file size in bytes (default 1MB)

    Returns:
        File content or empty string
    """
    if not os.path.exists(path):
        return ""

    try:
        size = os.path.getsize(path)
        if size > max_size:
            logger.warning(f"File too large ({size} bytes), truncating: {path}")

        # Try UTF-8 first
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read(max_size)
        except UnicodeDecodeError:
            # Fall back to latin-1
            with open(path, "r", encoding="latin-1") as f:
                content = f.read(max_size)

        return content

    except Exception as e:
        logger.error(f"Failed to load file {path}: {e}")
        return ""


def parse_memory_file(content: str) -> Dict[str, Any]:
    """
    Parse structured data from memory file.

    Args:
        content: Markdown content

    Returns:
        Dictionary of parsed sections
    """
    result = {
        "metadata": {},
        "sections": {},
        "raw": content
    }

    if not content:
        return result

    lines = content.split("\n")
    current_section = None
    section_content = []

    # Check for YAML frontmatter
    if lines[0].strip() == "---":
        frontmatter_end = -1
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == "---":
                frontmatter_end = i
                break

        if frontmatter_end > 0:
            try:
                import yaml
                frontmatter = "\n".join(lines[1:frontmatter_end])
                result["metadata"] = yaml.safe_load(frontmatter) or {}
                lines = lines[frontmatter_end + 1:]
            except Exception:
                pass

    # Parse sections by heading
    for line in lines:
        heading_match = re.match(r"^(#+)\s+(.+)$", line)
        if heading_match:
            # Save previous section
            if current_section:
                result["sections"][current_section] = "\n".join(section_content).strip()

            current_section = heading_match.group(2).strip()
            section_content = []
        else:
            section_content.append(line)

    # Save last section
    if current_section:
        result["sections"][current_section] = "\n".join(section_content).strip()

    return result


def find_feature_memory_file(
    memory_path: str,
    issue_number: int
) -> Optional[str]:
    """
    Find the memory file for a feature/issue.

    Args:
        memory_path: Path to memory directory
        issue_number: Issue number to find

    Returns:
        Path to memory file or None
    """
    if not os.path.isdir(memory_path):
        return None

    # Search all markdown files for the issue number
    pattern = os.path.join(memory_path, "*.md")
    for filepath in glob.glob(pattern):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read(5000)  # Read first 5KB

            # Look for issue reference
            if f"#{issue_number}" in content or f"Issue: {issue_number}" in content:
                return filepath
        except Exception:
            continue

    return None


def extract_acceptance_criteria(body: str) -> List[Dict[str, str]]:
    """
    Extract acceptance criteria from issue body.

    Args:
        body: Issue body markdown

    Returns:
        List of criteria dictionaries
    """
    criteria = []

    # Look for acceptance criteria section
    pattern = r"(?:acceptance criteria|requirements|tasks)[\s:]*\n((?:[-*]\s+.+\n?)+)"
    match = re.search(pattern, body, re.IGNORECASE)

    if match:
        items_text = match.group(1)
        for line in items_text.split("\n"):
            line = line.strip()
            if line.startswith(("-", "*", "+")):
                text = line.lstrip("-*+ ").strip()
                if text:
                    # Check for checkbox
                    checked = False
                    if text.startswith("[ ]") or text.startswith("[x]"):
                        checked = text.startswith("[x]")
                        text = text[3:].strip()

                    criteria.append({
                        "text": text,
                        "checked": checked
                    })

    return criteria


# =============================================================================
# GITHUB HELPER
# =============================================================================

class GitHubHelper:
    """
    Simplified GitHub API client for agents.

    Provides high-level methods for common operations:
    - Fetching issues
    - Adding comments
    - Updating labels
    - Creating issues

    Uses the GitHub API token from environment.
    """

    BASE_URL = "https://api.github.com"

    def __init__(self, token: str = None, repo: str = None):
        """
        Initialize GitHub helper.

        Args:
            token: GitHub API token (default: from GITHUB_TOKEN env)
            repo: Repository name (default: from GITHUB_REPO env)
        """
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.repo = repo or os.environ.get("GITHUB_REPO")

        if not self.token:
            raise ValueError("GitHub token not provided")
        if not self.repo:
            raise ValueError("GitHub repository not provided")

        # Set up session with retries
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28"
        })

        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504]
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def get_issue(self, issue_number: int) -> Dict[str, Any]:
        """
        Fetch issue details.

        Args:
            issue_number: Issue number

        Returns:
            Issue data dictionary
        """
        url = f"{self.BASE_URL}/repos/{self.repo}/issues/{issue_number}"
        response = self.session.get(url)
        response.raise_for_status()

        data = response.json()

        return {
            "number": data["number"],
            "title": data["title"],
            "body": data.get("body", "") or "",
            "labels": [label["name"] for label in data.get("labels", [])],
            "state": data["state"],
            "created_at": data["created_at"],
            "updated_at": data["updated_at"],
            "assignee": data.get("assignee", {}).get("login") if data.get("assignee") else None,
            "html_url": data.get("html_url", ""),
            "user": data.get("user", {}).get("login", ""),
            "milestone": data.get("milestone", {}).get("title") if data.get("milestone") else None
        }

    def get_issue_comments(self, issue_number: int) -> List[Dict[str, Any]]:
        """
        Fetch comments for an issue.

        Args:
            issue_number: Issue number

        Returns:
            List of comment dictionaries
        """
        url = f"{self.BASE_URL}/repos/{self.repo}/issues/{issue_number}/comments"
        response = self.session.get(url)
        response.raise_for_status()

        comments = []
        for comment in response.json():
            comments.append({
                "id": comment["id"],
                "body": comment["body"],
                "user": comment["user"]["login"],
                "created_at": comment["created_at"]
            })

        return comments

    def add_comment(self, issue_number: int, body: str) -> Dict[str, Any]:
        """
        Add comment to issue.

        Args:
            issue_number: Issue number
            body: Comment body (markdown)

        Returns:
            Created comment data
        """
        url = f"{self.BASE_URL}/repos/{self.repo}/issues/{issue_number}/comments"
        response = self.session.post(url, json={"body": body})
        response.raise_for_status()
        return response.json()

    def update_labels(
        self,
        issue_number: int,
        add: List[str] = None,
        remove: List[str] = None
    ) -> List[str]:
        """
        Update issue labels.

        Args:
            issue_number: Issue number
            add: Labels to add
            remove: Labels to remove

        Returns:
            Current label list after update
        """
        # Get current labels
        issue = self.get_issue(issue_number)
        current_labels = set(issue["labels"])

        # Apply changes
        if remove:
            current_labels -= set(remove)
        if add:
            current_labels |= set(add)

        # Update labels
        url = f"{self.BASE_URL}/repos/{self.repo}/issues/{issue_number}"
        response = self.session.patch(url, json={"labels": list(current_labels)})
        response.raise_for_status()

        return list(current_labels)

    def create_issue(
        self,
        title: str,
        body: str,
        labels: List[str] = None,
        assignee: str = None,
        milestone: int = None
    ) -> int:
        """
        Create new issue.

        Args:
            title: Issue title
            body: Issue body (markdown)
            labels: Label names
            assignee: Assignee username
            milestone: Milestone number

        Returns:
            Created issue number
        """
        url = f"{self.BASE_URL}/repos/{self.repo}/issues"

        payload = {
            "title": title,
            "body": body
        }

        if labels:
            payload["labels"] = labels
        if assignee:
            payload["assignee"] = assignee
        if milestone:
            payload["milestone"] = milestone

        response = self.session.post(url, json=payload)
        response.raise_for_status()

        return response.json()["number"]

    def close_issue(self, issue_number: int) -> bool:
        """
        Close an issue.

        Args:
            issue_number: Issue number

        Returns:
            True if successful
        """
        url = f"{self.BASE_URL}/repos/{self.repo}/issues/{issue_number}"
        response = self.session.patch(url, json={"state": "closed"})
        response.raise_for_status()
        return True

    def get_rate_limit(self) -> Dict[str, int]:
        """
        Get current rate limit status.

        Returns:
            Dictionary with limit, remaining, reset timestamp
        """
        url = f"{self.BASE_URL}/rate_limit"
        response = self.session.get(url)
        response.raise_for_status()

        data = response.json()["resources"]["core"]
        return {
            "limit": data["limit"],
            "remaining": data["remaining"],
            "reset": data["reset"]
        }
