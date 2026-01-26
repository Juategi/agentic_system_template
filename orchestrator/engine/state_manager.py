# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - STATE MANAGER
# =============================================================================
"""
State Manager Module

This module handles persistent state storage and retrieval for the orchestrator.
It enables the system to:
1. Persist workflow state across restarts
2. Recover interrupted workflows
3. Maintain audit trails
4. Support multiple storage backends

Supported Backends:
    - File: JSON file storage (default, simple)
    - Redis: In-memory with persistence (recommended for production)
    - PostgreSQL: Full relational storage (for complex queries)
"""

import os
import json
import asyncio
import logging
import shutil
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import threading
import hashlib


# =============================================================================
# EXCEPTIONS
# =============================================================================

class StateManagerError(Exception):
    """Base exception for state manager errors."""
    pass


class StateNotFoundError(StateManagerError):
    """Raised when state for an issue is not found."""
    pass


class StateConcurrencyError(StateManagerError):
    """Raised when concurrent modification is detected."""
    pass


class StateBackendError(StateManagerError):
    """Raised when backend operation fails."""
    pass


class StateValidationError(StateManagerError):
    """Raised when state validation fails."""
    pass


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class IssueState(Enum):
    """Possible states for an issue in the workflow."""
    TRIAGE = "TRIAGE"
    PLANNING = "PLANNING"
    AWAIT_SUBTASKS = "AWAIT_SUBTASKS"
    DEVELOPMENT = "DEVELOPMENT"
    QA = "QA"
    QA_FAILED = "QA_FAILED"
    REVIEW = "REVIEW"
    DOCUMENTATION = "DOCUMENTATION"
    DONE = "DONE"
    BLOCKED = "BLOCKED"


@dataclass
class TransitionRecord:
    """Record of a state transition for audit trail."""
    timestamp: str
    from_state: str
    to_state: str
    trigger: str
    agent_type: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransitionRecord':
        return cls(**data)


@dataclass
class WorkflowState:
    """
    State object for an issue in the workflow.

    This is the main data structure that flows through the LangGraph workflow
    and is persisted by the state manager.
    """
    issue_number: int
    issue_state: str = IssueState.TRIAGE.value
    issue_type: str = ""
    current_agent: Optional[str] = None
    iteration_count: int = 0
    max_iterations: int = 5
    last_agent_output: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    parent_issue: Optional[int] = None
    child_issues: List[int] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowState':
        data = data.copy()
        if 'child_issues' not in data:
            data['child_issues'] = []
        if 'history' not in data:
            data['history'] = []
        if 'metadata' not in data:
            data['metadata'] = {}
        return cls(**data)

    def add_transition(
        self,
        from_state: str,
        to_state: str,
        trigger: str,
        agent_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Add a transition record to history."""
        record = TransitionRecord(
            timestamp=datetime.utcnow().isoformat(),
            from_state=from_state,
            to_state=to_state,
            trigger=trigger,
            agent_type=agent_type,
            details=details or {}
        )
        self.history.append(record.to_dict())
        self.issue_state = to_state
        self.updated_at = datetime.utcnow().isoformat()
        self.version += 1


# =============================================================================
# STATE MANAGER INTERFACE
# =============================================================================

class StateManagerInterface(ABC):
    """
    Abstract interface for state persistence backends.

    All backends must implement this interface to ensure
    consistent behavior across storage options.
    """

    @abstractmethod
    async def get_state(self, issue_number: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve state for an issue.

        Args:
            issue_number: GitHub issue number

        Returns:
            State dictionary or None if not found
        """
        pass

    @abstractmethod
    async def save_state(self, issue_number: int, state: Dict[str, Any]) -> bool:
        """
        Save state for an issue.

        Args:
            issue_number: GitHub issue number
            state: State dictionary to save

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def delete_state(self, issue_number: int) -> bool:
        """
        Delete state for an issue (when completed).

        Args:
            issue_number: GitHub issue number

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def list_active_issues(self) -> List[int]:
        """
        List all issues with active state.

        Returns:
            List of issue numbers with stored state
        """
        pass

    @abstractmethod
    async def get_issues_by_state(self, workflow_state: str) -> List[int]:
        """
        Find issues in a specific workflow state.

        Args:
            workflow_state: State to filter by (e.g., "DEVELOPMENT")

        Returns:
            List of matching issue numbers
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Check backend health.

        Returns:
            Health status dictionary
        """
        pass


# =============================================================================
# FILE BACKEND
# =============================================================================

class FileStateManager(StateManagerInterface):
    """
    File-based state persistence using JSON.

    Simple backend suitable for:
    - Development environments
    - Single-instance deployments
    - Low-volume production

    State is stored in a JSON file with structure:
    {
        "issues": {
            "123": { ... state ... },
            "124": { ... state ... }
        },
        "metadata": {
            "last_updated": "2024-01-01T00:00:00Z",
            "version": "1.0"
        }
    }
    """

    STATE_VERSION = "1.0"

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize file state manager.

        Args:
            config: Configuration containing:
                - file.path: Path to state file
                - file.backup_enabled: Enable backups
                - file.backup_count: Number of backups to keep
        """
        file_config = config.get("file", config.get("persistence", {}).get("file", {}))

        self.file_path = Path(file_config.get("path", "state/workflow_state.json"))
        self.backup_enabled = file_config.get("backup_enabled", True)
        self.backup_count = file_config.get("backup_count", 5)

        self.logger = logging.getLogger("orchestrator.state.file")
        self._lock = threading.RLock()
        self._cache: Optional[Dict[str, Any]] = None
        self._cache_hash: Optional[str] = None

        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Create state file if it doesn't exist."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.file_path.exists():
            initial_state = {
                "issues": {},
                "metadata": {
                    "version": self.STATE_VERSION,
                    "created_at": datetime.utcnow().isoformat(),
                    "last_updated": datetime.utcnow().isoformat()
                }
            }
            self._write_file(initial_state)
            self.logger.info(f"Created state file: {self.file_path}")

    def _read_file(self) -> Dict[str, Any]:
        """Read state file with caching."""
        with self._lock:
            try:
                content = self.file_path.read_text(encoding="utf-8")
                content_hash = hashlib.md5(content.encode()).hexdigest()

                if self._cache is not None and self._cache_hash == content_hash:
                    return self._cache

                data = json.loads(content)
                self._cache = data
                self._cache_hash = content_hash
                return data

            except json.JSONDecodeError as e:
                self.logger.error(f"Invalid JSON in state file: {e}")
                raise StateBackendError(f"Invalid state file: {e}")
            except Exception as e:
                self.logger.error(f"Error reading state file: {e}")
                raise StateBackendError(f"Failed to read state: {e}")

    def _write_file(self, data: Dict[str, Any]):
        """Write state file atomically with backup."""
        with self._lock:
            if self.backup_enabled and self.file_path.exists():
                self._create_backup()

            data["metadata"]["last_updated"] = datetime.utcnow().isoformat()

            temp_path = self.file_path.with_suffix(".tmp")
            try:
                temp_path.write_text(
                    json.dumps(data, indent=2, default=str),
                    encoding="utf-8"
                )
                temp_path.replace(self.file_path)

                content = json.dumps(data, indent=2, default=str)
                self._cache = data
                self._cache_hash = hashlib.md5(content.encode()).hexdigest()

            except Exception as e:
                if temp_path.exists():
                    temp_path.unlink()
                raise StateBackendError(f"Failed to write state: {e}")

    def _create_backup(self):
        """Create a backup of the state file."""
        if not self.file_path.exists():
            return

        backup_dir = self.file_path.parent / "backups"
        backup_dir.mkdir(exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"state_{timestamp}.json"

        try:
            shutil.copy2(self.file_path, backup_path)
            self.logger.debug(f"Created backup: {backup_path}")

            backups = sorted(backup_dir.glob("state_*.json"))
            while len(backups) > self.backup_count:
                oldest = backups.pop(0)
                oldest.unlink()
                self.logger.debug(f"Removed old backup: {oldest}")

        except Exception as e:
            self.logger.warning(f"Failed to create backup: {e}")

    async def get_state(self, issue_number: int) -> Optional[Dict[str, Any]]:
        """Retrieve state from JSON file."""
        try:
            data = self._read_file()
            issue_key = str(issue_number)

            if issue_key in data.get("issues", {}):
                return data["issues"][issue_key]
            return None

        except StateBackendError:
            raise
        except Exception as e:
            self.logger.error(f"Error getting state for issue {issue_number}: {e}")
            return None

    async def save_state(self, issue_number: int, state: Dict[str, Any]) -> bool:
        """Save state to JSON file."""
        try:
            data = self._read_file()
            issue_key = str(issue_number)

            state["updated_at"] = datetime.utcnow().isoformat()

            if issue_key in data.get("issues", {}):
                existing_version = data["issues"][issue_key].get("version", 0)
                new_version = state.get("version", 1)
                if new_version <= existing_version:
                    state["version"] = existing_version + 1

            data["issues"][issue_key] = state
            self._write_file(data)

            self.logger.debug(f"Saved state for issue {issue_number}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving state for issue {issue_number}: {e}")
            return False

    async def delete_state(self, issue_number: int) -> bool:
        """Delete state for an issue."""
        try:
            data = self._read_file()
            issue_key = str(issue_number)

            if issue_key in data.get("issues", {}):
                del data["issues"][issue_key]
                self._write_file(data)
                self.logger.info(f"Deleted state for issue {issue_number}")
                return True
            return False

        except Exception as e:
            self.logger.error(f"Error deleting state for issue {issue_number}: {e}")
            return False

    async def list_active_issues(self) -> List[int]:
        """List all issues with active state."""
        try:
            data = self._read_file()
            issues = []

            for issue_key, state in data.get("issues", {}).items():
                if state.get("issue_state") not in [IssueState.DONE.value, IssueState.BLOCKED.value]:
                    issues.append(int(issue_key))

            return sorted(issues)

        except Exception as e:
            self.logger.error(f"Error listing active issues: {e}")
            return []

    async def get_issues_by_state(self, workflow_state: str) -> List[int]:
        """Find issues in a specific workflow state."""
        try:
            data = self._read_file()
            issues = []

            for issue_key, state in data.get("issues", {}).items():
                if state.get("issue_state") == workflow_state:
                    issues.append(int(issue_key))

            return sorted(issues)

        except Exception as e:
            self.logger.error(f"Error getting issues by state {workflow_state}: {e}")
            return []

    async def health_check(self) -> Dict[str, Any]:
        """Check file backend health."""
        try:
            data = self._read_file()
            issue_count = len(data.get("issues", {}))

            return {
                "healthy": True,
                "backend": "file",
                "file_path": str(self.file_path),
                "issue_count": issue_count,
                "last_updated": data.get("metadata", {}).get("last_updated"),
                "version": data.get("metadata", {}).get("version")
            }

        except Exception as e:
            return {
                "healthy": False,
                "backend": "file",
                "error": str(e)
            }


# =============================================================================
# REDIS BACKEND
# =============================================================================

class RedisStateManager(StateManagerInterface):
    """
    Redis-based state persistence.

    Recommended for production:
    - Fast read/write operations
    - Built-in expiration
    - Atomic operations
    - Supports clustering

    Key structure:
        orchestrator:{project_id}:state:{issue_number} -> JSON state
        orchestrator:{project_id}:index:state:{workflow_state} -> Set of issues
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Redis state manager.

        Args:
            config: Configuration containing:
                - redis.url: Redis connection URL
                - redis.key_prefix: Key prefix
                - redis.ttl_seconds: State TTL (0 = no expiry)
        """
        redis_config = config.get("redis", config.get("persistence", {}).get("redis", {}))

        self.redis_url = redis_config.get("url", os.environ.get("REDIS_URL", "redis://localhost:6379"))
        self.key_prefix = redis_config.get("key_prefix", "orchestrator")
        self.ttl_seconds = redis_config.get("ttl_seconds", 0)
        self.project_id = config.get("project_id", os.environ.get("PROJECT_ID", "default"))

        self.logger = logging.getLogger("orchestrator.state.redis")
        self._client = None
        self._initialized = False

    async def _get_client(self):
        """Get or create Redis client."""
        if self._client is None:
            try:
                import redis.asyncio as aioredis
                self._client = aioredis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                self._initialized = True
                self.logger.info("Redis connection established")
            except ImportError:
                raise StateBackendError("redis package not installed. Install with: pip install redis")
            except Exception as e:
                raise StateBackendError(f"Failed to connect to Redis: {e}")
        return self._client

    def _state_key(self, issue_number: int) -> str:
        """Build key for state storage."""
        return f"{self.key_prefix}:{self.project_id}:state:{issue_number}"

    def _index_key(self, workflow_state: str) -> str:
        """Build key for state index."""
        return f"{self.key_prefix}:{self.project_id}:index:state:{workflow_state}"

    def _all_issues_key(self) -> str:
        """Build key for all issues set."""
        return f"{self.key_prefix}:{self.project_id}:all_issues"

    async def get_state(self, issue_number: int) -> Optional[Dict[str, Any]]:
        """Retrieve state from Redis."""
        try:
            client = await self._get_client()
            data = await client.get(self._state_key(issue_number))

            if data:
                return json.loads(data)
            return None

        except Exception as e:
            self.logger.error(f"Error getting state for issue {issue_number}: {e}")
            raise StateBackendError(f"Redis get failed: {e}")

    async def save_state(self, issue_number: int, state: Dict[str, Any]) -> bool:
        """Save state to Redis with transaction."""
        try:
            client = await self._get_client()
            state_key = self._state_key(issue_number)

            old_state = await self.get_state(issue_number)
            old_workflow_state = old_state.get("issue_state") if old_state else None
            new_workflow_state = state.get("issue_state")

            state["updated_at"] = datetime.utcnow().isoformat()
            state_json = json.dumps(state, default=str)

            async with client.pipeline(transaction=True) as pipe:
                if self.ttl_seconds > 0:
                    pipe.setex(state_key, self.ttl_seconds, state_json)
                else:
                    pipe.set(state_key, state_json)

                pipe.sadd(self._all_issues_key(), issue_number)

                if old_workflow_state and old_workflow_state != new_workflow_state:
                    pipe.srem(self._index_key(old_workflow_state), issue_number)

                if new_workflow_state:
                    pipe.sadd(self._index_key(new_workflow_state), issue_number)

                await pipe.execute()

            self.logger.debug(f"Saved state for issue {issue_number}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving state for issue {issue_number}: {e}")
            return False

    async def delete_state(self, issue_number: int) -> bool:
        """Delete state from Redis."""
        try:
            client = await self._get_client()

            old_state = await self.get_state(issue_number)
            old_workflow_state = old_state.get("issue_state") if old_state else None

            async with client.pipeline(transaction=True) as pipe:
                pipe.delete(self._state_key(issue_number))
                pipe.srem(self._all_issues_key(), issue_number)

                if old_workflow_state:
                    pipe.srem(self._index_key(old_workflow_state), issue_number)

                await pipe.execute()

            self.logger.info(f"Deleted state for issue {issue_number}")
            return True

        except Exception as e:
            self.logger.error(f"Error deleting state for issue {issue_number}: {e}")
            return False

    async def list_active_issues(self) -> List[int]:
        """List all issues with active state."""
        try:
            client = await self._get_client()

            terminal_states = [IssueState.DONE.value, IssueState.BLOCKED.value]
            terminal_keys = [self._index_key(s) for s in terminal_states]

            all_issues = await client.smembers(self._all_issues_key())

            terminal_issues = set()
            for key in terminal_keys:
                members = await client.smembers(key)
                terminal_issues.update(members)

            active = [int(i) for i in all_issues if i not in terminal_issues]
            return sorted(active)

        except Exception as e:
            self.logger.error(f"Error listing active issues: {e}")
            return []

    async def get_issues_by_state(self, workflow_state: str) -> List[int]:
        """Get issues in a workflow state using index."""
        try:
            client = await self._get_client()
            members = await client.smembers(self._index_key(workflow_state))
            return sorted([int(i) for i in members])

        except Exception as e:
            self.logger.error(f"Error getting issues by state {workflow_state}: {e}")
            return []

    async def health_check(self) -> Dict[str, Any]:
        """Check Redis backend health."""
        try:
            client = await self._get_client()
            await client.ping()

            all_issues = await client.smembers(self._all_issues_key())

            return {
                "healthy": True,
                "backend": "redis",
                "url": self.redis_url.split("@")[-1] if "@" in self.redis_url else self.redis_url,
                "issue_count": len(all_issues),
                "project_id": self.project_id
            }

        except Exception as e:
            return {
                "healthy": False,
                "backend": "redis",
                "error": str(e)
            }

    async def close(self):
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None


# =============================================================================
# POSTGRESQL BACKEND
# =============================================================================

class PostgreSQLStateManager(StateManagerInterface):
    """
    PostgreSQL-based state persistence.

    Best for:
    - Complex queries and reporting
    - Long-term state storage
    - Audit trail requirements
    - Integration with existing databases

    Schema:
        CREATE TABLE workflow_state (
            issue_number INTEGER PRIMARY KEY,
            project_id VARCHAR(255) NOT NULL,
            workflow_state VARCHAR(50) NOT NULL,
            state_json JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );

        CREATE INDEX idx_workflow_state ON workflow_state(workflow_state);
        CREATE INDEX idx_project_id ON workflow_state(project_id);
    """

    SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS workflow_state (
        issue_number INTEGER NOT NULL,
        project_id VARCHAR(255) NOT NULL,
        workflow_state VARCHAR(50) NOT NULL,
        state_json JSONB NOT NULL,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW(),
        PRIMARY KEY (project_id, issue_number)
    );

    CREATE INDEX IF NOT EXISTS idx_workflow_state ON workflow_state(workflow_state);
    CREATE INDEX IF NOT EXISTS idx_project_id ON workflow_state(project_id);
    CREATE INDEX IF NOT EXISTS idx_updated_at ON workflow_state(updated_at);
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PostgreSQL state manager.

        Args:
            config: Configuration containing:
                - database.url: Connection string
                - database.pool_size: Connection pool size
        """
        db_config = config.get("database", config.get("persistence", {}).get("database", {}))

        self.database_url = db_config.get(
            "url",
            os.environ.get("DATABASE_URL", "postgresql://localhost/orchestrator")
        )
        self.pool_size = db_config.get("pool_size", 5)
        self.project_id = config.get("project_id", os.environ.get("PROJECT_ID", "default"))

        self.logger = logging.getLogger("orchestrator.state.postgresql")
        self._pool = None
        self._initialized = False

    async def _get_pool(self):
        """Get or create connection pool."""
        if self._pool is None:
            try:
                import asyncpg
                self._pool = await asyncpg.create_pool(
                    self.database_url,
                    min_size=1,
                    max_size=self.pool_size
                )

                async with self._pool.acquire() as conn:
                    await conn.execute(self.SCHEMA_SQL)

                self._initialized = True
                self.logger.info("PostgreSQL connection pool created")

            except ImportError:
                raise StateBackendError("asyncpg package not installed. Install with: pip install asyncpg")
            except Exception as e:
                raise StateBackendError(f"Failed to connect to PostgreSQL: {e}")

        return self._pool

    async def get_state(self, issue_number: int) -> Optional[Dict[str, Any]]:
        """Retrieve state from PostgreSQL."""
        try:
            pool = await self._get_pool()

            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT state_json FROM workflow_state
                    WHERE project_id = $1 AND issue_number = $2
                    """,
                    self.project_id,
                    issue_number
                )

                if row:
                    return json.loads(row["state_json"])
                return None

        except Exception as e:
            self.logger.error(f"Error getting state for issue {issue_number}: {e}")
            raise StateBackendError(f"PostgreSQL query failed: {e}")

    async def save_state(self, issue_number: int, state: Dict[str, Any]) -> bool:
        """Save state with upsert."""
        try:
            pool = await self._get_pool()

            state["updated_at"] = datetime.utcnow().isoformat()
            workflow_state = state.get("issue_state", "UNKNOWN")
            state_json = json.dumps(state, default=str)

            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO workflow_state (issue_number, project_id, workflow_state, state_json)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (project_id, issue_number)
                    DO UPDATE SET
                        workflow_state = EXCLUDED.workflow_state,
                        state_json = EXCLUDED.state_json,
                        updated_at = NOW()
                    """,
                    issue_number,
                    self.project_id,
                    workflow_state,
                    state_json
                )

            self.logger.debug(f"Saved state for issue {issue_number}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving state for issue {issue_number}: {e}")
            return False

    async def delete_state(self, issue_number: int) -> bool:
        """Delete state from PostgreSQL."""
        try:
            pool = await self._get_pool()

            async with pool.acquire() as conn:
                result = await conn.execute(
                    """
                    DELETE FROM workflow_state
                    WHERE project_id = $1 AND issue_number = $2
                    """,
                    self.project_id,
                    issue_number
                )

            deleted = result.split()[-1] != "0"
            if deleted:
                self.logger.info(f"Deleted state for issue {issue_number}")
            return deleted

        except Exception as e:
            self.logger.error(f"Error deleting state for issue {issue_number}: {e}")
            return False

    async def list_active_issues(self) -> List[int]:
        """List all issues with active state."""
        try:
            pool = await self._get_pool()
            terminal_states = [IssueState.DONE.value, IssueState.BLOCKED.value]

            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT issue_number FROM workflow_state
                    WHERE project_id = $1 AND workflow_state != ALL($2)
                    ORDER BY issue_number
                    """,
                    self.project_id,
                    terminal_states
                )

            return [row["issue_number"] for row in rows]

        except Exception as e:
            self.logger.error(f"Error listing active issues: {e}")
            return []

    async def get_issues_by_state(self, workflow_state: str) -> List[int]:
        """Find issues in a specific workflow state."""
        try:
            pool = await self._get_pool()

            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT issue_number FROM workflow_state
                    WHERE project_id = $1 AND workflow_state = $2
                    ORDER BY issue_number
                    """,
                    self.project_id,
                    workflow_state
                )

            return [row["issue_number"] for row in rows]

        except Exception as e:
            self.logger.error(f"Error getting issues by state {workflow_state}: {e}")
            return []

    async def health_check(self) -> Dict[str, Any]:
        """Check PostgreSQL backend health."""
        try:
            pool = await self._get_pool()

            async with pool.acquire() as conn:
                count = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM workflow_state
                    WHERE project_id = $1
                    """,
                    self.project_id
                )

            return {
                "healthy": True,
                "backend": "postgresql",
                "issue_count": count,
                "project_id": self.project_id,
                "pool_size": self.pool_size
            }

        except Exception as e:
            return {
                "healthy": False,
                "backend": "postgresql",
                "error": str(e)
            }

    async def close(self):
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None


# =============================================================================
# STATE MANAGER FACTORY
# =============================================================================

class StateManager:
    """
    Factory and wrapper class for state management.

    Usage:
        state_manager = await StateManager.create(config)
        state = await state_manager.get_state(123)
    """

    def __init__(self, backend: StateManagerInterface):
        """Initialize with a backend."""
        self._backend = backend
        self.logger = logging.getLogger("orchestrator.state")

    @classmethod
    async def create(cls, config: Dict[str, Any]) -> 'StateManager':
        """
        Create state manager based on configuration.

        Args:
            config: Configuration with persistence settings

        Returns:
            StateManager instance with appropriate backend
        """
        backend_type = config.get("persistence", {}).get("backend", "file")

        if backend_type == "file":
            backend = FileStateManager(config)
        elif backend_type == "redis":
            backend = RedisStateManager(config)
        elif backend_type == "postgresql":
            backend = PostgreSQLStateManager(config)
        else:
            raise ValueError(f"Unknown state backend: {backend_type}")

        return cls(backend)

    async def get_state(self, issue_number: int) -> Optional[WorkflowState]:
        """Get workflow state for an issue."""
        data = await self._backend.get_state(issue_number)
        if data:
            return WorkflowState.from_dict(data)
        return None

    async def save_state(self, issue_number: int, state: WorkflowState) -> bool:
        """Save workflow state for an issue."""
        return await self._backend.save_state(issue_number, state.to_dict())

    async def delete_state(self, issue_number: int) -> bool:
        """Delete state for an issue."""
        return await self._backend.delete_state(issue_number)

    async def list_active_issues(self) -> List[int]:
        """List all active issues."""
        return await self._backend.list_active_issues()

    async def get_issues_by_state(self, workflow_state: str) -> List[int]:
        """Get issues in a specific state."""
        return await self._backend.get_issues_by_state(workflow_state)

    async def health_check(self) -> Dict[str, Any]:
        """Check backend health."""
        return await self._backend.health_check()

    async def create_initial_state(self, issue_number: int, issue_type: str = "") -> WorkflowState:
        """
        Create initial state for a new issue.

        Args:
            issue_number: GitHub issue number
            issue_type: Type of issue (feature, task, bug)

        Returns:
            Created WorkflowState
        """
        state = WorkflowState(
            issue_number=issue_number,
            issue_state=IssueState.TRIAGE.value,
            issue_type=issue_type
        )

        await self.save_state(issue_number, state)
        self.logger.info(f"Created initial state for issue {issue_number}")

        return state

    async def transition_state(
        self,
        issue_number: int,
        to_state: str,
        trigger: str,
        agent_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        expected_from_state: Optional[str] = None
    ) -> WorkflowState:
        """
        Transition issue to a new state with history recording.

        Args:
            issue_number: GitHub issue number
            to_state: Target state
            trigger: What triggered the transition
            agent_type: Agent that triggered (if any)
            details: Additional transition details
            expected_from_state: Expected current state (for validation)

        Returns:
            Updated WorkflowState

        Raises:
            StateNotFoundError: If issue has no state
            StateValidationError: If current state doesn't match expected
        """
        state = await self.get_state(issue_number)

        if state is None:
            raise StateNotFoundError(f"No state found for issue {issue_number}")

        from_state = state.issue_state

        if expected_from_state and from_state != expected_from_state:
            raise StateValidationError(
                f"Expected state {expected_from_state} but found {from_state}"
            )

        state.add_transition(
            from_state=from_state,
            to_state=to_state,
            trigger=trigger,
            agent_type=agent_type,
            details=details
        )

        await self.save_state(issue_number, state)

        self.logger.info(
            f"Issue {issue_number}: {from_state} -> {to_state} (trigger: {trigger})"
        )

        return state

    async def increment_iteration(self, issue_number: int) -> WorkflowState:
        """
        Increment iteration count for an issue.

        Args:
            issue_number: GitHub issue number

        Returns:
            Updated WorkflowState
        """
        state = await self.get_state(issue_number)

        if state is None:
            raise StateNotFoundError(f"No state found for issue {issue_number}")

        state.iteration_count += 1
        state.updated_at = datetime.utcnow().isoformat()

        await self.save_state(issue_number, state)

        self.logger.debug(
            f"Issue {issue_number}: iteration count = {state.iteration_count}"
        )

        return state

    async def set_agent_output(
        self,
        issue_number: int,
        agent_type: str,
        output: Dict[str, Any]
    ) -> WorkflowState:
        """
        Store agent output in state.

        Args:
            issue_number: GitHub issue number
            agent_type: Type of agent
            output: Agent output to store

        Returns:
            Updated WorkflowState
        """
        state = await self.get_state(issue_number)

        if state is None:
            raise StateNotFoundError(f"No state found for issue {issue_number}")

        state.last_agent_output = output
        state.current_agent = None
        state.metadata[f"last_{agent_type}_output"] = output
        state.updated_at = datetime.utcnow().isoformat()

        await self.save_state(issue_number, state)

        return state

    async def set_error(self, issue_number: int, error_message: str) -> WorkflowState:
        """
        Set error message on state.

        Args:
            issue_number: GitHub issue number
            error_message: Error message to store

        Returns:
            Updated WorkflowState
        """
        state = await self.get_state(issue_number)

        if state is None:
            raise StateNotFoundError(f"No state found for issue {issue_number}")

        state.error_message = error_message
        state.updated_at = datetime.utcnow().isoformat()

        await self.save_state(issue_number, state)

        self.logger.error(f"Issue {issue_number}: error set - {error_message}")

        return state

    async def close(self):
        """Close backend connections."""
        if hasattr(self._backend, 'close'):
            await self._backend.close()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

async def recover_orphaned_states(
    state_manager: StateManager,
    agent_launcher=None
) -> List[int]:
    """
    Find and handle orphaned states after restart.

    Orphaned states occur when:
    - Orchestrator crashed during processing
    - Agent container died unexpectedly
    - Network issues caused incomplete updates

    Args:
        state_manager: StateManager instance
        agent_launcher: Optional AgentLauncher to check running containers

    Returns:
        List of recovered issue numbers
    """
    logger = logging.getLogger("orchestrator.state.recovery")
    recovered = []

    in_progress = await state_manager.get_issues_by_state(IssueState.DEVELOPMENT.value)
    in_progress.extend(await state_manager.get_issues_by_state(IssueState.QA.value))
    in_progress.extend(await state_manager.get_issues_by_state(IssueState.REVIEW.value))
    in_progress.extend(await state_manager.get_issues_by_state(IssueState.DOCUMENTATION.value))

    for issue_number in in_progress:
        state = await state_manager.get_state(issue_number)
        if not state:
            continue

        is_running = False
        if agent_launcher and state.current_agent:
            container_id = agent_launcher.container_pool.get_by_issue(issue_number)
            if container_id:
                try:
                    status = await agent_launcher.get_agent_status(container_id)
                    is_running = status.get("running", False)
                except Exception:
                    is_running = False

        if not is_running:
            previous_state = IssueState.DEVELOPMENT.value
            if state.history:
                for record in reversed(state.history):
                    if record.get("to_state") == state.issue_state:
                        previous_state = record.get("from_state", IssueState.DEVELOPMENT.value)
                        break

            await state_manager.transition_state(
                issue_number=issue_number,
                to_state=previous_state,
                trigger="recovery",
                details={"reason": "orphaned_state_recovery"}
            )

            recovered.append(issue_number)
            logger.warning(
                f"Recovered orphaned issue {issue_number}: "
                f"{state.issue_state} -> {previous_state}"
            )

    return recovered


async def cleanup_completed_states(
    state_manager: StateManager,
    retention_days: int = 7
) -> int:
    """
    Remove state for completed issues after retention period.

    Args:
        state_manager: StateManager instance
        retention_days: Days to keep completed state

    Returns:
        Number of states cleaned up
    """
    logger = logging.getLogger("orchestrator.state.cleanup")
    cleaned = 0

    cutoff = datetime.utcnow() - timedelta(days=retention_days)

    for terminal_state in [IssueState.DONE.value, IssueState.BLOCKED.value]:
        issues = await state_manager.get_issues_by_state(terminal_state)

        for issue_number in issues:
            state = await state_manager.get_state(issue_number)
            if not state:
                continue

            try:
                updated_at = datetime.fromisoformat(state.updated_at.replace("Z", "+00:00"))
                if updated_at.replace(tzinfo=None) < cutoff:
                    await state_manager.delete_state(issue_number)
                    cleaned += 1
                    logger.info(f"Cleaned up state for issue {issue_number}")
            except (ValueError, AttributeError):
                continue

    return cleaned


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main classes
    "StateManager",
    "StateManagerInterface",
    "FileStateManager",
    "RedisStateManager",
    "PostgreSQLStateManager",
    # Data structures
    "WorkflowState",
    "TransitionRecord",
    "IssueState",
    # Exceptions
    "StateManagerError",
    "StateNotFoundError",
    "StateConcurrencyError",
    "StateBackendError",
    "StateValidationError",
    # Utilities
    "recover_orphaned_states",
    "cleanup_completed_states",
]