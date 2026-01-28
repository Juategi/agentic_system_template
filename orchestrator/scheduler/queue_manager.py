# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - QUEUE MANAGER
# =============================================================================
"""
Queue Manager

Manages the queue of issues waiting to be processed.
Handles prioritization and scheduling of work.

Features:
    - Priority-based ordering
    - FIFO within same priority
    - Dependency awareness
    - Rate limiting
    - Multiple backend support (memory, Redis)
    - Concurrency control
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from typing import (
    Dict,
    List,
    Any,
    Optional,
    Set,
    Tuple,
    Callable,
    TYPE_CHECKING,
)
import json

# Optional Redis support
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

if TYPE_CHECKING:
    from orchestrator.github.client import GitHubClient


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class Priority(IntEnum):
    """Issue priority levels (lower number = higher priority)."""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    DEFAULT = 2  # Default to medium


class QueueStatus(Enum):
    """Status of an item in the queue."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


# Label to priority mapping
PRIORITY_LABELS = {
    "priority:critical": Priority.CRITICAL,
    "priority:high": Priority.HIGH,
    "priority:medium": Priority.MEDIUM,
    "priority:low": Priority.LOW,
    "critical": Priority.CRITICAL,
    "high-priority": Priority.HIGH,
    "urgent": Priority.CRITICAL,
}


# =============================================================================
# EXCEPTIONS
# =============================================================================


class QueueError(Exception):
    """Base exception for queue errors."""
    pass


class QueueFullError(QueueError):
    """Queue has reached maximum capacity."""
    pass


class RateLimitError(QueueError):
    """Rate limit exceeded."""
    pass


class DependencyError(QueueError):
    """Dependency not satisfied."""
    pass


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass(order=True)
class QueueItem:
    """
    An item in the priority queue.

    Uses dataclass ordering for heap operations.
    Priority is first for min-heap behavior (lower = higher priority).
    """
    priority: int
    timestamp: float = field(compare=True)  # For FIFO within same priority
    issue_number: int = field(compare=False)
    issue_data: Dict[str, Any] = field(compare=False, default_factory=dict)
    status: str = field(compare=False, default=QueueStatus.PENDING.value)
    dependencies: List[int] = field(compare=False, default_factory=list)
    retry_count: int = field(compare=False, default=0)
    max_retries: int = field(compare=False, default=3)
    created_at: str = field(compare=False, default="")
    updated_at: str = field(compare=False, default="")

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "priority": self.priority,
            "timestamp": self.timestamp,
            "issue_number": self.issue_number,
            "issue_data": self.issue_data,
            "status": self.status,
            "dependencies": self.dependencies,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueueItem':
        """Create from dictionary."""
        return cls(
            priority=data.get("priority", Priority.DEFAULT),
            timestamp=data.get("timestamp", time.time()),
            issue_number=data["issue_number"],
            issue_data=data.get("issue_data", {}),
            status=data.get("status", QueueStatus.PENDING.value),
            dependencies=data.get("dependencies", []),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    max_concurrent: int = 5  # Maximum concurrent processing
    requests_per_minute: int = 30  # Max issues started per minute
    cooldown_seconds: int = 60  # Cooldown after hitting limit
    burst_limit: int = 10  # Max burst of requests


@dataclass
class QueueStats:
    """Queue statistics."""
    total_items: int = 0
    pending_items: int = 0
    processing_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    blocked_items: int = 0
    avg_wait_time: float = 0.0
    avg_process_time: float = 0.0


# =============================================================================
# QUEUE BACKEND INTERFACE
# =============================================================================


class QueueBackendInterface(ABC):
    """Abstract interface for queue backends."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the backend."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the backend connection."""
        pass

    @abstractmethod
    async def push(self, item: QueueItem) -> bool:
        """Push an item to the queue."""
        pass

    @abstractmethod
    async def pop(self) -> Optional[QueueItem]:
        """Pop the highest priority item from the queue."""
        pass

    @abstractmethod
    async def peek(self) -> Optional[QueueItem]:
        """Peek at the highest priority item without removing it."""
        pass

    @abstractmethod
    async def get(self, issue_number: int) -> Optional[QueueItem]:
        """Get a specific item by issue number."""
        pass

    @abstractmethod
    async def update(self, item: QueueItem) -> bool:
        """Update an item in the queue."""
        pass

    @abstractmethod
    async def remove(self, issue_number: int) -> bool:
        """Remove an item from the queue."""
        pass

    @abstractmethod
    async def list_all(self, status: Optional[str] = None) -> List[QueueItem]:
        """List all items, optionally filtered by status."""
        pass

    @abstractmethod
    async def count(self, status: Optional[str] = None) -> int:
        """Count items, optionally filtered by status."""
        pass

    @abstractmethod
    async def clear(self) -> int:
        """Clear all items from the queue."""
        pass


# =============================================================================
# IN-MEMORY QUEUE BACKEND
# =============================================================================


class MemoryQueueBackend(QueueBackendInterface):
    """
    In-memory priority queue backend.

    Uses a heap for priority ordering and a dict for O(1) lookups.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize memory backend."""
        self.config = config
        self._heap: List[QueueItem] = []
        self._items: Dict[int, QueueItem] = {}  # issue_number -> item
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the backend."""
        logger.info("Memory queue backend initialized")

    async def close(self) -> None:
        """Close the backend."""
        pass

    async def push(self, item: QueueItem) -> bool:
        """Push an item to the queue."""
        async with self._lock:
            # Check if already exists
            if item.issue_number in self._items:
                # Update existing
                return await self._update_internal(item)

            # Add to heap and dict
            heapq.heappush(self._heap, item)
            self._items[item.issue_number] = item
            return True

    async def pop(self) -> Optional[QueueItem]:
        """Pop the highest priority pending item."""
        async with self._lock:
            while self._heap:
                item = heapq.heappop(self._heap)
                # Check if item is still valid and pending
                if (item.issue_number in self._items and
                    self._items[item.issue_number].status == QueueStatus.PENDING.value):
                    return item
            return None

    async def peek(self) -> Optional[QueueItem]:
        """Peek at the highest priority pending item."""
        async with self._lock:
            for item in self._heap:
                if (item.issue_number in self._items and
                    self._items[item.issue_number].status == QueueStatus.PENDING.value):
                    return self._items[item.issue_number]
            return None

    async def get(self, issue_number: int) -> Optional[QueueItem]:
        """Get a specific item."""
        async with self._lock:
            return self._items.get(issue_number)

    async def update(self, item: QueueItem) -> bool:
        """Update an item."""
        async with self._lock:
            return await self._update_internal(item)

    async def _update_internal(self, item: QueueItem) -> bool:
        """Internal update without lock."""
        if item.issue_number not in self._items:
            return False

        item.updated_at = datetime.utcnow().isoformat()
        self._items[item.issue_number] = item

        # Re-heapify if priority changed
        # Note: This is O(n) but updates should be infrequent
        self._heap = [i for i in self._heap if i.issue_number in self._items]
        self._heap = [self._items[i.issue_number] for i in self._heap]
        heapq.heapify(self._heap)

        return True

    async def remove(self, issue_number: int) -> bool:
        """Remove an item."""
        async with self._lock:
            if issue_number not in self._items:
                return False

            del self._items[issue_number]
            # Lazy removal from heap (will be skipped in pop)
            return True

    async def list_all(self, status: Optional[str] = None) -> List[QueueItem]:
        """List all items."""
        async with self._lock:
            items = list(self._items.values())
            if status:
                items = [i for i in items if i.status == status]
            return sorted(items, key=lambda x: (x.priority, x.timestamp))

    async def count(self, status: Optional[str] = None) -> int:
        """Count items."""
        async with self._lock:
            if status:
                return sum(1 for i in self._items.values() if i.status == status)
            return len(self._items)

    async def clear(self) -> int:
        """Clear all items."""
        async with self._lock:
            count = len(self._items)
            self._heap.clear()
            self._items.clear()
            return count


# =============================================================================
# REDIS QUEUE BACKEND
# =============================================================================


class RedisQueueBackend(QueueBackendInterface):
    """
    Redis-based queue backend.

    Uses sorted sets for priority ordering.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize Redis backend."""
        self.config = config
        self._client: Optional[redis.Redis] = None
        self._prefix = config.get("redis_prefix", "queue")

    async def initialize(self) -> None:
        """Initialize Redis connection."""
        if not REDIS_AVAILABLE:
            raise QueueError("Redis package not installed")

        redis_url = self.config.get("redis_url", "redis://localhost:6379/0")
        self._client = redis.from_url(redis_url, decode_responses=True)
        await self._client.ping()
        logger.info("Redis queue backend initialized")

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()

    def _queue_key(self) -> str:
        """Get the sorted set key."""
        return f"{self._prefix}:queue"

    def _item_key(self, issue_number: int) -> str:
        """Get the hash key for an item."""
        return f"{self._prefix}:item:{issue_number}"

    def _score(self, item: QueueItem) -> float:
        """Calculate Redis score from priority and timestamp."""
        # Priority is the integer part, timestamp fraction is decimal
        return item.priority + (item.timestamp / 10**12)

    async def push(self, item: QueueItem) -> bool:
        """Push an item to the queue."""
        if not self._client:
            raise QueueError("Redis not initialized")

        # Store item data
        await self._client.hset(
            self._item_key(item.issue_number),
            mapping={"data": json.dumps(item.to_dict())}
        )

        # Add to sorted set
        await self._client.zadd(
            self._queue_key(),
            {str(item.issue_number): self._score(item)}
        )

        return True

    async def pop(self) -> Optional[QueueItem]:
        """Pop the highest priority pending item."""
        if not self._client:
            raise QueueError("Redis not initialized")

        # Get items in priority order
        members = await self._client.zrange(self._queue_key(), 0, -1)

        for issue_str in members:
            issue_number = int(issue_str)
            item = await self.get(issue_number)

            if item and item.status == QueueStatus.PENDING.value:
                # Remove from sorted set
                await self._client.zrem(self._queue_key(), issue_str)
                return item

        return None

    async def peek(self) -> Optional[QueueItem]:
        """Peek at the highest priority pending item."""
        if not self._client:
            raise QueueError("Redis not initialized")

        members = await self._client.zrange(self._queue_key(), 0, -1)

        for issue_str in members:
            issue_number = int(issue_str)
            item = await self.get(issue_number)

            if item and item.status == QueueStatus.PENDING.value:
                return item

        return None

    async def get(self, issue_number: int) -> Optional[QueueItem]:
        """Get a specific item."""
        if not self._client:
            raise QueueError("Redis not initialized")

        data = await self._client.hget(self._item_key(issue_number), "data")
        if not data:
            return None

        return QueueItem.from_dict(json.loads(data))

    async def update(self, item: QueueItem) -> bool:
        """Update an item."""
        if not self._client:
            raise QueueError("Redis not initialized")

        item.updated_at = datetime.utcnow().isoformat()

        # Update item data
        await self._client.hset(
            self._item_key(item.issue_number),
            mapping={"data": json.dumps(item.to_dict())}
        )

        # Update score in sorted set
        await self._client.zadd(
            self._queue_key(),
            {str(item.issue_number): self._score(item)}
        )

        return True

    async def remove(self, issue_number: int) -> bool:
        """Remove an item."""
        if not self._client:
            raise QueueError("Redis not initialized")

        # Remove from sorted set
        await self._client.zrem(self._queue_key(), str(issue_number))

        # Delete item data
        await self._client.delete(self._item_key(issue_number))

        return True

    async def list_all(self, status: Optional[str] = None) -> List[QueueItem]:
        """List all items."""
        if not self._client:
            raise QueueError("Redis not initialized")

        members = await self._client.zrange(self._queue_key(), 0, -1)
        items = []

        for issue_str in members:
            item = await self.get(int(issue_str))
            if item:
                if status is None or item.status == status:
                    items.append(item)

        return items

    async def count(self, status: Optional[str] = None) -> int:
        """Count items."""
        if status is None and self._client:
            return await self._client.zcard(self._queue_key())
        return len(await self.list_all(status))

    async def clear(self) -> int:
        """Clear all items."""
        if not self._client:
            raise QueueError("Redis not initialized")

        count = await self._client.zcard(self._queue_key())
        members = await self._client.zrange(self._queue_key(), 0, -1)

        # Delete all item keys
        for issue_str in members:
            await self._client.delete(self._item_key(int(issue_str)))

        # Delete the sorted set
        await self._client.delete(self._queue_key())

        return count


# =============================================================================
# RATE LIMITER
# =============================================================================


class RateLimiter:
    """Token bucket rate limiter with sliding window."""

    def __init__(self, config: RateLimitConfig):
        """Initialize rate limiter."""
        self.config = config
        self._tokens: float = config.burst_limit
        self._last_update = time.time()
        self._requests_this_minute: List[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """
        Try to acquire a token.

        Returns True if allowed, False if rate limited.
        """
        async with self._lock:
            now = time.time()

            # Refill tokens
            elapsed = now - self._last_update
            self._tokens = min(
                self.config.burst_limit,
                self._tokens + elapsed * (self.config.requests_per_minute / 60)
            )
            self._last_update = now

            # Clean old requests
            minute_ago = now - 60
            self._requests_this_minute = [
                t for t in self._requests_this_minute if t > minute_ago
            ]

            # Check rate limit
            if len(self._requests_this_minute) >= self.config.requests_per_minute:
                return False

            # Check tokens
            if self._tokens < 1:
                return False

            # Consume token
            self._tokens -= 1
            self._requests_this_minute.append(now)
            return True

    async def wait_for_token(self, timeout: float = 60.0) -> bool:
        """Wait for a token to become available."""
        start = time.time()
        while time.time() - start < timeout:
            if await self.acquire():
                return True
            await asyncio.sleep(0.5)
        return False

    @property
    def available_tokens(self) -> float:
        """Get current available tokens."""
        return self._tokens


# =============================================================================
# QUEUE MANAGER CLASS
# =============================================================================


class QueueManager:
    """
    Manages the issue processing queue.

    Features:
    - Priority-based ordering
    - FIFO within same priority
    - Dependency awareness
    - Rate limiting
    - Concurrency control

    Attributes:
        backend: Queue storage backend
        github_client: GitHub API client
        rate_limiter: Rate limiting controller
        processing: Set of currently processing issues
        completed: Set of completed issue numbers
    """

    def __init__(
        self,
        github_client: 'GitHubClient',
        config: Dict[str, Any],
        backend: Optional[QueueBackendInterface] = None,
    ):
        """
        Initialize queue manager.

        Args:
            github_client: GitHub API client for fetching issues
            config: Queue configuration
            backend: Optional queue backend (defaults to memory)
        """
        self.github_client = github_client
        self.config = config

        # Initialize backend
        if backend:
            self.backend = backend
        else:
            backend_type = config.get("backend", "memory")
            if backend_type == "redis":
                self.backend = RedisQueueBackend(config)
            else:
                self.backend = MemoryQueueBackend(config)

        # Rate limiting
        rate_config = RateLimitConfig(
            max_concurrent=config.get("max_concurrent", 5),
            requests_per_minute=config.get("requests_per_minute", 30),
            cooldown_seconds=config.get("cooldown_seconds", 60),
            burst_limit=config.get("burst_limit", 10),
        )
        self.rate_limiter = RateLimiter(rate_config)

        # Tracking sets
        self._processing: Set[int] = set()
        self._completed: Set[int] = set()
        self._lock = asyncio.Lock()

        # Configuration
        self.max_concurrent = config.get("max_concurrent", 5)
        self.ready_label = config.get("ready_label", "READY")
        self.auto_refresh = config.get("auto_refresh", True)
        self.refresh_interval = config.get("refresh_interval", 300)  # 5 minutes

    async def initialize(self) -> None:
        """Initialize the queue manager."""
        await self.backend.initialize()
        logger.info("Queue manager initialized")

    async def close(self) -> None:
        """Close the queue manager."""
        await self.backend.close()

    # =========================================================================
    # QUEUE OPERATIONS
    # =========================================================================

    async def refresh_queue(self) -> int:
        """
        Refresh queue from GitHub.

        Fetches all READY issues and updates queue.

        Returns:
            Number of new issues added
        """
        logger.info("Refreshing queue from GitHub")
        added = 0

        try:
            # Fetch issues with READY label
            issues = await self.github_client.list_issues(
                labels=[self.ready_label],
                state="open",
            )

            for issue in issues:
                issue_number = issue["number"]

                # Skip if already in queue or processing
                existing = await self.backend.get(issue_number)
                if existing and existing.status in (
                    QueueStatus.PENDING.value,
                    QueueStatus.PROCESSING.value
                ):
                    continue

                # Skip if already completed
                if issue_number in self._completed:
                    continue

                # Add to queue
                item = self._create_queue_item(issue)
                await self.backend.push(item)
                added += 1
                logger.debug(f"Added issue #{issue_number} to queue (priority={item.priority})")

            logger.info(f"Queue refresh complete: {added} new issues added")
            return added

        except Exception as e:
            logger.error(f"Failed to refresh queue: {e}")
            raise QueueError(f"Queue refresh failed: {e}") from e

    async def enqueue(self, issue: Dict[str, Any]) -> QueueItem:
        """
        Add an issue to the queue.

        Args:
            issue: Issue data from GitHub

        Returns:
            Created queue item
        """
        item = self._create_queue_item(issue)
        await self.backend.push(item)
        logger.info(f"Enqueued issue #{item.issue_number} with priority {item.priority}")
        return item

    async def get_next(self) -> Optional[Dict[str, Any]]:
        """
        Get next issue to process.

        Returns highest priority issue that:
        - Is not currently being processed
        - Has no unmet dependencies
        - Is within rate limits
        - Concurrent limit not exceeded

        Returns:
            Issue data dict or None if no issues available
        """
        async with self._lock:
            # Check concurrent limit
            if len(self._processing) >= self.max_concurrent:
                logger.debug(f"Concurrent limit reached ({len(self._processing)}/{self.max_concurrent})")
                return None

            # Check rate limit
            if not await self.rate_limiter.acquire():
                logger.debug("Rate limit exceeded")
                return None

            # Get all pending items
            pending = await self.backend.list_all(status=QueueStatus.PENDING.value)

            for item in pending:
                # Skip if already processing
                if item.issue_number in self._processing:
                    continue

                # Check dependencies
                if not await self._check_dependencies(item):
                    continue

                # Found a valid item
                item.status = QueueStatus.PROCESSING.value
                item.updated_at = datetime.utcnow().isoformat()
                await self.backend.update(item)

                self._processing.add(item.issue_number)
                logger.info(f"Dequeued issue #{item.issue_number} for processing")

                return item.issue_data

            return None

    async def mark_processing(self, issue_number: int) -> bool:
        """
        Mark issue as being processed.

        Args:
            issue_number: Issue number

        Returns:
            True if marked successfully
        """
        async with self._lock:
            item = await self.backend.get(issue_number)
            if not item:
                return False

            item.status = QueueStatus.PROCESSING.value
            await self.backend.update(item)
            self._processing.add(issue_number)

            return True

    async def mark_complete(self, issue_number: int) -> bool:
        """
        Mark issue as completed.

        Args:
            issue_number: Issue number

        Returns:
            True if marked successfully
        """
        async with self._lock:
            item = await self.backend.get(issue_number)
            if not item:
                return False

            item.status = QueueStatus.COMPLETED.value
            await self.backend.update(item)

            self._processing.discard(issue_number)
            self._completed.add(issue_number)

            logger.info(f"Issue #{issue_number} marked complete")
            return True

    async def mark_failed(
        self,
        issue_number: int,
        retry: bool = True,
    ) -> bool:
        """
        Mark issue as failed.

        Args:
            issue_number: Issue number
            retry: Whether to retry the issue

        Returns:
            True if marked successfully
        """
        async with self._lock:
            item = await self.backend.get(issue_number)
            if not item:
                return False

            self._processing.discard(issue_number)

            if retry and item.retry_count < item.max_retries:
                # Requeue with incremented retry count
                item.retry_count += 1
                item.status = QueueStatus.PENDING.value
                # Lower priority slightly on retry
                item.priority = min(item.priority + 1, Priority.LOW)
                await self.backend.update(item)
                logger.info(
                    f"Issue #{issue_number} requeued for retry "
                    f"({item.retry_count}/{item.max_retries})"
                )
            else:
                item.status = QueueStatus.FAILED.value
                await self.backend.update(item)
                logger.warning(f"Issue #{issue_number} marked as failed (max retries reached)")

            return True

    async def mark_blocked(self, issue_number: int, reason: str = "") -> bool:
        """
        Mark issue as blocked.

        Args:
            issue_number: Issue number
            reason: Blocking reason

        Returns:
            True if marked successfully
        """
        async with self._lock:
            item = await self.backend.get(issue_number)
            if not item:
                return False

            item.status = QueueStatus.BLOCKED.value
            item.issue_data["block_reason"] = reason
            await self.backend.update(item)

            self._processing.discard(issue_number)
            logger.warning(f"Issue #{issue_number} marked as blocked: {reason}")

            return True

    # =========================================================================
    # DEPENDENCY MANAGEMENT
    # =========================================================================

    async def add_dependency(
        self,
        issue_number: int,
        depends_on: int,
    ) -> bool:
        """
        Add a dependency between issues.

        Args:
            issue_number: Issue that depends on another
            depends_on: Issue that must complete first

        Returns:
            True if dependency added
        """
        item = await self.backend.get(issue_number)
        if not item:
            return False

        if depends_on not in item.dependencies:
            item.dependencies.append(depends_on)
            await self.backend.update(item)
            logger.debug(f"Added dependency: #{issue_number} depends on #{depends_on}")

        return True

    async def remove_dependency(
        self,
        issue_number: int,
        depends_on: int,
    ) -> bool:
        """
        Remove a dependency.

        Args:
            issue_number: Issue with dependency
            depends_on: Dependency to remove

        Returns:
            True if dependency removed
        """
        item = await self.backend.get(issue_number)
        if not item:
            return False

        if depends_on in item.dependencies:
            item.dependencies.remove(depends_on)
            await self.backend.update(item)

        return True

    async def _check_dependencies(self, item: QueueItem) -> bool:
        """
        Check if all dependencies are satisfied.

        Args:
            item: Queue item to check

        Returns:
            True if all dependencies are complete
        """
        for dep_number in item.dependencies:
            # Check if dependency is completed
            if dep_number not in self._completed:
                dep_item = await self.backend.get(dep_number)
                if not dep_item or dep_item.status != QueueStatus.COMPLETED.value:
                    logger.debug(
                        f"Issue #{item.issue_number} blocked by dependency #{dep_number}"
                    )
                    return False
        return True

    # =========================================================================
    # PRIORITY MANAGEMENT
    # =========================================================================

    def _create_queue_item(self, issue: Dict[str, Any]) -> QueueItem:
        """Create a queue item from issue data."""
        priority = self._calculate_priority(issue)
        dependencies = self._extract_dependencies(issue)

        return QueueItem(
            priority=priority,
            timestamp=time.time(),
            issue_number=issue["number"],
            issue_data=issue,
            status=QueueStatus.PENDING.value,
            dependencies=dependencies,
        )

    def _calculate_priority(self, issue: Dict[str, Any]) -> int:
        """
        Calculate priority score for an issue.

        Factors:
        - Priority label (critical > high > medium > low)
        - Age (older issues get slight boost)
        - Is a blocker (blocks other issues)

        Args:
            issue: Issue data

        Returns:
            Priority score (lower = higher priority)
        """
        # Start with default priority
        priority = Priority.DEFAULT

        # Check labels for priority
        labels = [l.get("name", "").lower() for l in issue.get("labels", [])]

        for label in labels:
            if label in PRIORITY_LABELS:
                priority = min(priority, PRIORITY_LABELS[label])
                break

        # Age bonus (issues older than 7 days get slight priority boost)
        created_at = issue.get("created_at", "")
        if created_at:
            try:
                created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                age_days = (datetime.now(created.tzinfo) - created).days
                if age_days > 7:
                    priority = max(0, priority - 1)  # Boost but don't exceed critical
            except (ValueError, TypeError):
                pass

        return priority

    def _extract_dependencies(self, issue: Dict[str, Any]) -> List[int]:
        """
        Extract dependencies from issue body.

        Looks for patterns like:
        - Depends on #123
        - Blocked by #456
        - Requires #789

        Args:
            issue: Issue data

        Returns:
            List of dependency issue numbers
        """
        import re

        dependencies = []
        body = issue.get("body", "") or ""

        # Patterns to match
        patterns = [
            r"depends\s+on\s+#(\d+)",
            r"blocked\s+by\s+#(\d+)",
            r"requires\s+#(\d+)",
            r"after\s+#(\d+)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, body, re.IGNORECASE)
            for match in matches:
                dep_number = int(match)
                if dep_number not in dependencies:
                    dependencies.append(dep_number)

        return dependencies

    async def update_priority(
        self,
        issue_number: int,
        priority: Priority,
    ) -> bool:
        """
        Update the priority of an issue.

        Args:
            issue_number: Issue number
            priority: New priority

        Returns:
            True if updated successfully
        """
        item = await self.backend.get(issue_number)
        if not item:
            return False

        item.priority = priority
        await self.backend.update(item)
        logger.info(f"Updated priority of issue #{issue_number} to {priority.name}")

        return True

    # =========================================================================
    # STATS AND MONITORING
    # =========================================================================

    async def get_stats(self) -> QueueStats:
        """Get queue statistics."""
        pending = await self.backend.count(QueueStatus.PENDING.value)
        processing = await self.backend.count(QueueStatus.PROCESSING.value)
        completed = await self.backend.count(QueueStatus.COMPLETED.value)
        failed = await self.backend.count(QueueStatus.FAILED.value)
        blocked = await self.backend.count(QueueStatus.BLOCKED.value)

        return QueueStats(
            total_items=pending + processing + completed + failed + blocked,
            pending_items=pending,
            processing_items=processing,
            completed_items=completed,
            failed_items=failed,
            blocked_items=blocked,
        )

    async def list_pending(self) -> List[QueueItem]:
        """List all pending items."""
        return await self.backend.list_all(QueueStatus.PENDING.value)

    async def list_processing(self) -> List[QueueItem]:
        """List all processing items."""
        return await self.backend.list_all(QueueStatus.PROCESSING.value)

    async def list_failed(self) -> List[QueueItem]:
        """List all failed items."""
        return await self.backend.list_all(QueueStatus.FAILED.value)

    async def list_blocked(self) -> List[QueueItem]:
        """List all blocked items."""
        return await self.backend.list_all(QueueStatus.BLOCKED.value)

    @property
    def processing_count(self) -> int:
        """Get number of currently processing issues."""
        return len(self._processing)

    @property
    def processing_issues(self) -> Set[int]:
        """Get set of currently processing issue numbers."""
        return self._processing.copy()


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


async def create_queue_manager(
    github_client: 'GitHubClient',
    config: Dict[str, Any],
) -> QueueManager:
    """
    Create and initialize a queue manager.

    Args:
        github_client: GitHub API client
        config: Queue configuration

    Returns:
        Initialized QueueManager instance
    """
    manager = QueueManager(github_client, config)
    await manager.initialize()
    return manager


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main class
    "QueueManager",
    "create_queue_manager",
    # Backend interface
    "QueueBackendInterface",
    "MemoryQueueBackend",
    "RedisQueueBackend",
    # Data structures
    "QueueItem",
    "QueueStats",
    "RateLimitConfig",
    # Enums
    "Priority",
    "QueueStatus",
    # Rate limiter
    "RateLimiter",
    # Exceptions
    "QueueError",
    "QueueFullError",
    "RateLimitError",
    "DependencyError",
    # Constants
    "PRIORITY_LABELS",
    "REDIS_AVAILABLE",
]