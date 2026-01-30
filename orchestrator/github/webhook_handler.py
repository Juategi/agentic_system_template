# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - WEBHOOK HANDLER
# =============================================================================
"""
GitHub Webhook Handler

Processes incoming webhooks from GitHub to enable real-time
issue processing instead of polling.

Supported Events:
    - issues: Issue opened, closed, labeled, etc.
    - issue_comment: New comments on issues
    - label: Label created, edited, deleted

Security:
    - Validates webhook signature using HMAC-SHA256
    - Rejects requests from unknown sources
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from datetime import datetime, timezone
from typing import (
    Dict,
    Any,
    Optional,
    Callable,
    Awaitable,
    Tuple,
    TYPE_CHECKING,
)

# Third-party imports
try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

if TYPE_CHECKING:
    from orchestrator.github.issue_manager import IssueManager


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# EXCEPTIONS
# =============================================================================


class WebhookError(Exception):
    """Base exception for webhook errors."""
    pass


class WebhookValidationError(WebhookError):
    """Raised when webhook signature validation fails."""
    pass


class WebhookParseError(WebhookError):
    """Raised when webhook payload cannot be parsed."""
    pass


class WebhookConfigError(WebhookError):
    """Raised when webhook configuration is invalid."""
    pass


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

# Event handler function type
EventHandler = Callable[[str, Dict[str, Any]], Awaitable[Dict[str, Any]]]


# =============================================================================
# CONSTANTS
# =============================================================================

# GitHub event header names
HEADER_EVENT = "X-GitHub-Event"
HEADER_SIGNATURE = "X-Hub-Signature-256"
HEADER_DELIVERY = "X-GitHub-Delivery"

# Supported GitHub events
SUPPORTED_EVENTS = {"issues", "issue_comment", "label", "ping"}

# Bot user identifiers (to ignore bot-generated comments)
BOT_IDENTIFIERS = ["[bot]", "github-actions", "dependabot"]


# =============================================================================
# WEBHOOK HANDLER CLASS
# =============================================================================


class WebhookHandler:
    """
    Handles incoming GitHub webhooks.

    This class:
    1. Validates webhook signatures using HMAC-SHA256
    2. Parses event payloads
    3. Triggers appropriate actions in orchestrator

    Attributes:
        secret: Webhook secret for validation
        issue_manager: IssueManager for state changes
        orchestrator: Main orchestrator instance for processing
        event_handlers: Map of event types to handler methods
    """

    def __init__(
        self,
        secret: str,
        issue_manager: "IssueManager",
        orchestrator: Any,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize webhook handler.

        Args:
            secret: Webhook secret for signature validation
            issue_manager: IssueManager instance
            orchestrator: Main orchestrator instance
            config: Optional configuration dict
        """
        self.secret = secret.encode("utf-8") if secret else b""
        self.issue_manager = issue_manager
        self.orchestrator = orchestrator
        self.config = config or {}

        # Statistics
        self.stats = {
            "total_received": 0,
            "total_processed": 0,
            "total_ignored": 0,
            "total_errors": 0,
            "by_event": {},
            "last_received": None,
        }

        # Event handlers mapping
        self.event_handlers: Dict[str, EventHandler] = {
            "issues": self._handle_issues_event,
            "issue_comment": self._handle_issue_comment_event,
            "label": self._handle_label_event,
            "ping": self._handle_ping_event,
        }

        logger.info("WebhookHandler initialized")

    async def handle_webhook(
        self,
        headers: Dict[str, str],
        body: bytes,
    ) -> Dict[str, Any]:
        """
        Process an incoming webhook.

        Args:
            headers: HTTP headers including X-Hub-Signature-256
            body: Raw request body

        Returns:
            {"status": "processed", ...} or {"status": "ignored", "reason": "..."}

        Raises:
            WebhookValidationError: If signature is invalid
            WebhookParseError: If payload cannot be parsed
        """
        self.stats["total_received"] += 1
        self.stats["last_received"] = datetime.now(timezone.utc).isoformat()

        # Validate signature
        if self.secret and not self._validate_signature(headers, body):
            self.stats["total_errors"] += 1
            raise WebhookValidationError("Invalid webhook signature")

        # Parse event
        event_type, action, payload = self._parse_event(headers, body)

        # Track by event type
        event_key = f"{event_type}.{action}" if action else event_type
        self.stats["by_event"][event_key] = self.stats["by_event"].get(event_key, 0) + 1

        # Get delivery ID for logging
        delivery_id = headers.get(HEADER_DELIVERY, "unknown")
        logger.info(f"Webhook received: {event_key} (delivery: {delivery_id})")

        # Check if event is supported
        if event_type not in SUPPORTED_EVENTS:
            self.stats["total_ignored"] += 1
            logger.debug(f"Ignoring unsupported event type: {event_type}")
            return {
                "status": "ignored",
                "reason": f"Unsupported event type: {event_type}",
            }

        # Get handler
        handler = self.event_handlers.get(event_type)
        if not handler:
            self.stats["total_ignored"] += 1
            return {
                "status": "ignored",
                "reason": f"No handler for event type: {event_type}",
            }

        # Execute handler
        try:
            result = await handler(action, payload)
            self.stats["total_processed"] += 1
            return {
                "status": "processed",
                "event": event_type,
                "action": action,
                "result": result,
            }
        except Exception as e:
            self.stats["total_errors"] += 1
            logger.error(f"Error handling webhook: {e}", exc_info=True)
            return {
                "status": "error",
                "event": event_type,
                "action": action,
                "error": str(e),
            }

    def _validate_signature(self, headers: Dict[str, str], body: bytes) -> bool:
        """
        Validate webhook signature.

        Uses HMAC-SHA256 with configured secret.

        Args:
            headers: HTTP headers
            body: Raw request body

        Returns:
            True if signature is valid, False otherwise
        """
        signature_header = headers.get(HEADER_SIGNATURE, "")

        if not signature_header:
            logger.warning("Missing webhook signature header")
            return False

        if not signature_header.startswith("sha256="):
            logger.warning("Invalid signature format (expected sha256=...)")
            return False

        # Extract the signature
        expected_signature = signature_header[7:]  # Remove "sha256=" prefix

        # Compute HMAC
        computed = hmac.new(
            self.secret,
            body,
            hashlib.sha256,
        ).hexdigest()

        # Constant-time comparison to prevent timing attacks
        return hmac.compare_digest(computed, expected_signature)

    def _parse_event(
        self,
        headers: Dict[str, str],
        body: bytes,
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Parse event type and payload.

        Args:
            headers: HTTP headers
            body: Raw request body

        Returns:
            (event_type, action, payload)

        Raises:
            WebhookParseError: If parsing fails
        """
        # Get event type from header
        event_type = headers.get(HEADER_EVENT, "").lower()
        if not event_type:
            raise WebhookParseError("Missing X-GitHub-Event header")

        # Parse JSON body
        try:
            payload = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise WebhookParseError(f"Invalid JSON payload: {e}")

        # Extract action (most events have an action field)
        action = payload.get("action", "")

        return event_type, action, payload

    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================

    async def _handle_issues_event(
        self,
        action: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Handle issues events.

        Actions:
        - opened: New issue created
        - closed: Issue closed
        - reopened: Issue reopened
        - labeled: Label added
        - unlabeled: Label removed
        - edited: Issue edited
        - assigned/unassigned: Assignment changes

        Args:
            action: The specific action (opened, labeled, etc.)
            payload: Full event payload

        Returns:
            Result dict with action taken
        """
        issue = payload.get("issue", {})
        issue_number = issue.get("number")

        if not issue_number:
            return {"action": "ignored", "reason": "No issue number in payload"}

        logger.info(f"Handling issues.{action} for issue #{issue_number}")

        if action == "opened":
            return await self._on_issue_opened(payload)

        elif action == "reopened":
            return await self._on_issue_reopened(payload)

        elif action == "closed":
            return await self._on_issue_closed(payload)

        elif action == "labeled":
            return await self._on_issue_labeled(payload)

        elif action == "unlabeled":
            return await self._on_issue_unlabeled(payload)

        elif action == "edited":
            return await self._on_issue_edited(payload)

        else:
            logger.debug(f"Ignoring issues.{action} event")
            return {"action": "ignored", "reason": f"Unhandled action: {action}"}

    async def _handle_issue_comment_event(
        self,
        action: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Handle issue comment events.

        Actions:
        - created: New comment added
        - edited: Comment edited
        - deleted: Comment deleted

        Used to detect human feedback on issues.

        Args:
            action: The specific action
            payload: Full event payload

        Returns:
            Result dict
        """
        issue = payload.get("issue", {})
        issue_number = issue.get("number")

        if not issue_number:
            return {"action": "ignored", "reason": "No issue number in payload"}

        if action == "created":
            return await self._on_comment_created(payload)

        if action == "edited":
            # Could be used to track feedback updates
            logger.debug(f"Comment edited on issue #{issue_number}")
            return {"action": "noted", "issue": issue_number}

        return {"action": "ignored", "reason": f"Unhandled action: {action}"}

    async def _handle_label_event(
        self,
        action: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Handle label events (repository-level label changes).

        Actions:
        - created: New label created
        - edited: Label edited
        - deleted: Label deleted

        Args:
            action: The specific action
            payload: Full event payload

        Returns:
            Result dict
        """
        label = payload.get("label", {})
        label_name = label.get("name", "")

        logger.debug(f"Label event: {action} - {label_name}")

        # We mostly just note these events, no special handling needed
        return {
            "action": "noted",
            "label": label_name,
            "label_action": action,
        }

    async def _handle_ping_event(
        self,
        action: str,  # noqa: ARG002
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Handle ping event (sent when webhook is first configured).

        Args:
            action: Usually empty for ping (unused, required by handler signature)
            payload: Ping payload with zen message

        Returns:
            Acknowledgement
        """
        del action  # Unused but required by EventHandler signature
        zen = payload.get("zen", "")
        hook_id = payload.get("hook_id", "")

        logger.info(f"Webhook ping received. Hook ID: {hook_id}, Zen: {zen}")

        return {
            "action": "pong",
            "hook_id": hook_id,
            "zen": zen,
        }

    # =========================================================================
    # SPECIFIC EVENT HANDLERS
    # =========================================================================

    async def _on_issue_opened(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle new issue creation.

        If issue has appropriate labels (feature, task, bug), queue for processing.

        Args:
            payload: Full event payload

        Returns:
            Result dict
        """
        issue = payload.get("issue", {})
        issue_number = issue.get("number")
        title = issue.get("title", "")
        labels = [l.get("name", "") for l in issue.get("labels", [])]

        logger.info(f"New issue opened: #{issue_number} - {title}")

        # Check if issue has a type label we care about
        type_labels = {"feature", "task", "bug"}
        has_type_label = bool(type_labels & set(labels))

        if not has_type_label:
            logger.debug(f"Issue #{issue_number} has no type label, ignoring")
            return {
                "action": "ignored",
                "reason": "No type label (feature/task/bug)",
                "issue": issue_number,
            }

        # Add to queue for triage
        if self.orchestrator and hasattr(self.orchestrator, "queue_manager"):
            try:
                await self.orchestrator.queue_manager.enqueue(issue)
                logger.info(f"Issue #{issue_number} queued for processing")
                return {
                    "action": "queued",
                    "issue": issue_number,
                }
            except Exception as e:
                logger.error(f"Failed to queue issue #{issue_number}: {e}")
                return {
                    "action": "error",
                    "issue": issue_number,
                    "error": str(e),
                }

        return {
            "action": "noted",
            "issue": issue_number,
            "reason": "No orchestrator available",
        }

    async def _on_issue_reopened(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle issue reopening.

        Reset issue to READY state for reprocessing.

        Args:
            payload: Full event payload

        Returns:
            Result dict
        """
        issue = payload.get("issue", {})
        issue_number = issue.get("number")

        logger.info(f"Issue #{issue_number} reopened, transitioning to READY")

        try:
            await self.issue_manager.transition_to_ready(issue_number)
            return {
                "action": "transitioned",
                "issue": issue_number,
                "to_state": "READY",
            }
        except Exception as e:
            logger.error(f"Failed to transition issue #{issue_number}: {e}")
            return {
                "action": "error",
                "issue": issue_number,
                "error": str(e),
            }

    async def _on_issue_closed(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle issue closing.

        Remove from queue if present.

        Args:
            payload: Full event payload

        Returns:
            Result dict
        """
        issue = payload.get("issue", {})
        issue_number = issue.get("number")

        logger.info(f"Issue #{issue_number} closed externally")

        # Remove from queue if orchestrator available
        if self.orchestrator and hasattr(self.orchestrator, "queue_manager"):
            try:
                await self.orchestrator.queue_manager.mark_complete(issue_number)
            except Exception:
                pass  # May not be in queue

        return {
            "action": "noted",
            "issue": issue_number,
            "event": "closed",
        }

    async def _on_issue_labeled(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle label addition.

        Triggers workflow transitions based on label changes.
        Special handling for READY and BLOCKED labels.

        Args:
            payload: Full event payload

        Returns:
            Result dict
        """
        issue = payload.get("issue", {})
        label = payload.get("label", {})
        issue_number = issue.get("number")
        label_name = label.get("name", "")

        logger.info(f"Label '{label_name}' added to issue #{issue_number}")

        # Handle special labels
        if label_name == "READY":
            # Issue marked ready for processing
            if self.orchestrator and hasattr(self.orchestrator, "queue_manager"):
                try:
                    await self.orchestrator.queue_manager.enqueue(issue)
                    return {
                        "action": "queued",
                        "issue": issue_number,
                        "label": label_name,
                    }
                except Exception as e:
                    logger.error(f"Failed to queue issue #{issue_number}: {e}")

        elif label_name == "BLOCKED":
            # Issue blocked - remove from queue
            if self.orchestrator and hasattr(self.orchestrator, "queue_manager"):
                try:
                    await self.orchestrator.queue_manager.mark_blocked(
                        issue_number, "Manually blocked via label"
                    )
                except Exception:
                    pass

            return {
                "action": "blocked",
                "issue": issue_number,
            }

        return {
            "action": "labeled",
            "issue": issue_number,
            "label": label_name,
        }

    async def _on_issue_unlabeled(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle label removal.

        Special handling for BLOCKED label removal (may unblock issue).

        Args:
            payload: Full event payload

        Returns:
            Result dict
        """
        issue = payload.get("issue", {})
        label = payload.get("label", {})
        issue_number = issue.get("number")
        label_name = label.get("name", "")

        logger.info(f"Label '{label_name}' removed from issue #{issue_number}")

        # Handle BLOCKED removal - requeue the issue
        if label_name == "BLOCKED":
            current_labels = [l.get("name", "") for l in issue.get("labels", [])]

            # If issue still has READY or other actionable state, requeue
            if "READY" in current_labels:
                if self.orchestrator and hasattr(self.orchestrator, "queue_manager"):
                    try:
                        await self.orchestrator.queue_manager.enqueue(issue)
                        return {
                            "action": "unblocked_and_queued",
                            "issue": issue_number,
                        }
                    except Exception:
                        pass

        return {
            "action": "unlabeled",
            "issue": issue_number,
            "label": label_name,
        }

    async def _on_issue_edited(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle issue edit.

        May need to update cached issue data if requirements changed.

        Args:
            payload: Full event payload

        Returns:
            Result dict
        """
        issue = payload.get("issue", {})
        issue_number = issue.get("number")
        changes = payload.get("changes", {})

        logger.debug(f"Issue #{issue_number} edited. Changes: {list(changes.keys())}")

        return {
            "action": "noted",
            "issue": issue_number,
            "changes": list(changes.keys()),
        }

    async def _on_comment_created(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle new comment.

        If comment is from human (not bot), may provide feedback
        that unblocks a task.

        Args:
            payload: Full event payload

        Returns:
            Result dict
        """
        issue = payload.get("issue", {})
        comment = payload.get("comment", {})
        issue_number = issue.get("number")

        # Get comment author
        user = comment.get("user", {})
        username = user.get("login", "")
        user_type = user.get("type", "")

        # Check if this is a bot comment
        is_bot = (
            user_type == "Bot"
            or any(bot_id in username.lower() for bot_id in BOT_IDENTIFIERS)
        )

        if is_bot:
            logger.debug(f"Ignoring bot comment on issue #{issue_number}")
            return {
                "action": "ignored",
                "reason": "Bot comment",
                "issue": issue_number,
            }

        logger.info(f"Human comment on issue #{issue_number} by {username}")

        # Check if issue is blocked
        current_labels = [l.get("name", "") for l in issue.get("labels", [])]

        if "BLOCKED" in current_labels:
            # Human provided feedback on blocked issue
            comment_body = comment.get("body", "")

            # Check for unblock keywords
            unblock_keywords = ["unblock", "continue", "proceed", "fixed", "resolved"]
            should_unblock = any(
                keyword in comment_body.lower() for keyword in unblock_keywords
            )

            if should_unblock:
                logger.info(f"Unblocking issue #{issue_number} based on human comment")
                try:
                    await self.issue_manager.transition_to_ready(issue_number)
                    return {
                        "action": "unblocked",
                        "issue": issue_number,
                        "by": username,
                    }
                except Exception as e:
                    logger.error(f"Failed to unblock issue #{issue_number}: {e}")

            return {
                "action": "feedback_received",
                "issue": issue_number,
                "status": "blocked",
                "by": username,
            }

        return {
            "action": "comment_noted",
            "issue": issue_number,
            "by": username,
        }

    # =========================================================================
    # STATISTICS AND MONITORING
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        Get webhook statistics.

        Returns:
            Statistics dict
        """
        return self.stats.copy()

    def reset_stats(self) -> None:
        """Reset webhook statistics."""
        self.stats = {
            "total_received": 0,
            "total_processed": 0,
            "total_ignored": 0,
            "total_errors": 0,
            "by_event": {},
            "last_received": None,
        }


# =============================================================================
# WEBHOOK SERVER
# =============================================================================


class WebhookServer:
    """
    HTTP server for receiving webhooks.

    Runs as part of the orchestrator to receive GitHub webhooks.
    Uses aiohttp for async HTTP handling.

    Attributes:
        handler: WebhookHandler instance
        host: Host to bind to
        port: Port to listen on
        path: URL path for webhook endpoint
    """

    def __init__(
        self,
        handler: WebhookHandler,
        host: str = "0.0.0.0",
        port: int = 8080,
        path: str = "/webhooks/github",
    ):
        """
        Initialize webhook server.

        Args:
            handler: WebhookHandler instance
            host: Host to bind to (default: 0.0.0.0)
            port: Port to listen on (default: 8080)
            path: URL path for webhooks (default: /webhooks/github)

        Raises:
            WebhookConfigError: If aiohttp is not available
        """
        if not AIOHTTP_AVAILABLE:
            raise WebhookConfigError(
                "aiohttp is required for webhook server. "
                "Install with: pip install aiohttp"
            )

        self.handler = handler
        self.host = host
        self.port = port
        self.path = path

        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._running = False

        logger.info(f"WebhookServer initialized (will listen on {host}:{port}{path})")

    async def start(self) -> None:
        """
        Start the webhook server.

        Creates aiohttp application and starts listening for connections.
        """
        if self._running:
            logger.warning("Webhook server already running")
            return

        # Create aiohttp application
        self._app = web.Application()

        # Add routes
        self._app.router.add_post(self.path, self._handle_webhook)
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_get("/stats", self._handle_stats)

        # Create runner and site
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()

        self._running = True
        logger.info(f"Webhook server started on http://{self.host}:{self.port}")

    async def stop(self) -> None:
        """Stop the webhook server."""
        if not self._running:
            return

        logger.info("Stopping webhook server...")

        if self._site:
            await self._site.stop()

        if self._runner:
            await self._runner.cleanup()

        self._running = False
        self._app = None
        self._runner = None
        self._site = None

        logger.info("Webhook server stopped")

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running

    # =========================================================================
    # REQUEST HANDLERS
    # =========================================================================

    async def _handle_webhook(self, request: web.Request) -> web.Response:
        """
        Handle incoming webhook HTTP request.

        Args:
            request: aiohttp request object

        Returns:
            JSON response with processing result
        """
        try:
            # Read raw body
            body = await request.read()

            # Convert headers to dict (case-insensitive access)
            headers = {k: v for k, v in request.headers.items()}

            # Process webhook
            result = await self.handler.handle_webhook(headers, body)

            # Return result
            status = 200 if result.get("status") != "error" else 500
            return web.json_response(result, status=status)

        except WebhookValidationError as e:
            logger.warning(f"Webhook validation failed: {e}")
            return web.json_response(
                {"status": "error", "error": "Invalid signature"},
                status=401,
            )

        except WebhookParseError as e:
            logger.warning(f"Webhook parse error: {e}")
            return web.json_response(
                {"status": "error", "error": str(e)},
                status=400,
            )

        except Exception as e:
            logger.error(f"Unexpected error handling webhook: {e}", exc_info=True)
            return web.json_response(
                {"status": "error", "error": "Internal server error"},
                status=500,
            )

    async def _handle_health(self, request: web.Request) -> web.Response:
        """
        Handle health check request.

        Args:
            request: aiohttp request object (unused, required by aiohttp)

        Returns:
            Health status response
        """
        del request  # Unused but required by aiohttp
        return web.json_response({
            "status": "healthy",
            "server": "webhook",
            "running": self._running,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    async def _handle_stats(self, request: web.Request) -> web.Response:
        """
        Handle stats request.

        Args:
            request: aiohttp request object (unused, required by aiohttp)

        Returns:
            Webhook statistics response
        """
        del request  # Unused but required by aiohttp
        stats = self.handler.get_stats()
        return web.json_response({
            "status": "ok",
            "stats": stats,
        })


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_webhook_handler(
    secret: str,
    issue_manager: "IssueManager",
    orchestrator: Any,
    config: Optional[Dict[str, Any]] = None,
) -> WebhookHandler:
    """
    Create a configured webhook handler.

    Args:
        secret: Webhook secret for signature validation
        issue_manager: IssueManager instance
        orchestrator: Main orchestrator instance
        config: Optional configuration

    Returns:
        Configured WebhookHandler instance
    """
    return WebhookHandler(
        secret=secret,
        issue_manager=issue_manager,
        orchestrator=orchestrator,
        config=config,
    )


def create_webhook_server(
    handler: WebhookHandler,
    config: Optional[Dict[str, Any]] = None,
) -> WebhookServer:
    """
    Create a configured webhook server.

    Args:
        handler: WebhookHandler instance
        config: Optional configuration with host, port, path

    Returns:
        Configured WebhookServer instance

    Raises:
        WebhookConfigError: If aiohttp is not available
    """
    config = config or {}

    return WebhookServer(
        handler=handler,
        host=config.get("host", "0.0.0.0"),
        port=config.get("port", 8080),
        path=config.get("path", "/webhooks/github"),
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main classes
    "WebhookHandler",
    "WebhookServer",
    # Factory functions
    "create_webhook_handler",
    "create_webhook_server",
    # Exceptions
    "WebhookError",
    "WebhookValidationError",
    "WebhookParseError",
    "WebhookConfigError",
    # Constants
    "HEADER_EVENT",
    "HEADER_SIGNATURE",
    "HEADER_DELIVERY",
    "SUPPORTED_EVENTS",
]
