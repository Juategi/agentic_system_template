# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - STRUCTURED LOGGING
# =============================================================================
"""
Structured Logging Module

Provides consistent, structured logging across all components.
Uses structlog for JSON-formatted log output with fallback to stdlib logging.

Features:
    - JSON-formatted logs for easy parsing
    - Contextual information in all logs
    - Log levels with filtering
    - Sensitive data masking
    - File output with rotation
    - Audit trail logger
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Try to import structlog; fall back to stdlib if unavailable
try:
    import structlog
    from structlog.contextvars import bind_contextvars, unbind_contextvars, merge_contextvars
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False


# =============================================================================
# SENSITIVE DATA MASKING
# =============================================================================

# Keys whose values should be masked
SENSITIVE_KEYS = frozenset([
    "token", "api_key", "password", "secret", "credential",
    "private_key", "access_token", "refresh_token", "authorization",
    "github_token", "anthropic_api_key", "openai_api_key",
])


def _is_sensitive(key: str) -> bool:
    """Check if a key name indicates sensitive data."""
    key_lower = key.lower().replace("-", "_")
    return any(s in key_lower for s in SENSITIVE_KEYS)


def _mask_value(value: Any) -> str:
    """Mask a sensitive value, keeping first/last 4 chars if long enough."""
    if not isinstance(value, str):
        return "****"
    if len(value) > 8:
        return value[:4] + "****" + value[-4:]
    return "****"


def mask_sensitive_data(
    logger: Any, method_name: str, event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Structlog processor that masks sensitive data in log events.

    Recursively processes dictionaries to mask values whose keys
    match known sensitive patterns.
    """

    def _process(d: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for key, value in d.items():
            if _is_sensitive(key):
                result[key] = _mask_value(value)
            elif isinstance(value, dict):
                result[key] = _process(value)
            else:
                result[key] = value
        return result

    return _process(event_dict)


def mask_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Public helper to mask sensitive data in an arbitrary dict.

    Useful outside the structlog pipeline (e.g. audit events).
    """
    return mask_sensitive_data(None, "", data)


# =============================================================================
# STDLIB JSON FORMATTER (fallback when structlog is unavailable)
# =============================================================================


class JSONFormatter(logging.Formatter):
    """JSON log formatter for stdlib logging."""

    def __init__(self, mask_sensitive: bool = True):
        super().__init__()
        self.mask_sensitive = mask_sensitive

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }

        # Include extra fields added via `extra=` or LoggerAdapter
        for key in ("issue_number", "agent_type", "component", "project_id"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val

        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        if self.mask_sensitive:
            log_entry = mask_dict(log_entry)

        return json.dumps(log_entry, default=str)


# =============================================================================
# LOGGING SETUP
# =============================================================================


def setup_logging(
    level: str = "INFO",
    fmt: str = "json",
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
    mask_sensitive: bool = True,
    max_bytes: int = 100 * 1024 * 1024,  # 100 MB
    backup_count: int = 10,
) -> logging.Logger:
    """
    Configure structured logging for the application.

    Supports two back-ends:
      * **structlog** (preferred) – produces rich, contextual JSON logs.
      * **stdlib logging** – used when structlog is not installed.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        fmt: Output format – ``"json"`` or ``"text"``.
        log_file: Explicit log file path.  Overrides *log_dir*.
        log_dir: Directory for log files.  When set (and *log_file* is
            ``None``), logs are written to ``<log_dir>/orchestrator.log``.
        mask_sensitive: Mask sensitive values in logs.
        max_bytes: Max file size before rotation.
        backup_count: Number of rotated files to keep.

    Returns:
        Root logger instance.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Resolve log file path
    resolved_log_file: Optional[str] = log_file
    if resolved_log_file is None and log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        resolved_log_file = str(Path(log_dir) / "orchestrator.log")

    # ------------------------------------------------------------------
    # structlog path
    # ------------------------------------------------------------------
    if STRUCTLOG_AVAILABLE:
        processors: list = [
            merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ]

        if mask_sensitive:
            processors.append(mask_sensitive_data)

        if fmt == "json":
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())

        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )

    # ------------------------------------------------------------------
    # stdlib logging (always configured – structlog wraps it)
    # ------------------------------------------------------------------
    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates on re-init
    root.handlers.clear()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(numeric_level)

    if fmt == "json" and not STRUCTLOG_AVAILABLE:
        console.setFormatter(JSONFormatter(mask_sensitive=mask_sensitive))
    else:
        console.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    root.addHandler(console)

    # File handler (with rotation)
    if resolved_log_file:
        Path(resolved_log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            resolved_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)  # capture everything to file
        file_handler.setFormatter(JSONFormatter(mask_sensitive=mask_sensitive))
        root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for name in ("urllib3", "docker", "httpx", "httpcore", "github"):
        logging.getLogger(name).setLevel(logging.WARNING)

    return root


# =============================================================================
# LOG CONTEXT MANAGER
# =============================================================================


class LogContext:
    """
    Context manager that binds key-value pairs to all logs emitted
    inside the block.

    Uses structlog's contextvars when available; otherwise falls
    back to a no-op (context is not propagated with stdlib alone).

    Usage::

        with LogContext(issue_number=123, agent="developer"):
            logger.info("Starting work")
            # All logs include issue_number=123 and agent="developer"
    """

    def __init__(self, **kwargs: Any):
        self.context = kwargs

    def __enter__(self) -> "LogContext":
        if STRUCTLOG_AVAILABLE:
            bind_contextvars(**self.context)
        return self

    def __exit__(self, *args: Any) -> None:
        if STRUCTLOG_AVAILABLE:
            unbind_contextvars(*self.context.keys())


@contextmanager
def log_context(**kwargs: Any):
    """Functional alias for :class:`LogContext`."""
    ctx = LogContext(**kwargs)
    ctx.__enter__()
    try:
        yield ctx
    finally:
        ctx.__exit__(None, None, None)


# =============================================================================
# AUDIT LOGGER
# =============================================================================


class AuditLogger:
    """
    Special logger for audit trail events.

    Records structured events to a JSONL file (one JSON object per line)
    for compliance, debugging, and post-mortem analysis.

    Event categories:
        - ``state_transition``: Workflow state changes
        - ``agent_execution``: Agent runs
        - ``code_change``: File modifications
        - ``github_api``: GitHub API calls
        - ``decision``: LLM-based decisions

    Usage::

        audit = AuditLogger("./logs/audit.jsonl")
        audit.log_state_transition(123, "TRIAGE", "PLANNING", "triage_router")
    """

    def __init__(
        self,
        output_path: str = "./logs/audit.jsonl",
        max_bytes: int = 500 * 1024 * 1024,  # 500 MB
        backup_count: int = 30,
    ):
        self.output_path = output_path
        self._logger = logging.getLogger("audit")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False  # don't echo to root logger

        # Setup rotating file handler for audit log
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            output_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)

    # -----------------------------------------------------------------
    # Event writers
    # -----------------------------------------------------------------

    def _write_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Write a single audit event."""
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": event_type,
            **data,
        }
        self._logger.info(json.dumps(event, default=str))

    def log_state_transition(
        self,
        issue_number: int,
        from_state: str,
        to_state: str,
        trigger: str,
        agent_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a workflow state transition."""
        self._write_event("state_transition", {
            "issue_number": issue_number,
            "from_state": from_state,
            "to_state": to_state,
            "trigger": trigger,
            "agent_type": agent_type,
            **(details or {}),
        })

    def log_agent_execution(
        self,
        agent_type: str,
        issue_number: int,
        result: str,
        duration: float,
        output_summary: str = "",
    ) -> None:
        """Log an agent execution."""
        self._write_event("agent_execution", {
            "agent_type": agent_type,
            "issue_number": issue_number,
            "result": result,
            "duration_seconds": round(duration, 2),
            "output_summary": output_summary,
        })

    def log_code_change(
        self,
        issue_number: int,
        files_modified: List[str],
        agent_type: str,
        commit_hash: Optional[str] = None,
    ) -> None:
        """Log code changes made by an agent."""
        self._write_event("code_change", {
            "issue_number": issue_number,
            "agent_type": agent_type,
            "files_modified": files_modified,
            "file_count": len(files_modified),
            "commit_hash": commit_hash,
        })

    def log_github_api(
        self,
        endpoint: str,
        method: str,
        status: int,
        issue_number: Optional[int] = None,
    ) -> None:
        """Log a GitHub API call."""
        self._write_event("github_api", {
            "endpoint": endpoint,
            "method": method,
            "status": status,
            "issue_number": issue_number,
        })

    def log_decision(
        self,
        issue_number: int,
        decision_type: str,
        decision: str,
        reasoning: str = "",
    ) -> None:
        """Log an LLM-based decision."""
        self._write_event("decision", {
            "issue_number": issue_number,
            "decision_type": decision_type,
            "decision": decision,
            "reasoning": reasoning,
        })

    def log_llm_call(
        self,
        model: str,
        agent_type: str,
        input_tokens: int,
        output_tokens: int,
        duration: float,
        cost: float = 0.0,
    ) -> None:
        """Log an LLM API call."""
        self._write_event("llm_call", {
            "model": model,
            "agent_type": agent_type,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "duration_seconds": round(duration, 2),
            "estimated_cost": round(cost, 6),
        })

    def log_error(
        self,
        component: str,
        error_type: str,
        message: str,
        issue_number: Optional[int] = None,
    ) -> None:
        """Log an error event."""
        self._write_event("error", {
            "component": component,
            "error_type": error_type,
            "message": message,
            "issue_number": issue_number,
        })


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Setup
    "setup_logging",
    "STRUCTLOG_AVAILABLE",
    # Data masking
    "mask_sensitive_data",
    "mask_dict",
    # Context
    "LogContext",
    "log_context",
    # Formatters
    "JSONFormatter",
    # Audit
    "AuditLogger",
]