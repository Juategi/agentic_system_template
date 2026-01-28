# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - METRICS COLLECTION
# =============================================================================
"""
Metrics Collection Module

Collects and exports metrics for monitoring the AI agent system.
Supports Prometheus client library when available, with an in-memory
fallback for environments without Prometheus.

Metric Categories:
    - Issue metrics: Processing times, throughput, states
    - Agent metrics: Execution times, success rates
    - LLM metrics: Token usage, costs, latency
    - GitHub metrics: API requests, rate limits
    - System metrics: Queue depth, errors, uptime
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try to import prometheus_client; provide in-memory fallback
try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Info,
        start_http_server,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


# =============================================================================
# IN-MEMORY FALLBACK METRICS
# =============================================================================


class _InMemoryCounter:
    """Minimal counter that stores values in memory."""

    def __init__(self, name: str, description: str, labels: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self._label_names = labels or []
        self._values: Dict[tuple, float] = defaultdict(float)
        self._lock = threading.Lock()

    def labels(self, **kwargs) -> "_InMemoryCounter":
        key = tuple(kwargs.get(l, "") for l in self._label_names)
        view = _InMemoryCounterView(self, key)
        return view

    def inc(self, amount: float = 1) -> None:
        with self._lock:
            self._values[()] += amount

    def get(self) -> Dict[tuple, float]:
        with self._lock:
            return dict(self._values)


class _InMemoryCounterView:
    """Labelled view of an in-memory counter."""

    def __init__(self, parent: _InMemoryCounter, key: tuple):
        self._parent = parent
        self._key = key

    def inc(self, amount: float = 1) -> None:
        with self._parent._lock:
            self._parent._values[self._key] += amount


class _InMemoryGauge:
    """Minimal gauge that stores values in memory."""

    def __init__(self, name: str, description: str, labels: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self._label_names = labels or []
        self._values: Dict[tuple, float] = defaultdict(float)
        self._lock = threading.Lock()

    def labels(self, **kwargs) -> "_InMemoryGauge":
        return _InMemoryGaugeView(self, tuple(kwargs.get(l, "") for l in self._label_names))

    def set(self, value: float) -> None:
        with self._lock:
            self._values[()] = value

    def inc(self, amount: float = 1) -> None:
        with self._lock:
            self._values[()] += amount

    def dec(self, amount: float = 1) -> None:
        with self._lock:
            self._values[()] -= amount

    def get(self) -> Dict[tuple, float]:
        with self._lock:
            return dict(self._values)


class _InMemoryGaugeView:
    """Labelled view of an in-memory gauge."""

    def __init__(self, parent: _InMemoryGauge, key: tuple):
        self._parent = parent
        self._key = key

    def set(self, value: float) -> None:
        with self._parent._lock:
            self._parent._values[self._key] = value

    def inc(self, amount: float = 1) -> None:
        with self._parent._lock:
            self._parent._values[self._key] += amount

    def dec(self, amount: float = 1) -> None:
        with self._parent._lock:
            self._parent._values[self._key] -= amount


class _InMemoryHistogram:
    """Minimal histogram that stores observations in memory."""

    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None,
    ):
        self.name = name
        self.description = description
        self._label_names = labels or []
        self._buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
        self._observations: Dict[tuple, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def labels(self, **kwargs) -> "_InMemoryHistogram":
        return _InMemoryHistogramView(
            self, tuple(kwargs.get(l, "") for l in self._label_names)
        )

    def observe(self, value: float) -> None:
        with self._lock:
            self._observations[()].append(value)

    def get(self) -> Dict[tuple, List[float]]:
        with self._lock:
            return {k: list(v) for k, v in self._observations.items()}


class _InMemoryHistogramView:
    """Labelled view of an in-memory histogram."""

    def __init__(self, parent: _InMemoryHistogram, key: tuple):
        self._parent = parent
        self._key = key

    def observe(self, value: float) -> None:
        with self._parent._lock:
            self._parent._observations[self._key].append(value)


class _InMemoryInfo:
    """Minimal info metric."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self._info: Dict[str, str] = {}

    def info(self, data: Dict[str, str]) -> None:
        self._info.update(data)

    def get(self) -> Dict[str, str]:
        return dict(self._info)


# =============================================================================
# LLM COST ESTIMATOR
# =============================================================================

# Pricing per 1K tokens (USD) – update as providers adjust rates
LLM_PRICING: Dict[str, Dict[str, float]] = {
    # Anthropic
    "claude-opus-4-20250514": {"input": 0.015, "output": 0.075},
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-3-5-haiku-20241022": {"input": 0.001, "output": 0.005},
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
    # OpenAI
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
}

# Fallback pricing for unknown models
_DEFAULT_PRICING = {"input": 0.01, "output": 0.03}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Estimate the dollar cost of an LLM call.

    Args:
        model: Model identifier.
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.

    Returns:
        Estimated cost in USD.
    """
    # Try exact match, then prefix match
    rates = LLM_PRICING.get(model)
    if rates is None:
        for key in LLM_PRICING:
            if model.startswith(key.rsplit("-", 1)[0]):
                rates = LLM_PRICING[key]
                break
    if rates is None:
        rates = _DEFAULT_PRICING

    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1000


# =============================================================================
# METRICS COLLECTOR
# =============================================================================


class MetricsCollector:
    """
    Central metrics collector for the AI Agent Development System.

    Uses Prometheus client library when available; otherwise stores
    metrics in memory with the same API surface.

    Usage::

        metrics = MetricsCollector()
        metrics.record_issue_processed("feature", "success")
        metrics.record_agent_execution("developer", 120.5, "success")
        metrics.record_llm_call("claude-3-5-sonnet-20241022", "developer", 1500, 800, 3.2)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize all metric collectors.

        Args:
            config: Optional metrics configuration dict.
        """
        self.config = config or {}
        self._start_time = time.monotonic()
        self._lock = threading.Lock()

        # Snapshot counters for computing rolling rates
        self._qa_pass_count = 0
        self._qa_total_count = 0

        if PROMETHEUS_AVAILABLE:
            self._init_prometheus()
        else:
            self._init_in_memory()
            logger.info(
                "prometheus_client not installed – using in-memory metrics"
            )

    # -----------------------------------------------------------------
    # Prometheus initialization
    # -----------------------------------------------------------------

    def _init_prometheus(self) -> None:
        # Issue metrics
        self.issues_total = Counter(
            "issues_total",
            "Total number of issues processed",
            ["issue_type", "result"],
        )
        self.issues_in_progress = Gauge(
            "issues_in_progress",
            "Number of issues currently in progress",
            ["state"],
        )
        self.issue_processing_duration = Histogram(
            "issue_processing_duration_seconds",
            "Time to process an issue from READY to DONE",
            buckets=[60, 300, 600, 1800, 3600, 7200, 14400],
        )
        self.issue_iterations = Histogram(
            "issue_iterations_count",
            "Number of iterations before issue completion",
            buckets=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        )

        # Agent metrics
        self.agent_executions = Counter(
            "agent_executions_total",
            "Total agent executions",
            ["agent_type", "result"],
        )
        self.agent_duration = Histogram(
            "agent_execution_duration_seconds",
            "Agent execution duration",
            ["agent_type"],
            buckets=[30, 60, 120, 300, 600, 1200, 1800],
        )
        self.agents_active = Gauge(
            "agents_active",
            "Number of currently running agents",
            ["agent_type"],
        )

        # QA metrics
        self.qa_results = Counter(
            "qa_results_total",
            "QA validation results",
            ["result"],
        )
        self.qa_pass_rate = Gauge(
            "qa_pass_rate",
            "Rolling QA pass rate",
        )

        # LLM metrics
        self.llm_requests = Counter(
            "llm_requests_total",
            "Total LLM API requests",
            ["model", "agent_type"],
        )
        self.llm_tokens = Counter(
            "llm_tokens_total",
            "Total tokens used",
            ["model", "token_type"],
        )
        self.llm_latency = Histogram(
            "llm_request_duration_seconds",
            "LLM request duration",
            ["model"],
            buckets=[1, 2, 5, 10, 30, 60, 120],
        )
        self.llm_cost = Counter(
            "llm_cost_dollars",
            "Estimated LLM cost in dollars",
            ["model"],
        )

        # GitHub metrics
        self.github_requests = Counter(
            "github_api_requests_total",
            "Total GitHub API requests",
            ["endpoint", "method", "status"],
        )
        self.github_rate_limit = Gauge(
            "github_rate_limit_remaining",
            "Remaining GitHub API rate limit",
        )

        # System metrics
        self.queue_depth = Gauge(
            "queue_depth",
            "Number of issues waiting to be processed",
        )
        self.errors_total = Counter(
            "errors_total",
            "Total errors",
            ["component", "error_type"],
        )
        self.state_transitions = Counter(
            "state_transitions_total",
            "Total state machine transitions",
            ["from_state", "to_state"],
        )
        self.system_info = Info(
            "system",
            "System information",
        )

    # -----------------------------------------------------------------
    # In-memory fallback initialization
    # -----------------------------------------------------------------

    def _init_in_memory(self) -> None:
        self.issues_total = _InMemoryCounter(
            "issues_total", "Total issues", ["issue_type", "result"]
        )
        self.issues_in_progress = _InMemoryGauge(
            "issues_in_progress", "In progress", ["state"]
        )
        self.issue_processing_duration = _InMemoryHistogram(
            "issue_processing_duration_seconds", "Duration",
            buckets=[60, 300, 600, 1800, 3600, 7200, 14400],
        )
        self.issue_iterations = _InMemoryHistogram(
            "issue_iterations_count", "Iterations",
            buckets=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        )

        self.agent_executions = _InMemoryCounter(
            "agent_executions_total", "Agent executions", ["agent_type", "result"]
        )
        self.agent_duration = _InMemoryHistogram(
            "agent_execution_duration_seconds", "Agent duration", ["agent_type"],
            buckets=[30, 60, 120, 300, 600, 1200, 1800],
        )
        self.agents_active = _InMemoryGauge(
            "agents_active", "Active agents", ["agent_type"]
        )

        self.qa_results = _InMemoryCounter(
            "qa_results_total", "QA results", ["result"]
        )
        self.qa_pass_rate = _InMemoryGauge("qa_pass_rate", "QA pass rate")

        self.llm_requests = _InMemoryCounter(
            "llm_requests_total", "LLM requests", ["model", "agent_type"]
        )
        self.llm_tokens = _InMemoryCounter(
            "llm_tokens_total", "Tokens", ["model", "token_type"]
        )
        self.llm_latency = _InMemoryHistogram(
            "llm_request_duration_seconds", "LLM latency", ["model"],
            buckets=[1, 2, 5, 10, 30, 60, 120],
        )
        self.llm_cost = _InMemoryCounter(
            "llm_cost_dollars", "LLM cost", ["model"]
        )

        self.github_requests = _InMemoryCounter(
            "github_api_requests_total", "GitHub requests",
            ["endpoint", "method", "status"],
        )
        self.github_rate_limit = _InMemoryGauge(
            "github_rate_limit_remaining", "Rate limit"
        )

        self.queue_depth = _InMemoryGauge("queue_depth", "Queue depth")
        self.errors_total = _InMemoryCounter(
            "errors_total", "Errors", ["component", "error_type"]
        )
        self.state_transitions = _InMemoryCounter(
            "state_transitions_total", "Transitions", ["from_state", "to_state"]
        )
        self.system_info = _InMemoryInfo("system", "System info")

    # =====================================================================
    # RECORDING METHODS
    # =====================================================================

    # -- Issue metrics ----------------------------------------------------

    def record_issue_processed(self, issue_type: str, result: str) -> None:
        """Record an issue reaching a terminal state."""
        self.issues_total.labels(issue_type=issue_type, result=result).inc()

    def record_issue_duration(self, duration_seconds: float) -> None:
        """Record time taken from READY to DONE."""
        self.issue_processing_duration.observe(duration_seconds)

    def record_issue_iterations(self, iterations: int) -> None:
        """Record number of QA iterations an issue required."""
        self.issue_iterations.observe(float(iterations))

    def set_issues_in_progress(self, state: str, count: int) -> None:
        """Set the current number of issues in a given workflow state."""
        self.issues_in_progress.labels(state=state).set(count)

    # -- Agent metrics ----------------------------------------------------

    def record_agent_execution(
        self, agent_type: str, duration: float, result: str
    ) -> None:
        """Record an agent execution."""
        self.agent_executions.labels(agent_type=agent_type, result=result).inc()
        self.agent_duration.labels(agent_type=agent_type).observe(duration)

    def agent_started(self, agent_type: str) -> None:
        """Increment active agent count."""
        self.agents_active.labels(agent_type=agent_type).inc()

    def agent_finished(self, agent_type: str) -> None:
        """Decrement active agent count."""
        self.agents_active.labels(agent_type=agent_type).dec()

    # -- QA metrics -------------------------------------------------------

    def record_qa_result(self, passed: bool) -> None:
        """Record a QA result and update rolling pass rate."""
        result = "pass" if passed else "fail"
        self.qa_results.labels(result=result).inc()

        with self._lock:
            self._qa_total_count += 1
            if passed:
                self._qa_pass_count += 1
            rate = self._qa_pass_count / self._qa_total_count
        self.qa_pass_rate.set(round(rate, 4))

    # -- LLM metrics ------------------------------------------------------

    def record_llm_call(
        self,
        model: str,
        agent_type: str,
        input_tokens: int,
        output_tokens: int,
        duration: float,
    ) -> None:
        """Record an LLM API call with token counts and latency."""
        self.llm_requests.labels(model=model, agent_type=agent_type).inc()
        self.llm_tokens.labels(model=model, token_type="input").inc(input_tokens)
        self.llm_tokens.labels(model=model, token_type="output").inc(output_tokens)
        self.llm_latency.labels(model=model).observe(duration)

        cost = estimate_cost(model, input_tokens, output_tokens)
        self.llm_cost.labels(model=model).inc(cost)

    # -- GitHub metrics ----------------------------------------------------

    def record_github_request(
        self, endpoint: str, method: str, status: int
    ) -> None:
        """Record a GitHub API request."""
        self.github_requests.labels(
            endpoint=endpoint, method=method, status=str(status)
        ).inc()

    def set_github_rate_limit(self, remaining: int) -> None:
        """Update the remaining GitHub API rate limit."""
        self.github_rate_limit.set(remaining)

    # -- System metrics ----------------------------------------------------

    def set_queue_depth(self, depth: int) -> None:
        """Set the current queue depth."""
        self.queue_depth.set(depth)

    def record_error(self, component: str, error_type: str) -> None:
        """Record an error occurrence."""
        self.errors_total.labels(component=component, error_type=error_type).inc()

    def record_state_transition(self, from_state: str, to_state: str) -> None:
        """Record a workflow state transition."""
        self.state_transitions.labels(
            from_state=from_state, to_state=to_state
        ).inc()

    def set_system_info(self, **info: str) -> None:
        """Set system information labels."""
        self.system_info.info(info)

    def get_uptime(self) -> float:
        """Return seconds since this collector was created."""
        return time.monotonic() - self._start_time

    # =====================================================================
    # EXPORT / SNAPSHOT
    # =====================================================================

    def start_http_server(self, port: int = 8080) -> None:
        """
        Start an HTTP server that exposes Prometheus metrics.

        Only available when ``prometheus_client`` is installed.
        """
        if not PROMETHEUS_AVAILABLE:
            logger.warning(
                "Cannot start metrics HTTP server: prometheus_client not installed"
            )
            return
        start_http_server(port)
        logger.info(f"Metrics HTTP server started on port {port}")

    def snapshot(self) -> Dict[str, Any]:
        """
        Return a plain dict snapshot of all metrics.

        Useful for logging, health checks, or environments without
        Prometheus.
        """
        result: Dict[str, Any] = {
            "uptime_seconds": round(self.get_uptime(), 1),
            "qa_pass_rate": round(
                self._qa_pass_count / self._qa_total_count, 4
            )
            if self._qa_total_count > 0
            else None,
            "qa_total": self._qa_total_count,
        }

        # For in-memory backend, include raw values
        if not PROMETHEUS_AVAILABLE:
            for attr_name in (
                "issues_total",
                "agent_executions",
                "llm_requests",
                "llm_tokens",
                "llm_cost",
                "github_requests",
                "errors_total",
                "state_transitions",
            ):
                obj = getattr(self, attr_name, None)
                if obj is not None and hasattr(obj, "get"):
                    values = obj.get()
                    if values:
                        result[attr_name] = {
                            str(k): v for k, v in values.items()
                        }

            for attr_name in (
                "issues_in_progress",
                "agents_active",
                "queue_depth",
                "github_rate_limit",
            ):
                obj = getattr(self, attr_name, None)
                if obj is not None and hasattr(obj, "get"):
                    values = obj.get()
                    if values:
                        result[attr_name] = {
                            str(k): v for k, v in values.items()
                        }

        return result


# =============================================================================
# ALERTING
# =============================================================================


class AlertManager:
    """
    Rule-based alerting that evaluates conditions against current metrics.

    Alerts are dispatched to registered channels (log, callback).

    Usage::

        alerts = AlertManager(metrics)
        alerts.add_rule("high_failure", "qa_pass_rate < 0.5", severity="critical")
        alerts.check_all()
    """

    def __init__(self, metrics: MetricsCollector):
        self.metrics = metrics
        self._rules: List[AlertRule] = []
        self._logger = logging.getLogger("monitoring.alerts")
        self._callbacks: List[Any] = []
        self._fired: Dict[str, float] = {}  # rule_name -> last_fired_ts
        self._cooldown = 300  # seconds between repeated alerts

    def add_rule(
        self,
        name: str,
        check_fn,
        severity: str = "warning",
        message: str = "",
    ) -> None:
        """
        Register an alert rule.

        Args:
            name: Unique rule identifier.
            check_fn: Callable receiving ``MetricsCollector`` and returning
                ``True`` when the alert condition is met.
            severity: ``"info"``, ``"warning"``, ``"error"``, or ``"critical"``.
            message: Human-readable message template.
        """
        self._rules.append(AlertRule(name, check_fn, severity, message))

    def add_callback(self, callback) -> None:
        """
        Register a callback invoked when an alert fires.

        Signature: ``callback(name: str, severity: str, message: str)``.
        """
        self._callbacks.append(callback)

    def check_all(self) -> List[Dict[str, str]]:
        """
        Evaluate all rules and fire alerts for those that trigger.

        Returns:
            List of fired alert dicts.
        """
        fired = []
        now = time.time()

        for rule in self._rules:
            try:
                if not rule.check_fn(self.metrics):
                    continue
            except Exception:
                continue

            # Cooldown check
            last = self._fired.get(rule.name, 0)
            if now - last < self._cooldown:
                continue

            self._fired[rule.name] = now
            alert = {
                "name": rule.name,
                "severity": rule.severity,
                "message": rule.message,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            fired.append(alert)

            # Log
            log_fn = getattr(self._logger, rule.severity, self._logger.warning)
            log_fn(f"ALERT [{rule.severity.upper()}] {rule.name}: {rule.message}")

            # Callbacks
            for cb in self._callbacks:
                try:
                    cb(rule.name, rule.severity, rule.message)
                except Exception as e:
                    self._logger.error(f"Alert callback failed: {e}")

        return fired


class AlertRule:
    """A single alert rule definition."""

    __slots__ = ("name", "check_fn", "severity", "message")

    def __init__(self, name: str, check_fn, severity: str, message: str):
        self.name = name
        self.check_fn = check_fn
        self.severity = severity
        self.message = message


# =============================================================================
# HEALTH CHECK
# =============================================================================


class HealthCheck:
    """
    System health checker that aggregates component statuses.

    Usage::

        health = HealthCheck()
        health.register("github", github_check_fn)
        health.register("redis", redis_check_fn)
        result = await health.check_all()
    """

    def __init__(self):
        self._checks: Dict[str, Any] = {}
        self._critical: Dict[str, bool] = {}

    def register(self, name: str, check_fn, critical: bool = True) -> None:
        """
        Register a health check.

        Args:
            name: Component name.
            check_fn: Async callable returning a dict with at least
                ``{"healthy": bool}``.
            critical: Whether failure of this check means the system
                is unhealthy overall.
        """
        self._checks[name] = check_fn
        self._critical[name] = critical

    async def check_all(self) -> Dict[str, Any]:
        """
        Run all registered health checks.

        Returns:
            Aggregated health status dict.
        """
        results: Dict[str, Any] = {}
        overall_healthy = True

        for name, check_fn in self._checks.items():
            try:
                result = await check_fn()
                results[name] = result
                if not result.get("healthy", False) and self._critical.get(name, True):
                    overall_healthy = False
            except Exception as e:
                results[name] = {"healthy": False, "error": str(e)}
                if self._critical.get(name, True):
                    overall_healthy = False

        return {
            "healthy": overall_healthy,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "checks": results,
        }

    async def check_one(self, name: str) -> Dict[str, Any]:
        """Run a single named health check."""
        check_fn = self._checks.get(name)
        if check_fn is None:
            return {"healthy": False, "error": f"Unknown check: {name}"}
        try:
            return await check_fn()
        except Exception as e:
            return {"healthy": False, "error": str(e)}


# =============================================================================
# FACTORY
# =============================================================================


def create_metrics_collector(
    config: Optional[Dict[str, Any]] = None,
) -> MetricsCollector:
    """
    Create a MetricsCollector from configuration.

    Args:
        config: Metrics section of monitoring.yaml.

    Returns:
        Configured MetricsCollector.
    """
    config = config or {}
    collector = MetricsCollector(config)

    # Start Prometheus HTTP server if configured
    if config.get("enabled", True) and PROMETHEUS_AVAILABLE:
        endpoint = config.get("endpoint", {})
        port = endpoint.get("port", 8080)
        try:
            collector.start_http_server(port)
        except OSError as e:
            logger.warning(f"Could not start metrics server on port {port}: {e}")

    return collector


def create_alert_manager(
    metrics: MetricsCollector,
    config: Optional[Dict[str, Any]] = None,
) -> AlertManager:
    """
    Create an AlertManager with default rules from configuration.

    Args:
        metrics: MetricsCollector instance.
        config: Alerting section of monitoring.yaml.

    Returns:
        Configured AlertManager.
    """
    config = config or {}
    manager = AlertManager(metrics)

    if not config.get("enabled", True):
        return manager

    # Register default rules from config
    rules = config.get("rules", {})

    if "high_failure_rate" in rules:
        manager.add_rule(
            name="high_failure_rate",
            check_fn=lambda m: (
                m._qa_total_count > 0
                and (m._qa_pass_count / m._qa_total_count) < 0.5
            ),
            severity="critical",
            message="High QA failure rate detected (>50%)",
        )

    if "rate_limit_warning" in rules:
        manager.add_rule(
            name="rate_limit_warning",
            check_fn=lambda m: (
                hasattr(m.github_rate_limit, "get")
                and any(v < 500 for v in m.github_rate_limit.get().values())
            ) if not PROMETHEUS_AVAILABLE else False,
            severity="warning",
            message="GitHub API rate limit approaching",
        )

    return manager


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core
    "MetricsCollector",
    "create_metrics_collector",
    "PROMETHEUS_AVAILABLE",
    # Cost estimation
    "estimate_cost",
    "LLM_PRICING",
    # Alerting
    "AlertManager",
    "AlertRule",
    "create_alert_manager",
    # Health checks
    "HealthCheck",
]