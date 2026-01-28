# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - MONITORING PACKAGE
# =============================================================================
"""
Monitoring Package

This package provides monitoring, metrics, and logging infrastructure
for the AI Agent Development System.

Components:
    - Logger: Structured logging with structlog/stdlib fallback
    - Metrics: Prometheus-compatible metrics collection
    - Audit: Audit trail recording to JSONL
    - Alerts: Rule-based alerting with cooldown
    - Health: Aggregated component health checks

Usage:
    from monitoring import setup_logging, MetricsCollector, AuditLogger

    # Setup logging
    setup_logging(level="INFO", fmt="json", log_dir="./logs")

    # Metrics
    metrics = MetricsCollector()
    metrics.record_issue_processed("feature", "success")
    metrics.record_llm_call("claude-3-5-sonnet", "developer", 1500, 800, 3.2)

    # Audit trail
    audit = AuditLogger("./logs/audit.jsonl")
    audit.log_state_transition(123, "TRIAGE", "PLANNING", "triage_router")

    # Health checks
    health = HealthCheck()
    health.register("state_manager", state_manager.health_check)
    result = await health.check_all()
"""

# Logger
from monitoring.logger import (
    setup_logging,
    AuditLogger,
    LogContext,
    log_context,
    JSONFormatter,
    mask_sensitive_data,
    mask_dict,
    STRUCTLOG_AVAILABLE,
)

# Metrics
from monitoring.metrics import (
    MetricsCollector,
    create_metrics_collector,
    estimate_cost,
    LLM_PRICING,
    AlertManager,
    AlertRule,
    create_alert_manager,
    HealthCheck,
    PROMETHEUS_AVAILABLE,
)


__all__ = [
    # Logger
    "setup_logging",
    "AuditLogger",
    "LogContext",
    "log_context",
    "JSONFormatter",
    "mask_sensitive_data",
    "mask_dict",
    "STRUCTLOG_AVAILABLE",
    # Metrics
    "MetricsCollector",
    "create_metrics_collector",
    "estimate_cost",
    "LLM_PRICING",
    "PROMETHEUS_AVAILABLE",
    # Alerting
    "AlertManager",
    "AlertRule",
    "create_alert_manager",
    # Health
    "HealthCheck",
]