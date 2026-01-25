# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - MONITORING PACKAGE
# =============================================================================
"""
Monitoring Package

This package provides monitoring, metrics, and logging infrastructure.

Components:
    - Metrics: Prometheus metrics collection
    - Logger: Structured logging
    - Audit: Audit trail recording

Features:
    - Prometheus-compatible metrics export
    - Structured JSON logging
    - Audit trail for compliance
    - Health check endpoints
"""

__all__ = [
    "MetricsCollector",
    "setup_logging",
    "AuditLogger",
]
