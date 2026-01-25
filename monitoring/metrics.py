# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - METRICS COLLECTION
# =============================================================================
"""
Metrics Collection Module

Collects and exports metrics for monitoring the AI agent system.
Uses Prometheus client library for metrics export.

Metric Categories:
    - Issue metrics: Processing times, throughput, states
    - Agent metrics: Execution times, success rates
    - LLM metrics: Token usage, costs, latency
    - System metrics: Queue depth, errors, uptime
"""

# =============================================================================
# METRICS COLLECTOR
# =============================================================================
"""
from prometheus_client import Counter, Histogram, Gauge, Info

class MetricsCollector:
    '''
    Collects and exports system metrics.

    Usage:
        metrics = MetricsCollector()
        metrics.record_issue_processed(issue_number, result="success")
        metrics.record_agent_execution("developer", duration=120.5)
    '''

    def __init__(self):
        '''Initialize all metric collectors.'''

        # ---------------------------------------------------------------------
        # ISSUE METRICS
        # ---------------------------------------------------------------------
        self.issues_total = Counter(
            'issues_total',
            'Total number of issues processed',
            ['issue_type', 'result']
        )

        self.issues_in_progress = Gauge(
            'issues_in_progress',
            'Number of issues currently in progress',
            ['state']
        )

        self.issue_processing_duration = Histogram(
            'issue_processing_duration_seconds',
            'Time to process an issue from READY to DONE',
            buckets=[60, 300, 600, 1800, 3600, 7200, 14400]
        )

        self.issue_iterations = Histogram(
            'issue_iterations_count',
            'Number of iterations before issue completion',
            buckets=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        )

        # ---------------------------------------------------------------------
        # AGENT METRICS
        # ---------------------------------------------------------------------
        self.agent_executions = Counter(
            'agent_executions_total',
            'Total agent executions',
            ['agent_type', 'result']
        )

        self.agent_duration = Histogram(
            'agent_execution_duration_seconds',
            'Agent execution duration',
            ['agent_type'],
            buckets=[30, 60, 120, 300, 600, 1200, 1800]
        )

        self.agents_active = Gauge(
            'agents_active',
            'Number of currently running agents',
            ['agent_type']
        )

        # ---------------------------------------------------------------------
        # QA METRICS
        # ---------------------------------------------------------------------
        self.qa_results = Counter(
            'qa_results_total',
            'QA validation results',
            ['result']  # pass, fail
        )

        self.qa_pass_rate = Gauge(
            'qa_pass_rate',
            'Rolling QA pass rate'
        )

        # ---------------------------------------------------------------------
        # LLM METRICS
        # ---------------------------------------------------------------------
        self.llm_requests = Counter(
            'llm_requests_total',
            'Total LLM API requests',
            ['model', 'agent_type']
        )

        self.llm_tokens = Counter(
            'llm_tokens_total',
            'Total tokens used',
            ['model', 'token_type']  # input, output
        )

        self.llm_latency = Histogram(
            'llm_request_duration_seconds',
            'LLM request duration',
            ['model'],
            buckets=[1, 2, 5, 10, 30, 60, 120]
        )

        self.llm_cost = Counter(
            'llm_cost_dollars',
            'Estimated LLM cost in dollars',
            ['model']
        )

        # ---------------------------------------------------------------------
        # GITHUB METRICS
        # ---------------------------------------------------------------------
        self.github_requests = Counter(
            'github_api_requests_total',
            'Total GitHub API requests',
            ['endpoint', 'method', 'status']
        )

        self.github_rate_limit = Gauge(
            'github_rate_limit_remaining',
            'Remaining GitHub API rate limit'
        )

        # ---------------------------------------------------------------------
        # SYSTEM METRICS
        # ---------------------------------------------------------------------
        self.queue_depth = Gauge(
            'queue_depth',
            'Number of issues waiting to be processed'
        )

        self.errors = Counter(
            'errors_total',
            'Total errors',
            ['component', 'error_type']
        )

        self.uptime = Counter(
            'orchestrator_uptime_seconds',
            'Orchestrator uptime in seconds'
        )

        self.system_info = Info(
            'system',
            'System information'
        )

    # =========================================================================
    # RECORDING METHODS
    # =========================================================================

    def record_issue_processed(self, issue_type: str, result: str):
        '''Record an issue being processed.'''
        self.issues_total.labels(issue_type=issue_type, result=result).inc()

    def record_agent_execution(
        self,
        agent_type: str,
        duration: float,
        result: str
    ):
        '''Record agent execution metrics.'''
        self.agent_executions.labels(agent_type=agent_type, result=result).inc()
        self.agent_duration.labels(agent_type=agent_type).observe(duration)

    def record_llm_call(
        self,
        model: str,
        agent_type: str,
        input_tokens: int,
        output_tokens: int,
        duration: float
    ):
        '''Record LLM API call metrics.'''
        self.llm_requests.labels(model=model, agent_type=agent_type).inc()
        self.llm_tokens.labels(model=model, token_type='input').inc(input_tokens)
        self.llm_tokens.labels(model=model, token_type='output').inc(output_tokens)
        self.llm_latency.labels(model=model).observe(duration)

        # Estimate cost (example rates)
        cost = self._estimate_cost(model, input_tokens, output_tokens)
        self.llm_cost.labels(model=model).inc(cost)

    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        '''Estimate LLM cost based on token counts.'''
        # Example pricing (adjust for actual rates)
        pricing = {
            'claude-3-opus': {'input': 0.015, 'output': 0.075},
            'claude-3-sonnet': {'input': 0.003, 'output': 0.015},
            'claude-3-haiku': {'input': 0.00025, 'output': 0.00125},
            'gpt-4': {'input': 0.03, 'output': 0.06},
            'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
        }
        rates = pricing.get(model, {'input': 0.01, 'output': 0.03})
        return (input_tokens * rates['input'] + output_tokens * rates['output']) / 1000
'''
"""
