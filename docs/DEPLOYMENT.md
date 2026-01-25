# Deployment Guide

## Overview

This guide covers deploying the AI Agent Development System in production environments.

## Deployment Options

### 1. Local Development

Best for: Testing, development, small projects

```bash
docker-compose up
```

### 2. Single Server

Best for: Small teams, moderate volume

```bash
# Production compose file
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### 3. Kubernetes

Best for: High availability, auto-scaling, enterprise

See Kubernetes manifests below.

### 4. Cloud Managed

Best for: Minimal ops, serverless agents

- AWS ECS/Fargate
- Google Cloud Run
- Azure Container Instances

## Production Checklist

### Security

- [ ] Rotate all API keys and tokens
- [ ] Use secrets management (Vault, AWS Secrets Manager)
- [ ] Enable TLS for all endpoints
- [ ] Configure firewall rules
- [ ] Set up audit logging
- [ ] Review container security

### Reliability

- [ ] Configure health checks
- [ ] Set up monitoring and alerts
- [ ] Implement backup strategy
- [ ] Test recovery procedures
- [ ] Configure auto-restart

### Performance

- [ ] Size containers appropriately
- [ ] Configure connection pooling
- [ ] Set appropriate timeouts
- [ ] Enable caching where applicable

## Docker Production Configuration

### docker-compose.prod.yml

```yaml
version: '3.8'

services:
  orchestrator:
    restart: always
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
      - ENABLE_TRACING=true

  redis:
    restart: always
    command: redis-server --appendonly yes --maxmemory 1gb --maxmemory-policy allkeys-lru
```

## Kubernetes Deployment

### Namespace

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ai-agent-system
```

### Orchestrator Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: orchestrator
  namespace: ai-agent-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: orchestrator
  template:
    metadata:
      labels:
        app: orchestrator
    spec:
      containers:
      - name: orchestrator
        image: ai-agent-orchestrator:latest
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2"
        ports:
        - containerPort: 8080
        envFrom:
        - secretRef:
            name: ai-agent-secrets
        - configMapRef:
            name: ai-agent-config
        volumeMounts:
        - name: memory
          mountPath: /memory
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
      volumes:
      - name: memory
        persistentVolumeClaim:
          claimName: memory-pvc
```

### Secrets

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: ai-agent-secrets
  namespace: ai-agent-system
type: Opaque
stringData:
  GITHUB_TOKEN: "ghp_xxxxx"
  ANTHROPIC_API_KEY: "sk-ant-xxxxx"
  GITHUB_WEBHOOK_SECRET: "webhook-secret"
```

### ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ai-agent-config
  namespace: ai-agent-system
data:
  PROJECT_ID: "my-project"
  GITHUB_REPO: "owner/repo"
  LLM_PROVIDER: "anthropic"
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
```

## Monitoring Setup

### Prometheus

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'orchestrator'
    static_configs:
      - targets: ['orchestrator:8080']
```

### Grafana Dashboards

Import dashboards from `monitoring/grafana/provisioning/`.

### Alerts

```yaml
# alerts.yml
groups:
- name: ai-agent-alerts
  rules:
  - alert: OrchestratorDown
    expr: up{job="orchestrator"} == 0
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Orchestrator is down"

  - alert: HighFailureRate
    expr: rate(qa_results_total{result="fail"}[1h]) > 0.5
    for: 15m
    labels:
      severity: warning
    annotations:
      summary: "High QA failure rate"

  - alert: TaskBlocked
    expr: issues_in_progress{state="BLOCKED"} > 0
    for: 1h
    labels:
      severity: warning
    annotations:
      summary: "Task blocked for over 1 hour"
```

## Backup and Recovery

### Memory Backup

```bash
# Backup memory to S3
aws s3 sync ./memory s3://my-bucket/ai-agent-memory/$(date +%Y%m%d)

# Restore
aws s3 sync s3://my-bucket/ai-agent-memory/20240101 ./memory
```

### State Backup

For Redis:
```bash
redis-cli BGSAVE
```

For PostgreSQL:
```bash
pg_dump -h localhost -U orchestrator orchestrator > backup.sql
```

## Scaling

### Horizontal Scaling

For multiple projects:
- Deploy separate orchestrator per project
- Share Redis cluster for coordination
- Use different namespaces/networks

### Agent Concurrency

Adjust based on resources:
```bash
ORCHESTRATOR_MAX_CONCURRENT_AGENTS=5
```

### Rate Limits

Configure for your API limits:
```bash
# GitHub: 5000/hour authenticated
# Anthropic: varies by plan
# OpenAI: varies by plan
```

## Cost Optimization

### LLM Costs

- Use smaller models for simple tasks (Haiku for docs)
- Cache repeated queries
- Monitor token usage

### Compute Costs

- Right-size containers
- Use spot instances for agents
- Implement auto-scaling

## Troubleshooting Production

### Common Issues

1. **Agent timeouts**: Increase `AGENT_TIMEOUT`
2. **Rate limits**: Reduce concurrency, add delays
3. **State corruption**: Check Redis/PostgreSQL health
4. **Memory issues**: Increase container limits

### Debug Mode

```bash
LOG_LEVEL=DEBUG docker-compose up orchestrator
```

### Recovery Procedures

1. Stop orchestrator
2. Backup current state
3. Fix underlying issue
4. Restore state if needed
5. Restart orchestrator
6. Verify with health check
