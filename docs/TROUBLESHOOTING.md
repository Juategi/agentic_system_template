# Troubleshooting Guide

## Quick Diagnostics

Run the health check first:
```bash
make health
```

## Common Issues

### Orchestrator Won't Start

**Symptoms**: Container exits immediately or won't start

**Check**:
```bash
docker-compose logs orchestrator
```

**Common causes**:
1. **Missing environment variables**
   - Ensure `.env` exists and has required variables
   - Run `./scripts/init_project.sh` to regenerate

2. **Docker socket permissions**
   ```bash
   sudo chmod 666 /var/run/docker.sock
   ```

3. **Port already in use**
   ```bash
   lsof -i :8080
   # Change WEBHOOK_PORT in .env if needed
   ```

### GitHub API Errors

**401 Unauthorized**
- Token expired or invalid
- Generate new token with correct permissions
- Verify: `curl -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/user`

**403 Forbidden**
- Rate limit exceeded (check `X-RateLimit-Remaining` header)
- Repository access denied (check token permissions)

**404 Not Found**
- Repository name incorrect
- Repository is private and token lacks access

### Agent Failures

**Agent timeout**
- Increase `AGENT_TIMEOUT` in `.env`
- Check for infinite loops in prompts
- Verify LLM API is responding

**Agent exits with error**
```bash
docker-compose logs agent-runner
```

**No output from agent**
- Check `/output` volume is mounted
- Verify write permissions
- Check agent logs

### LLM API Errors

**Rate limit exceeded**
- Reduce `ORCHESTRATOR_MAX_CONCURRENT_AGENTS`
- Add delays between requests
- Upgrade API plan

**Invalid API key**
- Verify key format
- Check environment variable is set correctly
- Ensure no extra whitespace

**Model not found**
- Check model name is correct for provider
- Verify model access on your account

### State Issues

**Tasks stuck in IN_PROGRESS**
- Check if agent container is running
- May need manual state reset
- Verify orchestrator is running

**Duplicate task processing**
- Ensure only one orchestrator instance
- Check state backend (Redis/file) is accessible

**State not persisting**
- Check state backend configuration
- Verify write permissions on state file
- Check Redis connection

### Memory/Context Issues

**Agent lacks context**
- Verify memory files exist
- Check volume mounts in container
- Ensure files are readable

**Outdated memory**
- Memory files may need manual update
- Check Doc agent is running correctly

### Docker Issues

**Image not found**
```bash
make build
```

**Container can't pull image**
- Check Docker registry credentials
- Verify network connectivity

**Out of disk space**
```bash
docker system prune
```

**Network issues**
```bash
docker network create ai-agent-network
```

## Debugging Techniques

### Enable Debug Logging

```bash
LOG_LEVEL=DEBUG make start
```

### Inspect Agent Container

```bash
# Run agent interactively
docker-compose run --rm agent-runner /bin/bash

# Inside container
ls -la /memory
ls -la /repo
cat /input/input.json
```

### Check State

```bash
# File backend
cat ./data/orchestrator_state.json

# Redis backend
redis-cli
> KEYS orchestrator:*
> GET orchestrator:state:123
```

### Monitor LLM Calls

Enable tracing:
```bash
ENABLE_TRACING=true make start
```

### Test GitHub Connection

```bash
./scripts/health_check.sh
```

## Recovery Procedures

### Reset Stuck Task

1. Update issue labels manually in GitHub
2. Remove from state:
   ```bash
   # File backend
   # Edit orchestrator_state.json, remove issue entry

   # Redis backend
   redis-cli DEL orchestrator:state:123
   ```
3. Add READY label to retry

### Restart Clean

```bash
make stop
make clean-state  # WARNING: Loses all progress
make start
```

### Recover from Corrupted State

1. Stop orchestrator
2. Backup current state
3. Reset state:
   ```bash
   rm ./data/orchestrator_state.json
   ```
4. Restart orchestrator
5. Re-label issues as READY

## Getting Help

1. Check logs: `make logs`
2. Run health check: `make health`
3. Review this guide
4. Check GitHub Issues for known problems
5. Create new issue with:
   - Error message
   - Logs
   - Configuration (redact secrets)
   - Steps to reproduce
