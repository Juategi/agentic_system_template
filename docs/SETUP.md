# Setup Guide

## Prerequisites

Before setting up the AI Agent Development System, ensure you have:

- **Docker** (version 20.10+)
- **Docker Compose** (version 2.0+)
- **Git** (version 2.30+)
- **GitHub Account** with repository access
- **LLM API Key** (Anthropic or OpenAI)

## Quick Start

### 1. Clone the Template

```bash
git clone https://github.com/your-org/agentic_system_template.git my-project
cd my-project
```

### 2. Run Initialization

```bash
./scripts/init_project.sh
```

This interactive script will:
- Prompt for project configuration
- Create `.env` file
- Initialize memory structure
- Optionally setup GitHub labels
- Build Docker images

### 3. Configure Environment

Edit `.env` with your settings:

```bash
# Required
GITHUB_TOKEN=ghp_your_token
GITHUB_REPO=owner/repo
ANTHROPIC_API_KEY=sk-ant-your_key  # or OPENAI_API_KEY

# Optional customization
PROJECT_ID=my-project
AGENT_MAX_ITERATIONS=5
```

### 4. Start the System

```bash
make start
```

### 5. Create Your First Feature

1. Go to your GitHub repository
2. Create a new issue with:
   - Title: `[Feature] Your feature name`
   - Body: Description and acceptance criteria
   - Labels: `feature`, `READY`

The system will automatically begin processing!

## Manual Setup (Alternative)

If you prefer manual setup:

### Step 1: Copy Environment File

```bash
cp .env.template .env
```

### Step 2: Edit Configuration

Open `.env` and configure:

```bash
# Project identification
PROJECT_ID=my-project
PROJECT_NAME="My AI Project"

# GitHub (required)
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
GITHUB_REPO=your-username/your-repo

# LLM Provider (choose one)
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxx

# Or for OpenAI:
# LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-xxxxxxxxxxxx
```

### Step 3: Setup GitHub Labels

```bash
./scripts/setup_github.sh
```

### Step 4: Build Docker Images

```bash
make build
```

### Step 5: Start Orchestrator

```bash
make start
```

## Configuration Details

### GitHub Token Permissions

Your GitHub token needs these permissions:
- `repo` - Full repository access
- `write:discussion` - Comment on issues

For GitHub Apps, configure:
- Repository permissions: Issues (Read & Write)
- Repository permissions: Contents (Read & Write)

### LLM Provider Configuration

#### Anthropic (Claude)

```bash
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-xxxxx

# Model selection (optional)
LLM_MODEL_PLANNER=claude-sonnet-4-20250514
LLM_MODEL_DEVELOPER=claude-sonnet-4-20250514
LLM_MODEL_QA=claude-sonnet-4-20250514
```

#### OpenAI

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-xxxxx

# Model selection (optional)
LLM_MODEL_PLANNER=gpt-4-turbo
LLM_MODEL_DEVELOPER=gpt-4-turbo
```

### Resource Configuration

```bash
# Concurrency
ORCHESTRATOR_MAX_CONCURRENT_AGENTS=3

# Timeouts
AGENT_TIMEOUT=1800  # 30 minutes

# Iterations
AGENT_MAX_ITERATIONS=5
```

## Verification

### Check System Health

```bash
make health
```

Expected output:
```
Docker:
  ✓ Docker daemon is running
  ✓ Orchestrator container is running
Configuration:
  ✓ .env file exists
  ✓ GITHUB_TOKEN is set
  ✓ LLM API key is set
GitHub API:
  ✓ GitHub API accessible
```

### View Logs

```bash
make logs
```

## Troubleshooting

### Docker Issues

**Problem**: Docker daemon not running
```bash
# Linux
sudo systemctl start docker

# macOS
open -a Docker
```

**Problem**: Permission denied
```bash
sudo usermod -aG docker $USER
# Log out and back in
```

### GitHub Issues

**Problem**: API rate limit
- Wait for rate limit reset
- Use GitHub App instead of PAT for higher limits

**Problem**: 401 Unauthorized
- Verify token is correct
- Check token permissions
- Ensure token hasn't expired

### LLM Issues

**Problem**: API key invalid
- Verify key format
- Check provider is correct
- Ensure account has credits

## Next Steps

1. Review [Architecture Guide](ARCHITECTURE.md)
2. Understand [Agent Specifications](AGENTS.md)
3. Configure [Memory System](MEMORY_SYSTEM.md)
4. Read [Deployment Guide](DEPLOYMENT.md) for production
