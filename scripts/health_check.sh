#!/bin/bash
# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - HEALTH CHECK SCRIPT
# =============================================================================
# Checks the health of all system components.
#
# Usage:
#   ./scripts/health_check.sh
# =============================================================================

set -e

# Load .env if exists
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

pass() { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; }

echo ""
echo "=========================================="
echo "  AI Agent System Health Check"
echo "=========================================="
echo ""

ERRORS=0
WARNINGS=0

# -----------------------------------------------------------------------------
# DOCKER
# -----------------------------------------------------------------------------
echo "Docker:"
if docker info > /dev/null 2>&1; then
    pass "Docker daemon is running"
else
    fail "Docker daemon is not running"
    ERRORS=$((ERRORS + 1))
fi

# Check orchestrator container
if docker ps --format '{{.Names}}' | grep -q "orchestrator"; then
    pass "Orchestrator container is running"
else
    warn "Orchestrator container is not running"
    WARNINGS=$((WARNINGS + 1))
fi

# Check Docker network
if docker network ls --format '{{.Name}}' | grep -q "${DOCKER_NETWORK:-ai-agent-network}"; then
    pass "Docker network exists"
else
    warn "Docker network not found"
    WARNINGS=$((WARNINGS + 1))
fi

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
echo ""
echo "Configuration:"

if [ -f ".env" ]; then
    pass ".env file exists"
else
    fail ".env file not found"
    ERRORS=$((ERRORS + 1))
fi

if [ -n "$GITHUB_TOKEN" ]; then
    pass "GITHUB_TOKEN is set"
else
    fail "GITHUB_TOKEN is not set"
    ERRORS=$((ERRORS + 1))
fi

if [ -n "$GITHUB_REPO" ]; then
    pass "GITHUB_REPO is set"
else
    fail "GITHUB_REPO is not set"
    ERRORS=$((ERRORS + 1))
fi

if [ -n "$ANTHROPIC_API_KEY" ] || [ -n "$OPENAI_API_KEY" ]; then
    pass "LLM API key is set"
else
    fail "No LLM API key configured"
    ERRORS=$((ERRORS + 1))
fi

# -----------------------------------------------------------------------------
# FILE SYSTEM
# -----------------------------------------------------------------------------
echo ""
echo "File System:"

if [ -d "memory" ]; then
    pass "Memory directory exists"
else
    fail "Memory directory not found"
    ERRORS=$((ERRORS + 1))
fi

if [ -d "memory/features" ]; then
    pass "Features directory exists"
else
    warn "Features directory not found"
    WARNINGS=$((WARNINGS + 1))
fi

if [ -f "memory/PROJECT.md" ]; then
    pass "PROJECT.md exists"
else
    warn "PROJECT.md not found"
    WARNINGS=$((WARNINGS + 1))
fi

# -----------------------------------------------------------------------------
# GITHUB API
# -----------------------------------------------------------------------------
echo ""
echo "GitHub API:"

if [ -n "$GITHUB_TOKEN" ] && [ -n "$GITHUB_REPO" ]; then
    response=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "Authorization: Bearer $GITHUB_TOKEN" \
        "https://api.github.com/repos/$GITHUB_REPO")

    if [ "$response" = "200" ]; then
        pass "GitHub API accessible"
    else
        fail "GitHub API returned $response"
        ERRORS=$((ERRORS + 1))
    fi

    # Check rate limit
    rate_limit=$(curl -s -H "Authorization: Bearer $GITHUB_TOKEN" \
        "https://api.github.com/rate_limit" | grep -o '"remaining":[0-9]*' | head -1 | cut -d: -f2)

    if [ -n "$rate_limit" ]; then
        if [ "$rate_limit" -gt 100 ]; then
            pass "GitHub rate limit OK ($rate_limit remaining)"
        else
            warn "GitHub rate limit low ($rate_limit remaining)"
            WARNINGS=$((WARNINGS + 1))
        fi
    fi
else
    warn "Skipping GitHub API check (no credentials)"
    WARNINGS=$((WARNINGS + 1))
fi

# -----------------------------------------------------------------------------
# ORCHESTRATOR HEALTH
# -----------------------------------------------------------------------------
echo ""
echo "Orchestrator:"

ORCHESTRATOR_URL="http://localhost:${WEBHOOK_PORT:-8080}"

if curl -s -o /dev/null -w "%{http_code}" "$ORCHESTRATOR_URL/health" | grep -q "200"; then
    pass "Orchestrator health endpoint responding"
else
    warn "Orchestrator health endpoint not responding"
    WARNINGS=$((WARNINGS + 1))
fi

# -----------------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "  Summary"
echo "=========================================="
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}All checks passed!${NC}"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}$WARNINGS warnings${NC}"
    exit 0
else
    echo -e "${RED}$ERRORS errors, $WARNINGS warnings${NC}"
    exit 1
fi
