#!/bin/bash
# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - PROJECT INITIALIZATION SCRIPT
# =============================================================================
# This script initializes a new project from the template.
# Run this after cloning the template repository.
#
# Usage:
#   ./scripts/init_project.sh [project_name]
#
# What it does:
# 1. Prompts for project configuration
# 2. Creates .env file from template
# 3. Initializes empty memory structure
# 4. Sets up GitHub labels
# 5. Builds Docker images
# 6. Validates configuration
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# -----------------------------------------------------------------------------
# BANNER
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "  AI Agent Development System"
echo "  Project Initialization"
echo "=========================================="
echo ""

# -----------------------------------------------------------------------------
# CHECK PREREQUISITES
# -----------------------------------------------------------------------------
info "Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    error "Docker is not installed. Please install Docker first."
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    error "Docker Compose is not installed. Please install Docker Compose first."
fi

# Check git
if ! command -v git &> /dev/null; then
    error "Git is not installed. Please install Git first."
fi

success "All prerequisites met"

# -----------------------------------------------------------------------------
# GET PROJECT CONFIGURATION
# -----------------------------------------------------------------------------
echo ""
info "Project Configuration"
echo "---------------------"

# Project name
if [ -n "$1" ]; then
    PROJECT_NAME="$1"
else
    read -p "Project name: " PROJECT_NAME
fi
PROJECT_NAME=${PROJECT_NAME:-"my-ai-project"}

# Project ID (slug)
PROJECT_ID=$(echo "$PROJECT_NAME" | tr '[:upper:]' '[:lower:]' | tr ' ' '-' | tr -cd '[:alnum:]-')
info "Project ID: $PROJECT_ID"

# GitHub repository
read -p "GitHub repository (owner/repo): " GITHUB_REPO
if [ -z "$GITHUB_REPO" ]; then
    warning "No GitHub repository specified. You'll need to set this manually."
fi

# GitHub token
read -p "GitHub token (leave empty to set later): " GITHUB_TOKEN
if [ -z "$GITHUB_TOKEN" ]; then
    warning "GitHub token not provided. Set GITHUB_TOKEN in .env before starting."
fi

# LLM Provider
echo ""
echo "Select LLM Provider:"
echo "  1) Anthropic (Claude)"
echo "  2) OpenAI"
read -p "Choice [1]: " LLM_CHOICE
case "$LLM_CHOICE" in
    2) LLM_PROVIDER="openai" ;;
    *) LLM_PROVIDER="anthropic" ;;
esac

# LLM API Key
if [ "$LLM_PROVIDER" = "anthropic" ]; then
    read -p "Anthropic API Key (leave empty to set later): " API_KEY
    ANTHROPIC_API_KEY="$API_KEY"
    OPENAI_API_KEY=""
else
    read -p "OpenAI API Key (leave empty to set later): " API_KEY
    OPENAI_API_KEY="$API_KEY"
    ANTHROPIC_API_KEY=""
fi

# -----------------------------------------------------------------------------
# CREATE .ENV FILE
# -----------------------------------------------------------------------------
echo ""
info "Creating .env file..."

if [ -f ".env" ]; then
    warning ".env file already exists"
    read -p "Overwrite? (y/N): " OVERWRITE
    if [ "$OVERWRITE" != "y" ] && [ "$OVERWRITE" != "Y" ]; then
        info "Keeping existing .env file"
    else
        cp .env ".env.backup.$(date +%Y%m%d%H%M%S)"
        info "Backed up existing .env"
    fi
fi

# Generate .env from template
cat > .env << EOF
# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - ENVIRONMENT CONFIGURATION
# Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
# =============================================================================

# Project
PROJECT_ID=$PROJECT_ID
PROJECT_NAME="$PROJECT_NAME"
ENVIRONMENT=development

# GitHub
GITHUB_TOKEN=$GITHUB_TOKEN
GITHUB_REPO=$GITHUB_REPO

# LLM
LLM_PROVIDER=$LLM_PROVIDER
ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY
OPENAI_API_KEY=$OPENAI_API_KEY

# Orchestrator
ORCHESTRATOR_POLL_INTERVAL=30
ORCHESTRATOR_MAX_CONCURRENT_AGENTS=3

# Agent
AGENT_MAX_ITERATIONS=5
AGENT_TIMEOUT=1800

# Paths
LOCAL_MEMORY_PATH=./memory
LOCAL_REPO_PATH=./repo
LOCAL_OUTPUT_PATH=./output
LOG_PATH=./logs

# Logging
LOG_LEVEL=INFO
ENABLE_METRICS=true
ENABLE_TRACING=true

# Docker
DOCKER_AGENT_IMAGE=ai-agent:latest
DOCKER_NETWORK=ai-agent-network
EOF

success ".env file created"

# -----------------------------------------------------------------------------
# INITIALIZE MEMORY STRUCTURE
# -----------------------------------------------------------------------------
echo ""
info "Initializing memory structure..."

# Create memory directories
mkdir -p memory/features
mkdir -p logs
mkdir -p output
mkdir -p repo

# Update PROJECT.md with project name
sed -i.bak "s/\[PROJECT_NAME\]/$PROJECT_NAME/g" memory/PROJECT.md 2>/dev/null || \
    sed -i '' "s/\[PROJECT_NAME\]/$PROJECT_NAME/g" memory/PROJECT.md

# Create .gitkeep files
touch memory/features/.gitkeep
touch logs/.gitkeep
touch output/.gitkeep

success "Memory structure initialized"

# -----------------------------------------------------------------------------
# SETUP GITHUB LABELS (if token provided)
# -----------------------------------------------------------------------------
if [ -n "$GITHUB_TOKEN" ] && [ -n "$GITHUB_REPO" ]; then
    echo ""
    info "Setting up GitHub labels..."
    ./scripts/setup_github.sh 2>/dev/null || warning "Could not setup GitHub labels. Run ./scripts/setup_github.sh manually."
else
    warning "Skipping GitHub label setup (no credentials)"
fi

# -----------------------------------------------------------------------------
# BUILD DOCKER IMAGES
# -----------------------------------------------------------------------------
echo ""
info "Building Docker images..."

read -p "Build Docker images now? (Y/n): " BUILD_DOCKER
if [ "$BUILD_DOCKER" != "n" ] && [ "$BUILD_DOCKER" != "N" ]; then
    docker-compose build
    success "Docker images built"
else
    info "Skipping Docker build. Run 'make build' when ready."
fi

# -----------------------------------------------------------------------------
# FINAL VALIDATION
# -----------------------------------------------------------------------------
echo ""
info "Validating configuration..."

ERRORS=0

if [ -z "$GITHUB_TOKEN" ]; then
    warning "GITHUB_TOKEN not set in .env"
    ERRORS=$((ERRORS + 1))
fi

if [ -z "$GITHUB_REPO" ]; then
    warning "GITHUB_REPO not set in .env"
    ERRORS=$((ERRORS + 1))
fi

if [ -z "$ANTHROPIC_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
    warning "No LLM API key configured in .env"
    ERRORS=$((ERRORS + 1))
fi

if [ $ERRORS -gt 0 ]; then
    warning "$ERRORS configuration items need attention"
else
    success "Configuration validated"
fi

# -----------------------------------------------------------------------------
# COMPLETION
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "  Initialization Complete!"
echo "=========================================="
echo ""
echo "Project: $PROJECT_NAME"
echo "ID: $PROJECT_ID"
echo ""
echo "Next steps:"
echo "  1. Review and complete .env configuration"
echo "  2. Update memory/PROJECT.md with project details"
echo "  3. Run 'make start' to start the orchestrator"
echo "  4. Create a GitHub Issue with 'feature' label"
echo ""
echo "Useful commands:"
echo "  make build    - Build Docker images"
echo "  make start    - Start orchestrator"
echo "  make logs     - View orchestrator logs"
echo "  make health   - Check system health"
echo ""
success "Happy coding with AI agents!"
