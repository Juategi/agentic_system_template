#!/bin/bash
# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - GITHUB SETUP SCRIPT
# =============================================================================
# This script configures GitHub repository with required labels.
#
# Usage:
#   ./scripts/setup_github.sh
#
# Requires:
#   - GITHUB_TOKEN environment variable
#   - GITHUB_REPO environment variable (owner/repo)
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

info() { echo -e "[INFO] $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Validate environment
if [ -z "$GITHUB_TOKEN" ]; then
    error "GITHUB_TOKEN environment variable is not set"
fi

if [ -z "$GITHUB_REPO" ]; then
    error "GITHUB_REPO environment variable is not set"
fi

API_URL="https://api.github.com/repos/$GITHUB_REPO/labels"
AUTH_HEADER="Authorization: Bearer $GITHUB_TOKEN"

info "Setting up GitHub labels for $GITHUB_REPO"

# Function to create or update a label
create_label() {
    local name="$1"
    local color="$2"
    local description="$3"

    # Check if label exists
    response=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "$AUTH_HEADER" \
        "$API_URL/$(echo $name | sed 's/ /%20/g')")

    if [ "$response" = "200" ]; then
        # Update existing label
        curl -s -X PATCH \
            -H "$AUTH_HEADER" \
            -H "Content-Type: application/json" \
            "$API_URL/$(echo $name | sed 's/ /%20/g')" \
            -d "{\"color\":\"$color\",\"description\":\"$description\"}" > /dev/null
        info "Updated label: $name"
    else
        # Create new label
        curl -s -X POST \
            -H "$AUTH_HEADER" \
            -H "Content-Type: application/json" \
            "$API_URL" \
            -d "{\"name\":\"$name\",\"color\":\"$color\",\"description\":\"$description\"}" > /dev/null
        info "Created label: $name"
    fi
}

echo ""
info "Creating workflow state labels..."

# Workflow state labels
create_label "READY" "0E8A16" "Task is ready for agent assignment"
create_label "IN_PROGRESS" "FBCA04" "Agent is currently working on this task"
create_label "QA" "1D76DB" "Task is undergoing quality assurance"
create_label "QA_FAILED" "E99695" "QA failed, returning to development"
create_label "REVIEW" "5319E7" "Task is under code review"
create_label "BLOCKED" "D93F0B" "Task requires human intervention"
create_label "DONE" "0E8A16" "Task completed successfully"

echo ""
info "Creating issue type labels..."

# Issue type labels
create_label "feature" "A2EEEF" "A feature to be decomposed into tasks"
create_label "task" "D4C5F9" "An actionable development task"
create_label "bug" "D73A4A" "A bug to be fixed"
create_label "subtask" "BFD4F2" "A subtask of a larger feature"

echo ""
info "Creating priority labels..."

# Priority labels
create_label "priority:critical" "B60205" "Critical priority - immediate attention"
create_label "priority:high" "D93F0B" "High priority"
create_label "priority:medium" "FBCA04" "Medium priority"
create_label "priority:low" "0E8A16" "Low priority"

echo ""
info "Creating agent labels..."

# Agent labels
create_label "agent:planner" "C5DEF5" "Assigned to Planner Agent"
create_label "agent:developer" "C5DEF5" "Assigned to Developer Agent"
create_label "agent:qa" "C5DEF5" "Assigned to QA Agent"
create_label "agent:reviewer" "C5DEF5" "Assigned to Reviewer Agent"

echo ""
success "GitHub labels configured successfully!"
echo ""
echo "Your repository is ready for AI agent development."
echo "Create an issue with the 'feature' and 'READY' labels to get started."
