#!/bin/bash
# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - AGENT ENTRYPOINT
# =============================================================================
# This script is the entrypoint for all agent containers.
# It reads the AGENT_TYPE environment variable and runs the appropriate agent.
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
AGENT_TYPE="${AGENT_TYPE:-developer}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# -----------------------------------------------------------------------------
# VALIDATION
# -----------------------------------------------------------------------------

# Validate required environment variables
required_vars=(
    "AGENT_TYPE"
    "PROJECT_ID"
    "ISSUE_NUMBER"
    "GITHUB_TOKEN"
    "GITHUB_REPO"
)

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "ERROR: Required environment variable $var is not set"
        exit 1
    fi
done

# Validate agent type
valid_types=("planner" "developer" "qa" "reviewer" "doc")
if [[ ! " ${valid_types[*]} " =~ " ${AGENT_TYPE} " ]]; then
    echo "ERROR: Invalid AGENT_TYPE: $AGENT_TYPE"
    echo "Valid types: ${valid_types[*]}"
    exit 1
fi

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------

echo "========================================"
echo "AI Agent Development System"
echo "========================================"
echo "Agent Type: $AGENT_TYPE"
echo "Project ID: $PROJECT_ID"
echo "Issue Number: $ISSUE_NUMBER"
echo "Iteration: ${ITERATION:-1}/${MAX_ITERATIONS:-5}"
echo "Started at: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "========================================"

# Ensure output directory exists and is writable
mkdir -p "$OUTPUT_PATH/logs" "$OUTPUT_PATH/artifacts"

# Log file for this execution
LOG_FILE="$OUTPUT_PATH/logs/execution.log"

# Function to log messages
log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

# -----------------------------------------------------------------------------
# RUN AGENT
# -----------------------------------------------------------------------------

log "INFO" "Starting $AGENT_TYPE agent for issue #$ISSUE_NUMBER"

# Run the appropriate agent module
# The agent code reads configuration from environment and mounted volumes
python -m agents.${AGENT_TYPE}.${AGENT_TYPE}_agent \
    2>&1 | tee -a "$LOG_FILE"

# Capture exit code
EXIT_CODE=${PIPESTATUS[0]}

# -----------------------------------------------------------------------------
# CLEANUP
# -----------------------------------------------------------------------------

log "INFO" "Agent completed with exit code: $EXIT_CODE"
echo "Completed at: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"

# Exit with agent's exit code
exit $EXIT_CODE
