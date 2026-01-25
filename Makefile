# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - MAKEFILE
# =============================================================================
# Common commands for managing the AI agent development system
# Usage: make <target>
# =============================================================================

.PHONY: help init build start stop restart logs clean test lint health

# Default target
.DEFAULT_GOAL := help

# =============================================================================
# HELP
# =============================================================================
help: ## Show this help message
	@echo "AI Agent Development System - Available Commands"
	@echo "================================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# INITIALIZATION
# =============================================================================
init: ## Initialize a new project from template
	@echo "Initializing AI Agent Development System..."
	@./scripts/init_project.sh

setup-env: ## Create .env file from template
	@if [ ! -f .env ]; then \
		cp .env.template .env; \
		echo "Created .env file. Please edit it with your configuration."; \
	else \
		echo ".env file already exists. Skipping."; \
	fi

setup-github: ## Configure GitHub integration
	@./scripts/setup_github.sh

# =============================================================================
# DOCKER BUILD
# =============================================================================
build: ## Build all Docker images
	@echo "Building Docker images..."
	docker-compose build

build-orchestrator: ## Build only orchestrator image
	docker-compose build orchestrator

build-agent: ## Build only agent image
	docker-compose build agent-runner

rebuild: ## Rebuild all images without cache
	docker-compose build --no-cache

# =============================================================================
# RUN SERVICES
# =============================================================================
start: ## Start orchestrator (main service)
	@echo "Starting orchestrator..."
	docker-compose up -d orchestrator
	@echo "Orchestrator started. Use 'make logs' to view output."

start-all: ## Start all services including optional ones
	docker-compose --profile full up -d

start-dev: ## Start with development tools (webhook tunnel)
	docker-compose --profile dev up -d

start-monitoring: ## Start with monitoring stack
	docker-compose --profile monitoring up -d

stop: ## Stop all services
	docker-compose down

restart: ## Restart all services
	docker-compose restart

# =============================================================================
# LOGS AND DEBUGGING
# =============================================================================
logs: ## View orchestrator logs (follow mode)
	docker-compose logs -f orchestrator

logs-all: ## View all service logs
	docker-compose logs -f

logs-agent: ## View agent runner logs
	docker-compose logs -f agent-runner

# =============================================================================
# HEALTH AND STATUS
# =============================================================================
health: ## Check system health
	@./scripts/health_check.sh

status: ## Show status of all services
	docker-compose ps

# =============================================================================
# AGENT MANAGEMENT
# =============================================================================
run-planner: ## Run planner agent manually (requires ISSUE_NUMBER)
	docker-compose run --rm -e AGENT_TYPE=planner -e ISSUE_NUMBER=$(ISSUE_NUMBER) agent-runner

run-developer: ## Run developer agent manually (requires ISSUE_NUMBER)
	docker-compose run --rm -e AGENT_TYPE=developer -e ISSUE_NUMBER=$(ISSUE_NUMBER) agent-runner

run-qa: ## Run QA agent manually (requires ISSUE_NUMBER)
	docker-compose run --rm -e AGENT_TYPE=qa -e ISSUE_NUMBER=$(ISSUE_NUMBER) agent-runner

run-reviewer: ## Run reviewer agent manually (requires ISSUE_NUMBER)
	docker-compose run --rm -e AGENT_TYPE=reviewer -e ISSUE_NUMBER=$(ISSUE_NUMBER) agent-runner

run-doc: ## Run documentation agent manually (requires ISSUE_NUMBER)
	docker-compose run --rm -e AGENT_TYPE=doc -e ISSUE_NUMBER=$(ISSUE_NUMBER) agent-runner

# =============================================================================
# TESTING
# =============================================================================
test: ## Run all tests
	@echo "Running tests..."
	python -m pytest tests/ -v

test-orchestrator: ## Run orchestrator tests
	python -m pytest tests/test_orchestrator.py -v

test-agents: ## Run agent tests
	python -m pytest tests/test_agents.py -v

test-github: ## Run GitHub integration tests
	python -m pytest tests/test_github_integration.py -v

# =============================================================================
# CODE QUALITY
# =============================================================================
lint: ## Run linters
	@echo "Running linters..."
	ruff check .
	mypy orchestrator/ agents/

format: ## Format code
	@echo "Formatting code..."
	ruff format .
	isort .

# =============================================================================
# CLEANUP
# =============================================================================
clean: ## Clean up temporary files and containers
	@echo "Cleaning up..."
	docker-compose down -v --remove-orphans
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache 2>/dev/null || true

clean-logs: ## Clean log files
	rm -rf logs/*.log

clean-output: ## Clean agent output files
	rm -rf output/*

clean-state: ## Clean orchestrator state (WARNING: resets all progress)
	@echo "WARNING: This will reset all task progress. Are you sure? [y/N]"
	@read ans && [ $${ans:-N} = y ] && rm -f orchestrator_state.json || echo "Cancelled."

# =============================================================================
# DATABASE (if using PostgreSQL)
# =============================================================================
db-migrate: ## Run database migrations
	docker-compose exec orchestrator python -m alembic upgrade head

db-shell: ## Open database shell
	docker-compose exec postgres psql -U orchestrator -d orchestrator

# =============================================================================
# UTILITIES
# =============================================================================
shell-orchestrator: ## Open shell in orchestrator container
	docker-compose exec orchestrator /bin/bash

shell-agent: ## Open shell in agent container
	docker-compose run --rm agent-runner /bin/bash

# =============================================================================
# DEPLOYMENT
# =============================================================================
deploy-build: ## Build production images
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml build

deploy-push: ## Push images to registry
	docker-compose push

# =============================================================================
# DOCUMENTATION
# =============================================================================
docs-serve: ## Serve documentation locally
	@echo "Starting documentation server..."
	python -m http.server 8000 --directory docs/
