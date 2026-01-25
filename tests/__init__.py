# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - TEST PACKAGE
# =============================================================================
"""
Test Package

This package contains tests for the AI Agent Development System.

Test Structure:
    tests/
    ├── __init__.py              # This file
    ├── test_orchestrator.py     # Orchestrator tests
    ├── test_agents.py           # Agent tests
    ├── test_github_integration.py # GitHub integration tests
    └── fixtures/                 # Test fixtures
        ├── sample_issues.json
        └── sample_states.json

Running Tests:
    # Run all tests
    make test

    # Run specific test file
    pytest tests/test_orchestrator.py -v

    # Run with coverage
    pytest tests/ --cov=orchestrator --cov=agents

Test Categories:
    - Unit tests: Test individual components in isolation
    - Integration tests: Test component interactions
    - End-to-end tests: Test full workflows with mocks
"""
