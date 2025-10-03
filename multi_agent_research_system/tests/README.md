# Testing Suite for Multi-Agent Research System

## Test Organization

### 📁 Directory Structure

```
tests/
├── README.md                 # This file
├── conftest.py              # pytest configuration and fixtures
├── requirements.txt         # Test-specific dependencies
├── unit/                    # Unit tests for individual components
│   ├── __init__.py
│   ├── test_agents.py       # Test agent definitions
│   ├── test_tools.py        # Test individual tool functions
│   └── test_orchestrator.py # Test orchestrator methods
├── integration/             # Integration tests for component interaction
│   ├── __init__.py
│   ├── test_workflow.py     # Test agent coordination
│   └── test_session_mgmt.py # Test session management
├── functional/              # Functional tests with real LLM calls
│   ├── __init__.py
│   ├── test_research_workflow.py  # Complete research workflow
│   └── test_ui_integration.py     # UI + backend integration
├── utils/                   # Test utilities and helpers
│   ├── __init__.py
│   ├── mock_sdk.py          # Mock SDK for development
│   ├── test_helpers.py      # Common test functions
│   └── assertions.py        # Custom assertions
└── fixtures/                # Test data and configurations
    ├── sample_requests.json # Example research requests
    └── expected_outputs/    # Expected results for comparison
```

## Test Types

### 🔬 Unit Tests
- Test individual components in isolation
- Fast execution, no external dependencies
- Mock external services and SDK calls

### 🔗 Integration Tests
- Test component interactions
- Test workflow orchestration
- Mock LLM responses but test real logic

### ⚡ Functional Tests
- **Real LLM interactions** using actual Claude Agent SDK
- Test complete workflows end-to-end
- Validate actual research outputs
- **These tests require API credentials and real agents**

### 🛠️ Utility Tests
- Test helper functions and utilities
- Validate mock implementations
- Test custom assertions

## Running Tests

### All Tests
```bash
pytest tests/
```

### By Type
```bash
# Unit tests only (fast)
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Functional tests (requires API credentials)
pytest tests/functional/

# Skip tests that require real API
pytest tests/ -k "not functional"
```

### Development vs Production

**Development Mode:**
- Run unit and integration tests (no API calls)
- Fast feedback during development

**Production Validation:**
- Run full test suite including functional tests
- Requires Claude API credentials
- Tests real agent behavior and outputs

## Test Markers

- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.functional`: Functional tests (requires API)
- `@pytest.mark.slow`: Long-running tests

## Environment Setup

### For Unit/Integration Tests
```bash
pip install pytest pytest-asyncio pytest-mock
```

### For Functional Tests
```bash
pip install -r tests/requirements.txt
export ANTHROPIC_API_KEY="your-api-key"
```

## Test Data

All test data is stored in `tests/fixtures/`:
- `sample_requests.json`: Example research requests
- `expected_outputs/`: Expected outputs for validation

## Real Functional Testing

The functional tests in `tests/functional/` are designed to:
1. **Use real Claude models** via the Agent SDK
2. **Execute actual research workflows** with real web searches
3. **Generate real reports** using the multi-agent system
4. **Validate actual outputs** against expected quality standards

These tests ensure the system works with real LLM interactions, not mocks.