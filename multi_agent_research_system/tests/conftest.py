"""pytest configuration and fixtures for the test suite."""

import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

# Add parent directory to path for imports
sys.path.append('..')

# Test fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_research_request() -> dict[str, Any]:
    """Sample research request for testing."""
    return {
        "topic": "The Impact of Remote Work on Employee Productivity",
        "depth": "Standard Research",
        "audience": "Business",
        "format": "Business Brief",
        "requirements": "Focus on recent studies from 2020-2024, include statistics and company examples",
        "timeline": "Within 24 hours"
    }


@pytest.fixture
def sample_research_request_simple() -> dict[str, Any]:
    """Simple research request for quick testing."""
    return {
        "topic": "Benefits of Renewable Energy",
        "depth": "Quick Overview",
        "audience": "General Public",
        "format": "Standard Report",
        "requirements": "Basic overview with key benefits",
        "timeline": "ASAP"
    }


@pytest.fixture
def mock_session_data() -> dict[str, Any]:
    """Mock session data for testing."""
    return {
        "session_id": "test-session-123",
        "topic": "Test Topic",
        "user_requirements": {"depth": "Quick Overview"},
        "status": "completed",
        "created_at": "2024-01-01T12:00:00",
        "current_stage": "finalization",
        "workflow_history": [
            {"stage": "research", "completed_at": "2024-01-01T12:30:00", "results_count": 5},
            {"stage": "report_generation", "completed_at": "2024-01-01T13:00:00", "results_count": 1},
            {"stage": "editorial_review", "completed_at": "2024-01-01T13:30:00", "results_count": 3}
        ],
        "final_report": None
    }


@pytest.fixture
def researchmaterials_dir(temp_dir):
    """Create researchmaterials directory structure."""
    research_dir = temp_dir / "researchmaterials" / "sessions"
    research_dir.mkdir(parents=True)
    return research_dir


# Mock fixtures for development testing
@pytest.fixture
def mock_claude_sdk():
    """Mock Claude SDK for development testing."""
    import sys
    from unittest.mock import MagicMock

    # Mock the SDK components
    mock_client = MagicMock()
    mock_agent_options = MagicMock()
    mock_mcp_server = MagicMock()
    mock_agent_definition = MagicMock()

    # Create mock module
    mock_sdk = MagicMock()
    mock_sdk.ClaudeSDKClient = MagicMock(return_value=mock_client)
    mock_sdk.ClaudeAgentOptions = MagicMock(return_value=mock_agent_options)
    mock_sdk.create_sdk_mcp_server = MagicMock(return_value=mock_mcp_server)
    mock_sdk.AgentDefinition = MagicMock(return_value=mock_agent_definition)

    # Patch the imports
    sys.modules['claude_agent_sdk'] = mock_sdk

    return mock_sdk


# Real SDK fixture for functional testing
@pytest.fixture
def real_sdk_available():
    """Check if real Claude SDK is available for functional tests."""
    try:
        import claude_agent_sdk
        return True
    except ImportError:
        pytest.skip("Claude Agent SDK not available - install with pip install claude-agent-sdk")


@pytest.fixture
def api_credentials_available():
    """Check if API credentials are available for functional tests."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set - required for functional tests")
    return True


# Test data fixtures
@pytest.fixture
def sample_tool_args():
    """Sample arguments for tool testing."""
    return {
        "conduct_research": {
            "topic": "Artificial Intelligence in Healthcare",
            "depth": "medium",
            "focus_areas": ["diagnostics", "treatment", "ethics"],
            "max_sources": 5
        },
        "generate_report": {
            "research_data": {
                "topic": "AI in Healthcare",
                "findings": [
                    {"fact": "AI helps in medical diagnosis", "sources": ["source1"]},
                    {"fact": "AI improves treatment planning", "sources": ["source2"]}
                ]
            },
            "format": "markdown",
            "audience": "technical",
            "sections": ["summary", "findings", "conclusions"]
        },
        "review_report": {
            "report": {
                "title": "AI in Healthcare Report",
                "sections": {"summary": "Test content"}
            },
            "review_criteria": ["accuracy", "clarity", "completeness"]
        }
    }


# Async test helpers
@pytest.fixture
async def async_test_timeout():
    """Timeout for async tests to prevent hanging."""
    return 30.0  # 30 seconds


# Custom markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests (fast, no external deps)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (component interaction)"
    )
    config.addinivalue_line(
        "markers", "functional: marks tests as functional tests (requires real API)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow-running (may take > 30 seconds)"
    )
