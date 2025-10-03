"""Test utilities and helper functions for the multi-agent research system."""

from .assertions import (
    assert_valid_report_structure,
    assert_valid_research_results,
    assert_valid_session_data,
)
from .mock_sdk import MockAgentDefinition, MockClaudeSDKClient, create_mock_orchestrator
from .test_helpers import (
    async_test_wrapper,
    compare_tool_results,
    create_mock_session,
    validate_research_output,
)

__all__ = [
    # Test helpers
    "create_mock_session",
    "validate_research_output",
    "compare_tool_results",
    "async_test_wrapper",

    # Mock SDK
    "MockClaudeSDKClient",
    "MockAgentDefinition",
    "create_mock_orchestrator",

    # Assertions
    "assert_valid_session_data",
    "assert_valid_research_results",
    "assert_valid_report_structure"
]
