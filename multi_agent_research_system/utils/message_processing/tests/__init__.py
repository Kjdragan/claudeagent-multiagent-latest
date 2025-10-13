"""
Test Suite for Message Processing System

This package contains comprehensive tests for all components of the
message processing system, including unit tests, integration tests,
and performance benchmarks.

Test Coverage:
- Core message types and processing
- Rich display formatting
- Content enhancement and quality analysis
- Message routing and caching
- Serialization and persistence
- Integration with orchestrator
- Performance monitoring
- End-to-end workflows

Running Tests:
    python -m pytest tests/
    python -m pytest tests/test_message_processing.py -v
    python tests/test_message_processing.py
"""

from .test_message_processing import *

__version__ = "2.3.0"
__author__ = "Multi-Agent Research System"