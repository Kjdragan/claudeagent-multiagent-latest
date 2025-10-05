#!/usr/bin/env python3
"""
Simplified test script for the monitoring system core functionality.

This script tests the monitoring components without external dependencies.
"""

import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Test basic logging functionality that doesn't require psutil
async def test_basic_logging():
    """Test basic logging functionality."""
    print("🔍 Testing Basic Logging Functionality...")

    try:
        # Test that we can import and use the agent logging
        from agent_logging import (
            EditorAgentLogger,
            ReportAgentLogger,
            ResearchAgentLogger,
            UICoordinatorLogger,
            create_agent_logger,
        )

        session_id = f"test_logging_session_{uuid.uuid4().hex[:8]}"

        # Test creating different agent loggers
        research_logger = ResearchAgentLogger(session_id)
        report_logger = ReportAgentLogger(session_id)
        editor_logger = EditorAgentLogger(session_id)
        ui_logger = UICoordinatorLogger(session_id)

        print("✅ All agent loggers created successfully")

        # Test logging some activities
        research_logger.log_search_initiation(
            query="test query",
            search_params={"limit": 10},
            topic="test topic",
            estimated_results=5
        )

        report_logger.log_section_generation_start(
            section_name="Test Section",
            section_type="test",
            research_sources_count=3,
            target_word_count=500
        )

        editor_logger.log_review_initiation(
            document_title="Test Document",
            document_type="test",
            word_count=250,
            review_focus_areas=["content", "structure"]
        )

        ui_logger.log_workflow_initiation(
            workflow_id="test_workflow",
            user_request="Test request",
            workflow_type="test",
            estimated_stages=["test"],
            priority_level="normal"
        )

        print("✅ Agent activities logged successfully")

        # Get summaries
        research_summary = research_logger.get_research_summary()
        report_summary = report_logger.get_report_summary()
        editor_summary = editor_logger.get_editor_summary()
        ui_summary = ui_logger.get_coordinator_summary()

        print(f"✅ Research summary: {research_summary.get('research_metrics', {}).get('total_searches', 0)} searches")
        print(f"✅ Report summary: {report_summary.get('report_metrics', {}).get('total_sections_generated', 0)} sections")
        print(f"✅ Editor summary: {editor_summary.get('editor_metrics', {}).get('total_reviews_completed', 0)} reviews")
        print(f"✅ UI summary: {ui_summary.get('coordinator_metrics', {}).get('total_workflows_managed', 0)} workflows")

        # Test factory function
        factory_logger = create_agent_logger("research_agent", session_id)
        print(f"✅ Factory logger created: {type(factory_logger).__name__}")

        return True

    except Exception as e:
        print(f"❌ Basic logging test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_structured_logger():
    """Test the structured logger functionality."""
    print("\n📝 Testing StructuredLogger...")

    try:
        from agent_logging.structured_logger import StructuredLogger

        # Create a test structured logger
        logger = StructuredLogger(
            name="test_logger",
            log_dir=Path("test_structured_logs")
        )

        # Test logging different levels
        logger.info("Test info message", test_data={"key": "value"})
        logger.warning("Test warning message", warning_level="low")
        logger.error("Test error message", error_code="TEST_ERROR")

        print("✅ StructuredLogger messages logged successfully")

        # Test creating a session context
        session_id = str(uuid.uuid4())
        logger.info("Session started", session_id=session_id, action="start")

        # Test performance logging
        start_time = time.time()
        await asyncio.sleep(0.1)  # Simulate some work
        duration = time.time() - start_time

        logger.info("Performance test",
                   session_id=session_id,
                   action="performance_test",
                   duration_seconds=duration,
                   performance_type="test_operation")

        print(f"✅ Performance logging successful: {duration:.3f}s")

        # Test cleanup
        if hasattr(logger, 'cleanup'):
            logger.cleanup()

        return True

    except Exception as e:
        print(f"❌ StructuredLogger test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_agent_logger_factory():
    """Test the agent logger factory functionality."""
    print("\n🏭 Testing Agent Logger Factory...")

    try:
        from agent_logging import create_agent_logger

        session_id = f"factory_test_session_{uuid.uuid4().hex[:8]}"

        # Test creating different types of loggers
        agent_types = [
            "research_agent",
            "report_agent",
            "editor_agent",
            "ui_coordinator"
        ]

        created_loggers = {}
        for agent_type in agent_types:
            logger = create_agent_logger(agent_type, session_id)
            created_loggers[agent_type] = logger
            print(f"✅ Created {agent_type} logger: {type(logger).__name__}")

        # Test invalid agent type (should fallback to generic)
        invalid_logger = create_agent_logger("invalid_agent_type", session_id)
        created_loggers["invalid_agent"] = invalid_logger
        print(f"✅ Invalid agent type fallback: {type(invalid_logger).__name__}")

        # Test that all loggers have the expected methods
        expected_methods = [
            'log_activity',
            'get_session_summary',
            'export_session_data'
        ]

        for agent_type, logger in created_loggers.items():
            for method in expected_methods:
                if hasattr(logger, method):
                    print(f"✅ {agent_type} has {method} method")
                else:
                    print(f"❌ {agent_type} missing {method} method")
                    return False

        return True

    except Exception as e:
        print(f"❌ Agent logger factory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_session_management():
    """Test session management functionality."""
    print("\n🔄 Testing Session Management...")

    try:
        from agent_logging import ResearchAgentLogger

        session_id = f"session_test_{uuid.uuid4().hex[:8]}"
        logger = ResearchAgentLogger(session_id)

        # Test session start logging
        session_data = {
            "user_request": "Test session management",
            "research_depth": "standard",
            "start_time": datetime.now().isoformat()
        }

        logger.log_session_start(session_data)
        print("✅ Session start logged")

        # Test some activities
        logger.log_search_initiation(
            query="session test query",
            search_params={},
            topic="session testing",
            estimated_results=3
        )

        logger.log_source_analysis(
            source_url="https://example.com/test",
            source_title="Test Source",
            credibility_score=0.9,
            content_relevance=0.85,
            content_type="test",
            extraction_method="manual"
        )

        print("✅ Session activities logged")

        # Test session end logging
        session_summary = {
            "total_searches": 1,
            "total_sources_analyzed": 1,
            "session_duration_seconds": 5,
            "status": "completed"
        }

        logger.log_session_end(session_summary)
        print("✅ Session end logged")

        # Test getting session summary
        summary = logger.get_session_summary()
        print(f"✅ Session summary: {summary.get('total_activities', 0)} activities")

        # Test session data export
        export_path = logger.export_session_data()
        if Path(export_path).exists():
            print(f"✅ Session data exported to: {export_path}")

            # Verify exported data
            with open(export_path) as f:
                exported_data = json.load(f)

            if exported_data.get('session_id') == session_id:
                print("✅ Exported session data verification successful")
            else:
                print("❌ Exported session data verification failed")
                return False

        return True

    except Exception as e:
        print(f"❌ Session management test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_error_handling():
    """Test error handling in the logging system."""
    print("\n⚠️  Testing Error Handling...")

    try:
        from agent_logging import ResearchAgentLogger

        session_id = f"error_test_{uuid.uuid4().hex[:8]}"
        logger = ResearchAgentLogger(session_id)

        # Test logging activities with error information
        logger.log_search_initiation(
            query="error test query",
            search_params={"invalid_param": "test"},
            topic="error testing",
            estimated_results=0
        )

        # Log an error activity
        logger.log_activity(
            agent_name="research_agent",
            activity_type="error",
            stage="search_execution",
            error="Search API timeout",
            metadata={"error_code": "TIMEOUT", "retry_count": 3}
        )

        print("✅ Error activity logged successfully")

        # Test logging activities with missing/invalid data
        try:
            logger.log_search_initiation(
                query="",  # Empty query
                search_params=None,
                topic="",
                estimated_results=-1  # Invalid negative number
            )
            print("✅ Handled invalid data gracefully")
        except Exception as e:
            print(f"⚠️  Expected error with invalid data: {e}")

        # Test getting summaries when no data exists
        empty_logger = ResearchAgentLogger(f"empty_session_{uuid.uuid4().hex[:8]}")
        empty_summary = empty_logger.get_research_summary()

        if empty_summary:
            print("✅ Empty session summary generated")
        else:
            print("❌ Empty session summary failed")
            return False

        return True

    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_simple_monitoring_tests():
    """Run simplified monitoring system tests."""
    print("🚀 Starting Simplified Monitoring System Tests")
    print("=" * 60)

    test_results = []

    # Run individual tests
    tests = [
        ("Basic Logging", test_basic_logging),
        ("Structured Logger", test_structured_logger),
        ("Agent Logger Factory", test_agent_logger_factory),
        ("Session Management", test_session_management),
        ("Error Handling", test_error_handling)
    ]

    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        try:
            result = await test_func()
            test_results.append((test_name, result))
            if result:
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            print(f"❌ {test_name} test FAILED with exception: {e}")
            test_results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("🎯 SIMPLIFIED MONITORING TEST SUMMARY")
    print("=" * 60)

    passed_count = sum(1 for _, result in test_results if result)
    total_count = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")

    print(f"\n📊 Overall Result: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("🎉 ALL SIMPLIFIED MONITORING TESTS PASSED")
        success = True
    else:
        print("❌ SOME SIMPLIFIED MONITORING TESTS FAILED")
        success = False

    print("=" * 60)

    return success


def main():
    """Main function to run simplified monitoring tests."""
    try:
        success = asyncio.run(run_simple_monitoring_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
