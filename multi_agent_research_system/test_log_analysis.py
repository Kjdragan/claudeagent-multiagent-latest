#!/usr/bin/env python3
"""
Test script for the comprehensive log analysis and reporting system.

This script tests the LogAggregator, LogSearchEngine, AnalyticsEngine,
ReportGenerator, and AuditTrailManager components.
"""

import asyncio
import json
import sys
import os
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from log_analysis import (
    LogAggregator,
    LogSearchEngine,
    AnalyticsEngine,
    ReportGenerator,
    AuditTrailManager
)


async def test_log_aggregator():
    """Test the LogAggregator functionality."""
    print("📊 Testing LogAggregator...")

    session_id = f"test_agg_session_{uuid.uuid4().hex[:8]}"
    aggregator = LogAggregator(
        session_id=session_id,
        aggregation_dir="test_log_aggregation",
        max_entries=1000,
        retention_days=1
    )

    try:
        # Test log source management
        from log_analysis.log_aggregator import LogSource
        test_source = LogSource(
            name="test_source",
            path=Path("test_logs"),
            pattern="*.log",
            format="plain",
            priority=1
        )

        aggregator.add_log_source(test_source)
        print("✅ Log source added successfully")

        # Test manual log entry creation
        from log_analysis.log_aggregator import LogEntry
        test_entry = LogEntry(
            timestamp=datetime.now(),
            level="INFO",
            source="test_source",
            session_id=session_id,
            agent_name="test_agent",
            activity_type="test_activity",
            message="Test log message",
            metadata={"test": True}
        )

        await aggregator._process_log_entry(test_entry)
        print("✅ Manual log entry processed successfully")

        # Test aggregation (without actual files)
        stats = aggregator.get_aggregation_stats()
        print(f"✅ Aggregation stats: {stats['total_entries']} entries")

        # Test entry filtering
        entries = aggregator.get_entries(limit=10)
        print(f"✅ Retrieved {len(entries)} entries")

        # Test export functionality
        export_path = aggregator.export_aggregated_logs(format='json')
        print(f"✅ Aggregated logs exported to: {export_path}")

        return aggregator

    except Exception as e:
        print(f"❌ LogAggregator test failed: {e}")
        raise

    finally:
        await aggregator.stop_aggregation()


async def test_log_search_engine():
    """Test the LogSearchEngine functionality."""
    print("\n🔍 Testing LogSearchEngine...")

    try:
        # Create test log entries
        from log_analysis.log_aggregator import LogEntry
        from log_analysis.log_search import SearchQuery, SearchOperator

        test_entries = [
            LogEntry(
                timestamp=datetime.now() - timedelta(minutes=i),
                level="INFO" if i % 2 == 0 else "ERROR",
                source=f"source_{i % 3}",
                session_id="test_session",
                agent_name=f"agent_{i % 2}",
                activity_type=f"activity_{i % 4}",
                message=f"Test message {i} with content",
                metadata={"execution_time": i * 0.1}
            )
            for i in range(20)
        ]

        search_engine = LogSearchEngine()
        search_engine.build_index(test_entries)
        print("✅ Search index built successfully")

        # Test basic text search
        results, stats = search_engine.search(
            test_entries,
            "content",
            limit=5
        )
        print(f"✅ Text search: {len(results)} results in {stats.execution_time_ms:.2f}ms")

        # Test field-based search
        field_query = SearchQuery(
            field="level",
            operator=SearchOperator.EQUALS,
            value="ERROR"
        )
        results, stats = search_engine.search(
            test_entries,
            field_query
        )
        print(f"✅ Field search: {len(results)} ERROR level entries")

        # Test complex query
        complex_query = [
            SearchQuery(
                field="agent_name",
                operator=SearchOperator.EQUALS,
                value="agent_0"
            ),
            SearchQuery(
                field="level",
                operator=SearchOperator.EQUALS,
                value="INFO"
            )
        ]
        results, stats = search_engine.search(
            test_entries,
            complex_query
        )
        print(f"✅ Complex query: {len(results)} matching entries")

        # Test search stats
        search_stats = search_engine.get_search_stats()
        print(f"✅ Search stats: {search_stats['indexed_fields']} indexed fields")

        return search_engine

    except Exception as e:
        print(f"❌ LogSearchEngine test failed: {e}")
        raise


async def test_analytics_engine():
    """Test the AnalyticsEngine functionality."""
    print("\n📈 Testing AnalyticsEngine...")

    try:
        session_id = f"test_analytics_session_{uuid.uuid4().hex[:8]}"
        analytics = AnalyticsEngine(
            session_id=session_id,
            analytics_dir="test_analytics"
        )

        # Create test log entries with performance data
        from log_analysis.log_aggregator import LogEntry
        test_entries = []

        for i in range(50):
            entry = LogEntry(
                timestamp=datetime.now() - timedelta(minutes=i),
                level="INFO" if i % 5 != 0 else "ERROR",
                source="test_system",
                session_id=session_id,
                agent_name=f"agent_{i % 3}",
                activity_type=f"process_data",
                message=f"Processing item {i}",
                metadata={
                    "execution_time": 0.5 + (i % 10) * 0.1,
                    "processed_items": i % 20 + 1,
                    "tool_name": f"tool_{i % 4}"
                }
            )
            test_entries.append(entry)

        # Test log analysis
        analysis = await analytics.analyze_logs(
            test_entries,
            ['performance', 'usage', 'errors', 'trends']
        )
        print("✅ Log analysis completed")

        # Check analysis results
        if 'performance' in analysis['analysis_results']:
            perf = analysis['analysis_results']['performance']
            if 'response_times' in perf:
                avg_time = perf['response_times']['avg']
                print(f"✅ Performance analysis: avg response time {avg_time:.2f}s")

        if 'errors' in analysis['analysis_results']:
            errors = analysis['analysis_results']['errors']
            print(f"✅ Error analysis: {errors['total_errors']} errors found")

        if 'insights' in analysis:
            insights = analysis['insights']
            print(f"✅ Generated {len(insights)} insights")

        # Test analytics summary
        summary = analytics.get_analytics_summary()
        print(f"✅ Analytics summary: {summary['total_metrics']} metrics")

        # Test export
        export_path = analytics.export_analytics_data()
        print(f"✅ Analytics data exported to: {export_path}")

        return analytics

    except Exception as e:
        print(f"❌ AnalyticsEngine test failed: {e}")
        raise


async def test_audit_trail():
    """Test the AuditTrailManager functionality."""
    print("\n🔒 Testing AuditTrailManager...")

    try:
        session_id = f"test_audit_session_{uuid.uuid4().hex[:8]}"
        audit = AuditTrailManager(
            session_id=session_id,
            audit_dir="test_audit",
            retention_days=30
        )

        # Test audit event logging
        from log_analysis.audit_trail import AuditEventType, ComplianceStandard

        event_id = audit.log_audit_event(
            event_type=AuditEventType.USER_ACTION,
            action="login_attempt",
            actor="test_user",
            resource="system",
            outcome="success",
            details={"ip_address": "192.168.1.1"},
            compliance_tags=["authentication"],
            data_classification="internal"
        )
        print(f"✅ Audit event logged: {event_id}")

        # Test different event types
        audit.log_audit_event(
            event_type=AuditEventType.DATA_ACCESS,
            action="read_sensitive_data",
            actor="test_agent",
            resource="customer_data",
            outcome="success",
            details={"record_count": 10},
            compliance_tags=["gdpr", "data_access"],
            data_classification="confidential"
        )

        audit.log_audit_event(
            event_type=AuditEventType.SECURITY_EVENT,
            action="failed_login",
            actor="unknown_user",
            resource="system",
            outcome="failure",
            details={"reason": "invalid_credentials"},
            compliance_tags=["security"],
            data_classification="internal"
        )

        print("✅ Multiple audit events logged")

        # Test search functionality
        user_events = audit.search_audit_trail(
            actor="test_user",
            limit=10
        )
        print(f"✅ Found {len(user_events)} events for test_user")

        security_events = audit.search_audit_trail(
            event_type=AuditEventType.SECURITY_EVENT,
            limit=10
        )
        print(f"✅ Found {len(security_events)} security events")

        # Test integrity verification
        integrity_result = audit.verify_integrity()
        print(f"✅ Integrity check: {integrity_result['verification_passed']} (verified {integrity_result['total_events_verified']} events)")

        # Test compliance report generation
        compliance_report = await audit.generate_compliance_report(
            standard=ComplianceStandard.GDPR,
            period_start=datetime.now() - timedelta(days=1),
            period_end=datetime.now()
        )
        print(f"✅ GDPR compliance report: {compliance_report.compliance_score:.1f}% compliance score")

        # Test audit summary
        summary = audit.get_audit_summary()
        print(f"✅ Audit summary: {summary['total_events']} total events")

        # Test export
        export_path = audit.export_audit_trail()
        print(f"✅ Audit trail exported to: {export_path}")

        return audit

    except Exception as e:
        print(f"❌ AuditTrailManager test failed: {e}")
        raise


async def test_report_generator():
    """Test the ReportGenerator functionality."""
    print("\n📄 Testing ReportGenerator...")

    try:
        session_id = f"test_report_session_{uuid.uuid4().hex[:8]}"
        generator = ReportGenerator(
            session_id=session_id,
            reports_dir="test_reports"
        )

        # Create mock data sources (simplified)
        from log_analysis.log_aggregator import LogEntry
        test_entries = [
            LogEntry(
                timestamp=datetime.now() - timedelta(hours=i),
                level="INFO",
                source="test_system",
                session_id=session_id,
                agent_name="test_agent",
                activity_type="test_activity",
                message=f"Test message {i}",
                metadata={"execution_time": i * 0.1}
            )
            for i in range(10)
        ]

        # Test daily summary report generation
        from log_analysis.report_generator import ReportType, ReportFormat

        daily_report = await generator.generate_report(
            report_type=ReportType.DAILY_SUMMARY,
            period_start=datetime.now() - timedelta(days=1),
            period_end=datetime.now(),
            formats=[ReportFormat.JSON, ReportFormat.HTML]
        )
        print(f"✅ Daily report generated: {daily_report.report_id}")

        # Test performance analysis report
        perf_report = await generator.generate_report(
            report_type=ReportType.PERFORMANCE_ANALYSIS,
            period_start=datetime.now() - timedelta(hours=6),
            period_end=datetime.now(),
            formats=[ReportFormat.JSON]
        )
        print(f"✅ Performance report generated: {perf_report.report_id}")

        # Test report structure
        print(f"✅ Report has {len(daily_report.sections)} sections")
        print(f"✅ Report generated in {len(daily_report.file_paths)} formats")

        # Test report summary
        summary = generator.get_report_summary()
        print(f"✅ Report generator summary: {summary['total_reports']} reports generated")

        return generator

    except Exception as e:
        print(f"❌ ReportGenerator test failed: {e}")
        raise


async def test_integration():
    """Test integration of all log analysis components."""
    print("\n🔗 Testing Integration...")

    try:
        session_id = f"test_integration_session_{uuid.uuid4().hex[:8]}"

        # Initialize all components
        aggregator = LogAggregator(
            session_id=session_id,
            aggregation_dir="test_integration_aggregation"
        )

        search_engine = LogSearchEngine()
        analytics = AnalyticsEngine(
            session_id=session_id,
            analytics_dir="test_integration_analytics"
        )

        audit = AuditTrailManager(
            session_id=session_id,
            audit_dir="test_integration_audit"
        )

        generator = ReportGenerator(
            session_id=session_id,
            reports_dir="test_integration_reports"
        )

        print("✅ All components initialized")

        # Create test data
        from log_analysis.log_aggregator import LogEntry
        from log_analysis.audit_trail import AuditEventType, ComplianceStandard
        from log_analysis.report_generator import ReportType, ReportFormat
        from log_analysis.log_search import SearchQuery, SearchOperator

        test_entries = []
        for i in range(30):
            entry = LogEntry(
                timestamp=datetime.now() - timedelta(minutes=i),
                level="INFO" if i % 4 != 0 else "ERROR",
                source="integration_test",
                session_id=session_id,
                agent_name=f"agent_{i % 2}",
                activity_type="integration_activity",
                message=f"Integration test message {i}",
                metadata={
                    "execution_time": 0.1 + (i % 5) * 0.05,
                    "test_id": i
                }
            )
            test_entries.append(entry)

        # Log audit events for integration
        audit.log_audit_event(
            event_type=AuditEventType.AGENT_HANDOFF,
            action="handoff_test",
            actor="orchestrator",
            resource="workflow_123",
            outcome="success",
            details={"test_phase": "integration"}
        )

        print("✅ Test data created")

        # Test integrated workflow
        # 1. Search for specific entries
        search_engine.build_index(test_entries)
        error_query = SearchQuery(
            field="level",
            operator=SearchOperator.EQUALS,
            value="ERROR"
        )
        error_results, search_stats = search_engine.search(
            test_entries,
            error_query,
            limit=10
        )
        print(f"✅ Found {len(error_results)} error entries")

        # 2. Analyze the data
        analysis = await analytics.analyze_logs(
            test_entries,
            ['performance', 'errors', 'usage']
        )
        print(f"✅ Analysis completed with {len(analysis.get('insights', []))} insights")

        # 3. Generate comprehensive report
        report = await generator.generate_report(
            report_type=ReportType.WEEKLY_ANALYSIS,
            period_start=datetime.now() - timedelta(days=7),
            period_end=datetime.now(),
            formats=[ReportFormat.JSON]
        )
        print(f"✅ Integration report generated: {report.report_id}")

        # Test data flow
        integration_success = (
            len(test_entries) > 0 and
            len(error_results) > 0 and
            'insights' in analysis and
            report.file_paths
        )

        if integration_success:
            print("✅ Integration test successful")
        else:
            print("❌ Integration test failed")
            return False

        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_log_analysis_tests():
    """Run all log analysis system tests."""
    print("🚀 Starting Comprehensive Log Analysis System Tests")
    print("=" * 60)

    try:
        # Create test directories
        test_dirs = [
            "test_log_aggregation",
            "test_analytics",
            "test_audit",
            "test_reports",
            "test_integration_aggregation",
            "test_integration_analytics",
            "test_integration_audit",
            "test_integration_reports"
        ]

        for test_dir in test_dirs:
            Path(test_dir).mkdir(exist_ok=True)

        # Run individual tests
        aggregator = await test_log_aggregator()
        search_engine = await test_log_search_engine()
        analytics = await test_analytics_engine()
        audit = await test_audit_trail()
        generator = await test_report_generator()

        # Test integration
        integration_success = await test_integration()

        print("\n" + "=" * 60)
        if integration_success:
            print("🎉 ALL LOG ANALYSIS TESTS PASSED")
        else:
            print("❌ SOME LOG ANALYSIS TESTS FAILED")
        print("=" * 60)

        return integration_success

    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Clean up test directories
        test_dirs = [
            "test_log_aggregation",
            "test_analytics",
            "test_audit",
            "test_reports",
            "test_integration_aggregation",
            "test_integration_analytics",
            "test_integration_audit",
            "test_integration_reports"
        ]

        for test_dir in test_dirs:
            test_path = Path(test_dir)
            if test_path.exists():
                import shutil
                try:
                    shutil.rmtree(test_path)
                    print(f"🧹 Cleaned up test directory: {test_dir}")
                except Exception as e:
                    print(f"⚠️  Could not clean up {test_dir}: {e}")


def main():
    """Main function to run log analysis tests."""
    try:
        success = asyncio.run(run_log_analysis_tests())
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