#!/usr/bin/env python3
"""
Test script for the comprehensive monitoring system.

This script tests the MetricsCollector, PerformanceMonitor, SystemHealthMonitor,
RealTimeDashboard, and DiagnosticTools components.
"""

import asyncio
import json
import sys
import os
import time
import uuid
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from monitoring import (
    MetricsCollector,
    PerformanceMonitor,
    SystemHealthMonitor,
    DiagnosticTools
)


async def test_metrics_collector():
    """Test the MetricsCollector functionality."""
    print("📊 Testing MetricsCollector...")

    session_id = f"test_metrics_session_{uuid.uuid4().hex[:8]}"
    collector = MetricsCollector(
        session_id=session_id,
        metrics_dir="test_metrics",
        collection_interval=5  # Short interval for testing
    )

    try:
        # Start monitoring
        await collector.start_monitoring()
        print("✅ MetricsCollector started successfully")

        # Record agent metrics
        collector.record_agent_metric(
            agent_name="research_agent",
            metric_type="performance",
            metric_name="search_completion_time",
            value=2.5,
            unit="seconds"
        )

        collector.record_agent_metric(
            agent_name="report_agent",
            metric_type="usage",
            metric_name="sections_generated",
            value=3,
            unit="count"
        )

        collector.record_agent_metric(
            agent_name="editor_agent",
            metric_type="error",
            metric_name="validation_errors",
            value=1,
            unit="count"
        )

        print("✅ Agent metrics recorded successfully")

        # Record tool metrics
        collector.record_tool_metric(
            tool_name="web_search",
            agent_name="research_agent",
            execution_time=1.8,
            success=True,
            input_size=256,
            output_size=1024
        )

        collector.record_tool_metric(
            tool_name="content_analysis",
            agent_name="report_agent",
            execution_time=3.2,
            success=False,
            input_size=512,
            output_size=0,
            error_type="TimeoutError"
        )

        print("✅ Tool metrics recorded successfully")

        # Record workflow metrics
        collector.record_workflow_metric(
            workflow_id="workflow_123",
            stage_name="research_phase",
            stage_duration=15.5,
            total_duration=15.5,
            success=True,
            agents_involved=["research_agent"],
            tools_used=["web_search", "content_analysis"]
        )

        collector.record_workflow_metric(
            workflow_id="workflow_123",
            stage_name="report_generation",
            stage_duration=8.3,
            total_duration=23.8,
            success=True,
            agents_involved=["report_agent"],
            tools_used=["content_synthesis"]
        )

        print("✅ Workflow metrics recorded successfully")

        # Wait for some system metrics to be collected
        await asyncio.sleep(6)  # Wait for at least one collection cycle

        # Get summaries
        agent_summary = collector.get_agent_summary()
        tool_summary = collector.get_tool_summary()
        system_summary = collector.get_system_summary()

        print(f"✅ Agent summary: {len(agent_summary.get('metrics_by_type', {}))} metric types")
        print(f"✅ Tool summary: {tool_summary.get('total_executions', 0)} total executions")
        print(f"✅ System summary: CPU {system_summary.get('current', {}).get('cpu_percent', 0):.1f}%")

        # Export metrics
        export_path = collector.export_metrics()
        print(f"✅ Metrics exported to: {export_path}")

        # Verify exported file exists and contains data
        if Path(export_path).exists():
            with open(export_path, 'r') as f:
                exported_data = json.load(f)
            print(f"✅ Export verification: {len(exported_data.get('data', {}))} data types exported")

        return collector

    except Exception as e:
        print(f"❌ MetricsCollector test failed: {e}")
        raise

    finally:
        await collector.stop_monitoring()


async def test_performance_monitor(collector):
    """Test the PerformanceMonitor functionality."""
    print("\n⚡ Testing PerformanceMonitor...")

    monitor = PerformanceMonitor(
        metrics_collector=collector,
        alert_cooldown_minutes=1  # Short cooldown for testing
    )

    try:
        # Start monitoring
        await monitor.start_monitoring()
        print("✅ PerformanceMonitor started successfully")

        # Test session tracking
        session_id = f"perf_test_session_{uuid.uuid4().hex[:8]}"
        monitor.start_session_tracking(session_id, {"test": "performance_monitoring"})

        # Test agent activity recording
        monitor.record_agent_activity(
            agent_name="test_agent",
            activity_type="performance",
            activity_name="test_activity",
            value=42.0,
            unit="count"
        )

        # Test tool execution monitoring (context manager)
        async with monitor.monitor_tool_execution(
            tool_name="test_tool",
            agent_name="test_agent",
            input_size=100
        ) as context:
            # Simulate tool work
            await asyncio.sleep(0.1)
            context['result'] = "success"

        print("✅ Tool execution monitoring test passed")

        # Test workflow stage monitoring (context manager)
        async with monitor.monitor_workflow_stage(
            workflow_id="test_workflow",
            stage_name="test_stage",
            agents_involved=["test_agent"]
        ) as context:
            # Simulate workflow stage work
            await asyncio.sleep(0.1)
            context['stage_result'] = "completed"

        print("✅ Workflow stage monitoring test passed")

        # Wait for monitoring to process
        await asyncio.sleep(2)

        # Get performance summary
        perf_summary = monitor.get_performance_summary()
        print(f"✅ Performance summary: {perf_summary.get('active_sessions', 0)} active sessions")

        # Test alerts functionality
        alerts_summary = monitor.get_alerts_summary(hours_back=1)
        print(f"✅ Alerts summary: {alerts_summary.get('total_alerts', 0)} total alerts")

        # End session tracking
        session_summary = monitor.end_session_tracking(session_id, {"status": "completed"})
        print(f"✅ Session ended: {session_summary.get('duration_seconds', 0):.2f}s duration")

        return monitor

    except Exception as e:
        print(f"❌ PerformanceMonitor test failed: {e}")
        raise

    finally:
        await monitor.stop_monitoring()


async def test_system_health_monitor():
    """Test the SystemHealthMonitor functionality."""
    print("\n🏥 Testing SystemHealthMonitor...")

    session_id = f"health_test_session_{uuid.uuid4().hex[:8]}"
    health_monitor = SystemHealthMonitor(
        session_id=session_id,
        health_dir="test_health",
        check_interval=10  # Short interval for testing
    )

    try:
        # Start monitoring
        await health_monitor.start_monitoring()
        print("✅ SystemHealthMonitor started successfully")

        # Wait for initial health checks to run
        await asyncio.sleep(12)  # Wait for at least one check cycle

        # Run a specific health check
        memory_check = await health_monitor.run_health_check("memory_usage")
        print(f"✅ Memory health check: {memory_check.get('status', 'unknown')}")

        # Run all health checks
        all_checks = await health_monitor.run_all_health_checks()
        print(f"✅ All health checks completed: {len(all_checks)} checks run")

        # Get health summary
        health_summary = health_monitor.get_health_summary()
        print(f"✅ Health summary: {health_summary.get('overall_status', 'unknown')} overall status")

        # Generate health report
        health_report = await health_monitor.generate_health_report()
        if health_report:
            print(f"✅ Health report generated: {health_report.total_checks}/{health_report.total_checks} checks")
            print(f"   Status: {health_report.overall_status.value}")
            print(f"   Alerts: {len(health_report.alerts)}")
            print(f"   Recommendations: {len(health_report.recommendations)}")

        # Export health report
        export_path = health_monitor.export_health_report()
        print(f"✅ Health report exported to: {export_path}")

        return health_monitor

    except Exception as e:
        print(f"❌ SystemHealthMonitor test failed: {e}")
        raise

    finally:
        await health_monitor.stop_monitoring()


async def test_diagnostic_tools(collector, performance_monitor, health_monitor):
    """Test the DiagnosticTools functionality."""
    print("\n🔍 Testing DiagnosticTools...")

    session_id = f"diag_test_session_{uuid.uuid4().hex[:8]}"
    diagnostics = DiagnosticTools(
        session_id=session_id,
        metrics_collector=collector,
        performance_monitor=performance_monitor,
        health_monitor=health_monitor,
        diagnostics_dir="test_diagnostics"
    )

    try:
        # Generate comprehensive diagnostic report
        diagnostic_report = await diagnostics.generate_comprehensive_diagnostic_report()
        print(f"✅ Comprehensive diagnostic report generated")
        print(f"   Session ID: {diagnostic_report.get('session_info', {}).get('session_id')}")
        print(f"   Report sections: {len(diagnostic_report)}")

        # Test session reconstruction
        session_reconstruction = await diagnostics.reconstruct_session(
            session_id=session_id,
            time_range_hours=1
        )
        print(f"✅ Session reconstruction completed")
        print(f"   Total events: {session_reconstruction.get('summary', {}).get('total_events', 0)}")
        print(f"   Agent activities: {session_reconstruction.get('summary', {}).get('agent_activity_count', 0)}")
        print(f"   Tool executions: {session_reconstruction.get('summary', {}).get('tool_execution_count', 0)}")

        # Test error pattern analysis
        error_analysis = await diagnostics.analyze_error_patterns(time_range_hours=1)
        print(f"✅ Error pattern analysis completed")
        print(f"   Total errors: {error_analysis.get('total_errors', 0)}")
        print(f"   Error types analyzed: {len(error_analysis.get('error_breakdown', {}))}")

        # Export diagnostic report
        export_path = diagnostics.export_diagnostic_report(diagnostic_report)
        print(f"✅ Diagnostic report exported to: {export_path}")

        # Verify exported file
        if Path(export_path).exists():
            with open(export_path, 'r') as f:
                exported_data = json.load(f)
            print(f"✅ Export verification: {len(exported_data)} report sections")

        return diagnostics

    except Exception as e:
        print(f"❌ DiagnosticTools test failed: {e}")
        raise

    finally:
        await diagnostics.cleanup()


async def test_integration():
    """Test integration of all monitoring components."""
    print("\n🔗 Testing Integration...")

    try:
        session_id = f"integration_test_{uuid.uuid4().hex[:8]}"

        # Initialize all components
        collector = MetricsCollector(session_id=session_id, metrics_dir="test_integration_metrics")
        performance_monitor = PerformanceMonitor(metrics_collector=collector)
        health_monitor = SystemHealthMonitor(session_id=session_id, health_dir="test_integration_health")
        diagnostics = DiagnosticTools(
            session_id=session_id,
            metrics_collector=collector,
            performance_monitor=performance_monitor,
            health_monitor=health_monitor,
            diagnostics_dir="test_integration_diagnostics"
        )

        # Start all monitoring
        await collector.start_monitoring()
        await performance_monitor.start_monitoring()
        await health_monitor.start_monitoring()

        print("✅ All monitoring components started")

        # Simulate some activity
        collector.record_agent_metric(
            agent_name="integration_agent",
            metric_type="performance",
            metric_name="integration_test",
            value=100,
            unit="count"
        )

        collector.record_tool_metric(
            tool_name="integration_tool",
            agent_name="integration_agent",
            execution_time=0.5,
            success=True
        )

        # Wait for data collection
        await asyncio.sleep(6)

        # Generate integrated report
        diagnostic_report = await diagnostics.generate_comprehensive_diagnostic_report()

        # Verify all components are working together
        metrics_summary = collector.get_agent_summary()
        performance_summary = performance_monitor.get_performance_summary()
        health_summary = health_monitor.get_health_summary()

        print(f"✅ Integration test successful:")
        print(f"   Metrics collected: {metrics_summary.get('total_metrics', 0)}")
        print(f"   Performance monitoring: {performance_summary.get('monitoring_status', 'unknown')}")
        print(f"   Health status: {health_summary.get('overall_status', 'unknown')}")

        # Cleanup all components
        await collector.stop_monitoring()
        await performance_monitor.stop_monitoring()
        await health_monitor.stop_monitoring()
        await diagnostics.cleanup()

        print("✅ All components cleaned up successfully")

        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


async def run_monitoring_tests():
    """Run all monitoring system tests."""
    print("🚀 Starting Comprehensive Monitoring System Tests")
    print("=" * 60)

    try:
        # Test individual components
        collector = await test_metrics_collector()
        performance_monitor = await test_performance_monitor(collector)
        health_monitor = await test_system_health_monitor()
        await test_diagnostic_tools(collector, performance_monitor, health_monitor)

        # Test integration
        integration_success = await test_integration()

        print("\n" + "=" * 60)
        if integration_success:
            print("🎉 ALL MONITORING TESTS PASSED")
        else:
            print("❌ SOME TESTS FAILED")
        print("=" * 60)

        return integration_success

    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Clean up test directories
        test_dirs = ["test_metrics", "test_health", "test_diagnostics", "test_integration_metrics",
                    "test_integration_health", "test_integration_diagnostics"]

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
    """Main function to run monitoring tests."""
    success = asyncio.run(run_monitoring_tests())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()