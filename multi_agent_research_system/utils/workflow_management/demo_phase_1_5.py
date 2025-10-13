"""
Phase 1.5 Demonstration: Enhanced Success Tracking and Early Termination

This demo script showcases the complete Phase 1.5 implementation including:
- EnhancedSuccessTracker with comprehensive failure analysis
- WorkflowState with early termination logic
- SimpleLifecycleManager for task coordination
- Integration with AsyncScrapingOrchestrator from Phase 1.4
- Comprehensive performance metrics and monitoring

Run this demo to see the Phase 1.5 workflow management in action.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import List, Dict, Any

# Import Phase 1.5 components
from .success_tracker import (
    TaskResult, TaskType, EnhancedSuccessTracker, FailureCategory
)
from .workflow_state import (
    WorkflowState, WorkflowStatus, TargetType, TerminationReason
)
from .lifecycle_manager import (
    SimpleLifecycleManager, TaskPriority, TaskDependency
)
from .integration import (
    OrchestratorIntegration, create_workflow_integration,
    run_orchestrator_with_workflow_management
)

# Import AsyncScrapingOrchestrator for integration demo
try:
    from ..scraping_pipeline.async_orchestrator import AsyncScrapingOrchestrator, PipelineConfig
    ASYNC_ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ASYNC_ORCHESTRATOR_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockAsyncScrapingOrchestrator:
    """Mock AsyncScrapingOrchestrator for demonstration purposes."""

    def __init__(self):
        self.session_id = "demo_session_123"
        self.config = PipelineConfig()
        self.config.max_scrape_workers = 5
        self.config.max_clean_workers = 3
        self.config.worker_timeout_seconds = 120
        self.config.min_acceptable_quality = 0.6

        self.running = False
        self.submitted_tasks = []

    async def start(self):
        """Start the mock orchestrator."""
        self.running = True
        logger.info("Mock AsyncScrapingOrchestrator started")

    async def shutdown(self):
        """Shutdown the mock orchestrator."""
        self.running = False
        logger.info("Mock AsyncScrapingOrchestrator shutdown")

    async def submit_scraping_task(self, request, priority=0):
        """Mock submit scraping task."""
        if not self.running:
            return False

        self.submitted_tasks.append(('scraping', request.url, priority))
        logger.info(f"Mock submitted scraping task for {request.url}")
        return True

    async def submit_cleaning_task(self, request, priority=0):
        """Mock submit cleaning task."""
        if not self.running:
            return False

        self.submitted_tasks.append(('cleaning', request.url, priority))
        logger.info(f"Mock submitted cleaning task for {request.url}")
        return True

    def get_statistics(self):
        """Mock statistics."""
        return {
            'session_id': self.session_id,
            'total_submitted': len(self.submitted_tasks),
            'scraping_tasks': len([t for t in self.submitted_tasks if t[0] == 'scraping']),
            'cleaning_tasks': len([t for t in self.submitted_tasks if t[0] == 'cleaning'])
        }

    async def get_health_status(self):
        """Mock health status."""
        return {
            'overall_health': 'healthy',
            'health_issues': [],
            'active_workers': {'scrape': 5, 'clean': 3}
        }


async def demo_enhanced_success_tracker():
    """Demonstrate EnhancedSuccessTracker capabilities."""
    logger.info("=== Demo: Enhanced Success Tracker ===")

    # Create success tracker
    tracker = EnhancedSuccessTracker("demo_session_tracker")

    # Simulate various task results
    tasks_data = [
        {"task_type": TaskType.SCRAPING, "url": "https://example1.com", "success": True, "quality": 0.8, "duration": 2.5},
        {"task_type": TaskType.SCRAPING, "url": "https://example2.com", "success": False, "quality": 0.0, "duration": 5.0, "error": "Anti-bot detection"},
        {"task_type": TaskType.CLEANING, "url": "https://example1.com", "success": True, "quality": 0.9, "duration": 1.2},
        {"task_type": TaskType.SCRAPING, "url": "https://example3.com", "success": True, "quality": 0.7, "duration": 3.1},
        {"task_type": TaskType.SCRAPING, "url": "https://example4.com", "success": False, "quality": 0.0, "duration": 10.0, "error": "Timeout"},
        {"task_type": TaskType.CLEANING, "url": "https://example3.com", "success": True, "quality": 0.85, "duration": 0.8},
        {"task_type": TaskType.SCRAPING, "url": "https://example5.com", "success": True, "quality": 0.75, "duration": 2.8},
        {"task_type": TaskType.SCRAPING, "url": "https://example6.com", "success": False, "quality": 0.0, "duration": 4.2, "error": "Network error"},
    ]

    # Process tasks
    for i, task_data in enumerate(tasks_data):
        task_id = f"task_{i+1}"

        # Start tracking
        tracker.start_task_tracking(
            task_id=task_id,
            task_type=task_data["task_type"],
            url=task_data.get("url")
        )

        # Simulate processing time
        await asyncio.sleep(0.1)

        # Complete task
        success = task_data["success"]
        tracker.complete_task(
            task_id=task_id,
            success=success,
            error_message=task_data.get("error") if not success else None,
            duration_seconds=task_data["duration"],
            quality_score=task_data["quality"],
            content_length=1000 if success else 0
        )

    # Check early termination recommendation
    should_terminate, reason = tracker.should_terminate_early()
    logger.info(f"Early termination recommendation: {should_terminate} - {reason}")

    # Get comprehensive report
    report = tracker.get_comprehensive_report()

    logger.info(f"Success Tracker Results:")
    logger.info(f"  Total tasks: {report['total_tasks']}")
    logger.info(f"  Overall success rate: {report['overall_success_rate']:.1%}")
    logger.info(f"  Failure summary: {report['failure_summary']}")
    logger.info(f"  Optimization suggestions: {len(report['optimization_suggestions'])}")

    for suggestion in report['optimization_suggestions']:
        logger.info(f"    - {suggestion}")

    return tracker


async def demo_workflow_state():
    """Demonstrate WorkflowState with early termination."""
    logger.info("=== Demo: Workflow State with Early Termination ===")

    # Create workflow state
    workflow = WorkflowState("demo_session_workflow")

    # Add targets
    task_count_target = workflow.add_target(
        target_type=TargetType.TASK_COUNT,
        target_value=10,
        name="Task Completion Target",
        is_primary=True
    )

    success_rate_target = workflow.add_target(
        target_type=TargetType.SUCCESS_RATE,
        target_value=0.8,
        name="Success Rate Target",
        quality_threshold=0.7
    )

    quality_target = workflow.add_target(
        target_type=TargetType.QUALITY_THRESHOLD,
        target_value=0.75,
        name="Quality Target"
    )

    # Start workflow
    workflow.transition_to(WorkflowStatus.RUNNING, "Demo workflow started")

    # Simulate task progress
    for i in range(15):  # More tasks than target to show early termination
        task_result = TaskResult(
            task_id=f"workflow_task_{i+1}",
            task_type=TaskType.SCRAPING,
            success=i < 10,  # First 10 succeed, last 5 fail
            duration_seconds=2.0 + (i * 0.1),
            url=f"https://demo{i+1}.com",
            domain=f"demo{i+1}.com",
            content_length=1000 if i < 10 else 0,
            quality_score=0.8 if i < 10 else 0.2
        )

        workflow.update_task_progress(task_result)

        # Check early termination periodically
        if i % 3 == 0:
            should_terminate, reason, termination_reason = workflow.check_early_termination()
            if should_terminate:
                logger.info(f"Early termination triggered: {reason}")
                workflow.terminate_early(reason, termination_reason)
                break

        await asyncio.sleep(0.05)

    # Get final status
    progress_summary = workflow.get_progress_summary()

    logger.info(f"Workflow State Results:")
    logger.info(f"  Final status: {progress_summary['status']}")
    logger.info(f"  Overall progress: {progress_summary['overall_progress_percentage']:.1f}%")
    logger.info(f"  Success rate: {progress_summary['success_rate']:.1%}")
    logger.info(f"  Targets achieved: {progress_summary['targets_achieved']}")

    for target_id, target_info in progress_summary['targets_progress'].items():
        logger.info(f"    Target {target_id}: {target_info['progress_percentage']:.1f}% (achieved: {target_info['achieved']})")

    return workflow


async def demo_lifecycle_manager():
    """Demonstrate SimpleLifecycleManager task coordination."""
    logger.info("=== Demo: Simple Lifecycle Manager ===")

    # Create lifecycle manager
    lifecycle = SimpleLifecycleManager(
        workflow_id="demo_session_lifecycle",
        max_concurrent_tasks=3
    )

    # Define task handlers
    async def mock_scraping_handler(task_lifecycle):
        """Mock scraping task handler."""
        await asyncio.sleep(0.5)  # Simulate work
        return TaskResult(
            task_id=task_lifecycle.task_id,
            task_type=TaskType.SCRAPING,
            success=True,
            duration_seconds=0.5,
            url=task_lifecycle.context.get("url", ""),
            content_length=1000,
            quality_score=0.8
        )

    async def mock_cleaning_handler(task_lifecycle):
        """Mock cleaning task handler."""
        await asyncio.sleep(0.3)  # Simulate work
        return TaskResult(
            task_id=task_lifecycle.task_id,
            task_type=TaskType.CLEANING,
            success=True,
            duration_seconds=0.3,
            url=task_lifecycle.context.get("url", ""),
            content_length=800,
            quality_score=0.9
        )

    # Register handlers
    lifecycle.set_task_handler(TaskType.SCRAPING, mock_scraping_handler)
    lifecycle.set_task_handler(TaskType.CLEANING, mock_cleaning_handler)

    # Create tasks with dependencies
    scraping_tasks = []
    cleaning_tasks = []

    # Create scraping tasks
    for i in range(8):
        dependencies = []
        if i > 0 and i % 3 == 0:  # Every 3rd task depends on previous
            dependencies.append(TaskDependency(
                task_id=f"scrape_{i}",
                depends_on=[f"scrape_{i-1}"],
                dependency_type="completion"
            ))

        task_id = lifecycle.create_task(
            task_type=TaskType.SCRAPING,
            name=f"Scraping Task {i+1}",
            description=f"Scrape content from site {i+1}",
            priority=TaskPriority.HIGH if i < 3 else TaskPriority.NORMAL,
            dependencies=dependencies,
            context={"url": f"https://site{i+1}.com"}
        )
        scraping_tasks.append(task_id)

    # Create cleaning tasks that depend on scraping
    for i in range(6):
        dependencies = [TaskDependency(
            task_id=f"clean_{i}",
            depends_on=[f"scrape_{i}"],
            dependency_type="success"
        )]

        task_id = lifecycle.create_task(
            task_type=TaskType.CLEANING,
            name=f"Cleaning Task {i+1}",
            description=f"Clean content from site {i+1}",
            priority=TaskPriority.NORMAL,
            dependencies=dependencies,
            context={"url": f"https://site{i+1}.com", "content": f"Mock content {i+1}"}
        )
        cleaning_tasks.append(task_id)

    # Start workflow
    await lifecycle.start_workflow()

    # Wait for completion
    await asyncio.sleep(5)  # Wait for tasks to complete

    # Get status
    status = lifecycle.get_workflow_status()
    summary = lifecycle.get_workflow_summary()

    logger.info(f"Lifecycle Manager Results:")
    logger.info(f"  Total tasks: {summary['total_tasks']}")
    logger.info(f"  Successful tasks: {summary['successful_tasks']}")
    logger.info(f"  Failed tasks: {summary['failed_tasks']}")
    logger.info(f"  Success rate: {summary['success_rate']:.1%}")
    logger.info(f"  Average duration: {summary['average_task_duration_seconds']:.2f}s")

    # Stop workflow
    await lifecycle.stop_workflow("Demo completed")

    return lifecycle


async def demo_orchestrator_integration():
    """Demonstrate integration with AsyncScrapingOrchestrator."""
    logger.info("=== Demo: Orchestrator Integration ===")

    # Create mock orchestrator (since we may not have the real one)
    orchestrator = MockAsyncScrapingOrchestrator()

    # Create integration
    if ASYNC_ORCHESTRATOR_AVAILABLE:
        # Use real orchestrator if available
        config = PipelineConfig()
        real_orchestrator = AsyncScrapingOrchestrator(config)
        integration = create_workflow_integration(real_orchestrator)
    else:
        # Use mock integration
        from .integration import WorkflowIntegrationMixin
        class MockIntegration(WorkflowIntegrationMixin):
            def __init__(self):
                self.session_id = "mock_integration_session"
                self.max_concurrent_tasks = 8
                super().__init__()

            def _setup_task_handlers(self):
                async def mock_handler(task_lifecycle):
                    await asyncio.sleep(0.2)
                    return TaskResult(
                        task_id=task_lifecycle.task_id,
                        task_type=task_lifecycle.context.get("task_type", TaskType.SCRAPING),
                        success=True,
                        duration_seconds=0.2,
                        url=task_lifecycle.context.get("url", ""),
                        content_length=1000,
                        quality_score=0.8
                    )

                self.lifecycle_manager.set_task_handler(TaskType.SCRAPING, mock_handler)
                self.lifecycle_manager.set_task_handler(TaskType.CLEANING, mock_handler)

        integration = MockIntegration()

    try:
        # Start integrated workflow
        await integration.start_integrated_workflow()

        # Submit scraping workflow
        urls = [f"https://demosite{i+1}.com" for i in range(6)]
        task_ids = await integration.submit_scraping_workflow(
            urls=urls,
            search_query="demo search query",
            priority=TaskPriority.HIGH,
            sequential_processing=False
        )

        logger.info(f"Submitted {len(task_ids)} scraping tasks")

        # Wait for some processing
        await asyncio.sleep(3)

        # Monitor health
        health_report = integration.monitor_workflow_health()

        logger.info(f"Orchestrator Integration Results:")
        logger.info(f"  Overall health: {health_report['overall_health']}")
        logger.info(f"  Health issues: {len(health_report['health_issues'])}")

        for issue in health_report['health_issues']:
            logger.info(f"    - {issue}")

        logger.info(f"  Recommendations: {len(health_report['recommendations'])}")
        for rec in health_report['recommendations']:
            logger.info(f"    - {rec}")

        # Get comprehensive status
        status = integration.get_comprehensive_status()

        logger.info(f"  Workflow state: {status['workflow_state']['status']}")
        logger.info(f"  Active tasks: {status['lifecycle_manager']['active_tasks_count']}")
        logger.info(f"  Success rate: {status['success_tracker']['overall_success_rate']:.1%}")

        # Stop workflow
        await integration.stop_integrated_workflow("Demo integration completed")

    except Exception as e:
        logger.error(f"Integration demo failed: {e}")
        import traceback
        traceback.print_exc()


async def demo_performance_monitoring():
    """Demonstrate comprehensive performance monitoring."""
    logger.info("=== Demo: Performance Monitoring ===")

    # Create all components for monitoring demo
    tracker = EnhancedSuccessTracker("monitoring_demo")
    workflow = WorkflowState("monitoring_demo_workflow")
    lifecycle = SimpleLifecycleManager("monitoring_demo_lifecycle", max_concurrent_tasks=5)

    # Connect components
    lifecycle.set_success_tracker(tracker)
    lifecycle.set_workflow_state(workflow)

    # Configure performance monitoring
    workflow.update_termination_criteria(
        min_success_rate=0.6,
        max_failure_rate=0.5,
        max_consecutive_failures=3,
        max_avg_task_duration=5.0,
        evaluation_interval_seconds=1.0
    )

    # Set up task handler for monitoring
    async def variable_performance_handler(task_lifecycle):
        """Handler with variable performance for monitoring demo."""
        # Simulate variable performance
        base_duration = 0.5
        performance_variation = (task_lifecycle.task_id[-1]) if task_lifecycle.task_id[-1].isdigit() else '0'
        duration = base_duration + (int(performance_variation) * 0.1)

        await asyncio.sleep(duration)

        # Simulate variable success rates
        success_chance = 0.8 if int(performance_variation) < 7 else 0.4
        success = int(performance_variation) % 10 < int(success_chance * 10)

        quality_score = 0.9 if success else 0.3

        return TaskResult(
            task_id=task_lifecycle.task_id,
            task_type=task_lifecycle.task_type,
            success=success,
            duration_seconds=duration,
            url=task_lifecycle.context.get("url", ""),
            content_length=1000 if success else 0,
            quality_score=quality_score
        )

    lifecycle.set_task_handler(TaskType.SCRAPING, variable_performance_handler)

    # Add monitoring targets
    workflow.add_target(
        target_type=TargetType.SUCCESS_RATE,
        target_value=0.7,
        name="Performance Target",
        quality_threshold=0.6
    )

    # Start monitoring
    await lifecycle.start_workflow()

    # Create tasks that will show performance patterns
    for i in range(12):
        task_id = lifecycle.create_task(
            task_type=TaskType.SCRAPING,
            name=f"Performance Test Task {i+1}",
            priority=TaskPriority.NORMAL,
            context={"url": f"https://perf-test-{i+1}.com"}
        )

    # Monitor performance over time
    monitoring_duration = 8
    start_time = time.time()

    while time.time() - start_time < monitoring_duration:
        await asyncio.sleep(1)

        # Get current metrics
        tracker_report = tracker.get_comprehensive_report()
        workflow_status = workflow.get_progress_summary()
        lifecycle_status = lifecycle.get_workflow_status()

        logger.info(f"--- Monitoring Update (t={int(time.time() - start_time)}s) ---")
        logger.info(f"  Success rate: {tracker_report['overall_success_rate']:.1%}")
        logger.info(f"  Avg duration: {tracker_report['performance_summary']['avg_duration']:.2f}s")
        logger.info(f"  Active tasks: {lifecycle_status['active_tasks_count']}")
        logger.info(f"  Failed tasks: {lifecycle_status['failed_tasks']}")

        # Check early termination
        should_terminate, reason, termination_reason = workflow.check_early_termination()
        if should_terminate:
            logger.info(f"Early termination: {reason}")
            break

    # Get final monitoring report
    final_tracker_report = tracker.get_comprehensive_report()
    final_workflow_status = workflow.get_progress_summary()
    final_lifecycle_status = lifecycle.get_workflow_status()

    logger.info(f"=== Final Performance Monitoring Results ===")
    logger.info(f"  Total tasks processed: {final_tracker_report['total_tasks']}")
    logger.info(f"  Final success rate: {final_tracker_report['overall_success_rate']:.1%}")
    logger.info(f"  Performance pattern: {final_tracker_report['performance_summary']['patterns']}")
    logger.info(f"  Optimization suggestions: {len(final_tracker_report['optimization_suggestions'])}")

    # Stop monitoring
    await lifecycle.stop_workflow("Performance monitoring demo completed")

    return tracker, workflow, lifecycle


async def run_complete_phase_1_5_demo():
    """Run complete Phase 1.5 demonstration."""
    logger.info("🚀 Starting Phase 1.5: Enhanced Success Tracking and Early Termination Demo")
    logger.info("=" * 80)

    try:
        # Demo 1: Enhanced Success Tracker
        logger.info("\n📊 Demo 1: Enhanced Success Tracker")
        await demo_enhanced_success_tracker()

        # Demo 2: Workflow State with Early Termination
        logger.info("\n🎯 Demo 2: Workflow State with Early Termination")
        await demo_workflow_state()

        # Demo 3: Simple Lifecycle Manager
        logger.info("\n🔄 Demo 3: Simple Lifecycle Manager")
        await demo_lifecycle_manager()

        # Demo 4: Orchestrator Integration
        logger.info("\n🔗 Demo 4: Orchestrator Integration")
        await demo_orchestrator_integration()

        # Demo 5: Performance Monitoring
        logger.info("\n📈 Demo 5: Performance Monitoring")
        await demo_performance_monitoring()

        logger.info("\n✅ Phase 1.5 Demo Completed Successfully!")
        logger.info("\nKey Features Demonstrated:")
        logger.info("  ✅ EnhancedSuccessTracker with comprehensive failure analysis")
        logger.info("  ✅ WorkflowState with intelligent early termination")
        logger.info("  ✅ SimpleLifecycleManager for task coordination")
        logger.info("  ✅ Integration with AsyncScrapingOrchestrator")
        logger.info("  ✅ Comprehensive performance monitoring and metrics")

    except Exception as e:
        logger.error(f"❌ Phase 1.5 Demo Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    """Run the Phase 1.5 demonstration."""
    asyncio.run(run_complete_phase_1_5_demo())