"""
Phase 1.5 Comprehensive Tests and Validation

This module provides comprehensive tests and validation for all Phase 1.5 components:
- EnhancedSuccessTracker with comprehensive failure analysis
- WorkflowState with early termination logic
- SimpleLifecycleManager for task coordination
- Integration with AsyncScrapingOrchestrator
- Performance metrics and monitoring

Run these tests to validate the Phase 1.5 implementation.
"""

import asyncio
import logging
import pytest
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Import Phase 1.5 components
from .success_tracker import (
    TaskResult, TaskType, EnhancedSuccessTracker, FailureAnalysis,
    FailureCategory, SuccessMetrics, PerformancePattern
)
from .workflow_state import (
    WorkflowState, WorkflowStatus, TargetType, TargetDefinition,
    TerminationCriteria, TerminationReason, StateTransition
)
from .lifecycle_manager import (
    SimpleLifecycleManager, TaskLifecycle, TaskPriority,
    TaskDependency, LifecycleStage
)
from .integration import (
    OrchestratorIntegration, WorkflowIntegrationMixin
)

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestEnhancedSuccessTracker:
    """Test suite for EnhancedSuccessTracker."""

    @pytest.fixture
    def success_tracker(self):
        """Create a test success tracker."""
        return EnhancedSuccessTracker("test_session")

    @pytest.fixture
    def sample_task_results(self):
        """Create sample task results for testing."""
        return [
            TaskResult(
                task_id=f"task_{i}",
                task_type=TaskType.SCRAPING,
                success=i % 3 != 0,  # 2 out of 3 succeed
                duration_seconds=2.0 + (i * 0.1),
                url=f"https://test{i}.com",
                domain=f"test{i}.com",
                content_length=1000 if i % 3 != 0 else 0,
                quality_score=0.8 if i % 3 != 0 else 0.0,
                error_message="Mock error" if i % 3 == 0 else None
            )
            for i in range(10)
        ]

    def test_tracker_initialization(self, success_tracker):
        """Test success tracker initialization."""
        assert success_tracker.session_id == "test_session"
        assert len(success_tracker._tasks) == 0
        assert len(success_tracker._failures) == 0
        assert success_tracker.early_termination_enabled is True

    def test_task_tracking(self, success_tracker, sample_task_results):
        """Test task tracking functionality."""
        # Start tracking tasks
        for result in sample_task_results:
            success_tracker.start_task_tracking(
                task_id=result.task_id,
                task_type=result.task_type,
                url=result.url
            )

        assert len(success_tracker._active_tasks) == len(sample_task_results)

        # Complete tasks
        for result in sample_task_results:
            success_tracker.complete_task(
                task_id=result.task_id,
                success=result.success,
                error_message=result.error_message,
                duration_seconds=result.duration_seconds,
                quality_score=result.quality_score
            )

        assert len(success_tracker._active_tasks) == 0
        assert len(success_tracker._tasks) == len(sample_task_results)

    def test_failure_analysis(self, success_tracker):
        """Test failure analysis generation."""
        # Create failing task
        task_id = "failing_task"
        success_tracker.start_task_tracking(
            task_id=task_id,
            task_type=TaskType.SCRAPING,
            url="https://fail.com"
        )

        # Complete with failure
        success_tracker.complete_task(
            task_id=task_id,
            success=False,
            error_message="Connection timeout",
            error_type="timeout_error"
        )

        # Check failure analysis
        assert len(success_tracker._failures) == 1
        failure = success_tracker._failures[0]
        assert failure.failure_category == FailureCategory.TIMEOUT_FAILURE
        assert failure.error_message == "Connection timeout"

    def test_early_termination_logic(self, success_tracker):
        """Test early termination logic."""
        # Add successful tasks
        for i in range(8):
            task_id = f"success_task_{i}"
            success_tracker.start_task_tracking(task_id, TaskType.SCRAPING)
            success_tracker.complete_task(task_id, True, duration_seconds=2.0, quality_score=0.8)

        # Add failing tasks to trigger early termination
        for i in range(6):
            task_id = f"fail_task_{i}"
            success_tracker.start_task_tracking(task_id, TaskType.SCRAPING)
            success_tracker.complete_task(
                task_id, False,
                error_message="Mock error",
                duration_seconds=1.0
            )

        # Check early termination recommendation
        should_terminate, reason = success_tracker.should_terminate_early()
        assert should_terminate is True
        assert "failure rate" in reason.lower()

    def test_comprehensive_report(self, success_tracker, sample_task_results):
        """Test comprehensive report generation."""
        # Process sample results
        for result in sample_task_results:
            success_tracker.start_task_tracking(result.task_id, result.task_type, result.url)
            success_tracker.complete_task(
                result.task_id, result.success,
                duration_seconds=result.duration_seconds,
                quality_score=result.quality_score
            )

        # Generate report
        report = success_tracker.get_comprehensive_report()

        # Validate report structure
        assert 'session_id' in report
        assert 'total_tasks' in report
        assert 'overall_success_rate' in report
        assert 'failure_summary' in report
        assert 'anti_bot_summary' in report
        assert 'performance_summary' in report
        assert 'optimization_suggestions' in report
        assert 'early_termination' in report

        # Validate values
        assert report['total_tasks'] == len(sample_task_results)
        assert 0 <= report['overall_success_rate'] <= 1
        assert report['failure_summary']['total_failures'] == 3  # 1 out of 3 fail

    def test_performance_pattern_detection(self, success_tracker):
        """Test performance pattern detection."""
        # Create tasks with improving performance
        durations = [5.0, 4.0, 3.0, 2.0, 1.0, 0.8, 0.6, 0.5]

        for i, duration in enumerate(durations):
            task_id = f"perf_task_{i}"
            success_tracker.start_task_tracking(task_id, TaskType.SCRAPING)
            success_tracker.complete_task(task_id, True, duration_seconds=duration, quality_score=0.8)

        # Check pattern detection
        report = success_tracker.get_comprehensive_report()
        patterns = report['performance_summary']['patterns']

        # Should detect improving pattern
        assert any(pattern == PerformancePattern.IMPROVING.value for pattern in patterns.values())


class TestWorkflowState:
    """Test suite for WorkflowState."""

    @pytest.fixture
    def workflow_state(self):
        """Create a test workflow state."""
        return WorkflowState("test_workflow")

    def test_workflow_initialization(self, workflow_state):
        """Test workflow state initialization."""
        assert workflow_state.workflow_id == "test_workflow"
        assert workflow_state.status == WorkflowStatus.INITIALIZING
        assert len(workflow_state.targets) == 0
        assert workflow_state.primary_target_id is None

    def test_target_management(self, workflow_state):
        """Test target creation and management."""
        # Add primary target
        target_id = workflow_state.add_target(
            target_type=TargetType.TASK_COUNT,
            target_value=10,
            name="Test Target",
            is_primary=True
        )

        assert target_id in workflow_state.targets
        assert workflow_state.primary_target_id == target_id
        assert workflow_state.targets[target_id].target_value == 10

        # Add secondary target
        secondary_id = workflow_state.add_target(
            target_type=TargetType.SUCCESS_RATE,
            target_value=0.8,
            name="Success Target"
        )

        assert secondary_id in workflow_state.targets
        assert workflow_state.primary_target_id == target_id  # Primary unchanged

    def test_state_transitions(self, workflow_state):
        """Test workflow state transitions."""
        # Initial state
        assert workflow_state.status == WorkflowStatus.INITIALIZING
        assert len(workflow_state.state_history) == 0

        # Transition to running
        workflow_state.transition_to(WorkflowStatus.RUNNING, "Test start")
        assert workflow_state.status == WorkflowStatus.RUNNING
        assert workflow_state.started_at is not None
        assert len(workflow_state.state_history) == 1

        # Transition to completed
        workflow_state.transition_to(WorkflowStatus.COMPLETED, "Test complete")
        assert workflow_state.status == WorkflowStatus.COMPLETED
        assert workflow_state.completed_at is not None
        assert len(workflow_state.state_history) == 2

        # Check transition history
        history = workflow_state.get_state_history()
        assert len(history) == 2
        assert history[0]['to_status'] == WorkflowStatus.RUNNING.value
        assert history[1]['to_status'] == WorkflowStatus.COMPLETED.value

    def test_target_progress_tracking(self, workflow_state):
        """Test target progress tracking."""
        # Add task count target
        target_id = workflow_state.add_target(
            target_type=TargetType.TASK_COUNT,
            target_value=5,
            name="Task Count Target"
        )

        # Update progress with task results
        for i in range(5):
            task_result = TaskResult(
                task_id=f"task_{i}",
                task_type=TaskType.SCRAPING,
                success=True,
                duration_seconds=1.0,
                url=f"https://test{i}.com",
                content_length=1000,
                quality_score=0.8
            )
            workflow_state.update_task_progress(task_result)

        # Check target achievement
        target = workflow_state.targets[target_id]
        assert target.achieved is True
        assert target.progress_percentage == 100.0
        assert target.achievement_time is not None

    def test_early_termination_criteria(self, workflow_state):
        """Test early termination criteria evaluation."""
        # Configure strict termination criteria
        workflow_state.update_termination_criteria(
            min_success_rate=0.9,
            max_failure_rate=0.1,
            max_consecutive_failures=2
        )

        # Add failing tasks to trigger termination
        for i in range(3):
            task_result = TaskResult(
                task_id=f"fail_task_{i}",
                task_type=TaskType.SCRAPING,
                success=False,
                duration_seconds=1.0,
                url=f"https://fail{i}.com",
                error_message="Mock error"
            )
            workflow_state.update_task_progress(task_result)

        # Check early termination
        should_terminate, reason, termination_reason = workflow_state.check_early_termination()
        assert should_terminate is True
        assert termination_reason in [TerminationReason.ERROR_THRESHOLD, TerminationReason.PERFORMANCE_THRESHOLD]

    def test_progress_summary(self, workflow_state):
        """Test progress summary generation."""
        # Add targets
        workflow_state.add_target(
            target_type=TargetType.TASK_COUNT,
            target_value=3,
            name="Task Target",
            is_primary=True
        )

        # Add some task results
        for i in range(2):
            task_result = TaskResult(
                task_id=f"task_{i}",
                task_type=TaskType.SCRAPING,
                success=i == 0,  # First succeeds, second fails
                duration_seconds=1.0,
                url=f"https://test{i}.com",
                content_length=1000 if i == 0 else 0,
                quality_score=0.8 if i == 0 else 0.0
            )
            workflow_state.update_task_progress(task_result)

        # Get progress summary
        summary = workflow_state.get_progress_summary()

        # Validate summary
        assert summary['workflow_id'] == "test_workflow"
        assert summary['total_tasks'] == 2
        assert summary['successful_tasks'] == 1
        assert summary['failed_tasks'] == 1
        assert summary['success_rate'] == 0.5
        assert summary['overall_progress_percentage'] == 66.7  # 2/3 tasks completed


class TestSimpleLifecycleManager:
    """Test suite for SimpleLifecycleManager."""

    @pytest.fixture
    def lifecycle_manager(self):
        """Create a test lifecycle manager."""
        return SimpleLifecycleManager("test_lifecycle", max_concurrent_tasks=3)

    def test_lifecycle_initialization(self, lifecycle_manager):
        """Test lifecycle manager initialization."""
        assert lifecycle_manager.workflow_id == "test_lifecycle"
        assert lifecycle_manager.max_concurrent_tasks == 3
        assert lifecycle_manager.running is False
        assert len(lifecycle_manager.tasks) == 0

    def test_task_creation(self, lifecycle_manager):
        """Test task creation."""
        task_id = lifecycle_manager.create_task(
            task_type=TaskType.SCRAPING,
            name="Test Task",
            description="A test scraping task",
            priority=TaskPriority.HIGH
        )

        assert task_id in lifecycle_manager.tasks
        task = lifecycle_manager.tasks[task_id]
        assert task.name == "Test Task"
        assert task.task_type == TaskType.SCRAPING
        assert task.priority == TaskPriority.HIGH
        assert task.current_stage == LifecycleStage.CREATED

    def test_task_dependencies(self, lifecycle_manager):
        """Test task dependency management."""
        # Create first task
        task1_id = lifecycle_manager.create_task(
            task_type=TaskType.SCRAPING,
            name="Task 1"
        )

        # Create second task with dependency
        dependency = TaskDependency(
            task_id="task_2",
            depends_on=[task1_id],
            dependency_type="completion"
        )

        task2_id = lifecycle_manager.create_task(
            task_type=TaskType.SCRAPING,
            name="Task 2",
            dependencies=[dependency]
        )

        # Check dependency satisfaction
        task2 = lifecycle_manager.tasks[task2_id]
        assert not task2.has_dependencies_satisfied(set(), set())  # Not satisfied initially

        # Mark first task as completed
        lifecycle_manager.completed_tasks.add(task1_id)
        assert task2.has_dependencies_satisfied({task1_id}, set())  # Now satisfied

    def test_task_execution(self, lifecycle_manager):
        """Test task execution with handlers."""
        # Set up task handler
        async def mock_handler(task_lifecycle):
            await asyncio.sleep(0.1)  # Simulate work
            return TaskResult(
                task_id=task_lifecycle.task_id,
                task_type=task_lifecycle.task_type,
                success=True,
                duration_seconds=0.1,
                url="https://test.com",
                content_length=1000,
                quality_score=0.8
            )

        lifecycle_manager.set_task_handler(TaskType.SCRAPING, mock_handler)

        # Create and start task
        task_id = lifecycle_manager.create_task(
            task_type=TaskType.SCRAPING,
            name="Test Task"
        )

        # Start workflow
        asyncio.create_task(lifecycle_manager.start_workflow())

        # Wait for task completion
        await asyncio.sleep(0.5)

        # Check task status
        task = lifecycle_manager.tasks[task_id]
        assert task.is_finished()
        assert task.current_stage == LifecycleStage.COMPLETED
        assert task.result is not None

        # Stop workflow
        await lifecycle_manager.stop_workflow("Test completed")

    def test_concurrent_task_limiting(self, lifecycle_manager):
        """Test concurrent task limiting."""
        executed_tasks = []

        async def slow_handler(task_lifecycle):
            executed_tasks.append(task_lifecycle.task_id)
            await asyncio.sleep(0.2)  # Slow task
            return TaskResult(
                task_id=task_lifecycle.task_id,
                task_type=task_lifecycle.task_type,
                success=True,
                duration_seconds=0.2
            )

        lifecycle_manager.set_task_handler(TaskType.SCRAPING, slow_handler)

        # Create more tasks than concurrent limit
        task_ids = []
        for i in range(5):  # 5 tasks, limit is 3
            task_id = lifecycle_manager.create_task(
                task_type=TaskType.SCRAPING,
                name=f"Task {i}"
            )
            task_ids.append(task_id)

        # Start workflow
        asyncio.create_task(lifecycle_manager.start_workflow())

        # Wait a bit and check concurrent execution
        await asyncio.sleep(0.1)
        assert len(executed_tasks) <= 3  # Should not exceed concurrent limit

        # Wait for completion
        await asyncio.sleep(0.5)

        # Stop workflow
        await lifecycle_manager.stop_workflow("Test completed")

    def test_workflow_status_tracking(self, lifecycle_manager):
        """Test workflow status tracking."""
        status = lifecycle_manager.get_workflow_status()

        assert 'workflow_id' in status
        assert 'running' in status
        assert 'total_tasks' in status
        assert 'active_tasks_count' in status
        assert 'success_rate' in status
        assert 'queue_status' in status

        # Initial status
        assert status['running'] is False
        assert status['total_tasks'] == 0

        # Add tasks and check status update
        lifecycle_manager.create_task(TaskType.SCRAPING, "Test 1")
        lifecycle_manager.create_task(TaskType.CLEANING, "Test 2")

        status = lifecycle_manager.get_workflow_status()
        assert status['total_tasks'] == 2


class TestIntegration:
    """Test suite for integration components."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator for testing."""
        class MockOrchestrator:
            def __init__(self):
                self.session_id = "mock_session"
                self.config = type('Config', (), {
                    'max_scrape_workers': 5,
                    'max_clean_workers': 3,
                    'worker_timeout_seconds': 120,
                    'min_acceptable_quality': 0.6
                })()

            async def start(self):
                pass

            async def shutdown(self):
                pass

            async def submit_scraping_task(self, request, priority=0):
                return True

            async def submit_cleaning_task(self, request, priority=0):
                return True

            def get_statistics(self):
                return {'session_id': self.session_id}

            async def get_health_status(self):
                return {'overall_health': 'healthy', 'health_issues': []}

        return MockOrchestrator()

    def test_workflow_integration_mixin(self):
        """Test WorkflowIntegrationMixin functionality."""
        class TestOrchestrator(WorkflowIntegrationMixin):
            def __init__(self):
                self.session_id = "test_integration"
                self.max_concurrent_tasks = 5
                super().__init__()

        orchestrator = TestOrchestrator()

        # Check component initialization
        assert orchestrator.success_tracker is not None
        assert orchestrator.workflow_state is not None
        assert orchestrator.lifecycle_manager is not None

        # Check integration setup
        assert orchestrator.lifecycle_manager.success_tracker is orchestrator.success_tracker
        assert orchestrator.lifecycle_manager.workflow_state is orchestrator.workflow_state

    def test_orchestrator_integration(self, mock_orchestrator):
        """Test OrchestratorIntegration functionality."""
        from .integration import OrchestratorIntegration

        # Patch the import check
        import utils.workflow_management.integration as integration_module
        original_available = integration_module.ASYNC_ORCHESTRATOR_AVAILABLE
        integration_module.ASYNC_ORCHESTRATOR_AVAILABLE = True

        try:
            # Create integration
            integration = OrchestratorIntegration(mock_orchestrator)

            # Check integration setup
            assert integration.orchestrator is mock_orchestrator
            assert integration.success_tracker is not None
            assert integration.workflow_state is not None
            assert integration.lifecycle_manager is not None

            # Check default targets
            assert len(integration.workflow_state.targets) > 0
            assert integration.workflow_state.primary_target_id is not None

            # Check early termination configuration
            assert integration.workflow_state.early_termination_enabled is True

        finally:
            # Restore original value
            integration_module.ASYNC_ORCHESTRATOR_AVAILABLE = original_available

    @pytest.mark.asyncio
    async def test_workflow_task_submission(self, mock_orchestrator):
        """Test workflow task submission."""
        class TestIntegration(WorkflowIntegrationMixin):
            def __init__(self):
                self.session_id = "test_workflow"
                self.max_concurrent_tasks = 3
                super().__init__()

            def _setup_task_handlers(self):
                async def mock_handler(task_lifecycle):
                    return TaskResult(
                        task_id=task_lifecycle.task_id,
                        task_type=task_lifecycle.task_type,
                        success=True,
                        duration_seconds=0.1
                    )
                self.lifecycle_manager.set_task_handler(TaskType.SCRAPING, mock_handler)

        integration = TestIntegration()

        # Submit scraping workflow
        urls = ["https://test1.com", "https://test2.com", "https://test3.com"]
        task_ids = await integration.submit_scraping_workflow(
            urls=urls,
            search_query="test query"
        )

        assert len(task_ids) == len(urls)
        assert all(task_id in integration.lifecycle_manager.tasks for task_id in task_ids)

        # Check task details
        for task_id in task_ids:
            task = integration.lifecycle_manager.tasks[task_id]
            assert task.task_type == TaskType.SCRAPING
            assert task.context['url'] in urls

    def test_comprehensive_status_reporting(self):
        """Test comprehensive status reporting."""
        class TestOrchestrator(WorkflowIntegrationMixin):
            def __init__(self):
                self.session_id = "test_status"
                self.max_concurrent_tasks = 3
                super().__init__()

        orchestrator = TestOrchestrator()

        # Get comprehensive status
        status = orchestrator.get_comprehensive_status()

        # Validate status structure
        assert 'workflow_state' in status
        assert 'success_tracker' in status
        assert 'lifecycle_manager' in status
        assert 'integration_status' in status

        # Validate integration status
        integration_status = status['integration_status']
        assert integration_status['components_initialized'] is True
        assert integration_status['early_termination_enabled'] is True


class TestPerformanceMonitoring:
    """Test suite for performance monitoring and metrics."""

    @pytest.mark.asyncio
    async def test_real_time_monitoring(self):
        """Test real-time performance monitoring."""
        # Create components
        tracker = EnhancedSuccessTracker("perf_test")
        workflow = WorkflowState("perf_test_workflow")
        lifecycle = SimpleLifecycleManager("perf_test_lifecycle", max_concurrent_tasks=3)

        # Connect components
        lifecycle.set_success_tracker(tracker)
        lifecycle.set_workflow_state(workflow)

        # Set up performance monitoring
        workflow.update_termination_criteria(
            min_success_rate=0.7,
            max_avg_task_duration=2.0,
            evaluation_interval_seconds=0.5
        )

        # Set up variable performance handler
        async def variable_handler(task_lifecycle):
            # Variable performance based on task ID
            task_num = int(task_lifecycle.task_id.split('_')[-1])
            duration = 0.5 + (task_num * 0.1)
            success = task_num % 4 != 0  # 3 out of 4 succeed

            await asyncio.sleep(duration)

            return TaskResult(
                task_id=task_lifecycle.task_id,
                task_type=task_lifecycle.task_type,
                success=success,
                duration_seconds=duration,
                quality_score=0.8 if success else 0.3
            )

        lifecycle.set_task_handler(TaskType.SCRAPING, variable_handler)

        # Add monitoring targets
        workflow.add_target(
            target_type=TargetType.SUCCESS_RATE,
            target_value=0.75,
            name="Performance Target"
        )

        # Start monitoring
        await lifecycle.start_workflow()

        # Create tasks
        task_ids = []
        for i in range(8):
            task_id = lifecycle.create_task(
                task_type=TaskType.SCRAPING,
                name=f"Perf Task {i+1}",
                context={"url": f"https://perf{i+1}.com"}
            )
            task_ids.append(task_id)

        # Monitor for a period
        monitoring_data = []
        for _ in range(5):
            await asyncio.sleep(0.5)

            # Collect metrics
            tracker_report = tracker.get_comprehensive_report()
            workflow_status = workflow.get_progress_summary()
            lifecycle_status = lifecycle.get_workflow_status()

            monitoring_data.append({
                'timestamp': datetime.now(),
                'success_rate': tracker_report['overall_success_rate'],
                'avg_duration': tracker_report['performance_summary']['avg_duration'],
                'active_tasks': lifecycle_status['active_tasks_count'],
                'workflow_progress': workflow_status['overall_progress_percentage']
            })

        # Stop monitoring
        await lifecycle.stop_workflow("Performance test completed")

        # Validate monitoring data
        assert len(monitoring_data) == 5
        assert all(0 <= data['success_rate'] <= 1 for data in monitoring_data)
        assert all(data['avg_duration'] >= 0 for data in monitoring_data)

        # Check for performance trends
        success_rates = [data['success_rate'] for data in monitoring_data]
        assert len(set(success_rates)) > 1  # Should have variation


# Test runner function
async def run_all_tests():
    """Run all Phase 1.5 tests."""
    logger.info("🧪 Starting Phase 1.5 Comprehensive Tests")
    logger.info("=" * 60)

    test_results = {}

    try:
        # Test EnhancedSuccessTracker
        logger.info("\n📊 Testing EnhancedSuccessTracker...")
        tracker = EnhancedSuccessTracker("test_tracker")

        # Basic functionality test
        tracker.start_task_tracking("test_task", TaskType.SCRAPING, "https://test.com")
        tracker.complete_task("test_task", True, duration_seconds=1.0, quality_score=0.8)

        report = tracker.get_comprehensive_report()
        assert report['total_tasks'] == 1
        assert report['overall_success_rate'] == 1.0

        test_results['success_tracker'] = "✅ PASSED"
        logger.info("EnhancedSuccessTracker tests passed")

        # Test WorkflowState
        logger.info("\n🎯 Testing WorkflowState...")
        workflow = WorkflowState("test_workflow")

        # Target and progress test
        target_id = workflow.add_target(
            TargetType.TASK_COUNT, 3, "Test Target", is_primary=True
        )

        workflow.transition_to(WorkflowStatus.RUNNING, "Test start")

        for i in range(3):
            task_result = TaskResult(
                task_id=f"task_{i}",
                task_type=TaskType.SCRAPING,
                success=True,
                duration_seconds=1.0,
                url=f"https://test{i}.com",
                content_length=1000,
                quality_score=0.8
            )
            workflow.update_task_progress(task_result)

        summary = workflow.get_progress_summary()
        assert summary['total_tasks'] == 3
        assert summary['success_rate'] == 1.0
        assert summary['overall_progress_percentage'] == 100.0

        test_results['workflow_state'] = "✅ PASSED"
        logger.info("WorkflowState tests passed")

        # Test SimpleLifecycleManager
        logger.info("\n🔄 Testing SimpleLifecycleManager...")
        lifecycle = SimpleLifecycleManager("test_lifecycle", max_concurrent_tasks=2)

        # Task handler setup
        async def test_handler(task_lifecycle):
            await asyncio.sleep(0.1)
            return TaskResult(
                task_id=task_lifecycle.task_id,
                task_type=task_lifecycle.task_type,
                success=True,
                duration_seconds=0.1
            )

        lifecycle.set_task_handler(TaskType.SCRAPING, test_handler)

        # Task creation and execution test
        task_id = lifecycle.create_task(TaskType.SCRAPING, "Test Task")
        assert task_id in lifecycle.tasks

        await lifecycle.start_workflow()
        await asyncio.sleep(0.5)  # Wait for execution
        await lifecycle.stop_workflow("Test complete")

        status = lifecycle.get_workflow_status()
        assert status['total_tasks'] == 1

        test_results['lifecycle_manager'] = "✅ PASSED"
        logger.info("SimpleLifecycleManager tests passed")

        # Test Integration
        logger.info("\n🔗 Testing Integration...")

        class TestIntegration(WorkflowIntegrationMixin):
            def __init__(self):
                self.session_id = "test_integration"
                self.max_concurrent_tasks = 3
                super().__init__()

            def _setup_task_handlers(self):
                async def test_handler(task_lifecycle):
                    return TaskResult(
                        task_id=task_lifecycle.task_id,
                        task_type=task_lifecycle.task_type,
                        success=True,
                        duration_seconds=0.1
                    )
                self.lifecycle_manager.set_task_handler(TaskType.SCRAPING, test_handler)

        integration = TestIntegration()

        # Test task submission
        task_ids = await integration.submit_scraping_workflow(
            ["https://test1.com", "https://test2.com"],
            "test query"
        )
        assert len(task_ids) == 2

        # Test comprehensive status
        status = integration.get_comprehensive_status()
        assert 'workflow_state' in status
        assert 'success_tracker' in status
        assert 'lifecycle_manager' in status
        assert status['integration_status']['components_initialized'] is True

        test_results['integration'] = "✅ PASSED"
        logger.info("Integration tests passed")

        # Test Performance Monitoring
        logger.info("\n📈 Testing Performance Monitoring...")

        # Create monitoring setup
        tracker = EnhancedSuccessTracker("monitor_test")
        workflow = WorkflowState("monitor_test_workflow")
        lifecycle = SimpleLifecycleManager("monitor_test_lifecycle")

        # Connect components
        lifecycle.set_success_tracker(tracker)
        lifecycle.set_workflow_state(workflow)

        # Configure monitoring
        workflow.update_termination_criteria(min_success_rate=0.6)
        workflow.add_target(TargetType.SUCCESS_RATE, 0.7, "Monitor Target")

        # Test monitoring data collection
        workflow.transition_to(WorkflowStatus.RUNNING, "Monitor test start")

        task_result = TaskResult(
            task_id="monitor_task",
            task_type=TaskType.SCRAPING,
            success=True,
            duration_seconds=1.5,
            url="https://monitor.com",
            content_length=1000,
            quality_score=0.8
        )

        workflow.update_task_progress(task_result)
        tracker.complete_task("monitor_task", True, duration_seconds=1.5, quality_score=0.8)

        # Check monitoring reports
        tracker_report = tracker.get_comprehensive_report()
        workflow_summary = workflow.get_progress_summary()

        assert tracker_report['total_tasks'] == 1
        assert workflow_summary['total_tasks'] == 1
        assert tracker_report['overall_success_rate'] == 1.0

        test_results['performance_monitoring'] = "✅ PASSED"
        logger.info("Performance monitoring tests passed")

        logger.info("\n✅ All Phase 1.5 Tests Passed Successfully!")

    except Exception as e:
        logger.error(f"❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
        test_results['error'] = str(e)

    # Print test summary
    logger.info("\n📋 Test Summary:")
    for test_name, result in test_results.items():
        logger.info(f"  {test_name}: {result}")

    return test_results


if __name__ == "__main__":
    """Run all Phase 1.5 tests."""
    asyncio.run(run_all_tests())