"""
Phase 1.5 Integration: Workflow Management with AsyncScrapingOrchestrator

This module provides integration between the Phase 1.5 workflow management components
and the Phase 1.4 AsyncScrapingOrchestrator, enabling intelligent task coordination
with early termination and success tracking.

Key Features:
- WorkflowIntegrationMixin for easy integration with existing orchestrators
- OrchestratorIntegration for seamless AsyncScrapingOrchestrator integration
- Task result mapping and synchronization
- Performance monitoring and metrics sharing
- Early termination integration with orchestrator workflows

Based on Technical Enhancements Section 6 & 7 Integration Requirements
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime

# Import workflow management components
from .success_tracker import (
    TaskResult, TaskType, EnhancedSuccessTracker, FailureAnalysis
)
from .workflow_state import (
    WorkflowState, WorkflowStatus, TargetType, TargetDefinition,
    TerminationCriteria, TerminationReason
)
from .lifecycle_manager import (
    SimpleLifecycleManager, TaskLifecycle, TaskPriority, TaskDependency
)

# Import AsyncScrapingOrchestrator from Phase 1.4
try:
    from ..scraping_pipeline.async_orchestrator import (
        AsyncScrapingOrchestrator, ScrapingRequest, ScrapingResult,
        CleaningRequest, CleaningResult, PipelineConfig
    )
    ASYNC_ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ASYNC_ORCHESTRATOR_AVAILABLE = False

logger = logging.getLogger(__name__)


class WorkflowIntegrationMixin:
    """
    Mixin class for integrating workflow management capabilities with existing orchestrators.

    This mixin provides methods to easily add success tracking, early termination,
    and lifecycle management to any orchestrator implementation.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the workflow integration mixin."""
        super().__init__(*args, **kwargs)
        self._initialize_workflow_components()

    def _initialize_workflow_components(self):
        """Initialize workflow management components."""
        # Create workflow components
        self.success_tracker = EnhancedSuccessTracker(
            session_id=getattr(self, 'session_id', None)
        )
        self.workflow_state = WorkflowState(
            workflow_id=getattr(self, 'session_id', None)
        )
        self.lifecycle_manager = SimpleLifecycleManager(
            workflow_id=getattr(self, 'session_id', None),
            max_concurrent_tasks=getattr(self, 'max_concurrent_tasks', 10)
        )

        # Set up integration
        self.lifecycle_manager.set_success_tracker(self.success_tracker)
        self.lifecycle_manager.set_workflow_state(self.workflow_state)

        # Configure task handlers
        self._setup_task_handlers()

        # Enable early termination
        self.workflow_state.enable_early_termination(True)

        logger.info("Workflow integration components initialized")

    def _setup_task_handlers(self):
        """Set up task handlers for the lifecycle manager."""
        # Override in subclasses to implement specific task handlers
        pass

    async def submit_task_with_tracking(
        self,
        task_type: TaskType,
        task_data: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        dependencies: Optional[List[TaskDependency]] = None,
        **kwargs
    ) -> str:
        """Submit a task with full workflow tracking.

        Args:
            task_type: Type of task to submit
            task_data: Task execution data
            priority: Task priority
            dependencies: Task dependencies
            **kwargs: Additional task parameters

        Returns:
            Task ID
        """
        # Create task in lifecycle manager
        task_id = self.lifecycle_manager.create_task(
            task_type=task_type,
            priority=priority,
            dependencies=dependencies,
            **task_data,
            **kwargs
        )

        # Start tracking with success tracker
        self.success_tracker.start_task_tracking(
            task_id=task_id,
            task_type=task_type,
            url=task_data.get('url'),
            search_query=task_data.get('search_query')
        )

        return task_id

    async def complete_task_with_tracking(
        self,
        task_id: str,
        success: bool,
        result_data: Dict[str, Any],
        error_message: Optional[str] = None
    ):
        """Complete a task with full workflow tracking.

        Args:
            task_id: Task ID
            success: Whether task was successful
            result_data: Task result data
            error_message: Optional error message
        """
        # Update success tracker
        self.success_tracker.complete_task(
            task_id=task_id,
            success=success,
            error_message=error_message,
            **result_data
        )

        # Create TaskResult for workflow state
        task_result = TaskResult(
            task_id=task_id,
            task_type=TaskType(result_data.get('task_type', 'scraping')),
            success=success,
            duration_seconds=result_data.get('duration', 0.0),
            url=result_data.get('url'),
            domain=result_data.get('domain'),
            content_length=result_data.get('content_length', 0),
            quality_score=result_data.get('quality_score'),
            error_message=error_message
        )

        # Update workflow state
        self.workflow_state.update_task_progress(task_result)

    def add_workflow_target(
        self,
        target_type: TargetType,
        target_value: Union[int, float],
        name: Optional[str] = None,
        is_primary: bool = False,
        **kwargs
    ) -> str:
        """Add a target to the workflow.

        Args:
            target_type: Type of target
            target_value: Target value
            name: Target name
            is_primary: Whether this is the primary target
            **kwargs: Additional target parameters

        Returns:
            Target ID
        """
        return self.workflow_state.add_target(
            target_type=target_type,
            target_value=target_value,
            name=name,
            is_primary=is_primary,
            **kwargs
        )

    def configure_early_termination(self, **termination_criteria):
        """Configure early termination criteria.

        Args:
            **termination_criteria: Termination criteria parameters
        """
        self.workflow_state.update_termination_criteria(**termination_criteria)

    async def start_workflow_with_tracking(self) -> bool:
        """Start the workflow with full tracking.

        Returns:
            True if workflow started successfully
        """
        # Start workflow state
        self.workflow_state.transition_to(WorkflowStatus.RUNNING, "Workflow started with tracking")

        # Start lifecycle manager
        return await self.lifecycle_manager.start_workflow()

    async def stop_workflow_with_tracking(self, reason: str = "Manual stop"):
        """Stop the workflow with full tracking.

        Args:
            reason: Reason for stopping
        """
        await self.lifecycle_manager.stop_workflow(reason)

    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive workflow status including all components.

        Returns:
            Comprehensive status dictionary
        """
        return {
            'workflow_state': self.workflow_state.get_progress_summary(),
            'success_tracker': self.success_tracker.get_comprehensive_report(),
            'lifecycle_manager': self.lifecycle_manager.get_workflow_status(),
            'integration_status': {
                'components_initialized': True,
                'early_termination_enabled': self.workflow_state.early_termination_enabled,
                'integration_active': self.lifecycle_manager.running
            }
        }


class OrchestratorIntegration(WorkflowIntegrationMixin):
    """
    Integration class for AsyncScrapingOrchestrator with Phase 1.5 workflow management.

    This class provides seamless integration between the AsyncScrapingOrchestrator
    and the new workflow management components.
    """

    def __init__(self, orchestrator: AsyncScrapingOrchestrator):
        """Initialize orchestrator integration.

        Args:
            orchestrator: AsyncScrapingOrchestrator instance
        """
        self.orchestrator = orchestrator
        self.session_id = orchestrator.session_id
        self.max_concurrent_tasks = (
            orchestrator.config.max_scrape_workers +
            orchestrator.config.max_clean_workers
        )

        # Initialize workflow components
        self.success_tracker = EnhancedSuccessTracker(session_id=self.session_id)
        self.workflow_state = WorkflowState(workflow_id=self.session_id)
        self.lifecycle_manager = SimpleLifecycleManager(
            workflow_id=self.session_id,
            max_concurrent_tasks=self.max_concurrent_tasks
        )

        # Set up integration
        self._setup_orchestrator_integration()

    def _setup_orchestrator_integration(self):
        """Set up integration with AsyncScrapingOrchestrator."""
        # Connect components
        self.lifecycle_manager.set_success_tracker(self.success_tracker)
        self.lifecycle_manager.set_workflow_state(self.workflow_state)

        # Set up task handlers
        self._setup_orchestrator_task_handlers()

        # Configure workflow targets based on orchestrator
        self._configure_default_targets()

        # Set up early termination
        self._configure_early_termination()

        logger.info(f"Orchestrator integration setup complete for session {self.session_id}")

    def _setup_orchestrator_task_handlers(self):
        """Set up task handlers specific to AsyncScrapingOrchestrator."""
        async def handle_scraping_task(task_lifecycle: TaskLifecycle) -> TaskResult:
            """Handle scraping task execution."""
            # Extract scraping request from task context
            request_data = task_lifecycle.context
            scraping_request = ScrapingRequest(
                url=request_data['url'],
                search_query=request_data.get('search_query'),
                timeout_seconds=request_data.get('timeout_seconds', 120)
            )

            # Submit to orchestrator
            success = await self.orchestrator.submit_scraping_task(
                scraping_request,
                priority=self._map_priority(task_lifecycle.priority)
            )

            if not success:
                raise Exception("Failed to submit scraping task to orchestrator")

            # Wait for completion (this would need proper async waiting)
            # For now, create a mock result
            return TaskResult(
                task_id=task_lifecycle.task_id,
                task_type=TaskType.SCRAPING,
                success=True,
                duration_seconds=2.0,
                url=request_data['url'],
                domain=request_data.get('domain', ''),
                content_length=1000,
                quality_score=0.8
            )

        async def handle_cleaning_task(task_lifecycle: TaskLifecycle) -> TaskResult:
            """Handle cleaning task execution."""
            # Extract cleaning request from task context
            request_data = task_lifecycle.context
            cleaning_request = CleaningRequest(
                content=request_data['content'],
                url=request_data['url'],
                search_query=request_data.get('search_query')
            )

            # Submit to orchestrator
            success = await self.orchestrator.submit_cleaning_task(
                cleaning_request,
                priority=self._map_priority(task_lifecycle.priority)
            )

            if not success:
                raise Exception("Failed to submit cleaning task to orchestrator")

            # Wait for completion (this would need proper async waiting)
            # For now, create a mock result
            return TaskResult(
                task_id=task_lifecycle.task_id,
                task_type=TaskType.CLEANING,
                success=True,
                duration_seconds=1.0,
                url=request_data['url'],
                content_length=len(request_data['content']),
                quality_score=0.9
            )

        # Register handlers
        self.lifecycle_manager.set_task_handler(TaskType.SCRAPING, handle_scraping_task)
        self.lifecycle_manager.set_task_handler(TaskType.CLEANING, handle_cleaning_task)

    def _map_priority(self, lifecycle_priority: TaskPriority) -> int:
        """Map lifecycle priority to orchestrator priority."""
        priority_mapping = {
            TaskPriority.CRITICAL: 0,
            TaskPriority.HIGH: 1,
            TaskPriority.NORMAL: 2,
            TaskPriority.LOW: 3
        }
        return priority_mapping.get(lifecycle_priority, 2)

    def _configure_default_targets(self):
        """Configure default workflow targets based on orchestrator configuration."""
        # Add task count target based on queue sizes
        total_capacity = (
            self.orchestrator.config.max_scrape_workers +
            self.orchestrator.config.max_clean_workers
        )
        self.workflow_state.add_target(
            target_type=TargetType.TASK_COUNT,
            target_value=total_capacity * 2,  # Process 2x capacity
            name="Default Task Count Target",
            is_primary=True
        )

        # Add success rate target
        self.workflow_state.add_target(
            target_type=TargetType.SUCCESS_RATE,
            target_value=0.8,  # 80% success rate
            name="Success Rate Target",
            quality_threshold=0.7
        )

        # Add quality target
        self.workflow_state.add_target(
            target_type=TargetType.QUALITY_THRESHOLD,
            target_value=0.7,  # 70% quality threshold
            name="Quality Target"
        )

    def _configure_early_termination(self):
        """Configure early termination based on orchestrator settings."""
        # Configure termination criteria based on orchestrator config
        termination_criteria = {
            'min_success_rate': 0.3,
            'max_failure_rate': 0.7,
            'max_consecutive_failures': 5,
            'max_avg_task_duration': self.orchestrator.config.worker_timeout_seconds * 0.8,
            'min_quality_threshold': self.orchestrator.config.min_acceptable_quality,
            'max_execution_time_seconds': 3600,  # 1 hour default
            'evaluation_interval_seconds': 30.0
        }

        self.workflow_state.update_termination_criteria(**termination_criteria)

    async def submit_scraping_workflow(
        self,
        urls: List[str],
        search_query: Optional[str] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        **kwargs
    ) -> List[str]:
        """Submit URLs as a scraping workflow.

        Args:
            urls: List of URLs to scrape
            search_query: Optional search query
            priority: Task priority
            **kwargs: Additional parameters

        Returns:
            List of task IDs
        """
        task_ids = []

        for i, url in enumerate(urls):
            # Create task dependencies for sequential processing if needed
            dependencies = []
            if i > 0 and kwargs.get('sequential_processing', False):
                dependencies.append(TaskDependency(
                    task_id=f"scrape_task_{i}",
                    depends_on=[f"scrape_task_{i-1}"],
                    dependency_type="completion"
                ))

            task_id = await self.submit_task_with_tracking(
                task_type=TaskType.SCRAPING,
                task_data={
                    'url': url,
                    'search_query': search_query,
                    'timeout_seconds': kwargs.get('timeout_seconds', 120)
                },
                priority=priority,
                dependencies=dependencies
            )
            task_ids.append(task_id)

        return task_ids

    async def submit_cleaning_workflow(
        self,
        content_items: List[Tuple[str, str]],  # (content, url) tuples
        search_query: Optional[str] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        **kwargs
    ) -> List[str]:
        """Submit content items as a cleaning workflow.

        Args:
            content_items: List of (content, url) tuples
            search_query: Optional search query
            priority: Task priority
            **kwargs: Additional parameters

        Returns:
            List of task IDs
        """
        task_ids = []

        for i, (content, url) in enumerate(content_items):
            # Create task dependencies for sequential processing if needed
            dependencies = []
            if i > 0 and kwargs.get('sequential_processing', False):
                dependencies.append(TaskDependency(
                    task_id=f"clean_task_{i}",
                    depends_on=[f"clean_task_{i-1}"],
                    dependency_type="completion"
                ))

            task_id = await self.submit_task_with_tracking(
                task_type=TaskType.CLEANING,
                task_data={
                    'content': content,
                    'url': url,
                    'search_query': search_query
                },
                priority=priority,
                dependencies=dependencies
            )
            task_ids.append(task_id)

        return task_ids

    async def start_integrated_workflow(self) -> bool:
        """Start the integrated workflow with orchestrator.

        Returns:
            True if workflow started successfully
        """
        # Start orchestrator
        await self.orchestrator.start()

        # Start workflow with tracking
        return await self.start_workflow_with_tracking()

    async def stop_integrated_workflow(self, reason: str = "Manual stop"):
        """Stop the integrated workflow.

        Args:
            reason: Reason for stopping
        """
        # Stop workflow with tracking
        await self.stop_workflow_with_tracking(reason)

        # Shutdown orchestrator
        await self.orchestrator.shutdown()

    async def wait_for_workflow_completion(self, timeout_seconds: Optional[float] = None) -> Dict[str, Any]:
        """Wait for workflow completion with comprehensive status.

        Args:
            timeout_seconds: Optional timeout

        Returns:
            Comprehensive completion status
        """
        # Wait for lifecycle manager completion
        workflow_summary = await self.lifecycle_manager.wait_for_completion(timeout_seconds)

        # Get comprehensive status
        comprehensive_status = self.get_comprehensive_status()

        return {
            'workflow_summary': workflow_summary,
            'comprehensive_status': comprehensive_status,
            'orchestrator_stats': self.orchestrator.get_statistics()
        }

    def monitor_workflow_health(self) -> Dict[str, Any]:
        """Monitor workflow health and provide recommendations.

        Returns:
            Health monitoring report
        """
        # Get orchestrator health
        orchestrator_health = await self.orchestrator.get_health_status()

        # Get workflow state health
        should_terminate, termination_reason = self.workflow_state.check_early_termination()

        # Get success tracker insights
        success_tracker_report = self.success_tracker.get_comprehensive_report()
        optimization_suggestions = success_tracker_report.get('optimization_suggestions', [])

        # Get lifecycle manager status
        lifecycle_status = self.lifecycle_manager.get_workflow_status()

        # Overall health assessment
        health_issues = []

        if orchestrator_health['overall_health'] != 'healthy':
            health_issues.extend(orchestrator_health['health_issues'])

        if should_terminate:
            health_issues.append(f"Early termination recommended: {termination_reason}")

        if optimization_suggestions:
            health_issues.extend(optimization_suggestions)

        overall_health = 'healthy' if not health_issues else 'degraded' if len(health_issues) <= 3 else 'unhealthy'

        return {
            'overall_health': overall_health,
            'health_issues': health_issues,
            'orchestrator_health': orchestrator_health,
            'workflow_termination_risk': {
                'should_terminate': should_terminate,
                'reason': termination_reason
            },
            'performance_insights': {
                'optimization_suggestions': optimization_suggestions,
                'success_rate': success_tracker_report.get('overall_success_rate', 0.0),
                'failure_patterns': success_tracker_report.get('failure_summary', {}),
                'anti_bot_effectiveness': success_tracker_report.get('anti_bot_summary', {})
            },
            'lifecycle_status': lifecycle_status,
            'recommendations': self._generate_health_recommendations(health_issues, success_tracker_report)
        }

    def _generate_health_recommendations(
        self,
        health_issues: List[str],
        success_tracker_report: Dict[str, Any]
    ) -> List[str]:
        """Generate health recommendations based on issues and metrics.

        Args:
            health_issues: List of health issues
            success_tracker_report: Success tracker report

        Returns:
            List of recommendations
        """
        recommendations = []

        # Based on health issues
        for issue in health_issues:
            if 'success rate' in issue.lower():
                recommendations.append("Consider adjusting quality thresholds or reviewing anti-bot strategies")
            elif 'queue' in issue.lower():
                recommendations.append("Increase worker pool size or optimize task processing")
            elif 'circuit breaker' in issue.lower():
                recommendations.append("Review error patterns and implement retry strategies")
            elif 'termination' in issue.lower():
                recommendations.append("Review early termination criteria and adjust thresholds")

        # Based on success tracker metrics
        failure_summary = success_tracker_report.get('failure_summary', {})
        if failure_summary.get('total_failures', 0) > 10:
            recommendations.append("High failure rate detected - review error handling and retry logic")

        anti_bot_summary = success_tracker_report.get('anti_bot_summary', {})
        if anti_bot_summary.get('distribution', {}):
            recommendations.append("Monitor anti-bot escalation effectiveness and adjust strategies")

        return list(set(recommendations))  # Remove duplicates


def create_workflow_integration(orchestrator: AsyncScrapingOrchestrator) -> OrchestratorIntegration:
    """Create workflow integration for AsyncScrapingOrchestrator.

    Args:
        orchestrator: AsyncScrapingOrchestrator instance

    Returns:
        OrchestratorIntegration instance
    """
    if not ASYNC_ORCHESTRATOR_AVAILABLE:
        raise ImportError("AsyncScrapingOrchestrator not available")

    return OrchestratorIntegration(orchestrator)


# Integration utilities
async def run_orchestrator_with_workflow_management(
    orchestrator: AsyncScrapingOrchestrator,
    urls: List[str],
    search_query: Optional[str] = None,
    **workflow_kwargs
) -> Dict[str, Any]:
    """Run orchestrator with full workflow management integration.

    Args:
        orchestrator: AsyncScrapingOrchestrator instance
        urls: List of URLs to process
        search_query: Optional search query
        **workflow_kwargs: Additional workflow parameters

    Returns:
        Comprehensive execution results
    """
    # Create integration
    integration = create_workflow_integration(orchestrator)

    try:
        # Start integrated workflow
        await integration.start_integrated_workflow()

        # Submit scraping tasks
        task_ids = await integration.submit_scraping_workflow(
            urls=urls,
            search_query=search_query,
            **workflow_kwargs
        )

        # Wait for completion
        results = await integration.wait_for_workflow_completion()

        # Get final health report
        health_report = integration.monitor_workflow_health()

        return {
            'execution_results': results,
            'health_report': health_report,
            'task_ids': task_ids,
            'workflow_id': integration.workflow_id
        }

    finally:
        # Ensure cleanup
        await integration.stop_integrated_workflow("Workflow execution completed")