"""
Phase 1.5.3: Simple Lifecycle Manager for Task Coordination

This module implements SimpleLifecycleManager for task coordination and lifecycle management
without complex rollback mechanisms, focusing on simple and effective task orchestration.

Key Features:
- Simple task lifecycle management without rollback complexity
- Task coordination with dependency resolution
- Lifecycle event tracking and monitoring
- Resource management and cleanup
- Error recovery with retry logic
- Performance monitoring and optimization

Based on Technical Enhancements Section 7: Lifecycle Management & Edge Cases
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Set, Tuple, Union, Callable,
    DefaultDict, Deque
)
from uuid import uuid4

# Import enhanced logging from Phase 1.1
try:
    from ...agent_logging.enhanced_logger import (
        get_enhanced_logger, LogLevel, LogCategory, AgentEventType
    )
    ENHANCED_LOGGING_AVAILABLE = True
except ImportError:
    ENHANCED_LOGGING_AVAILABLE = False

# Import data contracts from Phase 1.4
try:
    from ..scraping_pipeline.data_contracts import (
        TaskStatus, TaskContext, PipelineStage
    )
    DATA_CONTRACTS_AVAILABLE = True
except ImportError:
    DATA_CONTRACTS_AVAILABLE = False

# Import workflow management from Phase 1.5
try:
    from .success_tracker import TaskResult, TaskType, EnhancedSuccessTracker
    from .workflow_state import WorkflowState, WorkflowStatus, TerminationReason
    WORKFLOW_MANAGEMENT_AVAILABLE = True
except ImportError:
    WORKFLOW_MANAGEMENT_AVAILABLE = False

logger = logging.getLogger(__name__)


class LifecycleStage(str, Enum):
    """Task lifecycle stages."""
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CLEANUP = "cleanup"


class LifecycleEvent(str, Enum):
    """Lifecycle event types."""
    TASK_CREATED = "task_created"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"
    TASK_RETRY = "task_retry"
    TASK_CLEANUP = "task_cleanup"
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    WORKFLOW_CANCELLED = "workflow_cancelled"


class TaskPriority(str, Enum):
    """Task priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TaskDependency:
    """Task dependency definition."""
    task_id: str
    depends_on: List[str] = field(default_factory=list)
    dependency_type: str = "completion"  # completion, success, data
    required: bool = True

    def is_satisfied(self, completed_tasks: Set[str], successful_tasks: Set[str]) -> bool:
        """Check if dependency is satisfied."""
        if not self.depends_on:
            return True

        if self.dependency_type == "completion":
            return all(dep_id in completed_tasks for dep_id in self.depends_on)
        elif self.dependency_type == "success":
            return all(dep_id in successful_tasks for dep_id in self.depends_on)
        else:
            # Default to completion requirement
            return all(dep_id in completed_tasks for dep_id in self.depends_on)


@dataclass
class TaskLifecycle:
    """Complete task lifecycle information."""

    # Basic task information
    task_id: str = field(default_factory=lambda: str(uuid4()))
    task_type: TaskType = TaskType.SCRAPING
    name: str = ""
    description: str = ""

    # Lifecycle stages
    current_stage: LifecycleStage = LifecycleStage.CREATED
    previous_stages: List[LifecycleStage] = field(default_factory=list)

    # Timing information
    created_at: datetime = field(default_factory=datetime.now)
    queued_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None

    # Task execution data
    priority: TaskPriority = TaskPriority.NORMAL
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: Optional[float] = None

    # Dependencies
    dependencies: List[TaskDependency] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)

    # Resource requirements
    estimated_memory_mb: Optional[float] = None
    estimated_duration_seconds: Optional[float] = None
    required_resources: Dict[str, Any] = field(default_factory=dict)

    # Execution context
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Results and errors
    result: Optional[TaskResult] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None

    # Cleanup
    cleanup_required: bool = False
    cleanup_actions: List[Callable] = field(default_factory=list)
    cleanup_completed: bool = False

    def transition_to(self, new_stage: LifecycleStage, timestamp: Optional[datetime] = None):
        """Transition task to a new lifecycle stage."""
        timestamp = timestamp or datetime.now()

        # Record previous stage
        if new_stage != self.current_stage:
            self.previous_stages.append(self.current_stage)

        # Update stage and timestamp
        self.current_stage = new_stage

        if new_stage == LifecycleStage.QUEUED:
            self.queued_at = timestamp
        elif new_stage == LifecycleStage.RUNNING:
            self.started_at = timestamp
        elif new_stage == LifecycleStage.COMPLETED:
            self.completed_at = timestamp
        elif new_stage == LifecycleStage.FAILED:
            self.failed_at = timestamp
        elif new_stage == LifecycleStage.CANCELLED:
            self.cancelled_at = timestamp
        elif new_stage == LifecycleStage.CLEANUP:
            # No specific timestamp for cleanup
            pass

    def get_duration(self) -> float:
        """Get total task duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            return (datetime.now() - self.started_at).total_seconds()
        else:
            return 0.0

    def is_finished(self) -> bool:
        """Check if task is in a finished state."""
        return self.current_stage in [LifecycleStage.COMPLETED, LifecycleStage.FAILED, LifecycleStage.CANCELLED]

    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return (
            self.current_stage == LifecycleStage.FAILED and
            self.retry_count < self.max_retries
        )

    def has_dependencies_satisfied(self, completed_tasks: Set[str], successful_tasks: Set[str]) -> bool:
        """Check if all task dependencies are satisfied."""
        return all(dep.is_satisfied(completed_tasks, successful_tasks) for dep in self.dependencies)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type.value,
            'name': self.name,
            'description': self.description,
            'current_stage': self.current_stage.value,
            'previous_stages': [stage.value for stage in self.previous_stages],
            'created_at': self.created_at.isoformat(),
            'queued_at': self.queued_at.isoformat() if self.queued_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'failed_at': self.failed_at.isoformat() if self.failed_at else None,
            'cancelled_at': self.cancelled_at.isoformat() if self.cancelled_at else None,
            'priority': self.priority.value,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'timeout_seconds': self.timeout_seconds,
            'dependencies': [
                {
                    'task_id': dep.task_id,
                    'depends_on': dep.depends_on,
                    'dependency_type': dep.dependency_type,
                    'required': dep.required
                }
                for dep in self.dependencies
            ],
            'dependents': self.dependents,
            'estimated_memory_mb': self.estimated_memory_mb,
            'estimated_duration_seconds': self.estimated_duration_seconds,
            'required_resources': self.required_resources,
            'context': self.context,
            'metadata': self.metadata,
            'result': self.result.to_dict() if self.result else None,
            'error_message': self.error_message,
            'error_type': self.error_type,
            'cleanup_required': self.cleanup_required,
            'cleanup_completed': self.cleanup_completed,
            'duration_seconds': self.get_duration(),
            'is_finished': self.is_finished(),
            'can_retry': self.can_retry()
        }


class SimpleLifecycleManager:
    """
    Simple lifecycle manager for task coordination and workflow orchestration.

    This manager provides simple and effective task lifecycle management without
    complex rollback mechanisms, focusing on reliability and performance.
    """

    def __init__(
        self,
        workflow_id: Optional[str] = None,
        max_concurrent_tasks: int = 10,
        enable_retry: bool = True,
        cleanup_interval_seconds: float = 60.0
    ):
        """Initialize the simple lifecycle manager.

        Args:
            workflow_id: Unique workflow identifier
            max_concurrent_tasks: Maximum concurrent tasks
            enable_retry: Enable task retry functionality
            cleanup_interval_seconds: Interval for cleanup operations
        """
        self.workflow_id = workflow_id or str(uuid4())
        self.max_concurrent_tasks = max_concurrent_tasks
        self.enable_retry = enable_retry
        self.cleanup_interval_seconds = cleanup_interval_seconds

        # Initialize enhanced logging
        self._setup_logging()

        # Task management
        self.tasks: Dict[str, TaskLifecycle] = {}
        self.active_tasks: Set[str] = set()
        self.completed_tasks: Set[str] = set()
        self.successful_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()

        # Queue management
        self.task_queue: Deque[str] = deque()
        self.running_tasks: Set[str] = set()
        self.priority_queue: Dict[TaskPriority, Deque[str]] = {
            priority: deque() for priority in TaskPriority
        }

        # Dependency tracking
        self.dependency_graph: Dict[str, List[str]] = defaultdict(list)
        self.reverse_dependency_graph: Dict[str, List[str]] = defaultdict(list)

        # Event tracking
        self.event_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = defaultdict(float)

        # Retry management
        self.retry_queue: Deque[str] = deque()
        self.retry_delays: List[float] = [1.0, 2.0, 5.0, 10.0]

        # Cleanup management
        self.cleanup_queue: Deque[str] = deque()
        self.last_cleanup_time = datetime.now()

        # Task execution
        self.task_handlers: Dict[TaskType, Callable] = {}
        self.running = False

        # Integration with other components
        self.success_tracker: Optional[EnhancedSuccessTracker] = None
        self.workflow_state: Optional[WorkflowState] = None

        logger.info(f"SimpleLifecycleManager initialized for workflow {self.workflow_id}")

    def _setup_logging(self):
        """Setup enhanced logging integration."""
        if ENHANCED_LOGGING_AVAILABLE:
            self.enhanced_logger = get_enhanced_logger("simple_lifecycle_manager")
            self.enhanced_logger.log_event(
                LogLevel.INFO,
                LogCategory.SYSTEM,
                AgentEventType.SESSION_START,
                "Simple Lifecycle Manager initialized",
                workflow_id=self.workflow_id
            )
        else:
            self.enhanced_logger = None

    def set_task_handler(self, task_type: TaskType, handler: Callable):
        """Set task handler for specific task type.

        Args:
            task_type: Type of task
            handler: Async function to handle the task
        """
        self.task_handlers[task_type] = handler
        logger.debug(f"Set handler for task type {task_type.value}")

    def set_success_tracker(self, success_tracker: EnhancedSuccessTracker):
        """Set success tracker integration."""
        self.success_tracker = success_tracker
        logger.debug("Integrated with EnhancedSuccessTracker")

    def set_workflow_state(self, workflow_state: WorkflowState):
        """Set workflow state integration."""
        self.workflow_state = workflow_state
        logger.debug("Integrated with WorkflowState")

    def create_task(
        self,
        task_type: TaskType,
        name: str = "",
        description: str = "",
        priority: TaskPriority = TaskPriority.NORMAL,
        dependencies: Optional[List[TaskDependency]] = None,
        max_retries: int = 3,
        timeout_seconds: Optional[float] = None,
        **kwargs
    ) -> str:
        """Create a new task.

        Args:
            task_type: Type of task
            name: Task name
            description: Task description
            priority: Task priority
            dependencies: List of task dependencies
            max_retries: Maximum retry attempts
            timeout_seconds: Task timeout
            **kwargs: Additional task parameters

        Returns:
            Task ID
        """
        task = TaskLifecycle(
            task_type=task_type,
            name=name or f"{task_type.value}_task",
            description=description,
            priority=priority,
            dependencies=dependencies or [],
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            **kwargs
        )

        self.tasks[task.task_id] = task

        # Update dependency graphs
        for dep in task.dependencies:
            for dep_id in dep.depends_on:
                self.dependency_graph[task.task_id].append(dep_id)
                self.reverse_dependency_graph[dep_id].append(task.task_id)

        # Add to appropriate queue
        self._add_task_to_queue(task.task_id, priority)

        # Log event
        self._log_event(LifecycleEvent.TASK_CREATED, task.task_id, {
            'task_type': task_type.value,
            'name': name,
            'priority': priority.value,
            'dependencies': [dep.depends_on for dep in (dependencies or [])]
        })

        logger.debug(f"Created task {task.task_id}: {task.name} ({task_type.value})")
        return task.task_id

    def _add_task_to_queue(self, task_id: str, priority: TaskPriority):
        """Add task to appropriate queue based on priority."""
        self.priority_queue[priority].append(task_id)

        # Also add to main queue for processing
        self.task_queue.append(task_id)

    def can_start_task(self, task_id: str) -> bool:
        """Check if task can be started.

        Args:
            task_id: Task ID

        Returns:
            True if task can be started
        """
        if task_id not in self.tasks:
            return False

        task = self.tasks[task_id]

        # Check if task is already running or finished
        if task_id in self.running_tasks or task.is_finished():
            return False

        # Check concurrent task limit
        if len(self.running_tasks) >= self.max_concurrent_tasks:
            return False

        # Check dependencies
        if not task.has_dependencies_satisfied(self.completed_tasks, self.successful_tasks):
            return False

        return True

    async def start_task(self, task_id: str) -> bool:
        """Start executing a task.

        Args:
            task_id: Task ID

        Returns:
            True if task was started successfully
        """
        if not self.can_start_task(task_id):
            return False

        task = self.tasks[task_id]
        task.transition_to(LifecycleStage.RUNNING)

        self.running_tasks.add(task_id)
        self.active_tasks.add(task_id)

        # Update workflow state
        if self.workflow_state:
            self.workflow_state.add_active_task(task_id)

        # Log event
        self._log_event(LifecycleEvent.TASK_STARTED, task_id, {
            'task_type': task.task_type.value,
            'priority': task.priority.value
        })

        # Start task execution
        asyncio.create_task(self._execute_task(task_id))

        logger.debug(f"Started task {task_id}: {task.name}")
        return True

    async def _execute_task(self, task_id: str):
        """Execute a task with timeout and error handling.

        Args:
            task_id: Task ID
        """
        task = self.tasks[task_id]
        start_time = time.time()

        try:
            # Get task handler
            handler = self.task_handlers.get(task.task_type)
            if not handler:
                raise ValueError(f"No handler registered for task type {task.task_type.value}")

            # Execute task with timeout
            if task.timeout_seconds:
                result = await asyncio.wait_for(
                    handler(task),
                    timeout=task.timeout_seconds
                )
            else:
                result = await handler(task)

            # Process successful result
            await self._complete_task(task_id, result, True)

        except asyncio.TimeoutError:
            await self._fail_task(task_id, "Task timeout", "TIMEOUT")
        except Exception as e:
            await self._fail_task(task_id, str(e), "EXECUTION_ERROR")

        # Update performance metrics
        duration = time.time() - start_time
        self.performance_metrics[f'avg_duration_{task.task_type.value}'] = (
            (self.performance_metrics[f'avg_duration_{task.task_type.value}'] + duration) / 2
        )

    async def _complete_task(self, task_id: str, result: Any, success: bool):
        """Complete a task with result.

        Args:
            task_id: Task ID
            result: Task result
            success: Whether task was successful
        """
        task = self.tasks[task_id]

        if success:
            task.transition_to(LifecycleStage.COMPLETED)
            self.completed_tasks.add(task_id)
            self.successful_tasks.add(task_id)

            # Store result if it's a TaskResult
            if isinstance(result, TaskResult):
                task.result = result

            # Update success tracker
            if self.success_tracker:
                self.success_tracker.complete_task(task_id, True, TaskStatus.COMPLETED)

            # Update workflow state
            if self.workflow_state and task.result:
                self.workflow_state.update_task_progress(task.result)

            # Log event
            self._log_event(LifecycleEvent.TASK_COMPLETED, task_id, {
                'task_type': task.task_type.value,
                'duration_seconds': task.get_duration()
            })

            logger.debug(f"Completed task {task_id}: {task.name}")

        else:
            await self._fail_task(task_id, str(result), "EXECUTION_ERROR")

        # Remove from running tasks
        self.running_tasks.discard(task_id)
        self.active_tasks.discard(task_id)

        # Update workflow state
        if self.workflow_state:
            self.workflow_state.remove_active_task(task_id)

        # Add to cleanup queue if needed
        if task.cleanup_required:
            self.cleanup_queue.append(task_id)

        # Start dependent tasks
        await self._start_dependent_tasks(task_id)

    async def _fail_task(self, task_id: str, error_message: str, error_type: str):
        """Fail a task with error information.

        Args:
            task_id: Task ID
            error_message: Error message
            error_type: Type of error
        """
        task = self.tasks[task_id]
        task.transition_to(LifecycleStage.FAILED)
        task.error_message = error_message
        task.error_type = error_type

        self.failed_tasks.add(task_id)
        self.completed_tasks.add(task_id)

        # Update success tracker
        if self.success_tracker:
            self.success_tracker.complete_task(
                task_id, False, TaskStatus.FAILED,
                error_message=error_message
            )

        # Update workflow state
        if self.workflow_state:
            # Create a basic task result for workflow state
            result = TaskResult(
                task_id=task_id,
                task_type=task.task_type,
                success=False,
                error_message=error_message,
                duration_seconds=task.get_duration()
            )
            self.workflow_state.update_task_progress(result)

        # Log event
        self._log_event(LifecycleEvent.TASK_FAILED, task_id, {
            'task_type': task.task_type.value,
            'error_message': error_message,
            'error_type': error_type,
            'retry_count': task.retry_count,
            'can_retry': task.can_retry()
        })

        logger.warning(f"Failed task {task_id}: {task.name} - {error_message}")

        # Handle retry
        if self.enable_retry and task.can_retry():
            await self._retry_task(task_id)

        # Remove from running tasks
        self.running_tasks.discard(task_id)
        self.active_tasks.discard(task_id)

        # Update workflow state
        if self.workflow_state:
            self.workflow_state.remove_active_task(task_id)

    async def _retry_task(self, task_id: str):
        """Retry a failed task.

        Args:
            task_id: Task ID
        """
        task = self.tasks[task_id]
        task.retry_count += 1

        # Calculate retry delay
        retry_delay = self.retry_delays[min(task.retry_count - 1, len(self.retry_delays) - 1)]

        # Schedule retry
        await asyncio.sleep(retry_delay)

        # Reset task state for retry
        task.transition_to(LifecycleStage.QUEUED)
        task.error_message = None
        task.error_type = None

        # Add back to queue
        self._add_task_to_queue(task_id, task.priority)

        # Log event
        self._log_event(LifecycleEvent.TASK_RETRY, task_id, {
            'retry_count': task.retry_count,
            'retry_delay': retry_delay,
            'max_retries': task.max_retries
        })

        logger.info(f"Retrying task {task_id}: {task.name} (attempt {task.retry_count}/{task.max_retries})")

    async def _start_dependent_tasks(self, completed_task_id: str):
        """Start tasks that depend on the completed task.

        Args:
            completed_task_id: ID of completed task
        """
        dependent_task_ids = self.reverse_dependency_graph.get(completed_task_id, [])

        for task_id in dependent_task_ids:
            if task_id in self.tasks and self.can_start_task(task_id):
                await self.start_task(task_id)

    def cancel_task(self, task_id: str, reason: str = "Manual cancellation") -> bool:
        """Cancel a task.

        Args:
            task_id: Task ID
            reason: Cancellation reason

        Returns:
            True if task was cancelled
        """
        if task_id not in self.tasks:
            return False

        task = self.tasks[task_id]

        # Can only cancel queued or running tasks
        if task.current_stage not in [LifecycleStage.QUEUED, LifecycleStage.RUNNING]:
            return False

        task.transition_to(LifecycleStage.CANCELLED)
        task.error_message = reason

        # Remove from active queues
        self.running_tasks.discard(task_id)
        self.active_tasks.discard(task_id)

        # Update workflow state
        if self.workflow_state:
            self.workflow_state.remove_active_task(task_id)

        # Log event
        self._log_event(LifecycleEvent.TASK_CANCELLED, task_id, {
            'reason': reason,
            'previous_stage': task.previous_stages[-1].value if task.previous_stages else None
        })

        logger.info(f"Cancelled task {task_id}: {task.name} - {reason}")
        return True

    async def start_workflow(self) -> bool:
        """Start the workflow execution.

        Returns:
            True if workflow started successfully
        """
        if self.running:
            return False

        self.running = True

        # Update workflow state
        if self.workflow_state:
            self.workflow_state.transition_to(WorkflowStatus.RUNNING, "Workflow started by lifecycle manager")

        # Log event
        self._log_event(LifecycleEvent.WORKFLOW_STARTED, self.workflow_id, {
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'total_tasks': len(self.tasks)
        })

        # Start task processing loop
        asyncio.create_task(self._process_tasks())

        # Start cleanup loop
        asyncio.create_task(self._cleanup_loop())

        logger.info(f"Started workflow {self.workflow_id}")
        return True

    async def stop_workflow(self, reason: str = "Manual stop"):
        """Stop the workflow execution.

        Args:
            reason: Reason for stopping
        """
        self.running = False

        # Cancel all running tasks
        running_task_ids = list(self.running_tasks)
        for task_id in running_task_ids:
            self.cancel_task(task_id, reason)

        # Update workflow state
        if self.workflow_state:
            final_status = WorkflowStatus.CANCELLED if reason != "Workflow completed" else WorkflowStatus.COMPLETED
            self.workflow_state.transition_to(final_status, reason)

        # Log event
        self._log_event(LifecycleEvent.WORKFLOW_CANCELLED if reason != "Workflow completed" else LifecycleEvent.WORKFLOW_COMPLETED, self.workflow_id, {
            'reason': reason,
            'total_tasks': len(self.tasks),
            'completed_tasks': len(self.completed_tasks),
            'successful_tasks': len(self.successful_tasks),
            'failed_tasks': len(self.failed_tasks)
        })

        logger.info(f"Stopped workflow {self.workflow_id}: {reason}")

    async def _process_tasks(self):
        """Main task processing loop."""
        while self.running:
            try:
                # Process tasks by priority
                for priority in [TaskPriority.CRITICAL, TaskPriority.HIGH, TaskPriority.NORMAL, TaskPriority.LOW]:
                    queue = self.priority_queue[priority]
                    while queue and len(self.running_tasks) < self.max_concurrent_tasks:
                        task_id = queue.popleft()
                        if self.can_start_task(task_id):
                            await self.start_task(task_id)

                # Check early termination
                if self.workflow_state:
                    should_terminate, reason, termination_reason = self.workflow_state.check_early_termination()
                    if should_terminate:
                        logger.info(f"Early termination triggered: {reason}")
                        await self.stop_workflow(f"Early termination: {reason}")
                        break

                # Wait before next iteration
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in task processing loop: {e}")
                await asyncio.sleep(1.0)

    async def _cleanup_loop(self):
        """Cleanup loop for finished tasks."""
        while self.running:
            try:
                current_time = datetime.now()

                # Check if cleanup is needed
                if (current_time - self.last_cleanup_time).total_seconds() >= self.cleanup_interval_seconds:
                    await self._perform_cleanup()
                    self.last_cleanup_time = current_time

                # Process cleanup queue
                while self.cleanup_queue:
                    task_id = self.cleanup_queue.popleft()
                    await self._cleanup_task(task_id)

                await asyncio.sleep(5.0)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(10.0)

    async def _perform_cleanup(self):
        """Perform periodic cleanup of finished tasks."""
        # Remove very old finished tasks to prevent memory buildup
        cutoff_time = datetime.now() - timedelta(hours=1)

        old_task_ids = [
            task_id for task_id, task in self.tasks.items()
            if task.is_finished() and task.completed_at and task.completed_at < cutoff_time
        ]

        for task_id in old_task_ids:
            await self._cleanup_task(task_id)

        if old_task_ids:
            logger.debug(f"Cleaned up {len(old_task_ids)} old tasks")

    async def _cleanup_task(self, task_id: str):
        """Cleanup a specific task.

        Args:
            task_id: Task ID
        """
        if task_id not in self.tasks:
            return

        task = self.tasks[task_id]
        task.transition_to(LifecycleStage.CLEANUP)

        # Execute cleanup actions
        for cleanup_action in task.cleanup_actions:
            try:
                if asyncio.iscoroutinefunction(cleanup_action):
                    await cleanup_action(task)
                else:
                    cleanup_action(task)
            except Exception as e:
                logger.warning(f"Cleanup action failed for task {task_id}: {e}")

        task.cleanup_completed = True

        # Log event
        self._log_event(LifecycleEvent.TASK_CLEANUP, task_id, {
            'cleanup_actions_count': len(task.cleanup_actions)
        })

        # Optionally remove task from memory (keep for debugging)
        # self.tasks.pop(task_id, None)

    def _log_event(self, event_type: LifecycleEvent, subject_id: str, metadata: Dict[str, Any]):
        """Log a lifecycle event.

        Args:
            event_type: Type of event
            subject_id: ID of the subject (task or workflow)
            metadata: Event metadata
        """
        event = {
            'event_type': event_type.value,
            'subject_id': subject_id,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata
        }

        self.event_history.append(event)

        # Keep only recent events
        if len(self.event_history) > 1000:
            self.event_history = self.event_history[-500:]

        # Log with enhanced logger if available
        if self.enhanced_logger:
            log_level = LogLevel.INFO if event_type in [LifecycleEvent.TASK_COMPLETED, LifecycleEvent.WORKFLOW_COMPLETED] else LogLevel.WARNING if event_type in [LifecycleEvent.TASK_FAILED, LifecycleEvent.TASK_CANCELLED] else LogLevel.DEBUG

            self.enhanced_logger.log_event(
                log_level,
                LogCategory.PERFORMANCE,
                AgentEventType.TASK_END if event_type == LifecycleEvent.TASK_COMPLETED else AgentEventType.TASK_START if event_type == LifecycleEvent.TASK_STARTED else AgentEventType.ERROR if event_type == LifecycleEvent.TASK_FAILED else AgentEventType.SESSION_START if event_type == LifecycleEvent.WORKFLOW_STARTED else AgentEventType.SESSION_END,
                f"Lifecycle event: {event_type.value}",
                subject_id=subject_id,
                event_type=event_type.value,
                metadata=metadata
            )

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get comprehensive workflow status.

        Returns:
            Workflow status dictionary
        """
        # Calculate task statistics
        total_tasks = len(self.tasks)
        finished_tasks = len(self.completed_tasks)
        running_tasks = len(self.running_tasks)
        queued_tasks = sum(len(queue) for queue in self.priority_queue.values())

        # Calculate success rate
        success_rate = len(self.successful_tasks) / finished_tasks if finished_tasks > 0 else 0.0

        # Get recent events
        recent_events = self.event_history[-20:] if self.event_history else []

        return {
            'workflow_id': self.workflow_id,
            'running': self.running,
            'timestamp': datetime.now().isoformat(),

            # Task statistics
            'total_tasks': total_tasks,
            'finished_tasks': finished_tasks,
            'successful_tasks': len(self.successful_tasks),
            'failed_tasks': len(self.failed_tasks),
            'running_tasks': running_tasks,
            'queued_tasks': queued_tasks,
            'success_rate': success_rate,

            # Performance metrics
            'performance_metrics': dict(self.performance_metrics),

            # Queue status
            'queue_status': {
                priority.value: len(queue) for priority, queue in self.priority_queue.items()
            },

            # Recent events
            'recent_events': recent_events,

            # Cleanup status
            'cleanup_queue_size': len(self.cleanup_queue),
            'last_cleanup_time': self.last_cleanup_time.isoformat()
        }

    def get_task_details(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a task.

        Args:
            task_id: Task ID

        Returns:
            Task details or None if not found
        """
        task = self.tasks.get(task_id)
        if not task:
            return None

        # Add dependency information
        dependencies_info = []
        for dep in task.dependencies:
            dependencies_info.append({
                'depends_on': dep.depends_on,
                'dependency_type': dep.dependency_type,
                'required': dep.required,
                'satisfied': dep.is_satisfied(self.completed_tasks, self.successful_tasks)
            })

        # Get dependent tasks
        dependent_tasks = [
            dep_id for dep_id in self.reverse_dependency_graph.get(task_id, [])
            if dep_id in self.tasks
        ]

        return {
            **task.to_dict(),
            'dependencies_info': dependencies_info,
            'dependent_tasks': dependent_tasks,
            'can_start': self.can_start_task(task_id)
        }

    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get workflow execution summary.

        Returns:
            Workflow summary dictionary
        """
        # Task type breakdown
        task_type_stats = defaultdict(lambda: {'total': 0, 'successful': 0, 'failed': 0})
        for task in self.tasks.values():
            task_type_stats[task.task_type.value]['total'] += 1
            if task.task_id in self.successful_tasks:
                task_type_stats[task.task_type.value]['successful'] += 1
            if task.task_id in self.failed_tasks:
                task_type_stats[task.task_type.value]['failed'] += 1

        # Priority breakdown
        priority_stats = defaultdict(int)
        for task in self.tasks.values():
            priority_stats[task.priority.value] += 1

        # Performance summary
        avg_duration = 0.0
        completed_tasks_with_duration = [
            task for task in self.tasks.values()
            if task.is_finished() and task.get_duration() > 0
        ]
        if completed_tasks_with_duration:
            avg_duration = sum(task.get_duration() for task in completed_tasks_with_duration) / len(completed_tasks_with_duration)

        return {
            'workflow_id': self.workflow_id,
            'execution_time_seconds': (datetime.now() - self.tasks[next(iter(self.tasks))].created_at).total_seconds() if self.tasks else 0.0,
            'total_tasks': len(self.tasks),
            'successful_tasks': len(self.successful_tasks),
            'failed_tasks': len(self.failed_tasks),
            'success_rate': len(self.successful_tasks) / len(self.tasks) if self.tasks else 0.0,
            'average_task_duration_seconds': avg_duration,
            'task_type_breakdown': dict(task_type_stats),
            'priority_breakdown': dict(priority_stats),
            'total_events': len(self.event_history),
            'performance_metrics': dict(self.performance_metrics)
        }

    async def wait_for_completion(self, timeout_seconds: Optional[float] = None) -> Dict[str, Any]:
        """Wait for workflow completion.

        Args:
            timeout_seconds: Optional timeout

        Returns:
            Completion status
        """
        start_time = time.time()

        while self.running:
            if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                await self.stop_workflow("Timeout waiting for completion")
                break

            # Check if all tasks are finished
            if len(self.completed_tasks) == len(self.tasks):
                await self.stop_workflow("Workflow completed")
                break

            await asyncio.sleep(1.0)

        return self.get_workflow_summary()

    def add_cleanup_action(self, task_id: str, cleanup_action: Callable):
        """Add cleanup action to a task.

        Args:
            task_id: Task ID
            cleanup_action: Cleanup function to execute
        """
        if task_id in self.tasks:
            self.tasks[task_id].cleanup_actions.append(cleanup_action)
            self.tasks[task_id].cleanup_required = True
            logger.debug(f"Added cleanup action to task {task_id}")

    def set_max_concurrent_tasks(self, max_concurrent: int):
        """Update maximum concurrent tasks.

        Args:
            max_concurrent: New maximum concurrent tasks
        """
        self.max_concurrent_tasks = max_concurrent
        logger.info(f"Updated max concurrent tasks to {max_concurrent}")