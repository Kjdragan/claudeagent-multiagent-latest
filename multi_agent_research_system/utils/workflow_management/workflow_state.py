"""
Phase 1.5.2: Workflow State with Early Termination Logic

This module implements WorkflowState with intelligent early termination logic,
target achievement detection, and state management for optimizing workflow execution.

Key Features:
- TargetDefinition with configurable success criteria
- TerminationCriteria with intelligent early termination
- StateTransition tracking with rollback capabilities
- Target achievement detection and monitoring
- Performance-based termination logic
- Resource utilization monitoring

Based on Technical Enhancements Section 7: Lifecycle Management & Edge Cases
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Set, Tuple, Union, Callable,
    NamedTuple
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
        TaskStatus, PipelineStage
    )
    DATA_CONTRACTS_AVAILABLE = True
except ImportError:
    DATA_CONTRACTS_AVAILABLE = False

# Import success tracking from Phase 1.5.1
try:
    from .success_tracker import TaskResult, TaskType, SuccessMetrics
    SUCCESS_TRACKING_AVAILABLE = True
except ImportError:
    SUCCESS_TRACKING_AVAILABLE = False

logger = logging.getLogger(__name__)


class WorkflowStatus(str, Enum):
    """Workflow execution status."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    TERMINATING = "terminating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TerminationReason(str, Enum):
    """Reason for workflow termination."""
    TARGET_ACHIEVED = "target_achieved"
    PERFORMANCE_THRESHOLD = "performance_threshold"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    ERROR_THRESHOLD = "error_threshold"
    TIME_LIMIT = "time_limit"
    MANUAL_TERMINATION = "manual_termination"
    EARLY_TERMINATION = "early_termination"
    SYSTEM_SHUTDOWN = "system_shutdown"


class TargetType(str, Enum):
    """Types of targets for workflow completion."""
    TASK_COUNT = "task_count"
    SUCCESS_RATE = "success_rate"
    QUALITY_THRESHOLD = "quality_threshold"
    TIME_LIMIT = "time_limit"
    RESOURCE_LIMIT = "resource_limit"
    CUSTOM = "custom"


class StateTransition(NamedTuple):
    """State transition record."""
    from_status: WorkflowStatus
    to_status: WorkflowStatus
    timestamp: datetime
    reason: str
    metadata: Dict[str, Any]


@dataclass
class TargetDefinition:
    """Definition of workflow completion targets."""

    # Basic target identification
    target_id: str = field(default_factory=lambda: str(uuid4()))
    target_type: TargetType = TargetType.TASK_COUNT
    name: str = "Default Target"
    description: str = ""

    # Target values and thresholds
    target_value: Union[int, float] = 0
    current_value: Union[int, float] = 0
    min_threshold: Optional[Union[int, float]] = None
    max_threshold: Optional[Union[int, float]] = None

    # Progress tracking
    progress_percentage: float = 0.0
    achieved: bool = False
    achievement_time: Optional[datetime] = None

    # Quality requirements
    quality_threshold: Optional[float] = None
    current_quality: Optional[float] = None

    # Time constraints
    time_limit_seconds: Optional[float] = None
    start_time: Optional[datetime] = None
    time_remaining: Optional[float] = None

    # Custom evaluation
    custom_evaluator: Optional[Callable[[Any], bool]] = None
    evaluation_data: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_progress(self, new_value: Union[int, float], quality_score: Optional[float] = None):
        """Update target progress and check achievement."""
        self.current_value = new_value
        self.updated_at = datetime.now()

        # Update quality if provided
        if quality_score is not None:
            self.current_quality = quality_score

        # Calculate progress percentage
        if self.target_value != 0:
            self.progress_percentage = min(100.0, (new_value / self.target_value) * 100)

        # Update time remaining
        if self.start_time and self.time_limit_seconds:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            self.time_remaining = max(0, self.time_limit_seconds - elapsed)

        # Check achievement
        self._check_achievement()

    def _check_achievement(self):
        """Check if target has been achieved."""
        if self.achieved:
            return

        # Check basic target achievement
        target_met = False

        if self.target_type == TargetType.TASK_COUNT:
            target_met = self.current_value >= self.target_value
        elif self.target_type == TargetType.SUCCESS_RATE:
            target_met = self.current_value >= self.target_value
        elif self.target_type == TargetType.QUALITY_THRESHOLD:
            target_met = (self.current_quality or 0) >= self.target_value
        elif self.target_type == TargetType.TIME_LIMIT:
            target_met = self.time_remaining is not None and self.time_remaining <= 0
        elif self.target_type == TargetType.RESOURCE_LIMIT:
            target_met = self.current_value >= self.target_value
        elif self.target_type == TargetType.CUSTOM and self.custom_evaluator:
            target_met = self.custom_evaluator(self.evaluation_data)

        # Check quality threshold if required
        if target_met and self.quality_threshold:
            target_met = (self.current_quality or 0) >= self.quality_threshold

        # Check thresholds
        if target_met:
            if self.min_threshold is not None and self.current_value < self.min_threshold:
                target_met = False
            if self.max_threshold is not None and self.current_value > self.max_threshold:
                target_met = False

        if target_met:
            self.achieved = True
            self.achievement_time = datetime.now()

    def is_time_exceeded(self) -> bool:
        """Check if time limit has been exceeded."""
        if not self.start_time or not self.time_limit_seconds:
            return False

        elapsed = (datetime.now() - self.start_time).total_seconds()
        return elapsed > self.time_limit_seconds

    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive target status summary."""
        return {
            'target_id': self.target_id,
            'name': self.name,
            'target_type': self.target_type.value,
            'target_value': self.target_value,
            'current_value': self.current_value,
            'progress_percentage': self.progress_percentage,
            'achieved': self.achieved,
            'achievement_time': self.achievement_time.isoformat() if self.achievement_time else None,
            'quality_threshold': self.quality_threshold,
            'current_quality': self.current_quality,
            'time_limit_seconds': self.time_limit_seconds,
            'time_remaining': self.time_remaining,
            'is_time_exceeded': self.is_time_exceeded(),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


@dataclass
class TerminationCriteria:
    """Criteria for workflow termination."""

    # Performance-based criteria
    min_success_rate: float = 0.3
    max_failure_rate: float = 0.7
    max_consecutive_failures: int = 5
    max_avg_task_duration: float = 300.0  # 5 minutes
    min_quality_threshold: float = 0.5

    # Resource-based criteria
    max_memory_usage_mb: float = 2048.0
    max_cpu_usage_percent: float = 90.0
    max_disk_usage_mb: float = 10240.0  # 10GB
    max_network_errors: int = 10

    # Time-based criteria
    max_execution_time_seconds: Optional[float] = None
    max_idle_time_seconds: float = 300.0  # 5 minutes
    task_timeout_seconds: float = 600.0  # 10 minutes

    # Quality-based criteria
    min_avg_quality_score: float = 0.6
    quality_degradation_threshold: float = 0.2
    max_quality_volatility: float = 0.3

    # Error-based criteria
    max_error_rate: float = 0.5
    max_critical_errors: int = 3
    error_pattern_threshold: int = 5

    # Custom criteria
    custom_termination_checks: List[Callable[[], Tuple[bool, str]]] = field(default_factory=list)

    # Monitoring settings
    evaluation_interval_seconds: float = 30.0
    enable_adaptive_thresholds: bool = True
    adaptive_sensitivity: float = 0.1

    def should_terminate(self, metrics: Dict[str, Any]) -> Tuple[bool, str, TerminationReason]:
        """Check if workflow should terminate based on criteria.

        Args:
            metrics: Current workflow metrics

        Returns:
            Tuple of (should_terminate, reason, termination_reason)
        """
        # Performance-based checks
        success_rate = metrics.get('success_rate', 1.0)
        if success_rate < self.min_success_rate:
            return True, f"Success rate ({success_rate:.1%}) below minimum ({self.min_success_rate:.1%})", TerminationReason.PERFORMANCE_THRESHOLD

        failure_rate = metrics.get('failure_rate', 0.0)
        if failure_rate > self.max_failure_rate:
            return True, f"Failure rate ({failure_rate:.1%}) above maximum ({self.max_failure_rate:.1%})", TerminationReason.ERROR_THRESHOLD

        consecutive_failures = metrics.get('consecutive_failures', 0)
        if consecutive_failures >= self.max_consecutive_failures:
            return True, f"Too many consecutive failures ({consecutive_failures})", TerminationReason.ERROR_THRESHOLD

        avg_duration = metrics.get('avg_task_duration', 0.0)
        if avg_duration > self.max_avg_task_duration:
            return True, f"Average task duration ({avg_duration:.1f}s) exceeds maximum ({self.max_avg_task_duration:.1f}s)", TerminationReason.PERFORMANCE_THRESHOLD

        avg_quality = metrics.get('avg_quality_score', 1.0)
        if avg_quality < self.min_quality_threshold:
            return True, f"Average quality score ({avg_quality:.2f}) below minimum ({self.min_quality_threshold:.2f})", TerminationReason.PERFORMANCE_THRESHOLD

        # Resource-based checks
        memory_usage = metrics.get('memory_usage_mb', 0.0)
        if memory_usage > self.max_memory_usage_mb:
            return True, f"Memory usage ({memory_usage:.1f}MB) exceeds maximum ({self.max_memory_usage_mb:.1f}MB)", TerminationReason.RESOURCE_EXHAUSTED

        cpu_usage = metrics.get('cpu_usage_percent', 0.0)
        if cpu_usage > self.max_cpu_usage_percent:
            return True, f"CPU usage ({cpu_usage:.1f}%) exceeds maximum ({self.max_cpu_usage_percent:.1f}%)", TerminationReason.RESOURCE_EXHAUSTED

        network_errors = metrics.get('network_errors', 0)
        if network_errors > self.max_network_errors:
            return True, f"Too many network errors ({network_errors})", TerminationReason.ERROR_THRESHOLD

        # Time-based checks
        execution_time = metrics.get('execution_time_seconds', 0.0)
        if self.max_execution_time_seconds and execution_time > self.max_execution_time_seconds:
            return True, f"Execution time ({execution_time:.1f}s) exceeds maximum ({self.max_execution_time_seconds:.1f}s)", TerminationReason.TIME_LIMIT

        idle_time = metrics.get('idle_time_seconds', 0.0)
        if idle_time > self.max_idle_time_seconds:
            return True, f"Idle time ({idle_time:.1f}s) exceeds maximum ({self.max_idle_time_seconds:.1f}s)", TerminationReason.TIME_LIMIT

        # Quality-based checks
        quality_volatility = metrics.get('quality_volatility', 0.0)
        if quality_volatility > self.max_quality_volatility:
            return True, f"Quality volatility ({quality_volatility:.2f}) exceeds maximum ({self.max_quality_volatility:.2f})", TerminationReason.PERFORMANCE_THRESHOLD

        quality_degradation = metrics.get('quality_degradation', 0.0)
        if quality_degradation > self.quality_degradation_threshold:
            return True, f"Quality degradation ({quality_degradation:.2f}) exceeds threshold ({self.quality_degradation_threshold:.2f})", TerminationReason.PERFORMANCE_THRESHOLD

        # Error-based checks
        error_rate = metrics.get('error_rate', 0.0)
        if error_rate > self.max_error_rate:
            return True, f"Error rate ({error_rate:.1%}) exceeds maximum ({self.max_error_rate:.1%})", TerminationReason.ERROR_THRESHOLD

        critical_errors = metrics.get('critical_errors', 0)
        if critical_errors >= self.max_critical_errors:
            return True, f"Too many critical errors ({critical_errors})", TerminationReason.ERROR_THRESHOLD

        # Custom termination checks
        for check_func in self.custom_termination_checks:
            try:
                should_terminate, reason = check_func()
                if should_terminate:
                    return True, reason, TerminationReason.CUSTOM
            except Exception as e:
                logger.warning(f"Custom termination check failed: {e}")

        return False, "All termination criteria within acceptable ranges", TerminationReason.TARGET_ACHIEVED

    def adapt_thresholds(self, performance_history: List[float]):
        """Adapt termination thresholds based on performance history."""
        if not self.enable_adaptive_thresholds or len(performance_history) < 10:
            return

        # Calculate performance trend
        recent_performance = performance_history[-5:]
        older_performance = performance_history[-10:-5]

        recent_avg = sum(recent_performance) / len(recent_performance)
        older_avg = sum(older_performance) / len(older_performance)

        # Adjust thresholds based on trend
        if recent_avg < older_avg:
            # Performance degrading, tighten thresholds
            self.min_success_rate = min(0.8, self.min_success_rate + self.adaptive_sensitivity)
            self.max_failure_rate = max(0.2, self.max_failure_rate - self.adaptive_sensitivity)
            logger.info(f"Adapted thresholds due to performance degradation: success_rate={self.min_success_rate:.2f}, failure_rate={self.max_failure_rate:.2f}")
        elif recent_avg > older_avg:
            # Performance improving, relax thresholds slightly
            self.min_success_rate = max(0.1, self.min_success_rate - self.adaptive_sensitivity * 0.5)
            self.max_failure_rate = min(0.9, self.max_failure_rate + self.adaptive_sensitivity * 0.5)
            logger.info(f"Adapted thresholds due to performance improvement: success_rate={self.min_success_rate:.2f}, failure_rate={self.max_failure_rate:.2f}")


class WorkflowState:
    """
    Workflow state manager with intelligent early termination logic and target achievement detection.

    This class provides comprehensive state management, progress tracking, and intelligent
    early termination to optimize workflow execution and prevent wasted work.
    """

    def __init__(
        self,
        workflow_id: Optional[str] = None,
        termination_criteria: Optional[TerminationCriteria] = None
    ):
        """Initialize workflow state manager.

        Args:
            workflow_id: Unique workflow identifier
            termination_criteria: Custom termination criteria
        """
        self.workflow_id = workflow_id or str(uuid4())
        self.termination_criteria = termination_criteria or TerminationCriteria()

        # Initialize enhanced logging
        self._setup_logging()

        # State management
        self.status = WorkflowStatus.INITIALIZING
        self.state_history: List[StateTransition] = []
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None

        # Target management
        self.targets: Dict[str, TargetDefinition] = {}
        self.primary_target_id: Optional[str] = None

        # Progress tracking
        self.total_tasks = 0
        self.completed_tasks = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.active_tasks: Set[str] = set()

        # Performance metrics
        self.performance_history: List[float] = []
        self.last_activity_time = datetime.now()
        self.termination_checks_count = 0

        # Early termination data
        self.termination_recommendations: List[Tuple[bool, str, TerminationReason]] = []
        self.early_termination_enabled = True

        # Resource monitoring
        self.current_metrics: Dict[str, Any] = {}
        self.metrics_history: List[Dict[str, Any]] = []

        # Task results tracking
        self.task_results: Dict[str, TaskResult] = {}

        logger.info(f"WorkflowState initialized for workflow {self.workflow_id}")

    def _setup_logging(self):
        """Setup enhanced logging integration."""
        if ENHANCED_LOGGING_AVAILABLE:
            self.enhanced_logger = get_enhanced_logger("workflow_state")
            self.enhanced_logger.log_event(
                LogLevel.INFO,
                LogCategory.SYSTEM,
                AgentEventType.SESSION_START,
                "Workflow State initialized",
                workflow_id=self.workflow_id
            )
        else:
            self.enhanced_logger = None

    def add_target(
        self,
        target_type: TargetType,
        target_value: Union[int, float],
        name: Optional[str] = None,
        is_primary: bool = False,
        **kwargs
    ) -> str:
        """Add a new target to the workflow.

        Args:
            target_type: Type of target
            target_value: Target value to achieve
            name: Optional target name
            is_primary: Whether this is the primary target
            **kwargs: Additional target parameters

        Returns:
            Target ID
        """
        target = TargetDefinition(
            target_type=target_type,
            target_value=target_value,
            name=name or f"{target_type.value}_target",
            **kwargs
        )

        self.targets[target.target_id] = target

        if is_primary or not self.primary_target_id:
            self.primary_target_id = target.target_id

        logger.debug(f"Added target {target.target_id}: {target.name} ({target_type.value} = {target_value})")
        return target.target_id

    def transition_to(self, new_status: WorkflowStatus, reason: str = "", metadata: Optional[Dict[str, Any]] = None):
        """Transition workflow to a new status.

        Args:
            new_status: New workflow status
            reason: Reason for transition
            metadata: Optional transition metadata
        """
        old_status = self.status
        self.status = new_status

        # Record transition
        transition = StateTransition(
            from_status=old_status,
            to_status=new_status,
            timestamp=datetime.now(),
            reason=reason or f"Transition from {old_status.value} to {new_status.value}",
            metadata=metadata or {}
        )
        self.state_history.append(transition)

        # Handle special transitions
        if new_status == WorkflowStatus.RUNNING and not self.started_at:
            self.started_at = datetime.now()
            # Set start time for time-based targets
            for target in self.targets.values():
                if target.time_limit_seconds:
                    target.start_time = self.started_at

        elif new_status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]:
            self.completed_at = datetime.now()

        # Log transition
        if self.enhanced_logger:
            self.enhanced_logger.log_event(
                LogLevel.INFO,
                LogCategory.SYSTEM,
                AgentEventType.SESSION_START if new_status == WorkflowStatus.RUNNING else AgentEventType.SESSION_END,
                f"Workflow transitioned from {old_status.value} to {new_status.value}",
                workflow_id=self.workflow_id,
                old_status=old_status.value,
                new_status=new_status.value,
                reason=reason
            )

        logger.info(f"Workflow {self.workflow_id} transitioned: {old_status.value} → {new_status.value} ({reason})")

    def update_task_progress(self, task_result: TaskResult):
        """Update workflow progress based on task result.

        Args:
            task_result: Result of completed task
        """
        # Store task result
        self.task_results[task_result.task_id] = task_result

        # Update task counters
        self.total_tasks += 1
        if task_result.success:
            self.successful_tasks += 1
        else:
            self.failed_tasks += 1

        # Update active tasks
        self.active_tasks.discard(task_result.task_id)
        self.completed_tasks += 1

        # Update last activity time
        self.last_activity_time = datetime.now()

        # Update performance history
        if task_result.success:
            self.performance_history.append(1.0)
        else:
            self.performance_history.append(0.0)

        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-50:]

        # Update targets
        self._update_targets(task_result)

        # Update current metrics
        self._update_current_metrics()

        # Check if primary target achieved
        self._check_target_achievement()

    def _update_targets(self, task_result: TaskResult):
        """Update all targets based on task result."""
        # Update task count target
        for target in self.targets.values():
            if target.target_type == TargetType.TASK_COUNT:
                target.update_progress(self.completed_tasks, task_result.quality_score)

            elif target.target_type == TargetType.SUCCESS_RATE:
                success_rate = self.successful_tasks / self.total_tasks if self.total_tasks > 0 else 0.0
                target.update_progress(success_rate, task_result.quality_score)

            elif target.target_type == TargetType.QUALITY_THRESHOLD:
                avg_quality = self._calculate_average_quality()
                target.update_progress(avg_quality, avg_quality)

            elif target.target_type == TargetType.CUSTOM:
                # Update evaluation data for custom targets
                target.evaluation_data.update({
                    'task_result': task_result.to_dict(),
                    'completed_tasks': self.completed_tasks,
                    'successful_tasks': self.successful_tasks,
                    'failed_tasks': self.failed_tasks
                })

    def _calculate_average_quality(self) -> float:
        """Calculate average quality score from all task results."""
        quality_scores = [
            result.quality_score for result in self.task_results.values()
            if result.quality_score is not None
        ]
        return sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    def _check_target_achievement(self):
        """Check if any targets have been achieved."""
        for target in self.targets.values():
            if target.achieved and target.target_id == self.primary_target_id:
                self.transition_to(
                    WorkflowStatus.COMPLETED,
                    f"Primary target achieved: {target.name}",
                    {'target_id': target.target_id, 'achievement_time': target.achievement_time.isoformat()}
                )
                break

    def _update_current_metrics(self):
        """Update current workflow metrics."""
        # Calculate basic metrics
        success_rate = self.successful_tasks / self.total_tasks if self.total_tasks > 0 else 1.0
        failure_rate = self.failed_tasks / self.total_tasks if self.total_tasks > 0 else 0.0

        # Calculate consecutive failures
        consecutive_failures = 0
        recent_results = sorted(
            [r for r in self.task_results.values() if r.completed_at],
            key=lambda x: x.completed_at or datetime.min,
            reverse=True
        )
        for result in recent_results:
            if not result.success:
                consecutive_failures += 1
            else:
                break

        # Calculate average task duration
        completed_results = [r for r in self.task_results.values() if r.duration_seconds > 0]
        avg_duration = sum(r.duration_seconds for r in completed_results) / len(completed_results) if completed_results else 0.0

        # Calculate idle time
        idle_time = (datetime.now() - self.last_activity_time).total_seconds()

        # Calculate execution time
        execution_time = 0.0
        if self.started_at:
            execution_time = (datetime.now() - self.started_at).total_seconds()

        # Calculate quality metrics
        avg_quality = self._calculate_average_quality()
        quality_volatility = self._calculate_quality_volatility()

        # Store metrics
        self.current_metrics = {
            'success_rate': success_rate,
            'failure_rate': failure_rate,
            'consecutive_failures': consecutive_failures,
            'avg_task_duration': avg_duration,
            'avg_quality_score': avg_quality,
            'quality_volatility': quality_volatility,
            'idle_time_seconds': idle_time,
            'execution_time_seconds': execution_time,
            'total_tasks': self.total_tasks,
            'completed_tasks': self.completed_tasks,
            'successful_tasks': self.successful_tasks,
            'failed_tasks': self.failed_tasks,
            'active_tasks_count': len(self.active_tasks)
        }

        # Add resource metrics if available
        self.current_metrics.update(self._get_resource_metrics())

        # Store in history
        self.metrics_history.append(self.current_metrics.copy())
        if len(self.metrics_history) > 50:
            self.metrics_history = self.metrics_history[-25:]

    def _calculate_quality_volatility(self) -> float:
        """Calculate quality score volatility."""
        quality_scores = [
            result.quality_score for result in self.task_results.values()
            if result.quality_score is not None
        ]

        if len(quality_scores) < 5:
            return 0.0

        avg_quality = sum(quality_scores) / len(quality_scores)
        variance = sum((score - avg_quality) ** 2 for score in quality_scores) / len(quality_scores)
        return (variance ** 0.5) / avg_quality if avg_quality > 0 else 0.0

    def _get_resource_metrics(self) -> Dict[str, float]:
        """Get current resource usage metrics."""
        # This would integrate with system monitoring
        # For now, return placeholder values
        return {
            'memory_usage_mb': 0.0,
            'cpu_usage_percent': 0.0,
            'disk_usage_mb': 0.0,
            'network_errors': 0
        }

    def check_early_termination(self) -> Tuple[bool, str, TerminationReason]:
        """Check if workflow should terminate early.

        Returns:
            Tuple of (should_terminate, reason, termination_reason)
        """
        if not self.early_termination_enabled or self.status not in [WorkflowStatus.RUNNING]:
            return False, "Early termination not applicable in current state", TerminationReason.TARGET_ACHIEVED

        self.termination_checks_count += 1

        # Use termination criteria
        should_terminate, reason, termination_reason = self.termination_criteria.should_terminate(self.current_metrics)

        # Adapt thresholds based on performance
        if len(self.performance_history) >= 10:
            self.termination_criteria.adapt_thresholds(self.performance_history)

        # Store recommendation
        self.termination_recommendations.append((should_terminate, reason, termination_reason))

        # Log if termination recommended
        if should_terminate and self.enhanced_logger:
            self.enhanced_logger.log_event(
                LogLevel.WARNING,
                LogCategory.SYSTEM,
                AgentEventType.ERROR,
                f"Early termination recommended: {reason}",
                workflow_id=self.workflow_id,
                termination_reason=termination_reason.value,
                current_metrics=self.current_metrics
            )

        return should_terminate, reason, termination_reason

    def terminate_early(self, reason: str, termination_reason: TerminationReason):
        """Terminate workflow early.

        Args:
            reason: Reason for early termination
            termination_reason: Type of termination reason
        """
        self.transition_to(
            WorkflowStatus.TERMINATING,
            f"Early termination: {reason}",
            {
                'termination_reason': termination_reason.value,
                'termination_checks_count': self.termination_checks_count,
                'final_metrics': self.current_metrics.copy()
            }
        )

        # Immediately move to final state
        final_status = WorkflowStatus.FAILED if termination_reason != TerminationReason.TARGET_ACHIEVED else WorkflowStatus.COMPLETED
        self.transition_to(
            final_status,
            f"Workflow terminated early: {reason}",
            {
                'termination_reason': termination_reason.value,
                'execution_time_seconds': self.current_metrics.get('execution_time_seconds', 0.0)
            }
        )

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get comprehensive progress summary."""
        # Target progress
        targets_progress = {
            target_id: target.get_status_summary()
            for target_id, target in self.targets.items()
        }

        # Calculate overall progress
        overall_progress = 0.0
        if self.primary_target_id and self.primary_target_id in self.targets:
            overall_progress = self.targets[self.primary_target_id].progress_percentage
        elif self.targets:
            overall_progress = sum(target.progress_percentage for target in self.targets.values()) / len(self.targets)

        return {
            'workflow_id': self.workflow_id,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,

            # Progress metrics
            'overall_progress_percentage': overall_progress,
            'total_tasks': self.total_tasks,
            'completed_tasks': self.completed_tasks,
            'successful_tasks': self.successful_tasks,
            'failed_tasks': self.failed_tasks,
            'active_tasks_count': len(self.active_tasks),

            # Success rates
            'success_rate': self.current_metrics.get('success_rate', 0.0),
            'failure_rate': self.current_metrics.get('failure_rate', 0.0),

            # Performance metrics
            'avg_task_duration': self.current_metrics.get('avg_task_duration', 0.0),
            'avg_quality_score': self.current_metrics.get('avg_quality_score', 0.0),
            'quality_volatility': self.current_metrics.get('quality_volatility', 0.0),

            # Time metrics
            'execution_time_seconds': self.current_metrics.get('execution_time_seconds', 0.0),
            'idle_time_seconds': self.current_metrics.get('idle_time_seconds', 0.0),

            # Targets
            'primary_target_id': self.primary_target_id,
            'targets_progress': targets_progress,
            'targets_achieved': [tid for tid, target in self.targets.items() if target.achieved],

            # Early termination
            'early_termination_enabled': self.early_termination_enabled,
            'termination_checks_count': self.termination_checks_count,
            'latest_termination_recommendation': self.termination_recommendations[-1] if self.termination_recommendations else None
        }

    def get_state_history(self) -> List[Dict[str, Any]]:
        """Get workflow state history."""
        return [
            {
                'from_status': transition.from_status.value,
                'to_status': transition.to_status.value,
                'timestamp': transition.timestamp.isoformat(),
                'reason': transition.reason,
                'metadata': transition.metadata
            }
            for transition in self.state_history
        ]

    def add_active_task(self, task_id: str):
        """Add a task to active tasks list."""
        self.active_tasks.add(task_id)
        self.last_activity_time = datetime.now()

    def remove_active_task(self, task_id: str):
        """Remove a task from active tasks list."""
        self.active_tasks.discard(task_id)

    def enable_early_termination(self, enabled: bool = True):
        """Enable or disable early termination."""
        self.early_termination_enabled = enabled
        logger.info(f"Early termination {'enabled' if enabled else 'disabled'} for workflow {self.workflow_id}")

    def update_termination_criteria(self, **criteria_updates):
        """Update termination criteria."""
        for key, value in criteria_updates.items():
            if hasattr(self.termination_criteria, key):
                setattr(self.termination_criteria, key, value)
        logger.info(f"Updated termination criteria for workflow {self.workflow_id}: {criteria_updates}")

    def add_custom_termination_check(self, check_func: Callable[[], Tuple[bool, str]]):
        """Add custom termination check function.

        Args:
            check_func: Function that returns (should_terminate, reason)
        """
        self.termination_criteria.custom_termination_checks.append(check_func)
        logger.info(f"Added custom termination check for workflow {self.workflow_id}")

    def reset_state(self):
        """Reset workflow state for new execution."""
        self.status = WorkflowStatus.INITIALIZING
        self.state_history.clear()
        self.started_at = None
        self.completed_at = None

        # Reset targets
        for target in self.targets.values():
            target.current_value = 0
            target.progress_percentage = 0.0
            target.achieved = False
            target.achievement_time = None
            target.start_time = None
            target.time_remaining = None

        # Reset progress
        self.total_tasks = 0
        self.completed_tasks = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.active_tasks.clear()

        # Reset metrics
        self.performance_history.clear()
        self.current_metrics.clear()
        self.metrics_history.clear()
        self.task_results.clear()
        self.termination_recommendations.clear()
        self.termination_checks_count = 0

        logger.info(f"Reset workflow state for {self.workflow_id}")