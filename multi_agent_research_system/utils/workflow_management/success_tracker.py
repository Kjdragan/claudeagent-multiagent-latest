"""
Phase 1.5.1: Enhanced Success Tracker with Comprehensive Failure Analysis

This module implements the EnhancedSuccessTracker with comprehensive failure analysis,
pattern detection, and performance metrics tracking for intelligent task management.

Key Features:
- TaskResult dataclass with comprehensive failure analysis
- Failure pattern detection and classification
- Performance metrics and optimization suggestions
- Success rate tracking by task type and session
- Anti-bot escalation distribution analysis
- Timeout and performance bottleneck detection

Based on Technical Enhancements Section 6: Enhanced Success Tracking & Resource Management
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
        ErrorType, TaskStatus, TaskContext, PipelineStage
    )
    DATA_CONTRACTS_AVAILABLE = True
except ImportError:
    DATA_CONTRACTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class FailureCategory(str, Enum):
    """Categorized failure types for pattern analysis."""
    NETWORK_FAILURE = "network_failure"
    ANTI_BOT_FAILURE = "anti_bot_failure"
    CONTENT_FAILURE = "content_failure"
    QUALITY_FAILURE = "quality_failure"
    TIMEOUT_FAILURE = "timeout_failure"
    SYSTEM_FAILURE = "system_failure"
    VALIDATION_FAILURE = "validation_failure"
    RATE_LIMIT_FAILURE = "rate_limit_failure"
    UNKNOWN_FAILURE = "unknown_failure"


class TaskType(str, Enum):
    """Task type enumeration for success tracking."""
    SCRAPING = "scraping"
    CLEANING = "cleaning"
    SEARCH = "search"
    VALIDATION = "validation"
    RETRY = "retry"
    BATCH_PROCESSING = "batch_processing"


class PerformancePattern(str, Enum):
    """Performance pattern types for optimization."""
    CONSISTENT = "consistent"
    IMPROVING = "improving"
    DEGRADING = "degrading"
    VOLATILE = "volatile"
    STAGNANT = "stagnant"
    UNKNOWN = "unknown"


@dataclass
class FailureAnalysis:
    """Comprehensive failure analysis for pattern detection."""

    # Basic failure information
    failure_type: ErrorType
    failure_category: FailureCategory
    stage: PipelineStage
    timestamp: datetime
    error_message: str

    # Context information
    task_type: TaskType
    url: Optional[str] = None
    domain: Optional[str] = None
    session_id: Optional[str] = None
    retry_count: int = 0

    # Anti-bot specific data
    anti_bot_level: Optional[int] = None
    escalation_used: bool = False
    escalation_triggers: List[str] = field(default_factory=list)

    # Performance data
    duration_seconds: float = 0.0
    timeout_occurred: bool = False
    memory_usage_mb: Optional[float] = None

    # Content quality data
    content_length: int = 0
    quality_score: Optional[float] = None
    cleanliness_score: Optional[float] = None

    # Retry analysis
    is_retry: bool = False
    original_failure_time: Optional[datetime] = None
    retry_delay: float = 0.0

    # System context
    worker_id: Optional[str] = None
    queue_size_at_failure: int = 0
    concurrent_tasks: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging and analysis."""
        return {
            'failure_type': self.failure_type.value,
            'failure_category': self.failure_category.value,
            'stage': self.stage.value,
            'timestamp': self.timestamp.isoformat(),
            'error_message': self.error_message,
            'task_type': self.task_type.value,
            'url': self.url,
            'domain': self.domain,
            'session_id': self.session_id,
            'retry_count': self.retry_count,
            'anti_bot_level': self.anti_bot_level,
            'escalation_used': self.escalation_used,
            'escalation_triggers': self.escalation_triggers,
            'duration_seconds': self.duration_seconds,
            'timeout_occurred': self.timeout_occurred,
            'memory_usage_mb': self.memory_usage_mb,
            'content_length': self.content_length,
            'quality_score': self.quality_score,
            'cleanliness_score': self.cleanliness_score,
            'is_retry': self.is_retry,
            'original_failure_time': self.original_failure_time.isoformat() if self.original_failure_time else None,
            'retry_delay': self.retry_delay,
            'worker_id': self.worker_id,
            'queue_size_at_failure': self.queue_size_at_failure,
            'concurrent_tasks': self.concurrent_tasks
        }


@dataclass
class SuccessMetrics:
    """Success metrics for performance tracking."""

    # Basic success metrics
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    cancelled_tasks: int = 0

    # Performance metrics
    total_duration: float = 0.0
    avg_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0

    # Quality metrics
    avg_quality_score: float = 0.0
    high_quality_count: int = 0
    quality_threshold_met: int = 0

    # Anti-bot metrics
    escalation_success_rate: float = 0.0
    avg_anti_bot_level: float = 0.0
    escalation_distribution: Dict[int, int] = field(default_factory=dict)

    # Retry metrics
    retry_success_rate: float = 0.0
    avg_retries_per_task: float = 0.0
    retry_effectiveness: float = 0.0

    # Resource utilization
    avg_memory_usage: float = 0.0
    avg_queue_size: float = 0.0
    avg_concurrent_tasks: float = 0.0

    def calculate_success_rate(self) -> float:
        """Calculate overall success rate."""
        if self.total_tasks == 0:
            return 0.0
        return self.successful_tasks / self.total_tasks

    def calculate_failure_rate(self) -> float:
        """Calculate overall failure rate."""
        if self.total_tasks == 0:
            return 0.0
        return self.failed_tasks / self.total_tasks

    def update_duration_stats(self, duration: float):
        """Update duration statistics."""
        self.total_duration += duration
        self.total_tasks += 1
        self.avg_duration = self.total_duration / self.total_tasks
        self.min_duration = min(self.min_duration, duration)
        self.max_duration = max(self.max_duration, duration)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            'total_tasks': self.total_tasks,
            'successful_tasks': self.successful_tasks,
            'failed_tasks': self.failed_tasks,
            'cancelled_tasks': self.cancelled_tasks,
            'success_rate': self.calculate_success_rate(),
            'failure_rate': self.calculate_failure_rate(),
            'total_duration': self.total_duration,
            'avg_duration': self.avg_duration,
            'min_duration': self.min_duration if self.min_duration != float('inf') else 0.0,
            'max_duration': self.max_duration,
            'avg_quality_score': self.avg_quality_score,
            'high_quality_count': self.high_quality_count,
            'quality_threshold_met': self.quality_threshold_met,
            'escalation_success_rate': self.escalation_success_rate,
            'avg_anti_bot_level': self.avg_anti_bot_level,
            'escalation_distribution': self.escalation_distribution,
            'retry_success_rate': self.retry_success_rate,
            'avg_retries_per_task': self.avg_retries_per_task,
            'retry_effectiveness': self.retry_effectiveness,
            'avg_memory_usage': self.avg_memory_usage,
            'avg_queue_size': self.avg_queue_size,
            'avg_concurrent_tasks': self.avg_concurrent_tasks
        }


@dataclass
class TaskResult:
    """Comprehensive task result with failure analysis and performance tracking."""

    # Core result data
    task_id: str = field(default_factory=lambda: str(uuid4()))
    task_type: TaskType = TaskType.SCRAPING
    success: bool = False
    status: TaskStatus = TaskStatus.PENDING

    # Timing information
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0

    # Task context
    session_id: str = field(default_factory=lambda: str(uuid4()))
    url: Optional[str] = None
    domain: Optional[str] = None
    search_query: Optional[str] = None

    # Content and quality data
    content_length: int = 0
    quality_score: Optional[float] = None
    cleanliness_score: Optional[float] = None

    # Anti-bot data
    anti_bot_level: Optional[int] = None
    escalation_used: bool = False
    escalation_triggers: List[str] = field(default_factory=list)

    # Error and failure data
    error_type: Optional[ErrorType] = None
    error_message: Optional[str] = None
    failure_analysis: Optional[FailureAnalysis] = None

    # Performance data
    memory_usage_mb: Optional[float] = None
    worker_id: Optional[str] = None
    retry_count: int = 0
    is_retry: bool = False

    # System context
    queue_size_at_start: int = 0
    queue_size_at_end: int = 0
    concurrent_tasks: int = 0

    # Success indicators
    quality_threshold_met: bool = False
    content_requirements_met: bool = False
    performance_acceptable: bool = True

    def mark_completed(self, success: bool, status: TaskStatus = TaskStatus.COMPLETED):
        """Mark task as completed with final status."""
        self.success = success
        self.status = status
        self.completed_at = datetime.now()
        self.duration_seconds = (self.completed_at - self.started_at).total_seconds()

    def create_failure_analysis(self) -> FailureAnalysis:
        """Create comprehensive failure analysis."""
        if not self.error_type:
            raise ValueError("Cannot create failure analysis without error type")

        # Determine failure category
        failure_category = self._categorize_failure()

        return FailureAnalysis(
            failure_type=self.error_type,
            failure_category=failure_category,
            stage=PipelineStage.SCRAPING,  # Default, should be set by caller
            timestamp=self.completed_at or datetime.now(),
            error_message=self.error_message or "Unknown error",
            task_type=self.task_type,
            url=self.url,
            domain=self.domain,
            session_id=self.session_id,
            retry_count=self.retry_count,
            anti_bot_level=self.anti_bot_level,
            escalation_used=self.escalation_used,
            escalation_triggers=self.escalation_triggers,
            duration_seconds=self.duration_seconds,
            timeout_occurred=self.error_type == ErrorType.TIMEOUT_ERROR,
            memory_usage_mb=self.memory_usage_mb,
            content_length=self.content_length,
            quality_score=self.quality_score,
            cleanliness_score=self.cleanliness_score,
            is_retry=self.is_retry,
            worker_id=self.worker_id,
            queue_size_at_failure=self.queue_size_at_end,
            concurrent_tasks=self.concurrent_tasks
        )

    def _categorize_failure(self) -> FailureCategory:
        """Categorize failure type for pattern analysis."""
        if not self.error_type:
            return FailureCategory.UNKNOWN_FAILURE

        category_mapping = {
            ErrorType.NETWORK_ERROR: FailureCategory.NETWORK_FAILURE,
            ErrorType.ANTI_BOT_DETECTION: FailureCategory.ANTI_BOT_FAILURE,
            ErrorType.CONTENT_EXTRACTION_ERROR: FailureCategory.CONTENT_FAILURE,
            ErrorType.CLEANING_ERROR: FailureCategory.CONTENT_FAILURE,
            ErrorType.TIMEOUT_ERROR: FailureCategory.TIMEOUT_FAILURE,
            ErrorType.RATE_LIMIT_ERROR: FailureCategory.RATE_LIMIT_FAILURE,
            ErrorType.VALIDATION_ERROR: FailureCategory.VALIDATION_FAILURE,
            ErrorType.UNKNOWN_ERROR: FailureCategory.UNKNOWN_FAILURE
        }

        return category_mapping.get(self.error_type, FailureCategory.UNKNOWN_FAILURE)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging and analysis."""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type.value,
            'success': self.success,
            'status': self.status.value,
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'duration_seconds': self.duration_seconds,
            'session_id': self.session_id,
            'url': self.url,
            'domain': self.domain,
            'search_query': self.search_query,
            'content_length': self.content_length,
            'quality_score': self.quality_score,
            'cleanliness_score': self.cleanliness_score,
            'anti_bot_level': self.anti_bot_level,
            'escalation_used': self.escalation_used,
            'escalation_triggers': self.escalation_triggers,
            'error_type': self.error_type.value if self.error_type else None,
            'error_message': self.error_message,
            'failure_analysis': self.failure_analysis.to_dict() if self.failure_analysis else None,
            'memory_usage_mb': self.memory_usage_mb,
            'worker_id': self.worker_id,
            'retry_count': self.retry_count,
            'is_retry': self.is_retry,
            'queue_size_at_start': self.queue_size_at_start,
            'queue_size_at_end': self.queue_size_at_end,
            'concurrent_tasks': self.concurrent_tasks,
            'quality_threshold_met': self.quality_threshold_met,
            'content_requirements_met': self.content_requirements_met,
            'performance_acceptable': self.performance_acceptable
        }


class EnhancedSuccessTracker:
    """
    Enhanced success tracker with comprehensive failure analysis and pattern detection.

    This tracker provides intelligent success tracking, failure pattern analysis,
    performance optimization suggestions, and early termination recommendations.
    """

    def __init__(self, session_id: Optional[str] = None):
        """Initialize the enhanced success tracker.

        Args:
            session_id: Optional session ID for tracking
        """
        self.session_id = session_id or str(uuid4())
        self.created_at = datetime.now()

        # Initialize enhanced logging
        self._setup_logging()

        # Task tracking
        self._tasks: Dict[str, TaskResult] = {}
        self._active_tasks: Set[str] = set()

        # Failure analysis storage
        self._failures: List[FailureAnalysis] = []
        self._failure_patterns: DefaultDict[str, List[FailureAnalysis]] = defaultdict(list)
        self._domain_failures: DefaultDict[str, List[FailureAnalysis]] = defaultdict(list)

        # Performance tracking
        self._performance_history: Deque[float] = deque(maxlen=100)
        self._success_metrics: Dict[TaskType, SuccessMetrics] = {
            task_type: SuccessMetrics() for task_type in TaskType
        }

        # Anti-bot tracking
        self._anti_bot_distribution: DefaultDict[int, int] = defaultdict(int)
        self._escalation_effectiveness: DefaultDict[int, float] = defaultdict(float)

        # Pattern detection
        self._performance_patterns: Dict[str, PerformancePattern] = {}
        self._optimization_suggestions: List[str] = []

        # Early termination data
        self._termination_thresholds: Dict[str, float] = {
            'min_success_rate': 0.3,
            'max_failure_rate': 0.7,
            'max_consecutive_failures': 5,
            'max_avg_duration': 300.0,  # 5 minutes
            'min_quality_threshold': 0.5
        }

        logger.info(f"EnhancedSuccessTracker initialized for session {self.session_id}")

    def _setup_logging(self):
        """Setup enhanced logging integration."""
        if ENHANCED_LOGGING_AVAILABLE:
            self.enhanced_logger = get_enhanced_logger("enhanced_success_tracker")
            self.enhanced_logger.log_event(
                LogLevel.INFO,
                LogCategory.SYSTEM,
                AgentEventType.SESSION_START,
                "Enhanced Success Tracker initialized",
                session_id=self.session_id
            )
        else:
            self.enhanced_logger = None

    def start_task_tracking(
        self,
        task_id: str,
        task_type: TaskType,
        url: Optional[str] = None,
        **kwargs
    ) -> TaskResult:
        """Start tracking a new task.

        Args:
            task_id: Unique task identifier
            task_type: Type of task being tracked
            url: Optional URL for the task
            **kwargs: Additional task parameters

        Returns:
            Created TaskResult instance
        """
        task_result = TaskResult(
            task_id=task_id,
            task_type=task_type,
            url=url,
            domain=self._extract_domain(url) if url else None,
            session_id=self.session_id,
            **kwargs
        )

        self._tasks[task_id] = task_result
        self._active_tasks.add(task_id)

        # Log task start
        if self.enhanced_logger:
            self.enhanced_logger.log_event(
                LogLevel.INFO,
                LogCategory.PERFORMANCE,
                AgentEventType.TASK_START,
                f"Started tracking {task_type.value} task",
                task_id=task_id,
                url=url,
                session_id=self.session_id
            )

        logger.debug(f"Started tracking task {task_id} ({task_type.value})")
        return task_result

    def complete_task(
        self,
        task_id: str,
        success: bool,
        status: TaskStatus = TaskStatus.COMPLETED,
        error_type: Optional[ErrorType] = None,
        error_message: Optional[str] = None,
        **kwargs
    ) -> Optional[TaskResult]:
        """Complete a task with results.

        Args:
            task_id: Task identifier
            success: Whether task was successful
            status: Final task status
            error_type: Optional error type if failed
            error_message: Optional error message if failed
            **kwargs: Additional result data

        Returns:
            Updated TaskResult or None if task not found
        """
        if task_id not in self._tasks:
            logger.warning(f"Task {task_id} not found for completion")
            return None

        task_result = self._tasks[task_id]

        # Update task result
        task_result.mark_completed(success, status)
        task_result.error_type = error_type
        task_result.error_message = error_message

        # Update additional fields
        for key, value in kwargs.items():
            if hasattr(task_result, key):
                setattr(task_result, key, value)

        # Create failure analysis if needed
        if not success and error_type:
            task_result.failure_analysis = task_result.create_failure_analysis()
            self._analyze_failure(task_result.failure_analysis)

        # Update metrics
        self._update_success_metrics(task_result)
        self._update_performance_history(task_result)
        self._update_anti_bot_tracking(task_result)

        # Remove from active tasks
        self._active_tasks.discard(task_id)

        # Log task completion
        if self.enhanced_logger:
            self.enhanced_logger.log_event(
                LogLevel.INFO if success else LogLevel.WARNING,
                LogCategory.PERFORMANCE,
                AgentEventType.TASK_END if success else AgentEventType.ERROR,
                f"Completed {task_result.task_type.value} task: {'success' if success else 'failure'}",
                task_id=task_id,
                success=success,
                duration=task_result.duration_seconds,
                quality_score=task_result.quality_score,
                error_type=error_type.value if error_type else None
            )

        logger.debug(f"Completed task {task_id}: success={success}, duration={task_result.duration_seconds:.2f}s")
        return task_result

    def _analyze_failure(self, failure: FailureAnalysis):
        """Analyze failure for patterns and trends."""
        self._failures.append(failure)

        # Categorize failures by type
        failure_key = f"{failure.failure_category.value}_{failure.stage.value}"
        self._failure_patterns[failure_key].append(failure)

        # Track domain-specific failures
        if failure.domain:
            self._domain_failures[failure.domain].append(failure)

        # Analyze for patterns
        self._detect_failure_patterns()
        self._generate_optimization_suggestions()

    def _update_success_metrics(self, task_result: TaskResult):
        """Update success metrics for task type."""
        metrics = self._success_metrics[task_result.task_type]

        if task_result.success:
            metrics.successful_tasks += 1

            # Update quality metrics
            if task_result.quality_score:
                metrics.avg_quality_score = (
                    (metrics.avg_quality_score * (metrics.successful_tasks - 1) + task_result.quality_score) /
                    metrics.successful_tasks
                )

                if task_result.quality_score >= 0.8:
                    metrics.high_quality_count += 1
                if task_result.quality_score >= 0.7:
                    metrics.quality_threshold_met += 1

            # Update anti-bot metrics
            if task_result.escalation_used:
                if task_result.anti_bot_level is not None:
                    self._anti_bot_distribution[task_result.anti_bot_level] += 1

                    # Calculate escalation effectiveness
                    level_success = self._calculate_escalation_effectiveness(task_result.anti_bot_level)
                    self._escalation_effectiveness[task_result.anti_bot_level] = level_success

        else:
            metrics.failed_tasks += 1

        # Update duration metrics
        metrics.update_duration_stats(task_result.duration_seconds)

    def _update_performance_history(self, task_result: TaskResult):
        """Update performance history for pattern detection."""
        self._performance_history.append(task_result.duration_seconds)

        # Update performance patterns
        if len(self._performance_history) >= 10:
            pattern = self._detect_performance_pattern()
            self._performance_patterns[task_result.task_type.value] = pattern

    def _update_anti_bot_tracking(self, task_result: TaskResult):
        """Update anti-bot tracking data."""
        if task_result.anti_bot_level is not None:
            self._anti_bot_distribution[task_result.anti_bot_level] += 1

        # Calculate escalation effectiveness
        if task_result.escalation_used and task_result.success:
            if task_result.anti_bot_level is not None:
                level_success = self._calculate_escalation_effectiveness(task_result.anti_bot_level)
                self._escalation_effectiveness[task_result.anti_bot_level] = level_success

    def _calculate_escalation_effectiveness(self, level: int) -> float:
        """Calculate effectiveness of anti-bot escalation level."""
        level_tasks = [t for t in self._tasks.values() if t.anti_bot_level == level]
        if not level_tasks:
            return 0.0

        successful_tasks = [t for t in level_tasks if t.success]
        return len(successful_tasks) / len(level_tasks)

    def _detect_performance_pattern(self) -> PerformancePattern:
        """Detect performance pattern from recent history."""
        if len(self._performance_history) < 5:
            return PerformancePattern.UNKNOWN

        recent_times = list(self._performance_history)[-10:]

        # Calculate trend
        if len(recent_times) >= 5:
            first_half = recent_times[:len(recent_times)//2]
            second_half = recent_times[len(recent_times)//2:]

            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)

            # Calculate variance for volatility
            variance = sum((x - sum(recent_times)/len(recent_times))**2 for x in recent_times) / len(recent_times)
            std_dev = variance ** 0.5
            coefficient_of_variation = std_dev / (sum(recent_times) / len(recent_times)) if sum(recent_times) > 0 else 0

            # Determine pattern
            if coefficient_of_variation > 0.5:
                return PerformancePattern.VOLATILE
            elif second_avg < first_avg * 0.9:
                return PerformancePattern.IMPROVING
            elif second_avg > first_avg * 1.1:
                return PerformancePattern.DEGRADING
            elif abs(second_avg - first_avg) / first_avg < 0.1:
                return PerformancePattern.CONSISTENT
            else:
                return PerformancePattern.STAGNANT

        return PerformancePattern.UNKNOWN

    def _detect_failure_patterns(self):
        """Detect patterns in failures for optimization."""
        # Check for repeated failure types
        for failure_key, failures in self._failure_patterns.items():
            if len(failures) >= 3:  # Pattern threshold
                recent_failures = failures[-3:]

                # Check if failures are consecutive
                timestamps = [f.timestamp for f in recent_failures]
                if all((timestamps[i+1] - timestamps[i]) < timedelta(minutes=5) for i in range(len(timestamps)-1)):
                    logger.warning(f"Detected consecutive failure pattern: {failure_key}")

                    if self.enhanced_logger:
                        self.enhanced_logger.log_event(
                            LogLevel.WARNING,
                            LogCategory.SYSTEM,
                            AgentEventType.ERROR,
                            f"Consecutive failure pattern detected: {failure_key}",
                            failure_key=failure_key,
                            failure_count=len(recent_failures)
                        )

        # Check for domain-specific patterns
        for domain, failures in self._domain_failures.items():
            if len(failures) >= 5:  # Domain pattern threshold
                failure_rate = len(failures) / max(1, len([t for t in self._tasks.values() if t.domain == domain]))
                if failure_rate > 0.7:  # 70% failure rate for domain
                    logger.warning(f"High failure rate detected for domain {domain}: {failure_rate:.1%}")

    def _generate_optimization_suggestions(self):
        """Generate optimization suggestions based on patterns."""
        suggestions = []

        # Check anti-bot effectiveness
        for level, effectiveness in self._escalation_effectiveness.items():
            if effectiveness < 0.3 and self._anti_bot_distribution[level] > 5:
                suggestions.append(f"Anti-bot level {level} has low effectiveness ({effectiveness:.1%}). Consider escalation strategy adjustments.")

        # Check performance patterns
        for task_type, pattern in self._performance_patterns.items():
            if pattern == PerformancePattern.DEGRADING:
                suggestions.append(f"Performance degrading for {task_type} tasks. Consider resource optimization.")
            elif pattern == PerformancePattern.VOLATILE:
                suggestions.append(f"High performance volatility for {task_type} tasks. Consider load balancing.")

        # Check failure patterns
        total_tasks = len(self._tasks)
        if total_tasks > 10:
            failure_rate = len(self._failures) / total_tasks
            if failure_rate > 0.5:
                suggestions.append(f"High overall failure rate ({failure_rate:.1%}). Review system configuration and error handling.")

        self._optimization_suggestions = suggestions

    def should_terminate_early(self) -> Tuple[bool, str]:
        """Check if workflow should terminate early based on performance.

        Returns:
            Tuple of (should_terminate, reason)
        """
        if len(self._tasks) < 5:  # Need minimum data
            return False, "Insufficient data for early termination decision"

        # Check success rate
        total_completed = len([t for t in self._tasks.values() if t.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]])
        if total_completed > 0:
            success_rate = len([t for t in self._tasks.values() if t.success]) / total_completed

            if success_rate < self._termination_thresholds['min_success_rate']:
                return True, f"Success rate ({success_rate:.1%}) below minimum threshold ({self._termination_thresholds['min_success_rate']:.1%})"

            failure_rate = 1.0 - success_rate
            if failure_rate > self._termination_thresholds['max_failure_rate']:
                return True, f"Failure rate ({failure_rate:.1%}) above maximum threshold ({self._termination_thresholds['max_failure_rate']:.1%})"

        # Check consecutive failures
        recent_tasks = sorted([t for t in self._tasks.values() if t.completed_at], key=lambda x: x.completed_at or datetime.min)
        consecutive_failures = 0
        for task in reversed(recent_tasks[-10:]):  # Check last 10 tasks
            if not task.success:
                consecutive_failures += 1
            else:
                break

        if consecutive_failures >= self._termination_thresholds['max_consecutive_failures']:
            return True, f"Too many consecutive failures ({consecutive_failures})"

        # Check average duration
        if self._performance_history:
            avg_duration = sum(self._performance_history) / len(self._performance_history)
            if avg_duration > self._termination_thresholds['max_avg_duration']:
                return True, f"Average task duration ({avg_duration:.1f}s) exceeds maximum threshold ({self._termination_thresholds['max_avg_duration']:.1f}s)"

        # Check quality scores
        quality_scores = [t.quality_score for t in self._tasks.values() if t.quality_score is not None]
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            if avg_quality < self._termination_thresholds['min_quality_threshold']:
                return True, f"Average quality score ({avg_quality:.2f}) below minimum threshold ({self._termination_thresholds['min_quality_threshold']:.2f})"

        return False, "Performance metrics within acceptable ranges"

    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive success tracking report."""
        total_tasks = len(self._tasks)
        completed_tasks = [t for t in self._tasks.values() if t.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]]

        # Overall metrics
        overall_success_rate = len([t for t in completed_tasks if t.success]) / len(completed_tasks) if completed_tasks else 0.0

        # Task type breakdown
        task_type_metrics = {}
        for task_type, metrics in self._success_metrics.items():
            if metrics.total_tasks > 0:
                task_type_metrics[task_type.value] = metrics.to_dict()

        # Failure analysis
        failure_summary = {
            'total_failures': len(self._failures),
            'failure_categories': self._summarize_failure_categories(),
            'domain_failures': {domain: len(failures) for domain, failures in self._domain_failures.items()},
            'failure_patterns': {pattern: len(failures) for pattern, failures in self._failure_patterns.items()}
        }

        # Anti-bot analysis
        anti_bot_summary = {
            'distribution': dict(self._anti_bot_distribution),
            'effectiveness': dict(self._escalation_effectiveness)
        }

        # Performance analysis
        performance_summary = {
            'patterns': {task_type: pattern.value for task_type, pattern in self._performance_patterns.items()},
            'avg_duration': sum(self._performance_history) / len(self._performance_history) if self._performance_history else 0.0,
            'performance_history': list(self._performance_history)[-20:]  # Last 20 tasks
        }

        # Early termination assessment
        should_terminate, termination_reason = self.should_terminate_early()

        return {
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat(),
            'report_generated_at': datetime.now().isoformat(),

            # Overall metrics
            'total_tasks': total_tasks,
            'active_tasks': len(self._active_tasks),
            'completed_tasks': len(completed_tasks),
            'overall_success_rate': overall_success_rate,

            # Task type breakdown
            'task_type_metrics': task_type_metrics,

            # Failure analysis
            'failure_summary': failure_summary,

            # Anti-bot analysis
            'anti_bot_summary': anti_bot_summary,

            # Performance analysis
            'performance_summary': performance_summary,

            # Optimization
            'optimization_suggestions': self._optimization_suggestions,

            # Early termination
            'early_termination': {
                'should_terminate': should_terminate,
                'reason': termination_reason
            },

            # Termination thresholds
            'termination_thresholds': self._termination_thresholds.copy()
        }

    def _summarize_failure_categories(self) -> Dict[str, int]:
        """Summarize failures by category."""
        category_counts = defaultdict(int)
        for failure in self._failures:
            category_counts[failure.failure_category.value] += 1
        return dict(category_counts)

    def _extract_domain(self, url: str) -> Optional[str]:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc
        except Exception:
            return None

    def get_active_tasks(self) -> List[TaskResult]:
        """Get list of currently active tasks."""
        return [self._tasks[task_id] for task_id in self._active_tasks if task_id in self._tasks]

    def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """Get task result by ID."""
        return self._tasks.get(task_id)

    def update_termination_thresholds(self, **thresholds):
        """Update early termination thresholds."""
        self._termination_thresholds.update(thresholds)
        logger.info(f"Updated termination thresholds: {thresholds}")

    def reset_metrics(self):
        """Reset all metrics for new session."""
        self._tasks.clear()
        self._active_tasks.clear()
        self._failures.clear()
        self._failure_patterns.clear()
        self._domain_failures.clear()
        self._performance_history.clear()
        self._performance_patterns.clear()
        self._optimization_suggestions.clear()

        # Reset success metrics
        for task_type in TaskType:
            self._success_metrics[task_type] = SuccessMetrics()

        logger.info("Reset all success tracking metrics")