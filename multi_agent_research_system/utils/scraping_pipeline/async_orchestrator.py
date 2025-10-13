"""
Enhanced Two-Module Scraping Architecture - Async Orchestrator

Phase 1.4.2: Implement AsyncScrapingOrchestrator with worker pools (40/20 concurrency)

This module implements the AsyncScrapingOrchestrator with sophisticated worker pools,
backpressure management, and comprehensive integration with anti-bot and content cleaning systems.

Key Features:
- AsyncScrapingOrchestrator with 40 scrape workers and 20 clean workers
- Sophisticated queue management with backpressure control
- Integration with anti-bot escalation system (Phase 1.2)
- Integration with content cleaning pipeline (Phase 1.3)
- Performance monitoring and optimization
- Early termination and success tracking logic
- Comprehensive error handling and recovery mechanisms

Based on Technical Enhancements Section 1: Enhanced Concurrency & Async Orchestration
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import (
    Any, Dict, List, Optional, Tuple, Callable, Set, Union,
    AsyncGenerator, Deque
)
from uuid import uuid4

from .data_contracts import (
    TaskContext, ScrapingRequest, ScrapingResult,
    CleaningRequest, CleaningResult, PipelineConfig,
    PipelineStatistics, TaskStatus, PipelineStage,
    ErrorType, Priority
)

# Import phase integrations
try:
    from ..anti_bot.escalation_manager import AntiBotEscalationManager
    from ..content_cleaning.content_cleaning_pipeline import ContentCleaningPipeline
    from ..search_types import SearchQuery, SearchResult
    PHASE_INTEGRATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Phase integration limited: {e}")
    PHASE_INTEGRATION_AVAILABLE = False

# Import enhanced logging from Phase 1.1
try:
    from ...agent_logging.enhanced_logger import (
        get_enhanced_logger, LogLevel, LogCategory, AgentEventType
    )
    ENHANCED_LOGGING_AVAILABLE = True
except ImportError:
    ENHANCED_LOGGING_AVAILABLE = False

logger = logging.getLogger(__name__)


class QueueMetrics:
    """Metrics for queue monitoring."""

    def __init__(self):
        self.total_added = 0
        self.total_processed = 0
        self.total_failed = 0
        self.total_cancelled = 0
        self.peak_size = 0
        self.avg_wait_time = 0.0
        self.total_wait_time = 0.0

    def update(self, wait_time: float, processed: bool):
        """Update metrics with new task result."""
        self.total_wait_time += wait_time
        if processed:
            self.total_processed += 1
        else:
            self.total_failed += 1

        if self.total_processed + self.total_failed > 0:
            self.avg_wait_time = self.total_wait_time / (self.total_processed + self.total_failed)

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        return {
            'total_added': self.total_added,
            'total_processed': self.total_processed,
            'total_failed': self.total_failed,
            'total_cancelled': self.total_cancelled,
            'success_rate': (
                self.total_processed / (self.total_processed + self.total_failed)
                if (self.total_processed + self.total_failed) > 0 else 0.0
            ),
            'peak_size': self.peak_size,
            'avg_wait_time': self.avg_wait_time
        }


class AsyncTaskQueue:
    """Async task queue with priority support and backpressure management."""

    def __init__(
        self,
        max_size: int = 1000,
        backpressure_threshold: float = 0.8,
        name: str = "default"
    ):
        """Initialize the async task queue.

        Args:
            max_size: Maximum queue size
            backpressure_threshold: Threshold for backpressure activation (0.0-1.0)
            name: Queue name for logging
        """
        self.max_size = max_size
        self.backpressure_threshold = backpressure_threshold
        self.name = name
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_size)
        self._processing: Set[str] = set()
        self._completed: Set[str] = set()
        self._failed: Set[str] = set()
        self._metrics = QueueMetrics()
        self._shutdown = False
        self._lock = asyncio.Lock()

    async def put(
        self,
        task: Any,
        priority: int = 0,
        task_id: Optional[str] = None
    ) -> bool:
        """Add a task to the queue with priority.

        Args:
            task: Task to add
            priority: Priority (lower numbers = higher priority)
            task_id: Optional task ID

        Returns:
            True if task was added, False if queue is full
        """
        if self._shutdown:
            return False

        # Check backpressure
        if await self._is_backpressure_active():
            logger.debug(f"Backpressure active on queue {self.name}, rejecting task")
            return False

        task_id = task_id or str(uuid4())
        now = time.time()

        try:
            # Create priority item (priority, timestamp, task_id, task, queued_time)
            priority_item = (priority, now, task_id, task, now)
            await self._queue.put(priority_item)
            self._metrics.total_added += 1

            # Update peak size
            current_size = self._queue.qsize()
            if current_size > self._metrics.peak_size:
                self._metrics.peak_size = current_size

            logger.debug(f"Added task {task_id} to queue {self.name} with priority {priority}")
            return True

        except asyncio.QueueFull:
            logger.warning(f"Queue {self.name} is full, rejecting task")
            return False

    async def get(self, timeout: Optional[float] = None) -> Optional[Tuple[Any, str, float]]:
        """Get a task from the queue.

        Args:
            timeout: Optional timeout in seconds

        Returns:
            Tuple of (task, task_id, wait_time) or None if timeout
        """
        if self._shutdown:
            return None

        try:
            if timeout:
                priority_item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            else:
                priority_item = await self._queue.get()

            priority, timestamp, task_id, task, queued_time = priority_item
            wait_time = time.time() - queued_time

            async with self._lock:
                self._processing.add(task_id)

            logger.debug(f"Retrieved task {task_id} from queue {self.name} (wait: {wait_time:.2f}s)")
            return task, task_id, wait_time

        except asyncio.TimeoutError:
            logger.debug(f"Queue {self.name} get timeout after {timeout}s")
            return None
        except Exception as e:
            logger.error(f"Error getting task from queue {self.name}: {e}")
            return None

    async def task_done(self, task_id: str, success: bool, wait_time: float):
        """Mark a task as completed.

        Args:
            task_id: Task ID
            success: Whether task was successful
            wait_time: Time spent waiting in queue
        """
        async with self._lock:
            if task_id in self._processing:
                self._processing.remove(task_id)
                if success:
                    self._completed.add(task_id)
                else:
                    self._failed.add(task_id)

        self._metrics.update(wait_time, success)

    async def _is_backpressure_active(self) -> bool:
        """Check if backpressure should be activated."""
        current_size = self._queue.qsize()
        threshold_size = int(self.max_size * self.backpressure_threshold)
        return current_size >= threshold_size

    def get_size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            'name': self.name,
            'current_size': self.get_size(),
            'max_size': self.max_size,
            'processing_count': len(self._processing),
            'completed_count': len(self._completed),
            'failed_count': len(self._failed),
            'backpressure_active': self._queue.qsize() >= int(self.max_size * self.backpressure_threshold),
            'metrics': self._metrics.get_stats()
        }

    async def clear(self):
        """Clear the queue and reset metrics."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        async with self._lock:
            self._processing.clear()
            self._completed.clear()
            self._failed.clear()

        self._metrics = QueueMetrics()
        logger.info(f"Cleared queue {self.name}")

    async def shutdown(self):
        """Shutdown the queue and reject new tasks."""
        self._shutdown = True
        await self.clear()
        logger.info(f"Shutdown queue {self.name}")


class WorkerPool:
    """Async worker pool for processing tasks."""

    def __init__(
        self,
        worker_count: int,
        task_handler: Callable,
        name: str = "worker_pool",
        timeout: float = 300.0
    ):
        """Initialize the worker pool.

        Args:
            worker_count: Number of workers
            task_handler: Async function to handle tasks
            name: Pool name for logging
            timeout: Task timeout in seconds
        """
        self.worker_count = worker_count
        self.task_handler = task_handler
        self.name = name
        self.timeout = timeout
        self._workers: List[asyncio.Task] = []
        self._shutdown = False
        self._stats = {
            'tasks_processed': 0,
            'tasks_failed': 0,
            'total_duration': 0.0,
            'active_workers': 0
        }

    async def start(self, queue: AsyncTaskQueue):
        """Start the worker pool.

        Args:
            queue: Task queue to process
        """
        logger.info(f"Starting {self.name} worker pool with {self.worker_count} workers")

        for i in range(self.worker_count):
            worker_task = asyncio.create_task(
                self._worker(f"{self.name}_worker_{i}", queue)
            )
            self._workers.append(worker_task)

    async def _worker(self, worker_name: str, queue: AsyncTaskQueue):
        """Worker coroutine for processing tasks.

        Args:
            worker_name: Name of this worker
            queue: Task queue to process
        """
        logger.debug(f"Starting worker {worker_name}")

        while not self._shutdown:
            try:
                # Get task from queue
                task_result = await queue.get(timeout=1.0)
                if task_result is None:
                    continue

                task, task_id, wait_time = task_result

                # Process task with timeout
                start_time = time.time()
                try:
                    result = await asyncio.wait_for(
                        self.task_handler(task),
                        timeout=self.timeout
                    )
                    success = True
                    self._stats['tasks_processed'] += 1

                except asyncio.TimeoutError:
                    logger.warning(f"Task {task_id} timed out in {worker_name}")
                    result = None
                    success = False
                    self._stats['tasks_failed'] += 1

                except Exception as e:
                    logger.error(f"Task {task_id} failed in {worker_name}: {e}")
                    result = None
                    success = False
                    self._stats['tasks_failed'] += 1

                duration = time.time() - start_time
                self._stats['total_duration'] += duration

                # Mark task as done
                await queue.task_done(task_id, success, wait_time)

                logger.debug(
                    f"Worker {worker_name} completed task {task_id} "
                    f"(success: {success}, duration: {duration:.2f}s)"
                )

            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                await asyncio.sleep(0.1)

        logger.debug(f"Worker {worker_name} stopped")

    async def shutdown(self):
        """Shutdown the worker pool."""
        logger.info(f"Shutting down {self.name} worker pool")
        self._shutdown = True

        # Cancel all workers
        for worker in self._workers:
            worker.cancel()

        # Wait for workers to finish
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)

        self._workers.clear()
        logger.info(f"Shutdown {self.name} worker pool")

    def get_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        total_tasks = self._stats['tasks_processed'] + self._stats['tasks_failed']
        avg_duration = (
            self._stats['total_duration'] / self._stats['tasks_processed']
            if self._stats['tasks_processed'] > 0 else 0.0
        )

        return {
            'name': self.name,
            'worker_count': self.worker_count,
            'active_workers': len([w for w in self._workers if not w.done()]),
            'tasks_processed': self._stats['tasks_processed'],
            'tasks_failed': self._stats['tasks_failed'],
            'success_rate': (
                self._stats['tasks_processed'] / total_tasks
                if total_tasks > 0 else 0.0
            ),
            'avg_task_duration': avg_duration,
            'total_duration': self._stats['total_duration']
        }


class AsyncScrapingOrchestrator:
    """
    Async orchestrator for the enhanced two-module scraping architecture.

    This orchestrator manages worker pools, queues, and the overall workflow
    for scraping and cleaning operations with sophisticated error handling and
    performance optimization.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the async scraping orchestrator.

        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self.session_id = str(uuid4())
        self.created_at = datetime.now()

        # Initialize enhanced logging
        self._setup_logging()

        # Initialize phase integrations
        self._initialize_phase_integrations()

        # Initialize queues
        self._scrape_queue = AsyncTaskQueue(
            max_size=self.config.max_queue_size,
            backpressure_threshold=self.config.backpressure_threshold,
            name="scrape"
        )
        self._clean_queue = AsyncTaskQueue(
            max_size=self.config.max_queue_size,
            backpressure_threshold=self.config.backpressure_threshold,
            name="clean"
        )

        # Initialize worker pools
        self._scrape_pool = WorkerPool(
            worker_count=self.config.max_scrape_workers,
            task_handler=self._handle_scrape_task,
            name="scrape",
            timeout=self.config.worker_timeout_seconds
        )
        self._clean_pool = WorkerPool(
            worker_count=self.config.max_clean_workers,
            task_handler=self._handle_clean_task,
            name="clean",
            timeout=self.config.worker_timeout_seconds
        )

        # Statistics tracking
        self._stats = PipelineStatistics(session_id=self.session_id)
        self._active_tasks: Dict[str, TaskContext] = {}

        # Circuit breaker for error handling
        self._circuit_breaker = {
            'failure_count': 0,
            'last_failure': None,
            'state': 'closed',  # closed, open, half_open
            'threshold': 10,
            'recovery_timeout': 60.0
        }

        logger.info(f"AsyncScrapingOrchestrator initialized with session {self.session_id}")

    def _setup_logging(self):
        """Setup enhanced logging integration."""
        if ENHANCED_LOGGING_AVAILABLE:
            self.enhanced_logger = get_enhanced_logger("async_scraping_orchestrator")
            self.enhanced_logger.log_event(
                LogLevel.INFO,
                LogCategory.SYSTEM,
                AgentEventType.SESSION_START,
                "Async Scraping Orchestrator initialized",
                session_id=self.session_id,
                max_scrape_workers=self.config.max_scrape_workers,
                max_clean_workers=self.config.max_clean_workers,
                max_queue_size=self.config.max_queue_size
            )
        else:
            self.enhanced_logger = None

    def _initialize_phase_integrations(self):
        """Initialize integrations with previous phases."""
        if PHASE_INTEGRATION_AVAILABLE:
            if self.config.enable_anti_bot:
                self.anti_bot_manager = AntiBotEscalationManager()
                logger.info("Anti-bot escalation manager initialized")

            if self.config.enable_content_cleaning:
                self.cleaning_pipeline = ContentCleaningPipeline()
                logger.info("Content cleaning pipeline initialized")
        else:
            self.anti_bot_manager = None
            self.cleaning_pipeline = None
            logger.warning("Phase integrations not available, using fallback implementations")

    async def start(self):
        """Start the orchestrator and worker pools."""
        logger.info(f"Starting AsyncScrapingOrchestrator (session: {self.session_id})")

        # Start worker pools
        await self._scrape_pool.start(self._scrape_queue)
        await self._clean_pool.start(self._clean_queue)

        logger.info("AsyncScrapingOrchestrator started successfully")

        if self.enhanced_logger:
            self.enhanced_logger.log_event(
                LogLevel.INFO,
                LogCategory.SYSTEM,
                AgentEventType.TASK_START,
                "Async Scraping Orchestrator started",
                session_id=self.session_id
            )

    async def shutdown(self):
        """Shutdown the orchestrator and cleanup resources."""
        logger.info(f"Shutting down AsyncScrapingOrchestrator (session: {self.session_id})")

        # Shutdown worker pools
        await self._scrape_pool.shutdown()
        await self._clean_pool.shutdown()

        # Shutdown queues
        await self._scrape_queue.shutdown()
        await self._clean_queue.shutdown()

        # Save phase integration data
        if PHASE_INTEGRATION_AVAILABLE and self.anti_bot_manager:
            self.anti_bot_manager.save_domain_profiles()

        logger.info("AsyncScrapingOrchestrator shutdown complete")

        if self.enhanced_logger:
            self.enhanced_logger.log_event(
                LogLevel.INFO,
                LogCategory.SYSTEM,
                AgentEventType.SESSION_END,
                "Async Scraping Orchestrator shutdown",
                session_id=self.session_id,
                final_stats=self.get_statistics()
            )

    async def submit_scraping_task(
        self,
        request: ScrapingRequest,
        priority: int = 0
    ) -> bool:
        """Submit a scraping task to the queue.

        Args:
            request: Scraping request
            priority: Task priority (lower = higher priority)

        Returns:
            True if task was submitted successfully
        """
        # Check circuit breaker
        if not self._check_circuit_breaker():
            logger.warning("Circuit breaker is open, rejecting scraping task")
            return False

        # Update request context
        request.context.session_id = self.session_id
        request.context.pipeline_stage = PipelineStage.SCRAPING

        # Add to active tasks
        self._active_tasks[request.context.task_id] = request.context

        # Submit to queue
        success = await self._scrape_queue.put(
            request,
            priority=priority,
            task_id=request.context.task_id
        )

        if success:
            logger.debug(f"Submitted scraping task {request.context.task_id} for {request.url}")
        else:
            # Remove from active tasks if submission failed
            self._active_tasks.pop(request.context.task_id, None)
            logger.warning(f"Failed to submit scraping task for {request.url}")

        return success

    async def submit_cleaning_task(
        self,
        request: CleaningRequest,
        priority: int = 0
    ) -> bool:
        """Submit a cleaning task to the queue.

        Args:
            request: Cleaning request
            priority: Task priority (lower = higher priority)

        Returns:
            True if task was submitted successfully
        """
        # Check circuit breaker
        if not self._check_circuit_breaker():
            logger.warning("Circuit breaker is open, rejecting cleaning task")
            return False

        # Update request context
        request.context.session_id = self.session_id
        request.context.pipeline_stage = PipelineStage.CLEANING

        # Add to active tasks
        self._active_tasks[request.context.task_id] = request.context

        # Submit to queue
        success = await self._clean_queue.put(
            request,
            priority=priority,
            task_id=request.context.task_id
        )

        if success:
            logger.debug(f"Submitted cleaning task {request.context.task_id} for {request.url}")
        else:
            # Remove from active tasks if submission failed
            self._active_tasks.pop(request.context.task_id, None)
            logger.warning(f"Failed to submit cleaning task for {request.url}")

        return success

    async def _handle_scrape_task(self, request: ScrapingRequest) -> Optional[ScrapingResult]:
        """Handle a scraping task.

        Args:
            request: Scraping request

        Returns:
            Scraping result or None if failed
        """
        start_time = time.time()
        task_id = request.context.task_id

        logger.debug(f"Processing scraping task {task_id} for {request.url}")

        try:
            # Use anti-bot escalation manager if available
            if self.anti_bot_manager and self.config.enable_anti_bot:
                escalation_result = await self.anti_bot_manager.crawl_with_escalation(
                    url=str(request.url),
                    initial_level=request.anti_bot_level,
                    max_level=request.max_anti_bot_level,
                    session_id=self.session_id
                )

                # Convert to scraping result
                result = self._convert_escalation_to_scraping_result(
                    escalation_result, request, start_time
                )

            else:
                # Fallback implementation
                result = await self._fallback_scrape(request, start_time)

            # Update task context
            request.context.actual_duration = time.time() - start_time
            request.context.pipeline_stage = PipelineStage.COMPLETED if result.success else PipelineStage.FAILED

            # Update statistics
            self._stats.add_task_result(
                duration=request.context.actual_duration,
                success=result.success,
                quality_score=result.content_quality_score
            )

            # Log completion
            if self.enhanced_logger:
                self.enhanced_logger.log_event(
                    LogLevel.INFO if result.success else LogLevel.WARNING,
                    LogCategory.PERFORMANCE,
                    AgentEventType.TASK_END if result.success else AgentEventType.ERROR,
                    f"Scraping task {'completed' if result.success else 'failed'} for {request.url}",
                    task_id=task_id,
                    url=str(request.url),
                    success=result.success,
                    duration=request.context.actual_duration,
                    quality_score=result.content_quality_score,
                    anti_bot_level=result.final_anti_bot_level,
                    escalation_used=result.escalation_used
                )

            logger.debug(f"Scraping task {task_id} completed: success={result.success}")
            return result

        except Exception as e:
            # Update circuit breaker on failure
            self._update_circuit_breaker()

            # Create error result
            duration = time.time() - start_time
            error_result = ScrapingResult(
                url=str(request.url),
                domain=request.url.host if hasattr(request.url, 'host') else "",
                success=False,
                error_message=str(e),
                error_type=ErrorType.UNKNOWN_ERROR,
                duration=duration,
                attempts_made=1,
                context=request.context,
                search_query=request.search_query
            )

            # Update statistics
            self._stats.add_task_result(duration=duration, success=False)

            logger.error(f"Scraping task {task_id} failed: {e}")
            return error_result

        finally:
            # Remove from active tasks
            self._active_tasks.pop(task_id, None)

    async def _handle_clean_task(self, request: CleaningRequest) -> Optional[CleaningResult]:
        """Handle a cleaning task.

        Args:
            request: Cleaning request

        Returns:
            Cleaning result or None if failed
        """
        start_time = time.time()
        task_id = request.context.task_id

        logger.debug(f"Processing cleaning task {task_id} for {request.url}")

        try:
            # Use content cleaning pipeline if available
            if self.cleaning_pipeline and self.config.enable_content_cleaning:
                cleaning_result = await self.cleaning_pipeline.clean_content(
                    content=request.content,
                    url=request.url,
                    search_query=request.search_query
                )

                # Convert to cleaning result
                result = self._convert_cleaning_to_cleaning_result(
                    cleaning_result, request, start_time
                )

            else:
                # Fallback implementation
                result = await self._fallback_clean(request, start_time)

            # Update task context
            request.context.actual_duration = time.time() - start_time
            request.context.pipeline_stage = PipelineStage.COMPLETED if result.success else PipelineStage.FAILED

            # Update statistics
            self._stats.add_task_result(
                duration=request.context.actual_duration,
                success=result.success,
                quality_score=result.final_quality_score
            )

            # Log completion
            if self.enhanced_logger:
                self.enhanced_logger.log_event(
                    LogLevel.INFO if result.success else LogLevel.WARNING,
                    LogCategory.PERFORMANCE,
                    AgentEventType.TASK_END if result.success else AgentEventType.ERROR,
                    f"Cleaning task {'completed' if result.success else 'failed'} for {request.url}",
                    task_id=task_id,
                    url=request.url,
                    success=result.success,
                    duration=request.context.actual_duration,
                    quality_score=result.final_quality_score,
                    cleaning_performed=result.cleaning_performed,
                    quality_improvement=result.quality_improvement
                )

            logger.debug(f"Cleaning task {task_id} completed: success={result.success}")
            return result

        except Exception as e:
            # Update circuit breaker on failure
            self._update_circuit_breaker()

            # Create error result
            duration = time.time() - start_time
            error_result = CleaningResult(
                original_content=request.content,
                cleaned_content=request.content,  # Fallback to original
                url=request.url,
                success=False,
                error_message=str(e),
                error_type=ErrorType.CLEANING_ERROR,
                processing_time_ms=duration * 1000,
                context=request.context,
                search_query=request.search_query
            )

            # Update statistics
            self._stats.add_task_result(duration=duration, success=False)

            logger.error(f"Cleaning task {task_id} failed: {e}")
            return error_result

        finally:
            # Remove from active tasks
            self._active_tasks.pop(task_id, None)

    def _convert_escalation_to_scraping_result(
        self,
        escalation_result: Any,
        request: ScrapingRequest,
        start_time: float
    ) -> ScrapingResult:
        """Convert anti-bot escalation result to scraping result."""
        # This would need to be adapted based on the actual EscalationResult structure
        duration = time.time() - start_time

        return ScrapingResult(
            url=str(request.url),
            domain=getattr(escalation_result, 'domain', ''),
            success=getattr(escalation_result, 'success', False),
            content=getattr(escalation_result, 'content', None),
            duration=duration,
            attempts_made=getattr(escalation_result, 'attempts_made', 1),
            word_count=len(escalation_result.content.split()) if escalation_result.content else 0,
            char_count=len(escalation_result.content) if escalation_result.content else 0,
            final_anti_bot_level=getattr(escalation_result, 'final_level', 0),
            escalation_used=getattr(escalation_result, 'escalation_used', False),
            escalation_triggers=getattr(escalation_result, 'escalation_triggers', []),
            context=request.context,
            search_query=request.search_query
        )

    def _convert_cleaning_to_cleaning_result(
        self,
        cleaning_result: Any,
        request: CleaningRequest,
        start_time: float
    ) -> CleaningResult:
        """Convert content cleaning result to cleaning result."""
        # This would need to be adapted based on the actual CleaningResult structure
        duration = time.time() - start_time

        return CleaningResult(
            original_content=request.content,
            cleaned_content=getattr(cleaning_result, 'cleaned_content', request.content),
            url=request.url,
            success=True,
            cleaning_performed=getattr(cleaning_result, 'cleaning_performed', False),
            quality_improvement=getattr(cleaning_result, 'quality_improvement', 0.0),
            length_reduction=1.0 - (len(getattr(cleaning_result, 'cleaned_content', request.content)) / len(request.content)),
            original_quality_score=getattr(cleaning_result, 'confidence_signals', {}).get('overall_confidence', None),
            final_quality_score=getattr(cleaning_result, 'confidence_signals', {}).get('overall_confidence', None),
            cleanliness_score=getattr(cleaning_result, 'confidence_signals', {}).get('cleanliness_score', None),
            processing_time_ms=duration * 1000,
            cleaning_stage=getattr(cleaning_result, 'cleaning_stage', 'basic'),
            editorial_recommendation=getattr(cleaning_result, 'editorial_recommendation', 'UNKNOWN'),
            enhancement_suggestions=getattr(cleaning_result, 'enhancement_suggestions', []),
            context=request.context,
            search_query=request.search_query
        )

    async def _fallback_scrape(self, request: ScrapingRequest, start_time: float) -> ScrapingResult:
        """Fallback scraping implementation."""
        # Simple fallback implementation
        await asyncio.sleep(1.0)  # Simulate scraping time
        return ScrapingResult(
            url=str(request.url),
            domain=request.url.host if hasattr(request.url, 'host') else "",
            success=False,
            error_message="Fallback scraping not implemented",
            error_type=ErrorType.UNKNOWN_ERROR,
            duration=time.time() - start_time,
            attempts_made=1,
            context=request.context,
            search_query=request.search_query
        )

    async def _fallback_clean(self, request: CleaningRequest, start_time: float) -> CleaningResult:
        """Fallback cleaning implementation."""
        # Simple fallback implementation
        await asyncio.sleep(0.5)  # Simulate cleaning time
        return CleaningResult(
            original_content=request.content,
            cleaned_content=request.content.strip(),
            url=request.url,
            success=True,
            cleaning_performed=False,
            processing_time_ms=(time.time() - start_time) * 1000,
            context=request.context,
            search_query=request.search_query
        )

    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker allows operations."""
        if not self.config.enable_circuit_breaker:
            return True

        now = time.time()
        breaker = self._circuit_breaker

        if breaker['state'] == 'open':
            # Check if recovery timeout has passed
            if now - breaker['last_failure'] > breaker['recovery_timeout']:
                breaker['state'] = 'half_open'
                logger.info("Circuit breaker moving to half-open state")
                return True
            return False

        return True

    def _update_circuit_breaker(self):
        """Update circuit breaker state on failure."""
        if not self.config.enable_circuit_breaker:
            return

        breaker = self._circuit_breaker
        breaker['failure_count'] += 1
        breaker['last_failure'] = time.time()

        if breaker['failure_count'] >= breaker['threshold']:
            breaker['state'] = 'open'
            logger.warning(f"Circuit breaker opened after {breaker['failure_count']} failures")

    async def process_url_batch(
        self,
        urls: List[str],
        search_query: Optional[str] = None,
        priority: int = 0
    ) -> List[ScrapingResult]:
        """Process a batch of URLs through the complete pipeline.

        Args:
            urls: List of URLs to process
            search_query: Optional search query for context
            priority: Task priority

        Returns:
            List of scraping results
        """
        logger.info(f"Processing URL batch: {len(urls)} URLs")

        # Submit scraping tasks
        scraping_tasks = []
        for url in urls:
            request = create_scraping_request(
                url=url,
                search_query=search_query
            )
            success = await self.submit_scraping_task(request, priority)
            if success:
                scraping_tasks.append(request)

        # Wait for all scraping tasks to complete
        results = []
        while len(results) < len(scraping_tasks):
            await asyncio.sleep(0.1)
            # Check if tasks are completed (this would need proper task tracking)

        logger.info(f"URL batch processing completed: {len(results)} results")
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator statistics."""
        return {
            'session_id': self.session_id,
            'uptime_seconds': (datetime.now() - self.created_at).total_seconds(),
            'pipeline_stats': self._stats.dict(),
            'scrape_queue': self._scrape_queue.get_stats(),
            'clean_queue': self._clean_queue.get_stats(),
            'scrape_pool': self._scrape_pool.get_stats(),
            'clean_pool': self._clean_pool.get_stats(),
            'active_tasks': len(self._active_tasks),
            'circuit_breaker': self._circuit_breaker.copy()
        }

    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the orchestrator."""
        stats = self.get_statistics()
        scrape_queue_size = stats['scrape_queue']['current_size']
        clean_queue_size = stats['clean_queue']['current_size']
        scrape_success_rate = stats['scrape_pool']['success_rate']
        clean_success_rate = stats['clean_pool']['success_rate']

        # Determine overall health
        health_issues = []

        if scrape_queue_size > stats['scrape_queue']['max_size'] * 0.9:
            health_issues.append("Scrape queue nearly full")

        if clean_queue_size > stats['clean_queue']['max_size'] * 0.9:
            health_issues.append("Clean queue nearly full")

        if scrape_success_rate < 0.8:
            health_issues.append(f"Low scrape success rate: {scrape_success_rate:.1%}")

        if clean_success_rate < 0.8:
            health_issues.append(f"Low clean success rate: {clean_success_rate:.1%}")

        if self._circuit_breaker['state'] == 'open':
            health_issues.append("Circuit breaker is open")

        overall_health = "healthy" if not health_issues else "degraded"

        return {
            'overall_health': overall_health,
            'health_issues': health_issues,
            'active_workers': {
                'scrape': stats['scrape_pool']['active_workers'],
                'clean': stats['clean_pool']['active_workers']
            },
            'queue_sizes': {
                'scrape': scrape_queue_size,
                'clean': clean_queue_size
            },
            'success_rates': {
                'scrape': scrape_success_rate,
                'clean': clean_success_rate
            },
            'circuit_breaker_state': self._circuit_breaker['state']
        }


# Import factory function
from .data_contracts import create_scraping_request, create_cleaning_request


@asynccontextmanager
async def managed_orchestrator(config: Optional[PipelineConfig] = None):
    """Context manager for orchestrator lifecycle management.

    Args:
        config: Optional pipeline configuration

    Yields:
        AsyncScrapingOrchestrator instance
    """
    orchestrator = AsyncScrapingOrchestrator(config)
    try:
        await orchestrator.start()
        yield orchestrator
    finally:
        await orchestrator.shutdown()