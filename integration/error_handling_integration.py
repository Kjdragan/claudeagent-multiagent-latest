"""
Error Handling & Recovery Integration for Multi-Agent Research System.

This module provides comprehensive error handling and recovery mechanisms for the agent-based
research system, including retry logic with escalation, graceful degradation, circuit breaker
patterns, and system resilience for the comprehensive research workflow.

Phase 2.4 Implementation: Create Error Handling & Recovery for agent-based system
"""

import asyncio
import logging
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Type
from pathlib import Path

# Import system components
from integration.kevin_directory_integration import KevinDirectoryIntegration
from integration.mcp_tool_integration import MCPToolIntegration
from integration.quality_assurance_integration import QualityAssuranceIntegration


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification and handling."""
    NETWORK = "network"
    API = "api"
    SYSTEM = "system"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    AUTHENTICATION = "authentication"
    CONTENT = "content"
    QUALITY = "quality"
    WORKFLOW = "workflow"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY = "retry"
    ESCALATE = "escalate"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    CIRCUIT_BREAKER = "circuit_breaker"
    MANUAL_INTERVENTION = "manual_intervention"
    SKIP_OPERATION = "skip_operation"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    session_id: str
    operation: str
    component: str
    stage: str
    timestamp: datetime
    additional_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemError:
    """Structured error representation."""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    original_exception: Optional[Exception]
    context: ErrorContext
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    recovery_strategy: Optional[RecoveryStrategy] = None
    resolved: bool = False
    resolution_details: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True
    retry_on_exceptions: List[Type[Exception]] = field(default_factory=list)
    stop_on_exceptions: List[Type[Exception]] = field(default_factory=list)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern."""
    failure_threshold: int = 5
    timeout: float = 60.0
    expected_exception: Type[Exception] = Exception
    recovery_timeout: float = 30.0


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, calls fail immediately
    HALF_OPEN = "half_open"  # Testing if service has recovered


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""

    def __init__(self, config: CircuitBreakerConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self.next_attempt = None

    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt to reset."""
        return (
            self.next_attempt is None or
            datetime.now() >= self.next_attempt
        )

    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
        self.logger.debug("Circuit breaker reset to CLOSED")

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.next_attempt = datetime.now() + timedelta(seconds=self.config.recovery_timeout)
            self.logger.warning(f"Circuit breaker opened due to {self.failure_count} failures")


class ErrorHandlingIntegration:
    """
    Comprehensive error handling and recovery integration for the agent-based research system.

    Provides intelligent error classification, recovery strategies, retry logic with escalation,
    circuit breaker patterns, and system resilience mechanisms.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Error Handling Integration component.

        Args:
            config: Optional configuration for error handling behavior
        """
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)

        # Initialize system integrations
        self.kevin_integration = KevinDirectoryIntegration()
        self.mcp_integration = MCPToolIntegration()
        self.quality_integration = QualityAssuranceIntegration()

        # Error tracking and recovery
        self.error_history: List[SystemError] = []
        self.active_errors: Dict[str, SystemError] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

        # Recovery strategies and handlers
        self.recovery_handlers: Dict[ErrorCategory, List[Callable]] = {
            category: [] for category in ErrorCategory
        }

        # Metrics and monitoring
        self.error_metrics: Dict[str, Any] = {
            "total_errors": 0,
            "resolved_errors": 0,
            "escalated_errors": 0,
            "errors_by_category": {},
            "errors_by_severity": {},
            "recovery_success_rate": 0.0
        }

        self.logger.info("ErrorHandlingIntegration initialized with comprehensive recovery strategies")

    async def handle_error(
        self,
        exception: Exception,
        context: ErrorContext,
        recovery_attempts: int = 0
    ) -> Dict[str, Any]:
        """
        Handle an error with comprehensive recovery logic.

        Args:
            exception: The exception that occurred
            context: Error context information
            recovery_attempts: Number of recovery attempts already made

        Returns:
            Error handling result with recovery strategy and outcome
        """
        self.logger.error(f"Handling error in {context.operation}: {str(exception)}")

        try:
            # Classify and create structured error
            system_error = await self._classify_error(exception, context)
            system_error.recovery_attempts = recovery_attempts

            # Record error in history
            await self._record_error(system_error)

            # Determine recovery strategy
            recovery_strategy = await self._determine_recovery_strategy(system_error)
            system_error.recovery_strategy = recovery_strategy

            # Execute recovery strategy
            recovery_result = await self._execute_recovery_strategy(
                system_error, recovery_strategy
            )

            # Update error metrics
            await self._update_error_metrics(system_error, recovery_result)

            # Log error handling outcome
            await self._log_error_handling(system_error, recovery_result)

            return {
                "success": recovery_result["success"],
                "error_id": system_error.error_id,
                "recovery_strategy": recovery_strategy.value,
                "recovery_result": recovery_result,
                "error_details": {
                    "category": system_error.category.value,
                    "severity": system_error.severity.value,
                    "message": system_error.message
                },
                "timestamp": datetime.now().isoformat()
            }

        except Exception as handling_error:
            self.logger.critical(f"Error handling itself failed: {str(handling_error)}")
            return {
                "success": False,
                "error": "Error handling failure",
                "original_error": str(exception),
                "handling_error": str(handling_error),
                "timestamp": datetime.now().isoformat()
            }

    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        retry_config: Optional[RetryConfig] = None,
        context: Optional[ErrorContext] = None,
        **kwargs
    ) -> Any:
        """
        Execute a function with intelligent retry logic.

        Args:
            func: Function to execute
            *args: Function arguments
            retry_config: Retry configuration
            context: Error context for retry attempts
            **kwargs: Function keyword arguments

        Returns:
            Function result or raises last exception

        Raises:
            Last exception if all retries are exhausted
        """
        config = retry_config or self._get_default_retry_config()
        context = context or self._create_default_context()

        last_exception = None

        for attempt in range(config.max_attempts):
            try:
                # Check if we should use circuit breaker
                circuit_breaker_key = f"{context.component}_{context.operation}"
                if circuit_breaker_key in self.circuit_breakers:
                    circuit_breaker = self.circuit_breakers[circuit_breaker_key]
                    return await circuit_breaker.call(func, *args, **kwargs)
                else:
                    return await func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                # Check if we should stop retrying on this exception
                if any(isinstance(e, exc_type) for exc_type in config.stop_on_exceptions):
                    self.logger.error(f"Stopping retry due to non-retryable exception: {str(e)}")
                    raise e

                # Check if this is a retryable exception
                if config.retry_on_exceptions and not any(isinstance(e, exc_type) for exc_type in config.retry_on_exceptions):
                    self.logger.warning(f"Exception not configured for retry: {str(e)}")
                    raise e

                if attempt < config.max_attempts - 1:
                    # Calculate delay with exponential backoff
                    delay = self._calculate_retry_delay(config, attempt)

                    self.logger.warning(
                        f"Attempt {attempt + 1} failed for {context.operation}: {str(e)}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )

                    # Log retry attempt
                    await self._log_retry_attempt(context, e, attempt + 1, delay)

                    # Wait before retry
                    await asyncio.sleep(delay)

        # All retries exhausted
        self.logger.error(
            f"All {config.max_attempts} retry attempts failed for {context.operation}"
        )
        raise last_exception

    async def execute_with_circuit_breaker(
        self,
        func: Callable,
        *args,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        context: Optional[ErrorContext] = None,
        **kwargs
    ) -> Any:
        """
        Execute a function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Function arguments
            circuit_breaker_config: Circuit breaker configuration
            context: Error context
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception if circuit breaker is open or function fails
        """
        config = circuit_breaker_config or self._get_default_circuit_breaker_config()
        context = context or self._create_default_context()

        # Create or get circuit breaker
        cb_key = f"{context.component}_{context.operation}"
        if cb_key not in self.circuit_breakers:
            self.circuit_breakers[cb_key] = CircuitBreaker(config, self.logger)

        circuit_breaker = self.circuit_breakers[cb_key]

        try:
            return await circuit_breaker.call(func, *args, **kwargs)
        except Exception as e:
            self.logger.error(f"Circuit breaker protected call failed: {str(e)}")
            raise e

    async def execute_with_fallback(
        self,
        primary_func: Callable,
        fallback_func: Callable,
        *args,
        context: Optional[ErrorContext] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a function with fallback support.

        Args:
            primary_func: Primary function to execute
            fallback_func: Fallback function to execute if primary fails
            *args: Function arguments
            context: Error context
            **kwargs: Function keyword arguments

        Returns:
            Execution result with fallback information
        """
        context = context or self._create_default_context()

        try:
            # Try primary function
            result = await primary_func(*args, **kwargs)

            return {
                "success": True,
                "result": result,
                "used_fallback": False,
                "fallback_reason": None,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as primary_error:
            self.logger.warning(
                f"Primary function failed for {context.operation}: {str(primary_error)}. "
                "Executing fallback function."
            )

            try:
                # Execute fallback function
                fallback_result = await fallback_func(*args, **kwargs)

                return {
                    "success": True,
                    "result": fallback_result,
                    "used_fallback": True,
                    "fallback_reason": str(primary_error),
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as fallback_error:
                self.logger.error(
                    f"Both primary and fallback functions failed for {context.operation}. "
                    f"Primary: {str(primary_error)}, Fallback: {str(fallback_error)}"
                )

                return {
                    "success": False,
                    "error": "Both primary and fallback failed",
                    "primary_error": str(primary_error),
                    "fallback_error": str(fallback_error),
                    "timestamp": datetime.now().isoformat()
                }

    async def handle_graceful_degradation(
        self,
        operation: str,
        full_functionality: Callable,
        reduced_functionality: Callable,
        minimal_functionality: Callable,
        context: Optional[ErrorContext] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute operation with graceful degradation levels.

        Args:
            operation: Operation identifier
            full_functionality: Full functionality function
            reduced_functionality: Reduced functionality function
            minimal_functionality: Minimal functionality function
            context: Error context
            **kwargs: Function keyword arguments

        Returns:
            Execution result with degradation level
        """
        context = context or self._create_default_context()
        context.operation = operation

        degradation_levels = [
            ("full", full_functionality, "Full functionality"),
            ("reduced", reduced_functionality, "Reduced functionality"),
            ("minimal", minimal_functionality, "Minimal functionality")
        ]

        for level, func, description in degradation_levels:
            try:
                self.logger.info(f"Attempting {description} for {operation}")
                result = await func(**kwargs)

                return {
                    "success": True,
                    "result": result,
                    "degradation_level": level,
                    "description": description,
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                self.logger.warning(
                    f"{description} failed for {operation}: {str(e)}. "
                    f"Degrading to next level."
                )

        # All levels failed
        return {
            "success": False,
            "error": "All degradation levels failed",
            "operation": operation,
            "timestamp": datetime.now().isoformat()
        }

    async def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive error statistics and metrics.

        Returns:
            Error statistics and performance metrics
        """
        total_errors = len(self.error_history)
        resolved_errors = len([e for e in self.error_history if e.resolved])

        # Calculate resolution rate
        resolution_rate = (resolved_errors / total_errors * 100) if total_errors > 0 else 0

        # Categorize errors
        errors_by_category = {}
        errors_by_severity = {}

        for error in self.error_history:
            category = error.category.value
            severity = error.severity.value

            errors_by_category[category] = errors_by_category.get(category, 0) + 1
            errors_by_severity[severity] = errors_by_severity.get(severity, 0) + 1

        # Calculate error trends
        recent_errors = [
            e for e in self.error_history
            if e.timestamp > datetime.now() - timedelta(hours=24)
        ]

        error_trends = {
            "last_24_hours": len(recent_errors),
            "last_hour": len([
                e for e in recent_errors
                if e.timestamp > datetime.now() - timedelta(hours=1)
            ]),
            "trend_direction": self._calculate_error_trend(recent_errors)
        }

        return {
            "total_errors": total_errors,
            "resolved_errors": resolved_errors,
            "resolution_rate": resolution_rate,
            "errors_by_category": errors_by_category,
            "errors_by_severity": errors_by_severity,
            "error_trends": error_trends,
            "active_circuit_breakers": len([
                cb for cb in self.circuit_breakers.values()
                if cb.state == CircuitBreakerState.OPEN
            ]),
            "total_circuit_breakers": len(self.circuit_breakers),
            "last_updated": datetime.now().isoformat()
        }

    async def get_active_errors(self) -> List[Dict[str, Any]]:
        """
        Get list of currently active (unresolved) errors.

        Returns:
            List of active error details
        """
        active_errors = [
            error for error in self.error_history
            if not error.resolved
        ]

        return [
            {
                "error_id": error.error_id,
                "category": error.category.value,
                "severity": error.severity.value,
                "message": error.message,
                "operation": error.context.operation,
                "component": error.context.component,
                "recovery_attempts": error.recovery_attempts,
                "timestamp": error.timestamp.isoformat(),
                "recovery_strategy": error.recovery_strategy.value if error.recovery_strategy else None
            }
            for error in active_errors
        ]

    async def clear_resolved_errors(self, older_than_hours: int = 24) -> Dict[str, Any]:
        """
        Clear resolved errors from history.

        Args:
            older_than_hours: Clear errors older than this many hours

        Returns:
            Clear operation result
        """
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)

        original_count = len(self.error_history)

        # Remove old resolved errors
        self.error_history = [
            error for error in self.error_history
            if not (error.resolved and error.timestamp < cutoff_time)
        ]

        cleared_count = original_count - len(self.error_history)

        self.logger.info(f"Cleared {cleared_count} resolved errors older than {older_than_hours} hours")

        return {
            "success": True,
            "original_count": original_count,
            "cleared_count": cleared_count,
            "remaining_count": len(self.error_history),
            "timestamp": datetime.now().isoformat()
        }

    async def register_recovery_handler(
        self,
        category: ErrorCategory,
        handler: Callable
    ) -> Dict[str, Any]:
        """
        Register a custom recovery handler for an error category.

        Args:
            category: Error category to handle
            handler: Recovery handler function

        Returns:
            Registration result
        """
        self.recovery_handlers[category].append(handler)

        self.logger.info(f"Registered recovery handler for {category.value} category")

        return {
            "success": True,
            "category": category.value,
            "handler_count": len(self.recovery_handlers[category]),
            "timestamp": datetime.now().isoformat()
        }

    # Private helper methods

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default error handling configuration."""
        return {
            "max_recovery_attempts": 3,
            "default_retry_attempts": 3,
            "circuit_breaker_failure_threshold": 5,
            "circuit_breaker_timeout": 60.0,
            "enable_graceful_degradation": True,
            "enable_circuit_breakers": True,
            "log_all_errors": True,
            "persist_error_history": True
        }

    def _get_default_retry_config(self) -> RetryConfig:
        """Get default retry configuration."""
        return RetryConfig(
            max_attempts=3,
            base_delay=1.0,
            max_delay=30.0,
            backoff_factor=2.0,
            jitter=True,
            retry_on_exceptions=[
                ConnectionError,
                TimeoutError,
                OSError
            ],
            stop_on_exceptions=[
                ValueError,
                TypeError,
                KeyError
            ]
        )

    def _get_default_circuit_breaker_config(self) -> CircuitBreakerConfig:
        """Get default circuit breaker configuration."""
        return CircuitBreakerConfig(
            failure_threshold=5,
            timeout=60.0,
            expected_exception=Exception,
            recovery_timeout=30.0
        )

    def _create_default_context(self) -> ErrorContext:
        """Create default error context."""
        return ErrorContext(
            session_id="unknown",
            operation="unknown_operation",
            component="unknown_component",
            stage="unknown_stage",
            timestamp=datetime.now()
        )

    async def _classify_error(self, exception: Exception, context: ErrorContext) -> SystemError:
        """Classify and create structured error."""
        error_id = f"error_{int(time.time() * 1000)}_{id(exception)}"

        # Determine error category
        category = self._determine_error_category(exception)

        # Determine error severity
        severity = self._determine_error_severity(exception, context)

        return SystemError(
            error_id=error_id,
            category=category,
            severity=severity,
            message=str(exception),
            original_exception=exception,
            context=context
        )

    def _determine_error_category(self, exception: Exception) -> ErrorCategory:
        """Determine error category from exception type and message."""
        message = str(exception).lower()
        exception_type = type(exception).__name__.lower()

        # Network-related errors
        if any(keyword in message or keyword in exception_type for keyword in [
            "connection", "network", "timeout", "socket", "dns", "http"
        ]):
            return ErrorCategory.NETWORK

        # API-related errors
        if any(keyword in message or keyword in exception_type for keyword in [
            "api", "key", "authentication", "unauthorized", "forbidden", "rate limit"
        ]):
            return ErrorCategory.API

        # System-related errors
        if any(keyword in message or keyword in exception_type for keyword in [
            "memory", "disk", "system", "resource", "permission"
        ]):
            return ErrorCategory.SYSTEM

        # Validation errors
        if any(keyword in message or keyword in exception_type for keyword in [
            "validation", "invalid", "missing", "required", "format"
        ]):
            return ErrorCategory.VALIDATION

        # Content errors
        if any(keyword in message or keyword in exception_type for keyword in [
            "content", "parse", "encoding", "format"
        ]):
            return ErrorCategory.CONTENT

        # Quality errors
        if any(keyword in message or keyword in exception_type for keyword in [
            "quality", "assessment", "score", "threshold"
        ]):
            return ErrorCategory.QUALITY

        # Workflow errors
        if any(keyword in message or keyword in exception_type for keyword in [
            "workflow", "stage", "process", "pipeline"
        ]):
            return ErrorCategory.WORKFLOW

        return ErrorCategory.UNKNOWN

    def _determine_error_severity(self, exception: Exception, context: ErrorContext) -> ErrorSeverity:
        """Determine error severity based on exception and context."""
        # Critical errors
        if any(keyword in str(exception).lower() for keyword in [
            "critical", "fatal", "crash", "corrupt"
        ]):
            return ErrorSeverity.CRITICAL

        # High severity errors
        if any(keyword in context.stage.lower() for keyword in [
            "final", "critical", "production"
        ]):
            return ErrorSeverity.HIGH

        # Medium severity errors
        if any(keyword in str(exception).lower() for keyword in [
            "failed", "error", "exception"
        ]):
            return ErrorSeverity.MEDIUM

        return ErrorSeverity.LOW

    async def _determine_recovery_strategy(self, error: SystemError) -> RecoveryStrategy:
        """Determine appropriate recovery strategy for an error."""
        # Check if we've exceeded recovery attempts
        if error.recovery_attempts >= error.max_recovery_attempts:
            return RecoveryStrategy.MANUAL_INTERVENTION

        # Determine strategy based on error category and severity
        if error.category == ErrorCategory.NETWORK:
            return RecoveryStrategy.RETRY

        elif error.category == ErrorCategory.API:
            if error.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
                return RecoveryStrategy.ESCALATE
            else:
                return RecoveryStrategy.RETRY

        elif error.category == ErrorCategory.RESOURCE:
            return RecoveryStrategy.GRACEFUL_DEGRADATION

        elif error.category == ErrorCategory.QUALITY:
            return RecoveryStrategy.FALLBACK

        elif error.severity == ErrorSeverity.CRITICAL:
            return RecoveryStrategy.ESCALATE

        elif error.severity == ErrorSeverity.HIGH:
            return RecoveryStrategy.CIRCUIT_BREAKER

        else:
            return RecoveryStrategy.RETRY

    async def _execute_recovery_strategy(
        self,
        error: SystemError,
        strategy: RecoveryStrategy
    ) -> Dict[str, Any]:
        """Execute the determined recovery strategy."""
        try:
            if strategy == RecoveryStrategy.RETRY:
                return await self._execute_retry_recovery(error)
            elif strategy == RecoveryStrategy.FALLBACK:
                return await self._execute_fallback_recovery(error)
            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                return await self._execute_graceful_degradation_recovery(error)
            elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                return await self._execute_circuit_breaker_recovery(error)
            elif strategy == RecoveryStrategy.ESCALATE:
                return await self._execute_escalation_recovery(error)
            elif strategy == RecoveryStrategy.MANUAL_INTERVENTION:
                return await self._execute_manual_intervention_recovery(error)
            else:
                return await self._execute_skip_operation_recovery(error)

        except Exception as recovery_error:
            self.logger.error(f"Recovery strategy execution failed: {str(recovery_error)}")
            return {
                "success": False,
                "recovery_error": str(recovery_error),
                "strategy": strategy.value
            }

    async def _execute_retry_recovery(self, error: SystemError) -> Dict[str, Any]:
        """Execute retry recovery strategy."""
        self.logger.info(f"Executing retry recovery for error {error.error_id}")

        # This would implement retry logic based on the operation that failed
        # For now, return a placeholder result
        return {
            "success": False,
            "strategy": "retry",
            "message": "Retry recovery not yet implemented for this operation type",
            "recommended_action": "Implement retry logic for specific operations"
        }

    async def _execute_fallback_recovery(self, error: SystemError) -> Dict[str, Any]:
        """Execute fallback recovery strategy."""
        self.logger.info(f"Executing fallback recovery for error {error.error_id}")

        return {
            "success": False,
            "strategy": "fallback",
            "message": "Fallback recovery not yet implemented for this operation type",
            "recommended_action": "Implement fallback logic for specific operations"
        }

    async def _execute_graceful_degradation_recovery(self, error: SystemError) -> Dict[str, Any]:
        """Execute graceful degradation recovery strategy."""
        self.logger.info(f"Executing graceful degradation recovery for error {error.error_id}")

        return {
            "success": False,
            "strategy": "graceful_degradation",
            "message": "Graceful degradation recovery not yet implemented for this operation type",
            "recommended_action": "Implement graceful degradation logic for specific operations"
        }

    async def _execute_circuit_breaker_recovery(self, error: SystemError) -> Dict[str, Any]:
        """Execute circuit breaker recovery strategy."""
        self.logger.info(f"Executing circuit breaker recovery for error {error.error_id}")

        # Create or update circuit breaker
        cb_key = f"{error.context.component}_{error.context.operation}"
        if cb_key not in self.circuit_breakers:
            config = self._get_default_circuit_breaker_config()
            self.circuit_breakers[cb_key] = CircuitBreaker(config, self.logger)

        circuit_breaker = self.circuit_breakers[cb_key]
        circuit_breaker._on_failure()  # Manually trigger failure

        return {
            "success": True,
            "strategy": "circuit_breaker",
            "message": f"Circuit breaker activated for {cb_key}",
            "circuit_breaker_state": circuit_breaker.state.value
        }

    async def _execute_escalation_recovery(self, error: SystemError) -> Dict[str, Any]:
        """Execute escalation recovery strategy."""
        self.logger.info(f"Executing escalation recovery for error {error.error_id}")

        # Log escalation to KEVIN directory
        await self._log_error_escalation(error)

        return {
            "success": True,
            "strategy": "escalation",
            "message": "Error has been escalated for manual review",
            "escalation_details": {
                "error_id": error.error_id,
                "severity": error.severity.value,
                "category": error.category.value,
                "requires_manual_intervention": True
            }
        }

    async def _execute_manual_intervention_recovery(self, error: SystemError) -> Dict[str, Any]:
        """Execute manual intervention recovery strategy."""
        self.logger.info(f"Executing manual intervention recovery for error {error.error_id}")

        # Create manual intervention request
        await self._create_manual_intervention_request(error)

        return {
            "success": False,
            "strategy": "manual_intervention",
            "message": "Manual intervention required - all automated recovery attempts exhausted",
            "manual_intervention_request": {
                "error_id": error.error_id,
                "created_at": datetime.now().isoformat(),
                "status": "pending"
            }
        }

    async def _execute_skip_operation_recovery(self, error: SystemError) -> Dict[str, Any]:
        """Execute skip operation recovery strategy."""
        self.logger.info(f"Executing skip operation recovery for error {error.error_id}")

        return {
            "success": True,
            "strategy": "skip_operation",
            "message": f"Operation {error.context.operation} skipped due to unrecoverable error",
            "skipped_operation": error.context.operation
        }

    def _calculate_retry_delay(self, config: RetryConfig, attempt: int) -> float:
        """Calculate retry delay with exponential backoff."""
        delay = config.base_delay * (config.backoff_factor ** attempt)
        delay = min(delay, config.max_delay)

        if config.jitter:
            # Add jitter to prevent thundering herd
            import random
            jitter_factor = random.uniform(0.8, 1.2)
            delay *= jitter_factor

        return delay

    async def _record_error(self, error: SystemError):
        """Record error in history and active errors."""
        self.error_history.append(error)
        self.active_errors[error.error_id] = error

        # Persist error to KEVIN directory if enabled
        if self.config.get("persist_error_history", True):
            await self._persist_error(error)

    async def _update_error_metrics(self, error: SystemError, recovery_result: Dict[str, Any]):
        """Update error metrics based on recovery result."""
        self.error_metrics["total_errors"] += 1

        if recovery_result.get("success", False):
            self.error_metrics["resolved_errors"] += 1
            error.resolved = True
            error.resolution_details = recovery_result.get("message", "Resolved")
        else:
            if error.recovery_strategy == RecoveryStrategy.ESCALATE:
                self.error_metrics["escalated_errors"] += 1

        # Update category and severity metrics
        category = error.category.value
        severity = error.severity.value

        self.error_metrics["errors_by_category"][category] = \
            self.error_metrics["errors_by_category"].get(category, 0) + 1
        self.error_metrics["errors_by_severity"][severity] = \
            self.error_metrics["errors_by_severity"].get(severity, 0) + 1

        # Calculate recovery success rate
        if self.error_metrics["total_errors"] > 0:
            self.error_metrics["recovery_success_rate"] = \
                (self.error_metrics["resolved_errors"] / self.error_metrics["total_errors"]) * 100

    async def _log_error_handling(self, error: SystemError, recovery_result: Dict[str, Any]):
        """Log error handling outcome."""
        log_level = logging.ERROR if not recovery_result.get("success", False) else logging.INFO

        self.logger.log(
            log_level,
            f"Error {error.error_id} handling completed: "
            f"Strategy={error.recovery_strategy.value if error.recovery_strategy else 'None'}, "
            f"Success={recovery_result.get('success', False)}, "
            f"Details={recovery_result.get('message', 'No details')}"
        )

    async def _log_retry_attempt(
        self,
        context: ErrorContext,
        exception: Exception,
        attempt: int,
        delay: float
    ):
        """Log retry attempt details."""
        await self.kevin_integration.create_log_file(
            context.session_id,
            f"retry_attempt_{context.operation}.log"
        )

    async def _persist_error(self, error: SystemError):
        """Persist error to KEVIN directory."""
        try:
            error_data = {
                "error_id": error.error_id,
                "category": error.category.value,
                "severity": error.severity.value,
                "message": error.message,
                "context": {
                    "session_id": error.context.session_id,
                    "operation": error.context.operation,
                    "component": error.context.component,
                    "stage": error.context.stage
                },
                "recovery_attempts": error.recovery_attempts,
                "recovery_strategy": error.recovery_strategy.value if error.recovery_strategy else None,
                "resolved": error.resolved,
                "resolution_details": error.resolution_details,
                "timestamp": error.timestamp.isoformat()
            }

            log_file_path = await self.kevin_integration.create_log_file(
                error.context.session_id,
                f"error_{error.error_id}.json"
            )

            import json
            with open(log_file_path, 'w') as f:
                json.dump(error_data, f, indent=2)

        except Exception as persist_error:
            self.logger.warning(f"Failed to persist error {error.error_id}: {str(persist_error)}")

    async def _log_error_escalation(self, error: SystemError):
        """Log error escalation to KEVIN directory."""
        try:
            escalation_data = {
                "error_id": error.error_id,
                "category": error.category.value,
                "severity": error.severity.value,
                "message": error.message,
                "escalation_timestamp": datetime.now().isoformat(),
                "requires_manual_intervention": True,
                "context": {
                    "session_id": error.context.session_id,
                    "operation": error.context.operation,
                    "component": error.context.component,
                    "stage": error.context.stage
                }
            }

            escalation_file_path = await self.kevin_integration.create_log_file(
                error.context.session_id,
                f"escalation_{error.error_id}.json"
            )

            import json
            with open(escalation_file_path, 'w') as f:
                json.dump(escalation_data, f, indent=2)

        except Exception as escalation_error:
            self.logger.error(f"Failed to log error escalation: {str(escalation_error)}")

    async def _create_manual_intervention_request(self, error: SystemError):
        """Create manual intervention request."""
        try:
            intervention_data = {
                "request_id": f"intervention_{error.error_id}",
                "error_id": error.error_id,
                "category": error.category.value,
                "severity": error.severity.value,
                "message": error.message,
                "created_at": datetime.now().isoformat(),
                "status": "pending",
                "context": {
                    "session_id": error.context.session_id,
                    "operation": error.context.operation,
                    "component": error.context.component,
                    "stage": error.context.stage
                },
                "recovery_history": {
                    "attempts": error.recovery_attempts,
                    "strategies_tried": [error.recovery_strategy.value] if error.recovery_strategy else []
                }
            }

            intervention_file_path = await self.kevin_integration.create_log_file(
                error.context.session_id,
                f"manual_intervention_{error.error_id}.json"
            )

            import json
            with open(intervention_file_path, 'w') as f:
                json.dump(intervention_data, f, indent=2)

        except Exception as intervention_error:
            self.logger.error(f"Failed to create manual intervention request: {str(intervention_error)}")

    def _calculate_error_trend(self, recent_errors: List[SystemError]) -> str:
        """Calculate error trend direction."""
        if len(recent_errors) < 2:
            return "insufficient_data"

        # Group errors by hour
        errors_by_hour = {}
        for error in recent_errors:
            hour_key = error.timestamp.replace(minute=0, second=0, microsecond=0)
            errors_by_hour[hour_key] = errors_by_hour.get(hour_key, 0) + 1

        # Get last two hours
        sorted_hours = sorted(errors_by_hour.keys())
        if len(sorted_hours) < 2:
            return "insufficient_data"

        recent_hour = sorted_hours[-1]
        previous_hour = sorted_hours[-2]

        recent_count = errors_by_hour[recent_hour]
        previous_count = errors_by_hour[previous_hour]

        if recent_count > previous_count * 1.2:
            return "increasing"
        elif recent_count < previous_count * 0.8:
            return "decreasing"
        else:
            return "stable"


# Convenience function for quick error handling
async def handle_system_error(
    exception: Exception,
    session_id: str,
    operation: str,
    component: str = "unknown",
    stage: str = "unknown"
) -> Dict[str, Any]:
    """
    Quick error handling function.

    Args:
        exception: The exception that occurred
        session_id: Session identifier
        operation: Operation that failed
        component: Component where error occurred
        stage: Workflow stage

    Returns:
        Error handling result
    """
    error_integration = ErrorHandlingIntegration()

    context = ErrorContext(
        session_id=session_id,
        operation=operation,
        component=component,
        stage=stage,
        timestamp=datetime.now()
    )

    return await error_integration.handle_error(exception, context)


# Decorator for automatic error handling
def with_error_handling(
    session_id_param: str = "session_id",
    operation_param: str = "operation",
    component: str = "unknown",
    stage: str = "unknown"
):
    """
    Decorator for automatic error handling.

    Args:
        session_id_param: Parameter name containing session ID
        operation_param: Parameter name containing operation name
        component: Component name
        stage: Stage name
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Extract context from function parameters
                session_id = kwargs.get(session_id_param, "unknown")
                operation = kwargs.get(operation_param, func.__name__)

                # Handle error
                result = await handle_system_error(e, session_id, operation, component, stage)

                # Raise original exception if error handling failed
                if not result.get("success", False):
                    raise e

                # Return error handling result
                return {
                    "error_occurred": True,
                    "error_handling_result": result,
                    "original_function": func.__name__
                }

        return wrapper
    return decorator