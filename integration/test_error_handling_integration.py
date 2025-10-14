"""
Test Suite for Error Handling Integration

This test suite validates the comprehensive error handling and recovery mechanisms
for the agent-based research system.
"""

import asyncio
import pytest
import tempfile
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

# Import the component to test
from integration.error_handling_integration import (
    ErrorHandlingIntegration,
    ErrorSeverity,
    ErrorCategory,
    RecoveryStrategy,
    ErrorContext,
    SystemError,
    RetryConfig,
    CircuitBreakerConfig,
    CircuitBreaker,
    CircuitBreakerState,
    handle_system_error,
    with_error_handling
)


class TestErrorHandlingIntegration:
    """Comprehensive test suite for ErrorHandlingIntegration."""

    @pytest.fixture
    def error_integration(self):
        """Create an ErrorHandlingIntegration instance for testing."""
        return ErrorHandlingIntegration()

    @pytest.fixture
    def sample_error_context(self):
        """Sample error context for testing."""
        return ErrorContext(
            session_id="test_session_001",
            operation="research_execution",
            component="mcp_tool_integration",
            stage="research",
            timestamp=datetime.now(),
            additional_context={"test": True}
        )

    @pytest.fixture
    def sample_exceptions(self):
        """Sample exceptions for testing."""
        return {
            "network_error": ConnectionError("Network connection failed"),
            "timeout_error": TimeoutError("Operation timed out"),
            "validation_error": ValueError("Invalid input parameter"),
            "system_error": OSError("System resource unavailable"),
            "api_error": Exception("API rate limit exceeded"),
            "unknown_error": Exception("Unknown error occurred")
        }

    def test_initialization(self, error_integration):
        """Test ErrorHandlingIntegration initialization."""
        assert error_integration.config is not None
        assert error_integration.kevin_integration is not None
        assert error_integration.mcp_integration is not None
        assert error_integration.quality_integration is not None
        assert isinstance(error_integration.error_history, list)
        assert isinstance(error_integration.active_errors, dict)
        assert isinstance(error_integration.circuit_breakers, dict)
        assert isinstance(error_integration.recovery_handlers, dict)
        assert isinstance(error_integration.error_metrics, dict)

    def test_default_configurations(self, error_integration):
        """Test default configuration values."""
        retry_config = error_integration._get_default_retry_config()
        assert isinstance(retry_config, RetryConfig)
        assert retry_config.max_attempts == 3
        assert retry_config.base_delay == 1.0
        assert retry_config.backoff_factor == 2.0

        circuit_config = error_integration._get_default_circuit_breaker_config()
        assert isinstance(circuit_config, CircuitBreakerConfig)
        assert circuit_config.failure_threshold == 5
        assert circuit_config.timeout == 60.0

    def test_error_context_creation(self, error_integration):
        """Test default error context creation."""
        context = error_integration._create_default_context()
        assert isinstance(context, ErrorContext)
        assert context.session_id == "unknown"
        assert context.operation == "unknown_operation"
        assert context.component == "unknown_component"
        assert context.stage == "unknown_stage"

    def test_error_classification(self, error_integration, sample_error_context, sample_exceptions):
        """Test error classification logic."""
        test_cases = [
            (sample_exceptions["network_error"], ErrorCategory.NETWORK),
            (sample_exceptions["timeout_error"], ErrorCategory.NETWORK),
            (sample_exceptions["validation_error"], ErrorCategory.VALIDATION),
            (sample_exceptions["system_error"], ErrorCategory.SYSTEM),
            (sample_exceptions["api_error"], ErrorCategory.API),
            (sample_exceptions["unknown_error"], ErrorCategory.UNKNOWN)
        ]

        for exception, expected_category in test_cases:
            error = asyncio.run(error_integration._classify_error(exception, sample_error_context))
            assert isinstance(error, SystemError)
            assert error.category == expected_category
            assert error.context == sample_error_context
            assert error.original_exception == exception

    def test_error_severity_determination(self, error_integration):
        """Test error severity determination."""
        # Test different severity scenarios
        critical_context = ErrorContext(
            session_id="test", operation="test", component="test",
            stage="critical_production", timestamp=datetime.now()
        )
        critical_error = Exception("Critical system failure occurred")

        error = asyncio.run(error_integration._classify_error(critical_error, critical_context))
        assert error.severity == ErrorSeverity.HIGH

    def test_recovery_strategy_determination(self, error_integration, sample_error_context):
        """Test recovery strategy determination."""
        # Test different error categories and recovery strategies
        test_cases = [
            (ErrorCategory.NETWORK, RecoveryStrategy.RETRY),
            (ErrorCategory.RESOURCE, RecoveryStrategy.GRACEFUL_DEGRADATION),
            (ErrorCategory.QUALITY, RecoveryStrategy.FALLBACK),
        ]

        for category, expected_strategy in test_cases:
            error = SystemError(
                error_id="test_error",
                category=category,
                severity=ErrorSeverity.MEDIUM,
                message="Test error",
                original_exception=Exception("Test"),
                context=sample_error_context
            )

            strategy = asyncio.run(error_integration._determine_recovery_strategy(error))
            assert strategy == expected_strategy

    def test_retry_delay_calculation(self, error_integration):
        """Test retry delay calculation with exponential backoff."""
        config = RetryConfig(
            base_delay=1.0,
            backoff_factor=2.0,
            max_delay=10.0,
            jitter=False
        )

        # Test exponential backoff
        delay_1 = error_integration._calculate_retry_delay(config, 0)
        delay_2 = error_integration._calculate_retry_delay(config, 1)
        delay_3 = error_integration._calculate_retry_delay(config, 2)

        assert delay_1 == 1.0
        assert delay_2 == 2.0
        assert delay_3 == 4.0

        # Test max delay limit
        config.max_delay = 3.0
        delay_limited = error_integration._calculate_retry_delay(config, 5)
        assert delay_limited <= 3.0

    @pytest.mark.asyncio
    async def test_handle_error_success(self, error_integration, sample_error_context):
        """Test successful error handling."""
        exception = ConnectionError("Test network error")

        result = await error_integration.handle_error(exception, sample_error_context)

        assert result["success"] is True
        assert "error_id" in result
        assert "recovery_strategy" in result
        assert "recovery_result" in result
        assert "error_details" in result
        assert "timestamp" in result

        # Verify error was recorded
        assert len(error_integration.error_history) == 1
        assert error.error_id in error_integration.active_errors

    @pytest.mark.asyncio
    async def test_handle_error_with_recovery_attempts(self, error_integration, sample_error_context):
        """Test error handling with existing recovery attempts."""
        exception = ConnectionError("Test network error")

        # Handle error with existing attempts
        result = await error_integration.handle_error(
            exception, sample_error_context, recovery_attempts=2
        )

        assert result["success"] is True
        # Should escalate due to high recovery attempts
        assert result["recovery_strategy"] in [
            RecoveryStrategy.MANUAL_INTERVENTION.value,
            RecoveryStrategy.ESCALATE.value
        ]

    @pytest.mark.asyncio
    async def test_execute_with_retry_success(self, error_integration, sample_error_context):
        """Test successful execution with retry logic."""
        call_count = 0

        async def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Temporary failure")
            return "success"

        result = await error_integration.execute_with_retry(
            failing_function, context=sample_error_context
        )

        assert result == "success"
        assert call_count == 2  # Function called twice

    @pytest.mark.asyncio
    async def test_execute_with_retry_exhaustion(self, error_integration, sample_error_context):
        """Test retry exhaustion when all attempts fail."""
        async def always_failing_function():
            raise ConnectionError("Persistent failure")

        with pytest.raises(ConnectionError):
            await error_integration.execute_with_retry(
                always_failing_function, context=sample_error_context
            )

    @pytest.mark.asyncio
    async def test_execute_with_retry_custom_config(self, error_integration, sample_error_context):
        """Test retry execution with custom configuration."""
        call_count = 0

        async def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        custom_config = RetryConfig(max_attempts=3, base_delay=0.1)
        result = await error_integration.execute_with_retry(
            failing_function, retry_config=custom_config, context=sample_error_context
        )

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_execute_with_fallback_success(self, error_integration, sample_error_context):
        """Test successful execution with fallback."""
        async def primary_function():
            raise ConnectionError("Primary failed")

        async def fallback_function():
            return "fallback_success"

        result = await error_integration.execute_with_fallback(
            primary_function, fallback_function, context=sample_error_context
        )

        assert result["success"] is True
        assert result["result"] == "fallback_success"
        assert result["used_fallback"] is True
        assert "fallback_reason" in result

    @pytest.mark.asyncio
    async def test_execute_with_fallback_primary_success(self, error_integration, sample_error_context):
        """Test fallback execution when primary succeeds."""
        async def primary_function():
            return "primary_success"

        async def fallback_function():
            return "fallback_success"

        result = await error_integration.execute_with_fallback(
            primary_function, fallback_function, context=sample_error_context
        )

        assert result["success"] is True
        assert result["result"] == "primary_success"
        assert result["used_fallback"] is False

    @pytest.mark.asyncio
    async def test_execute_with_fallback_both_fail(self, error_integration, sample_error_context):
        """Test fallback execution when both functions fail."""
        async def primary_function():
            raise ConnectionError("Primary failed")

        async def fallback_function():
            raise TimeoutError("Fallback failed")

        result = await error_integration.execute_with_fallback(
            primary_function, fallback_function, context=sample_error_context
        )

        assert result["success"] is False
        assert "primary_error" in result
        assert "fallback_error" in result

    @pytest.mark.asyncio
    async def test_handle_graceful_degradation(self, error_integration, sample_error_context):
        """Test graceful degradation execution."""
        async def full_functionality():
            raise Exception("Full functionality unavailable")

        async def reduced_functionality():
            return "reduced_result"

        async def minimal_functionality():
            return "minimal_result"

        result = await error_integration.handle_graceful_degradation(
            operation="test_operation",
            full_functionality=full_functionality,
            reduced_functionality=reduced_functionality,
            minimal_functionality=minimal_functionality,
            context=sample_error_context
        )

        assert result["success"] is True
        assert result["result"] == "reduced_result"
        assert result["degradation_level"] == "reduced"

    @pytest.mark.asyncio
    async def test_execute_with_circuit_breaker(self, error_integration, sample_error_context):
        """Test circuit breaker execution."""
        call_count = 0

        async def sometimes_failing_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 3:  # Fail first 3 times
                raise ConnectionError("Service unavailable")
            return "success"

        # Configure circuit breaker to open after 2 failures
        cb_config = CircuitBreakerConfig(failure_threshold=2, timeout=1.0)

        # First call should work (circuit not yet open)
        result1 = await error_integration.execute_with_circuit_breaker(
            sometimes_failing_function,
            circuit_breaker_config=cb_config,
            context=sample_error_context
        )
        assert result1 == "success"

        # Reset call count for next test
        call_count = 0

        # Circuit should open after failures
        with pytest.raises(Exception):
            await error_integration.execute_with_circuit_breaker(
                sometimes_failing_function,
                circuit_breaker_config=cb_config,
                context=sample_error_context
            )

    def test_circuit_breaker_states(self):
        """Test circuit breaker state transitions."""
        logger = MagicMock()
        config = CircuitBreakerConfig(failure_threshold=3, timeout=1.0)
        circuit_breaker = CircuitBreaker(config, logger)

        # Initial state should be closed
        assert circuit_breaker.state == CircuitBreakerState.CLOSED

        # Simulate failures
        async def failing_function():
            raise Exception("Service failure")

        # Trigger failures to open circuit
        for _ in range(3):
            try:
                asyncio.run(circuit_breaker.call(failing_function))
            except:
                pass

        # Circuit should now be open
        assert circuit_breaker.state == CircuitBreakerState.OPEN

    @pytest.mark.asyncio
    async def test_get_error_statistics(self, error_integration):
        """Test error statistics calculation."""
        # Add some sample errors
        for i in range(5):
            error = SystemError(
                error_id=f"test_error_{i}",
                category=ErrorCategory.NETWORK if i % 2 == 0 else ErrorCategory.API,
                severity=ErrorSeverity.MEDIUM,
                message=f"Test error {i}",
                original_exception=Exception(f"Test {i}"),
                context=ErrorContext(
                    session_id="test", operation="test", component="test",
                    stage="test", timestamp=datetime.now()
                )
            )
            error_integration.error_history.append(error)

        # Mark some as resolved
        error_integration.error_history[0].resolved = True
        error_integration.error_history[2].resolved = True

        stats = await error_integration.get_error_statistics()

        assert stats["total_errors"] == 5
        assert stats["resolved_errors"] == 2
        assert stats["resolution_rate"] == 40.0  # 2/5 * 100
        assert "errors_by_category" in stats
        assert "errors_by_severity" in stats
        assert "error_trends" in stats

    @pytest.mark.asyncio
    async def test_get_active_errors(self, error_integration):
        """Test getting active errors."""
        # Add sample errors (some resolved, some active)
        for i in range(3):
            error = SystemError(
                error_id=f"active_error_{i}",
                category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.MEDIUM,
                message=f"Active error {i}",
                original_exception=Exception(f"Active {i}"),
                context=ErrorContext(
                    session_id="test", operation="test", component="test",
                    stage="test", timestamp=datetime.now()
                )
            )
            error_integration.error_history.append(error)

        # Mark one as resolved
        error_integration.error_history[1].resolved = True

        active_errors = await error_integration.get_active_errors()

        assert len(active_errors) == 2  # Only unresolved errors
        for error_info in active_errors:
            assert "error_id" in error_info
            assert "category" in error_info
            assert "severity" in error_info
            assert error_info["resolved"] is False

    @pytest.mark.asyncio
    async def test_clear_resolved_errors(self, error_integration):
        """Test clearing resolved errors."""
        # Add sample errors with different timestamps
        now = datetime.now()
        old_time = now - timedelta(hours=25)  # Older than 24 hours
        recent_time = now - timedelta(hours=12)  # Within 24 hours

        old_error = SystemError(
            error_id="old_error",
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            message="Old error",
            original_exception=Exception("Old"),
            context=ErrorContext(
                session_id="test", operation="test", component="test",
                stage="test", timestamp=old_time
            )
        )
        old_error.resolved = True

        recent_error = SystemError(
            error_id="recent_error",
            category=ErrorCategory.API,
            severity=ErrorSeverity.LOW,
            message="Recent error",
            original_exception=Exception("Recent"),
            context=ErrorContext(
                session_id="test", operation="test", component="test",
                stage="test", timestamp=recent_time
            )
        )
        recent_error.resolved = True

        error_integration.error_history.extend([old_error, recent_error])

        result = await error_integration.clear_resolved_errors(older_than_hours=24)

        assert result["success"] is True
        assert result["cleared_count"] == 1  # Only the old error
        assert len(error_integration.error_history) == 1  # Recent error remains

    @pytest.mark.asyncio
    async def test_register_recovery_handler(self, error_integration):
        """Test registering custom recovery handlers."""
        async def custom_recovery_handler(error):
            return {"success": True, "custom": True}

        result = await error_integration.register_recovery_handler(
            ErrorCategory.NETWORK, custom_recovery_handler
        )

        assert result["success"] is True
        assert result["category"] == ErrorCategory.NETWORK.value
        assert result["handler_count"] == 1

        # Verify handler was registered
        assert ErrorCategory.NETWORK in error_integration.recovery_handlers
        assert len(error_integration.recovery_handlers[ErrorCategory.NETWORK]) == 1

    def test_error_trend_calculation(self, error_integration):
        """Test error trend calculation."""
        now = datetime.now()

        # Create errors with increasing frequency (increasing trend)
        errors = []
        for i in range(6):
            error = SystemError(
                error_id=f"trend_error_{i}",
                category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.MEDIUM,
                message=f"Trend error {i}",
                original_exception=Exception(f"Trend {i}"),
                context=ErrorContext(
                    session_id="test", operation="test", component="test",
                    stage="test", timestamp=now - timedelta(minutes=i*10)
                )
            )
            errors.append(error)

        trend = error_integration._calculate_error_trend(errors)
        assert trend in ["increasing", "decreasing", "stable", "insufficient_data"]

    def test_convenience_function(self, sample_exceptions):
        """Test the convenience function for error handling."""
        exception = sample_exceptions["network_error"]

        result = asyncio.run(handle_system_error(
            exception=exception,
            session_id="test_session",
            operation="test_operation",
            component="test_component",
            stage="test_stage"
        ))

        assert result["success"] is True
        assert "error_id" in result
        assert "recovery_strategy" in result

    def test_error_handling_decorator(self, sample_exceptions):
        """Test the error handling decorator."""
        @with_error_handling(
            session_id_param="session_id",
            operation_param="operation",
            component="test_component",
            stage="test_stage"
        )
        async def test_function(session_id, operation):
            if operation == "fail":
                raise ConnectionError("Test failure")
            return "success"

        # Test successful execution
        result = asyncio.run(test_function(session_id="test_001", operation="success"))
        assert result == "success"

        # Test failed execution with error handling
        result = asyncio.run(test_function(session_id="test_002", operation="fail"))
        assert result["error_occurred"] is True
        assert "error_handling_result" in result


class TestErrorHandlingIntegrationEdgeCases:
    """Test edge cases and boundary conditions for ErrorHandlingIntegration."""

    @pytest.fixture
    def error_integration(self):
        """Create an ErrorHandlingIntegration instance for testing."""
        return ErrorHandlingIntegration()

    @pytest.mark.asyncio
    async def test_empty_error_history(self, error_integration):
        """Test error statistics with empty history."""
        stats = await error_integration.get_error_statistics()
        assert stats["total_errors"] == 0
        assert stats["resolved_errors"] == 0
        assert stats["resolution_rate"] == 0

    @pytest.mark.asyncio
    async def test_no_active_errors(self, error_integration):
        """Test getting active errors with no errors."""
        active_errors = await error_integration.get_active_errors()
        assert len(active_errors) == 0

    @pytest.mark.asyncio
    async def test_circular_error_handling(self, error_integration):
        """Test error handling when error handling itself fails."""
        # Create an error context
        context = ErrorContext(
            session_id="test",
            operation="test",
            component="test",
            stage="test",
            timestamp=datetime.now()
        )

        # Patch the _classify_error method to raise an exception
        with patch.object(error_integration, '_classify_error', side_effect=Exception("Classification failed")):
            exception = ConnectionError("Original error")
            result = await error_integration.handle_error(exception, context)

            assert result["success"] is False
            assert "error" in result
            assert "handling_error" in result

    @pytest.mark.asyncio
    async def test_max_recovery_attempts_exceeded(self, error_integration):
        """Test behavior when maximum recovery attempts are exceeded."""
        context = ErrorContext(
            session_id="test",
            operation="test",
            component="test",
            stage="test",
            timestamp=datetime.now()
        )

        exception = Exception("Persistent error")

        # Handle with maximum recovery attempts already reached
        result = await error_integration.handle_error(exception, context, recovery_attempts=10)

        assert result["success"] is True
        assert result["recovery_strategy"] == RecoveryStrategy.MANUAL_INTERVENTION.value

    @pytest.mark.asyncio
    async def test_unknown_error_category_handling(self, error_integration):
        """Test handling of completely unknown error types."""
        context = ErrorContext(
            session_id="test",
            operation="test",
            component="test",
            stage="test",
            timestamp=datetime.now()
        )

        # Create a very unusual exception
        class CustomException(Exception):
            pass

        exception = CustomException("Very unusual error")
        result = await error_integration.handle_error(exception, context)

        assert result["success"] is True
        assert result["error_details"]["category"] == ErrorCategory.UNKNOWN.value

    @pytest.mark.asyncio
    async def test_concurrent_error_handling(self, error_integration):
        """Test handling multiple errors concurrently."""
        context = ErrorContext(
            session_id="test_concurrent",
            operation="test",
            component="test",
            stage="test",
            timestamp=datetime.now()
        )

        # Create multiple error handling tasks
        tasks = []
        for i in range(5):
            exception = Exception(f"Concurrent error {i}")
            task = error_integration.handle_error(exception, context)
            tasks.append(task)

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed
        assert all(isinstance(result, dict) and result.get("success", False) for result in results)
        assert len(results) == 5
        assert len(error_integration.error_history) == 5

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_state(self, error_integration):
        """Test circuit breaker half-open state behavior."""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.1)
        circuit_breaker = CircuitBreaker(config, MagicMock())

        # Force circuit to open
        circuit_breaker.failure_count = config.failure_threshold
        circuit_breaker.state = CircuitBreakerState.OPEN
        circuit_breaker.next_attempt = datetime.now() - timedelta(seconds=1)  # Past recovery timeout

        async def test_function():
            return "success"

        # Should transition to half-open and succeed
        result = await circuit_breaker.call(test_function)
        assert result == "success"
        assert circuit_breaker.state == CircuitBreakerState.CLOSED  # Should reset on success

    def test_retry_delay_with_jitter(self, error_integration):
        """Test retry delay calculation with jitter enabled."""
        config = RetryConfig(
            base_delay=1.0,
            backoff_factor=2.0,
            max_delay=10.0,
            jitter=True
        )

        # Calculate multiple delays
        delays = []
        for attempt in range(5):
            delay = error_integration._calculate_retry_delay(config, attempt)
            delays.append(delay)

        # With jitter, delays should vary but be within reasonable bounds
        assert all(0 < delay <= 10.0 for delay in delays)
        # Base exponential growth should be preserved (with some variation)
        assert delays[1] > delays[0] * 1.5  # Allow for jitter variation

    @pytest.mark.asyncio
    async def test_graceful_degradation_all_levels_fail(self, error_integration):
        """Test graceful degradation when all levels fail."""
        context = ErrorContext(
            session_id="test",
            operation="test_operation",
            component="test",
            stage="test",
            timestamp=datetime.now()
        )

        async def failing_function():
            raise Exception("Function failed")

        result = await error_integration.handle_graceful_degradation(
            operation="test_operation",
            full_functionality=failing_function,
            reduced_functionality=failing_function,
            minimal_functionality=failing_function,
            context=context
        )

        assert result["success"] is False
        assert result["error"] == "All degradation levels failed"

    @pytest.mark.asyncio
    async def test_persistence_error_handling(self, error_integration):
        """Test error handling when persistence fails."""
        context = ErrorContext(
            session_id="test",
            operation="test",
            component="test",
            stage="test",
            timestamp=datetime.now()
        )

        exception = Exception("Test error")

        # Mock persistence to fail
        with patch.object(error_integration, '_persist_error', side_effect=Exception("Persistence failed")):
            result = await error_integration.handle_error(exception, context)

            # Error handling should still succeed despite persistence failure
            assert result["success"] is True


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])