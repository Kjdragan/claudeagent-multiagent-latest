"""
Enhanced Two-Module Scraping Architecture - Comprehensive Tests

Phase 1.4: Complete testing suite for the enhanced two-module scraping architecture

This module provides comprehensive testing for all components implemented in Phase 1.4,
including data contracts, async orchestrator, validation, and error recovery mechanisms.

Test Coverage:
- Phase 1.4.1: Pydantic data contracts validation
- Phase 1.4.2: AsyncScrapingOrchestrator with worker pools
- Phase 1.4.3: Data contract validation and error recovery
- Integration testing with previous phases
- Performance and load testing
- Error handling and recovery validation
"""

import asyncio
import logging
import pytest
import time
from datetime import datetime
from typing import Any, Dict, List
from uuid import uuid4

from .data_contracts import (
    TaskContext, ScrapingRequest, ScrapingResult, CleaningRequest, CleaningResult,
    PipelineConfig, PipelineStatistics, TaskStatus, PipelineStage, ErrorType,
    Priority, ValidationLevel, create_scraping_request, create_cleaning_request,
    DataContractValidator, ValidationError
)

from .async_orchestrator import (
    AsyncScrapingOrchestrator, AsyncTaskQueue, WorkerPool,
    managed_orchestrator
)

from .validation_recovery import (
    ErrorRecoveryManager, ScrapingRequestValidator, ScrapingResultValidator,
    CleaningRequestValidator, CleaningResultValidator, ValidationLevel,
    RecoveryStrategy, ValidationResult, create_validation_system
)

from .integration import (
    ScrapingPipelineAPI, quick_scrape_and_clean, batch_process_urls,
    run_demo
)

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestDataContracts:
    """Test suite for Phase 1.4.1: Pydantic data contracts."""

    def test_task_context_creation(self):
        """Test TaskContext creation and validation."""
        context = TaskContext()
        assert context.session_id is not None
        assert context.task_id is not None
        assert context.priority == Priority.NORMAL
        assert context.pipeline_stage == PipelineStage.INITIALIZATION
        assert context.retry_count == 0
        assert context.max_retries == 3

    def test_scraping_request_validation(self):
        """Test ScrapingRequest validation."""
        # Valid request
        request = ScrapingRequest(
            url="https://example.com",
            search_query="test query"
        )
        assert request.url == "https://example.com"
        assert request.search_query == "test query"
        assert request.anti_bot_level is None
        assert request.max_anti_bot_level == 3

        # Invalid request - missing URL
        with pytest.raises(ValidationError):
            ScrapingRequest(url="invalid-url")

        # Invalid anti-bot levels
        with pytest.raises(ValidationError):
            ScrapingRequest(
                url="https://example.com",
                anti_bot_level=5  # Invalid level
            )

        with pytest.raises(ValidationError):
            ScrapingRequest(
                url="https://example.com",
                anti_bot_level=2,
                max_anti_bot_level=1  # Less than starting level
            )

    def test_scraping_result_validation(self):
        """Test ScrapingResult validation."""
        context = TaskContext()

        # Valid successful result
        result = ScrapingResult(
            url="https://example.com",
            domain="example.com",
            success=True,
            content="Sample content",
            duration=1.5,
            attempts_made=1,
            word_count=2,
            char_count=15,
            final_anti_bot_level=1,
            context=context
        )
        assert result.success
        assert result.content == "Sample content"
        assert result.word_count == 2
        assert result.char_count == 15

        # Invalid result - success without content
        with pytest.raises(ValidationError):
            ScrapingResult(
                url="https://example.com",
                domain="example.com",
                success=True,
                content=None,  # Missing content for successful result
                duration=1.5,
                attempts_made=1,
                context=context
            )

        # Invalid result - failure without error message
        with pytest.raises(ValidationError):
            ScrapingResult(
                url="https://example.com",
                domain="example.com",
                success=False,
                content=None,
                duration=1.5,
                attempts_made=1,
                context=context,
                error_message=None  # Missing error message for failed result
            )

        # Valid failed result
        failed_result = ScrapingResult(
            url="https://example.com",
            domain="example.com",
            success=False,
            content=None,
            duration=1.5,
            attempts_made=1,
            context=context,
            error_message="Network error",
            error_type=ErrorType.NETWORK_ERROR
        )
        assert not failed_result.success
        assert failed_result.error_message == "Network error"

    def test_cleaning_request_validation(self):
        """Test CleaningRequest validation."""
        # Valid request
        request = CleaningRequest(
            content="Sample content to clean",
            url="https://example.com",
            search_query="test query"
        )
        assert request.content == "Sample content to clean"
        assert request.url == "https://example.com"
        assert request.cleaning_intensity == "medium"

        # Invalid request - content too short
        with pytest.raises(ValidationError):
            CleaningRequest(
                content="short",  # Less than 10 characters
                url="https://example.com"
            )

        # Invalid quality threshold
        with pytest.raises(ValidationError):
            CleaningRequest(
                content="Sample content to clean",
                url="https://example.com",
                quality_threshold=1.5  # Invalid range
            )

    def test_cleaning_result_validation(self):
        """Test CleaningResult validation."""
        context = TaskContext()

        # Valid result
        result = CleaningResult(
            original_content="Original content",
            cleaned_content="Cleaned content",
            url="https://example.com",
            success=True,
            cleaning_performed=True,
            quality_improvement=0.2,
            length_reduction=0.1,
            processing_time_ms=500.0,
            context=context
        )
        assert result.success
        assert result.original_content == "Original content"
        assert result.cleaned_content == "Cleaned content"
        assert result.quality_improvement == 0.2

        # Invalid result - missing content
        with pytest.raises(ValidationError):
            CleaningResult(
                original_content=None,  # Missing
                cleaned_content="Cleaned content",
                url="https://example.com",
                success=True,
                context=context
            )

    def test_pipeline_config_validation(self):
        """Test PipelineConfig validation."""
        # Valid config
        config = PipelineConfig()
        assert config.max_scrape_workers == 40
        assert config.max_clean_workers == 20
        assert config.max_queue_size == 1000
        assert config.backpressure_threshold == 0.8

        # Custom config
        custom_config = PipelineConfig(
            max_scrape_workers=10,
            max_clean_workers=5,
            enable_quality_gates=True,
            default_quality_threshold=0.8
        )
        assert custom_config.max_scrape_workers == 10
        assert custom_config.max_clean_workers == 5
        assert custom_config.enable_quality_gates == True

        # Invalid retry delays (not increasing)
        with pytest.raises(ValidationError):
            PipelineConfig(retry_delays=[5.0, 2.0, 1.0])

    def test_pipeline_statistics(self):
        """Test PipelineStatistics functionality."""
        stats = PipelineStatistics(session_id=str(uuid4()))
        assert stats.total_tasks == 0
        assert stats.completed_tasks == 0
        assert stats.success_rate == 0.0

        # Add successful task
        stats.add_task_result(duration=1.5, success=True, quality_score=0.8)
        assert stats.total_tasks == 1
        assert stats.completed_tasks == 1
        assert stats.success_rate == 1.0
        assert stats.avg_quality_score == 0.8

        # Add failed task
        stats.add_task_result(duration=2.0, success=False)
        assert stats.total_tasks == 2
        assert stats.completed_tasks == 1
        assert stats.failed_tasks == 1
        assert stats.success_rate == 0.5

    def test_factory_functions(self):
        """Test factory functions."""
        # Test scraping request factory
        request = create_scraping_request(
            url="https://example.com",
            search_query="test query",
            anti_bot_level=1
        )
        assert isinstance(request, ScrapingRequest)
        assert str(request.url) == "https://example.com"
        assert request.search_query == "test query"
        assert request.anti_bot_level == 1

        # Test cleaning request factory
        clean_request = create_cleaning_request(
            content="Test content",
            url="https://example.com",
            cleaning_intensity="aggressive"
        )
        assert isinstance(clean_request, CleaningRequest)
        assert clean_request.content == "Test content"
        assert clean_request.cleaning_intensity == "aggressive"

        # Test pipeline config factory
        config = create_pipeline_config(
            max_scrape_workers=20,
            enable_caching=False
        )
        assert isinstance(config, PipelineConfig)
        assert config.max_scrape_workers == 20
        assert config.enable_caching == False


class TestAsyncOrchestrator:
    """Test suite for Phase 1.4.2: AsyncScrapingOrchestrator."""

    @pytest.mark.asyncio
    async def test_async_task_queue(self):
        """Test AsyncTaskQueue functionality."""
        queue = AsyncTaskQueue(max_size=10, backpressure_threshold=0.8, name="test")

        # Test adding tasks
        task1 = "test_task_1"
        task2 = "test_task_2"

        success1 = await queue.put(task1, priority=1)
        success2 = await queue.put(task2, priority=0)  # Higher priority

        assert success1
        assert success2
        assert queue.get_size() == 2

        # Test getting tasks (priority order)
        retrieved_task, task_id, wait_time = await queue.get()
        assert retrieved_task == task2  # Higher priority task first
        assert wait_time >= 0

        retrieved_task, task_id, wait_time = await queue.get()
        assert retrieved_task == task1

        # Test empty queue
        empty_result = await queue.get(timeout=0.1)
        assert empty_result is None

        await queue.shutdown()

    @pytest.mark.asyncio
    async def test_worker_pool(self):
        """Test WorkerPool functionality."""
        async def simple_task(task_data):
            await asyncio.sleep(0.1)  # Simulate work
            return f"processed_{task_data}"

        queue = AsyncTaskQueue(max_size=10, name="worker_test")
        pool = WorkerPool(worker_count=2, task_handler=simple_task, name="test_pool")

        # Add tasks
        await queue.put("task1")
        await queue.put("task2")
        await queue.put("task3")

        # Start pool
        await pool.start(queue)

        # Wait for processing
        await asyncio.sleep(0.5)

        # Check stats
        stats = pool.get_stats()
        assert stats['tasks_processed'] >= 2
        assert stats['success_rate'] >= 0.5

        await pool.shutdown()
        await queue.shutdown()

    @pytest.mark.asyncio
    async def test_async_scraping_orchestrator_lifecycle(self):
        """Test AsyncScrapingOrchestrator lifecycle."""
        config = PipelineConfig(
            max_scrape_workers=2,
            max_clean_workers=1,
            max_queue_size=10
        )

        orchestrator = AsyncScrapingOrchestrator(config)

        # Test initialization
        await orchestrator.start()
        assert orchestrator.session_id is not None
        assert orchestrator._stats.session_id == orchestrator.session_id

        # Test statistics
        stats = orchestrator.get_statistics()
        assert 'session_id' in stats
        assert 'pipeline_stats' in stats
        assert 'scrape_queue' in stats
        assert 'clean_queue' in stats

        # Test health status
        health = await orchestrator.get_health_status()
        assert 'overall_health' in health
        assert 'health_issues' in health
        assert 'active_workers' in health

        # Test shutdown
        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_task_submission(self):
        """Test task submission to orchestrator."""
        config = PipelineConfig(
            max_scrape_workers=2,
            max_clean_workers=1,
            max_queue_size=10
        )

        orchestrator = AsyncScrapingOrchestrator(config)
        await orchestrator.start()

        # Create scraping request
        request = create_scraping_request(
            url="https://example.com",
            search_query="test query"
        )

        # Submit scraping task
        success = await orchestrator.submit_scraping_task(request, priority=1)
        assert success
        assert request.context.task_id in orchestrator._active_tasks

        # Create cleaning request
        clean_request = create_cleaning_request(
            content="Test content",
            url="https://example.com"
        )

        # Submit cleaning task
        clean_success = await orchestrator.submit_cleaning_task(clean_request, priority=1)
        assert clean_success
        assert clean_request.context.task_id in orchestrator._active_tasks

        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_managed_orchestrator_context(self):
        """Test managed orchestrator context manager."""
        config = PipelineConfig(max_scrape_workers=2, max_clean_workers=1)

        async with managed_orchestrator(config) as orchestrator:
            assert orchestrator is not None
            assert orchestrator.session_id is not None

            # Test functionality within context
            request = create_scraping_request(url="https://example.com")
            success = await orchestrator.submit_scraping_task(request)
            assert success

        # Orchestrator should be shutdown after context exit
        # This is tested implicitly - no exception should be raised


class TestValidationRecovery:
    """Test suite for Phase 1.4.3: Validation and error recovery."""

    def test_scraping_request_validator(self):
        """Test ScrapingRequestValidator."""
        config = PipelineConfig()
        validator = ScrapingRequestValidator(config)

        # Valid request
        request = create_scraping_request(url="https://example.com")
        result = validator.validate(request, ValidationLevel.BASIC)
        assert result.is_valid
        assert len(result.errors) == 0

        # Invalid request
        invalid_request = ScrapingRequest(
            url="invalid-url",
            anti_bot_level=5  # Invalid
        )
        result = validator.validate(invalid_request, ValidationLevel.STANDARD)
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_scraping_result_validator(self):
        """Test ScrapingResultValidator."""
        config = PipelineConfig()
        validator = ScrapingResultValidator(config)

        # Valid result
        result = ScrapingResult(
            url="https://example.com",
            domain="example.com",
            success=True,
            content="Test content",
            duration=1.0,
            attempts_made=1,
            context=TaskContext()
        )
        validation_result = validator.validate(result, ValidationLevel.BASIC)
        assert validation_result.is_valid

        # Invalid result
        invalid_result = ScrapingResult(
            url="https://example.com",
            domain="example.com",
            success=True,
            content=None,  # Missing content
            duration=1.0,
            attempts_made=1,
            context=TaskContext()
        )
        validation_result = validator.validate(invalid_result, ValidationLevel.STANDARD)
        assert not validation_result.is_valid
        assert len(validation_result.errors) > 0

    def test_cleaning_request_validator(self):
        """Test CleaningRequestValidator."""
        config = PipelineConfig()
        validator = CleaningRequestValidator(config)

        # Valid request
        request = create_cleaning_request(
            content="Sufficient content for testing",
            url="https://example.com"
        )
        result = validator.validate(request, ValidationLevel.BASIC)
        assert result.is_valid

        # Invalid request - content too short
        invalid_request = CleaningRequest(
            content="short",  # Too short
            url="https://example.com"
        )
        result = validator.validate(invalid_request, ValidationLevel.STANDARD)
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_cleaning_result_validator(self):
        """Test CleaningResultValidator."""
        config = PipelineConfig()
        validator = CleaningResultValidator(config)

        # Valid result
        result = CleaningResult(
            original_content="Original",
            cleaned_content="Cleaned",
            url="https://example.com",
            success=True,
            context=TaskContext()
        )
        validation_result = validator.validate(result, ValidationLevel.BASIC)
        assert validation_result.is_valid

        # Invalid result - missing original content
        invalid_result = CleaningResult(
            original_content=None,  # Missing
            cleaned_content="Cleaned",
            url="https://example.com",
            success=True,
            context=TaskContext()
        )
        validation_result = validator.validate(invalid_result, ValidationLevel.STANDARD)
        assert not validation_result.is_valid

    @pytest.mark.asyncio
    async def test_error_recovery_manager(self):
        """Test ErrorRecoveryManager functionality."""
        config = PipelineConfig()
        recovery_manager = ErrorRecoveryManager(config)

        # Test validation
        request = create_scraping_request(url="https://example.com")
        validation_result = recovery_manager.validate('scraping_request', request)
        assert isinstance(validation_result, ValidationResult)

        # Test recovery strategy determination
        context = TaskContext()
        network_error = Exception("Network connection failed")
        strategy = recovery_manager.determine_recovery_strategy(
            network_error, context, 'scraping_request', 0
        )
        assert strategy is not None
        assert strategy.strategy == RecoveryStrategy.RETRY

        # Test recovery execution
        async def dummy_func():
            return "success"

        success, result = await recovery_manager.execute_recovery(
            strategy, dummy_func
        )
        assert success
        assert result == "success"

        # Test statistics
        stats = recovery_manager.get_recovery_statistics()
        assert 'total_recoveries' in stats
        assert 'success_rate' in stats

    def test_validation_system_factory(self):
        """Test validation system factory function."""
        config = PipelineConfig()
        validators, recovery_manager = create_validation_system(config)

        # Check validators
        assert 'scraping_request' in validators
        assert 'scraping_result' in validators
        assert 'cleaning_request' in validators
        assert 'cleaning_result' in validators

        # Check recovery manager
        assert recovery_manager is not None
        assert isinstance(recovery_manager, ErrorRecoveryManager)


class TestIntegration:
    """Test suite for integration layer."""

    @pytest.mark.asyncio
    async def test_scraping_pipeline_api_lifecycle(self):
        """Test ScrapingPipelineAPI lifecycle."""
        api = ScrapingPipelineAPI()

        # Test initialization
        await api.initialize()
        assert api.is_initialized
        assert api.orchestrator is not None
        assert api.recovery_manager is not None

        # Test shutdown
        await api.shutdown()
        assert not api.is_initialized

    @pytest.mark.asyncio
    async def test_single_url_processing(self):
        """Test single URL processing through API."""
        api = ScrapingPipelineAPI(
            PipelineConfig(
                max_scrape_workers=2,
                max_clean_workers=1,
                max_queue_size=10
            )
        )

        await api.initialize()

        try:
            success, result = await api.scrape_and_clean(
                url="https://example.com",
                search_query="test query",
                validation_level=ValidationLevel.BASIC
            )

            assert isinstance(success, bool)
            if success:
                assert isinstance(result, CleaningResult)
            else:
                assert isinstance(result, list)  # Error list

        finally:
            await api.shutdown()

    @pytest.mark.asyncio
    async def test_batch_url_processing(self):
        """Test batch URL processing through API."""
        api = ScrapingPipelineAPI(
            PipelineConfig(
                max_scrape_workers=2,
                max_clean_workers=1,
                max_queue_size=20
            )
        )

        await api.initialize()

        try:
            urls = [
                "https://example.com/page1",
                "https://example.com/page2",
                "https://example.com/page3"
            ]

            result = await api.process_url_batch(
                urls=urls,
                search_query="batch test",
                validation_level=ValidationLevel.BASIC
            )

            assert isinstance(result, dict)
            assert 'total_urls' in result
            assert 'successful_urls' in result
            assert 'failed_urls' in result
            assert 'success_rate' in result
            assert 'results' in result
            assert 'errors' in result

            assert result['total_urls'] == len(urls)
            assert 0 <= result['success_rate'] <= 1.0

        finally:
            await api.shutdown()

    @pytest.mark.asyncio
    async def test_pipeline_status(self):
        """Test pipeline status reporting."""
        api = ScrapingPipelineAPI(
            PipelineConfig(max_scrape_workers=2, max_clean_workers=1)
        )

        await api.initialize()

        try:
            status = await api.get_pipeline_status()
            assert isinstance(status, dict)
            assert 'status' in status
            assert 'orchestrator_stats' in status
            assert 'recovery_stats' in status
            assert 'health_status' in status
            assert 'config' in status

        finally:
            await api.shutdown()

    @pytest.mark.asyncio
    async def test_convenience_functions(self):
        """Test convenience functions."""
        # Test quick_scrape_and_clean
        success, result = await quick_scrape_and_clean(
            url="https://example.com",
            search_query="quick test",
            max_scrape_workers=1,
            max_clean_workers=1
        )
        assert isinstance(success, bool)

        # Test batch_process_urls
        urls = ["https://example.com/page1", "https://example.com/page2"]
        batch_result = await batch_process_urls(
            urls=urls,
            search_query="batch test",
            max_scrape_workers=1,
            max_clean_workers=1
        )
        assert isinstance(batch_result, dict)
        assert 'total_urls' in batch_result


class TestPerformance:
    """Performance and load testing."""

    @pytest.mark.asyncio
    async def test_concurrent_task_processing(self):
        """Test concurrent task processing performance."""
        config = PipelineConfig(
            max_scrape_workers=5,
            max_clean_workers=3,
            max_queue_size=50
        )

        orchestrator = AsyncScrapingOrchestrator(config)
        await orchestrator.start()

        try:
            # Submit multiple tasks
            tasks = []
            for i in range(10):
                request = create_scraping_request(
                    url=f"https://example.com/page{i}",
                    search_query=f"test query {i}"
                )
                success = await orchestrator.submit_scraping_task(request)
                if success:
                    tasks.append(request)

            # Wait for processing
            await asyncio.sleep(2.0)

            # Check statistics
            stats = orchestrator.get_statistics()
            assert stats['pipeline_stats']['total_tasks'] >= len(tasks)

        finally:
            await orchestrator.shutdown()

    def test_memory_usage_validation(self):
        """Test memory usage with large data volumes."""
        # Create large amounts of test data
        large_content = "Test content " * 10000  # ~140KB

        request = CleaningRequest(
            content=large_content,
            url="https://example.com/large",
            context=TaskContext()
        )

        # Should handle large content without issues
        assert len(request.content) > 100000

        # Validate that the model can handle it
        try:
            validated = CleaningRequest.parse_obj(request.dict())
            assert len(validated.content) == len(large_content)
        except Exception as e:
            pytest.fail(f"Large content validation failed: {e}")

    @pytest.mark.asyncio
    async def test_backpressure_handling(self):
        """Test backpressure handling in queues."""
        config = PipelineConfig(
            max_queue_size=5,
            backpressure_threshold=0.6  # 60% threshold
        )

        orchestrator = AsyncScrapingOrchestrator(config)
        await orchestrator.start()

        try:
            # Fill queue beyond backpressure threshold
            submitted_tasks = []
            for i in range(10):
                request = create_scraping_request(
                    url=f"https://example.com/page{i}"
                )
                success = await orchestrator.submit_scraping_task(request)
                if success:
                    submitted_tasks.append(request)

            # Some tasks should be rejected due to backpressure
            assert len(submitted_tasks) < 10

        finally:
            await orchestrator.shutdown()


# Test runner function
async def run_all_tests():
    """Run all tests and report results."""
    print("=== Running Phase 1.4 Comprehensive Tests ===\n")

    test_classes = [
        TestDataContracts,
        TestAsyncOrchestrator,
        TestValidationRecovery,
        TestIntegration,
        TestPerformance
    ]

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    for test_class in test_classes:
        print(f"--- Running {test_class.__name__} ---")
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]

        for test_method_name in test_methods:
            total_tests += 1
            test_instance = test_class()
            test_method = getattr(test_instance, test_method_name)

            try:
                if asyncio.iscoroutinefunction(test_method):
                    await test_method()
                else:
                    test_method()
                passed_tests += 1
                print(f"  ✓ {test_method_name}")
            except Exception as e:
                failed_tests.append(f"{test_class.__name__}.{test_method_name}: {str(e)}")
                print(f"  ✗ {test_method_name}: {e}")

        print()

    # Summary
    print("=== Test Summary ===")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")

    if failed_tests:
        print("\nFailed tests:")
        for failure in failed_tests:
            print(f"  - {failure}")

    print(f"\nPhase 1.4 testing completed: {passed_tests}/{total_tests} tests passed")
    return len(failed_tests) == 0


if __name__ == "__main__":
    # Run tests when module is executed directly
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)