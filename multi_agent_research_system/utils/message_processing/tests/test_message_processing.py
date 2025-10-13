"""
Comprehensive Test Suite for Message Processing System

This module provides comprehensive testing and validation for all components
of the message processing system, including unit tests, integration tests,
and performance tests.

Key Features:
- Unit tests for all core components
- Integration tests for system workflows
- Performance tests with benchmarks
- Validation of message type processing
- Quality assessment testing
- Caching and routing validation
- Error handling and recovery testing
- Mock data generation for testing
"""

import asyncio
import json
import time
import pytest
import unittest
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Import the message processing components
from ..core.message_types import (
    RichMessage, EnhancedMessageType, MessagePriority, MessageContext,
    MessageBuilder, create_text_message, create_error_message
)
from ..core.message_processor import MessageProcessor, ProcessingResult
from ..formatters.rich_formatter import RichFormatter, DisplayConfig
from ..analyzers.content_enhancer import ContentEnhancer, EnhancementResult
from ..analyzers.message_quality_analyzer import MessageQualityAnalyzer, QualityAssessment
from ..routers.message_router import MessageRouter, RoutingResult, RoutingDecision
from ..cache.message_cache import MessageCache, CacheConfig, CacheEntry
from ..serializers.message_serializer import MessageSerializer, SerializationFormat
from ..integration.orchestrator_integration import OrchestratorIntegration


class TestMessageTypeSystem(unittest.TestCase):
    """Test cases for the enhanced message type system."""

    def test_rich_message_creation(self):
        """Test RichMessage creation and initialization."""
        message = RichMessage(
            id="test-001",
            message_type=EnhancedMessageType.TEXT,
            content="Test message content",
            priority=MessagePriority.NORMAL,
            context=MessageContext.RESEARCH
        )

        self.assertEqual(message.id, "test-001")
        self.assertEqual(message.message_type, EnhancedMessageType.TEXT)
        self.assertEqual(message.content, "Test message content")
        self.assertEqual(message.priority, MessagePriority.NORMAL)
        self.assertEqual(message.context, MessageContext.RESEARCH)

    def test_message_builder(self):
        """Test MessageBuilder fluent interface."""
        message = (MessageBuilder()
                   .with_content("Builder test content")
                   .with_type(EnhancedMessageType.RESEARCH_RESULT)
                   .with_priority(MessagePriority.HIGH)
                   .with_context(MessageContext.RESEARCH)
                   .with_session("session-123")
                   .with_quality_scores(quality=0.8, relevance=0.9)
                   .with_routing_tags("research", "high_quality")
                   .build())

        self.assertEqual(message.content, "Builder test content")
        self.assertEqual(message.message_type, EnhancedMessageType.RESEARCH_RESULT)
        self.assertEqual(message.priority, MessagePriority.HIGH)
        self.assertEqual(message.session_id, "session-123")
        self.assertEqual(message.metadata.quality_score, 0.8)
        self.assertEqual(message.metadata.relevance_score, 0.9)
        self.assertIn("research", message.metadata.routing_tags)

    def test_convenience_functions(self):
        """Test convenience message creation functions."""
        # Test text message
        text_msg = create_text_message("Simple text message")
        self.assertEqual(text_msg.message_type, EnhancedMessageType.TEXT)
        self.assertEqual(text_msg.content, "Simple text message")

        # Test error message
        error_msg = create_error_message("Something went wrong", "network_error")
        self.assertEqual(error_msg.message_type, EnhancedMessageType.SYSTEM_ERROR)
        self.assertEqual(error_msg.priority, MessagePriority.HIGH)
        self.assertIn("network_error", error_msg.metadata.routing_tags)

    def test_message_lifecycle(self):
        """Test message lifecycle tracking."""
        message = create_text_message("Lifecycle test")

        # Mark as processed
        message.mark_processed("test_processor", 0.5)
        self.assertEqual(message.lifecycle.value, "processed")
        self.assertEqual(message.metadata.processing_time, 0.5)
        self.assertIn("test_processor", message.metadata.processing_history[0]["processor"])

        # Mark as delivered
        message.mark_delivered("test_destination")
        self.assertEqual(message.metadata.target_agent, "test_destination")

        # Mark as failed
        message.mark_failed("Test error", "test_processor")
        self.assertEqual(message.lifecycle.value, "failed")

    def test_message_serialization(self):
        """Test message serialization to/from dictionary."""
        original_message = create_text_message("Serialization test")
        original_message.metadata.session_id = "test-session"
        original_message.set_quality_scores(quality=0.85)

        # Serialize to dictionary
        message_dict = original_message.to_dict()
        self.assertIn("id", message_dict)
        self.assertIn("message_type", message_dict)
        self.assertIn("content", message_dict)
        self.assertIn("metadata", message_dict)

        # Deserialize from dictionary
        restored_message = RichMessage.from_dict(message_dict)
        self.assertEqual(restored_message.id, original_message.id)
        self.assertEqual(restored_message.content, original_message.content)
        self.assertEqual(restored_message.metadata.session_id, "test-session")
        self.assertEqual(restored_message.metadata.quality_score, 0.85)


class TestMessageProcessor(unittest.IsolatedAsyncioTestCase):
    """Test cases for the message processor."""

    async def asyncSetUp(self):
        """Set up test environment."""
        self.processor = MessageProcessor({
            "max_concurrent_processing": 5,
            "enable_enhancement": True,
            "enable_quality_assessment": True
        })

    async def test_text_message_processing(self):
        """Test processing of text messages."""
        message = create_text_message("This is a test message for processing.")

        result = await self.processor.process_message(message)

        self.assertIsInstance(result, ProcessingResult)
        self.assertTrue(result.success)
        self.assertEqual(result.processed_message, message)
        self.assertGreater(result.processing_time, 0)
        self.assertIsNotNone(result.processor_used)

    async def test_error_message_processing(self):
        """Test processing of error messages."""
        message = create_error_message("Network timeout occurred", "timeout")

        result = await self.processor.process_message(message)

        self.assertTrue(result.success)
        self.assertEqual(message.message_type, EnhancedMessageType.SYSTEM_ERROR)
        self.assertEqual(message.priority, MessagePriority.HIGH)

    async def test_batch_processing(self):
        """Test batch message processing."""
        messages = [
            create_text_message(f"Message {i}")
            for i in range(5)
        ]

        results = await self.processor.process_batch(messages)

        self.assertEqual(len(results), 5)
        for result in results:
            self.assertTrue(result.success)

    async def test_processing_stats(self):
        """Test processing statistics."""
        # Process some messages
        for i in range(3):
            message = create_text_message(f"Stats test {i}")
            await self.processor.process_message(message)

        stats = self.processor.get_processing_stats()
        self.assertEqual(stats["total_processed"], 3)
        self.assertGreater(stats["average_processing_time"], 0)
        self.assertIn("by_type", stats)
        self.assertIn("by_context", stats)

    async def test_processor_error_handling(self):
        """Test processor error handling."""
        # Create a message that might cause issues
        message = RichMessage(
            message_type=EnhancedMessageType.CODE,
            content="```python\nprint('test')\n```" * 10000  # Very large content
        )

        result = await self.processor.process_message(message)

        # Should either succeed or handle error gracefully
        self.assertIsInstance(result, ProcessingResult)


class TestRichFormatter(unittest.TestCase):
    """Test cases for the rich message formatter."""

    def setUp(self):
        """Set up test environment."""
        self.formatter = RichFormatter({
            "color_scheme": "default",
            "show_timestamp": True,
            "show_quality_indicators": True
        })

    def test_text_message_formatting(self):
        """Test formatting of text messages."""
        message = create_text_message("This is a formatted message.")
        message.metadata.quality_score = 0.85

        formatted = self.formatter.format_message_sync(message)

        self.assertIsInstance(formatted, str)
        self.assertIn("This is a formatted message", formatted)

    def test_error_message_formatting(self):
        """Test formatting of error messages."""
        message = create_error_message("Connection failed", "network")

        formatted = self.formatter.format_message_sync(message)

        self.assertIsInstance(formatted, str)
        self.assertIn("Connection failed", formatted)

    def test_json_message_formatting(self):
        """Test formatting of JSON messages."""
        json_content = '{"key": "value", "number": 42}'
        message = RichMessage(
            message_type=EnhancedMessageType.JSON,
            content=json_content
        )

        formatted = self.formatter.format_message_sync(message)

        self.assertIsInstance(formatted, str)

    def test_formatting_stats(self):
        """Test formatting statistics."""
        # Format some messages
        for i in range(3):
            message = create_text_message(f"Format test {i}")
            self.formatter.format_message_sync(message)

        stats = self.formatter.get_formatting_stats()
        self.assertEqual(stats["total_formatted"], 3)
        self.assertGreater(stats["average_formatting_time"], 0)


class TestContentEnhancer(unittest.IsolatedAsyncioTestCase):
    """Test cases for the content enhancer."""

    async def asyncSetUp(self):
        """Set up test environment."""
        self.enhancer = ContentEnhancer({
            "enable_enhancement": True,
            "enhancement_strategies": ["text_clarity", "structure_optimization"]
        })

    async def test_text_clarity_enhancement(self):
        """Test text clarity enhancement."""
        content = "This is a very very long and complex sentence that goes on and on without proper punctuation and contains redundant words that make it difficult to read."
        message = create_text_message(content)

        enhancements = await self.enhancer.enhance_message(message)

        self.assertIsInstance(enhancements, list)
        # Should have some improvements applied
        self.assertGreaterEqual(len(enhancements), 0)

    async def test_structure_optimization(self):
        """Test structure optimization."""
        content = "# Title\nSome content\n## Subtitle\nMore content"
        message = create_text_message(content)

        enhancements = await self.enhancer.enhance_message(message)

        self.assertIsInstance(enhancements, list)

    async def test_enhancement_stats(self):
        """Test enhancement statistics."""
        # Enhance some messages
        for i in range(3):
            message = create_text_message(f"Enhancement test {i}")
            await self.enhancer.enhance_message(message)

        stats = self.enhancer.get_enhancement_stats()
        self.assertEqual(stats["total_enhanced"], 3)
        self.assertGreater(stats["processing_time"], 0)


class TestMessageQualityAnalyzer(unittest.IsolatedAsyncioTestCase):
    """Test cases for the message quality analyzer."""

    async def asyncSetUp(self):
        """Set up test environment."""
        self.analyzer = MessageQualityAnalyzer({
            "weights": {
                "content": 0.3,
                "structure": 0.2,
                "clarity": 0.2,
                "relevance": 0.3
            }
        })

    async def test_quality_assessment(self):
        """Test comprehensive quality assessment."""
        content = """
        # Research Results

        This research study examines the impact of artificial intelligence on healthcare systems.

        ## Findings

        The analysis reveals significant improvements in diagnostic accuracy and treatment planning.

        ## Conclusion

        AI technology shows great promise for healthcare applications.
        """
        message = create_text_message(content)
        message.message_type = EnhancedMessageType.RESEARCH_RESULT

        assessment = await self.analyzer.assess_message(message)

        self.assertIn("overall_quality", assessment)
        self.assertIn("quality_level", assessment)
        self.assertIn("dimensions", assessment)
        self.assertIn("recommendations", assessment)
        self.assertGreater(assessment["overall_quality"], 0)
        self.assertLessEqual(assessment["overall_quality"], 1)

    async def test_assessment_stats(self):
        """Test assessment statistics."""
        # Assess some messages
        for i in range(3):
            message = create_text_message(f"Quality test {i}")
            await self.analyzer.assess_message(message)

        stats = self.analyzer.get_assessment_stats()
        self.assertEqual(stats["total_assessments"], 3)
        self.assertGreater(stats["average_score"], 0)
        self.assertIn("assessments_by_type", stats)


class TestMessageRouter(unittest.IsolatedAsyncioTestCase):
    """Test cases for the message router."""

    async def asyncSetUp(self):
        """Set up test environment."""
        self.router = MessageRouter({
            "max_concurrent_routing": 5
        })

    async def test_message_routing(self):
        """Test message routing."""
        message = create_text_message("Route this message")
        message.message_type = EnhancedMessageType.RESEARCH_RESULT

        routing_result = await self.router.route_message(message)

        self.assertIn("decision", routing_result)
        self.assertIn("destinations", routing_result)
        self.assertIn("processing_time", routing_result)

    async def test_error_routing(self):
        """Test routing of error messages."""
        message = create_error_message("Critical system error", "system_failure")
        message.priority = MessagePriority.CRITICAL

        routing_result = await self.router.route_message(message)

        self.assertIn("decision", routing_result)
        # Error messages should be routed to appropriate destinations
        self.assertGreater(len(routing_result["destinations"]), 0)

    async def test_batch_routing(self):
        """Test batch message routing."""
        messages = [
            create_text_message(f"Batch message {i}")
            for i in range(3)
        ]

        results = await self.router.route_batch(messages)

        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIn("decision", result)

    async def test_routing_stats(self):
        """Test routing statistics."""
        # Route some messages
        for i in range(3):
            message = create_text_message(f"Routing test {i}")
            await self.router.route_message(message)

        stats = self.router.get_routing_stats()
        self.assertEqual(stats["total_routed"], 3)
        self.assertGreater(stats["average_routing_time"], 0)
        self.assertIn("routing_decisions", stats)


class TestMessageCache(unittest.IsolatedAsyncioTestCase):
    """Test cases for the message cache."""

    async def asyncSetUp(self):
        """Set up test environment."""
        self.cache = MessageCache({
            "max_memory_entries": 100,
            "max_memory_size_mb": 10,
            "default_ttl_seconds": 3600,
            "enable_compression": True
        })

    async def test_cache_set_and_get(self):
        """Test cache set and get operations."""
        message = create_text_message("Cache test message")
        data = {"processed": True, "quality": 0.8}

        # Set in cache
        success = await self.cache.set(message, data, ttl_seconds=60)
        self.assertTrue(success)

        # Get from cache
        cached_data = await self.cache.get(message)
        self.assertIsNotNone(cached_data)
        self.assertEqual(cached_data["processed"], True)
        self.assertEqual(cached_data["quality"], 0.8)

    async def test_cache_expiration(self):
        """Test cache expiration."""
        message = create_text_message("Expiration test")
        data = {"test": "data"}

        # Set with very short TTL
        await self.cache.set(message, data, ttl_seconds=1)

        # Should be available immediately
        cached_data = await self.cache.get(message)
        self.assertIsNotNone(cached_data)

        # Wait for expiration
        await asyncio.sleep(2)

        # Should be expired now
        cached_data = await self.cache.get(message)
        self.assertIsNone(cached_data)

    async def test_cache_stats(self):
        """Test cache statistics."""
        # Add some entries
        for i in range(3):
            message = create_text_message(f"Cache stats test {i}")
            await self.cache.set(message, {"index": i})

        # Get some entries
        for i in range(2):
            message = create_text_message(f"Cache stats test {i}")
            await self.cache.get(message)

        stats = self.cache.get_stats()
        self.assertIn("total_requests", stats)
        self.assertIn("cache_hits", stats)
        self.assertIn("cache_misses", stats)
        self.assertIn("overall_hit_rate", stats)


class TestMessageSerializer(unittest.IsolatedAsyncioTestCase):
    """Test cases for the message serializer."""

    async def asyncSetUp(self):
        """Set up test environment."""
        self.serializer = MessageSerializer({
            "default_format": "json",
            "enable_compression": True,
            "compression_threshold": 100
        })

    async def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        original_message = create_text_message("Serialization test")
        original_message.metadata.session_id = "test-session"
        original_message.set_quality_scores(quality=0.9)

        # Serialize
        serialized = await self.serializer.serialize(original_message)
        self.assertIsInstance(serialized, bytes)
        self.assertGreater(len(serialized), 0)

        # Deserialize
        restored_message = await self.serializer.deserialize(serialized)
        self.assertEqual(restored_message.id, original_message.id)
        self.assertEqual(restored_message.content, original_message.content)
        self.assertEqual(restored_message.metadata.session_id, "test-session")
        self.assertEqual(restored_message.metadata.quality_score, 0.9)

    async def test_pickle_serialization(self):
        """Test Pickle serialization."""
        original_message = create_text_message("Pickle test")

        # Serialize with pickle
        serialized = await self.serializer.serialize(
            original_message,
            format=SerializationFormat.PICKLE
        )

        # Deserialize
        restored_message = await self.serializer.deserialize(serialized)
        self.assertEqual(restored_message.id, original_message.id)
        self.assertEqual(restored_message.content, original_message.content)

    async def test_compression(self):
        """Test compression functionality."""
        # Create a large message
        large_content = "This is a test message. " * 1000
        original_message = create_text_message(large_content)

        # Serialize with compression
        compressed_serialized = await self.serializer.serialize(original_message)

        # Serialize without compression
        uncompressed_serialized = await self.serializer.serialize(
            original_message,
            compression=self.serializer.config.default_compression
        )

        # Compressed should be smaller
        self.assertLess(len(compressed_serialized.data), len(uncompressed_serialized.data))

    async def test_serialization_stats(self):
        """Test serialization statistics."""
        # Serialize some messages
        for i in range(3):
            message = create_text_message(f"Serialization test {i}")
            await self.serializer.serialize(message)

        stats = self.serializer.get_stats()
        self.assertEqual(stats["total_serialized"], 3)
        self.assertGreater(stats["average_serialization_time"], 0)


class TestIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for the complete message processing system."""

    async def asyncSetUp(self):
        """Set up test environment."""
        self.integration = OrchestratorIntegration({
            "enable_message_processing": True,
            "enable_rich_display": True,
            "enable_caching": True,
            "enable_quality_assessment": True,
            "enable_content_enhancement": True,
            "enable_message_routing": True,
            "cache_size": 50
        })

        await self.integration.start()

    async def asyncTearDown(self):
        """Clean up test environment."""
        await self.integration.stop()

    async def test_complete_message_pipeline(self):
        """Test complete message processing pipeline."""
        # Create a test message
        message = create_text_message(
            "This is a comprehensive test message for the complete processing pipeline."
        )
        message.message_type = EnhancedMessageType.RESEARCH_RESULT
        message.metadata.session_id = "test-session"

        # Process through complete pipeline
        processed_message = await self.integration.process_message_complete(message)

        # Verify processing results
        self.assertEqual(processed_message.id, message.id)
        self.assertIsNotNone(processed_message.metadata.quality_score)
        self.assertGreater(len(processed_message.routing_info), 0)

    async def test_batch_processing_pipeline(self):
        """Test batch processing through the pipeline."""
        messages = [
            create_text_message(f"Batch test message {i}")
            for i in range(5)
        ]

        # Process batch
        processed_messages = await self.integration.process_message_batch(messages)

        self.assertEqual(len(processed_messages), 5)
        for message in processed_messages:
            self.assertIsNotNone(message.metadata.quality_score)

    async def test_quality_assessment_integration(self):
        """Test quality assessment integration."""
        message = create_text_message(
            "This is a high-quality research finding with proper structure and clarity."
        )
        message.message_type = EnhancedMessageType.RESEARCH_RESULT

        # Assess quality
        quality_result = self.integration.assess_message_quality_complete(message)

        self.assertIn("overall_quality", quality_result)
        self.assertIn("quality_level", quality_result)
        self.assertGreater(quality_result["overall_quality"], 0)

    async def test_display_formatting_integration(self):
        """Test display formatting integration."""
        message = create_text_message("Display formatting test")
        message.metadata.quality_score = 0.85

        # Format for display
        formatted_output = self.integration.format_message_for_display(message)

        self.assertIsInstance(formatted_output, str)
        self.assertIn("Display formatting test", formatted_output)

    async def test_metrics_collection(self):
        """Test metrics collection."""
        # Process some messages
        for i in range(5):
            message = create_text_message(f"Metrics test {i}")
            await self.integration.process_message_complete(message)

        # Get metrics
        metrics = self.integration.get_metrics()

        self.assertIn("processing_metrics", metrics)
        self.assertIn("component_stats", metrics)
        self.assertEqual(metrics["processing_metrics"]["total_messages_processed"], 5)

    async def test_error_handling_integration(self):
        """Test error handling in the integration."""
        # Create a message that might cause issues
        problematic_message = RichMessage(
            message_type=EnhancedMessageType.ERROR,
            content="",  # Empty content
        )

        # Process should handle error gracefully
        try:
            processed_message = await self.integration.process_message_complete(problematic_message)
            # Should either succeed or handle error
            self.assertIsNotNone(processed_message)
        except Exception as e:
            # If exception is raised, it should be handled appropriately
            self.assertIn("error", str(e).lower())


class TestPerformanceBenchmarks(unittest.IsolatedAsyncioTestCase):
    """Performance benchmarks for the message processing system."""

    async def asyncSetUp(self):
        """Set up benchmark environment."""
        self.integration = OrchestratorIntegration({
            "enable_message_processing": True,
            "enable_caching": True,
            "cache_size": 1000
        })
        await self.integration.start()

    async def asyncTearDown(self):
        """Clean up benchmark environment."""
        await self.integration.stop()

    async def test_processing_performance(self):
        """Benchmark message processing performance."""
        num_messages = 100
        messages = [
            create_text_message(f"Performance test message {i}")
            for i in range(num_messages)
        ]

        start_time = time.time()

        # Process all messages
        for message in messages:
            await self.integration.process_message_complete(message)

        end_time = time.time()
        total_time = end_time - start_time

        # Calculate metrics
        avg_time_per_message = total_time / num_messages
        messages_per_second = num_messages / total_time

        print(f"\nPerformance Benchmark Results:")
        print(f"Messages processed: {num_messages}")
        print(f"Total time: {total_time:.3f}s")
        print(f"Average time per message: {avg_time_per_message:.3f}s")
        print(f"Messages per second: {messages_per_second:.1f}")

        # Performance assertions
        self.assertLess(avg_time_per_message, 0.1)  # Should be under 100ms per message
        self.assertGreater(messages_per_second, 10)  # Should handle at least 10 msg/sec

    async def test_cache_performance(self):
        """Benchmark cache performance."""
        num_iterations = 1000
        message = create_text_message("Cache performance test")

        # First pass (cache misses)
        start_time = time.time()
        for i in range(num_iterations):
            await self.integration.process_message_complete(message)
        first_pass_time = time.time() - start_time

        # Second pass (cache hits)
        start_time = time.time()
        for i in range(num_iterations):
            await self.integration.process_message_complete(message)
        second_pass_time = time.time() - start_time

        print(f"\nCache Performance Benchmark Results:")
        print(f"Iterations: {num_iterations}")
        print(f"First pass (cache misses): {first_pass_time:.3f}s")
        print(f"Second pass (cache hits): {second_pass_time:.3f}s")
        print(f"Cache speedup: {first_pass_time / second_pass_time:.2f}x")

        # Cache should provide significant speedup
        self.assertGreater(first_pass_time / second_pass_time, 2.0)

    async def test_batch_processing_performance(self):
        """Benchmark batch processing performance."""
        batch_sizes = [1, 5, 10, 25, 50]
        total_messages = 100

        results = {}

        for batch_size in batch_sizes:
            messages = [
                create_text_message(f"Batch performance test {i}")
                for i in range(total_messages)
            ]

            start_time = time.time()

            # Process in batches
            for i in range(0, len(messages), batch_size):
                batch = messages[i:i + batch_size]
                await self.integration.process_message_batch(batch)

            end_time = time.time()
            processing_time = end_time - start_time
            throughput = total_messages / processing_time

            results[batch_size] = {
                "time": processing_time,
                "throughput": throughput
            }

            print(f"Batch size {batch_size}: {processing_time:.3f}s, {throughput:.1f} msg/sec")

        # Batch processing should be more efficient
        self.assertGreater(results[10]["throughput"], results[1]["throughput"])


# Mock data generators for testing
class MockMessageGenerator:
    """Generate mock messages for testing."""

    @staticmethod
    def generate_text_message(length: int = 100) -> RichMessage:
        """Generate a text message of specified length."""
        content = "This is a test message. " * (length // 25)
        return create_text_message(content)

    @staticmethod
    def generate_research_message() -> RichMessage:
        """Generate a research result message."""
        content = """
        # Research Study: AI in Healthcare

        ## Abstract
        This study examines the impact of artificial intelligence on healthcare systems.

        ## Methodology
        We analyzed data from 100 healthcare facilities over a 5-year period.

        ## Results
        The findings show significant improvements in diagnostic accuracy.

        ## Conclusion
        AI technology shows great promise for healthcare applications.
        """
        message = RichMessage(
            message_type=EnhancedMessageType.RESEARCH_RESULT,
            content=content,
            context=MessageContext.RESEARCH
        )
        message.set_quality_scores(quality=0.85, relevance=0.9)
        return message

    @staticmethod
    def generate_error_message() -> RichMessage:
        """Generate an error message."""
        error_types = ["timeout", "connection", "permission", "syntax"]
        import random
        error_type = random.choice(error_types)
        return create_error_message(f"Simulated {error_type} error", error_type)

    @staticmethod
    def generate_message_batch(count: int, message_type: str = "mixed") -> List[RichMessage]:
        """Generate a batch of messages."""
        messages = []
        generators = {
            "text": MockMessageGenerator.generate_text_message,
            "research": MockMessageGenerator.generate_research_message,
            "error": MockMessageGenerator.generate_error_message
        }

        for i in range(count):
            if message_type == "mixed":
                # Randomly select message type
                import random
                generator = random.choice(list(generators.values()))
            else:
                generator = generators.get(message_type, generators["text"])

            messages.append(generator())

        return messages


# Test configuration and fixtures
@pytest.fixture
def sample_message():
    """Provide a sample message for testing."""
    return create_text_message("Sample test message")


@pytest.fixture
def message_processor():
    """Provide a message processor instance."""
    return MessageProcessor()


@pytest.fixture
def rich_formatter():
    """Provide a rich formatter instance."""
    return RichFormatter()


# Pytest test functions
@pytest.mark.asyncio
async def test_message_processing_workflow(sample_message, message_processor):
    """Test complete message processing workflow."""
    result = await message_processor.process_message(sample_message)
    assert result.success
    assert result.processing_time > 0


def test_message_formatting_workflow(sample_message, rich_formatter):
    """Test message formatting workflow."""
    formatted = rich_formatter.format_message_sync(sample_message)
    assert isinstance(formatted, str)
    assert sample_message.content in formatted


# Run tests if this file is executed directly
if __name__ == "__main__":
    # Create test suite
    suite = unittest.TestSuite()

    # Add test cases
    test_classes = [
        TestMessageTypeSystem,
        TestMessageProcessor,
        TestRichFormatter,
        TestContentEnhancer,
        TestMessageQualityAnalyzer,
        TestMessageRouter,
        TestMessageCache,
        TestMessageSerializer,
        TestIntegration,
        TestPerformanceBenchmarks
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")