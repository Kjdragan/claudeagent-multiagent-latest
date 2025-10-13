"""
Main Message Processing System Entry Point

This module provides the main entry point and integration layer for the
comprehensive message processing system, bringing together all components
into a cohesive, production-ready system.

Key Features:
- Complete system initialization and configuration
- Easy-to-use API for message processing
- Component integration and coordination
- Performance monitoring and optimization
- Comprehensive error handling and recovery
- Production-ready deployment patterns
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path

# Import all message processing components
from .core.message_types import (
    RichMessage, EnhancedMessageType, MessagePriority, MessageContext,
    MessageBuilder, create_text_message, create_error_message
)
from .core.message_processor import MessageProcessor
from .formatters.rich_formatter import RichFormatter
from .analyzers.content_enhancer import ContentEnhancer
from .analyzers.message_quality_analyzer import MessageQualityAnalyzer
from .routers.message_router import MessageRouter
from .cache.message_cache import MessageCache
from .serializers.message_serializer import MessageSerializer, MessagePersistence
from .integration.orchestrator_integration import OrchestratorIntegration
from .monitoring.performance_monitor import (
    PerformanceMonitor, get_performance_monitor,
    start_global_monitoring, stop_global_monitoring
)

# Version information
__version__ = "2.3.0"
__author__ = "Multi-Agent Research System"
__description__ = "Comprehensive Rich Message Processing and Display System"


class MessageProcessingSystem:
    """
    Main message processing system that integrates all components
    into a cohesive, production-ready solution.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the complete message processing system."""
        self.config = config or {}
        self.logger = self._setup_logging()

        # Initialize all components
        self._initialize_components()

        # Integration layer
        self.integration = None

        # Performance monitoring
        self.performance_monitor = None

        # System state
        self._initialized = False
        self._running = False

    def _setup_logging(self) -> logging.Logger:
        """Set up comprehensive logging for the system."""
        logger = logging.getLogger("message_processing")
        logger.setLevel(self.config.get("log_level", logging.INFO))

        if not logger.handlers:
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # Create file handler if log directory is specified
            log_dir = self.config.get("log_directory")
            if log_dir:
                log_path = Path(log_dir) / "message_processing.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(log_path)
                file_handler.setLevel(logging.DEBUG)
                logger.addHandler(file_handler)

            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            if log_dir:
                file_handler.setFormatter(formatter)

            logger.addHandler(console_handler)

        return logger

    def _initialize_components(self):
        """Initialize all system components."""
        try:
            # Core processing components
            self.message_processor = MessageProcessor(
                self.config.get("processor", {})
            )

            self.rich_formatter = RichFormatter(
                self.config.get("formatter", {})
            )

            self.content_enhancer = ContentEnhancer(
                self.config.get("enhancer", {})
            )

            self.quality_analyzer = MessageQualityAnalyzer(
                self.config.get("quality", {})
            )

            self.message_router = MessageRouter(
                self.config.get("router", {})
            )

            self.message_cache = MessageCache(
                self.config.get("cache", {})
            )

            self.message_serializer = MessageSerializer(
                self.config.get("serializer", {})
            )

            self.message_persistence = MessagePersistence(
                backend=self.config.get("persistence_backend", "file"),
                config=self.config.get("persistence", {})
            )

            # Performance monitoring
            if self.config.get("enable_monitoring", True):
                self.performance_monitor = PerformanceMonitor(
                    self.config.get("monitoring", {})
                )

            # Integration layer
            if self.config.get("enable_integration", True):
                self.integration = OrchestratorIntegration(
                    self.config.get("integration", {})
                )

            self._initialized = True
            self.logger.info("Message processing system initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize message processing system: {str(e)}")
            raise

    async def start(self):
        """Start the message processing system."""
        if not self._initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")

        if self._running:
            self.logger.warning("System is already running")
            return

        try:
            # Start performance monitoring
            if self.performance_monitor:
                await self.performance_monitor.start_monitoring()

            # Start integration layer
            if self.integration:
                await self.integration.start()

            # Start cache background cleanup
            await self.message_cache.start_background_cleanup()

            self._running = True
            self.logger.info("Message processing system started successfully")

        except Exception as e:
            self.logger.error(f"Failed to start message processing system: {str(e)}")
            raise

    async def stop(self):
        """Stop the message processing system."""
        if not self._running:
            self.logger.warning("System is not running")
            return

        try:
            # Stop integration layer
            if self.integration:
                await self.integration.stop()

            # Stop cache background cleanup
            await self.message_cache.stop_background_cleanup()

            # Stop performance monitoring
            if self.performance_monitor:
                await self.performance_monitor.stop_monitoring()

            # Close persistence connections
            self.message_persistence.close()

            self._running = False
            self.logger.info("Message processing system stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping message processing system: {str(e)}")

    async def process_message(self, message: RichMessage) -> RichMessage:
        """
        Process a single message through the complete pipeline.

        Args:
            message: The message to process

        Returns:
            The processed message
        """
        if not self._running:
            raise RuntimeError("System not started. Call start() first.")

        start_time = datetime.now()

        try:
            # Record processing start
            if self.performance_monitor:
                self.performance_monitor.record_metric(
                    "messages_processing_total",
                    self.performance_monitor.counters.get("messages_processing", 0) + 1,
                    self.performance_monitor.MetricType.COUNTER
                )

            # Process message through the pipeline
            if self.integration:
                processed_message = await self.integration.process_message_complete(message)
            else:
                processed_message = await self.message_processor.process_message(message).processed_message

            # Record processing metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            if self.performance_monitor:
                self.performance_monitor.record_message_processing(
                    processed_message, processing_time, True
                )

            return processed_message

        except Exception as e:
            # Record error metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            if self.performance_monitor:
                self.performance_monitor.record_message_processing(
                    message, processing_time, False
                )

            self.logger.error(f"Error processing message {message.id}: {str(e)}")
            raise

    async def process_batch(self, messages: List[RichMessage]) -> List[RichMessage]:
        """
        Process multiple messages in batch.

        Args:
            messages: List of messages to process

        Returns:
            List of processed messages
        """
        if not self._running:
            raise RuntimeError("System not started. Call start() first.")

        if self.integration:
            return await self.integration.process_message_batch(messages)
        else:
            results = await self.message_processor.process_batch(messages)
            return [r.processed_message for r in results]

    def format_message(self, message: RichMessage) -> str:
        """
        Format a message for rich display.

        Args:
            message: The message to format

        Returns:
            Formatted message string
        """
        if self.integration:
            return self.integration.format_message_for_display(message)
        else:
            return self.rich_formatter.format_message_sync(message)

    def assess_quality(self, message: RichMessage) -> Dict[str, Any]:
        """
        Assess the quality of a message.

        Args:
            message: The message to assess

        Returns:
            Quality assessment results
        """
        if self.integration:
            return self.integration.assess_message_quality_complete(message)
        else:
            return asyncio.run(self.quality_analyzer.assess_message(message))

    async def route_message(self, message: RichMessage) -> Dict[str, Any]:
        """
        Route a message to appropriate destinations.

        Args:
            message: The message to route

        Returns:
            Routing results
        """
        return await self.message_router.route_message(message)

    async def cache_message(self, message: RichMessage, data: Dict[str, Any], ttl_seconds: int = None) -> bool:
        """
        Cache a message with associated data.

        Args:
            message: The message to cache
            data: Data to cache with the message
            ttl_seconds: Time to live in seconds

        Returns:
            True if cached successfully
        """
        return await self.message_cache.set(message, data, ttl_seconds)

    async def get_cached_message(self, message: RichMessage) -> Optional[Dict[str, Any]]:
        """
        Get cached data for a message.

        Args:
            message: The message to get cached data for

        Returns:
            Cached data or None if not found
        """
        return await self.message_cache.get(message)

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status and metrics.

        Returns:
            System status information
        """
        status = {
            "system": {
                "initialized": self._initialized,
                "running": self._running,
                "version": __version__,
                "uptime": self._get_uptime()
            },
            "components": {
                "message_processor": self.message_processor.get_processing_stats(),
                "rich_formatter": self.rich_formatter.get_formatting_stats(),
                "content_enhancer": self.content_enhancer.get_enhancement_stats(),
                "quality_analyzer": self.quality_analyzer.get_assessment_stats(),
                "message_router": self.message_router.get_routing_stats(),
                "message_cache": self.message_cache.get_stats(),
                "message_serializer": self.message_serializer.get_stats()
            }
        }

        # Add performance monitor stats
        if self.performance_monitor:
            status["performance"] = self.performance_monitor.get_performance_summary()
            status["monitoring"] = self.performance_monitor.get_monitoring_stats()

        # Add integration stats
        if self.integration:
            status["integration"] = self.integration.get_metrics()

        return status

    def _get_uptime(self) -> str:
        """Get system uptime as formatted string."""
        # This would track actual uptime in a real implementation
        return "Not tracked"

    # Convenience methods for common operations
    def create_text_message(self, content: str, **kwargs) -> RichMessage:
        """Create a text message with default settings."""
        return create_text_message(content, **kwargs)

    def create_error_message(self, content: str, error_type: str = "general", **kwargs) -> RichMessage:
        """Create an error message with default settings."""
        return create_error_message(content, error_type, **kwargs)

    def create_message_builder(self) -> MessageBuilder:
        """Get a message builder for fluent message creation."""
        return MessageBuilder()

    async def create_and_process_message(self, content: str, message_type: str = "text", **kwargs) -> RichMessage:
        """
        Create and process a message in one step.

        Args:
            content: Message content
            message_type: Type of message
            **kwargs: Additional message parameters

        Returns:
            Processed message
        """
        if message_type == "text":
            message = self.create_text_message(content, **kwargs)
        elif message_type == "error":
            message = self.create_error_message(content, **kwargs)
        else:
            # Create generic message
            message = RichMessage(
                message_type=EnhancedMessageType(message_type),
                content=content,
                **kwargs
            )

        return await self.process_message(message)

    # Configuration management
    def update_config(self, new_config: Dict[str, Any]):
        """Update system configuration."""
        self.config.update(new_config)

        # Update component configurations
        if "processor" in new_config:
            self.message_processor.config.update(new_config["processor"])

        if "formatter" in new_config:
            self.rich_formatter.config.update(new_config["formatter"])

        if "cache" in new_config:
            self.message_cache.config.update(new_config["cache"])

        self.logger.info("System configuration updated")

    def get_config(self) -> Dict[str, Any]:
        """Get current system configuration."""
        return {
            "system": self.config,
            "components": {
                "processor": self.message_processor.config,
                "formatter": self.rich_formatter.config,
                "cache": self.message_cache.config
            }
        }

    # Export and reporting
    def export_metrics(self, format: str = "json") -> str:
        """Export system metrics."""
        if self.performance_monitor:
            return self.performance_monitor.export_metrics(format)
        else:
            return '{"error": "Performance monitoring not enabled"}'

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive system report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "system_status": self.get_system_status(),
            "performance_metrics": self.export_metrics() if self.performance_monitor else {},
            "optimization_suggestions": (
                self.performance_monitor.get_optimization_suggestions()
                if self.performance_monitor else []
            )
        }


# Convenience functions for easy usage
async def create_system(config: Dict[str, Any] = None) -> MessageProcessingSystem:
    """
    Create and start a message processing system.

    Args:
        config: System configuration

    Returns:
        Started message processing system
    """
    system = MessageProcessingSystem(config)
    await system.start()
    return system


async def process_message_simple(content: str, config: Dict[str, Any] = None) -> str:
    """
    Simple one-off message processing.

    Args:
        content: Message content to process
        config: Optional configuration

    Returns:
        Formatted processed message
    """
    system_config = {
        "enable_monitoring": False,
        "enable_caching": False,
        "enable_integration": False,
        **(config or {})
    }

    async with create_system(system_config) as system:
        message = system.create_text_message(content)
        processed_message = await system.process_message(message)
        formatted_message = system.format_message(processed_message)
        return formatted_message


# Context manager for easy system lifecycle management
class MessageProcessingContext:
    """Context manager for message processing system lifecycle."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.system = None

    async def __aenter__(self) -> MessageProcessingSystem:
        """Enter context manager."""
        self.system = MessageProcessingSystem(self.config)
        await self.system.start()
        return self.system

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if self.system:
            await self.system.stop()


# Example usage and demonstration
async def demo_system():
    """Demonstrate the message processing system capabilities."""
    print("🚀 Starting Message Processing System Demo")
    print("=" * 60)

    # Create system with demo configuration
    config = {
        "log_level": "INFO",
        "enable_monitoring": True,
        "enable_caching": True,
        "cache_size": 100,
        "processor": {
            "enable_enhancement": True,
            "enable_quality_assessment": True
        },
        "formatter": {
            "color_scheme": "default",
            "show_quality_indicators": True
        }
    }

    async with MessageProcessingContext(config) as system:
        print("\n📝 Processing a text message:")
        text_message = system.create_text_message(
            "This is a demonstration of the comprehensive message processing system.",
            message_type=EnhancedMessageType.TEXT,
            context=MessageContext.RESEARCH
        )

        processed_text = await system.process_message(text_message)
        formatted_text = system.format_message(processed_text)
        print("✅ Text message processed successfully")
        print(f"   Quality score: {processed_text.metadata.quality_score:.2f}")
        print(f"   Processing time: {processed_text.performance_metrics.get('total_processing_time', 0):.3f}s")

        print("\n🔍 Processing a research result:")
        research_content = """
        # AI in Healthcare Research Results

        Our comprehensive study reveals significant improvements in diagnostic accuracy
        through AI implementation in healthcare systems.

        Key Findings:
        - 95% improvement in early detection
        - 50% reduction in diagnostic time
        - 87% patient satisfaction rate
        """
        research_message = system.create_text_message(
            research_content,
            message_type=EnhancedMessageType.RESEARCH_RESULT,
            priority=MessagePriority.HIGH
        )

        processed_research = await system.process_message(research_message)
        quality_assessment = system.assess_quality(processed_research)
        print("✅ Research message processed successfully")
        print(f"   Overall quality: {quality_assessment.get('overall_quality', 0):.2%}")
        print(f"   Quality level: {quality_assessment.get('quality_level', 'unknown')}")

        print("\n⚠️ Processing an error message:")
        error_message = system.create_error_message(
            "Network timeout occurred while connecting to research database",
            "network_error"
        )

        processed_error = await system.process_message(error_message)
        formatted_error = system.format_message(processed_error)
        print("✅ Error message processed successfully")
        print(f"   Priority: {processed_error.priority.value}")

        print("\n📊 Batch processing demo:")
        batch_messages = [
            system.create_text_message(f"Batch message {i + 1}")
            for i in range(5)
        ]

        processed_batch = await system.process_batch(batch_messages)
        print(f"✅ Batch of {len(processed_batch)} messages processed successfully")

        print("\n📈 System Status:")
        status = system.get_system_status()
        print(f"   System running: {status['system']['running']}")
        print(f"   Components active: {len(status['components'])}")
        if status.get('performance'):
            print(f"   Performance monitoring: Active")

        print("\n🎯 System Report:")
        report = system.generate_report()
        print(f"   Report generated at: {report['timestamp']}")
        print(f"   System health: {report['performance_metrics'].get('system_health', {}).get('status', 'unknown')}")

    print("\n✅ Demo completed successfully!")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_system())