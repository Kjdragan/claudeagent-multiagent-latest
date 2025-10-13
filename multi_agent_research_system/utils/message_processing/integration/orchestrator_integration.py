"""
Orchestrator Integration - Seamless Integration with Enhanced Orchestrator and Sub-Agents

This module provides comprehensive integration between the message processing system
and the enhanced orchestrator, enabling rich message processing throughout the
research workflow.

Key Features:
- Seamless integration with enhanced orchestrator
- Sub-agent communication enhancement
- Rich message processing for all workflow stages
- Performance monitoring and optimization
- Error handling and recovery integration
- Configuration management and synchronization
- Message lifecycle management
- Workflow stage-specific processing
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field

from ..core.message_processor import MessageProcessor
from ..core.message_types import RichMessage, EnhancedMessageType, MessagePriority, MessageContext
from ..formatters.rich_formatter import RichFormatter
from ..analyzers.content_enhancer import ContentEnhancer
from ..analyzers.message_quality_analyzer import MessageQualityAnalyzer
from ..routers.message_router import MessageRouter
from ..cache.message_cache import MessageCache
from ..serializers.message_serializer import MessageSerializer, MessagePersistence

# Import orchestrator components (assuming they exist)
try:
    from ...core.enhanced_orchestrator import EnhancedOrchestrator, RichMessageProcessor as OrchestratorMessageProcessor
    from ...utils.sub_agents.sub_agent_coordinator import SubAgentCoordinator
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False
    EnhancedOrchestrator = None
    SubAgentCoordinator = None


@dataclass
class IntegrationConfig:
    """Configuration for orchestrator integration."""

    enable_message_processing: bool = True
    enable_rich_display: bool = True
    enable_caching: bool = True
    enable_quality_assessment: bool = True
    enable_content_enhancement: bool = True
    enable_message_routing: bool = True
    enable_persistence: bool = False
    cache_size: int = 1000
    processing_timeout: float = 30.0
    batch_processing: bool = True
    batch_size: int = 10
    performance_monitoring: bool = True
    error_recovery: bool = True
    log_level: str = "INFO"


@dataclass
class ProcessingMetrics:
    """Metrics for message processing performance."""

    total_messages_processed: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    cache_hit_rate: float = 0.0
    quality_improvement_rate: float = 0.0
    error_rate: float = 0.0
    by_stage: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_agent: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class OrchestratorIntegration:
    """Integration layer for message processing with orchestrator."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize orchestrator integration."""
        self.config = self._create_integration_config(config or {})
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.config.log_level)

        # Core message processing components
        self.message_processor = MessageProcessor(self.config.get("processor", {}))
        self.rich_formatter = RichFormatter(self.config.get("formatter", {}))
        self.content_enhancer = ContentEnhancer(self.config.get("enhancer", {}))
        self.quality_analyzer = MessageQualityAnalyzer(self.config.get("quality", {}))
        self.message_router = MessageRouter(self.config.get("router", {}))
        self.message_cache = MessageCache(self.config.get("cache", {}))
        self.message_serializer = MessageSerializer(self.config.get("serializer", {}))
        self.message_persistence = MessagePersistence(
            backend=self.config.get("persistence_backend", "file"),
            config=self.config.get("persistence", {})
        )

        # Orchestrator integration
        self.orchestrator = None
        self.sub_agent_coordinator = None

        # Processing metrics
        self.metrics = ProcessingMetrics()

        # Integration state
        self._running = False
        self._background_tasks = []

        # Message hooks for orchestrator
        self._setup_message_hooks()

    def _create_integration_config(self, config: Dict[str, Any]) -> IntegrationConfig:
        """Create integration configuration from settings."""
        return IntegrationConfig(
            enable_message_processing=config.get("enable_message_processing", True),
            enable_rich_display=config.get("enable_rich_display", True),
            enable_caching=config.get("enable_caching", True),
            enable_quality_assessment=config.get("enable_quality_assessment", True),
            enable_content_enhancement=config.get("enable_content_enhancement", True),
            enable_message_routing=config.get("enable_message_routing", True),
            enable_persistence=config.get("enable_persistence", False),
            cache_size=config.get("cache_size", 1000),
            processing_timeout=config.get("processing_timeout", 30.0),
            batch_processing=config.get("batch_processing", True),
            batch_size=config.get("batch_size", 10),
            performance_monitoring=config.get("performance_monitoring", True),
            error_recovery=config.get("error_recovery", True),
            log_level=config.get("log_level", "INFO")
        )

    def _setup_message_hooks(self):
        """Setup message processing hooks for orchestrator."""
        if not ORCHESTRATOR_AVAILABLE:
            self.logger.warning("Orchestrator not available - integration hooks disabled")
            return

        # Define hook functions
        self.message_hooks = {
            "pre_process": self._hook_pre_process_message,
            "post_process": self._hook_post_process_message,
            "format_display": self._hook_format_display_message,
            "assess_quality": self._hook_assess_quality_message,
            "route_message": self._hook_route_message,
            "cache_message": self._hook_cache_message,
            "persist_message": self._hook_persist_message
        }

    def integrate_with_orchestrator(self, orchestrator) -> bool:
        """Integrate with enhanced orchestrator."""
        if not ORCHESTRATOR_AVAILABLE:
            self.logger.error("Cannot integrate - orchestrator not available")
            return False

        try:
            self.orchestrator = orchestrator

            # Replace orchestrator's message processor with enhanced version
            if hasattr(orchestrator, 'message_processor'):
                orchestrator.message_processor = self._create_enhanced_message_processor()

            # Hook into orchestrator's workflow stages
            self._integrate_workflow_stages()

            # Setup sub-agent coordinator integration
            if hasattr(orchestrator, 'sub_agent_coordinator'):
                self.integrate_with_sub_agents(orchestrator.sub_agent_coordinator)

            self.logger.info("Successfully integrated with enhanced orchestrator")
            return True

        except Exception as e:
            self.logger.error(f"Failed to integrate with orchestrator: {str(e)}")
            return False

    def integrate_with_sub_agents(self, sub_agent_coordinator) -> bool:
        """Integrate with sub-agent coordinator."""
        if not ORCHESTRATOR_AVAILABLE or not sub_agent_coordinator:
            self.logger.warning("Cannot integrate with sub-agents - components not available")
            return False

        try:
            self.sub_agent_coordinator = sub_agent_coordinator

            # Hook into sub-agent communication
            self._setup_sub_agent_hooks()

            # Enhance sub-agent message processing
            self._enhance_sub_agent_communication()

            self.logger.info("Successfully integrated with sub-agent coordinator")
            return True

        except Exception as e:
            self.logger.error(f"Failed to integrate with sub-agents: {str(e)}")
            return False

    def _create_enhanced_message_processor(self):
        """Create enhanced message processor for orchestrator."""
        class EnhancedMessageProcessor:
            """Enhanced message processor with rich processing capabilities."""

            def __init__(self, integration):
                self.integration = integration

            async def process_message(self, message: RichMessage) -> RichMessage:
                """Process message through enhanced pipeline."""
                return await self.integration.process_message_complete(message)

            async def process_batch(self, messages: List[RichMessage]) -> List[RichMessage]:
                """Process batch of messages."""
                return await self.integration.process_message_batch(messages)

            def format_for_display(self, message: RichMessage) -> str:
                """Format message for rich display."""
                return self.integration.format_message_for_display(message)

            def assess_message_quality(self, message: RichMessage) -> Dict[str, Any]:
                """Assess message quality."""
                return self.integration.assess_message_quality_complete(message)

        return EnhancedMessageProcessor(self)

    def _integrate_workflow_stages(self):
        """Integrate message processing into orchestrator workflow stages."""
        if not self.orchestrator:
            return

        # Hook into key workflow stages
        workflow_stages = {
            "research_stage": self._enhance_research_stage,
            "report_generation_stage": self._enhance_report_stage,
            "editorial_review_stage": self._enhance_editorial_stage,
            "quality_assessment_stage": self._enhance_quality_stage,
            "final_output_stage": self._enhance_output_stage
        }

        for stage_name, enhancer in workflow_stages.items():
            if hasattr(self.orchestrator, stage_name):
                original_method = getattr(self.orchestrator, stage_name)
                setattr(self.orchestrator, stage_name, enhancer(original_method))

    def _setup_sub_agent_hooks(self):
        """Setup hooks for sub-agent communication."""
        if not self.sub_agent_coordinator:
            return

        # Hook into agent communication
        if hasattr(self.sub_agent_coordinator, 'send_message'):
            original_send = self.sub_agent_coordinator.send_message
            self.sub_agent_coordinator.send_message = self._enhance_agent_send_message(original_send)

        if hasattr(self.sub_agent_coordinator, 'receive_message'):
            original_receive = self.sub_agent_coordinator.receive_message
            self.sub_agent_coordinator.receive_message = self._enhance_agent_receive_message(original_receive)

    async def process_message_complete(self, message: RichMessage) -> RichMessage:
        """Process message through complete pipeline."""
        if not self.config.enable_message_processing:
            return message

        start_time = datetime.now()

        try:
            # Apply pre-processing hooks
            await self._apply_hook("pre_process", message)

            # Check cache first
            if self.config.enable_caching:
                cached_result = await self.message_cache.get(message)
                if cached_result:
                    self.metrics.cache_hit_rate = (self.metrics.cache_hit_rate + 1) / max(self.metrics.total_messages_processed + 1, 1)
                    self.logger.debug(f"Cache hit for message {message.id}")
                    return message

            # Route message if enabled
            if self.config.enable_message_routing:
                routing_result = await self.message_router.route_message(message)
                message.routing_info.update(routing_result)

            # Process through message processor
            processing_result = await self.message_processor.process_message(message)

            if processing_result.success:
                message = processing_result.processed_message

                # Assess quality if enabled
                if self.config.enable_quality_assessment:
                    quality_result = await self.quality_analyzer.assess_message(message)
                    message.metadata.quality_score = quality_result.get("overall_quality", 0.5)

                    # Track quality improvement
                    if processing_result.quality_improvement > 0:
                        self.metrics.quality_improvement_rate = (
                            self.metrics.quality_improvement_rate + 1
                        ) / max(self.metrics.total_messages_processed + 1, 1)

                # Cache the processed message
                if self.config.enable_caching:
                    cache_data = {
                        "processing_result": processing_result.__dict__,
                        "quality_score": message.metadata.quality_score,
                        "routing_info": message.routing_info
                    }
                    await self.message_cache.set(message, cache_data)

                # Persist message if enabled
                if self.config.enable_persistence:
                    await self.message_persistence.save_message(message)

            # Apply post-processing hooks
            await self._apply_hook("post_process", message)

            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_metrics(message, processing_time, True)

            return message

        except Exception as e:
            # Handle processing errors
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_metrics(message, processing_time, False)

            if self.config.error_recovery:
                return await self._handle_processing_error(message, e)
            else:
                raise

    async def process_message_batch(self, messages: List[RichMessage]) -> List[RichMessage]:
        """Process batch of messages."""
        if not self.config.batch_processing or len(messages) <= self.config.batch_size:
            # Process individually
            results = []
            for message in messages:
                result = await self.process_message_complete(message)
                results.append(result)
            return results

        # Process in batches
        batch_results = []
        for i in range(0, len(messages), self.config.batch_size):
            batch = messages[i:i + self.config.batch_size]
            batch_result = await self.message_processor.process_batch(batch)
            batch_results.extend([r.processed_message for r in batch_result])

        return batch_results

    def format_message_for_display(self, message: RichMessage) -> str:
        """Format message for rich display."""
        if not self.config.enable_rich_display:
            return message.content

        try:
            # Apply display formatting hook
            asyncio.create_task(self._apply_hook("format_display", message))
            return self.rich_formatter.format_message_sync(message)
        except Exception as e:
            self.logger.error(f"Display formatting failed: {str(e)}")
            return message.content

    def assess_message_quality_complete(self, message: RichMessage) -> Dict[str, Any]:
        """Assess message quality with comprehensive analysis."""
        if not self.config.enable_quality_assessment:
            return {"overall_quality": 0.5, "quality_level": "unknown"}

        try:
            # Apply quality assessment hook
            asyncio.create_task(self._apply_hook("assess_quality", message))
            return asyncio.run(self.quality_analyzer.assess_message(message))
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {str(e)}")
            return {"overall_quality": 0.5, "error": str(e)}

    # Hook implementations
    async def _hook_pre_process_message(self, message: RichMessage):
        """Pre-processing hook for messages."""
        # Validate message structure
        if not message.content and message.message_type not in [
            EnhancedMessageType.SYSTEM_INFO,
            EnhancedMessageType.PROGRESS_UPDATE
        ]:
            self.logger.warning(f"Message {message.id} has no content")

        # Set default values if missing
        if not message.session_id and self.orchestrator:
            # Try to get current session from orchestrator
            if hasattr(self.orchestrator, 'current_session_id'):
                message.session_id = self.orchestrator.current_session_id

    async def _hook_post_process_message(self, message: RichMessage):
        """Post-processing hook for messages."""
        # Update timestamp
        message.metadata.updated_at = datetime.now()

        # Log processing completion
        if message.metadata.quality_score:
            self.logger.debug(f"Processed message {message.id} with quality {message.metadata.quality_score:.2f}")

    async def _hook_format_display_message(self, message: RichMessage):
        """Display formatting hook for messages."""
        # Apply rich display formatting
        if self.config.enable_rich_display:
            display_config = {
                "show_quality_indicators": True,
                "show_timestamp": True,
                "show_metadata": self.config.performance_monitoring
            }
            # Note: RichFormatter doesn't have a way to apply config post-initialization
            # This would require modification to the RichFormatter class

    async def _hook_assess_quality_message(self, message: RichMessage):
        """Quality assessment hook for messages."""
        # Additional quality checks can be added here
        pass

    async def _hook_route_message(self, message: RichMessage):
        """Message routing hook."""
        # Apply custom routing logic if needed
        pass

    async def _hook_cache_message(self, message: RichMessage):
        """Message caching hook."""
        # Custom caching logic can be added here
        pass

    async def _hook_persist_message(self, message: RichMessage):
        """Message persistence hook."""
        # Custom persistence logic can be added here
        pass

    # Enhancement methods for orchestrator stages
    def _enhance_research_stage(self, original_method):
        """Enhance research stage with message processing."""
        async def enhanced_research_stage(*args, **kwargs):
            # Call original method
            result = await original_method(*args, **kwargs)

            # Process research results through message system
            if isinstance(result, dict) and "research_data" in result:
                research_message = RichMessage(
                    message_type=EnhancedMessageType.RESEARCH_RESULT,
                    content=str(result["research_data"]),
                    context=MessageContext.RESEARCH
                )
                processed_message = await self.process_message_complete(research_message)
                result["processed_research_data"] = processed_message

            return result

        return enhanced_research_stage

    def _enhance_report_stage(self, original_method):
        """Enhance report generation stage with message processing."""
        async def enhanced_report_stage(*args, **kwargs):
            # Call original method
            result = await original_method(*args, **kwargs)

            # Process report through message system
            if isinstance(result, dict) and "report_content" in result:
                report_message = RichMessage(
                    message_type=EnhancedMessageType.TEXT,
                    content=result["report_content"],
                    context=MessageContext.REPORTING
                )
                processed_message = await self.process_message_complete(report_message)
                result["enhanced_report_content"] = processed_message.content

            return result

        return enhanced_report_stage

    def _enhance_editorial_stage(self, original_method):
        """Enhance editorial review stage with message processing."""
        async def enhanced_editorial_stage(*args, **kwargs):
            # Call original method
            result = await original_method(*args, **kwargs)

            # Process editorial review through message system
            if isinstance(result, dict) and "editorial_review" in result:
                editorial_message = RichMessage(
                    message_type=EnhancedMessageType.QUALITY_ASSESSMENT,
                    content=str(result["editorial_review"]),
                    context=MessageContext.EDITORIAL
                )
                processed_message = await self.process_message_complete(editorial_message)
                result["enhanced_editorial_review"] = processed_message.content

            return result

        return enhanced_editorial_stage

    def _enhance_quality_stage(self, original_method):
        """Enhance quality assessment stage with message processing."""
        async def enhanced_quality_stage(*args, **kwargs):
            # Call original method
            result = await original_method(*args, **kwargs)

            # Process quality assessment through message system
            if isinstance(result, dict) and "quality_assessment" in result:
                quality_message = RichMessage(
                    message_type=EnhancedMessageType.QUALITY_ASSESSMENT,
                    content=str(result["quality_assessment"]),
                    context=MessageContext.QUALITY
                )
                processed_message = await self.process_message_complete(quality_message)
                result["enhanced_quality_assessment"] = processed_message.content

            return result

        return enhanced_quality_stage

    def _enhance_output_stage(self, original_method):
        """Enhance final output stage with message processing."""
        async def enhanced_output_stage(*args, **kwargs):
            # Call original method
            result = await original_method(*args, **kwargs)

            # Process final output through message system
            if isinstance(result, dict) and "final_output" in result:
                output_message = RichMessage(
                    message_type=EnhancedMessageType.TEXT,
                    content=result["final_output"],
                    context=MessageContext.REPORTING,
                    priority=MessagePriority.HIGH
                )
                processed_message = await self.process_message_complete(output_message)
                result["enhanced_final_output"] = processed_message.content

            return result

        return enhanced_output_stage

    # Sub-agent communication enhancement
    def _enhance_agent_send_message(self, original_send):
        """Enhance agent message sending with rich processing."""
        async def enhanced_send(to_agent: str, message: RichMessage, **kwargs):
            # Process message before sending
            processed_message = await self.process_message_complete(message)

            # Add routing information
            processed_message.metadata.target_agent = to_agent

            # Send processed message
            result = await original_send(to_agent, processed_message, **kwargs)

            return result

        return enhanced_send

    def _enhance_agent_receive_message(self, original_receive):
        """Enhance agent message receiving with rich processing."""
        async def enhanced_receive(from_agent: str, message: RichMessage, **kwargs):
            # Add source information
            message.metadata.source_agent = from_agent

            # Process received message
            processed_message = await self.process_message_complete(message)

            # Return processed message
            result = await original_receive(from_agent, processed_message, **kwargs)

            return result

        return enhanced_receive

    # Error handling and recovery
    async def _handle_processing_error(self, message: RichMessage, error: Exception) -> RichMessage:
        """Handle processing errors with recovery."""
        self.logger.error(f"Processing error for message {message.id}: {str(error)}")

        # Mark message as having processing error
        message.add_processing_step("error_handler", 0.0, f"failed: {str(error)}")

        # Try to apply minimal processing
        try:
            # Basic formatting only
            if message.content:
                message.formatting.update({
                    "style": "error",
                    "error_info": str(error)
                })

            return message

        except Exception as recovery_error:
            self.logger.error(f"Error recovery also failed: {str(recovery_error)}")
            return message

    async def _apply_hook(self, hook_name: str, message: RichMessage):
        """Apply a specific hook to a message."""
        if hook_name in self.message_hooks:
            try:
                await self.message_hooks[hook_name](message)
            except Exception as e:
                self.logger.warning(f"Hook {hook_name} failed: {str(e)}")

    # Metrics and monitoring
    def _update_metrics(self, message: RichMessage, processing_time: float, success: bool):
        """Update processing metrics."""
        self.metrics.total_messages_processed += 1
        self.metrics.total_processing_time += processing_time
        self.metrics.average_processing_time = (
            self.metrics.total_processing_time / self.metrics.total_messages_processed
        )

        if not success:
            self.metrics.error_rate = (self.metrics.error_rate + 1) / self.metrics.total_messages_processed

        # Update stage-specific metrics
        stage = message.metadata.workflow_stage or "unknown"
        if stage not in self.metrics.by_stage:
            self.metrics.by_stage[stage] = {
                "count": 0,
                "total_time": 0.0,
                "average_time": 0.0,
                "average_quality": 0.0
            }

        stage_metrics = self.metrics.by_stage[stage]
        stage_metrics["count"] += 1
        stage_metrics["total_time"] += processing_time
        stage_metrics["average_time"] = stage_metrics["total_time"] / stage_metrics["count"]

        if message.metadata.quality_score:
            stage_metrics["average_quality"] = (
                (stage_metrics["average_quality"] * (stage_metrics["count"] - 1) +
                 message.metadata.quality_score) / stage_metrics["count"]
            )

        # Update agent-specific metrics
        agent = message.metadata.source_agent or "unknown"
        if agent not in self.metrics.by_agent:
            self.metrics.by_agent[agent] = {
                "count": 0,
                "total_time": 0.0,
                "average_time": 0.0
            }

        agent_metrics = self.metrics.by_agent[agent]
        agent_metrics["count"] += 1
        agent_metrics["total_time"] += processing_time
        agent_metrics["average_time"] = agent_metrics["total_time"] / agent_metrics["count"]

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive processing metrics."""
        metrics_data = {
            "processing_metrics": {
                "total_messages_processed": self.metrics.total_messages_processed,
                "total_processing_time": self.metrics.total_processing_time,
                "average_processing_time": self.metrics.average_processing_time,
                "cache_hit_rate": self.metrics.cache_hit_rate,
                "quality_improvement_rate": self.metrics.quality_improvement_rate,
                "error_rate": self.metrics.error_rate
            },
            "by_stage": self.metrics.by_stage,
            "by_agent": self.metrics.by_agent,
            "component_stats": {
                "message_processor": self.message_processor.get_processing_stats(),
                "rich_formatter": self.rich_formatter.get_formatting_stats(),
                "content_enhancer": self.content_enhancer.get_enhancement_stats(),
                "quality_analyzer": self.quality_analyzer.get_assessment_stats(),
                "message_router": self.message_router.get_routing_stats(),
                "message_cache": self.message_cache.get_stats(),
                "message_serializer": self.message_serializer.get_stats()
            }
        }

        return metrics_data

    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics = ProcessingMetrics()

        # Reset component metrics
        self.message_processor.reset_stats()
        self.rich_formatter.reset_stats()
        self.content_enhancer.reset_stats()
        self.quality_analyzer.reset_stats()
        self.message_router.reset_stats()
        self.message_cache.reset_stats()
        self.message_serializer.reset_stats()

    # Lifecycle management
    async def start(self):
        """Start integration services."""
        if self._running:
            return

        self._running = True

        # Start background services
        if self.config.enable_caching:
            await self.message_cache.start_background_cleanup()

        self.logger.info("Message processing integration started")

    async def stop(self):
        """Stop integration services."""
        if not self._running:
            return

        self._running = False

        # Stop background services
        await self.message_cache.stop_background_cleanup()

        # Close persistence connections
        self.message_persistence.close()

        self.logger.info("Message processing integration stopped")

    # Configuration management
    def update_config(self, new_config: Dict[str, Any]):
        """Update integration configuration."""
        self.config = self._create_integration_config({**self.config.__dict__, **new_config})

        # Update component configurations
        self.message_processor.config.update(new_config.get("processor", {}))
        self.rich_formatter.display_config = self.rich_formatter._create_display_config(
            new_config.get("formatter", {})
        )

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            "integration": self.config.__dict__,
            "components": {
                "processor": self.message_processor.config,
                "formatter": self.rich_formatter.config,
                "cache": self.message_cache.config,
                "router": self.message_router.config
            }
        }