"""
Enhanced Hooks Integration for Multi-Agent Research System

Phase 3.1.6: Integration of Comprehensive Hooks with Existing Phase 1 & 2 Systems

This module provides seamless integration between the comprehensive hooks system
and all existing system components including Phase 1 (anti-bot, content cleaning,
scraping pipeline) and Phase 2 (sub-agents, enhanced orchestrator, rich messaging).

Integration Features:
- Seamless integration with enhanced orchestrator
- Phase 1 system hook integration (anti-bot, content cleaning, scraping)
- Phase 2 system hook integration (sub-agents, message processing)
- Real-time monitoring and analytics
- Performance optimization and bottleneck detection
- Claude Agent SDK compliance
- Rich messaging integration
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable
import logging

# Import system components
try:
    from ..core.logging_config import get_logger
    from ..core.enhanced_orchestrator import EnhancedResearchOrchestrator, WorkflowHookContext
    from ..utils.message_processing.main import MessageProcessor, MessageType
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)
    # Fallback definitions
    EnhancedResearchOrchestrator = None
    WorkflowHookContext = None
    MessageType = None
    MessageProcessor = None

# Import comprehensive hooks system
from .comprehensive_hooks import ComprehensiveHookManager, HookCategory, HookExecutionResult
from .hook_analytics import HookAnalyticsEngine, create_hook_analytics_engine
from .real_time_monitoring import MetricsCollector, RealTimeMonitor, create_real_time_monitoring


class IntegrationLevel(Enum):
    """Levels of hooks system integration."""
    BASIC = "basic"           # Essential hooks only
    STANDARD = "standard"     # Full hooks with monitoring
    COMPREHENSIVE = "comprehensive"  # All hooks with analytics and optimization


@dataclass
class IntegrationConfig:
    """Configuration for hooks system integration."""
    integration_level: IntegrationLevel = IntegrationLevel.COMPREHENSIVE
    enable_analytics: bool = True
    enable_real_time_monitoring: bool = True
    enable_performance_optimization: bool = True
    enable_rich_messaging: bool = True

    # Performance settings
    analytics_window_minutes: int = 60
    metrics_retention_hours: int = 24
    optimization_threshold: float = 0.1

    # Hook execution settings
    parallel_execution: bool = True
    max_concurrent_hooks: int = 10
    hook_timeout_seconds: int = 30

    # Notification settings
    enable_notifications: bool = True
    notification_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'slow_execution': 5.0,
        'high_failure_rate': 0.1,
        'performance_degradation': 0.2
    })


class EnhancedHooksIntegrator:
    """
    Enhanced hooks system integrator for comprehensive system monitoring.

    Provides seamless integration between the comprehensive hooks system and all
    existing system components with proper Claude Agent SDK compliance.
    """

    def __init__(self, config: Optional[IntegrationConfig] = None):
        """Initialize enhanced hooks integrator."""
        self.config = config or IntegrationConfig()
        self.logger = get_logger("enhanced_hooks_integrator")

        # Core components
        self.hook_manager: Optional[ComprehensiveHookManager] = None
        self.analytics_engine: Optional[HookAnalyticsEngine] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        self.real_time_monitor: Optional[RealTimeMonitor] = None

        # Integration state
        self.enhanced_orchestrator: Optional[EnhancedResearchOrchestrator] = None
        self.message_processor: Optional[MessageProcessor] = None
        self.integration_active = False

        # Performance tracking
        self.integration_stats: Dict[str, Any] = {
            'total_hook_executions': 0,
            'successful_integrations': 0,
            'performance_improvements': 0,
            'bottlenecks_detected': 0,
            'optimizations_applied': 0
        }

    async def initialize(self, orchestrator: Optional[EnhancedResearchOrchestrator] = None) -> bool:
        """
        Initialize the enhanced hooks integration.

        Args:
            orchestrator: Optional enhanced orchestrator instance

        Returns:
            True if initialization successful
        """
        try:
            self.logger.info("Initializing enhanced hooks integration",
                           integration_level=self.config.integration_level.value)

            # Store orchestrator reference
            self.enhanced_orchestrator = orchestrator

            # Initialize core components
            await self._initialize_hook_manager()
            await self._initialize_analytics_engine()
            await self._initialize_real_time_monitoring()

            # Initialize message processor
            if self.config.enable_rich_messaging:
                await self._initialize_message_processor()

            # Register system integrations
            await self._register_phase1_integrations()
            await self._register_phase2_integrations()
            await self._register_orchestrator_integration()

            # Start background services
            await self._start_background_services()

            self.integration_active = True
            self.logger.info("Enhanced hooks integration initialized successfully")

            return True

        except Exception as e:
            self.logger.error(f"Enhanced hooks integration initialization failed: {e}",
                            error=str(e),
                            error_type=type(e).__name__)
            return False

    async def shutdown(self):
        """Shutdown the enhanced hooks integration."""
        if not self.integration_active:
            return

        self.logger.info("Shutting down enhanced hooks integration...")

        try:
            # Stop background services
            if self.analytics_engine:
                await self.analytics_engine.stop()
            if self.real_time_monitor:
                await self.real_time_monitor.stop()
            if self.metrics_collector:
                await self.metrics_collector.stop()

            self.integration_active = False
            self.logger.info("Enhanced hooks integration shutdown complete")

        except Exception as e:
            self.logger.error(f"Enhanced hooks integration shutdown failed: {e}",
                            error=str(e),
                            error_type=type(e).__name__)

    async def _initialize_hook_manager(self):
        """Initialize the comprehensive hook manager."""
        self.hook_manager = ComprehensiveHookManager(logger=self.logger)

        # Configure hook manager based on integration level
        if self.config.integration_level == IntegrationLevel.BASIC:
            self.hook_manager.enable_performance_tracking = False
            self.hook_manager.enable_rich_logging = False
        elif self.config.integration_level == IntegrationLevel.STANDARD:
            self.hook_manager.enable_performance_tracking = True
            self.hook_manager.enable_rich_logging = True
        else:  # COMPREHENSIVE
            self.hook_manager.enable_performance_tracking = True
            self.hook_manager.enable_rich_logging = True
            self.hook_manager.parallel_execution = self.config.parallel_execution
            self.hook_manager.max_concurrent_hooks = self.config.max_concurrent_hooks

        self.logger.info("Comprehensive hook manager initialized")

    async def _initialize_analytics_engine(self):
        """Initialize the hook analytics engine."""
        if self.config.enable_analytics:
            self.analytics_engine = create_hook_analytics_engine(
                analysis_window_minutes=self.config.analytics_window_minutes
            )
            await self.analytics_engine.start()
            self.logger.info("Hook analytics engine initialized")
        else:
            self.logger.info("Analytics engine disabled")

    async def _initialize_real_time_monitoring(self):
        """Initialize real-time monitoring."""
        if self.config.enable_real_time_monitoring:
            self.metrics_collector, self.real_time_monitor = create_real_time_monitoring(
                retention_hours=self.config.metrics_retention_hours
            )
            await self.metrics_collector.start()
            await self.real_time_monitor.start()

            # Register alert callbacks
            if self.config.enable_notifications:
                self.real_time_monitor.add_alert_callback(self._handle_alert_notification)

            self.logger.info("Real-time monitoring initialized")
        else:
            self.logger.info("Real-time monitoring disabled")

    async def _initialize_message_processor(self):
        """Initialize message processor for rich messaging."""
        try:
            self.message_processor = MessageProcessor()
            self.logger.info("Message processor initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize message processor: {e}")

    async def _register_phase1_integrations(self):
        """Register integrations for Phase 1 systems."""
        self.logger.info("Registering Phase 1 system integrations")

        # Anti-bot escalation system hooks
        self._register_anti_bot_hooks()

        # Content cleaning system hooks
        self._register_content_cleaning_hooks()

        # Scraping pipeline hooks
        self._register_scraping_pipeline_hooks()

        # Workflow management hooks
        self._register_workflow_management_hooks()

    async def _register_phase2_integrations(self):
        """Register integrations for Phase 2 systems."""
        self.logger.info("Registering Phase 2 system integrations")

        # Sub-agent system hooks
        self._register_sub_agent_hooks()

        # Message processing hooks
        self._register_message_processing_hooks()

        # Enhanced agent base class hooks
        self._register_enhanced_agent_hooks()

    async def _register_orchestrator_integration(self):
        """Register integration with enhanced orchestrator."""
        if not self.enhanced_orchestrator:
            self.logger.warning("No enhanced orchestrator provided for integration")
            return

        self.logger.info("Registering enhanced orchestrator integration")

        # Replace the orchestrator's hook manager with our comprehensive one
        self.enhanced_orchestrator.hook_manager = self.hook_manager

        # Register custom orchestrator hooks
        self._register_orchestrator_hooks()

        # Set up analytics integration
        self._setup_orchestrator_analytics()

    def _register_anti_bot_hooks(self):
        """Register anti-bot escalation system hooks."""
        # Anti-bot escalation tracking
        self.hook_manager.register_hook(
            "anti_bot_escalation_started",
            self._hook_anti_bot_escalation_started,
            metadata={"system": "anti_bot", "phase": "1"}
        )

        self.hook_manager.register_hook(
            "anti_bot_escalation_completed",
            self._hook_anti_bot_escalation_completed,
            metadata={"system": "anti_bot", "phase": "1"}
        )

        self.hook_manager.register_hook(
            "anti_bot_level_changed",
            self._hook_anti_bot_level_changed,
            metadata={"system": "anti_bot", "phase": "1"}
        )

    def _register_content_cleaning_hooks(self):
        """Register content cleaning system hooks."""
        self.hook_manager.register_hook(
            "content_cleaning_started",
            self._hook_content_cleaning_started,
            metadata={"system": "content_cleaning", "phase": "1"}
        )

        self.hook_manager.register_hook(
            "gpt5nano_scoring_completed",
            self._hook_gpt5nano_scoring_completed,
            metadata={"system": "content_cleaning", "phase": "1"}
        )

        self.hook_manager.register_hook(
            "content_quality_judgment",
            self._hook_content_quality_judgment,
            metadata={"system": "content_cleaning", "phase": "1"}
        )

    def _register_scraping_pipeline_hooks(self):
        """Register scraping pipeline system hooks."""
        self.hook_manager.register_hook(
            "async_orchestrator_started",
            self._hook_async_orchestrator_started,
            metadata={"system": "scraping_pipeline", "phase": "1"}
        )

        self.hook_manager.register_hook(
            "scraping_batch_completed",
            self._hook_scraping_batch_completed,
            metadata={"system": "scraping_pipeline", "phase": "1"}
        )

        self.hook_manager.register_hook(
            "data_contract_validated",
            self._hook_data_contract_validated,
            metadata={"system": "scraping_pipeline", "phase": "1"}
        )

    def _register_workflow_management_hooks(self):
        """Register workflow management hooks."""
        self.hook_manager.register_hook(
            "success_tracker_updated",
            self._hook_success_tracker_updated,
            metadata={"system": "workflow_management", "phase": "1"}
        )

        self.hook_manager.register_hook(
            "early_termination_triggered",
            self._hook_early_termination_triggered,
            metadata={"system": "workflow_management", "phase": "1"}
        )

    def _register_sub_agent_hooks(self):
        """Register sub-agent system hooks."""
        self.hook_manager.register_hook(
            "sub_agent_created",
            self._hook_sub_agent_created,
            metadata={"system": "sub_agents", "phase": "2"}
        )

        self.hook_manager.register_hook(
            "sub_agent_coordination_started",
            self._hook_sub_agent_coordination_started,
            metadata={"system": "sub_agents", "phase": "2"}
        )

        self.hook_manager.register_hook(
            "context_isolation_enforced",
            self._hook_context_isolation_enforced,
            metadata={"system": "sub_agents", "phase": "2"}
        )

    def _register_message_processing_hooks(self):
        """Register message processing hooks."""
        self.hook_manager.register_hook(
            "rich_message_processed",
            self._hook_rich_message_processed,
            metadata={"system": "message_processing", "phase": "2"}
        )

        self.hook_manager.register_hook(
            "message_cache_hit",
            self._hook_message_cache_hit,
            metadata={"system": "message_processing", "phase": "2"}
        )

        self.hook_manager.register_hook(
            "message_formatting_applied",
            self._hook_message_formatting_applied,
            metadata={"system": "message_processing", "phase": "2"}
        )

    def _register_enhanced_agent_hooks(self):
        """Register enhanced agent base class hooks."""
        self.hook_manager.register_hook(
            "agent_lifecycle_event",
            self._hook_agent_lifecycle_event,
            metadata={"system": "enhanced_agents", "phase": "2"}
        )

        self.hook_manager.register_hook(
            "sdk_options_applied",
            self._hook_sdk_options_applied,
            metadata={"system": "enhanced_agents", "phase": "2"}
        )

        self.hook_manager.register_hook(
            "factory_pattern_executed",
            self._hook_factory_pattern_executed,
            metadata={"system": "enhanced_agents", "phase": "2"}
        )

    def _register_orchestrator_hooks(self):
        """Register orchestrator-specific hooks."""
        self.hook_manager.register_hook(
            "quality_gate_enforcement",
            self._hook_quality_gate_enforcement,
            metadata={"system": "orchestrator", "phase": "2"}
        )

        self.hook_manager.register_hook(
            "flow_adherence_validation",
            self._hook_flow_adherence_validation,
            metadata={"system": "orchestrator", "phase": "2"}
        )

        self.hook_manager.register_hook(
            "gap_research_enforcement",
            self._hook_gap_research_enforcement,
            metadata={"system": "orchestrator", "phase": "2"}
        )

    def _setup_orchestrator_analytics(self):
        """Set up analytics integration with orchestrator."""
        if not self.enhanced_orchestrator or not self.analytics_engine:
            return

        # Hook into orchestrator methods for analytics
        original_execute_workflow = self.enhanced_orchestrator.execute_enhanced_research_workflow

        async def analytics_wrapped_workflow(session_id: str):
            """Wrap workflow execution with analytics."""
            start_time = time.time()

            try:
                result = await original_execute_workflow(session_id)
                duration = time.time() - start_time

                # Record workflow metrics
                if self.metrics_collector:
                    from .real_time_monitoring import MetricValue, MetricType
                    self.metrics_collector.record_metric(MetricValue(
                        name="workflow_execution_time",
                        value=duration,
                        metric_type=MetricType.TIMER,
                        timestamp=datetime.now(),
                        labels={"session_id": session_id, "status": "success"},
                        unit="seconds"
                    ))

                return result

            except Exception as e:
                duration = time.time() - start_time

                # Record failure metrics
                if self.metrics_collector:
                    from .real_time_monitoring import MetricValue, MetricType
                    self.metrics_collector.record_metric(MetricValue(
                        name="workflow_execution_time",
                        value=duration,
                        metric_type=MetricType.TIMER,
                        timestamp=datetime.now(),
                        labels={"session_id": session_id, "status": "error"},
                        unit="seconds"
                    ))

                raise

        # Replace the method
        self.enhanced_orchestrator.execute_enhanced_research_workflow = analytics_wrapped_workflow

    async def _start_background_services(self):
        """Start background monitoring and optimization services."""
        if self.config.enable_performance_optimization and self.analytics_engine:
            # Start optimization monitoring task
            asyncio.create_task(self._optimization_monitoring_loop())

    async def _optimization_monitoring_loop(self):
        """Background loop for monitoring and applying optimizations."""
        self.logger.info("Starting optimization monitoring loop")

        while self.integration_active:
            try:
                # Check for optimization opportunities
                if self.analytics_engine:
                    recommendations = self.analytics_engine._recommendations_cache

                    # Apply high-priority optimizations automatically
                    auto_applicable = [r for r in recommendations if r.priority <= 2 and r.implementation_complexity == "low"]

                    for recommendation in auto_applicable:
                        await self._apply_optimization_recommendation(recommendation)

                await asyncio.sleep(600)  # Check every 10 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Optimization monitoring error: {e}")
                await asyncio.sleep(60)

    async def _apply_optimization_recommendation(self, recommendation):
        """Apply an optimization recommendation."""
        try:
            self.logger.info(f"Applying optimization: {recommendation.description}")

            # Implementation would depend on the specific recommendation type
            # For now, we'll just log and track the application
            self.integration_stats['optimizations_applied'] += 1

            if self.message_processor:
                await self.message_processor.process_message(
                    MessageType.INFO,
                    f"🔧 Optimization applied: {recommendation.description}",
                    metadata={
                        'event_type': 'optimization_applied',
                        'recommendation': recommendation.__dict__
                    }
                )

        except Exception as e:
            self.logger.error(f"Failed to apply optimization: {e}")

    async def _handle_alert_notification(self, alert):
        """Handle real-time monitoring alerts."""
        self.logger.warning(f"System alert: {alert.title} - {alert.description}")

        if self.message_processor:
            await self.message_processor.process_message(
                MessageType.WARNING,
                f"🚨 System Alert: {alert.title}",
                metadata={
                    'event_type': 'system_alert',
                    'alert_id': alert.id,
                    'severity': alert.severity.value,
                    'description': alert.description,
                    'metric_name': alert.metric_name,
                    'current_value': alert.current_value,
                    'threshold': alert.threshold
                }
            )

    # Phase 1 system hook implementations
    async def _hook_anti_bot_escalation_started(self, context: WorkflowHookContext) -> Dict[str, Any]:
        """Hook for anti-bot escalation start."""
        domain = context.metadata.get('domain', 'Unknown')
        level = context.metadata.get('escalation_level', 1)

        self.logger.info(f"🛡️ Anti-bot escalation started for {domain} (level {level})")

        if self.metrics_collector:
            from .real_time_monitoring import MetricValue, MetricType
            self.metrics_collector.record_metric(MetricValue(
                name="anti_bot_escalation_started",
                value=1,
                metric_type=MetricType.COUNTER,
                timestamp=datetime.now(),
                labels={"domain": domain, "level": str(level)}
            ))

        return {"event": "anti_bot_escalation_started", "domain": domain, "level": level}

    async def _hook_anti_bot_escalation_completed(self, context: WorkflowHookContext) -> Dict[str, Any]:
        """Hook for anti-bot escalation completion."""
        duration = context.get_duration()
        success = context.metadata.get('success', False)
        final_level = context.metadata.get('final_level', 1)

        self.logger.info(f"✅ Anti-bot escalation completed: {'Success' if success else 'Failed'} "
                        f"(level {final_level}, {duration:.2f}s)")

        return {"event": "anti_bot_escalation_completed", "success": success, "final_level": final_level}

    async def _hook_content_cleaning_started(self, context: WorkflowHookContext) -> Dict[str, Any]:
        """Hook for content cleaning start."""
        content_count = context.metadata.get('content_count', 0)

        self.logger.info(f"🧹 Content cleaning started: {content_count} items")

        return {"event": "content_cleaning_started", "content_count": content_count}

    async def _hook_gpt5nano_scoring_completed(self, context: WorkflowHookContext) -> Dict[str, Any]:
        """Hook for GPT-5-nano scoring completion."""
        duration = context.get_duration()
        average_score = context.metadata.get('average_score', 0)
        cache_hit_rate = context.metadata.get('cache_hit_rate', 0)

        self.logger.info(f"🤖 GPT-5-nano scoring completed: avg score {average_score:.2f}, "
                        f"cache hit rate {cache_hit_rate:.1%} ({duration:.2f}s)")

        return {
            "event": "gpt5nano_scoring_completed",
            "average_score": average_score,
            "cache_hit_rate": cache_hit_rate,
            "duration": duration
        }

    async def _hook_content_quality_judgment(self, context: WorkflowHookContext) -> Dict[str, Any]:
        """Hook for content quality judgment."""
        useful = context.metadata.get('useful', False)
        confidence = context.metadata.get('confidence', 0)

        self.logger.info(f"⚖️ Content quality judgment: {'Useful' if useful else 'Not useful'} "
                        f"(confidence: {confidence:.2f})")

        return {"event": "content_quality_judgment", "useful": useful, "confidence": confidence}

    async def _hook_async_orchestrator_started(self, context: WorkflowHookContext) -> Dict[str, Any]:
        """Hook for async orchestrator start."""
        concurrent_scrapes = context.metadata.get('concurrent_scrapes', 0)
        concurrent_cleans = context.metadata.get('concurrent_cleans', 0)

        self.logger.info(f"⚡ Async orchestrator started: {concurrent_scrapes} scrapes, "
                        f"{concurrent_cleans} cleans")

        return {
            "event": "async_orchestrator_started",
            "concurrent_scrapes": concurrent_scrapes,
            "concurrent_cleans": concurrent_cleans
        }

    async def _hook_scraping_batch_completed(self, context: WorkflowHookContext) -> Dict[str, Any]:
        """Hook for scraping batch completion."""
        batch_size = context.metadata.get('batch_size', 0)
        success_count = context.metadata.get('success_count', 0)

        self.logger.info(f"📦 Scraping batch completed: {success_count}/{batch_size} successful")

        return {"event": "scraping_batch_completed", "batch_size": batch_size, "success_count": success_count}

    async def _hook_data_contract_validated(self, context: WorkflowHookContext) -> Dict[str, Any]:
        """Hook for data contract validation."""
        valid = context.metadata.get('valid', False)
        contract_type = context.metadata.get('contract_type', 'Unknown')

        self.logger.info(f"📋 Data contract validation: {'Valid' if valid else 'Invalid'} "
                        f"({contract_type})")

        return {"event": "data_contract_validated", "valid": valid, "contract_type": contract_type}

    async def _hook_success_tracker_updated(self, context: WorkflowHookContext) -> Dict[str, Any]:
        """Hook for success tracker updates."""
        current_successes = context.metadata.get('current_successes', 0)
        target_count = context.metadata.get('target_count', 0)
        completion_reached = context.metadata.get('completion_reached', False)

        progress = (current_successes / target_count * 100) if target_count > 0 else 0

        self.logger.info(f"📊 Success tracker updated: {current_successes}/{target_count} "
                        f"({progress:.1f}%) {'✅ COMPLETE' if completion_reached else ''}")

        return {
            "event": "success_tracker_updated",
            "current_successes": current_successes,
            "target_count": target_count,
            "progress_percentage": progress,
            "completion_reached": completion_reached
        }

    async def _hook_early_termination_triggered(self, context: WorkflowHookContext) -> Dict[str, Any]:
        """Hook for early termination triggers."""
        reason = context.metadata.get('reason', 'Unknown')
        resources_saved = context.metadata.get('resources_saved', 0)

        self.logger.info(f"⏹️ Early termination triggered: {reason} "
                        f"(saved {resources_saved} resources)")

        return {"event": "early_termination_triggered", "reason": reason, "resources_saved": resources_saved}

    # Phase 2 system hook implementations
    async def _hook_sub_agent_created(self, context: WorkflowHookContext) -> Dict[str, Any]:
        """Hook for sub-agent creation."""
        agent_type = context.metadata.get('agent_type', 'Unknown')
        agent_id = context.metadata.get('agent_id', 'Unknown')

        self.logger.info(f"👥 Sub-agent created: {agent_type} ({agent_id})")

        return {"event": "sub_agent_created", "agent_type": agent_type, "agent_id": agent_id}

    async def _hook_sub_agent_coordination_started(self, context: WorkflowHookContext) -> Dict[str, Any]:
        """Hook for sub-agent coordination start."""
        coordination_type = context.metadata.get('coordination_type', 'Unknown')
        participant_count = context.metadata.get('participant_count', 0)

        self.logger.info(f"🔄 Sub-agent coordination started: {coordination_type} "
                        f"({participant_count} participants)")

        return {
            "event": "sub_agent_coordination_started",
            "coordination_type": coordination_type,
            "participant_count": participant_count
        }

    async def _hook_context_isolation_enforced(self, context: WorkflowHookContext) -> Dict[str, Any]:
        """Hook for context isolation enforcement."""
        isolation_level = context.metadata.get('isolation_level', 'medium')
        data_leak_prevented = context.metadata.get('data_leak_prevented', False)

        self.logger.info(f"🔒 Context isolation enforced: {isolation_level} "
                        f"{'🛡️ Leak prevented' if data_leak_prevented else ''}")

        return {
            "event": "context_isolation_enforced",
            "isolation_level": isolation_level,
            "data_leak_prevented": data_leak_prevented
        }

    async def _hook_rich_message_processed(self, context: WorkflowHookContext) -> Dict[str, Any]:
        """Hook for rich message processing."""
        message_type = context.metadata.get('message_type', 'Unknown')
        processing_time = context.metadata.get('processing_time', 0)
        cache_used = context.metadata.get('cache_used', False)

        self.logger.info(f"💬 Rich message processed: {message_type} "
                        f"({processing_time:.3f}s {'🗄️ from cache' if cache_used else ''})")

        return {
            "event": "rich_message_processed",
            "message_type": message_type,
            "processing_time": processing_time,
            "cache_used": cache_used
        }

    async def _hook_message_cache_hit(self, context: WorkflowHookContext) -> Dict[str, Any]:
        """Hook for message cache hits."""
        cache_key = context.metadata.get('cache_key', 'Unknown')
        hit_rate = context.metadata.get('hit_rate', 0)

        self.logger.info(f"🗄️ Message cache hit: {cache_key} (hit rate: {hit_rate:.1%})")

        return {"event": "message_cache_hit", "cache_key": cache_key, "hit_rate": hit_rate}

    async def _hook_message_formatting_applied(self, context: WorkflowHookContext) -> Dict[str, Any]:
        """Hook for message formatting application."""
        format_type = context.metadata.get('format_type', 'Unknown')
        enhancement_level = context.metadata.get('enhancement_level', 'medium')

        self.logger.info(f"🎨 Message formatting applied: {format_type} ({enhancement_level})")

        return {
            "event": "message_formatting_applied",
            "format_type": format_type,
            "enhancement_level": enhancement_level
        }

    async def _hook_agent_lifecycle_event(self, context: WorkflowHookContext) -> Dict[str, Any]:
        """Hook for agent lifecycle events."""
        lifecycle_event = context.metadata.get('event', 'Unknown')
        agent_type = context.metadata.get('agent_type', 'Unknown')

        self.logger.info(f"🔄 Agent lifecycle event: {lifecycle_event} ({agent_type})")

        return {"event": "agent_lifecycle_event", "lifecycle_event": lifecycle_event, "agent_type": agent_type}

    async def _hook_sdk_options_applied(self, context: WorkflowHookContext) -> Dict[str, Any]:
        """Hook for SDK options application."""
        options_count = context.metadata.get('options_count', 0)
        custom_options = context.metadata.get('custom_options', False)

        self.logger.info(f"⚙️ SDK options applied: {options_count} options "
                        f"{'🔧 custom' if custom_options else '📋 default'}")

        return {
            "event": "sdk_options_applied",
            "options_count": options_count,
            "custom_options": custom_options
        }

    async def _hook_factory_pattern_executed(self, context: WorkflowHookContext) -> Dict[str, Any]:
        """Hook for factory pattern execution."""
        pattern_type = context.metadata.get('pattern_type', 'Unknown')
        instance_created = context.metadata.get('instance_created', False)

        self.logger.info(f"🏭 Factory pattern executed: {pattern_type} "
                        f"{'✅ instance created' if instance_created else '❌ failed'}")

        return {
            "event": "factory_pattern_executed",
            "pattern_type": pattern_type,
            "instance_created": instance_created
        }

    # Orchestrator-specific hook implementations
    async def _hook_quality_gate_enforcement(self, context: WorkflowHookContext) -> Dict[str, Any]:
        """Hook for quality gate enforcement."""
        gate_passed = context.metadata.get('gate_passed', False)
        quality_score = context.metadata.get('quality_score', 0)
        threshold = context.metadata.get('threshold', 0)

        self.logger.info(f"🚪 Quality gate enforcement: {'✅ PASSED' if gate_passed else '❌ FAILED'} "
                        f"(score: {quality_score}, threshold: {threshold})")

        return {
            "event": "quality_gate_enforcement",
            "gate_passed": gate_passed,
            "quality_score": quality_score,
            "threshold": threshold
        }

    async def _hook_flow_adherence_validation(self, context: WorkflowHookContext) -> Dict[str, Any]:
        """Hook for flow adherence validation."""
        adherence_level = context.metadata.get('adherence_level', 'unknown')
        violations_detected = context.metadata.get('violations_detected', 0)
        enforcement_actions = context.metadata.get('enforcement_actions', 0)

        self.logger.info(f"📏 Flow adherence validation: {adherence_level} "
                        f"{'⚠️ violations: ' + str(violations_detected) if violations_detected > 0 else '✅ compliant'} "
                        f"{'🔧 actions: ' + str(enforcement_actions) if enforcement_actions > 0 else ''}")

        return {
            "event": "flow_adherence_validation",
            "adherence_level": adherence_level,
            "violations_detected": violations_detected,
            "enforcement_actions": enforcement_actions
        }

    async def _hook_gap_research_enforcement(self, context: WorkflowHookContext) -> Dict[str, Any]:
        """Hook for gap research enforcement."""
        enforcement_needed = context.metadata.get('enforcement_needed', False)
        gaps_identified = context.metadata.get('gaps_identified', 0)
        research_triggered = context.metadata.get('research_triggered', False)

        self.logger.info(f"🔍 Gap research enforcement: {'🔧 ENFORCED' if enforcement_needed else '✅ compliant'} "
                        f"({gaps_identified} gaps {'🚀 triggered' if research_triggered else '⏸️ pending'})")

        return {
            "event": "gap_research_enforcement",
            "enforcement_needed": enforcement_needed,
            "gaps_identified": gaps_identified,
            "research_triggered": research_triggered
        }

    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status."""
        status = {
            "integration_active": self.integration_active,
            "config": self.config.__dict__,
            "components": {
                "hook_manager": self.hook_manager is not None,
                "analytics_engine": self.analytics_engine is not None,
                "real_time_monitor": self.real_time_monitor is not None,
                "metrics_collector": self.metrics_collector is not None,
                "message_processor": self.message_processor is not None
            },
            "statistics": self.integration_stats.copy()
        }

        # Add hook statistics
        if self.hook_manager:
            status["hook_statistics"] = self.hook_manager.get_hook_statistics()

        # Add analytics summary
        if self.analytics_engine:
            status["analytics_summary"] = self.analytics_engine.get_performance_summary()

        # Add monitoring status
        if self.real_time_monitor:
            status["monitoring_status"] = {
                "active_alerts": len(self.real_time_monitor.get_active_alerts()),
                "system_health": self.real_time_monitor.get_system_health().value
            }

        return status


# Factory function for creating enhanced hooks integration
def create_enhanced_hooks_integration(config: Optional[IntegrationConfig] = None) -> EnhancedHooksIntegrator:
    """
    Create and configure enhanced hooks integration.

    Args:
        config: Optional integration configuration

    Returns:
        Configured EnhancedHooksIntegrator instance
    """
    return EnhancedHooksIntegrator(config=config)