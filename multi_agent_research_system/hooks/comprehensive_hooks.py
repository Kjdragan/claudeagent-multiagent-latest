"""
Comprehensive Hooks System for Multi-Agent Research System

Phase 3.1: Enhanced Hooks System with Claude Agent SDK Integration

This module provides comprehensive hooks for all major system operations including:
- Research operations (searching, scraping, content cleaning)
- Content processing and analysis
- Quality management and enhancement
- Editorial workflows and gap research
- System performance and monitoring
- Agent coordination and communication
- Session management and lifecycle

Designed to integrate seamlessly with the enhanced orchestrator and follow Claude Agent SDK patterns.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
import logging

# Import SDK components when available
try:
    from claude_agent_sdk.types import HookContext as SDKHookContext
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    SDKHookContext = None

# Import system components
try:
    from ..core.logging_config import get_logger
    from ..utils.message_processing.main import MessageProcessor, MessageType
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)
    MessageType = None
    MessageProcessor = None

# Import hook context from enhanced orchestrator
try:
    from ..core.enhanced_orchestrator import WorkflowHookContext
except ImportError:
    # Fallback definition
    @dataclass
    class WorkflowHookContext:
        session_id: str
        workflow_stage: str
        agent_name: str
        operation: str
        start_time: datetime
        metadata: Dict[str, Any] = field(default_factory=dict)
        error_context: Optional[str] = None
        quality_metrics: Dict[str, Any] = field(default_factory=dict)

        def get_duration(self) -> float:
            return (datetime.now() - self.start_time).total_seconds()


class HookCategory(Enum):
    """Categories of hooks for organization."""
    RESEARCH_OPERATIONS = "research_operations"
    CONTENT_PROCESSING = "content_processing"
    QUALITY_MANAGEMENT = "quality_management"
    EDITORIAL_WORKFLOW = "editorial_workflow"
    AGENT_COORDINATION = "agent_coordination"
    SYSTEM_MONITORING = "system_monitoring"
    SESSION_MANAGEMENT = "session_management"
    PERFORMANCE_TRACKING = "performance_tracking"
    ERROR_HANDLING = "error_handling"
    MCP_INTEGRATION = "mcp_integration"


class HookPriority(Enum):
    """Hook execution priorities."""
    LOWEST = 0
    LOW = 25
    NORMAL = 50
    HIGH = 75
    HIGHEST = 100


@dataclass
class HookExecutionResult:
    """Result of hook execution with comprehensive tracking."""
    hook_name: str
    hook_category: HookCategory
    hook_type: str
    success: bool
    execution_time: float
    result_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class ComprehensiveHookManager:
    """
    Comprehensive hook management system with Claude Agent SDK integration.

    Provides extensive monitoring and coordination capabilities for all system operations
    with proper SDK integration and rich messaging support.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize comprehensive hook manager."""
        self.logger = logger or get_logger("comprehensive_hook_manager")

        # Enhanced hook registry with categorization
        self.hooks: Dict[str, List[Dict[str, Any]]] = {}
        self.hook_categories: Dict[str, HookCategory] = {}

        # Performance tracking
        self.execution_stats: Dict[str, Dict[str, Any]] = {}
        self.performance_history: Dict[str, List[float]] = {}

        # Message processing for rich output
        self.message_processor: Optional[MessageProcessor] = None
        if MessageProcessor:
            try:
                self.message_processor = MessageProcessor()
            except Exception as e:
                self.logger.warning(f"Failed to initialize message processor: {e}")

        # Hook configuration
        self.enable_performance_tracking = True
        self.enable_rich_logging = True
        self.parallel_execution = True
        self.max_concurrent_hooks = 10

        # Initialize default hooks
        self._initialize_hook_categories()
        self._register_default_hooks()

    def _initialize_hook_categories(self):
        """Initialize hook categories and event types."""
        hook_definitions = {
            # Research Operations
            "search_initiated": HookCategory.RESEARCH_OPERATIONS,
            "search_completed": HookCategory.RESEARCH_OPERATIONS,
            "scraping_started": HookCategory.RESEARCH_OPERATIONS,
            "scraping_progress": HookCategory.RESEARCH_OPERATIONS,
            "scraping_completed": HookCategory.RESEARCH_OPERATIONS,
            "content_cleaning_started": HookCategory.RESEARCH_OPERATIONS,
            "content_cleaning_completed": HookCategory.RESEARCH_OPERATIONS,
            "research_success_tracking": HookCategory.RESEARCH_OPERATIONS,

            # Content Processing
            "content_analysis_started": HookCategory.CONTENT_PROCESSING,
            "content_analysis_completed": HookCategory.CONTENT_PROCESSING,
            "content_integration_started": HookCategory.CONTENT_PROCESSING,
            "content_integration_completed": HookCategory.CONTENT_PROCESSING,
            "content_enhancement_started": HookCategory.CONTENT_PROCESSING,
            "content_enhancement_completed": HookCategory.CONTENT_PROCESSING,

            # Quality Management
            "quality_assessment_started": HookCategory.QUALITY_MANAGEMENT,
            "quality_assessment_completed": HookCategory.QUALITY_MANAGEMENT,
            "quality_gate_evaluation": HookCategory.QUALITY_MANAGEMENT,
            "quality_enhancement_initiated": HookCategory.QUALITY_MANAGEMENT,
            "quality_enhancement_completed": HookCategory.QUALITY_MANAGEMENT,
            "progressive_enhancement_cycle": HookCategory.QUALITY_MANAGEMENT,

            # Editorial Workflow
            "editorial_review_started": HookCategory.EDITORIAL_WORKFLOW,
            "editorial_review_completed": HookCategory.EDITORIAL_WORKFLOW,
            "gap_research_decision": HookCategory.EDITORIAL_WORKFLOW,
            "gap_research_initiated": HookCategory.EDITORIAL_WORKFLOW,
            "gap_research_completed": HookCategory.EDITORIAL_WORKFLOW,
            "editorial_recommendations_generated": HookCategory.EDITORIAL_WORKFLOW,

            # Agent Coordination
            "agent_communication_started": HookCategory.AGENT_COORDINATION,
            "agent_communication_completed": HookCategory.AGENT_COORDINATION,
            "agent_handoff_initiated": HookCategory.AGENT_COORDINATION,
            "agent_handoff_completed": HookCategory.AGENT_COORDINATION,
            "sub_agent_coordination": HookCategory.AGENT_COORDINATION,
            "agent_state_change": HookCategory.AGENT_COORDINATION,

            # System Monitoring
            "system_health_check": HookCategory.SYSTEM_MONITORING,
            "performance_metrics_collection": HookCategory.SYSTEM_MONITORING,
            "resource_monitoring": HookCategory.SYSTEM_MONITORING,
            "system_alert_triggered": HookCategory.SYSTEM_MONITORING,
            "bottleneck_detected": HookCategory.SYSTEM_MONITORING,

            # Session Management
            "session_created": HookCategory.SESSION_MANAGEMENT,
            "session_initialized": HookCategory.SESSION_MANAGEMENT,
            "session_updated": HookCategory.SESSION_MANAGEMENT,
            "session_completed": HookCategory.SESSION_MANAGEMENT,
            "session_archived": HookCategory.SESSION_MANAGEMENT,
            "sub_session_created": HookCategory.SESSION_MANAGEMENT,

            # Performance Tracking
            "performance_baseline_established": HookCategory.PERFORMANCE_TRACKING,
            "performance_degradation_detected": HookCategory.PERFORMANCE_TRACKING,
            "performance_optimization_applied": HookCategory.PERFORMANCE_TRACKING,
            "execution_time_tracked": HookCategory.PERFORMANCE_TRACKING,

            # Error Handling
            "error_detected": HookCategory.ERROR_HANDLING,
            "error_recovery_initiated": HookCategory.ERROR_HANDLING,
            "error_recovery_completed": HookCategory.ERROR_HANDLING,
            "fallback_activated": HookCategory.ERROR_HANDLING,
            "circuit_breaker_triggered": HookCategory.ERROR_HANDLING,

            # MCP Integration
            "mcp_tool_execution": HookCategory.MCP_INTEGRATION,
            "mcp_message_processing": HookCategory.MCP_INTEGRATION,
            "mcp_session_management": HookCategory.MCP_INTEGRATION,
            "sdk_hook_execution": HookCategory.MCP_INTEGRATION
        }

        for hook_type, category in hook_definitions.items():
            self.hooks[hook_type] = []
            self.hook_categories[hook_type] = category
            self.execution_stats[hook_type] = {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "total_duration": 0.0,
                "average_duration": 0.0,
                "last_execution": None
            }
            self.performance_history[hook_type] = []

    def register_hook(self,
                     hook_type: str,
                     hook_func: Callable,
                     priority: HookPriority = HookPriority.NORMAL,
                     enabled: bool = True,
                     metadata: Optional[Dict[str, Any]] = None):
        """
        Register a hook function with comprehensive configuration.

        Args:
            hook_type: Type of hook event
            hook_func: Hook function to register
            priority: Execution priority
            enabled: Whether hook is enabled
            metadata: Additional metadata for the hook
        """
        if hook_type not in self.hooks:
            self.logger.warning(f"Unknown hook type: {hook_type}")
            self.hooks[hook_type] = []
            self.hook_categories[hook_type] = HookCategory.SYSTEM_MONITORING

        hook_info = {
            "function": hook_func,
            "name": hook_func.__name__,
            "priority": priority.value,
            "enabled": enabled,
            "metadata": metadata or {},
            "registered_at": datetime.now()
        }

        self.hooks[hook_type].append(hook_info)

        # Sort by priority
        self.hooks[hook_type].sort(key=lambda x: x["priority"], reverse=True)

        category = self.hook_categories[hook_type]
        self.logger.info(f"Registered {category.value} hook: {hook_func.__name__} for {hook_type} "
                        f"(priority: {priority.name})")

    async def execute_hooks(self,
                          hook_type: str,
                          context: WorkflowHookContext,
                          category: Optional[HookCategory] = None) -> List[HookExecutionResult]:
        """
        Execute hooks for a given type with comprehensive tracking.

        Args:
            hook_type: Type of hook to execute
            context: Hook context information
            category: Optional category override

        Returns:
            List of hook execution results
        """
        if hook_type not in self.hooks:
            self.logger.debug(f"No hooks registered for: {hook_type}")
            return []

        start_time = time.time()
        hook_category = category or self.hook_categories.get(hook_type, HookCategory.SYSTEM_MONITORING)

        # Get enabled hooks
        enabled_hooks = [h for h in self.hooks[hook_type] if h["enabled"]]
        if not enabled_hooks:
            return []

        self.logger.debug(f"Executing {len(enabled_hooks)} hooks for {hook_type} "
                         f"(category: {hook_category.value})")

        # Execute hooks
        if self.parallel_execution and len(enabled_hooks) > 1:
            results = await self._execute_hooks_parallel(enabled_hooks, hook_type, hook_category, context)
        else:
            results = await self._execute_hooks_sequential(enabled_hooks, hook_type, hook_category, context)

        # Update statistics
        total_duration = time.time() - start_time
        self._update_execution_stats(hook_type, results, total_duration)

        # Log rich message if available
        if self.enable_rich_logging and self.message_processor:
            await self._log_hook_execution(hook_type, results, total_duration)

        return results

    async def _execute_hooks_sequential(self,
                                      hooks: List[Dict[str, Any]],
                                      hook_type: str,
                                      category: HookCategory,
                                      context: WorkflowHookContext) -> List[HookExecutionResult]:
        """Execute hooks sequentially."""
        results = []

        for hook_info in hooks:
            result = await self._execute_single_hook(hook_info, hook_type, category, context)
            results.append(result)

        return results

    async def _execute_hooks_parallel(self,
                                     hooks: List[Dict[str, Any]],
                                     hook_type: str,
                                     category: HookCategory,
                                     context: WorkflowHookContext) -> List[HookExecutionResult]:
        """Execute hooks in parallel with concurrency control."""
        semaphore = asyncio.Semaphore(self.max_concurrent_hooks)

        async def execute_with_semaphore(hook_info):
            async with semaphore:
                return await self._execute_single_hook(hook_info, hook_type, category, context)

        tasks = [execute_with_semaphore(hook_info) for hook_info in hooks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(HookExecutionResult(
                    hook_name=hooks[i]["name"],
                    hook_category=category,
                    hook_type=hook_type,
                    success=False,
                    execution_time=0.0,
                    error_message=str(result)
                ))
            else:
                processed_results.append(result)

        return processed_results

    async def _execute_single_hook(self,
                                 hook_info: Dict[str, Any],
                                 hook_type: str,
                                 category: HookCategory,
                                 context: WorkflowHookContext) -> HookExecutionResult:
        """Execute a single hook with comprehensive error handling."""
        hook_func = hook_info["function"]
        hook_name = hook_info["name"]

        start_time = time.time()

        try:
            # Execute the hook
            if asyncio.iscoroutinefunction(hook_func):
                result_data = await hook_func(context)
            else:
                result_data = hook_func(context)

            execution_time = time.time() - start_time

            # Track performance
            if self.enable_performance_tracking:
                self.performance_history[hook_type].append(execution_time)
                # Keep only last 100 executions
                if len(self.performance_history[hook_type]) > 100:
                    self.performance_history[hook_type].pop(0)

            return HookExecutionResult(
                hook_name=hook_name,
                hook_category=category,
                hook_type=hook_type,
                success=True,
                execution_time=execution_time,
                result_data=result_data or {},
                metadata=hook_info["metadata"]
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Hook {hook_name} failed: {str(e)}",
                            hook_type=hook_type,
                            error_type=type(e).__name__)

            return HookExecutionResult(
                hook_name=hook_name,
                hook_category=category,
                hook_type=hook_type,
                success=False,
                execution_time=execution_time,
                error_message=str(e),
                metadata=hook_info["metadata"]
            )

    def _update_execution_stats(self, hook_type: str, results: List[HookExecutionResult], total_duration: float):
        """Update execution statistics for hook type."""
        stats = self.execution_stats[hook_type]

        stats["total_executions"] += len(results)
        stats["successful_executions"] += sum(1 for r in results if r.success)
        stats["failed_executions"] += sum(1 for r in results if not r.success)
        stats["total_duration"] += total_duration
        stats["average_duration"] = stats["total_duration"] / stats["total_executions"]
        stats["last_execution"] = datetime.now()

    async def _log_hook_execution(self, hook_type: str, results: List[HookExecutionResult], duration: float):
        """Log hook execution with rich message formatting."""
        if not self.message_processor:
            return

        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful

        # Create rich message
        message_data = {
            "type": "hook_execution",
            "hook_type": hook_type,
            "total_hooks": len(results),
            "successful_hooks": successful,
            "failed_hooks": failed,
            "total_duration": duration,
            "average_hook_duration": sum(r.execution_time for r in results) / len(results) if results else 0,
            "results": [
                {
                    "name": r.hook_name,
                    "success": r.success,
                    "duration": r.execution_time,
                    "error": r.error_message
                }
                for r in results
            ]
        }

        try:
            await self.message_processor.process_message(
                message_type=MessageType.INFO,
                content=f"🔗 Hook Execution: {hook_type} ({successful}/{len(results)} successful)",
                metadata=message_data
            )
        except Exception as e:
            self.logger.warning(f"Failed to process hook execution message: {e}")

    def _register_default_hooks(self):
        """Register default comprehensive hooks for system monitoring."""

        # Research operations hooks
        self.register_hook("search_initiated", self._hook_search_initiated, HookPriority.HIGH)
        self.register_hook("search_completed", self._hook_search_completed, HookPriority.HIGH)
        self.register_hook("scraping_progress", self._hook_scraping_progress, HookPriority.NORMAL)
        self.register_hook("content_cleaning_completed", self._hook_content_cleaning_completed, HookPriority.HIGH)

        # Quality management hooks
        self.register_hook("quality_assessment_completed", self._hook_quality_assessment_completed, HookPriority.HIGH)
        self.register_hook("quality_gate_evaluation", self._hook_quality_gate_evaluation, HookPriority.HIGHEST)
        self.register_hook("progressive_enhancement_cycle", self._hook_progressive_enhancement_cycle, HookPriority.NORMAL)

        # Editorial workflow hooks
        self.register_hook("gap_research_decision", self._hook_gap_research_decision, HookPriority.HIGHEST)
        self.register_hook("gap_research_completed", self._hook_gap_research_completed, HookPriority.HIGH)

        # Performance monitoring hooks
        self.register_hook("performance_metrics_collection", self._hook_performance_metrics_collection, HookPriority.NORMAL)
        self.register_hook("bottleneck_detected", self._hook_bottleneck_detected, HookPriority.HIGH)

        # Error handling hooks
        self.register_hook("error_detected", self._hook_error_detected, HookPriority.HIGH)
        self.register_hook("error_recovery_completed", self._hook_error_recovery_completed, HookPriority.NORMAL)

    # Default hook implementations
    async def _hook_search_initiated(self, context: WorkflowHookContext) -> Dict[str, Any]:
        """Hook for search initiation."""
        self.logger.info(f"🔍 Search initiated for session {context.session_id}")

        # Record search start metric
        if self.message_processor:
            await self.message_processor.process_message(
                MessageType.INFO,
                f"🔍 Starting search: {context.metadata.get('query', 'Unknown query')}",
                metadata={
                    "session_id": context.session_id,
                    "operation": "search_initiated",
                    "query": context.metadata.get('query'),
                    "search_type": context.metadata.get('search_type', 'standard')
                }
            )

        return {"event": "search_initiated", "timestamp": datetime.now().isoformat()}

    async def _hook_search_completed(self, context: WorkflowHookContext) -> Dict[str, Any]:
        """Hook for search completion."""
        duration = context.get_duration()
        results_count = context.metadata.get('results_count', 0)

        self.logger.info(f"✅ Search completed in {duration:.2f}s: {results_count} results")

        if self.message_processor:
            await self.message_processor.process_message(
                MessageType.SUCCESS,
                f"✅ Search completed: {results_count} results in {duration:.2f}s",
                metadata={
                    "session_id": context.session_id,
                    "operation": "search_completed",
                    "duration": duration,
                    "results_count": results_count,
                    "success_rate": context.metadata.get('success_rate', 0)
                }
            )

        return {
            "event": "search_completed",
            "duration": duration,
            "results_count": results_count,
            "success_rate": context.metadata.get('success_rate', 0)
        }

    async def _hook_scraping_progress(self, context: WorkflowHookContext) -> Dict[str, Any]:
        """Hook for scraping progress updates."""
        progress = context.metadata.get('progress', 0)
        success_count = context.metadata.get('success_count', 0)
        target_count = context.metadata.get('target_count', 0)

        self.logger.debug(f"📊 Scraping progress: {progress:.1f}% ({success_count}/{target_count})")

        return {
            "event": "scraping_progress",
            "progress": progress,
            "success_count": success_count,
            "target_count": target_count
        }

    async def _hook_content_cleaning_completed(self, context: WorkflowHookContext) -> Dict[str, Any]:
        """Hook for content cleaning completion."""
        duration = context.get_duration()
        cleaned_count = context.metadata.get('cleaned_count', 0)
        quality_score = context.metadata.get('average_quality_score', 0)

        self.logger.info(f"🧹 Content cleaning completed: {cleaned_count} items, avg quality: {quality_score:.2f}")

        return {
            "event": "content_cleaning_completed",
            "duration": duration,
            "cleaned_count": cleaned_count,
            "average_quality_score": quality_score
        }

    async def _hook_quality_assessment_completed(self, context: WorkflowHookContext) -> Dict[str, Any]:
        """Hook for quality assessment completion."""
        overall_score = context.quality_metrics.get('overall_score', 0)
        quality_level = context.quality_metrics.get('quality_level', 'Unknown')

        self.logger.info(f"📊 Quality assessment: {quality_level} ({overall_score}/100)")

        if self.message_processor:
            await self.message_processor.process_message(
                MessageType.INFO,
                f"📊 Quality assessment: {quality_level} ({overall_score}/100)",
                metadata={
                    "session_id": context.session_id,
                    "operation": "quality_assessment",
                    "overall_score": overall_score,
                    "quality_level": quality_level,
                    "criteria_scores": context.quality_metrics.get('criteria_scores', {})
                }
            )

        return {
            "event": "quality_assessment_completed",
            "overall_score": overall_score,
            "quality_level": quality_level,
            "criteria_scores": context.quality_metrics.get('criteria_scores', {})
        }

    async def _hook_quality_gate_evaluation(self, context: WorkflowHookContext) -> Dict[str, Any]:
        """Hook for quality gate evaluation."""
        decision = context.metadata.get('gate_decision', 'UNKNOWN')
        threshold = context.metadata.get('threshold', 0)
        actual_score = context.metadata.get('actual_score', 0)

        self.logger.info(f"🚪 Quality gate: {decision} (score: {actual_score}, threshold: {threshold})")

        if self.message_processor:
            message_type = MessageType.SUCCESS if decision == 'PROCEED' else MessageType.WARNING
            await self.message_processor.process_message(
                message_type,
                f"🚪 Quality gate: {decision}",
                metadata={
                    "session_id": context.session_id,
                    "operation": "quality_gate_evaluation",
                    "decision": decision,
                    "threshold": threshold,
                    "actual_score": actual_score,
                    "gap": actual_score - threshold
                }
            )

        return {
            "event": "quality_gate_evaluation",
            "decision": decision,
            "threshold": threshold,
            "actual_score": actual_score
        }

    async def _hook_progressive_enhancement_cycle(self, context: WorkflowHookContext) -> Dict[str, Any]:
        """Hook for progressive enhancement cycles."""
        cycle_number = context.metadata.get('cycle_number', 1)
        improvement = context.metadata.get('improvement', 0)
        stage = context.metadata.get('enhancement_stage', 'Unknown')

        self.logger.info(f"🔄 Enhancement cycle {cycle_number}: {stage} (+{improvement:.1f} improvement)")

        return {
            "event": "progressive_enhancement_cycle",
            "cycle_number": cycle_number,
            "enhancement_stage": stage,
            "improvement": improvement
        }

    async def _hook_gap_research_decision(self, context: WorkflowHookContext) -> Dict[str, Any]:
        """Hook for gap research decisions."""
        should_research = context.metadata.get('should_research', False)
        confidence = context.metadata.get('confidence', 0)
        gap_areas = context.metadata.get('gap_areas', [])

        self.logger.info(f"🔍 Gap research decision: {'YES' if should_research else 'NO'} "
                        f"(confidence: {confidence:.2f}, gaps: {len(gap_areas)})")

        if self.message_processor:
            message_type = MessageType.INFO if should_research else MessageType.SUCCESS
            await self.message_processor.process_message(
                message_type,
                f"🔍 Gap research: {'Required' if should_research else 'Not required'}",
                metadata={
                    "session_id": context.session_id,
                    "operation": "gap_research_decision",
                    "should_research": should_research,
                    "confidence": confidence,
                    "gap_areas": gap_areas
                }
            )

        return {
            "event": "gap_research_decision",
            "should_research": should_research,
            "confidence": confidence,
            "gap_areas": gap_areas
        }

    async def _hook_gap_research_completed(self, context: WorkflowHookContext) -> Dict[str, Any]:
        """Hook for gap research completion."""
        duration = context.get_duration()
        gap_results = context.metadata.get('gap_results', {})
        success_count = context.metadata.get('success_count', 0)

        self.logger.info(f"✅ Gap research completed in {duration:.2f}s: {success_count} additional results")

        return {
            "event": "gap_research_completed",
            "duration": duration,
            "success_count": success_count,
            "gap_results": gap_results
        }

    async def _hook_performance_metrics_collection(self, context: WorkflowHookContext) -> Dict[str, Any]:
        """Hook for performance metrics collection."""
        metrics = context.metadata.get('metrics', {})

        self.logger.debug(f"📈 Performance metrics collected: {len(metrics)} metrics")

        return {
            "event": "performance_metrics_collection",
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }

    async def _hook_bottleneck_detected(self, context: WorkflowHookContext) -> Dict[str, Any]:
        """Hook for bottleneck detection."""
        bottleneck_type = context.metadata.get('bottleneck_type', 'Unknown')
        severity = context.metadata.get('severity', 'medium')
        affected_component = context.metadata.get('affected_component', 'Unknown')

        self.logger.warning(f"⚠️ Bottleneck detected: {bottleneck_type} in {affected_component} ({severity})")

        if self.message_processor:
            await self.message_processor.process_message(
                MessageType.WARNING,
                f"⚠️ Performance bottleneck: {bottleneck_type}",
                metadata={
                    "session_id": context.session_id,
                    "operation": "bottleneck_detection",
                    "bottleneck_type": bottleneck_type,
                    "severity": severity,
                    "affected_component": affected_component
                }
            )

        return {
            "event": "bottleneck_detected",
            "bottleneck_type": bottleneck_type,
            "severity": severity,
            "affected_component": affected_component
        }

    async def _hook_error_detected(self, context: WorkflowHookContext) -> Dict[str, Any]:
        """Hook for error detection."""
        error_type = context.metadata.get('error_type', 'Unknown')
        component = context.metadata.get('component', 'Unknown')
        recoverable = context.metadata.get('recoverable', True)

        self.logger.error(f"❌ Error detected in {component}: {error_type} "
                         f"{'(recoverable)' if recoverable else '(non-recoverable)'}")

        if self.message_processor:
            await self.message_processor.process_message(
                MessageType.ERROR,
                f"❌ Error in {component}: {error_type}",
                metadata={
                    "session_id": context.session_id,
                    "operation": "error_detection",
                    "error_type": error_type,
                    "component": component,
                    "recoverable": recoverable,
                    "error_context": context.error_context
                }
            )

        return {
            "event": "error_detected",
            "error_type": error_type,
            "component": component,
            "recoverable": recoverable
        }

    async def _hook_error_recovery_completed(self, context: WorkflowHookContext) -> Dict[str, Any]:
        """Hook for error recovery completion."""
        recovery_strategy = context.metadata.get('recovery_strategy', 'Unknown')
        success = context.metadata.get('recovery_success', False)
        duration = context.metadata.get('recovery_duration', 0)

        status = "✅ Recovery successful" if success else "❌ Recovery failed"
        self.logger.info(f"{status}: {recovery_strategy} ({duration:.2f}s)")

        if self.message_processor:
            message_type = MessageType.SUCCESS if success else MessageType.ERROR
            await self.message_processor.process_message(
                message_type,
                f"{status}: {recovery_strategy}",
                metadata={
                    "session_id": context.session_id,
                    "operation": "error_recovery",
                    "recovery_strategy": recovery_strategy,
                    "success": success,
                    "duration": duration
                }
            )

        return {
            "event": "error_recovery_completed",
            "recovery_strategy": recovery_strategy,
            "success": success,
            "duration": duration
        }

    def get_hook_statistics(self) -> Dict[str, Any]:
        """Get comprehensive hook execution statistics."""
        return {
            "hook_types": list(self.hooks.keys()),
            "execution_stats": self.execution_stats.copy(),
            "performance_history": {
                hook_type: {
                    "recent_count": len(history),
                    "average_execution_time": sum(history) / len(history) if history else 0,
                    "min_execution_time": min(history) if history else 0,
                    "max_execution_time": max(history) if history else 0
                }
                for hook_type, history in self.performance_history.items()
            },
            "category_counts": {
                category.value: sum(1 for cat in self.hook_categories.values() if cat == category)
                for category in HookCategory
            }
        }

    def get_hooks_by_category(self, category: HookCategory) -> Dict[str, List[Dict[str, Any]]]:
        """Get all hooks of a specific category."""
        return {
            hook_type: hooks
            for hook_type, hooks in self.hooks.items()
            if self.hook_categories.get(hook_type) == category
        }

    def enable_hook(self, hook_type: str, hook_name: str):
        """Enable a specific hook."""
        if hook_type in self.hooks:
            for hook in self.hooks[hook_type]:
                if hook["name"] == hook_name:
                    hook["enabled"] = True
                    self.logger.info(f"Enabled hook: {hook_name} for {hook_type}")

    def disable_hook(self, hook_type: str, hook_name: str):
        """Disable a specific hook."""
        if hook_type in self.hooks:
            for hook in self.hooks[hook_type]:
                if hook["name"] == hook_name:
                    hook["enabled"] = False
                    self.logger.info(f"Disabled hook: {hook_name} for {hook_type}")


# Factory function for creating comprehensive hook manager
def create_comprehensive_hook_manager(logger: Optional[logging.Logger] = None) -> ComprehensiveHookManager:
    """
    Create and configure a comprehensive hook manager.

    Args:
        logger: Optional logger instance

    Returns:
        Configured ComprehensiveHookManager instance
    """
    return ComprehensiveHookManager(logger=logger)