"""
Comprehensive Hook System for Multi-Agent Research System

Phase 3.1: Enhanced Hooks System with Claude Agent SDK Integration

This module provides extensive hook monitoring and coordination capabilities
for tracking agent interactions, tool execution, and workflow orchestration
with comprehensive real-time monitoring, analytics, and optimization.

New Phase 3.1 Features:
- Comprehensive hooks system with Claude Agent SDK integration
- Real-time monitoring infrastructure with centralized metrics collection
- Performance analytics with bottleneck detection and optimization
- Intelligent alerting system with adaptive thresholds
- Hook analytics and optimization with automated tuning
- Complete integration with Phase 1 & 2 systems
"""

# Legacy hooks (existing from before Phase 3.1)
from .agent_hooks import AgentCommunicationHook, AgentHandoffHook, AgentStateMonitor
from .base_hooks import HookContext, HookManager, HookResult
from .hook_integration_manager import HookIntegrationConfig, HookIntegrationManager
from .mcp_hooks import MCPMessageHook, MCPSessionHook, MCPToolExecutionHook
from .monitoring_hooks import (
    ErrorTrackingHook,
    PerformanceMonitorHook,
    SystemHealthHook,
)
from .sdk_integration import SDKHookBridge, SDKHookIntegration, SDKMessageProcessingHook
from .session_hooks import SessionLifecycleHook
from .tool_hooks import ToolExecutionHook, ToolPerformanceMonitor
from .workflow_hooks import (
    DecisionPointHook,
    StageTransitionHook,
    WorkflowOrchestrationHook,
)

# Phase 3.1 Enhanced Hooks System
from .comprehensive_hooks import (
    ComprehensiveHookManager,
    HookCategory,
    HookPriority,
    HookExecutionResult,
    create_comprehensive_hook_manager,
)
from .hook_analytics import (
    HookAnalyticsEngine,
    OptimizationType,
    PerformanceLevel,
    HookPerformanceMetrics,
    OptimizationRecommendation,
    BottleneckAnalysis,
    create_hook_analytics_engine,
)
from .real_time_monitoring import (
    MetricsCollector,
    RealTimeMonitor,
    MetricType,
    MetricValue,
    AlertSeverity,
    HealthLevel,
    PerformanceThreshold,
    Alert,
    SystemHealthSnapshot,
    create_real_time_monitoring,
)
from .enhanced_integration import (
    EnhancedHooksIntegrator,
    IntegrationConfig,
    IntegrationLevel,
    create_enhanced_hooks_integration,
)

__all__ = [
    # Legacy Base hook infrastructure
    "HookManager",
    "HookContext",
    "HookResult",

    # Legacy Tool execution hooks
    "ToolExecutionHook",
    "ToolPerformanceMonitor",

    # Legacy Agent communication hooks
    "AgentCommunicationHook",
    "AgentHandoffHook",
    "AgentStateMonitor",

    # Legacy Workflow orchestration hooks
    "WorkflowOrchestrationHook",
    "StageTransitionHook",
    "DecisionPointHook",

    # Legacy Session management hooks
    "SessionLifecycleHook",

    # Legacy System monitoring hooks
    "SystemHealthHook",
    "PerformanceMonitorHook",
    "ErrorTrackingHook",

    # Legacy SDK integration hooks
    "SDKMessageProcessingHook",
    "SDKHookIntegration",
    "SDKHookBridge",

    # Legacy MCP-aware hooks
    "MCPMessageHook",
    "MCPToolExecutionHook",
    "MCPSessionHook",

    # Legacy Integration management
    "HookIntegrationManager",
    "HookIntegrationConfig",

    # === PHASE 3.1 COMPREHENSIVE HOOKS SYSTEM ===

    # Comprehensive Hook Management
    "ComprehensiveHookManager",
    "HookCategory",
    "HookPriority",
    "HookExecutionResult",
    "create_comprehensive_hook_manager",

    # Hook Analytics and Optimization
    "HookAnalyticsEngine",
    "OptimizationType",
    "PerformanceLevel",
    "HookPerformanceMetrics",
    "OptimizationRecommendation",
    "BottleneckAnalysis",
    "create_hook_analytics_engine",

    # Real-Time Monitoring Infrastructure
    "MetricsCollector",
    "RealTimeMonitor",
    "MetricType",
    "MetricValue",
    "AlertSeverity",
    "HealthLevel",
    "PerformanceThreshold",
    "Alert",
    "SystemHealthSnapshot",
    "create_real_time_monitoring",

    # Enhanced Integration System
    "EnhancedHooksIntegrator",
    "IntegrationConfig",
    "IntegrationLevel",
    "create_enhanced_hooks_integration",
]
