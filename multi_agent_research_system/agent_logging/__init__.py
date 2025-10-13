"""
Enhanced Logging System for Multi-Agent Research System

This module provides comprehensive logging capabilities for monitoring
agent activities, tool execution, and orchestrator operations.

Enhanced with Phase 1.1.3: Advanced logging infrastructure with structured logging,
performance monitoring, real-time metrics, and observability features.
"""

# Legacy logging components
from .agent_logger import AgentLogger
from .agent_loggers import (
    EditorAgentLogger,
    ReportAgentLogger,
    ResearchAgentLogger,
    UICoordinatorLogger,
    create_agent_logger,
)
from .hook_logger import (
    AgentCommunicationLogger,
    HookLogger,
    SessionLifecycleLogger,
    ToolUseLogger,
    WorkflowLogger,
)
from .structured_logger import StructuredLogger, get_logger

# Enhanced logging and monitoring components (Phase 1.1.3)
from .enhanced_logger import (
    EnhancedLogger,
    get_enhanced_logger,
    setup_logging_for_session,
    get_session_performance_summary,
    export_all_logs,
    LogLevel,
    LogCategory,
    AgentEventType,
    LogEvent,
    PerformanceMetrics,
)
from .monitoring import (
    MetricsCollector,
    AlertManager,
    HealthChecker,
    MonitoringSystem,
    Alert,
    AlertSeverity,
    Metric,
    MetricType,
    get_monitoring_system,
    start_monitoring,
    stop_monitoring,
    record_agent_task,
    record_tool_execution,
    check_system_resources,
    check_process_health,
)

__all__ = [
    # Legacy logging components
    "StructuredLogger",
    "get_logger",
    "AgentLogger",
    "HookLogger",
    "ToolUseLogger",
    "AgentCommunicationLogger",
    "SessionLifecycleLogger",
    "WorkflowLogger",
    "ResearchAgentLogger",
    "ReportAgentLogger",
    "EditorAgentLogger",
    "UICoordinatorLogger",
    "create_agent_logger",

    # Enhanced logging components (Phase 1.1.3)
    "EnhancedLogger",
    "get_enhanced_logger",
    "setup_logging_for_session",
    "get_session_performance_summary",
    "export_all_logs",
    "LogLevel",
    "LogCategory",
    "AgentEventType",
    "LogEvent",
    "PerformanceMetrics",

    # Monitoring and metrics components
    "MetricsCollector",
    "AlertManager",
    "HealthChecker",
    "MonitoringSystem",
    "Alert",
    "AlertSeverity",
    "Metric",
    "MetricType",
    "get_monitoring_system",
    "start_monitoring",
    "stop_monitoring",
    "record_agent_task",
    "record_tool_execution",
    "check_system_resources",
    "check_process_health",
]
