"""
Comprehensive Hook System for Multi-Agent Research System

This module provides extensive hook monitoring and coordination capabilities
for tracking agent interactions, tool execution, and workflow orchestration.
"""

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

__all__ = [
    # Base hook infrastructure
    "HookManager",
    "HookContext",
    "HookResult",

    # Tool execution hooks
    "ToolExecutionHook",
    "ToolPerformanceMonitor",

    # Agent communication hooks
    "AgentCommunicationHook",
    "AgentHandoffHook",
    "AgentStateMonitor",

    # Workflow orchestration hooks
    "WorkflowOrchestrationHook",
    "StageTransitionHook",
    "DecisionPointHook",

    # Session management hooks
    "SessionLifecycleHook",

    # System monitoring hooks
    "SystemHealthHook",
    "PerformanceMonitorHook",
    "ErrorTrackingHook",

    # SDK integration hooks
    "SDKMessageProcessingHook",
    "SDKHookIntegration",
    "SDKHookBridge",

    # MCP-aware hooks
    "MCPMessageHook",
    "MCPToolExecutionHook",
    "MCPSessionHook",

    # Integration management
    "HookIntegrationManager",
    "HookIntegrationConfig",
]
