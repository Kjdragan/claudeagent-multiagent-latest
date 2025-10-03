"""
Comprehensive Hook System for Multi-Agent Research System

This module provides extensive hook monitoring and coordination capabilities
for tracking agent interactions, tool execution, and workflow orchestration.
"""

from .base_hooks import HookManager, HookContext, HookResult
from .tool_hooks import ToolExecutionHook, ToolPerformanceMonitor
from .agent_hooks import AgentCommunicationHook, AgentHandoffHook, AgentStateMonitor
from .workflow_hooks import WorkflowOrchestrationHook, StageTransitionHook, DecisionPointHook
from .session_hooks import SessionLifecycleHook
from .monitoring_hooks import SystemHealthHook, PerformanceMonitorHook, ErrorTrackingHook
from .sdk_integration import (
    SDKMessageProcessingHook,
    SDKHookIntegration,
    SDKHookBridge
)
from .mcp_hooks import (
    MCPMessageHook,
    MCPToolExecutionHook,
    MCPSessionHook
)
from .hook_integration_manager import (
    HookIntegrationManager,
    HookIntegrationConfig
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