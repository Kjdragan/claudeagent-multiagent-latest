"""
Comprehensive Hook Integration Manager for Multi-Agent Research System

This manager integrates all hook systems with the orchestrator using proper
MCP structure and SDK patterns. It provides a unified interface for hook
management, execution, and monitoring across all hook categories.
"""

import os
import sys
from dataclasses import dataclass
from typing import Any

from .agent_hooks import AgentCommunicationHook, AgentHandoffHook, AgentStateMonitor
from .base_hooks import HookContext, HookManager, HookResult
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

# Add parent directory to path for proper imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent_logging import get_logger


@dataclass
class HookIntegrationConfig:
    """Configuration for hook integration manager."""
    enable_tool_hooks: bool = True
    enable_agent_hooks: bool = True
    enable_workflow_hooks: bool = True
    enable_session_hooks: bool = True
    enable_monitoring_hooks: bool = True
    enable_sdk_hooks: bool = True
    enable_mcp_hooks: bool = True

    # Hook-specific settings
    tool_execution_timeout: float = 30.0
    agent_communication_timeout: float = 15.0
    workflow_orchestration_timeout: float = 20.0
    session_lifecycle_timeout: float = 10.0
    system_health_timeout: float = 5.0
    sdk_integration_timeout: float = 10.0
    mcp_message_timeout: float = 10.0
    mcp_tool_timeout: float = 30.0
    mcp_session_timeout: float = 15.0

    # Performance settings
    parallel_execution: bool = True
    max_concurrent_hooks: int = 10
    enable_hook_caching: bool = True


class HookIntegrationManager:
    """
    Comprehensive manager for integrating all hook systems with the orchestrator.

    This manager provides a unified interface for hook management, ensuring
    proper MCP structure compliance and SDK pattern integration.
    """

    def __init__(self, config: HookIntegrationConfig | None = None):
        """Initialize the hook integration manager."""
        self.config = config or HookIntegrationConfig()
        self.logger = get_logger("hook_integration_manager")

        # Initialize hook managers for different categories
        self.hook_managers: dict[str, HookManager] = {}

        # Initialize hooks
        self.tool_hooks: dict[str, Any] = {}
        self.agent_hooks: dict[str, Any] = {}
        self.workflow_hooks: dict[str, Any] = {}
        self.session_hooks: dict[str, Any] = {}
        self.monitoring_hooks: dict[str, Any] = {}
        self.sdk_hooks: dict[str, Any] = {}
        self.mcp_hooks: dict[str, Any] = {}

        # SDK bridge for integration
        self.sdk_bridge: SDKHookBridge | None = None

        # Statistics and monitoring
        self.integration_stats: dict[str, Any] = {}
        self.initialized = False

    async def initialize(self) -> bool:
        """Initialize all hook systems and integrations."""
        try:
            self.logger.info("Initializing hook integration manager...",
                           config=self.config.__dict__)

            # Initialize hook managers
            await self._initialize_hook_managers()

            # Initialize all hook categories
            await self._initialize_tool_hooks()
            await self._initialize_agent_hooks()
            await self._initialize_workflow_hooks()
            await self._initialize_session_hooks()
            await self._initialize_monitoring_hooks()
            await self._initialize_sdk_hooks()
            await self._initialize_mcp_hooks()

            # Initialize SDK bridge
            await self._initialize_sdk_bridge()

            # Set up cross-hook integration
            await self._setup_cross_hook_integration()

            self.initialized = True
            self.logger.info("Hook integration manager initialized successfully",
                           total_hooks=self._get_total_hook_count())

            return True

        except Exception as e:
            self.logger.error(f"Hook integration manager initialization failed: {str(e)}",
                            error=str(e),
                            error_type=type(e).__name__)
            return False

    async def _initialize_hook_managers(self):
        """Initialize hook managers for different categories."""
        categories = [
            "tool_execution", "agent_communication", "workflow_orchestration",
            "session_lifecycle", "system_monitoring", "sdk_integration", "mcp_processing"
        ]

        for category in categories:
            self.hook_managers[category] = HookManager()
            self.logger.debug(f"Hook manager initialized: {category}")

    async def _initialize_tool_hooks(self):
        """Initialize tool execution hooks."""
        if not self.config.enable_tool_hooks:
            return

        # Create and register tool hooks
        tool_execution_hook = ToolExecutionHook(
            enabled=True,
            timeout=self.config.tool_execution_timeout
        )

        tool_performance_monitor = ToolPerformanceMonitor(
            enabled=True,
            timeout=self.config.tool_execution_timeout
        )

        # Register with tool execution manager
        manager = self.hook_managers["tool_execution"]
        manager.register_hook(tool_execution_hook, ["PreToolUse", "PostToolUse", "tool_start", "tool_complete"])
        manager.register_hook(tool_performance_monitor, ["tool_execution", "tool_performance"])

        self.tool_hooks = {
            "execution": tool_execution_hook,
            "performance": tool_performance_monitor
        }

        self.logger.info("Tool hooks initialized", hooks=list(self.tool_hooks.keys()))

    async def _initialize_agent_hooks(self):
        """Initialize agent communication hooks."""
        if not self.config.enable_agent_hooks:
            return

        # Create agent hooks
        agent_communication_hook = AgentCommunicationHook(
            enabled=True,
            timeout=self.config.agent_communication_timeout
        )

        agent_handoff_hook = AgentHandoffHook(
            enabled=True,
            timeout=self.config.agent_communication_timeout
        )

        agent_state_monitor = AgentStateMonitor(
            enabled=True,
            timeout=self.config.agent_communication_timeout
        )

        # Register with agent communication manager
        manager = self.hook_managers["agent_communication"]
        manager.register_hook(agent_communication_hook, ["agent_communication", "UserPromptSubmit", "Stop"])
        manager.register_hook(agent_handoff_hook, ["agent_handoff", "handoff"])
        manager.register_hook(agent_state_monitor, ["agent_state", "state_monitor"])

        self.agent_hooks = {
            "communication": agent_communication_hook,
            "handoff": agent_handoff_hook,
            "state_monitor": agent_state_monitor
        }

        self.logger.info("Agent hooks initialized", hooks=list(self.agent_hooks.keys()))

    async def _initialize_workflow_hooks(self):
        """Initialize workflow orchestration hooks."""
        if not self.config.enable_workflow_hooks:
            return

        # Create workflow hooks
        workflow_orchestration_hook = WorkflowOrchestrationHook(
            enabled=True,
            timeout=self.config.workflow_orchestration_timeout
        )

        stage_transition_hook = StageTransitionHook(
            enabled=True,
            timeout=self.config.workflow_orchestration_timeout
        )

        decision_point_hook = DecisionPointHook(
            enabled=True,
            timeout=self.config.workflow_orchestration_timeout
        )

        # Register with workflow orchestration manager
        manager = self.hook_managers["workflow_orchestration"]
        manager.register_hook(workflow_orchestration_hook, ["workflow_orchestration", "workflow"])
        manager.register_hook(stage_transition_hook, ["stage_transition", "transition"])
        manager.register_hook(decision_point_hook, ["decision_point", "decision"])

        self.workflow_hooks = {
            "orchestration": workflow_orchestration_hook,
            "stage_transition": stage_transition_hook,
            "decision_point": decision_point_hook
        }

        self.logger.info("Workflow hooks initialized", hooks=list(self.workflow_hooks.keys()))

    async def _initialize_session_hooks(self):
        """Initialize session lifecycle hooks."""
        if not self.config.enable_session_hooks:
            return

        # Create session hooks
        session_lifecycle_hook = SessionLifecycleHook(
            enabled=True,
            timeout=self.config.session_lifecycle_timeout
        )

        # TODO: SessionStateMonitor and SessionRecoveryHook are not yet implemented
        # session_state_monitor = SessionStateMonitor(
        #     enabled=True,
        #     timeout=self.config.session_lifecycle_timeout
        # )

        # session_recovery_hook = SessionRecoveryHook(
        #     enabled=True,
        #     timeout=self.config.session_lifecycle_timeout
        # )

        # Register with session lifecycle manager
        manager = self.hook_managers["session_lifecycle"]
        manager.register_hook(session_lifecycle_hook, ["session_lifecycle", "session"])
        # manager.register_hook(session_state_monitor, ["session_state", "state"])
        # manager.register_hook(session_recovery_hook, ["session_recovery", "recovery"])

        self.session_hooks = {
            "lifecycle": session_lifecycle_hook,
            # "state_monitor": session_state_monitor,
            # "recovery": session_recovery_hook
        }

        self.logger.info("Session hooks initialized", hooks=list(self.session_hooks.keys()))

    async def _initialize_monitoring_hooks(self):
        """Initialize system monitoring hooks."""
        if not self.config.enable_monitoring_hooks:
            return

        # Create monitoring hooks
        system_health_hook = SystemHealthHook(
            enabled=True,
            timeout=self.config.system_health_timeout
        )

        performance_monitor_hook = PerformanceMonitorHook(
            enabled=True,
            timeout=self.config.system_health_timeout
        )

        error_tracking_hook = ErrorTrackingHook(
            enabled=True,
            timeout=self.config.system_health_timeout
        )

        # Register with system monitoring manager
        manager = self.hook_managers["system_monitoring"]
        manager.register_hook(system_health_hook, ["system_health", "health"])
        manager.register_hook(performance_monitor_hook, ["performance_monitor", "performance"])
        manager.register_hook(error_tracking_hook, ["error_tracking", "error"])

        self.monitoring_hooks = {
            "system_health": system_health_hook,
            "performance_monitor": performance_monitor_hook,
            "error_tracking": error_tracking_hook
        }

        self.logger.info("Monitoring hooks initialized", hooks=list(self.monitoring_hooks.keys()))

    async def _initialize_sdk_hooks(self):
        """Initialize SDK integration hooks."""
        if not self.config.enable_sdk_hooks:
            return

        # Create SDK hooks
        sdk_message_processing_hook = SDKMessageProcessingHook(
            enabled=True,
            timeout=self.config.sdk_integration_timeout
        )

        sdk_hook_integration = SDKHookIntegration(
            enabled=True,
            timeout=self.config.sdk_integration_timeout
        )

        # Register with SDK integration manager
        manager = self.hook_managers["sdk_integration"]
        manager.register_hook(sdk_message_processing_hook, ["sdk_message_processing", "sdk_message"])
        manager.register_hook(sdk_hook_integration, ["sdk_hook_integration", "sdk_hook"])

        self.sdk_hooks = {
            "message_processing": sdk_message_processing_hook,
            "hook_integration": sdk_hook_integration
        }

        self.logger.info("SDK hooks initialized", hooks=list(self.sdk_hooks.keys()))

    async def _initialize_mcp_hooks(self):
        """Initialize MCP-aware hooks."""
        if not self.config.enable_mcp_hooks:
            return

        # Create MCP hooks
        mcp_message_hook = MCPMessageHook(
            enabled=True,
            timeout=self.config.mcp_message_timeout
        )

        mcp_tool_execution_hook = MCPToolExecutionHook(
            enabled=True,
            timeout=self.config.mcp_tool_timeout
        )

        mcp_session_hook = MCPSessionHook(
            enabled=True,
            timeout=self.config.mcp_session_timeout
        )

        # Register with MCP processing manager
        manager = self.hook_managers["mcp_processing"]
        manager.register_hook(mcp_message_hook, ["mcp_message_processing", "mcp_message"])
        manager.register_hook(mcp_tool_execution_hook, ["mcp_tool_execution", "mcp_tool"])
        manager.register_hook(mcp_session_hook, ["mcp_session_management", "mcp_session"])

        self.mcp_hooks = {
            "message": mcp_message_hook,
            "tool_execution": mcp_tool_execution_hook,
            "session": mcp_session_hook
        }

        self.logger.info("MCP hooks initialized", hooks=list(self.mcp_hooks.keys()))

    async def _initialize_sdk_bridge(self):
        """Initialize SDK bridge for integration."""
        try:
            # Create a combined hook manager for SDK bridge
            combined_manager = HookManager()

            # Register all hooks with the combined manager for SDK integration
            for hook_category, hooks in [
                ("tool", self.tool_hooks),
                ("agent", self.agent_hooks),
                ("workflow", self.workflow_hooks),
                ("session", self.session_hooks),
                ("monitoring", self.monitoring_hooks),
                ("sdk", self.sdk_hooks),
                ("mcp", self.mcp_hooks)
            ]:
                for hook_name, hook_instance in hooks.items():
                    hook_types = [f"{hook_category}_{hook_name}", hook_category]
                    combined_manager.register_hook(hook_instance, hook_types)

            # Create SDK bridge
            self.sdk_bridge = SDKHookBridge(combined_manager)

            self.logger.info("SDK bridge initialized successfully")

        except Exception as e:
            self.logger.error(f"SDK bridge initialization failed: {str(e)}",
                            error=str(e),
                            error_type=type(e).__name__)

    async def _setup_cross_hook_integration(self):
        """Set up integration between different hook categories."""
        # This would set up cross-hook communication and data sharing
        # For now, we'll just log that it's been set up
        self.logger.info("Cross-hook integration set up")

    def _get_total_hook_count(self) -> int:
        """Get total number of registered hooks."""
        return sum(len(hooks) for hooks in [
            self.tool_hooks, self.agent_hooks, self.workflow_hooks,
            self.session_hooks, self.monitoring_hooks, self.sdk_hooks, self.mcp_hooks
        ])

    async def execute_hooks(
        self,
        hook_type: str,
        context: HookContext,
        category: str | None = None
    ) -> list[HookResult]:
        """
        Execute hooks for a given type and optionally category.

        Args:
            hook_type: The type of hook to execute
            context: Hook context information
            category: Optional hook category to limit execution

        Returns:
            List of hook execution results
        """
        if not self.initialized:
            self.logger.warning("Hook integration manager not initialized")
            return []

        try:
            results = []

            if category and category in self.hook_managers:
                # Execute hooks in specific category
                manager = self.hook_managers[category]
                category_results = await manager.execute_hooks(
                    hook_type,
                    context,
                    parallel=self.config.parallel_execution
                )
                results.extend(category_results)
            else:
                # Execute hooks in all relevant categories
                for category_name, manager in self.hook_managers.items():
                    try:
                        category_results = await manager.execute_hooks(
                            hook_type,
                            context,
                            parallel=self.config.parallel_execution
                        )
                        results.extend(category_results)
                    except Exception as e:
                        self.logger.error(f"Hook execution failed in category {category_name}: {str(e)}",
                                        category=category_name,
                                        hook_type=hook_type,
                                        error=str(e))

            # Update integration statistics
            self._update_integration_stats(hook_type, category, results)

            return results

        except Exception as e:
            self.logger.error(f"Hook execution failed: {str(e)}",
                            hook_type=hook_type,
                            category=category,
                            error=str(e),
                            error_type=type(e).__name__)
            return []

    def _update_integration_stats(
        self,
        hook_type: str,
        category: str | None,
        results: list[HookResult]
    ):
        """Update integration statistics."""
        if "total_executions" not in self.integration_stats:
            self.integration_stats["total_executions"] = 0
            self.integration_stats["successful_executions"] = 0
            self.integration_stats["failed_executions"] = 0
            self.integration_stats["hook_type_stats"] = {}
            self.integration_stats["category_stats"] = {}

        self.integration_stats["total_executions"] += len(results)
        self.integration_stats["successful_executions"] += sum(1 for r in results if r.success)
        self.integration_stats["failed_executions"] += sum(1 for r in results if r.failed)

        # Update hook type statistics
        if hook_type not in self.integration_stats["hook_type_stats"]:
            self.integration_stats["hook_type_stats"][hook_type] = {
                "total": 0,
                "successful": 0,
                "failed": 0
            }

        stats = self.integration_stats["hook_type_stats"][hook_type]
        stats["total"] += len(results)
        stats["successful"] += sum(1 for r in results if r.success)
        stats["failed"] += sum(1 for r in results if r.failed)

        # Update category statistics
        if category:
            if category not in self.integration_stats["category_stats"]:
                self.integration_stats["category_stats"][category] = {
                    "total": 0,
                    "successful": 0,
                    "failed": 0
                }

            cat_stats = self.integration_stats["category_stats"][category]
            cat_stats["total"] += len(results)
            cat_stats["successful"] += sum(1 for r in results if r.success)
            cat_stats["failed"] += sum(1 for r in results if r.failed)

    def get_sdk_hooks(self) -> dict[str, Any] | None:
        """
        Get SDK-compatible hooks for integration with ClaudeAgentOptions.

        Returns:
            Dictionary of hooks compatible with SDK format, or None if not available
        """
        if not self.sdk_bridge or not self.initialized:
            return None

        try:
            # Create hook configuration for SDK
            hook_config = {
                "PreToolUse": ["tool_execution", "mcp_tool_execution"],
                "PostToolUse": ["tool_execution", "mcp_tool_execution"],
                "UserPromptSubmit": ["agent_communication", "workflow_orchestration"],
                "Stop": ["agent_communication", "session_lifecycle"],
                "SubagentStop": ["agent_communication", "agent_handoff"]
            }

            # Create SDK hook matchers
            sdk_hooks = self.sdk_bridge.create_hook_matchers(hook_config)

            self.logger.info("SDK hooks generated for integration",
                           hook_types=list(sdk_hooks.keys()))

            return sdk_hooks

        except Exception as e:
            self.logger.error(f"SDK hooks generation failed: {str(e)}",
                            error=str(e),
                            error_type=type(e).__name__)
            return None

    def get_hook_statistics(self) -> dict[str, Any]:
        """Get comprehensive hook system statistics."""
        if not self.initialized:
            return {"message": "Hook integration manager not initialized"}

        stats = {
            "integration_stats": self.integration_stats.copy(),
            "hook_managers": {},
            "category_details": {}
        }

        # Get statistics from each hook manager
        for category, manager in self.hook_managers.items():
            manager_stats = manager.get_hook_stats()
            stats["hook_managers"][category] = manager_stats

        # Get detailed statistics from specific hooks
        if self.tool_hooks.get("execution"):
            stats["category_details"]["tool_execution"] = self.tool_hooks["execution"].get_tool_stats()

        if self.mcp_hooks.get("message"):
            stats["category_details"]["mcp_messages"] = self.mcp_hooks["message"].get_mcp_message_stats()

        if self.mcp_hooks.get("tool_execution"):
            stats["category_details"]["mcp_tools"] = self.mcp_hooks["tool_execution"].get_mcp_tool_stats()

        if self.mcp_hooks.get("session"):
            stats["category_details"]["mcp_sessions"] = self.mcp_hooks["session"].get_mcp_session_stats()

        return stats

    async def shutdown(self):
        """Shutdown all hook systems and cleanup resources."""
        try:
            self.logger.info("Shutting down hook integration manager...")

            # Cleanup hook managers
            for category, manager in self.hook_managers.items():
                # Get final statistics before cleanup
                stats = manager.get_hook_stats()
                self.logger.info(f"Hook manager shutdown: {category}",
                                total_executions=stats.get("total_executions", 0))

            # Clear all hooks and managers
            self.hook_managers.clear()
            self.tool_hooks.clear()
            self.agent_hooks.clear()
            self.workflow_hooks.clear()
            self.session_hooks.clear()
            self.monitoring_hooks.clear()
            self.sdk_hooks.clear()
            self.mcp_hooks.clear()

            self.sdk_bridge = None
            self.initialized = False

            self.logger.info("Hook integration manager shutdown complete")

        except Exception as e:
            self.logger.error(f"Hook integration manager shutdown failed: {str(e)}",
                            error=str(e),
                            error_type=type(e).__name__)
