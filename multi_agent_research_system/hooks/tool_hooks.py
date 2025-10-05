"""
Tool Execution Hooks for Multi-Agent Research System

Provides comprehensive monitoring and tracking of tool execution,
including performance metrics, success rates, and detailed logging.
"""

import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

# Add parent directory to path for proper imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent_logging import ToolUseLogger

from .base_hooks import BaseHook, HookContext, HookPriority, HookResult, HookStatus


@dataclass
class ToolExecutionMetrics:
    """Metrics for tool execution performance."""
    tool_name: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    max_execution_time: float = 0.0
    total_data_processed: int = 0
    last_execution: datetime | None = None
    error_types: dict[str, int] = None

    def __post_init__(self):
        if self.error_types is None:
            self.error_types = {}

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_executions == 0:
            return 0.0
        return (self.successful_executions / self.total_executions) * 100

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate as percentage."""
        return 100.0 - self.success_rate

    def update_execution(self, execution_time: float, success: bool, error_type: str | None = None, data_size: int = 0):
        """Update metrics with new execution data."""
        self.total_executions += 1
        self.last_execution = datetime.now()
        self.total_data_processed += data_size

        if success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1
            if error_type:
                self.error_types[error_type] = self.error_types.get(error_type, 0) + 1

        # Update execution time statistics
        self.min_execution_time = min(self.min_execution_time, execution_time)
        self.max_execution_time = max(self.max_execution_time, execution_time)

        if self.total_executions == 1:
            self.average_execution_time = execution_time
        else:
            self.average_execution_time = (
                (self.average_execution_time * (self.total_executions - 1) + execution_time) /
                self.total_executions
            )


class ToolExecutionHook(BaseHook):
    """Hook for monitoring tool execution events."""

    def __init__(self, enabled: bool = True, timeout: float = 30.0):
        super().__init__(
            name="tool_execution_monitor",
            hook_type="tool_execution",
            priority=HookPriority.HIGH,
            timeout=timeout,
            enabled=enabled,
            retry_count=1
        )
        self.tool_logger = ToolUseLogger()
        self.tool_metrics: dict[str, ToolExecutionMetrics] = {}
        self.active_executions: dict[str, dict[str, Any]] = {}

    async def execute(self, context: HookContext) -> HookResult:
        """Execute tool execution monitoring."""
        try:
            tool_name = context.metadata.get("tool_name", "unknown")
            tool_input = context.metadata.get("tool_input", {})
            tool_use_id = context.metadata.get("tool_use_id", "unknown")
            execution_phase = context.metadata.get("execution_phase", "unknown")

            self.logger.info(f"Tool execution monitoring: {tool_name} - {execution_phase}",
                           tool_name=tool_name,
                           tool_use_id=tool_use_id,
                           execution_phase=execution_phase,
                           session_id=context.session_id)

            if execution_phase == "start":
                return await self._handle_tool_start(context, tool_name, tool_input, tool_use_id)
            elif execution_phase == "complete":
                return await self._handle_tool_complete(context, tool_name, tool_use_id)
            elif execution_phase == "error":
                return await self._handle_tool_error(context, tool_name, tool_use_id)
            else:
                return HookResult(
                    hook_name=self.name,
                    hook_type=self.hook_type,
                    status=HookStatus.COMPLETED,
                    execution_id=context.execution_id,
                    start_time=context.timestamp,
                    end_time=datetime.now(),
                    result_data={"message": f"Unknown execution phase: {execution_phase}"}
                )

        except Exception as e:
            self.logger.error(f"Tool execution hook failed: {str(e)}",
                            error=str(e),
                            error_type=type(e).__name__,
                            session_id=context.session_id)
            return HookResult(
                hook_name=self.name,
                hook_type=self.hook_type,
                status=HookStatus.FAILED,
                execution_id=context.execution_id,
                start_time=context.timestamp,
                end_time=datetime.now(),
                error_message=str(e),
                error_type=type(e).__name__
            )

    async def _handle_tool_start(
        self,
        context: HookContext,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_use_id: str
    ) -> HookResult:
        """Handle tool execution start event."""
        start_time = time.time()

        # Initialize metrics if needed
        if tool_name not in self.tool_metrics:
            self.tool_metrics[tool_name] = ToolExecutionMetrics(tool_name=tool_name)

        # Track active execution
        self.active_executions[tool_use_id] = {
            "tool_name": tool_name,
            "start_time": start_time,
            "context": context,
            "input_size": len(json.dumps(tool_input, default=str))
        }

        # Use tool logger for detailed logging
        agent_context = {
            "agent_name": context.agent_name,
            "agent_type": context.agent_type
        }
        self.tool_logger.log_pre_tool_use(
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            session_id=context.session_id,
            input_data=tool_input,
            agent_context=agent_context
        )

        return HookResult(
            hook_name=self.name,
            hook_type=self.hook_type,
            status=HookStatus.COMPLETED,
            execution_id=context.execution_id,
            start_time=context.timestamp,
            end_time=datetime.now(),
            result_data={
                "tool_name": tool_name,
                "tool_use_id": tool_use_id,
                "execution_phase": "start",
                "input_size": self.active_executions[tool_use_id]["input_size"]
            }
        )

    async def _handle_tool_complete(self, context: HookContext, tool_name: str, tool_use_id: str) -> HookResult:
        """Handle tool execution completion event."""
        end_time = time.time()
        result_data = context.metadata.get("tool_result", {})
        execution_time = 0.0

        # Calculate execution time
        if tool_use_id in self.active_executions:
            start_time = self.active_executions[tool_use_id]["start_time"]
            execution_time = end_time - start_time
            del self.active_executions[tool_use_id]

        # Update metrics
        if tool_name in self.tool_metrics:
            result_size = len(json.dumps(result_data, default=str))
            self.tool_metrics[tool_name].update_execution(execution_time, True, data_size=result_size)

        # Use tool logger for completion logging
        agent_context = {
            "agent_name": context.agent_name,
            "agent_type": context.agent_type
        }
        self.tool_logger.log_post_tool_use(
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            session_id=context.session_id,
            input_data=context.metadata.get("tool_input", {}),
            result_data=result_data,
            execution_time=execution_time,
            success=True,
            agent_context=agent_context
        )

        return HookResult(
            hook_name=self.name,
            hook_type=self.hook_type,
            status=HookStatus.COMPLETED,
            execution_id=context.execution_id,
            start_time=context.timestamp,
            end_time=datetime.now(),
            execution_time=execution_time,
            result_data={
                "tool_name": tool_name,
                "tool_use_id": tool_use_id,
                "execution_phase": "complete",
                "execution_time": execution_time,
                "result_size": len(json.dumps(result_data, default=str)),
                "success": True
            }
        )

    async def _handle_tool_error(self, context: HookContext, tool_name: str, tool_use_id: str) -> HookResult:
        """Handle tool execution error event."""
        end_time = time.time()
        error_message = context.metadata.get("error_message", "Unknown error")
        error_type = context.metadata.get("error_type", "UnknownError")
        execution_time = 0.0

        # Calculate execution time
        if tool_use_id in self.active_executions:
            start_time = self.active_executions[tool_use_id]["start_time"]
            execution_time = end_time - start_time
            del self.active_executions[tool_use_id]

        # Update metrics
        if tool_name in self.tool_metrics:
            self.tool_metrics[tool_name].update_execution(execution_time, False, error_type)

        # Use tool logger for error logging
        agent_context = {
            "agent_name": context.agent_name,
            "agent_type": context.agent_type
        }
        self.tool_logger.log_post_tool_use(
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            session_id=context.session_id,
            input_data=context.metadata.get("tool_input", {}),
            result_data={"error": error_message},
            execution_time=execution_time,
            success=False,
            agent_context=agent_context
        )

        return HookResult(
            hook_name=self.name,
            hook_type=self.hook_type,
            status=HookStatus.COMPLETED,
            execution_id=context.execution_id,
            start_time=context.timestamp,
            end_time=datetime.now(),
            execution_time=execution_time,
            result_data={
                "tool_name": tool_name,
                "tool_use_id": tool_use_id,
                "execution_phase": "error",
                "execution_time": execution_time,
                "error_message": error_message,
                "error_type": error_type,
                "success": False
            }
        )

    def get_tool_metrics(self, tool_name: str | None = None) -> dict[str, Any]:
        """Get performance metrics for tools."""
        if tool_name:
            if tool_name not in self.tool_metrics:
                return {}
            metrics = self.tool_metrics[tool_name]
            return {
                "tool_name": metrics.tool_name,
                "total_executions": metrics.total_executions,
                "successful_executions": metrics.successful_executions,
                "failed_executions": metrics.failed_executions,
                "success_rate": round(metrics.success_rate, 2),
                "failure_rate": round(metrics.failure_rate, 2),
                "average_execution_time": round(metrics.average_execution_time, 3),
                "min_execution_time": metrics.min_execution_time if metrics.min_execution_time != float('inf') else 0.0,
                "max_execution_time": metrics.max_execution_time,
                "total_data_processed": metrics.total_data_processed,
                "last_execution": metrics.last_execution.isoformat() if metrics.last_execution else None,
                "error_types": metrics.error_types.copy()
            }
        else:
            # Return metrics for all tools
            return {
                tool_name: self.get_tool_metrics(tool_name)
                for tool_name in self.tool_metrics.keys()
            }

    def get_active_executions(self) -> dict[str, dict[str, Any]]:
        """Get currently active tool executions."""
        current_time = time.time()
        active = {}

        for tool_use_id, execution in self.active_executions.items():
            execution_duration = current_time - execution["start_time"]
            active[tool_use_id] = {
                **execution,
                "execution_duration": execution_duration,
                "start_time_iso": datetime.fromtimestamp(execution["start_time"]).isoformat()
            }

        return active

    def get_slow_tools(self, threshold_seconds: float = 5.0) -> list[str]:
        """Get list of tools with average execution time above threshold."""
        slow_tools = []
        for tool_name, metrics in self.tool_metrics.items():
            if metrics.average_execution_time > threshold_seconds:
                slow_tools.append(tool_name)
        return sorted(slow_tools, key=lambda t: self.tool_metrics[t].average_execution_time, reverse=True)

    def get_problematic_tools(self, failure_rate_threshold: float = 20.0) -> list[str]:
        """Get list of tools with failure rate above threshold."""
        problematic_tools = []
        for tool_name, metrics in self.tool_metrics.items():
            if metrics.failure_rate > failure_rate_threshold:
                problematic_tools.append(tool_name)
        return sorted(problematic_tools, key=lambda t: self.tool_metrics[t].failure_rate, reverse=True)


class ToolPerformanceMonitor(BaseHook):
    """Hook for monitoring overall tool performance and generating alerts."""

    def __init__(self, slow_threshold: float = 10.0, failure_rate_threshold: float = 25.0, enabled: bool = True, timeout: float = 15.0):
        super().__init__(
            name="tool_performance_monitor",
            hook_type="tool_performance",
            priority=HookPriority.NORMAL,
            timeout=timeout,
            enabled=enabled,
            retry_count=0
        )
        self.slow_threshold = slow_threshold
        self.failure_rate_threshold = failure_rate_threshold
        self.performance_alerts: list[dict[str, Any]] = []
        self.max_alerts = 100

    async def execute(self, context: HookContext) -> HookResult:
        """Execute tool performance monitoring."""
        try:
            # This would typically be called periodically or on tool completion
            current_time = time.time()

            # Get tool execution metrics (would be passed in context or retrieved from a shared store)
            tool_metrics = context.metadata.get("tool_metrics", {})

            # Generate performance alerts
            alerts = self._generate_performance_alerts(tool_metrics, current_time)

            # Log performance summary
            self.logger.info("Tool performance monitoring completed",
                           total_tools=len(tool_metrics),
                           alerts_generated=len(alerts),
                           session_id=context.session_id)

            return HookResult(
                hook_name=self.name,
                hook_type=self.hook_type,
                status=HookStatus.COMPLETED,
                execution_id=context.execution_id,
                start_time=context.timestamp,
                end_time=datetime.now(),
                result_data={
                    "tools_monitored": len(tool_metrics),
                    "alerts_generated": len(alerts),
                    "alerts": alerts,
                    "performance_summary": self._generate_performance_summary(tool_metrics)
                }
            )

        except Exception as e:
            self.logger.error(f"Tool performance monitor failed: {str(e)}",
                            error=str(e),
                            error_type=type(e).__name__,
                            session_id=context.session_id)
            return HookResult(
                hook_name=self.name,
                hook_type=self.hook_type,
                status=HookStatus.FAILED,
                execution_id=context.execution_id,
                start_time=context.timestamp,
                end_time=datetime.now(),
                error_message=str(e),
                error_type=type(e).__name__
            )

    def _generate_performance_alerts(self, tool_metrics: dict[str, Any], current_time: float) -> list[dict[str, Any]]:
        """Generate performance alerts based on tool metrics."""
        alerts = []

        for tool_name, metrics in tool_metrics.items():
            # Check for slow execution
            avg_time = metrics.get("average_execution_time", 0.0)
            if avg_time > self.slow_threshold:
                alert = {
                    "type": "slow_tool",
                    "tool_name": tool_name,
                    "average_execution_time": avg_time,
                    "threshold": self.slow_threshold,
                    "severity": "high" if avg_time > self.slow_threshold * 2 else "medium",
                    "timestamp": current_time,
                    "message": f"Tool {tool_name} is slow (avg: {avg_time:.2f}s, threshold: {self.slow_threshold}s)"
                }
                alerts.append(alert)

            # Check for high failure rate
            failure_rate = metrics.get("failure_rate", 0.0)
            if failure_rate > self.failure_rate_threshold:
                alert = {
                    "type": "high_failure_rate",
                    "tool_name": tool_name,
                    "failure_rate": failure_rate,
                    "threshold": self.failure_rate_threshold,
                    "severity": "critical" if failure_rate > 50.0 else "high",
                    "timestamp": current_time,
                    "message": f"Tool {tool_name} has high failure rate ({failure_rate:.1f}%, threshold: {self.failure_rate_threshold}%)"
                }
                alerts.append(alert)

        # Store alerts
        self.performance_alerts.extend(alerts)
        if len(self.performance_alerts) > self.max_alerts:
            self.performance_alerts = self.performance_alerts[-self.max_alerts:]

        return alerts

    def _generate_performance_summary(self, tool_metrics: dict[str, Any]) -> dict[str, Any]:
        """Generate overall performance summary."""
        if not tool_metrics:
            return {"message": "No tool metrics available"}

        total_executions = sum(m.get("total_executions", 0) for m in tool_metrics.values())
        total_successful = sum(m.get("successful_executions", 0) for m in tool_metrics.values())
        total_failed = sum(m.get("failed_executions", 0) for m in tool_metrics.values())
        avg_execution_time = sum(m.get("average_execution_time", 0.0) for m in tool_metrics.values()) / len(tool_metrics)

        slow_tools = [name for name, metrics in tool_metrics.items()
                     if metrics.get("average_execution_time", 0.0) > self.slow_threshold]
        problematic_tools = [name for name, metrics in tool_metrics.items()
                            if metrics.get("failure_rate", 0.0) > self.failure_rate_threshold]

        return {
            "total_tools": len(tool_metrics),
            "total_executions": total_executions,
            "total_successful": total_successful,
            "total_failed": total_failed,
            "overall_success_rate": (total_successful / total_executions * 100) if total_executions > 0 else 0.0,
            "average_execution_time": round(avg_execution_time, 3),
            "slow_tools": slow_tools,
            "problematic_tools": problematic_tools,
            "performance_grade": self._calculate_performance_grade(tool_metrics)
        }

    def _calculate_performance_grade(self, tool_metrics: dict[str, Any]) -> str:
        """Calculate overall performance grade."""
        if not tool_metrics:
            return "N/A"

        total_success_rate = 0.0
        avg_execution_time = 0.0

        for metrics in tool_metrics.values():
            total_success_rate += metrics.get("success_rate", 0.0)
            avg_execution_time += metrics.get("average_execution_time", 0.0)

        total_success_rate /= len(tool_metrics)
        avg_execution_time /= len(tool_metrics)

        # Grade based on success rate and speed
        if total_success_rate >= 95.0 and avg_execution_time <= 2.0:
            return "A"
        elif total_success_rate >= 90.0 and avg_execution_time <= 5.0:
            return "B"
        elif total_success_rate >= 80.0 and avg_execution_time <= 10.0:
            return "C"
        elif total_success_rate >= 70.0:
            return "D"
        else:
            return "F"

    def get_recent_alerts(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent performance alerts."""
        return self.performance_alerts[-limit:] if self.performance_alerts else []
