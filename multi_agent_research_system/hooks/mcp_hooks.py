"""
MCP-Aware Hooks for Multi-Agent Research System

Provides comprehensive hooks specifically designed to work within the
Model Context Protocol (MCP) structure as required by the Claude Agent SDK.
These hooks understand MCP message formats, tool execution patterns, and
control protocol integration.
"""

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .base_hooks import BaseHook, HookContext, HookPriority, HookResult, HookStatus

# Add parent directory to path for proper imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@dataclass
class MCPMessageInfo:
    """Information about an MCP message with detailed context."""
    message_type: str
    subtype: str
    timestamp: datetime
    session_id: str
    request_id: str | None = None
    parent_tool_use_id: str | None = None
    content_size: int = 0
    processing_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPToolExecution:
    """MCP tool execution information with comprehensive tracking."""
    tool_name: str
    tool_use_id: str
    input_data: dict[str, Any]
    start_time: datetime
    end_time: datetime | None = None
    execution_time: float | None = None
    result_size: int = 0
    success: bool = False
    error_message: str | None = None
    mcp_server: str | None = None
    permission_required: bool = False
    permission_granted: bool | None = None


class MCPMessageHook(BaseHook):
    """
    Hook for monitoring MCP message flow and ensuring proper MCP structure compliance.

    This hook tracks all MCP messages including control requests, responses,
    and content messages to ensure they follow proper MCP protocols.
    """

    def __init__(self, enabled: bool = True, timeout: float = 10.0):
        super().__init__(
            name="mcp_message_monitor",
            hook_type="mcp_message_processing",
            priority=HookPriority.HIGH,
            timeout=timeout,
            enabled=enabled,
            retry_count=1
        )
        self.message_history: list[MCPMessageInfo] = []
        self.max_history = 5000
        self.message_type_stats: dict[str, int] = {}
        self.session_message_counts: dict[str, int] = {}

    async def execute(self, context: HookContext) -> HookResult:
        """Execute MCP message processing and validation."""
        try:
            mcp_message = context.metadata.get("mcp_message", {})
            message_type = mcp_message.get("type", "unknown")
            subtype = mcp_message.get("subtype", "unknown")
            session_id = context.session_id

            self.logger.info(f"MCP message processing: {message_type}/{subtype}",
                           message_type=message_type,
                           subtype=subtype,
                           session_id=session_id)

            # Create message info
            message_info = MCPMessageInfo(
                message_type=message_type,
                subtype=subtype,
                timestamp=datetime.now(),
                session_id=session_id,
                request_id=mcp_message.get("request_id"),
                parent_tool_use_id=mcp_message.get("parent_tool_use_id"),
                content_size=len(json.dumps(mcp_message, default=str)),
                metadata=mcp_message.copy()
            )

            # Validate MCP structure
            validation_result = await self._validate_mcp_structure(message_info)

            # Track message statistics
            self._update_message_stats(message_info)

            # Store in history
            self._store_message_info(message_info)

            return HookResult(
                hook_name=self.name,
                hook_type=self.hook_type,
                status=HookStatus.COMPLETED,
                execution_id=context.execution_id,
                start_time=context.timestamp,
                end_time=datetime.now(),
                result_data={
                    "message_type": message_type,
                    "subtype": subtype,
                    "validation_passed": validation_result["valid"],
                    "validation_issues": validation_result["issues"],
                    "session_message_count": self.session_message_counts.get(session_id, 0),
                    "total_messages_processed": len(self.message_history)
                }
            )

        except Exception as e:
            self.logger.error(f"MCP message processing failed: {str(e)}",
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

    async def _validate_mcp_structure(self, message_info: MCPMessageInfo) -> dict[str, Any]:
        """Validate MCP message structure and compliance."""
        issues = []
        valid = True

        # Validate control message structure
        if message_info.message_type == "control_request":
            if not message_info.request_id:
                issues.append("Missing request_id in control_request")
                valid = False

            request = message_info.metadata.get("request", {})
            if not request.get("subtype"):
                issues.append("Missing subtype in control request")
                valid = False

        # Validate response message structure
        elif message_info.message_type == "control_response":
            response = message_info.metadata.get("response", {})
            if not response.get("subtype"):
                issues.append("Missing subtype in control response")
                valid = False

        # Validate content message structure
        elif message_info.message_type in ["user", "assistant", "system"]:
            message = message_info.metadata.get("message", {})
            if not message.get("role"):
                issues.append(f"Missing role in {message_info.message_type} message")
                valid = False

            content = message.get("content")
            if not content:
                issues.append(f"Missing content in {message_info.message_type} message")
                valid = False

        return {
            "valid": valid,
            "issues": issues,
            "message_type": message_info.message_type,
            "subtype": message_info.subtype
        }

    def _update_message_stats(self, message_info: MCPMessageInfo):
        """Update message statistics."""
        # Update message type stats
        key = f"{message_info.message_type}/{message_info.subtype}"
        self.message_type_stats[key] = self.message_type_stats.get(key, 0) + 1

        # Update session message counts
        self.session_message_counts[message_info.session_id] = \
            self.session_message_counts.get(message_info.session_id, 0) + 1

    def _store_message_info(self, message_info: MCPMessageInfo):
        """Store message info and maintain history size."""
        self.message_history.append(message_info)
        if len(self.message_history) > self.max_history:
            self.message_history = self.message_history[-self.max_history:]

    def get_mcp_message_stats(self) -> dict[str, Any]:
        """Get comprehensive MCP message statistics."""
        if not self.message_history:
            return {"message": "No MCP messages processed"}

        # Calculate statistics
        total_messages = len(self.message_history)
        active_sessions = len(self.session_message_counts)
        avg_messages_per_session = total_messages / active_sessions if active_sessions > 0 else 0

        # Message type distribution
        message_types = {}
        for key, count in self.message_type_stats.items():
            message_type, subtype = key.split("/", 1)
            if message_type not in message_types:
                message_types[message_type] = {}
            message_types[message_type][subtype] = count

        # Recent messages (last hour)
        cutoff_time = datetime.now().timestamp() - 3600
        recent_messages = [
            msg for msg in self.message_history
            if msg.timestamp.timestamp() > cutoff_time
        ]

        return {
            "total_messages": total_messages,
            "active_sessions": active_sessions,
            "avg_messages_per_session": round(avg_messages_per_session, 2),
            "message_type_distribution": message_types,
            "recent_messages_last_hour": len(recent_messages),
            "messages_per_hour": len(recent_messages),
            "most_active_session": max(self.session_message_counts.items(),
                                     key=lambda x: x[1])[0] if self.session_message_counts else None,
            "message_types_by_frequency": dict(sorted(self.message_type_stats.items(),
                                                     key=lambda x: x[1], reverse=True)[:10])
        }


class MCPToolExecutionHook(BaseHook):
    """
    Hook for monitoring MCP tool execution within the MCP framework.

    This hook tracks tool usage, execution times, success rates, and
    ensures tools are executed properly within the MCP structure.
    """

    def __init__(self, enabled: bool = True, timeout: float = 30.0):
        super().__init__(
            name="mcp_tool_execution_monitor",
            hook_type="mcp_tool_execution",
            priority=HookPriority.HIGH,
            timeout=timeout,
            enabled=enabled,
            retry_count=0
        )
        self.tool_executions: list[MCPToolExecution] = []
        self.max_executions = 2000
        self.tool_stats: dict[str, dict[str, Any]] = {}

    async def execute(self, context: HookContext) -> HookResult:
        """Execute MCP tool execution monitoring."""
        try:
            execution_phase = context.metadata.get("execution_phase", "unknown")
            tool_name = context.metadata.get("tool_name", "unknown")
            tool_use_id = context.metadata.get("tool_use_id")
            tool_input = context.metadata.get("tool_input", {})

            self.logger.info(f"MCP tool execution: {tool_name} - {execution_phase}",
                           tool_name=tool_name,
                           execution_phase=execution_phase,
                           tool_use_id=tool_use_id,
                           session_id=context.session_id)

            if execution_phase == "start":
                return await self._handle_tool_start(context, tool_name, tool_use_id, tool_input)
            elif execution_phase == "complete":
                return await self._handle_tool_complete(context, tool_name, tool_use_id)
            else:
                return await self._handle_tool_generic(context, tool_name, execution_phase)

        except Exception as e:
            self.logger.error(f"MCP tool execution monitoring failed: {str(e)}",
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
        tool_use_id: str,
        tool_input: dict[str, Any]
    ) -> HookResult:
        """Handle tool execution start."""
        execution = MCPToolExecution(
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            input_data=tool_input,
            start_time=datetime.now(),
            mcp_server=context.metadata.get("mcp_server"),
            permission_required=context.metadata.get("permission_required", False)
        )

        # Store execution for completion tracking
        self.tool_executions.append(execution)

        # Log tool start
        self.logger.info(f"MCP tool execution started: {tool_name}",
                        tool_name=tool_name,
                        tool_use_id=tool_use_id,
                        input_size=len(json.dumps(tool_input, default=str)),
                        mcp_server=execution.mcp_server,
                        permission_required=execution.permission_required)

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
                "input_size": len(json.dumps(tool_input, default=str)),
                "mcp_server": execution.mcp_server,
                "permission_required": execution.permission_required
            }
        )

    async def _handle_tool_complete(
        self,
        context: HookContext,
        tool_name: str,
        tool_use_id: str
    ) -> HookResult:
        """Handle tool execution completion."""
        # Find the corresponding execution
        execution = None
        for exec_record in reversed(self.tool_executions):
            if exec_record.tool_use_id == tool_use_id and exec_record.end_time is None:
                execution = exec_record
                break

        if not execution:
            # Create a new execution record if not found
            execution = MCPToolExecution(
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                input_data={},
                start_time=datetime.now()
            )
            self.tool_executions.append(execution)

        # Update completion info
        execution.end_time = datetime.now()
        execution.execution_time = (execution.end_time - execution.start_time).total_seconds()
        execution.success = context.metadata.get("success", True)
        execution.error_message = context.metadata.get("error_message")
        execution.result_size = len(str(context.metadata.get("result", "")))
        execution.permission_granted = context.metadata.get("permission_granted")

        # Update tool statistics
        self._update_tool_stats(execution)

        # Log completion
        self.logger.info(f"MCP tool execution completed: {tool_name}",
                        tool_name=tool_name,
                        tool_use_id=tool_use_id,
                        execution_time=execution.execution_time,
                        success=execution.success,
                        result_size=execution.result_size)

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
                "execution_phase": "complete",
                "execution_time": execution.execution_time,
                "success": execution.success,
                "result_size": execution.result_size,
                "error_message": execution.error_message
            }
        )

    async def _handle_tool_generic(
        self,
        context: HookContext,
        tool_name: str,
        execution_phase: str
    ) -> HookResult:
        """Handle generic tool execution phases."""
        return HookResult(
            hook_name=self.name,
            hook_type=self.hook_type,
            status=HookStatus.COMPLETED,
            execution_id=context.execution_id,
            start_time=context.timestamp,
            end_time=datetime.now(),
            result_data={
                "tool_name": tool_name,
                "execution_phase": execution_phase,
                "message": f"Generic tool execution phase: {execution_phase}"
            }
        )

    def _update_tool_stats(self, execution: MCPToolExecution):
        """Update tool execution statistics."""
        tool_name = execution.tool_name

        if tool_name not in self.tool_stats:
            self.tool_stats[tool_name] = {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "total_execution_time": 0.0,
                "average_execution_time": 0.0,
                "min_execution_time": float('inf'),
                "max_execution_time": 0.0,
                "total_result_size": 0,
                "average_result_size": 0.0,
                "last_execution": None,
                "mcp_servers": set(),
                "permission_required_count": 0
            }

        stats = self.tool_stats[tool_name]
        stats["total_executions"] += 1
        stats["last_execution"] = execution.end_time

        if execution.success:
            stats["successful_executions"] += 1
        else:
            stats["failed_executions"] += 1

        # Update execution time statistics
        if execution.execution_time is not None:
            stats["total_execution_time"] += execution.execution_time
            stats["min_execution_time"] = min(stats["min_execution_time"], execution.execution_time)
            stats["max_execution_time"] = max(stats["max_execution_time"], execution.execution_time)
            stats["average_execution_time"] = (
                stats["total_execution_time"] / stats["total_executions"]
            )

        # Update result size statistics
        stats["total_result_size"] += execution.result_size
        stats["average_result_size"] = stats["total_result_size"] / stats["total_executions"]

        # Track MCP servers
        if execution.mcp_server:
            stats["mcp_servers"].add(execution.mcp_server)

        # Track permission requirements
        if execution.permission_required:
            stats["permission_required_count"] += 1

        # Maintain execution history size
        if len(self.tool_executions) > self.max_executions:
            self.tool_executions = self.tool_executions[-self.max_executions:]

    def get_mcp_tool_stats(self) -> dict[str, Any]:
        """Get comprehensive MCP tool execution statistics."""
        if not self.tool_stats:
            return {"message": "No MCP tool executions recorded"}

        total_executions = sum(stats["total_executions"] for stats in self.tool_stats.values())
        successful_executions = sum(stats["successful_executions"] for stats in self.tool_stats.values())

        # Calculate tool performance rankings
        tool_performance = []
        for tool_name, stats in self.tool_stats.items():
            success_rate = (stats["successful_executions"] / stats["total_executions"] * 100) if stats["total_executions"] > 0 else 0
            tool_performance.append({
                "tool_name": tool_name,
                "executions": stats["total_executions"],
                "success_rate": round(success_rate, 2),
                "avg_execution_time": round(stats["average_execution_time"], 3),
                "avg_result_size": round(stats["average_result_size"], 0)
            })

        # Sort by execution count
        tool_performance.sort(key=lambda x: x["executions"], reverse=True)

        # Find slowest tools
        slowest_tools = sorted(
            [(name, stats["average_execution_time"]) for name, stats in self.tool_stats.items() if stats["average_execution_time"] > 0],
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return {
            "total_tools": len(self.tool_stats),
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "overall_success_rate": round((successful_executions / total_executions * 100) if total_executions > 0 else 0, 2),
            "average_execution_time": round(
                sum(stats["average_execution_time"] for stats in self.tool_stats.values()) / len(self.tool_stats), 3
            ),
            "most_used_tools": tool_performance[:10],
            "slowest_tools": [
                {"tool_name": name, "avg_time": round(time, 3)}
                for name, time in slowest_tools
            ],
            "tools_with_permissions": [
                name for name, stats in self.tool_stats.items()
                if stats["permission_required_count"] > 0
            ],
            "detailed_stats": {
                name: {
                    "executions": stats["total_executions"],
                    "success_rate": round((stats["successful_executions"] / stats["total_executions"] * 100) if stats["total_executions"] > 0 else 0, 2),
                    "avg_time": round(stats["average_execution_time"], 3),
                    "min_time": round(stats["min_execution_time"], 3) if stats["min_execution_time"] != float('inf') else 0,
                    "max_time": round(stats["max_execution_time"], 3),
                    "avg_result_size": round(stats["average_result_size"], 0),
                    "mcp_servers": list(stats["mcp_servers"]),
                    "permission_required": stats["permission_required_count"] > 0
                }
                for name, stats in self.tool_stats.items()
            }
        }


class MCPSessionHook(BaseHook):
    """
    Hook for monitoring MCP session lifecycle and state management.

    This hook tracks session creation, resumption, termination, and ensures
    proper MCP session protocol compliance.
    """

    def __init__(self, timeout: float = 30.0, enabled: bool = True):
        super().__init__(
            name="mcp_session_monitor",
            hook_type="mcp_session_management",
            priority=HookPriority.NORMAL,
            timeout=15.0,
            enabled=enabled,
            retry_count=1
        )
        self.active_sessions: dict[str, dict[str, Any]] = {}
        self.session_history: list[dict[str, Any]] = []
        self.max_history = 1000

    async def execute(self, context: HookContext) -> HookResult:
        """Execute MCP session management monitoring."""
        try:
            session_event = context.metadata.get("session_event", "unknown")
            session_id = context.session_id

            self.logger.info(f"MCP session event: {session_event}",
                           session_event=session_event,
                           session_id=session_id)

            if session_event == "creation":
                return await self._handle_session_creation(context, session_id)
            elif session_event == "resumption":
                return await self._handle_session_resumption(context, session_id)
            elif session_event == "termination":
                return await self._handle_session_termination(context, session_id)
            elif session_event == "activity":
                return await self._handle_session_activity(context, session_id)
            else:
                return await self._handle_session_generic(context, session_id, session_event)

        except Exception as e:
            self.logger.error(f"MCP session monitoring failed: {str(e)}",
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

    async def _handle_session_creation(self, context: HookContext, session_id: str) -> HookResult:
        """Handle session creation."""
        session_info = {
            "session_id": session_id,
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "status": "active",
            "message_count": 0,
            "tool_executions": 0,
            "agent_handoffs": 0,
            "total_execution_time": 0.0,
            "metadata": context.metadata.copy()
        }

        self.active_sessions[session_id] = session_info

        self.logger.info(f"MCP session created: {session_id}",
                        session_id=session_id,
                        initial_metadata=session_info["metadata"])

        return HookResult(
            hook_name=self.name,
            hook_type=self.hook_type,
            status=HookStatus.COMPLETED,
            execution_id=context.execution_id,
            start_time=context.timestamp,
            end_time=datetime.now(),
            result_data={
                "session_event": "creation",
                "session_id": session_id,
                "active_sessions": len(self.active_sessions)
            }
        )

    async def _handle_session_resumption(self, context: HookContext, session_id: str) -> HookResult:
        """Handle session resumption."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["last_activity"] = datetime.now()
            self.active_sessions[session_id]["status"] = "resumed"
            self.active_sessions[session_id]["resumption_count"] = \
                self.active_sessions[session_id].get("resumption_count", 0) + 1
        else:
            # Create session info for resumed session that wasn't tracked
            await self._handle_session_creation(context, session_id)

        self.logger.info(f"MCP session resumed: {session_id}",
                        session_id=session_id,
                        resumption_count=self.active_sessions[session_id].get("resumption_count", 1))

        return HookResult(
            hook_name=self.name,
            hook_type=self.hook_type,
            status=HookStatus.COMPLETED,
            execution_id=context.execution_id,
            start_time=context.timestamp,
            end_time=datetime.now(),
            result_data={
                "session_event": "resumption",
                "session_id": session_id,
                "resumption_count": self.active_sessions[session_id].get("resumption_count", 1)
            }
        )

    async def _handle_session_termination(self, context: HookContext, session_id: str) -> HookResult:
        """Handle session termination."""
        if session_id in self.active_sessions:
            session_info = self.active_sessions[session_id]
            session_info["terminated_at"] = datetime.now()
            session_info["status"] = "terminated"
            session_info["termination_reason"] = context.metadata.get("termination_reason", "unknown")

            # Calculate session duration
            duration = (session_info["terminated_at"] - session_info["created_at"]).total_seconds()
            session_info["duration_seconds"] = duration

            # Move to history
            self.session_history.append(session_info.copy())
            del self.active_sessions[session_id]

            # Maintain history size
            if len(self.session_history) > self.max_history:
                self.session_history = self.session_history[-self.max_history:]

            self.logger.info(f"MCP session terminated: {session_id}",
                            session_id=session_id,
                            duration_seconds=duration,
                            termination_reason=session_info["termination_reason"])

        return HookResult(
            hook_name=self.name,
            hook_type=self.hook_type,
            status=HookStatus.COMPLETED,
            execution_id=context.execution_id,
            start_time=context.timestamp,
            end_time=datetime.now(),
            result_data={
                "session_event": "termination",
                "session_id": session_id,
                "active_sessions": len(self.active_sessions)
            }
        )

    async def _handle_session_activity(self, context: HookContext, session_id: str) -> HookResult:
        """Handle session activity updates."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["last_activity"] = datetime.now()

            activity_type = context.metadata.get("activity_type", "unknown")
            if activity_type == "message":
                self.active_sessions[session_id]["message_count"] += 1
            elif activity_type == "tool_execution":
                self.active_sessions[session_id]["tool_executions"] += 1
            elif activity_type == "agent_handoff":
                self.active_sessions[session_id]["agent_handoffs"] += 1

        return HookResult(
            hook_name=self.name,
            hook_type=self.hook_type,
            status=HookStatus.COMPLETED,
            execution_id=context.execution_id,
            start_time=context.timestamp,
            end_time=datetime.now(),
            result_data={
                "session_event": "activity",
                "session_id": session_id
            }
        )

    async def _handle_session_generic(self, context: HookContext, session_id: str, session_event: str) -> HookResult:
        """Handle generic session events."""
        return HookResult(
            hook_name=self.name,
            hook_type=self.hook_type,
            status=HookStatus.COMPLETED,
            execution_id=context.execution_id,
            start_time=context.timestamp,
            end_time=datetime.now(),
            result_data={
                "session_event": session_event,
                "session_id": session_id,
                "message": f"Generic session event: {session_event}"
            }
        )

    def get_mcp_session_stats(self) -> dict[str, Any]:
        """Get comprehensive MCP session statistics."""
        return {
            "active_sessions": len(self.active_sessions),
            "total_sessions_in_history": len(self.session_history),
            "active_session_details": {
                session_id: {
                    "created_at": info["created_at"].isoformat(),
                    "last_activity": info["last_activity"].isoformat(),
                    "status": info["status"],
                    "message_count": info["message_count"],
                    "tool_executions": info["tool_executions"],
                    "agent_handoffs": info["agent_handoffs"],
                    "duration_minutes": round((datetime.now() - info["created_at"]).total_seconds() / 60, 2),
                    "resumption_count": info.get("resumption_count", 0)
                }
                for session_id, info in self.active_sessions.items()
            },
            "recent_terminated_sessions": [
                {
                    "session_id": session["session_id"],
                    "duration_minutes": round(session.get("duration_seconds", 0) / 60, 2),
                    "message_count": session["message_count"],
                    "tool_executions": session["tool_executions"],
                    "termination_reason": session.get("termination_reason", "unknown")
                }
                for session in self.session_history[-10:]
            ],
            "session_averages": self._calculate_session_averages()
        }

    def _calculate_session_averages(self) -> dict[str, float]:
        """Calculate average session metrics."""
        if not self.session_history:
            return {}

        total_sessions = len(self.session_history)
        total_duration = sum(session.get("duration_seconds", 0) for session in self.session_history)
        total_messages = sum(session["message_count"] for session in self.session_history)
        total_tools = sum(session["tool_executions"] for session in self.session_history)

        return {
            "avg_duration_minutes": round((total_duration / total_sessions) / 60, 2) if total_sessions > 0 else 0,
            "avg_messages_per_session": round(total_messages / total_sessions, 2) if total_sessions > 0 else 0,
            "avg_tools_per_session": round(total_tools / total_sessions, 2) if total_sessions > 0 else 0
        }
