"""
Session Lifecycle Hooks for Multi-Agent Research System

Provides comprehensive monitoring of session creation, state changes,
recovery operations, and lifecycle management throughout the research process.
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from .base_hooks import BaseHook, HookContext, HookPriority, HookResult, HookStatus

# Add parent directory to path for proper imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent_logging import SessionLifecycleLogger


class SessionState(Enum):
    """Enumeration of session states."""
    CREATING = "creating"
    INITIALIZED = "initialized"
    ACTIVE = "active"
    PAUSED = "paused"
    RESUMING = "resuming"
    COMPLETING = "completing"
    COMPLETED = "completed"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class SessionEventType(Enum):
    """Enumeration of session event types."""
    CREATION = "session_creation"
    RESUMPTION = "session_resumption"
    PAUSE = "session_pause"
    TERMINATION = "session_termination"
    ERROR = "session_error"
    TIMEOUT = "session_timeout"
    STATE_CHANGE = "state_change"
    DATA_UPDATE = "data_update"


@dataclass
class SessionEvent:
    """Represents a session lifecycle event."""
    event_id: str
    timestamp: datetime
    session_id: str
    event_type: SessionEventType
    previous_state: SessionState | None
    new_state: SessionState | None
    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    correlation_id: str | None = None


@dataclass
class SessionMetrics:
    """Metrics for session performance and lifecycle."""
    session_id: str
    created_at: datetime
    last_activity: datetime
    total_duration: float = 0.0
    active_duration: float = 0.0
    paused_duration: float = 0.0
    state_changes: int = 0
    errors_count: int = 0
    recovery_attempts: int = 0
    successful_recoveries: int = 0
    data_size_bytes: int = 0
    workflow_stages_completed: list[str] = field(default_factory=list)
    agents_involved: set[str] = field(default_factory=set)
    tools_executed: dict[str, int] = field(default_factory=dict)

    @property
    def current_duration(self) -> float:
        """Calculate duration from creation to now."""
        return (datetime.now() - self.created_at).total_seconds()

    @property
    def recovery_success_rate(self) -> float:
        """Calculate recovery success rate."""
        if self.recovery_attempts == 0:
            return 100.0
        return (self.successful_recoveries / self.recovery_attempts) * 100

    @property
    def is_active(self) -> bool:
        """Check if session is currently active."""
        return (datetime.now() - self.last_activity).total_seconds() < 300  # 5 minutes


class SessionLifecycleHook(BaseHook):
    """Hook for monitoring session lifecycle events and state management."""

    def __init__(self, timeout: float = 30.0, enabled: bool = True):
        super().__init__(
            name="session_lifecycle_monitor",
            hook_type="session_lifecycle",
            priority=HookPriority.HIGHEST,
            timeout=30.0,
            enabled=enabled,
            retry_count=1
        )
        self.session_logger = SessionLifecycleLogger()
        self.session_events: list[SessionEvent] = []
        self.session_metrics: dict[str, SessionMetrics] = {}
        self.active_sessions: dict[str, dict[str, Any]] = {}
        self.session_snapshots: dict[str, list[dict[str, Any]]] = {}
        self.max_events = 5000
        self.max_snapshots_per_session = 50

    async def execute(self, context: HookContext) -> HookResult:
        """Execute session lifecycle monitoring."""
        try:
            session_id = context.session_id
            event_type_str = context.metadata.get("session_event", "unknown")

            try:
                event_type = SessionEventType(event_type_str)
            except ValueError:
                event_type = SessionEventType.STATE_CHANGE

            previous_state_str = context.metadata.get("previous_state")
            new_state_str = context.metadata.get("new_state")

            try:
                previous_state = SessionState(previous_state_str) if previous_state_str else None
                new_state = SessionState(new_state_str) if new_state_str else None
            except ValueError:
                previous_state = None
                new_state = None

            self.logger.info(f"Session lifecycle event: {session_id} - {event_type.value}",
                           session_id=session_id,
                           event_type=event_type.value,
                           previous_state=previous_state.value if previous_state else None,
                           new_state=new_state.value if new_state else None)

            # Create session event
            event = SessionEvent(
                event_id=f"session_{int(time.time())}_{event_type.value}_{session_id[:8]}",
                timestamp=datetime.now(),
                session_id=session_id,
                event_type=event_type,
                previous_state=previous_state,
                new_state=new_state,
                data=context.metadata.copy(),
                metadata=context.metadata.copy(),
                correlation_id=context.correlation_id
            )

            # Store event
            self._store_session_event(event)

            # Update session metrics
            await self._update_session_metrics(event)

            # Handle specific event types
            if event_type == SessionEventType.CREATION:
                return await self._handle_session_creation(context, event)
            elif event_type == SessionEventType.RESUMPTION:
                return await self._handle_session_resumption(context, event)
            elif event_type == SessionEventType.PAUSE:
                return await self._handle_session_pause(context, event)
            elif event_type == SessionEventType.TERMINATION:
                return await self._handle_session_termination(context, event)
            elif event_type == SessionEventType.ERROR:
                return await self._handle_session_error(context, event)
            elif event_type == SessionEventType.TIMEOUT:
                return await self._handle_session_timeout(context, event)
            elif event_type == SessionEventType.STATE_CHANGE:
                return await self._handle_session_state_change(context, event)
            else:
                return await self._handle_generic_session_event(context, event)

        except Exception as e:
            self.logger.error(f"Session lifecycle hook failed: {str(e)}",
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

    async def _handle_session_creation(self, context: HookContext, event: SessionEvent) -> HookResult:
        """Handle session creation event."""
        session_config = event.data.get("session_config", {})
        user_context = event.data.get("user_context", {})

        # Use session logger for creation
        self.session_logger.log_session_creation(
            session_id=event.session_id,
            session_config=session_config,
            user_context=user_context
        )

        # Initialize session metrics
        self.session_metrics[event.session_id] = SessionMetrics(
            session_id=event.session_id,
            created_at=event.timestamp,
            last_activity=event.timestamp
        )

        # Track active session
        self.active_sessions[event.session_id] = {
            "created_at": event.timestamp,
            "state": SessionState.INITIALIZED.value,
            "config": session_config,
            "user_context": user_context
        }

        # Create initial session snapshot
        await self._create_session_snapshot(event.session_id, "initialization", {
            "config": session_config,
            "user_context": user_context,
            "state": SessionState.INITIALIZED.value
        })

        return HookResult(
            hook_name=self.name,
            hook_type=self.hook_type,
            status=HookStatus.COMPLETED,
            execution_id=context.execution_id,
            start_time=context.timestamp,
            end_time=datetime.now(),
            result_data={
                "event_type": "session_creation",
                "session_id": event.session_id,
                "config_size": len(str(session_config)),
                "active_sessions": len(self.active_sessions)
            }
        )

    async def _handle_session_resumption(self, context: HookContext, event: SessionEvent) -> HookResult:
        """Handle session resumption event."""
        previous_state = event.data.get("previous_state", {})
        resumption_context = event.data.get("resumption_context", {})

        # Use session logger for resumption
        self.session_logger.log_session_resumption(
            session_id=event.session_id,
            previous_state=previous_state,
            resumption_context=resumption_context
        )

        # Update session metrics
        if event.session_id in self.session_metrics:
            metrics = self.session_metrics[event.session_id]
            metrics.last_activity = event.timestamp
            metrics.recovery_attempts += 1
            metrics.successful_recoveries += 1

        # Update active session status
        if event.session_id in self.active_sessions:
            self.active_sessions[event.session_id]["state"] = SessionState.ACTIVE.value
            self.active_sessions[event.session_id]["last_resumed"] = event.timestamp

        # Create session snapshot
        await self._create_session_snapshot(event.session_id, "resumption", {
            "previous_state_size": len(str(previous_state)),
            "resumption_context": resumption_context,
            "resumed_at": event.timestamp.isoformat()
        })

        return HookResult(
            hook_name=self.name,
            hook_type=self.hook_type,
            status=HookStatus.COMPLETED,
            execution_id=context.execution_id,
            start_time=context.timestamp,
            end_time=datetime.now(),
            result_data={
                "event_type": "session_resumption",
                "session_id": event.session_id,
                "previous_state_size": len(str(previous_state)),
                "recovery_success_rate": self.session_metrics[event.session_id].recovery_success_rate if event.session_id in self.session_metrics else 0.0
            }
        )

    async def _handle_session_pause(self, context: HookContext, event: SessionEvent) -> HookResult:
        """Handle session pause event."""
        pause_reason = event.data.get("pause_reason", "unknown")
        state_snapshot = event.data.get("state_snapshot", {})

        # Use session logger for pause
        self.session_logger.log_session_pause(
            session_id=event.session_id,
            pause_reason=pause_reason,
            state_snapshot=state_snapshot
        )

        # Update session metrics
        if event.session_id in self.session_metrics:
            self.session_metrics[event.session_id].last_activity = event.timestamp

        # Update active session status
        if event.session_id in self.active_sessions:
            self.active_sessions[event.session_id]["state"] = SessionState.PAUSED.value
            self.active_sessions[event.session_id]["paused_at"] = event.timestamp
            self.active_sessions[event.session_id]["pause_reason"] = pause_reason

        # Create session snapshot
        await self._create_session_snapshot(event.session_id, "pause", {
            "pause_reason": pause_reason,
            "state_snapshot_size": len(str(state_snapshot)),
            "paused_at": event.timestamp.isoformat()
        })

        return HookResult(
            hook_name=self.name,
            hook_type=self.hook_type,
            status=HookStatus.COMPLETED,
            execution_id=context.execution_id,
            start_time=context.timestamp,
            end_time=datetime.now(),
            result_data={
                "event_type": "session_pause",
                "session_id": event.session_id,
                "pause_reason": pause_reason,
                "state_snapshot_size": len(str(state_snapshot))
            }
        )

    async def _handle_session_termination(self, context: HookContext, event: SessionEvent) -> HookResult:
        """Handle session termination event."""
        termination_reason = event.data.get("termination_reason", "unknown")
        final_state = event.data.get("final_state", {})

        # Use session logger for termination
        self.session_logger.log_session_termination(
            session_id=event.session_id,
            termination_reason=termination_reason,
            final_state=final_state
        )

        # Update session metrics
        if event.session_id in self.session_metrics:
            metrics = self.session_metrics[event.session_id]
            metrics.total_duration = (event.timestamp - metrics.created_at).total_seconds()
            metrics.last_activity = event.timestamp

        # Remove from active sessions
        if event.session_id in self.active_sessions:
            self.active_sessions[event.session_id]["state"] = SessionState.COMPLETED.value
            self.active_sessions[event.session_id]["terminated_at"] = event.timestamp
            self.active_sessions[event.session_id]["termination_reason"] = termination_reason

        # Create final session snapshot
        await self._create_session_snapshot(event.session_id, "termination", {
            "termination_reason": termination_reason,
            "final_state_size": len(str(final_state)),
            "total_duration": self.session_metrics[event.session_id].total_duration if event.session_id in self.session_metrics else 0.0,
            "terminated_at": event.timestamp.isoformat()
        })

        return HookResult(
            hook_name=self.name,
            hook_type=self.hook_type,
            status=HookStatus.COMPLETED,
            execution_id=context.execution_id,
            start_time=context.timestamp,
            end_time=datetime.now(),
            result_data={
                "event_type": "session_termination",
                "session_id": event.session_id,
                "termination_reason": termination_reason,
                "total_duration": self.session_metrics[event.session_id].total_duration if event.session_id in self.session_metrics else 0.0,
                "remaining_active_sessions": len([s for s in self.active_sessions.values() if s["state"] not in [SessionState.COMPLETED.value, SessionState.CANCELLED.value]])
            }
        )

    async def _handle_session_error(self, context: HookContext, event: SessionEvent) -> HookResult:
        """Handle session error event."""
        error_type = event.data.get("error_type", "UnknownError")
        error_context = event.data.get("error_context", {})
        recovery_action = event.data.get("recovery_action")

        # Use session logger for error
        self.session_logger.log_session_error(
            session_id=event.session_id,
            error_type=error_type,
            error_context=error_context,
            recovery_action=recovery_action
        )

        # Update session metrics
        if event.session_id in self.session_metrics:
            self.session_metrics[event.session_id].last_activity = event.timestamp
            self.session_metrics[event.session_id].errors_count += 1
            if recovery_action:
                self.session_metrics[event.session_id].recovery_attempts += 1

        # Update active session status
        if event.session_id in self.active_sessions:
            self.active_sessions[event.session_id]["state"] = SessionState.ERROR.value
            self.active_sessions[event.session_id]["last_error"] = {
                "type": error_type,
                "context": error_context,
                "timestamp": event.timestamp,
                "recovery_action": recovery_action
            }

        # Create session snapshot
        await self._create_session_snapshot(event.session_id, "error", {
            "error_type": error_type,
            "error_context": error_context,
            "recovery_action": recovery_action,
            "error_timestamp": event.timestamp.isoformat()
        })

        return HookResult(
            hook_name=self.name,
            hook_type=self.hook_type,
            status=HookStatus.COMPLETED,
            execution_id=context.execution_id,
            start_time=context.timestamp,
            end_time=datetime.now(),
            result_data={
                "event_type": "session_error",
                "session_id": event.session_id,
                "error_type": error_type,
                "total_errors": self.session_metrics[event.session_id].errors_count if event.session_id in self.session_metrics else 1,
                "recovery_attempted": recovery_action is not None
            }
        )

    async def _handle_session_timeout(self, context: HookContext, event: SessionEvent) -> HookResult:
        """Handle session timeout event."""
        timeout_duration = event.data.get("timeout_duration", 0.0)
        last_activity = event.data.get("last_activity")

        # Update session metrics
        if event.session_id in self.session_metrics:
            self.session_metrics[event.session_id].last_activity = event.timestamp

        # Update active session status
        if event.session_id in self.active_sessions:
            self.active_sessions[event.session_id]["state"] = SessionState.TIMEOUT.value
            self.active_sessions[event.session_id]["timeout_at"] = event.timestamp
            self.active_sessions[event.session_id]["timeout_duration"] = timeout_duration

        # Create session snapshot
        await self._create_session_snapshot(event.session_id, "timeout", {
            "timeout_duration": timeout_duration,
            "last_activity": last_activity,
            "timeout_timestamp": event.timestamp.isoformat()
        })

        return HookResult(
            hook_name=self.name,
            hook_type=self.hook_type,
            status=HookStatus.COMPLETED,
            execution_id=context.execution_id,
            start_time=context.timestamp,
            end_time=datetime.now(),
            result_data={
                "event_type": "session_timeout",
                "session_id": event.session_id,
                "timeout_duration": timeout_duration,
                "last_activity": last_activity
            }
        )

    async def _handle_session_state_change(self, context: HookContext, event: SessionEvent) -> HookResult:
        """Handle generic session state change event."""
        if event.session_id in self.session_metrics:
            self.session_metrics[event.session_id].last_activity = event.timestamp
            self.session_metrics[event.session_id].state_changes += 1

        if event.session_id in self.active_sessions and event.new_state:
            self.active_sessions[event.session_id]["state"] = event.new_state.value

        return HookResult(
            hook_name=self.name,
            hook_type=self.hook_type,
            status=HookStatus.COMPLETED,
            execution_id=context.execution_id,
            start_time=context.timestamp,
            end_time=datetime.now(),
            result_data={
                "event_type": "session_state_change",
                "session_id": event.session_id,
                "previous_state": event.previous_state.value if event.previous_state else None,
                "new_state": event.new_state.value if event.new_state else None,
                "total_state_changes": self.session_metrics[event.session_id].state_changes if event.session_id in self.session_metrics else 0
            }
        )

    async def _handle_generic_session_event(self, context: HookContext, event: SessionEvent) -> HookResult:
        """Handle generic session event."""
        if event.session_id in self.session_metrics:
            self.session_metrics[event.session_id].last_activity = event.timestamp

        return HookResult(
            hook_name=self.name,
            hook_type=self.hook_type,
            status=HookStatus.COMPLETED,
            execution_id=context.execution_id,
            start_time=context.timestamp,
            end_time=datetime.now(),
            result_data={
                "event_type": "generic_session",
                "session_id": event.session_id,
                "session_event": event.event_type.value,
                "data_keys": list(event.data.keys())
            }
        )

    def _store_session_event(self, event: SessionEvent):
        """Store session event and maintain history size."""
        self.session_events.append(event)
        if len(self.session_events) > self.max_events:
            self.session_events = self.session_events[-self.max_events:]

    async def _update_session_metrics(self, event: SessionEvent):
        """Update session metrics based on event."""
        session_id = event.session_id

        if session_id not in self.session_metrics:
            self.session_metrics[session_id] = SessionMetrics(
                session_id=session_id,
                created_at=event.timestamp,
                last_activity=event.timestamp
            )

        metrics = self.session_metrics[session_id]
        metrics.last_activity = event.timestamp

        # Update specific metrics based on event type
        if event.event_type == SessionEventType.STATE_CHANGE and event.new_state:
            metrics.state_changes += 1

        # Update data size
        metrics.data_size_bytes += len(json.dumps(event.data, default=str))

        # Update agents involved
        if "agent_name" in event.data:
            metrics.agents_involved.add(event.data["agent_name"])

        # Update workflow stages completed
        if "workflow_stage" in event.data and event.event_type in [SessionEventType.STATE_CHANGE, SessionEventType.TERMINATION]:
            stage = event.data["workflow_stage"]
            if stage not in metrics.workflow_stages_completed:
                metrics.workflow_stages_completed.append(stage)

        # Update tools executed
        if "tool_name" in event.data:
            tool_name = event.data["tool_name"]
            metrics.tools_executed[tool_name] = metrics.tools_executed.get(tool_name, 0) + 1

    async def _create_session_snapshot(self, session_id: str, snapshot_type: str, data: dict[str, Any]):
        """Create a session snapshot for recovery purposes."""
        if session_id not in self.session_snapshots:
            self.session_snapshots[session_id] = []

        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "type": snapshot_type,
            "data": data.copy(),
            "session_metrics": self.session_metrics[session_id].__dict__.copy() if session_id in self.session_metrics else {}
        }

        self.session_snapshots[session_id].append(snapshot)

        # Maintain snapshot limit
        if len(self.session_snapshots[session_id]) > self.max_snapshots_per_session:
            self.session_snapshots[session_id] = self.session_snapshots[session_id][-self.max_snapshots_per_session:]

    def get_session_events(
        self,
        session_id: str | None = None,
        event_type: str | None = None,
        limit: int = 100
    ) -> list[SessionEvent]:
        """Get filtered session events."""
        events = self.session_events.copy()

        # Apply filters
        if session_id:
            events = [e for e in events if e.session_id == session_id]

        if event_type:
            try:
                event_enum = SessionEventType(event_type)
                events = [e for e in events if e.event_type == event_enum]
            except ValueError:
                pass

        # Sort by timestamp (most recent first) and limit
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]

    def get_session_metrics(self, session_id: str | None = None) -> dict[str, Any]:
        """Get session performance metrics."""
        if session_id:
            if session_id not in self.session_metrics:
                return {}
            metrics = self.session_metrics[session_id]
            return {
                "session_id": metrics.session_id,
                "created_at": metrics.created_at.isoformat(),
                "last_activity": metrics.last_activity.isoformat(),
                "current_duration": metrics.current_duration,
                "total_duration": metrics.total_duration,
                "active_duration": metrics.active_duration,
                "paused_duration": metrics.paused_duration,
                "state_changes": metrics.state_changes,
                "errors_count": metrics.errors_count,
                "recovery_attempts": metrics.recovery_attempts,
                "successful_recoveries": metrics.successful_recoveries,
                "recovery_success_rate": metrics.recovery_success_rate,
                "data_size_bytes": metrics.data_size_bytes,
                "workflow_stages_completed": metrics.workflow_stages_completed.copy(),
                "agents_involved": list(metrics.agents_involved),
                "tools_executed": metrics.tools_executed.copy(),
                "is_active": metrics.is_active
            }
        else:
            # Return metrics for all sessions
            return {
                session_id: self.get_session_metrics(session_id)
                for session_id in self.session_metrics.keys()
            }

    def get_active_sessions(self) -> dict[str, dict[str, Any]]:
        """Get currently active sessions."""
        active = {}
        for session_id, metrics in self.session_metrics.items():
            if metrics.is_active and session_id in self.active_sessions:
                active[session_id] = {
                    **self.active_sessions[session_id],
                    "metrics": self.get_session_metrics(session_id)
                }
        return active

    def get_session_snapshots(self, session_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """Get session snapshots for recovery."""
        if session_id not in self.session_snapshots:
            return []

        snapshots = self.session_snapshots[session_id].copy()
        snapshots.sort(key=lambda s: s["timestamp"], reverse=True)
        return snapshots[:limit]

    def get_session_summary(self) -> dict[str, Any]:
        """Get comprehensive session summary."""
        total_sessions = len(self.session_metrics)
        active_sessions = len([s for s in self.session_metrics.values() if s.is_active])
        completed_sessions = len([s for s in self.session_metrics.values() if s.created_at + timedelta(hours=24) < datetime.now()])
        error_sessions = len([s for s in self.session_metrics.values() if s.errors_count > 0])

        # Calculate average session duration
        completed_metrics = [s for s in self.session_metrics.values() if s.total_duration > 0]
        avg_duration = sum(s.total_duration for s in completed_metrics) / len(completed_metrics) if completed_metrics else 0.0

        # Find most active agents
        all_agents = set()
        for metrics in self.session_metrics.values():
            all_agents.update(metrics.agents_involved)

        # Find most used tools
        tool_usage = {}
        for metrics in self.session_metrics.values():
            for tool, count in metrics.tools_executed.items():
                tool_usage[tool] = tool_usage.get(tool, 0) + count

        return {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "completed_sessions": completed_sessions,
            "error_sessions": error_sessions,
            "average_duration": round(avg_duration, 2),
            "unique_agents": len(all_agents),
            "unique_tools": len(tool_usage),
            "most_used_tools": dict(sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)[:10]),
            "total_events": len(self.session_events),
            "total_snapshots": sum(len(snapshots) for snapshots in self.session_snapshots.values())
        }
