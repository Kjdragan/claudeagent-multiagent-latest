"""
Agent Communication Hooks for Multi-Agent Research System

Provides comprehensive monitoring of agent interactions, communication,
handoffs, and state changes throughout the research workflow.
"""

import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

from .base_hooks import BaseHook, HookContext, HookResult, HookStatus, HookPriority
import sys
import os
# Add parent directory to path for proper imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent_logging import get_logger, AgentCommunicationLogger


class AgentState(Enum):
    """Enumeration of possible agent states."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    PROCESSING = "processing"
    WAITING = "waiting"
    COMMUNICATING = "communicating"
    HANDING_OFF = "handing_off"
    ERROR = "error"
    COMPLETED = "completed"


class HandoffReason(Enum):
    """Enumeration of agent handoff reasons."""
    WORKFLOW_STAGE_COMPLETE = "workflow_stage_complete"
    ERROR_RECOVERY = "error_recovery"
    TASK_DELEGATION = "task_delegation"
    RESOURCE_CONFLICT = "resource_conflict"
    TIMEOUT = "timeout"
    USER_REQUEST = "user_request"
    COLLABORATION_NEEDED = "collaboration_needed"


@dataclass
class AgentCommunicationEvent:
    """Represents an agent communication event."""
    event_id: str
    timestamp: datetime
    from_agent: str
    to_agent: Optional[str]
    event_type: str  # message, handoff, state_change, error
    content: Dict[str, Any]
    session_id: str
    workflow_stage: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentHandoffEvent:
    """Represents an agent handoff event."""
    event_id: str
    timestamp: datetime
    from_agent: str
    to_agent: str
    reason: HandoffReason
    context_data: Dict[str, Any]
    session_id: str
    workflow_stage: str
    success: bool = True
    completion_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentStateSnapshot:
    """Snapshot of agent state at a point in time."""
    agent_name: str
    agent_type: str
    state: AgentState
    timestamp: datetime
    session_id: str
    current_task: Optional[str] = None
    processing_time: float = 0.0
    memory_usage: Optional[float] = None
    error_count: int = 0
    last_activity: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentCommunicationHook(BaseHook):
    """Hook for monitoring agent communication and interactions."""

    def __init__(self, enabled: bool = True, timeout: float = 10.0):
        super().__init__(
            name="agent_communication_monitor",
            hook_type="agent_communication",
            priority=HookPriority.HIGH,
            timeout=timeout,
            enabled=enabled,
            retry_count=1
        )
        self.comm_logger = AgentCommunicationLogger()
        self.communication_events: List[AgentCommunicationEvent] = []
        self.agent_states: Dict[str, AgentStateSnapshot] = {}
        self.communication_graph: Dict[str, Set[str]] = {}
        self.max_events = 1000

    async def execute(self, context: HookContext) -> HookResult:
        """Execute agent communication monitoring."""
        try:
            event_type = context.metadata.get("communication_type", "unknown")
            from_agent = context.agent_name or "unknown"
            to_agent = context.metadata.get("to_agent")
            content = context.metadata.get("content", {})

            self.logger.info(f"Agent communication monitoring: {from_agent} -> {to_agent or 'system'} ({event_type})",
                           from_agent=from_agent,
                           to_agent=to_agent,
                           event_type=event_type,
                           session_id=context.session_id)

            # Create communication event
            event = AgentCommunicationEvent(
                event_id=f"comm_{int(time.time())}_{from_agent}_{event_type}",
                timestamp=datetime.now(),
                from_agent=from_agent,
                to_agent=to_agent,
                event_type=event_type,
                content=content,
                session_id=context.session_id,
                workflow_stage=context.workflow_stage,
                correlation_id=context.correlation_id,
                metadata=context.metadata.copy()
            )

            # Store event
            self._store_communication_event(event)

            # Update communication graph
            self._update_communication_graph(from_agent, to_agent)

            # Use specialized communication logger
            if event_type == "message":
                return await self._handle_message_event(context, event)
            elif event_type == "handoff":
                return await self._handle_handoff_event(context, event)
            elif event_type == "state_change":
                return await self._handle_state_change_event(context, event)
            elif event_type == "error":
                return await self._handle_error_event(context, event)
            else:
                return await self._handle_generic_event(context, event)

        except Exception as e:
            self.logger.error(f"Agent communication hook failed: {str(e)}",
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

    async def _handle_message_event(self, context: HookContext, event: AgentCommunicationEvent) -> HookResult:
        """Handle agent message event."""
        if event.to_agent:
            self.comm_logger.log_agent_message_send(
                session_id=event.session_id,
                message_content=str(event.content.get("message", "")),
                recipient_agent=event.to_agent,
                sender_context={"agent_name": event.from_agent, "agent_type": context.agent_type}
            )

            # Simulate message receipt (in real system, this would be handled by the receiving agent)
            self.comm_logger.log_agent_message_receive(
                session_id=event.session_id,
                message_content=str(event.content.get("message", "")),
                sender_agent=event.from_agent,
                receiver_context={"agent_name": event.to_agent}
            )

        return HookResult(
            hook_name=self.name,
            hook_type=self.hook_type,
            status=HookStatus.COMPLETED,
            execution_id=context.execution_id,
            start_time=context.timestamp,
            end_time=datetime.now(),
            result_data={
                "event_type": "message",
                "from_agent": event.from_agent,
                "to_agent": event.to_agent,
                "message_logged": True
            }
        )

    async def _handle_handoff_event(self, context: HookContext, event: AgentCommunicationEvent) -> HookResult:
        """Handle agent handoff event."""
        if not event.to_agent:
            return HookResult(
                hook_name=self.name,
                hook_type=self.hook_type,
                status=HookStatus.FAILED,
                execution_id=context.execution_id,
                start_time=context.timestamp,
                end_time=datetime.now(),
                error_message="Handoff event missing target agent"
            )

        handoff_reason = HandoffReason(event.content.get("reason", "workflow_stage_complete"))
        context_data = event.content.get("context", {})

        self.comm_logger.log_agent_handoff(
            session_id=event.session_id,
            from_agent=event.from_agent,
            to_agent=event.to_agent,
            handoff_reason=handoff_reason.value,
            context_data=context_data
        )

        # Update agent states
        self._update_agent_state(event.from_agent, AgentState.HANDING_OFF, context.session_id)
        self._update_agent_state(event.to_agent, AgentState.PROCESSING, context.session_id)

        return HookResult(
            hook_name=self.name,
            hook_type=self.hook_type,
            status=HookStatus.COMPLETED,
            execution_id=context.execution_id,
            start_time=context.timestamp,
            end_time=datetime.now(),
            result_data={
                "event_type": "handoff",
                "from_agent": event.from_agent,
                "to_agent": event.to_agent,
                "handoff_reason": handoff_reason.value,
                "context_size": len(str(context_data))
            }
        )

    async def _handle_state_change_event(self, context: HookContext, event: AgentCommunicationEvent) -> HookResult:
        """Handle agent state change event."""
        new_state_str = event.content.get("new_state", "unknown")
        try:
            new_state = AgentState(new_state_str)
        except ValueError:
            new_state = AgentState.IDLE

        self._update_agent_state(
            event.from_agent,
            new_state,
            event.session_id,
            current_task=event.content.get("current_task"),
            processing_time=event.content.get("processing_time", 0.0)
        )

        return HookResult(
            hook_name=self.name,
            hook_type=self.hook_type,
            status=HookStatus.COMPLETED,
            execution_id=context.execution_id,
            start_time=context.timestamp,
            end_time=datetime.now(),
            result_data={
                "event_type": "state_change",
                "agent": event.from_agent,
                "new_state": new_state.value,
                "previous_state": event.content.get("previous_state", "unknown")
            }
        )

    async def _handle_error_event(self, context: HookContext, event: AgentCommunicationEvent) -> HookResult:
        """Handle agent error event."""
        error_message = event.content.get("error", "Unknown error")
        error_type = event.content.get("error_type", "UnknownError")

        # Update agent state to error
        self._update_agent_state(event.from_agent, AgentState.ERROR, event.session_id)

        # Increment error count for the agent
        if event.from_agent in self.agent_states:
            self.agent_states[event.from_agent].error_count += 1

        return HookResult(
            hook_name=self.name,
            hook_type=self.hook_type,
            status=HookStatus.COMPLETED,
            execution_id=context.execution_id,
            start_time=context.timestamp,
            end_time=datetime.now(),
            result_data={
                "event_type": "error",
                "agent": event.from_agent,
                "error_message": error_message,
                "error_type": error_type,
                "total_errors": self.agent_states.get(event.from_agent, AgentStateSnapshot("", "", AgentState.IDLE, datetime.now(), "")).error_count
            }
        )

    async def _handle_generic_event(self, context: HookContext, event: AgentCommunicationEvent) -> HookResult:
        """Handle generic communication event."""
        return HookResult(
            hook_name=self.name,
            hook_type=self.hook_type,
            status=HookStatus.COMPLETED,
            execution_id=context.execution_id,
            start_time=context.timestamp,
            end_time=datetime.now(),
            result_data={
                "event_type": "generic",
                "agent": event.from_agent,
                "event_type": event.event_type,
                "content_keys": list(event.content.keys())
            }
        )

    def _store_communication_event(self, event: AgentCommunicationEvent):
        """Store communication event and maintain history size."""
        self.communication_events.append(event)
        if len(self.communication_events) > self.max_events:
            self.communication_events = self.communication_events[-self.max_events:]

    def _update_communication_graph(self, from_agent: str, to_agent: Optional[str]):
        """Update communication graph with new interaction."""
        if from_agent not in self.communication_graph:
            self.communication_graph[from_agent] = set()

        if to_agent:
            self.communication_graph[from_agent].add(to_agent)
            if to_agent not in self.communication_graph:
                self.communication_graph[to_agent] = set()

    def _update_agent_state(
        self,
        agent_name: str,
        state: AgentState,
        session_id: str,
        current_task: Optional[str] = None,
        processing_time: float = 0.0
    ):
        """Update agent state snapshot."""
        snapshot = AgentStateSnapshot(
            agent_name=agent_name,
            agent_type="",  # Would be determined from context
            state=state,
            timestamp=datetime.now(),
            session_id=session_id,
            current_task=current_task,
            processing_time=processing_time,
            last_activity=datetime.now()
        )

        # If agent already exists, preserve some data
        if agent_name in self.agent_states:
            old_snapshot = self.agent_states[agent_name]
            snapshot.agent_type = old_snapshot.agent_type
            snapshot.memory_usage = old_snapshot.memory_usage
            snapshot.error_count = old_snapshot.error_count
            if state == AgentState.ERROR:
                snapshot.error_count += 1

        self.agent_states[agent_name] = snapshot

    def get_communication_history(
        self,
        session_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[AgentCommunicationEvent]:
        """Get filtered communication history."""
        events = self.communication_events.copy()

        # Apply filters
        if session_id:
            events = [e for e in events if e.session_id == session_id]

        if agent_name:
            events = [e for e in events if e.from_agent == agent_name or e.to_agent == agent_name]

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        # Sort by timestamp (most recent first) and limit
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]

    def get_agent_states(self, session_id: Optional[str] = None) -> Dict[str, AgentStateSnapshot]:
        """Get current agent states."""
        if session_id:
            return {name: state for name, state in self.agent_states.items() if state.session_id == session_id}
        return self.agent_states.copy()

    def get_communication_graph(self) -> Dict[str, List[str]]:
        """Get communication graph as adjacency list."""
        return {agent: list(connections) for agent, connections in self.communication_graph.items()}

    def get_active_agents(self, session_id: Optional[str] = None) -> List[str]:
        """Get list of currently active agents."""
        active_states = {AgentState.PROCESSING, AgentState.COMMUNICATING, AgentState.HANDING_OFF}
        active_agents = []

        for agent_name, snapshot in self.agent_states.items():
            if snapshot.state in active_states:
                if session_id is None or snapshot.session_id == session_id:
                    # Check if agent was recently active (within last 5 minutes)
                    if snapshot.last_activity and (datetime.now() - snapshot.last_activity) < timedelta(minutes=5):
                        active_agents.append(agent_name)

        return active_agents

    def get_error_summary(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of agent errors."""
        error_events = [e for e in self.communication_events if e.event_type == "error"]
        if session_id:
            error_events = [e for e in error_events if e.session_id == session_id]

        # Count errors by agent
        error_counts = {}
        error_types = {}
        recent_errors = []

        for event in error_events[-20:]:  # Last 20 errors
            agent = event.from_agent
            error_counts[agent] = error_counts.get(agent, 0) + 1

            error_type = event.content.get("error_type", "Unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1

            recent_errors.append({
                "timestamp": event.timestamp.isoformat(),
                "agent": agent,
                "error_type": error_type,
                "error_message": event.content.get("error", "Unknown")
            })

        return {
            "total_errors": len(error_events),
            "errors_by_agent": error_counts,
            "error_types": error_types,
            "recent_errors": recent_errors,
            "agents_with_errors": list(error_counts.keys())
        }


class AgentHandoffHook(BaseHook):
    """Specialized hook for monitoring agent handoffs with detailed tracking."""

    def __init__(self, enabled: bool = True, timeout: float = 15.0):
        super().__init__(
            name="agent_handoff_monitor",
            hook_type="agent_handoff",
            priority=HookPriority.HIGH,
            timeout=timeout,
            enabled=enabled,
            retry_count=1
        )
        self.handoff_events: List[AgentHandoffEvent] = []
        self.handoff_metrics: Dict[str, Dict[str, Any]] = {}
        self.max_handoffs = 500

    async def execute(self, context: HookContext) -> HookResult:
        """Execute agent handoff monitoring."""
        try:
            from_agent = context.agent_name or "unknown"
            to_agent = context.metadata.get("to_agent")
            reason_str = context.metadata.get("reason", "workflow_stage_complete")
            context_data = context.metadata.get("context", {})

            if not to_agent:
                return HookResult(
                    hook_name=self.name,
                    hook_type=self.hook_type,
                    status=HookStatus.FAILED,
                    execution_id=context.execution_id,
                    start_time=context.timestamp,
                    end_time=datetime.now(),
                    error_message="Handoff event missing target agent"
                )

            # Create handoff event
            handoff = AgentHandoffEvent(
                event_id=f"handoff_{int(time.time())}_{from_agent}_{to_agent}",
                timestamp=datetime.now(),
                from_agent=from_agent,
                to_agent=to_agent,
                reason=HandoffReason(reason_str),
                context_data=context_data,
                session_id=context.session_id,
                workflow_stage=context.workflow_stage or "unknown",
                metadata=context.metadata.copy()
            )

            # Store handoff event
            self._store_handoff_event(handoff)

            # Update metrics
            self._update_handoff_metrics(handoff)

            self.logger.info(f"Agent handoff monitored: {from_agent} -> {to_agent} ({reason_str})",
                           from_agent=from_agent,
                           to_agent=to_agent,
                           reason=reason_str,
                           workflow_stage=context.workflow_stage,
                           session_id=context.session_id)

            return HookResult(
                hook_name=self.name,
                hook_type=self.hook_type,
                status=HookStatus.COMPLETED,
                execution_id=context.execution_id,
                start_time=context.timestamp,
                end_time=datetime.now(),
                result_data={
                    "handoff_id": handoff.event_id,
                    "from_agent": from_agent,
                    "to_agent": to_agent,
                    "reason": reason_str,
                    "workflow_stage": context.workflow_stage,
                    "context_size": len(str(context_data)),
                    "success": handoff.success
                }
            )

        except Exception as e:
            self.logger.error(f"Agent handoff hook failed: {str(e)}",
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

    def _store_handoff_event(self, handoff: AgentHandoffEvent):
        """Store handoff event and maintain history size."""
        self.handoff_events.append(handoff)
        if len(self.handoff_events) > self.max_handoffs:
            self.handoff_events = self.handoff_events[-self.max_handoffs:]

    def _update_handoff_metrics(self, handoff: AgentHandoffEvent):
        """Update handoff metrics."""
        pair_key = f"{handoff.from_agent}->{handoff.to_agent}"
        reason_key = handoff.reason.value

        # Initialize metrics if needed
        if pair_key not in self.handoff_metrics:
            self.handoff_metrics[pair_key] = {
                "total_handoffs": 0,
                "successful_handoffs": 0,
                "failed_handoffs": 0,
                "average_completion_time": 0.0,
                "reasons": {}
            }

        # Update pair metrics
        metrics = self.handoff_metrics[pair_key]
        metrics["total_handoffs"] += 1
        if handoff.success:
            metrics["successful_handoffs"] += 1
        else:
            metrics["failed_handoffs"] += 1

        if handoff.completion_time is not None:
            if metrics["total_handoffs"] == 1:
                metrics["average_completion_time"] = handoff.completion_time
            else:
                total_time = metrics["average_completion_time"] * (metrics["total_handoffs"] - 1)
                metrics["average_completion_time"] = (total_time + handoff.completion_time) / metrics["total_handoffs"]

        # Update reason metrics
        if reason_key not in metrics["reasons"]:
            metrics["reasons"][reason_key] = 0
        metrics["reasons"][reason_key] += 1

    def get_handoff_history(
        self,
        session_id: Optional[str] = None,
        from_agent: Optional[str] = None,
        to_agent: Optional[str] = None,
        reason: Optional[str] = None,
        limit: int = 50
    ) -> List[AgentHandoffEvent]:
        """Get filtered handoff history."""
        handoffs = self.handoff_events.copy()

        # Apply filters
        if session_id:
            handoffs = [h for h in handoffs if h.session_id == session_id]

        if from_agent:
            handoffs = [h for h in handoffs if h.from_agent == from_agent]

        if to_agent:
            handoffs = [h for h in handoffs if h.to_agent == to_agent]

        if reason:
            handoffs = [h for h in handoffs if h.reason.value == reason]

        # Sort by timestamp (most recent first) and limit
        handoffs.sort(key=lambda h: h.timestamp, reverse=True)
        return handoffs[:limit]

    def get_handoff_metrics(self) -> Dict[str, Any]:
        """Get comprehensive handoff metrics."""
        summary = {
            "total_handoffs": sum(m["total_handoffs"] for m in self.handoff_metrics.values()),
            "successful_handoffs": sum(m["successful_handoffs"] for m in self.handoff_metrics.values()),
            "failed_handoffs": sum(m["failed_handoffs"] for m in self.handoff_metrics.values()),
            "overall_success_rate": 0.0,
            "handoff_pairs": self.handoff_metrics.copy(),
            "most_common_pairs": [],
            "common_reasons": {}
        }

        # Calculate overall success rate
        if summary["total_handoffs"] > 0:
            summary["overall_success_rate"] = (summary["successful_handoffs"] / summary["total_handoffs"]) * 100

        # Find most common handoff pairs
        sorted_pairs = sorted(self.handoff_metrics.items(), key=lambda x: x[1]["total_handoffs"], reverse=True)
        summary["most_common_pairs"] = [
            {"pair": pair, "count": metrics["total_handoffs"], "success_rate": (metrics["successful_handoffs"] / metrics["total_handoffs"]) * 100 if metrics["total_handoffs"] > 0 else 0}
            for pair, metrics in sorted_pairs[:10]
        ]

        # Aggregate common reasons
        reason_counts = {}
        for metrics in self.handoff_metrics.values():
            for reason, count in metrics["reasons"].items():
                reason_counts[reason] = reason_counts.get(reason, 0) + count

        summary["common_reasons"] = dict(sorted(reason_counts.items(), key=lambda x: x[1], reverse=True))

        return summary


class AgentStateMonitor(BaseHook):
    """Hook for monitoring agent state changes and health."""

    def __init__(self, check_interval: float = 30.0, timeout: float = 30.0, enabled: bool = True):
        super().__init__(
            name="agent_state_monitor",
            hook_type="agent_state",
            priority=HookPriority.NORMAL,
            timeout=10.0,
            enabled=enabled,
            retry_count=0
        )
        self.check_interval = check_interval
        self.agent_health_status: Dict[str, Dict[str, Any]] = {}
        self.state_change_history: List[Dict[str, Any]] = []
        self.max_history = 200

    async def execute(self, context: HookContext) -> HookResult:
        """Execute agent state monitoring."""
        try:
            agent_name = context.agent_name or "unknown"
            new_state_str = context.metadata.get("new_state", "unknown")
            previous_state_str = context.metadata.get("previous_state", "unknown")

            try:
                new_state = AgentState(new_state_str)
                previous_state = AgentState(previous_state_str)
            except ValueError:
                new_state = AgentState.IDLE
                previous_state = AgentState.IDLE

            # Record state change
            state_change = {
                "timestamp": datetime.now().isoformat(),
                "agent_name": agent_name,
                "previous_state": previous_state.value,
                "new_state": new_state.value,
                "session_id": context.session_id,
                "workflow_stage": context.workflow_stage,
                "reason": context.metadata.get("reason", "unknown")
            }

            self._store_state_change(state_change)

            # Update health status
            self._update_health_status(agent_name, new_state, context)

            # Check for concerning state patterns
            alerts = self._check_state_patterns(agent_name, new_state, previous_state)

            self.logger.info(f"Agent state change monitored: {agent_name} {previous_state.value} -> {new_state.value}",
                           agent_name=agent_name,
                           previous_state=previous_state.value,
                           new_state=new_state.value,
                           session_id=context.session_id,
                           alerts_generated=len(alerts))

            return HookResult(
                hook_name=self.name,
                hook_type=self.hook_type,
                status=HookStatus.COMPLETED,
                execution_id=context.execution_id,
                start_time=context.timestamp,
                end_time=datetime.now(),
                result_data={
                    "agent_name": agent_name,
                    "state_change": state_change,
                    "health_status": self.agent_health_status.get(agent_name, {}),
                    "alerts": alerts
                }
            )

        except Exception as e:
            self.logger.error(f"Agent state monitor failed: {str(e)}",
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

    def _store_state_change(self, state_change: Dict[str, Any]):
        """Store state change in history."""
        self.state_change_history.append(state_change)
        if len(self.state_change_history) > self.max_history:
            self.state_change_history = self.state_change_history[-self.max_history:]

    def _update_health_status(self, agent_name: str, state: AgentState, context: HookContext):
        """Update agent health status based on state."""
        if agent_name not in self.agent_health_status:
            self.agent_health_status[agent_name] = {
                "agent_name": agent_name,
                "current_state": state.value,
                "last_state_change": datetime.now(),
                "state_changes_count": 0,
                "error_count": 0,
                "total_processing_time": 0.0,
                "average_processing_time": 0.0,
                "health_score": 100.0,
                "last_activity": datetime.now(),
                "issues": []
            }

        status = self.agent_health_status[agent_name]
        previous_state = status["current_state"]

        # Update status
        status["current_state"] = state.value
        status["last_state_change"] = datetime.now()
        status["state_changes_count"] += 1
        status["last_activity"] = datetime.now()

        # Update error count
        if state == AgentState.ERROR:
            status["error_count"] += 1

        # Calculate health score
        self._calculate_health_score(agent_name, status)

        # Check for issues
        issues = []
        if state == AgentState.ERROR:
            issues.append("Agent in error state")
        if status["error_count"] > 5:
            issues.append(f"High error count: {status['error_count']}")
        if status["state_changes_count"] > 50:  # Frequent state changes
            issues.append("Excessive state changes")

        status["issues"] = issues

    def _calculate_health_score(self, agent_name: str, status: Dict[str, Any]):
        """Calculate agent health score."""
        score = 100.0

        # Deduct points for errors
        error_penalty = min(status["error_count"] * 5, 50)
        score -= error_penalty

        # Deduct points for being in error state
        if status["current_state"] == AgentState.ERROR.value:
            score -= 20

        # Deduct points for frequent state changes (indicates instability)
        if status["state_changes_count"] > 30:
            score -= 10
        elif status["state_changes_count"] > 20:
            score -= 5

        # Ensure score doesn't go below 0
        status["health_score"] = max(0.0, score)

    def _check_state_patterns(self, agent_name: str, new_state: AgentState, previous_state: AgentState) -> List[str]:
        """Check for concerning state patterns and generate alerts."""
        alerts = []

        # Check for error loops
        if new_state == AgentState.ERROR and previous_state == AgentState.ERROR:
            alerts.append("Agent stuck in error state")

        # Check for rapid state changes
        recent_changes = [sc for sc in self.state_change_history[-10:] if sc["agent_name"] == agent_name]
        if len(recent_changes) > 5:  # More than 5 state changes in recent history
            alerts.append("Agent experiencing rapid state changes")

        # Check for long processing times
        if new_state == AgentState.PROCESSING:
            # Check if agent has been processing for too long
            long_processing_threshold = 300  # 5 minutes
            # This would need timestamp tracking - simplified for now

        return alerts

    def get_agent_health_summary(self) -> Dict[str, Any]:
        """Get overall agent health summary."""
        if not self.agent_health_status:
            return {"message": "No agent health data available"}

        total_agents = len(self.agent_health_status)
        healthy_agents = sum(1 for status in self.agent_health_status.values() if status["health_score"] >= 80)
        unhealthy_agents = sum(1 for status in self.agent_health_status.values() if status["health_score"] < 50)

        # Find agents with issues
        agents_with_issues = {
            name: status["issues"]
            for name, status in self.agent_health_status.items()
            if status["issues"]
        }

        # Calculate overall health score
        overall_score = sum(status["health_score"] for status in self.agent_health_status.values()) / total_agents

        return {
            "total_agents": total_agents,
            "healthy_agents": healthy_agents,
            "unhealthy_agents": unhealthy_agents,
            "overall_health_score": round(overall_score, 1),
            "agents_with_issues": agents_with_issues,
            "health_grade": self._calculate_health_grade(overall_score)
        }

    def _calculate_health_grade(self, score: float) -> str:
        """Calculate health grade based on score."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"