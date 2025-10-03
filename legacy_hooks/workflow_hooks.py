"""
Workflow Orchestration Hooks for Multi-Agent Research System

Provides comprehensive monitoring of workflow execution, stage transitions,
decision points, and orchestration patterns throughout the research process.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum

from .base_hooks import BaseHook, HookContext, HookResult, HookStatus, HookPriority
import sys
import os
# Add parent directory to path for proper imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent_logging import get_logger, WorkflowLogger


class WorkflowStage(Enum):
    """Enumeration of workflow stages."""
    INITIALIZATION = "initialization"
    RESEARCH = "research"
    REPORT_GENERATION = "report_generation"
    EDITORIAL_REVIEW = "editorial_review"
    FINALIZATION = "finalization"
    COMPLETED = "completed"
    ERROR = "error"


class DecisionType(Enum):
    """Enumeration of decision types in workflow."""
    ROUTING_DECISION = "routing_decision"
    QUALITY_GATE = "quality_gate"
    ERROR_RECOVERY = "error_recovery"
    RESOURCE_ALLOCATION = "resource_allocation"
    CONTINUATION_DECISION = "continuation_decision"


@dataclass
class WorkflowEvent:
    """Represents a workflow event."""
    event_id: str
    timestamp: datetime
    workflow_type: str
    session_id: str
    stage: WorkflowStage
    event_type: str  # stage_start, stage_complete, decision, error
    data: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageMetrics:
    """Metrics for workflow stage performance."""
    stage_name: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    max_execution_time: float = 0.0
    last_execution: Optional[datetime] = None
    common_errors: Dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_executions == 0:
            return 0.0
        return (self.successful_executions / self.total_executions) * 100


@dataclass
class DecisionPoint:
    """Represents a decision point in the workflow."""
    decision_id: str
    timestamp: datetime
    workflow_type: str
    session_id: str
    stage: WorkflowStage
    decision_type: DecisionType
    available_options: List[str]
    chosen_option: str
    decision_context: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    outcome: Optional[str] = None


class WorkflowOrchestrationHook(BaseHook):
    """Hook for monitoring workflow orchestration and stage transitions."""

    def __init__(self, enabled: bool = True, timeout: float = 20.0):
        super().__init__(
            name="workflow_orchestration_monitor",
            hook_type="workflow_orchestration",
            priority=HookPriority.HIGH,
            timeout=timeout,
            enabled=enabled,
            retry_count=1
        )
        self.workflow_logger = WorkflowLogger()
        self.workflow_events: List[WorkflowEvent] = []
        self.stage_metrics: Dict[str, StageMetrics] = {}
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.max_events = 2000

    async def execute(self, context: HookContext) -> HookResult:
        """Execute workflow orchestration monitoring."""
        try:
            workflow_type = context.metadata.get("workflow_type", "research_workflow")
            event_type = context.metadata.get("workflow_event", "unknown")
            stage_str = context.metadata.get("stage", "unknown")

            try:
                stage = WorkflowStage(stage_str)
            except ValueError:
                stage = WorkflowStage.INITIALIZATION

            self.logger.info(f"Workflow orchestration monitoring: {workflow_type} - {event_type} ({stage.value})",
                           workflow_type=workflow_type,
                           event_type=event_type,
                           stage=stage.value,
                           session_id=context.session_id)

            # Create workflow event
            event = WorkflowEvent(
                event_id=f"workflow_{int(time.time())}_{workflow_type}_{event_type}",
                timestamp=datetime.now(),
                workflow_type=workflow_type,
                session_id=context.session_id,
                stage=stage,
                event_type=event_type,
                data=context.metadata.copy(),
                correlation_id=context.correlation_id,
                metadata=context.metadata.copy()
            )

            # Store event
            self._store_workflow_event(event)

            # Handle different event types
            if event_type == "stage_start":
                return await self._handle_stage_start(context, event)
            elif event_type == "stage_complete":
                return await self._handle_stage_complete(context, event)
            elif event_type == "decision_point":
                return await self._handle_decision_point(context, event)
            elif event_type == "workflow_start":
                return await self._handle_workflow_start(context, event)
            elif event_type == "workflow_complete":
                return await self._handle_workflow_complete(context, event)
            elif event_type == "workflow_error":
                return await self._handle_workflow_error(context, event)
            else:
                return await self._handle_generic_workflow_event(context, event)

        except Exception as e:
            self.logger.error(f"Workflow orchestration hook failed: {str(e)}",
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

    async def _handle_workflow_start(self, context: HookContext, event: WorkflowEvent) -> HookResult:
        """Handle workflow start event."""
        workflow_type = event.workflow_type
        participants = event.data.get("participants", [])
        estimated_duration = event.data.get("estimated_duration", 600.0)

        # Use workflow logger for orchestration
        self.workflow_logger.log_workflow_orchestration(
            workflow_type=workflow_type,
            participants=participants,
            estimated_duration=estimated_duration,
            session_id=event.session_id,
            stage_count=event.data.get("stage_count", 4)
        )

        # Track active workflow
        self.active_workflows[event.session_id] = {
            "workflow_type": workflow_type,
            "start_time": event.timestamp,
            "current_stage": event.stage.value,
            "participants": participants,
            "estimated_duration": estimated_duration,
            "status": "running"
        }

        return HookResult(
            hook_name=self.name,
            hook_type=self.hook_type,
            status=HookStatus.COMPLETED,
            execution_id=context.execution_id,
            start_time=context.timestamp,
            end_time=datetime.now(),
            result_data={
                "event_type": "workflow_start",
                "workflow_type": workflow_type,
                "participants": participants,
                "estimated_duration": estimated_duration,
                "active_workflows": len(self.active_workflows)
            }
        )

    async def _handle_stage_start(self, context: HookContext, event: WorkflowEvent) -> HookResult:
        """Handle workflow stage start event."""
        stage_name = event.stage.value
        stage_config = event.data.get("stage_config", {})
        estimated_duration = event.data.get("estimated_duration", 120.0)

        # Use workflow logger for stage start
        self.workflow_logger.log_workflow_stage_start(
            session_id=event.session_id,
            workflow_type=event.workflow_type,
            stage_name=stage_name,
            stage_config=stage_config,
            estimated_duration=estimated_duration
        )

        # Update active workflow stage
        if event.session_id in self.active_workflows:
            self.active_workflows[event.session_id]["current_stage"] = stage_name

        # Initialize stage metrics if needed
        if stage_name not in self.stage_metrics:
            self.stage_metrics[stage_name] = StageMetrics(stage_name=stage_name)

        return HookResult(
            hook_name=self.name,
            hook_type=self.hook_type,
            status=HookStatus.COMPLETED,
            execution_id=context.execution_id,
            start_time=context.timestamp,
            end_time=datetime.now(),
            result_data={
                "event_type": "stage_start",
                "stage_name": stage_name,
                "estimated_duration": estimated_duration,
                "config_keys": list(stage_config.keys())
            }
        )

    async def _handle_stage_complete(self, context: HookContext, event: WorkflowEvent) -> HookResult:
        """Handle workflow stage completion event."""
        stage_name = event.stage.value
        stage_result = event.data.get("stage_result", {})
        execution_time = event.data.get("execution_time", 0.0)

        # Use workflow logger for stage completion
        self.workflow_logger.log_workflow_stage_complete(
            session_id=event.session_id,
            workflow_type=event.workflow_type,
            stage_name=stage_name,
            stage_result=stage_result,
            execution_time=execution_time
        )

        # Update stage metrics
        if stage_name in self.stage_metrics:
            success = stage_result.get("success", True)
            self.stage_metrics[stage_name].total_executions += 1
            self.stage_metrics[stage_name].last_execution = event.timestamp

            if success:
                self.stage_metrics[stage_name].successful_executions += 1
            else:
                self.stage_metrics[stage_name].failed_executions += 1
                error_type = stage_result.get("error_type", "Unknown")
                self.stage_metrics[stage_name].common_errors[error_type] = \
                    self.stage_metrics[stage_name].common_errors.get(error_type, 0) + 1

            # Update execution time metrics
            self.stage_metrics[stage_name].min_execution_time = \
                min(self.stage_metrics[stage_name].min_execution_time, execution_time)
            self.stage_metrics[stage_name].max_execution_time = \
                max(self.stage_metrics[stage_name].max_execution_time, execution_time)

            if self.stage_metrics[stage_name].total_executions == 1:
                self.stage_metrics[stage_name].average_execution_time = execution_time
            else:
                total = self.stage_metrics[stage_name].average_execution_time * \
                       (self.stage_metrics[stage_name].total_executions - 1)
                self.stage_metrics[stage_name].average_execution_time = \
                    (total + execution_time) / self.stage_metrics[stage_name].total_executions

        return HookResult(
            hook_name=self.name,
            hook_type=self.hook_type,
            status=HookStatus.COMPLETED,
            execution_id=context.execution_id,
            start_time=context.timestamp,
            end_time=datetime.now(),
            result_data={
                "event_type": "stage_complete",
                "stage_name": stage_name,
                "execution_time": execution_time,
                "success": stage_result.get("success", True),
                "result_keys": list(stage_result.keys())
            }
        )

    async def _handle_decision_point(self, context: HookContext, event: WorkflowEvent) -> HookResult:
        """Handle workflow decision point event."""
        decision_data = event.data.get("decision", {})
        available_options = decision_data.get("available_options", [])
        chosen_option = decision_data.get("chosen_option", "unknown")
        decision_context = decision_data.get("decision_context", {})

        try:
            decision_type = DecisionType(decision_data.get("decision_type", "routing_decision"))
        except ValueError:
            decision_type = DecisionType.ROUTING_DECISION

        # Use workflow logger for decision point
        self.workflow_logger.log_workflow_decision_point(
            session_id=event.session_id,
            workflow_type=event.workflow_type,
            decision_point=decision_data.get("decision_point", "unknown"),
            available_options=available_options,
            chosen_option=chosen_option,
            decision_context=decision_context
        )

        # Create decision point record
        decision = DecisionPoint(
            decision_id=f"decision_{int(time.time())}",
            timestamp=event.timestamp,
            workflow_type=event.workflow_type,
            session_id=event.session_id,
            stage=event.stage,
            decision_type=decision_type,
            available_options=available_options,
            chosen_option=chosen_option,
            decision_context=decision_context
        )

        return HookResult(
            hook_name=self.name,
            hook_type=self.hook_type,
            status=HookStatus.COMPLETED,
            execution_id=context.execution_id,
            start_time=context.timestamp,
            end_time=datetime.now(),
            result_data={
                "event_type": "decision_point",
                "decision_type": decision_type.value,
                "available_options": available_options,
                "chosen_option": chosen_option,
                "decision_id": decision.decision_id
            }
        )

    async def _handle_workflow_complete(self, context: HookContext, event: WorkflowEvent) -> HookResult:
        """Handle workflow completion event."""
        duration = event.data.get("duration", 0.0)
        final_result = event.data.get("final_result", {})

        # Update active workflow status
        if event.session_id in self.active_workflows:
            self.active_workflows[event.session_id]["status"] = "completed"
            self.active_workflows[event.session_id]["end_time"] = event.timestamp
            self.active_workflows[event.session_id]["duration"] = duration

        return HookResult(
            hook_name=self.name,
            hook_type=self.hook_type,
            status=HookStatus.COMPLETED,
            execution_id=context.execution_id,
            start_time=context.timestamp,
            end_time=datetime.now(),
            result_data={
                "event_type": "workflow_complete",
                "duration": duration,
                "final_result_keys": list(final_result.keys()),
                "active_workflows": len([w for w in self.active_workflows.values() if w["status"] == "running"])
            }
        )

    async def _handle_workflow_error(self, context: HookContext, event: WorkflowEvent) -> HookResult:
        """Handle workflow error event."""
        error_type = event.data.get("error_type", "UnknownError")
        error_message = event.data.get("error_message", "Unknown error")
        recovery_attempted = event.data.get("recovery_attempted", False)

        # Use workflow logger for error
        self.workflow_logger.log_workflow_error(
            session_id=event.session_id,
            workflow_type=event.workflow_type,
            error_stage=event.stage.value,
            error_type=error_type,
            error_context={"error_message": error_message},
            recovery_attempted=recovery_attempted
        )

        # Update active workflow status
        if event.session_id in self.active_workflows:
            self.active_workflows[event.session_id]["status"] = "error"
            self.active_workflows[event.session_id]["error"] = {
                "type": error_type,
                "message": error_message,
                "timestamp": event.timestamp
            }

        return HookResult(
            hook_name=self.name,
            hook_type=self.hook_type,
            status=HookStatus.COMPLETED,
            execution_id=context.execution_id,
            start_time=context.timestamp,
            end_time=datetime.now(),
            result_data={
                "event_type": "workflow_error",
                "error_type": error_type,
                "error_message": error_message,
                "recovery_attempted": recovery_attempted
            }
        )

    async def _handle_generic_workflow_event(self, context: HookContext, event: WorkflowEvent) -> HookResult:
        """Handle generic workflow event."""
        return HookResult(
            hook_name=self.name,
            hook_type=self.hook_type,
            status=HookStatus.COMPLETED,
            execution_id=context.execution_id,
            start_time=context.timestamp,
            end_time=datetime.now(),
            result_data={
                "event_type": "generic_workflow",
                "workflow_type": event.workflow_type,
                "stage": event.stage.value,
                "data_keys": list(event.data.keys())
            }
        )

    def _store_workflow_event(self, event: WorkflowEvent):
        """Store workflow event and maintain history size."""
        self.workflow_events.append(event)
        if len(self.workflow_events) > self.max_events:
            self.workflow_events = self.workflow_events[-self.max_events:]

    def get_workflow_events(
        self,
        session_id: Optional[str] = None,
        workflow_type: Optional[str] = None,
        event_type: Optional[str] = None,
        stage: Optional[str] = None,
        limit: int = 100
    ) -> List[WorkflowEvent]:
        """Get filtered workflow events."""
        events = self.workflow_events.copy()

        # Apply filters
        if session_id:
            events = [e for e in events if e.session_id == session_id]

        if workflow_type:
            events = [e for e in events if e.workflow_type == workflow_type]

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if stage:
            events = [e for e in events if e.stage.value == stage]

        # Sort by timestamp (most recent first) and limit
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]

    def get_stage_metrics(self, stage_name: Optional[str] = None) -> Dict[str, Any]:
        """Get stage performance metrics."""
        if stage_name:
            if stage_name not in self.stage_metrics:
                return {}
            metrics = self.stage_metrics[stage_name]
            return {
                "stage_name": metrics.stage_name,
                "total_executions": metrics.total_executions,
                "successful_executions": metrics.successful_executions,
                "failed_executions": metrics.failed_executions,
                "success_rate": round(metrics.success_rate, 2),
                "average_execution_time": round(metrics.average_execution_time, 3),
                "min_execution_time": metrics.min_execution_time if metrics.min_execution_time != float('inf') else 0.0,
                "max_execution_time": metrics.max_execution_time,
                "last_execution": metrics.last_execution.isoformat() if metrics.last_execution else None,
                "common_errors": metrics.common_errors.copy()
            }
        else:
            # Return metrics for all stages
            return {
                stage_name: self.get_stage_metrics(stage_name)
                for stage_name in self.stage_metrics.keys()
            }

    def get_active_workflows(self) -> Dict[str, Dict[str, Any]]:
        """Get currently active workflows."""
        return {
            session_id: workflow_data.copy()
            for session_id, workflow_data in self.active_workflows.items()
            if workflow_data.get("status") == "running"
        }

    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get comprehensive workflow summary."""
        total_workflows = len(self.active_workflows)
        active_workflows = len([w for w in self.active_workflows.values() if w.get("status") == "running"])
        completed_workflows = len([w for w in self.active_workflows.values() if w.get("status") == "completed"])
        error_workflows = len([w for w in self.active_workflows.values() if w.get("status") == "error"])

        # Calculate stage performance
        stage_performance = {}
        for stage_name, metrics in self.stage_metrics.items():
            if metrics.total_executions > 0:
                stage_performance[stage_name] = {
                    "success_rate": metrics.success_rate,
                    "avg_time": metrics.average_execution_time,
                    "total_executions": metrics.total_executions
                }

        return {
            "total_workflows": total_workflows,
            "active_workflows": active_workflows,
            "completed_workflows": completed_workflows,
            "error_workflows": error_workflows,
            "completion_rate": (completed_workflows / total_workflows * 100) if total_workflows > 0 else 0.0,
            "stage_performance": stage_performance,
            "total_events": len(self.workflow_events)
        }


class StageTransitionHook(BaseHook):
    """Specialized hook for monitoring stage transitions in detail."""

    def __init__(self, enabled: bool = True, timeout: float = 15.0):
        super().__init__(
            name="stage_transition_monitor",
            hook_type="stage_transition",
            priority=HookPriority.HIGH,
            timeout=timeout,
            enabled=enabled,
            retry_count=1
        )
        self.transition_history: List[Dict[str, Any]] = []
        self.transition_patterns: Dict[str, int] = {}
        self.max_history = 500

    async def execute(self, context: HookContext) -> HookResult:
        """Execute stage transition monitoring."""
        try:
            from_stage = context.metadata.get("from_stage", "unknown")
            to_stage = context.metadata.get("to_stage", "unknown")
            transition_context = context.metadata.get("context", {})

            self.logger.info(f"Stage transition monitored: {from_stage} -> {to_stage}",
                           from_stage=from_stage,
                           to_stage=to_stage,
                           session_id=context.session_id)

            # Record transition
            transition = {
                "timestamp": datetime.now().isoformat(),
                "session_id": context.session_id,
                "workflow_type": context.metadata.get("workflow_type", "research_workflow"),
                "from_stage": from_stage,
                "to_stage": to_stage,
                "transition_context": transition_context,
                "transition_id": f"transition_{int(time.time())}_{from_stage}_{to_stage}"
            }

            # Store transition
            self.transition_history.append(transition)
            if len(self.transition_history) > self.max_history:
                self.transition_history = self.transition_history[-self.max_history]

            # Update transition patterns
            pattern_key = f"{from_stage}->{to_stage}"
            self.transition_patterns[pattern_key] = self.transition_patterns.get(pattern_key, 0) + 1

            # Check for unusual transitions
            unusual_patterns = self._check_unusual_patterns(from_stage, to_stage)

            return HookResult(
                hook_name=self.name,
                hook_type=self.hook_type,
                status=HookStatus.COMPLETED,
                execution_id=context.execution_id,
                start_time=context.timestamp,
                end_time=datetime.now(),
                result_data={
                    "transition_id": transition["transition_id"],
                    "from_stage": from_stage,
                    "to_stage": to_stage,
                    "context_size": len(str(transition_context)),
                    "unusual_patterns": unusual_patterns,
                    "transition_count": self.transition_patterns[pattern_key]
                }
            )

        except Exception as e:
            self.logger.error(f"Stage transition hook failed: {str(e)}",
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

    def _check_unusual_patterns(self, from_stage: str, to_stage: str) -> List[str]:
        """Check for unusual or problematic transition patterns."""
        unusual = []

        # Check for backward transitions (might indicate retries or issues)
        stage_order = ["initialization", "research", "report_generation", "editorial_review", "finalization", "completed"]
        try:
            from_index = stage_order.index(from_stage)
            to_index = stage_order.index(to_stage)

            if to_index < from_index and to_stage != "error":
                unusual.append("Backward transition detected")
        except ValueError:
            unusual.append("Unknown stage in transition")

        # Check for transitions to error state
        if to_stage == "error":
            unusual.append("Transition to error state")

        # Check for skipped stages
        if from_index + 1 < to_index:
            unusual.append("Stage skipped in transition")

        return unusual

    def get_transition_history(
        self,
        session_id: Optional[str] = None,
        from_stage: Optional[str] = None,
        to_stage: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get filtered transition history."""
        transitions = self.transition_history.copy()

        # Apply filters
        if session_id:
            transitions = [t for t in transitions if t["session_id"] == session_id]

        if from_stage:
            transitions = [t for t in transitions if t["from_stage"] == from_stage]

        if to_stage:
            transitions = [t for t in transitions if t["to_stage"] == to_stage]

        # Sort by timestamp (most recent first) and limit
        transitions.sort(key=lambda t: t["timestamp"], reverse=True)
        return transitions[:limit]

    def get_transition_patterns(self) -> Dict[str, Any]:
        """Get transition pattern analysis."""
        total_transitions = sum(self.transition_patterns.values())

        # Find most common transitions
        sorted_patterns = sorted(self.transition_patterns.items(), key=lambda x: x[1], reverse=True)
        most_common = sorted_patterns[:10]

        # Calculate transition frequencies
        pattern_frequencies = {
            pattern: (count / total_transitions * 100) if total_transitions > 0 else 0.0
            for pattern, count in self.transition_patterns.items()
        }

        return {
            "total_transitions": total_transitions,
            "unique_patterns": len(self.transition_patterns),
            "most_common_transitions": [
                {"pattern": pattern, "count": count, "frequency": pattern_frequencies[pattern]}
                for pattern, count in most_common
            ],
            "pattern_frequencies": pattern_frequencies,
            "all_patterns": self.transition_patterns.copy()
        }


class DecisionPointHook(BaseHook):
    """Specialized hook for monitoring decision points in workflow execution."""

    def __init__(self, timeout: float = 30.0, enabled: bool = True):
        super().__init__(
            name="decision_point_monitor",
            hook_type="decision_point",
            priority=HookPriority.NORMAL,
            timeout=10.0,
            enabled=enabled,
            retry_count=0
        )
        self.decision_points: List[DecisionPoint] = []
        self.decision_patterns: Dict[str, Dict[str, Any]] = {}
        self.max_decisions = 300

    async def execute(self, context: HookContext) -> HookResult:
        """Execute decision point monitoring."""
        try:
            decision_data = context.metadata.get("decision", {})
            decision_point = decision_data.get("decision_point", "unknown")
            available_options = decision_data.get("available_options", [])
            chosen_option = decision_data.get("chosen_option", "unknown")
            decision_context = decision_data.get("decision_context", {})

            try:
                decision_type = DecisionType(decision_data.get("decision_type", "routing_decision"))
            except ValueError:
                decision_type = DecisionType.ROUTING_DECISION

            self.logger.info(f"Decision point monitored: {decision_point} -> {chosen_option}",
                           decision_point=decision_point,
                           chosen_option=chosen_option,
                           decision_type=decision_type.value,
                           session_id=context.session_id)

            # Create decision point record
            decision = DecisionPoint(
                decision_id=f"decision_{int(time.time())}_{decision_point}",
                timestamp=datetime.now(),
                workflow_type=context.metadata.get("workflow_type", "research_workflow"),
                session_id=context.session_id,
                stage=context.metadata.get("stage", WorkflowStage.INITIALIZATION),
                decision_type=decision_type,
                available_options=available_options,
                chosen_option=chosen_option,
                decision_context=decision_context
            )

            # Store decision
            self.decision_points.append(decision)
            if len(self.decision_points) > self.max_decisions:
                self.decision_points = self.decision_points[-self.max_decisions]

            # Update decision patterns
            self._update_decision_patterns(decision_point, decision_type, available_options, chosen_option)

            # Analyze decision quality
            decision_analysis = self._analyze_decision(decision)

            return HookResult(
                hook_name=self.name,
                hook_type=self.hook_type,
                status=HookStatus.COMPLETED,
                execution_id=context.execution_id,
                start_time=context.timestamp,
                end_time=datetime.now(),
                result_data={
                    "decision_id": decision.decision_id,
                    "decision_point": decision_point,
                    "decision_type": decision_type.value,
                    "available_options": available_options,
                    "chosen_option": chosen_option,
                    "decision_analysis": decision_analysis
                }
            )

        except Exception as e:
            self.logger.error(f"Decision point hook failed: {str(e)}",
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

    def _update_decision_patterns(
        self,
        decision_point: str,
        decision_type: DecisionType,
        available_options: List[str],
        chosen_option: str
    ):
        """Update decision pattern statistics."""
        key = f"{decision_point}:{decision_type.value}"

        if key not in self.decision_patterns:
            self.decision_patterns[key] = {
                "decision_point": decision_point,
                "decision_type": decision_type.value,
                "total_decisions": 0,
                "option_counts": {},
                "most_common_choice": None
            }

        pattern = self.decision_patterns[key]
        pattern["total_decisions"] += 1

        # Update option counts
        pattern["option_counts"][chosen_option] = pattern["option_counts"].get(chosen_option, 0) + 1

        # Update most common choice
        if pattern["option_counts"][chosen_option] > pattern["option_counts"].get(pattern["most_common_choice"], 0):
            pattern["most_common_choice"] = chosen_option

    def _analyze_decision(self, decision: DecisionPoint) -> Dict[str, Any]:
        """Analyze decision quality and patterns."""
        analysis = {
            "decision_quality": "unknown",
            "confidence_level": "medium",
            "risk_factors": [],
            "recommendations": []
        }

        # Analyze based on number of options
        if len(decision.available_options) == 1:
            analysis["confidence_level"] = "high"
            analysis["decision_quality"] = "deterministic"
        elif len(decision.available_options) > 5:
            analysis["confidence_level"] = "low"
            analysis["risk_factors"].append("Too many options may indicate unclear requirements")

        # Analyze based on decision type
        if decision.decision_type == DecisionType.ERROR_RECOVERY:
            analysis["risk_factors"].append("Error recovery decision")
            analysis["confidence_level"] = "medium"

        # Analyze context complexity
        if len(decision.decision_context) > 10:
            analysis["risk_factors"].append("Complex decision context")

        return analysis

    def get_decision_history(
        self,
        session_id: Optional[str] = None,
        decision_type: Optional[str] = None,
        decision_point: Optional[str] = None,
        limit: int = 100
    ) -> List[DecisionPoint]:
        """Get filtered decision history."""
        decisions = self.decision_points.copy()

        # Apply filters
        if session_id:
            decisions = [d for d in decisions if d.session_id == session_id]

        if decision_type:
            decisions = [d for d in decisions if d.decision_type.value == decision_type]

        if decision_point:
            decisions = [d for d in decisions if decision_point in str(d.decision_id)]

        # Sort by timestamp (most recent first) and limit
        decisions.sort(key=lambda d: d.timestamp, reverse=True)
        return decisions[:limit]

    def get_decision_patterns(self) -> Dict[str, Any]:
        """Get decision pattern analysis."""
        total_decisions = sum(pattern["total_decisions"] for pattern in self.decision_patterns.values())

        # Find most complex decision points
        complex_decisions = [
            {"point": pattern["decision_point"], "options": len(pattern["option_counts"])}
            for pattern in self.decision_patterns.values()
            if len(pattern["option_counts"]) > 3
        ]

        # Calculate decision diversity
        diverse_decisions = [
            pattern["decision_point"]
            for pattern in self.decision_patterns.values()
            if len(pattern["option_counts"]) >= 3
        ]

        return {
            "total_decision_points": len(self.decision_patterns),
            "total_decisions": total_decisions,
            "patterns": self.decision_patterns.copy(),
            "complex_decisions": sorted(complex_decisions, key=lambda x: x["options"], reverse=True)[:5],
            "diverse_decision_points": diverse_decisions,
            "average_options_per_decision": sum(len(p["option_counts"]) for p in self.decision_patterns.values()) / len(self.decision_patterns) if self.decision_patterns else 0
        }