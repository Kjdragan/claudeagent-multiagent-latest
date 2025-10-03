"""
Hook Logger for Multi-Agent Research System

Provides specialized logging capabilities for monitoring hook execution,
including tool use monitoring, agent communication tracking, and session lifecycle events.
"""

from typing import Any, Dict, Optional, List
from datetime import datetime
from pathlib import Path
import json

from .structured_logger import StructuredLogger, get_logger


class HookLogger:
    """Specialized logger for hook execution tracking and monitoring."""

    def __init__(self, hook_name: str, log_dir: Optional[Path] = None):
        """Initialize hook logger."""
        self.hook_name = hook_name
        self.logger = get_logger(f"hook.{hook_name}", log_dir=log_dir)

    def log_hook_execution_start(
        self,
        hook_type: str,
        session_id: str,
        trigger_event: str,
        **metadata
    ) -> None:
        """Log hook execution start event."""
        self.logger.info(
            f"Hook execution started: {self.hook_name} - {hook_type}",
            event_type="hook_execution_start",
            hook_name=self.hook_name,
            hook_type=hook_type,
            session_id=session_id,
            trigger_event=trigger_event,
            **metadata
        )

    def log_hook_execution_complete(
        self,
        hook_type: str,
        session_id: str,
        execution_time: float,
        result: Optional[Dict[str, Any]] = None,
        **metadata
    ) -> None:
        """Log hook execution completion event."""
        self.logger.info(
            f"Hook execution completed: {self.hook_name} - {hook_type}",
            event_type="hook_execution_complete",
            hook_name=self.hook_name,
            hook_type=hook_type,
            session_id=session_id,
            execution_time_seconds=execution_time,
            result=result,
            **metadata
        )

    def log_hook_execution_error(
        self,
        hook_type: str,
        session_id: str,
        error_type: str,
        error_message: str,
        execution_time: float,
        **metadata
    ) -> None:
        """Log hook execution error event."""
        self.logger.error(
            f"Hook execution failed: {self.hook_name} - {hook_type} - {error_type}",
            event_type="hook_execution_error",
            hook_name=self.hook_name,
            hook_type=hook_type,
            session_id=session_id,
            error_type=error_type,
            error_message=error_message,
            execution_time_seconds=execution_time,
            **metadata
        )


class ToolUseLogger(HookLogger):
    """Specialized logger for tool use hook monitoring."""

    def __init__(self, log_dir: Optional[Path] = None):
        super().__init__("tool_use_monitor", log_dir)

    def log_pre_tool_use(
        self,
        tool_name: str,
        tool_use_id: str,
        session_id: str,
        input_data: Dict[str, Any],
        agent_context: Dict[str, Any],
        **metadata
    ) -> None:
        """Log pre-tool-use hook execution."""
        self.log_hook_execution_start(
            hook_type="pre_tool_use",
            session_id=session_id,
            trigger_event="tool_execution_requested",
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            input_data=input_data,
            input_size_bytes=len(json.dumps(input_data, default=str)),
            agent_name=agent_context.get("agent_name", "unknown"),
            agent_type=agent_context.get("agent_type", "unknown"),
            **metadata
        )

    def log_post_tool_use(
        self,
        tool_name: str,
        tool_use_id: str,
        session_id: str,
        input_data: Dict[str, Any],
        result_data: Any,
        execution_time: float,
        success: bool,
        agent_context: Dict[str, Any],
        **metadata
    ) -> None:
        """Log post-tool-use hook execution."""
        # Calculate result size if possible
        result_size = None
        if result_data is not None:
            try:
                result_size = len(json.dumps(result_data, default=str))
            except (TypeError, ValueError):
                result_size = len(str(result_data))

        self.log_hook_execution_complete(
            hook_type="post_tool_use",
            session_id=session_id,
            execution_time=execution_time,
            result={"success": success, "result_size": result_size},
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            input_data=input_data,
            result_data=result_data,
            result_size_bytes=result_size,
            success=success,
            agent_name=agent_context.get("agent_name", "unknown"),
            agent_type=agent_context.get("agent_type", "unknown"),
            **metadata
        )

    def log_tool_use_blocked(
        self,
        tool_name: str,
        tool_use_id: str,
        session_id: str,
        block_reason: str,
        agent_context: Dict[str, Any],
        **metadata
    ) -> None:
        """Log tool use blocked event."""
        self.logger.warning(
            f"Tool use blocked: {tool_name} - {block_reason}",
            event_type="tool_use_blocked",
            hook_name=self.hook_name,
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            session_id=session_id,
            block_reason=block_reason,
            agent_name=agent_context.get("agent_name", "unknown"),
            agent_type=agent_context.get("agent_type", "unknown"),
            **metadata
        )


class AgentCommunicationLogger(HookLogger):
    """Specialized logger for agent communication hook monitoring."""

    def __init__(self, log_dir: Optional[Path] = None):
        super().__init__("agent_communication_monitor", log_dir)

    def log_user_prompt_submit(
        self,
        session_id: str,
        prompt_content: str,
        agent_context: Dict[str, Any],
        **metadata
    ) -> None:
        """Log user prompt submission hook."""
        self.log_hook_execution_start(
            hook_type="user_prompt_submit",
            session_id=session_id,
            trigger_event="user_input_received",
            prompt_content=prompt_content,
            prompt_length=len(prompt_content),
            agent_name=agent_context.get("agent_name", "unknown"),
            agent_type=agent_context.get("agent_type", "unknown"),
            **metadata
        )

    def log_agent_message_send(
        self,
        session_id: str,
        message_content: str,
        recipient_agent: str,
        sender_context: Dict[str, Any],
        **metadata
    ) -> None:
        """Log agent message sending hook."""
        self.log_hook_execution_start(
            hook_type="agent_message_send",
            session_id=session_id,
            trigger_event="agent_communication",
            message_content=message_content,
            message_length=len(message_content),
            recipient_agent=recipient_agent,
            sender_agent=sender_context.get("agent_name", "unknown"),
            sender_type=sender_context.get("agent_type", "unknown"),
            **metadata
        )

    def log_agent_message_receive(
        self,
        session_id: str,
        message_content: str,
        sender_agent: str,
        receiver_context: Dict[str, Any],
        **metadata
    ) -> None:
        """Log agent message receiving hook."""
        self.log_hook_execution_complete(
            hook_type="agent_message_receive",
            session_id=session_id,
            execution_time=0.0,  # Receive is typically instant
            result={"message_received": True},
            message_content=message_content,
            message_length=len(message_content),
            sender_agent=sender_agent,
            receiver_agent=receiver_context.get("agent_name", "unknown"),
            receiver_type=receiver_context.get("agent_type", "unknown"),
            **metadata
        )

    def log_agent_handoff(
        self,
        session_id: str,
        from_agent: str,
        to_agent: str,
        handoff_reason: str,
        context_data: Dict[str, Any],
        **metadata
    ) -> None:
        """Log agent handoff hook."""
        self.logger.info(
            f"Agent handoff: {from_agent} -> {to_agent} ({handoff_reason})",
            event_type="agent_handoff",
            hook_name=self.hook_name,
            session_id=session_id,
            from_agent=from_agent,
            to_agent=to_agent,
            handoff_reason=handoff_reason,
            context_size=len(json.dumps(context_data, default=str)),
            **context_data,
            **metadata
        )


class SessionLifecycleLogger(HookLogger):
    """Specialized logger for session lifecycle hook monitoring."""

    def __init__(self, log_dir: Optional[Path] = None):
        super().__init__("session_lifecycle_monitor", log_dir)

    def log_session_creation(
        self,
        session_id: str,
        session_config: Dict[str, Any],
        user_context: Dict[str, Any],
        **metadata
    ) -> None:
        """Log session creation hook."""
        self.log_hook_execution_start(
            hook_type="session_creation",
            session_id=session_id,
            trigger_event="system_initialization",
            session_config=session_config,
            user_context=user_context,
            **metadata
        )

    def log_session_resumption(
        self,
        session_id: str,
        previous_state: Dict[str, Any],
        resumption_context: Dict[str, Any],
        **metadata
    ) -> None:
        """Log session resumption hook."""
        self.log_hook_execution_start(
            hook_type="session_resumption",
            session_id=session_id,
            trigger_event="session_restore",
            previous_state_size=len(json.dumps(previous_state, default=str)),
            resumption_context=resumption_context,
            **metadata
        )

    def log_session_pause(
        self,
        session_id: str,
        pause_reason: str,
        state_snapshot: Dict[str, Any],
        **metadata
    ) -> None:
        """Log session pause hook."""
        self.log_hook_execution_complete(
            hook_type="session_pause",
            session_id=session_id,
            execution_time=0.0,
            result={"pause_reason": pause_reason},
            pause_reason=pause_reason,
            state_size=len(json.dumps(state_snapshot, default=str)),
            **metadata
        )

    def log_session_termination(
        self,
        session_id: str,
        termination_reason: str,
        final_state: Dict[str, Any],
        **metadata
    ) -> None:
        """Log session termination hook."""
        self.log_hook_execution_complete(
            hook_type="session_termination",
            session_id=session_id,
            execution_time=0.0,
            result={"termination_reason": termination_reason},
            termination_reason=termination_reason,
            final_state_size=len(json.dumps(final_state, default=str)),
            **metadata
        )

    def log_session_error(
        self,
        session_id: str,
        error_type: str,
        error_context: Dict[str, Any],
        recovery_action: Optional[str] = None,
        **metadata
    ) -> None:
        """Log session error hook."""
        self.log_hook_execution_error(
            hook_type="session_error",
            session_id=session_id,
            error_type=error_type,
            error_message=str(error_context.get("error", "Unknown error")),
            execution_time=0.0,
            error_context=error_context,
            recovery_action=recovery_action,
            **metadata
        )


class WorkflowLogger(HookLogger):
    """Specialized logger for workflow execution hook monitoring."""

    def __init__(self, log_dir: Optional[Path] = None):
        super().__init__("workflow_monitor", log_dir)

    def log_workflow_stage_start(
        self,
        session_id: str,
        workflow_type: str,
        stage_name: str,
        stage_config: Dict[str, Any],
        **metadata
    ) -> None:
        """Log workflow stage start hook."""
        self.log_hook_execution_start(
            hook_type="workflow_stage_start",
            session_id=session_id,
            trigger_event="workflow_progression",
            workflow_type=workflow_type,
            stage_name=stage_name,
            stage_config=stage_config,
            **metadata
        )

    def log_workflow_stage_complete(
        self,
        session_id: str,
        workflow_type: str,
        stage_name: str,
        stage_result: Dict[str, Any],
        execution_time: float,
        **metadata
    ) -> None:
        """Log workflow stage completion hook."""
        self.log_hook_execution_complete(
            hook_type="workflow_stage_complete",
            session_id=session_id,
            execution_time=execution_time,
            result=stage_result,
            workflow_type=workflow_type,
            stage_name=stage_name,
            stage_result=stage_result,
            **metadata
        )

    def log_workflow_decision_point(
        self,
        session_id: str,
        workflow_type: str,
        decision_point: str,
        available_options: List[str],
        chosen_option: str,
        decision_context: Dict[str, Any],
        **metadata
    ) -> None:
        """Log workflow decision point hook."""
        self.logger.info(
            f"Workflow decision: {decision_point} -> {chosen_option}",
            event_type="workflow_decision",
            hook_name=self.hook_name,
            session_id=session_id,
            workflow_type=workflow_type,
            decision_point=decision_point,
            available_options=available_options,
            chosen_option=chosen_option,
            options_count=len(available_options),
            decision_context=decision_context,
            **metadata
        )

    def log_workflow_error(
        self,
        session_id: str,
        workflow_type: str,
        error_stage: str,
        error_type: str,
        error_context: Dict[str, Any],
        recovery_attempted: bool,
        **metadata
    ) -> None:
        """Log workflow error hook."""
        self.log_hook_execution_error(
            hook_type="workflow_error",
            session_id=session_id,
            error_type=error_type,
            error_message=str(error_context.get("error", "Unknown workflow error")),
            execution_time=0.0,
            workflow_type=workflow_type,
            error_stage=error_stage,
            error_context=error_context,
            recovery_attempted=recovery_attempted,
            **metadata
        )