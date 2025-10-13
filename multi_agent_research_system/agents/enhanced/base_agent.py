"""Enhanced Agent Base Classes - Comprehensive Claude Agent SDK Integration

This module provides enhanced base classes for all agents in the research system,
featuring comprehensive Claude Agent SDK integration, advanced configuration management,
performance monitoring, and sophisticated communication protocols.

Key Features:
- Comprehensive SDK Options Configuration
- Advanced Performance Monitoring
- Rich Message Processing with Phase 2.3 integration
- Agent Lifecycle Management
- Factory Pattern Implementation
- Error Recovery and Resilience
"""

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    tool,
    Hook,
    Message as SDKMessage,
)


class AgentStatus(Enum):
    """Agent lifecycle status."""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    TERMINATED = "terminated"


class AgentPriority(Enum):
    """Agent execution priority."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AgentConfiguration:
    """Comprehensive agent configuration."""
    agent_type: str
    agent_id: str
    max_turns: int = 50
    continue_conversation: bool = True
    include_partial_messages: bool = True
    enable_hooks: bool = True
    timeout_seconds: int = 300
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    priority: AgentPriority = AgentPriority.NORMAL
    allowed_tools: List[str] = field(default_factory=list)
    blocked_tools: List[str] = field(default_factory=list)
    mcp_servers: Dict[str, Any] = field(default_factory=dict)
    custom_options: Dict[str, Any] = field(default_factory=dict)
    performance_monitoring: bool = True
    debug_mode: bool = False
    quality_threshold: float = 0.75
    max_memory_mb: int = 512
    concurrent_sessions: int = 5


@dataclass
class AgentPerformanceMetrics:
    """Performance metrics for agent monitoring."""
    agent_id: str
    session_id: str
    start_time: datetime
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    tool_usage_count: Dict[str, int] = field(default_factory=dict)
    error_count: int = 0
    last_activity: Optional[datetime] = None
    quality_scores: List[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    @property
    def average_quality_score(self) -> float:
        """Calculate average quality score."""
        if not self.quality_scores:
            return 0.0
        return sum(self.quality_scores) / len(self.quality_scores)


@dataclass
class RichMessage:
    """Enhanced message structure with rich metadata and processing capabilities."""

    sender: str
    recipient: str
    message_type: str
    payload: Dict[str, Any]
    session_id: str
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    priority: AgentPriority = AgentPriority.NORMAL
    requires_response: bool = False
    response_timeout_seconds: int = 30
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    tracking_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "message_type": self.message_type,
            "payload": self.payload,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.value,
            "requires_response": self.requires_response,
            "response_timeout_seconds": self.response_timeout_seconds,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "metadata": self.metadata,
            "attachments": self.attachments,
            "tracking_info": self.tracking_info
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RichMessage":
        """Create message from dictionary."""
        # Handle timestamp conversion
        timestamp = datetime.fromisoformat(data["timestamp"]) if isinstance(data.get("timestamp"), str) else data.get("timestamp", datetime.now())

        # Handle priority conversion
        priority = AgentPriority(data["priority"]) if isinstance(data.get("priority"), int) else data.get("priority", AgentPriority.NORMAL)

        msg = cls(
            sender=data["sender"],
            recipient=data["recipient"],
            message_type=data["message_type"],
            payload=data["payload"],
            session_id=data["session_id"],
            correlation_id=data.get("correlation_id", str(uuid.uuid4())),
            timestamp=timestamp,
            priority=priority,
            requires_response=data.get("requires_response", False),
            response_timeout_seconds=data.get("response_timeout_seconds", 30),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            metadata=data.get("metadata", {}),
            attachments=data.get("attachments", []),
            tracking_info=data.get("tracking_info", {})
        )
        return msg


class EnhancedBaseAgent(ABC):
    """Enhanced base agent class with comprehensive SDK integration and advanced features."""

    def __init__(self, config: AgentConfiguration):
        self.config = config
        self.agent_id = config.agent_id
        self.agent_type = config.agent_type
        self.status = AgentStatus.INITIALIZING
        self.logger = logging.getLogger(f"agent.{self.agent_id}")

        # SDK Components
        self.client: Optional[ClaudeSDKClient] = None
        self.sdk_options: Optional[ClaudeAgentOptions] = None

        # Message Handling
        self.message_handlers: Dict[str, Callable] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.pending_responses: Dict[str, RichMessage] = {}

        # Session Management
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_locks: Dict[str, asyncio.Lock] = {}

        # Performance Monitoring
        self.performance_metrics: Dict[str, AgentPerformanceMetrics] = {}
        self.monitoring_enabled = config.performance_monitoring

        # Error Handling
        self.error_handlers: Dict[str, Callable] = {}
        self.circuit_breaker_active = False
        self.circuit_breaker_reset_time: Optional[datetime] = None

        # Hooks Integration
        self.hooks: Dict[str, List[Callable]] = {}

        # Communication
        self.agent_registry: Optional["AgentRegistry"] = None

        self.logger.info(f"Enhanced agent {self.agent_id} initialized with type {self.agent_type}")

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        pass

    @abstractmethod
    def get_default_tools(self) -> List[str]:
        """Get the default tools for this agent type."""
        pass

    async def initialize(self, agent_registry: Optional["AgentRegistry"] = None) -> None:
        """Initialize the agent with comprehensive SDK setup."""
        try:
            self.status = AgentStatus.INITIALIZING
            self.agent_registry = agent_registry

            # Create SDK options
            self.sdk_options = self._create_sdk_options()

            # Initialize SDK client
            self.client = ClaudeSDKClient(options=self.sdk_options)
            await self.client.connect()

            # Register message handlers
            self._register_default_handlers()

            # Setup hooks if enabled
            if self.config.enable_hooks:
                self._setup_hooks()

            # Start message processing
            asyncio.create_task(self._process_message_queue())

            # Start performance monitoring
            if self.monitoring_enabled:
                asyncio.create_task(self._monitor_performance())

            self.status = AgentStatus.READY
            self.logger.info(f"Agent {self.agent_id} initialized successfully")

        except Exception as e:
            self.status = AgentStatus.ERROR
            self.logger.error(f"Failed to initialize agent {self.agent_id}: {e}")
            raise

    def _create_sdk_options(self) -> ClaudeAgentOptions:
        """Create comprehensive SDK options."""
        allowed_tools = self.config.allowed_tools or self.get_default_tools()

        options = ClaudeAgentOptions(
            max_turns=self.config.max_turns,
            continue_conversation=self.config.continue_conversation,
            include_partial_messages=self.config.include_partial_messages,
            enable_hooks=self.config.enable_hooks,
            system_prompt=self.get_system_prompt(),
            allowed_tools=allowed_tools,
            blocked_tools=self.config.blocked_tools,
            mcp_servers=self.config.mcp_servers,
            timeout=self.config.timeout_seconds,
            **self.config.custom_options
        )

        return options

    def _register_default_handlers(self) -> None:
        """Register default message handlers."""
        self.register_message_handler("ping", self._handle_ping)
        self.register_message_handler("status", self._handle_status)
        self.register_message_handler("shutdown", self._handle_shutdown)
        self.register_message_handler("health_check", self._handle_health_check)

    def _setup_hooks(self) -> None:
        """Setup SDK hooks for monitoring and validation."""
        if "PreToolUse" not in self.hooks:
            self.hooks["PreToolUse"] = []

        # Add performance monitoring hooks
        self.hooks["PreToolUse"].append(self._hook_pre_tool_use)
        self.hooks["PostToolUse"] = [self._hook_post_tool_use]
        self.hooks["PreMessageSend"] = [self._hook_pre_message_send]
        self.hooks["PostMessageReceive"] = [self._hook_post_message_receive]

    def register_message_handler(self, message_type: str, handler: Callable) -> None:
        """Register a handler for a specific message type."""
        self.message_handlers[message_type] = handler
        self.logger.debug(f"Registered handler for message type: {message_type}")

    def register_error_handler(self, error_type: str, handler: Callable) -> None:
        """Register an error handler for specific error types."""
        self.error_handlers[error_type] = handler
        self.logger.debug(f"Registered error handler for: {error_type}")

    async def send_message(self, message: RichMessage) -> Optional[RichMessage]:
        """Send a message to another agent with rich features."""
        if self.status != AgentStatus.READY:
            raise RuntimeError(f"Agent {self.agent_id} is not ready for sending messages")

        try:
            # Add tracking information
            message.tracking_info.update({
                "sender_agent_id": self.agent_id,
                "send_timestamp": datetime.now().isoformat(),
                "message_id": str(uuid.uuid4())
            })

            # Route message through registry or direct
            if self.agent_registry:
                await self.agent_registry.route_message(message)
            else:
                # Direct sending (fallback)
                await self._send_message_direct(message)

            # Log the send
            self.logger.info(f"Sent message type {message.message_type} to {message.recipient}")

            return message

        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            await self._handle_send_error(message, e)
            return None

    async def receive_message(self, message: RichMessage) -> None:
        """Receive and queue a message for processing."""
        await self.message_queue.put(message)
        self.logger.debug(f"Queued message type {message.message_type} from {message.sender}")

    async def _process_message_queue(self) -> None:
        """Process messages from the queue."""
        while self.status not in [AgentStatus.SHUTTING_DOWN, AgentStatus.TERMINATED]:
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )

                # Process the message
                await self._handle_message(message)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing message queue: {e}")

    async def _handle_message(self, message: RichMessage) -> None:
        """Handle an incoming message."""
        session_id = message.session_id

        # Update status
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {
                "start_time": datetime.now(),
                "message_count": 0,
                "last_activity": datetime.now()
            }

        self.active_sessions[session_id]["message_count"] += 1
        self.active_sessions[session_id]["last_activity"] = datetime.now()

        # Update performance metrics
        if self.monitoring_enabled:
            await self._update_message_metrics(session_id)

        # Set status to busy during processing
        old_status = self.status
        self.status = AgentStatus.BUSY

        try:
            # Check for handler
            handler = self.message_handlers.get(message.message_type)
            if handler:
                self.logger.info(f"Processing message type {message.message_type} from {message.sender}")

                # Execute handler with timeout
                await asyncio.wait_for(
                    handler(message),
                    timeout=message.response_timeout_seconds
                )
            else:
                self.logger.warning(f"No handler found for message type: {message.message_type}")
                await self._handle_unknown_message(message)

            # Send response if required
            if message.requires_response:
                await self._send_response(message)

        except asyncio.TimeoutError:
            self.logger.error(f"Timeout processing message type {message.message_type}")
            await self._handle_message_timeout(message)
        except Exception as e:
            self.logger.error(f"Error handling message type {message.message_type}: {e}")
            await self._handle_message_error(message, e)
        finally:
            # Restore status
            self.status = old_status

    async def execute_with_session(self, session_id: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Execute agent task within a session context."""
        if self.status != AgentStatus.READY:
            raise RuntimeError(f"Agent {self.agent_id} is not ready")

        if self.circuit_breaker_active:
            raise RuntimeError("Circuit breaker is active - agent temporarily unavailable")

        # Get or create session lock
        if session_id not in self.session_locks:
            self.session_locks[session_id] = asyncio.Lock()

        async with self.session_locks[session_id]:
            try:
                # Initialize session if needed
                if session_id not in self.active_sessions:
                    self.active_sessions[session_id] = {
                        "start_time": datetime.now(),
                        "message_count": 0,
                        "last_activity": datetime.now(),
                        "requests": 0,
                        "errors": 0
                    }

                # Update session metrics
                self.active_sessions[session_id]["requests"] += 1
                self.active_sessions[session_id]["last_activity"] = datetime.now()

                # Execute with SDK client
                start_time = datetime.now()
                result = await self.client.query(prompt)
                execution_time = (datetime.now() - start_time).total_seconds()

                # Update performance metrics
                if self.monitoring_enabled:
                    await self._update_execution_metrics(session_id, execution_time, True)

                return {
                    "success": True,
                    "result": result,
                    "execution_time": execution_time,
                    "session_id": session_id,
                    "agent_id": self.agent_id
                }

            except Exception as e:
                # Update error metrics
                if session_id in self.active_sessions:
                    self.active_sessions[session_id]["errors"] += 1

                if self.monitoring_enabled:
                    await self._update_execution_metrics(session_id, 0, False)

                # Check error handlers
                error_handler = self.error_handlers.get(type(e).__name__)
                if error_handler:
                    return await error_handler(e, session_id, prompt, **kwargs)

                # Default error handling
                return await self._handle_execution_error(e, session_id, prompt, **kwargs)

    # Default Message Handlers

    async def _handle_ping(self, message: RichMessage) -> None:
        """Handle ping message."""
        response = RichMessage(
            sender=self.agent_id,
            recipient=message.sender,
            message_type="pong",
            payload={"timestamp": datetime.now().isoformat()},
            session_id=message.session_id,
            correlation_id=message.correlation_id
        )
        await self.send_message(response)

    async def _handle_status(self, message: RichMessage) -> None:
        """Handle status request."""
        status_info = {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": self.status.value,
            "active_sessions": len(self.active_sessions),
            "queue_size": self.message_queue.qsize(),
            "circuit_breaker_active": self.circuit_breaker_active,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds() if hasattr(self, 'start_time') else 0
        }

        response = RichMessage(
            sender=self.agent_id,
            recipient=message.sender,
            message_type="status_response",
            payload=status_info,
            session_id=message.session_id,
            correlation_id=message.correlation_id
        )
        await self.send_message(response)

    async def _handle_health_check(self, message: RichMessage) -> None:
        """Handle health check request."""
        health_status = {
            "healthy": self.status == AgentStatus.READY,
            "status": self.status.value,
            "memory_usage_mb": self._get_memory_usage(),
            "active_sessions": len(self.active_sessions),
            "error_rate": self._calculate_error_rate(),
            "last_activity": self._get_last_activity()
        }

        response = RichMessage(
            sender=self.agent_id,
            recipient=message.sender,
            message_type="health_check_response",
            payload=health_status,
            session_id=message.session_id,
            correlation_id=message.correlation_id
        )
        await self.send_message(response)

    async def _handle_shutdown(self, message: RichMessage) -> None:
        """Handle shutdown request."""
        self.logger.info(f"Received shutdown request from {message.sender}")
        await self.shutdown()

    # Hook Methods

    async def _hook_pre_tool_use(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Hook called before tool use."""
        if self.monitoring_enabled:
            session_id = self._get_current_session_id()
            if session_id and session_id in self.performance_metrics:
                metrics = self.performance_metrics[session_id]
                metrics.tool_usage_count[tool_name] = metrics.tool_usage_count.get(tool_name, 0) + 1

        self.logger.debug(f"Pre-tool use hook: {tool_name}")
        return arguments

    async def _hook_post_tool_use(self, tool_name: str, arguments: Dict[str, Any], result: Any) -> Any:
        """Hook called after tool use."""
        self.logger.debug(f"Post-tool use hook: {tool_name}")
        return result

    async def _hook_pre_message_send(self, message: RichMessage) -> RichMessage:
        """Hook called before message send."""
        self.logger.debug(f"Pre-message send hook: {message.message_type}")
        return message

    async def _hook_post_message_receive(self, message: RichMessage) -> RichMessage:
        """Hook called after message receive."""
        self.logger.debug(f"Post-message receive hook: {message.message_type}")
        return message

    # Performance Monitoring Methods

    async def _monitor_performance(self) -> None:
        """Monitor agent performance continuously."""
        while self.status not in [AgentStatus.SHUTTING_DOWN, AgentStatus.TERMINATED]:
            try:
                # Update metrics for all active sessions
                for session_id in list(self.active_sessions.keys()):
                    await self._update_session_metrics(session_id)

                # Check circuit breaker
                await self._check_circuit_breaker()

                # Cleanup inactive sessions
                await self._cleanup_inactive_sessions()

                await asyncio.sleep(30)  # Monitor every 30 seconds

            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(60)  # Back off on error

    async def _update_session_metrics(self, session_id: str) -> None:
        """Update performance metrics for a session."""
        if not self.monitoring_enabled or session_id not in self.performance_metrics:
            return

        metrics = self.performance_metrics[session_id]
        metrics.memory_usage_mb = self._get_memory_usage()
        metrics.cpu_usage_percent = self._get_cpu_usage()
        metrics.last_activity = datetime.now()

    async def _update_execution_metrics(self, session_id: str, execution_time: float, success: bool) -> None:
        """Update execution metrics."""
        if not self.monitoring_enabled:
            return

        if session_id not in self.performance_metrics:
            self.performance_metrics[session_id] = AgentPerformanceMetrics(
                agent_id=self.agent_id,
                session_id=session_id,
                start_time=datetime.now()
            )

        metrics = self.performance_metrics[session_id]
        metrics.total_requests += 1

        if success:
            metrics.successful_requests += 1
        else:
            metrics.failed_requests += 1
            metrics.error_count += 1

        # Update average response time
        total_time = metrics.average_response_time * (metrics.total_requests - 1) + execution_time
        metrics.average_response_time = total_time / metrics.total_requests

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil
            process = psutil.Process()
            return process.cpu_percent()
        except ImportError:
            return 0.0

    def _calculate_error_rate(self) -> float:
        """Calculate current error rate."""
        total_requests = sum(m.total_requests for m in self.performance_metrics.values())
        total_errors = sum(m.error_count for m in self.performance_metrics.values())

        if total_requests == 0:
            return 0.0
        return total_errors / total_requests

    def _get_last_activity(self) -> Optional[datetime]:
        """Get last activity timestamp."""
        if not self.active_sessions:
            return None

        return max(
            session["last_activity"]
            for session in self.active_sessions.values()
        )

    # Lifecycle Management

    async def shutdown(self) -> None:
        """Shutdown the agent gracefully."""
        self.status = AgentStatus.SHUTTING_DOWN
        self.logger.info(f"Shutting down agent {self.agent_id}")

        try:
            # Stop accepting new messages
            # (message queue processing will stop naturally)

            # Close SDK client
            if self.client:
                await self.client.close()

            # Cleanup sessions
            for session_id in list(self.active_sessions.keys()):
                await self._cleanup_session(session_id)

            self.status = AgentStatus.TERMINATED
            self.logger.info(f"Agent {self.agent_id} shutdown completed")

        except Exception as e:
            self.status = AgentStatus.ERROR
            self.logger.error(f"Error during agent shutdown: {e}")

    async def _cleanup_session(self, session_id: str) -> None:
        """Cleanup a session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]

        if session_id in self.session_locks:
            del self.session_locks[session_id]

        if session_id in self.performance_metrics:
            del self.performance_metrics[session_id]

    async def _cleanup_inactive_sessions(self) -> None:
        """Cleanup inactive sessions."""
        cutoff_time = datetime.now() - timedelta(hours=1)

        inactive_sessions = [
            session_id for session_id, session_data in self.active_sessions.items()
            if session_data["last_activity"] < cutoff_time
        ]

        for session_id in inactive_sessions:
            await self._cleanup_session(session_id)
            self.logger.info(f"Cleaned up inactive session: {session_id}")

    # Error Handling

    async def _handle_execution_error(self, error: Exception, session_id: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Handle execution error with recovery strategies."""
        self.logger.error(f"Execution error in session {session_id}: {error}")

        # Check for circuit breaker trigger
        error_rate = self._calculate_error_rate()
        if error_rate > 0.5:  # 50% error rate triggers circuit breaker
            self.circuit_breaker_active = True
            self.circuit_breaker_reset_time = datetime.now() + timedelta(minutes=5)
            self.logger.warning("Circuit breaker activated due to high error rate")

        return {
            "success": False,
            "error": str(error),
            "error_type": type(error).__name__,
            "session_id": session_id,
            "agent_id": self.agent_id,
            "circuit_breaker_active": self.circuit_breaker_active
        }

    async def _handle_unknown_message(self, message: RichMessage) -> None:
        """Handle unknown message type."""
        self.logger.warning(f"Unknown message type: {message.message_type}")

    async def _handle_message_timeout(self, message: RichMessage) -> None:
        """Handle message processing timeout."""
        self.logger.error(f"Timeout processing message type {message.message_type}")

    async def _handle_message_error(self, message: RichMessage, error: Exception) -> None:
        """Handle message processing error."""
        self.logger.error(f"Error processing message type {message.message_type}: {error}")

    async def _handle_send_error(self, message: RichMessage, error: Exception) -> None:
        """Handle message send error."""
        self.logger.error(f"Error sending message to {message.recipient}: {error}")

    async def _send_response(self, original_message: RichMessage) -> None:
        """Send response for messages that require it."""
        response = RichMessage(
            sender=self.agent_id,
            recipient=original_message.sender,
            message_type=f"{original_message.message_type}_response",
            payload={"status": "processed", "timestamp": datetime.now().isoformat()},
            session_id=original_message.session_id,
            correlation_id=original_message.correlation_id
        )
        await self.send_message(response)

    async def _send_message_direct(self, message: RichMessage) -> None:
        """Send message directly (fallback when registry not available)."""
        # This would implement direct message sending
        # For now, just log the attempt
        self.logger.info(f"Direct message send to {message.recipient}: {message.message_type}")

    def _get_current_session_id(self) -> Optional[str]:
        """Get current session ID from context."""
        # This would be implemented based on context management
        return None

    async def _check_circuit_breaker(self) -> None:
        """Check and reset circuit breaker if needed."""
        if self.circuit_breaker_active and self.circuit_breaker_reset_time:
            if datetime.now() >= self.circuit_breaker_reset_time:
                self.circuit_breaker_active = False
                self.circuit_breaker_reset_time = None
                self.logger.info("Circuit breaker reset - agent available again")

    async def _update_message_metrics(self, session_id: str) -> None:
        """Update message processing metrics."""
        if not self.monitoring_enabled:
            return

        if session_id not in self.performance_metrics:
            self.performance_metrics[session_id] = AgentPerformanceMetrics(
                agent_id=self.agent_id,
                session_id=session_id,
                start_time=datetime.now()
            )

        metrics = self.performance_metrics[session_id]
        metrics.last_activity = datetime.now()

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.monitoring_enabled:
            return {"monitoring_enabled": False}

        total_sessions = len(self.performance_metrics)
        if total_sessions == 0:
            return {
                "monitoring_enabled": True,
                "total_sessions": 0,
                "agent_status": self.status.value
            }

        # Aggregate metrics across all sessions
        total_requests = sum(m.total_requests for m in self.performance_metrics.values())
        total_successful = sum(m.successful_requests for m in self.performance_metrics.values())
        total_errors = sum(m.error_count for m in self.performance_metrics.values())
        avg_response_time = sum(m.average_response_time for m in self.performance_metrics.values()) / total_sessions
        avg_memory = sum(m.memory_usage_mb for m in self.performance_metrics.values()) / total_sessions
        avg_quality = sum(m.average_quality_score for m in self.performance_metrics.values()) / total_sessions

        return {
            "monitoring_enabled": True,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "agent_status": self.status.value,
            "total_sessions": total_sessions,
            "active_sessions": len(self.active_sessions),
            "total_requests": total_requests,
            "successful_requests": total_successful,
            "failed_requests": total_requests - total_successful,
            "error_count": total_errors,
            "overall_success_rate": total_successful / total_requests if total_requests > 0 else 0,
            "average_response_time": avg_response_time,
            "average_memory_usage_mb": avg_memory,
            "average_quality_score": avg_quality,
            "circuit_breaker_active": self.circuit_breaker_active,
            "tool_usage": self._aggregate_tool_usage(),
            "last_activity": self._get_last_activity().isoformat() if self._get_last_activity() else None
        }

    def _aggregate_tool_usage(self) -> Dict[str, int]:
        """Aggregate tool usage across all sessions."""
        tool_usage = {}
        for metrics in self.performance_metrics.values():
            for tool_name, count in metrics.tool_usage_count.items():
                tool_usage[tool_name] = tool_usage.get(tool_name, 0) + count
        return tool_usage