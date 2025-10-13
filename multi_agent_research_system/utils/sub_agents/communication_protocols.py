"""
Sub-Agent Communication Protocols

This module provides communication protocols and message passing mechanisms
for coordinating between different sub-agents while maintaining context
isolation and proper message routing.
"""

import asyncio
import uuid
import json
from typing import Dict, List, Any, Optional, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from asyncio import Queue, PriorityQueue


logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages between sub-agents."""

    # Direct communication
    DIRECT_MESSAGE = "direct_message"
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"

    # Workflow coordination
    TASK_ASSIGNMENT = "task_assignment"
    TASK_COMPLETION = "task_completion"
    WORKFLOW_HANDOFF = "workflow_handoff"
    STATUS_UPDATE = "status_update"

    # Data sharing
    DATA_REQUEST = "data_request"
    DATA_RESPONSE = "data_response"
    DATA_SHARE = "data_share"

    # Error handling
    ERROR_REPORT = "error_report"
    ERROR_RECOVERY = "error_recovery"

    # System messages
    HEARTBEAT = "heartbeat"
    SHUTDOWN = "shutdown"
    PING = "ping"
    PONG = "pong"


class MessagePriority(Enum):
    """Message priority levels."""

    CRITICAL = 1    # System critical messages
    HIGH = 2        # High priority workflow messages
    NORMAL = 3      # Normal priority messages
    LOW = 4         # Low priority background messages
    BULK = 5        # Bulk data transfer messages


@dataclass
class SubAgentMessage:
    """Message structure for sub-agent communication."""

    message_id: str
    message_type: MessageType
    priority: MessagePriority
    sender_id: str
    sender_type: str
    recipient_id: Optional[str]  # None for broadcast
    recipient_type: Optional[str]  # None for broadcast to all types
    session_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    context_data: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300
    requires_response: bool = False
    response_received: bool = False

    def is_expired(self) -> bool:
        """Check if the message has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def can_retry(self) -> bool:
        """Check if the message can be retried."""
        return self.retry_count < self.max_retries

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "priority": self.priority.value,
            "sender_id": self.sender_id,
            "sender_type": self.sender_type,
            "recipient_id": self.recipient_id,
            "recipient_type": self.recipient_type,
            "session_id": self.session_id,
            "payload": self.payload,
            "context_data": self.context_data,
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "requires_response": self.requires_response,
            "response_received": self.response_received
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubAgentMessage":
        """Create message from dictionary."""
        message = cls(
            message_id=data["message_id"],
            message_type=MessageType(data["message_type"]),
            priority=MessagePriority(data["priority"]),
            sender_id=data["sender_id"],
            sender_type=data["sender_type"],
            recipient_id=data.get("recipient_id"),
            recipient_type=data.get("recipient_type"),
            session_id=data["session_id"],
            payload=data.get("payload", {}),
            context_data=data.get("context_data", {}),
            correlation_id=data.get("correlation_id"),
            reply_to=data.get("reply_to"),
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            timeout_seconds=data.get("timeout_seconds", 300),
            requires_response=data.get("requires_response", False),
            response_received=data.get("response_received", False)
        )
        return message


@dataclass
class MessageHandler:
    """Message handler registration."""

    handler_id: str
    agent_id: str
    agent_type: str
    message_types: Set[MessageType]
    handler_function: Callable
    priority_filter: Optional[Set[MessagePriority]] = None
    context_filter: Optional[Dict[str, Any]] = None
    active: bool = True
    message_count: int = 0
    last_activity: datetime = field(default_factory=datetime.now)

    def can_handle_message(self, message: SubAgentMessage) -> bool:
        """Check if this handler can handle the given message."""
        if not self.active:
            return False

        if message.message_type not in self.message_types:
            return False

        if self.priority_filter and message.priority not in self.priority_filter:
            return False

        # Check context filter if present
        if self.context_filter:
            for key, expected_value in self.context_filter.items():
                if message.context_data.get(key) != expected_value:
                    return False

        return True

    async def handle_message(self, message: SubAgentMessage) -> Optional[SubAgentMessage]:
        """Handle a message and optionally return a response."""
        try:
            self.message_count += 1
            self.last_activity = datetime.now()

            if asyncio.iscoroutinefunction(self.handler_function):
                return await self.handler_function(message)
            else:
                return self.handler_function(message)

        except Exception as e:
            logger.error(f"Error in message handler {self.handler_id}: {e}")
            # Create error response
            return SubAgentMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.ERROR_REPORT,
                priority=MessagePriority.HIGH,
                sender_id=self.agent_id,
                sender_type=self.agent_type,
                recipient_id=message.sender_id,
                recipient_type=message.sender_type,
                session_id=message.session_id,
                payload={
                    "error": str(e),
                    "original_message_id": message.message_id,
                    "handler_id": self.handler_id
                },
                correlation_id=message.correlation_id,
                reply_to=message.message_id
            )


class SubAgentCommunicationManager:
    """
    Manages communication protocols and message passing between sub-agents
    while maintaining proper routing, prioritization, and isolation.
    """

    def __init__(self):
        self.message_handlers: Dict[str, MessageHandler] = {}
        self.message_queue: PriorityQueue = PriorityQueue()
        self.pending_responses: Dict[str, SubAgentMessage] = {}
        self.broadcast_subscribers: Dict[str, Set[str]] = {}  # message_type -> handler_ids
        self.type_subscribers: Dict[str, Set[str]] = {}  # agent_type -> handler_ids
        self.communication_stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "messages_failed": 0,
            "handlers_registered": 0,
            "broadcast_messages": 0
        }
        self.communication_config = {
            "max_queue_size": 1000,
            "message_timeout_seconds": 300,
            "max_pending_responses": 100,
            "cleanup_interval": 60,  # seconds
            "enable_broadcast": True,
            "enable_priority_routing": True,
            "retry_failed_messages": True,
            "max_message_size": 1048576  # 1MB
        }
        self._processing_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

    async def initialize(self):
        """Initialize the communication manager."""
        logger.info("Initializing Sub-Agent Communication Manager")
        self._running = True

        # Start processing tasks
        self._processing_task = asyncio.create_task(self._message_processing_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        # Setup system message handlers
        await self._setup_system_handlers()

    async def shutdown(self):
        """Shutdown the communication manager."""
        logger.info("Shutting down Sub-Agent Communication Manager")
        self._running = False

        # Cancel processing tasks
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Send shutdown messages to all handlers
        await self._broadcast_shutdown()

        # Cleanup pending responses
        self.pending_responses.clear()

    async def register_message_handler(
        self,
        agent_id: str,
        agent_type: str,
        message_types: List[MessageType],
        handler_function: Callable,
        priority_filter: Optional[List[MessagePriority]] = None,
        context_filter: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register a message handler for specific message types.

        Args:
            agent_id: ID of the agent
            agent_type: Type of the agent
            message_types: List of message types to handle
            handler_function: Function to handle messages
            priority_filter: Optional priority filter
            context_filter: Optional context filter

        Returns:
            Handler ID for the registered handler
        """

        handler_id = str(uuid.uuid4())

        handler = MessageHandler(
            handler_id=handler_id,
            agent_id=agent_id,
            agent_type=agent_type,
            message_types=set(message_types),
            handler_function=handler_function,
            priority_filter=set(priority_filter) if priority_filter else None,
            context_filter=context_filter
        )

        self.message_handlers[handler_id] = handler

        # Update subscriptions
        for message_type in message_types:
            if message_type not in self.broadcast_subscribers:
                self.broadcast_subscribers[message_type] = set()
            self.broadcast_subscribers[message_type].add(handler_id)

        # Update type subscriptions
        if agent_type not in self.type_subscribers:
            self.type_subscribers[agent_type] = set()
        self.type_subscribers[agent_type].add(handler_id)

        self.communication_stats["handlers_registered"] += 1

        logger.info(f"Registered message handler {handler_id} for agent {agent_id} ({agent_type})")
        return handler_id

    async def unregister_message_handler(self, handler_id: str):
        """Unregister a message handler."""
        if handler_id not in self.message_handlers:
            return

        handler = self.message_handlers[handler_id]

        # Remove from subscriptions
        for message_type in handler.message_types:
            if message_type in self.broadcast_subscribers:
                self.broadcast_subscribers[message_type].discard(handler_id)

        if handler.agent_type in self.type_subscribers:
            self.type_subscribers[handler.agent_type].discard(handler_id)

        # Remove handler
        del self.message_handlers[handler_id]

        logger.info(f"Unregistered message handler {handler_id}")

    async def send_message(
        self,
        message_type: MessageType,
        sender_id: str,
        sender_type: str,
        recipient_id: Optional[str],
        recipient_type: Optional[str],
        session_id: str,
        payload: Dict[str, Any],
        context_data: Optional[Dict[str, Any]] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        correlation_id: Optional[str] = None,
        reply_to: Optional[str] = None,
        requires_response: bool = False,
        timeout_seconds: Optional[int] = None
    ) -> str:
        """
        Send a message to one or more sub-agents.

        Args:
            message_type: Type of the message
            sender_id: ID of the sender
            sender_type: Type of the sender
            recipient_id: ID of the recipient (None for broadcast)
            recipient_type: Type of the recipient (None for broadcast to all types)
            session_id: Session ID
            payload: Message payload
            context_data: Additional context data
            priority: Message priority
            correlation_id: Optional correlation ID for related messages
            reply_to: Optional message ID this is replying to
            requires_response: Whether a response is required
            timeout_seconds: Optional timeout for response

        Returns:
            Message ID of the sent message
        """

        # Check queue size
        if self.message_queue.qsize() >= self.communication_config["max_queue_size"]:
            raise RuntimeError("Message queue is full")

        # Create message
        message = SubAgentMessage(
            message_id=str(uuid.uuid4()),
            message_type=message_type,
            priority=priority,
            sender_id=sender_id,
            sender_type=sender_type,
            recipient_id=recipient_id,
            recipient_type=recipient_type,
            session_id=session_id,
            payload=payload,
            context_data=context_data or {},
            correlation_id=correlation_id,
            reply_to=reply_to,
            timeout_seconds=timeout_seconds or self.communication_config["message_timeout_seconds"],
            requires_response=requires_response
        )

        # Set expiration
        message.expires_at = datetime.now() + timedelta(seconds=message.timeout_seconds)

        # Add to pending responses if response required
        if requires_response:
            self.pending_responses[message.message_id] = message

        # Queue message
        await self.message_queue.put((priority.value, message))

        self.communication_stats["messages_sent"] += 1

        logger.debug(f"Queued message {message.message_id} from {sender_id} to {recipient_id or 'broadcast'}")
        return message.message_id

    async def send_direct_message(
        self,
        sender_id: str,
        sender_type: str,
        recipient_id: str,
        recipient_type: str,
        session_id: str,
        payload: Dict[str, Any],
        **kwargs
    ) -> str:
        """Send a direct message to a specific agent."""
        return await self.send_message(
            message_type=MessageType.DIRECT_MESSAGE,
            sender_id=sender_id,
            sender_type=sender_type,
            recipient_id=recipient_id,
            recipient_type=recipient_type,
            session_id=session_id,
            payload=payload,
            **kwargs
        )

    async def send_task_assignment(
        self,
        sender_id: str,
        sender_type: str,
        recipient_id: str,
        recipient_type: str,
        session_id: str,
        task_description: str,
        task_data: Dict[str, Any],
        **kwargs
    ) -> str:
        """Send a task assignment message."""
        payload = {
            "task_description": task_description,
            "task_data": task_data,
            "assigned_at": datetime.now().isoformat()
        }

        return await self.send_message(
            message_type=MessageType.TASK_ASSIGNMENT,
            sender_id=sender_id,
            sender_type=sender_type,
            recipient_id=recipient_id,
            recipient_type=recipient_type,
            session_id=session_id,
            payload=payload,
            priority=MessagePriority.HIGH,
            requires_response=True,
            **kwargs
        )

    async def send_workflow_handoff(
        self,
        sender_id: str,
        sender_type: str,
        recipient_id: str,
        recipient_type: str,
        session_id: str,
        handoff_data: Dict[str, Any],
        next_stage: str,
        **kwargs
    ) -> str:
        """Send a workflow handoff message."""
        payload = {
            "handoff_data": handoff_data,
            "next_stage": next_stage,
            "handoff_at": datetime.now().isoformat(),
            "previous_stage": sender_type
        }

        return await self.send_message(
            message_type=MessageType.WORKFLOW_HANDOFF,
            sender_id=sender_id,
            sender_type=sender_type,
            recipient_id=recipient_id,
            recipient_type=recipient_type,
            session_id=session_id,
            payload=payload,
            priority=MessagePriority.HIGH,
            requires_response=True,
            **kwargs
        )

    async def broadcast_message(
        self,
        sender_id: str,
        sender_type: str,
        session_id: str,
        payload: Dict[str, Any],
        recipient_type_filter: Optional[str] = None,
        **kwargs
    ) -> str:
        """Broadcast a message to all agents or agents of a specific type."""
        return await self.send_message(
            message_type=MessageType.NOTIFICATION,
            sender_id=sender_id,
            sender_type=sender_type,
            recipient_id=None,
            recipient_type=recipient_type_filter,
            session_id=session_id,
            payload=payload,
            **kwargs
        )

    async def wait_for_response(
        self,
        message_id: str,
        timeout_seconds: Optional[int] = None
    ) -> Optional[SubAgentMessage]:
        """Wait for a response to a specific message."""
        timeout = timeout_seconds or self.communication_config["message_timeout_seconds"]
        start_time = datetime.now()

        while (datetime.now() - start_time).total_seconds() < timeout:
            if message_id in self.pending_responses:
                message = self.pending_responses[message_id]
                if message.response_received:
                    del self.pending_responses[message_id]
                    return message
            await asyncio.sleep(0.1)

        # Timeout
        if message_id in self.pending_responses:
            del self.pending_responses[message_id]
        return None

    async def _message_processing_loop(self):
        """Main message processing loop."""
        while self._running:
            try:
                # Get next message (with timeout)
                priority_value, message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )

                # Process message
                await self._process_message(message)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in message processing loop: {e}")

    async def _process_message(self, message: SubAgentMessage):
        """Process a single message."""
        try:
            # Check if message is expired
            if message.is_expired():
                logger.warning(f"Message {message.message_id} expired, discarding")
                self.communication_stats["messages_failed"] += 1
                return

            # Find appropriate handlers
            handlers = await self._find_handlers_for_message(message)

            if not handlers:
                logger.warning(f"No handlers found for message {message.message_id}")
                self.communication_stats["messages_failed"] += 1
                return

            # Dispatch to handlers
            tasks = []
            for handler in handlers:
                task = asyncio.create_task(handler.handle_message(message))
                tasks.append(task)

            # Wait for handlers to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process responses
            for i, result in enumerate(results):
                handler = handlers[i]

                if isinstance(result, Exception):
                    logger.error(f"Handler {handler.handler_id} failed: {result}")
                    continue

                if isinstance(result, SubAgentMessage):
                    # This is a response message
                    await self._handle_response_message(result)

            self.communication_stats["messages_received"] += 1

        except Exception as e:
            logger.error(f"Error processing message {message.message_id}: {e}")
            self.communication_stats["messages_failed"] += 1

    async def _find_handlers_for_message(self, message: SubAgentMessage) -> List[MessageHandler]:
        """Find handlers that can process the given message."""
        handlers = []

        for handler in self.message_handlers.values():
            if handler.can_handle_message(message):
                # Check recipient filtering
                if message.recipient_id and handler.agent_id != message.recipient_id:
                    continue

                if message.recipient_type and handler.agent_type != message.recipient_type:
                    continue

                handlers.append(handler)

        return handlers

    async def _handle_response_message(self, response: SubAgentMessage):
        """Handle a response message."""
        if response.reply_to and response.reply_to in self.pending_responses:
            original_message = self.pending_responses[response.reply_to]
            original_message.response_received = True

            # Could add more sophisticated response handling here
            logger.debug(f"Received response for message {response.reply_to}")

    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while self._running:
            try:
                await asyncio.sleep(self.communication_config["cleanup_interval"])
                await self._cleanup_expired_messages()
                await self._cleanup_inactive_handlers()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    async def _cleanup_expired_messages(self):
        """Cleanup expired messages and pending responses."""
        current_time = datetime.now()

        # Cleanup pending responses
        expired_responses = [
            message_id for message_id, message in self.pending_responses.items()
            if message.expires_at and current_time > message.expires_at
        ]

        for message_id in expired_responses:
            del self.pending_responses[message_id]
            logger.debug(f"Cleaned up expired pending response {message_id}")

    async def _cleanup_inactive_handlers(self):
        """Cleanup inactive handlers (optional - based on activity)."""
        # This could be implemented to remove handlers that haven't been active
        # for a certain period, though currently we keep all registered handlers
        pass

    async def _setup_system_handlers(self):
        """Setup system message handlers."""
        # Register ping/pong handler
        await self.register_message_handler(
            agent_id="system",
            agent_type="system",
            message_types=[MessageType.PING],
            handler_function=self._handle_ping_message
        )

    async def _handle_ping_message(self, message: SubAgentMessage) -> SubAgentMessage:
        """Handle ping messages with pong responses."""
        return SubAgentMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.PONG,
            priority=message.priority,
            sender_id="system",
            sender_type="system",
            recipient_id=message.sender_id,
            recipient_type=message.sender_type,
            session_id=message.session_id,
            payload={"ping_time": message.created_at.isoformat()},
            correlation_id=message.correlation_id,
            reply_to=message.message_id
        )

    async def _broadcast_shutdown(self):
        """Broadcast shutdown message to all handlers."""
        if not self.communication_config["enable_broadcast"]:
            return

        shutdown_message = SubAgentMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.SHUTDOWN,
            priority=MessagePriority.CRITICAL,
            sender_id="system",
            sender_type="system",
            recipient_id=None,
            recipient_type=None,
            session_id="system",
            payload={"shutdown_at": datetime.now().isoformat()}
        )

        await self.message_queue.put((MessagePriority.CRITICAL.value, shutdown_message))

    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics."""
        return {
            **self.communication_stats,
            "active_handlers": len(self.message_handlers),
            "pending_responses": len(self.pending_responses),
            "queue_size": self.message_queue.qsize(),
            "broadcast_subscribers": {
                msg_type.value: len(subscribers)
                for msg_type, subscribers in self.broadcast_subscribers.items()
            }
        }