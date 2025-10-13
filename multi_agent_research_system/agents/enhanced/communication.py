"""Agent Communication Protocols with Rich Messaging

This module provides sophisticated communication protocols for enhanced agents,
including message routing, delivery guarantees, priority handling, and
integration with Phase 2.3 rich message processing.

Key Features:
- Rich Message Processing with Phase 2.3 Integration
- Message Routing and Discovery
- Delivery Guarantees and Acknowledgments
- Priority-Based Message Handling
- Message Serialization and Compression
- Communication Monitoring and Analytics
"""

import asyncio
import json
import logging
import pickle
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Callable
from uuid import UUID, uuid4

from .base_agent import RichMessage, AgentPriority


class DeliveryGuarantee(Enum):
    """Message delivery guarantee levels."""
    AT_MOST_ONCE = "at_most_once"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"


class MessageStatus(Enum):
    """Message processing status."""
    PENDING = "pending"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"
    PROCESSED = "processed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class CommunicationProtocol(Enum):
    """Communication protocol types."""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    FIRE_AND_FORGET = "fire_and_forget"
    REQUEST_RESPONSE = "request_response"
    PUB_SUB = "pub_sub"


@dataclass
class MessageEnvelope:
    """Envelope for rich messages with metadata and tracking."""
    message: RichMessage
    envelope_id: str = field(default_factory=lambda: str(uuid4()))
    protocol: CommunicationProtocol = CommunicationProtocol.ASYNCHRONOUS
    delivery_guarantee: DeliveryGuarantee = DeliveryGuarantee.AT_LEAST_ONCE
    created_at: datetime = field(default_factory=datetime.now)
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 30
    acknowledgment_required: bool = True
    compressed: bool = False
    encrypted: bool = False
    routing_key: Optional[str] = None
    correlation_chain: List[str] = field(default_factory=list)
    delivery_attempts: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MessageAck:
    """Message acknowledgment."""
    envelope_id: str
    message_id: str
    status: MessageStatus
    agent_id: str
    timestamp: datetime
    processing_time_ms: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CommunicationMetrics:
    """Communication metrics and analytics."""
    total_messages_sent: int = 0
    total_messages_delivered: int = 0
    total_messages_processed: int = 0
    total_messages_failed: int = 0
    total_messages_timeout: int = 0
    average_delivery_time_ms: float = 0.0
    average_processing_time_ms: float = 0.0
    messages_by_priority: Dict[str, int] = field(default_factory=dict)
    messages_by_status: Dict[str, int] = field(default_factory=dict)
    messages_by_protocol: Dict[str, int] = field(default_factory=dict)
    active_conversations: int = 0
    failed_deliveries_by_agent: Dict[str, int] = field(default_factory=dict)
    throughput_per_minute: float = 0.0


class MessageQueue:
    """Priority-based message queue with persistence."""

    def __init__(self, max_size: int = 10000, persistence_dir: Optional[Path] = None):
        self.max_size = max_size
        self.persistence_dir = persistence_dir
        self._queue = asyncio.PriorityQueue(maxsize=max_size)
        self._pending_messages: Dict[str, MessageEnvelope] = {}
        self._lock = asyncio.Lock()
        self.logger = logging.getLogger("message_queue")

    async def put(self, envelope: MessageEnvelope) -> bool:
        """Add message to queue with priority."""
        async with self._lock:
            if self._queue.full():
                self.logger.warning("Message queue is full, dropping oldest message")
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass

            # Priority value (lower number = higher priority)
            priority_value = 5 - envelope.message.priority.value

            try:
                await self._queue.put((priority_value, envelope.envelope_id, envelope))
                self._pending_messages[envelope.envelope_id] = envelope

                # Persist if enabled
                if self.persistence_dir:
                    await self._persist_message(envelope)

                return True
            except asyncio.QueueFull:
                self.logger.error("Failed to add message to queue")
                return False

    async def get(self, timeout: Optional[float] = None) -> Optional[MessageEnvelope]:
        """Get message from queue."""
        try:
            _, envelope_id, envelope = await asyncio.wait_for(
                self._queue.get(), timeout=timeout
            )

            async with self._lock:
                if envelope_id in self._pending_messages:
                    del self._pending_messages[envelope_id]

            return envelope
        except asyncio.TimeoutError:
            return None

    async def get_by_id(self, envelope_id: str) -> Optional[MessageEnvelope]:
        """Get specific message by ID."""
        async with self._lock:
            return self._pending_messages.get(envelope_id)

    async def size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()

    async def pending_count(self) -> int:
        """Get count of pending messages."""
        async with self._lock:
            return len(self._pending_messages)

    async def _persist_message(self, envelope: MessageEnvelope) -> None:
        """Persist message to disk."""
        if not self.persistence_dir:
            return

        try:
            self.persistence_dir.mkdir(parents=True, exist_ok=True)
            message_file = self.persistence_dir / f"{envelope.envelope_id}.msg"

            # Serialize envelope
            data = {
                "envelope": envelope,
                "serialized_at": datetime.now().isoformat()
            }

            with open(message_file, 'wb') as f:
                compressed_data = zlib.compress(pickle.dumps(data))
                f.write(compressed_data)

        except Exception as e:
            self.logger.error(f"Failed to persist message {envelope.envelope_id}: {e}")


class RichMessageProcessor:
    """Advanced message processor with Phase 2.3 integration."""

    def __init__(self):
        self.logger = logging.getLogger("rich_message_processor")
        self.processors: Dict[str, Callable] = {}
        self.filters: List[Callable] = []
        self.transformers: List[Callable] = []

    def register_processor(self, message_type: str, processor: Callable) -> None:
        """Register a message processor."""
        self.processors[message_type] = processor
        self.logger.debug(f"Registered processor for message type: {message_type}")

    def add_filter(self, filter_func: Callable) -> None:
        """Add a message filter."""
        self.filters.append(filter_func)

    def add_transformer(self, transformer_func: Callable) -> None:
        """Add a message transformer."""
        self.transformers.append(transformer_func)

    async def process_message(self, envelope: MessageEnvelope) -> MessageEnvelope:
        """Process a message envelope."""
        message = envelope.message

        try:
            # Apply filters
            for filter_func in self.filters:
                if not await filter_func(message):
                    self.logger.debug(f"Message filtered out: {message.message_type}")
                    return envelope

            # Apply transformers
            for transformer in self.transformers:
                message = await transformer(message)

            # Apply type-specific processor
            if message.message_type in self.processors:
                processor = self.processors[message.message_type]
                await processor(message)

            return envelope

        except Exception as e:
            self.logger.error(f"Error processing message {message.message_type}: {e}")
            raise

    async def compress_message(self, envelope: MessageEnvelope) -> MessageEnvelope:
        """Compress message payload if beneficial."""
        if len(str(envelope.message.payload)) < 1000:  # Don't compress small messages
            return envelope

        try:
            payload_str = json.dumps(envelope.message.payload)
            compressed_payload = zlib.compress(payload_str.encode())

            if len(compressed_payload) < len(payload_str.encode()):
                envelope.message.payload = {
                    "_compressed": True,
                    "_original_type": type(envelope.message.payload).__name__,
                    "_data": compressed_payload.hex()
                }
                envelope.compressed = True
                self.logger.debug(f"Compressed message {envelope.envelope_id}")

        except Exception as e:
            self.logger.error(f"Failed to compress message: {e}")

        return envelope

    async def decompress_message(self, envelope: MessageEnvelope) -> MessageEnvelope:
        """Decompress message payload if compressed."""
        if (envelope.compressed and
            isinstance(envelope.message.payload, dict) and
            envelope.message.payload.get("_compressed")):

            try:
                compressed_data = bytes.fromhex(envelope.message.payload["_data"])
                decompressed = zlib.decompress(compressed_data)
                envelope.message.payload = json.loads(decompressed.decode())
                envelope.compressed = False
                self.logger.debug(f"Decompressed message {envelope.envelope_id}")

            except Exception as e:
                self.logger.error(f"Failed to decompress message: {e}")

        return envelope


class AgentCommunicationManager:
    """Comprehensive communication manager for enhanced agents."""

    def __init__(self, agent_id: str, persistence_dir: Optional[Path] = None):
        self.agent_id = agent_id
        self.logger = logging.getLogger(f"comm_manager.{agent_id}")
        self.persistence_dir = persistence_dir or Path("data/agent_communication") / agent_id
        self.persistence_dir.mkdir(parents=True, exist_ok=True)

        # Communication components
        self.message_queue = MessageQueue(persistence_dir=self.persistence_dir / "queue")
        self.message_processor = RichMessageProcessor()
        self.active_conversations: Dict[str, Dict[str, Any]] = {}

        # Message tracking
        self.sent_messages: Dict[str, MessageEnvelope] = {}
        self.received_messages: Dict[str, MessageEnvelope] = {}
        self.pending_acks: Dict[str, MessageAck] = {}

        # Communication metrics
        self.metrics = CommunicationMetrics()

        # Event handlers
        self.message_handlers: Dict[str, Callable] = {}
        self.delivery_handlers: List[Callable] = []
        self.error_handlers: List[Callable] = []

        # Background tasks
        self.processing_task: Optional[asyncio.Task] = None
        self.ack_monitor_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None

        # State
        self.running = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the communication manager."""
        if self.running:
            return

        self.running = True
        self.logger.info(f"Starting communication manager for {self.agent_id}")

        # Load persisted state
        await self._load_persisted_state()

        # Start background tasks
        self.processing_task = asyncio.create_task(self._processing_loop())
        self.ack_monitor_task = asyncio.create_task(self._ack_monitor_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())

        self.logger.info("Communication manager started")

    async def stop(self) -> None:
        """Stop the communication manager."""
        if not self.running:
            return

        self.running = False
        self.logger.info(f"Stopping communication manager for {self.agent_id}")

        # Cancel background tasks
        tasks = [self.processing_task, self.ack_monitor_task, self.cleanup_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Save state
        await self._save_persisted_state()

        self.logger.info("Communication manager stopped")

    async def send_message(self, message: RichMessage,
                         protocol: CommunicationProtocol = CommunicationProtocol.ASYNCHRONOUS,
                         delivery_guarantee: DeliveryGuarantee = DeliveryGuarantee.AT_LEAST_ONCE,
                         timeout_seconds: int = 30) -> str:
        """Send a message with rich protocol support."""
        try:
            # Create envelope
            envelope = MessageEnvelope(
                message=message,
                protocol=protocol,
                delivery_guarantee=delivery_guarantee,
                timeout_seconds=timeout_seconds,
                expires_at=datetime.now() + timedelta(seconds=timeout_seconds)
            )

            # Process message
            envelope = await self.message_processor.process_message(envelope)
            envelope = await self.message_processor.compress_message(envelope)

            # Set sent timestamp
            envelope.sent_at = datetime.now()

            # Track message
            self.sent_messages[envelope.envelope_id] = envelope
            self.metrics.total_messages_sent += 1

            # Update priority metrics
            priority_name = message.priority.name
            self.metrics.messages_by_priority[priority_name] = \
                self.metrics.messages_by_priority.get(priority_name, 0) + 1

            # Update protocol metrics
            protocol_name = protocol.value
            self.metrics.messages_by_protocol[protocol_name] = \
                self.metrics.messages_by_protocol.get(protocol_name, 0) + 1

            # Route message based on protocol
            if protocol == CommunicationProtocol.SYNCHRONOUS:
                return await self._send_synchronous(envelope)
            elif protocol == CommunicationProtocol.REQUEST_RESPONSE:
                return await self._send_request_response(envelope)
            else:
                return await self._send_async(envelope)

        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            self.metrics.total_messages_failed += 1
            raise

    async def _send_async(self, envelope: MessageEnvelope) -> str:
        """Send message asynchronously."""
        # This would integrate with the agent registry or message router
        # For now, simulate successful send
        envelope.delivered_at = datetime.now()
        self.metrics.total_messages_delivered += 1
        return envelope.envelope_id

    async def _send_synchronous(self, envelope: MessageEnvelope) -> str:
        """Send message synchronously and wait for response."""
        # Implement synchronous communication
        return await self._send_async(envelope)

    async def _send_request_response(self, envelope: MessageEnvelope) -> str:
        """Send request-response message."""
        # Start conversation tracking
        self.active_conversations[envelope.message.correlation_id] = {
            "started_at": datetime.now(),
            "message_id": envelope.envelope_id,
            "status": "awaiting_response"
        }
        self.metrics.active_conversations += 1

        return await self._send_async(envelope)

    async def receive_message(self, envelope: MessageEnvelope) -> None:
        """Receive and process a message."""
        try:
            # Decompress if needed
            envelope = await self.message_processor.decompress_message(envelope)

            # Track received message
            self.received_messages[envelope.envelope_id] = envelope
            envelope.delivered_at = datetime.now()
            self.metrics.total_messages_delivered += 1

            # Process message
            await self._process_received_message(envelope)

            # Send acknowledgment if required
            if envelope.acknowledgment_required:
                await self._send_acknowledgment(envelope, MessageStatus.PROCESSED)

        except Exception as e:
            self.logger.error(f"Failed to receive message: {e}")
            self.metrics.total_messages_failed += 1

            # Send error acknowledgment
            if envelope.acknowledgment_required:
                await self._send_acknowledgment(envelope, MessageStatus.FAILED, str(e))

    async def _process_received_message(self, envelope: MessageEnvelope) -> None:
        """Process a received message."""
        message = envelope.message
        start_time = datetime.now()

        try:
            # Add to processing queue
            await self.message_queue.put(envelope)

            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_processing_time_metrics(processing_time)

        except Exception as e:
            self.logger.error(f"Failed to queue message for processing: {e}")
            raise

    async def _processing_loop(self) -> None:
        """Main message processing loop."""
        while self.running:
            try:
                # Get message from queue
                envelope = await self.message_queue.get(timeout=1.0)
                if envelope is None:
                    continue

                # Process message
                await self._handle_message_processing(envelope)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")

    async def _handle_message_processing(self, envelope: MessageEnvelope) -> None:
        """Handle individual message processing."""
        message = envelope.message
        start_time = datetime.now()

        try:
            # Check for message expiration
            if envelope.expires_at and datetime.now() > envelope.expires_at:
                self.logger.warning(f"Message {envelope.envelope_id} expired")
                await self._send_acknowledgment(envelope, MessageStatus.TIMEOUT)
                return

            # Find and execute handler
            handler = self.message_handlers.get(message.message_type)
            if handler:
                self.logger.info(f"Processing message type: {message.message_type}")
                await handler(message)
            else:
                self.logger.warning(f"No handler for message type: {message.message_type}")

            # Update status
            envelope.processed_at = datetime.now()
            self.metrics.total_messages_processed += 1

            # Update status metrics
            status_name = MessageStatus.PROCESSED.value
            self.metrics.messages_by_status[status_name] = \
                self.metrics.messages_by_status.get(status_name, 0) + 1

            # Handle request-response
            if (envelope.protocol == CommunicationProtocol.REQUEST_RESPONSE and
                message.correlation_id in self.active_conversations):

                conversation = self.active_conversations[message.correlation_id]
                conversation["status"] = "completed"
                conversation["completed_at"] = datetime.now()
                self.metrics.active_conversations -= 1

        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            self.metrics.total_messages_failed += 1

            # Call error handlers
            for error_handler in self.error_handlers:
                try:
                    await error_handler(envelope, e)
                except Exception as handler_error:
                    self.logger.error(f"Error in error handler: {handler_error}")

        finally:
            # Update processing time metrics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_processing_time_metrics(processing_time)

    async def _send_acknowledgment(self, envelope: MessageEnvelope,
                                 status: MessageStatus,
                                 error_message: Optional[str] = None) -> None:
        """Send message acknowledgment."""
        ack = MessageAck(
            envelope_id=envelope.envelope_id,
            message_id=envelope.message.correlation_id,
            status=status,
            agent_id=self.agent_id,
            timestamp=datetime.now(),
            processing_time_ms=0.0,
            error_message=error_message
        )

        # Store acknowledgment
        self.pending_acks[envelope.envelope_id] = ack

        # This would send the acknowledgment via the message router
        self.logger.debug(f"Sent acknowledgment for {envelope.envelope_id}: {status.value}")

    async def _ack_monitor_loop(self) -> None:
        """Monitor pending acknowledgments."""
        while self.running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                # Check for expired acknowledgments
                current_time = datetime.now()
                expired_acks = []

                for envelope_id, ack in self.pending_acks.items():
                    if (current_time - ack.timestamp).total_seconds() > 300:  # 5 minutes
                        expired_acks.append(envelope_id)

                # Clean up expired acks
                for envelope_id in expired_acks:
                    del self.pending_acks[envelope_id]
                    self.logger.debug(f"Cleaned up expired acknowledgment: {envelope_id}")

            except Exception as e:
                self.logger.error(f"Error in ack monitor loop: {e}")

    async def _cleanup_loop(self) -> None:
        """Cleanup old messages and data."""
        while self.running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes

                cutoff_time = datetime.now() - timedelta(hours=24)

                # Cleanup old sent messages
                old_sent = [
                    envelope_id for envelope_id, envelope in self.sent_messages.items()
                    if envelope.sent_at and envelope.sent_at < cutoff_time
                ]
                for envelope_id in old_sent:
                    del self.sent_messages[envelope_id]

                # Cleanup old received messages
                old_received = [
                    envelope_id for envelope_id, envelope in self.received_messages.items()
                    if envelope.delivered_at and envelope.delivered_at < cutoff_time
                ]
                for envelope_id in old_received:
                    del self.received_messages[envelope_id]

                # Cleanup old conversations
                old_conversations = [
                    corr_id for corr_id, conv in self.active_conversations.items()
                    if conv.get("completed_at") and conv["completed_at"] < cutoff_time
                ]
                for corr_id in old_conversations:
                    del self.active_conversations[corr_id]

                self.logger.debug(f"Cleaned up {len(old_sent)} sent, {len(old_received)} received, {len(old_conversations)} conversations")

            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")

    def _update_processing_time_metrics(self, processing_time_ms: float) -> None:
        """Update processing time metrics."""
        total_processed = self.metrics.total_messages_processed
        if total_processed == 1:
            self.metrics.average_processing_time_ms = processing_time_ms
        else:
            # Calculate running average
            current_avg = self.metrics.average_processing_time_ms
            self.metrics.average_processing_time_ms = (
                (current_avg * (total_processed - 1) + processing_time_ms) / total_processed
            )

    def register_message_handler(self, message_type: str, handler: Callable) -> None:
        """Register a message handler."""
        self.message_handlers[message_type] = handler
        self.message_processor.register_processor(message_type, handler)

    def add_delivery_handler(self, handler: Callable) -> None:
        """Add a delivery event handler."""
        self.delivery_handlers.append(handler)

    def add_error_handler(self, handler: Callable) -> None:
        """Add an error handler."""
        self.error_handlers.append(handler)

    async def _load_persisted_state(self) -> None:
        """Load persisted state from disk."""
        try:
            # Load metrics
            metrics_file = self.persistence_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    # Restore metrics
                    for key, value in data.items():
                        if hasattr(self.metrics, key):
                            setattr(self.metrics, key, value)

            self.logger.info("Loaded persisted communication state")
        except Exception as e:
            self.logger.error(f"Failed to load persisted state: {e}")

    async def _save_persisted_state(self) -> None:
        """Save current state to disk."""
        try:
            # Save metrics
            metrics_file = self.persistence_dir / "metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics.__dict__, f, indent=2, default=str)

            self.logger.info("Saved communication state")
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")

    def get_communication_statistics(self) -> Dict[str, Any]:
        """Get comprehensive communication statistics."""
        return {
            "agent_id": self.agent_id,
            "metrics": self.metrics.__dict__,
            "queue_size": self.message_queue._queue.qsize(),
            "pending_messages": len(self.message_queue._pending_messages),
            "active_conversations": len(self.active_conversations),
            "sent_messages_count": len(self.sent_messages),
            "received_messages_count": len(self.received_messages),
            "pending_acks_count": len(self.pending_acks),
            "registered_handlers": list(self.message_handlers.keys()),
            "running": self.running
        }


# Global communication manager registry
_communication_managers: Dict[str, AgentCommunicationManager] = {}


def get_communication_manager(agent_id: str,
                            persistence_dir: Optional[Path] = None) -> AgentCommunicationManager:
    """Get or create communication manager for an agent."""
    if agent_id not in _communication_managers:
        _communication_managers[agent_id] = AgentCommunicationManager(agent_id, persistence_dir)
    return _communication_managers[agent_id]


async def shutdown_all_communication_managers() -> None:
    """Shutdown all communication managers."""
    for manager in _communication_managers.values():
        await manager.stop()
    _communication_managers.clear()