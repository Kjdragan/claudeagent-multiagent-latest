"""
Enhanced Message Type System - Comprehensive Message Classification and Metadata

This module provides an enhanced message type system with comprehensive classification,
metadata support, and context awareness for sophisticated message processing.

Key Features:
- Comprehensive message type enumeration with specific categories
- Message priority levels for processing order
- Context classification for routing and filtering
- Enhanced message structure with rich metadata
- Message quality and relevance scoring
- Message lifecycle tracking and audit trail
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import uuid
import json


class EnhancedMessageType(Enum):
    """Enhanced message types with comprehensive classification."""

    # Core message types
    TEXT = "text"
    MARKDOWN = "markdown"
    CODE = "code"
    JSON = "json"
    XML = "xml"
    YAML = "yaml"

    # Agent communication types
    AGENT_MESSAGE = "agent_message"
    AGENT_REQUEST = "agent_request"
    AGENT_RESPONSE = "agent_response"
    AGENT_HANDOFF = "agent_handoff"
    AGENT_STATUS = "agent_status"

    # Tool interaction types
    TOOL_USE = "tool_use"
    TOOL_REQUEST = "tool_request"
    TOOL_RESPONSE = "tool_response"
    TOOL_RESULT = "tool_result"
    TOOL_ERROR = "tool_error"

    # Research and analysis types
    RESEARCH_QUERY = "research_query"
    RESEARCH_RESULT = "research_result"
    ANALYSIS_REQUEST = "analysis_request"
    ANALYSIS_RESULT = "analysis_result"
    CONTENT_SUMMARY = "content_summary"
    CONTENT_EXTRACTION = "content_extraction"

    # Quality and assessment types
    QUALITY_ASSESSMENT = "quality_assessment"
    QUALITY_SCORE = "quality_score"
    QUALITY_FEEDBACK = "quality_feedback"
    VALIDATION_RESULT = "validation_result"
    ERROR_ANALYSIS = "error_analysis"

    # Workflow and orchestration types
    WORKFLOW_STAGE = "workflow_stage"
    WORKFLOW_STATUS = "workflow_status"
    WORKFLOW_COMMAND = "workflow_command"
    PROGRESS_UPDATE = "progress_update"
    STAGE_TRANSITION = "stage_transition"

    # System and infrastructure types
    SYSTEM_NOTIFICATION = "system_notification"
    SYSTEM_ERROR = "system_error"
    SYSTEM_WARNING = "system_warning"
    SYSTEM_INFO = "system_info"
    PERFORMANCE_METRIC = "performance_metric"

    # Data and content types
    DATA_REQUEST = "data_request"
    DATA_RESPONSE = "data_response"
    FILE_OPERATION = "file_operation"
    CACHE_OPERATION = "cache_operation"
    DATABASE_OPERATION = "database_operation"

    # User interaction types
    USER_INPUT = "user_input"
    USER_FEEDBACK = "user_feedback"
    USER_NOTIFICATION = "user_notification"
    USER_PROMPT = "user_prompt"

    # Debug and development types
    DEBUG_MESSAGE = "debug_message"
    LOG_MESSAGE = "log_message"
    TRACE_MESSAGE = "trace_message"
    PROFILING_DATA = "profiling_data"

    # Specialized research types
    GAP_RESEARCH = "gap_research"
    GAP_ANALYSIS = "gap_analysis"
    RECOMMENDATION = "recommendation"
    INSIGHT = "insight"
    FINDING = "finding"

    # Session and state types
    SESSION_MESSAGE = "session_message"
    STATE_UPDATE = "state_update"
    CHECKPOINT = "checkpoint"
    RECOVERY = "recovery"


class MessagePriority(Enum):
    """Message priority levels for processing order."""

    CRITICAL = 1    # System-critical messages, errors, security
    HIGH = 2        # Important user messages, quality gates
    NORMAL = 3      # Standard processing messages
    LOW = 4         # Background tasks, metrics, debug
    BULK = 5        # Bulk data, logs, non-urgent


class MessageContext(Enum):
    """Message context classification for routing and filtering."""

    # Workflow contexts
    RESEARCH = "research"
    ANALYSIS = "analysis"
    REPORTING = "reporting"
    EDITORIAL = "editorial"
    QUALITY = "quality"
    ENHANCEMENT = "enhancement"

    # Agent contexts
    COORDINATION = "coordination"
    HANDOFF = "handoff"
    COLLABORATION = "collaboration"
    DELEGATION = "delegation"

    # System contexts
    MONITORING = "monitoring"
    MAINTENANCE = "maintenance"
    DEBUGGING = "debugging"
    TESTING = "testing"

    # User contexts
    INTERACTION = "interaction"
    FEEDBACK = "feedback"
    NOTIFICATION = "notification"
    SUPPORT = "support"


class MessageLifecycle(Enum):
    """Message lifecycle stages for tracking and audit."""

    CREATED = "created"
    QUEUED = "queued"
    PROCESSING = "processing"
    PROCESSED = "processed"
    ROUTED = "routed"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"
    ARCHIVED = "archived"
    FAILED = "failed"


@dataclass
class MessageMetadata:
    """Comprehensive message metadata for processing and analysis."""

    # Basic metadata
    source_agent: str = ""
    target_agent: str = ""
    session_id: str = ""
    workflow_stage: str = ""
    correlation_id: str = ""

    # Content metadata
    content_type: str = ""
    content_length: int = 0
    content_hash: str = ""
    language: str = "en"
    encoding: str = "utf-8"

    # Processing metadata
    processing_time: float = 0.0
    processing_attempts: int = 0
    last_processor: str = ""
    processing_history: List[Dict[str, Any]] = field(default_factory=list)

    # Quality and relevance
    quality_score: Optional[float] = None
    relevance_score: Optional[float] = None
    confidence_score: Optional[float] = None
    accuracy_score: Optional[float] = None

    # Routing and filtering
    routing_tags: List[str] = field(default_factory=list)
    filtering_criteria: Dict[str, Any] = field(default_factory=dict)
    delivery_requirements: Dict[str, Any] = field(default_factory=dict)

    # Performance and optimization
    cache_hit: bool = False
    compression_ratio: Optional[float] = None
    optimization_applied: List[str] = field(default_factory=list)

    # Security and compliance
    sensitivity_level: str = "public"
    access_controls: List[str] = field(default_factory=list)
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)

    # System metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    ttl_seconds: Optional[int] = None


@dataclass
class RichMessage:
    """Enhanced message structure with comprehensive metadata and processing capabilities."""

    # Core message fields
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: EnhancedMessageType = EnhancedMessageType.TEXT
    content: str = ""
    priority: MessagePriority = MessagePriority.NORMAL
    context: MessageContext = MessageContext.RESEARCH

    # Enhanced metadata
    metadata: MessageMetadata = field(default_factory=MessageMetadata)

    # Processing fields
    formatting: Dict[str, Any] = field(default_factory=dict)
    display_options: Dict[str, Any] = field(default_factory=dict)
    routing_info: Dict[str, Any] = field(default_factory=dict)

    # Lifecycle tracking
    lifecycle: MessageLifecycle = MessageLifecycle.CREATED
    lifecycle_history: List[Dict[str, Any]] = field(default_factory=list)

    # Performance tracking
    timestamps: Dict[str, datetime] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

    # Relationships
    parent_message_id: Optional[str] = None
    child_message_ids: List[str] = field(default_factory=list)
    related_message_ids: List[str] = field(default_factory=list)

    # Content analysis
    content_analysis: Dict[str, Any] = field(default_factory=dict)
    extracted_entities: List[Dict[str, Any]] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize message with default values and validation."""
        # Set initial timestamps
        current_time = datetime.now()
        self.timestamps["created"] = current_time
        self.timestamps["updated"] = current_time

        # Initialize metadata with basic information
        if not self.metadata.content_length:
            self.metadata.content_length = len(self.content)

        # Update metadata with message type info
        self.metadata.content_type = self.message_type.value

        # Add initial lifecycle entry
        self._add_lifecycle_entry(MessageLifecycle.CREATED, "Message created")

    def _add_lifecycle_entry(self, stage: MessageLifecycle, details: str = ""):
        """Add entry to lifecycle history."""
        entry = {
            "stage": stage.value,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        self.lifecycle_history.append(entry)
        self.lifecycle = stage
        self.timestamps["updated"] = datetime.now()

    def update_content(self, new_content: str, reason: str = ""):
        """Update message content with audit trail."""
        old_content = self.content
        self.content = new_content
        self.metadata.content_length = len(new_content)
        self.metadata.updated_at = datetime.now()
        self.timestamps["updated"] = datetime.now()

        # Add to audit trail
        self.metadata.audit_trail.append({
            "action": "content_update",
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "old_length": len(old_content),
            "new_length": len(new_content)
        })

    def add_processing_step(self, processor: str, duration: float = 0.0, result: str = "success"):
        """Add processing step to history."""
        step = {
            "processor": processor,
            "duration": duration,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        self.metadata.processing_history.append(step)
        self.metadata.last_processor = processor
        self.metadata.processing_attempts += 1

        if duration > 0:
            self.performance_metrics[f"{processor}_duration"] = duration

    def add_routing_tag(self, tag: str):
        """Add routing tag for filtering and routing."""
        if tag not in self.metadata.routing_tags:
            self.metadata.routing_tags.append(tag)

    def set_quality_scores(self, quality: float = None, relevance: float = None,
                          confidence: float = None, accuracy: float = None):
        """Set quality scores with validation."""
        if quality is not None:
            self.metadata.quality_score = max(0.0, min(1.0, quality))
        if relevance is not None:
            self.metadata.relevance_score = max(0.0, min(1.0, relevance))
        if confidence is not None:
            self.metadata.confidence_score = max(0.0, min(1.0, confidence))
        if accuracy is not None:
            self.metadata.accuracy_score = max(0.0, min(1.0, accuracy))

    def add_relationship(self, related_id: str, relationship_type: str = "related"):
        """Add relationship to another message."""
        if relationship_type == "parent":
            self.parent_message_id = related_id
        elif relationship_type == "child":
            if related_id not in self.child_message_ids:
                self.child_message_ids.append(related_id)
        else:  # related
            if related_id not in self.related_message_ids:
                self.related_message_ids.append(related_id)

    def mark_processed(self, processor: str, duration: float = 0.0):
        """Mark message as processed with performance metrics."""
        self._add_lifecycle_entry(MessageLifecycle.PROCESSED, f"Processed by {processor}")
        self.add_processing_step(processor, duration, "success")

        if duration > 0:
            self.metadata.processing_time += duration
            self.performance_metrics["total_processing_time"] = self.metadata.processing_time

    def mark_delivered(self, target: str = ""):
        """Mark message as delivered."""
        self._add_lifecycle_entry(MessageLifecycle.DELIVERED, f"Delivered to {target}")
        if target:
            self.metadata.target_agent = target

    def mark_failed(self, error: str, processor: str = ""):
        """Mark message as failed with error information."""
        self._add_lifecycle_entry(MessageLifecycle.FAILED, f"Failed: {error}")
        if processor:
            self.add_processing_step(processor, 0.0, f"failed: {error}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "id": self.id,
            "message_type": self.message_type.value,
            "content": self.content,
            "priority": self.priority.value,
            "context": self.context.value,
            "metadata": self._serialize_metadata(),
            "formatting": self.formatting,
            "display_options": self.display_options,
            "routing_info": self.routing_info,
            "lifecycle": self.lifecycle.value,
            "lifecycle_history": self.lifecycle_history,
            "timestamps": {k: v.isoformat() for k, v in self.timestamps.items()},
            "performance_metrics": self.performance_metrics,
            "parent_message_id": self.parent_message_id,
            "child_message_ids": self.child_message_ids,
            "related_message_ids": self.related_message_ids,
            "content_analysis": self.content_analysis,
            "extracted_entities": self.extracted_entities,
            "keywords": self.keywords
        }

    def _serialize_metadata(self) -> Dict[str, Any]:
        """Serialize metadata with datetime handling."""
        metadata_dict = {}
        for key, value in self.metadata.__dict__.items():
            if isinstance(value, datetime):
                metadata_dict[key] = value.isoformat()
            elif isinstance(value, list):
                metadata_dict[key] = value
            else:
                metadata_dict[key] = value
        return metadata_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RichMessage':
        """Create message from dictionary."""
        # Create basic message
        message = cls(
            id=data.get("id", str(uuid.uuid4())),
            message_type=EnhancedMessageType(data.get("message_type", "text")),
            content=data.get("content", ""),
            priority=MessagePriority(data.get("priority", 3)),
            context=MessageContext(data.get("context", "research"))
        )

        # Restore complex fields
        if "metadata" in data:
            message.metadata = MessageMetadata()
            for key, value in data["metadata"].items():
                if key in ["created_at", "updated_at", "expires_at"] and value:
                    setattr(message.metadata, key, datetime.fromisoformat(value))
                else:
                    setattr(message.metadata, key, value)

        # Restore other fields
        for field in ["formatting", "display_options", "routing_info", "content_analysis",
                      "extracted_entities", "keywords", "performance_metrics"]:
            if field in data:
                setattr(message, field, data[field])

        # Restore timestamps
        if "timestamps" in data:
            for key, value in data["timestamps"].items():
                message.timestamps[key] = datetime.fromisoformat(value)

        # Restore lifecycle
        if "lifecycle" in data:
            message.lifecycle = MessageLifecycle(data["lifecycle"])
        if "lifecycle_history" in data:
            message.lifecycle_history = data["lifecycle_history"]

        # Restore relationships
        message.parent_message_id = data.get("parent_message_id")
        message.child_message_ids = data.get("child_message_ids", [])
        message.related_message_ids = data.get("related_message_ids", [])

        return message

    def copy(self) -> 'RichMessage':
        """Create a deep copy of the message."""
        return RichMessage.from_dict(self.to_dict())

    def __str__(self) -> str:
        """String representation of message."""
        return f"RichMessage(id={self.id}, type={self.message_type.value}, priority={self.priority.value})"

    def __repr__(self) -> str:
        """Detailed string representation of message."""
        return (f"RichMessage(id={self.id}, type={self.message_type.value}, "
                f"priority={self.priority.value}, context={self.context.value}, "
                f"content_length={len(self.content)})")


class MessageBuilder:
    """Builder pattern for creating rich messages with fluent interface."""

    def __init__(self):
        self.message = RichMessage()

    def with_content(self, content: str) -> 'MessageBuilder':
        """Set message content."""
        self.message.content = content
        return self

    def with_type(self, message_type: EnhancedMessageType) -> 'MessageBuilder':
        """Set message type."""
        self.message.message_type = message_type
        return self

    def with_priority(self, priority: MessagePriority) -> 'MessageBuilder':
        """Set message priority."""
        self.message.priority = priority
        return self

    def with_context(self, context: MessageContext) -> 'MessageBuilder':
        """Set message context."""
        self.message.context = context
        return self

    def with_session(self, session_id: str) -> 'MessageBuilder':
        """Set session ID."""
        self.message.metadata.session_id = session_id
        return self

    def with_agent(self, source_agent: str, target_agent: str = "") -> 'MessageBuilder':
        """Set source and target agents."""
        self.message.metadata.source_agent = source_agent
        self.message.metadata.target_agent = target_agent
        return self

    def with_workflow_stage(self, stage: str) -> 'MessageBuilder':
        """Set workflow stage."""
        self.message.metadata.workflow_stage = stage
        return self

    def with_quality_scores(self, quality: float = None, relevance: float = None,
                           confidence: float = None, accuracy: float = None) -> 'MessageBuilder':
        """Set quality scores."""
        self.message.set_quality_scores(quality, relevance, confidence, accuracy)
        return self

    def with_routing_tags(self, *tags: str) -> 'MessageBuilder':
        """Add routing tags."""
        for tag in tags:
            self.message.add_routing_tag(tag)
        return self

    def with_formatting(self, **formatting_options) -> 'MessageBuilder':
        """Set formatting options."""
        self.message.formatting.update(formatting_options)
        return self

    def with_display_options(self, **display_options) -> 'MessageBuilder':
        """Set display options."""
        self.message.display_options.update(display_options)
        return self

    def with_correlation_id(self, correlation_id: str) -> 'MessageBuilder':
        """Set correlation ID."""
        self.message.metadata.correlation_id = correlation_id
        return self

    def with_sensitivity(self, level: str = "public") -> 'MessageBuilder':
        """Set sensitivity level."""
        self.message.metadata.sensitivity_level = level
        return self

    def with_relationship(self, related_id: str, relationship_type: str = "related") -> 'MessageBuilder':
        """Add relationship to another message."""
        self.message.add_relationship(related_id, relationship_type)
        return self

    def build(self) -> RichMessage:
        """Build the final message."""
        return self.message.copy()


# Convenience functions for common message types
def create_text_message(content: str, **kwargs) -> RichMessage:
    """Create a simple text message."""
    return MessageBuilder().with_content(content).with_type(EnhancedMessageType.TEXT).build(**kwargs)


def create_error_message(content: str, error_type: str = "general", **kwargs) -> RichMessage:
    """Create an error message."""
    return (MessageBuilder()
            .with_content(content)
            .with_type(EnhancedMessageType.SYSTEM_ERROR)
            .with_priority(MessagePriority.HIGH)
            .with_routing_tags("error", error_type)
            .build(**kwargs))


def create_tool_result_message(tool_name: str, result: Any, success: bool = True, **kwargs) -> RichMessage:
    """Create a tool result message."""
    content = json.dumps(result, indent=2) if not isinstance(result, str) else result
    message_type = EnhancedMessageType.TOOL_RESULT if success else EnhancedMessageType.TOOL_ERROR
    priority = MessagePriority.NORMAL if success else MessagePriority.HIGH

    return (MessageBuilder()
            .with_content(content)
            .with_type(message_type)
            .with_priority(priority)
            .with_routing_tags("tool", tool_name)
            .with_formatting(tool_name=tool_name, success=success)
            .build(**kwargs))


def create_quality_assessment_message(assessment: Dict[str, Any], **kwargs) -> RichMessage:
    """Create a quality assessment message."""
    content = json.dumps(assessment, indent=2)
    quality_score = assessment.get("overall_score", 0.0)

    return (MessageBuilder()
            .with_content(content)
            .with_type(EnhancedMessageType.QUALITY_ASSESSMENT)
            .with_quality_scores(quality=quality_score)
            .with_routing_tags("quality", "assessment")
            .build(**kwargs))


def create_progress_message(stage: str, progress: float, details: str = "", **kwargs) -> RichMessage:
    """Create a progress update message."""
    content = f"{stage}: {progress:.1%} complete"
    if details:
        content += f"\n{details}"

    return (MessageBuilder()
            .with_content(content)
            .with_type(EnhancedMessageType.PROGRESS_UPDATE)
            .with_routing_tags("progress", stage)
            .with_formatting(stage=stage, progress=progress)
            .build(**kwargs))


def create_agent_handoff_message(from_agent: str, to_agent: str, context: Dict[str, Any], **kwargs) -> RichMessage:
    """Create an agent handoff message."""
    content = f"Handoff from {from_agent} to {to_agent}"
    if "reason" in context:
        content += f"\nReason: {context['reason']}"

    return (MessageBuilder()
            .with_content(content)
            .with_type(EnhancedMessageType.AGENT_HANDOFF)
            .with_agent(from_agent, to_agent)
            .with_context(MessageContext.HANDOFF)
            .with_routing_tags("handoff", from_agent, to_agent)
            .with_formatting(from_agent=from_agent, to_agent=to_agent, **context)
            .build(**kwargs))