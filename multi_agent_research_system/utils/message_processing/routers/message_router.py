"""
Message Router and Filtering System - Advanced Message Routing and Context-Aware Processing

This module provides sophisticated message routing and filtering capabilities with
context-aware processing, intelligent routing decisions, and flexible filter systems.

Key Features:
- Context-aware message routing with intelligent decision making
- Advanced filtering system with multiple filter types
- Dynamic routing rules with priority and condition handling
- Message transformation and enrichment during routing
- Load balancing and failover routing capabilities
- Routing statistics and monitoring
- Message batching and aggregation support
- Integration with processing pipelines and caching
"""

import asyncio
import re
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..core.message_types import RichMessage, EnhancedMessageType, MessagePriority, MessageContext


class RoutingDecision(Enum):
    """Routing decision types."""

    ROUTE = "route"
    FILTER = "filter"
    TRANSFORM = "transform"
    BATCH = "batch"
    DEFER = "defer"
    REJECT = "reject"
    BROADCAST = "broadcast"


class FilterType(Enum):
    """Filter types for message processing."""

    PRIORITY = "priority"
    CONTENT = "content"
    CONTEXT = "context"
    METADATA = "metadata"
    TEMPORAL = "temporal"
    CUSTOM = "custom"


@dataclass
class RoutingRule:
    """Routing rule with conditions and actions."""

    id: str
    name: str
    priority: int
    conditions: Dict[str, Any]
    actions: List[Dict[str, Any]]
    enabled: bool = True
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    hit_count: int = 0
    last_hit: Optional[datetime] = None


@dataclass
class FilterRule:
    """Filter rule for message filtering."""

    id: str
    name: str
    filter_type: FilterType
    conditions: Dict[str, Any]
    action: str  # "allow", "block", "transform"
    transformation: Optional[Dict[str, Any]] = None
    enabled: bool = True
    priority: int = 0
    description: str = ""
    hit_count: int = 0


@dataclass
class RoutingResult:
    """Result of message routing with routing information."""

    decision: RoutingDecision
    destinations: List[str]
    transformations: List[Dict[str, Any]]
    routing_path: List[str]
    rules_applied: List[str]
    filters_applied: List[str]
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class MessageRouter:
    """Advanced message router with context-aware processing and filtering."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize message router with configuration."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Routing components
        self.routing_rules: Dict[str, RoutingRule] = {}
        self.filter_rules: Dict[str, FilterRule] = {}
        self.destinations: Dict[str, Any] = {}
        self.transformers: Dict[str, Callable] = {}

        # Routing statistics
        self.routing_stats = {
            "total_routed": 0,
            "total_filtered": 0,
            "routing_decisions": {},
            "rule_hits": {},
            "filter_hits": {},
            "average_routing_time": 0.0,
            "failed_routes": 0
        }

        # Initialize default routing rules
        self._initialize_default_rules()
        self._initialize_default_transformers()

    def _initialize_default_rules(self):
        """Initialize default routing rules."""
        default_rules = [
            # High-priority error routing
            RoutingRule(
                id="error_high_priority",
                name="Route High Priority Errors",
                priority=100,
                conditions={
                    "message_type": ["system_error", "tool_error"],
                    "priority": ["critical", "high"]
                },
                actions=[
                    {"type": "route", "destination": "error_handler"},
                    {"type": "transform", "transformer": "error_enrichment"},
                    {"type": "broadcast", "destinations": ["monitoring", "alerting"]}
                ],
                description="Route high-priority errors to specialized handlers"
            ),

            # Quality assessment routing
            RoutingRule(
                id="quality_assessment",
                name="Route Quality Assessments",
                priority=80,
                conditions={
                    "message_type": ["quality_assessment", "validation_result"]
                },
                actions=[
                    {"type": "route", "destination": "quality_processor"},
                    {"type": "transform", "transformer": "quality_analysis"}
                ],
                description="Route quality assessments for specialized processing"
            ),

            # Research results routing
            RoutingRule(
                id="research_results",
                name="Route Research Results",
                priority=70,
                conditions={
                    "message_type": ["research_result", "analysis_result"]
                },
                actions=[
                    {"type": "route", "destination": "research_processor"},
                    {"type": "transform", "transformer": "research_enrichment"}
                ],
                description="Route research results for analysis and storage"
            ),

            # Agent handoff routing
            RoutingRule(
                id="agent_handoff",
                name="Route Agent Handoffs",
                priority=90,
                conditions={
                    "message_type": ["agent_handoff"]
                },
                actions=[
                    {"type": "route", "destination": "agent_coordinator"},
                    {"type": "transform", "transformer": "handoff_tracking"}
                ],
                description="Route agent handoffs to coordinator"
            ),

            # Progress updates routing
            RoutingRule(
                id="progress_updates",
                name="Route Progress Updates",
                priority=50,
                conditions={
                    "message_type": ["progress_update", "workflow_stage"]
                },
                actions=[
                    {"type": "route", "destination": "progress_tracker"},
                    {"type": "broadcast", "destinations": ["ui_updates"]}
                ],
                description="Route progress updates to tracking systems"
            ),

            # Default routing for unknown types
            RoutingRule(
                id="default_routing",
                name="Default Routing",
                priority=1,
                conditions={},  # No conditions - matches all
                actions=[
                    {"type": "route", "destination": "default_processor"}
                ],
                description="Default routing for all messages"
            )
        ]

        for rule in default_rules:
            self.routing_rules[rule.id] = rule

    def _initialize_default_transformers(self):
        """Initialize default message transformers."""
        self.transformers = {
            "error_enrichment": self._transform_error_enrichment,
            "quality_analysis": self._transform_quality_analysis,
            "research_enrichment": self._transform_research_enrichment,
            "handoff_tracking": self._transform_handoff_tracking,
            "priority_boost": self._transform_priority_boost,
            "metadata_enrichment": self._transform_metadata_enrichment
        }

    def _initialize_default_filters(self):
        """Initialize default filter rules."""
        default_filters = [
            # Spam filter
            FilterRule(
                id="spam_filter",
                name="Spam Filter",
                filter_type=FilterType.CONTENT,
                conditions={
                    "content_patterns": [r"spam", r"advertisement", r"promotion"],
                    "max_frequency": 10  # Max 10 messages per minute
                },
                action="block",
                priority=90,
                description="Block spam messages"
            ),

            # Size filter
            FilterRule(
                id="size_filter",
                name="Size Filter",
                filter_type=FilterType.CONTENT,
                conditions={
                    "max_length": 10000,  # Max 10k characters
                    "min_length": 1      # Min 1 character
                },
                action="allow",
                priority=80,
                description="Filter messages by size"
            ),

            # Priority filter
            FilterRule(
                id="priority_filter",
                name="Priority Filter",
                filter_type=FilterType.PRIORITY,
                conditions={
                    "allowed_priorities": ["critical", "high", "normal", "low"]
                },
                action="allow",
                priority=70,
                description="Filter by message priority"
            )
        ]

        for filter_rule in default_filters:
            self.filter_rules[filter_rule.id] = filter_rule

    async def route_message(self, message: RichMessage) -> Dict[str, Any]:
        """Route a message through the routing system."""
        start_time = datetime.now()

        try:
            # Apply filters first
            filter_result = await self._apply_filters(message)
            if filter_result["blocked"]:
                self.routing_stats["total_filtered"] += 1
                return {
                    "decision": "filtered",
                    "blocked": True,
                    "reason": filter_result["reason"],
                    "processing_time": (datetime.now() - start_time).total_seconds()
                }

            # Find matching routing rules
            matching_rules = self._find_matching_rules(message)

            # Apply routing rules in priority order
            routing_result = await self._apply_routing_rules(message, matching_rules)

            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_routing_stats(message, routing_result, processing_time)

            return {
                "decision": routing_result.decision.value,
                "destinations": routing_result.destinations,
                "transformations": routing_result.transformations,
                "rules_applied": routing_result.rules_applied,
                "filters_applied": routing_result.filters_applied,
                "processing_time": processing_time,
                "routing_path": routing_result.routing_path
            }

        except Exception as e:
            self.logger.error(f"Routing failed for message {message.id}: {str(e)}")
            self.routing_stats["failed_routes"] += 1

            return {
                "decision": "error",
                "error": str(e),
                "processing_time": (datetime.now() - start_time).total_seconds()
            }

    async def route_batch(self, messages: List[RichMessage]) -> List[Dict[str, Any]]:
        """Route multiple messages in batch."""
        routing_results = []

        # Group messages for potential batch routing
        batch_groups = self._group_messages_for_batch(messages)

        for group_id, group_messages in batch_groups.items():
            if len(group_messages) == 1:
                # Single message routing
                result = await self.route_message(group_messages[0])
                routing_results.append(result)
            else:
                # Batch routing
                result = await self._route_batch_group(group_messages)
                routing_results.extend(result)

        return routing_results

    async def _apply_filters(self, message: RichMessage) -> Dict[str, Any]:
        """Apply filter rules to message."""
        filter_results = []

        # Sort filter rules by priority
        sorted_filters = sorted(
            [f for f in self.filter_rules.values() if f.enabled],
            key=lambda f: f.priority,
            reverse=True
        )

        for filter_rule in sorted_filters:
            if await self._matches_filter_conditions(message, filter_rule):
                filter_rule.hit_count += 1
                filter_results.append(filter_rule.id)

                # Apply filter action
                if filter_rule.action == "block":
                    return {
                        "blocked": True,
                        "reason": f"Blocked by filter: {filter_rule.name}",
                        "filter_id": filter_rule.id
                    }
                elif filter_rule.action == "transform" and filter_rule.transformation:
                    await self._apply_transformation(message, filter_rule.transformation)

        return {
            "blocked": False,
            "applied_filters": filter_results
        }

    async def _matches_filter_conditions(self, message: RichMessage, filter_rule: FilterRule) -> bool:
        """Check if message matches filter conditions."""
        conditions = filter_rule.conditions

        # Content-based filtering
        if filter_rule.filter_type == FilterType.CONTENT:
            if "content_patterns" in conditions:
                content_lower = message.content.lower()
                for pattern in conditions["content_patterns"]:
                    if re.search(pattern, content_lower, re.IGNORECASE):
                        return True

            if "max_length" in conditions and len(message.content) > conditions["max_length"]:
                return True

            if "min_length" in conditions and len(message.content) < conditions["min_length"]:
                return True

        # Priority-based filtering
        elif filter_rule.filter_type == FilterType.PRIORITY:
            if "allowed_priorities" in conditions:
                return message.priority.value in conditions["allowed_priorities"]

            if "blocked_priorities" in conditions:
                return message.priority.value in conditions["blocked_priorities"]

        # Context-based filtering
        elif filter_rule.filter_type == FilterType.CONTEXT:
            if "allowed_contexts" in conditions:
                return message.context.value in conditions["allowed_contexts"]

            if "blocked_contexts" in conditions:
                return message.context.value in conditions["blocked_contexts"]

        # Metadata-based filtering
        elif filter_rule.filter_type == FilterType.METADATA:
            if "required_metadata" in conditions:
                for key, value in conditions["required_metadata"].items():
                    if getattr(message.metadata, key, None) != value:
                        return False
                return True

        # Temporal filtering
        elif filter_rule.filter_type == FilterType.TEMPORAL:
            if "time_window" in conditions:
                # Check if message is within time window
                message_time = message.timestamps.get("created", datetime.now())
                current_time = datetime.now()
                time_diff = (current_time - message_time).total_seconds()

                if time_diff > conditions["time_window"]:
                    return False
                return True

            if "max_frequency" in conditions:
                # Check frequency (simplified - would need message history in practice)
                return True

        # Custom filtering
        elif filter_rule.filter_type == FilterType.CUSTOM:
            custom_filter = conditions.get("custom_filter")
            if custom_filter and callable(custom_filter):
                return custom_filter(message)

        return False

    def _find_matching_rules(self, message: RichMessage) -> List[RoutingRule]:
        """Find routing rules that match the message."""
        matching_rules = []

        for rule in self.routing_rules.values():
            if not rule.enabled:
                continue

            if self._matches_routing_conditions(message, rule):
                matching_rules.append(rule)

        # Sort by priority (higher priority first)
        matching_rules.sort(key=lambda r: r.priority, reverse=True)

        return matching_rules

    def _matches_routing_conditions(self, message: RichMessage, rule: RoutingRule) -> bool:
        """Check if message matches routing rule conditions."""
        conditions = rule.conditions

        # No conditions means match all
        if not conditions:
            return True

        # Message type conditions
        if "message_type" in conditions:
            allowed_types = conditions["message_type"]
            if isinstance(allowed_types, str):
                allowed_types = [allowed_types]

            if message.message_type.value not in allowed_types:
                return False

        # Priority conditions
        if "priority" in conditions:
            allowed_priorities = conditions["priority"]
            if isinstance(allowed_priorities, str):
                allowed_priorities = [allowed_priorities]

            if message.priority.value not in allowed_priorities:
                return False

        # Context conditions
        if "context" in conditions:
            allowed_contexts = conditions["context"]
            if isinstance(allowed_contexts, str):
                allowed_contexts = [allowed_contexts]

            if message.context.value not in allowed_contexts:
                return False

        # Content conditions
        if "content_patterns" in conditions:
            content_lower = message.content.lower()
            for pattern in conditions["content_patterns"]:
                if not re.search(pattern, content_lower, re.IGNORECASE):
                    return False

        # Metadata conditions
        if "metadata" in conditions:
            for key, value in conditions["metadata"].items():
                metadata_value = getattr(message.metadata, key, None)
                if metadata_value != value:
                    return False

        # Quality score conditions
        if "quality_score" in conditions:
            quality_condition = conditions["quality_score"]
            message_quality = message.metadata.quality_score

            if "min" in quality_condition and message_quality < quality_condition["min"]:
                return False

            if "max" in quality_condition and message_quality > quality_condition["max"]:
                return False

        # Session conditions
        if "session_id" in conditions:
            if message.metadata.session_id != conditions["session_id"]:
                return False

        # Agent conditions
        if "source_agent" in conditions:
            if message.metadata.source_agent != conditions["source_agent"]:
                return False

        return True

    async def _apply_routing_rules(self, message: RichMessage, matching_rules: List[RoutingRule]) -> RoutingResult:
        """Apply routing rules to message."""
        destinations = []
        transformations = []
        rules_applied = []
        routing_path = []
        decision = RoutingDecision.ROUTE

        for rule in matching_rules:
            rule.hit_count += 1
            rule.last_hit = datetime.now()
            rules_applied.append(rule.id)

            # Apply rule actions
            for action in rule.actions:
                action_type = action.get("type")

                if action_type == "route":
                    destination = action.get("destination")
                    if destination:
                        destinations.append(destination)
                        routing_path.append(f"route:{destination}")

                elif action_type == "broadcast":
                    broadcast_dests = action.get("destinations", [])
                    destinations.extend(broadcast_dests)
                    for dest in broadcast_dests:
                        routing_path.append(f"broadcast:{dest}")

                elif action_type == "transform":
                    transformer_name = action.get("transformer")
                    if transformer_name in self.transformers:
                        transformation_result = await self.transformers[transformer_name](message, action)
                        if transformation_result:
                            transformations.append(transformation_result)
                        routing_path.append(f"transform:{transformer_name}")

                elif action_type == "filter":
                    # Additional filtering during routing
                    filter_conditions = action.get("conditions", {})
                    if not self._matches_routing_conditions(message,
                        RoutingRule("temp", "temp", 0, filter_conditions, [])):
                        decision = RoutingDecision.FILTER
                        break

                elif action_type == "defer":
                    decision = RoutingDecision.DEFER
                    break

                elif action_type == "reject":
                    decision = RoutingDecision.REJECT
                    destinations = []
                    break

            # Stop processing if we have a non-continue decision
            if decision in [RoutingDecision.DEFER, RoutingDecision.REJECT]:
                break

        # Default destination if none specified
        if not destinations and decision == RoutingDecision.ROUTE:
            destinations = ["default_processor"]

        return RoutingResult(
            decision=decision,
            destinations=destinations,
            transformations=transformations,
            routing_path=routing_path,
            rules_applied=rules_applied,
            filters_applied=[],
            processing_time=0.0
        )

    async def _route_batch_group(self, messages: List[RichMessage]) -> List[Dict[str, Any]]:
        """Route a batch of messages together."""
        # Use first message's routing for the batch
        if not messages:
            return []

        primary_message = messages[0]
        routing_result = await self.route_message(primary_message)

        # Apply same routing to all messages in batch
        batch_results = []
        for message in messages:
            # Create result for each message based on primary routing
            result = routing_result.copy()
            result["batch_size"] = len(messages)
            result["batch_id"] = f"batch_{hash(tuple(m.id for m in messages))}"
            batch_results.append(result)

        return batch_results

    def _group_messages_for_batch(self, messages: List[RichMessage]) -> Dict[str, List[RichMessage]]:
        """Group messages for batch routing."""
        groups = {}

        # Group by message type and priority for potential batching
        for message in messages:
            group_key = f"{message.message_type.value}_{message.priority.value}"

            if group_key not in groups:
                groups[group_key] = []

            groups[group_key].append(message)

        return groups

    async def _apply_transformation(self, message: RichMessage, transformation: Dict[str, Any]):
        """Apply transformation to message."""
        transform_type = transformation.get("type")

        if transform_type == "priority_boost":
            boost_amount = transformation.get("boost", 1)
            current_priority_value = message.priority.value
            new_priority_value = max(1, current_priority_value - boost_amount)

            # Map back to MessagePriority enum
            priority_mapping = {1: "critical", 2: "high", 3: "normal", 4: "low", 5: "bulk"}
            new_priority_name = priority_mapping.get(new_priority_value, "normal")
            message.priority = MessagePriority(new_priority_name)

        elif transform_type == "add_metadata":
            metadata_to_add = transformation.get("metadata", {})
            for key, value in metadata_to_add.items():
                setattr(message.metadata, key, value)

        elif transform_type == "add_routing_tags":
            tags = transformation.get("tags", [])
            for tag in tags:
                message.add_routing_tag(tag)

    # Transformer implementations
    async def _transform_error_enrichment(self, message: RichMessage, action: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich error messages with additional context."""
        # Add error categorization
        error_categories = self._categorize_error(message.content)
        message.metadata.error_category = error_categories["category"]
        message.metadata.error_severity = error_categories["severity"]

        # Add routing tags for error handling
        message.add_routing_tag("error", error_categories["category"])

        return {
            "transformer": "error_enrichment",
            "changes": ["error_categorization", "severity_assessment"]
        }

    async def _transform_quality_analysis(self, message: RichMessage, action: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and enrich quality assessment messages."""
        try:
            import json
            quality_data = json.loads(message.content)

            # Extract key quality metrics
            if "overall_quality" in quality_data:
                message.metadata.quality_score = quality_data["overall_quality"]

            # Add quality-based routing
            quality_score = message.metadata.quality_score or 0.0
            if quality_score >= 0.8:
                message.add_routing_tag("high_quality")
            elif quality_score < 0.5:
                message.add_routing_tag("low_quality")

            return {
                "transformer": "quality_analysis",
                "changes": ["quality_extraction", "routing_tags"]
            }

        except json.JSONDecodeError:
            return {
                "transformer": "quality_analysis",
                "changes": [],
                "error": "Invalid JSON in quality data"
            }

    async def _transform_research_enrichment(self, message: RichMessage, action: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich research messages with additional metadata."""
        # Add research-specific metadata
        word_count = len(message.content.split())
        message.metadata.research_word_count = word_count

        # Estimate research depth
        if word_count > 500:
            message.metadata.research_depth = "comprehensive"
        elif word_count > 200:
            message.metadata.research_depth = "detailed"
        else:
            message.metadata.research_depth = "summary"

        # Add research routing tags
        message.add_routing_tag("research", message.metadata.research_depth)

        return {
            "transformer": "research_enrichment",
            "changes": ["research_depth", "word_count", "routing_tags"]
        }

    async def _transform_handoff_tracking(self, message: RichMessage, action: Dict[str, Any]) -> Dict[str, Any]:
        """Track and enrich agent handoff messages."""
        # Extract handoff information
        handoff_info = self._extract_handoff_details(message.content)

        if handoff_info["from_agent"]:
            message.metadata.handoff_from = handoff_info["from_agent"]
        if handoff_info["to_agent"]:
            message.metadata.handoff_to = handoff_info["to_agent"]

        # Add handoff routing tags
        if handoff_info["from_agent"]:
            message.add_routing_tag("handoff_from", handoff_info["from_agent"])
        if handoff_info["to_agent"]:
            message.add_routing_tag("handoff_to", handoff_info["to_agent"])

        return {
            "transformer": "handoff_tracking",
            "changes": ["handoff_extraction", "routing_tags"]
        }

    async def _transform_priority_boost(self, message: RichMessage, action: Dict[str, Any]) -> Dict[str, Any]:
        """Boost message priority based on conditions."""
        boost_amount = action.get("boost", 1)
        current_priority_value = message.priority.value

        new_priority_value = max(1, current_priority_value - boost_amount)
        priority_mapping = {1: "critical", 2: "high", 3: "normal", 4: "low", 5: "bulk"}
        new_priority_name = priority_mapping.get(new_priority_value, "normal")

        old_priority = message.priority
        message.priority = MessagePriority(new_priority_name)

        return {
            "transformer": "priority_boost",
            "changes": [f"priority_{old_priority.value}_to_{new_priority_name}"]
        }

    async def _transform_metadata_enrichment(self, message: RichMessage, action: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich message with additional metadata."""
        metadata_to_add = action.get("metadata", {})
        changes = []

        for key, value in metadata_to_add.items():
            if hasattr(message.metadata, key):
                old_value = getattr(message.metadata, key)
                setattr(message.metadata, key, value)
                changes.append(f"{key}_{old_value}_to_{value}")
            else:
                setattr(message.metadata, key, value)
                changes.append(f"{key}_added")

        return {
            "transformer": "metadata_enrichment",
            "changes": changes
        }

    # Helper methods
    def _categorize_error(self, content: str) -> Dict[str, str]:
        """Categorize error content."""
        content_lower = content.lower()

        if "timeout" in content_lower:
            return {"category": "timeout", "severity": "medium"}
        elif "permission" in content_lower or "access" in content_lower:
            return {"category": "permission", "severity": "high"}
        elif "connection" in content_lower:
            return {"category": "connection", "severity": "medium"}
        elif "file not found" in content_lower:
            return {"category": "file_error", "severity": "medium"}
        elif "syntax" in content_lower or "parse" in content_lower:
            return {"category": "syntax", "severity": "low"}
        else:
            return {"category": "general", "severity": "medium"}

    def _extract_handoff_details(self, content: str) -> Dict[str, str]:
        """Extract handoff details from content."""
        details = {"from_agent": None, "to_agent": None}

        # Simple regex extraction
        from_match = re.search(r'from\s+([^\s]+)', content, re.IGNORECASE)
        if from_match:
            details["from_agent"] = from_match.group(1)

        to_match = re.search(r'to\s+([^\s]+)', content, re.IGNORECASE)
        if to_match:
            details["to_agent"] = to_match.group(1)

        return details

    def _update_routing_stats(self, message: RichMessage, result: RoutingResult, processing_time: float):
        """Update routing statistics."""
        self.routing_stats["total_routed"] += 1
        self.routing_stats["average_routing_time"] = (
            (self.routing_stats["average_routing_time"] * (self.routing_stats["total_routed"] - 1) + processing_time) /
            self.routing_stats["total_routed"]
        )

        # Update decision stats
        decision_key = result.decision.value
        if decision_key not in self.routing_stats["routing_decisions"]:
            self.routing_stats["routing_decisions"][decision_key] = 0
        self.routing_stats["routing_decisions"][decision_key] += 1

        # Update rule hit stats
        for rule_id in result.rules_applied:
            if rule_id not in self.routing_stats["rule_hits"]:
                self.routing_stats["rule_hits"][rule_id] = 0
            self.routing_stats["rule_hits"][rule_id] += 1

    # Configuration management
    def add_routing_rule(self, rule: RoutingRule):
        """Add a new routing rule."""
        self.routing_rules[rule.id] = rule

    def remove_routing_rule(self, rule_id: str):
        """Remove a routing rule."""
        if rule_id in self.routing_rules:
            del self.routing_rules[rule_id]

    def add_filter_rule(self, filter_rule: FilterRule):
        """Add a new filter rule."""
        self.filter_rules[filter_rule.id] = filter_rule

    def remove_filter_rule(self, filter_id: str):
        """Remove a filter rule."""
        if filter_id in self.filter_rules:
            del self.filter_rules[filter_id]

    def add_transformer(self, name: str, transformer: Callable):
        """Add a custom transformer."""
        self.transformers[name] = transformer

    def add_destination(self, name: str, destination: Any):
        """Add a routing destination."""
        self.destinations[name] = destination

    # Statistics and monitoring
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics."""
        stats = self.routing_stats.copy()

        # Add rule statistics
        stats["rule_details"] = {
            rule_id: {
                "name": rule.name,
                "hit_count": rule.hit_count,
                "last_hit": rule.last_hit.isoformat() if rule.last_hit else None,
                "priority": rule.priority,
                "enabled": rule.enabled
            }
            for rule_id, rule in self.routing_rules.items()
        }

        # Add filter statistics
        stats["filter_details"] = {
            filter_id: {
                "name": filter_rule.name,
                "hit_count": filter_rule.hit_count,
                "filter_type": filter_rule.filter_type.value,
                "action": filter_rule.action,
                "enabled": filter_rule.enabled
            }
            for filter_id, filter_rule in self.filter_rules.items()
        }

        return stats

    def reset_stats(self):
        """Reset routing statistics."""
        self.routing_stats = {
            "total_routed": 0,
            "total_filtered": 0,
            "routing_decisions": {},
            "rule_hits": {},
            "filter_hits": {},
            "average_routing_time": 0.0,
            "failed_routes": 0
        }

        # Reset rule hit counts
        for rule in self.routing_rules.values():
            rule.hit_count = 0
            rule.last_hit = None

        # Reset filter hit counts
        for filter_rule in self.filter_rules.values():
            filter_rule.hit_count = 0

    # Configuration export/import
    def export_config(self) -> Dict[str, Any]:
        """Export router configuration."""
        return {
            "routing_rules": {
                rule_id: {
                    "name": rule.name,
                    "priority": rule.priority,
                    "conditions": rule.conditions,
                    "actions": rule.actions,
                    "enabled": rule.enabled,
                    "description": rule.description
                }
                for rule_id, rule in self.routing_rules.items()
            },
            "filter_rules": {
                filter_id: {
                    "name": filter_rule.name,
                    "filter_type": filter_rule.filter_type.value,
                    "conditions": filter_rule.conditions,
                    "action": filter_rule.action,
                    "transformation": filter_rule.transformation,
                    "enabled": filter_rule.enabled,
                    "priority": filter_rule.priority,
                    "description": filter_rule.description
                }
                for filter_id, filter_rule in self.filter_rules.items()
            },
            "config": self.config
        }

    def import_config(self, config: Dict[str, Any]):
        """Import router configuration."""
        if "routing_rules" in config:
            for rule_id, rule_data in config["routing_rules"].items():
                rule = RoutingRule(
                    id=rule_id,
                    name=rule_data["name"],
                    priority=rule_data["priority"],
                    conditions=rule_data["conditions"],
                    actions=rule_data["actions"],
                    enabled=rule_data.get("enabled", True),
                    description=rule_data.get("description", "")
                )
                self.routing_rules[rule_id] = rule

        if "filter_rules" in config:
            for filter_id, filter_data in config["filter_rules"].items():
                filter_rule = FilterRule(
                    id=filter_id,
                    name=filter_data["name"],
                    filter_type=FilterType(filter_data["filter_type"]),
                    conditions=filter_data["conditions"],
                    action=filter_data["action"],
                    transformation=filter_data.get("transformation"),
                    enabled=filter_data.get("enabled", True),
                    priority=filter_data.get("priority", 0),
                    description=filter_data.get("description", "")
                )
                self.filter_rules[filter_id] = filter_rule