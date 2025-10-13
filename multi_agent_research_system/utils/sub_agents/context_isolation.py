"""
Context Isolation Manager

This module provides context isolation mechanisms for sub-agents to ensure
proper separation of concerns and prevent unintended information sharing
between different sub-agent instances.
"""

import asyncio
import uuid
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
import json
import hashlib
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class IsolationContext:
    """Represents an isolated context for a sub-agent."""

    context_id: str
    agent_type: str
    session_id: str
    parent_context: Optional[str]
    created_at: datetime
    last_accessed: datetime
    data_store: Dict[str, Any] = field(default_factory=dict)
    allowed_data_keys: Set[str] = field(default_factory=set)
    forbidden_data_keys: Set[str] = field(default_factory=set)
    access_log: List[Dict[str, Any]] = field(default_factory=list)
    isolation_level: str = "moderate"  # strict, moderate, permissive
    expires_at: Optional[datetime] = None

    def is_expired(self) -> bool:
        """Check if the isolation context has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def update_access(self, operation: str, data_key: str = None):
        """Update the access log and last accessed timestamp."""
        self.last_accessed = datetime.now()

        log_entry = {
            "timestamp": self.last_accessed.isoformat(),
            "operation": operation,
            "data_key": data_key
        }

        self.access_log.append(log_entry)

        # Keep only last 100 access entries
        if len(self.access_log) > 100:
            self.access_log = self.access_log[-100:]

    def can_access_data(self, data_key: str) -> bool:
        """Check if the context can access a specific data key."""
        if self.isolation_level == "permissive":
            return True

        if data_key in self.forbidden_data_keys:
            return False

        if self.isolation_level == "strict":
            return data_key in self.allowed_data_keys

        # moderate level - allow unless explicitly forbidden
        return True

    def store_data(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None):
        """Store data in the isolated context."""
        if not self.can_access_data(key):
            raise PermissionError(f"Access to key '{key}' is forbidden in this isolation context")

        self.data_store[key] = {
            "value": value,
            "metadata": metadata or {},
            "stored_at": datetime.now().isoformat(),
            "access_count": 0
        }

        self.update_access("store", key)

    def retrieve_data(self, key: str) -> Any:
        """Retrieve data from the isolated context."""
        if not self.can_access_data(key):
            raise PermissionError(f"Access to key '{key}' is forbidden in this isolation context")

        if key not in self.data_store:
            raise KeyError(f"Key '{key}' not found in isolation context")

        data_entry = self.data_store[key]
        data_entry["access_count"] += 1
        data_entry["last_accessed"] = datetime.now().isoformat()

        self.update_access("retrieve", key)
        return data_entry["value"]

    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of the isolation context."""
        return {
            "context_id": self.context_id,
            "agent_type": self.agent_type,
            "session_id": self.session_id,
            "parent_context": self.parent_context,
            "isolation_level": self.isolation_level,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "data_store_size": len(self.data_store),
            "allowed_keys_count": len(self.allowed_data_keys),
            "forbidden_keys_count": len(self.forbidden_data_keys),
            "access_log_entries": len(self.access_log),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_expired": self.is_expired()
        }


@dataclass
class DataSharingRule:
    """Defines rules for sharing data between isolation contexts."""

    rule_id: str
    source_context_pattern: str
    target_context_pattern: str
    data_key_pattern: str
    sharing_allowed: bool
    conditions: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

    def is_expired(self) -> bool:
        """Check if the sharing rule has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def applies_to_contexts(self, source_context: str, target_context: str, data_key: str) -> bool:
        """Check if this rule applies to the given contexts and data key."""
        if self.is_expired():
            return False

        import fnmatch

        # Check patterns
        if not fnmatch.fnmatch(source_context, self.source_context_pattern):
            return False

        if not fnmatch.fnmatch(target_context, self.target_context_pattern):
            return False

        if not fnmatch.fnmatch(data_key, self.data_key_pattern):
            return False

        # Check conditions
        for condition_key, condition_value in self.conditions.items():
            # This is a simplified condition check
            # In practice, you might want more sophisticated condition evaluation
            pass

        return True


class ContextIsolationManager:
    """
    Manages context isolation for sub-agents, ensuring proper separation
    and controlled data sharing between different sub-agent instances.
    """

    def __init__(self):
        self.active_contexts: Dict[str, IsolationContext] = {}
        self.sharing_rules: List[DataSharingRule] = {}
        self.context_hierarchy: Dict[str, List[str]] = {}  # parent -> [children]
        self.isolation_config = {
            "default_isolation_level": "moderate",
            "default_expiry_hours": 24,
            "max_contexts_per_session": 10,
            "cleanup_interval": 300,  # 5 minutes
            "access_log_retention": 100,
            "enable_data_sharing": True,
            "persistent_storage": False,
            "storage_path": "isolated_contexts"
        }
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

    async def initialize(self):
        """Initialize the context isolation manager."""
        logger.info("Initializing Context Isolation Manager")
        self._running = True

        # Create storage directory if persistent storage is enabled
        if self.isolation_config["persistent_storage"]:
            storage_path = Path(self.isolation_config["storage_path"])
            storage_path.mkdir(parents=True, exist_ok=True)

        # Load existing contexts if persistent storage is enabled
        if self.isolation_config["persistent_storage"]:
            await self._load_persistent_contexts()

        # Setup default sharing rules
        await self._setup_default_sharing_rules()

        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def shutdown(self):
        """Shutdown the context isolation manager."""
        logger.info("Shutting down Context Isolation Manager")
        self._running = False

        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Save contexts if persistent storage is enabled
        if self.isolation_config["persistent_storage"]:
            await self._save_persistent_contexts()

        # Cleanup all contexts
        await self.cleanup_all_contexts()

    async def create_isolation_context(
        self,
        agent_type: str,
        session_id: str,
        parent_context: Optional[str] = None,
        isolation_level: Optional[str] = None,
        expiry_hours: Optional[int] = None,
        allowed_data_keys: Optional[Set[str]] = None,
        forbidden_data_keys: Optional[Set[str]] = None
    ) -> str:
        """
        Create a new isolation context.

        Args:
            agent_type: Type of agent creating the context
            session_id: Session ID for the context
            parent_context: Optional parent context ID
            isolation_level: Isolation level (strict, moderate, permissive)
            expiry_hours: Hours until context expires
            allowed_data_keys: Set of allowed data keys (for strict isolation)
            forbidden_data_keys: Set of forbidden data keys

        Returns:
            Context ID for the created isolation context
        """

        # Check session limit
        session_contexts = [
            ctx for ctx in self.active_contexts.values()
            if ctx.session_id == session_id
        ]

        if len(session_contexts) >= self.isolation_config["max_contexts_per_session"]:
            # Cleanup expired contexts first
            await self._cleanup_expired_contexts()

            # Check again
            session_contexts = [
                ctx for ctx in self.active_contexts.values()
                if ctx.session_id == session_id
            ]

            if len(session_contexts) >= self.isolation_config["max_contexts_per_session"]:
                raise RuntimeError(f"Maximum contexts per session ({self.isolation_config['max_contexts_per_session']}) reached")

        # Generate context ID
        context_id = str(uuid.uuid4())

        # Set defaults
        isolation_level = isolation_level or self.isolation_config["default_isolation_level"]
        expiry_hours = expiry_hours or self.isolation_config["default_expiry_hours"]
        expires_at = datetime.now() + timedelta(hours=expiry_hours)

        # Create context
        context = IsolationContext(
            context_id=context_id,
            agent_type=agent_type,
            session_id=session_id,
            parent_context=parent_context,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            allowed_data_keys=allowed_data_keys or set(),
            forbidden_data_keys=forbidden_data_keys or set(),
            isolation_level=isolation_level,
            expires_at=expires_at
        )

        # Store context
        self.active_contexts[context_id] = context

        # Update hierarchy
        if parent_context:
            if parent_context not in self.context_hierarchy:
                self.context_hierarchy[parent_context] = []
            self.context_hierarchy[parent_context].append(context_id)

        # Set default allowed keys based on agent type
        await self._setup_default_permissions(context)

        logger.info(f"Created isolation context {context_id} for agent {agent_type}")
        return context_id

    async def cleanup_isolation_context(self, context_id: str):
        """Cleanup a specific isolation context."""
        if context_id not in self.active_contexts:
            return

        context = self.active_contexts[context_id]

        # Remove from hierarchy
        if context.parent_context and context.parent_context in self.context_hierarchy:
            self.context_hierarchy[context.parent_context] = [
                ctx_id for ctx_id in self.context_hierarchy[context.parent_context]
                if ctx_id != context_id
            ]

        # Remove child contexts
        if context_id in self.context_hierarchy:
            for child_context_id in self.context_hierarchy[context_id]:
                await self.cleanup_isolation_context(child_context_id)
            del self.context_hierarchy[context_id]

        # Remove from active contexts
        del self.active_contexts[context_id]

        logger.info(f"Cleaned up isolation context {context_id}")

    async def cleanup_all_contexts(self):
        """Cleanup all active isolation contexts."""
        context_ids = list(self.active_contexts.keys())
        for context_id in context_ids:
            await self.cleanup_isolation_context(context_id)

    async def get_context(self, context_id: str) -> Optional[IsolationContext]:
        """Get an isolation context by ID."""
        context = self.active_contexts.get(context_id)

        if context and context.is_expired():
            await self.cleanup_isolation_context(context_id)
            return None

        return context

    async def store_data_in_context(
        self,
        context_id: str,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Store data in a specific isolation context."""
        context = await self.get_context(context_id)
        if not context:
            raise ValueError(f"Isolation context {context_id} not found or expired")

        context.store_data(key, value, metadata)

    async def retrieve_data_from_context(self, context_id: str, key: str) -> Any:
        """Retrieve data from a specific isolation context."""
        context = await self.get_context(context_id)
        if not context:
            raise ValueError(f"Isolation context {context_id} not found or expired")

        return context.retrieve_data(key)

    async def share_data_between_contexts(
        self,
        source_context_id: str,
        target_context_id: str,
        data_key: str,
        override_rules: bool = False
    ) -> bool:
        """
        Share data between two isolation contexts based on sharing rules.

        Args:
            source_context_id: Source context ID
            target_context_id: Target context ID
            data_key: Data key to share
            override_rules: Whether to override sharing rules

        Returns:
            True if data was shared, False otherwise
        """

        source_context = await self.get_context(source_context_id)
        target_context = await self.get_context(target_context_id)

        if not source_context or not target_context:
            return False

        # Check sharing rules
        if not override_rules and self.isolation_config["enable_data_sharing"]:
            sharing_allowed = await self._check_sharing_rules(
                source_context_id, target_context_id, data_key
            )
            if not sharing_allowed:
                logger.warning(f"Data sharing not allowed: {source_context_id} -> {target_context_id} ({data_key})")
                return False

        try:
            # Retrieve data from source
            value = source_context.retrieve_data(data_key)

            # Get metadata from source
            metadata = source_context.data_store[data_key].get("metadata", {})

            # Add sharing metadata
            metadata.update({
                "shared_from": source_context_id,
                "shared_at": datetime.now().isoformat(),
                "original_key": data_key
            })

            # Store in target context
            shared_key = f"shared_{data_key}_{source_context_id[:8]}"
            target_context.store_data(shared_key, value, metadata)

            logger.info(f"Shared data {data_key} from {source_context_id} to {target_context_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to share data between contexts: {e}")
            return False

    async def add_sharing_rule(self, rule: DataSharingRule):
        """Add a data sharing rule."""
        self.sharing_rules.append(rule)
        logger.info(f"Added sharing rule {rule.rule_id}")

    async def remove_sharing_rule(self, rule_id: str):
        """Remove a data sharing rule."""
        self.sharing_rules = [rule for rule in self.sharing_rules if rule.rule_id != rule_id]
        logger.info(f"Removed sharing rule {rule_id}")

    async def get_context_summary(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of an isolation context."""
        context = await self.get_context(context_id)
        return context.get_context_summary() if context else None

    async def get_session_contexts(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all contexts for a specific session."""
        contexts = [
            context for context in self.active_contexts.values()
            if context.session_id == session_id and not context.is_expired()
        ]

        return [context.get_context_summary() for context in contexts]

    async def _cleanup_expired_contexts(self):
        """Cleanup expired isolation contexts."""
        expired_contexts = [
            context_id for context_id, context in self.active_contexts.items()
            if context.is_expired()
        ]

        for context_id in expired_contexts:
            await self.cleanup_isolation_context(context_id)

    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while self._running:
            try:
                await asyncio.sleep(self.isolation_config["cleanup_interval"])
                await self._cleanup_expired_contexts()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    async def _setup_default_permissions(self, context: IsolationContext):
        """Setup default permissions for a context based on agent type."""
        agent_type = context.agent_type.lower()

        # Define default permissions by agent type
        default_permissions = {
            "researcher": {
                "allowed": ["research_data", "search_results", "sources", "findings"],
                "forbidden": ["user_credentials", "system_config"]
            },
            "report_writer": {
                "allowed": ["research_findings", "content_data", "structure_info"],
                "forbidden": ["raw_search_results", "system_internal"]
            },
            "editorial_reviewer": {
                "allowed": ["content_data", "research_findings", "quality_metrics"],
                "forbidden": ["raw_search_data", "processing_logs"]
            },
            "quality_assessor": {
                "allowed": ["content_data", "assessment_criteria", "metrics"],
                "forbidden": ["user_data", "system_config"]
            }
        }

        if agent_type in default_permissions:
            permissions = default_permissions[agent_type]
            context.allowed_data_keys.update(permissions["allowed"])
            context.forbidden_data_keys.update(permissions["forbidden"])

    async def _setup_default_sharing_rules(self):
        """Setup default data sharing rules."""

        # Allow researchers to share findings with report writers
        await self.add_sharing_rule(DataSharingRule(
            rule_id="researcher_to_report_writer",
            source_context_pattern="*researcher*",
            target_context_pattern="*report_writer*",
            data_key_pattern="research_findings",
            sharing_allowed=True
        ))

        # Allow report writers to share content with editorial reviewers
        await self.add_sharing_rule(DataSharingRule(
            rule_id="report_writer_to_editorial",
            source_context_pattern="*report_writer*",
            target_context_pattern="*editorial*",
            data_key_pattern="content_data",
            sharing_allowed=True
        ))

        # Deny sharing of sensitive data
        await self.add_sharing_rule(DataSharingRule(
            rule_id="block_sensitive_data",
            source_context_pattern="*",
            target_context_pattern="*",
            data_key_pattern="*credentials*|*password*|*token*",
            sharing_allowed=False
        ))

    async def _check_sharing_rules(
        self,
        source_context_id: str,
        target_context_id: str,
        data_key: str
    ) -> bool:
        """Check if data sharing is allowed based on rules."""

        source_context = self.active_contexts.get(source_context_id)
        target_context = self.active_contexts.get(target_context_id)

        if not source_context or not target_context:
            return False

        # Check all applicable rules
        for rule in self.sharing_rules:
            if rule.applies_to_contexts(
                f"{source_context.agent_type}_{source_context.context_id}",
                f"{target_context.agent_type}_{target_context.context_id}",
                data_key
            ):
                return rule.sharing_allowed

        # Default: allow sharing if no rules apply
        return True

    async def _save_persistent_contexts(self):
        """Save contexts to persistent storage."""
        if not self.isolation_config["persistent_storage"]:
            return

        storage_path = Path(self.isolation_config["storage_path"])

        for context_id, context in self.active_contexts.items():
            try:
                context_file = storage_path / f"{context_id}.json"

                # Convert context to serializable format
                context_data = {
                    "context_id": context.context_id,
                    "agent_type": context.agent_type,
                    "session_id": context.session_id,
                    "parent_context": context.parent_context,
                    "created_at": context.created_at.isoformat(),
                    "last_accessed": context.last_accessed.isoformat(),
                    "data_store": context.data_store,
                    "allowed_data_keys": list(context.allowed_data_keys),
                    "forbidden_data_keys": list(context.forbidden_data_keys),
                    "access_log": context.access_log,
                    "isolation_level": context.isolation_level,
                    "expires_at": context.expires_at.isoformat() if context.expires_at else None
                }

                with open(context_file, 'w') as f:
                    json.dump(context_data, f, indent=2)

            except Exception as e:
                logger.error(f"Failed to save context {context_id}: {e}")

    async def _load_persistent_contexts(self):
        """Load contexts from persistent storage."""
        if not self.isolation_config["persistent_storage"]:
            return

        storage_path = Path(self.isolation_config["storage_path"])

        if not storage_path.exists():
            return

        for context_file in storage_path.glob("*.json"):
            try:
                with open(context_file, 'r') as f:
                    context_data = json.load(f)

                # Reconstruct context
                context = IsolationContext(
                    context_id=context_data["context_id"],
                    agent_type=context_data["agent_type"],
                    session_id=context_data["session_id"],
                    parent_context=context_data["parent_context"],
                    created_at=datetime.fromisoformat(context_data["created_at"]),
                    last_accessed=datetime.fromisoformat(context_data["last_accessed"]),
                    data_store=context_data["data_store"],
                    allowed_data_keys=set(context_data["allowed_data_keys"]),
                    forbidden_data_keys=set(context_data["forbidden_data_keys"]),
                    access_log=context_data["access_log"],
                    isolation_level=context_data["isolation_level"],
                    expires_at=datetime.fromisoformat(context_data["expires_at"]) if context_data["expires_at"] else None
                )

                # Only load if not expired
                if not context.is_expired():
                    self.active_contexts[context.context_id] = context
                    logger.info(f"Loaded persistent context {context.context_id}")
                else:
                    # Remove expired context file
                    context_file.unlink()

            except Exception as e:
                logger.error(f"Failed to load context from {context_file}: {e}")

    def get_isolation_status(self) -> Dict[str, Any]:
        """Get the current status of the isolation manager."""

        contexts_by_type = {}
        for context in self.active_contexts.values():
            agent_type = context.agent_type
            if agent_type not in contexts_by_type:
                contexts_by_type[agent_type] = 0
            contexts_by_type[agent_type] += 1

        return {
            "running": self._running,
            "active_contexts": len(self.active_contexts),
            "contexts_by_type": contexts_by_type,
            "sharing_rules": len(self.sharing_rules),
            "persistent_storage": self.isolation_config["persistent_storage"],
            "max_contexts_per_session": self.isolation_config["max_contexts_per_session"]
        }