"""
Sub-Session Management System for Gap Research Coordination

Phase 3.2.5: Sophisticated sub-session system with parent-child linking,
resource management, and coordination capabilities.

This module provides comprehensive sub-session management for gap research
operations, including hierarchical session relationships, resource allocation,
status tracking, and coordination mechanisms.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import asyncio
import json
import uuid
import hashlib
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor
import time

from pydantic import BaseModel, Field


class SubSessionStatus(str, Enum):
    """Status of sub-sessions"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    RESEARCHING = "researching"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    MERGED = "merged"


class SubSessionType(str, Enum):
    """Types of sub-sessions"""
    GAP_RESEARCH = "gap_research"
    QUALITY_ENHANCEMENT = "quality_enhancement"
    COVERAGE_EXPANSION = "coverage_expansion"
    SOURCE_VERIFICATION = "source_verification"
    CONTENT_UPDATE = "content_update"
    INTEGRATION = "integration"


class ResourcePriority(str, Enum):
    """Priority levels for resource allocation"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CoordinationStatus(str, Enum):
    """Coordination status between sub-sessions"""
    INDEPENDENT = "independent"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    SYNCHRONIZED = "synchronized"
    MERGE_REQUIRED = "merge_required"


@dataclass
class SubSessionMetadata:
    """Metadata for sub-sessions"""
    sub_session_id: str
    parent_session_id: str
    session_type: SubSessionType
    title: str
    description: str
    created_at: datetime
    updated_at: datetime
    created_by: str
    priority: ResourcePriority
    estimated_duration: str
    estimated_resources: Dict[str, Any]
    tags: Set[str]
    context: Dict[str, Any]

    def __post_init__(self):
        if isinstance(self.tags, list):
            self.tags = set(self.tags)


@dataclass
class SubSessionState:
    """Current state of a sub-session"""
    status: SubSessionStatus
    progress_percentage: float  # 0-100
    current_stage: str
    stages_completed: List[str]
    stages_remaining: List[str]
    error_count: int
    last_error: Optional[str]
    success_criteria_met: List[str]
    success_criteria_remaining: List[str]
    resource_usage: Dict[str, Any]
    performance_metrics: Dict[str, float]

    @property
    def is_active(self) -> bool:
        """Check if sub-session is currently active"""
        return self.status in [SubSessionStatus.ACTIVE, SubSessionStatus.RESEARCHING, SubSessionStatus.ANALYZING]

    @property
    def is_completed(self) -> bool:
        """Check if sub-session is completed"""
        return self.status == SubSessionStatus.COMPLETED

    @property
    def has_errors(self) -> bool:
        """Check if sub-session has errors"""
        return self.error_count > 0


@dataclass
class SubSessionLink:
    """Link between sub-sessions"""
    link_id: str
    from_session_id: str
    to_session_id: str
    link_type: str  # "dependency", "coordination", "merge", "synchronization"
    link_strength: float  # 0-1
    description: str
    created_at: datetime
    coordination_requirements: Dict[str, Any]
    data_flow_specification: Dict[str, Any]

    @property
    def is_dependency(self) -> bool:
        """Check if this is a dependency link"""
        return self.link_type == "dependency"

    @property
    def requires_synchronization(self) -> bool:
        """Check if synchronization is required"""
        return self.link_type in ["synchronization", "merge"]


@dataclass
class ResourceAllocation:
    """Resource allocation for sub-sessions"""
    allocation_id: str
    sub_session_id: str
    resource_type: str
    resource_amount: float
    priority: ResourcePriority
    allocated_at: datetime
    expires_at: Optional[datetime]
    utilization_rate: float  # 0-1
    performance_metrics: Dict[str, float]

    @property
    def is_expired(self) -> bool:
        """Check if allocation has expired"""
        return self.expires_at and datetime.now() > self.expires_at

    @property
    def is_underutilized(self) -> bool:
        """Check if allocation is underutilized"""
        return self.utilization_rate < 0.5


@dataclass
class SubSession:
    """Complete sub-session representation"""
    metadata: SubSessionMetadata
    state: SubSessionState
    data: Dict[str, Any]
    links: List[SubSessionLink]
    resource_allocations: List[ResourceAllocation]
    output_products: List[Dict[str, Any]]
    integration_points: List[str]
    merge_strategy: Optional[Dict[str, Any]]

    @property
    def sub_session_id(self) -> str:
        """Get sub-session ID"""
        return self.metadata.sub_session_id

    @property
    def parent_session_id(self) -> str:
        """Get parent session ID"""
        return self.metadata.parent_session_id

    @property
    def is_ready_for_merge(self) -> bool:
        """Check if sub-session is ready for merging"""
        return (self.state.is_completed and
                len(self.state.success_criteria_remaining) == 0 and
                not self.state.has_errors)

    @property
    def total_resource_usage(self) -> Dict[str, float]:
        """Calculate total resource usage"""
        usage = defaultdict(float)
        for allocation in self.resource_allocations:
            usage[allocation.resource_type] += allocation.resource_amount * allocation.utilization_rate
        return dict(usage)


@dataclass
class SubSessionGroup:
    """Group of related sub-sessions"""
    group_id: str
    group_name: str
    description: str
    sub_session_ids: List[str]
    coordination_status: CoordinationStatus
    coordination_strategy: Dict[str, Any]
    shared_resources: List[str]
    merge_configuration: Dict[str, Any]
    created_at: datetime

    @property
    def is_parallel_executable(self) -> bool:
        """Check if group can be executed in parallel"""
        return self.coordination_status in [CoordinationStatus.PARALLEL, CoordinationStatus.INDEPENDENT]

    @property
    def requires_merge(self) -> bool:
        """Check if group requires merging"""
        return self.coordination_status == CoordinationStatus.MERGE_REQUIRED


class SubSessionManager:
    """
    Advanced sub-session management system for gap research coordination.

    This manager provides comprehensive capabilities for creating, monitoring,
    coordinating, and merging sub-sessions within a hierarchical session structure.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self._sub_sessions: Dict[str, SubSession] = {}
        self._session_groups: Dict[str, SubSessionGroup] = {}
        self._parent_children_map: Dict[str, List[str]] = defaultdict(list)
        self._resource_pools: Dict[str, Dict[str, Any]] = {}
        self._coordination_handlers: Dict[str, Any] = {}
        self._monitoring_active = False
        self._monitoring_thread = None
        self._lock = threading.RLock()

        # Configuration
        self._max_concurrent_sub_sessions = self.config.get('max_concurrent_sub_sessions', 10)
        self._default_sub_session_timeout = self.config.get('default_sub_session_timeout', 3600)  # 1 hour
        self._resource_limits = self.config.get('resource_limits', {})
        self._coordination_settings = self.config.get('coordination_settings', {})

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for sub-session manager"""
        return {
            'max_concurrent_sub_sessions': 10,
            'default_sub_session_timeout': 3600,  # 1 hour
            'resource_limits': {
                'max_memory_per_session': '2GB',
                'max_cpu_per_session': '50%',
                'max_concurrent_research': 5,
                'max_api_calls_per_minute': 100
            },
            'coordination_settings': {
                'auto_coordination': True,
                'merge_timeout': 300,  # 5 minutes
                'coordination_retry_attempts': 3,
                'synchronization_tolerance': 0.1  # 10% tolerance
            },
            'monitoring_settings': {
                'monitoring_interval': 30,  # seconds
                'progress_reporting': True,
                'error_alerting': True,
                'performance_tracking': True
            },
            'persistence_settings': {
                'auto_save': True,
                'save_interval': 60,  # seconds
                'backup_count': 5
            }
        }

    async def create_sub_session(
        self,
        parent_session_id: str,
        session_type: SubSessionType,
        title: str,
        description: str,
        gap_query: Optional[str] = None,
        priority: ResourcePriority = ResourcePriority.MEDIUM,
        estimated_duration: str = "1-2 hours",
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[Set[str]] = None
    ) -> str:
        """
        Create a new sub-session

        Args:
            parent_session_id: ID of the parent session
            session_type: Type of sub-session
            title: Title for the sub-session
            description: Description of sub-session purpose
            gap_query: Specific gap research query (if applicable)
            priority: Resource allocation priority
            estimated_duration: Estimated duration for completion
            context: Additional context data
            tags: Tags for categorization

        Returns:
            ID of the created sub-session
        """
        sub_session_id = self._generate_sub_session_id(parent_session_id, session_type)

        # Create metadata
        metadata = SubSessionMetadata(
            sub_session_id=sub_session_id,
            parent_session_id=parent_session_id,
            session_type=session_type,
            title=title,
            description=description,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            created_by="sub_session_manager",
            priority=priority,
            estimated_duration=estimated_duration,
            estimated_resources=self._estimate_initial_resources(session_type),
            tags=tags or set(),
            context=context or {}
        )

        # Add gap query to context if provided
        if gap_query:
            metadata.context['gap_query'] = gap_query

        # Create initial state
        state = SubSessionState(
            status=SubSessionStatus.INITIALIZING,
            progress_percentage=0.0,
            current_stage="initialization",
            stages_completed=[],
            stages_remaining=self._get_default_stages(session_type),
            error_count=0,
            last_error=None,
            success_criteria_met=[],
            success_criteria_remaining=self._get_default_success_criteria(session_type),
            resource_usage={},
            performance_metrics={}
        )

        # Create sub-session
        sub_session = SubSession(
            metadata=metadata,
            state=state,
            data={},
            links=[],
            resource_allocations=[],
            output_products=[],
            integration_points=[],
            merge_strategy=self._get_default_merge_strategy(session_type)
        )

        # Store sub-session
        with self._lock:
            self._sub_sessions[sub_session_id] = sub_session
            self._parent_children_map[parent_session_id].append(sub_session_id)

        # Log creation
        await self._log_sub_session_event(sub_session_id, "created", {
            "parent_session_id": parent_session_id,
            "session_type": session_type.value,
            "priority": priority.value
        })

        # Start monitoring if not already active
        if not self._monitoring_active:
            await self.start_monitoring()

        return sub_session_id

    async def get_sub_session(self, sub_session_id: str) -> Optional[SubSession]:
        """Get sub-session by ID"""
        with self._lock:
            return self._sub_sessions.get(sub_session_id)

    async def update_sub_session_state(
        self,
        sub_session_id: str,
        status: Optional[SubSessionStatus] = None,
        progress_percentage: Optional[float] = None,
        current_stage: Optional[str] = None,
        error_message: Optional[str] = None,
        success_criteria_met: Optional[List[str]] = None,
        performance_metrics: Optional[Dict[str, float]] = None
    ) -> bool:
        """Update sub-session state"""
        sub_session = await self.get_sub_session(sub_session_id)
        if not sub_session:
            return False

        with self._lock:
            # Update state fields
            if status:
                sub_session.state.status = status
                sub_session.metadata.updated_at = datetime.now()

            if progress_percentage is not None:
                sub_session.state.progress_percentage = max(0, min(100, progress_percentage))

            if current_stage:
                # Move from remaining to completed if stage is advancing
                if current_stage in sub_session.state.stages_remaining:
                    sub_session.state.stages_remaining.remove(current_stage)
                    if current_stage not in sub_session.state.stages_completed:
                        sub_session.state.stages_completed.append(current_stage)
                sub_session.state.current_stage = current_stage

            if error_message:
                sub_session.state.last_error = error_message
                sub_session.state.error_count += 1

            if success_criteria_met:
                for criterion in success_criteria_met:
                    if criterion in sub_session.state.success_criteria_remaining:
                        sub_session.state.success_criteria_remaining.remove(criterion)
                        if criterion not in sub_session.state.success_criteria_met:
                            sub_session.state.success_criteria_met.append(criterion)

            if performance_metrics:
                sub_session.state.performance_metrics.update(performance_metrics)

        # Log state update
        await self._log_sub_session_event(sub_session_id, "state_updated", {
            "status": status.value if status else None,
            "progress": progress_percentage,
            "stage": current_stage,
            "error": error_message
        })

        return True

    async def add_sub_session_data(
        self,
        sub_session_id: str,
        data_key: str,
        data_value: Any,
        data_type: str = "general"
    ) -> bool:
        """Add data to sub-session"""
        sub_session = await self.get_sub_session(sub_session_id)
        if not sub_session:
            return False

        with self._lock:
            if 'data' not in sub_session.data:
                sub_session.data = {}

            sub_session.data[data_key] = {
                'value': data_value,
                'type': data_type,
                'added_at': datetime.now().isoformat(),
                'size': len(str(data_value)) if isinstance(data_value, str) else 1
            }

        await self._log_sub_session_event(sub_session_id, "data_added", {
            "data_key": data_key,
            "data_type": data_type
        })

        return True

    async def link_sub_sessions(
        self,
        from_session_id: str,
        to_session_id: str,
        link_type: str,
        description: str,
        coordination_requirements: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a link between two sub-sessions"""
        link_id = self._generate_link_id()

        link = SubSessionLink(
            link_id=link_id,
            from_session_id=from_session_id,
            to_session_id=to_session_id,
            link_type=link_type,
            link_strength=self._calculate_link_strength(from_session_id, to_session_id, link_type),
            description=description,
            created_at=datetime.now(),
            coordination_requirements=coordination_requirements or {},
            data_flow_specification=self._create_data_flow_specification(link_type)
        )

        # Add link to both sub-sessions
        for session_id in [from_session_id, to_session_id]:
            sub_session = await self.get_sub_session(session_id)
            if sub_session:
                with self._lock:
                    sub_session.links.append(link)

        await self._log_sub_session_event(link_id, "link_created", {
            "from_session": from_session_id,
            "to_session": to_session_id,
            "link_type": link_type
        })

        return link_id

    async def create_sub_session_group(
        self,
        group_name: str,
        description: str,
        sub_session_ids: List[str],
        coordination_status: CoordinationStatus = CoordinationStatus.INDEPENDENT
    ) -> str:
        """Create a group of related sub-sessions"""
        group_id = self._generate_group_id()

        group = SubSessionGroup(
            group_id=group_id,
            group_name=group_name,
            description=description,
            sub_session_ids=sub_session_ids.copy(),
            coordination_status=coordination_status,
            coordination_strategy=self._create_coordination_strategy(coordination_status),
            shared_resources=self._identify_shared_resources(sub_session_ids),
            merge_configuration=self._create_merge_configuration(sub_session_ids),
            created_at=datetime.now()
        )

        with self._lock:
            self._session_groups[group_id] = group

        await self._log_sub_session_event(group_id, "group_created", {
            "group_name": group_name,
            "coordination_status": coordination_status.value,
            "sub_session_count": len(sub_session_ids)
        })

        return group_id

    async def allocate_resources(
        self,
        sub_session_id: str,
        resource_type: str,
        amount: float,
        priority: ResourcePriority = ResourcePriority.MEDIUM,
        expires_in: Optional[int] = None
    ) -> str:
        """Allocate resources to sub-session"""
        allocation_id = self._generate_allocation_id()

        expires_at = None
        if expires_in:
            expires_at = datetime.now() + timedelta(seconds=expires_in)

        allocation = ResourceAllocation(
            allocation_id=allocation_id,
            sub_session_id=sub_session_id,
            resource_type=resource_type,
            resource_amount=amount,
            priority=priority,
            allocated_at=datetime.now(),
            expires_at=expires_at,
            utilization_rate=0.0,
            performance_metrics={}
        )

        # Add allocation to sub-session
        sub_session = await self.get_sub_session(sub_session_id)
        if sub_session:
            with self._lock:
                sub_session.resource_allocations.append(allocation)

        await self._log_sub_session_event(allocation_id, "resource_allocated", {
            "sub_session_id": sub_session_id,
            "resource_type": resource_type,
            "amount": amount,
            "priority": priority.value
        })

        return allocation_id

    async def update_resource_utilization(
        self,
        allocation_id: str,
        utilization_rate: float,
        performance_metrics: Optional[Dict[str, float]] = None
    ) -> bool:
        """Update resource utilization metrics"""
        found = False
        for sub_session in self._sub_sessions.values():
            for allocation in sub_session.resource_allocations:
                if allocation.allocation_id == allocation_id:
                    with self._lock:
                        allocation.utilization_rate = max(0, min(1, utilization_rate))
                        if performance_metrics:
                            allocation.performance_metrics.update(performance_metrics)
                    found = True
                    break
            if found:
                break

        return found

    async def get_parent_children(self, parent_session_id: str) -> List[str]:
        """Get all child sub-session IDs for a parent session"""
        with self._lock:
            return self._parent_children_map.get(parent_session_id, []).copy()

    async def get_session_hierarchy(self, root_session_id: str) -> Dict[str, List[str]]:
        """Get complete session hierarchy"""
        hierarchy = {root_session_id: []}

        def _build_hierarchy(session_id: str, level: int = 0):
            children = await self.get_parent_children(session_id)
            if children:
                hierarchy[session_id] = children
                for child_id in children:
                    _build_hierarchy(child_id, level + 1)

        await _build_hierarchy(root_session_id)
        return hierarchy

    async def get_ready_for_merge(self, parent_session_id: str) -> List[str]:
        """Get sub-sessions ready for merging"""
        ready_sessions = []

        for sub_session_id in await self.get_parent_children(parent_session_id):
            sub_session = await self.get_sub_session(sub_session_id)
            if sub_session and sub_session.is_ready_for_merge:
                ready_sessions.append(sub_session_id)

        return ready_sessions

    async def merge_sub_sessions(
        self,
        sub_session_ids: List[str],
        merge_strategy: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Merge multiple sub-sessions"""
        merge_id = self._generate_merge_id()

        # Validate all sub-sessions exist and are ready
        sub_sessions = []
        for session_id in sub_session_ids:
            sub_session = await self.get_sub_session(session_id)
            if not sub_session:
                raise ValueError(f"Sub-session {session_id} not found")
            if not sub_session.is_ready_for_merge:
                raise ValueError(f"Sub-session {session_id} is not ready for merge")
            sub_sessions.append(sub_session)

        # Perform merge
        merge_result = await self._perform_merge(sub_sessions, merge_strategy or {})

        # Update sub-session states
        for sub_session in sub_sessions:
            await self.update_sub_session_state(
                sub_session.sub_session_id,
                status=SubSessionStatus.MERGED
            )

        await self._log_sub_session_event(merge_id, "merge_completed", {
            "sub_sessions": sub_session_ids,
            "merge_result_size": len(str(merge_result))
        })

        return merge_result

    async def coordinate_sub_sessions(self, group_id: str) -> bool:
        """Coordinate sub-sessions within a group"""
        group = self._session_groups.get(group_id)
        if not group:
            return False

        coordination_result = await self._execute_coordination_strategy(group)

        await self._log_sub_session_event(group_id, "coordination_executed", {
            "strategy": group.coordination_strategy,
            "result": coordination_result
        })

        return coordination_result

    async def start_monitoring(self) -> None:
        """Start background monitoring of sub-sessions"""
        if self._monitoring_active:
            return

        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()

    async def stop_monitoring(self) -> None:
        """Stop background monitoring"""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)

    def _monitoring_loop(self) -> None:
        """Background monitoring loop"""
        while self._monitoring_active:
            try:
                asyncio.run(self._monitoring_cycle())
                time.sleep(self.config.get('monitoring_settings', {}).get('monitoring_interval', 30))
            except Exception as e:
                # Log error but continue monitoring
                print(f"Monitoring error: {e}")

    async def _monitoring_cycle(self) -> None:
        """Single monitoring cycle"""
        current_time = datetime.now()

        # Check for expired allocations
        for sub_session in self._sub_sessions.values():
            expired_allocations = [
                allocation for allocation in sub_session.resource_allocations
                if allocation.is_expired
            ]

            for allocation in expired_allocations:
                await self._log_sub_session_event(allocation.allocation_id, "allocation_expired", {
                    "sub_session_id": sub_session.sub_session_id,
                    "resource_type": allocation.resource_type
                })

        # Check for stuck sub-sessions
        timeout_threshold = self._default_sub_session_timeout
        for sub_session in self._sub_sessions.values():
            if (sub_session.state.is_active and
                (current_time - sub_session.metadata.updated_at).total_seconds() > timeout_threshold):
                await self.update_sub_session_state(
                    sub_session.sub_session_id,
                    status=SubSessionStatus.PAUSED,
                    error_message="Session timeout - paused for review"
                )

        # Check resource utilization
        for sub_session in self._sub_sessions.values():
            underutilized_allocations = [
                allocation for allocation in sub_session.resource_allocations
                if allocation.is_underutilized and allocation.priority == ResourcePriority.HIGH
            ]

            if underutilized_allocations:
                await self._log_sub_session_event(sub_session.sub_session_id, "underutilized_resources", {
                    "allocations": [alloc.allocation_id for alloc in underutilized_allocations]
                })

    def _generate_sub_session_id(self, parent_session_id: str, session_type: SubSessionType) -> str:
        """Generate unique sub-session ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_hash = hashlib.md5(f"{parent_session_id}{session_type}{timestamp}".encode()).hexdigest()[:8]
        return f"sub_{session_type.value}_{timestamp}_{unique_hash}"

    def _generate_link_id(self) -> str:
        """Generate unique link ID"""
        return f"link_{uuid.uuid4().hex[:12]}"

    def _generate_group_id(self) -> str:
        """Generate unique group ID"""
        return f"group_{uuid.uuid4().hex[:12]}"

    def _generate_allocation_id(self) -> str:
        """Generate unique allocation ID"""
        return f"alloc_{uuid.uuid4().hex[:12]}"

    def _generate_merge_id(self) -> str:
        """Generate unique merge ID"""
        return f"merge_{uuid.uuid4().hex[:12]}"

    def _estimate_initial_resources(self, session_type: SubSessionType) -> Dict[str, Any]:
        """Estimate initial resource requirements"""
        estimates = {
            SubSessionType.GAP_RESEARCH: {
                "cpu": "30%",
                "memory": "1GB",
                "api_calls": 50,
                "storage": "100MB"
            },
            SubSessionType.QUALITY_ENHANCEMENT: {
                "cpu": "20%",
                "memory": "512MB",
                "api_calls": 20,
                "storage": "50MB"
            },
            SubSessionType.COVERAGE_EXPANSION: {
                "cpu": "25%",
                "memory": "768MB",
                "api_calls": 35,
                "storage": "75MB"
            },
            SubSessionType.SOURCE_VERIFICATION: {
                "cpu": "15%",
                "memory": "256MB",
                "api_calls": 15,
                "storage": "25MB"
            },
            SubSessionType.CONTENT_UPDATE: {
                "cpu": "10%",
                "memory": "128MB",
                "api_calls": 5,
                "storage": "10MB"
            },
            SubSessionType.INTEGRATION: {
                "cpu": "20%",
                "memory": "512MB",
                "api_calls": 10,
                "storage": "30MB"
            }
        }

        return estimates.get(session_type, {
            "cpu": "20%",
            "memory": "512MB",
            "api_calls": 25,
            "storage": "50MB"
        })

    def _get_default_stages(self, session_type: SubSessionType) -> List[str]:
        """Get default stages for session type"""
        stage_map = {
            SubSessionType.GAP_RESEARCH: [
                "initialization", "query_formulation", "research_execution",
                "content_analysis", "integration_preparation"
            ],
            SubSessionType.QUALITY_ENHANCEMENT: [
                "initialization", "quality_assessment", "improvement_planning",
                "enhancement_execution", "validation"
            ],
            SubSessionType.COVERAGE_EXPANSION: [
                "initialization", "gap_identification", "research_planning",
                "content_expansion", "coverage_validation"
            ],
            SubSessionType.SOURCE_VERIFICATION: [
                "initialization", "source_identification", "verification_execution",
                "quality_assessment", "results_documentation"
            ],
            SubSessionType.CONTENT_UPDATE: [
                "initialization", "content_analysis", "update_planning",
                "content_modification", "quality_check"
            ],
            SubSessionType.INTEGRATION: [
                "initialization", "data_collection", "integration_planning",
                "execution", "validation", "finalization"
            ]
        }

        return stage_map.get(session_type, ["initialization", "execution", "completion"])

    def _get_default_success_criteria(self, session_type: SubSessionType) -> List[str]:
        """Get default success criteria for session type"""
        criteria_map = {
            SubSessionType.GAP_RESEARCH: [
                "gap_query_researched", "relevant_sources_found", "information_extracted",
                "quality_threshold_met", "integration_ready"
            ],
            SubSessionType.QUALITY_ENHANCEMENT: [
                "quality_issues_identified", "improvements_implemented",
                "quality_metrics_improved", "validation_passed"
            ],
            SubSessionType.COVERAGE_EXPANSION: [
                "coverage_gaps_identified", "missing_content_added",
                "coverage_completeness_achieved", "integration_verified"
            ],
            SubSessionType.SOURCE_VERIFICATION: [
                "sources_identified", "verification_completed",
                "quality_assessed", "findings_documented"
            ],
            SubSessionType.CONTENT_UPDATE: [
                "content_analyzed", "updates_planned", "modifications_completed",
                "quality_validated"
            ],
            SubSessionType.INTEGRATION: [
                "data_collected", "integration_planned", "integration_executed",
                "validation_completed", "finalization_approved"
            ]
        }

        return criteria_map.get(session_type, ["task_completed", "quality_achieved"])

    def _get_default_merge_strategy(self, session_type: SubSessionType) -> Dict[str, Any]:
        """Get default merge strategy for session type"""
        strategy_map = {
            SubSessionType.GAP_RESEARCH: {
                "merge_type": "content_integration",
                "priority": "high",
                "conflict_resolution": "merge_all",
                "validation_required": True
            },
            SubSessionType.QUALITY_ENHANCEMENT: {
                "merge_type": "quality_overlay",
                "priority": "medium",
                "conflict_resolution": "highest_quality",
                "validation_required": True
            },
            SubSessionType.COVERAGE_EXPANSION: {
                "merge_type": "content_expansion",
                "priority": "medium",
                "conflict_resolution": "comprehensive",
                "validation_required": True
            },
            SubSessionType.SOURCE_VERIFICATION: {
                "merge_type": "metadata_update",
                "priority": "low",
                "conflict_resolution": "replace_existing",
                "validation_required": False
            },
            SubSessionType.CONTENT_UPDATE: {
                "merge_type": "content_replace",
                "priority": "high",
                "conflict_resolution": "use_newest",
                "validation_required": True
            },
            SubSessionType.INTEGRATION: {
                "merge_type": "full_integration",
                "priority": "critical",
                "conflict_resolution": "manual_review",
                "validation_required": True
            }
        }

        return strategy_map.get(session_type, {
            "merge_type": "standard",
            "priority": "medium",
            "conflict_resolution": "merge_all",
            "validation_required": True
        })

    def _calculate_link_strength(
        self,
        from_session_id: str,
        to_session_id: str,
        link_type: str
    ) -> float:
        """Calculate link strength between sub-sessions"""
        # Base strength by link type
        base_strengths = {
            "dependency": 0.9,
            "coordination": 0.7,
            "merge": 0.8,
            "synchronization": 0.6
        }

        base_strength = base_strengths.get(link_type, 0.5)

        # Adjust based on session types
        from_session = self._sub_sessions.get(from_session_id)
        to_session = self._sub_sessions.get(to_session_id)

        if from_session and to_session:
            # Stronger links for same type sessions
            if from_session.metadata.session_type == to_session.metadata.session_type:
                base_strength += 0.1

            # Stronger links for gap research sessions
            if (from_session.metadata.session_type == SubSessionType.GAP_RESEARCH or
                to_session.metadata.session_type == SubSessionType.GAP_RESEARCH):
                base_strength += 0.1

        return min(base_strength, 1.0)

    def _create_data_flow_specification(self, link_type: str) -> Dict[str, Any]:
        """Create data flow specification for link"""
        specifications = {
            "dependency": {
                "flow_direction": "unidirectional",
                "data_types": ["status", "completion"],
                "timing": "immediate",
                "validation": "required"
            },
            "coordination": {
                "flow_direction": "bidirectional",
                "data_types": ["status", "progress", "resources"],
                "timing": "periodic",
                "validation": "optional"
            },
            "merge": {
                "flow_direction": "convergent",
                "data_types": ["content", "metadata", "results"],
                "timing": "completion",
                "validation": "required"
            },
            "synchronization": {
                "flow_direction": "bidirectional",
                "data_types": ["status", "data", "timing"],
                "timing": "real_time",
                "validation": "required"
            }
        }

        return specifications.get(link_type, {
            "flow_direction": "unidirectional",
            "data_types": ["basic"],
            "timing": "immediate",
            "validation": "optional"
        })

    def _create_coordination_strategy(self, coordination_status: CoordinationStatus) -> Dict[str, Any]:
        """Create coordination strategy for group"""
        strategies = {
            CoordinationStatus.INDEPENDENT: {
                "execution_mode": "independent",
                "coordination_required": False,
                "synchronization_points": [],
                "resource_sharing": "none"
            },
            CoordinationStatus.SEQUENTIAL: {
                "execution_mode": "sequential",
                "coordination_required": True,
                "synchronization_points": ["completion"],
                "resource_sharing": "sequential"
            },
            CoordinationStatus.PARALLEL: {
                "execution_mode": "parallel",
                "coordination_required": True,
                "synchronization_points": ["start", "completion"],
                "resource_sharing": "pooled"
            },
            CoordinationStatus.SYNCHRONIZED: {
                "execution_mode": "synchronized",
                "coordination_required": True,
                "synchronization_points": ["start", "milestones", "completion"],
                "resource_sharing": "coordinated"
            },
            CoordinationStatus.MERGE_REQUIRED: {
                "execution_mode": "independent_until_merge",
                "coordination_required": True,
                "synchronization_points": ["merge_point"],
                "resource_sharing": "independent"
            }
        }

        return strategies.get(coordination_status, strategies[CoordinationStatus.INDEPENDENT])

    def _identify_shared_resources(self, sub_session_ids: List[str]) -> List[str]:
        """Identify shared resources among sub-sessions"""
        resource_usage = defaultdict(list)

        for session_id in sub_session_ids:
            sub_session = self._sub_sessions.get(session_id)
            if sub_session:
                for allocation in sub_session.resource_allocations:
                    resource_usage[allocation.resource_type].append(session_id)

        # Resources shared by multiple sessions
        shared_resources = [
            resource_type for resource_type, sessions in resource_usage.items()
            if len(sessions) > 1
        ]

        return shared_resources

    def _create_merge_configuration(self, sub_session_ids: List[str]) -> Dict[str, Any]:
        """Create merge configuration for sub-sessions"""
        return {
            "merge_strategy": "intelligent_merge",
            "conflict_resolution": "quality_priority",
            "validation_required": True,
            "merge_order": "priority_based",
            "integration_points": self._identify_integration_points(sub_session_ids),
            "merge_timeout": self.config.get('coordination_settings', {}).get('merge_timeout', 300)
        }

    def _identify_integration_points(self, sub_session_ids: List[str]) -> List[str]:
        """Identify integration points between sub-sessions"""
        integration_points = []

        for session_id in sub_session_ids:
            sub_session = self._sub_sessions.get(session_id)
            if sub_session:
                integration_points.extend(sub_session.integration_points)

        return list(set(integration_points))

    async def _perform_merge(
        self,
        sub_sessions: List[SubSession],
        merge_strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform actual merge of sub-sessions"""
        merge_result = {
            "merged_data": {},
            "merged_metadata": {
                "merge_timestamp": datetime.now().isoformat(),
                "sub_sessions_merged": [s.sub_session_id for s in sub_sessions],
                "merge_strategy": merge_strategy
            },
            "validation_results": {},
            "integration_summary": {}
        }

        # Collect all data
        all_data = {}
        for sub_session in sub_sessions:
            all_data[sub_session.sub_session_id] = sub_session.data

        # Apply merge strategy
        merge_type = merge_strategy.get("merge_type", "standard")

        if merge_type == "content_integration":
            merge_result["merged_data"] = await self._merge_content_integration(all_data)
        elif merge_type == "quality_overlay":
            merge_result["merged_data"] = await self._merge_quality_overlay(all_data)
        elif merge_type == "content_expansion":
            merge_result["merged_data"] = await self._merge_content_expansion(all_data)
        else:
            merge_result["merged_data"] = await self._merge_standard(all_data)

        # Validate merge result
        merge_result["validation_results"] = await self._validate_merge_result(
            merge_result["merged_data"], sub_sessions
        )

        # Create integration summary
        merge_result["integration_summary"] = await self._create_integration_summary(
            sub_sessions, merge_result
        )

        return merge_result

    async def _merge_content_integration(self, all_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Merge data with content integration strategy"""
        merged = {}

        # Collect all content
        all_content = []
        for session_id, data in all_data.items():
            for key, value in data.items():
                if isinstance(value, dict) and 'value' in value:
                    if value.get('type') == 'content':
                        all_content.append({
                            'session_id': session_id,
                            'key': key,
                            'content': value['value'],
                            'added_at': value.get('added_at')
                        })

        # Sort by addition time
        all_content.sort(key=lambda x: x.get('added_at', ''))

        # Merge content
        merged['integrated_content'] = {
            'sections': [item['content'] for item in all_content],
            'sources': list(set(item['session_id'] for item in all_content)),
            'integration_timestamp': datetime.now().isoformat()
        }

        # Add non-content data
        for session_id, data in all_data.items():
            for key, value in data.items():
                if isinstance(value, dict) and value.get('type') != 'content':
                    merged[f"{session_id}_{key}"] = value['value']

        return merged

    async def _merge_quality_overlay(self, all_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Merge data with quality overlay strategy"""
        merged = {}

        # Find highest quality data for each key
        quality_scores = {}

        for session_id, data in all_data.items():
            for key, value in data.items():
                if isinstance(value, dict):
                    quality_key = f"{value.get('type', 'unknown')}_{key}"
                    current_quality = quality_scores.get(quality_key, {}).get('quality', 0)

                    # Simple quality assessment (can be enhanced)
                    new_quality = self._assess_data_quality(value)

                    if new_quality > current_quality:
                        quality_scores[quality_key] = {
                            'data': value['value'],
                            'quality': new_quality,
                            'source_session': session_id
                        }

        # Build merged result
        for quality_key, info in quality_scores.items():
            merged[quality_key] = {
                'value': info['data'],
                'quality_score': info['quality'],
                'source_session': info['source_session']
            }

        return merged

    async def _merge_content_expansion(self, all_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Merge data with content expansion strategy"""
        merged = {}

        # Start with base content and expand
        base_content = None
        expansions = []

        for session_id, data in all_data.items():
            for key, value in data.items():
                if isinstance(value, dict) and value.get('type') == 'content':
                    if not base_content:
                        base_content = value['value']
                    else:
                        expansions.append({
                            'session_id': session_id,
                            'expansion': value['value']
                        })

        if base_content:
            merged['expanded_content'] = {
                'base_content': base_content,
                'expansions': expansions,
                'expansion_count': len(expansions),
                'expansion_timestamp': datetime.now().isoformat()
            }

        # Add other data
        for session_id, data in all_data.items():
            for key, value in data.items():
                if isinstance(value, dict) and value.get('type') != 'content':
                    merged[f"{session_id}_{key}"] = value['value']

        return merged

    async def _merge_standard(self, all_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Standard merge strategy"""
        merged = {}

        for session_id, data in all_data.items():
            for key, value in data.items():
                if isinstance(value, dict):
                    merged[f"{session_id}_{key}"] = {
                        'value': value['value'],
                        'type': value.get('type', 'unknown'),
                        'source_session': session_id,
                        'added_at': value.get('added_at')
                    }

        return merged

    def _assess_data_quality(self, data_value: Dict[str, Any]) -> float:
        """Simple data quality assessment"""
        quality = 0.5  # Base quality

        # Size factor
        size = data_value.get('size', 0)
        if size > 1000:
            quality += 0.2
        elif size > 100:
            quality += 0.1

        # Type factor
        data_type = data_value.get('type', 'unknown')
        if data_type in ['content', 'research', 'analysis']:
            quality += 0.2
        elif data_type in ['metadata', 'status']:
            quality += 0.1

        return min(quality, 1.0)

    async def _validate_merge_result(
        self,
        merged_data: Dict[str, Any],
        original_sessions: List[SubSession]
    ) -> Dict[str, Any]:
        """Validate merge result"""
        validation_results = {
            "validation_passed": True,
            "validation_errors": [],
            "validation_warnings": [],
            "data_integrity_check": {},
            "completeness_check": {}
        }

        # Check data integrity
        original_data_keys = set()
        for session in original_sessions:
            original_data_keys.update(session.data.keys())

        merged_data_keys = set()
        for key in merged_data.keys():
            if isinstance(merged_data[key], dict) and 'value' in merged_data[key]:
                merged_data_keys.add(key)
            else:
                merged_data_keys.add(key)

        # Check for missing data
        missing_keys = original_data_keys - merged_data_keys
        if missing_keys:
            validation_results["validation_warnings"].append(f"Missing keys in merge: {missing_keys}")

        # Check for unexpected data
        unexpected_keys = merged_data_keys - original_data_keys
        if unexpected_keys:
            validation_results["validation_warnings"].append(f"Unexpected keys in merge: {unexpected_keys}")

        # Check merge size
        merged_size = len(str(merged_data))
        if merged_size == 0:
            validation_results["validation_errors"].append("Merge resulted in empty data")
            validation_results["validation_passed"] = False

        validation_results["data_integrity_check"] = {
            "original_keys": len(original_data_keys),
            "merged_keys": len(merged_data_keys),
            "missing_keys": len(missing_keys),
            "unexpected_keys": len(unexpected_keys)
        }

        return validation_results

    async def _create_integration_summary(
        self,
        original_sessions: List[SubSession],
        merge_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create integration summary"""
        summary = {
            "integration_timestamp": datetime.now().isoformat(),
            "source_sessions": [s.sub_session_id for s in original_sessions],
            "session_types": [s.metadata.session_type.value for s in original_sessions],
            "total_data_size": len(str(merge_result.get("merged_data", {}))),
            "merge_quality": self._assess_merge_quality(merge_result),
            "integration_points_used": self._count_integration_points(original_sessions),
            "resource_efficiency": self._calculate_resource_efficiency(original_sessions)
        }

        return summary

    def _assess_merge_quality(self, merge_result: Dict[str, Any]) -> float:
        """Assess quality of merge result"""
        quality = 0.5

        # Validation results
        validation = merge_result.get("validation_results", {})
        if validation.get("validation_passed", False):
            quality += 0.3

        # Data size
        data_size = len(str(merge_result.get("merged_data", {})))
        if data_size > 1000:
            quality += 0.1
        elif data_size > 100:
            quality += 0.05

        # No validation errors
        if not validation.get("validation_errors", []):
            quality += 0.1

        return min(quality, 1.0)

    def _count_integration_points(self, sessions: List[SubSession]) -> int:
        """Count integration points used"""
        return sum(len(s.integration_points) for s in sessions)

    def _calculate_resource_efficiency(self, sessions: List[SubSession]) -> float:
        """Calculate resource efficiency of sessions"""
        total_allocated = sum(
            sum(alloc.resource_amount for alloc in s.resource_allocations)
            for s in sessions
        )

        total_utilized = sum(
            sum(alloc.resource_amount * alloc.utilization_rate for alloc in s.resource_allocations)
            for s in sessions
        )

        if total_allocated > 0:
            return total_utilized / total_allocated
        return 0.0

    async def _execute_coordination_strategy(self, group: SubSessionGroup) -> bool:
        """Execute coordination strategy for a group"""
        strategy = group.coordination_strategy
        execution_mode = strategy.get("execution_mode", "independent")

        if execution_mode == "independent":
            return True  # No coordination needed
        elif execution_mode == "parallel":
            return await self._coordinate_parallel_execution(group)
        elif execution_mode == "sequential":
            return await self._coordinate_sequential_execution(group)
        elif execution_mode == "synchronized":
            return await self._coordinate_synchronized_execution(group)

        return True

    async def _coordinate_parallel_execution(self, group: SubSessionGroup) -> bool:
        """Coordinate parallel execution of sub-sessions"""
        # Check if all sessions can run in parallel
        ready_sessions = []
        for session_id in group.sub_session_ids:
            sub_session = await self.get_sub_session(session_id)
            if sub_session and sub_session.state.status in [SubSessionStatus.INITIALIZING, SubSessionStatus.ACTIVE]:
                ready_sessions.append(session_id)

        # Start all ready sessions
        for session_id in ready_sessions:
            await self.update_sub_session_state(session_id, status=SubSessionStatus.ACTIVE)

        return len(ready_sessions) > 0

    async def _coordinate_sequential_execution(self, group: SubSessionGroup) -> bool:
        """Coordinate sequential execution of sub-sessions"""
        # Find first active/inactive session
        for session_id in group.sub_session_ids:
            sub_session = await self.get_sub_session(session_id)
            if sub_session:
                if sub_session.state.status == SubSessionStatus.COMPLETED:
                    continue  # Move to next
                elif sub_session.state.status in [SubSessionStatus.INITIALIZING, SubSessionStatus.ACTIVE]:
                    # This is the current active session
                    return True
                else:
                    # Start this session
                    await self.update_sub_session_state(session_id, status=SubSessionStatus.ACTIVE)
                    return True

        return False

    async def _coordinate_synchronized_execution(self, group: SubSessionGroup) -> bool:
        """Coordinate synchronized execution of sub-sessions"""
        # Check if all sessions are at the same progress level
        progress_levels = []
        for session_id in group.sub_session_ids:
            sub_session = await self.get_sub_session(session_id)
            if sub_session:
                progress_levels.append(sub_session.state.progress_percentage)

        if not progress_levels:
            return False

        # Check tolerance
        max_progress = max(progress_levels)
        min_progress = min(progress_levels)
        tolerance = self.config.get('coordination_settings', {}).get('synchronization_tolerance', 0.1)

        if (max_progress - min_progress) / 100 <= tolerance:
            # Within tolerance, can proceed
            for session_id in group.sub_session_ids:
                sub_session = await self.get_sub_session(session_id)
                if sub_session and sub_session.state.status == SubSessionStatus.INITIALIZING:
                    await self.update_sub_session_state(session_id, status=SubSessionStatus.ACTIVE)
            return True

        # Need to wait for slower sessions
        return False

    async def _log_sub_session_event(self, entity_id: str, event_type: str, event_data: Dict[str, Any]) -> None:
        """Log sub-session related events"""
        # This would integrate with the logging system
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "entity_id": entity_id,
            "event_type": event_type,
            "event_data": event_data
        }

        # For now, just print (would integrate with proper logging system)
        print(f"SubSession Event: {event_type} for {entity_id}")

    async def get_sub_session_statistics(self, parent_session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about sub-sessions"""
        sessions_to_analyze = []

        if parent_session_id:
            sessions_to_analyze = [
                self._sub_sessions[sid] for sid in await self.get_parent_children(parent_session_id)
                if sid in self._sub_sessions
            ]
        else:
            sessions_to_analyze = list(self._sub_sessions.values())

        if not sessions_to_analyze:
            return {}

        stats = {
            "total_sub_sessions": len(sessions_to_analyze),
            "status_distribution": {},
            "type_distribution": {},
            "priority_distribution": {},
            "average_progress": 0.0,
            "total_resource_usage": defaultdict(float),
            "completion_rate": 0.0,
            "error_rate": 0.0
        }

        total_progress = 0
        completed_count = 0
        error_count = 0

        for session in sessions_to_analyze:
            # Status distribution
            status = session.state.status.value
            stats["status_distribution"][status] = stats["status_distribution"].get(status, 0) + 1

            # Type distribution
            session_type = session.metadata.session_type.value
            stats["type_distribution"][session_type] = stats["type_distribution"].get(session_type, 0) + 1

            # Priority distribution
            priority = session.metadata.priority.value
            stats["priority_distribution"][priority] = stats["priority_distribution"].get(priority, 0) + 1

            # Progress
            total_progress += session.state.progress_percentage

            # Completion and errors
            if session.state.is_completed:
                completed_count += 1
            if session.state.has_errors:
                error_count += 1

            # Resource usage
            for resource_type, amount in session.total_resource_usage.items():
                stats["total_resource_usage"][resource_type] += amount

        # Calculate averages and rates
        if sessions_to_analyze:
            stats["average_progress"] = total_progress / len(sessions_to_analyze)
            stats["completion_rate"] = completed_count / len(sessions_to_analyze)
            stats["error_rate"] = error_count / len(sessions_to_analyze)

        stats["total_resource_usage"] = dict(stats["total_resource_usage"])

        return stats


# Factory function for easy instantiation
def create_sub_session_manager(config: Optional[Dict[str, Any]] = None) -> SubSessionManager:
    """Create a configured sub-session manager"""
    return SubSessionManager(config)


# Utility functions for common operations
async def create_gap_research_sub_session(
    manager: SubSessionManager,
    parent_session_id: str,
    gap_query: str,
    priority: ResourcePriority = ResourcePriority.HIGH
) -> str:
    """Create a gap research sub-session with standard configuration"""
    return await manager.create_sub_session(
        parent_session_id=parent_session_id,
        session_type=SubSessionType.GAP_RESEARCH,
        title=f"Gap Research: {gap_query}",
        description=f"Conduct targeted research to address gap: {gap_query}",
        gap_query=gap_query,
        priority=priority,
        estimated_duration="2-4 hours",
        context={"gap_query": gap_query, "research_type": "gap_research"},
        tags={"gap_research", "targeted"}
    )


async def create_quality_enhancement_sub_session(
    manager: SubSessionManager,
    parent_session_id: str,
    quality_issues: List[str],
    priority: ResourcePriority = ResourcePriority.MEDIUM
) -> str:
    """Create a quality enhancement sub-session"""
    return await manager.create_sub_session(
        parent_session_id=parent_session_id,
        session_type=SubSessionType.QUALITY_ENHANCEMENT,
        title=f"Quality Enhancement: {len(quality_issues)} issues identified",
        description=f"Address quality issues: {', '.join(quality_issues[:3])}",
        priority=priority,
        estimated_duration="1-3 hours",
        context={"quality_issues": quality_issues, "enhancement_type": "quality_improvement"},
        tags={"quality", "enhancement"}
    )


async def get_sub_session_summary(manager: SubSessionManager, parent_session_id: str) -> Dict[str, Any]:
    """Get summary of sub-sessions for a parent session"""
    children = await manager.get_parent_children(parent_session_id)
    statistics = await manager.get_sub_session_statistics(parent_session_id)

    ready_for_merge = await manager.get_ready_for_merge(parent_session_id)

    return {
        "parent_session_id": parent_session_id,
        "total_sub_sessions": len(children),
        "sub_session_ids": children,
        "ready_for_merge": ready_for_merge,
        "statistics": statistics,
        "hierarchy": await manager.get_session_hierarchy(parent_session_id)
    }