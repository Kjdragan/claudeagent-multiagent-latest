"""
Enhanced KEVIN Session Management System for Multi-Agent Research System v3.2.

This module provides comprehensive session management with KEVIN directory structure,
organized data storage, sub-session coordination, quality assurance integration, and
comprehensive tracking capabilities following the enhanced architectural patterns.

Phase 3.5: Implement Session Management with KEVIN directory structure

Integration with:
- Enhanced Quality Assurance Framework (Phase 3.4)
- Gap Research Enforcement System (Phase 3.3)
- Enhanced Editorial Intelligence System
- Progressive Enhancement Pipeline
- Multi-dimensional Quality Management
"""

import json
import logging
import os
import shutil
import uuid
from datetime import datetime, timedelta
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .workflow_state import WorkflowSession, WorkflowStage, StageStatus


class SessionStatus(Enum):
    """Session status enumeration."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


class DataType(Enum):
    """Data type enumeration for organization."""
    WORKING = "working"
    RESEARCH = "research"
    COMPLETE = "complete"
    AGENT_LOGS = "agent_logs"
    SUB_SESSIONS = "sub_sessions"
    QUALITY_REPORTS = "quality_reports"
    GAP_RESEARCH = "gap_research"


@dataclass
class SessionMetadata:
    """Session metadata for tracking and management."""
    session_id: str
    topic: str
    user_requirements: Dict[str, Any]
    created_at: datetime
    status: SessionStatus
    workflow_stage: WorkflowStage
    parent_session_id: Optional[str] = None
    sub_session_count: int = 0
    quality_score: Optional[float] = None
    processing_statistics: Dict[str, Any] = field(default_factory=dict)
    file_mappings: Dict[str, str] = field(default_factory=dict)
    session_state: Dict[str, Any] = field(default_factory=dict)
    kevin_directory: Optional[str] = None
    last_activity: datetime = field(default_factory=datetime.now)

    # Enhanced Quality Assurance Integration (Phase 3.4)
    quality_assessment_reports: List[Dict[str, Any]] = field(default_factory=list)
    quality_metrics_history: List[Dict[str, Any]] = field(default_factory=list)
    enhancement_cycles_completed: int = 0
    final_quality_level: Optional[str] = None

    # Gap Research Enforcement Integration (Phase 3.3)
    gap_research_enforcement: Dict[str, Any] = field(default_factory=dict)
    compliance_records: List[Dict[str, Any]] = field(default_factory=list)
    enforced_requirements: List[str] = field(default_factory=list)
    compliance_score: Optional[float] = None

    # Editorial Intelligence Integration
    editorial_decisions: List[Dict[str, Any]] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    gap_research_decisions: List[Dict[str, Any]] = field(default_factory=list)
    editorial_recommendations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_activity'] = self.last_activity.isoformat()
        data['status'] = self.status.value
        data['workflow_stage'] = self.workflow_stage.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionMetadata':
        """Create from dictionary."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_activity'] = datetime.fromisoformat(data['last_activity'])
        data['status'] = SessionStatus(data['status'])
        data['workflow_stage'] = WorkflowStage(data['workflow_stage'])
        return cls(**data)


@dataclass
class SubSessionInfo:
    """Information about a sub-session."""
    sub_session_id: str
    parent_session_id: str
    gap_topic: str
    created_at: datetime
    status: SessionStatus
    work_directory: str
    result_files: List[str] = field(default_factory=list)
    integration_status: str = "pending"
    quality_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['status'] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SubSessionInfo':
        """Create from dictionary."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['status'] = SessionStatus(data['status'])
        return cls(**data)


class KevinSessionManager:
    """
    Enhanced Session Management System with KEVIN directory structure.

    This comprehensive session manager provides:
    - Organized data storage with KEVIN directory structure
    - Session lifecycle management with persistence
    - Sub-session coordination for gap research
    - File management and tracking
    - Quality assurance integration
    - Comprehensive audit trails
    """

    def __init__(self, kevin_base_path: Optional[str] = None):
        """
        Initialize the KEVIN Session Manager.

        Args:
            kevin_base_path: Base path for KEVIN directory structure
        """
        self.logger = logging.getLogger(__name__)

        # Set up KEVIN directory structure
        if kevin_base_path:
            self.kevin_base_path = Path(kevin_base_path)
        else:
            # Default to KEVIN directory in project root
            self.kevin_base_path = Path(__file__).parent.parent.parent / "KEVIN"

        self.sessions_path = self.kevin_base_path / "sessions"
        self.logs_path = self.kevin_base_path / "logs"
        self.reports_path = self.kevin_base_path / "reports"
        self.monitoring_path = self.kevin_base_path / "monitoring"

        # Session tracking
        self.active_sessions: Dict[str, SessionMetadata] = {}
        self.session_workflows: Dict[str, WorkflowSession] = {}
        self.sub_sessions: Dict[str, SubSessionInfo] = {}
        self.parent_child_links: Dict[str, List[str]] = {}

        # File management
        self.file_mappings: Dict[str, Dict[str, str]] = {}
        self.file_registry: Dict[str, str] = field(default_factory=dict)

        # Initialize directory structure
        self._initialize_kevin_structure()

        self.logger.info(f"KEVIN Session Manager initialized with base path: {self.kevin_base_path}")

    def _initialize_kevin_structure(self) -> None:
        """Initialize the KEVIN directory structure."""
        directories = [
            self.sessions_path,
            self.logs_path,
            self.reports_path,
            self.monitoring_path
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Ensured directory exists: {directory}")

        self.logger.info("KEVIN directory structure initialized")

    async def create_session(
        self,
        topic: str,
        user_requirements: Dict[str, Any],
        session_id: Optional[str] = None,
        parent_session_id: Optional[str] = None
    ) -> str:
        """
        Create a new research session with KEVIN directory structure.

        Args:
            topic: Research topic
            user_requirements: User requirements and preferences
            session_id: Optional existing session ID
            parent_session_id: Optional parent session ID for sub-sessions

        Returns:
            Session ID
        """
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())

        # Create session metadata
        metadata = SessionMetadata(
            session_id=session_id,
            topic=topic,
            user_requirements=user_requirements,
            created_at=datetime.now(),
            status=SessionStatus.INITIALIZING,
            workflow_stage=WorkflowStage.RESEARCH,
            parent_session_id=parent_session_id,
            kevin_directory=str(self.sessions_path / session_id)
        )

        # Create workflow session
        workflow_session = WorkflowSession(
            session_id=session_id,
            topic=topic,
            user_requirements=user_requirements,
            current_stage=WorkflowStage.RESEARCH,
            overall_status=StageStatus.PENDING
        )

        # Create KEVIN directory structure for session
        await self._create_session_directory_structure(session_id, parent_session_id)

        # Store session information
        self.active_sessions[session_id] = metadata
        self.session_workflows[session_id] = workflow_session

        # Handle parent-child relationships
        if parent_session_id:
            if parent_session_id not in self.parent_child_links:
                self.parent_child_links[parent_session_id] = []
            self.parent_child_links[parent_session_id].append(session_id)

            # Update parent session metadata
            if parent_session_id in self.active_sessions:
                self.active_sessions[parent_session_id].sub_session_count += 1

        # Save session metadata
        await self._save_session_metadata(session_id)

        # Update session status to active
        metadata.status = SessionStatus.ACTIVE
        await self._save_session_metadata(session_id)

        self.logger.info(f"Created session: {session_id} (Topic: {topic})")
        if parent_session_id:
            self.logger.info(f"Created sub-session {session_id} under parent {parent_session_id}")

        return session_id

    async def _create_session_directory_structure(self, session_id: str, parent_session_id: Optional[str] = None) -> None:
        """Create the directory structure for a session."""
        session_path = self.sessions_path / session_id

        # Create main session directories
        directories = [
            session_path / "working",
            session_path / "research",
            session_path / "complete",
            session_path / "agent_logs",
            session_path / "quality_reports",
            session_path / "sub_sessions"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        # Create sub-session specific directories if this is a sub-session
        if parent_session_id:
            sub_session_research_path = session_path / "research" / "gap_research"
            sub_session_research_path.mkdir(exist_ok=True)

        self.logger.debug(f"Created directory structure for session: {session_id}")

    async def update_session_status(
        self,
        session_id: str,
        status: SessionStatus,
        workflow_stage: Optional[WorkflowStage] = None,
        metadata_updates: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update session status and related information.

        Args:
            session_id: Session ID
            status: New session status
            workflow_stage: Optional workflow stage update
            metadata_updates: Optional metadata updates
        """
        if session_id not in self.active_sessions:
            self.logger.warning(f"Attempted to update non-existent session: {session_id}")
            return

        metadata = self.active_sessions[session_id]
        metadata.status = status
        metadata.last_activity = datetime.now()

        if workflow_stage:
            metadata.workflow_stage = workflow_stage

        if metadata_updates:
            for key, value in metadata_updates.items():
                if hasattr(metadata, key):
                    setattr(metadata, key, value)
                else:
                    metadata.session_state[key] = value

        # Update workflow session if available
        if session_id in self.session_workflows and workflow_stage:
            workflow_session = self.session_workflows[session_id]
            workflow_session.current_stage = workflow_stage

            # Update stage status based on session status
            if status == SessionStatus.COMPLETED:
                workflow_session.overall_status = StageStatus.COMPLETED
                workflow_session.end_time = datetime.now()
            elif status == SessionStatus.FAILED:
                workflow_session.overall_status = StageStatus.FAILED
                workflow_session.end_time = datetime.now()

        # Save updated metadata
        await self._save_session_metadata(session_id)

        self.logger.info(f"Updated session {session_id} status to {status.value}")

    async def create_sub_session(
        self,
        gap_topic: str,
        parent_session_id: str,
        sub_session_id: Optional[str] = None
    ) -> str:
        """
        Create a sub-session for gap research.

        Args:
            gap_topic: Research gap topic
            parent_session_id: Parent session ID
            sub_session_id: Optional sub-session ID

        Returns:
            Sub-session ID
        """
        if parent_session_id not in self.active_sessions:
            raise ValueError(f"Parent session {parent_session_id} does not exist")

        # Generate sub-session ID if not provided
        if not sub_session_id:
            sub_session_id = str(uuid.uuid4())

        # Create sub-session metadata
        sub_metadata = SessionMetadata(
            session_id=sub_session_id,
            topic=gap_topic,
            user_requirements={"gap_research": True, "parent_topic": self.active_sessions[parent_session_id].topic},
            created_at=datetime.now(),
            status=SessionStatus.INITIALIZING,
            workflow_stage=WorkflowStage.RESEARCH,
            parent_session_id=parent_session_id,
            kevin_directory=str(self.sessions_path / sub_session_id)
        )

        # Create sub-session info
        sub_session_info = SubSessionInfo(
            sub_session_id=sub_session_id,
            parent_session_id=parent_session_id,
            gap_topic=gap_topic,
            created_at=datetime.now(),
            status=SessionStatus.ACTIVE,
            work_directory=str(self.sessions_path / sub_session_id)
        )

        # Create sub-session directory structure
        await self._create_session_directory_structure(sub_session_id, parent_session_id)

        # Store sub-session information
        self.active_sessions[sub_session_id] = sub_metadata
        self.sub_sessions[sub_session_id] = sub_session_info

        # Update parent-child links
        if parent_session_id not in self.parent_child_links:
            self.parent_child_links[parent_session_id] = []
        self.parent_child_links[parent_session_id].append(sub_session_id)

        # Update parent session metadata
        parent_metadata = self.active_sessions[parent_session_id]
        parent_metadata.sub_session_count += 1
        parent_metadata.last_activity = datetime.now()

        # Save metadata
        await self._save_session_metadata(sub_session_id)
        await self._save_session_metadata(parent_session_id)

        self.logger.info(f"Created sub-session: {sub_session_id} for gap topic: {gap_topic}")

        return sub_session_id

    async def store_session_file(
        self,
        session_id: str,
        data_type: DataType,
        content: str,
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a file in the appropriate KEVIN session directory.

        Args:
            session_id: Session ID
            data_type: Type of data being stored
            content: File content
            filename: Optional filename (auto-generated if not provided)
            metadata: Optional metadata for the file

        Returns:
            File path
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} does not exist")

        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{data_type.value.upper()}_{timestamp}.md"

        # Determine directory based on data type
        session_path = self.sessions_path / session_id

        if data_type == DataType.WORKING:
            file_dir = session_path / "working"
        elif data_type == DataType.RESEARCH:
            file_dir = session_path / "research"
        elif data_type == DataType.COMPLETE:
            file_dir = session_path / "complete"
        elif data_type == DataType.AGENT_LOGS:
            file_dir = session_path / "agent_logs"
        elif data_type == DataType.QUALITY_REPORTS:
            file_dir = session_path / "quality_reports"
        elif data_type == DataType.SUB_SESSIONS:
            file_dir = session_path / "sub_sessions"
        else:
            file_dir = session_path

        # Ensure directory exists
        file_dir.mkdir(exist_ok=True)

        # Create file path
        file_path = file_dir / filename

        # Write content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # Create metadata file
        if metadata:
            metadata_file = file_path.with_suffix('.json')
            metadata_content = {
                "session_id": session_id,
                "data_type": data_type.value,
                "created_at": datetime.now().isoformat(),
                "file_size": len(content),
                "metadata": metadata
            }

            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata_content, f, indent=2)

        # Update file mappings
        if session_id not in self.file_mappings:
            self.file_mappings[session_id] = {}
        self.file_mappings[session_id][filename] = str(file_path)

        # Update session metadata
        session_metadata = self.active_sessions[session_id]
        session_metadata.file_mappings[filename] = str(file_path)
        session_metadata.last_activity = datetime.now()

        # Save updated metadata
        await self._save_session_metadata(session_id)

        self.logger.info(f"Stored file {filename} for session {session_id} in {data_type.value}")

        return str(file_path)

    async def get_session_file(
        self,
        session_id: str,
        filename: str,
        data_type: Optional[DataType] = None
    ) -> Optional[str]:
        """
        Retrieve a file from the session directory.

        Args:
            session_id: Session ID
            filename: Filename to retrieve
            data_type: Optional data type to narrow search

        Returns:
            File content or None if not found
        """
        if session_id not in self.active_sessions:
            self.logger.warning(f"Session {session_id} does not exist")
            return None

        # Check file mappings first
        if session_id in self.file_mappings and filename in self.file_mappings[session_id]:
            file_path = Path(self.file_mappings[session_id][filename])
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()

        # Search in appropriate directory if data type is specified
        if data_type:
            session_path = self.sessions_path / session_id

            if data_type == DataType.WORKING:
                search_dir = session_path / "working"
            elif data_type == DataType.RESEARCH:
                search_dir = session_path / "research"
            elif data_type == DataType.COMPLETE:
                search_dir = session_path / "complete"
            elif data_type == DataType.AGENT_LOGS:
                search_dir = session_path / "agent_logs"
            elif data_type == DataType.QUALITY_REPORTS:
                search_dir = session_path / "quality_reports"
            elif data_type == DataType.SUB_SESSIONS:
                search_dir = session_path / "sub_sessions"
            else:
                search_dir = session_path

            file_path = search_dir / filename
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()

        # Search all session directories
        session_path = self.sessions_path / session_id
        for root, dirs, files in os.walk(session_path):
            if filename in files:
                file_path = Path(root) / filename
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return f.read()
                except (UnicodeDecodeError, IOError):
                    continue

        self.logger.warning(f"File {filename} not found for session {session_id}")
        return None

    async def get_session_files(
        self,
        session_id: str,
        data_type: Optional[DataType] = None,
        include_metadata: bool = False
    ) -> Dict[str, Any]:
        """
        Get all files for a session, optionally filtered by data type.

        Args:
            session_id: Session ID
            data_type: Optional data type filter
            include_metadata: Whether to include file metadata

        Returns:
            Dictionary of files with their content and metadata
        """
        if session_id not in self.active_sessions:
            return {}

        files = {}
        session_path = self.sessions_path / session_id

        # Determine search directory
        if data_type:
            if data_type == DataType.WORKING:
                search_dir = session_path / "working"
            elif data_type == DataType.RESEARCH:
                search_dir = session_path / "research"
            elif data_type == DataType.COMPLETE:
                search_dir = session_path / "complete"
            elif data_type == DataType.AGENT_LOGS:
                search_dir = session_path / "agent_logs"
            elif data_type == DataType.QUALITY_REPORTS:
                search_dir = session_path / "quality_reports"
            elif data_type == DataType.SUB_SESSIONS:
                search_dir = session_path / "sub_sessions"
            else:
                search_dir = session_path

            search_dirs = [search_dir]
        else:
            search_dirs = [
                session_path / "working",
                session_path / "research",
                session_path / "complete",
                session_path / "agent_logs",
                session_path / "quality_reports",
                session_path / "sub_sessions"
            ]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue

            for file_path in search_dir.iterdir():
                if file_path.is_file() and not file_path.name.endswith('.json'):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        file_info = {"content": content, "path": str(file_path)}

                        if include_metadata:
                            metadata_file = file_path.with_suffix('.json')
                            if metadata_file.exists():
                                with open(metadata_file, 'r', encoding='utf-8') as f:
                                    file_info["metadata"] = json.load(f)

                        files[file_path.name] = file_info

                    except (UnicodeDecodeError, IOError) as e:
                        self.logger.warning(f"Could not read file {file_path}: {e}")
                        continue

        return files

    async def store_quality_assessment_report(
        self,
        session_id: str,
        assessment_report: Dict[str, Any],
        stage: Optional[str] = None
    ) -> str:
        """
        Store quality assurance report in KEVIN structure.

        Args:
            session_id: Session ID
            assessment_report: Quality assessment report from QA framework
            stage: Optional workflow stage identifier

        Returns:
            File path
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} does not exist")

        # Create QA report content with enhanced metadata
        qa_content = {
            "session_id": session_id,
            "assessment_timestamp": assessment_report.get("timestamp", datetime.now().isoformat()),
            "overall_quality_score": assessment_report.get("overall_quality_score"),
            "quality_level": assessment_report.get("quality_level"),
            "enhancement_cycles_completed": assessment_report.get("enhancement_cycles_completed", 0),
            "total_improvement": assessment_report.get("total_improvement", 0),
            "gate_compliance_rate": assessment_report.get("gate_compliance_rate"),
            "recommendations": assessment_report.get("recommendations", []),
            "issues_identified": assessment_report.get("issues_identified", []),
            "performance_summary": assessment_report.get("performance_summary", {}),
            "compliance_status": assessment_report.get("compliance_status", {}),
            "stage": stage,
            "kevin_integration": {
                "stored_at": datetime.now().isoformat(),
                "session_phase": self.active_sessions[session_id].workflow_stage.value,
                "quality_enhancement_enabled": self.active_sessions[session_id].user_requirements.get("progressive_enhancement", False)
            }
        }

        # Store in quality reports directory
        qa_content_str = json.dumps(qa_content, indent=2)
        filename = f"quality_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        file_path = await self.store_session_file(
            session_id, DataType.QUALITY_REPORTS, qa_content_str, filename
        )

        # Update session metadata with QA information
        metadata = self.active_sessions[session_id]
        metadata.quality_assessment_reports.append(qa_content)
        metadata.quality_score = assessment_report.get("overall_quality_score")
        metadata.final_quality_level = assessment_report.get("quality_level")
        metadata.last_activity = datetime.now()

        # Update enhancement cycles if available
        if "enhancement_cycles_completed" in assessment_report:
            metadata.enhancement_cycles_completed = assessment_report["enhancement_cycles_completed"]

        await self._save_session_metadata(session_id)

        self.logger.info(f"Stored quality assessment report for session {session_id} - Quality Score: {metadata.quality_score}")

        return file_path

    async def store_gap_research_enforcement_report(
        self,
        session_id: str,
        enforcement_report: Dict[str, Any]
    ) -> str:
        """
        Store gap research enforcement report in KEVIN structure.

        Args:
            session_id: Session ID
            enforcement_report: Gap research enforcement report

        Returns:
            File path
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} does not exist")

        # Create enforcement report content
        enforcement_content = {
            "session_id": session_id,
            "enforcement_timestamp": datetime.now().isoformat(),
            "overall_compliance_score": enforcement_report.get("overall_compliance_score"),
            "requirements_met": enforcement_report.get("requirements_met", []),
            "requirements_failed": enforcement_report.get("requirements_failed", []),
            "enforcement_actions": enforcement_report.get("enforcement_actions", []),
            "audit_trail": enforcement_report.get("audit_trail", []),
            "compliance_records": enforcement_report.get("compliance_records", []),
            "kevin_integration": {
                "stored_at": datetime.now().isoformat(),
                "session_phase": self.active_sessions[session_id].workflow_stage.value,
                "gap_research_enabled": self.active_sessions[session_id].user_requirements.get("gap_research_enabled", False)
            }
        }

        # Store in research directory
        enforcement_content_str = json.dumps(enforcement_content, indent=2)
        filename = f"gap_research_enforcement_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        file_path = await self.store_session_file(
            session_id, DataType.GAP_RESEARCH, enforcement_content_str, filename
        )

        # Update session metadata with enforcement information
        metadata = self.active_sessions[session_id]
        metadata.gap_research_enforcement = enforcement_report
        metadata.compliance_records = enforcement_report.get("compliance_records", [])
        metadata.enforced_requirements = enforcement_report.get("requirements_met", [])
        metadata.compliance_score = enforcement_report.get("overall_compliance_score")
        metadata.last_activity = datetime.now()

        await self._save_session_metadata(session_id)

        self.logger.info(f"Stored gap research enforcement report for session {session_id} - Compliance Score: {metadata.compliance_score}")

        return file_path

    async def store_editorial_decision_report(
        self,
        session_id: str,
        editorial_report: Dict[str, Any]
    ) -> str:
        """
        Store editorial decision report in KEVIN structure.

        Args:
            session_id: Session ID
            editorial_report: Editorial decision and analysis report

        Returns:
            File path
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} does not exist")

        # Create editorial report content
        editorial_content = {
            "session_id": session_id,
            "decision_timestamp": editorial_report.get("timestamp", datetime.now().isoformat()),
            "gap_research_decision": editorial_report.get("gap_research_decision", {}),
            "confidence_scores": editorial_report.get("confidence_scores", {}),
            "editorial_recommendations": editorial_report.get("editorial_recommendations", []),
            "corpus_analysis": editorial_report.get("corpus_analysis", {}),
            "kevin_integration": {
                "stored_at": datetime.now().isoformat(),
                "session_phase": self.active_sessions[session_id].workflow_stage.value,
                "editorial_intelligence_enabled": self.active_sessions[session_id].user_requirements.get("editorial_intelligence_enabled", False)
            }
        }

        # Store in working directory for editorial analysis
        editorial_content_str = json.dumps(editorial_content, indent=2)
        filename = f"editorial_decision_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        file_path = await self.store_session_file(
            session_id, DataType.WORKING, editorial_content_str, filename
        )

        # Update session metadata with editorial information
        metadata = self.active_sessions[session_id]
        metadata.editorial_decisions.append(editorial_content)
        metadata.confidence_scores = editorial_report.get("confidence_scores", {})
        metadata.gap_research_decisions.append(editorial_report.get("gap_research_decision", {}))
        metadata.editorial_recommendations = editorial_report.get("editorial_recommendations", [])
        metadata.last_activity = datetime.now()

        await self._save_session_metadata(session_id)

        self.logger.info(f"Stored editorial decision report for session {session_id}")

        return file_path

    async def integrate_sub_session_results(
        self,
        parent_session_id: str,
        integration_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhanced sub-session result integration with quality tracking.

        Args:
            parent_session_id: Parent session ID
            integration_data: Data from sub-sessions

        Returns:
            Enhanced integration results with quality metrics
        """
        if parent_session_id not in self.active_sessions:
            raise ValueError(f"Parent session {parent_session_id} does not exist")

        if parent_session_id not in self.parent_child_links:
            return {"error": "No sub-sessions found for parent session"}

        child_session_ids = self.parent_child_links[parent_session_id]
        integrated_results = []
        quality_analysis = []
        compliance_analysis = []

        for child_id in child_session_ids:
            if child_id in self.sub_sessions:
                child_info = self.sub_sessions[child_id]

                # Get sub-session files
                sub_files = await self.get_session_files(child_id)

                # Enhanced quality analysis
                quality_score = integration_data.get(child_id, {}).get("quality_score", 0.0)
                compliance_score = integration_data.get(child_id, {}).get("compliance_score", 1.0)

                quality_analysis.append({
                    "sub_session_id": child_id,
                    "gap_topic": child_info.gap_topic,
                    "quality_score": quality_score,
                    "compliance_score": compliance_score,
                    "file_count": len(sub_files),
                    "status": child_info.status.value,
                    "integration_timestamp": datetime.now().isoformat()
                })

                integrated_results.append({
                    "sub_session_id": child_id,
                    "gap_topic": child_info.gap_topic,
                    "sub_session_info": child_info.to_dict(),
                    "files": sub_files,
                    "integration_data": integration_data.get(child_id, {}),
                    "integration_status": "integrated",
                    "quality_metrics": {
                        "quality_score": quality_score,
                        "compliance_score": compliance_score,
                        "overall_score": (quality_score + compliance_score) / 2
                    }
                })

                # Update sub-session status and metrics
                child_info.status = SessionStatus.COMPLETED
                child_info.integration_status = "completed"
                child_info.quality_score = quality_score

        # Enhanced integration quality calculation
        integration_quality = self._calculate_enhanced_integration_quality(quality_analysis, compliance_analysis)

        # Store enhanced integration results
        integration_content = {
            "parent_session_id": parent_session_id,
            "integration_timestamp": datetime.now().isoformat(),
            "integrated_results": integrated_results,
            "quality_analysis": quality_analysis,
            "compliance_analysis": compliance_analysis,
            "integration_quality": integration_quality,
            "total_sub_sessions": len(child_session_ids),
            "successful_integrations": len([r for r in integrated_results if r["integration_status"] == "integrated"]),
            "average_quality_score": sum(item["quality_score"] for item in quality_analysis) / len(quality_analysis) if quality_analysis else 0,
            "average_compliance_score": sum(item["compliance_score"] for item in compliance_analysis) / len(compliance_analysis) if compliance_analysis else 0.0,
            "kevin_enhancement": {
                "quality_assurance_integration": True,
                "gap_research_enforcement_tracking": True,
                "editorial_intelligence_coordination": True
            }
        }

        await self.store_session_file(
            parent_session_id,
            DataType.SUB_SESSIONS,
            json.dumps(integration_content, indent=2),
            f"ENHANCED_SUB_SESSION_INTEGRATION_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        # Update parent session metadata with enhanced tracking
        parent_metadata = self.active_sessions[parent_session_id]
        parent_metadata.last_activity = datetime.now()
        parent_metadata.session_state["sub_session_integration"] = {
            "completed_at": datetime.now().isoformat(),
            "integrated_count": len(integrated_results),
            "integration_quality": integration_quality,
            "average_quality_score": integration_content["average_quality_score"],
            "average_compliance_score": integration_content["average_compliance_score"],
            "enhanced_integration": True
        }

        # Update quality metrics if available
        if integration_content["average_quality_score"] > 0:
            parent_metadata.quality_score = integration_content["average_quality_score"]

        await self._save_session_metadata(parent_session_id)

        self.logger.info(f"Enhanced integration of {len(integrated_results)} sub-sessions into parent {parent_session_id}")
        self.logger.info(f"Average Quality Score: {integration_content['average_quality_score']:.2f}, Average Compliance Score: {integration_content['average_compliance_score']:.2f}")

        return {
            "parent_session_id": parent_session_id,
            "integrated_results": integrated_results,
            "total_sub_sessions": len(child_session_ids),
            "successful_integrations": len([r for r in integrated_results if r["integration_status"] == "integrated"]),
            "quality_analysis": quality_analysis,
            "compliance_analysis": compliance_analysis,
            "integration_quality": integration_quality,
            "average_quality_score": integration_content["average_quality_score"],
            "average_compliance_score": integration_content["average_compliance_score"],
            "enhanced_integration": True
        }

    def _calculate_integration_quality(self, quality_analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate integration quality metrics."""
        if not quality_analysis:
            return {"overall_score": 0.0, "average_quality": 0.0, "completion_rate": 0.0}

        total_quality = sum(item["quality_score"] for item in quality_analysis)
        average_quality = total_quality / len(quality_analysis)
        completion_rate = len([item for item in quality_analysis if item["status"] == "completed"]) / len(quality_analysis)

        return {
            "overall_score": average_quality * completion_rate,
            "average_quality": average_quality,
            "completion_rate": completion_rate,
            "total_sub_sessions": len(quality_analysis),
            "completed_sub_sessions": len([item for item in quality_analysis if item["status"] == "completed"])
        }

    async def _save_session_metadata(self, session_id: str) -> None:
        """Save session metadata to file."""
        if session_id not in self.active_sessions:
            return

        metadata = self.active_sessions[session_id]
        session_path = self.sessions_path / session_id

        # Ensure session directory exists
        session_path.mkdir(exist_ok=True)

        # Save metadata
        metadata_file = session_path / "session_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata.to_dict(), f, indent=2)

    async def load_session_metadata(self, session_id: str) -> Optional[SessionMetadata]:
        """Load session metadata from file."""
        session_path = self.sessions_path / session_id
        metadata_file = session_path / "session_metadata.json"

        if not metadata_file.exists():
            return None

        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return SessionMetadata.from_dict(data)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.error(f"Could not load metadata for session {session_id}: {e}")
            return None

    async def get_active_sessions(self, status_filter: Optional[SessionStatus] = None) -> List[SessionMetadata]:
        """Get list of active sessions, optionally filtered by status."""
        sessions = list(self.active_sessions.values())

        if status_filter:
            sessions = [s for s in sessions if s.status == status_filter]

        return sessions

    async def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a comprehensive summary of a session."""
        if session_id not in self.active_sessions:
            return None

        metadata = self.active_sessions[session_id]

        # Get file statistics
        files = await self.get_session_files(session_id)
        file_stats = {
            "total_files": len(files),
            "working_files": len([f for f in files if "working" in str(f.get("path", ""))]),
            "research_files": len([f for f in files if "research" in str(f.get("path", ""))]),
            "complete_files": len([f for f in files if "complete" in str(f.get("path", ""))]),
            "agent_logs": len([f for f in files if "agent_logs" in str(f.get("path", ""))]),
            "quality_reports": len([f for f in files if "quality_reports" in str(f.get("path", ""))])
        }

        # Get sub-session information
        sub_session_info = []
        if session_id in self.parent_child_links:
            for child_id in self.parent_child_links[session_id]:
                if child_id in self.sub_sessions:
                    sub_session_info.append(self.sub_sessions[child_id].to_dict())

        return {
            "session_metadata": metadata.to_dict(),
            "file_statistics": file_stats,
            "sub_sessions": sub_session_info,
            "workflow_stage": metadata.workflow_stage.value,
            "session_duration": (datetime.now() - metadata.created_at).total_seconds(),
            "last_activity": metadata.last_activity.isoformat()
        }

    async def archive_session(self, session_id: str) -> bool:
        """Archive a completed session."""
        if session_id not in self.active_sessions:
            self.logger.warning(f"Session {session_id} does not exist")
            return False

        metadata = self.active_sessions[session_id]

        if metadata.status != SessionStatus.COMPLETED:
            self.logger.warning(f"Cannot archive session {session_id} - status is {metadata.status.value}")
            return False

        # Move session to archived directory
        archive_path = self.sessions_path / "archived"
        archive_path.mkdir(exist_ok=True)

        current_path = self.sessions_path / session_id
        archived_session_path = archive_path / session_id

        try:
            # Move session directory
            if current_path.exists():
                shutil.move(str(current_path), str(archived_session_path))

            # Update metadata
            metadata.status = SessionStatus.ARCHIVED
            await self._save_session_metadata(session_id)

            # Remove from active sessions
            del self.active_sessions[session_id]
            if session_id in self.session_workflows:
                del self.session_workflows[session_id]

            self.logger.info(f"Archived session: {session_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to archive session {session_id}: {e}")
            return False

    async def cleanup_expired_sessions(self, max_age_hours: int = 24) -> int:
        """Clean up expired sessions."""
        current_time = datetime.now()
        expired_sessions = []

        for session_id, metadata in self.active_sessions.items():
            age_hours = (current_time - metadata.created_at).total_seconds() / 3600

            if age_hours > max_age_hours and metadata.status in [SessionStatus.COMPLETED, SessionStatus.FAILED]:
                expired_sessions.append(session_id)

        archived_count = 0
        for session_id in expired_sessions:
            if await self.archive_session(session_id):
                archived_count += 1

        self.logger.info(f"Cleaned up {archived_count} expired sessions")
        return archived_count

    async def get_kevin_statistics(self) -> Dict[str, Any]:
        """Get comprehensive KEVIN system statistics."""
        total_sessions = len(self.active_sessions)

        status_counts = {}
        stage_counts = {}

        for metadata in self.active_sessions.values():
            status_counts[metadata.status.value] = status_counts.get(metadata.status.value, 0) + 1
            stage_counts[metadata.workflow_stage.value] = stage_counts.get(metadata.workflow_stage.value, 0) + 1

        # File statistics
        total_files = 0
        total_size = 0

        for session_id in self.active_sessions:
            files = await self.get_session_files(session_id)
            total_files += len(files)

            for file_info in files.values():
                total_size += len(file_info.get("content", ""))

        return {
            "total_sessions": total_sessions,
            "status_distribution": status_counts,
            "stage_distribution": stage_counts,
            "total_files": total_files,
            "total_size_bytes": total_size,
            "sub_sessions": len(self.sub_sessions),
            "parent_sessions": len(self.parent_child_links),
            "kevin_directory": str(self.kevin_base_path),
            "sessions_directory": str(self.sessions_path),
            "last_updated": datetime.now().isoformat()
        }

    async def _calculate_enhanced_integration_quality(self, quality_analysis: list, compliance_analysis: list) -> float:
        """Calculate enhanced integration quality with multi-dimensional assessment"""
        try:
            quality_components = []

            # Quality assessment component (40% weight)
            if quality_analysis:
                avg_quality_score = sum(item.get("quality_score", 0) for item in quality_analysis) / len(quality_analysis)
                completion_rate = len([item for item in quality_analysis if item.get("status") == "completed"]) / len(quality_analysis)
                quality_component = (avg_quality_score * 0.7) + (completion_rate * 0.3)
                quality_components.append(("quality", quality_component, 0.4))

            # Compliance assessment component (35% weight)
            if compliance_analysis:
                avg_compliance_score = sum(item.get("compliance_score", 0) for item in compliance_analysis) / len(compliance_analysis)
                enforcement_success_rate = len([item for item in compliance_analysis if item.get("enforcement_success", False)]) / len(compliance_analysis)
                compliance_component = (avg_compliance_score * 0.6) + (enforcement_success_rate * 0.4)
                quality_components.append(("compliance", compliance_component, 0.35))

            # Integration efficiency component (15% weight)
            integration_metrics = self._calculate_integration_efficiency(quality_analysis, compliance_analysis)
            quality_components.append(("efficiency", integration_metrics, 0.15))

            # Data consistency component (10% weight)
            consistency_score = self._calculate_data_consistency(quality_analysis, compliance_analysis)
            quality_components.append(("consistency", consistency_score, 0.10))

            # Calculate weighted overall score
            overall_score = 0.0
            total_weight = 0.0

            for component_name, component_score, weight in quality_components:
                overall_score += component_score * weight
                total_weight += weight

            # Normalize if total weight is not exactly 1.0
            if total_weight > 0:
                overall_score = overall_score / total_weight

            return min(max(overall_score, 0.0), 1.0)  # Ensure score is between 0 and 1

        except Exception as e:
            self.logger.error(f"Error calculating enhanced integration quality: {e}")
            return 0.5  # Return moderate default score on error

    def _calculate_integration_efficiency(self, quality_analysis: list, compliance_analysis: list) -> float:
        """Calculate integration efficiency metrics"""
        try:
            # Time efficiency (how quickly sessions are integrated)
            time_efficiency = 0.8  # Placeholder for actual time-based calculation

            # Resource efficiency (how well resources are utilized)
            resource_efficiency = 0.85  # Placeholder for resource utilization analysis

            # Success efficiency (success rate of integrations)
            success_efficiency = 0.9  # Placeholder for success rate calculation

            # Combine efficiency metrics
            return (time_efficiency * 0.3) + (resource_efficiency * 0.3) + (success_efficiency * 0.4)

        except Exception as e:
            self.logger.error(f"Error calculating integration efficiency: {e}")
            return 0.7  # Return moderate default efficiency

    def _calculate_data_consistency(self, quality_analysis: list, compliance_analysis: list) -> float:
        """Calculate data consistency across integrated components"""
        try:
            # Check for consistency in quality scores
            if quality_analysis:
                quality_scores = [item.get("quality_score", 0) for item in quality_analysis]
                quality_variance = sum((score - sum(quality_scores)/len(quality_scores))**2 for score in quality_scores) / len(quality_scores)
                quality_consistency = 1.0 - min(quality_variance, 1.0)  # Lower variance = higher consistency
            else:
                quality_consistency = 0.8

            # Check for consistency in compliance scores
            if compliance_analysis:
                compliance_scores = [item.get("compliance_score", 0) for item in compliance_analysis]
                compliance_variance = sum((score - sum(compliance_scores)/len(compliance_scores))**2 for score in compliance_scores) / len(compliance_scores)
                compliance_consistency = 1.0 - min(compliance_variance, 1.0)
            else:
                compliance_consistency = 0.8

            # Check data integrity consistency
            data_integrity_consistency = 0.85  # Placeholder for data integrity checks

            # Combine consistency metrics
            return (quality_consistency * 0.4) + (compliance_consistency * 0.3) + (data_integrity_consistency * 0.3)

        except Exception as e:
            self.logger.error(f"Error calculating data consistency: {e}")
            return 0.75  # Return moderate default consistency


# Convenience function for quick session creation
async def create_kevin_session(
    topic: str,
    user_requirements: Dict[str, Any],
    kevin_base_path: Optional[str] = None
) -> str:
    """
    Quick session creation function.

    Args:
        topic: Research topic
        user_requirements: User requirements
        kevin_base_path: Optional KEVIN base path

    Returns:
        Session ID
    """
    manager = KevinSessionManager(kevin_base_path)
    return await manager.create_session(topic, user_requirements)