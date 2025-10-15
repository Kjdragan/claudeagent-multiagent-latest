"""
Enhanced Session State Management for Multi-Agent Research System

Provides comprehensive session state management with real-time updates,
workflow stage tracking, and metadata persistence for the multi-agent research system.

Key Features:
- Real-time session state updates
- Workflow stage progression tracking
- File-based session metadata persistence
- Integration with KEVIN directory structure
- Comprehensive session lifecycle management
- Multi-dimensional session state tracking
"""

import json
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
from uuid import uuid4

logger = logging.getLogger(__name__)


class EnhancedSessionStateManager:
    """
    Enhanced session state manager with real-time updates and workflow tracking.

    Provides comprehensive session management with file-based persistence,
    workflow stage tracking, and integration with the KEVIN directory structure.
    """

    def __init__(self, base_dir: str = "KEVIN/sessions"):
        """
        Initialize the enhanced session state manager.

        Args:
            base_dir: Base directory for session storage
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._active_sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

        logger.info(f"EnhancedSessionStateManager initialized with base directory: {base_dir}")

    async def create_session(self, initial_query: str, user_requirements: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new enhanced session with comprehensive metadata.

        Args:
            initial_query: The initial research query
            user_requirements: Optional user requirements dictionary

        Returns:
            New session ID
        """
        async with self._lock:
            session_id = str(uuid4())

            # Create session directory structure
            session_dir = self.base_dir / session_id
            await self._create_session_directory_structure(session_dir)

            # Initialize session metadata
            session_metadata = {
                "session_id": session_id,
                "initial_query": initial_query,
                "user_requirements": user_requirements or {},
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "initialized",
                "version": "3.2",
                "editorial_workflow_enabled": True,
                "stages": {
                    "target_generation": {"status": "pending"},
                    "initial_research": {"status": "pending"},
                    "first_draft": {"status": "pending"},
                    "enhanced_editorial_analysis": {"status": "pending"},
                    "gap_research_decision": {"status": "pending"},
                    "gap_research_execution": {"status": "pending"},
                    "editorial_recommendations": {"status": "pending"},
                    "workflow_integration": {"status": "pending"},
                    "final_report": {"status": "pending"}
                },
                "sub_sessions": [],
                "editorial_decisions": {},
                "gap_research_decisions": {},
                "confidence_scores": {},
                "file_mappings": {},
                "research_metrics": {
                    "total_urls_processed": 0,
                    "successful_scrapes": 0,
                    "successful_cleans": 0,
                    "useful_content_count": 0,
                    "editorial_quality_score": 0.0,
                    "confidence_scores_calculated": 0
                },
                "workflow_integration": {
                    "orchestrator_integration": False,
                    "hooks_integration": False,
                    "quality_integration": False,
                    "sdk_integration": False
                }
            }

            # Store session metadata
            await self._save_session_metadata(session_dir, session_metadata)
            self._active_sessions[session_id] = session_metadata

            logger.info(f"Created enhanced session: {session_id} for query: {initial_query[:100]}...")
            return session_id

    async def update_stage_status(self, session_id: str, stage: str, status: str,
                                 metadata: Optional[Dict[str, Any]] = None):
        """
        Update the status of a workflow stage with real-time persistence.

        Args:
            session_id: Session ID to update
            stage: Workflow stage name
            status: New status ("pending", "running", "completed", "failed", "skipped")
            metadata: Optional additional metadata for the stage
        """
        async with self._lock:
            session_dir = self.base_dir / session_id

            if not session_dir.exists():
                logger.error(f"Session directory not found: {session_id}")
                return

            # Load current metadata
            session_metadata = await self._load_session_metadata(session_dir)
            if not session_metadata:
                logger.error(f"Failed to load session metadata: {session_id}")
                return

            # Update stage status
            if stage not in session_metadata["stages"]:
                logger.warning(f"Unknown stage: {stage} for session: {session_id}")
                return

            stage_info = session_metadata["stages"][stage]
            stage_info["status"] = status

            # Add timestamp
            if status == "running":
                stage_info["started_at"] = datetime.now(timezone.utc).isoformat()
            elif status in ["completed", "failed", "skipped"]:
                stage_info["completed_at"] = datetime.now(timezone.utc).isoformat()
                if "started_at" in stage_info:
                    start_time = datetime.fromisoformat(stage_info["started_at"])
                    end_time = datetime.fromisoformat(stage_info["completed_at"])
                    stage_info["duration_seconds"] = (end_time - start_time).total_seconds()

            # Add additional metadata
            if metadata:
                stage_info.update(metadata)

            # Calculate overall progress
            session_metadata = await self._calculate_session_progress(session_metadata)

            # Update current stage
            if status == "running":
                session_metadata["current_stage"] = stage
            elif status == "completed":
                # Determine next stage
                current_stage_index = list(session_metadata["stages"].keys()).index(stage)
                if current_stage_index < len(session_metadata["stages"]) - 1:
                    next_stage = list(session_metadata["stages"].keys())[current_stage_index + 1]
                    session_metadata["current_stage"] = next_stage

            # Save updated metadata
            await self._save_session_metadata(session_dir, session_metadata)
            self._active_sessions[session_id] = session_metadata

            logger.info(f"Updated stage {stage} to {status} for session {session_id}")

    async def add_file_mapping(self, session_id: str, file_type: str, file_path: str,
                              metadata: Optional[Dict[str, Any]] = None):
        """
        Add a file mapping to the session metadata.

        Args:
            session_id: Session ID
            file_type: Type of file (e.g., "research_workproduct", "final_report")
            file_path: Relative path to the file
            metadata: Optional file metadata
        """
        async with self._lock:
            session_dir = self.base_dir / session_id

            if not session_dir.exists():
                logger.error(f"Session directory not found: {session_id}")
                return

            # Load current metadata
            session_metadata = await self._load_session_metadata(session_dir)
            if not session_metadata:
                logger.error(f"Failed to load session metadata: {session_id}")
                return

            # Add file mapping
            file_info = {
                "file_path": file_path,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "file_type": file_type
            }

            if metadata:
                file_info.update(metadata)

            if file_type not in session_metadata["file_mappings"]:
                session_metadata["file_mappings"][file_type] = []

            session_metadata["file_mappings"][file_type].append(file_info)

            # Save updated metadata
            await self._save_session_metadata(session_dir, session_metadata)
            self._active_sessions[session_id] = session_metadata

            logger.info(f"Added file mapping {file_type}: {file_path} for session {session_id}")

    async def update_research_metrics(self, session_id: str, metrics: Dict[str, Any]):
        """
        Update research metrics for the session.

        Args:
            session_id: Session ID
            metrics: Metrics to update
        """
        async with self._lock:
            session_dir = self.base_dir / session_id

            if not session_dir.exists():
                logger.error(f"Session directory not found: {session_id}")
                return

            # Load current metadata
            session_metadata = await self._load_session_metadata(session_dir)
            if not session_metadata:
                logger.error(f"Failed to load session metadata: {session_id}")
                return

            # Update metrics
            session_metadata["research_metrics"].update(metrics)

            # Save updated metadata
            await self._save_session_metadata(session_dir, session_metadata)
            self._active_sessions[session_id] = session_metadata

            logger.debug(f"Updated research metrics for session {session_id}")

    async def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive session status.

        Args:
            session_id: Session ID

        Returns:
            Session status dictionary or None if not found
        """
        async with self._lock:
            session_dir = self.base_dir / session_id

            if not session_dir.exists():
                return None

            # Load metadata
            session_metadata = await self._load_session_metadata(session_dir)
            if not session_metadata:
                return None

            # Calculate progress if not already calculated
            if "progress_percentage" not in session_metadata:
                session_metadata = await self._calculate_session_progress(session_metadata)

            return session_metadata

    async def get_session_file_path(self, session_id: str) -> Path:
        """
        Get the session metadata file path.

        Args:
            session_id: Session ID

        Returns:
            Path to session metadata file
        """
        return self.base_dir / session_id / "session_metadata.json"

    async def _create_session_directory_structure(self, session_dir: Path):
        """Create the standard session directory structure."""
        directories = [
            "working",
            "research",
            "complete",
            "agent_logs",
            "sub_sessions"
        ]

        for directory in directories:
            (session_dir / directory).mkdir(parents=True, exist_ok=True)

        logger.debug(f"Created session directory structure: {session_dir}")

    async def _save_session_metadata(self, session_dir: Path, metadata: Dict[str, Any]):
        """Save session metadata to file."""
        metadata_file = session_dir / "session_metadata.json"

        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save session metadata: {e}")
            raise

    async def _load_session_metadata(self, session_dir: Path) -> Optional[Dict[str, Any]]:
        """Load session metadata from file."""
        metadata_file = session_dir / "session_metadata.json"

        try:
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load session metadata: {e}")

        return None

    async def _calculate_session_progress(self, session_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall session progress percentage."""
        stages = session_metadata["stages"]

        completed_stages = sum(1 for stage in stages.values() if stage["status"] == "completed")
        total_stages = len(stages)

        progress_percentage = (completed_stages / total_stages * 100) if total_stages > 0 else 0

        session_metadata["progress_percentage"] = round(progress_percentage, 2)
        session_metadata["completed_stages"] = completed_stages
        session_metadata["total_stages"] = total_stages

        # Determine overall status
        if completed_stages == total_stages:
            session_metadata["status"] = "completed"
        elif any(stage["status"] == "failed" for stage in stages.values()):
            session_metadata["status"] = "failed"
        elif any(stage["status"] == "running" for stage in stages.values()):
            session_metadata["status"] = "in_progress"
        else:
            session_metadata["status"] = "initialized"

        return session_metadata

    async def cleanup_expired_sessions(self, max_age_hours: int = 24) -> int:
        """
        Clean up expired sessions from disk and memory.

        Args:
            max_age_hours: Maximum age in hours before cleanup

        Returns:
            Number of sessions cleaned up
        """
        current_time = datetime.now(timezone.utc)
        max_age_seconds = max_age_hours * 3600

        cleaned_count = 0

        async with self._lock:
            for session_dir in self.base_dir.iterdir():
                if not session_dir.is_dir():
                    continue

                try:
                    metadata_file = session_dir / "session_metadata.json"
                    if not metadata_file.exists():
                        continue

                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)

                    created_at = datetime.fromisoformat(metadata["created_at"])
                    age_seconds = (current_time - created_at).total_seconds()

                    if age_seconds > max_age_seconds:
                        # Remove from active sessions
                        session_id = session_dir.name
                        if session_id in self._active_sessions:
                            del self._active_sessions[session_id]

                        # Remove directory (optional - keep for archival)
                        # import shutil
                        # shutil.rmtree(session_dir)

                        cleaned_count += 1
                        logger.info(f"Marked expired session for cleanup: {session_id}")

                except Exception as e:
                    logger.error(f"Error processing session {session_dir}: {e}")

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} expired sessions")

        return cleaned_count


# Global enhanced session manager instance
_enhanced_session_manager = None


def get_enhanced_session_manager(base_dir: str = "KEVIN/sessions") -> EnhancedSessionStateManager:
    """
    Get or create the global enhanced session manager instance.

    Args:
        base_dir: Base directory for session storage

    Returns:
        EnhancedSessionStateManager instance
    """
    global _enhanced_session_manager
    if _enhanced_session_manager is None:
        _enhanced_session_manager = EnhancedSessionStateManager(base_dir)
    return _enhanced_session_manager