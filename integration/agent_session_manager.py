#!/usr/bin/env python3
"""
Agent Session Manager - Bridge between Claude Agent SDK and KEVIN Directory Structure

This module provides the integration layer between Claude Agent SDK sessions and the
existing KEVIN directory structure. It handles session creation, lifecycle management,
and seamless data persistence for agent-based research workflows.

Key Features:
- Bridge Claude Agent SDK sessions to KEVIN directory structure
- Session lifecycle management with proper initialization and cleanup
- File organization and metadata tracking
- Integration with existing workflow management systems
- Support for sub-session coordination (gap research)
- Real-time session monitoring and status tracking

Session Management Capabilities:
- Automatic session directory creation with standardized structure
- Session metadata persistence and recovery
- Integration with KEVIN's session-based organization
- Support for session linking and hierarchical organization
- Comprehensive session state tracking
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Import existing session management components with graceful fallback
LEGACY_SESSION_AVAILABLE = False
try:
    # Try importing without triggering the full system initialization
    import sys
    import importlib.util

    # Load KevinSessionManager if available
    kevin_spec = importlib.util.find_spec("multi_agent_research_system.core.kevin_session_manager")
    if kevin_spec and kevin_spec.loader:
        kevin_module = importlib.util.module_from_spec(kevin_spec)
        sys.modules["multi_agent_research_system.core.kevin_session_manager"] = kevin_module
        kevin_spec.loader.exec_module(kevin_module)
        KevinSessionManager = kevin_module.KevinSessionManager
        LEGACY_SESSION_AVAILABLE = True

    # Load WorkflowStateManager if available
    workflow_spec = importlib.util.find_spec("multi_agent_research_system.core.workflow_state")
    if workflow_spec and workflow_spec.loader:
        workflow_module = importlib.util.module_from_spec(workflow_spec)
        sys.modules["multi_agent_research_system.core.workflow_state"] = workflow_module
        workflow_spec.loader.exec_module(workflow_module)
        WorkflowStateManager = workflow_module.WorkflowStateManager

except Exception as e:
    logging.warning(f"Legacy session management not available: {e}")
    KevinSessionManager = None
    WorkflowStateManager = None
    LEGACY_SESSION_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


class AgentSessionManager:
    """
    Advanced session management for Claude Agent SDK integration with KEVIN directory structure.

    This class provides comprehensive session management capabilities that bridge Claude Agent
    SDK sessions with the existing KEVIN directory organization system.
    """

    def __init__(self, kevin_base_dir: str = "KEVIN"):
        """
        Initialize the agent session manager.

        Args:
            kevin_base_dir: Base directory for KEVIN session storage
        """

        self.kevin_base_dir = Path(kevin_base_dir)
        self.sessions_dir = self.kevin_base_dir / "sessions"
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_history: List[Dict[str, Any]] = []

        # Initialize directory structure
        self._ensure_directory_structure()

        # Initialize legacy components if available
        self.legacy_session_manager = None
        self.workflow_state_manager = None

        if LEGACY_SESSION_AVAILABLE and KevinSessionManager is not None:
            try:
                self.legacy_session_manager = KevinSessionManager(str(self.kevin_base_dir))
                logger.info("✅ KevinSessionManager integrated")
            except Exception as e:
                logger.warning(f"Failed to integrate KevinSessionManager: {e}")

        if LEGACY_SESSION_AVAILABLE and WorkflowStateManager is not None:
            try:
                self.workflow_state_manager = WorkflowStateManager()
                logger.info("✅ WorkflowStateManager integrated")
            except Exception as e:
                logger.warning(f"Failed to integrate WorkflowStateManager: {e}")

        if self.legacy_session_manager or self.workflow_state_manager:
            logger.info("✅ Legacy session components integrated")
        else:
            logger.info("ℹ️  Using standalone session management (no legacy integration)")

        logger.info(f"🔧 Agent session manager initialized with KEVIN base: {kevin_base_dir}")

    def _ensure_directory_structure(self):
        """Ensure the KEVIN directory structure exists."""

        directories = [
            self.sessions_dir,
            self.kevin_base_dir / "logs",
            self.kevin_base_dir / "monitoring"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        logger.debug("📁 KEVIN directory structure ensured")

    async def create_session(self, topic: str, user_requirements: Dict[str, Any],
                           session_id: Optional[str] = None) -> str:
        """
        Create a new research session with KEVIN directory integration.

        Args:
            topic: Research topic or query
            user_requirements: User requirements and preferences
            session_id: Optional predefined session ID

        Returns:
            str: Created session ID
        """

        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())

        logger.info(f"🆔 Creating agent session: {session_id}")
        logger.info(f"📝 Research topic: {topic[:100]}...")

        try:
            # Create session directory structure
            session_dir = await self._create_session_directory(session_id, topic)

            # Initialize session metadata
            session_metadata = await self._initialize_session_metadata(
                session_id, topic, user_requirements
            )

            # Create session state
            session_state = {
                "session_id": session_id,
                "topic": topic,
                "user_requirements": user_requirements,
                "session_dir": str(session_dir),
                "created_at": datetime.now().isoformat(),
                "status": "active",
                "stage": "initialization",
                "metadata": session_metadata,
                "agent_interactions": [],
                "files_created": [],
                "sub_sessions": []
            }

            # Store session
            self.active_sessions[session_id] = session_state

            # Save session metadata to file
            await self._save_session_metadata(session_id, session_metadata)

            # Integrate with legacy session manager if available
            if self.legacy_session_manager:
                try:
                    await self._integrate_with_legacy_session(session_id, topic, user_requirements)
                except Exception as e:
                    logger.warning(f"Legacy integration failed: {e}")

            logger.info(f"✅ Session created successfully: {session_id}")
            logger.info(f"📁 Session directory: {session_dir}")

            return session_id

        except Exception as e:
            logger.error(f"❌ Failed to create session {session_id}: {e}")
            raise

    async def _create_session_directory(self, session_id: str, topic: str) -> Path:
        """Create the standardized session directory structure."""

        session_dir = self.sessions_dir / session_id

        # Create session subdirectories
        subdirs = [
            "working",        # Active work in progress
            "research",       # Research data and sources
            "complete",       # Completed workproducts
            "agent_logs",     # Agent activity logs
            "sub_sessions"    # Gap research sub-sessions
        ]

        for subdir in subdirs:
            (session_dir / subdir).mkdir(parents=True, exist_ok=True)

        logger.debug(f"📁 Session directories created: {session_dir}")
        return session_dir

    async def _initialize_session_metadata(self, session_id: str, topic: str,
                                         user_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize comprehensive session metadata."""

        metadata = {
            "session_info": {
                "session_id": session_id,
                "topic": topic,
                "created_at": datetime.now().isoformat(),
                "status": "initialized",
                "agent_type": "comprehensive_research_agent"
            },
            "user_requirements": user_requirements,
            "workflow_stages": {
                "initialization": {
                    "status": "completed",
                    "completed_at": datetime.now().isoformat()
                },
                "query_processing": {"status": "pending"},
                "research_execution": {"status": "pending"},
                "content_analysis": {"status": "pending"},
                "report_generation": {"status": "pending"},
                "quality_assessment": {"status": "pending"},
                "finalization": {"status": "pending"}
            },
            "research_configuration": {
                "target_urls": user_requirements.get("target_results", 50),
                "concurrent_processing": True,
                "anti_bot_level": 1,
                "quality_threshold": 0.8,
                "session_prefix": "comprehensive_research"
            },
            "file_tracking": {
                "working_files": [],
                "research_files": [],
                "complete_files": [],
                "log_files": []
            },
            "agent_interactions": {
                "total_interactions": 0,
                "research_queries": 0,
                "tool_executions": 0,
                "error_count": 0
            },
            "session_metrics": {
                "duration_seconds": 0,
                "total_urls_processed": 0,
                "successful_scrapes": 0,
                "quality_score": None,
                "completion_percentage": 0
            }
        }

        return metadata

    async def _save_session_metadata(self, session_id: str, metadata: Dict[str, Any]):
        """Save session metadata to file."""

        session_dir = self.sessions_dir / session_id
        metadata_file = session_dir / "session_metadata.json"

        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.debug(f"💾 Session metadata saved: {metadata_file}")

        except Exception as e:
            logger.error(f"Failed to save session metadata: {e}")
            raise

    async def _integrate_with_legacy_session(self, session_id: str, topic: str,
                                           user_requirements: Dict[str, Any]):
        """Integrate with legacy session management systems."""

        if not self.legacy_session_manager:
            return

        try:
            # Use the same session ID for legacy session to avoid duplication
            legacy_session_id = await self.legacy_session_manager.create_session(
                topic=topic,
                user_requirements=user_requirements,
                session_id=session_id  # Use the same session ID
            )

            # Link sessions
            await self._link_legacy_session(session_id, legacy_session_id)

            logger.info(f"🔗 Legacy session created with shared ID: {legacy_session_id}")

        except Exception as e:
            logger.warning(f"Legacy session integration failed: {e}")

    async def _link_legacy_session(self, agent_session_id: str, legacy_session_id: str):
        """Link agent session with legacy session."""

        link_data = {
            "agent_session_id": agent_session_id,
            "legacy_session_id": legacy_session_id,
            "linked_at": datetime.now().isoformat(),
            "link_type": "agent_to_legacy"
        }

        # Store link information
        session_dir = self.sessions_dir / agent_session_id
        link_file = session_dir / "legacy_session_link.json"

        try:
            with open(link_file, 'w', encoding='utf-8') as f:
                json.dump(link_data, f, indent=2)

            logger.debug(f"🔗 Session link saved: {link_file}")

        except Exception as e:
            logger.error(f"Failed to save session link: {e}")

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session information.

        Args:
            session_id: Session ID to retrieve

        Returns:
            Optional[Dict[str, Any]]: Session data or None if not found
        """

        # Check active sessions first
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]

        # Try to load from file
        return await self._load_session_from_file(session_id)

    async def _load_session_from_file(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session data from file storage."""

        session_dir = self.sessions_dir / session_id
        metadata_file = session_dir / "session_metadata.json"

        if not metadata_file.exists():
            logger.warning(f"Session metadata not found: {session_id}")
            return None

        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # Reconstruct session state
            session_state = {
                "session_id": session_id,
                "topic": metadata["session_info"]["topic"],
                "user_requirements": metadata["user_requirements"],
                "session_dir": str(session_dir),
                "created_at": metadata["session_info"]["created_at"],
                "status": metadata["session_info"]["status"],
                "metadata": metadata,
                "loaded_from_file": True
            }

            return session_state

        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    async def update_session_stage(self, session_id: str, stage: str,
                                 status: str = "running", **kwargs):
        """
        Update session stage status.

        Args:
            session_id: Session ID
            stage: Workflow stage name
            status: Stage status
            **kwargs: Additional stage data
        """

        session = await self.get_session(session_id)
        if not session:
            logger.warning(f"Session not found for stage update: {session_id}")
            return

        try:
            # Update metadata
            if "workflow_stages" in session["metadata"]:
                session["metadata"]["workflow_stages"][stage] = {
                    "status": status,
                    "updated_at": datetime.now().isoformat(),
                    **kwargs
                }

                # Update overall session status
                if status == "completed":
                    session["metadata"]["session_info"]["status"] = f"{stage}_completed"
                elif status == "error":
                    session["metadata"]["session_info"]["status"] = "error"
                    session["metadata"]["session_info"]["error"] = kwargs.get("error", "Unknown error")

                # Save updated metadata
                await self._save_session_metadata(session_id, session["metadata"])

                # Update active session if in memory
                if session_id in self.active_sessions:
                    self.active_sessions[session_id].update(session)

                logger.info(f"📝 Session {session_id} stage updated: {stage} -> {status}")

        except Exception as e:
            logger.error(f"Failed to update session stage: {e}")

    async def create_sub_session(self, parent_session_id: str, sub_session_type: str,
                               query: str, **kwargs) -> str:
        """
        Create a sub-session for gap research or specialized tasks.

        Args:
            parent_session_id: Parent session ID
            sub_session_type: Type of sub-session (e.g., "gap_research", "validation")
            query: Sub-session query or task
            **kwargs: Additional sub-session parameters

        Returns:
            str: Created sub-session ID
        """

        sub_session_id = str(uuid.uuid4())
        sub_session_name = f"{sub_session_type}_{sub_session_id[:8]}"

        logger.info(f"🔗 Creating sub-session: {sub_session_name} (parent: {parent_session_id})")

        try:
            # Get parent session
            parent_session = await self.get_session(parent_session_id)
            if not parent_session:
                raise ValueError(f"Parent session not found: {parent_session_id}")

            # Create sub-session directory
            session_dir = Path(parent_session["session_dir"])
            sub_session_dir = session_dir / "sub_sessions" / sub_session_name
            sub_session_dir.mkdir(parents=True, exist_ok=True)

            # Initialize sub-session metadata
            sub_metadata = {
                "sub_session_info": {
                    "sub_session_id": sub_session_id,
                    "sub_session_name": sub_session_name,
                    "parent_session_id": parent_session_id,
                    "sub_session_type": sub_session_type,
                    "query": query,
                    "created_at": datetime.now().isoformat(),
                    "status": "initialized"
                },
                "sub_session_config": kwargs,
                "results": {},
                "files_created": []
            }

            # Save sub-session metadata
            sub_metadata_file = sub_session_dir / "sub_session_metadata.json"
            with open(sub_metadata_file, 'w', encoding='utf-8') as f:
                json.dump(sub_metadata, f, indent=2)

            # Update parent session
            if "sub_sessions" not in parent_session["metadata"]:
                parent_session["metadata"]["sub_sessions"] = []

            parent_session["metadata"]["sub_sessions"].append({
                "sub_session_id": sub_session_id,
                "sub_session_name": sub_session_name,
                "sub_session_type": sub_session_type,
                "query": query,
                "created_at": datetime.now().isoformat()
            })

            await self._save_session_metadata(parent_session_id, parent_session["metadata"])

            logger.info(f"✅ Sub-session created: {sub_session_name}")

            return sub_session_id

        except Exception as e:
            logger.error(f"Failed to create sub-session: {e}")
            raise

    async def log_agent_interaction(self, session_id: str, interaction_type: str,
                                  details: Dict[str, Any]):
        """
        Log agent interaction for session tracking.

        Args:
            session_id: Session ID
            interaction_type: Type of interaction (e.g., "query", "tool_execution", "error")
            details: Interaction details
        """

        try:
            session = await self.get_session(session_id)
            if not session:
                logger.warning(f"Session not found for interaction logging: {session_id}")
                return

            # Create interaction log entry
            interaction = {
                "timestamp": datetime.now().isoformat(),
                "interaction_type": interaction_type,
                "details": details
            }

            # Update session interaction tracking
            if "agent_interactions" not in session["metadata"]:
                session["metadata"]["agent_interactions"] = []

            # Check if agent_interactions is a list
            if not isinstance(session["metadata"]["agent_interactions"], list):
                # Convert to list if it's not (might be dict from earlier version)
                session["metadata"]["agent_interactions"] = []

            session["metadata"]["agent_interactions"].append(interaction)

            # Update interaction counts in session_metrics
            if "session_metrics" not in session["metadata"]:
                session["metadata"]["session_metrics"] = {}

            session["metadata"]["session_metrics"]["total_interactions"] = len(
                [i for i in session["metadata"]["agent_interactions"]
                 if isinstance(i, dict) and "timestamp" in i]
            )

            # Save updated metadata
            await self._save_session_metadata(session_id, session["metadata"])

            # Also log to agent_logs directory
            await self._log_to_agent_file(session_id, interaction)

            logger.debug(f"📝 Agent interaction logged: {interaction_type} for session {session_id}")

        except Exception as e:
            logger.error(f"Failed to log agent interaction: {e}")

    async def _log_to_agent_file(self, session_id: str, interaction: Dict[str, Any]):
        """Log interaction to agent-specific log file."""

        session_dir = self.sessions_dir / session_id
        agent_logs_dir = session_dir / "agent_logs"

        # Create log file for interaction type
        log_file = agent_logs_dir / f"{interaction['interaction_type']}.log"

        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{interaction['timestamp']} - {interaction['interaction_type']}\n")
                f.write(f"Details: {json.dumps(interaction['details'], indent=2)}\n")
                f.write("-" * 80 + "\n")

        except Exception as e:
            logger.error(f"Failed to write agent log file: {e}")

    async def close_session(self, session_id: str, final_status: str = "completed"):
        """
        Close and finalize a session.

        Args:
            session_id: Session ID to close
            final_status: Final session status
        """

        try:
            session = await self.get_session(session_id)
            if not session:
                logger.warning(f"Session not found for closing: {session_id}")
                return

            # Update final status
            session["metadata"]["session_info"]["status"] = final_status
            session["metadata"]["session_info"]["completed_at"] = datetime.now().isoformat()

            # Calculate session duration
            created_at = datetime.fromisoformat(session["created_at"])
            completed_at = datetime.now()
            duration = completed_at - created_at

            session["metadata"]["session_metrics"]["duration_seconds"] = int(duration.total_seconds())

            # Save final metadata
            await self._save_session_metadata(session_id, session["metadata"])

            # Move from active to history
            if session_id in self.active_sessions:
                session_history_entry = self.active_sessions[session_id].copy()
                session_history_entry["closed_at"] = completed_at.isoformat()
                self.session_history.append(session_history_entry)
                del self.active_sessions[session_id]

            logger.info(f"🏁 Session closed: {session_id} (status: {final_status}, duration: {duration})")

        except Exception as e:
            logger.error(f"Failed to close session: {e}")

    async def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of session activity and results.

        Args:
            session_id: Session ID

        Returns:
            Optional[Dict[str, Any]]: Session summary
        """

        session = await self.get_session(session_id)
        if not session:
            return None

        try:
            metadata = session["metadata"]

            summary = {
                "session_id": session_id,
                "topic": session["topic"],
                "status": metadata["session_info"]["status"],
                "created_at": metadata["session_info"]["created_at"],
                "duration_seconds": metadata["session_metrics"].get("duration_seconds", 0),
                "completed_stages": [
                    stage for stage, info in metadata["workflow_stages"].items()
                    if info.get("status") == "completed"
                ],
                "total_interactions": len([
                    i for i in metadata.get("agent_interactions", [])
                    if isinstance(i, dict) and "timestamp" in i
                ]),
                "sub_sessions_count": len(metadata.get("sub_sessions", [])),
                "files_created": sum(
                    len(files) for files in metadata["file_tracking"].values()
                ),
                "research_metrics": metadata["session_metrics"]
            }

            return summary

        except Exception as e:
            logger.error(f"Failed to generate session summary: {e}")
            return None

    async def cleanup_old_sessions(self, max_age_days: int = 30):
        """Clean up old completed sessions."""

        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        cleaned_count = 0

        try:
            for session_dir in self.sessions_dir.iterdir():
                if not session_dir.is_dir():
                    continue

                metadata_file = session_dir / "session_metadata.json"
                if not metadata_file.exists():
                    continue

                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)

                    created_at = datetime.fromisoformat(metadata["session_info"]["created_at"])

                    # Clean old completed sessions
                    if (created_at < cutoff_date and
                        metadata["session_info"]["status"] in ["completed", "error"]):

                        # Archive or remove session
                        await self._archive_session(session_dir)
                        cleaned_count += 1

                except Exception as e:
                    logger.warning(f"Failed to process session {session_dir.name}: {e}")

            if cleaned_count > 0:
                logger.info(f"🧹 Cleaned up {cleaned_count} old sessions")

        except Exception as e:
            logger.error(f"Session cleanup failed: {e}")

    async def _archive_session(self, session_dir: Path):
        """Archive a session directory."""

        archive_dir = self.kevin_base_dir / "archived_sessions"
        archive_dir.mkdir(exist_ok=True)

        try:
            # Create archive name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_name = f"{session_dir.name}_archived_{timestamp}"
            archive_path = archive_dir / archive_name

            # Move session directory to archive
            session_dir.rename(archive_path)

            logger.debug(f"📦 Session archived: {session_dir.name} -> {archive_name}")

        except Exception as e:
            logger.error(f"Failed to archive session {session_dir.name}: {e}")

    def get_active_sessions_count(self) -> int:
        """Get count of active sessions."""
        return len(self.active_sessions)

    def get_session_history_count(self) -> int:
        """Get count of sessions in history."""
        return len(self.session_history)


# Fallback session manager for when integration components are not available
class FallbackSessionManager:
    """Simplified session manager for fallback operations."""

    def __init__(self, kevin_base_dir: str = "KEVIN"):
        self.kevin_base_dir = Path(kevin_base_dir)
        self.sessions_dir = self.kevin_base_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    async def create_session(self, topic: str, user_requirements: Dict[str, Any]) -> str:
        """Create a simple fallback session."""
        session_id = str(uuid.uuid4())

        session_dir = self.sessions_dir / session_id
        session_dir.mkdir(exist_ok=True)

        # Create simple metadata
        metadata = {
            "session_id": session_id,
            "topic": topic,
            "user_requirements": user_requirements,
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }

        metadata_file = session_dir / "session_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        return session_id

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get fallback session data."""
        session_dir = self.sessions_dir / session_id
        metadata_file = session_dir / "session_metadata.json"

        if not metadata_file.exists():
            return None

        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)