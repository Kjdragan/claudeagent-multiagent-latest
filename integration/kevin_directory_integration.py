#!/usr/bin/env python3
"""
KEVIN Directory Integration - Comprehensive KEVIN Directory Structure Management

This module provides comprehensive integration with the KEVIN directory structure,
ensuring all system components properly organize their data according to the established
KEVIN session-based organization system.

Key Features:
- KEVIN directory structure validation and management
- Session-based file organization and naming conventions
- Integration with existing KEVIN workflows
- File path generation and validation
- Session state synchronization
- Workproduct management and tracking

Integration Capabilities:
- Seamless integration with existing KEVIN directory structure
- Session lifecycle management with directory organization
- File naming conventions and path management
- Cross-component file organization and access
- Session metadata synchronization
- Workproduct generation and storage coordination
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Configure logging
logger = logging.getLogger(__name__)


class KevinDirectoryIntegration:
    """
    Comprehensive KEVIN directory integration for agent-based research system.

    This class provides seamless integration with the existing KEVIN directory structure,
    ensuring all system components properly organize data according to the established
    session-based organization system.
    """

    def __init__(self, kevin_base_dir: str = "KEVIN"):
        """
        Initialize KEVIN directory integration.

        Args:
            kevin_base_dir: Base directory for KEVIN storage
        """

        self.kevin_base_dir = Path(kevin_base_dir)
        self.sessions_dir = self.kevin_base_dir / "sessions"
        self.templates_dir = self.sessions_dir / "templates"

        # Directory structure configuration
        self.directory_structure = {
            "sessions": self.sessions_dir,
            "templates": self.templates_dir,
            "work_products": self.kevin_base_dir / "work_products",
            "logs": self.kevin_base_dir / "logs",
            "monitoring": self.kevin_base_dir / "monitoring"
        }

        # Session subdirectory structure
        self.session_subdirs = {
            "working": "working",           # Active work in progress
            "research": "research",         # Research data and sources
            "complete": "complete",         # Completed workproducts
            "agent_logs": "agent_logs",     # Agent activity logs
            "sub_sessions": "sub_sessions"   # Gap research sub-sessions
        }

        # File naming conventions
        self.naming_conventions = {
            "working_prefixes": {
                "INITIAL_RESEARCH_DRAFT": "INITIAL_RESEARCH_DRAFT_",
                "REPORT": "REPORT_",
                "EDITORIAL_REVIEW": "EDITORIAL_REVIEW_",
                "EDITORIAL_RECOMMENDATIONS": "EDITORIAL_RECOMMENDATIONS_",
                "FINAL_REPORT": "FINAL_REPORT_",
                "QUALITY_ASSESSMENT": "QUALITY_ASSESSMENT_",
                "ENHANCED_REPORT": "ENHANCED_REPORT_"
            },
            "research_prefixes": {
                "INITIAL_SEARCH": "INITIAL_SEARCH_",
                "EDITORIAL_RESEARCH": "EDITORIAL_RESEARCH_",
                "GAP_RESEARCH": "GAP_RESEARCH_",
                "QUALITY_ENHANCEMENT": "QUALITY_ENHANCEMENT_"
            },
            "log_prefixes": {
                "session": "session_",
                "agent": "agent_",
                "research": "research_",
                "editorial": "editorial_",
                "quality": "quality_",
                "workflow": "workflow_"
            }
        }

        # Initialize directory structure
        self._ensure_directory_structure()

        logger.info(f"🏗️  KEVIN directory integration initialized with base: {kevin_base_dir}")

    def _ensure_directory_structure(self):
        """Ensure the complete KEVIN directory structure exists."""

        for dir_name, dir_path in self.directory_structure.items():
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"📁 Directory ensured: {dir_path}")

    def get_session_directory(self, session_id: str) -> Path:
        """
        Get the main directory path for a session.

        Args:
            session_id: Session identifier

        Returns:
            Path: Session directory path
        """

        session_dir = self.sessions_dir / session_id
        return session_dir

    def get_session_subdirectory(self, session_id: str, subdir_name: str) -> Path:
        """
        Get a specific subdirectory within a session.

        Args:
            session_id: Session identifier
            subdir_name: Subdirectory name (working, research, complete, agent_logs, sub_sessions)

        Returns:
            Path: Subdirectory path
        """

        if subdir_name not in self.session_subdirs:
            raise ValueError(f"Invalid subdirectory: {subdir_name}. Valid options: {list(self.session_subdirs.keys())}")

        session_dir = self.get_session_directory(session_id)
        subdir_path = session_dir / self.session_subdirs[subdir_name]
        return subdir_path

    def create_session_structure(self, session_id: str) -> Dict[str, Path]:
        """
        Create the complete directory structure for a session.

        Args:
            session_id: Session identifier

        Returns:
            Dict[str, Path]: Dictionary of created directory paths
        """

        logger.info(f"🏗️  Creating session structure: {session_id}")

        created_dirs = {}

        # Create main session directory
        session_dir = self.get_session_directory(session_id)
        session_dir.mkdir(exist_ok=True)
        created_dirs["session"] = session_dir

        # Create subdirectories
        for subdir_name, subdir_path_name in self.session_subdirs.items():
            subdir_path = session_dir / subdir_path_name
            subdir_path.mkdir(exist_ok=True)
            created_dirs[subdir_name] = subdir_path

        logger.debug(f"📁 Session structure created with {len(created_dirs)} directories")
        return created_dirs

    def generate_workproduct_filename(self, session_id: str, stage: str,
                                     description: Optional[str] = None) -> str:
        """
        Generate a standardized workproduct filename.

        Args:
            session_id: Session identifier
            stage: Research stage
            description: Optional description for the filename

        Returns:
            str: Generated filename
        """

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Determine prefix based on stage
        if stage == "working":
            if "INITIAL_RESEARCH" in (description or "").upper():
                prefix = self.naming_conventions["working_prefixes"]["INITIAL_RESEARCH_DRAFT"]
            elif "EDITORIAL" in (description or "").upper():
                prefix = self.naming_conventions["working_prefixes"]["EDITORIAL_REVIEW"]
            elif "REPORT" in (description or "").upper():
                prefix = self.naming_conventions["working_prefixes"]["REPORT"]
            elif "FINAL" in (description or "").upper():
                prefix = self.naming_conventions_prefixes["working_prefixes"]["FINAL_REPORT"]
            else:
                prefix = self.naming_conventions["working_prefixes"]["INITIAL_RESEARCH_DRAFT"]

        elif stage == "research":
            if "INITIAL" in (description or "").upper():
                prefix = self.naming_conventions["research_prefixes"]["INITIAL_SEARCH"]
            elif "EDITORIAL" in (description or "").upper():
                prefix = self.naming_conventions["research_prefixes"]["EDITORIAL_RESEARCH"]
            elif "GAP" in (description or "").upper():
                prefix = self.naming_conventions["research_prefixes"]["GAP_RESEARCH"]
            else:
                prefix = self.naming_conventions["research_prefixes"]["INITIAL_SEARCH"]

        elif stage == "complete":
            prefix = "FINAL_ENHANCED_"

        else:
            # Default prefix
            prefix = f"{stage.upper()}_"

        filename = f"{prefix}{timestamp}.md"
        return filename

    def get_workproduct_path(self, session_id: str, stage: str,
                            description: Optional[str] = None) -> Path:
        """
        Get the full path for a workproduct file.

        Args:
            session_id: Session identifier
            stage: Research stage
            description: Optional description

        Returns:
            Path: Full file path
        """

        if stage == "complete":
            subdir = "complete"
        elif stage in ["working", "research"]:
            subdir = stage
        else:
            subdir = "working"  # Default to working for unknown stages

        subdir_path = self.get_session_subdirectory(session_id, subdir)
        filename = self.generate_workproduct_filename(session_id, stage, description)
        return subdir_path / filename

    def get_log_file_path(self, session_id: str, log_type: str,
                          component: Optional[str] = None) -> Path:
        """
        Get the full path for a log file.

        Args:
            session_id: Session identifier
            log_type: Type of log (session, agent, research, editorial, quality, workflow)
            component: Optional component name

        Returns:
            Path: Full log file path
        """

        timestamp = datetime.now().strftime("%Y%m%d")
        subdir_path = self.get_session_subdirectory(session_id, "agent_logs")

        if log_type in self.naming_conventions["log_prefixes"]:
            prefix = self.naming_conventions["log_prefixes"][log_type]
        else:
            prefix = f"{log_type}_"

        if component:
            filename = f"{prefix}{component}_{timestamp}.log"
        else:
            filename = f"{prefix}{timestamp}.log"

        return subdir_path / filename

    def save_session_metadata(self, session_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Save session metadata to file.

        Args:
            session_id: Session identifier
            metadata: Session metadata dictionary

        Returns:
            bool: Success status
        """

        try:
            session_dir = self.get_session_directory(session_id)
            metadata_file = session_dir / "session_metadata.json"

            # Add/update metadata fields
            metadata["last_updated"] = datetime.now().isoformat()
            metadata["kevin_integration_version"] = "1.0"

            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.debug(f"💾 Session metadata saved: {metadata_file}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to save session metadata: {e}")
            return False

    def load_session_metadata(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load session metadata from file.

        Args:
            session_id: Session identifier

        Returns:
            Optional[Dict[str, Any]]: Session metadata or None if not found
        """

        try:
            session_dir = self.get_session_directory(session_id)
            metadata_file = session_dir / "session_metadata.json"

            if not metadata_file.exists():
                logger.warning(f"⚠️  Session metadata file not found: {metadata_file}")
                return None

            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            logger.debug(f"📖 Session metadata loaded: {metadata_file}")
            return metadata

        except Exception as e:
            logger.error(f"❌ Failed to load session metadata: {e}")
            return None

    def update_session_stage(self, session_id: str, stage: str,
                           status: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update session stage information.

        Args:
            session_id: Session identifier
            stage: Stage name
            status: Stage status
            metadata: Optional additional metadata

        Returns:
            bool: Success status
        """

        try:
            # Load existing metadata
            session_metadata = self.load_session_metadata(session_id)
            if not session_metadata:
                session_metadata = {
                    "session_id": session_id,
                    "created_at": datetime.now().isoformat(),
                    "stages": {}
                }

            # Update stage information
            if "stages" not in session_metadata:
                session_metadata["stages"] = {}

            session_metadata["stages"][stage] = {
                "status": status,
                "updated_at": datetime.now().isoformat()
            }

            # Add additional metadata if provided
            if metadata:
                session_metadata["stages"][stage].update(metadata)

            # Update current stage if it's the most recent
            session_metadata["current_stage"] = stage
            session_metadata["current_status"] = status

            # Save updated metadata
            return self.save_session_metadata(session_id, session_metadata)

        except Exception as e:
            logger.error(f"❌ Failed to update session stage: {e}")
            return False

    def get_session_files(self, session_id: str, stage: Optional[str] = None) -> Dict[str, List[Path]]:
        """
        Get all files for a session, optionally filtered by stage.

        Args:
            session_id: Session identifier
            stage: Optional stage filter

        Returns:
            Dict[str, List[Path]]: Dictionary of file lists by type
        """

        files = {
            "working_files": [],
            "research_files": [],
            "complete_files": [],
            "log_files": [],
            "metadata_files": []
        }

        try:
            session_dir = self.get_session_directory(session_id)

            # Get files by subdirectory
            for subdir_name, subdir_key in [
                ("working", "working_files"),
                ("research", "research_files"),
                ("complete", "complete_files"),
                ("agent_logs", "log_files")
            ]:
                subdir_path = session_dir / subdir_name
                if subdir_path.exists():
                    files[subdir_key] = list(subdir_path.glob("*"))

            # Get metadata file
            metadata_file = session_dir / "session_metadata.json"
            if metadata_file.exists():
                files["metadata_files"] = [metadata_file]

            # Filter by stage if specified
            if stage:
                # Determine which subdirectory to filter
                if stage == "working":
                    return {"working_files": files["working_files"]}
                elif stage == "research":
                    return {"research_files": files["research_files"]}
                elif stage == "complete":
                    return {"complete_files": files["complete_files"]}
                elif stage == "logs":
                    return {"log_files": files["log_files"]}

            return files

        except Exception as e:
            logger.error(f"❌ Failed to get session files: {e}")
            return files

    def create_session_backup(self, session_id: str, backup_name: Optional[str] = None) -> bool:
        """
        Create a backup of a session directory.

        Args:
            session_id: Session identifier
            backup_name: Optional backup name (auto-generated if not provided)

        Returns:
            bool: Success status
        """

        try:
            import shutil

            session_dir = self.get_session_directory(session_id)
            if not session_dir.exists():
                logger.warning(f"⚠️  Session directory not found: {session_dir}")
                return False

            # Generate backup name
            if not backup_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"backup_{session_id}_{timestamp}"

            backup_dir = self.kevin_base_dir / "backups" / backup_name
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Copy session directory
            shutil.copytree(session_dir, backup_dir / session_id)

            logger.info(f"💾 Session backup created: {backup_dir}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to create session backup: {e}")
            return False

    def cleanup_old_sessions(self, max_age_days: int = 30, dry_run: bool = False) -> Dict[str, Any]:
        """
        Clean up old sessions based on age.

        Args:
            max_age_days: Maximum age in days
            dry_run: Whether to perform dry run (without actual deletion)

        Returns:
            Dict[str, Any]: Cleanup results
        """

        cleanup_results = {
            "sessions_checked": 0,
            "sessions_to_delete": [],
            "sessions_deleted": [],
            "errors": []
        }

        try:
            cutoff_date = datetime.now().timestamp() - (max_age_days * 24 * 3600)

            for session_dir in self.sessions_dir.iterdir():
                if not session_dir.is_dir():
                    continue

                session_id = session_dir.name
                cleanup_results["sessions_checked"] += 1

                # Check session age
                metadata = self.load_session_metadata(session_id)
                if metadata:
                    created_at = metadata.get("created_at", "")
                    if created_at:
                        try:
                            created_timestamp = datetime.fromisoformat(created_at).timestamp()
                            if created_timestamp < cutoff_date:
                                cleanup_results["sessions_to_delete"].append(session_id)
                        except ValueError:
                            cleanup_results["errors"].append(f"Invalid date format in metadata for session {session_id}")

                # Check directory modification time as fallback
                elif session_dir.stat().st_mtime < cutoff_date:
                    cleanup_results["sessions_to_delete"].append(session_id)

            # Delete old sessions if not dry run
            if not dry_run:
                for session_id in cleanup_results["sessions_to_delete"]:
                    try:
                        session_dir = self.get_session_directory(session_id)
                        import shutil
                        shutil.rmtree(session_dir)
                        cleanup_results["sessions_deleted"].append(session_id)
                        logger.info(f"🗑️  Deleted old session: {session_id}")
                    except Exception as e:
                        cleanup_results["errors"].append(f"Failed to delete session {session_id}: {e}")

            logger.info(f"🧹 Session cleanup completed: "
                       f"checked={cleanup_results['sessions_checked']}, "
                       f"to_delete={len(cleanup_results['sessions_to_delete'])}, "
                       f"deleted={len(cleanup_results['sessions_deleted'])}")

            return cleanup_results

        except Exception as e:
            logger.error(f"❌ Session cleanup failed: {e}")
            cleanup_results["errors"].append(str(e))
            return cleanup_results

    def get_session_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about KEVIN sessions.

        Returns:
            Dict[str, Any]: Session statistics
        """

        stats = {
            "total_sessions": 0,
            "active_sessions": 0,
            "total_files": 0,
            "total_size_bytes": 0,
            "sessions_by_status": {},
            "file_types": {},
            "storage_usage": {}
        }

        try:
            if not self.sessions_dir.exists():
                return stats

            # Count sessions and files
            for session_dir in self.sessions_dir.iterdir():
                if not session_dir.is_dir():
                    continue

                stats["total_sessions"] += 1

                # Get session metadata
                metadata = self.load_session_metadata(session_dir.name)
                if metadata:
                    status = metadata.get("status", "unknown")
                    stats["sessions_by_status"][status] = stats["sessions_by_status"].get(status, 0) + 1

                # Count files
                session_files = self.get_session_files(session_dir.name)
                for file_type, files in session_files.items():
                    stats["total_files"] += len(files)
                    stats["file_types"][file_type] = stats["file_types"].get(file_type, 0) + len(files)

                    # Calculate size
                    for file_path in files:
                        if file_path.is_file():
                            stats["total_size_bytes"] += file_path.stat().st_size

            # Calculate storage usage
            stats["storage_usage"] = {
                "total_size_mb": round(stats["total_size_bytes"] / (1024 * 1024), 2),
                "average_size_mb": round(stats["total_size_bytes"] / max(stats["total_sessions"], 1) / (1024 * 1024), 2) if stats["total_sessions"] > 0 else 0
            }

            return stats

        except Exception as e:
            logger.error(f"❌ Failed to get session statistics: {e}")
            return stats

    def validate_session_structure(self, session_id: str) -> Dict[str, Any]:
        """
        Validate the structure of a session directory.

        Args:
            session_id: Session identifier

        Returns:
            Dict[str, Any]: Validation results
        """

        validation_result = {
            "is_valid": True,
            "issues": [],
            "missing_directories": [],
            "missing_files": [],
            "structure_errors": []
        }

        try:
            session_dir = self.get_session_directory(session_id)

            if not session_dir.exists():
                validation_result["is_valid"] = False
                validation_result["issues"].append("Session directory does not exist")
                return validation_result

            # Check required subdirectories
            for subdir_name in self.session_subdirs.values():
                subdir_path = session_dir / subdir_name
                if not subdir_path.exists():
                    validation_result["missing_directories"].append(subdir_name)
                    validation_result["issues"].append(f"Missing subdirectory: {subdir_name}")

            # Check metadata file
            metadata_file = session_dir / "session_metadata.json"
            if not metadata_file.exists():
                validation_result["missing_files"].append("session_metadata.json")
                validation_result["issues"].append("Missing session metadata file")

            # Validate directory structure integrity
            if validation_result["issues"]:
                validation_result["is_valid"] = False

            return validation_result

        except Exception as e:
            validation_result["is_valid"] = False
            validation_result["structure_errors"].append(str(e))
            return validation_result


# Fallback KEVIN integration for when directory operations fail
class FallbackKevinIntegration:
    """Fallback KEVIN integration for basic operations."""

    def __init__(self, kevin_base_dir: str = "KEVIN"):
        self.kevin_base_dir = Path(kevin_base_dir)
        self.sessions_dir = self.kevin_base_dir / "sessions"

    def get_session_directory(self, session_id: str) -> Path:
        """Get session directory path (fallback)."""
        return self.sessions_dir / session_id

    def create_session_structure(self, session_id: str) -> Dict[str, Path]:
        """Create basic session structure (fallback)."""
        session_dir = self.get_session_directory(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        return {"session": session_dir}