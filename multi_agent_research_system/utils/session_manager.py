"""
Session Management Utility for Multi-Agent Research System

Provides session ID generation and management for tracking research activities.
"""

import uuid
import time
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages session IDs and basic session metadata for research tracking."""

    def __init__(self):
        """Initialize the session manager."""
        self._active_sessions: Dict[str, Dict[str, Any]] = {}
        logger.info("SessionManager initialized")

    def get_session_id(self, session_id: Optional[str] = None) -> str:
        """
        Get or generate a session ID.

        Args:
            session_id: Optional existing session ID to reuse

        Returns:
            Session ID string
        """
        if session_id and session_id in self._active_sessions:
            # Reuse existing session
            self._active_sessions[session_id]['last_accessed'] = time.time()
            logger.debug(f"Reusing existing session: {session_id}")
            return session_id

        # Generate new session ID
        new_session_id = str(uuid.uuid4())
        self._active_sessions[new_session_id] = {
            'created_at': time.time(),
            'last_accessed': time.time(),
            'purpose': 'research_session'
        }

        logger.info(f"Generated new session ID: {new_session_id}")
        return new_session_id

    def update_session_metadata(self, session_id: str, metadata: Dict[str, Any]) -> None:
        """
        Update metadata for a session.

        Args:
            session_id: Session ID to update
            metadata: Metadata to update
        """
        if session_id in self._active_sessions:
            self._active_sessions[session_id].update(metadata)
            self._active_sessions[session_id]['last_accessed'] = time.time()
            logger.debug(f"Updated metadata for session: {session_id}")
        else:
            logger.warning(f"Attempted to update non-existent session: {session_id}")

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a session.

        Args:
            session_id: Session ID to query

        Returns:
            Session metadata dictionary or None if not found
        """
        if session_id in self._active_sessions:
            return self._active_sessions[session_id].copy()
        return None

    def cleanup_expired_sessions(self, max_age_hours: int = 24) -> int:
        """
        Clean up expired sessions.

        Args:
            max_age_hours: Maximum age in hours before cleanup

        Returns:
            Number of sessions cleaned up
        """
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        expired_sessions = []
        for session_id, session_data in self._active_sessions.items():
            age = current_time - session_data['created_at']
            if age > max_age_seconds:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            del self._active_sessions[session_id]
            logger.debug(f"Cleaned up expired session: {session_id}")

        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

        return len(expired_sessions)

    def get_active_session_count(self) -> int:
        """Get the number of currently active sessions."""
        return len(self._active_sessions)


# Global session manager instance
_session_manager = None


def get_session_manager() -> SessionManager:
    """
    Get or create the global session manager instance.

    Returns:
        SessionManager instance
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager