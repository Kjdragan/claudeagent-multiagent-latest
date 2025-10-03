"""
Base Agent Logger class for multi-agent research system.

This module provides the base AgentLogger class that all specialized
agent loggers inherit from.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .structured_logger import StructuredLogger


class AgentLogger:
    """Base class for all agent-specific loggers."""

    def __init__(self, agent_name: str, session_id: Optional[str] = None, base_log_dir: str = "logs"):
        """Initialize the agent logger.

        Args:
            agent_name: Name of the agent (e.g., "research_agent", "report_agent")
            session_id: Optional session identifier for tracking
            base_log_dir: Base directory for log files
        """
        self.agent_name = agent_name
        self.session_id = session_id or str(uuid.uuid4())
        self.base_log_dir = Path(base_log_dir)

        # Create agent-specific log directory
        self.log_dir = self.base_log_dir / self.agent_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize structured logger for this agent
        self.structured_logger = StructuredLogger(
            name=f"{self.agent_name}_logger",
            log_dir=self.log_dir
        )

        # Activity tracking
        self.activities: list[Dict[str, Any]] = []

    def log_activity(self,
                    agent_name: str,
                    activity_type: str,
                    stage: str,
                    input_data: Optional[Dict[str, Any]] = None,
                    output_data: Optional[Dict[str, Any]] = None,
                    tool_used: Optional[str] = None,
                    tool_result: Optional[Any] = None,
                    execution_time: Optional[float] = None,
                    error: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a general agent activity.

        Args:
            agent_name: Name of the agent performing the activity
            activity_type: Type of activity (e.g., "search", "analysis", "generation")
            stage: Stage or phase of the activity
            input_data: Input data for the activity
            output_data: Output data from the activity
            tool_used: Name of tool used (if applicable)
            tool_result: Result from tool execution
            execution_time: Time taken to complete the activity
            error: Error message if activity failed
            metadata: Additional metadata about the activity
        """
        timestamp = datetime.now().isoformat()

        activity_entry = {
            "timestamp": timestamp,
            "session_id": self.session_id,
            "agent_name": agent_name,
            "activity_type": activity_type,
            "stage": stage,
            "input_data": input_data,
            "output_data": output_data,
            "tool_used": tool_used,
            "tool_result": tool_result,
            "execution_time": execution_time,
            "error": error,
            "metadata": metadata or {},
            "success": error is None
        }

        self.activities.append(activity_entry)

        # Log to structured logger
        log_level = "error" if error else "info"
        getattr(self.structured_logger, log_level)(
            f"{agent_name} activity: {activity_type}",
            event_type="agent_activity",
            **activity_entry
        )

        # Also write to activity log file
        self._write_activity_log(activity_entry)

    def _write_activity_log(self, activity_entry: Dict[str, Any]) -> None:
        """Write activity entry to activity log file."""
        try:
            activity_log_file = self.log_dir / f"activities_{self.session_id}.jsonl"

            with open(activity_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(activity_entry) + '\n')

        except Exception as e:
            # Don't let logging errors break the main flow
            print(f"Warning: Failed to write activity log: {e}")

    def log_agent_initialization(self,
                                agent_name: str,
                                config: Dict[str, Any]) -> None:
        """Log agent initialization.

        Args:
            agent_name: Name of the agent being initialized
            config: Configuration parameters for the agent
        """
        self.structured_logger.info(f"{agent_name} initialized",
                                   event_type="agent_initialization",
                                   agent_name=agent_name,
                                   session_id=self.session_id,
                                   config=config)

    def log_session_start(self, session_data: Dict[str, Any]) -> None:
        """Log the start of a new session.

        Args:
            session_data: Information about the session
        """
        self.structured_logger.info(f"{self.agent_name} session started",
                                   event_type="session_start",
                                   agent_name=self.agent_name,
                                   session_id=self.session_id,
                                   session_data=session_data)

    def log_session_end(self, session_summary: Dict[str, Any]) -> None:
        """Log the end of a session.

        Args:
            session_summary: Summary of session activities and results
        """
        self.structured_logger.info(f"{self.agent_name} session ended",
                                   event_type="session_end",
                                   agent_name=self.agent_name,
                                   session_id=self.session_id,
                                   total_activities=len(self.activities),
                                   session_summary=session_summary)

    def get_activities(self,
                      activity_type: Optional[str] = None,
                      limit: Optional[int] = None) -> list[Dict[str, Any]]:
        """Get activities with optional filtering.

        Args:
            activity_type: Filter by specific activity type
            limit: Maximum number of activities to return

        Returns:
            List of activity entries
        """
        activities = self.activities

        if activity_type:
            activities = [a for a in activities if a.get("activity_type") == activity_type]

        if limit:
            activities = activities[-limit:]

        return activities

    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session.

        Returns:
            Dictionary containing session statistics and summary
        """
        successful_activities = len([a for a in self.activities if a.get("success", False)])
        failed_activities = len(self.activities) - successful_activities

        # Calculate average execution time for successful activities
        execution_times = [a.get("execution_time", 0) for a in self.activities
                          if a.get("execution_time") is not None and a.get("success", False)]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0.0

        # Get activity type counts
        activity_types = {}
        for activity in self.activities:
            activity_type = activity.get("activity_type", "unknown")
            activity_types[activity_type] = activity_types.get(activity_type, 0) + 1

        return {
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "total_activities": len(self.activities),
            "successful_activities": successful_activities,
            "failed_activities": failed_activities,
            "success_rate": successful_activities / len(self.activities) if self.activities else 0.0,
            "average_execution_time": avg_execution_time,
            "activity_type_counts": activity_types,
            "log_directory": str(self.log_dir)
        }

    def export_session_data(self, file_path: Optional[str] = None) -> str:
        """Export all session data to a JSON file.

        Args:
            file_path: Optional custom file path. If not provided, generates one.

        Returns:
            Path to the exported file
        """
        if not file_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = str(self.log_dir / f"session_export_{self.session_id}_{timestamp}.json")

        session_data = {
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "export_timestamp": datetime.now().isoformat(),
            "session_summary": self.get_session_summary(),
            "activities": self.activities
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2)

        self.structured_logger.info(f"Session data exported",
                                   event_type="session_export",
                                   file_path=file_path,
                                   activities_count=len(self.activities))

        return file_path

    def cleanup(self) -> None:
        """Clean up resources and close loggers."""
        if hasattr(self.structured_logger, 'cleanup'):
            self.structured_logger.cleanup()