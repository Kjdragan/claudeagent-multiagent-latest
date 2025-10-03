"""Agent activity logging system for debugging and monitoring multi-agent workflows.

This module provides comprehensive logging of agent activities, including:
- Agent initialization and configuration
- Tool usage and results
- Conversation flows between agents
- Input/output tracking for debugging
- Performance metrics and timing
"""

import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
import asyncio
import threading


@dataclass
class AgentActivity:
    """Represents a single agent activity event."""
    timestamp: str
    agent_name: str
    activity_type: str
    stage: str
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    tool_used: Optional[str] = None
    tool_result: Optional[Any] = None
    execution_time: Optional[float] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AgentLogger:
    """Centralized logger for tracking all agent activities in a research session."""

    def __init__(self, session_id: str, base_log_dir: str = "KEVIN/sessions"):
        self.session_id = session_id
        self.base_log_dir = Path(base_log_dir)
        self.session_dir = self.base_log_dir / session_id
        self.agent_logs_dir = self.session_dir / "agent_logs"

        # Create directories
        self.agent_logs_dir.mkdir(parents=True, exist_ok=True)

        # Activity storage
        self.activities: List[AgentActivity] = []
        self.conversation_flow: List[Dict[str, Any]] = []

        # Individual agent log files
        self.agent_log_files: Dict[str, Path] = {}

        # Flow log file (complete conversation)
        self.flow_log_file = self.agent_logs_dir / "conversation_flow.jsonl"

        # Summary log file
        self.summary_log_file = self.agent_logs_dir / "agent_summary.json"

        # Thread safety
        self._lock = threading.Lock()

        # Session metadata
        self.session_start_time = datetime.now()
        self.session_metadata = {
            "session_id": session_id,
            "start_time": self.session_start_time.isoformat(),
            "total_activities": 0,
            "agents_involved": [],
            "tools_used": [],
            "errors": []
        }

        self._write_session_metadata()

    def _write_session_metadata(self):
        """Write initial session metadata."""
        with open(self.summary_log_file, 'w') as f:
            json.dump({
                **self.session_metadata,
                "agents_involved": list(self.session_metadata["agents_involved"]),
                "tools_used": list(self.session_metadata["tools_used"])
            }, f, indent=2)

    def _update_session_metadata(self):
        """Update session metadata with current statistics."""
        self.session_metadata.update({
            "total_activities": len(self.activities),
            "last_activity": datetime.now().isoformat()
        })

        with open(self.summary_log_file, 'w') as f:
            json.dump(self.session_metadata, f, indent=2)

    def _get_agent_log_file(self, agent_name: str) -> Path:
        """Get or create log file for a specific agent."""
        if agent_name not in self.agent_log_files:
            self.agent_log_files[agent_name] = self.agent_logs_dir / f"{agent_name}.jsonl"
        return self.agent_log_files[agent_name]

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
                    metadata: Optional[Dict[str, Any]] = None):
        """Log an agent activity."""

        timestamp = datetime.now().isoformat()

        # Create activity record
        activity = AgentActivity(
            timestamp=timestamp,
            agent_name=agent_name,
            activity_type=activity_type,
            stage=stage,
            input_data=input_data,
            output_data=output_data,
            tool_used=tool_used,
            tool_result=tool_result,
            execution_time=execution_time,
            error=error,
            metadata=metadata or {}
        )

        with self._lock:
            self.activities.append(activity)

            # Update session metadata
            if agent_name not in self.session_metadata["agents_involved"]:
                self.session_metadata["agents_involved"].append(agent_name)
            if tool_used and tool_used not in self.session_metadata["tools_used"]:
                self.session_metadata["tools_used"].append(tool_used)
            if error:
                self.session_metadata["errors"].append({
                    "timestamp": timestamp,
                    "agent": agent_name,
                    "error": error
                })

            # Write to agent-specific log file
            agent_log_file = self._get_agent_log_file(agent_name)
            with open(agent_log_file, 'a') as f:
                f.write(json.dumps(asdict(activity), default=str) + '\n')

            # Write to conversation flow log
            flow_entry = {
                "timestamp": timestamp,
                "agent": agent_name,
                "activity_type": activity_type,
                "stage": stage,
                "summary": f"{agent_name} performed {activity_type} in {stage}",
                "input_preview": self._create_preview(input_data) if input_data else None,
                "output_preview": self._create_preview(output_data) if output_data else None,
                "tool_used": tool_used,
                "execution_time": execution_time,
                "has_error": bool(error)
            }

            with open(self.flow_log_file, 'a') as f:
                f.write(json.dumps(flow_entry, default=str) + '\n')

            # Update summary metadata
            self._update_session_metadata()

    def _create_preview(self, data: Any, max_length: int = 200) -> str:
        """Create a preview string of data for logging."""
        if data is None:
            return None

        try:
            if isinstance(data, str):
                return data[:max_length] + "..." if len(data) > max_length else data
            elif isinstance(data, dict):
                preview = {k: str(v)[:100] for k, v in list(data.items())[:5]}
                return json.dumps(preview)
            elif isinstance(data, list):
                return f"List with {len(data)} items"
            else:
                return str(data)[:max_length]
        except Exception:
            return f"[{type(data).__name__} - preview failed]"

    def log_agent_initialization(self, agent_name: str, config: Dict[str, Any]):
        """Log agent initialization."""
        self.log_activity(
            agent_name=agent_name,
            activity_type="agent_initialization",
            stage="initialization",
            input_data={"config": config},
            output_data={"status": "initialized"},
            metadata={"event": "agent_created"}
        )

    def log_tool_usage(self,
                      agent_name: str,
                      tool_name: str,
                      tool_input: Dict[str, Any],
                      tool_result: Any,
                      execution_time: float,
                      stage: str):
        """Log tool usage by an agent."""
        self.log_activity(
            agent_name=agent_name,
            activity_type="tool_usage",
            stage=stage,
            input_data={"tool_input": tool_input},
            output_data={"tool_result": tool_result},
            tool_used=tool_name,
            tool_result=tool_result,
            execution_time=execution_time
        )

    def log_query_response(self,
                          agent_name: str,
                          query: str,
                          response: Any,
                          execution_time: float,
                          stage: str):
        """Log agent query and response."""
        self.log_activity(
            agent_name=agent_name,
            activity_type="query_response",
            stage=stage,
            input_data={"query": query},
            output_data={"response_type": type(response).__name__, "response": str(response)[:1000] if response else None},
            execution_time=execution_time
        )

    def log_error(self,
                 agent_name: str,
                 error: Exception,
                 context: Dict[str, Any],
                 stage: str):
        """Log an error that occurred during agent execution."""
        self.log_activity(
            agent_name=agent_name,
            activity_type="error",
            stage=stage,
            input_data=context,
            error=str(error),
            metadata={
                "error_type": type(error).__name__,
                "traceback": traceback.format_exc()
            }
        )

    def log_stage_transition(self,
                           from_stage: str,
                           to_stage: str,
                           agent_name: str,
                           context: Optional[Dict[str, Any]] = None):
        """Log workflow stage transitions."""
        self.log_activity(
            agent_name=agent_name,
            activity_type="stage_transition",
            stage=to_stage,
            input_data={"from_stage": from_stage, "to_stage": to_stage, "context": context},
            metadata={"event": "workflow_transition"}
        )

    def log_work_product_transfer(self,
                                from_agent: str,
                                to_agent: str,
                                work_product: Dict[str, Any],
                                stage: str):
        """Log transfer of work products between agents."""
        self.log_activity(
            agent_name=to_agent,
            activity_type="work_product_received",
            stage=stage,
            input_data={"from_agent": from_agent, "work_product": work_product},
            metadata={"event": "agent_handoff"}
        )

        # Also log the sending side
        self.log_activity(
            agent_name=from_agent,
            activity_type="work_product_sent",
            stage=stage,
            output_data={"to_agent": to_agent, "work_product": work_product},
            metadata={"event": "agent_handoff"}
        )

    def get_agent_activities(self, agent_name: Optional[str] = None) -> List[AgentActivity]:
        """Get activities for a specific agent or all agents."""
        with self._lock:
            if agent_name:
                return [a for a in self.activities if a.agent_name == agent_name]
            return self.activities.copy()

    def get_conversation_flow(self) -> List[Dict[str, Any]]:
        """Get the complete conversation flow."""
        if not self.flow_log_file.exists():
            return []

        flow = []
        try:
            with open(self.flow_log_file, 'r') as f:
                for line in f:
                    if line.strip():
                        flow.append(json.loads(line))
        except Exception as e:
            print(f"Error reading conversation flow: {e}")

        return flow

    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the session activities."""
        with self._lock:
            return {
                **self.session_metadata,
                "total_activities": len(self.activities),
                "session_duration": (datetime.now() - self.session_start_time).total_seconds(),
                "error_count": len(self.session_metadata["errors"])
            }

    def export_debug_report(self, output_path: Optional[str] = None) -> str:
        """Export a comprehensive debug report."""
        if not output_path:
            output_path = self.agent_logs_dir / f"debug_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        report = {
            "session_summary": self.get_session_summary(),
            "conversation_flow": self.get_conversation_flow(),
            "agent_activities": [asdict(a) for a in self.activities],
            "errors": self.session_metadata["errors"],
            "export_timestamp": datetime.now().isoformat()
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        return str(output_path)

    def finalize_session(self):
        """Finalize the session and write final metadata."""
        self.session_metadata.update({
            "end_time": datetime.now().isoformat(),
            "total_duration": (datetime.now() - self.session_start_time).total_seconds(),
            "final_activity_count": len(self.activities)
        })

        self._update_session_metadata()

        # Create final summary
        final_summary = self.get_session_summary()
        summary_file = self.agent_logs_dir / "final_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(final_summary, f, indent=2)


class AgentLoggerFactory:
    """Factory for creating and managing agent loggers."""

    _loggers: Dict[str, AgentLogger] = {}
    _lock = threading.Lock()

    @classmethod
    def get_logger(cls, session_id: str, base_log_dir: str = "KEVIN/sessions") -> AgentLogger:
        """Get or create an agent logger for a session."""
        with cls._lock:
            if session_id not in cls._loggers:
                cls._loggers[session_id] = AgentLogger(session_id, base_log_dir)
            return cls._loggers[session_id]

    @classmethod
    def remove_logger(cls, session_id: str):
        """Remove a logger from the factory."""
        with cls._lock:
            if session_id in cls._loggers:
                del cls._loggers[session_id]