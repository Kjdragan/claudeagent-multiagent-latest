"""
Enhanced Logging Infrastructure for Multi-Agent Research System

This module provides comprehensive logging infrastructure with structured logging,
performance monitoring, and observability features based on the redesign plan
specifications.

Key Features:
- Structured logging with correlation IDs
- Performance metrics tracking
- Agent execution logging
- Flow adherence compliance logging
- Rich message processing logs
- Error tracking and analysis
- Export capabilities for monitoring

Based on Redesign Plan PLUS SDK Implementation (October 13, 2025)
"""

import os
import json
import time
import uuid
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import logging.handlers
import threading
from contextlib import contextmanager

import structlog
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..config.sdk_config import get_sdk_config, LogLevel


class LogLevel(str, Enum):
    """Enhanced log levels with additional agent-specific levels."""
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogCategory(str, Enum):
    """Log categories for better organization."""
    AGENT = "agent"
    TOOL = "tool"
    WORKFLOW = "workflow"
    QUALITY = "quality"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"
    ERROR = "error"
    SYSTEM = "system"


class AgentEventType(str, Enum):
    """Agent event types for structured logging."""
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    TASK_START = "task_start"
    TASK_END = "task_end"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    QUALITY_ASSESSMENT = "quality_assessment"
    FLOW_COMPLIANCE = "flow_compliance"
    MESSAGE_PROCESSED = "message_processed"
    HANDOFF = "handoff"


@dataclass
class LogEvent:
    """Structured log event with rich metadata."""

    # Basic event information
    timestamp: datetime
    level: LogLevel
    category: LogCategory
    event_type: AgentEventType
    message: str

    # Context information
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    agent_type: Optional[str] = None
    task_id: Optional[str] = None
    correlation_id: Optional[str] = None

    # Performance metrics
    execution_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None

    # Quality and compliance
    quality_score: Optional[float] = None
    compliance_status: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert log event to dictionary."""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        return result


@dataclass
class PerformanceMetrics:
    """Performance metrics for agent execution."""

    session_id: str
    agent_id: str
    agent_type: str

    # Timing metrics
    session_start_time: datetime
    task_start_time: Optional[datetime] = None
    last_activity_time: Optional[datetime] = None

    # Execution metrics
    total_tasks_completed: int = 0
    total_tools_executed: int = 0
    total_errors: int = 0

    # Performance metrics
    average_task_duration_ms: float = 0.0
    average_tool_duration_ms: float = 0.0
    peak_memory_usage_mb: float = 0.0

    # Quality metrics
    average_quality_score: float = 0.0
    compliance_violations: int = 0

    # Resource usage
    total_tokens_used: int = 0
    total_api_calls: int = 0

    def update_task_completion(self, duration_ms: float, quality_score: Optional[float] = None):
        """Update metrics after task completion."""
        self.total_tasks_completed += 1
        self.last_activity_time = datetime.now(timezone.utc)

        # Update average duration
        total_duration = self.average_task_duration_ms * (self.total_tasks_completed - 1) + duration_ms
        self.average_task_duration_ms = total_duration / self.total_tasks_completed

        # Update quality score if provided
        if quality_score is not None:
            total_quality = self.average_quality_score * (self.total_tasks_completed - 1) + quality_score
            self.average_quality_score = total_quality / self.total_tasks_completed

    def update_tool_execution(self, duration_ms: float):
        """Update metrics after tool execution."""
        self.total_tools_executed += 1

        # Update average tool duration
        total_duration = self.average_tool_duration_ms * (self.total_tools_executed - 1) + duration_ms
        self.average_tool_duration_ms = total_duration / self.total_tools_executed

    def update_error(self):
        """Update error metrics."""
        self.total_errors += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        result = asdict(self)
        for key, value in result.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
        return result


class EnhancedLogger:
    """Enhanced logger with structured logging and performance tracking."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self._session_id: Optional[str] = None
        self._agent_id: Optional[str] = None
        self._agent_type: Optional[str] = None
        self._correlation_id: Optional[str] = None

        # Performance tracking
        self._performance_metrics: Dict[str, PerformanceMetrics] = {}
        self._current_task_start_time: Optional[float] = None

        # Setup structured logging
        self._setup_structured_logging()

        # Setup rich console for terminal output
        self._console = Console()

        # Setup file logging
        self._setup_file_logging()

        # Thread safety
        self._lock = threading.Lock()

    def _setup_structured_logging(self):
        """Setup structured logging with structlog."""
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                self._add_custom_fields,
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        self._structlog_logger = structlog.get_logger(self.name)

    def _setup_file_logging(self):
        """Setup file logging with rotation."""
        sdk_config = get_sdk_config()
        logs_dir = Path(sdk_config.logs_directory or "KEVIN/logs")
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Create session-based log file
        if self._session_id:
            log_file = logs_dir / f"{self.name}_{self._session_id}.json"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = logs_dir / f"{self.name}_{timestamp}.json"

        # Setup rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=5
        )

        file_handler.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)

        # Add handler to root logger
        root_logger = logging.getLogger(self.name)
        root_logger.addHandler(file_handler)
        root_logger.setLevel(logging.DEBUG)

        self._log_file = log_file

    def _add_custom_fields(self, logger, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Add custom fields to log events."""
        # Add session context
        if self._session_id:
            event_dict["session_id"] = self._session_id
        if self._agent_id:
            event_dict["agent_id"] = self._agent_id
        if self._agent_type:
            event_dict["agent_type"] = self._agent_type
        if self._correlation_id:
            event_dict["correlation_id"] = self._correlation_id

        return event_dict

    def set_session_context(self, session_id: str, agent_id: Optional[str] = None,
                           agent_type: Optional[str] = None):
        """Set session context for logging."""
        with self._lock:
            self._session_id = session_id
            self._agent_id = agent_id
            self._agent_type = agent_type
            self._correlation_id = str(uuid.uuid4())

            # Initialize performance metrics
            if session_id and agent_id:
                self._performance_metrics[session_id] = PerformanceMetrics(
                    session_id=session_id,
                    agent_id=agent_id,
                    agent_type=agent_type or "unknown",
                    session_start_time=datetime.now(timezone.utc)
                )

    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID for request tracking."""
        self._correlation_id = correlation_id

    def log_event(self, level: LogLevel, category: LogCategory, event_type: AgentEventType,
                  message: str, **kwargs) -> LogEvent:
        """Log a structured event."""
        # Create log event
        event = LogEvent(
            timestamp=datetime.now(timezone.utc),
            level=level,
            category=category,
            event_type=event_type,
            message=message,
            session_id=self._session_id,
            agent_id=self._agent_id,
            agent_type=self._agent_type,
            correlation_id=self._correlation_id,
            **kwargs
        )

        # Log to structured logger
        log_method = getattr(self._structlog_logger, level.value.lower())
        log_method(
            message,
            category=category.value,
            event_type=event_type.value,
            **event.metadata
        )

        # Log to rich console if enabled
        sdk_config = get_sdk_config()
        if sdk_config.debug_mode or level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            self._log_to_console(event)

        return event

    def _log_to_console(self, event: LogEvent):
        """Log event to rich console."""
        # Create color-coded level indicator
        level_colors = {
            LogLevel.TRACE: "dim white",
            LogLevel.DEBUG: "blue",
            LogLevel.INFO: "green",
            LogLevel.WARNING: "yellow",
            LogLevel.ERROR: "red",
            LogLevel.CRITICAL: "bold red"
        }

        level_color = level_colors.get(event.level, "white")
        category_color = {
            LogCategory.AGENT: "cyan",
            LogCategory.TOOL: "magenta",
            LogCategory.WORKFLOW: "blue",
            LogCategory.QUALITY: "green",
            LogCategory.PERFORMANCE: "yellow",
            LogCategory.COMPLIANCE: "purple",
            LogCategory.ERROR: "red",
            LogCategory.SYSTEM: "white"
        }.get(event.category, "white")

        # Format message
        timestamp = event.timestamp.strftime("%H:%M:%S")
        level_str = f"[{level_color}]{event.level.value}[/{level_color}]"
        category_str = f"[{category_color}]{event.category.value}[/{category_color}]"
        event_type_str = f"[dim]{event.event_type.value}[/dim]"

        message_parts = [timestamp, level_str, category_str, event_type_str]
        if event.agent_type:
            message_parts.append(f"[cyan]{event.agent_type}[/cyan]")
        if event.agent_id:
            message_parts.append(f"[dim]({event.agent_id[:8]})[/dim]")

        message = " ".join(message_parts) + f": {event.message}"

        # Add performance information if available
        if event.execution_time_ms:
            message += f" [yellow]({event.execution_time_ms:.0f}ms)[/yellow]"
        if event.quality_score:
            message += f" [green]({event.quality_score:.1f}/10)[/green]"

        self._console.print(message)

    @contextmanager
    def task_timer(self, task_name: str, **metadata):
        """Context manager for timing task execution."""
        start_time = time.time()
        task_id = str(uuid.uuid4())

        self.log_event(
            LogLevel.INFO,
            LogCategory.AGENT,
            AgentEventType.TASK_START,
            f"Starting task: {task_name}",
            task_id=task_id,
            **metadata
        )

        self._current_task_start_time = start_time

        try:
            yield task_id
            duration_ms = (time.time() - start_time) * 1000

            self.log_event(
                LogLevel.INFO,
                LogCategory.AGENT,
                AgentEventType.TASK_END,
                f"Completed task: {task_name}",
                task_id=task_id,
                execution_time_ms=duration_ms,
                **metadata
            )

            # Update performance metrics
            if self._session_id and self._session_id in self._performance_metrics:
                self._performance_metrics[self._session_id].update_task_completion(duration_ms)

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            self.log_event(
                LogLevel.ERROR,
                LogCategory.ERROR,
                AgentEventType.ERROR,
                f"Task failed: {task_name} - {str(e)}",
                task_id=task_id,
                execution_time_ms=duration_ms,
                error_details={"error_type": type(e).__name__, "error_message": str(e)},
                **metadata
            )

            # Update error metrics
            if self._session_id and self._session_id in self._performance_metrics:
                self._performance_metrics[self._session_id].update_error()

            raise
        finally:
            self._current_task_start_time = None

    @contextmanager
    def tool_timer(self, tool_name: str, **parameters):
        """Context manager for timing tool execution."""
        start_time = time.time()

        self.log_event(
            LogLevel.DEBUG,
            LogCategory.TOOL,
            AgentEventType.TOOL_CALL,
            f"Executing tool: {tool_name}",
            tool_name=tool_name,
            tool_parameters=parameters
        )

        try:
            yield
            duration_ms = (time.time() - start_time) * 1000

            self.log_event(
                LogLevel.DEBUG,
                LogCategory.TOOL,
                AgentEventType.TOOL_RESULT,
                f"Tool completed: {tool_name}",
                tool_name=tool_name,
                execution_time_ms=duration_ms
            )

            # Update performance metrics
            if self._session_id and self._session_id in self._performance_metrics:
                self._performance_metrics[self._session_id].update_tool_execution(duration_ms)

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            self.log_event(
                LogLevel.ERROR,
                LogCategory.TOOL,
                AgentEventType.ERROR,
                f"Tool failed: {tool_name} - {str(e)}",
                tool_name=tool_name,
                execution_time_ms=duration_ms,
                error_details={"error_type": type(e).__name__, "error_message": str(e)},
                tool_parameters=parameters
            )

            raise

    def log_quality_assessment(self, content: str, quality_score: float,
                             dimensions: Dict[str, float], **metadata):
        """Log quality assessment results."""
        self.log_event(
            LogLevel.INFO,
            LogCategory.QUALITY,
            AgentEventType.QUALITY_ASSESSMENT,
            f"Quality assessment completed: {quality_score:.1f}/10",
            quality_score=quality_score,
            content_length=len(content),
            **{**metadata, **dimensions}
        )

        # Update performance metrics
        if self._session_id and self._session_id in self._performance_metrics:
            self._performance_metrics[self._session_id].update_task_completion(0, quality_score)

    def log_flow_compliance(self, compliance_status: str, violations: List[str] = None,
                          enforcement_actions: List[str] = None, **metadata):
        """Log flow adherence compliance results."""
        self.log_event(
            LogLevel.INFO if compliance_status == "compliant" else LogLevel.WARNING,
            LogCategory.COMPLIANCE,
            AgentEventType.FLOW_COMPLIANCE,
            f"Flow compliance: {compliance_status}",
            compliance_status=compliance_status,
            violations=violations or [],
            enforcement_actions=enforcement_actions or [],
            **metadata
        )

        # Update compliance metrics
        if self._session_id and self._session_id in self._performance_metrics:
            if compliance_status != "compliant":
                self._performance_metrics[self._session_id].compliance_violations += 1

    def log_agent_handoff(self, from_agent: str, to_agent: str, reason: str, **metadata):
        """Log agent handoff events."""
        self.log_event(
            LogLevel.INFO,
            LogCategory.WORKFLOW,
            AgentEventType.HANDOFF,
            f"Agent handoff: {from_agent} -> {to_agent}",
            from_agent=from_agent,
            to_agent=to_agent,
            handoff_reason=reason,
            **metadata
        )

    def get_performance_metrics(self, session_id: Optional[str] = None) -> Optional[PerformanceMetrics]:
        """Get performance metrics for a session."""
        if session_id is None:
            session_id = self._session_id

        if session_id and session_id in self._performance_metrics:
            return self._performance_metrics[session_id]

        return None

    def get_session_summary(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get session summary with performance metrics."""
        if session_id is None:
            session_id = self._session_id

        if not session_id or session_id not in self._performance_metrics:
            return {}

        metrics = self._performance_metrics[session_id]

        # Calculate session duration
        now = datetime.now(timezone.utc)
        session_duration_ms = (now - metrics.session_start_time).total_seconds() * 1000

        return {
            "session_id": session_id,
            "agent_id": metrics.agent_id,
            "agent_type": metrics.agent_type,
            "session_duration_ms": session_duration_ms,
            "total_tasks_completed": metrics.total_tasks_completed,
            "total_tools_executed": metrics.total_tools_executed,
            "total_errors": metrics.total_errors,
            "average_task_duration_ms": metrics.average_task_duration_ms,
            "average_tool_duration_ms": metrics.average_tool_duration_ms,
            "average_quality_score": metrics.average_quality_score,
            "compliance_violations": metrics.compliance_violations,
            "total_tokens_used": metrics.total_tokens_used,
            "total_api_calls": metrics.total_api_calls,
            "tasks_per_minute": (metrics.total_tasks_completed / (session_duration_ms / 60000)) if session_duration_ms > 0 else 0,
            "error_rate": (metrics.total_errors / max(metrics.total_tasks_completed, 1)) * 100,
            "compliance_rate": ((metrics.total_tasks_completed - metrics.compliance_violations) / max(metrics.total_tasks_completed, 1)) * 100
        }

    def display_performance_dashboard(self):
        """Display a performance dashboard in the console."""
        if not self._session_id or self._session_id not in self._performance_metrics:
            self._console.print("[yellow]No performance data available[/yellow]")
            return

        summary = self.get_session_summary()

        # Create dashboard table
        table = Table(title=f"Performance Dashboard - {summary['agent_type']} ({summary['agent_id'][:8]})")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        # Session metrics
        table.add_row("Session Duration", f"{summary['session_duration_ms'] / 1000:.1f}s")
        table.add_row("Tasks Completed", str(summary['total_tasks_completed']))
        table.add_row("Tools Executed", str(summary['total_tools_executed']))
        table.add_row("Total Errors", str(summary['total_errors']))

        # Performance metrics
        table.add_row("Avg Task Duration", f"{summary['average_task_duration_ms']:.0f}ms")
        table.add_row("Avg Tool Duration", f"{summary['average_tool_duration_ms']:.0f}ms")
        table.add_row("Tasks/Minute", f"{summary['tasks_per_minute']:.1f}")

        # Quality metrics
        table.add_row("Avg Quality Score", f"{summary['average_quality_score']:.1f}/10")
        table.add_row("Error Rate", f"{summary['error_rate']:.1f}%")
        table.add_row("Compliance Rate", f"{summary['compliance_rate']:.1f}%")

        # Resource metrics
        table.add_row("Tokens Used", str(summary['total_tokens_used']))
        table.add_row("API Calls", str(summary['total_api_calls']))

        self._console.print(table)

    def export_logs(self, export_path: Union[str, Path], format: str = "json") -> bool:
        """Export logs to file."""
        try:
            export_path = Path(export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)

            if format.lower() == "json":
                # Export performance metrics
                metrics_data = {
                    session_id: metrics.to_dict()
                    for session_id, metrics in self._performance_metrics.items()
                }

                # Export session summaries
                summaries_data = {}
                for session_id in self._performance_metrics:
                    summaries_data[session_id] = self.get_session_summary(session_id)

                export_data = {
                    "export_timestamp": datetime.now(timezone.utc).isoformat(),
                    "logger_name": self.name,
                    "performance_metrics": metrics_data,
                    "session_summaries": summaries_data
                }

                with open(export_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)

            self._console.print(f"[green]Logs exported to: {export_path}[/green]")
            return True

        except Exception as e:
            self.log_event(
                LogLevel.ERROR,
                LogCategory.SYSTEM,
                AgentEventType.ERROR,
                f"Failed to export logs: {str(e)}",
                export_path=str(export_path),
                format=format,
                error_details={"error_type": type(e).__name__, "error_message": str(e)}
            )
            return False


# Global logger registry
_loggers: Dict[str, EnhancedLogger] = {}


def get_enhanced_logger(name: str) -> EnhancedLogger:
    """Get or create an enhanced logger."""
    if name not in _loggers:
        _loggers[name] = EnhancedLogger(name)
    return _loggers[name]


def setup_logging_for_session(session_id: str, agent_id: str, agent_type: str) -> EnhancedLogger:
    """Setup logging for a specific session."""
    logger_name = f"agent_{agent_type}"
    logger = get_enhanced_logger(logger_name)
    logger.set_session_context(session_id, agent_id, agent_type)
    return logger


def get_session_performance_summary(session_id: str) -> Dict[str, Any]:
    """Get performance summary for all agents in a session."""
    summaries = {}
    for logger in _loggers.values():
        summary = logger.get_session_summary(session_id)
        if summary:
            summaries[summary['agent_type']] = summary
    return summaries


def export_all_logs(export_dir: Union[str, Path]) -> bool:
    """Export logs from all loggers."""
    try:
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for logger_name, logger in _loggers.items():
            export_file = export_dir / f"{logger_name}_{timestamp}.json"
            logger.export_logs(export_file)

        return True

    except Exception as e:
        print(f"Failed to export logs: {e}")
        return False