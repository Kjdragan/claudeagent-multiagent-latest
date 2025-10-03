"""
Structured Logger for Multi-Agent Research System

Provides JSON-formatted logging with correlation IDs and comprehensive
event tracking for agent activities, tool execution, and workflow monitoring.
"""

import json
import logging
import logging.handlers
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from contextvars import ContextVar

# Context variable for correlation ID
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs JSON-structured log messages."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        # Get correlation ID from context
        corr_id = correlation_id.get()

        # Build structured log entry
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "correlation_id": corr_id,
            "thread": record.thread,
            "process": record.process
        }

        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }

        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                'filename', 'module', 'lineno', 'funcName', 'created',
                'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'getMessage', 'exc_info',
                'exc_text', 'stack_info'
            }:
                log_entry[key] = value

        return json.dumps(log_entry, default=str)


class StructuredLogger:
    """Enhanced logger with structured output and correlation tracking."""

    def __init__(
        self,
        name: str,
        log_dir: Optional[Path] = None,
        log_level: str = "INFO",
        enable_console: bool = True,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ):
        """Initialize structured logger."""
        self.name = name
        self.logger = logging.getLogger(f"multi_agent.{name}")
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Clear any existing handlers
        self.logger.handlers.clear()

        # Create formatters
        json_formatter = StructuredFormatter()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Add console handler if enabled
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

        # Add file handler if log directory specified
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_dir / f"{name}.json",
                maxBytes=max_file_size,
                backupCount=backup_count
            )
            file_handler.setFormatter(json_formatter)
            self.logger.addHandler(file_handler)

    def set_correlation_id(self, session_id: str) -> None:
        """Set correlation ID for current context."""
        correlation_id.set(session_id)

    def clear_correlation_id(self) -> None:
        """Clear correlation ID from current context."""
        correlation_id.set(None)

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with structured data."""
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log info message with structured data."""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with structured data."""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log error message with structured data."""
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        """Log critical message with structured data."""
        self._log(logging.CRITICAL, message, **kwargs)

    def _log(self, level: int, message: str, **kwargs) -> None:
        """Internal logging method with structured data."""
        # Add structured data as extra fields
        extra = {}
        for key, value in kwargs.items():
            # Sanitize key names for logging
            clean_key = key.replace('.', '_').replace('-', '_')
            extra[clean_key] = value

        self.logger.log(level, message, extra=extra)

    def log_workflow_start(
        self,
        session_id: str,
        workflow_type: str,
        **metadata
    ) -> None:
        """Log workflow start event."""
        self.set_correlation_id(session_id)
        self.info(
            f"Workflow started: {workflow_type}",
            event_type="workflow_start",
            session_id=session_id,
            workflow_type=workflow_type,
            **metadata
        )

    def log_workflow_complete(
        self,
        session_id: str,
        workflow_type: str,
        duration: float,
        **metadata
    ) -> None:
        """Log workflow completion event."""
        self.info(
            f"Workflow completed: {workflow_type}",
            event_type="workflow_complete",
            session_id=session_id,
            workflow_type=workflow_type,
            duration_seconds=duration,
            **metadata
        )
        self.clear_correlation_id()

    def log_agent_activity(
        self,
        agent_name: str,
        activity_type: str,
        session_id: str,
        **metadata
    ) -> None:
        """Log agent activity event."""
        self.set_correlation_id(session_id)
        self.info(
            f"Agent activity: {agent_name} - {activity_type}",
            event_type="agent_activity",
            agent_name=agent_name,
            activity_type=activity_type,
            session_id=session_id,
            **metadata
        )

    def log_tool_execution(
        self,
        tool_name: str,
        tool_use_id: str,
        session_id: str,
        execution_time: Optional[float] = None,
        success: Optional[bool] = None,
        **metadata
    ) -> None:
        """Log tool execution event."""
        self.set_correlation_id(session_id)
        log_data = {
            "event_type": "tool_execution",
            "tool_name": tool_name,
            "tool_use_id": tool_use_id,
            "session_id": session_id,
            **metadata
        }

        if execution_time is not None:
            log_data["execution_time_seconds"] = execution_time

        if success is not None:
            log_data["success"] = success
            level = logging.INFO if success else logging.ERROR
            self._log(level, f"Tool executed: {tool_name}", **log_data)
        else:
            self.info(f"Tool executing: {tool_name}", **log_data)

    def log_session_transition(
        self,
        session_id: str,
        from_stage: str,
        to_stage: str,
        **metadata
    ) -> None:
        """Log session stage transition."""
        self.set_correlation_id(session_id)
        self.info(
            f"Stage transition: {from_stage} -> {to_stage}",
            event_type="stage_transition",
            session_id=session_id,
            from_stage=from_stage,
            to_stage=to_stage,
            **metadata
        )

    def log_error_with_context(
        self,
        error: Exception,
        session_id: str,
        context: Dict[str, Any],
        **metadata
    ) -> None:
        """Log error with full context information."""
        self.set_correlation_id(session_id)
        self.error(
            f"Error occurred: {str(error)}",
            event_type="error",
            session_id=session_id,
            error_type=type(error).__name__,
            error_message=str(error),
            context=context,
            **metadata
        )


# Global logger registry
_loggers: Dict[str, StructuredLogger] = {}


def get_logger(
    name: str,
    log_dir: Optional[Path] = None,
    log_level: str = "INFO",
    enable_console: bool = True
) -> StructuredLogger:
    """Get or create a structured logger instance."""
    if name not in _loggers:
        # Default log directory to KEVIN/logs if not specified
        if log_dir is None:
            log_dir = Path.home() / "lrepos" / "claude-agent-sdk-python" / "KEVIN" / "logs"

        _loggers[name] = StructuredLogger(
            name=name,
            log_dir=log_dir,
            log_level=log_level,
            enable_console=enable_console
        )

    return _loggers[name]


def configure_global_logging(
    log_dir: Optional[Path] = None,
    log_level: str = "INFO",
    enable_console: bool = True
) -> None:
    """Configure global logging settings."""
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

        # Add file handler for all logs
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "multi_agent_system.json",
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10
        )
        file_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(file_handler)

    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        root_logger.addHandler(console_handler)