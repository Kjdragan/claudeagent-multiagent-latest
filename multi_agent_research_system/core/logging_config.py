"""Logging configuration for the multi-agent research system.

This module provides a centralized logging system with configurable levels,
file rotation, and cleanup capabilities.
"""

import argparse
import logging
import logging.handlers
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class LogLevel(Enum):
    """Enumeration of available log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class MultiAgentLogger:
    """Centralized logging system for the multi-agent research system."""

    def __init__(self, log_level: str = "INFO", log_dir: str = "KEVIN/logs"):
        """Initialize the logging system.

        Args:
            log_level: Initial log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory to store log files (defaults to KEVIN/logs)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_level = log_level.upper()
        self.loggers: dict[str, logging.Logger] = {}
        self.start_time = datetime.now()

        # Clean up old log files on startup
        self._cleanup_old_logs()

        # Setup root logger
        self._setup_root_logger()

        # Create system logger
        self.system_logger = self.get_logger("system")
        self.system_logger.info(f"Logging system initialized with level: {self.log_level}")

    def _cleanup_old_logs(self):
        """Clean up old log files on startup."""
        try:
            if self.log_dir.exists():
                log_files = list(self.log_dir.glob("*.log"))
                for log_file in log_files:
                    log_file.unlink()
                self.log_files_deleted = len(log_files)
        except Exception as e:
            print(f"Warning: Could not clean up old log files: {e}")
            self.log_files_deleted = 0

    def _setup_root_logger(self):
        """Setup the root logger with console and file handlers."""
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.log_level))

        # Clear existing handlers
        root_logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.log_level))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # File handler with timestamp
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"multi_agent_research_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        self.current_log_file = log_file
        self.system_logger = logging.getLogger("system")

    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance for a specific component.

        Args:
            name: Name of the component/module

        Returns:
            Logger instance
        """
        if name not in self.loggers:
            logger = logging.getLogger(name)
            logger.setLevel(getattr(logging, self.log_level))
            self.loggers[name] = logger

        return self.loggers[name]

    def set_log_level(self, level: str):
        """Change the log level for all loggers.

        Args:
            level: New log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        level = level.upper()
        if level not in [l.value for l in LogLevel]:
            raise ValueError(f"Invalid log level: {level}")

        self.log_level = level

        # Update all loggers
        for logger in self.loggers.values():
            logger.setLevel(getattr(logging, level))

        # Update console handler
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(getattr(logging, level))
                break

        if hasattr(self, 'system_logger'):
            self.system_logger.info(f"Log level changed to: {level}")

    def get_log_level(self) -> str:
        """Get current log level.

        Returns:
            Current log level
        """
        return self.log_level

    def get_log_files(self) -> list:
        """Get list of current log files.

        Returns:
            List of log file paths
        """
        try:
            return list(self.log_dir.glob("*.log"))
        except Exception:
            return []

    def get_current_log_file(self) -> Path | None:
        """Get the current log file path.

        Returns:
            Path to current log file or None
        """
        return getattr(self, 'current_log_file', None)

    def get_log_summary(self) -> dict[str, Any]:
        """Get a summary of logging information.

        Returns:
            Dictionary with logging information
        """
        return {
            "log_level": self.log_level,
            "log_directory": str(self.log_dir),
            "current_log_file": str(self.current_log_file) if self.current_log_file else None,
            "log_files_deleted": getattr(self, 'log_files_deleted', 0),
            "start_time": self.start_time.isoformat(),
            "active_loggers": len(self.loggers)
        }


# Global logger instance
_logger_instance: MultiAgentLogger | None = None


def setup_logging(log_level: str = "INFO", log_dir: str = "KEVIN/logs") -> MultiAgentLogger:
    """Setup the global logging system.

    Args:
        log_level: Log level to use
        log_dir: Directory for log files

    Returns:
        MultiAgentLogger instance
    """
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = MultiAgentLogger(log_level, log_dir)
    return _logger_instance


def get_logger(name: str = "system") -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Name of the logger

    Returns:
        Logger instance
    """
    if _logger_instance is None:
        setup_logging()
    return _logger_instance.get_logger(name)


def set_log_level(level: str):
    """Set the global log level.

    Args:
        level: New log level
    """
    if _logger_instance:
        _logger_instance.set_log_level(level)


def get_log_level() -> str:
    """Get the current log level.

    Returns:
        Current log level
    """
    if _logger_instance:
        return _logger_instance.get_log_level()
    return "INFO"


def get_log_summary() -> dict[str, Any]:
    """Get logging system summary.

    Returns:
        Dictionary with logging information
    """
    if _logger_instance:
        return _logger_instance.get_log_summary()
    return {}


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser with logging options.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Multi-Agent Research System with configurable logging"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO)"
    )
    parser.add_argument(
        "--log-dir",
        default="KEVIN/logs",
        help="Directory to store log files (default: KEVIN/logs)"
    )
    parser.add_argument(
        "--no-console-log",
        action="store_true",
        help="Disable console logging (file logging only)"
    )
    parser.add_argument(
        "--debug-agents",
        action="store_true",
        help="Enable agent debugging with stderr capture and tool tracing"
    )
    return parser


def parse_args_and_setup_logging(args: list | None = None) -> argparse.Namespace:
    """Parse command line arguments and setup logging.

    Args:
        args: Command line arguments (defaults to sys.argv[1:])

    Returns:
        Parsed arguments
    """
    parser = create_argument_parser()
    parsed_args = parser.parse_args(args)

    # Setup logging
    setup_logging(parsed_args.log_level, parsed_args.log_dir)

    # Configure console logging
    if parsed_args.no_console_log:
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler):
                root_logger.removeHandler(handler)

    logger = get_logger("cli")
    logger.info(f"Started with log level: {parsed_args.log_level}")
    logger.info(f"Log directory: {parsed_args.log_dir}")

    return parsed_args


# Convenience functions for direct import
def log_debug(message: str, logger_name: str = "system"):
    """Log a debug message."""
    get_logger(logger_name).debug(message)


def log_info(message: str, logger_name: str = "system"):
    """Log an info message."""
    get_logger(logger_name).info(message)


def log_warning(message: str, logger_name: str = "system"):
    """Log a warning message."""
    get_logger(logger_name).warning(message)


def log_error(message: str, logger_name: str = "system"):
    """Log an error message."""
    get_logger(logger_name).error(message)


def log_critical(message: str, logger_name: str = "system"):
    """Log a critical message."""
    get_logger(logger_name).critical(message)
