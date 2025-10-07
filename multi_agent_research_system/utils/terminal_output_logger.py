"""
Terminal Output Logger - Captures stdout/stderr to session files.

This module provides functionality to capture all terminal output and save it
to timestamped files in the session's working directory for later review.
"""

import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import TextIO, Optional
import threading


logger = logging.getLogger(__name__)


class TerminalOutputLogger:
    """
    Captures terminal output (stdout/stderr) and writes it to both console and file.

    Thread-safe implementation that allows multiple sessions to log simultaneously.
    """

    def __init__(self, session_id: str, working_dir: str | Path):
        """
        Initialize terminal output logger.

        Args:
            session_id: Unique session identifier
            working_dir: Path to session's working directory
        """
        self.session_id = session_id
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = self.working_dir / f"terminal_output_{timestamp}.log"

        # File handle for writing
        self.file_handle: Optional[TextIO] = None

        # Original stdout/stderr for restoration
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        # Thread lock for thread-safe writing
        self.lock = threading.Lock()

        # Track if logger is active
        self.active = False

        logger.info(f"Terminal output logger initialized for session {session_id}")
        logger.info(f"Output will be saved to: {self.output_file}")

    def start(self):
        """Start capturing terminal output."""
        if self.active:
            logger.warning(f"Terminal output logger already active for session {self.session_id}")
            return

        try:
            # Open file for writing
            self.file_handle = open(self.output_file, 'w', encoding='utf-8', buffering=1)

            # Write header to file
            self.file_handle.write(f"=" * 80 + "\n")
            self.file_handle.write(f"Multi-Agent Research System - Terminal Output Log\n")
            self.file_handle.write(f"Session ID: {self.session_id}\n")
            self.file_handle.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.file_handle.write(f"=" * 80 + "\n\n")
            self.file_handle.flush()

            # Redirect stdout and stderr to our tee writers
            sys.stdout = TeeWriter(self.original_stdout, self.file_handle, self.lock)
            sys.stderr = TeeWriter(self.original_stderr, self.file_handle, self.lock, prefix="[STDERR] ")

            self.active = True
            logger.info(f"✅ Terminal output capture started for session {self.session_id}")

        except Exception as e:
            logger.error(f"Failed to start terminal output logging: {e}")
            self.stop()  # Cleanup on failure
            raise

    def stop(self):
        """Stop capturing terminal output and close file."""
        if not self.active:
            return

        try:
            # Restore original stdout/stderr
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr

            # Write footer and close file
            if self.file_handle and not self.file_handle.closed:
                self.file_handle.write("\n" + "=" * 80 + "\n")
                self.file_handle.write(f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                self.file_handle.write("=" * 80 + "\n")
                self.file_handle.flush()
                self.file_handle.close()

            self.active = False
            logger.info(f"✅ Terminal output capture stopped for session {self.session_id}")
            logger.info(f"📄 Output saved to: {self.output_file}")

        except Exception as e:
            logger.error(f"Error stopping terminal output logging: {e}")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False  # Don't suppress exceptions


class TeeWriter:
    """
    Writer that duplicates output to both console and file.

    Thread-safe implementation for concurrent writing.
    """

    def __init__(self, console: TextIO, file: TextIO, lock: threading.Lock, prefix: str = ""):
        """
        Initialize tee writer.

        Args:
            console: Original console stream (stdout or stderr)
            file: File stream to write to
            lock: Thread lock for synchronized writing
            prefix: Optional prefix for each line (e.g., "[STDERR] ")
        """
        self.console = console
        self.file = file
        self.lock = lock
        self.prefix = prefix

    def write(self, message: str):
        """Write message to both console and file."""
        with self.lock:
            # Write to console (original stream)
            self.console.write(message)

            # Write to file with optional prefix
            if self.file and not self.file.closed:
                if self.prefix and message.strip():  # Only add prefix to non-empty lines
                    # Handle multi-line messages
                    lines = message.split('\n')
                    prefixed_message = '\n'.join(
                        self.prefix + line if line.strip() else line
                        for line in lines
                    )
                    self.file.write(prefixed_message)
                else:
                    self.file.write(message)
                self.file.flush()

    def flush(self):
        """Flush both streams."""
        with self.lock:
            self.console.flush()
            if self.file and not self.file.closed:
                self.file.flush()

    def isatty(self):
        """Check if console is a TTY."""
        return self.console.isatty()

    def fileno(self):
        """Get file descriptor of console."""
        return self.console.fileno()


# Global registry of active loggers (one per session)
_active_loggers: dict[str, TerminalOutputLogger] = {}
_registry_lock = threading.Lock()


def start_session_output_logging(session_id: str, working_dir: str | Path) -> TerminalOutputLogger:
    """
    Start terminal output logging for a session.

    Args:
        session_id: Unique session identifier
        working_dir: Path to session's working directory

    Returns:
        TerminalOutputLogger instance

    Raises:
        RuntimeError: If logging already active for this session
    """
    with _registry_lock:
        if session_id in _active_loggers:
            raise RuntimeError(f"Terminal output logging already active for session {session_id}")

        output_logger = TerminalOutputLogger(session_id, working_dir)
        output_logger.start()
        _active_loggers[session_id] = output_logger

        return output_logger


def stop_session_output_logging(session_id: str):
    """
    Stop terminal output logging for a session.

    Args:
        session_id: Unique session identifier
    """
    with _registry_lock:
        if session_id in _active_loggers:
            output_logger = _active_loggers[session_id]
            output_logger.stop()
            del _active_loggers[session_id]
        else:
            logger.warning(f"No active terminal output logging found for session {session_id}")


def get_session_output_logger(session_id: str) -> Optional[TerminalOutputLogger]:
    """
    Get the active output logger for a session.

    Args:
        session_id: Unique session identifier

    Returns:
        TerminalOutputLogger instance or None if not active
    """
    with _registry_lock:
        return _active_loggers.get(session_id)


def cleanup_all_loggers():
    """Stop all active output loggers (for cleanup on shutdown)."""
    with _registry_lock:
        for session_id, output_logger in list(_active_loggers.items()):
            logger.info(f"Cleaning up terminal output logger for session {session_id}")
            output_logger.stop()
        _active_loggers.clear()


# Register cleanup on module unload
import atexit
atexit.register(cleanup_all_loggers)
