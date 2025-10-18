"""
Session-based logging configuration for multi-agent research system.

This module provides utilities to consolidate all logs under session directories,
replacing the scattered logging across multiple root directories.

Structure:
    /KEVIN/sessions/{session_id}/
        logs/
            orchestrator.log        # Main workflow log
            system.log              # System initialization
            validation.log          # Tracker & validation events
            agents/
                research_agent.log
                report_agent.log
                editor_agent.log
            tools/
                search_tools.log
                research_tools.log
                workproduct_tools.log
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


def get_session_log_dir(session_id: str) -> Path:
    """Get the log directory for a specific session.
    
    Args:
        session_id: The session UUID
        
    Returns:
        Path to session log directory
    """
    return Path(f"KEVIN/sessions/{session_id}/logs")


def create_session_log_structure(session_id: str) -> Dict[str, Path]:
    """Create complete log directory structure for a session.
    
    Args:
        session_id: The session UUID
        
    Returns:
        Dictionary mapping log names to paths
    """
    log_dir = get_session_log_dir(session_id)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    agents_dir = log_dir / "agents"
    tools_dir = log_dir / "tools"
    agents_dir.mkdir(exist_ok=True)
    tools_dir.mkdir(exist_ok=True)
    
    # Define all log paths
    log_paths = {
        "orchestrator": log_dir / "orchestrator.log",
        "system": log_dir / "system.log",
        "validation": log_dir / "validation.log",
        "research_agent": agents_dir / "research_agent.log",
        "report_agent": agents_dir / "report_agent.log",
        "enhanced_report_agent": agents_dir / "enhanced_report_agent.log",
        "editor_agent": agents_dir / "editor_agent.log",
        "finalization": agents_dir / "finalization.log",
        "search_tools": tools_dir / "search_tools.log",
        "research_tools": tools_dir / "research_tools.log",
        "workproduct_tools": tools_dir / "workproduct_tools.log",
        "tool_execution": tools_dir / "execution.log",
    }
    
    return log_paths


def create_file_handler(
    log_path: Path,
    level: int = logging.DEBUG,
    format_string: Optional[str] = None
) -> logging.FileHandler:
    """Create a file handler for logging.
    
    Args:
        log_path: Path to log file
        level: Logging level (default: DEBUG)
        format_string: Optional custom format string
        
    Returns:
        Configured FileHandler
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handler = logging.FileHandler(log_path, encoding='utf-8')
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(format_string))
    
    return handler


def create_session_logger(
    logger_name: str,
    log_path: Path,
    level: int = logging.DEBUG
) -> logging.Logger:
    """Create a logger for a specific component within a session.
    
    Args:
        logger_name: Name for the logger
        log_path: Path to log file
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Add file handler
    file_handler = create_file_handler(log_path, level)
    logger.addHandler(file_handler)
    
    # Prevent propagation to avoid duplicate logs
    logger.propagate = False
    
    return logger


def configure_validation_logger(session_id: str) -> logging.Logger:
    """Create dedicated validation logger for tracker and validation events.
    
    Args:
        session_id: The session UUID
        
    Returns:
        Validation logger
    """
    log_paths = create_session_log_structure(session_id)
    validation_log = log_paths["validation"]
    
    # Create logger with specific format for validation events
    format_string = '%(asctime)s - VALIDATION - %(levelname)s - %(message)s'
    logger = logging.getLogger(f"validation.{session_id}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    handler = create_file_handler(validation_log, format_string=format_string)
    logger.addHandler(handler)
    logger.propagate = False
    
    return logger


def configure_tool_logger(session_id: str, tool_category: str = "execution") -> logging.Logger:
    """Create dedicated tool execution logger.
    
    Args:
        session_id: The session UUID
        tool_category: Category of tool (execution, search, research, workproduct)
        
    Returns:
        Tool logger
    """
    log_paths = create_session_log_structure(session_id)
    
    # Map category to log path
    category_map = {
        "execution": log_paths["tool_execution"],
        "search": log_paths["search_tools"],
        "research": log_paths["research_tools"],
        "workproduct": log_paths["workproduct_tools"],
    }
    
    tool_log = category_map.get(tool_category, log_paths["tool_execution"])
    
    # Create logger with tool-specific format
    format_string = '%(asctime)s - TOOL - %(message)s'
    logger = logging.getLogger(f"tools.{tool_category}.{session_id}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    handler = create_file_handler(tool_log, format_string=format_string)
    logger.addHandler(handler)
    logger.propagate = False
    
    return logger


def get_session_log_summary(session_id: str) -> Dict[str, any]:
    """Get summary information about session logs.
    
    Args:
        session_id: The session UUID
        
    Returns:
        Dictionary with log file information
    """
    log_dir = get_session_log_dir(session_id)
    
    if not log_dir.exists():
        return {"exists": False, "session_id": session_id}
    
    summary = {
        "exists": True,
        "session_id": session_id,
        "log_dir": str(log_dir),
        "created": datetime.fromtimestamp(log_dir.stat().st_ctime).isoformat(),
        "log_files": {},
        "total_size_bytes": 0,
    }
    
    # Scan all log files
    for log_file in log_dir.rglob("*.log"):
        rel_path = log_file.relative_to(log_dir)
        size = log_file.stat().st_size
        summary["log_files"][str(rel_path)] = {
            "size_bytes": size,
            "size_kb": round(size / 1024, 2),
            "modified": datetime.fromtimestamp(log_file.stat().st_mtime).isoformat(),
        }
        summary["total_size_bytes"] += size
    
    summary["total_size_kb"] = round(summary["total_size_bytes"] / 1024, 2)
    summary["file_count"] = len(summary["log_files"])
    
    return summary


def log_with_context(logger: logging.Logger, level: int, message: str, **context):
    """Log a message with structured context.
    
    Args:
        logger: The logger to use
        level: Logging level (logging.INFO, etc.)
        message: The log message
        **context: Additional context as key=value pairs
    """
    if context:
        context_str = " | ".join(f"{k}={v}" for k, v in context.items())
        full_message = f"{message} | {context_str}"
    else:
        full_message = message
    
    logger.log(level, full_message)


# Convenience functions for structured logging
def log_validation(session_id: str, stage: str, result: dict):
    """Log validation results in structured format.
    
    Args:
        session_id: The session UUID
        stage: Workflow stage being validated
        result: Validation result dictionary
    """
    logger = configure_validation_logger(session_id)
    
    status = "PASS" if result.get("valid") else "FAIL"
    logger.info(f"{'='*60}")
    logger.info(f"VALIDATION: {stage}")
    logger.info(f"Status: {status}")
    
    if "required_tools" in result:
        logger.info(f"Required tools: {result['required_tools']}")
    
    if "found_tools" in result:
        logger.info(f"Found tools: {list(result['found_tools'].keys())}")
    
    if "missing_tools" in result:
        logger.info(f"Missing tools: {result['missing_tools']}")
    
    if "successful_tool_count" in result:
        logger.info(f"Successful: {result['successful_tool_count']}/{result.get('required_tool_count', '?')}")
    
    logger.info(f"{'='*60}")


def log_tool_execution(session_id: str, tool_name: str, duration: float, success: bool, **details):
    """Log tool execution in structured format.
    
    Args:
        session_id: The session UUID
        tool_name: Name of the tool
        duration: Execution time in seconds
        success: Whether execution succeeded
        **details: Additional execution details
    """
    logger = configure_tool_logger(session_id, "execution")
    
    status = "SUCCESS" if success else "FAILED"
    message = f"[{status}] {tool_name} | Duration: {duration:.2f}s"
    
    if details:
        details_str = " | ".join(f"{k}={v}" for k, v in details.items())
        message += f" | {details_str}"
    
    logger.info(message)


def log_stage_summary(session_id: str, stage: str, summary: dict):
    """Log stage completion summary.
    
    Args:
        session_id: The session UUID
        stage: Stage name
        summary: Summary dictionary
    """
    log_paths = create_session_log_structure(session_id)
    logger = create_session_logger(f"stage.{stage}.{session_id}", log_paths["orchestrator"])
    
    logger.info(f"{'='*60}")
    logger.info(f"STAGE SUMMARY: {stage}")
    
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")
    
    logger.info(f"{'='*60}")
