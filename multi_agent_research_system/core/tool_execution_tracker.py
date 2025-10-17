"""
MCP Tool Lifecycle Management - Tool Execution Tracker

This module provides monitoring and management for MCP tool execution lifecycle,
addressing issues with long-running tools, stalled states, and timeout handling.

Phase 3 Implementation from repair-edited.md:
- Detect long-running tool calls (>2 minutes)
- Enhance timeout handling with visibility
- Track tool execution state
- Report indeterminate states
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Any, List
from enum import Enum

logger = logging.getLogger(__name__)


class ToolExecutionState(Enum):
    """Tool execution states for lifecycle tracking."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    INDETERMINATE = "indeterminate"  # Tool state unknown after timeout


@dataclass
class ToolExecution:
    """Tracks a single tool execution."""
    tool_name: str
    tool_use_id: str
    session_id: str
    start_time: float
    input_data: Dict[str, Any]
    
    # State tracking
    state: ToolExecutionState = ToolExecutionState.PENDING
    end_time: Optional[float] = None
    execution_time: Optional[float] = None
    result_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    # Timeout tracking
    timeout_seconds: int = 120
    timeout_warnings: List[str] = field(default_factory=list)
    exceeded_timeout: bool = False
    
    # Metadata
    agent_context: Optional[Dict[str, Any]] = None
    retries: int = 0
    
    def elapsed_time(self) -> float:
        """Get elapsed time since tool execution started."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    def is_long_running(self, threshold_seconds: int = 120) -> bool:
        """Check if tool has been running longer than threshold."""
        return self.state == ToolExecutionState.RUNNING and self.elapsed_time() > threshold_seconds
    
    def is_active(self) -> bool:
        """Check if tool execution is still active."""
        return self.state in [ToolExecutionState.PENDING, ToolExecutionState.RUNNING]
    
    def mark_completed(self, result_data: Optional[Dict[str, Any]] = None):
        """Mark tool execution as completed."""
        self.state = ToolExecutionState.COMPLETED
        self.end_time = time.time()
        self.execution_time = self.elapsed_time()
        self.result_data = result_data
    
    def mark_failed(self, error: str):
        """Mark tool execution as failed."""
        self.state = ToolExecutionState.FAILED
        self.end_time = time.time()
        self.execution_time = self.elapsed_time()
        self.error = error
    
    def mark_timeout(self):
        """Mark tool execution as timed out."""
        self.state = ToolExecutionState.TIMEOUT
        self.end_time = time.time()
        self.execution_time = self.elapsed_time()
        self.exceeded_timeout = True
        self.error = f"Tool execution exceeded {self.timeout_seconds}s timeout"
    
    def mark_indeterminate(self):
        """Mark tool execution state as indeterminate (unknown after timeout)."""
        self.state = ToolExecutionState.INDETERMINATE
        self.end_time = time.time()
        self.execution_time = self.elapsed_time()
        self.error = "Tool state became indeterminate after timeout"


class ToolExecutionTracker:
    """
    Tracks MCP tool execution lifecycle with timeout monitoring and state management.
    
    Addresses Phase 3 requirements:
    - Detects long-running tool calls
    - Provides timeout handling with visibility
    - Tracks tool execution states
    - Reports indeterminate states
    """
    
    def __init__(self, default_timeout: int = 120, warning_threshold: int = 60):
        """
        Initialize tool execution tracker.
        
        Args:
            default_timeout: Default timeout for tool execution in seconds
            warning_threshold: Emit warning after this many seconds
        """
        self.default_timeout = default_timeout
        self.warning_threshold = warning_threshold
        
        # Active executions: {tool_use_id: ToolExecution}
        self.active_executions: Dict[str, ToolExecution] = {}
        
        # Completed executions history (for analysis)
        self.execution_history: List[ToolExecution] = []
        
        # Statistics
        self.stats = {
            "total_executions": 0,
            "completed": 0,
            "failed": 0,
            "timeout": 0,
            "cancelled": 0,
            "indeterminate": 0,
            "long_running_detected": 0
        }
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def track_tool_start(
        self, 
        tool_name: str, 
        tool_use_id: str,
        session_id: str,
        input_data: Dict[str, Any],
        agent_context: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[int] = None
    ) -> ToolExecution:
        """
        Start tracking a tool execution.
        
        Args:
            tool_name: Name of the tool being executed
            tool_use_id: Unique identifier for this tool use
            session_id: Session identifier
            input_data: Tool input parameters
            agent_context: Optional agent context information
            timeout_seconds: Optional custom timeout (uses default if not provided)
        
        Returns:
            ToolExecution object for tracking
        """
        execution = ToolExecution(
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            session_id=session_id,
            start_time=time.time(),
            input_data=input_data,
            agent_context=agent_context,
            timeout_seconds=timeout_seconds or self.default_timeout,
            state=ToolExecutionState.RUNNING
        )
        
        self.active_executions[tool_use_id] = execution
        self.stats["total_executions"] += 1
        
        self.logger.info(
            f"🔧 Tool execution started: {tool_name} "
            f"(id: {tool_use_id[:8]}, timeout: {execution.timeout_seconds}s)"
        )
        
        return execution
    
    def track_tool_completion(
        self, 
        tool_use_id: str, 
        result_data: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error: Optional[str] = None
    ):
        """
        Mark a tool execution as completed.
        
        Args:
            tool_use_id: Tool execution identifier
            result_data: Optional result data from tool
            success: Whether execution was successful
            error: Optional error message if failed
        """
        if tool_use_id not in self.active_executions:
            self.logger.warning(f"Attempted to complete untracked tool: {tool_use_id[:8]}")
            return
        
        execution = self.active_executions[tool_use_id]
        
        if success:
            execution.mark_completed(result_data)
            self.stats["completed"] += 1
            self.logger.info(
                f"✅ Tool completed: {execution.tool_name} "
                f"(time: {execution.execution_time:.2f}s)"
            )
        else:
            execution.mark_failed(error or "Unknown error")
            self.stats["failed"] += 1
            self.logger.error(
                f"❌ Tool failed: {execution.tool_name} "
                f"(time: {execution.execution_time:.2f}s, error: {error})"
            )
        
        # Move to history
        self.execution_history.append(execution)
        del self.active_executions[tool_use_id]
    
    def check_tool_timeout(self, tool_use_id: str, max_duration: Optional[int] = None) -> bool:
        """
        Check if a tool has exceeded its timeout.
        
        Args:
            tool_use_id: Tool execution identifier
            max_duration: Optional override for timeout duration
        
        Returns:
            True if tool has timed out, False otherwise
        """
        if tool_use_id not in self.active_executions:
            return False
        
        execution = self.active_executions[tool_use_id]
        timeout = max_duration or execution.timeout_seconds
        elapsed = execution.elapsed_time()
        
        # Check for timeout
        if elapsed > timeout:
            if not execution.exceeded_timeout:
                execution.mark_timeout()
                self.stats["timeout"] += 1
                
                self.logger.error(
                    f"⏱️ TIMEOUT: {execution.tool_name} exceeded {timeout}s "
                    f"(elapsed: {elapsed:.2f}s, id: {tool_use_id[:8]})"
                )
                
                # Move to history
                self.execution_history.append(execution)
                del self.active_executions[tool_use_id]
            
            return True
        
        # Check for warning threshold
        if elapsed > self.warning_threshold and len(execution.timeout_warnings) == 0:
            warning = f"Tool running for {elapsed:.1f}s (warning threshold: {self.warning_threshold}s)"
            execution.timeout_warnings.append(warning)
            
            self.logger.warning(
                f"⚠️ Long-running tool: {execution.tool_name} "
                f"({elapsed:.1f}s elapsed, id: {tool_use_id[:8]})"
            )
        
        return False
    
    def check_all_active_tools(self) -> Dict[str, List[ToolExecution]]:
        """
        Check all active tools for timeouts and long-running states.
        
        Returns:
            Dictionary with 'timeout', 'long_running', and 'active' tool lists
        """
        timed_out = []
        long_running = []
        active = []
        
        for tool_use_id, execution in list(self.active_executions.items()):
            # Check timeout
            if self.check_tool_timeout(tool_use_id):
                timed_out.append(execution)
                continue
            
            # Check long-running
            if execution.is_long_running(self.warning_threshold):
                long_running.append(execution)
                # Increment counter (it's an integer, not a list)
                self.stats["long_running_detected"] += 1
            else:
                active.append(execution)
        
        return {
            "timeout": timed_out,
            "long_running": long_running,
            "active": active
        }
    
    def handle_orphaned_tools(self, session_id: str):
        """
        Handle tools that may be orphaned (session ended but tool still running).
        
        Args:
            session_id: Session identifier to check
        """
        orphaned = [
            execution for execution in self.active_executions.values()
            if execution.session_id == session_id
        ]
        
        if orphaned:
            self.logger.warning(
                f"🔍 Found {len(orphaned)} potentially orphaned tools for session {session_id}"
            )
            
            for execution in orphaned:
                execution.mark_indeterminate()
                self.stats["indeterminate"] += 1
                
                self.logger.warning(
                    f"   - {execution.tool_name} (elapsed: {execution.elapsed_time():.1f}s)"
                )
                
                # Move to history
                self.execution_history.append(execution)
                if execution.tool_use_id in self.active_executions:
                    del self.active_executions[execution.tool_use_id]
    
    def get_execution_state(self, tool_use_id: str) -> Optional[ToolExecutionState]:
        """Get the current state of a tool execution."""
        if tool_use_id in self.active_executions:
            return self.active_executions[tool_use_id].state
        
        # Check history
        for execution in reversed(self.execution_history):
            if execution.tool_use_id == tool_use_id:
                return execution.state
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        return {
            **self.stats,
            "active_count": len(self.active_executions),
            "history_count": len(self.execution_history),
            "average_execution_time": self._calculate_average_execution_time(),
            "timeout_rate": self._calculate_timeout_rate()
        }
    
    def _calculate_average_execution_time(self) -> float:
        """Calculate average execution time for completed tools."""
        completed_executions = [
            e for e in self.execution_history 
            if e.state == ToolExecutionState.COMPLETED and e.execution_time
        ]
        
        if not completed_executions:
            return 0.0
        
        return sum(e.execution_time for e in completed_executions) / len(completed_executions)
    
    def _calculate_timeout_rate(self) -> float:
        """Calculate timeout rate as percentage."""
        total = self.stats["total_executions"]
        if total == 0:
            return 0.0
        
        return (self.stats["timeout"] / total) * 100
    
    def get_active_tool_summary(self) -> str:
        """Get human-readable summary of active tools."""
        if not self.active_executions:
            return "No active tools"
        
        lines = [f"Active tools: {len(self.active_executions)}"]
        
        for execution in self.active_executions.values():
            elapsed = execution.elapsed_time()
            status = "⚠️ LONG" if execution.is_long_running() else "✓"
            lines.append(
                f"  {status} {execution.tool_name}: {elapsed:.1f}s "
                f"(id: {execution.tool_use_id[:8]})"
            )
        
        return "\n".join(lines)


# Global tracker instance
_tracker: Optional[ToolExecutionTracker] = None


def get_tool_execution_tracker(
    default_timeout: int = 120,
    warning_threshold: int = 60
) -> ToolExecutionTracker:
    """
    Get or create the global tool execution tracker instance.
    
    Args:
        default_timeout: Default timeout for tools in seconds
        warning_threshold: Warning threshold in seconds
    
    Returns:
        ToolExecutionTracker instance
    """
    global _tracker
    if _tracker is None:
        _tracker = ToolExecutionTracker(default_timeout, warning_threshold)
    return _tracker


async def monitor_active_tools(interval: int = 30):
    """
    Background task to monitor active tools periodically.
    
    Args:
        interval: Check interval in seconds
    """
    tracker = get_tool_execution_tracker()
    
    while True:
        await asyncio.sleep(interval)
        
        results = tracker.check_all_active_tools()
        
        if results["timeout"]:
            logger.error(f"⏱️ {len(results['timeout'])} tools timed out")
        
        if results["long_running"]:
            logger.warning(f"⚠️ {len(results['long_running'])} long-running tools detected")
            logger.warning(tracker.get_active_tool_summary())
