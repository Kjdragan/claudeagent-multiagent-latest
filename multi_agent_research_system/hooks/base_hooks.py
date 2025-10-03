"""
Base Hook Infrastructure for Multi-Agent Research System

Provides foundational classes and interfaces for hook management,
context tracking, and result handling across the hook system.
"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
from pathlib import Path

# Import logging with proper path handling
try:
    from agent_logging import get_logger
except ImportError:
    # Fallback for when running as module
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Try to import from the agent_logging module we created
    try:
        from agent_logging import get_logger
    except ImportError:
        # Final fallback - create a simple logger
        import logging
        def get_logger(name):
            return logging.getLogger(name)

# Forward reference for type hints
HookManager = None


class HookStatus(Enum):
    """Enumeration of hook execution statuses."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class HookPriority(Enum):
    """Enumeration of hook execution priorities."""
    LOWEST = 0
    LOW = 25
    NORMAL = 50
    HIGH = 75
    HIGHEST = 100


@dataclass
class HookContext:
    """Context information passed to hooks during execution."""
    hook_name: str
    hook_type: str
    session_id: str
    agent_name: Optional[str] = None
    agent_type: Optional[str] = None
    workflow_stage: Optional[str] = None
    correlation_id: Optional[str] = None
    execution_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    previous_contexts: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Initialize derived fields."""
        if not self.correlation_id:
            self.correlation_id = str(uuid.uuid4())
        if not self.execution_id:
            self.execution_id = f"{self.hook_name}_{int(time.time())}_{str(uuid.uuid4())[:8]}"


@dataclass
class HookResult:
    """Result of hook execution with comprehensive tracking."""
    hook_name: str
    hook_type: str
    status: HookStatus
    execution_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    execution_time: Optional[float] = None
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    next_hooks: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if hook execution was successful."""
        return self.status == HookStatus.COMPLETED

    @property
    def failed(self) -> bool:
        """Check if hook execution failed."""
        return self.status in [HookStatus.FAILED, HookStatus.TIMEOUT, HookStatus.CANCELLED]


class BaseHook(ABC):
    """Abstract base class for all hooks in the system."""

    def __init__(
        self,
        name: str,
        hook_type: str,
        priority: HookPriority = HookPriority.NORMAL,
        timeout: Optional[float] = None,
        enabled: bool = True,
        retry_count: int = 0,
        retry_delay: float = 1.0
    ):
        """Initialize base hook."""
        self.name = name
        self.hook_type = hook_type
        self.priority = priority
        self.timeout = timeout
        self.enabled = enabled
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.logger = get_logger(f"hook.{name}")
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.last_execution: Optional[datetime] = None
        self.average_execution_time = 0.0

    @abstractmethod
    async def execute(self, context: HookContext) -> HookResult:
        """Execute the hook with given context."""
        pass

    def can_execute(self, context: HookContext) -> bool:
        """Check if hook can be executed for given context."""
        return self.enabled

    async def safe_execute(self, context: HookContext) -> HookResult:
        """Safely execute hook with error handling and retries."""
        if not self.can_execute(context):
            return HookResult(
                hook_name=self.name,
                hook_type=self.hook_type,
                status=HookStatus.CANCELLED,
                execution_id=context.execution_id,
                start_time=context.timestamp,
                metadata={"reason": "Hook cannot execute for this context"}
            )

        start_time = time.time()
        self.execution_count += 1
        self.last_execution = datetime.now()

        for attempt in range(self.retry_count + 1):
            attempt_start = time.time()

            try:
                self.logger.info(f"Executing hook: {self.name} (attempt {attempt + 1})",
                               hook_name=self.name,
                               attempt=attempt + 1,
                               max_attempts=self.retry_count + 1,
                               session_id=context.session_id)

                # Execute with timeout if specified
                if self.timeout:
                    result = await asyncio.wait_for(
                        self.execute(context),
                        timeout=self.timeout
                    )
                else:
                    result = await self.execute(context)

                # Update statistics
                execution_time = time.time() - attempt_start
                self._update_execution_stats(execution_time, True)

                self.logger.info(f"Hook execution completed: {self.name}",
                               hook_name=self.name,
                               execution_time=execution_time,
                               status=result.status.value,
                               session_id=context.session_id)

                return result

            except asyncio.TimeoutError:
                execution_time = time.time() - attempt_start
                self._update_execution_stats(execution_time, False)

                self.logger.warning(f"Hook execution timeout: {self.name} (attempt {attempt + 1})",
                                   hook_name=self.name,
                                   execution_time=execution_time,
                                   timeout=self.timeout,
                                   attempt=attempt + 1,
                                   session_id=context.session_id)

                if attempt < self.retry_count:
                    await asyncio.sleep(self.retry_delay)
                    continue

                return HookResult(
                    hook_name=self.name,
                    hook_type=self.hook_type,
                    status=HookStatus.TIMEOUT,
                    execution_id=context.execution_id,
                    start_time=context.timestamp,
                    end_time=datetime.now(),
                    execution_time=time.time() - start_time,
                    error_message=f"Hook execution timed out after {self.timeout}s",
                    metadata={"attempts": attempt + 1, "timeout": self.timeout}
                )

            except Exception as e:
                execution_time = time.time() - attempt_start
                self._update_execution_stats(execution_time, False)

                self.logger.error(f"Hook execution failed: {self.name} (attempt {attempt + 1})",
                                hook_name=self.name,
                                error=str(e),
                                error_type=type(e).__name__,
                                execution_time=execution_time,
                                attempt=attempt + 1,
                                session_id=context.session_id)

                if attempt < self.retry_count:
                    await asyncio.sleep(self.retry_delay)
                    continue

                return HookResult(
                    hook_name=self.name,
                    hook_type=self.hook_type,
                    status=HookStatus.FAILED,
                    execution_id=context.execution_id,
                    start_time=context.timestamp,
                    end_time=datetime.now(),
                    execution_time=time.time() - start_time,
                    error_message=str(e),
                    error_type=type(e).__name__,
                    metadata={"attempts": attempt + 1}
                )

        # Should never reach here
        return HookResult(
            hook_name=self.name,
            hook_type=self.hook_type,
            status=HookStatus.FAILED,
            execution_id=context.execution_id,
            start_time=context.timestamp,
            end_time=datetime.now(),
            execution_time=time.time() - start_time,
            error_message="Unexpected hook execution failure",
            metadata={"attempts": self.retry_count + 1}
        )

    def _update_execution_stats(self, execution_time: float, success: bool):
        """Update hook execution statistics."""
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

        # Update average execution time
        total_executions = self.success_count + self.failure_count
        if total_executions == 1:
            self.average_execution_time = execution_time
        else:
            self.average_execution_time = (
                (self.average_execution_time * (total_executions - 1) + execution_time) /
                total_executions
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get hook execution statistics."""
        total_executions = self.success_count + self.failure_count
        success_rate = (self.success_count / total_executions * 100) if total_executions > 0 else 0.0

        return {
            "hook_name": self.name,
            "hook_type": self.hook_type,
            "enabled": self.enabled,
            "priority": self.priority.value,
            "total_executions": total_executions,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": round(success_rate, 2),
            "average_execution_time": round(self.average_execution_time, 3),
            "last_execution": self.last_execution.isoformat() if self.last_execution else None,
            "timeout": self.timeout,
            "retry_count": self.retry_count
        }


class HookManager:
    """Central manager for hook registration, execution, and monitoring."""

    def __init__(self):
        """Initialize hook manager."""
        self.hooks: Dict[str, List[BaseHook]] = {}
        self.global_hooks: List[BaseHook] = []
        self.logger = get_logger("hook_manager")
        self.execution_history: List[HookResult] = []
        self.max_history_size = 1000

    def register_hook(self, hook: BaseHook, hook_types: Optional[List[str]] = None):
        """Register a hook for specific hook types or globally."""
        if hook_types:
            for hook_type in hook_types:
                if hook_type not in self.hooks:
                    self.hooks[hook_type] = []
                self.hooks[hook_type].append(hook)
        else:
            self.global_hooks.append(hook)

        # Sort hooks by priority (highest first)
        if hook_types:
            for hook_type in hook_types:
                if hook_type in self.hooks:
                    self.hooks[hook_type].sort(key=lambda h: h.priority.value, reverse=True)
        else:
            self.global_hooks.sort(key=lambda h: h.priority.value, reverse=True)

        self.logger.info(f"Hook registered: {hook.name}",
                        hook_name=hook.name,
                        hook_type=hook.hook_type,
                        priority=hook.priority.value,
                        global_hook=hook_types is None)

    def unregister_hook(self, hook_name: str, hook_type: Optional[str] = None):
        """Unregister a hook by name and optionally by type."""
        removed_count = 0

        if hook_type and hook_type in self.hooks:
            original_count = len(self.hooks[hook_type])
            self.hooks[hook_type] = [h for h in self.hooks[hook_type] if h.name != hook_name]
            removed_count += original_count - len(self.hooks[hook_type])

        if not hook_type:
            # Search in all hook types
            for ht in self.hooks:
                original_count = len(self.hooks[ht])
                self.hooks[ht] = [h for h in self.hooks[ht] if h.name != hook_name]
                removed_count += original_count - len(self.hooks[ht])

            # Also remove from global hooks
            original_count = len(self.global_hooks)
            self.global_hooks = [h for h in self.global_hooks if h.name != hook_name]
            removed_count += original_count - len(self.global_hooks)

        self.logger.info(f"Hook unregistered: {hook_name}",
                        hook_name=hook_name,
                        hook_type=hook_type,
                        removed_count=removed_count)

    async def execute_hooks(
        self,
        hook_type: str,
        context: HookContext,
        parallel: bool = False
    ) -> List[HookResult]:
        """Execute all registered hooks for a given type."""
        # Get relevant hooks
        relevant_hooks = self.hooks.get(hook_type, []) + self.global_hooks
        relevant_hooks.sort(key=lambda h: h.priority.value, reverse=True)

        if not relevant_hooks:
            self.logger.debug(f"No hooks registered for type: {hook_type}",
                            hook_type=hook_type,
                            session_id=context.session_id)
            return []

        self.logger.info(f"Executing {len(relevant_hooks)} hooks for type: {hook_type}",
                        hook_type=hook_type,
                        hook_count=len(relevant_hooks),
                        session_id=context.session_id)

        start_time = time.time()

        try:
            if parallel:
                # Execute hooks in parallel
                tasks = [hook.safe_execute(context) for hook in relevant_hooks]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Convert exceptions to failed HookResult objects
                processed_results = []
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        processed_results.append(HookResult(
                            hook_name=relevant_hooks[i].name,
                            hook_type=relevant_hooks[i].hook_type,
                            status=HookStatus.FAILED,
                            execution_id=context.execution_id,
                            start_time=context.timestamp,
                            end_time=datetime.now(),
                            execution_time=time.time() - start_time,
                            error_message=str(result),
                            error_type=type(result).__name__
                        ))
                    else:
                        processed_results.append(result)

                results = processed_results
            else:
                # Execute hooks sequentially
                results = []
                for hook in relevant_hooks:
                    result = await hook.safe_execute(context)
                    results.append(result)

                    # Stop execution if hook indicates failure and is critical
                    if not result.success and hook.priority == HookPriority.HIGHEST:
                        self.logger.warning(f"Critical hook failed, stopping execution: {hook.name}",
                                          hook_name=hook.name,
                                          error=result.error_message,
                                          session_id=context.session_id)
                        break

        except Exception as e:
            self.logger.error(f"Hook execution failed: {hook_type}",
                            hook_type=hook_type,
                            error=str(e),
                            error_type=type(e).__name__,
                            session_id=context.session_id)
            results = []

        # Store results in history
        self._add_to_history(results)

        execution_time = time.time() - start_time
        successful_hooks = sum(1 for r in results if r.success)

        self.logger.info(f"Hook execution completed: {hook_type}",
                        hook_type=hook_type,
                        execution_time=execution_time,
                        total_hooks=len(relevant_hooks),
                        successful_hooks=successful_hooks,
                        failed_hooks=len(results) - successful_hooks,
                        session_id=context.session_id)

        return results

    def _add_to_history(self, results: List[HookResult]):
        """Add hook results to execution history."""
        self.execution_history.extend(results)

        # Trim history if it exceeds maximum size
        if len(self.execution_history) > self.max_history_size:
            excess = len(self.execution_history) - self.max_history_size
            self.execution_history = self.execution_history[excess:]

    def get_hook_stats(self) -> Dict[str, Any]:
        """Get statistics for all registered hooks."""
        all_hooks = []
        for hooks_list in self.hooks.values():
            all_hooks.extend(hooks_list)
        all_hooks.extend(self.global_hooks)

        return {
            "total_hooks": len(all_hooks),
            "hook_types": list(self.hooks.keys()),
            "hooks_per_type": {ht: len(hooks) for ht, hooks in self.hooks.items()},
            "global_hooks": len(self.global_hooks),
            "total_executions": len(self.execution_history),
            "hook_details": [hook.get_stats() for hook in all_hooks]
        }

    def get_execution_history(
        self,
        limit: Optional[int] = None,
        hook_type: Optional[str] = None,
        session_id: Optional[str] = None,
        status: Optional[HookStatus] = None
    ) -> List[HookResult]:
        """Get filtered hook execution history."""
        history = self.execution_history.copy()

        # Apply filters
        if hook_type:
            history = [r for r in history if r.hook_type == hook_type]

        if session_id:
            # Need to filter by session_id from context (would need to store context)
            # For now, this is a placeholder
            pass

        if status:
            history = [r for r in history if r.status == status]

        # Sort by execution time (most recent first)
        history.sort(key=lambda r: r.start_time, reverse=True)

        # Apply limit
        if limit:
            history = history[:limit]

        return history