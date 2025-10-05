"""
Error Recovery Mechanisms for Multi-Agent Research System

Provides resilient workflow execution with fallback mechanisms and recovery strategies.
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class RecoveryStrategy(Enum):
    """Types of recovery strategies."""
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    FALLBACK_FUNCTION = "fallback_function"
    MINIMAL_EXECUTION = "minimal_execution"
    SKIP_STAGE = "skip_stage"
    ABORT_WORKFLOW = "abort_workflow"


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""
    strategy: RecoveryStrategy
    success: bool
    result: Any | None = None
    error: str | None = None
    attempt_count: int = 0
    recovery_time: datetime | None = None


@dataclass
class StageCheckpoint:
    """Checkpoint data for a workflow stage."""
    stage_name: str
    result: dict[str, Any]
    context: dict[str, Any]
    timestamp: datetime
    attempt_count: int = 1
    recovery_attempts: list[RecoveryResult] = field(default_factory=list)


class ResilientWorkflowManager:
    """Manages resilient workflow execution with comprehensive recovery options."""

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)
        self.checkpoints: dict[str, StageCheckpoint] = {}
        self.fallback_strategies: dict[str, Callable] = {}
        self.max_attempts_per_stage = 3
        self.base_retry_delay = 1.0  # seconds
        self.max_retry_delay = 30.0  # seconds

    def register_fallback_strategy(self, stage_name: str, fallback_function: Callable):
        """Register a fallback strategy for a specific stage."""
        self.fallback_strategies[stage_name] = fallback_function
        self.logger.info(f"Registered fallback strategy for stage: {stage_name}")

    async def execute_stage_with_recovery(
        self,
        stage_name: str,
        stage_function: Callable,
        context: dict[str, Any],
        timeout_seconds: int = 300
    ) -> dict[str, Any]:
        """
        Execute a workflow stage with comprehensive recovery options.

        Args:
            stage_name: Name of the workflow stage
            stage_function: Async function to execute for this stage
            context: Context data for the stage execution
            timeout_seconds: Timeout for stage execution

        Returns:
            Stage execution result with recovery metadata
        """
        self.logger.info(f"🚀 Executing stage: {stage_name}")

        for attempt in range(self.max_attempts_per_stage):
            try:
                self.logger.info(f"📝 {stage_name} - Attempt {attempt + 1}/{self.max_attempts_per_stage}")

                # Execute stage with timeout
                result = await asyncio.wait_for(
                    stage_function(context),
                    timeout=timeout_seconds
                )

                # Stage succeeded - save checkpoint
                checkpoint = StageCheckpoint(
                    stage_name=stage_name,
                    result=result,
                    context=context,
                    timestamp=datetime.now(),
                    attempt_count=attempt + 1
                )
                self.checkpoints[stage_name] = checkpoint

                self.logger.info(f"✅ {stage_name} completed successfully on attempt {attempt + 1}")
                return {
                    "stage_name": stage_name,
                    "success": True,
                    "result": result,
                    "attempt_count": attempt + 1,
                    "recovery_attempts": [],
                    "execution_time": datetime.now().isoformat()
                }

            except asyncio.TimeoutError:
                error_msg = f"{stage_name} timed out after {timeout_seconds}s on attempt {attempt + 1}"
                self.logger.error(f"⏰ {error_msg}")
                recovery_result = await self._handle_stage_failure(
                    stage_name, error_msg, attempt, context
                )
                if recovery_result.success:
                    return recovery_result.result

            except Exception as e:
                error_msg = f"{stage_name} failed on attempt {attempt + 1}: {str(e)}"
                self.logger.error(f"❌ {error_msg}")

                if attempt < self.max_attempts_per_stage - 1:
                    # Try recovery strategy
                    recovery_result = await self._handle_stage_failure(
                        stage_name, error_msg, attempt, context
                    )
                    if recovery_result.success:
                        return recovery_result.result
                else:
                    # Final attempt - try minimal execution
                    self.logger.warning(f"🔧 {stage_name} - Final attempt: trying minimal execution")
                    minimal_result = await self._try_minimal_execution(stage_name, context, e)
                    return minimal_result

        # All attempts failed
        failure_result = {
            "stage_name": stage_name,
            "success": False,
            "error": f"All {self.max_attempts_per_stage} attempts failed for {stage_name}",
            "attempt_count": self.max_attempts_per_stage,
            "execution_time": datetime.now().isoformat()
        }
        self.logger.error(f"💥 {stage_name} failed completely after {self.max_attempts_per_stage} attempts")
        return failure_result

    async def _handle_stage_failure(
        self,
        stage_name: str,
        error_msg: str,
        attempt_num: int,
        context: dict[str, Any]
    ) -> RecoveryResult:
        """Handle stage failure with appropriate recovery strategy."""

        # Strategy 1: Retry with exponential backoff
        if attempt_num < self.max_attempts_per_stage - 1:
            backoff_delay = min(
                self.base_retry_delay * (2 ** attempt_num),
                self.max_retry_delay
            )

            self.logger.info(f"⏳ {stage_name} - Retrying in {backoff_delay:.1f}s...")
            await asyncio.sleep(backoff_delay)

            return RecoveryResult(
                strategy=RecoveryStrategy.RETRY_WITH_BACKOFF,
                success=False,  # Will be determined by next attempt
                error=error_msg,
                attempt_count=attempt_num + 1,
                recovery_time=datetime.now()
            )

        # Strategy 2: Try registered fallback function
        if stage_name in self.fallback_strategies:
            try:
                self.logger.info(f"🔄 {stage_name} - Trying fallback strategy...")
                fallback_result = await self.fallback_strategies[stage_name](context, error_msg)

                return RecoveryResult(
                    strategy=RecoveryStrategy.FALLBACK_FUNCTION,
                    success=True,
                    result={
                        "stage_name": stage_name,
                        "success": True,
                        "result": fallback_result,
                        "fallback_used": True,
                        "attempt_count": attempt_num + 1,
                        "execution_time": datetime.now().isoformat()
                    },
                    attempt_count=attempt_num + 1,
                    recovery_time=datetime.now()
                )
            except Exception as fallback_error:
                self.logger.error(f"❌ Fallback strategy failed for {stage_name}: {fallback_error}")
                return RecoveryResult(
                    strategy=RecoveryStrategy.FALLBACK_FUNCTION,
                    success=False,
                    error=f"Fallback failed: {fallback_error}",
                    attempt_count=attempt_num + 1,
                    recovery_time=datetime.now()
                )

        # No fallback available
        return RecoveryResult(
            strategy=RecoveryStrategy.MINIMAL_EXECUTION,
            success=False,
            error="No fallback strategy available",
            attempt_count=attempt_num + 1,
            recovery_time=datetime.now()
        )

    async def _try_minimal_execution(
        self,
        stage_name: str,
        context: dict[str, Any],
        original_error: Exception
    ) -> dict[str, Any]:
        """Try minimal execution strategy for final recovery attempt."""

        try:
            minimal_result = {
                "stage_name": stage_name,
                "success": False,
                "minimal_execution": True,
                "error": str(original_error),
                "fallback_used": False,
                "attempt_count": self.max_attempts_per_stage,
                "execution_time": datetime.now().isoformat(),
                "partial_result": self._extract_partial_result(stage_name, context)
            }

            self.logger.warning(f"🔧 {stage_name} - Minimal execution completed")
            return minimal_result

        except Exception as e:
            # Even minimal execution failed
            failure_result = {
                "stage_name": stage_name,
                "success": False,
                "minimal_execution": False,
                "error": f"Minimal execution also failed: {str(e)}",
                "original_error": str(original_error),
                "attempt_count": self.max_attempts_per_stage,
                "execution_time": datetime.now().isoformat()
            }
            self.logger.error(f"💥 {stage_name} - Even minimal execution failed")
            return failure_result

    def _extract_partial_result(self, stage_name: str, context: dict[str, Any]) -> dict[str, Any]:
        """Extract any partial results from context for minimal recovery."""
        partial_result = {}

        # Extract common partial results based on stage
        if stage_name == "research":
            partial_result.update({
                "has_search_data": bool(context.get("search_results")),
                "url_count": len(context.get("search_results", [])),
                "has_files": bool(context.get("session_id"))
            })
        elif stage_name == "report_generation":
            partial_result.update({
                "has_research_data": bool(context.get("research_result")),
                "content_available": bool(context.get("research_result", {}).get("files_created", 0))
            })
        elif stage_name == "editorial_review":
            partial_result.update({
                "has_content_to_review": bool(context.get("report_content")),
                "files_available": bool(context.get("report_files"))
            })

        return partial_result

    def get_checkpoint(self, stage_name: str) -> StageCheckpoint | None:
        """Get checkpoint data for a stage."""
        return self.checkpoints.get(stage_name)

    def has_completed_stage(self, stage_name: str) -> bool:
        """Check if a stage has completed successfully."""
        return stage_name in self.checkpoints

    def clear_checkpoints(self):
        """Clear all checkpoints."""
        self.checkpoints.clear()
        self.logger.info("All checkpoints cleared")

    def get_recovery_summary(self) -> dict[str, Any]:
        """Get summary of recovery attempts and checkpoints."""
        summary = {
            "total_stages": len(self.checkpoints),
            "completed_stages": list(self.checkpoints.keys()),
            "recovery_attempts": sum(
                len(checkpoint.recovery_attempts)
                for checkpoint in self.checkpoints.values()
            ),
            "timestamp": datetime.now().isoformat()
        }
        return summary


class RecoveryManager:
    """Manages recovery strategies for different types of failures."""

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)
        self.recovery_strategies = {}

    def register_strategy(self, failure_type: str, strategy: Callable):
        """Register a recovery strategy for a specific failure type."""
        self.recovery_strategies[failure_type] = strategy
        self.logger.info(f"Registered recovery strategy for: {failure_type}")

    async def recover_from_failure(
        self,
        workflow_session: Any,
        stage_name: str,
        error: Exception,
        attempt_num: int,
        fallback_data: Any | None = None
    ) -> Any | None:
        """
        Attempt to recover from a stage failure.

        Args:
            workflow_session: Current workflow session
            stage_name: Name of the failed stage
            error: The error that caused the failure
            attempt_num: Current attempt number
            fallback_data: Optional fallback data to use

        Returns:
            Recovery result or None if recovery failed
        """
        failure_type = type(error).__name__

        # Check for specific recovery strategy
        if failure_type in self.recovery_strategies:
            try:
                self.logger.info(f"🔄 Attempting recovery for {failure_type} in {stage_name}")
                recovery_result = await self.recovery_strategies[failure_type](
                    workflow_session, stage_name, error, attempt_num, fallback_data
                )
                self.logger.info(f"✅ Recovery successful for {stage_name}")
                return recovery_result
            except Exception as recovery_error:
                self.logger.error(f"❌ Recovery strategy failed for {stage_name}: {recovery_error}")

        # Check for generic recovery strategy
        if "generic" in self.recovery_strategies:
            try:
                self.logger.info(f"🔄 Attempting generic recovery for {stage_name}")
                recovery_result = await self.recovery_strategies["generic"](
                    workflow_session, stage_name, error, attempt_num, fallback_data
                )
                self.logger.info(f"✅ Generic recovery successful for {stage_name}")
                return recovery_result
            except Exception as recovery_error:
                self.logger.error(f"❌ Generic recovery failed for {stage_name}: {recovery_error}")

        self.logger.warning(f"⚠️ No recovery strategy available for {failure_type} in {stage_name}")
        return None
