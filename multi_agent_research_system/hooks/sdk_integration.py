"""
SDK Integration Hooks for Multi-Agent Research System

Provides comprehensive integration with Claude Agent SDK types including
AssistantMessage, ContentBlock, HookContext, and other SDK objects for
proper monitoring and interaction with the multi-agent system.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic
from dataclasses import dataclass, field
from pathlib import Path

# Try to import SDK types, with fallback for when SDK is not available
try:
    from claude_agent_sdk.types import (
        AssistantMessage,
        UserMessage,
        ResultMessage,
        ContentBlock,
        TextBlock,
        ThinkingBlock,
        ToolUseBlock,
        ToolResultBlock,
        HookContext as SDKHookContext,
        HookMatcher,
        HookCallback,
        HookEvent,
        HookJSONOutput
    )
    SDK_AVAILABLE = True
except ImportError:
    # Create fallback types for when SDK is not available
    SDK_AVAILABLE = False

    @dataclass
    class AssistantMessage:
        content: List[Any]
        model: str
        parent_tool_use_id: Optional[str] = None

    @dataclass
    class ContentBlock:
        pass

    @dataclass
    class TextBlock(ContentBlock):
        text: str

    @dataclass
    class ThinkingBlock(ContentBlock):
        thinking: str
        signature: str

    @dataclass
    class ToolUseBlock(ContentBlock):
        id: str
        name: str
        input: Dict[str, Any]

    @dataclass
    class ToolResultBlock(ContentBlock):
        tool_use_id: str
        content: Optional[Union[str, List[Dict[str, Any]]]] = None
        is_error: Optional[bool] = None

    @dataclass
    class SDKHookContext:
        signal: Any = None

    class HookMatcher:
        def __init__(self, matcher=None, hooks=None):
            self.matcher = matcher
            self.hooks = hooks or []

    HookEvent = str
    HookJSONOutput = Dict[str, Any]

from .base_hooks import BaseHook, HookContext, HookResult, HookStatus, HookPriority
import sys
import os
# Add parent directory to path for proper imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent_logging import get_logger


class SDKHookBridge:
    """
    Bridge between our internal hook system and the Claude Agent SDK hook system.

    This class provides the proper integration layer that converts between our
    internal hook format and the SDK's expected HookCallback format.
    """

    def __init__(self, hook_manager: 'HookManager'):
        """Initialize the SDK hook bridge."""
        self.hook_manager = hook_manager
        self.logger = get_logger("sdk_hook_bridge")

    def create_sdk_hook_callback(
        self,
        hook_type: str,
        internal_hook_name: Optional[str] = None
    ) -> HookCallback:
        """
        Create a SDK-compatible hook callback that delegates to our internal hooks.

        Args:
            hook_type: The type of hook (e.g., "PreToolUse", "PostToolUse")
            internal_hook_name: Optional specific internal hook to execute

        Returns:
            A SDK-compatible HookCallback function
        """
        async def sdk_callback(
            input_data: Dict[str, Any],
            tool_use_id: Optional[str],
            context: SDKHookContext
        ) -> HookJSONOutput:
            """SDK hook callback that delegates to internal hook system."""
            try:
                # Convert SDK hook context to our internal hook context
                internal_context = HookContext(
                    hook_name=f"sdk_{hook_type.lower()}",
                    hook_type=hook_type,
                    session_id=getattr(context, 'session_id', 'unknown'),
                    agent_name=getattr(context, 'agent_name', None),
                    metadata={
                        "sdk_input_data": input_data,
                        "tool_use_id": tool_use_id,
                        "sdk_context": context,
                        "hook_type": hook_type,
                        "internal_hook_name": internal_hook_name
                    }
                )

                # Execute internal hooks
                if internal_hook_name:
                    # Execute specific hook
                    results = await self.hook_manager.execute_hooks(
                        hook_type,
                        internal_context,
                        parallel=False
                    )
                else:
                    # Execute all hooks of this type
                    results = await self.hook_manager.execute_hooks(
                        hook_type,
                        internal_context,
                        parallel=True
                    )

                # Convert results to SDK format
                successful_hooks = sum(1 for r in results if r.success)
                failed_hooks = len(results) - successful_hooks

                # Log hook execution
                self.logger.info(f"SDK hook execution completed: {hook_type}",
                                hook_type=hook_type,
                                tool_use_id=tool_use_id,
                                total_hooks=len(results),
                                successful_hooks=successful_hooks,
                                failed_hooks=failed_hooks)

                # Return SDK-compatible response
                if failed_hooks > 0:
                    # Some hooks failed - include error information
                    error_messages = [r.error_message for r in results if r.error_message]
                    return {
                        "systemMessage": f"Hook execution completed with {failed_hooks} errors: {'; '.join(error_messages[:3])}",
                        "hookSpecificOutput": {
                            "execution_results": [
                                {
                                    "hook_name": r.hook_name,
                                    "status": r.status.value,
                                    "execution_time": r.execution_time
                                }
                                for r in results
                            ]
                        }
                    }
                else:
                    # All hooks succeeded
                    return {
                        "hookSpecificOutput": {
                            "execution_results": [
                                {
                                    "hook_name": r.hook_name,
                                    "status": r.status.value,
                                    "execution_time": r.execution_time,
                                    "result_data": r.result_data
                                }
                                for r in results
                            ]
                        }
                    }

            except Exception as e:
                self.logger.error(f"SDK hook callback failed: {hook_type}",
                                hook_type=hook_type,
                                error=str(e),
                                error_type=type(e).__name__)

                # Return error response
                return {
                    "systemMessage": f"Hook execution failed: {str(e)}",
                    "hookSpecificOutput": {
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                }

        return sdk_callback

    def create_hook_matchers(
        self,
        hook_config: Dict[str, List[str]]
    ) -> Dict[HookEvent, List[HookMatcher]]:
        """
        Create SDK HookMatcher objects from our hook configuration.

        Args:
            hook_config: Dictionary mapping hook types to lists of internal hook names

        Returns:
            Dictionary compatible with SDK hooks configuration
        """
        sdk_hooks = {}

        for hook_type, internal_hooks in hook_config.items():
            if not SDK_AVAILABLE:
                continue

            hook_matchers = []

            for internal_hook_name in internal_hooks:
                # Create a SDK hook callback for each internal hook
                sdk_callback = self.create_sdk_hook_callback(
                    hook_type,
                    internal_hook_name
                )

                # Create HookMatcher with the callback
                hook_matcher = HookMatcher(
                    matcher=None,  # Execute for all tools/events
                    hooks=[sdk_callback]
                )
                hook_matchers.append(hook_matcher)

            if hook_matchers:
                sdk_hooks[hook_type] = hook_matchers

        return sdk_hooks

    async def process_sdk_message(
        self,
        message: Union[AssistantMessage, UserMessage, ResultMessage],
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Process SDK messages through our internal hook system.

        Args:
            message: The SDK message to process
            session_id: Session identifier

        Returns:
            Processing results from internal hooks
        """
        try:
            # Determine message type
            if isinstance(message, AssistantMessage):
                message_type = "assistant_message"
            elif isinstance(message, UserMessage):
                message_type = "user_message"
            elif isinstance(message, ResultMessage):
                message_type = "result_message"
            else:
                message_type = "unknown"

            # Create hook context
            context = HookContext(
                hook_name="sdk_message_processor",
                hook_type="sdk_message_processing",
                session_id=session_id,
                metadata={
                    "sdk_message": message,
                    "message_type": message_type
                }
            )

            # Execute message processing hooks
            results = await self.hook_manager.execute_hooks(
                "sdk_message_processing",
                context,
                parallel=True
            )

            # Return combined results
            return {
                "message_type": message_type,
                "processing_results": [
                    {
                        "hook_name": r.hook_name,
                        "status": r.status.value,
                        "result_data": r.result_data
                    }
                    for r in results if r.success
                ],
                "total_hooks": len(results),
                "successful_hooks": sum(1 for r in results if r.success)
            }

        except Exception as e:
            self.logger.error(f"SDK message processing failed",
                            message_type=type(message).__name__,
                            error=str(e),
                            error_type=type(e).__name__)
            return {
                "error": str(e),
                "error_type": type(e).__name__
            }


@dataclass
class MessageAnalysis:
    """Analysis of an SDK message with extracted information."""
    message_id: str
    message_type: str
    timestamp: datetime
    model: Optional[str] = None
    content_summary: str = ""
    text_content: List[str] = field(default_factory=list)
    tool_uses: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    thinking_blocks: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContentBlockAnalysis:
    """Analysis of a content block with detailed information."""
    block_id: str
    block_type: str
    timestamp: datetime
    content: Any
    size_bytes: int = 0
    processing_time: float = 0.0
    extracted_data: Dict[str, Any] = field(default_factory=dict)


class SDKMessageProcessingHook(BaseHook):
    """Hook for processing and analyzing SDK messages (AssistantMessage, UserMessage, etc.)."""

    def __init__(self, enabled: bool = True, timeout: float = 15.0):
        super().__init__(
            name="sdk_message_processor",
            hook_type="sdk_message_processing",
            priority=HookPriority.HIGH,
            timeout=timeout,
            enabled=enabled,
            retry_count=1
        )
        self.message_analyses: List[MessageAnalysis] = []
        self.block_analyses: List[ContentBlockAnalysis] = []
        self.message_patterns: Dict[str, Dict[str, Any]] = {}
        self.max_analyses = 5000

    async def execute(self, context: HookContext) -> HookResult:
        """Execute SDK message processing."""
        try:
            message_data = context.metadata.get("sdk_message", {})
            message_type = context.metadata.get("message_type", "unknown")

            self.logger.info(f"SDK message processing: {message_type}",
                           message_type=message_type,
                           session_id=context.session_id)

            if not SDK_AVAILABLE:
                return HookResult(
                    hook_name=self.name,
                    hook_type=self.hook_type,
                    status=HookStatus.COMPLETED,
                    execution_id=context.execution_id,
                    start_time=context.timestamp,
                    end_time=datetime.now(),
                    result_data={
                        "message_type": message_type,
                        "status": "sdk_not_available",
                        "message": "SDK not available for detailed message processing"
                    }
                )

            # Process based on message type
            if message_type == "assistant_message":
                result = await self._process_assistant_message(context, message_data)
            elif message_type == "user_message":
                result = await self._process_user_message(context, message_data)
            elif message_type == "result_message":
                result = await self._process_result_message(context, message_data)
            elif message_type == "content_blocks":
                result = await self._process_content_blocks(context, message_data)
            else:
                result = await self._process_generic_message(context, message_data)

            return HookResult(
                hook_name=self.name,
                hook_type=self.hook_type,
                status=HookStatus.COMPLETED,
                execution_id=context.execution_id,
                start_time=context.timestamp,
                end_time=datetime.now(),
                result_data=result
            )

        except Exception as e:
            self.logger.error(f"SDK message processing failed: {str(e)}",
                            error=str(e),
                            error_type=type(e).__name__,
                            session_id=context.session_id)
            return HookResult(
                hook_name=self.name,
                hook_type=self.hook_type,
                status=HookStatus.FAILED,
                execution_id=context.execution_id,
                start_time=context.timestamp,
                end_time=datetime.now(),
                error_message=str(e),
                error_type=type(e).__name__
            )

    async def _process_assistant_message(self, context: HookContext, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process AssistantMessage and extract detailed information."""
        if not isinstance(message_data.get("message"), AssistantMessage):
            return {"status": "invalid_message_type", "expected": "AssistantMessage"}

        message: AssistantMessage = message_data["message"]
        start_time = time.time()

        # Create message analysis
        analysis = MessageAnalysis(
            message_id=f"assistant_msg_{int(time.time())}",
            message_type="AssistantMessage",
            timestamp=datetime.now(),
            model=message.model,
            metadata=message_data.copy()
        )

        # Process content blocks
        for i, block in enumerate(message.content):
            block_analysis = await self._analyze_content_block(block, f"block_{i}")
            self.block_analyses.append(block_analysis)

            # Update analysis based on block type
            if isinstance(block, TextBlock):
                analysis.text_content.append(block.text)
                analysis.content_summary += f"Text: {block.text[:100]}..."

            elif isinstance(block, ThinkingBlock):
                analysis.thinking_blocks.append({
                    "thinking": block.thinking,
                    "signature": block.signature,
                    "size": len(block.thinking)
                })

            elif isinstance(block, ToolUseBlock):
                tool_info = {
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                    "input_size": len(json.dumps(block.input, default=str))
                }
                analysis.tool_uses.append(tool_info)

            elif isinstance(block, ToolResultBlock):
                result_info = {
                    "tool_use_id": block.tool_use_id,
                    "content": block.content,
                    "is_error": block.is_error,
                    "content_size": len(str(block.content)) if block.content else 0
                }
                analysis.tool_results.append(result_info)

        # Calculate final metrics
        processing_time = time.time() - start_time
        analysis.metadata["processing_time"] = processing_time
        analysis.metadata["content_blocks_count"] = len(message.content)
        analysis.metadata["text_blocks_count"] = len(analysis.text_content)
        analysis.metadata["tool_uses_count"] = len(analysis.tool_uses)
        analysis.metadata["tool_results_count"] = len(analysis.tool_results)
        analysis.metadata["thinking_blocks_count"] = len(analysis.thinking_blocks)

        # Store analysis
        self._store_message_analysis(analysis)

        # Update message patterns
        self._update_message_patterns(analysis)

        return {
            "message_type": "AssistantMessage",
            "message_id": analysis.message_id,
            "model": message.model,
            "content_blocks": len(message.content),
            "text_blocks": len(analysis.text_content),
            "tool_uses": len(analysis.tool_uses),
            "tool_results": len(analysis.tool_results),
            "thinking_blocks": len(analysis.thinking_blocks),
            "processing_time": processing_time,
            "content_summary": analysis.content_summary[:200] + "..." if len(analysis.content_summary) > 200 else analysis.content_summary
        }

    async def _process_user_message(self, context: HookContext, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process UserMessage and extract relevant information."""
        # UserMessage processing would be implemented here
        return {
            "message_type": "UserMessage",
            "status": "processed",
            "message": "User message processing not yet implemented"
        }

    async def _process_result_message(self, context: HookContext, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process ResultMessage and extract completion information."""
        # ResultMessage processing would be implemented here
        return {
            "message_type": "ResultMessage",
            "status": "processed",
            "message": "Result message processing not yet implemented"
        }

    async def _process_content_blocks(self, context: HookContext, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a list of content blocks."""
        blocks = message_data.get("blocks", [])
        if not blocks:
            return {"message_type": "content_blocks", "status": "no_blocks"}

        analyses = []
        for i, block in enumerate(blocks):
            block_analysis = await self._analyze_content_block(block, f"standalone_block_{i}")
            analyses.append(block_analysis)

        return {
            "message_type": "content_blocks",
            "blocks_processed": len(analyses),
            "block_types": list(set(a.block_type for a in analyses)),
            "total_size": sum(a.size_bytes for a in analyses),
            "analyses": [
                {
                    "block_id": a.block_id,
                    "block_type": a.block_type,
                    "size_bytes": a.size_bytes,
                    "processing_time": a.processing_time
                }
                for a in analyses
            ]
        }

    async def _process_generic_message(self, context: HookContext, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process generic message data."""
        return {
            "message_type": "generic",
            "status": "processed",
            "data_keys": list(message_data.keys()),
            "data_size": len(json.dumps(message_data, default=str))
        }

    async def _analyze_content_block(self, block: ContentBlock, block_id: str) -> ContentBlockAnalysis:
        """Analyze a content block and extract detailed information."""
        start_time = time.time()

        block_type = type(block).__name__
        content_size = len(json.dumps(block, default=str))

        extracted_data = {}

        # Extract data based on block type
        if isinstance(block, TextBlock):
            extracted_data = {
                "text_length": len(block.text),
                "word_count": len(block.text.split()),
                "has_code": "```" in block.text,
                "has_urls": "http" in block.text
            }

        elif isinstance(block, ThinkingBlock):
            extracted_data = {
                "thinking_length": len(block.thinking),
                "signature": block.signature,
                "complexity_score": len(block.thinking) / 100  # Simple complexity metric
            }

        elif isinstance(block, ToolUseBlock):
            extracted_data = {
                "tool_name": block.name,
                "input_keys": list(block.input.keys()),
                "input_complexity": len(block.input),
                "estimated_duration": self._estimate_tool_duration(block.name, block.input)
            }

        elif isinstance(block, ToolResultBlock):
            extracted_data = {
                "tool_use_id": block.tool_use_id,
                "content_type": type(block.content).__name__ if block.content else "None",
                "is_error": block.is_error,
                "content_size": len(str(block.content)) if block.content else 0
            }

        processing_time = time.time() - start_time

        analysis = ContentBlockAnalysis(
            block_id=block_id,
            block_type=block_type,
            timestamp=datetime.now(),
            content=block,
            size_bytes=content_size,
            processing_time=processing_time,
            extracted_data=extracted_data
        )

        return analysis

    def _estimate_tool_duration(self, tool_name: str, tool_input: Dict[str, Any]) -> float:
        """Estimate tool execution duration based on tool type and input."""
        # Simple duration estimation based on tool type
        base_durations = {
            "Read": 0.1,
            "Write": 0.2,
            "Grep": 0.5,
            "Glob": 0.2,
            "Bash": 2.0,
            "WebFetch": 3.0,
            "Search": 1.0
        }

        base_duration = base_durations.get(tool_name, 1.0)

        # Adjust based on input complexity
        input_size = len(json.dumps(tool_input, default=str))
        complexity_factor = min(input_size / 1000, 5.0)  # Cap at 5x complexity

        return base_duration * complexity_factor

    def _store_message_analysis(self, analysis: MessageAnalysis):
        """Store message analysis and maintain history size."""
        self.message_analyses.append(analysis)
        if len(self.message_analyses) > self.max_analyses:
            self.message_analyses = self.message_analyses[-self.max_analyses]

    def _update_message_patterns(self, analysis: MessageAnalysis):
        """Update message pattern statistics."""
        pattern_key = f"{analysis.message_type}:{analysis.model}" if analysis.model else analysis.message_type

        if pattern_key not in self.message_patterns:
            self.message_patterns[pattern_key] = {
                "message_type": analysis.message_type,
                "model": analysis.model,
                "count": 0,
                "first_occurrence": analysis.timestamp,
                "last_occurrence": analysis.timestamp,
                "avg_content_blocks": 0,
                "avg_tool_uses": 0,
                "total_text_blocks": 0,
                "total_tool_uses": 0
            }

        pattern = self.message_patterns[pattern_key]
        pattern["count"] += 1
        pattern["last_occurrence"] = analysis.timestamp

        # Update averages
        pattern["total_text_blocks"] += analysis.metadata.get("text_blocks_count", 0)
        pattern["total_tool_uses"] += analysis.metadata.get("tool_uses_count", 0)
        pattern["avg_content_blocks"] = (
            (pattern["avg_content_blocks"] * (pattern["count"] - 1) + analysis.metadata.get("content_blocks_count", 0)) /
            pattern["count"]
        )
        pattern["avg_tool_uses"] = (
            (pattern["avg_tool_uses"] * (pattern["count"] - 1) + analysis.metadata.get("tool_uses_count", 0)) /
            pattern["count"]
        )

    def get_message_analysis_history(
        self,
        message_type: Optional[str] = None,
        model: Optional[str] = None,
        hours: int = 24,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Get filtered message analysis history."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        analyses = [a for a in self.message_analyses if a.timestamp > cutoff_time]

        # Apply filters
        if message_type:
            analyses = [a for a in analyses if a.message_type == message_type]

        if model:
            analyses = [a for a in analyses if a.model == model]

        # Sort by timestamp (most recent first) and limit
        analyses.sort(key=lambda a: a.timestamp, reverse=True)
        analyses = analyses[:limit]

        return {
            "filter_criteria": {
                "message_type": message_type,
                "model": model,
                "hours": hours,
                "limit": limit
            },
            "total_matching": len(analyses),
            "analyses": [
                {
                    "message_id": a.message_id,
                    "timestamp": a.timestamp.isoformat(),
                    "message_type": a.message_type,
                    "model": a.model,
                    "content_blocks": a.metadata.get("content_blocks_count", 0),
                    "text_blocks": a.metadata.get("text_blocks_count", 0),
                    "tool_uses": a.metadata.get("tool_uses_count", 0),
                    "tool_results": a.metadata.get("tool_results_count", 0),
                    "thinking_blocks": a.metadata.get("thinking_blocks_count", 0),
                    "processing_time": a.metadata.get("processing_time", 0)
                }
                for a in analyses
            ]
        }

    def get_content_block_statistics(self) -> Dict[str, Any]:
        """Get comprehensive content block statistics."""
        if not self.block_analyses:
            return {"message": "No content block analyses available"}

        # Calculate statistics by block type
        block_type_stats = {}
        total_size = 0
        total_processing_time = 0

        for analysis in self.block_analyses:
            block_type = analysis.block_type
            if block_type not in block_type_stats:
                block_type_stats[block_type] = {
                    "count": 0,
                    "total_size": 0,
                    "total_processing_time": 0,
                    "avg_size": 0,
                    "avg_processing_time": 0
                }

            stats = block_type_stats[block_type]
            stats["count"] += 1
            stats["total_size"] += analysis.size_bytes
            stats["total_processing_time"] += analysis.processing_time
            stats["avg_size"] = stats["total_size"] / stats["count"]
            stats["avg_processing_time"] = stats["total_processing_time"] / stats["count"]

            total_size += analysis.size_bytes
            total_processing_time += analysis.processing_time

        return {
            "total_blocks": len(self.block_analyses),
            "total_size_bytes": total_size,
            "total_processing_time": total_processing_time,
            "average_block_size": total_size / len(self.block_analyses),
            "average_processing_time": total_processing_time / len(self.block_analyses),
            "block_type_distribution": dict(sorted(block_type_stats.items(), key=lambda x: x[1]["count"], reverse=True)),
            "largest_blocks": [
                {
                    "block_id": a.block_id,
                    "block_type": a.block_type,
                    "size_bytes": a.size_bytes,
                    "processing_time": a.processing_time
                }
                for a in sorted(self.block_analyses, key=lambda x: x.size_bytes, reverse=True)[:10]
            ]
        }

    def get_message_patterns(self) -> Dict[str, Any]:
        """Get message pattern analysis."""
        return {
            "total_patterns": len(self.message_patterns),
            "patterns": self.message_patterns.copy(),
            "most_common_patterns": [
                {
                    "pattern": key,
                    "count": pattern["count"],
                    "message_type": pattern["message_type"],
                    "model": pattern["model"],
                    "avg_content_blocks": pattern["avg_content_blocks"],
                    "avg_tool_uses": pattern["avg_tool_uses"]
                }
                for key, pattern in sorted(self.message_patterns.items(), key=lambda x: x[1]["count"], reverse=True)[:10]
            ]
        }


class SDKHookIntegration(BaseHook):
    """Hook for integrating with SDK hooks system (HookMatcher, HookCallback, etc.)."""

    def __init__(self, timeout: float = 30.0, enabled: bool = True):
        super().__init__(
            name="sdk_hook_integration",
            hook_type="sdk_hook_integration",
            priority=HookPriority.NORMAL,
            timeout=10.0,
            enabled=enabled,
            retry_count=0
        )
        self.sdk_hook_executions: List[Dict[str, Any]] = []
        self.hook_performance_metrics: Dict[str, Dict[str, Any]] = {}
        self.max_executions = 2000

    async def execute(self, context: HookContext) -> HookResult:
        """Execute SDK hook integration monitoring."""
        try:
            hook_type = context.metadata.get("sdk_hook_type", "unknown")
            hook_name = context.metadata.get("hook_name", "unknown")
            execution_result = context.metadata.get("execution_result", {})

            self.logger.info(f"SDK hook integration: {hook_type} - {hook_name}",
                           hook_type=hook_type,
                           hook_name=hook_name,
                           session_id=context.session_id)

            # Record SDK hook execution
            execution_record = {
                "timestamp": datetime.now().isoformat(),
                "hook_type": hook_type,
                "hook_name": hook_name,
                "session_id": context.session_id,
                "execution_result": execution_result,
                "context": context.metadata.copy()
            }

            self.sdk_hook_executions.append(execution_record)
            if len(self.sdk_hook_executions) > self.max_executions:
                self.sdk_hook_executions = self.sdk_hook_executions[-self.max_executions]

            # Update performance metrics
            self._update_hook_performance_metrics(hook_name, hook_type, execution_result)

            return HookResult(
                hook_name=self.name,
                hook_type=self.hook_type,
                status=HookStatus.COMPLETED,
                execution_id=context.execution_id,
                start_time=context.timestamp,
                end_time=datetime.now(),
                result_data={
                    "hook_type": hook_type,
                    "hook_name": hook_name,
                    "execution_recorded": True,
                    "total_executions": len(self.sdk_hook_executions)
                }
            )

        except Exception as e:
            self.logger.error(f"SDK hook integration failed: {str(e)}",
                            error=str(e),
                            error_type=type(e).__name__,
                            session_id=context.session_id)
            return HookResult(
                hook_name=self.name,
                hook_type=self.hook_type,
                status=HookStatus.FAILED,
                execution_id=context.execution_id,
                start_time=context.timestamp,
                end_time=datetime.now(),
                error_message=str(e),
                error_type=type(e).__name__
            )

    def _update_hook_performance_metrics(self, hook_name: str, hook_type: str, execution_result: Dict[str, Any]):
        """Update performance metrics for SDK hooks."""
        key = f"{hook_name}:{hook_type}"

        if key not in self.hook_performance_metrics:
            self.hook_performance_metrics[key] = {
                "hook_name": hook_name,
                "hook_type": hook_type,
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "total_execution_time": 0.0,
                "average_execution_time": 0.0,
                "min_execution_time": float('inf'),
                "max_execution_time": 0.0,
                "last_execution": None,
                "error_types": {}
            }

        metrics = self.hook_performance_metrics[key]
        metrics["total_executions"] += 1
        metrics["last_execution"] = datetime.now()

        execution_time = execution_result.get("execution_time", 0.0)
        metrics["total_execution_time"] += execution_time

        # Update min/max execution times
        metrics["min_execution_time"] = min(metrics["min_execution_time"], execution_time)
        metrics["max_execution_time"] = max(metrics["max_execution_time"], execution_time)

        # Update average execution time
        metrics["average_execution_time"] = metrics["total_execution_time"] / metrics["total_executions"]

        # Update success/failure counts
        if execution_result.get("success", True):
            metrics["successful_executions"] += 1
        else:
            metrics["failed_executions"] += 1
            error_type = execution_result.get("error_type", "Unknown")
            metrics["error_types"][error_type] = metrics["error_types"].get(error_type, 0) + 1

    def get_sdk_hook_performance(self) -> Dict[str, Any]:
        """Get comprehensive SDK hook performance metrics."""
        if not self.hook_performance_metrics:
            return {"message": "No SDK hook performance data available"}

        total_hooks = len(self.hook_performance_metrics)
        total_executions = sum(m["total_executions"] for m in self.hook_performance_metrics.values())
        successful_executions = sum(m["successful_executions"] for m in self.hook_performance_metrics.values())

        # Find slowest hooks
        slowest_hooks = sorted(
            self.hook_performance_metrics.items(),
            key=lambda x: x[1]["average_execution_time"],
            reverse=True
        )[:10]

        # Find most error-prone hooks
        error_prone_hooks = sorted(
            [(key, metrics) for key, metrics in self.hook_performance_metrics.items() if metrics["failed_executions"] > 0],
            key=lambda x: x[1]["failed_executions"] / x[1]["total_executions"],
            reverse=True
        )[:10]

        return {
            "total_hooks": total_hooks,
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "overall_success_rate": (successful_executions / total_executions * 100) if total_executions > 0 else 0,
            "average_execution_time": sum(m["average_execution_time"] for m in self.hook_performance_metrics.values()) / total_hooks,
            "slowest_hooks": [
                {
                    "hook": key,
                    "average_time": metrics["average_execution_time"],
                    "max_time": metrics["max_execution_time"],
                    "executions": metrics["total_executions"]
                }
                for key, metrics in slowest_hooks
            ],
            "error_prone_hooks": [
                {
                    "hook": key,
                    "failure_rate": (metrics["failed_executions"] / metrics["total_executions"] * 100),
                    "failed_executions": metrics["failed_executions"],
                    "total_executions": metrics["total_executions"],
                    "common_errors": dict(sorted(metrics["error_types"].items(), key=lambda x: x[1], reverse=True)[:3])
                }
                for key, metrics in error_prone_hooks
            ]
        }

    def get_hook_execution_history(
        self,
        hook_name: Optional[str] = None,
        hook_type: Optional[str] = None,
        hours: int = 24,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Get filtered SDK hook execution history."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        executions = [
            e for e in self.sdk_hook_executions
            if datetime.fromisoformat(e["timestamp"]) > cutoff_time
        ]

        # Apply filters
        if hook_name:
            executions = [e for e in executions if e["hook_name"] == hook_name]

        if hook_type:
            executions = [e for e in executions if e["hook_type"] == hook_type]

        # Sort by timestamp (most recent first) and limit
        executions.sort(key=lambda e: e["timestamp"], reverse=True)
        executions = executions[:limit]

        return {
            "filter_criteria": {
                "hook_name": hook_name,
                "hook_type": hook_type,
                "hours": hours,
                "limit": limit
            },
            "total_matching": len(executions),
            "executions": executions
        }