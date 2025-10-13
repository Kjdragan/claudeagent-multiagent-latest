"""
Core Message Processor - Comprehensive Message Processing Engine

This module provides the core message processing engine with type-specific handlers,
content analysis, enhancement capabilities, and performance optimization.

Key Features:
- Type-specific message processing with comprehensive handlers
- Content analysis and enhancement pipeline
- Performance monitoring and optimization
- Error handling and recovery mechanisms
- Message lifecycle management
- Integration with caching and routing systems
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime
from dataclasses import dataclass

from .message_types import RichMessage, EnhancedMessageType, MessagePriority, MessageContext
from ..analyzers.content_enhancer import ContentEnhancer
from ..analyzers.message_quality_analyzer import MessageQualityAnalyzer
from ..routers.message_router import MessageRouter
from ..cache.message_cache import MessageCache


@dataclass
class ProcessingResult:
    """Result of message processing with comprehensive metadata."""

    success: bool
    processed_message: RichMessage
    processing_time: float
    processor_used: str
    error: Optional[str] = None
    enhancements_applied: List[str] = None
    quality_improvement: float = 0.0

    def __post_init__(self):
        if self.enhancements_applied is None:
            self.enhancements_applied = []


class MessageProcessor:
    """Comprehensive message processing engine with type-specific handlers."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize message processor with configuration."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Core components
        self.content_enhancer = ContentEnhancer(self.config.get("enhancement", {}))
        self.quality_analyzer = MessageQualityAnalyzer(self.config.get("quality", {}))
        self.message_router = MessageRouter(self.config.get("routing", {}))
        self.message_cache = MessageCache(self.config.get("cache", {}))

        # Processing registry
        self.processors: Dict[EnhancedMessageType, Callable] = {}
        self.pre_processors: List[Callable] = []
        self.post_processors: List[Callable] = []

        # Performance tracking
        self.processing_stats: Dict[str, Any] = {
            "total_processed": 0,
            "total_time": 0.0,
            "by_type": {},
            "by_context": {},
            "errors": 0,
            "cache_hits": 0
        }

        # Initialize default processors
        self._initialize_default_processors()
        self._initialize_pipeline_processors()

    def _initialize_default_processors(self):
        """Initialize default type-specific processors."""
        self.processors = {
            # Core content types
            EnhancedMessageType.TEXT: self._process_text_message,
            EnhancedMessageType.MARKDOWN: self._process_markdown_message,
            EnhancedMessageType.CODE: self._process_code_message,
            EnhancedMessageType.JSON: self._process_json_message,

            # Agent communication
            EnhancedMessageType.AGENT_MESSAGE: self._process_agent_message,
            EnhancedMessageType.AGENT_HANDOFF: self._process_agent_handoff,
            EnhancedMessageType.AGENT_STATUS: self._process_agent_status,

            # Tool interactions
            EnhancedMessageType.TOOL_USE: self._process_tool_use,
            EnhancedMessageType.TOOL_RESULT: self._process_tool_result,
            EnhancedMessageType.TOOL_ERROR: self._process_tool_error,

            # Research and analysis
            EnhancedMessageType.RESEARCH_QUERY: self._process_research_query,
            EnhancedMessageType.RESEARCH_RESULT: self._process_research_result,
            EnhancedMessageType.ANALYSIS_RESULT: self._process_analysis_result,
            EnhancedMessageType.CONTENT_SUMMARY: self._process_content_summary,

            # Quality and assessment
            EnhancedMessageType.QUALITY_ASSESSMENT: self._process_quality_assessment,
            EnhancedMessageType.QUALITY_SCORE: self._process_quality_score,
            EnhancedMessageType.VALIDATION_RESULT: self._process_validation_result,

            # Workflow and orchestration
            EnhancedMessageType.WORKFLOW_STAGE: self._process_workflow_stage,
            EnhancedMessageType.PROGRESS_UPDATE: self._process_progress_update,
            EnhancedMessageType.STAGE_TRANSITION: self._process_stage_transition,

            # System messages
            EnhancedMessageType.SYSTEM_ERROR: self._process_system_error,
            EnhancedMessageType.SYSTEM_WARNING: self._process_system_warning,
            EnhancedMessageType.SYSTEM_INFO: self._process_system_info,
            EnhancedMessageType.PERFORMANCE_METRIC: self._process_performance_metric,

            # Specialized research types
            EnhancedMessageType.GAP_RESEARCH: self._process_gap_research,
            EnhancedMessageType.GAP_ANALYSIS: self._process_gap_analysis,
            EnhancedMessageType.RECOMMENDATION: self._process_recommendation,
            EnhancedMessageType.INSIGHT: self._process_insight,
        }

    def _initialize_pipeline_processors(self):
        """Initialize pre-processing and post-processing pipeline."""
        # Pre-processors (run before type-specific processing)
        self.pre_processors = [
            self._validate_message_structure,
            self._apply_routing_rules,
            self._check_cache_hit,
            self._analyze_content_preprocessing,
        ]

        # Post-processors (run after type-specific processing)
        self.post_processors = [
            self._enhance_content,
            self._assess_quality,
            self._update_cache,
            self._track_performance,
            self._update_metadata,
        ]

    async def process_message(self, message: RichMessage) -> ProcessingResult:
        """Process a message through the complete pipeline."""
        start_time = time.time()
        processor_name = "unknown"
        enhancements_applied = []

        try:
            self.logger.debug(f"Processing message {message.id} of type {message.message_type.value}")

            # Mark as processing
            message.mark_processed("MessageProcessor", 0.0)

            # Pre-processing pipeline
            for pre_processor in self.pre_processors:
                try:
                    await pre_processor(message)
                except Exception as e:
                    self.logger.warning(f"Pre-processor {pre_processor.__name__} failed: {str(e)}")

            # Type-specific processing
            processor = self.processors.get(message.message_type, self._process_default_message)
            processor_name = processor.__name__

            pre_quality = message.metadata.quality_score or 0.0
            await processor(message)
            post_quality = message.metadata.quality_score or 0.0

            # Post-processing pipeline
            for post_processor in self.post_processors:
                try:
                    result = await post_processor(message)
                    if result and isinstance(result, list):
                        enhancements_applied.extend(result)
                except Exception as e:
                    self.logger.warning(f"Post-processor {post_processor.__name__} failed: {str(e)}")

            # Calculate processing time
            processing_time = time.time() - start_time
            quality_improvement = post_quality - pre_quality

            # Update message processing metadata
            message.add_processing_step(processor_name, processing_time, "success")
            message.mark_processed(processor_name, processing_time)

            # Update statistics
            self._update_processing_stats(message, processing_time, True)

            self.logger.debug(f"Successfully processed message {message.id} in {processing_time:.3f}s")

            return ProcessingResult(
                success=True,
                processed_message=message,
                processing_time=processing_time,
                processor_used=processor_name,
                enhancements_applied=enhancements_applied,
                quality_improvement=quality_improvement
            )

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Failed to process message {message.id}: {str(e)}"

            self.logger.error(error_msg)
            message.mark_failed(error_msg, processor_name)

            # Update statistics
            self._update_processing_stats(message, processing_time, False)

            return ProcessingResult(
                success=False,
                processed_message=message,
                processing_time=processing_time,
                processor_used=processor_name,
                error=error_msg
            )

    async def process_batch(self, messages: List[RichMessage]) -> List[ProcessingResult]:
        """Process multiple messages in parallel."""
        if not messages:
            return []

        # Process messages concurrently with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.config.get("max_concurrent_processing", 10))

        async def process_with_semaphore(message):
            async with semaphore:
                return await self.process_message(message)

        # Create tasks and run them
        tasks = [process_with_semaphore(msg) for msg in messages]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ProcessingResult(
                    success=False,
                    processed_message=messages[i],
                    processing_time=0.0,
                    processor_used="batch_processor",
                    error=str(result)
                ))
            else:
                processed_results.append(result)

        return processed_results

    # Default processor
    async def _process_default_message(self, message: RichMessage):
        """Default processor for unknown message types."""
        self.logger.warning(f"Using default processor for message type: {message.message_type.value}")

        # Basic content enhancement for unknown types
        if message.content:
            message.formatting.update({
                "style": "text",
                "max_width": 100,
                "word_wrap": True
            })

    # Core content processors
    async def _process_text_message(self, message: RichMessage):
        """Process text messages with formatting and analysis."""
        message.formatting.update({
            "style": "text",
            "max_width": 100,
            "word_wrap": True,
            "highlight_urls": True,
            "highlight_mentions": True
        })

        # Analyze text properties
        words = message.content.split()
        message.metadata.content_length = len(message.content)
        message.content_analysis.update({
            "word_count": len(words),
            "char_count": len(message.content),
            "estimated_reading_time": len(words) / 200,  # words per minute
            "has_urls": "http" in message.content.lower(),
            "has_code_blocks": "```" in message.content,
            "language": self._detect_language(message.content)
        })

    async def _process_markdown_message(self, message: RichMessage):
        """Process markdown messages with enhanced formatting."""
        message.formatting.update({
            "style": "markdown",
            "enable_syntax_highlighting": True,
            "enable_tables": True,
            "enable_links": True,
            "enable_images": True
        })

        # Analyze markdown structure
        content = message.content
        message.content_analysis.update({
            "has_headings": "#" in content,
            "has_lists": any(marker in content for marker in ["* ", "- ", "+ "]),
            "has_code_blocks": "```" in content,
            "has_tables": "|" in content and "\n" in content,
            "has_links": "[" in content and "](" in content,
            "heading_count": content.count("#"),
            "code_block_count": content.count("```") // 2
        })

    async def _process_code_message(self, message: RichMessage):
        """Process code messages with syntax highlighting."""
        # Detect programming language
        language = self._detect_programming_language(message.content)

        message.formatting.update({
            "style": "code",
            "language": language,
            "syntax_highlighting": True,
            "line_numbers": True,
            "theme": "monokai"
        })

        message.content_analysis.update({
            "language": language,
            "line_count": len(message.content.split('\n')),
            "char_count": len(message.content),
            "has_comments": self._has_code_comments(message.content, language),
            "complexity_estimate": self._estimate_code_complexity(message.content)
        })

    async def _process_json_message(self, message: RichMessage):
        """Process JSON messages with validation and formatting."""
        import json

        try:
            # Parse and validate JSON
            parsed_data = json.loads(message.content)

            message.formatting.update({
                "style": "json",
                "syntax_highlighting": True,
                "theme": "monokai",
                "collapsible": True,
                "default_open": False
            })

            message.content_analysis.update({
                "is_valid_json": True,
                "json_type": type(parsed_data).__name__,
                "json_size": len(message.content),
                "has_nested_objects": self._has_nested_objects(parsed_data),
                "max_depth": self._calculate_json_depth(parsed_data)
            })

            # Pretty format JSON
            message.content = json.dumps(parsed_data, indent=2)

        except json.JSONDecodeError as e:
            message.formatting.update({
                "style": "error",
                "error_type": "invalid_json"
            })

            message.content_analysis.update({
                "is_valid_json": False,
                "parse_error": str(e)
            })

            # Add routing tag for error handling
            message.add_routing_tag("json_error")

    # Agent communication processors
    async def _process_agent_message(self, message: RichMessage):
        """Process agent communication messages."""
        message.formatting.update({
            "style": "agent_message",
            "show_agent_info": True,
            "show_timestamp": True,
            "show_priority": True
        })

        # Analyze agent communication
        message.content_analysis.update({
            "communication_type": self._classify_agent_communication(message.content),
            "urgency": self._assess_urgency(message.content),
            "requires_action": self._requires_action(message.content),
            "has_attachments": "attachment" in message.content.lower()
        })

    async def _process_agent_handoff(self, message: RichMessage):
        """Process agent handoff messages with enhanced tracking."""
        message.formatting.update({
            "style": "handoff",
            "show_handoff_info": True,
            "highlight_transitions": True,
            "show_timeline": True
        })

        # Extract handoff details
        handoff_info = self._extract_handoff_info(message.content)
        message.content_analysis.update(handoff_info)

        # Add routing tags for tracking
        if "from_agent" in handoff_info:
            message.add_routing_tag("handoff_from", handoff_info["from_agent"])
        if "to_agent" in handoff_info:
            message.add_routing_tag("handoff_to", handoff_info["to_agent"])

    async def _process_agent_status(self, message: RichMessage):
        """Process agent status messages."""
        message.formatting.update({
            "style": "status",
            "show_status_indicators": True,
            "show_performance_metrics": True
        })

        # Parse status information
        status_info = self._parse_status_message(message.content)
        message.content_analysis.update(status_info)

    # Tool interaction processors
    async def _process_tool_use(self, message: RichMessage):
        """Process tool use messages with execution tracking."""
        tool_info = self._extract_tool_info(message.content)

        message.formatting.update({
            "style": "tool_use",
            "show_tool_name": True,
            "show_parameters": True,
            "show_execution_time": True
        })

        message.content_analysis.update(tool_info)
        message.metadata.routing_tags.extend(["tool_use", tool_info.get("tool_name", "unknown")])

    async def _process_tool_result(self, message: RichMessage):
        """Process tool result messages with success analysis."""
        success = message.metadata.get("success", True)

        message.formatting.update({
            "style": "tool_result",
            "show_success_status": True,
            "collapsible": True,
            "default_open": not success
        })

        # Analyze result quality
        result_analysis = self._analyze_tool_result(message.content, success)
        message.content_analysis.update(result_analysis)

    async def _process_tool_error(self, message: RichMessage):
        """Process tool error messages with enhanced error information."""
        message.formatting.update({
            "style": "error",
            "error_panel": True,
            "show_stack_trace": False,
            "show_suggestions": True
        })

        # Analyze error
        error_analysis = self._analyze_tool_error(message.content)
        message.content_analysis.update(error_analysis)

        # Update message priority based on error severity
        if error_analysis.get("severity") == "critical":
            message.priority = MessagePriority.CRITICAL
        elif error_analysis.get("severity") == "high":
            message.priority = MessagePriority.HIGH

        message.add_routing_tag("tool_error", error_analysis.get("error_type", "unknown"))

    # Research and analysis processors
    async def _process_research_query(self, message: RichMessage):
        """Process research query messages with query analysis."""
        message.formatting.update({
            "style": "research_query",
            "highlight_keywords": True,
            "show_query_analysis": True
        })

        # Analyze research query
        query_analysis = self._analyze_research_query(message.content)
        message.content_analysis.update(query_analysis)

        # Add research-related routing tags
        message.add_routing_tags("research", "query")
        if query_analysis.get("query_type"):
            message.add_routing_tag(f"query_{query_analysis['query_type']}")

    async def _process_research_result(self, message: RichMessage):
        """Process research result messages with result analysis."""
        message.formatting.update({
            "style": "research_result",
            "show_summary": True,
            "show_sources": True,
            "collapsible_sections": True
        })

        # Analyze research result
        result_analysis = self._analyze_research_result(message.content)
        message.content_analysis.update(result_analysis)

        message.add_routing_tags("research", "result")

    async def _process_analysis_result(self, message: RichMessage):
        """Process analysis result messages."""
        message.formatting.update({
            "style": "analysis_result",
            "show_metrics": True,
            "show_visualizations": True
        })

        analysis_info = self._analyze_analysis_result(message.content)
        message.content_analysis.update(analysis_info)

        message.add_routing_tags("analysis", "result")

    async def _process_content_summary(self, message: RichMessage):
        """Process content summary messages."""
        message.formatting.update({
            "style": "summary",
            "show_key_points": True,
            "show_confidence": True
        })

        summary_analysis = self._analyze_content_summary(message.content)
        message.content_analysis.update(summary_analysis)

        message.add_routing_tags("summary", "content")

    # Quality and assessment processors
    async def _process_quality_assessment(self, message: RichMessage):
        """Process quality assessment messages with visual indicators."""
        # Extract quality score from content
        quality_score = self._extract_quality_score(message.content)

        message.formatting.update({
            "style": "quality_assessment",
            "show_score_gauge": True,
            "show_recommendations": True,
            "color_scheme": self._get_quality_color_scheme(quality_score)
        })

        message.set_quality_scores(quality=quality_score)
        message.content_analysis.update({
            "assessment_type": "comprehensive",
            "score": quality_score,
            "quality_level": self._get_quality_level(quality_score)
        })

        message.add_routing_tags("quality", "assessment")

    async def _process_quality_score(self, message: RichMessage):
        """Process quality score messages."""
        score = self._extract_quality_score(message.content)

        message.formatting.update({
            "style": "quality_score",
            "show_score_badge": True,
            "show_trend": True
        })

        message.set_quality_scores(quality=score)
        message.add_routing_tags("quality", "score")

    async def _process_validation_result(self, message: RichMessage):
        """Process validation result messages."""
        validation_info = self._parse_validation_result(message.content)

        message.formatting.update({
            "style": "validation",
            "show_validation_status": True,
            "show_errors": True,
            "show_warnings": True
        })

        message.content_analysis.update(validation_info)
        message.add_routing_tags("validation")

    # Workflow and orchestration processors
    async def _process_workflow_stage(self, message: RichMessage):
        """Process workflow stage messages."""
        stage_info = self._extract_workflow_stage_info(message.content)

        message.formatting.update({
            "style": "workflow_stage",
            "show_stage_info": True,
            "show_progress": True
        })

        message.content_analysis.update(stage_info)
        message.add_routing_tags("workflow", "stage", stage_info.get("stage_name", "unknown"))

    async def _process_progress_update(self, message: RichMessage):
        """Process progress update messages."""
        progress_info = self._extract_progress_info(message.content)

        message.formatting.update({
            "style": "progress",
            "show_progress_bar": True,
            "show_percentage": True,
            "animate_progress": True
        })

        message.content_analysis.update(progress_info)
        message.add_routing_tags("progress")

    async def _process_stage_transition(self, message: RichMessage):
        """Process stage transition messages."""
        transition_info = self._extract_stage_transition_info(message.content)

        message.formatting.update({
            "style": "transition",
            "show_transition_details": True,
            "highlight_changes": True
        })

        message.content_analysis.update(transition_info)
        message.add_routing_tags("workflow", "transition")

    # System message processors
    async def _process_system_error(self, message: RichMessage):
        """Process system error messages."""
        error_info = self._analyze_system_error(message.content)

        message.formatting.update({
            "style": "system_error",
            "error_panel": True,
            "show_system_info": True,
            "show_recovery_options": True
        })

        message.content_analysis.update(error_info)
        message.priority = MessagePriority.HIGH
        message.add_routing_tags("system", "error", error_info.get("error_type", "unknown"))

    async def _process_system_warning(self, message: RichMessage):
        """Process system warning messages."""
        warning_info = self._analyze_system_warning(message.content)

        message.formatting.update({
            "style": "system_warning",
            "warning_panel": True,
            "show_system_info": True
        })

        message.content_analysis.update(warning_info)
        message.add_routing_tags("system", "warning")

    async def _process_system_info(self, message: RichMessage):
        """Process system information messages."""
        message.formatting.update({
            "style": "system_info",
            "info_panel": True,
            "show_system_info": True
        })

        message.add_routing_tags("system", "info")

    async def _process_performance_metric(self, message: RichMessage):
        """Process performance metric messages."""
        metric_info = self._parse_performance_metric(message.content)

        message.formatting.update({
            "style": "performance_metric",
            "show_charts": True,
            "show_trends": True
        })

        message.content_analysis.update(metric_info)
        message.add_routing_tags("performance", "metrics")

    # Specialized research processors
    async def _process_gap_research(self, message: RichMessage):
        """Process gap research messages."""
        gap_info = self._analyze_gap_research(message.content)

        message.formatting.update({
            "style": "gap_research",
            "show_gap_analysis": True,
            "show_recommendations": True
        })

        message.content_analysis.update(gap_info)
        message.add_routing_tags("research", "gap", "gap_research")

    async def _process_gap_analysis(self, message: RichMessage):
        """Process gap analysis messages."""
        analysis_info = self._analyze_gap_analysis(message.content)

        message.formatting.update({
            "style": "gap_analysis",
            "show_analysis_details": True,
            "show_visualizations": True
        })

        message.content_analysis.update(analysis_info)
        message.add_routing_tags("research", "gap", "analysis")

    async def _process_recommendation(self, message: RichMessage):
        """Process recommendation messages."""
        rec_info = self._analyze_recommendation(message.content)

        message.formatting.update({
            "style": "recommendation",
            "show_priority": True,
            "show_impact": True
        })

        message.content_analysis.update(rec_info)
        message.add_routing_tags("recommendation")

    async def _process_insight(self, message: RichMessage):
        """Process insight messages."""
        insight_info = self._analyze_insight(message.content)

        message.formatting.update({
            "style": "insight",
            "highlight_insight": True,
            "show_confidence": True
        })

        message.content_analysis.update(insight_info)
        message.add_routing_tags("insight")

    # Pre-processing pipeline methods
    async def _validate_message_structure(self, message: RichMessage):
        """Validate message structure and required fields."""
        if not message.content and message.message_type not in [
            EnhancedMessageType.SYSTEM_INFO,
            EnhancedMessageType.PERFORMANCE_METRIC
        ]:
            raise ValueError(f"Message {message.id} has no content")

        if not message.id:
            raise ValueError(f"Message has no ID")

    async def _apply_routing_rules(self, message: RichMessage):
        """Apply routing rules to determine message destination."""
        routing_decision = await self.message_router.route_message(message)
        message.routing_info.update(routing_decision)

    async def _check_cache_hit(self, message: RichMessage):
        """Check if message can be served from cache."""
        cached_result = await self.message_cache.get(message)
        if cached_result:
            message.metadata.cache_hit = True
            # Apply cached processing results
            message.formatting.update(cached_result.get("formatting", {}))
            message.content_analysis.update(cached_result.get("content_analysis", {}))
            self.processing_stats["cache_hits"] += 1

    async def _analyze_content_preprocessing(self, message: RichMessage):
        """Analyze content for basic preprocessing information."""
        if message.content:
            message.content_analysis.update({
                "language": self._detect_language(message.content),
                "sentiment": self._analyze_sentiment(message.content),
                "complexity": self._analyze_complexity(message.content),
                "keywords": self._extract_keywords(message.content)
            })

    # Post-processing pipeline methods
    async def _enhance_content(self, message: RichMessage) -> List[str]:
        """Enhance message content based on analysis."""
        if not self.config.get("enable_enhancement", True):
            return []

        enhancements = await self.content_enhancer.enhance_message(message)
        return enhancements

    async def _assess_quality(self, message: RichMessage):
        """Assess message quality."""
        if not self.config.get("enable_quality_assessment", True):
            return

        quality_result = await self.quality_analyzer.assess_message(message)
        message.set_quality_scores(
            quality=quality_result.get("overall_quality"),
            relevance=quality_result.get("relevance"),
            confidence=quality_result.get("confidence")
        )

    async def _update_cache(self, message: RichMessage):
        """Update cache with processed message."""
        if not self.config.get("enable_caching", True):
            return

        await self.message_cache.set(message, {
            "formatting": message.formatting,
            "content_analysis": message.content_analysis,
            "quality_scores": {
                "quality": message.metadata.quality_score,
                "relevance": message.metadata.relevance_score,
                "confidence": message.metadata.confidence_score
            }
        })

    async def _track_performance(self, message: RichMessage):
        """Track processing performance."""
        processing_time = message.performance_metrics.get("total_processing_time", 0.0)
        message.performance_metrics.update({
            "processing_timestamp": datetime.now().isoformat(),
            "message_size": len(message.content),
            "processing_efficiency": len(message.content) / max(processing_time, 0.001)
        })

    async def _update_metadata(self, message: RichMessage):
        """Update message metadata with processing information."""
        message.metadata.updated_at = datetime.now()
        message.metadata.processing_time = message.performance_metrics.get("total_processing_time", 0.0)

    # Utility methods
    def _detect_language(self, text: str) -> str:
        """Detect language of text content."""
        # Simple language detection - could be enhanced with proper NLP
        if any(ord(char) > 127 for char in text):
            return "unknown"
        return "en"

    def _detect_programming_language(self, code: str) -> str:
        """Detect programming language from code content."""
        code_lower = code.lower()

        if "def " in code and "import " in code:
            return "python"
        elif "function" in code or "const " in code or "let " in code:
            return "javascript"
        elif "public class" in code or "private " in code:
            return "java"
        elif "#include" in code or "int main" in code:
            return "cpp"
        elif "package main" in code or "func " in code:
            return "go"
        else:
            return "text"

    def _has_code_comments(self, code: str, language: str) -> bool:
        """Check if code contains comments."""
        if language == "python":
            return "#" in code
        elif language in ["javascript", "java", "cpp"]:
            return "//" in code or "/*" in code
        return False

    def _estimate_code_complexity(self, code: str) -> str:
        """Estimate code complexity."""
        lines = len(code.split('\n'))
        if lines < 10:
            return "low"
        elif lines < 50:
            return "medium"
        else:
            return "high"

    def _has_nested_objects(self, data) -> bool:
        """Check if JSON data has nested objects."""
        if isinstance(data, dict):
            return any(isinstance(v, (dict, list)) for v in data.values())
        elif isinstance(data, list):
            return any(isinstance(item, (dict, list)) for item in data)
        return False

    def _calculate_json_depth(self, data, current_depth=0) -> int:
        """Calculate maximum depth of JSON structure."""
        if isinstance(data, dict):
            return max([self._calculate_json_depth(v, current_depth + 1) for v in data.values()], default=current_depth)
        elif isinstance(data, list):
            return max([self._calculate_json_depth(item, current_depth + 1) for item in data], default=current_depth)
        return current_depth

    def _analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of text."""
        positive_words = ["good", "great", "excellent", "success", "completed", "achieved"]
        negative_words = ["error", "failed", "bad", "poor", "issue", "problem"]

        positive_count = sum(1 for word in positive_words if word in text.lower())
        negative_count = sum(1 for word in negative_words if word in text.lower())

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        return "neutral"

    def _analyze_complexity(self, text: str) -> str:
        """Analyze text complexity."""
        sentences = text.count('.') + text.count('!') + text.count('?')
        words = len(text.split())

        if words == 0:
            return "simple"

        avg_words_per_sentence = words / max(sentences, 1)

        if avg_words_per_sentence < 10:
            return "simple"
        elif avg_words_per_sentence < 20:
            return "medium"
        return "complex"

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Simple keyword extraction
        import re
        words = re.findall(r'\b\w+\b', text.lower())

        # Filter out common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they"}

        keywords = [word for word in words if len(word) > 3 and word not in stop_words]

        # Return most common keywords
        from collections import Counter
        return [word for word, count in Counter(keywords).most_common(5)]

    def _update_processing_stats(self, message: RichMessage, processing_time: float, success: bool):
        """Update processing statistics."""
        self.processing_stats["total_processed"] += 1
        self.processing_stats["total_time"] += processing_time

        if not success:
            self.processing_stats["errors"] += 1

        # Update by type
        msg_type = message.message_type.value
        if msg_type not in self.processing_stats["by_type"]:
            self.processing_stats["by_type"][msg_type] = {
                "count": 0, "total_time": 0.0, "errors": 0
            }

        self.processing_stats["by_type"][msg_type]["count"] += 1
        self.processing_stats["by_type"][msg_type]["total_time"] += processing_time
        if not success:
            self.processing_stats["by_type"][msg_type]["errors"] += 1

        # Update by context
        ctx = message.context.value
        if ctx not in self.processing_stats["by_context"]:
            self.processing_stats["by_context"][ctx] = {
                "count": 0, "total_time": 0.0, "errors": 0
            }

        self.processing_stats["by_context"][ctx]["count"] += 1
        self.processing_stats["by_context"][ctx]["total_time"] += processing_time
        if not success:
            self.processing_stats["by_context"][ctx]["errors"] += 1

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        stats = self.processing_stats.copy()

        # Calculate averages
        if stats["total_processed"] > 0:
            stats["average_processing_time"] = stats["total_time"] / stats["total_processed"]
            stats["error_rate"] = stats["errors"] / stats["total_processed"]
        else:
            stats["average_processing_time"] = 0.0
            stats["error_rate"] = 0.0

        # Calculate cache hit rate
        if stats["total_processed"] > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / stats["total_processed"]
        else:
            stats["cache_hit_rate"] = 0.0

        return stats

    def reset_stats(self):
        """Reset processing statistics."""
        self.processing_stats = {
            "total_processed": 0,
            "total_time": 0.0,
            "by_type": {},
            "by_context": {},
            "errors": 0,
            "cache_hits": 0
        }

    # Additional placeholder methods for specialized processing
    def _classify_agent_communication(self, content: str) -> str:
        """Classify agent communication type."""
        return "general"

    def _assess_urgency(self, content: str) -> str:
        """Assess message urgency."""
        return "normal"

    def _requires_action(self, content: str) -> bool:
        """Check if message requires action."""
        return False

    def _extract_handoff_info(self, content: str) -> Dict[str, Any]:
        """Extract handoff information from content."""
        return {}

    def _parse_status_message(self, content: str) -> Dict[str, Any]:
        """Parse status message content."""
        return {}

    def _extract_tool_info(self, content: str) -> Dict[str, Any]:
        """Extract tool information from content."""
        return {}

    def _analyze_tool_result(self, content: str, success: bool) -> Dict[str, Any]:
        """Analyze tool result."""
        return {"success": success}

    def _analyze_tool_error(self, content: str) -> Dict[str, Any]:
        """Analyze tool error."""
        return {"severity": "medium"}

    def _analyze_research_query(self, content: str) -> Dict[str, Any]:
        """Analyze research query."""
        return {"query_type": "general"}

    def _analyze_research_result(self, content: str) -> Dict[str, Any]:
        """Analyze research result."""
        return {"result_type": "summary"}

    def _analyze_analysis_result(self, content: str) -> Dict[str, Any]:
        """Analyze analysis result."""
        return {}

    def _analyze_content_summary(self, content: str) -> Dict[str, Any]:
        """Analyze content summary."""
        return {}

    def _extract_quality_score(self, content: str) -> float:
        """Extract quality score from content."""
        import re
        score_match = re.search(r'(\d+\.?\d*)', content)
        return float(score_match.group(1)) if score_match else 0.0

    def _get_quality_color_scheme(self, score: float) -> str:
        """Get color scheme based on quality score."""
        if score >= 0.8:
            return "green"
        elif score >= 0.6:
            return "yellow"
        else:
            return "red"

    def _get_quality_level(self, score: float) -> str:
        """Get quality level based on score."""
        if score >= 0.9:
            return "excellent"
        elif score >= 0.8:
            return "good"
        elif score >= 0.6:
            return "fair"
        else:
            return "poor"

    def _parse_validation_result(self, content: str) -> Dict[str, Any]:
        """Parse validation result."""
        return {"valid": True}

    def _extract_workflow_stage_info(self, content: str) -> Dict[str, Any]:
        """Extract workflow stage information."""
        return {"stage_name": "unknown"}

    def _extract_progress_info(self, content: str) -> Dict[str, Any]:
        """Extract progress information."""
        return {"percentage": 0.0}

    def _extract_stage_transition_info(self, content: str) -> Dict[str, Any]:
        """Extract stage transition information."""
        return {"from_stage": "unknown", "to_stage": "unknown"}

    def _analyze_system_error(self, content: str) -> Dict[str, Any]:
        """Analyze system error."""
        return {"error_type": "general", "severity": "high"}

    def _analyze_system_warning(self, content: str) -> Dict[str, Any]:
        """Analyze system warning."""
        return {"warning_type": "general"}

    def _parse_performance_metric(self, content: str) -> Dict[str, Any]:
        """Parse performance metric."""
        return {}

    def _analyze_gap_research(self, content: str) -> Dict[str, Any]:
        """Analyze gap research."""
        return {}

    def _analyze_gap_analysis(self, content: str) -> Dict[str, Any]:
        """Analyze gap analysis."""
        return {}

    def _analyze_recommendation(self, content: str) -> Dict[str, Any]:
        """Analyze recommendation."""
        return {}

    def _analyze_insight(self, content: str) -> Dict[str, Any]:
        """Analyze insight."""
        return {}