"""
Rich Display Formatter - Advanced Message Formatting and Visualization

This module provides sophisticated message formatting and visualization capabilities
with rich display patterns, proper styling, and enhanced user experience.

Key Features:
- Rich text formatting with syntax highlighting
- Tool call visualization with input/output display
- Error categorization and formatted display
- Quality score visualization with visual indicators
- Progress bars and status indicators
- Structured data display (tables, lists, trees)
- Interactive elements where appropriate
- Color-coded message types with consistent theming
- Timestamp and attribution formatting
- Accessibility-compliant formatting
"""

import asyncio
import json
import re
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.tree import Tree
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    from rich.columns import Columns
    from rich.align import Align
    from rich.rule import Rule
    from rich.live import Live
    from rich.status import Status
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from ..core.message_types import RichMessage, EnhancedMessageType, MessagePriority


class DisplayStyle(Enum):
    """Display style options for different message types."""

    PANEL = "panel"
    INLINE = "inline"
    COLLAPSIBLE = "collapsible"
    EXPANDED = "expanded"
    COMPACT = "compact"
    DETAILED = "detailed"
    INTERACTIVE = "interactive"


class ColorScheme(Enum):
    """Color schemes for message formatting."""

    DEFAULT = "default"
    DARK = "dark"
    LIGHT = "light"
    HIGH_CONTRAST = "high_contrast"
    MONOCHROME = "monochrome"


@dataclass
class DisplayConfig:
    """Configuration for message display formatting."""

    style: DisplayStyle = DisplayStyle.PANEL
    color_scheme: ColorScheme = ColorScheme.DEFAULT
    show_timestamp: bool = True
    show_metadata: bool = False
    show_quality_indicators: bool = True
    max_content_length: int = 2000
    enable_syntax_highlighting: bool = True
    enable_progress_animation: bool = True
    accessibility_mode: bool = False
    compact_mode: bool = False


class RichFormatter:
    """Advanced rich message formatter with comprehensive visualization capabilities."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize rich formatter with configuration."""
        self.config = config or {}
        self.display_config = self._create_display_config()

        # Initialize Rich console if available
        if RICH_AVAILABLE:
            self.console = Console(
                color_system=self._get_color_system(),
                file=None,
                force_terminal=True,
                legacy_windows=False,
                no_color=self.display_config.color_scheme == ColorScheme.MONOCHROME,
                width=self.config.get("console_width", 120)
            )
        else:
            self.console = None

        # Style definitions
        self.styles = self._create_style_definitions()

        # Formatting statistics
        self.formatting_stats = {
            "total_formatted": 0,
            "by_type": {},
            "by_style": {},
            "formatting_time": 0.0
        }

    def _create_display_config(self) -> DisplayConfig:
        """Create display configuration from settings."""
        return DisplayConfig(
            style=DisplayStyle(self.config.get("default_style", "panel")),
            color_scheme=ColorScheme(self.config.get("color_scheme", "default")),
            show_timestamp=self.config.get("show_timestamp", True),
            show_metadata=self.config.get("show_metadata", False),
            show_quality_indicators=self.config.get("show_quality_indicators", True),
            max_content_length=self.config.get("max_content_length", 2000),
            enable_syntax_highlighting=self.config.get("enable_syntax_highlighting", True),
            enable_progress_animation=self.config.get("enable_progress_animation", True),
            accessibility_mode=self.config.get("accessibility_mode", False),
            compact_mode=self.config.get("compact_mode", False)
        )

    def _create_style_definitions(self) -> Dict[str, Any]:
        """Create style definitions for different message types."""
        return {
            # Message type styles
            EnhancedMessageType.TEXT: {
                "title": "Text Message",
                "border_style": "blue",
                "title_style": "bold blue",
                "content_style": "white",
                "background_color": None
            },
            EnhancedMessageType.ERROR: {
                "title": "Error",
                "border_style": "red",
                "title_style": "bold red",
                "content_style": "red",
                "background_color": "red3"
            },
            EnhancedMessageType.WARNING: {
                "title": "Warning",
                "border_style": "yellow",
                "title_style": "bold yellow",
                "content_style": "yellow",
                "background_color": "yellow3"
            },
            EnhancedMessageType.SUCCESS: {
                "title": "Success",
                "border_style": "green",
                "title_style": "bold green",
                "content_style": "green",
                "background_color": "green3"
            },
            EnhancedMessageType.TOOL_USE: {
                "title": "Tool Use",
                "border_style": "cyan",
                "title_style": "bold cyan",
                "content_style": "cyan",
                "background_color": None
            },
            EnhancedMessageType.TOOL_RESULT: {
                "title": "Tool Result",
                "border_style": "magenta",
                "title_style": "bold magenta",
                "content_style": "magenta",
                "background_color": None
            },
            EnhancedMessageType.QUALITY_ASSESSMENT: {
                "title": "Quality Assessment",
                "border_style": "purple",
                "title_style": "bold purple",
                "content_style": "purple",
                "background_color": None
            },
            EnhancedMessageType.PROGRESS_UPDATE: {
                "title": "Progress",
                "border_style": "white",
                "title_style": "bold white",
                "content_style": "white",
                "background_color": None
            },
            EnhancedMessageType.AGENT_HANDOFF: {
                "title": "Agent Handoff",
                "border_style": "orange1",
                "title_style": "bold orange1",
                "content_style": "orange1",
                "background_color": None
            },
            EnhancedMessageType.RESEARCH_RESULT: {
                "title": "Research Result",
                "border_style": "blue",
                "title_style": "bold blue",
                "content_style": "blue",
                "background_color": None
            },
            EnhancedMessageType.ANALYSIS_RESULT: {
                "title": "Analysis Result",
                "border_style": "cyan",
                "title_style": "bold cyan",
                "content_style": "cyan",
                "background_color": None
            },
        }

    def _get_color_system(self) -> str:
        """Get appropriate color system based on configuration."""
        if self.display_config.color_scheme == ColorScheme.MONOCHROME:
            return "monochrome"
        elif self.display_config.color_scheme == ColorScheme.LIGHT:
            return "standard"
        elif self.display_config.color_scheme == ColorScheme.HIGH_CONTRAST:
            return "256"
        else:  # DEFAULT or DARK
            return "auto"

    async def format_message(self, message: RichMessage) -> str:
        """Format a message for rich display."""
        start_time = datetime.now()

        try:
            if not RICH_AVAILABLE:
                return self._format_fallback(message)

            # Choose formatting method based on message type and configuration
            formatter = self._get_formatter(message)
            formatted_output = await formatter(message)

            # Update statistics
            formatting_time = (datetime.now() - start_time).total_seconds()
            self._update_formatting_stats(message, formatting_time)

            return formatted_output

        except Exception as e:
            # Fallback to simple formatting if rich formatting fails
            return self._format_error(message, str(e))

    def _get_formatter(self, message: RichMessage) -> callable:
        """Get appropriate formatter for message type."""
        formatters = {
            EnhancedMessageType.TEXT: self._format_text_message,
            EnhancedMessageType.MARKDOWN: self._format_markdown_message,
            EnhancedMessageType.CODE: self._format_code_message,
            EnhancedMessageType.JSON: self._format_json_message,
            EnhancedMessageType.ERROR: self._format_error_message,
            EnhancedMessageType.WARNING: self._format_warning_message,
            EnhancedMessageType.SUCCESS: self._format_success_message,
            EnhancedMessageType.TOOL_USE: self._format_tool_use_message,
            EnhancedMessageType.TOOL_RESULT: self._format_tool_result_message,
            EnhancedMessageType.QUALITY_ASSESSMENT: self._format_quality_assessment_message,
            EnhancedMessageType.PROGRESS_UPDATE: self._format_progress_message,
            EnhancedMessageType.AGENT_HANDOFF: self._format_agent_handoff_message,
            EnhancedMessageType.RESEARCH_RESULT: self._format_research_result_message,
            EnhancedMessageType.ANALYSIS_RESULT: self._format_analysis_result_message,
            EnhancedMessageType.RECOMMENDATION: self._format_recommendation_message,
            EnhancedMessageType.INSIGHT: self._format_insight_message,
        }

        return formatters.get(message.message_type, self._format_default_message)

    async def _format_text_message(self, message: RichMessage) -> str:
        """Format text message with rich styling."""
        style = self.styles.get(message.message_type, self.styles[EnhancedMessageType.TEXT])

        # Prepare content
        content = self._prepare_content(message.content)

        # Create text object with styling
        text_content = Text(content, style=style["content_style"])

        # Add timestamp if configured
        title = style["title"]
        if self.display_config.show_timestamp:
            timestamp = message.timestamps.get("created", datetime.now()).strftime("%H:%M:%S")
            title = f"[{timestamp}] {title}"

        # Add quality indicator if configured
        if self.display_config.show_quality_indicators and message.metadata.quality_score:
            quality_score = message.metadata.quality_score
            quality_emoji = self._get_quality_emoji(quality_score)
            title = f"{title} {quality_emoji}"

        # Create panel
        panel = Panel(
            text_content,
            title=title,
            border_style=style["border_style"],
            title_align="left",
            subtitle=self._get_subtitle(message) if self.display_config.show_metadata else None
        )

        # Capture output
        with self.console.capture() as capture:
            self.console.print(panel)

        return capture.get()

    async def _format_markdown_message(self, message: RichMessage) -> str:
        """Format markdown message with proper rendering."""
        content = self._prepare_content(message.content)

        # Create Markdown object
        markdown = Markdown(content)

        # Create panel with markdown content
        panel = Panel(
            markdown,
            title="📝 Markdown",
            border_style="blue",
            title_align="left"
        )

        with self.console.capture() as capture:
            self.console.print(panel)

        return capture.get()

    async def _format_code_message(self, message: RichMessage) -> str:
        """Format code message with syntax highlighting."""
        content = message.content
        language = message.formatting.get("language", "text")

        # Create syntax object
        syntax = Syntax(
            content,
            language,
            theme="monokai",
            line_numbers=True,
            word_wrap=True
        )

        # Create panel with syntax-highlighted code
        panel = Panel(
            syntax,
            title=f"💻 Code ({language})",
            border_style="cyan",
            title_align="left"
        )

        with self.console.capture() as capture:
            self.console.print(panel)

        return capture.get()

    async def _format_json_message(self, message: RichMessage) -> str:
        """Format JSON message with proper formatting."""
        content = message.content

        try:
            # Parse and reformat JSON
            parsed_json = json.loads(content)
            formatted_json = json.dumps(parsed_json, indent=2)

            # Create syntax object for JSON
            syntax = Syntax(
                formatted_json,
                "json",
                theme="monokai",
                line_numbers=False,
                word_wrap=True
            )

            panel = Panel(
                syntax,
                title="📄 JSON Data",
                border_style="magenta",
                title_align="left"
            )

        except json.JSONDecodeError:
            # Handle invalid JSON
            error_text = Text(f"❌ Invalid JSON:\n{content}", style="red")
            panel = Panel(
                error_text,
                title="📄 JSON Data (Parse Error)",
                border_style="red",
                title_align="left"
            )

        with self.console.capture() as capture:
            self.console.print(panel)

        return capture.get()

    async def _format_error_message(self, message: RichMessage) -> str:
        """Format error message with enhanced error display."""
        content = message.content

        # Analyze error content
        error_analysis = self._analyze_error_content(content)

        # Create error content with styling
        error_text = Text()
        error_text.append("❌ ", style="bold red")
        error_text.append("Error", style="bold red")
        error_text.append("\n\n", style="red")
        error_text.append(content, style="red")

        # Add error details if available
        if error_analysis["error_type"]:
            error_text.append(f"\n\n🏷️  Type: {error_analysis['error_type']}", style="yellow")

        if error_analysis["suggestions"]:
            error_text.append("\n\n💡 Suggestions:", style="green")
            for suggestion in error_analysis["suggestions"]:
                error_text.append(f"\n• {suggestion}", style="green")

        panel = Panel(
            error_text,
            title="🚨 Error",
            border_style="red",
            title_align="left",
            subtitle=error_analysis.get("severity", "Error")
        )

        with self.console.capture() as capture:
            self.console.print(panel)

        return capture.get()

    async def _format_warning_message(self, message: RichMessage) -> str:
        """Format warning message with appropriate styling."""
        content = message.content

        warning_text = Text()
        warning_text.append("⚠️ ", style="bold yellow")
        warning_text.append("Warning", style="bold yellow")
        warning_text.append("\n\n", style="yellow")
        warning_text.append(content, style="yellow")

        panel = Panel(
            warning_text,
            title="⚠️ Warning",
            border_style="yellow",
            title_align="left"
        )

        with self.console.capture() as capture:
            self.console.print(panel)

        return capture.get()

    async def _format_success_message(self, message: RichMessage) -> str:
        """Format success message with celebratory styling."""
        content = message.content

        success_text = Text()
        success_text.append("✅ ", style="bold green")
        success_text.append("Success", style="bold green")
        success_text.append("\n\n", style="green")
        success_text.append(content, style="green")

        panel = Panel(
            success_text,
            title="✅ Success",
            border_style="green",
            title_align="left"
        )

        with self.console.capture() as capture:
            self.console.print(panel)

        return capture.get()

    async def _format_tool_use_message(self, message: RichMessage) -> str:
        """Format tool use message with detailed tool information."""
        content = message.content
        tool_name = message.formatting.get("tool_name", "Unknown Tool")

        # Create tool information
        tool_text = Text()
        tool_text.append("🔧 ", style="bold cyan")
        tool_text.append(f"Tool: {tool_name}", style="bold cyan")
        tool_text.append("\n\n", style="cyan")
        tool_text.append("Input:", style="bold cyan")
        tool_text.append("\n", style="cyan")

        # Format tool input
        if self._is_json(content):
            try:
                tool_input = json.loads(content)
                formatted_input = json.dumps(tool_input, indent=2)
                syntax = Syntax(formatted_input, "json", theme="monokai", line_numbers=False)

                with self.console.capture() as capture:
                    self.console.print(syntax)

                tool_text.append(capture.get(), style="cyan")
            except json.JSONDecodeError:
                tool_text.append(content, style="cyan")
        else:
            tool_text.append(content, style="cyan")

        panel = Panel(
            tool_text,
            title=f"🔧 Tool Use: {tool_name}",
            border_style="cyan",
            title_align="left"
        )

        with self.console.capture() as capture:
            self.console.print(panel)

        return capture.get()

    async def _format_tool_result_message(self, message: RichMessage) -> str:
        """Format tool result message with success analysis."""
        content = message.content
        success = message.metadata.get("success", True)

        # Create result content
        result_text = Text()
        if success:
            result_text.append("✅ ", style="bold green")
            result_text.append("Tool Result", style="bold green")
        else:
            result_text.append("❌ ", style="bold red")
            result_text.append("Tool Result (Failed)", style="bold red")

        result_text.append("\n\n", style="green" if success else "red")

        # Format result content
        if len(content) > 500:
            # Show truncated content for long results
            result_text.append("Result (truncated):", style="bold")
            result_text.append("\n", style="")
            result_text.append(content[:500] + "...", style="")
            result_text.append(f"\n\n[Full result: {len(content)} characters]", style="dim")
        else:
            if self._is_json(content):
                try:
                    result_data = json.loads(content)
                    formatted_result = json.dumps(result_data, indent=2)
                    syntax = Syntax(formatted_result, "json", theme="monokai", line_numbers=False)

                    with self.console.capture() as capture:
                        self.console.print(syntax)

                    result_text.append(capture.get(), style="")
                except json.JSONDecodeError:
                    result_text.append(content, style="")
            else:
                result_text.append(content, style="")

        border_style = "green" if success else "red"
        panel = Panel(
            result_text,
            title="📊 Tool Result",
            border_style=border_style,
            title_align="left"
        )

        with self.console.capture() as capture:
            self.console.print(panel)

        return capture.get()

    async def _format_quality_assessment_message(self, message: RichMessage) -> str:
        """Format quality assessment message with visual indicators."""
        content = message.content
        quality_score = message.metadata.quality_score or 0.0

        # Create quality visualization
        quality_text = Text()
        quality_text.append("📊 ", style="bold purple")
        quality_text.append("Quality Assessment", style="bold purple")
        quality_text.append(f"\n\nScore: {quality_score:.1%}", style="bold purple")
        quality_text.append(f" {self._get_quality_emoji(quality_score)}", style="bold")

        # Add quality bar
        quality_text.append("\n", style="")
        quality_bar = self._create_quality_bar(quality_score)
        quality_text.append(quality_bar, style="")

        # Add assessment content
        quality_text.append("\n\n", style="")
        if self._is_json(content):
            try:
                assessment_data = json.loads(content)
                # Show key metrics
                if "overall_quality" in assessment_data:
                    quality_text.append(f"Overall Quality: {assessment_data['overall_quality']:.1%}\n", style="purple")
                if "recommendations" in assessment_data:
                    quality_text.append("\n💡 Recommendations:", style="bold green")
                    for rec in assessment_data["recommendations"][:3]:  # Show top 3
                        quality_text.append(f"\n• {rec}", style="green")
            except json.JSONDecodeError:
                quality_text.append(content, style="purple")
        else:
            quality_text.append(content, style="purple")

        panel = Panel(
            quality_text,
            title="📊 Quality Assessment",
            border_style="purple",
            title_align="left"
        )

        with self.console.capture() as capture:
            self.console.print(panel)

        return capture.get()

    async def _format_progress_message(self, message: RichMessage) -> str:
        """Format progress message with visual progress indicators."""
        content = message.content

        # Extract progress information
        progress_info = self._extract_progress_info(content)

        progress_text = Text()
        progress_text.append("⏳ ", style="bold blue")
        progress_text.append("Progress Update", style="bold blue")

        if progress_info["percentage"] is not None:
            progress_text.append(f"\n\n{progress_info['percentage']:.1%} Complete", style="bold blue")

            # Create progress bar
            progress_bar = self._create_progress_bar(progress_info["percentage"])
            progress_text.append(f"\n{progress_bar}", style="blue")

        if progress_info["stage"]:
            progress_text.append(f"\n\nStage: {progress_info['stage']}", style="blue")

        if progress_info["details"]:
            progress_text.append(f"\n\n{progress_info['details']}", style="blue")

        panel = Panel(
            progress_text,
            title="⏳ Progress",
            border_style="blue",
            title_align="left"
        )

        with self.console.capture() as capture:
            self.console.print(panel)

        return capture.get()

    async def _format_agent_handoff_message(self, message: RichMessage) -> str:
        """Format agent handoff message with clear handoff visualization."""
        content = message.content

        # Extract handoff information
        handoff_info = self._extract_handoff_info(content)

        handoff_text = Text()
        handoff_text.append("🔄 ", style="bold orange1")
        handoff_text.append("Agent Handoff", style="bold orange1")

        if handoff_info["from_agent"] and handoff_info["to_agent"]:
            handoff_text.append(f"\n\n{handoff_info['from_agent']} ➜ {handoff_info['to_agent']}", style="bold orange1")

        if handoff_info["reason"]:
            handoff_text.append(f"\n\nReason: {handoff_info['reason']}", style="orange1")

        handoff_text.append(f"\n\n{content}", style="orange1")

        panel = Panel(
            handoff_text,
            title="🔄 Agent Handoff",
            border_style="orange1",
            title_align="left"
        )

        with self.console.capture() as capture:
            self.console.print(panel)

        return capture.get()

    async def _format_research_result_message(self, message: RichMessage) -> str:
        """Format research result message with structured display."""
        content = message.content

        research_text = Text()
        research_text.append("🔍 ", style="bold blue")
        research_text.append("Research Result", style="bold blue")

        # Add research metadata if available
        if message.content_analysis.get("source_count"):
            research_text.append(f"\n\n📚 Sources: {message.content_analysis['source_count']}", style="blue")

        if message.content_analysis.get("confidence"):
            research_text.append(f"\n🎯 Confidence: {message.content_analysis['confidence']:.1%}", style="blue")

        research_text.append(f"\n\n{content}", style="blue")

        panel = Panel(
            research_text,
            title="🔍 Research Result",
            border_style="blue",
            title_align="left"
        )

        with self.console.capture() as capture:
            self.console.print(panel)

        return capture.get()

    async def _format_analysis_result_message(self, message: RichMessage) -> str:
        """Format analysis result message with structured display."""
        content = message.content

        analysis_text = Text()
        analysis_text.append("📈 ", style="bold cyan")
        analysis_text.append("Analysis Result", style="bold cyan")

        # Add analysis metadata
        if message.metadata.confidence_score:
            analysis_text.append(f"\n\n🎯 Confidence: {message.metadata.confidence_score:.1%}", style="cyan")

        analysis_text.append(f"\n\n{content}", style="cyan")

        panel = Panel(
            analysis_text,
            title="📈 Analysis Result",
            border_style="cyan",
            title_align="left"
        )

        with self.console.capture() as capture:
            self.console.print(panel)

        return capture.get()

    async def _format_recommendation_message(self, message: RichMessage) -> str:
        """Format recommendation message with prioritized display."""
        content = message.content

        recommendation_text = Text()
        recommendation_text.append("💡 ", style="bold green")
        recommendation_text.append("Recommendation", style="bold green")

        # Extract and format recommendations
        recommendations = self._extract_recommendations(content)

        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                recommendation_text.append(f"\n\n{i}. {rec}", style="green")
        else:
            recommendation_text.append(f"\n\n{content}", style="green")

        panel = Panel(
            recommendation_text,
            title="💡 Recommendation",
            border_style="green",
            title_align="left"
        )

        with self.console.capture() as capture:
            self.console.print(panel)

        return capture.get()

    async def _format_insight_message(self, message: RichMessage) -> str:
        """Format insight message with emphasis display."""
        content = message.content

        insight_text = Text()
        insight_text.append("💭 ", style="bold yellow")
        insight_text.append("Insight", style="bold yellow")

        if message.metadata.confidence_score:
            insight_text.append(f"\n\n🎯 Confidence: {message.metadata.confidence_score:.1%}", style="yellow")

        insight_text.append(f"\n\n{content}", style="yellow")

        panel = Panel(
            insight_text,
            title="💭 Insight",
            border_style="yellow",
            title_align="left"
        )

        with self.console.capture() as capture:
            self.console.print(panel)

        return capture.get()

    async def _format_default_message(self, message: RichMessage) -> str:
        """Format unknown message types with generic styling."""
        content = self._prepare_content(message.content)

        default_text = Text(content, style="white")

        panel = Panel(
            default_text,
            title=f"📨 {message.message_type.value.title()}",
            border_style="white",
            title_align="left"
        )

        with self.console.capture() as capture:
            self.console.print(panel)

        return capture.get()

    # Helper methods
    def _prepare_content(self, content: str) -> str:
        """Prepare content for display (truncation, cleaning)."""
        if len(content) > self.display_config.max_content_length:
            return content[:self.display_config.max_content_length] + f"\n\n... [Content truncated: {len(content)} total characters]"
        return content

    def _get_subtitle(self, message: RichMessage) -> Optional[str]:
        """Get subtitle for message panel."""
        if message.metadata.source_agent and message.metadata.target_agent:
            return f"{message.metadata.source_agent} → {message.metadata.target_agent}"
        elif message.metadata.source_agent:
            return f"From: {message.metadata.source_agent}"
        elif message.session_id:
            return f"Session: {message.session_id}"
        return None

    def _get_quality_emoji(self, score: float) -> str:
        """Get quality emoji based on score."""
        if score >= 0.9:
            return "🏆"
        elif score >= 0.8:
            return "✨"
        elif score >= 0.7:
            return "👍"
        elif score >= 0.6:
            return "👌"
        else:
            return "👎"

    def _create_quality_bar(self, score: float) -> str:
        """Create visual quality bar."""
        bar_length = 20
        filled_length = int(bar_length * score)

        if score >= 0.8:
            bar_char = "█"
            bar_color = "green"
        elif score >= 0.6:
            bar_char = "▓"
            bar_color = "yellow"
        else:
            bar_char = "░"
            bar_color = "red"

        bar = bar_char * filled_length + "░" * (bar_length - filled_length)
        return f"[{bar_color}]{bar}[/{bar_color}] {score:.1%}"

    def _create_progress_bar(self, percentage: float) -> str:
        """Create visual progress bar."""
        bar_length = 20
        filled_length = int(bar_length * percentage)

        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        return f"[blue]{bar}[/blue] {percentage:.1%}"

    def _analyze_error_content(self, content: str) -> Dict[str, Any]:
        """Analyze error content for categorization and suggestions."""
        error_analysis = {
            "error_type": "general",
            "severity": "error",
            "suggestions": []
        }

        content_lower = content.lower()

        # Error type detection
        if "timeout" in content_lower:
            error_analysis["error_type"] = "timeout"
            error_analysis["suggestions"] = ["Try increasing timeout settings", "Check network connectivity"]
        elif "permission" in content_lower or "access" in content_lower:
            error_analysis["error_type"] = "permission"
            error_analysis["suggestions"] = ["Check file permissions", "Verify access rights"]
        elif "connection" in content_lower:
            error_analysis["error_type"] = "connection"
            error_analysis["suggestions"] = ["Check network connection", "Verify service availability"]
        elif "file not found" in content_lower or "no such file" in content_lower:
            error_analysis["error_type"] = "file_not_found"
            error_analysis["suggestions"] = ["Check file path", "Verify file exists"]
        elif "syntax" in content_lower or "parse" in content_lower:
            error_analysis["error_type"] = "syntax"
            error_analysis["suggestions"] = ["Check syntax", "Validate input format"]

        return error_analysis

    def _extract_progress_info(self, content: str) -> Dict[str, Any]:
        """Extract progress information from content."""
        progress_info = {
            "percentage": None,
            "stage": None,
            "details": content
        }

        # Extract percentage
        percentage_match = re.search(r'(\d+(?:\.\d+)?)%', content)
        if percentage_match:
            progress_info["percentage"] = float(percentage_match.group(1)) / 100

        # Extract stage
        stage_match = re.search(r'(?:stage|phase):\s*([^\n]+)', content, re.IGNORECASE)
        if stage_match:
            progress_info["stage"] = stage_match.group(1).strip()

        return progress_info

    def _extract_handoff_info(self, content: str) -> Dict[str, Any]:
        """Extract handoff information from content."""
        handoff_info = {
            "from_agent": None,
            "to_agent": None,
            "reason": content
        }

        # Extract agent names
        from_match = re.search(r'from\s+([^\s]+)', content, re.IGNORECASE)
        if from_match:
            handoff_info["from_agent"] = from_match.group(1)

        to_match = re.search(r'to\s+([^\s]+)', content, re.IGNORECASE)
        if to_match:
            handoff_info["to_agent"] = to_match.group(1)

        return handoff_info

    def _extract_recommendations(self, content: str) -> List[str]:
        """Extract recommendations from content."""
        recommendations = []

        # Split by common recommendation delimiters
        delimiters = [r'\n\d+\.', r'\n•', r'\n-', r'\n\*']
        for delimiter in delimiters:
            parts = re.split(delimiter, content)
            if len(parts) > 1:
                recommendations = [part.strip() for part in parts[1:] if part.strip()]
                break

        # If no structured recommendations found, treat whole content as one
        if not recommendations and content.strip():
            recommendations = [content.strip()]

        return recommendations[:5]  # Return max 5 recommendations

    def _is_json(self, content: str) -> bool:
        """Check if content is valid JSON."""
        try:
            json.loads(content)
            return True
        except json.JSONDecodeError:
            return False

    def _format_fallback(self, message: RichMessage) -> str:
        """Fallback formatting when Rich is not available."""
        timestamp = message.timestamps.get("created", datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
        header = f"[{timestamp}] {message.message_type.value.upper()}:"

        if message.metadata.quality_score:
            header += f" (Quality: {message.metadata.quality_score:.1%})"

        return f"{header}\n{message.content}\n"

    def _format_error(self, message: RichMessage, error: str) -> str:
        """Format error when rich formatting fails."""
        return f"Formatting Error: {error}\nOriginal Message:\n{message.content}"

    def _update_formatting_stats(self, message: RichMessage, formatting_time: float):
        """Update formatting statistics."""
        self.formatting_stats["total_formatted"] += 1
        self.formatting_stats["formatting_time"] += formatting_time

        # Update by type
        msg_type = message.message_type.value
        if msg_type not in self.formatting_stats["by_type"]:
            self.formatting_stats["by_type"][msg_type] = {"count": 0, "total_time": 0.0}

        self.formatting_stats["by_type"][msg_type]["count"] += 1
        self.formatting_stats["by_type"][msg_type]["total_time"] += formatting_time

        # Update by style
        style = self.display_config.style.value
        if style not in self.formatting_stats["by_style"]:
            self.formatting_stats["by_style"][style] = {"count": 0, "total_time": 0.0}

        self.formatting_stats["by_style"][style]["count"] += 1
        self.formatting_stats["by_style"][style]["total_time"] += formatting_time

    def get_formatting_stats(self) -> Dict[str, Any]:
        """Get comprehensive formatting statistics."""
        stats = self.formatting_stats.copy()

        # Calculate averages
        if stats["total_formatted"] > 0:
            stats["average_formatting_time"] = stats["formatting_time"] / stats["total_formatted"]
        else:
            stats["average_formatting_time"] = 0.0

        return stats

    def reset_stats(self):
        """Reset formatting statistics."""
        self.formatting_stats = {
            "total_formatted": 0,
            "by_type": {},
            "by_style": {},
            "formatting_time": 0.0
        }

    # Utility methods for batch formatting
    async def format_messages(self, messages: List[RichMessage]) -> List[str]:
        """Format multiple messages."""
        formatted_messages = []

        for message in messages:
            formatted_message = await self.format_message(message)
            formatted_messages.append(formatted_message)

        return formatted_messages

    def format_message_sync(self, message: RichMessage) -> str:
        """Synchronous message formatting (fallback)."""
        if not RICH_AVAILABLE:
            return self._format_fallback(message)

        # Run async formatting in a simple way
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, use fallback
                return self._format_fallback(message)
            else:
                return loop.run_until_complete(self.format_message(message))
        except:
            return self._format_fallback(message)