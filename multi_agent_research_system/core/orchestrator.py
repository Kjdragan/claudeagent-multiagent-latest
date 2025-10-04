"""Main orchestrator for the multi-agent research system using Claude Agent SDK.

This module manages the research workflow using proper SDK patterns with
ClaudeSDKClient, agent definitions, and custom tools.
"""

import asyncio
import json
import os
import time

# Import from parent directory structure
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not found. Using environment variables only.")

# Set up API configuration
ANTHROPIC_BASE_URL = os.getenv('ANTHROPIC_BASE_URL', 'https://api.anthropic.com')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

if ANTHROPIC_API_KEY:
    # Set environment variables for the SDK
    os.environ['ANTHROPIC_BASE_URL'] = ANTHROPIC_BASE_URL
    os.environ['ANTHROPIC_API_KEY'] = ANTHROPIC_API_KEY
    print(f"✅ Using Anthropic API: {ANTHROPIC_BASE_URL}")
else:
    print("⚠️  No ANTHROPIC_API_KEY found in environment or .env file")

try:
    from claude_agent_sdk import (
        AgentDefinition,
        ClaudeAgentOptions,
        ClaudeSDKClient,
        create_sdk_mcp_server,
    )
    from claude_agent_sdk.types import HookMatcher, HookContext
except ImportError:
    # Fallback for when the SDK is not installed
    print("Warning: claude_agent_sdk not found. Please install the package.")
    AgentDefinition = None
    ClaudeSDKClient = None
    ClaudeAgentOptions = None
    create_sdk_mcp_server = None
    HookMatcher = None
    HookContext = None

from .agent_logger import AgentLoggerFactory
from .logging_config import get_logger
# Import agent_logging with proper path handling
try:
    from ..agent_logging import (
        StructuredLogger,
        get_logger as get_logger,
        AgentLogger,
        HookLogger,
        ResearchAgentLogger,
        ReportAgentLogger,
        EditorAgentLogger,
        UICoordinatorLogger,
        create_agent_logger
    )
except ImportError:
    # Fallback for when running as module
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from agent_logging import (
        StructuredLogger,
        get_logger as get_logger,
        AgentLogger,
        HookLogger,
        ResearchAgentLogger,
        ReportAgentLogger,
        EditorAgentLogger,
        UICoordinatorLogger,
        create_agent_logger
    )
# Hook system removed - using simplified implementation

# Use standard Python logging instead of complex agent_logging
import logging
StructuredLogger = logging.getLogger
AgentLogger = logging.getLogger
ResearchAgentLogger = logging.getLogger
ReportAgentLogger = logging.getLogger
EditorAgentLogger = logging.getLogger
UICoordinatorLogger = logging.getLogger


class SessionSearchBudget:
    """Manages search budget and limits for a research session."""

    def __init__(self, session_id: str):
        self.session_id = session_id

        # Primary research limits
        self.primary_successful_scrapes_limit = 10  # User requested limit
        self.primary_urls_processed = 0
        self.primary_successful_scrapes = 0
        self.primary_search_queries = 0

        # Editorial research limits
        self.editorial_successful_scrapes_limit = 5  # User requested limit
        self.editorial_search_queries_limit = 2  # User requested limit
        self.editorial_urls_processed = 0
        self.editorial_successful_scrapes = 0
        self.editorial_search_queries = 0

        # Global session limits
        self.total_urls_processed_limit = 100  # Safety limit
        self.total_urls_processed = 0

        self.logger = logging.getLogger(f"search_budget.{session_id}")

    def can_primary_research_proceed(self, urls_to_process: int = 1) -> tuple[bool, str]:
        """Check if primary research can proceed with given URL count."""
        # Check successful scrapes limit
        if self.primary_successful_scrapes >= self.primary_successful_scrapes_limit:
            return False, f"Primary research limit reached: {self.primary_successful_scrapes}/{self.primary_successful_scrapes_limit} successful scrapes"

        # Check total URL limit
        if self.total_urls_processed + urls_to_process > self.total_urls_processed_limit:
            return False, f"Session URL limit would be exceeded: {self.total_urls_processed + urls_to_process}/{self.total_urls_processed_limit}"

        return True, "Primary research can proceed"

    def can_editorial_research_proceed(self, urls_to_process: int = 1) -> tuple[bool, str]:
        """Check if editorial research can proceed."""
        # Check search queries limit
        if self.editorial_search_queries >= self.editorial_search_queries_limit:
            return False, f"Editorial search query limit reached: {self.editorial_search_queries}/{self.editorial_search_queries_limit}"

        # Check successful scrapes limit
        if self.editorial_successful_scrapes >= self.editorial_successful_scrapes_limit:
            return False, f"Editorial scrape limit reached: {self.editorial_successful_scrapes}/{self.editorial_successful_scrapes_limit} successful scrapes"

        # Check total URL limit
        if self.total_urls_processed + urls_to_process > self.total_urls_processed_limit:
            return False, f"Session URL limit would be exceeded: {self.total_urls_processed + urls_to_process}/{self.total_urls_processed_limit}"

        return True, "Editorial research can proceed"

    def record_primary_research(self, urls_processed: int, successful_scrapes: int, search_queries: int = 1):
        """Record primary research activity."""
        self.primary_urls_processed += urls_processed
        self.primary_successful_scrapes += successful_scrapes
        self.primary_search_queries += search_queries
        self.total_urls_processed += urls_processed

        self.logger.info(f"Primary research recorded: {urls_processed} URLs, {successful_scrapes} successful scrapes")

    def record_editorial_research(self, urls_processed: int, successful_scrapes: int, search_queries: int = 1):
        """Record editorial research activity."""
        self.editorial_urls_processed += urls_processed
        self.editorial_successful_scrapes += successful_scrapes
        self.editorial_search_queries += search_queries
        self.total_urls_processed += urls_processed

        self.logger.info(f"Editorial research recorded: {urls_processed} URLs, {successful_scrapes} successful scrapes")

    def get_budget_status(self) -> dict[str, Any]:
        """Get current budget status."""
        return {
            "session_id": self.session_id,
            "primary": {
                "successful_scrapes": f"{self.primary_successful_scrapes}/{self.primary_successful_scrapes_limit}",
                "urls_processed": self.primary_urls_processed,
                "search_queries": self.primary_search_queries,
                "can_proceed": self.can_primary_research_proceed()[0]
            },
            "editorial": {
                "successful_scrapes": f"{self.editorial_successful_scrapes}/{self.editorial_successful_scrapes_limit}",
                "search_queries": f"{self.editorial_search_queries}/{self.editorial_search_queries_limit}",
                "urls_processed": self.editorial_urls_processed,
                "can_proceed": self.can_editorial_research_proceed()[0]
            },
            "global": {
                "total_urls_processed": f"{self.total_urls_processed}/{self.total_urls_processed_limit}",
                "remaining_urls": self.total_urls_processed_limit - self.total_urls_processed
            }
        }


def create_agent_logger(session_id: str, agent_type: str) -> logging.Logger:
    """Create a simple logger for an agent."""
    return logging.getLogger(f"{agent_type}_{session_id}")
from .simple_research_tools import (
    create_research_report,
    get_session_data,
    save_research_findings,
)
from .search_analysis_tools import (
    capture_search_results,
    save_webfetch_content,
    create_search_verification_report,
)

# Import SERP API search tool, advanced scraping tools, intelligent research tool, and enhanced search MCP
try:
    from ..tools.serp_search_tool import serp_search
    from ..tools.advanced_scraping_tool import advanced_scrape_url, advanced_scrape_multiple_urls
    from ..tools.intelligent_research_tool import intelligent_research_with_advanced_scraping
    from ..mcp_tools.zplayground1_search import zplayground1_server
except ImportError:
    # Fallback for when the tools module is not available
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from tools.serp_search_tool import serp_search
    from tools.advanced_scraping_tool import advanced_scrape_url, advanced_scrape_multiple_urls
    from tools.intelligent_research_tool import intelligent_research_with_advanced_scraping
    try:
        from mcp_tools.zplayground1_search import zplayground1_server
    except ImportError:
        zplayground1_server = None
        print("Warning: zPlayground1 search MCP server not available")

# Import config module with fallback
try:
    from ..config.agents import get_all_agent_definitions
except ImportError:
    # Fallback for when running as a script
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from config.agents import get_all_agent_definitions


class ResearchOrchestrator:
    """Main orchestrator for the multi-agent research system."""

    def __init__(self, debug_mode: bool = False):
        # Initialize structured logging system
        self.logger = get_logger("orchestrator")
        self.structured_logger = get_logger("orchestrator")
        # Hook system disabled - no loggers initialized

        self.logger.info("Initializing ResearchOrchestrator")
        self.structured_logger.info("ResearchOrchestrator initialization started",
                                   event_type="orchestrator_initialization",
                                   debug_mode=debug_mode)

        self.debug_mode = debug_mode
        self.debug_output = []  # Store stderr debug output

        # Hook system configuration disabled
        self.hook_config = None  # Hook system removed
        self.logger.info("Hook system disabled for simplified operation")

        self.agent_definitions = get_all_agent_definitions()
        self.logger.debug(f"Loaded {len(self.agent_definitions)} agent definitions")
        self.structured_logger.info("Agent definitions loaded",
                                   agent_count=len(self.agent_definitions),
                                   agent_names=list(self.agent_definitions.keys()))

        self.active_sessions: dict[str, dict[str, Any]] = {}
        self.client: ClaudeSDKClient = None  # Single client for all agents (CORRECT PATTERN)
        self.agent_names: list[str] = []  # List of agent names for reference
        self.mcp_server = None

        # Agent-specific loggers for detailed activity tracking
        self.agent_logger = None  # General agent logger (legacy)
        self.agent_loggers: dict[str, Any] = {}  # Agent-specific loggers

        # Initialize agent-specific loggers
        self._initialize_agent_loggers()

        self.logger.info("ResearchOrchestrator initialized")
        self.structured_logger.info("ResearchOrchestrator initialization completed",
                                   event_type="orchestrator_ready",
                                   total_agents=len(self.agent_definitions))

    def _create_debug_callback(self, agent_name: str):
        """Create a debug callback for capturing stderr output."""
        def debug_callback(message: str):
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            formatted_msg = f"[{timestamp}] {agent_name}: {message.strip()}"
            self.debug_output.append(formatted_msg)

            # Log to agent-specific logger if available
            agent_logger = self.get_agent_logger(agent_name)
            if agent_logger:
                try:
                    # Parse the message to extract activity information for agent-specific logging
                    message_lower = message.lower()

                    if agent_name == "research_agent":
                        # Research agent specific logging
                        if "search" in message_lower and "results" in message_lower:
                            # Extract search information if available
                            agent_logger.log_search_initiation(
                                query="debug_search",  # Would be extracted from message in real implementation
                                search_params={"debug": True},
                                topic="debug_topic",
                                estimated_results=10
                            )
                        elif "source" in message_lower or "extract" in message_lower:
                            agent_logger.log_source_analysis(
                                source_url="debug_source.com",
                                source_title="Debug Source",
                                credibility_score=0.8,
                                content_relevance=0.9,
                                content_type="article",
                                extraction_method="debug"
                            )

                    elif agent_name == "report_agent":
                        # Report agent specific logging
                        if "section" in message_lower or "generating" in message_lower:
                            agent_logger.log_section_generation_start(
                                section_name="debug_section",
                                section_type="debug",
                                research_sources_count=5,
                                target_word_count=500
                            )

                    elif agent_name == "editor_agent":
                        # Editor agent specific logging
                        if "review" in message_lower or "analyzing" in message_lower:
                            agent_logger.log_review_initiation(
                                document_title="Debug Document",
                                document_type="report",
                                word_count=1000,
                                review_focus_areas=["content", "structure"]
                            )

                    elif agent_name == "ui_coordinator":
                        # UI coordinator specific logging
                        if "workflow" in message_lower or "coordinating" in message_lower:
                            agent_logger.log_workflow_initiation(
                                workflow_id=str(uuid.uuid4()),
                                user_request="debug request",
                                workflow_type="debug",
                                estimated_stages=["research", "report", "edit"],
                                priority_level="normal"
                            )

                    # Fallback to generic logging for all agents
                    if "tool" in message_lower:
                        agent_logger.log_activity(
                            agent_name=agent_name,
                            activity_type="debug_message",
                            stage="unknown",
                            input_data={"debug_message": message.strip()},
                            metadata={"message_type": "tool_debug"}
                        )
                    elif "error" in message_lower:
                        agent_logger.log_activity(
                            agent_name=agent_name,
                            activity_type="debug_error",
                            stage="unknown",
                            error=message.strip(),
                            metadata={"message_type": "error_debug"}
                        )
                    elif "response" in message_lower:
                        agent_logger.log_activity(
                            agent_name=agent_name,
                            activity_type="debug_response",
                            stage="unknown",
                            input_data={"debug_message": message.strip()},
                            metadata={"message_type": "response_debug"}
                        )

                except Exception as e:
                    # Don't let logging errors break the debug callback
                    self.logger.warning(f"Failed to log debug message to agent logger {agent_name}: {e}")

            # Log important messages
            if any(keyword in message for keyword in ["ERROR", "WARNING", "tool", "response", "message"]):
                self.logger.debug(f"SDK Debug ({agent_name}): {message.strip()}")

        return debug_callback

    def _get_simplified_hooks(self) -> dict:
        """Get simplified hooks configuration to avoid JavaScript parsing errors.

        Based on finally2.md analysis, the most valuable logging outputs are:
        1. session_state.json - Complete session narrative and success assessment
        2. orchestrator.jsonl - Workflow progression and root cause analysis
        3. agent_summary.json - Performance metrics and error patterns

        The hook system was causing 262 JavaScript parsing errors per session
        with limited additional value over our core logging system.
        """
        try:
            from claude_agent_sdk.types import HookMatcher, HookEvent

            # Only enable essential hooks that provide high value without causing parsing issues
            essential_hooks = {
                HookEvent.PreToolUse: [
                    HookMatcher(
                        matcher="serp_search|mcp__research_tools",
                        hooks=[self._create_essential_tool_hook()]
                    )
                ],
                HookEvent.PostToolUse: [
                    HookMatcher(
                        matcher="serp_search|mcp__research_tools",
                        hooks=[self._create_essential_completion_hook()]
                    )
                ]
            }

            self.logger.info("🔧 Using simplified hooks configuration (4 essential hooks vs 262 error-prone hooks)")
            return essential_hooks

        except Exception as e:
            self.logger.warning(f"⚠️  Hook system unavailable, using basic logging: {e}")
            return {}

    def _create_essential_tool_hook(self):
        """Create essential tool usage hook without complex parsing."""
        async def essential_tool_hook(input_data, tool_use_id, context):
            return {
                "decision": "continue",  # Always allow tools to continue
                "systemMessage": f"Tool executed: {tool_use_id}",
                "hookSpecificOutput": {
                    "tool_use_id": tool_use_id,
                    "timestamp": str(uuid.uuid4()),
                    "simplified": True
                }
            }
        return essential_tool_hook

    def _create_essential_completion_hook(self):
        """Create essential tool completion hook without complex parsing."""
        async def essential_completion_hook(input_data, tool_use_id, context):
            return {
                "decision": "continue",  # Always allow continuation
                "systemMessage": f"Tool completed: {tool_use_id}",
                "hookSpecificOutput": {
                    "tool_use_id": tool_use_id,
                    "completed": True,
                    "simplified": True
                }
            }
        return essential_completion_hook

    # Note: Tool execution monitoring is now handled by the comprehensive HookIntegrationManager
    # The old _debug_pre_tool_use and _debug_post_tool_use methods have been removed
    # as they are superseded by the integrated hook system with proper SDK pattern compliance

    def _initialize_agent_loggers(self):
        """Initialize agent-specific loggers for each agent type."""
        try:
            # Create session ID for this orchestrator instance
            session_id = str(uuid.uuid4())

            # Initialize agent-specific loggers
            for agent_name in self.agent_definitions.keys():
                try:
                    # Create appropriate agent logger based on agent type
                    agent_logger = create_agent_logger(agent_name, session_id)
                    self.agent_loggers[agent_name] = agent_logger

                    self.logger.debug(f"Agent logger initialized: {agent_name}")
                    self.structured_logger.debug("Agent logger initialized",
                                                agent_name=agent_name,
                                                session_id=session_id)

                except Exception as e:
                    self.logger.warning(f"Failed to initialize agent logger for {agent_name}: {e}")
                    self.structured_logger.warning("Agent logger initialization failed",
                                                 agent_name=agent_name,
                                                 error=str(e))

                    # Fallback to generic logger
                    self.agent_loggers[agent_name] = AgentLogger(agent_name, session_id)

            self.logger.info(f"Agent loggers initialized: {len(self.agent_loggers)}")
            self.structured_logger.info("Agent loggers initialization completed",
                                       agent_loggers_count=len(self.agent_loggers),
                                       agent_names=list(self.agent_loggers.keys()))

        except Exception as e:
            self.logger.error(f"Failed to initialize agent loggers: {e}")
            self.structured_logger.error("Agent loggers initialization failed",
                                       error=str(e),
                                       error_type=type(e).__name__)

    def get_agent_logger(self, agent_name: str) -> Any:
        """Get the appropriate agent logger for the specified agent."""
        return self.agent_loggers.get(agent_name, self.agent_logger)

    def get_all_agent_summaries(self) -> dict[str, Any]:
        """Get summaries from all agent loggers."""
        summaries = {}
        for agent_name, agent_logger in self.agent_loggers.items():
            try:
                if hasattr(agent_logger, 'get_research_summary'):
                    summaries[agent_name] = agent_logger.get_research_summary()
                elif hasattr(agent_logger, 'get_report_summary'):
                    summaries[agent_name] = agent_logger.get_report_summary()
                elif hasattr(agent_logger, 'get_editor_summary'):
                    summaries[agent_name] = agent_logger.get_editor_summary()
                elif hasattr(agent_logger, 'get_coordinator_summary'):
                    summaries[agent_name] = agent_logger.get_coordinator_summary()
                else:
                    summaries[agent_name] = {"message": "No summary method available"}
            except Exception as e:
                summaries[agent_name] = {"error": str(e)}

        return summaries

    async def initialize(self):
        """Initialize the orchestrator and all agent clients."""
        self.logger.info("Starting orchestrator initialization")

        try:
            # Hook system disabled
            self.logger.info("Hook system disabled - skipping initialization")

            # Create MCP server with custom tools including SERP API search
            self.logger.debug("Creating MCP server")

            # Create MCP server with properly decorated SdkMcpTool instances
            self.mcp_server = create_sdk_mcp_server(
                name="research_tools",
                version="1.0.0",
                tools=[
                    save_research_findings, create_research_report, get_session_data,
                    capture_search_results, save_webfetch_content, create_search_verification_report,
                    serp_search  # Add SERP API search tool to MCP server
                ]
            )
            self.logger.info("MCP server created successfully")

            # Log MCP server configuration
            self.logger.info("🔍 MCP Server Configuration:")
            self.logger.info(f"   Research Tools Server: {type(self.mcp_server).__name__} (TypedDict - correct)")
            self.logger.info(f"   Server Name: {self.mcp_server.get('name', 'Unknown')}")
            self.logger.info(f"   Available Tools: {len([save_research_findings, create_research_report, get_session_data, capture_search_results, save_webfetch_content, create_search_verification_report, serp_search])} tools")

            # Debug SERP API configuration
            serper_key = os.getenv('SERP_API_KEY', 'NOT_SET')
            serper_status = 'SET' if serper_key != 'NOT_SET' else 'NOT_SET'
            openai_key = os.getenv('OPENAI_API_KEY', 'NOT_SET')
            openai_status = 'SET' if openai_key != 'NOT_SET' else 'NOT_SET'
            self.logger.info(f"   SERP API Search: Enabled (high-performance replacement for WebPrime MCP)")
            self.logger.info(f"   SERP_API_KEY Status: {serper_status}")
            self.logger.info(f"   OPENAI_API_KEY Status: {openai_status}")
            self.logger.info(f"   Expected Tools: serp_search, research_tools")

            # Create a single client with all agents configured properly (CORRECT PATTERN)
            self.logger.debug("Creating single multi-agent client")

            # Check if required SDK components are available
            if AgentDefinition is None:
                raise ImportError("AgentDefinition is not available. Please ensure claude_agent_sdk is properly installed.")

            if ClaudeAgentOptions is None:
                raise ImportError("ClaudeAgentOptions is not available. Please ensure claude_agent_sdk is properly installed.")

            # Prepare agents configuration for ClaudeAgentOptions
            agents_config = {}
            for agent_name, agent_def in self.agent_definitions.items():
                # Create actual AgentDefinition instances with SERP API and research tools
                extended_tools = agent_def.tools + [
                    "mcp__research_tools__serp_search",  # High-performance SERP API search via MCP
                    "mcp__research_tools__save_research_findings",
                    "mcp__research_tools__create_research_report",
                    "mcp__research_tools__get_session_data",
                    "mcp__research_tools__capture_search_results",
                    "mcp__research_tools__save_webfetch_content",
                    "mcp__research_tools__create_search_verification_report",
                    "Read", "Write", "Glob", "Grep"
                ]

                # Add zPlayground1 search tool if available
                if zplayground1_server is not None:
                    zplayground1_tools = [
                        "mcp__zplayground1_search__zplayground1_search_scrape_clean"  # Single comprehensive tool
                    ]
                    extended_tools.extend(zplayground1_tools)
                    self.logger.info(f"✅ zPlayground1 search tool added to {agent_name}")
                else:
                    self.logger.warning(f"⚠️ zPlayground1 search tool not available for {agent_name}")

                agents_config[agent_name] = AgentDefinition(
                    description=agent_def.description,
                    prompt=agent_def.prompt,
                    tools=extended_tools,
                    model=agent_def.model
                )

            # Prepare MCP servers configuration
            mcp_servers_config = {
                "research_tools": self.mcp_server,
            }

            # Add zPlayground1 search server if available
            if zplayground1_server is not None:
                mcp_servers_config["zplayground1_search"] = zplayground1_server
                self.logger.info("✅ zPlayground1 search MCP server added to configuration")
            else:
                self.logger.warning("⚠️ zPlayground1 search MCP server not available, using standard tools")

            # Create single options with all agents configured properly
            options = ClaudeAgentOptions(
                agents=agents_config,
                mcp_servers=mcp_servers_config,
                # Use correct settings for proper response handling
                include_partial_messages=False,
                permission_mode="bypassPermissions",
                # Debugging features
                stderr=self._create_debug_callback("multi_agent"),
                extra_args={"debug-to-stderr": None},
                # Simplified logging approach - disable problematic hooks due to JavaScript parsing errors
                # Increase buffer size to handle large JSON outputs (DeepWiki recommendation)
                max_buffer_size=4096000,  # 4MB buffer
                hooks=self._get_simplified_hooks() if HookMatcher else {}
            )

            # Create and connect SINGLE client for all agents (CORRECT PATTERN)
            self.client = ClaudeSDKClient(options=options)
            await self.client.connect()
            self.logger.info("✅ Single multi-agent client created and connected")

            # Store all agent names for reference (no individual clients needed)
            self.agent_names = list(self.agent_definitions.keys())

            # Log agent initialization if agent logger is available
            if hasattr(self, 'agent_logger') and self.agent_logger:
                for agent_name, agent_def in self.agent_definitions.items():
                    self.agent_logger.log_agent_initialization(
                        agent_name=agent_name,
                        config={
                            "model": agent_def.model,
                            "tools": agent_def.tools,
                            "debug_mode": self.debug_mode,
                            "max_turns": 10,
                            "single_client_pattern": True
                        }
                    )

            self.logger.info(f"All {len(self.agent_names)} agents configured in single client")

        except Exception as e:
            self.logger.error(f"Initialization error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    @property
    def agent_clients(self) -> dict[str, Any]:
        """
        Backward compatibility property.
        Returns a dict mapping agent names to the shared client.
        """
        if self.client is None:
            return {}
        return {agent_name: self.client for agent_name in self.agent_names}

    async def check_agent_health(self) -> dict[str, Any]:
        """Check the health and connectivity of all agents using single client pattern."""
        self.logger.info("🔍 Performing agent health checks...")
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "total_agents": len(self.agent_names),
            "healthy_agents": 0,
            "unhealthy_agents": 0,
            "agent_status": {},
            "mcp_servers": {},
            "issues": []
        }

        for agent_name in self.agent_names:
            try:
                self.logger.debug(f"Checking health of {agent_name}...")

                # Test basic connectivity with natural language agent selection (CORRECT PATTERN)
                test_prompt = f"Use the {agent_name} agent to introduce yourself and describe your capabilities in 2-3 sentences."
                start_time = time.time()

                try:
                    # Send query to single client with natural language agent selection
                    await self.client.query(test_prompt)
                    response_time = time.time() - start_time

                    # Collect substantive response to verify agent capability
                    substantive_response = False
                    collected_messages = []

                    try:
                        async for message in self.client.receive_response():
                            collected_messages.append(message)

                            # Check for substantive content
                            if hasattr(message, 'content') and message.content:
                                for block in message.content:
                                    if hasattr(block, 'text') and len(block.text.strip()) > 10:
                                        substantive_response = True
                                        break
                            elif hasattr(message, 'total_cost_usd'):
                                # Found ResultMessage, we're done
                                break
                    except Exception as response_error:
                        self.logger.debug(f"Response collection error for {agent_name}: {response_error}")
                        # Continue with basic check if response collection fails

                    if substantive_response or len(collected_messages) > 0:
                        health_report["agent_status"][agent_name] = {
                            "status": "healthy",
                            "response_time": response_time,
                            "response_type": "messages_collected",
                            "message_count": len(collected_messages),
                            "substantive_response": substantive_response
                        }
                        health_report["healthy_agents"] += 1
                        self.logger.info(f"✅ {agent_name}: Healthy ({response_time:.2f}s, {len(collected_messages)} messages)")
                    else:
                        health_report["agent_status"][agent_name] = {
                            "status": "warning",
                            "issue": "no substantive response collected",
                            "response_time": response_time,
                            "message_count": len(collected_messages)
                        }
                        health_report["unhealthy_agents"] += 1
                        health_report["issues"].append(f"{agent_name}: No substantive response collected")
                        self.logger.warning(f"⚠️ {agent_name}: No substantive response ({response_time:.2f}s)")

                except Exception as query_error:
                    health_report["agent_status"][agent_name] = {
                        "status": "unhealthy",
                        "issue": str(query_error),
                        "response_time": time.time() - start_time
                    }
                    health_report["unhealthy_agents"] += 1
                    health_report["issues"].append(f"{agent_name}: {query_error}")
                    self.logger.error(f"❌ {agent_name}: {query_error}")

            except Exception as e:
                health_report["agent_status"][agent_name] = {
                    "status": "error",
                    "issue": f"Health check failed: {e}"
                }
                health_report["unhealthy_agents"] += 1
                health_report["issues"].append(f"{agent_name}: Health check failed - {e}")
                self.logger.error(f"❌ {agent_name}: Health check failed - {e}")

        # Check MCP server status
        if self.mcp_server:
            try:
                health_report["mcp_servers"]["research_tools"] = {
                    "status": "available",
                    "type": type(self.mcp_server).__name__
                }
                self.logger.info("✅ Research Tools MCP Server: Available")
            except Exception as e:
                health_report["mcp_servers"]["research_tools"] = {
                    "status": "error",
                    "issue": str(e)
                }
                health_report["issues"].append(f"Research Tools MCP: {e}")
                self.logger.error(f"❌ Research Tools MCP Server: {e}")

        # Check SERP API configuration
        serper_key = os.getenv('SERP_API_KEY')
        openai_key = os.getenv('OPENAI_API_KEY')
        if serper_key and openai_key:
            health_report["search_system"] = {
                "status": "configured",
                "type": "SERP_API",
                "serper_configured": True,
                "openai_configured": True
            }
            self.logger.info("✅ SERP API Search: Configured with API keys")
        else:
            health_report["search_system"] = {
                "status": "missing_keys",
                "type": "SERP_API",
                "serper_configured": bool(serper_key),
                "openai_configured": bool(openai_key)
            }
            if not serper_key:
                health_report["issues"].append("SERP API: Missing SERP_API_KEY")
                self.logger.error("❌ SERP API: Missing SERP_API_KEY")
            if not openai_key:
                health_report["issues"].append("SERP API: Missing OPENAI_API_KEY")
                self.logger.error("❌ SERP API: Missing OPENAI_API_KEY")

        # Summary
        health_report["summary"] = f"{health_report['healthy_agents']}/{health_report['total_agents']} agents healthy"
        self.logger.info(f"🏥 Agent Health Check Complete: {health_report['summary']}")

        if health_report["issues"]:
            self.logger.warning(f"🚨 Issues found: {len(health_report['issues'])} problems")
            for issue in health_report["issues"]:
                self.logger.warning(f"   - {issue}")

        return health_report

    async def execute_agent_query(
        self,
        agent_name: str,
        prompt: str,
        session_id: Optional[str] = None,
        timeout_seconds: int = 120
    ) -> dict[str, Any]:
        """Execute a query using natural language agent selection (CORRECT PATTERN).

        Args:
            agent_name: Name of the agent to use
            prompt: The task prompt for the agent
            session_id: Optional session ID for tracking
            timeout_seconds: Query timeout in seconds (default: 120)

        Returns:
            Dict with query results including messages, tool executions, and metadata
        """
        if not self.client:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        # Construct natural language agent selection prompt
        full_prompt = f"Use the {agent_name} agent to {prompt}"

        self.logger.info(f"🔍 Querying {agent_name} with natural language selection")

        query_result = {
            "agent_name": agent_name,
            "session_id": session_id,
            "prompt_sent": prompt,
            "messages_collected": [],
            "substantive_responses": 0,
            "tool_executions": [],
            "errors": [],
            "query_start_time": datetime.now().isoformat(),
            "success": False
        }

        try:
            # Send query to single client
            await self.client.query(full_prompt)

            # Collect responses
            async for message in self.client.receive_response():
                message_info = {
                    "message_type": type(message).__name__,
                    "timestamp": datetime.now().isoformat()
                }

                # Extract content information
                if hasattr(message, 'content') and message.content:
                    content_texts = []
                    for block in message.content:
                        if hasattr(block, 'text') and block.text:
                            content_texts.append(block.text)
                    if content_texts:
                        message_info["content_texts"] = content_texts
                        query_result["substantive_responses"] += 1

                # Extract tool use information
                if hasattr(message, 'tool_use') and message.tool_use:
                    message_info["tool_use"] = {
                        "name": message.tool_use.get("name"),
                        "id": message.tool_use.get("id")
                    }
                    query_result["tool_executions"].append(message_info["tool_use"])

                query_result["messages_collected"].append(message_info)

                # Stop when we get ResultMessage (conversation complete)
                if hasattr(message, 'total_cost_usd'):
                    self.logger.debug(f"ResultMessage received, stopping collection")
                    break

                # Safety limit to prevent infinite loops
                if len(query_result["messages_collected"]) >= 50:
                    self.logger.warning(f"Message limit reached for {agent_name} query")
                    break

            query_result["success"] = True
            query_result["query_end_time"] = datetime.now().isoformat()
            self.logger.info(f"✅ {agent_name} query completed: {len(query_result['messages_collected'])} messages")
            return query_result

        except Exception as e:
            query_result["errors"].append(str(e))
            query_result["success"] = False
            query_result["query_end_time"] = datetime.now().isoformat()
            self.logger.error(f"❌ {agent_name} query failed: {e}")
            raise

    def _validate_research_completion(self, research_result: dict[str, Any]) -> bool:
        """Validate that research stage completed successfully.

        Args:
            research_result: Result from execute_agent_query

        Returns:
            True if research completed successfully, False otherwise
        """
        # Check basic success
        if not research_result.get("success", False):
            return False

        # Check for substantive responses
        if research_result.get("substantive_responses", 0) < 1:
            return False

        # Check for tool executions (research should have used search tools)
        tool_executions = research_result.get("tool_executions", [])
        if len(tool_executions) < 1:
            return False

        # Check for required tools
        tool_names = [tool.get("name", "") for tool in tool_executions]
        required_tools = ["serp_search"]  # At minimum should do search

        has_required_tools = any(
            any(req_tool in tool_name for req_tool in required_tools)
            for tool_name in tool_names
        )

        if not has_required_tools:
            self.logger.warning(f"Research validation failed: required tools not found. Tools: {tool_names}")
            return False

        return True

    def _validate_report_completion(self, report_result: dict[str, Any]) -> bool:
        """Validate that report generation stage completed successfully.

        Args:
            report_result: Result from execute_agent_query

        Returns:
            True if report generation completed successfully, False otherwise
        """
        # Check basic success
        if not report_result.get("success", False):
            return False

        # Check for substantive responses
        if report_result.get("substantive_responses", 0) < 1:
            return False

        # Check for tool executions (should save findings/create report)
        tool_executions = report_result.get("tool_executions", [])
        if len(tool_executions) < 1:
            return False

        return True

    def _validate_editorial_completion(self, editorial_result: dict[str, Any]) -> bool:
        """Validate that editorial review stage completed successfully.

        Args:
            editorial_result: Result from execute_agent_query

        Returns:
            True if editorial review completed successfully, False otherwise
        """
        # Check basic success
        if not editorial_result.get("success", False):
            return False

        # Check for substantive responses
        if editorial_result.get("substantive_responses", 0) < 1:
            return False

        return True

    async def verify_tool_execution(self, agent_name: str = "research_agent") -> dict[str, Any]:
        """Verify that critical tools can be executed by agents."""
        self.logger.info(f"🔧 Verifying tool execution for {agent_name}...")

        tool_verification = {
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "tools_tested": [],
            "tools_working": [],
            "tools_failed": [],
            "issues": []
        }

        if not self.client:
            error_msg = "Client not initialized"
            tool_verification["issues"].append(error_msg)
            self.logger.error(f"❌ {error_msg}")
            return tool_verification

        if agent_name not in self.agent_names:
            error_msg = f"Agent {agent_name} not available"
            tool_verification["issues"].append(error_msg)
            self.logger.error(f"❌ {error_msg}")
            return tool_verification

        # Test critical tools with simple prompts
        critical_tools = [
            {
                "name": "SERP API Search",
                "test_prompt": "Use the mcp__research_tools__serp_search tool to search for 'test query' with num_results=1.",
                "expected_pattern": "serp_search"
            },
            {
                "name": "File Operations",
                "test_prompt": "Use the Read tool to read the file 'test.txt'.",
                "expected_pattern": "Read"
            }
        ]

        for tool_test in critical_tools:
            tool_name = tool_test["name"]
            test_prompt = tool_test["test_prompt"]

            tool_verification["tools_tested"].append(tool_name)

            try:
                self.logger.debug(f"Testing {tool_name} with {agent_name}...")
                start_time = time.time()

                # Use the new execute_agent_query method (CORRECT PATTERN)
                full_test_prompt = f"{test_prompt} Use the {agent_name} agent for this task."
                result = await self.execute_agent_query(agent_name, full_test_prompt)
                execution_time = time.time() - start_time

                # Check if tool was mentioned in responses
                tool_used = False
                for message_info in result["messages_collected"]:
                    if "content_texts" in message_info:
                        for text in message_info["content_texts"]:
                            if tool_test["expected_pattern"] in str(text):
                                tool_used = True
                                break
                        if tool_used:
                            break

                if tool_used or len(result["messages_collected"]) > 0:
                    tool_verification["tools_working"].append(tool_name)
                    self.logger.info(f"✅ {tool_name}: Working ({execution_time:.2f}s)")
                else:
                    tool_verification["tools_failed"].append(tool_name)
                    issue = f"{tool_name}: Tool not executed in response"
                    tool_verification["issues"].append(issue)
                    self.logger.warning(f"⚠️ {tool_name}: Tool not executed")

            except Exception as e:
                tool_verification["tools_failed"].append(tool_name)
                issue = f"{tool_name}: {str(e)}"
                tool_verification["issues"].append(issue)
                self.logger.error(f"❌ {tool_name}: {str(e)}")

        # Summary
        working_count = len(tool_verification["tools_working"])
        total_count = len(tool_verification["tools_tested"])
        tool_verification["summary"] = f"{working_count}/{total_count} tools working"
        tool_verification["success_rate"] = working_count / total_count if total_count > 0 else 0

        self.logger.info(f"🔧 Tool Verification Complete: {tool_verification['summary']}")

        if tool_verification["issues"]:
            self.logger.warning(f"🚨 Tool issues found: {len(tool_verification['issues'])} problems")
            for issue in tool_verification["issues"]:
                self.logger.warning(f"   - {issue}")

        return tool_verification

    async def execute_research_with_tool_enforcement(self, client, session_id: str, topic: str, user_requirements: dict[str, Any]) -> list:
        """Execute research with mandatory tool execution enforcement."""
        import json
        self.logger.info(f"🔧 Enforcing tool execution for research on: {topic}")

        required_tools = {"mcp__research_tools__serp_search", "mcp__research_tools__save_research_findings"}
        executed_tools = set()
        max_attempts = 3
        attempt = 0

        while attempt < max_attempts and not required_tools.issubset(executed_tools):
            attempt += 1
            self.logger.info(f"Research attempt {attempt}/{max_attempts}")

            # Create stronger prompt based on attempt
            if attempt == 1:
                research_prompt = f"""
                EXECUTE RESEARCH TASK: {topic}

                Session ID: {session_id}
                Requirements: {json.dumps(user_requirements)}

                CRITICAL REQUIREMENT: You MUST execute tools NOW, not respond with text first.

                REQUIRED IMMEDIATE ACTIONS:
                1. Execute mcp__research_tools__serp_search with:
                   - query: "{topic}"
                   - num_results: 15
                   - auto_crawl_top: 8
                   - crawl_threshold: 0.3
                   - session_id: "{session_id}"

                2. Execute mcp__research_tools__save_research_findings to save results

                DO NOT RESPOND WITH TEXT until AFTER executing the search tool.
                The search MUST be your first action.
                """
            else:
                # Stronger prompt for retry attempts
                missing_tools = required_tools - executed_tools
                research_prompt = f"""
                CRITICAL: Previous attempt failed to execute required tools.
                MISSING TOOLS: {missing_tools}

                You MUST execute these tools NOW:
                - mcp__research_tools__serp_search for "{topic}"
                - mcp__research_tools__save_research_findings

                DO NOT respond with "OK" or acknowledgments.
                EXECUTE THE TOOLS IMMEDIATELY.
                """

            # Send query and collect responses with tool tracking
            self.logger.info(f"Sending research query (attempt {attempt})...")
            query_start_time = time.time()

            try:
                response = await client.query(research_prompt)
                query_execution_time = time.time() - query_start_time
                self.logger.info(f"✅ Query sent: {type(response).__name__}")

                # Log the query
                if self.agent_logger:
                    self.agent_logger.log_query_response(
                        agent_name="research_agent",
                        query=research_prompt[:1000] + "..." if len(research_prompt) > 1000 else research_prompt,
                        response=response,
                        execution_time=query_execution_time,
                        stage="research"
                    )

            except Exception as e:
                query_execution_time = time.time() - query_start_time
                self.logger.error(f"❌ Query failed: {e}")
                if self.agent_logger:
                    self.agent_logger.log_error(
                        agent_name="research_agent",
                        error=e,
                        context={"query": research_prompt[:500], "session_id": session_id, "attempt": attempt},
                        stage="research"
                    )
                continue

            # Collect responses and track tool execution
            research_results = []
            tool_executed_this_attempt = False

            try:
                timeout_seconds = 60
                start_time = asyncio.get_event_loop().time()

                async for message in client.receive_response():
                    current_time = asyncio.get_event_loop().time()
                    elapsed = current_time - start_time

                    self.logger.debug(f"🔧 Research message type: {type(message).__name__} after {elapsed:.1f}s")
                    research_results.append(message)

                    # Track tool execution and handle results
                    if hasattr(message, 'content') and message.content:
                        for block in message.content:
                            block_type = type(block).__name__

                            # Handle tool execution (ToolUseBlock)
                            if hasattr(block, 'name') and block.name:
                                executed_tools.add(block.name)
                                tool_executed_this_attempt = True
                                self.logger.info(f"✅ Tool executed: {block.name}")

                            # Handle tool results (ToolResultBlock)
                            if block_type == 'ToolResultBlock':
                                self.logger.info(f"🔧 Found tool result block")
                                all_attrs = [attr for attr in dir(block) if not attr.startswith('_')]
                                self.logger.info(f"🔧 ToolResultBlock attributes: {all_attrs}")

                                # Check for result content - handle both string and list formats
                                if hasattr(block, 'content') and block.content:
                                    import json

                                    self.logger.info(f"🔧 ToolResultBlock has content: {type(block.content)}")

                                    # Case 1: Content is a string (JSON data or plain text)
                                    if isinstance(block.content, str):
                                        self.logger.info(f"🔧 Processing string content: {len(block.content)} characters")

                                        # First, try to parse as JSON
                                        try:
                                            parsed_data = json.loads(block.content)
                                            self.logger.info(f"🔧 Successfully parsed JSON string from tool result")
                                            await self._handle_tool_result_data(parsed_data, session_id)
                                        except json.JSONDecodeError:
                                            # If not JSON, treat as plain text result
                                            self.logger.info(f"🔧 Content is plain text, creating simple result structure")

                                            # Create a simple result structure for plain text content
                                            text_result = {
                                                "content": [{"type": "text", "text": block.content}],
                                                "success": True,
                                                "text_output": block.content,
                                                "tool_type": "text_response"
                                            }

                                            # Try to infer tool name from context or content
                                            tool_name = self._infer_tool_name_from_context(block.content)
                                            await self._handle_tool_file_creation(tool_name, text_result, session_id)

                                    # Case 2: Content is a list of content items
                                    elif isinstance(block.content, list):
                                        self.logger.info(f"🔧 Processing list content: {len(block.content)} items")
                                        for content_item in block.content:
                                            # Handle content items with text attribute
                                            if hasattr(content_item, 'text') and content_item.text:
                                                try:
                                                    parsed_data = json.loads(content_item.text)
                                                    self.logger.info(f"🔧 Successfully parsed JSON from content item text")
                                                    await self._handle_tool_result_data(parsed_data, session_id)
                                                except json.JSONDecodeError as e:
                                                    self.logger.warning(f"🔧 Could not parse content item text as JSON: {e}")
                                                    self.logger.debug(f"🔧 Raw content item: {content_item.text[:200]}...")

                                            # Handle content items that are already dictionaries
                                            elif isinstance(content_item, dict):
                                                self.logger.info(f"🔧 Processing dictionary content item")
                                                await self._handle_tool_result_data(content_item, session_id)

                                    # Case 3: Content is already a dictionary
                                    elif isinstance(block.content, dict):
                                        self.logger.info(f"🔧 Processing dictionary content directly")
                                        await self._handle_tool_result_data(block.content, session_id)

                                # Also check for direct result attribute
                                elif hasattr(block, 'result') and block.result:
                                    self.logger.info(f"🔧 Found direct result attribute")
                                    await self._handle_tool_file_creation('unknown_tool', block.result, session_id)

                    # Check for completion
                    if hasattr(message, 'result') or elapsed > timeout_seconds:
                        if elapsed > timeout_seconds:
                            self.logger.warning(f"⚠️ Research collection timeout after {elapsed:.1f}s")
                        break

                self.logger.info(f"🔧 Research attempt {attempt} completed: {len(research_results)} messages, tools: {executed_tools}")

            except Exception as e:
                self.logger.error(f"❌ Error collecting research responses: {e}")

            # Check if we have the required tools
            if required_tools.issubset(executed_tools):
                self.logger.info(f"✅ All required tools executed: {executed_tools}")
                break
            else:
                missing = required_tools - executed_tools
                self.logger.warning(f"⚠️ Missing required tools: {missing}")
                if attempt < max_attempts:
                    self.logger.info(f"🔄 Retrying with stronger tool enforcement...")
                await asyncio.sleep(1)  # Brief pause between attempts

        # Final status
        if not required_tools.issubset(executed_tools):
            self.logger.error(f"❌ Failed to execute required tools after {max_attempts} attempts: {required_tools - executed_tools}")
            # Add fallback response
            fallback_response = {
                "agent": "research_agent",
                "stage": "research",
                "timestamp": datetime.now().isoformat(),
                "response": f"Research tool execution failed. Required tools not executed: {required_tools - executed_tools}",
                "error_type": "tool_execution_failure",
                "attempts": attempt,
                "executed_tools": list(executed_tools),
                "fallback": True
            }
            research_results.append(fallback_response)

        return research_results

    def _create_session_id(self) -> str:
        """Create a new session ID."""
        return str(uuid.uuid4())

    def _cleanup_sessions_directory(self):
        """Clean up logs, work products, and sessions directory for clean slate."""
        import shutil

        # Define directories to clean
        cleanup_dirs = [
            "KEVIN/sessions",
            "logs",
            "work_products"  # Clean up any legacy work products
        ]

        for dir_path in cleanup_dirs:
            full_path = Path(dir_path)
            if full_path.exists():
                try:
                    self.logger.info(f"🧹 Cleaning up directory: {dir_path}")
                    shutil.rmtree(full_path)
                    self.logger.info(f"✅ Successfully cleaned: {dir_path}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to clean {dir_path}: {e}")

        self.logger.info("🧹 Session cleanup completed - ready for fresh start")

    async def start_research_session(self, topic: str, user_requirements: dict[str, Any]) -> str:
        """Start a new research session."""
        session_id = self._create_session_id()

        # Enhanced structured logging for session creation
        self.logger.info(f"Starting new research session for topic: {topic}")
        self.structured_logger.info(f"Starting research session: {topic}",
                                    event_type="session_creation_start",
                                    topic=topic,
                                    user_requirements=user_requirements)

        # Use specialized session lifecycle logger
        session_config = {
            "topic": topic,
            "user_requirements": user_requirements,
            "debug_mode": self.debug_mode,
            "available_agents": list(self.agent_definitions.keys())
        }
        # Hook system disabled - no session lifecycle logging

        self.logger.debug(f"Generated session ID: {session_id}")
        self.structured_logger.debug("Session ID generated", session_id=session_id)

        # Clean up previous sessions, logs, and work products for clean slate
        self._cleanup_sessions_directory()

        # Create session directory in KEVIN structure with absolute path
        # Get the project root directory (where KEVIN should be)
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent  # Go up from core/ to project root
        kevin_dir = project_root / "KEVIN"
        session_path = kevin_dir / "sessions" / session_id

        session_path.mkdir(parents=True, exist_ok=True)
        # Create subdirectories for organization
        (session_path / "research").mkdir(exist_ok=True)
        (session_path / "working").mkdir(exist_ok=True)
        (session_path / "final").mkdir(exist_ok=True)
        (session_path / "agent_logs").mkdir(exist_ok=True)

        self.logger.debug(f"Created session directory: {session_path}")
        self.logger.info(f"KEVIN directory structure: {kevin_dir}")
        self.structured_logger.info("Session directory structure created",
                                    session_path=str(session_path),
                                    kevin_dir=str(kevin_dir))

        # Initialize session state
        self.active_sessions[session_id] = {
            "session_id": session_id,
            "topic": topic,
            "user_requirements": user_requirements,
            "status": "initialized",
            "created_at": datetime.now().isoformat(),
            "current_stage": "research",
            "workflow_history": [],
            "final_report": None,
            "search_budget": SessionSearchBudget(session_id)  # Add search budget tracking
        }

        # Save session state
        await self.save_session_state(session_id)
        self.logger.debug(f"Session state saved for {session_id}")
        self.structured_logger.info("Session state saved and initialized",
                                    session_id=session_id,
                                    session_status="initialized")

        # Hook system disabled - no workflow logging

        # Start the research workflow
        asyncio.create_task(self.execute_research_workflow(session_id))
        self.logger.info(f"Research workflow started for session {session_id}")
        self.structured_logger.info("Research workflow task created",
                                    session_id=session_id,
                                    workflow_type="async_task")

        return session_id

    async def execute_research_workflow(self, session_id: str):
        """Execute the complete research workflow."""
        self.logger.info(f"Starting research workflow for session {session_id}")
        self.structured_logger.info(f"Research workflow started: {session_id}",
                                    event_type="workflow_start",
                                    session_id=session_id)

        # Hook system disabled - no workflow logging

        try:
            # Initialize agent logger for this session with correct path
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent  # Go up from core/ to project root
            kevin_sessions_dir = project_root / "KEVIN" / "sessions"
            self.agent_logger = AgentLoggerFactory.get_logger(session_id, str(kevin_sessions_dir))
            self.agent_logger.log_activity(
                agent_name="orchestrator",
                activity_type="workflow_start",
                stage="initialization",
                input_data={"session_id": session_id},
                metadata={"event": "research_workflow_started"}
            )

            # Enhanced structured logging for workflow initialization
            self.structured_logger.info("Agent logger initialized for session",
                                        session_id=session_id,
                                        log_directory=str(kevin_sessions_dir))

            # Health checks and tool verification moved to dedicated test suite
            # Use: python tests/test_startup_health.py to verify system health
            self.logger.info("🚀 Starting research workflow...")

            session_data = self.active_sessions[session_id]
            topic = session_data["topic"]
            user_requirements = session_data["user_requirements"]

            self.logger.debug(f"Session {session_id}: Topic={topic}, Requirements={user_requirements}")
            self.agent_logger.log_activity(
                agent_name="orchestrator",
                activity_type="session_parameters",
                stage="initialization",
                input_data={"topic": topic, "user_requirements": user_requirements},
                metadata={"event": "session_parameters_set"}
            )

            # Stage 1: Research
            self.logger.info(f"Session {session_id}: Starting research stage")
            self.agent_logger.log_stage_transition("initialization", "research", "orchestrator", {"topic": topic})

            # Hook system disabled - no workflow logging
            self.structured_logger.info("Research stage started",
                                        session_id=session_id,
                                        from_stage="initialization",
                                        to_stage="research",
                                        topic=topic)

            await self.stage_conduct_research(session_id, topic, user_requirements)

            # Stage 2: Report Generation
            self.logger.info(f"Session {session_id}: Starting report generation stage")
            self.agent_logger.log_stage_transition("research", "report_generation", "orchestrator")

            # Hook system disabled - no workflow logging
            self.structured_logger.info("Report generation stage started",
                                        session_id=session_id,
                                        from_stage="research",
                                        to_stage="report_generation")

            await self.stage_generate_report(session_id)

            # Stage 3: Editorial Review
            self.logger.info(f"Session {session_id}: Starting editorial review stage")
            self.agent_logger.log_stage_transition("report_generation", "editorial_review", "orchestrator")

            # Hook system disabled - no workflow logging
            self.structured_logger.info("Editorial review stage started",
                                        session_id=session_id,
                                        from_stage="report_generation",
                                        to_stage="editorial_review")

            await self.stage_editorial_review(session_id)

            # Stage 4: Finalization
            self.logger.info(f"Session {session_id}: Starting finalization stage")
            self.agent_logger.log_stage_transition("editorial_review", "finalization", "orchestrator")

            # Hook system disabled - no workflow logging
            self.structured_logger.info("Finalization stage started",
                                        session_id=session_id,
                                        from_stage="editorial_review",
                                        to_stage="finalization")

            await self.stage_finalize(session_id)

            self.logger.info(f"Session {session_id}: Research workflow completed successfully")
            self.structured_logger.info("Research workflow completed successfully",
                                        event_type="workflow_complete",
                                        session_id=session_id,
                                        final_stage="finalization")

            # Hook system disabled - no workflow logging

            # Use UI coordinator logger for orchestration summary
            ui_coordinator_logger = self.get_agent_logger("ui_coordinator")
            ui_coordinator_logger.log_progress_monitoring(
                current_stage="completed",
                progress_percentage=100.0,
                milestones_reached=["research", "report_generation", "editorial_review", "finalization"],
                session_id=session_id
            )

            self.agent_logger.log_activity(
                agent_name="orchestrator",
                activity_type="workflow_complete",
                stage="finalization",
                metadata={"event": "research_workflow_completed"}
            )

        except Exception as e:
            # Enhanced structured logging for workflow errors
            self.logger.error(f"Error in research workflow for session {session_id}: {e}")
            self.structured_logger.error("Research workflow error occurred",
                                        event_type="workflow_error",
                                        session_id=session_id,
                                        error_type=type(e).__name__,
                                        error_message=str(e),
                                        stage="workflow_execution")

            import traceback
            self.logger.error(traceback.format_exc())
            self.structured_logger.error("Workflow error traceback",
                                        traceback=traceback.format_exc(),
                                        session_id=session_id)

            # Hook system disabled - no workflow logging

            if self.agent_logger:
                self.agent_logger.log_error(
                    agent_name="orchestrator",
                    error=e,
                    context={"session_id": session_id, "stage": "workflow_execution"},
                    stage="error"
                )

            await self.update_session_status(session_id, "error", str(e))

        finally:
            # Finalize agent logger and export debug report
            if self.agent_logger:
                try:
                    self.agent_logger.finalize_session()
                    debug_report_path = self.agent_logger.export_debug_report()
                    self.logger.info(f"Agent debug report exported to: {debug_report_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to export agent debug report: {e}")

                # Clear the agent logger reference
                self.agent_logger = None

    async def execute_agent_query_with_response_collection(self, agent_client, agent_name: str, prompt: str, session_id: str, timeout_seconds: int = 120) -> dict[str, Any]:
        """Execute agent query with proper response collection."""
        self.logger.debug(f"Executing query for {agent_name}")

        query_result = {
            "agent_name": agent_name,
            "session_id": session_id,
            "prompt_sent": prompt,
            "messages_collected": [],
            "substantive_responses": 0,
            "tool_executions": [],
            "errors": [],
            "query_start_time": datetime.now().isoformat(),
            "success": False
        }

        try:
            # Send the query
            await agent_client.query(prompt, session_id=session_id)

            # Collect responses with timeout
            start_time = asyncio.get_event_loop().time()

            async def collect_responses():
                async for message in agent_client.receive_response():
                    current_time = asyncio.get_event_loop().time()
                    elapsed = current_time - start_time

                    message_info = {
                        "message_type": type(message).__name__,
                        "timestamp": datetime.now().isoformat(),
                        "elapsed_seconds": elapsed
                    }

                    # Extract content information
                    if hasattr(message, 'content') and message.content:
                        content_texts = []
                        for block in message.content:
                            if hasattr(block, 'text') and block.text:
                                content_texts.append(block.text)
                        if content_texts:
                            message_info["content_texts"] = content_texts
                            query_result["substantive_responses"] += 1
                            self.logger.debug(f"{agent_name} content: {content_texts[0][:100]}...")

                    # Extract tool use information
                    if hasattr(message, 'tool_use') and message.tool_use:
                        message_info["tool_use"] = {
                            "name": message.tool_use.get("name"),
                            "id": message.tool_use.get("id")
                        }
                        query_result["tool_executions"].append(message_info["tool_use"])
                        self.logger.info(f"{agent_name} executed tool: {message_info['tool_use']['name']}")

                    # Extract result information
                    if hasattr(message, 'result'):
                        message_info["has_result"] = True
                        self.logger.debug(f"{agent_name} received result message")

                    query_result["messages_collected"].append(message_info)

                    # Check for completion or timeout
                    if elapsed > timeout_seconds:
                        self.logger.warning(f"{agent_name} response collection timeout after {elapsed:.1f}s")
                        break

                    # Stop if we have substantive responses and a result
                    if query_result["substantive_responses"] > 0 and any(msg.get("has_result") for msg in query_result["messages_collected"]):
                        self.logger.info(f"{agent_name} completed with {len(query_result['messages_collected'])} messages")
                        break

            # Execute with timeout
            await asyncio.wait_for(collect_responses(), timeout=timeout_seconds + 10)

            query_result["success"] = True
            query_result["query_end_time"] = datetime.now().isoformat()

            self.logger.info(f"{agent_name} query completed: {len(query_result['messages_collected'])} messages, {query_result['substantive_responses']} substantive responses, {len(query_result['tool_executions'])} tool executions")

        except asyncio.TimeoutError:
            error_msg = f"{agent_name} query timeout after {timeout_seconds}s"
            self.logger.error(error_msg)
            query_result["errors"].append(error_msg)
        except Exception as e:
            error_msg = f"{agent_name} query error: {str(e)}"
            self.logger.error(error_msg)
            query_result["errors"].append(error_msg)

        return query_result

    async def stage_conduct_research(self, session_id: str, topic: str, user_requirements: dict[str, Any]):
        """Stage 1: Conduct research using Research Agent."""
        self.logger.info(f"Session {session_id}: Starting research on {topic}")

        await self.update_session_status(session_id, "researching", "Conducting initial research")

        # Validate search budget before starting
        search_budget = self.active_sessions[session_id]["search_budget"]
        can_proceed, budget_message = search_budget.can_primary_research_proceed(10)
        if not can_proceed:
            self.logger.error(f"Session {session_id}: Cannot proceed with research: {budget_message}")
            raise RuntimeError(f"Search budget limit reached: {budget_message}")

        max_attempts = 3
        research_successful = False
        research_result = None

        for attempt in range(max_attempts):
            try:
                self.logger.info(f"Session {session_id}: Research attempt {attempt + 1}/{max_attempts}")

                # Create comprehensive research prompt with natural language agent selection
                research_prompt = f"""
                Use the research_agent agent to conduct comprehensive research on the topic: "{topic}"

                User Requirements:
                {json.dumps(user_requirements, indent=2)}

                MANDATORY RESEARCH INSTRUCTIONS:
                1. IMMEDIATELY execute mcp__research_tools__serp_search with the topic
                2. Set num_results to 15 for comprehensive coverage
                3. Set auto_crawl_top to 8 for detailed content extraction
                4. Set crawl_threshold to 0.3 for relevance filtering
                5. Use mcp__research_tools__save_research_findings to save your findings
                6. Use mcp__research_tools__capture_search_results to structure results

                SEARCH BUDGET CONSTRAINTS:
                - **STRICT LIMIT**: Maximum 10 successful content extractions per session
                - **BUDGET AWARENESS**: Each search consumes from your session budget
                - **EFFICIENCY REQUIRED**: Make each search count with quality sources

                REQUIREMENTS:
                - Execute actual searches using the SERP API tool
                - Gather specific facts, data points, and expert opinions
                - Validate source credibility and authority
                - Save comprehensive findings to files for other agents
                - Do not acknowledge the task - EXECUTE the research immediately

                Session ID: {session_id}
                """

                # Execute research using the new single client pattern with natural language agent selection
                research_result = await self.execute_agent_query(
                    "research_agent", research_prompt, session_id, timeout_seconds=180
                )

                self.logger.info(f"✅ Research execution completed: {research_result['substantive_responses']} responses, {research_result['tool_executions']} tools")

                # Validate research completion
                if self._validate_research_completion(research_result):
                    research_successful = True
                    break
                else:
                    self.logger.warning(f"Session {session_id}: Research attempt {attempt + 1} did not complete required work")

            except Exception as e:
                self.logger.error(f"Session {session_id}: Research attempt {attempt + 1} failed: {e}")
                if attempt == max_attempts - 1:
                    raise
                await asyncio.sleep(2)  # Brief delay before retry

        if not research_successful:
            raise RuntimeError(f"Research stage failed after {max_attempts} attempts")

        # Record search budget usage
        if research_result and research_result.get("tool_executions"):
            # Estimate search usage based on tool executions
            estimated_scrapes = min(10, len(research_result["tool_executions"]))
            search_budget.record_primary_research(
                urls_processed=len(research_result["tool_executions"]) * 10,  # Estimate
                successful_scrapes=estimated_scrapes,
                search_queries=1
            )

        # Store research results
        session_data = self.active_sessions[session_id]
        session_data["research_results"] = research_result
        session_data["current_stage"] = "report_generation"
        session_data["workflow_history"].append({
            "stage": "research",
            "completed_at": datetime.now().isoformat(),
            "responses_count": research_result["substantive_responses"],
            "tools_executed": len(research_result["tool_executions"]),
            "success": research_result["success"],
            "attempts": attempt + 1
        })

        await self.save_session_state(session_id)
        self.logger.info(f"Session {session_id}: Research stage completed successfully")

    async def stage_generate_report(self, session_id: str):
        """Stage 2: Generate report using Report Agent."""
        self.logger.info(f"Session {session_id}: Generating report")

        await self.update_session_status(session_id, "generating_report", "Creating initial report")

        session_data = self.active_sessions[session_id]

        # Validate that research data exists before generating report
        if not session_data.get("research_results") or not session_data["research_results"].get("success"):
            raise RuntimeError("Cannot generate report: research stage did not complete successfully")

        max_attempts = 3
        report_successful = False
        report_result = None

        for attempt in range(max_attempts):
            try:
                self.logger.info(f"Session {session_id}: Report generation attempt {attempt + 1}/{max_attempts}")

                report_prompt = f"""
                Use the report_agent agent to generate a comprehensive report based on the research findings.

                Topic: {session_data['topic']}
                User Requirements: {json.dumps(session_data['user_requirements'], indent=2)}

                Research results have been collected and are available in the session data.

                Use the mcp__research_tools__get_session_data tool to retrieve research findings, then:

                1. Create a well-structured report on the topic
                2. Include all key findings from the research
                3. Organize content logically with clear sections
                4. Ensure proper citations and source attribution
                5. Target the report to the user's specified audience
                6. Use mcp__research_tools__create_research_report to create the report
                7. CRITICAL: Save the report using the provided filepath from the tool

                REQUIREMENTS:
                - Execute the create_research_report tool to generate the report
                - Use the Write tool to save the report to the exact filepath provided
                - Do not just describe the report - actually create and save it
                - Ensure the report is comprehensive and well-structured

                Session ID: {session_id}
                """

                # Execute report generation using the new single client pattern with natural language agent selection
                report_result = await self.execute_agent_query(
                    "report_agent", report_prompt, session_id, timeout_seconds=150
                )

                self.logger.info(f"✅ Report generation completed: {report_result['substantive_responses']} responses, {report_result['tool_executions']} tools")

                # Validate report completion
                if self._validate_report_completion(report_result):
                    report_successful = True
                    break
                else:
                    self.logger.warning(f"Session {session_id}: Report generation attempt {attempt + 1} did not complete required work")

            except Exception as e:
                self.logger.error(f"Session {session_id}: Report generation attempt {attempt + 1} failed: {e}")
                if attempt == max_attempts - 1:
                    raise
                await asyncio.sleep(2)  # Brief delay before retry

        if not report_successful:
            raise RuntimeError(f"Report generation stage failed after {max_attempts} attempts")

        # Store report results
        session_data["report_results"] = report_result
        session_data["current_stage"] = "editorial_review"
        session_data["workflow_history"].append({
            "stage": "report_generation",
            "completed_at": datetime.now().isoformat(),
            "responses_count": report_result["substantive_responses"],
            "tools_executed": len(report_result["tool_executions"]),
            "success": report_result["success"],
            "attempts": attempt + 1
        })

        await self.save_session_state(session_id)
        self.logger.info(f"Session {session_id}: Report generation stage completed successfully")

    async def stage_editorial_review(self, session_id: str):
        """Stage 3: Editorial review using Editor Agent with success-based search controls."""
        self.logger.info(f"Session {session_id}: Conducting editorial review")

        await self.update_session_status(session_id, "editorial_review", "Reviewing report quality")

        session_data = self.active_sessions[session_id]
        search_budget = session_data["search_budget"]

        # Validate search budget before starting editorial research
        can_proceed, budget_message = search_budget.can_editorial_research_proceed(5)
        if not can_proceed:
            self.logger.warning(f"Session {session_id}: Editorial search budget reached: {budget_message}")
            # Continue with review but no additional searches

        max_attempts = 3
        editorial_successful = False
        review_result = None

        for attempt in range(max_attempts):
            try:
                self.logger.info(f"Session {session_id}: Editorial review attempt {attempt + 1}/{max_attempts}")

                # Initialize editorial search tracking
                session_data["editorial_search_stats"] = {
                    "search_attempts": 0,
                    "successful_scrapes": 0,
                    "urls_attempted": 0,
                    "search_limit_reached": False
                }

                # Create current budget status for the prompt
                budget_status = search_budget.get_budget_status()
                editorial_remaining = budget_status["editorial"]["search_queries_remaining"]

                review_prompt = f"""
                Use the editor_agent agent to review the generated report for quality, accuracy, and completeness.

                Topic: {session_data['topic']}

                EDITORIAL SEARCH CONTROLS:
                **SUCCESS-BASED TERMINATION**: Continue searching until you achieve 5 successful scrapes total
                **SEARCH LIMITS**: Maximum {editorial_remaining} editorial search attempts remaining
                **WORK PRODUCT LABELING**: CRITICAL - Always use workproduct_prefix="editor research"
                **BUDGET AWARENESS**: Track your search usage to stay within limits

                CURRENT BUDGET STATUS:
                - Search queries remaining: {editorial_remaining}/2
                - Successful scrapes: {budget_status["editorial"]["successful_scrapes"]}
                - Search limit reached: {budget_status["editorial"]["search_queries_reached_limit"]}

                Use the Read tool to examine the generated report files, then provide comprehensive review:

                1. Assess report quality against professional standards
                2. Check accuracy and proper source attribution
                3. Evaluate clarity, organization, and completeness
                4. Identify specific gaps or areas needing improvement
                5. Provide specific, actionable feedback

                SEARCH GUIDELINES:
                - Only search for SPECIFIC identified gaps, not general "more information"
                - Use workproduct_prefix="editor research" for all editorial searches
                - STOP searching when you reach your search query limit or 5 successful scrapes
                - Each successful scrape should provide meaningful content for gap-filling

                If you identify research gaps and budget allows, conduct targeted searches following the guidelines above.

                Provide detailed feedback that will help improve the report to meet professional standards.
                """

                # Execute editorial review with extended timeout for search activities
                review_result = await self.execute_agent_query(
                    "editor_agent", review_prompt, session_id, timeout_seconds=300  # 5 minutes for editorial searches
                )

                self.logger.info(f"✅ Editorial review completed: {review_result['substantive_responses']} responses, {review_result['tool_executions']} tools")

                # Validate editorial completion
                if self._validate_editorial_completion(review_result):
                    editorial_successful = True
                    break
                else:
                    self.logger.warning(f"Session {session_id}: Editorial review attempt {attempt + 1} did not complete required work")

            except Exception as e:
                self.logger.error(f"Session {session_id}: Editorial review attempt {attempt + 1} failed: {e}")
                if attempt == max_attempts - 1:
                    raise
                await asyncio.sleep(2)  # Brief delay before retry

        if not editorial_successful:
            raise RuntimeError(f"Editorial review stage failed after {max_attempts} attempts")

        # Record editorial search statistics
        search_stats = session_data.get("editorial_search_stats", {})
        if search_stats.get("search_attempts", 0) > 0:
            search_budget.record_editorial_research(
                urls_processed=search_stats.get("urls_attempted", 0),
                successful_scrapes=search_stats.get("successful_scrapes", 0),
                search_queries=search_stats.get("search_attempts", 0)
            )

        # Log editorial search statistics
        self.logger.info(f"📊 Editorial search stats: {search_stats.get('search_attempts', 0)} attempts, {search_stats.get('successful_scrapes', 0)} successful scrapes")

        # Store review results
        session_data["review_results"] = review_result
        session_data["current_stage"] = "finalization"
        session_data["workflow_history"].append({
            "stage": "editorial_review",
            "completed_at": datetime.now().isoformat(),
            "responses_count": review_result["substantive_responses"],
            "tools_executed": len(review_result["tool_executions"]),
            "success": review_result["success"],
            "attempts": attempt + 1,
            "editorial_search_stats": search_stats
        })

        await self.save_session_state(session_id)
        self.logger.info(f"Session {session_id}: Editorial review stage completed successfully")

    async def stage_finalize(self, session_id: str):
        """Stage 4: Finalize the report and complete the session."""
        self.logger.info(f"Session {session_id}: Finalizing report")

        await self.update_session_status(session_id, "finalizing", "Completing final report")

        session_data = self.active_sessions[session_id]

        # Check if revisions are needed based on editorial review
        needs_revision = await self.check_if_revision_needed(session_id)

        if needs_revision:
            await self.stage_perform_revisions(session_id)
        else:
            await self.complete_session(session_id)

    async def check_if_revision_needed(self, session_id: str) -> bool:
        """Check if the report needs revisions based on editorial feedback."""
        session_data = self.active_sessions[session_id]
        review_results = session_data.get("review_results", [])

        # Analyze review results to determine if revisions are needed
        # This would parse the actual review feedback
        # For now, assume minor revisions are needed
        return len(review_results) > 0

    async def stage_perform_revisions(self, session_id: str):
        """Perform revisions based on editorial feedback."""
        self.logger.info(f"Session {session_id}: Performing revisions")

        await self.update_session_status(session_id, "revising", "Applying editorial feedback")

        session_data = self.active_sessions[session_id]

        revision_prompt = f"""
        Use the report_agent agent to revise the report based on the editorial feedback provided.

        Topic: {session_data['topic']}

        Use the Read tool to examine the current report and editorial feedback, then:

        1. Address all feedback from the editorial review
        2. Improve report quality based on specific recommendations
        3. Ensure all identified issues are resolved
        4. Maintain overall report coherence and quality
        5. Use the Write tool to save the improved report
        6. CRITICAL: Add "3-" prefix to your revised report title to indicate this is Stage 3 output

        Focus on implementing the feedback systematically and improving the report to meet professional standards.
        """

        # Execute revisions using the new single client pattern with natural language agent selection
        revision_result = await self.execute_agent_query(
            "report_agent", revision_prompt, session_id, timeout_seconds=120
        )

        self.logger.info(f"✅ Revision execution completed: {revision_result['substantive_responses']} responses, {revision_result['tool_executions']} tools")

        # Store revision results
        session_data["revision_results"] = revision_result
        session_data["workflow_history"].append({
            "stage": "revisions",
            "completed_at": datetime.now().isoformat(),
            "responses_count": revision_result["substantive_responses"],
            "tools_executed": len(revision_result["tool_executions"]),
            "success": revision_result["success"]
        })

        await self.save_session_state(session_id)

        # Complete the session after revisions
        await self.complete_session(session_id)

    async def complete_session(self, session_id: str):
        """Complete the research session."""
        self.logger.info(f"Session {session_id}: Completing session")

        await self.update_session_status(session_id, "completed", "Research workflow completed")

        session_data = self.active_sessions[session_id]
        session_data["completed_at"] = datetime.now().isoformat()
        session_data["workflow_history"].append({
            "stage": "completion",
            "completed_at": datetime.now().isoformat()
        })

        # Create final summary (Stage 4 work product)
        try:
            final_summary_prompt = f"""
            Create a final summary document for the completed research session.

            Topic: {session_data['topic']}

            Based on all the work completed (research, report, editorial review, and revisions),
            create a concise final summary that includes:
            1. Session overview and key findings
            2. Summary of the research workflow stages completed
            3. Location of all work products created
            4. Final quality assessment

            CRITICAL: Add "4-" prefix to your final summary title to indicate this is Stage 4 output
            Save this as a final summary document in the session directory.
            """

            # Execute final summary creation
            final_summary_result = await self.execute_agent_query(
                "report_agent", final_summary_prompt, session_id, timeout_seconds=60
            )

            self.logger.info(f"✅ Final summary created: {final_summary_result['substantive_responses']} responses")

        except Exception as e:
            self.logger.warning(f"Could not create final summary: {e}")

        await self.save_session_state(session_id)
        self.logger.info(f"Session {session_id}: Research workflow completed successfully")

    def get_debug_output(self) -> list[str]:
        """Get all debug output collected from stderr callbacks."""
        return self.debug_output.copy()

    def clear_debug_output(self):
        """Clear the debug output buffer."""
        self.debug_output.clear()

    def get_final_report(self, session_id: str) -> dict[str, Any]:
        """Get the final research report for a session."""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}

        session_data = self.active_sessions[session_id]

        # Check for KEVIN directory files
        kevin_dir = Path("/home/kjdragan/lrepos/claude-agent-sdk-python/KEVIN")
        if kevin_dir.exists():
            # Look for report files with this session
            report_files = list(kevin_dir.glob("research_report_*.md"))
            if report_files:
                latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
                try:
                    with open(latest_report, 'r', encoding='utf-8') as f:
                        content = f.read()
                    return {
                        "report_file": str(latest_report),
                        "report_content": content,
                        "report_length": len(content),
                        "session_id": session_id
                    }
                except Exception as e:
                    self.logger.error(f"Error reading report: {e}")

        return {"error": "No report found"}

    async def get_session_status(self, session_id: str) -> dict[str, Any]:
        """Get current status of a research session."""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}

        session_data = self.active_sessions[session_id]
        return {
            "session_id": session_id,
            "status": session_data.get("status"),
            "current_stage": session_data.get("current_stage"),
            "topic": session_data.get("topic"),
            "created_at": session_data.get("created_at"),
            "workflow_history": session_data.get("workflow_history", [])
        }

    async def update_session_status(self, session_id: str, status: str, message: str = ""):
        """Update session status and save state."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["status"] = status
            self.active_sessions[session_id]["status_message"] = message
            self.active_sessions[session_id]["last_updated"] = datetime.now().isoformat()
            await self.save_session_state(session_id)

    async def save_session_state(self, session_id: str):
        """Save session state to file."""
        session_data = self.active_sessions[session_id]

        # Use absolute path for session file
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent  # Go up from core/ to project root
        session_dir = project_root / "KEVIN" / "sessions" / session_id
        session_file = session_dir / "session_state.json"

        # Ensure session directory exists
        session_dir.mkdir(parents=True, exist_ok=True)

        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)

    def get_search_budget(self, session_id: str) -> Optional[SessionSearchBudget]:
        """Get the search budget for a session."""
        if session_id not in self.active_sessions:
            return None
        return self.active_sessions[session_id].get("search_budget")

    def validate_research_budget(self, session_id: str, agent_type: str, urls_to_process: int = 1) -> tuple[bool, str]:
        """Validate if research can proceed based on budget constraints."""
        budget = self.get_search_budget(session_id)
        if not budget:
            return False, "No search budget found for session"

        if agent_type == "research_agent":
            return budget.can_primary_research_proceed(urls_to_process)
        elif agent_type == "editor_agent":
            return budget.can_editorial_research_proceed(urls_to_process)
        else:
            return False, f"Unknown agent type: {agent_type}"

    def record_research_activity(self, session_id: str, agent_type: str, urls_processed: int, successful_scrapes: int, search_queries: int = 1):
        """Record research activity in the budget."""
        budget = self.get_search_budget(session_id)
        if not budget:
            self.logger.warning(f"No search budget found for session {session_id}")
            return

        if agent_type == "research_agent":
            budget.record_primary_research(urls_processed, successful_scrapes, search_queries)
        elif agent_type == "editor_agent":
            budget.record_editorial_research(urls_processed, successful_scrapes, search_queries)
        else:
            self.logger.warning(f"Unknown agent type for research recording: {agent_type}")

    def get_budget_summary(self, session_id: str) -> dict[str, Any]:
        """Get a summary of the search budget status."""
        budget = self.get_search_budget(session_id)
        if not budget:
            return {"error": "No search budget found for session"}

        return budget.get_budget_status()

    async def _diagnose_mcp_timeout_issue(self, agent_name: str, session_id: str, timeout_duration: int):
        """Diagnose MCP timeout issues and log detailed analysis."""
        self.logger.error(f"🔍 DIAGNOSING MCP TIMEOUT for {agent_name} (session: {session_id})")

        # Check MCP server status
        try:
            if self.mcp_server:
                mcp_status = await self._check_mcp_server_health()
                self.logger.error(f"🔍 MCP Server Status: {mcp_status}")
            else:
                self.logger.error("🔍 MCP Server: None initialized")
        except Exception as e:
            self.logger.error(f"🔍 MCP Server Health Check Failed: {e}")

        # Check single client connectivity
        try:
            if hasattr(self, 'client') and self.client:
                self.logger.error(f"🔍 Single Client: {type(self.client).__name__}")
                self.logger.error(f"🔍 Client Methods: {[m for m in dir(self.client) if not m.startswith('_')]}")
                self.logger.error(f"🔍 Available Agents: {self.agent_names}")
            else:
                self.logger.error(f"🔍 Single Client: Not initialized")
        except Exception as e:
            self.logger.error(f"🔍 Single Client Check Failed: {e}")

        # Check session state
        try:
            if session_id in self.active_sessions:
                session_data = self.active_sessions[session_id]
                self.logger.error(f"🔍 Session Stage: {session_data.get('current_stage')}")
                self.logger.error(f"🔍 Session Status: {session_data.get('status')}")
            else:
                self.logger.error(f"🔍 Session: {session_id} not found in active sessions")
        except Exception as e:
            self.logger.error(f"🔍 Session State Check Failed: {e}")

        # Log timeout analysis
        self.logger.error(f"🔍 Timeout Analysis:")
        self.logger.error(f"  - Duration: {timeout_duration}s")
        self.logger.error(f"  - Agent: {agent_name}")
        self.logger.error(f"  - Session: {session_id}")
        self.logger.error(f"  - Possible Causes: MCP server unresponsive, network issues, complex query processing")
        self.logger.error(f"  - Recommendation: Check MCP server logs and network connectivity")

    async def _diagnose_mcp_error_issue(self, agent_name: str, session_id: str, error: Exception):
        """Diagnose MCP error issues and log detailed analysis."""
        self.logger.error(f"🔍 DIAGNOSING MCP ERROR for {agent_name} (session: {session_id})")
        self.logger.error(f"🔍 Error Type: {type(error).__name__}")
        self.logger.error(f"🔍 Error Message: {str(error)}")

        # Check if it's an MCP-specific error
        error_str = str(error).lower()
        if any(keyword in error_str for keyword in ['mcp', 'server', 'connection', 'transport']):
            self.logger.error("🔍 MCP-related error detected")
            self.logger.error("🔍 Recommendation: Check MCP server configuration and connectivity")

        if any(keyword in error_str for keyword in ['tool', 'function', 'call']):
            self.logger.error("🔍 Tool execution error detected")
            self.logger.error("🔍 Recommendation: Verify tool availability and permissions")

        # Log error analysis
        self.logger.error(f"🔍 Error Analysis:")
        self.logger.error(f"  - Agent: {agent_name}")
        self.logger.error(f"  - Session: {session_id}")
        self.logger.error(f"  - Error Class: {error.__class__.__name__}")
        self.logger.error(f"  - Error Details: {str(error)}")

        # Try to get stack trace information
        import traceback
        self.logger.error(f"🔍 Stack Trace: {traceback.format_exc()}")

    async def _check_mcp_server_health(self) -> dict[str, Any]:
        """Check the health of the MCP server."""
        try:
            if not self.mcp_server:
                return {"status": "not_initialized", "error": "MCP server not created"}

            # Try to get server info
            server_info = self.mcp_server.get_server_info()
            return {
                "status": "running",
                "server_info": server_info,
                "agent_count": len(self.agent_names)
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }

    async def _handle_tool_file_creation(self, tool_name: str, tool_result: Any, session_id: str):
        """Handle file creation for tools that return data instead of writing files directly."""
        try:
            import json
            import os
            from pathlib import Path

            self.logger.info(f"🔧 Handling file creation for tool: {tool_name}")
            self.logger.info(f"🔧 Tool result type: {type(tool_result)}")

            # Convert tool result to expected format if needed
            parsed_result = None

            # If it's already a dict with success flag, use it directly
            if isinstance(tool_result, dict) and tool_result.get("success"):
                parsed_result = tool_result
                self.logger.info(f"🔧 Using tool result directly")

            # If it's a string, try to parse as JSON
            elif isinstance(tool_result, str):
                try:
                    parsed_result = json.loads(tool_result)
                    self.logger.info(f"🔧 Parsed tool result from JSON string")
                except json.JSONDecodeError:
                    self.logger.warning(f"🔧 Could not parse tool result as JSON")
                    return

            # If it has a 'text' attribute, try to parse that
            elif hasattr(tool_result, 'text'):
                try:
                    parsed_result = json.loads(tool_result.text)
                    self.logger.info(f"🔧 Parsed tool result from text attribute")
                except (json.JSONDecodeError, AttributeError):
                    self.logger.warning(f"🔧 Could not parse tool result text as JSON")
                    return

            else:
                self.logger.warning(f"🔧 Unrecognized tool result format")
                return

            # Parse tool result based on tool type
            if tool_name == "mcp__research_tools__save_research_findings":
                await self._create_research_findings_files(parsed_result, session_id)
            elif tool_name == "mcp__research_tools__create_research_report":
                await self._create_research_report_files(parsed_result, session_id)
            elif tool_name == "mcp__research_tools__serp_search":
                await self._create_serp_search_files(parsed_result, session_id)
            else:
                self.logger.debug(f"🔧 No file creation needed for tool: {tool_name}")

        except Exception as e:
            self.logger.error(f"❌ Error handling file creation for {tool_name}: {e}")
            import traceback
            self.logger.error(f"❌ Traceback: {traceback.format_exc()}")

    def _infer_tool_name_from_context(self, content: str) -> str:
        """Infer tool name from content text or current execution context."""
        content_lower = content.lower()

        # Method 1: Look for tool-specific keywords in content
        if 'search' in content_lower and ('result' in content_lower or 'found' in content_lower):
            return 'mcp__research_tools__serp_search'
        elif 'research' in content_lower and ('finding' in content_lower or 'saved' in content_lower):
            return 'mcp__research_tools__save_research_findings'
        elif 'report' in content_lower and ('created' in content_lower or 'generated' in content_lower):
            return 'mcp__research_tools__create_research_report'
        elif 'session' in content_lower and 'data' in content_lower:
            return 'mcp__research_tools__get_session_data'

        # Method 2: Default based on typical workflow order
        # If we can't determine from content, make an educated guess
        if len(content) > 100:  # Longer content is likely search results
            return 'mcp__research_tools__serp_search'
        else:
            return 'mcp__research_tools__save_research_findings'

    async def _handle_tool_result_data(self, tool_result_data: dict, session_id: str):
        """Extract tool name and route tool result data to appropriate file creation handler."""
        try:
            self.logger.info(f"🔧 Processing tool result data: {type(tool_result_data)}")

            # Try to extract tool name from the result data
            tool_name = None

            # Method 1: Look for explicit tool_name field
            if isinstance(tool_result_data, dict):
                tool_name = tool_result_data.get('tool_name')

                # Method 2: Try to infer from content structure
                if not tool_name:
                    content_text = str(tool_result_data.get('content', []))
                    if 'save_research_findings' in content_text:
                        tool_name = 'mcp__research_tools__save_research_findings'
                    elif 'create_research_report' in content_text:
                        tool_name = 'mcp__research_tools__create_research_report'
                    elif 'serp_search' in content_text:
                        tool_name = 'mcp__research_tools__serp_search'

                # Method 3: Check for success flag and file paths (research tools)
                if not tool_name and tool_result_data.get('success'):
                    if 'research_data' in tool_result_data or 'findings' in tool_result_data:
                        tool_name = 'mcp__research_tools__save_research_findings'
                    elif 'report_content' in tool_result_data:
                        tool_name = 'mcp__research_tools__create_research_report'

            # If we still don't have a tool name, log and exit
            if not tool_name:
                self.logger.warning(f"🔧 Could not determine tool name from result data: {list(tool_result_data.keys()) if isinstance(tool_result_data, dict) else 'Not a dict'}")
                return

            self.logger.info(f"🔧 Identified tool: {tool_name}")

            # Route to the appropriate file creation handler
            await self._handle_tool_file_creation(tool_name, tool_result_data, session_id)

        except Exception as e:
            self.logger.error(f"❌ Error processing tool result data: {e}")
            import traceback
            self.logger.error(f"❌ Traceback: {traceback.format_exc()}")

    async def _create_research_findings_files(self, tool_result: Any, session_id: str):
        """Create research findings files from tool result data."""
        try:
            import json
            from pathlib import Path
            from datetime import datetime

            # Extract data from tool result
            if isinstance(tool_result, dict) and tool_result.get("success"):
                research_data = tool_result.get("research_data")
                session_file_path = tool_result.get("session_file_path")
                kevin_file_path = tool_result.get("kevin_file_path")

                if research_data and session_file_path and kevin_file_path:
                    # Create session directory
                    session_dir = Path(session_file_path).parent
                    session_dir.mkdir(parents=True, exist_ok=True)

                    # Create KEVIN directory
                    kevin_dir = Path(kevin_file_path).parent
                    kevin_dir.mkdir(parents=True, exist_ok=True)

                    # Save to session path
                    with open(session_file_path, 'w', encoding='utf-8') as f:
                        json.dump(research_data, f, indent=2, ensure_ascii=False)
                    self.logger.info(f"✅ Created research findings file: {session_file_path}")

                    # Save to KEVIN directory
                    with open(kevin_file_path, 'w', encoding='utf-8') as f:
                        json.dump(research_data, f, indent=2, ensure_ascii=False)
                    self.logger.info(f"✅ Created research findings file in KEVIN: {kevin_file_path}")

                    # Store in session data for later access
                    if session_id in self.active_sessions:
                        self.active_sessions[session_id]["research_findings_file"] = session_file_path
                        self.active_sessions[session_id]["research_findings_kevin_file"] = kevin_file_path

        except Exception as e:
            self.logger.error(f"❌ Error creating research findings files: {e}")

    async def _create_research_report_files(self, tool_result: Any, session_id: str):
        """Create research report files from tool result data."""
        try:
            from pathlib import Path

            # Extract data from tool result
            if isinstance(tool_result, dict) and tool_result.get("success"):
                report_content = tool_result.get("report_content")
                session_file_path = tool_result.get("session_file_path")
                kevin_file_path = tool_result.get("kevin_file_path")

                if report_content and session_file_path and kevin_file_path:
                    # Create session directory
                    session_dir = Path(session_file_path).parent
                    session_dir.mkdir(parents=True, exist_ok=True)

                    # Create KEVIN directory
                    kevin_dir = Path(kevin_file_path).parent
                    kevin_dir.mkdir(parents=True, exist_ok=True)

                    # Save to session path
                    with open(session_file_path, 'w', encoding='utf-8') as f:
                        f.write(report_content)
                    self.logger.info(f"✅ Created research report file: {session_file_path}")

                    # Save to KEVIN directory
                    with open(kevin_file_path, 'w', encoding='utf-8') as f:
                        f.write(report_content)
                    self.logger.info(f"✅ Created research report file in KEVIN: {kevin_file_path}")

                    # Store in session data for later access
                    if session_id in self.active_sessions:
                        self.active_sessions[session_id]["research_report_file"] = session_file_path
                        self.active_sessions[session_id]["research_report_kevin_file"] = kevin_file_path

        except Exception as e:
            self.logger.error(f"❌ Error creating research report files: {e}")

    async def _create_serp_search_files(self, tool_result: Any, session_id: str):
        """Create SERP search result files from tool result data."""
        try:
            from pathlib import Path
            from datetime import datetime
            self.logger.info(f"🔧 Creating SERP search files for session {session_id}")

            # Extract data from tool result
            if isinstance(tool_result, dict):
                # Handle text-based tool results
                search_content = tool_result.get("text_output", "")
                if not search_content and "content" in tool_result:
                    content_list = tool_result.get("content", [])
                    if content_list and len(content_list) > 0:
                        search_content = content_list[0].get("text", "")

                if search_content:
                    # Create session directory
                    session_dir = Path(f"researchmaterials/sessions/{session_id}")
                    session_dir.mkdir(parents=True, exist_ok=True)

                    # Create KEVIN directory
                    kevin_dir = Path("/home/kjdragan/lrepos/claude-agent-sdk-python/KEVIN")
                    kevin_dir.mkdir(parents=True, exist_ok=True)

                    # Generate timestamp
                    timestamp = datetime.now().strftime('%H%M%S')

                    # Create file paths
                    session_file = session_dir / f"serp_search_results_{timestamp}.txt"
                    kevin_file = kevin_dir / f"serp_search_{session_id[:8]}_{timestamp}.txt"

                    # Save search results to session path
                    with open(session_file, 'w', encoding='utf-8') as f:
                        f.write(f"SERP Search Results\n")
                        f.write(f"Session ID: {session_id}\n")
                        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                        f.write(f"{'='*50}\n\n")
                        f.write(search_content)

                    self.logger.info(f"✅ Created SERP search file: {session_file}")

                    # Save search results to KEVIN directory
                    with open(kevin_file, 'w', encoding='utf-8') as f:
                        f.write(f"SERP Search Results\n")
                        f.write(f"Session ID: {session_id}\n")
                        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                        f.write(f"{'='*50}\n\n")
                        f.write(search_content)

                    self.logger.info(f"✅ Created SERP search file in KEVIN: {kevin_file}")

                    # Store in session data for later access
                    if session_id in self.active_sessions:
                        self.active_sessions[session_id]["serp_search_file"] = str(session_file)
                        self.active_sessions[session_id]["serp_search_kevin_file"] = str(kevin_file)

                    return str(session_file)
                else:
                    self.logger.warning(f"🔧 No search content found in tool result")
            else:
                self.logger.warning(f"🔧 Unexpected tool result format: {type(tool_result)}")

        except Exception as e:
            self.logger.error(f"❌ Error creating SERP search files: {e}")
            import traceback
            self.logger.error(f"❌ Traceback: {traceback.format_exc()}")

    def get_hook_statistics(self) -> Dict[str, Any]:
        """Get comprehensive hook system statistics and performance metrics."""
        if False:  # Hook system disabled
            return {"message": "Hook integration manager not available"}

        try:
            stats = self.hook_integration_manager.get_hook_statistics()
            self.logger.info("Hook statistics retrieved",
                            total_hooks=stats.get("integration_stats", {}).get("total_executions", 0))
            return stats
        except Exception as e:
            self.logger.error(f"Failed to get hook statistics: {str(e)}",
                            error=str(e),
                            error_type=type(e).__name__)
            return {"error": str(e), "error_type": type(e).__name__}

    async def execute_hooks(
        self,
        hook_type: str,
        metadata: Dict[str, Any],
        session_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute hooks of a specific type with given context.

        Args:
            hook_type: Type of hooks to execute
            metadata: Additional metadata for hook execution
            session_id: Session identifier (uses current session if not provided)
            agent_name: Agent name executing the hooks
            category: Optional hook category to limit execution

        Returns:
            List of hook execution results
        """
        if False:  # Hook system disabled
            self.logger.warning("Hook integration manager not available for hook execution")
            return []

        try:
            # Create hook context
            from ..hooks.base_hooks import HookContext
            context = HookContext(
                hook_name=f"orchestrator_{hook_type}",
                hook_type=hook_type,
                session_id=session_id or next(iter(self.active_sessions.keys()), "default"),
                agent_name=agent_name,
                metadata=metadata
            )

            # Hook system disabled - no hook execution

            # Hook system disabled - no results to process
            serializable_results = []

            self.logger.info(f"Hook execution skipped: {hook_type} - hook system disabled")

            return serializable_results

        except Exception as e:
            self.logger.error(f"Hook execution failed: {str(e)}",
                            hook_type=hook_type,
                            error=str(e),
                            error_type=type(e).__name__)
            return [{"error": str(e), "error_type": type(e).__name__}]

    async def cleanup(self):
        """Cleanup all agent clients and resources."""
        self.logger.info("Starting orchestrator cleanup")

        # Shutdown hook integration manager
        if hasattr(self, 'hook_integration_manager') and self.hook_integration_manager:
            try:
                await self.hook_integration_manager.shutdown()
                self.logger.info("HookIntegrationManager shutdown completed")
            except Exception as e:
                self.logger.error(f"HookIntegrationManager shutdown failed: {str(e)}",
                                error=str(e),
                                error_type=type(e).__name__)

        # Cleanup single client
        if hasattr(self, 'client') and self.client:
            try:
                await self.client.disconnect()
                self.logger.info("✅ Single client disconnected")
            except Exception as e:
                self.logger.warning(f"Failed to disconnect client: {str(e)}")
        self.active_sessions.clear()
        self.logger.info("Orchestrator cleanup completed")
