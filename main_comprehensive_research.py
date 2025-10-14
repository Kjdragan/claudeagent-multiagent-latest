#!/usr/bin/env python3
"""
Agent-Based Comprehensive Research System - Main Entry Point

This script provides the main entry point for the comprehensive research system that uses
Claude Agent SDK to orchestrate research workflows with access to advanced web scraping
and content analysis tools.

Key Features:
- Claude Agent SDK integration for agent-based research orchestration
- Access to comprehensive research tools (50+ URLs, concurrent cleaning)
- KEVIN directory structure integration for organized session management
- Real-time progress tracking and monitoring
- Quality assessment and enhancement workflows

Usage:
    python main_comprehensive_research.py "your research query here"
    python main_comprehensive_research.py "climate change impacts" --mode academic --num-results 30
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Add the multi_agent_research_system to the path
sys.path.insert(0, str(Path(__file__).parent / "multi_agent_research_system"))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️  python-dotenv not available, using environment variables only")

# Set up API configuration
ANTHROPIC_BASE_URL = os.getenv('ANTHROPIC_BASE_URL', 'https://api.anthropic.com')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

if ANTHROPIC_API_KEY:
    os.environ['ANTHROPIC_BASE_URL'] = ANTHROPIC_BASE_URL
    os.environ['ANTHROPIC_API_KEY'] = ANTHROPIC_API_KEY
    print(f"✅ Using Anthropic API: {ANTHROPIC_BASE_URL}")
else:
    print("❌ No ANTHROPIC_API_KEY found in environment or .env file")
    print("Please set ANTHROPIC_API_KEY in your environment variables")
    sys.exit(1)

# Try to import Claude Agent SDK
try:
    from claude_agent_sdk import (
        AgentDefinition,
        ClaudeAgentOptions,
        ClaudeSDKClient,
        create_sdk_mcp_server,
    )
    CLAUDE_SDK_AVAILABLE = True
    print("✅ Claude Agent SDK imported successfully")
except ImportError as e:
    print(f"❌ Claude Agent SDK not found: {e}")
    print("Please install the package: pip install claude-agent-sdk")
    sys.exit(1)

# Import system components
try:
    # Import Claude SDK components for proper tool integration
    from integration.agent_session_manager import AgentSessionManager
    from integration.query_processor import QueryProcessor
    from integration.research_orchestrator import ResearchOrchestrator
    from core.logging_config import get_logger
    SYSTEM_COMPONENTS_AVAILABLE = True
    print("✅ System components imported successfully")
except ImportError as e:
    print(f"⚠️  System components not fully available: {e}")
    print("Will use fallback implementations where needed")
    SYSTEM_COMPONENTS_AVAILABLE = False


class ComprehensiveResearchCLI:
    """Main CLI class for the comprehensive research system."""

    def __init__(self):
        self.client: Optional[ClaudeSDKClient] = None
        self.session_manager: Optional[Any] = None
        self.query_processor: Optional[Any] = None
        self.orchestrator: Optional[Any] = None
        self.logger = None
        self.session_id: Optional[str] = None
        self.start_time: Optional[datetime] = None

    def setup_logging(self, log_level: str = "INFO", log_file: Optional[str] = None):
        """Setup comprehensive logging system."""

        # Configure logging
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            force=True
        )

        self.logger = logging.getLogger("comprehensive_research")

        # Add file handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(log_format))
            self.logger.addHandler(file_handler)

        self.logger.info("🚀 Comprehensive Research System initialized")
        self.logger.info(f"📝 Log level: {log_level}")
        if log_file:
            self.logger.info(f"📄 Log file: {log_file}")

    def print_banner(self):
        """Print welcome banner."""
        banner = """
╔════════════════════════════════════════════════════════════════╗
║     Agent-Based Comprehensive Research System (Phase 1.1)              ║
║              Claude Agent SDK + Advanced Research Tools              ║
╚════════════════════════════════════════════════════════════════╝
        """
        print(banner)

    async def initialize_sdk_client(self):
        """Initialize Claude SDK client with comprehensive research capabilities."""
        self.logger.info("🔧 Initializing Claude SDK Client...")

        try:
            # Import required SDK components
            from claude_agent_sdk import create_sdk_mcp_server, ClaudeAgentOptions, ClaudeSDKClient
            from claude_agent_sdk import tool
            import sys
            import os

            # Add the project root to Python path to import tools
            project_root = Path(__file__).parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))

            # Import the zplayground1 search tool
            try:
                from multi_agent_research_system.mcp_tools.zplayground1_search import zplayground1_search_scrape_clean
                self.logger.info("✅ Imported zplayground1_search_scrape_clean tool")
            except ImportError as e:
                self.logger.error(f"❌ Failed to import zplayground1 tool: {e}")
                raise

            # Create SDK MCP server with the search tool
            search_server = create_sdk_mcp_server(
                name="search",
                version="1.0.0",
                tools=[zplayground1_search_scrape_clean]
            )
            self.logger.info("✅ Created search MCP server")

            # Configure agent options
            options = ClaudeAgentOptions(
                mcp_servers={"search": search_server},
                allowed_tools=["mcp__search__zplayground1_search_scrape_clean"],
                max_turns=50,
                continue_conversation=True
            )
            self.logger.info("✅ Configured Claude agent options")

            # Create client
            self.client = ClaudeSDKClient(options=options)
            self.logger.info("✅ Claude SDK client created with search tool")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize SDK client: {e}")
            raise

    async def initialize_system_components(self):
        """Initialize system components with fallback implementations."""
        self.logger.info("🔧 Initializing system components...")

        try:
            # Initialize session manager
            if SYSTEM_COMPONENTS_AVAILABLE:
                self.session_manager = AgentSessionManager()
                self.logger.info("✅ Agent session manager initialized")
            else:
                self.session_manager = FallbackSessionManager()
                self.logger.info("⚠️  Using fallback session manager")

            # Initialize query processor
            if SYSTEM_COMPONENTS_AVAILABLE:
                self.query_processor = QueryProcessor()
                self.logger.info("✅ Query processor initialized")
            else:
                self.query_processor = FallbackQueryProcessor()
                self.logger.info("⚠️  Using fallback query processor")

            # Initialize orchestrator
            if SYSTEM_COMPONENTS_AVAILABLE:
                self.orchestrator = ResearchOrchestrator()
                self.logger.info("✅ Research orchestrator initialized")
            else:
                self.orchestrator = FallbackOrchestrator()
                self.logger.info("⚠️  Using fallback orchestrator")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize system components: {e}")
            raise

    async def create_session(self, query: str, user_requirements: Dict[str, Any]) -> str:
        """Create a new research session."""
        self.logger.info(f"🆔 Creating research session for: {query[:100]}...")

        try:
            if hasattr(self.session_manager, 'create_session'):
                self.session_id = await self.session_manager.create_session(
                    topic=query,
                    user_requirements=user_requirements
                )
            else:
                # Generate session ID
                import uuid
                self.session_id = str(uuid.uuid4())
                self.logger.info(f"🆔 Generated session ID: {self.session_id}")

            self.logger.info(f"✅ Session created: {self.session_id}")
            return self.session_id

        except Exception as e:
            self.logger.error(f"❌ Failed to create session: {e}")
            raise

    async def process_query(self, query: str, mode: str, num_results: int,
                          user_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Process research query through the agent-based system."""
        self.start_time = datetime.now()
        self.logger.info(f"🔍 Processing query: {query}")
        self.logger.info(f"📊 Mode: {mode}, Target Results: {num_results}")

        try:
            # Create session
            session_id = await self.create_session(query, user_requirements)

            # Prepare research prompt for agent
            research_prompt = f"""
Conduct comprehensive research on: {query}

Research Parameters:
- Mode: {mode}
- Target Results: {num_results}
- Session ID: {session_id}

Instructions:
1. Use the comprehensive research tool to gather extensive data from multiple sources
2. Target {num_results} successful research results
3. Process URLs with concurrent scraping and AI content cleaning
4. Apply progressive anti-bot detection as needed
5. Generate a comprehensive research report with proper citations
6. Ensure all content is properly organized and analyzed

Available Tools:
- zplayground1_search_scrape_clean: Comprehensive web scraping with concurrent processing
- KEVIN session management for organized data storage

Please provide:
- Executive summary of findings
- Detailed analysis with multiple perspectives
- Key insights and implications
- Proper source attribution
- Quality assessment of findings

Session Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

            self.logger.info("🚀 Sending query to agent...")
            start_time = time.time()

            # Use proper Claude SDK pattern with async context manager
            async with self.client as client:
                await client.query(research_prompt)

                # Collect response from agent
                response_parts = []
                async for message in client.receive_response():
                    if hasattr(message, 'content'):
                        for block in message.content:
                            if hasattr(block, 'text'):
                                response_parts.append(block.text)
                                self.logger.info(f"📝 Received: {block.text[:100]}...")

                response = "\n".join(response_parts)

            processing_time = time.time() - start_time
            self.logger.info(f"✅ Query processing completed in {processing_time:.2f} seconds")

            # Calculate total session time
            total_time = (datetime.now() - self.start_time).total_seconds()
            self.logger.info(f"🏁 Total session time: {total_time:.2f} seconds")

            # Prepare result
            result = {
                "session_id": session_id,
                "query": query,
                "mode": mode,
                "target_results": num_results,
                "processing_time": processing_time,
                "total_time": total_time,
                "status": "completed",
                "response": response,
                "user_requirements": user_requirements
            }

            return result

        except Exception as e:
            self.logger.error(f"❌ Query processing failed: {e}")
            raise

    async def process_query_fallback(self, query: str, mode: str, num_results: int,
                                    session_id: str) -> str:
        """Fallback query processing when SDK methods are not available."""
        self.logger.warning("⚠️  Using fallback query processing")

        return f"""
# Comprehensive Research Results: {query}

**Session ID**: {session_id}
**Mode**: {mode}
**Target Results**: {num_results}
**Processing Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

The comprehensive research system has processed your query "{query}" using advanced web scraping and content analysis tools.

## Research Approach

The system utilized:
- Advanced web scraping with concurrent processing
- Progressive anti-bot detection (4-level escalation)
- AI-powered content cleaning and analysis
- KEVIN session management for organized storage
- Quality assessment and enhancement workflows

## Key Findings

This is a fallback response as the full agent-based processing encountered issues. The comprehensive research system is operational and ready for enhanced query processing.

## Next Steps

1. Ensure Claude Agent SDK is properly installed and configured
2. Verify API credentials are correctly set
3. Check system component integration status
4. Review error logs for detailed troubleshooting information

**Status**: Fallback Processing Complete
**Recommendation**: Check SDK integration and retry with full agent capabilities.
"""

    def print_results_summary(self, result: Dict[str, Any]):
        """Print summary of research results."""
        print("\n" + "="*60)
        print("🎉 COMPREHENSIVE RESEARCH COMPLETED")
        print("="*60)
        print(f"📝 Query: {result['query']}")
        print(f"🔍 Mode: {result['mode']}")
        print(f"🎯 Target Results: {result['target_results']}")
        print(f"⏱️  Processing Time: {result['processing_time']:.2f} seconds")
        print(f"🕐️ Total Session Time: {result['total_time']:.2f} seconds")
        print(f"🆔 Session ID: {result['session_id']}")
        print(f"✅ Status: {result['status'].upper()}")
        print("="*60)


class FallbackSessionManager:
    """Fallback session manager for when system components are not available."""

    async def create_session(self, topic: str, user_requirements: Dict[str, Any]) -> str:
        """Create fallback session."""
        import uuid
        return str(uuid.uuid4())


class FallbackQueryProcessor:
    """Fallback query processor for when system components are not available."""

    def analyze_query(self, query: str, mode: str) -> Dict[str, Any]:
        """Fallback query analysis."""
        return {
            "original_query": query,
            "mode": mode,
            "optimized_query": query,
            "research_strategy": "comprehensive",
            "estimated_complexity": "medium"
        }


class FallbackOrchestrator:
    """Fallback orchestrator for when system components are not available."""

    def __init__(self):
        self.logger = logging.getLogger("fallback_orchestrator")

    async def execute_comprehensive_research(self, query: str, mode: str,
                                           session_id: str, **kwargs) -> Dict[str, Any]:
        """Fallback orchestrator implementation."""
        self.logger.info("Using fallback orchestrator")
        return {
            "session_id": session_id,
            "query": query,
            "mode": mode,
            "status": "fallback_processing",
            "message": "Full comprehensive research not available in fallback mode"
        }


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Agent-Based Comprehensive Research System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_comprehensive_research.py "climate change impacts"
  python main_comprehensive_research.py "latest AI developments" --mode academic --num-results 30
  python main_comprehensive_research.py "quantum computing applications" --mode news --session-id custom-session
        """
    )

    # Required arguments
    parser.add_argument(
        "query",
        help="Research query to investigate (required)"
    )

    # Optional arguments
    parser.add_argument(
        "--mode",
        choices=["web", "news", "academic"],
        default="web",
        help="Research mode (default: web)"
    )

    parser.add_argument(
        "--num-results",
        type=int,
        default=50,
        help="Number of successful results to target (default: 50)"
    )

    parser.add_argument(
        "--session-id",
        help="Specific session ID to use (optional, auto-generated if not provided)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging"
    )

    parser.add_argument(
        "--log-file",
        help="Log file path (optional)"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode - process query without executing full research"
    )

    return parser


def create_user_requirements(args: argparse.Namespace) -> Dict[str, Any]:
    """Create user requirements dictionary from command line arguments."""
    return {
        "depth": "Comprehensive Analysis",
        "audience": "General",
        "format": "Detailed Report",
        "mode": args.mode,
        "target_results": args.num_results,
        "session_id": args.session_id,
        "debug_mode": args.debug,
        "dry_run": args.dry_run
    }


async def main():
    """Main function - entry point for the comprehensive research system."""

    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Initialize CLI
    cli = ComprehensiveResearchCLI()

    try:
        # Print banner
        cli.print_banner()

        # Setup logging
        cli.setup_logging(args.log_level, args.log_file)

        # Log startup information
        cli.logger.info(f"🔍 Research Query: {args.query}")
        cli.logger.info(f"📊 Mode: {args.mode}")
        cli.logger.info(f"🎯 Target Results: {args.num_results}")
        cli.logger.info(f"🆔 Session ID: {args.session_id or 'auto-generated'}")
        cli.logger.info(f"🐛 Debug Mode: {args.debug}")
        cli.logger.info(f"📄 Log File: {args.log_file or 'console only'}")

        if args.dry_run:
            cli.logger.info("🔍 DRY RUN MODE - Processing query without full execution")
            cli.logger.info("✅ Dry run completed successfully")
            return

        # Create user requirements
        user_requirements = create_user_requirements(args)

        # Initialize SDK client
        await cli.initialize_sdk_client()

        # Initialize system components
        await cli.initialize_system_components()

        # Process query
        result = await cli.process_query(
            query=args.query,
            mode=args.mode,
            num_results=args.num_results,
            user_requirements=user_requirements
        )

        # Print results summary
        cli.print_results_summary(result)

        cli.logger.info("🎉 Comprehensive research session completed successfully")

    except KeyboardInterrupt:
        cli.logger.info("\n⚠️  Research interrupted by user")
        sys.exit(1)
    except Exception as e:
        cli.logger.error(f"\n❌ Research failed: {str(e)}")
        if args.debug:
            import traceback
            cli.logger.error(f"🔍 Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())