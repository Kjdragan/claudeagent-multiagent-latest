#!/usr/bin/env python3
"""
Command Line Interface for Multi-Agent Research System

This provides a simple CLI to test the research system without Streamlit dependencies.
Perfect for debugging and testing the core functionality.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not found. Using environment variables only.")

# Set up API configuration
ANTHROPIC_BASE_URL = os.getenv('ANTHROPIC_BASE_URL', 'https://api.anthropic.com')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

if ANTHROPIC_API_KEY:
    os.environ['ANTHROPIC_BASE_URL'] = ANTHROPIC_BASE_URL
    os.environ['ANTHROPIC_API_KEY'] = ANTHROPIC_API_KEY
    print(f"✅ Connected to Anthropic API: {ANTHROPIC_BASE_URL}")
else:
    print("⚠️  No ANTHROPIC_API_KEY found in environment or .env file")
    sys.exit(1)

from core.logging_config import get_log_summary, get_logger, setup_logging
from core.orchestrator import ResearchOrchestrator


class ResearchCLI:
    """Command Line Interface for research system."""

    def __init__(self, debug_mode=True):
        setup_logging()
        self.logger = get_logger("cli")
        self.logger.info("CLI initialized")
        self.orchestrator = ResearchOrchestrator(debug_mode=debug_mode)
        self.current_session = None

    async def initialize(self):
        """Initialize the orchestrator."""
        print("🔧 Initializing research system...")
        try:
            await self.orchestrator.initialize()
            print("✅ System initialized with 4 agents")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize system: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    async def run_research(self, topic: str, requirements: dict = None):
        """Run a research session."""
        if not requirements:
            requirements = {
                "depth": "Standard Research",
                "audience": "General",
                "format": "Standard Report",
                "timeline": "ASAP"
            }

        print(f"🚀 Starting research on: {topic}")
        print(f"📋 Requirements: {json.dumps(requirements, indent=2)}")

        try:
            # Start research session
            session_id = await self.orchestrator.start_research_session(topic, requirements)
            self.current_session = session_id
            print(f"✅ Research session started: {session_id}")

            # Monitor progress
            await self.monitor_session(session_id)

            return session_id

        except Exception as e:
            print(f"❌ Research failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    async def monitor_session(self, session_id: str):
        """Monitor research session progress."""
        print("📊 Monitoring research progress...")

        # Check session status periodically
        max_wait_minutes = 10
        check_interval = 5  # seconds
        elapsed_time = 0

        while elapsed_time < max_wait_minutes * 60:
            session_data = self.orchestrator.active_sessions.get(session_id)
            if not session_data:
                print("❌ Session not found")
                return

            status = session_data.get('status')
            stage = session_data.get('current_stage', 'unknown')

            print(f"📈 Status: {status} | Stage: {stage} | Time: {elapsed_time}s")

            if status == 'completed':
                print("✅ Research completed!")
                await self.show_results(session_id)
                return
            elif status == 'error':
                print(f"❌ Research failed: {session_data.get('error', 'Unknown error')}")
                return

            await asyncio.sleep(check_interval)
            elapsed_time += check_interval

        print(f"⏰ Timeout after {max_wait_minutes} minutes")
        await self.show_partial_results(session_id)

    async def show_results(self, session_id: str):
        """Show research results."""
        print("\n" + "="*60)
        print("📄 RESEARCH RESULTS")
        print("="*60)

        # Show KEVIN directory results
        kevin_dir = Path("/home/kjdragan/lrepos/claude-agent-sdk-python/KEVIN")
        if kevin_dir.exists():
            print(f"\n📁 KEVIN Directory: {kevin_dir}")

            # Show web search results
            web_search_files = sorted(kevin_dir.glob("web_search_results_*.json"),
                                    key=lambda x: x.stat().st_mtime, reverse=True)

            if web_search_files:
                print(f"\n🔍 Found {len(web_search_files)} web search result files:")
                for i, search_file in enumerate(web_search_files[:3]):  # Show last 3
                    try:
                        with open(search_file, encoding='utf-8') as f:
                            data = json.load(f)
                        query = data.get('search_query', 'Unknown')
                        result_length = len(data.get('search_results', ''))
                        print(f"  {i+1}. {query[:60]}... ({result_length} chars)")
                    except Exception as e:
                        print(f"  {i+1}. Error reading {search_file.name}: {e}")
            else:
                print("\n⚠️  No web search result files found")

            # Show reports
            report_files = sorted(kevin_dir.glob("research_report_*.*"),
                                key=lambda x: x.stat().st_mtime, reverse=True)

            if report_files:
                print(f"\n📄 Found {len(report_files)} report files:")
                for i, report_file in enumerate(report_files[:3]):  # Show last 3
                    file_size = report_file.stat().st_size
                    print(f"  {i+1}. {report_file.name} ({file_size} bytes)")
            else:
                print("\n⚠️  No report files found")

        print("\n📋 Full log files:")
        log_summary = get_log_summary()
        if log_summary:
            current_log = log_summary.get('current_log_file')
            if current_log:
                print(f"  Current log: {current_log}")

        print("\n💾 All files saved to KEVIN directory")

    async def show_partial_results(self, session_id: str):
        """Show partial results when timeout occurs."""
        print("\n⚠️  Research incomplete - showing partial results")
        await self.show_results(session_id)

    def show_live_logs(self, lines: int = 50):
        """Show recent log entries."""
        print(f"\n📋 Recent {lines} log lines:")
        print("-" * 60)

        log_summary = get_log_summary()
        if log_summary and log_summary.get('current_log_file'):
            log_file = Path(log_summary['current_log_file'])
            if log_file.exists():
                with open(log_file, encoding='utf-8') as f:
                    all_lines = f.readlines()

                recent_lines = all_lines[-lines:]
                for line in recent_lines:
                    if "🔥" in line or "ERROR" in line or "session" in line:
                        print(line.strip())
                    else:
                        print(line.strip())
            else:
                print("Log file not found")
        else:
            print("No log files available")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Multi-Agent Research System CLI")
    parser.add_argument("topic", nargs="?", help="Research topic")
    parser.add_argument("--debug", action="store_true", default=True, help="Enable debug mode")
    parser.add_argument("--logs", action="store_true", help="Show recent logs only")
    parser.add_argument("--depth", default="Standard Research",
                       choices=["Quick Overview", "Standard Research", "Comprehensive Analysis"],
                       help="Research depth")
    parser.add_argument("--audience", default="General",
                       choices=["General Public", "Academic", "Business", "Technical", "Policy Makers"],
                       help="Target audience")
    parser.add_argument("--format", default="Standard Report",
                       choices=["Standard Report", "Academic Paper", "Business Brief", "Technical Documentation"],
                       help="Report format")

    args = parser.parse_args()

    cli = ResearchCLI(debug_mode=args.debug)

    # If --logs flag, just show logs and exit
    if args.logs:
        cli.show_live_logs(100)
        return

    # If no topic provided, show usage
    if not args.topic:
        print("🔬 Multi-Agent Research System CLI")
        print("="*50)
        print("\nUsage:")
        print(f"  {sys.argv[0]} \"your research topic\"")
        print(f"  {sys.argv[0]} \"latest news about Charlie Kirk\" --depth Standard")
        print(f"  {sys.argv[0]} --logs  # Show recent logs")
        print("\nOptions:")
        print("  --depth       Research depth (default: Standard Research)")
        print("  --audience    Target audience (default: General)")
        print("  --format      Report format (default: Standard Report)")
        print("  --logs        Show recent logs only")
        print("\nExamples:")
        print(f"  {sys.argv[0]} \"Russia Ukraine war latest developments\"")
        print(f"  {sys.argv[0]} \"artificial intelligence regulations 2025\" --depth Comprehensive")
        return

    async def run_research_async():
        """Async research runner."""
        # Initialize system
        if not await cli.initialize():
            return

        # Set up research requirements
        requirements = {
            "depth": args.depth,
            "audience": args.audience,
            "format": args.format,
            "timeline": "ASAP"
        }

        # Run research
        await cli.run_research(args.topic, requirements)

    # Run the research
    asyncio.run(run_research_async())


if __name__ == "__main__":
    main()
