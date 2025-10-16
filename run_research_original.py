#!/usr/bin/env python3
"""
Unified Research CLI - Enhanced terminal output with real-time visibility

This script provides a single command to run research with full terminal visibility
into the research process, organized output in KEVIN directory, and proper logging.

Usage:
    python run_research.py "your research topic" [--options]
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# CRITICAL: Load environment variables from .env file BEFORE importing modules
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️  python-dotenv not available, using environment variables only")

# Add the multi_agent_research_system to the path
sys.path.insert(0, str(Path(__file__).parent / "multi_agent_research_system"))

from core.logging_config import setup_logging, get_logger
from core.orchestrator import ResearchOrchestrator

class ResearchCLI:
    """Enhanced CLI with real-time visibility and organized output."""

    def __init__(self):
        self.logger = None
        self.orchestrator = None
        self.session_id = None
        self.start_time = None

    def setup_environment(self):
        """Setup the research environment with proper logging."""
        print("🚀 Initializing Multi-Agent Research System...")

        # Setup logging with KEVIN directory
        self.logger = setup_logging("INFO", "KEVIN/logs")
        self.logger = get_logger("cli")

        print("✅ Logging initialized - logs will be saved to KEVIN/logs/")
        print("✅ Research outputs will be organized in KEVIN/sessions/")
        print("✅ Final reports will be saved in session working directories")
        print()

    def print_banner(self):
        """Print a welcome banner."""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                 Multi-Agent Research System                    ║
║                   Enhanced CLI with Visibility                 ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)

    def print_session_info(self, session_id, topic):
        """Print session information."""
        self.session_id = session_id
        print(f"🔬 Research Session Started")
        print(f"   Session ID: {session_id}")
        print(f"   Topic: {topic}")
        print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("📁 Output Structure:")
        print(f"   KEVIN/sessions/{session_id}/")
        print(f"   ├── research/    (Raw search results)")
        print(f"   ├── working/     (Work-in-progress)")
        print(f"   └── final/       (Final reports)")
        print()

    def print_progress_update(self, stage, message):
        """Print progress updates with timestamp."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"⏱️  [{timestamp}] {stage}: {message}")

    async def run_research(self, topic, requirements=None, debug_agents=False, quick_start=False):
        """Run the research with full visibility."""
        self.start_time = datetime.now()

        try:
            # Initialize orchestrator
            if quick_start:
                self.print_progress_update("QUICK", "Quick start mode - minimal initialization")
                self.orchestrator = ResearchOrchestrator(debug_mode=debug_agents)
                await self.orchestrator.initialize()
                self.print_progress_update("QUICK", "Fast initialization complete")
            else:
                self.print_progress_update("INIT", "Creating research orchestrator")
                self.orchestrator = ResearchOrchestrator(debug_mode=debug_agents)

                if debug_agents:
                    self.print_progress_update("DEBUG", "Agent debugging enabled - full stderr capture active")

                # Initialize agent clients
                self.print_progress_update("AGENTS", "Initializing research agents...")
                await self.orchestrator.initialize()
                self.print_progress_update("AGENTS", f"Successfully initialized {len(self.orchestrator.agent_clients)} agents")

            # Start research session
            self.print_progress_update("START", f"Beginning research on: {topic}")
            session_id = await self.orchestrator.start_research_session(
                topic=topic,
                user_requirements=requirements if isinstance(requirements, dict) else {
                    "depth": "Comprehensive Research",
                    "audience": "General",
                    "format": "Standard Report",
                    "timeline": "ASAP",
                    "original_string_requirement": requirements if isinstance(requirements, str) else "None provided"
                }
            )

            self.print_session_info(session_id, topic)

            # Monitor progress
            await self.monitor_progress(session_id)

            # Get final results
            await self.get_final_results(session_id)

        except Exception as e:
            self.print_progress_update("ERROR", f"Research failed: {str(e)}")
            if self.logger:
                self.logger.error(f"Research failed: {str(e)}")
            raise

    async def monitor_progress(self, session_id):
        """Monitor research progress with real-time updates."""
        self.print_progress_update("MONITOR", "Monitoring research progress...")

        last_status = None
        no_change_count = 0

        while True:
            try:
                session_data = self.orchestrator.active_sessions.get(session_id)
                if not session_data:
                    self.print_progress_update("WARNING", "Session data not found")
                    break

                current_status = session_data.get("status", "unknown")
                status_message = session_data.get("status_message", "")

                if current_status != last_status:
                    self.print_progress_update("STATUS", f"Status changed to: {current_status}")
                    if status_message:
                        self.print_progress_update("DETAIL", status_message)
                    last_status = current_status
                    no_change_count = 0
                else:
                    no_change_count += 1

                # Check if research is complete
                if current_status in ["completed", "failed", "error"]:
                    self.print_progress_update("COMPLETE", f"Research finished with status: {current_status}")
                    break

                # Print periodic updates
                if no_change_count % 10 == 0:  # Every 10 checks
                    elapsed = datetime.now() - self.start_time
                    self.print_progress_update("PROGRESS",
                        f"Still working... ({elapsed.seconds}s elapsed, current status: {current_status})")

                await asyncio.sleep(2)  # Check every 2 seconds

            except Exception as e:
                self.print_progress_update("ERROR", f"Monitoring error: {str(e)}")
                await asyncio.sleep(5)

    async def get_final_results(self, session_id):
        """Get and display final research results."""
        self.print_progress_update("RESULTS", "Collecting final research results...")

        try:
            # Get session data
            session_data = self.orchestrator.active_sessions.get(session_id)
            if not session_data:
                self.print_progress_update("ERROR", "No session data found")
                return

            # Find reports
            session_path = Path(f"KEVIN/sessions/{session_id}")
            final_reports = list((session_path / "final").glob("*.md"))

            if final_reports:
                self.print_progress_update("SUCCESS", f"Found {len(final_reports)} report(s)")
                print()
                print("📊 Research Results Summary:")
                print("=" * 50)

                for report_file in final_reports:
                    print(f"📄 Report: {report_file.name}")
                    print(f"   Location: {report_file}")

                    # Read and display first few lines
                    try:
                        with open(report_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            lines = content.split('\n')[:10]  # First 10 lines
                            print("   Preview:")
                            for line in lines:
                                if line.strip():
                                    print(f"     {line}")
                            print(f"     ... ({len(content.split('\n'))} total lines)")
                    except Exception as e:
                        print(f"     Error reading preview: {str(e)}")
                    print()

                # Final reports are now kept in session working directory for consistency
                print("📁 Final reports saved in session working directory for easy access")

            else:
                self.print_progress_update("WARNING", "No final reports found")
                # Show what files do exist
                if session_path.exists():
                    all_files = []
                    for root, dirs, files in os.walk(session_path):
                        for file in files:
                            rel_path = Path(root) / file
                            all_files.append(str(rel_path))

                    if all_files:
                        print("📂 Available files in session:")
                        for file_path in all_files:
                            print(f"   {file_path}")

        except Exception as e:
            self.print_progress_update("ERROR", f"Error getting results: {str(e)}")

        # Print summary
        elapsed = datetime.now() - self.start_time
        print()
        print("🏁 Research Session Complete")
        print("=" * 50)
        print(f"⏱️  Total time: {elapsed}")
        print(f"🆔 Session ID: {session_id}")
        print(f"📁 Session directory: KEVIN/sessions/{session_id}/")
        print(f"📝 Logs: KEVIN/logs/")
        print(f"📊 Final reports: KEVIN/sessions/{session_id}/working/")
        print()

def main():
    """Main entry point for the enhanced CLI."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Research System with Enhanced Visibility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_research.py "latest AI developments"
  python run_research.py "climate change impact" --requirements "focus on economic aspects"
  python run_research.py "space exploration" --debug-agents
  python run_research.py "US foreign policy" --verify-startup    # Check system health first
  python run_research.py "quick research" --quick-start        # Fastest startup
        """
    )

    parser.add_argument(
        "topic",
        help="Research topic to investigate"
    )

    parser.add_argument(
        "--requirements", "-r",
        default="Comprehensive research with web search",
        help="Research requirements and constraints (default: comprehensive research)"
    )

    parser.add_argument(
        "--debug-agents", "-d",
        action="store_true",
        help="Enable agent debugging with stderr capture and tool tracing"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )

    parser.add_argument(
        "--verify-startup",
        action="store_true",
        help="Run startup health tests before starting research"
    )

    parser.add_argument(
        "--quick-start",
        action="store_true",
        help="Skip all non-essential initialization for fastest startup"
    )

    args = parser.parse_args()

    # Initialize and run CLI
    cli = ResearchCLI()

    try:
        cli.print_banner()
        cli.setup_environment()

        # Handle startup verification if requested
        if args.verify_startup:
            print("🔍 Running startup health tests...")
            try:
                from multi_agent_research_system.tests.test_startup_health import StartupHealthTester
                tester = StartupHealthTester(verbose=args.log_level == "DEBUG")
                health_report = asyncio.run(tester.run_all_tests())
                tester.print_report(health_report)

                if health_report["overall_status"] != "healthy":
                    print("\n⚠️  Health issues found. Use --quick-start to skip tests or fix issues before continuing.")
                    if health_report["overall_status"] == "initialization_failed":
                        sys.exit(1)

                    response = input("\nContinue with research despite health issues? (y/N): ")
                    if response.lower() != 'y':
                        print("Research cancelled.")
                        sys.exit(0)
                else:
                    print("\n✅ System health verified, starting research...")
            except ImportError as e:
                print(f"⚠️  Could not run health tests: {e}")
                print("Continuing with research...")

        # Run research
        asyncio.run(cli.run_research(
            topic=args.topic,
            requirements=args.requirements,
            debug_agents=args.debug_agents,
            quick_start=args.quick_start
        ))

    except KeyboardInterrupt:
        print("\n\n⚠️  Research interrupted by user")
        print("💾 Partial results may be available in the session directory")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Research failed: {str(e)}")
        print("📝 Check logs in KEVIN/logs/ for more details")
        sys.exit(1)

if __name__ == "__main__":
    main()