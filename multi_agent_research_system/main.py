#!/usr/bin/env python3
"""Main entry point for the Multi-Agent Research System with CLI support.

This script provides command-line interface for running the system with
configurable logging options.
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.logging_config import get_logger, parse_args_and_setup_logging


async def main():
    """Main entry point with CLI argument parsing."""
    # Parse CLI arguments and setup logging
    args = parse_args_and_setup_logging()
    logger = get_logger("main")

    logger.info("Starting Multi-Agent Research System")
    logger.info(f"Command line arguments: {vars(args)}")

    try:
        # Import after logging is setup
        from core.orchestrator import ResearchOrchestrator

        # Initialize orchestrator with debug mode if requested
        logger.info("Initializing research orchestrator")
        debug_mode = getattr(args, 'debug_agents', False)
        if debug_mode:
            logger.info("Agent debugging mode enabled - using stderr capture and tool tracing")
        orchestrator = ResearchOrchestrator(debug_mode=debug_mode)
        await orchestrator.initialize()

        # Start a demo research session
        logger.info("Starting demo research session")
        session_id = await orchestrator.start_research_session(
            "latest news about Charlie Kirk",
            {
                "depth": "Detailed Investigation",
                "audience": "General",
                "format": "Summary"
            }
        )

        logger.info(f"Research session started: {session_id}")

        # Monitor progress
        max_wait_time = 300  # 5 minutes
        wait_interval = 5
        elapsed_time = 0

        while elapsed_time < max_wait_time:
            status = await orchestrator.get_session_status(session_id)
            logger.debug(f"Session status: {status}")

            if status.get("status") == "completed":
                logger.info("Research session completed successfully")
                break
            elif status.get("status") == "error":
                logger.error(f"Research session failed: {status}")
                break

            await asyncio.sleep(wait_interval)
            elapsed_time += wait_interval

        if elapsed_time >= max_wait_time:
            logger.warning("Research session timed out")

        # Display research results
        logger.info("=== RESEARCH RESULTS ===")

        # Find the most recent research report in KEVIN directory
        kevin_dir = Path("/home/kjdragan/lrepos/claude-agent-sdk-python/KEVIN")
        if kevin_dir.exists():
            report_files = list(kevin_dir.glob("research_report_*.md"))
            if report_files:
                # Get the most recent report
                latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
                logger.info(f"Research Report Generated: {latest_report.name}")
                logger.info(f"Report Location: {latest_report}")

                # Display the first part of the report
                try:
                    with open(latest_report, encoding='utf-8') as f:
                        content = f.read()
                        logger.info(f"Report Length: {len(content)} characters")
                        logger.info("=== REPORT PREVIEW (first 1000 characters) ===")
                        logger.info(content[:1000])
                        if len(content) > 1000:
                            logger.info("... (report continues)")
                except Exception as e:
                    logger.error(f"Error reading report: {e}")
            else:
                logger.warning("No research reports found in KEVIN directory")

        # Also check for research findings
        findings_files = list(kevin_dir.glob("research_findings_*.json")) if kevin_dir.exists() else []
        if findings_files:
            latest_findings = max(findings_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Research Findings Saved: {latest_findings.name}")
            logger.info(f"Findings Location: {latest_findings}")

        # Display debug output if available
        if debug_mode:
            logger.info("=== AGENT DEBUG OUTPUT ===")
            debug_output = orchestrator.get_debug_output()
            if debug_output:
                logger.info(f"Debug Output: {len(debug_output)} lines captured")
                # Show key debug messages
                for line in debug_output:
                    if any(keyword in line for keyword in ["Tool Use Started", "Research content", "Report content", "✅", "🔧"]):
                        logger.info(f"DEBUG: {line}")
            else:
                logger.info("No debug output captured from agents")

        # Cleanup
        await orchestrator.cleanup()
        logger.info("Multi-Agent Research System completed")
        logger.info(f"Research results saved to: {kevin_dir}")

    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    return 0


def run_streamlit_app():
    """Run the Streamlit web interface."""
    logger = get_logger("streamlit")
    logger.info("Starting Streamlit web interface")

    import subprocess
    try:
        # Run streamlit with current python
        result = subprocess.run([
            sys.executable, "-m", "streamlit", "run", "ui/streamlit_app.py"
        ], cwd=Path(__file__).parent)
        return result.returncode
    except Exception as e:
        logger.error(f"Error starting Streamlit: {e}")
        return 1


if __name__ == "__main__":
    # Check if we should run Streamlit instead
    if len(sys.argv) > 1 and sys.argv[1] == "streamlit":
        # Remove 'streamlit' from args for proper parsing
        sys.argv.pop(1)
        exit_code = run_streamlit_app()
    else:
        exit_code = asyncio.run(main())

    sys.exit(exit_code)
