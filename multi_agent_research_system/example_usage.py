#!/usr/bin/env python3
"""
Example usage of the Multi-Agent Research System.

This script demonstrates how to use the system programmatically.
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.append('..')

from multi_agent_research_system import ResearchOrchestrator


async def main():
    """Example usage of the research system."""
    print("🔬 Multi-Agent Research System Example")
    print("=" * 40)

    # Initialize the orchestrator
    orchestrator = ResearchOrchestrator()

    print("📋 Available agents:")
    for agent_name in orchestrator.agent_definitions.keys():
        print(f"  - {agent_name}")

    # Define research requirements
    research_topic = "The Impact of Artificial Intelligence on Healthcare"
    user_requirements = {
        "depth": "Comprehensive Analysis",
        "audience": "Technical",
        "format": "Technical Documentation",
        "requirements": "Focus on recent developments, ethical considerations, and practical applications",
        "timeline": "Within 1 week"
    }

    print(f"\n🎯 Starting research on: {research_topic}")
    print(f"📝 Requirements: {user_requirements['depth']} for {user_requirements['audience']} audience")

    try:
        # Start a research session
        if hasattr(orchestrator, 'start_research_session'):
            session_id = await orchestrator.start_research_session(
                topic=research_topic,
                user_requirements=user_requirements
            )

            print(f"🚀 Research session started! Session ID: {session_id}")

            # Monitor progress (in a real implementation, you'd poll the status)
            print("📊 Monitoring research progress...")
            print("  🔍 Conducting research...")
            print("  📝 Generating report...")
            print("  ✅ Editorial review...")
            print("  🎯 Finalizing report...")

            # Get final status
            status = await orchestrator.get_session_status(session_id)
            print(f"\n✅ Research completed! Status: {status.get('status', 'Unknown')}")

        else:
            print("⚠️  SDK not available - showing workflow structure only")
            print("  🔍 Research phase: Collect and analyze sources")
            print("  📝 Report generation: Create structured report")
            print("  ✅ Editorial review: Quality assessment")
            print("  🎯 Finalization: Complete and deliver report")

    except Exception as e:
        print(f"❌ Error during research: {e}")

    print("\n💡 To run with full SDK functionality:")
    print("  1. Install claude-agent-sdk")
    print("  2. Set up your Claude API credentials")
    print("  3. Run this script again")


if __name__ == "__main__":
    print("Note: This example shows the system structure.")
    print("For full functionality, install the required dependencies.\n")

    # Run the example
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Research session cancelled.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")