#!/usr/bin/env python3
"""
Enhanced Research CLI - Uses the new enhanced orchestrator with all our improvements

This script uses the enhanced orchestrator with:
- Phase 3.3 Gap Research Enforcement System
- Phase 3.4 Quality Assurance Framework
- Phase 3.5 KEVIN Session Management
- New scraping and cleaning system
- Enhanced editorial workflow

Usage:
    uv run python run_enhanced_research.py "your research topic"
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️  python-dotenv not available, using environment variables only")

# Add the multi_agent_research_system to the path
sys.path.insert(0, str(Path(__file__).parent / "multi_agent_research_system"))

from core.enhanced_orchestrator import EnhancedResearchOrchestrator
from core.kevin_session_manager import KevinSessionManager


async def main():
    """Main enhanced research execution."""
    parser = argparse.ArgumentParser(description="Enhanced Multi-Agent Research System")
    parser.add_argument("topic", help="Research topic to investigate")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    print("🚀 Enhanced Multi-Agent Research System")
    print("=" * 50)
    print(f"🔍 Research Topic: {args.topic}")
    print("📋 Using Enhanced Orchestrator with:")
    print("   • Phase 3.3 Gap Research Enforcement")
    print("   • Phase 3.4 Quality Assurance Framework")
    print("   • Phase 3.5 KEVIN Session Management")
    print("   • New scraping & cleaning system")
    print("")

    try:
        # Initialize enhanced orchestrator
        print("⏱️  Initializing enhanced orchestrator...")
        orchestrator = EnhancedResearchOrchestrator(debug_mode=args.debug)
        await orchestrator.initialize()

        # Initialize KEVIN session manager
        kevin_manager = KevinSessionManager()

        # Create session
        print("🆔 Creating research session...")
        session_id = await kevin_manager.create_session(
            topic=args.topic,
            user_requirements={
                "depth": "Comprehensive Analysis",
                "audience": "General",
                "format": "Detailed Report"
            }
        )

        print(f"✅ Session created: {session_id}")
        print("🔬 Starting enhanced research workflow...")
        print("")

        # Execute enhanced research workflow
        result = await orchestrator.execute_enhanced_research_workflow(session_id)

        print("")
        print("🎉 Research completed!")
        print(f"📁 Results saved in KEVIN/sessions/{session_id}/")
        print(f"📊 Quality Score: {result.get('quality_score', 'N/A')}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())