#!/usr/bin/env python3
"""
Research System Entry Point - Working Pattern
Based on successful older repository implementation

Usage:
    python run_research.py "research topic" [requirements]
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# CRITICAL: Load environment variables from .env file BEFORE importing modules
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️  python-dotenv not available, using environment variables only")

# Add system path
sys.path.insert(0, str(Path(__file__).parent / "multi_agent_research_system"))

from core.orchestrator import ResearchOrchestrator

class ResearchCLI:
    def __init__(self):
        self.orchestrator = None

    async def run_research(self, topic: str, requirements: str = "comprehensive research"):
        """Run research using working pattern"""

        print(f"🔍 Starting research: {topic}")

        # Initialize orchestrator
        self.orchestrator = ResearchOrchestrator()
        await self.orchestrator.initialize()

        # Start research session
        session_id = await self.orchestrator.start_research_session(
            topic=topic,
            user_requirements={
                "depth": "Comprehensive Research",
                "audience": "General",
                "format": "Standard Report",
                "original_string_requirement": requirements
            }
        )

        print(f"🆔 Session started: {session_id}")

        # Monitor progress
        await self.monitor_progress(session_id)

        # Get final results
        await self.get_final_results(session_id)

        print(f"✅ Research completed for session: {session_id}")

    async def monitor_progress(self, session_id: str):
        """Monitor research progress"""

        while True:
            session_data = self.orchestrator.active_sessions.get(session_id)
            if not session_data:
                break

            status = session_data.get("status", "unknown")

            if status == "completed":
                print(f"✅ Research completed successfully")
                break
            elif status == "failed":
                print(f"❌ Research failed: {session_data.get('error', 'Unknown error')}")
                break

            await asyncio.sleep(5)  # Check every 5 seconds

    async def get_final_results(self, session_id: str):
        """Get and display final results"""

        # Find work products
        work_products = list(Path(f"KEVIN/sessions/{session_id}/research").glob("search_workproduct_*.md"))

        if work_products:
            print(f"📄 Found {len(work_products)} work product(s):")
            for wp in work_products:
                print(f"   - {wp.name}")
        else:
            print("⚠️  No work products found")

async def main():
    if len(sys.argv) < 2:
        print("Usage: python run_research.py 'research topic' [requirements]")
        sys.exit(1)

    topic = sys.argv[1]
    requirements = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "comprehensive research"

    cli = ResearchCLI()
    await cli.run_research(topic, requirements)

if __name__ == "__main__":
    asyncio.run(main())