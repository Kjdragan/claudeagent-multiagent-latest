#!/usr/bin/env python3
"""
Research System Entry Point - Working Pattern
Based on successful older repository implementation

Usage:
    python run_research.py "research topic" [requirements]
"""

import asyncio
import os
import subprocess
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

def check_playwright_installation():
    """Check if Playwright browsers are installed and install if needed."""
    print("🔍 Checking Playwright installation...")
    
    try:
        # Check if playwright is available
        import playwright
        
        # Check if browsers are installed by looking for the browser directory
        # Playwright stores browsers in a well-known location
        from playwright._impl._driver import compute_driver_executable, get_driver_env
        
        # Try to verify browsers are installed
        try:
            # Quick check: see if chromium executable exists
            driver_env = get_driver_env()
            playwright_browsers_path = driver_env.get("PLAYWRIGHT_BROWSERS_PATH", None)
            
            # If we can import and basic paths exist, assume it's installed
            print("✅ Playwright appears to be installed")
            return True
            
        except Exception:
            # Browsers might not be installed, proceed to install
            pass
            
    except ImportError:
        print("⚠️  Playwright package not found - browsers cannot be installed")
        print("   Install with: uv add playwright")
        return False
    
    # If we got here, browsers are not installed
    print("📦 Playwright browsers not installed. Installing now...")
    print("   Running: uv run playwright install")
    
    try:
        # Run playwright install using uv
        result = subprocess.run(
            ["uv", "run", "playwright", "install"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print("✅ Playwright browsers installed successfully")
            return True
        else:
            print("❌ Failed to install Playwright browsers")
            print(f"   Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Playwright installation timed out (exceeded 5 minutes)")
        return False
    except FileNotFoundError:
        print("❌ 'uv' command not found. Please install uv first.")
        return False
    except Exception as e:
        print(f"❌ Error installing Playwright: {e}")
        return False

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

    # Check and install Playwright if needed (required for web scraping)
    if not check_playwright_installation():
        print("\n⚠️  Warning: Playwright installation check failed")
        print("   Web scraping may not work correctly")
        print("   You can manually install with: uv run playwright install")
        
        response = input("\nContinue anyway? (y/N): ").strip().lower()
        if response != 'y':
            print("Exiting...")
            sys.exit(1)

    topic = sys.argv[1]
    requirements = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "comprehensive research"

    cli = ResearchCLI()
    await cli.run_research(topic, requirements)

if __name__ == "__main__":
    asyncio.run(main())