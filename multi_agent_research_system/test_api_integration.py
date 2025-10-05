#!/usr/bin/env python3
"""
Test script for API integration with the real Anthropic endpoint.
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up API configuration
ANTHROPIC_BASE_URL = os.getenv('ANTHROPIC_BASE_URL')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

if ANTHROPIC_API_KEY:
    os.environ['ANTHROPIC_BASE_URL'] = ANTHROPIC_BASE_URL
    os.environ['ANTHROPIC_API_KEY'] = ANTHROPIC_API_KEY
    print(f"✅ Using Anthropic API: {ANTHROPIC_BASE_URL}")
else:
    print("❌ No API key found")
    sys.exit(1)

from core.orchestrator import ResearchOrchestrator

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient


async def test_api_connection():
    """Test basic API connection."""
    print("🔍 Testing API connection...")

    try:
        # Create a simple client
        options = ClaudeAgentOptions(
            model="sonnet",
            max_turns=1
        )

        client = ClaudeSDKClient(options)
        await client.connect()

        # Simple test query
        response = await client.query("Say 'API connection successful!'")
        print(f"✅ API Response: {response}")

        await client.disconnect()
        print("✅ API connection test passed")
        return True

    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return False


async def test_research_system():
    """Test the research system with real API."""
    print("\n🔬 Testing research system...")

    try:
        # Initialize orchestrator
        orchestrator = ResearchOrchestrator()
        await orchestrator.initialize()
        print("✅ Research orchestrator initialized")

        # Start a simple research session
        session_id = await orchestrator.start_research_session(
            "Benefits of Solar Energy",
            {
                "depth": "Quick Overview",
                "audience": "General",
                "format": "Summary"
            }
        )
        print(f"✅ Research session started: {session_id}")

        # Get initial status
        status = await orchestrator.get_session_status(session_id)
        print(f"✅ Session status: {status['status']}")

        # Wait a moment for workflow to start
        await asyncio.sleep(2)

        # Check status again
        final_status = await orchestrator.get_session_status(session_id)
        print(f"✅ Final status: {final_status['status']}")

        await orchestrator.cleanup()
        print("✅ Research system test completed")
        return True

    except Exception as e:
        print(f"❌ Research system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("🚀 Multi-Agent Research System - API Integration Test")
    print("=" * 60)

    # Test basic API connection
    api_test = await test_api_connection()

    if not api_test:
        print("\n❌ Basic API connection failed. Skipping further tests.")
        return False

    # Test research system
    research_test = await test_research_system()

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"API Connection: {'✅ PASSED' if api_test else '❌ FAILED'}")
    print(f"Research System: {'✅ PASSED' if research_test else '❌ FAILED'}")

    if api_test and research_test:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ The system is ready for real research tasks with the Anthropic API")
        print("\n🚀 To run the full web interface:")
        print("   uv run streamlit run ui/streamlit_app.py")
        return True
    else:
        print("\n❌ SOME TESTS FAILED")
        print("🔧 Please check the configuration and API credentials")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
