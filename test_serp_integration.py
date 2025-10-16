#!/usr/bin/env python3
"""
Test script to verify SERP API integration is working correctly
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the multi_agent_research_system to the path
sys.path.insert(0, str(Path(__file__).parent / "multi_agent_research_system"))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️  python-dotenv not available, using environment variables only")

async def test_serp_integration():
    """Test SERP API integration using the working pattern"""

    print("🔍 Testing SERP API Integration...")

    # Check environment variables
    serp_api_key = os.getenv('SERP_API_KEY')
    if not serp_api_key:
        print("❌ SERP_API_KEY not found in environment variables")
        return False

    print(f"✅ SERP_API_KEY found: {serp_api_key[:10]}...")

    try:
        # Test SERP API directly using the utils
        from utils.serp_search_utils import execute_serper_search

        # Test search with simple query
        query = "artificial intelligence"
        print(f"🔍 Testing search query: {query}")

        results = await execute_serper_search(
            query=query,
            num_results=5,
            search_type="search"
        )

        if results:
            print(f"✅ SERP API search successful! Found {len(results)} results")
            print(f"📄 First result: {results[0].title if results[0].title else 'No title'}")
            return True
        else:
            print("❌ SERP API search returned no results")
            return False

    except Exception as e:
        print(f"❌ SERP API test failed: {str(e)}")
        return False

async def test_z_search_integration():
    """Test the integrated search and crawl utilities"""

    print("\n🔍 Testing z_search_crawl_utils integration...")

    try:
        from utils.z_search_crawl_utils import search_crawl_and_clean_direct

        query = "quantum computing"
        print(f"🔍 Testing search-crawl-clean pipeline for: {query}")

        # Note: This will attempt crawling which may fail, but search should work
        result = await search_crawl_and_clean_direct(
            query=query,
            search_type="search",
            num_results=3,
            auto_crawl_top=1,  # Minimal crawling for test
            session_id="test_session"
        )

        if result and len(result) > 100:
            print(f"✅ Search-crawl-clean pipeline working! Generated {len(result)} characters")
            return True
        else:
            print("❌ Search-crawl-clean pipeline failed or returned minimal content")
            return False

    except Exception as e:
        print(f"❌ z_search_crawl_utils test failed: {str(e)}")
        return False

async def main():
    """Run all integration tests"""

    print("🚀 Starting SERP API Integration Tests")
    print("=" * 50)

    # Test basic SERP API integration
    serp_success = await test_serp_integration()

    # Test integrated search pipeline
    search_success = await test_z_search_integration()

    print("\n" + "=" * 50)
    print("📊 TEST RESULTS:")
    print(f"   SERP API Integration: {'✅ PASS' if serp_success else '❌ FAIL'}")
    print(f"   Search Pipeline: {'✅ PASS' if search_success else '❌ FAIL'}")

    if serp_success and search_success:
        print("\n🎉 All tests passed! SERP API integration is working correctly.")
        return True
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)