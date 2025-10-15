#!/usr/bin/env python3
"""Direct test of the intelligent research system without MCP wrapper.

This script directly tests the z-playground1 intelligence functions
without the @tool decorator complications.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

async def test_intelligent_research_directly():
    """Test the intelligent research system directly without MCP wrapper."""
    print("\n" + "=" * 80)
    print("🧪 DIRECT INTELLIGENT RESEARCH TEST")
    print("=" * 80)
    print("Testing z-playground1 intelligence without MCP constraints")

    try:
        # Import the core functions directly
        from utils.serp_search_utils import serp_search_and_extract
        print("✅ serp_search_and_extract imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import serp_search_and_extract: {e}")
        return False

    try:
        # Import crawl4ai utilities to verify they work
        from utils.crawl4ai_utils import crawl_multiple_urls_with_cleaning
        print("✅ crawl4ai_utils imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import crawl4ai_utils: {e}")
        return False

    # Test configuration
    kevin_dir = Path.home() / "lrepos" / "claude-agent-sdk-python" / "KEVIN"
    test_session_id = "direct-test-intelligent-research"
    test_query = "latest military activities on both sides in the Russia Ukraine war"

    print(f"\n🔬 Testing with query: '{test_query}'")
    print(f"📁 Output directory: {kevin_dir}")
    print(f"🆔 Session ID: {test_session_id}")

    try:
        print("\n🚀 Starting intelligent research...")

        # This should use the advanced_content_extraction we integrated
        result = await serp_search_and_extract(
            query=test_query,
            search_type="search",
            num_results=15,  # Use z-playground1 default
            auto_crawl_top=8,   # Use z-playground1 default
            crawl_threshold=0.3,  # Use z-playground1 default
            session_id=test_session_id,
            kevin_dir=kevin_dir
        )

        print("\n📊 Results Analysis:")
        print(f"  - Result length: {len(result)} characters")
        print(f"  - Success: {'✅ Content extracted' if len(result) > 1000 else '❌ No content extracted'}")

        # Check for content extraction indicators
        has_crawled_content = "EXTRACTED CONTENT" in result
        has_work_product = "Work Product Saved" in result
        has_multiple_sources = len(result.split('\n#')) > 5

        print(f"  - Has crawled content: {'✅' if has_crawled_content else '❌'}")
        print(f"  - Has work products: {'✅' if has_work_product else '❌'}")
        print(f"  - Multiple sources: {'✅' if has_multiple_sources else '❌'}")

        # Show first 800 characters
        print("\n📄 Content Preview (first 800 chars):")
        print(result[:800])
        print("...")

        # Check work product directory
        work_product_dir = kevin_dir / "work_products" / test_session_id
        if work_product_dir.exists():
            work_files = list(work_product_dir.glob("*.md"))
            print(f"\n💾 Work Product Files: {len(work_files)}")
            for work_file in work_files[:3]:
                file_size = work_file.stat().st_size
                print(f"  - {work_file.name} ({file_size:,} bytes)")
        else:
            print(f"\n⚠️  No work product directory found at: {work_product_dir}")

        if len(result) > 2000 and has_crawled_content:
            print("\n🎉 SUCCESS: Intelligent research system working correctly!")
            print("✅ Content length indicates successful extraction")
            print("✅ 'EXTRACTED CONTENT' indicates advanced scraping worked")
            print("✅ System ready for agent use")
            return True
        else:
            print("\n⚠️  Limited success detected")
            print("🔍 System may still have issues to resolve")
            return False

    except Exception as e:
        print(f"\n❌ Error in intelligent research: {e}")
        print("\nCommon issues:")
        print("  - Check SERPER_API_KEY in .env file")
        print("  - Check OPENAI_API_KEY in .env file (for content cleaning)")
        print("  - Verify crawl4ai installation")
        return False


async def verify_crawl4ai_availability():
    """Verify Crawl4AI is properly installed."""
    print("\n" + "=" * 80)
    print("🔍 CRAWL4AI AVAILABILITY CHECK")
    print("=" * 80)

    try:
        # Test basic import
        from utils.crawl4ai_utils import SimpleCrawler
        print("✅ SimpleCrawler imported successfully")

        # Check if we can create an instance
        crawler = SimpleCrawler()
        print("✅ SimpleCrawler instance created successfully")

        # Check if we can access browser automation
        try:
            # Just test that the import works, no actual crawling
            from crawl4ai import AsyncWebCrawler
            print("✅ AsyncWebCrawler available")
        except ImportError:
            print("❌ AsyncWebCrawler not available")
            print("   Try: pip install crawl4ai playwright")
            print("   Then: playwright install chromium")
            return False

        print("✅ Crawl4AI is properly installed and ready")
        return True

    except ImportError as e:
        print(f"❌ Crawl4AI not available: {e}")
        print("\nTo install Crawl4AI:")
        print("  pip install crawl4ai playwright")
        print("  playwright install")
        print("  # Then try again")
        return False


async def run_direct_tests():
    """Run all direct tests."""
    print("\n" + "=" * 80)
    print("🧪 DIRECT INTELLIGENT RESEARCH SYSTEM TESTS")
    print("=" * 80)

    # Test 1: Verify Crawl4AI availability
    print("\n" + "=" * 20 + " STEP 1: CRAWL4AI CHECK " + "=" * 20)
    crawl4ai_available = await verify_crawl4ai_availability()

    if not crawl4ai_available:
        print("\n❌ Cannot proceed without Crawl4AI")
        return False

    # Test 2: Test intelligent research
    print("\n" + "=" * 20 + " STEP 2: INTELLIGENT RESEARCH " + "=" * 20)
    research_success = await test_intelligent_research_directly()

    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)

    results = [
        ("Crawl4AI Installation", crawl4ai_available),
        ("Intelligent Research", research_success)
    ]

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ✅ ALL TESTS PASSED!")
        print("✅ Z-Playground1 intelligent research system is working")
        print("✅ Ready to use with agents (when MCP wrapper is resolved)")
        print("✅ Advanced scraping and content cleaning implemented")
        print("=" * 80)
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        print("🔧 Check the failed components before agent integration")
        print("=" * 80)
        return False


if __name__ == "__main__":
    success = asyncio.run(run_direct_tests())
    exit(0 if success else 1)
