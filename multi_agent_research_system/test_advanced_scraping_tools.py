"""Test advanced scraping tools directly."""

import asyncio
from pathlib import Path


async def test_single_url_scraping():
    """Test advanced_scrape_url tool."""
    print("\n" + "=" * 80)
    print("TEST 1: SINGLE URL SCRAPING")
    print("=" * 80)

    from tools.advanced_scraping_tool import advanced_scrape_url

    # Test URL - Anthropic documentation
    args = {
        "url": "https://docs.anthropic.com/en/home",
        "session_id": "test-scraping",
        "search_query": "Claude API",
        "preserve_technical": True
    }

    print(f"\n📡 Testing URL: {args['url']}")
    print(f"🔍 Search Query: {args['search_query']}")

    result = await advanced_scrape_url(args)

    content_text = result['content'][0]['text']
    print("\n📊 Result:")
    print(f"  - Length: {len(content_text)} characters")
    print(f"  - Success: {'✅' if 'Success' in content_text else '❌'}")

    # Show preview
    print("\n📄 Content Preview (first 500 chars):")
    print(content_text[:500])
    print("...")

    # Verify success
    assert "Status" in content_text, "Should contain status information"
    assert len(content_text) > 1000, "Should extract substantial content"

    print("\n✅ Single URL scraping test PASSED")
    return True


async def test_serp_search_with_advanced_extraction():
    """Test SERP search with advanced extraction backend."""
    print("\n" + "=" * 80)
    print("TEST 2: SERP SEARCH WITH ADVANCED EXTRACTION")
    print("=" * 80)

    from utils.serp_search_utils import serp_search_and_extract

    kevin_dir = Path.home() / "lrepos" / "claude-agent-sdk-python" / "KEVIN"

    print("\n🔍 Searching for: 'Claude Agent SDK'")
    print(f"📁 Saving work products to: {kevin_dir}")

    result = await serp_search_and_extract(
        query="Claude Agent SDK",
        search_type="search",
        num_results=5,
        auto_crawl_top=2,  # Only crawl top 2 to save time in test
        crawl_threshold=0.3,
        session_id="test-serp-advanced",
        kevin_dir=kevin_dir
    )

    print("\n📊 Result:")
    print(f"  - Length: {len(result)} characters")
    print(f"  - Contains extracted content: {'EXTRACTED CONTENT' in result}")

    # Show preview
    print("\n📄 Content Preview (first 800 chars):")
    print(result[:800])
    print("...")

    # Verify improvement over old 2K limit
    assert len(result) > 5000, f"Should extract much more content than old 2K limit (got {len(result)} chars)"

    print("\n✅ SERP search with advanced extraction test PASSED")
    return True


async def test_import_utilities():
    """Test that advanced scraping utilities are importable."""
    print("\n" + "=" * 80)
    print("TEST 3: IMPORT VERIFICATION")
    print("=" * 80)

    try:
        print("\n📦 Testing imports...")

        from utils.crawl4ai_utils import (
            SimpleCrawler,
            crawl_multiple_urls_with_cleaning,
            scrape_and_clean_single_url_direct,
        )
        print("  ✅ crawl4ai_utils imports successful")

        from utils.content_cleaning import (
            assess_content_cleanliness,
            clean_content_with_gpt5_nano,
            clean_content_with_judge_optimization,
        )
        print("  ✅ content_cleaning imports successful")

        from tools.advanced_scraping_tool import (
            advanced_scrape_multiple_urls,
            advanced_scrape_url,
        )
        print("  ✅ advanced_scraping_tool imports successful")

        print("\n✅ All imports verified successfully")
        return True

    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        raise


async def run_all_tests():
    """Run all scraping tests."""
    print("\n" + "=" * 80)
    print("🧪 ADVANCED SCRAPING INTEGRATION TESTS")
    print("=" * 80 + "\n")

    tests_passed = 0
    tests_failed = 0

    try:
        # Test 1: Import verification
        if await test_import_utilities():
            tests_passed += 1
        else:
            tests_failed += 1
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}")
        tests_failed += 1

    try:
        # Test 2: Single URL scraping
        if await test_single_url_scraping():
            tests_passed += 1
        else:
            tests_failed += 1
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}")
        tests_failed += 1

    try:
        # Test 3: SERP search with advanced extraction
        if await test_serp_search_with_advanced_extraction():
            tests_passed += 1
        else:
            tests_failed += 1
    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}")
        tests_failed += 1

    # Print summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print(f"Tests Passed: {tests_passed}/3")
    print(f"Tests Failed: {tests_failed}/3")

    if tests_failed == 0:
        print("\n🎉 ✅ ALL TESTS PASSED!")
        print("=" * 80)
        return True
    else:
        print(f"\n⚠️  {tests_failed} test(s) failed")
        print("=" * 80)
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
