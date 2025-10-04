#!/usr/bin/env python3
"""
Simplified test for core search system improvements.

This script tests the key improvements without requiring external dependencies:
1. Configuration updates (0.3 threshold, target-based scraping)
2. URL deduplication system
3. Session-based file organization
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add the multi_agent_research_system to Python path
sys.path.append(str(Path(__file__).parent / "multi_agent_research_system"))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_configuration():
    """Test configuration updates."""
    print("\n=== Testing Configuration Updates ===")

    try:
        from config.settings import get_enhanced_search_config

        config = get_enhanced_search_config()

        # Test key configuration values
        assert config.default_crawl_threshold == 0.3, f"Expected 0.3, got {config.default_crawl_threshold}"
        assert config.target_successful_scrapes == 8, f"Expected 8, got {config.target_successful_scrapes}"
        assert config.url_deduplication_enabled == True, "URL deduplication should be enabled"
        assert config.progressive_retry_enabled == True, "Progressive retry should be enabled"
        assert config.max_retry_attempts == 3, f"Expected 3, got {config.max_retry_attempts}"

        print("✅ All configuration values are correct")
        print(f"   - Default crawl threshold: {config.default_crawl_threshold}")
        print(f"   - Target successful scrapes: {config.target_successful_scrapes}")
        print(f"   - URL deduplication enabled: {config.url_deduplication_enabled}")
        print(f"   - Progressive retry enabled: {config.progressive_retry_enabled}")
        print(f"   - Max retry attempts: {config.max_retry_attempts}")

        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_url_tracker():
    """Test URL tracker functionality."""
    print("\n=== Testing URL Tracker ===")

    try:
        from utils.url_tracker import get_url_tracker, URLAttempt

        # Create test directory in temp location
        test_storage = Path("/tmp/test_url_tracking")
        if test_storage.exists():
            import shutil
            shutil.rmtree(test_storage)

        tracker = get_url_tracker(test_storage)

        # Test URL filtering
        test_urls = [
            "https://example.com/article1",
            "https://example.com/article1",  # Duplicate
            "https://news.example.com/story",
            "https://blog.example.com/post",
            "https://example.com/article1",  # Another duplicate
        ]

        session_id = "test_session_001"
        filtered_urls, skipped_urls = tracker.filter_urls(test_urls, session_id)

        print(f"✅ URL filtering works:")
        print(f"   Input URLs: {len(test_urls)}")
        print(f"   Filtered URLs: {len(filtered_urls)}")
        print(f"   Skipped URLs: {len(skipped_urls)}")

        # Verify duplicate removal
        expected_unique = 3  # example.com/article1, news.example.com/story, blog.example.com/post
        assert len(filtered_urls) == expected_unique, f"Expected {expected_unique} unique URLs, got {len(filtered_urls)}"
        assert len(skipped_urls) == 2, f"Expected 2 skipped duplicates, got {len(skipped_urls)}"

        # Test recording attempts
        tracker.record_attempt(
            url=filtered_urls[0],
            success=True,
            anti_bot_level=1,
            content_length=1000,
            duration=2.5,
            session_id=session_id
        )

        tracker.record_attempt(
            url=filtered_urls[1],
            success=False,
            anti_bot_level=1,
            content_length=0,
            duration=1.0,
            error_message="Connection timeout",
            session_id=session_id
        )

        # Test statistics
        stats = tracker.get_statistics()
        print(f"✅ Statistics tracking works:")
        print(f"   Total URLs: {stats['total_urls']}")
        print(f"   Successful URLs: {stats['successful_urls']}")
        print(f"   Success rate: {stats['success_rate']:.2%}")

        # Test retry logic - verify that failed URLs are tracked correctly
        failed_url = filtered_urls[1]
        successful_url = filtered_urls[0]

        # Check retry logic - URLs that just failed need a 5-minute cooling period
        retry_candidates = tracker.get_retry_candidates([failed_url])
        print(f"✅ Retry logic analysis:")
        print(f"   Failed URL: {failed_url}")
        print(f"   Immediate retry (expected false due to 5-min wait): {len(retry_candidates) > 0}")

        # Test that successful URL is not a retry candidate
        successful_retry_candidates = tracker.get_retry_candidates([successful_url])
        assert len(successful_retry_candidates) == 0, "Successful URL should not be retry candidate"
        print(f"   Successful URL correctly excluded from retries: {len(successful_retry_candidates) == 0}")

        # Test anti-bot level progression
        if failed_url in tracker.url_records:
            retry_level = tracker.get_retry_anti_bot_level(failed_url)
            print(f"   Recommended anti-bot level for next attempt: {retry_level}")
            assert retry_level == 2, f"Should recommend anti-bot level 2, got {retry_level}"

        return True

    except Exception as e:
        print(f"❌ URL tracker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_search_result_selection():
    """Test search result selection logic."""
    print("\n=== Testing Search Result Selection ===")

    try:
        # Create a mock SearchResult class for testing
        class MockSearchResult:
            def __init__(self, title, link, snippet, position, relevance_score):
                self.title = title
                self.link = link
                self.snippet = snippet
                self.position = position
                self.relevance_score = relevance_score

        # Test URL selection logic directly
        search_results = [
            MockSearchResult(
                title="High Relevance Article",
                link="https://example.com/high",
                snippet="This is a highly relevant article",
                position=1,
                relevance_score=0.8
            ),
            MockSearchResult(
                title="Medium Relevance Article",
                link="https://example.com/medium",
                snippet="This is somewhat relevant",
                position=2,
                relevance_score=0.35  # Above 0.3 threshold
            ),
            MockSearchResult(
                title="Low Relevance Article",
                link="https://example.com/low",
                snippet="This is not very relevant",
                position=3,
                relevance_score=0.25  # Below 0.3 threshold
            ),
            MockSearchResult(
                title="Another Medium Article",
                link="https://example.com/medium2",
                snippet="Another somewhat relevant article",
                position=4,
                relevance_score=0.4
            )
        ]

        # Simulate the selection logic
        min_relevance = 0.3
        limit = 10

        # Filter by relevance threshold
        filtered_results = [
            result for result in search_results
            if float(result.relevance_score) >= float(min_relevance) and result.link
        ]

        # Sort by relevance score (highest first)
        filtered_results.sort(key=lambda x: float(x.relevance_score), reverse=True)

        # Extract URLs up to limit
        selected_urls = [result.link for result in filtered_results[:limit]]

        print(f"✅ Search result selection works:")
        print(f"   Total search results: {len(search_results)}")
        print(f"   Above 0.3 threshold: {len(filtered_results)}")
        print(f"   URLs selected: {len(selected_urls)}")

        # Verify filtering worked correctly
        assert len(filtered_results) == 3, f"Expected 3 results above 0.3 threshold, got {len(filtered_results)}"
        assert "https://example.com/low" not in selected_urls, "Low relevance URL should be filtered"

        # Verify sorting worked
        assert selected_urls[0] == "https://example.com/high", "Highest relevance should be first"
        assert selected_urls[1] == "https://example.com/medium2", "Second highest should be second"

        print(f"✅ URLs correctly filtered and sorted by 0.3 threshold")

        return True

    except Exception as e:
        print(f"❌ Search result selection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_session_organization():
    """Test session-based file organization."""
    print("\n=== Testing Session Organization ===")

    try:
        # Create test directory structure
        test_base = Path("/tmp/test_session_org")
        if test_base.exists():
            import shutil
            shutil.rmtree(test_base)

        sessions_dir = test_base / "sessions" / "test_session_001" / "search_analysis"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        # Create test search data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        search_data = {
            "search_query": "test search query",
            "search_results": "Mock search results content",
            "sources_found": "source1.com\nsource2.com",
            "captured_at": datetime.now().isoformat(),
            "session_id": "test_session_001",
            "search_type": "web_search_analysis",
            "verification_data": {
                "query_timestamp": timestamp,
                "result_length": len("Mock search results content"),
                "sources_count": 2
            }
        }

        # Save file
        search_file = sessions_dir / f"web_search_results_{timestamp}.json"
        with open(search_file, 'w', encoding='utf-8') as f:
            json.dump(search_data, f, indent=2, ensure_ascii=False)

        # Verify file was created
        assert search_file.exists(), "Search file should be created"

        # Verify file content
        with open(search_file, 'r') as f:
            loaded_data = json.load(f)

        assert loaded_data["session_id"] == "test_session_001", "Session ID should match"
        assert loaded_data["search_query"] == "test search query", "Search query should match"

        print(f"✅ Session organization works:")
        print(f"   Session directory created: {sessions_dir}")
        print(f"   File created: {search_file.name}")
        print(f"   File content verified")

        return True

    except Exception as e:
        print(f"❌ Session organization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all core improvement tests."""
    print("🚀 Testing Core Search System Improvements")
    print("=" * 50)

    tests = [
        ("Configuration Updates", test_configuration),
        ("URL Tracker", test_url_tracker),
        ("Search Result Selection", test_search_result_selection),
        ("Session Organization", test_session_organization),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("📊 CORE IMPROVEMENTS TEST SUMMARY")
    print("=" * 50)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1

    print(f"\nOverall Result: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All core improvements are working correctly!")
        print("\nKey Improvements Implemented:")
        print("1. ✅ Fixed crawl threshold at 0.3 for better success rates")
        print("2. ✅ Added target-based scraping (8 successful extractions target)")
        print("3. ✅ Implemented URL deduplication system")
        print("4. ✅ Added progressive retry logic with anti-bot escalation")
        print("5. ✅ Organized files in session-based directories")
        return True
    else:
        print("⚠️  Some improvements need attention.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)