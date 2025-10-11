#!/usr/bin/env python3
"""
Test script to verify that excluded sites tracking works in work products.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_excluded_sites_workproduct_generation():
    """Test that excluded sites are properly tracked and displayed in work products."""
    print("🧪 Testing Excluded Sites Tracking in Work Products")
    print("=" * 70)

    try:
        from multi_agent_research_system.utils.url_tracker import get_url_tracker
        from multi_agent_research_system.utils.z_search_crawl_utils import save_work_product
        from multi_agent_research_system.utils.z_search_crawl_utils import SearchResult

        print("✅ Successfully imported required modules")

        # Create a temporary URL tracker for testing
        temp_dir = Path("/tmp/test_url_tracker")
        temp_dir.mkdir(exist_ok=True)
        url_tracker = get_url_tracker(storage_dir=temp_dir)

        print(f"✅ URL tracker initialized with excluded domains: {url_tracker.get_excluded_domains()}")

        # Create test data simulating a research session
        test_query = "test query with excluded sites"
        session_id = "test_excluded_sites_session"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Test URLs including some that should be excluded
        test_search_results = [
            SearchResult(
                title="Test Result 1 - understandingwar.org",
                link="https://understandingwar.org/research/russian-offensive-campaign-assessment-october-8-2025/",
                snippet="This should be excluded due to domain block list",
                source="Institute for the Study of War",
                date="2 days ago",
                relevance_score=0.8
            ),
            SearchResult(
                title="Test Result 2 - understandingwar.org",
                link="https://understandingwar.org/research/russian-offensive-campaign-assessment-october-9-2025/",
                snippet="This should also be excluded",
                source="Institute for the Study of War",
                date="1 day ago",
                relevance_score=0.7
            ),
            SearchResult(
                title="Test Result 3 - Good Domain",
                link="https://kyivindependent.com/test-article",
                snippet="This should be processed normally",
                source="Kyiv Independent",
                date="1 day ago",
                relevance_score=0.6
            )
        ]

        # Test crawled content (only the non-excluded URL)
        test_crawled_content = [
            "This is cleaned content from the good domain",
        ]

        test_urls = ["https://kyivindependent.com/test-article"]

        # Create selection stats with excluded URLs
        test_selection_stats = {
            "pool_size": 3,
            "pool_target": 2,
            "filtered_candidates": 1,
            "trimmed_for_limit": 0,
            "fallback_applied": False,
            "domain_counts": {"understandingwar.org": 2, "kyivindependent.com": 1},
            "search_queries_executed": 1,
            "orthogonal_queries": [],
            "skipped_urls": [
                "https://understandingwar.org/research/russian-offensive-campaign-assessment-october-8-2025/",
                "https://understandingwar.org/research/russian-offensive-campaign-assessment-october-9-2025/"
            ]
        }

        print(f"\n📋 Test Data:")
        print(f"  - Search results: {len(test_search_results)}")
        print(f"  - Excluded URLs: {len(test_selection_stats['skipped_urls'])}")
        print(f"  - Processed URLs: {len(test_urls)}")

        # Create temporary workproduct directory
        workproduct_dir = Path("/tmp/test_workproducts")
        workproduct_dir.mkdir(exist_ok=True)

        print(f"\n🔧 Generating work product...")

        # Generate work product
        workproduct_path = save_work_product(
            search_results=test_search_results,
            crawled_content=test_crawled_content,
            urls=test_urls,
            query=test_query,
            selection_stats=test_selection_stats,
            attempted_crawls=1,
            successful_crawls=1,
            session_id=session_id,
            workproduct_dir=str(workproduct_dir),
            workproduct_prefix="test"
        )

        print(f"✅ Work product generated: {workproduct_path}")

        # Read and verify the work product content
        with open(workproduct_path, 'r', encoding='utf-8') as f:
            workproduct_content = f.read()

        print(f"\n📊 Work Product Analysis:")
        print(f"  - Total length: {len(workproduct_content)} characters")

        # Check for excluded sites section
        excluded_section_found = "🚫 Excluded Sites (Domain Block List)" in workproduct_content
        understandingwar_mentioned = "understandingwar.org" in workproduct_content
        domain_distribution_found = "🌍 Domain Distribution" in workproduct_content

        print(f"  - Excluded sites section: {'✅ Found' if excluded_section_found else '❌ Missing'}")
        print(f"  - understandingwar.org mentioned: {'✅ Yes' if understandingwar_mentioned else '❌ No'}")
        print(f"  - Domain distribution section: {'✅ Found' if domain_distribution_found else '❌ Missing'}")

        if excluded_section_found:
            # Extract and display the excluded sites section
            lines = workproduct_content.split('\n')
            excluded_section_start = None
            excluded_section_end = None

            for i, line in enumerate(lines):
                if "🚫 Excluded Sites (Domain Block List)" in line:
                    excluded_section_start = i
                elif excluded_section_start is not None and line.startswith("## ") and "Excluded" not in line:
                    excluded_section_end = i
                    break

            if excluded_section_start is not None:
                if excluded_section_end is None:
                    excluded_section_end = len(lines)

                excluded_section_lines = lines[excluded_section_start:excluded_section_end]
                excluded_section_text = '\n'.join(excluded_section_lines)

                print(f"\n📋 Extracted Excluded Sites Section:")
                print("-" * 50)
                print(excluded_section_text)
                print("-" * 50)

        # Show full work product preview
        print(f"\n📄 Full Work Product Preview:")
        print("-" * 50)
        print(workproduct_content[:1000])
        if len(workproduct_content) > 1000:
            print("...")
        print("-" * 50)

        # Clean up
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        if workproduct_path and os.path.exists(workproduct_path):
            os.remove(workproduct_path)
        if workproduct_dir.exists():
            shutil.rmtree(workproduct_dir, ignore_errors=True)

        print(f"\n✅ Test completed successfully!")
        print(f"   - Excluded sites tracking: {'✅ Working' if excluded_section_found else '❌ Not working'}")
        print(f"   - Domain exclusion: {'✅ Working' if understandingwar_mentioned else '❌ Not working'}")

        return excluded_section_found and understandingwar_mentioned

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_url_tracker_integration():
    """Test URL tracker integration with domain exclusion."""
    print(f"\n🔗 Testing URL Tracker Integration")
    print("-" * 40)

    try:
        from multi_agent_research_system.utils.url_tracker import get_url_tracker

        # Create test URL tracker
        temp_dir = Path("/tmp/test_url_tracker_integration")
        temp_dir.mkdir(exist_ok=True)
        url_tracker = get_url_tracker(storage_dir=temp_dir)

        # Test URLs
        test_urls = [
            "https://understandingwar.org/research/test-1",  # Should be excluded
            "https://understandingwar.org/research/test-2",  # Should be excluded
            "https://kyivindependent.com/article/test-1",    # Should be included
            "https://www.bbc.com/news/test-article",          # Should be included
            "https://understandingwar.org/research/test-3",  # Should be excluded
        ]

        print(f"Testing URL filtering with {len(test_urls)} URLs...")
        urls_to_crawl, skipped_urls = url_tracker.filter_urls(test_urls, "test_session")

        print(f"Results:")
        print(f"  - URLs to crawl: {len(urls_to_crawl)}")
        print(f"  - Skipped URLs: {len(skipped_urls)}")

        # Check exclusions
        excluded_count = len([url for url in skipped_urls if "understandingwar.org" in url])
        included_count = len([url for url in urls_to_crawl if "understandingwar.org" not in url])

        print(f"  - Excluded understandingwar.org URLs: {excluded_count}")
        print(f"  - Included non-excluded URLs: {included_count}")

        success = excluded_count == 3 and included_count == 2

        print(f"  - Integration test: {'✅ Passed' if success else '❌ Failed'}")

        # Clean up
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

        return success

    except Exception as e:
        print(f"❌ Error in URL tracker integration test: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Excluded Sites Tracking Implementation")
    print("=" * 70)

    # Test 1: Work product generation
    workproduct_success = test_excluded_sites_workproduct_generation()

    # Test 2: URL tracker integration
    tracker_success = test_url_tracker_integration()

    print(f"\n🏁 FINAL RESULTS:")
    print(f"  - Work product excluded sites tracking: {'✅ Working' if workproduct_success else '❌ Failed'}")
    print(f"  - URL tracker integration: {'✅ Working' if tracker_success else '❌ Failed'}")

    if workproduct_success and tracker_success:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"   The excluded sites tracking is working correctly.")
        print(f"   Future work products will include excluded sites information.")
    else:
        print(f"\n⚠️  SOME TESTS FAILED!")
        print(f"   Please review the implementation.")

    print(f"\n📈 EXPECTED BEHAVIOR:")
    print(f"   1. understandingwar.org URLs will be automatically excluded")
    print(f"   2. Work products will show '🚫 Excluded Sites (Domain Block List)' section")
    print(f"   3. Excluded URLs will be grouped by domain")
    print(f"   4. Users will have a record of what was excluded and why")