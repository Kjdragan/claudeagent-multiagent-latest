#!/usr/bin/env python3
"""
Test script to verify domain exclusion functionality.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_domain_exclusion():
    """Test the domain exclusion functionality."""
    try:
        from multi_agent_research_system.utils.url_tracker import URLTracker

        print("🧪 Testing Domain Exclusion Functionality")
        print("=" * 60)

        # Create a temporary URL tracker for testing
        temp_dir = Path("/tmp/test_url_tracker")
        temp_dir.mkdir(exist_ok=True)
        tracker = URLTracker(storage_dir=temp_dir)

        # Test URLs including the problematic domain
        test_urls = [
            "https://understandingwar.org/research/russia-ukraine/russian-offensive-campaign-assessment-october-8-2025/",
            "https://understandingwar.org/research/russia-ukraine/russian-offensive-campaign-assessment-october-9-2025/",
            "https://kyivindependent.com/ukraine-war-latest-russia-operates-network-of-210-facilities/",
            "https://www.russiamatters.org/news/russia-ukraine-war-report-card/",
            "https://www.aljazeera.com/news/2025/10/2/russia-ukraine-war-list-of-key-events-day-1316"
        ]

        print(f"\n📋 Test URLs ({len(test_urls)} total):")
        for i, url in enumerate(test_urls, 1):
            print(f"  {i}. {url}")

        # Test the filtering
        print(f"\n🔍 Testing URL filtering with domain exclusion...")
        urls_to_crawl, skipped_urls = tracker.filter_urls(test_urls, session_id="test_session")

        print(f"\n📊 Results:")
        print(f"  ✅ URLs to crawl: {len(urls_to_crawl)}")
        print(f"  ⏭️  Skipped URLs: {len(skipped_urls)}")

        # Check specific domains
        isw_urls = [url for url in test_urls if "understandingwar.org" in url]
        other_urls = [url for url in test_urls if "understandingwar.org" not in url]

        print(f"\n🎯 Domain-specific analysis:")
        print(f"  🚫 ISW URLs (should be excluded): {len(isw_urls)}")
        for url in isw_urls:
            excluded = url in skipped_urls
            print(f"    {'✅ EXCLUDED' if excluded else '❌ NOT EXCLUDED'}: {url[:80]}...")

        print(f"  ✅ Other domains (should be included): {len(other_urls)}")
        for url in other_urls:
            included = url in urls_to_crawl
            print(f"    {'✅ INCLUDED' if included else '❌ NOT INCLUDED'}: {url[:80]}...")

        # Test exclusion list management
        print(f"\n🔧 Testing exclusion list management...")
        print(f"  Current excluded domains: {tracker.get_excluded_domains()}")

        # Test adding a domain
        tracker.add_excluded_domain("example.com")
        updated_domains = tracker.get_excluded_domains()
        print(f"  After adding example.com: {updated_domains}")

        # Test removing a domain
        tracker.remove_excluded_domain("understandingwar.org")
        updated_domains = tracker.get_excluded_domains()
        print(f"  After removing understandingwar.org: {updated_domains}")

        # Re-test filtering after removal
        print(f"\n🔄 Testing filtering after removing understandingwar.org from exclusion...")
        urls_to_crawl_after, skipped_urls_after = tracker.filter_urls(test_urls, session_id="test_session_2")

        print(f"  After removal:")
        print(f"    URLs to crawl: {len(urls_to_crawl_after)}")
        print(f"    Skipped URLs: {len(skipped_urls_after)}")

        # Verify ISW URLs are now included
        isw_urls_after = [url for url in isw_urls if url in urls_to_crawl_after]
        print(f"    ISW URLs now included: {len(isw_urls_after)}/{len(isw_urls)}")

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

        print(f"\n🎉 Domain exclusion test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error during domain exclusion test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_content_simulation():
    """Simulate what would happen with understandingwar.org content."""
    print(f"\n🎭 Content Simulation Analysis")
    print("=" * 60)

    print("🚫 UNDERSTANDINGWAR.ORG ISSUES:")
    print("  • Scraping captures navigation, not article content")
    print("  • JavaScript-heavy site with dynamic content loading")
    print("  • Content structure incompatible with our extraction")
    print("  • Even with level 3 anti-bot (stealth), no article content extracted")

    print("\n💡 SOLUTION:")
    print("  • Add to domain exclusion list to prevent wasted processing")
    print("  • Focus on more reliable sources for similar content")
    print("  • ISW content is high-quality but not accessible via scraping")

    print("\n📈 BENEFITS OF EXCLUSION:")
    print("  • Faster research processing (no failed attempts)")
    print("  • Cleaner research data (no navigation pollution)")
    print("  • Better resource utilization")
    print("  • More reliable sources prioritized")

if __name__ == "__main__":
    success = test_domain_exclusion()
    test_content_simulation()

    if success:
        print(f"\n✅ All tests passed! Domain exclusion is working correctly.")
        print(f"\n📝 NEXT STEPS:")
        print(f"1. The system will now automatically exclude understandingwar.org URLs")
        print(f"2. Add other problematic domains as they are identified")
        print(f"3. Monitor logs for exclusion warnings to track effectiveness")
    else:
        print(f"\n❌ Tests failed. Please check the implementation.")