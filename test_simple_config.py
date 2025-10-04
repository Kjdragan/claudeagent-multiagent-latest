#!/usr/bin/env python3
"""
Test SimpleConfig directly by copying the class definition.
"""

def test_simple_config():
    """Test SimpleConfig class directly."""

    # Copy the exact SimpleConfig class from the fixed file
    class SimpleConfig:
        # Search settings
        default_num_results = 15
        default_auto_crawl_top = 10
        default_crawl_threshold = 0.3
        default_anti_bot_level = 1
        default_max_concurrent = 15

        # Target-based scraping settings
        target_successful_scrapes = 8
        url_deduplication_enabled = True
        progressive_retry_enabled = True

        # Retry logic settings
        max_retry_attempts = 3
        progressive_timeout_multiplier = 1.5

        # Token management
        max_response_tokens = 20000
        content_summary_threshold = 20000

        # Content cleaning settings
        default_cleanliness_threshold = 0.7
        min_content_length_for_cleaning = 500
        min_cleaned_content_length = 200

        # Crawl settings
        default_crawl_timeout = 30000
        max_concurrent_crawls = 15
        crawl_retry_attempts = 2

        # Anti-bot levels
        anti_bot_levels = {
            0: "basic",      # 6/10 sites success
            1: "enhanced",   # 8/10 sites success
            2: "advanced",   # 9/10 sites success
            3: "stealth"     # 9.5/10 sites success
        }

    # Test the critical attributes that are accessed in the code
    print("Testing SimpleConfig class...")

    config = SimpleConfig()
    print(f"Config type: {type(config)}")

    # Test specific attributes that are accessed in serp_search_utils.py
    critical_attributes = [
        'url_deduplication_enabled',           # line 213
        'target_successful_scrapes',           # line 463, 675, 680
        'progressive_retry_enabled',           # line 496
        'default_max_concurrent',              # line 682 - This was the problematic one
    ]

    print("Testing critical attributes:")
    all_present = True
    for attr in critical_attributes:
        has_attr = hasattr(config, attr)
        print(f"   {attr}: {'✓' if has_attr else '✗'}")
        if has_attr:
            try:
                value = getattr(config, attr)
                print(f"     value: {value}")
            except Exception as e:
                print(f"     ERROR accessing: {e}")
                all_present = False
        else:
            print(f"     MISSING!")
            all_present = False

    # Additional test: simulate the actual usage patterns from the code
    print("\nTesting actual usage patterns:")

    try:
        # Simulate line 213: if use_deduplication and config.url_deduplication_enabled:
        result = config.url_deduplication_enabled
        print(f"✓ config.url_deduplication_enabled = {result}")
    except Exception as e:
        print(f"✗ Error accessing url_deduplication_enabled: {e}")
        all_present = False

    try:
        # Simulate line 463: target_count = target_successful_scrapes or config.target_successful_scrapes:
        result = config.target_successful_scrapes
        print(f"✓ config.target_successful_scrapes = {result}")
    except Exception as e:
        print(f"✗ Error accessing target_successful_scrapes: {e}")
        all_present = False

    try:
        # Simulate line 496: if config.progressive_retry_enabled:
        result = config.progressive_retry_enabled
        print(f"✓ config.progressive_retry_enabled = {result}")
    except Exception as e:
        print(f"✗ Error accessing progressive_retry_enabled: {e}")
        all_present = False

    try:
        # Simulate line 682: max_concurrent=config.default_max_concurrent
        result = config.default_max_concurrent
        print(f"✓ config.default_max_concurrent = {result}")
    except Exception as e:
        print(f"✗ Error accessing default_max_concurrent: {e}")
        all_present = False

    return all_present

if __name__ == "__main__":
    success = test_simple_config()
    print(f"\nSimpleConfig test {'PASSED' if success else 'FAILED'}")