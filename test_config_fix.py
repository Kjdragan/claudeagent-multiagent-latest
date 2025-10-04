#!/usr/bin/env python3
"""
Test the fixed fallback configuration.
"""

import sys
import os
from pathlib import Path

# Add the multi_agent_research_system to the path
sys.path.insert(0, str(Path(__file__).parent / "multi_agent_research_system"))

def test_fixed_config():
    """Test the fixed configuration by forcing fallback usage."""
    try:
        print("Testing fixed configuration...")

        # Test by temporarily blocking the main import to force fallback
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'config.settings':
                raise ImportError("Forced import failure to test fallback")
            return original_import(name, *args, **kwargs)

        # Temporarily replace import to force fallback
        builtins.__import__ = mock_import

        try:
            # Clear any cached imports
            if 'utils.serp_search_utils' in sys.modules:
                del sys.modules['utils.serp_search_utils']

            # Now import and test the fallback
            from utils.serp_search_utils import get_enhanced_search_config

            config = get_enhanced_search_config()
            print(f"Config type: {type(config)}")
            print(f"Config class: {config.__class__.__name__}")

            # Test the critical attribute that was missing
            print(f"Has default_max_concurrent: {hasattr(config, 'default_max_concurrent')}")
            if hasattr(config, 'default_max_concurrent'):
                print(f"default_max_concurrent value: {config.default_max_concurrent}")
            else:
                print("ERROR: default_max_concurrent still missing!")
                return False

            # Test all the attributes we added
            attributes_to_check = [
                'default_num_results',
                'default_auto_crawl_top',
                'default_crawl_threshold',
                'default_anti_bot_level',
                'default_max_concurrent',
                'target_successful_scrapes',
                'url_deduplication_enabled',
                'progressive_retry_enabled',
                'max_retry_attempts',
                'progressive_timeout_multiplier',
                'max_response_tokens',
                'content_summary_threshold',
                'default_cleanliness_threshold',
                'min_content_length_for_cleaning',
                'min_cleaned_content_length',
                'default_crawl_timeout',
                'max_concurrent_crawls',
                'crawl_retry_attempts',
                'anti_bot_levels'
            ]

            print("Checking all attributes:")
            all_present = True
            for attr in attributes_to_check:
                has_attr = hasattr(config, attr)
                print(f"   {attr}: {'✓' if has_attr else '✗'}")
                if not has_attr:
                    all_present = False

            return all_present

        finally:
            # Restore original import
            builtins.__import__ = original_import

    except Exception as e:
        print(f"Error in fixed config test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fixed_config()
    print(f"\nFixed config test {'PASSED' if success else 'FAILED'}")