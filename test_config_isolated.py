#!/usr/bin/env python3
"""
Isolated test of just the config fallback logic.
"""

import sys
import os
from pathlib import Path

# Add the multi_agent_research_system to the path
sys.path.insert(0, str(Path(__file__).parent / "multi_agent_research_system"))

def test_fallback_config():
    """Test the fallback configuration directly."""
    try:
        print("Testing fallback configuration logic...")

        # Simulate the exact same logic as in serp_search_utils.py lines 30-43
        print("1. Testing main config import...")
        try:
            from config.settings import get_enhanced_search_config
            print("   Main config import successful")
            config = get_enhanced_search_config()
            print(f"   Config type: {type(config)}")
            print(f"   Config class: {config.__class__.__name__}")
        except ImportError as e:
            print(f"   Main config import failed: {e}")
            print("   Using fallback config...")

            # This is the exact fallback logic from lines 34-43
            def get_enhanced_search_config():
                # Simple fallback config
                class SimpleConfig:
                    default_crawl_threshold = 0.3
                    target_successful_scrapes = 8
                    url_deduplication_enabled = True
                    progressive_retry_enabled = True
                    max_retry_attempts = 3
                    default_max_concurrent = 15
                return SimpleConfig()

            config = get_enhanced_search_config()
            print(f"   Fallback config type: {type(config)}")

        # Test the required attributes
        print("2. Testing required attributes...")
        required_attrs = [
            'default_crawl_threshold',
            'target_successful_scrapes',
            'url_deduplication_enabled',
            'progressive_retry_enabled',
            'max_retry_attempts',
            'default_max_concurrent'
        ]

        all_present = True
        for attr in required_attrs:
            has_attr = hasattr(config, attr)
            print(f"   {attr}: {'✓' if has_attr else '✗'}")
            if has_attr:
                try:
                    value = getattr(config, attr)
                    print(f"     value: {value}")
                except Exception as e:
                    print(f"     ERROR: {e}")
                    all_present = False
            else:
                print(f"     MISSING!")
                all_present = False

        return all_present

    except Exception as e:
        print(f"Error in fallback test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fallback_config()
    print(f"\nFallback config test {'PASSED' if success else 'FAILED'}")