#!/usr/bin/env python3
"""
Simple debug script to test the SERP config issue without external dependencies.
"""

import sys
import os
from pathlib import Path

# Add the multi_agent_research_system to the path
sys.path.insert(0, str(Path(__file__).parent / "multi_agent_research_system"))

def test_config_import():
    """Test configuration import and attribute access."""
    try:
        print("Testing SERP search configuration...")

        # Test the specific import that's failing
        print("1. Testing config import from serp_search_utils...")
        from utils.serp_search_utils import get_enhanced_search_config

        print("2. Getting config...")
        config = get_enhanced_search_config()

        print(f"3. Config type: {type(config)}")
        print(f"4. Config class name: {config.__class__.__name__}")

        # Check all attributes that should be available
        attributes_to_check = [
            'default_crawl_threshold',
            'target_successful_scrapes',
            'url_deduplication_enabled',
            'progressive_retry_enabled',
            'max_retry_attempts',
            'default_max_concurrent'
        ]

        print("5. Checking required attributes:")
        for attr in attributes_to_check:
            has_attr = hasattr(config, attr)
            print(f"   - {attr}: {'✓' if has_attr else '✗'}")
            if has_attr:
                try:
                    value = getattr(config, attr)
                    print(f"     value: {value}")
                except Exception as e:
                    print(f"     error accessing: {e}")
            else:
                print(f"     MISSING!")

        if not hasattr(config, 'default_max_concurrent'):
            print("\n6. ERROR: default_max_concurrent attribute is missing!")
            print(f"   Available attributes: {[attr for attr in dir(config) if not attr.startswith('_')]}")
            return False
        else:
            print("\n6. SUCCESS: All required attributes are present!")
            return True

    except Exception as e:
        print(f"Error in config test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_config_import()
    print(f"\nConfig test {'PASSED' if success else 'FAILED'}")