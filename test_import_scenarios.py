#!/usr/bin/env python3
"""
Test different import scenarios to identify the configuration issue.
"""

import sys
import os
from pathlib import Path

# Add the multi_agent_research_system to the path
sys.path.insert(0, str(Path(__file__).parent / "multi_agent_research_system"))

def test_direct_import():
    """Test direct import of serp_search_utils."""
    try:
        print("1. Testing direct import...")

        # Test importing the module directly
        import utils.serp_search_utils

        # Get the config
        config = utils.serp_search_utils.get_enhanced_search_config()
        print(f"   Config type: {type(config)}")
        print(f"   Has default_max_concurrent: {hasattr(config, 'default_max_concurrent')}")

        return True

    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_from_import():
    """Test from import of serp_search_utils."""
    try:
        print("2. Testing from import...")

        # Test using from import
        from utils.serp_search_utils import get_enhanced_search_config

        config = get_enhanced_search_config()
        print(f"   Config type: {type(config)}")
        print(f"   Has default_max_concurrent: {hasattr(config, 'default_max_concurrent')}")

        return True

    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_imports():
    """Test multiple imports to check for caching issues."""
    try:
        print("3. Testing multiple imports...")

        # Import using different methods
        import utils.serp_search_utils as module1
        from utils.serp_search_utils import get_enhanced_search_config as func1

        # Get configs from different sources
        config1 = module1.get_enhanced_search_config()
        config2 = func1()

        print(f"   Config1 type: {type(config1)}")
        print(f"   Config2 type: {type(config2)}")
        print(f"   Config1 has default_max_concurrent: {hasattr(config1, 'default_max_concurrent')}")
        print(f"   Config2 has default_max_concurrent: {hasattr(config2, 'default_max_concurrent')}")
        print(f"   Same object: {config1 is config2}")

        return True

    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_module_reload():
    """Test module reloading scenarios."""
    try:
        print("4. Testing module reload...")

        # Import the module
        import utils.serp_search_utils

        # Get config before reload
        config1 = utils.serp_search_utils.get_enhanced_search_config()

        # Reload the module (if possible)
        try:
            import importlib
            importlib.reload(utils.serp_search_utils)
            print("   Module reloaded")
        except Exception as e:
            print(f"   Reload failed: {e}")

        # Get config after reload
        config2 = utils.serp_search_utils.get_enhanced_search_config()

        print(f"   Config1 type: {type(config1)}")
        print(f"   Config2 type: {type(config2)}")
        print(f"   Config1 has default_max_concurrent: {hasattr(config1, 'default_max_concurrent')}")
        print(f"   Config2 has default_max_concurrent: {hasattr(config2, 'default_max_concurrent')}")

        return True

    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all import scenario tests."""
    print("Testing import scenarios for SERP search configuration...")

    tests = [
        test_direct_import,
        test_from_import,
        test_multiple_imports,
        test_module_reload
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"   Test failed with exception: {e}")
            results.append(False)
        print()

    passed = sum(results)
    total = len(results)

    print(f"Results: {passed}/{total} tests passed")
    return passed == total

if __name__ == "__main__":
    success = main()
    print(f"\nImport scenario test {'PASSED' if success else 'FAILED'}")