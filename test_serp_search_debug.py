#!/usr/bin/env python3
"""
Debug script to test SERP search functionality directly.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the multi_agent_research_system to the path
sys.path.insert(0, str(Path(__file__).parent / "multi_agent_research_system"))

async def test_serp_search():
    """Test SERP search functionality directly."""
    try:
        print("Testing SERP search functionality...")

        # Import the function
        from utils.serp_search_utils import serp_search_and_extract
        from utils.serp_search_utils import get_enhanced_search_config

        # Test config first
        config = get_enhanced_search_config()
        print(f"Config type: {type(config)}")
        print(f"Has default_max_concurrent: {hasattr(config, 'default_max_concurrent')}")
        if hasattr(config, 'default_max_concurrent'):
            print(f"default_max_concurrent value: {config.default_max_concurrent}")
        else:
            print(f"Available attributes: {[attr for attr in dir(config) if not attr.startswith('_')]}")

        # Test basic SERP search (not the full extract function)
        from utils.serp_search_utils import execute_serp_search

        results = await execute_serp_search(
            query="test query",
            search_type="search",
            num_results=1
        )

        print(f"Search results: {len(results)} found")
        if results:
            result = results[0]
            print(f"First result: {result.title}")
            print(f"URL: {result.link}")

        return True

    except Exception as e:
        print(f"Error in test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_serp_search())
    print(f"Test {'PASSED' if success else 'FAILED'}")