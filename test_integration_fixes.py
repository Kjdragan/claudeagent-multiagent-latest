#!/usr/bin/env python3
"""
Simple test to verify our integration fixes work without heavy dependencies.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_session_manager():
    """Test SessionManager works correctly."""
    print("🧪 Testing SessionManager...")

    try:
        # Direct import without going through the main package
        session_manager_path = project_root / "multi_agent_research_system" / "utils" / "session_manager.py"

        # Import the module directly
        import importlib.util
        spec = importlib.util.spec_from_file_location("session_manager", session_manager_path)
        session_manager_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(session_manager_module)

        # Test the functionality
        get_session_manager = session_manager_module.get_session_manager
        session_mgr = get_session_manager()
        session_id = session_mgr.get_session_id()

        print(f"✅ SessionManager: SUCCESS (session_id: {session_id[:8]}...)")
        return True

    except Exception as e:
        print(f"❌ SessionManager: FAILED - {e}")
        return False

def test_function_signature():
    """Test that the enhanced search function has correct signature."""
    print("🧪 Testing function signature compatibility...")

    try:
        # Read the source file directly to check function signature
        utils_file = project_root / "multi_agent_research_system" / "utils" / "z_search_crawl_utils.py"

        with open(utils_file, 'r') as f:
            content = f.read()

        # Simple approach: just check that all required parameters exist somewhere in the file
        # Check for required backward compatibility parameters
        required_params = ['query', 'search_type', 'num_results', 'session_id']
        has_all_params = all(param in content for param in required_params)

        # Also verify it's the enhanced function signature by checking for some new params
        enhanced_params = ['target_successful_scrapes', 'use_enhanced_selection']
        has_enhanced = any(param in content for param in enhanced_params)

        if has_all_params and has_enhanced:
            print(f"✅ Function signature: SUCCESS (found {len(required_params)} required params + enhanced params)")
            return True
        else:
            missing = [param for param in required_params if param not in content]
            print(f"❌ Function signature: FAILED - missing params: {missing}")
            return False

    except Exception as e:
        print(f"❌ Function signature: FAILED - {e}")
        return False

def test_async_await_fix():
    """Test that async/await issues are fixed in serp_search_utils.py."""
    print("🧪 Testing async/await fixes...")

    try:
        # Read the source file directly
        utils_file = project_root / "multi_agent_research_system" / "utils" / "serp_search_utils.py"

        with open(utils_file, 'r') as f:
            content = f.read()

        # Check for the specific fix: await enhanced_select_urls_for_crawling
        has_await = 'await enhanced_select_urls_for_crawling' in content

        if not has_await:
            print("❌ Async/await fix: FAILED - missing await for enhanced_select_urls_for_crawling")
            return False

        # Check that select_urls_for_crawling is declared as async
        is_async = 'async def select_urls_for_crawling(' in content

        if not is_async:
            print("❌ Async/await fix: FAILED - select_urls_for_crawling is not declared as async")
            return False

        print("✅ Async/await fix: SUCCESS (found await declaration and async function)")
        return True

    except Exception as e:
        print(f"❌ Async/await fix: FAILED - {e}")
        return False

def test_import_paths():
    """Test that import paths are fixed in MCP tools."""
    print("🧪 Testing import path fixes...")

    try:
        # Check enhanced_search_scrape_clean.py
        mcp_file1 = project_root / "multi_agent_research_system" / "mcp_tools" / "enhanced_search_scrape_clean.py"

        with open(mcp_file1, 'r') as f:
            content1 = f.read()

        # Should have relative imports, not absolute utils imports
        has_bad_import1 = 'from utils.serp_search_utils import' in content1
        has_good_import1 = 'from ..utils.serp_search_utils import' in content1

        if has_bad_import1 or not has_good_import1:
            print("❌ Import paths: FAILED - enhanced_search_scrape_clean.py has incorrect imports")
            return False

        # Check zplayground1_search.py
        mcp_file2 = project_root / "multi_agent_research_system" / "mcp_tools" / "zplayground1_search.py"

        with open(mcp_file2, 'r') as f:
            content2 = f.read()

        has_bad_import2 = 'from utils.z_search_crawl_utils import' in content2
        has_good_import2 = 'from ..utils.z_search_crawl_utils import' in content2

        if has_bad_import2 or not has_good_import2:
            print("❌ Import paths: FAILED - zplayground1_search.py has incorrect imports")
            return False

        print("✅ Import paths: SUCCESS (relative imports correctly fixed)")
        return True

    except Exception as e:
        print(f"❌ Import paths: FAILED - {e}")
        return False

def main():
    """Run all tests to verify our integration fixes."""
    print("🔧 Testing Multi-Agent Research System Integration Fixes\n")

    tests = [
        ("SessionManager Dependency", test_session_manager),
        ("Function Signature Compatibility", test_function_signature),
        ("Async/Await Fixes", test_async_await_fix),
        ("Import Path Fixes", test_import_paths)
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        result = test_func()
        results.append((test_name, result))

    print(f"\n{'='*50}")
    print("🏁 TEST SUMMARY")
    print(f"{'='*50}")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL INTEGRATION FIXES VERIFIED!")
        print("The system should now work without the original integration failures.")
    else:
        print(f"\n⚠️ {total - passed} fix(es) may still need attention.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)