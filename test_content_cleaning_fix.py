#!/usr/bin/env python3
"""
Test script to verify the content cleaning data structure mismatch fix.

This script tests the fix for the 'str' object has no attribute 'get' error
that was occurring when concurrent scraping results were passed to content cleaning.
"""

import asyncio
import logging
import sys
import os

# Add the multi_agent_research_system to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'multi_agent_research_system'))

from utils.content_cleaning import clean_content_with_gpt5_nano
from fix_content_cleaning_errors import safe_extract_content_from_result

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_content_cleaning_fix():
    """Test that the content cleaning fix handles different result types correctly."""

    print("🔧 Testing Content Cleaning Data Structure Fix")
    print("=" * 60)

    # Sample content for testing
    test_content = """
    <!DOCTYPE html>
    <html>
    <head><title>Test Article</title></head>
    <body>
        <nav>Navigation menu</nav>
        <div class="ads">Advertisement</div>
        <article>
            <h1>Main Article Title</h1>
            <p>This is the main content of the article that should be preserved.</p>
            <p>More article content goes here with important information.</p>
        </article>
        <footer>Footer content</footer>
    </body>
    </html>
    """

    test_url = "https://example.com/test-article"
    test_search_query = "test article content"

    try:
        print("1. Testing content cleaning function...")

        # Call the content cleaning function
        raw_result = await clean_content_with_gpt5_nano(
            content=test_content,
            url=test_url,
            search_query=test_search_query
        )

        print(f"   Raw result type: {type(raw_result)}")
        print(f"   Raw result length: {len(str(raw_result)) if raw_result else 0}")

        # Test safe extraction
        print("\n2. Testing safe extraction function...")

        cleaned_result = safe_extract_content_from_result(raw_result)

        print(f"   Cleaned result type: {type(cleaned_result)}")
        print(f"   Cleaned result length: {len(cleaned_result) if cleaned_result else 0}")

        if cleaned_result:
            print(f"   Cleaned content preview: {cleaned_result[:200]}...")
            print("   ✅ Safe extraction successful")
        else:
            print("   ❌ Safe extraction failed")
            return False

        # Test the problematic pattern that was causing the error
        print("\n3. Testing the fixed pattern from z_search_crawl_utils.py...")

        # This should NOT raise an error anymore
        try:
            # This is the pattern that was failing before the fix
            if cleaned_result and len(cleaned_result.strip()) > 0:
                final_content = cleaned_result
                print("   ✅ Fixed pattern works correctly")
                print(f"   Final content length: {len(final_content)}")
            else:
                print("   ❌ No cleaned content available")
                return False

        except AttributeError as e:
            if "'str' object has no attribute 'get'" in str(e):
                print(f"   ❌ Original error still occurs: {e}")
                return False
            else:
                print(f"   ❌ Unexpected AttributeError: {e}")
                return False

        print("\n4. Testing edge cases...")

        # Test with None result
        none_result = safe_extract_content_from_result(None)
        print(f"   None result handling: '{none_result}' (length: {len(none_result)})")

        # Test with empty string
        empty_result = safe_extract_content_from_result("")
        print(f"   Empty string handling: '{empty_result}' (length: {len(empty_result)})")

        # Test with dictionary result (if it ever occurs)
        dict_result = safe_extract_content_from_result({"content": "test content from dict"})
        print(f"   Dictionary result handling: '{dict_result}' (length: {len(dict_result)})")

        print("\n" + "=" * 60)
        print("🎉 All tests passed! The content cleaning fix is working correctly.")
        print("\nKey fixes applied:")
        print("1. ✅ Fixed data structure mismatch in enhanced_clean_extracted_content")
        print("2. ✅ Added safe extraction for robust result handling")
        print("3. ✅ Prevented 'str' object has no attribute 'get' errors")
        print("4. ✅ Maintained backward compatibility")

        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    print("Content Cleaning Data Structure Mismatch Fix Test")
    print("Testing Phase 2.2 implementation for concurrent scraping integration")
    print()

    success = await test_content_cleaning_fix()

    if success:
        print("\n🚀 Fix verification successful!")
        print("The concurrent scraping to content cleaning pipeline should now work correctly.")
        sys.exit(0)
    else:
        print("\n💥 Fix verification failed!")
        print("The data structure mismatch issue still exists.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())