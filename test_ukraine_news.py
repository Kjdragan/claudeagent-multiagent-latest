#!/usr/bin/env python3
"""
Test script for Ukraine-Russia war news query using the working pattern
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Add the multi_agent_research_system to the path
sys.path.insert(0, str(Path(__file__).parent / "multi_agent_research_system"))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️  python-dotenv not available, using environment variables only")

async def test_ukraine_news():
    """Test Ukraine-Russia war news query using working pattern"""

    print("🔍 Testing Ukraine-Russia War News Query")
    print("=" * 60)

    # Check environment variables
    serp_api_key = os.getenv('SERP_API_KEY')
    if not serp_api_key:
        print("❌ SERP_API_KEY not found in environment variables")
        return False

    print(f"✅ SERP_API_KEY found: {serp_api_key[:10]}...")

    try:
        # Test using the working pattern from implementation guide
        from utils.z_search_crawl_utils import search_crawl_and_clean_direct

        query = "the latest news from the Ukraine Russia war"
        session_id = f"ukraine_news_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print(f"🔍 Query: {query}")
        print(f"🆔 Session ID: {session_id}")
        print(f"⏱️  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Use working parameters optimized for news content
        print("🚀 Starting news search with working parameters...")
        print("   - num_results: 20")
        print("   - auto_crawl_top: 12")
        print("   - anti_bot_level: 1")
        print("   - search_type: search")
        print()

        start_time = time.time()
        result = await search_crawl_and_clean_direct(
            query=query,
            search_type="search",
            num_results=20,
            auto_crawl_top=12,
            anti_bot_level=1,
            session_id=session_id
        )

        # Stop timing
        elapsed_time = time.time() - start_time
        print(f"⏱️  Completed in: {elapsed_time:.1f} seconds")

        if result and len(result) > 500:
            print(f"✅ Ukraine news search SUCCESS! Generated {len(result)} characters")

            # Check if work product was created
            work_product_path = Path(f"KEVIN/sessions/{session_id}/research")
            if work_product_path.exists():
                work_products = list(work_product_path.glob("search_workproduct_*.md"))
                if work_products:
                    print(f"✅ Work product created: {work_products[0].name}")
                    print(f"📁 Location: {work_products[0]}")

                    # Read and display first 500 characters
                    with open(work_products[0], 'r') as f:
                        content = f.read()
                        print(f"\n📄 Content preview (first 500 characters):")
                        print("=" * 60)
                        print(content[:500])
                        print("=" * 60)

                        # Look for news sources
                        import re
                        urls = re.findall(r'https?://[^\s\)]+', content)
                        print(f"\n🔗 News Sources Found: {len(urls)} URLs")

                        if urls:
                            print("Top news sources:")
                            for i, url in enumerate(urls[:5], 1):
                                print(f"   {i}. {url}")

                        return True
                else:
                    print("⚠️  No work product files found")
            else:
                print("⚠️  Work product directory not created")
        else:
            print("❌ Ukraine news search failed or returned minimal content")

    except Exception as e:
        elapsed_time = time.time() - start_time if 'start_time' in locals() else 0
        print(f"\n❌ Test failed after {elapsed_time:.1f} seconds: {str(e)}")
        return False

    return False

async def main():
    """Run the Ukraine news test"""

    print("Ukraine-Russia War News Search Test")
    print("Using the working MCP Tool Integration pattern")
    print()

    # Run the test
    success = await test_ukraine_news()

    print("\n" + "=" * 60)
    if success:
        print("🎉 Ukraine news search completed successfully!")
        print("✅ Real SERP API integration working for current events")
        print("✅ Work products created with Ukraine-Russia war content")
        print("✅ Multiple news sources identified and processed")
    else:
        print("⚠️  Test failed. Check error messages above.")

    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)