#!/usr/bin/env python3
"""
Simple test to verify SERP API is working with the working pattern
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the multi_agent_research_system to the path
sys.path.insert(0, str(Path(__file__).parent / "multi_agent_research_system"))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️  python-dotenv not available, using environment variables only")

async def test_working_serp_pattern():
    """Test the working SERP API pattern from implementation guide"""

    print("🔍 Testing Working SERP API Pattern...")

    # Check environment variables
    serp_api_key = os.getenv('SERP_API_KEY')
    if not serp_api_key:
        print("❌ SERP_API_KEY not found in environment variables")
        return False

    print(f"✅ SERP_API_KEY found: {serp_api_key[:10]}...")

    try:
        # Test using the working pattern from implementation guide
        from utils.z_search_crawl_utils import search_crawl_and_clean_direct

        query = "artificial intelligence in healthcare"
        print(f"🔍 Testing working search pattern for: {query}")

        # Use the working parameters from the implementation guide
        result = await search_crawl_and_clean_direct(
            query=query,
            search_type="search",
            num_results=5,
            auto_crawl_top=2,
            anti_bot_level=1,
            session_id="test_working_pattern"
        )

        if result and len(result) > 500:
            print(f"✅ Working search pattern SUCCESS! Generated {len(result)} characters")

            # Check if work product was created
            work_product_path = Path("KEVIN/sessions/test_working_pattern/research")
            if work_product_path.exists():
                work_products = list(work_product_path.glob("search_workproduct_*.md"))
                if work_products:
                    print(f"✅ Work product created: {work_products[0].name}")
                    print(f"📁 Location: {work_products[0]}")

                    # Read and display a snippet
                    with open(work_products[0], 'r') as f:
                        content = f.read()
                        print(f"📄 Content preview (first 200 chars):")
                        print(content[:200] + "...")
                    return True
                else:
                    print("⚠️  No work product files found")
            else:
                print("⚠️  Work product directory not created")
        else:
            print("❌ Working search pattern failed or returned minimal content")

    except Exception as e:
        print(f"❌ Working pattern test failed: {str(e)}")
        return False

    return False

async def main():
    """Run the working pattern test"""

    print("🚀 Testing Working SERP API Pattern")
    print("=" * 50)

    success = await test_working_serp_pattern()

    print("\n" + "=" * 50)
    if success:
        print("🎉 SUCCESS! SERP API integration is working correctly.")
        print("✅ Real SERP API calls are functional")
        print("✅ Work products are being created")
        print("✅ Search results are being processed")
        print("\nThe implementation guide patterns are working!")
    else:
        print("⚠️  Test failed. Check the error messages above.")

    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)