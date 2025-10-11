#!/usr/bin/env python3
"""
Debug script to test content cleaning and see what's happening to the content.
"""

import asyncio
import logging
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def test_content_cleaning():
    """Test content cleaning on a sample URL to see what's happening."""

    # Import the content cleaning function
    try:
        from multi_agent_research_system.utils.content_cleaning import clean_content_with_gpt5_nano
        logger.info("✅ Successfully imported content cleaning function")
    except ImportError as e:
        logger.error(f"❌ Failed to import content cleaning: {e}")
        return

    # Import crawling to get raw content
    try:
        from multi_agent_research_system.utils.crawl4ai_utils import scrape_and_clean_single_url_direct
        logger.info("✅ Successfully imported crawling function")
    except ImportError as e:
        logger.error(f"❌ Failed to import crawling: {e}")
        return

    # Test URL from the logs
    test_url = "https://understandingwar.org/research/russia-ukraine/russian-offensive-campaign-assessment-october-9-2025/"
    search_query = "Russia Ukraine war October 2025"

    print(f"\n{'='*80}")
    print(f"TESTING CONTENT CLEANING")
    print(f"URL: {test_url}")
    print(f"Search Query: {search_query}")
    print(f"{'='*80}\n")

    try:
        # Step 1: Get raw content
        print("🔍 Step 1: Getting raw content...")
        raw_result = await scrape_and_clean_single_url_direct(
            url=test_url,
            search_query=search_query,
            skip_llm_cleaning=True  # Get raw content
        )

        if hasattr(raw_result, 'content'):
            raw_content = raw_result.content
        else:
            raw_content = raw_result.get('content', '')

        print(f"✅ Raw content length: {len(raw_content)} characters")
        print(f"📝 Raw content preview (first 1000 chars):")
        print("-" * 60)
        print(raw_content[:1000])
        print("-" * 60)

        # Step 2: Clean the content
        print(f"\n🧹 Step 2: Cleaning content with GPT-5-nano...")
        cleaned_content = await clean_content_with_gpt5_nano(
            content=raw_content,
            url=test_url,
            search_query=search_query
        )

        print(f"✅ Cleaned content length: {len(cleaned_content)} characters")
        print(f"📉 Compression ratio: {len(cleaned_content)/len(raw_content):.2%}")

        print(f"\n📝 Cleaned content preview (first 1000 chars):")
        print("-" * 60)
        print(cleaned_content[:1000])
        print("-" * 60)

        # Step 3: Analysis
        print(f"\n📊 Step 3: Analysis")
        print(f"Original length: {len(raw_content)}")
        print(f"Cleaned length: {len(cleaned_content)}")
        print(f"Characters removed: {len(raw_content) - len(cleaned_content)}")
        print(f"Compression ratio: {len(cleaned_content)/len(raw_content):.2%}")

        if len(cleaned_content) < 1000:
            print("⚠️  WARNING: Cleaned content is very short!")
            print("This suggests over-aggressive cleaning.")

        # Check if key content is preserved
        key_terms = ["Russia", "Ukraine", "war", "October", "2025"]
        preserved_terms = [term for term in key_terms if term.lower() in cleaned_content.lower()]
        print(f"Key terms preserved: {preserved_terms}")

        if len(preserved_terms) < len(key_terms) * 0.5:
            print("⚠️  WARNING: Many key terms were removed!")

    except Exception as e:
        logger.error(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_content_cleaning())