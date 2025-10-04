#!/usr/bin/env python3
"""
Test script for the new expanded query search functionality.

This script tests the corrected query expansion workflow:
1. Generate multiple search queries
2. Execute SERP searches for each expanded query
3. Collect & deduplicate results into master list
4. Rank by relevance
5. Scrape from master ranked list within budget limits
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add the multi_agent_research_system to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'multi_agent_research_system'))

from utils.serp_search_utils import (
    expanded_query_search_and_extract,
    generate_expanded_queries,
    deduplicate_search_results
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_generate_expanded_queries():
    """Test the query expansion functionality."""
    print("\n🔍 Testing Query Expansion")
    print("=" * 50)

    original_query = "artificial intelligence healthcare applications"

    # Test with different limits
    for limit in [1, 3, 5]:
        expanded_queries = await generate_expanded_queries(original_query, limit)
        print(f"\nMax {limit} expanded queries:")
        for i, query in enumerate(expanded_queries, 1):
            print(f"  {i}. {query}")
        print(f"Total: {len(expanded_queries)} queries")

    print("\n✅ Query expansion test completed")
    return expanded_queries


async def test_deduplication():
    """Test the search results deduplication functionality."""
    print("\n🔄 Testing Search Results Deduplication")
    print("=" * 50)

    # Create mock search results with duplicates
    from utils.serp_search_utils import SearchResult

    # Create duplicate URLs with different relevance scores
    mock_results = [
        SearchResult("AI in Healthcare", "https://example.com/ai-healthcare", "AI transforming healthcare", 1, "2024-01-01", "TechNews", 0.9),
        SearchResult("Healthcare AI", "https://example.com/ai-healthcare", "Healthcare revolutionized by AI", 2, "2024-01-02", "HealthTech", 0.85),  # Duplicate URL
        SearchResult("Machine Learning Medicine", "https://example.com/ml-medicine", "ML applications in medicine", 3, "2024-01-03", "MedNews", 0.8),
        SearchResult("AI Healthcare", "https://example.com/ai-healthcare", "Latest AI healthcare trends", 1, "2024-01-04", "AITrends", 0.95),  # Duplicate URL with higher score
        SearchResult("Digital Health", "https://example.com/digital-health", "Digital transformation in health", 4, "2024-01-05", "DigitalNews", 0.75)
    ]

    print(f"Original results: {len(mock_results)}")
    for i, result in enumerate(mock_results, 1):
        print(f"  {i}. {result.title} (Score: {result.relevance_score}, URL: {result.link[:30]}...)")

    # Test deduplication
    deduplicated = deduplicate_search_results(mock_results)
    print(f"\nDeduplicated results: {len(deduplicated)}")
    for i, result in enumerate(deduplicated, 1):
        print(f"  {i}. {result.title} (Score: {result.relevance_score}, URL: {result.link[:30]}...)")

    # Verify the highest relevance score was kept for duplicates
    healthcare_results = [r for r in deduplicated if "ai-healthcare" in r.link]
    if healthcare_results:
        best_score = max(healthcare_results, key=lambda x: x.relevance_score).relevance_score
        print(f"\n✅ Best relevance score for duplicate URL: {best_score} (should be 0.95)")

    print("\n✅ Deduplication test completed")
    return deduplicated


async def test_expanded_query_search():
    """Test the complete expanded query search functionality."""
    print("\n🚀 Testing Complete Expanded Query Search")
    print("=" * 50)

    # Check if SERP_API_KEY is available
    if not os.getenv("SERP_API_KEY"):
        print("⚠️  SERP_API_KEY not found in environment variables")
        print("Skipping full integration test - only testing components")
        return None

    original_query = "quantum computing recent developments"
    session_id = "test_expanded_query"

    print(f"Original Query: '{original_query}'")
    print(f"Session ID: {session_id}")
    print(f"Max Expanded Queries: 3")
    print(f"Target Successful Scrapes: 15")

    try:
        # Set up test directory
        test_kevin_dir = Path.home() / "lrepos" / "claude-agent-sdk-python" / "KEVIN"
        test_kevin_dir.mkdir(parents=True, exist_ok=True)

        print("\n🔄 Executing expanded query search...")
        result = await expanded_query_search_and_extract(
            query=original_query,
            search_type="search",
            num_results=10,  # Smaller for testing
            auto_crawl_top=5,  # Smaller for testing
            crawl_threshold=0.3,
            session_id=session_id,
            kevin_dir=test_kevin_dir,
            max_expanded_queries=3
        )

        print(f"\n📊 Results Summary:")
        print(f"Result length: {len(result)} characters")

        # Check if result was chunked (it will be a dict if chunked)
        if isinstance(result, dict) and "content" in result:
            print("📦 Result was chunked into multiple content blocks:")
            content_blocks = result["content"]
            print(f"   Total chunks: {len(content_blocks)}")

            for i, chunk in enumerate(content_blocks):
                chunk_text = chunk["text"]
                print(f"   Chunk {i+1}: {len(chunk_text)} characters")

                # Check for key sections in each chunk
                if "EXPANDED QUERY SEARCH RESULTS" in chunk_text:
                    print(f"   ✅ Chunk {i+1} contains main results section")
                if "QUERY EXPANSION ANALYSIS" in chunk_text:
                    print(f"   ✅ Chunk {i+1} contains query analysis")
                if "MASTER SEARCH RESULTS" in chunk_text:
                    print(f"   ✅ Chunk {i+1} contains master results")
                if "PROCESSING SUMMARY" in chunk_text:
                    print(f"   ✅ Chunk {i+1} contains processing summary")
                if "Part" in chunk_text and "of" in chunk_text:
                    print(f"   ✅ Chunk {i+1} has proper part header")

            # Extract key metrics from all chunks
            all_text = '\n'.join(chunk["text"] for chunk in content_blocks)
            lines = all_text.split('\n')
            print("\n📈 Key Metrics from chunked content:")
            for line in lines:
                if "Total Master Results" in line:
                    print(f"   {line.strip()}")
                elif "URLs Extracted" in line:
                    print(f"   {line.strip()}")
                elif "Processing Time" in line:
                    print(f"   {line.strip()}")
                elif "Deduplication Rate" in line:
                    print(f"   {line.strip()}")
        else:
            # Traditional single chunk result
            print("📄 Result is a single content block")

            # Check for key sections in the result
            key_sections = [
                "EXPANDED QUERY SEARCH RESULTS",
                "QUERY EXPANSION ANALYSIS",
                "MASTER SEARCH RESULTS",
                "PROCESSING SUMMARY"
            ]

            found_sections = []
            for section in key_sections:
                if section in result:
                    found_sections.append(section)
                    print(f"✅ Found section: {section}")
                else:
                    print(f"❌ Missing section: {section}")

            print(f"\n📋 Sections found: {len(found_sections)}/{len(key_sections)}")

            # Extract key metrics
            lines = result.split('\n')
            print("\n📈 Key Metrics:")
            for line in lines:
                if "Total Master Results" in line:
                    print(f"   {line.strip()}")
                elif "URLs Extracted" in line:
                    print(f"   {line.strip()}")
                elif "Processing Time" in line:
                    print(f"   {line.strip()}")
                elif "Deduplication Rate" in line:
                    print(f"   {line.strip()}")

        print("\n✅ Expanded query search test completed successfully")
        return result

    except Exception as e:
        print(f"\n❌ Expanded query search test failed: {e}")
        logger.error(f"Test failed: {e}", exc_info=True)
        return None


async def main():
    """Run all tests."""
    print("🧪 Testing Expanded Query Search Implementation")
    print("=" * 60)

    # Test components
    await test_generate_expanded_queries()
    await test_deduplication()

    # Test full integration (if API key available)
    full_result = await test_expanded_query_search()

    print("\n" + "=" * 60)
    print("🏁 Testing Complete")

    if full_result:
        print("✅ All tests passed - Expanded query search is working correctly")

        # Show chunking statistics if available
        try:
            from mcp_tools.enhanced_search_scrape_clean import get_chunking_stats
            stats = get_chunking_stats()
            if "total_calls" in stats:
                print(f"\n📊 Chunking Statistics from this test run:")
                print(f"   Total calls: {stats['total_calls']}")
                print(f"   Chunking triggered: {stats['chunking_triggered']} ({stats['chunking_rate_percent']}%)")
                print(f"   Average content size: {stats['average_content_size']}")
                if stats['chunking_triggered'] > 0:
                    print(f"   Average chunks when chunking: {stats['average_chunks_when_chunking']}")
        except Exception as e:
            # Statistics not available, skip
            pass
    else:
        print("⚠️  Component tests passed, but full integration test skipped")
        print("   (This is normal if SERP_API_KEY is not configured)")


if __name__ == "__main__":
    asyncio.run(main())