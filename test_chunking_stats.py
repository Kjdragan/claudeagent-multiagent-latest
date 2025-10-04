#!/usr/bin/env python3
"""
Test script to demonstrate the chunking statistics functionality.

This script shows how the statistics tracking works and what kind of
information we can gather about chunking usage patterns.
"""

import sys
import os
from pathlib import Path

# Add the multi_agent_research_system to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'multi_agent_research_system'))

from mcp_tools.enhanced_search_scrape_clean import create_adaptive_chunks, get_chunking_stats

def test_chunking_statistics():
    """Test the chunking statistics functionality."""
    print("🧪 Testing Chunking Statistics")
    print("=" * 50)

    # Test 1: Small content (no chunking)
    print("\n📝 Test 1: Small content (no chunking)")
    small_content = """# Small Search Results

**Query**: test query
**Results**: 3 sources found

## Article 1
Short content here.
"""

    chunks = create_adaptive_chunks(small_content, "test query")
    print(f"Content: {len(small_content)} chars -> {len(chunks)} chunks")

    # Test 2: Medium content (no chunking but larger)
    print("\n📝 Test 2: Medium content (no chunking)")
    medium_content = "# Medium Results\n" + "Content line.\n" * 1000  # ~15k chars
    chunks = create_adaptive_chunks(medium_content, "medium test")
    print(f"Content: {len(medium_content)} chars -> {len(chunks)} chunks")

    # Test 3: Large content (chunking triggered)
    print("\n📝 Test 3: Large content (chunking triggered)")
    large_content = "# Large Results\n" + "Long content line with substantial text.\n" * 2000  # ~45k chars
    chunks = create_adaptive_chunks(large_content, "large test")
    print(f"Content: {len(large_content)} chars -> {len(chunks)} chunks")

    # Test 4: Another large content (multiple chunking instances)
    print("\n📝 Test 4: Another large content")
    very_large_content = "# Very Large Results\n" + "Very long content line with extensive text and details.\n" * 1500  # ~60k chars
    chunks = create_adaptive_chunks(very_large_content, "very large test")
    print(f"Content: {len(very_large_content)} chars -> {len(chunks)} chunks")

    # Get and display statistics
    print("\n" + "=" * 50)
    print("📊 CURRENT CHUNKING STATISTICS")
    print("=" * 50)

    stats = get_chunking_stats()
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    # Analysis and insights
    print("\n" + "=" * 50)
    print("🔍 STATISTICS ANALYSIS")
    print("=" * 50)

    if stats.get("chunking_rate_percent", 0) > 50:
        print("⚠️  High chunking rate detected! Consider:")
        print("   - Increasing chunk size limit")
        print("   - Investigating content size patterns")
        print("   - Optimizing content generation")
    elif stats.get("chunking_rate_percent", 0) > 20:
        print("📊 Moderate chunking rate. Monitor usage patterns.")
    else:
        print("✅ Low chunking rate - system is efficient.")

    avg_chunks = stats.get("average_chunks_when_chunking", 0)
    if avg_chunks > 4:
        print("⚠️  High average chunks per request may impact:")
        print("   - Agent processing time")
        print("   - Context window utilization")
        print("   - Response delivery performance")
    else:
        print("✅ Reasonable chunk count per request.")

    print(f"\n💡 Key Insights:")
    print(f"   - Total API calls: {stats.get('total_calls', 0)}")
    print(f"   - Chunking frequency: {stats.get('chunking_rate_percent', 0)}%")
    print(f"   - Average content size: {stats.get('average_content_size', 'N/A')}")
    print(f"   - Total data processed: {stats.get('total_content_processed', 'N/A')}")

    print("\n✅ Statistics test completed successfully")
    return True

if __name__ == "__main__":
    success = test_chunking_statistics()
    sys.exit(0 if success else 1)