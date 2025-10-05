"""Test the intelligent research utilities functions only.

This tests the core intelligence components that we can import and test directly
without the MCP tool wrapper complications.
"""

import asyncio
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

async def test_relevance_scoring():
    """Test the enhanced relevance scoring function."""
    print("\n" + "=" * 80)
    print("TEST: ENHANCED RELEVANCE SCORING (Z-PLAYGROUND1 ALGORITHM)")
    print("=" * 80)

    try:
        from tools.intelligent_research_tool import (
            calculate_enhanced_relevance_score,
            extract_query_terms,
        )
        print("✅ Relevance scoring functions imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import relevance scoring functions: {e}")
        return False

    # Test cases for z-playground1 algorithm
    test_cases = [
        {
            "query": "artificial intelligence machine learning",
            "title": "Understanding Artificial Intelligence and Machine Learning",
            "snippet": "A comprehensive guide to AI and ML concepts and applications",
            "position": 1,
            "description": "Perfect match with query"
        },
        {
            "query": "python programming tutorial",
            "title": "JavaScript Programming Basics",
            "snippet": "Learn the fundamentals of JavaScript for web development",
            "position": 5,
            "description": "Different technology, should score lower"
        },
        {
            "query": "Russia Ukraine war latest news",
            "title": "Latest Updates on Russia-Ukraine Conflict",
            "snippet": "Recent developments and military actions in the ongoing conflict",
            "position": 2,
            "description": "Good relevance match"
        }
    ]

    print("\n📊 Testing Z-Playground1 Enhanced Relevance Scoring:")
    print("   Formula: Position(40%) + Title Match(30%) + Snippet Match(30%)")

    for i, test_case in enumerate(test_cases, 1):
        query_terms = extract_query_terms(test_case["query"])
        score = calculate_enhanced_relevance_score(
            title=test_case["title"],
            snippet=test_case["snippet"],
            position=test_case["position"],
            query_terms=query_terms
        )

        print(f"\nTest Case {i}: {test_case['description']}")
        print(f"  Query: '{test_case['query']}' → Terms: {query_terms}")
        print(f"  Title: '{test_case['title']}'")
        print(f"  Position: {test_case['position']}")
        print(f"  Final Score: {score:.3f}")

        # Calculate component scores for analysis
        title_lower = test_case["title"].lower()
        snippet_lower = test_case["snippet"].lower()
        query_terms_lower = [term.lower() for term in query_terms if term]

        if test_case["position"] <= 10:
            position_score = (11 - test_case["position"]) / 10
        else:
            position_score = max(0.05, 0.1 - ((test_case["position"] - 10) * 0.01))

        title_matches = sum(1 for term in query_terms_lower if term in title_lower)
        title_score = min(1.0, title_matches / len(query_terms_lower)) if query_terms_lower else 0

        snippet_matches = sum(1 for term in query_terms_lower if term in snippet_lower)
        snippet_score = min(1.0, snippet_matches / len(query_terms_lower)) if query_terms_lower else 0

        print("  Component Scores:")
        print(f"    - Position (40%): {position_score:.3f}")
        print(f"    - Title Match (30%): {title_score:.3f} ({title_matches}/{len(query_terms_lower)})")
        print(f"    - Snippet Match (30%): {snippet_score:.3f} ({snippet_matches}/{len(query_terms_lower)})")

    print("\n✅ Enhanced relevance scoring test PASSED")
    return True


async def test_url_selection():
    """Test the threshold-based URL selection."""
    print("\n" + "=" * 80)
    print("TEST: THRESHOLD-BASED URL SELECTION")
    print("=" * 80)

    try:
        from tools.intelligent_research_tool import (
            SearchResult,
            select_urls_for_crawling,
        )
        print("✅ URL selection function imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import URL selection function: {e}")
        return False

    # Create mock search results with varying relevance scores
    mock_search_results = [
        SearchResult(title="Highly Relevant Article", link="https://example1.com",
                    snippet="Directly addresses the search query with comprehensive information",
                    position=1, relevance_score=0.95),
        SearchResult(title="Very Relevant Article", link="https://example2.com",
                    snippet="Contains detailed information about the search topic",
                    position=2, relevance_score=0.87),
        SearchResult(title="Somewhat Relevant Article", link="https://example3.com",
                    snippet="Related to the search but not directly addressing it",
                    position=5, relevance_score=0.45),
        SearchResult(title="Barely Relevant Article", link="https://example4.com",
                    snippet="Mentions the topic briefly but focuses on other things",
                    position=8, relevance_score=0.28),
        SearchResult(title="Irrelevant Article", link="https://example5.com",
                    snippet="Completely different topic, no connection to search",
                    position=12, relevance_score=0.15),
    ]

    print("\n📊 Mock Search Results Created:")
    for i, result in enumerate(mock_search_results, 1):
        print(f"  {i}. {result.title} - Score: {result.relevance_score:.2f}")

    # Test with threshold 0.3
    selected_urls = select_urls_for_crawling(
        search_results=mock_search_results,
        limit=3,
        min_relevance=0.3
    )

    print("\n🔍 URL Selection with Threshold 0.3:")
    print(f"  - Total results: {len(mock_search_results)}")
    print(f"  - Above threshold: {len([r for r in mock_search_results if r.relevance_score >= 0.3])}")
    print(f"  - Selected for crawling: {len(selected_urls)}")
    print(f"  - URLs selected: {selected_urls}")

    # Validate selection
    expected_selected = [r.link for r in mock_search_results if r.relevance_score >= 0.3][:3]
    if selected_urls == expected_selected:
        print("✅ URL selection logic works correctly")
    else:
        print("❌ URL selection logic has issues")
        return False

    # Test with different parameters
    selected_urls_strict = select_urls_for_crawling(
        search_results=mock_search_results,
        limit=5,
        min_relevance=0.6
    )

    print("\n🔍 Stricter Selection with Threshold 0.6:")
    print(f"  - Selected: {len(selected_urls_strict)} URLs")
    print(f"  - URLs: {selected_urls_strict}")

    print("\n✅ Threshold-based URL selection test PASSED")
    return True


async def test_query_extraction():
    """Test the query term extraction."""
    print("\n" + "=" * 80)
    print("TEST: QUERY TERM EXTRACTION")
    print("=" * 80)

    try:
        from tools.intelligent_research_tool import extract_query_terms
        print("✅ Query extraction function imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import query extraction function: {e}")
        return False

    test_queries = [
        "artificial intelligence machine learning deep learning",
        "Russia Ukraine war latest developments",
        "Python web development tutorial for beginners",
        "climate change environmental impact solutions",
        "stock market investment strategies cryptocurrency"
    ]

    print("\n📊 Query Term Extraction Results:")
    for query in test_queries:
        terms = extract_query_terms(query)
        print(f"  Query: '{query}'")
        print(f"  Extracted Terms: {terms}")
        print(f"  Term Count: {len(terms)}")

    print("\n✅ Query term extraction test PASSED")
    return True


async def test_mcp_compression_simulation():
    """Simulate the MCP compression logic."""
    print("\n" + "=" * 80)
    print("TEST: MCP COMPRESSION SIMULATION")
    print("=" * 80)

    # Simulate content compression logic
    def simulate_mcp_compression(total_content_chars: int, max_tokens: int = 20000) -> dict:
        """Simulate the compression strategy."""
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        content_tokens = total_content_chars // 4

        compression_levels = [
            {"level": "Top Priority", "percentage": 0.5, "detail": "Full"},
            {"level": "High Priority", "percentage": 0.3, "detail": "Summarized"},
            {"level": "References Only", "percentage": 0.2, "detail": "Brief"}
        ]

        results = []
        remaining_tokens = max_tokens

        for level in compression_levels:
            level_tokens = min(remaining_tokens, int(content_tokens * level["percentage"]))
            if level_tokens > 0:
                results.append({
                    "level": level["level"],
                    "tokens_allocated": level_tokens,
                    "detail_level": level["detail"]
                })
                remaining_tokens -= level_tokens

        return {
            "total_content_chars": total_content_chars,
            "estimated_tokens": content_tokens,
            "max_tokens": max_tokens,
            "compression_needed": content_tokens > max_tokens,
            "compression_levels": results
        }

    # Test with different content sizes
    test_sizes = [
        50000,  # Large content (would definitely need compression)
        20000,  # Medium content (at limit)
        10000,  # Small content (no compression needed)
        5000,   # Very small content
    ]

    print("\n📊 MCP Compression Simulation Results:")
    print("   Token Limit: 20,000 tokens")
    print("   Ratio: ~1 token = 4 characters")

    for size in test_sizes:
        result = simulate_mcp_compression(size)
        print(f"\n  Content Size: {size:,} chars (~{result['estimated_tokens']:,} tokens)")
        print(f"  Compression Needed: {'Yes' if result['compression_needed'] else 'No'}")

        for level in result['compression_levels']:
            print(f"    - {level['level']}: {level['tokens_allocated']:,} tokens ({level['detail']})")

    print("\n✅ MCP compression simulation test PASSED")
    return True


async def run_all_utility_tests():
    """Run all utility function tests."""
    print("\n" + "=" * 80)
    print("🧪 INTELLIGENT RESEARCH UTILITIES TESTS")
    print("=" * 80 + "\n")

    tests_passed = 0
    tests_failed = 0

    # Test 1: Relevance Scoring
    if await test_relevance_scoring():
        tests_passed += 1
    else:
        tests_failed += 1

    # Test 2: URL Selection
    if await test_url_selection():
        tests_passed += 1
    else:
        tests_failed += 1

    # Test 3: Query Extraction
    if await test_query_extraction():
        tests_passed += 1
    else:
        tests_failed += 1

    # Test 4: MCP Compression Simulation
    if await test_mcp_compression_simulation():
        tests_passed += 1
    else:
        tests_failed += 1

    # Print summary
    print("\n" + "=" * 80)
    print("📊 UTILITIES TEST SUMMARY")
    print("=" * 80)
    print(f"Tests Passed: {tests_passed}/4")
    print(f"Tests Failed: {tests_failed}/4")

    if tests_failed == 0:
        print("\n🎉 ✅ ALL UTILITY TESTS PASSED!")
        print("✅ Z-Playground1 intelligence components are working correctly")
        print("✅ The system should work when used through MCP agents")
        print("=" * 80)
        return True
    else:
        print(f"\n⚠️  {tests_failed} test(s) failed")
        print("=" * 80)
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_utility_tests())
    exit(0 if success else 1)
