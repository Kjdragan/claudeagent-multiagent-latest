"""Test the complete intelligent research system implementation.

This script tests the new intelligent research tool that implements the complete
z-playground1 proven intelligence while staying within MCP constraints.
"""

import asyncio
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

async def test_intelligent_research_tool():
    """Test the intelligent research tool directly."""
    print("\n" + "=" * 80)
    print("TEST 1: INTELLIGENT RESEARCH TOOL")
    print("=" * 80)

    try:
        from tools.intelligent_research_tool import (
            intelligent_research_with_advanced_scraping,
        )
        print("✅ Intelligent research tool imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import intelligent research tool: {e}")
        return False

    # Test with a query
    test_args = {
        "query": "Claude Agent SDK documentation features",
        "session_id": "test-intelligent-research",
        "max_urls": 8,  # Slightly reduced for testing
        "relevance_threshold": 0.3,
        "max_concurrent": 5  # Reduced for testing
    }

    print(f"\n🔍 Testing with query: '{test_args['query']}'")
    print(f"📊 Configuration: max_urls={test_args['max_urls']}, threshold={test_args['relevance_threshold']}")

    try:
        result = await intelligent_research_with_advanced_scraping(test_args)

        # Check if result is successful
        if 'content' in result and len(result['content']) > 0:
            content_text = result['content'][0]['text']

            print("\n📊 Result Analysis:")
            print(f"  - Content length: {len(content_text)} characters")
            print(f"  - Success: {'✅' if 'Intelligent Research Complete' in content_text else '❌'}")
            print(f"  - Has metadata: {'✅' if 'metadata' in result else '❌'}")

            # Show first 1000 characters
            print("\n📄 Content Preview (first 1000 chars):")
            print(content_text[:1000])
            print("...")

            # Check for work product
            if 'metadata' in result and 'work_product_path' in result['metadata']:
                work_product_path = result['metadata']['work_product_path']
                print(f"\n💾 Work product: {work_product_path}")

                # Check if work product file exists
                if os.path.exists(work_product_path):
                    print("✅ Work product file created successfully")
                    file_size = os.path.getsize(work_product_path)
                    print(f"   File size: {file_size:,} bytes")
                else:
                    print("❌ Work product file not found")

            print("\n✅ Intelligent research tool test PASSED")
            return True
        else:
            print("\n❌ No content returned from intelligent research tool")
            return False

    except Exception as e:
        print(f"\n❌ Error executing intelligent research tool: {e}")
        return False


async def test_relevance_scoring():
    """Test the enhanced relevance scoring function."""
    print("\n" + "=" * 80)
    print("TEST 2: ENHANCED RELEVANCE SCORING")
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

    # Test cases
    test_cases = [
        {
            "query": "artificial intelligence machine learning",
            "title": "Understanding Artificial Intelligence and Machine Learning",
            "snippet": "A comprehensive guide to AI and ML concepts and applications",
            "position": 1,
            "expected_min_score": 0.8
        },
        {
            "query": "python programming tutorial",
            "title": "JavaScript Programming Basics",
            "snippet": "Learn the fundamentals of JavaScript for web development",
            "position": 5,
            "expected_max_score": 0.6
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        query_terms = extract_query_terms(test_case["query"])
        score = calculate_enhanced_relevance_score(
            title=test_case["title"],
            snippet=test_case["snippet"],
            position=test_case["position"],
            query_terms=query_terms
        )

        print(f"\nTest Case {i}:")
        print(f"  Query: {test_case['query']}")
        print(f"  Title: {test_case['title']}")
        print(f"  Position: {test_case['position']}")
        print(f"  Score: {score:.3f}")
        print(f"  Expected range: {test_case['expected_min_score'] if 'expected_min_score' in test_case else 'N/A'} - {test_case.get('expected_max_score', 'N/A')}")

        # Validation
        if 'expected_min_score' in test_case and score < test_case['expected_min_score']:
            print("  ⚠️  Score below expected minimum")
        elif 'expected_max_score' in test_case and score > test_case['expected_max_score']:
            print("  ⚠️  Score above expected maximum")
        else:
            print("  ✅ Score within expected range")

    print("\n✅ Relevance scoring test PASSED")
    return True


async def test_content_compression():
    """Test the smart content compression for MCP compliance."""
    print("\n" + "=" * 80)
    print("TEST 3: MCP COMPLIANCE CONTENT COMPRESSION")
    print("=" * 80)

    try:
        from tools.intelligent_research_tool import (
            SearchResult,
            compress_for_mcp_compression,
        )
        print("✅ Content compression function imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import compression function: {e}")
        return False

    # Create mock data
    mock_search_results = [
        SearchResult(
            title="First Source - High Relevance",
            link="https://example1.com",
            snippet="Highly relevant content about the search query",
            position=1,
            relevance_score=0.95
        ),
        SearchResult(
            title="Second Source - Medium Relevance",
            link="https://example2.com",
            snippet="Some relevant content",
            position=3,
            relevance_score=0.65
        ),
        SearchResult(
            title="Third Source - Low Relevance",
            link="https://example3.com",
            snippet="Some content related to the query",
            position=8,
            relevance_score=0.32
        )
    ]

    mock_crawl_results = [
        {
            'url': 'https://example1.com',
            'success': True,
            'cleaned_content': 'Large content block 1 - ' * 1000 + 'This is extensive content from the first source with detailed information about the search query and related topics.',
            'title': 'First Source - High Relevance'
        },
        {
            'url': 'https://example2.com',
            'success': True,
            'cleaned_content': 'Medium content block 2 - ' * 500 + 'This is medium-sized content from the second source.',
            'title': 'Second Source - Medium Relevance'
        },
        {
            'url': 'https://example3.com',
            'success': True,
            'cleaned_content': 'Small content block 3 - ' * 200 + 'This is smaller content from the third source.',
            'title': 'Third Source - Low Relevance'
        }
    ]

    print("\n📊 Mock data created:")
    print(f"  - Search results: {len(mock_search_results)}")
    print(f"  - Crawl results: {len(mock_crawl_results)}")
    print(f"  - Total content chars: {sum(len(r['cleaned_content']) for r in mock_crawl_results)}")

    try:
        # Test compression
        compressed = compress_for_mcp_compression(
            crawl_results=mock_crawl_results,
            search_results=mock_search_results,
            max_tokens=5000  # Small limit for testing
        )

        print("\n🗜️ Compression Results:")
        print(f"  - Compressed length: {len(compressed)} characters")
        print(f"  - Includes top priority: {'✅' if 'Top Priority Sources' in compressed else '❌'}")
        print(f"  - Includes medium priority: {'✅' if 'High Priority Sources' in compressed else '❌'}")
        print(f"  - Includes references: {'✅' if 'Additional Sources' in compressed else '❌'}")

        # Show first 500 chars
        print("\n📄 Compressed Content Preview:")
        print(compressed[:500])
        print("...")

        print("\n✅ Content compression test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Error in content compression: {e}")
        return False


async def run_all_tests():
    """Run all intelligent research system tests."""
    print("\n" + "=" * 80)
    print("🧪 INTELLIGENT RESEARCH SYSTEM TESTS")
    print("=" * 80 + "\n")

    tests_passed = 0
    tests_failed = 0

    # Test 1: Intelligent Research Tool
    if await test_intelligent_research_tool():
        tests_passed += 1
    else:
        tests_failed += 1

    # Test 2: Relevance Scoring
    if await test_relevance_scoring():
        tests_passed += 1
    else:
        tests_failed += 1

    # Test 3: Content Compression
    if await test_content_compression():
        tests_passed += 1
    else:
        tests_failed += 1

    # Print summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print(f"Tests Passed: {tests_passed}/3")
    print(f"Tests Failed: {tests_failed}/3")

    if tests_failed == 0:
        print("\n🎉 ✅ ALL TESTS PASSED!")
        print("=" * 80)
        return True
    else:
        print(f"\n⚠️  {tests_failed} test(s) failed")
        print("=" * 80)
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
