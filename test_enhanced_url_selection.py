#!/usr/bin/env python3
"""
Test Script for Enhanced URL Selection System

This script tests the new enhanced URL selection system that uses GPT-5 Nano
for query optimization and intelligent ranking algorithms.

Usage:
    python test_enhanced_url_selection.py
"""

import asyncio
import logging
import os
import sys
from datetime import datetime

# Add the multi_agent_research_system to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'multi_agent_research_system'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_query_enhancement():
    """Test the query enhancement module."""
    print("\n" + "="*60)
    print("TESTING QUERY ENHANCEMENT MODULE")
    print("="*60)

    try:
        from multi_agent_research_system.utils.query_enhancer import enhance_user_query

        test_queries = [
            "quantum computing applications in healthcare",
            "climate change impact on global agriculture",
            "artificial intelligence ethics in business"
        ]

        for query in test_queries:
            print(f"\n🔍 Original Query: '{query}'")

            try:
                enhanced_queries = await enhance_user_query(query, "test_session")

                print(f"✅ Enhanced Primary Query: '{enhanced_queries.primary_query}'")
                print(f"✅ Orthogonal Query 1: '{enhanced_queries.orthogonal_query_1}'")
                print(f"✅ Orthogonal Query 2: '{enhanced_queries.orthogonal_query_2}'")

            except Exception as e:
                print(f"❌ Query enhancement failed: {e}")
                print("   This may be due to missing OpenAI API key or network issues")

    except ImportError as e:
        print(f"❌ Failed to import query enhancer: {e}")


async def test_multi_stream_search():
    """Test the multi-stream search execution."""
    print("\n" + "="*60)
    print("TESTING MULTI-STREAM SEARCH")
    print("="*60)

    try:
        from multi_agent_research_system.utils.multi_stream_search import execute_multi_stream_search
        from multi_agent_research_system.utils.query_enhancer import enhance_user_query

        test_query = "renewable energy trends 2024"
        print(f"🔍 Test Query: '{test_query}'")

        # Step 1: Enhance the query
        try:
            enhanced_queries = await enhance_user_query(test_query, "test_session")
            print(f"✅ Query enhancement successful")
        except Exception as e:
            print(f"❌ Query enhancement failed: {e}")
            # Create mock enhanced queries for testing
            from multi_agent_research_system.utils.query_enhancer import EnhancedQueries
            enhanced_queries = EnhancedQueries(
                primary_query=test_query,
                orthogonal_query_1=f"{test_query} applications",
                orthogonal_query_2=f"{test_query} challenges",
                enhancement_metadata={"fallback": True}
            )
            print(f"🔄 Using mock enhanced queries for testing")

        # Step 2: Execute multi-stream search
        try:
            multi_results = await execute_multi_stream_search(
                enhanced_queries=enhanced_queries,
                session_id="test_session",
                search_type="search",
                result_distribution={"primary": 10, "orthogonal_1": 5, "orthogonal_2": 5}
            )

            print(f"✅ Multi-stream search completed")
            print(f"   Total results: {multi_results.total_results}")
            print(f"   Successful streams: {multi_results.successful_streams}")
            print(f"   Failed streams: {multi_results.failed_streams}")
            print(f"   Execution time: {multi_results.total_execution_time:.2f}s")

            # Show results by stream
            for priority, stream_result in multi_results.stream_results.items():
                print(f"   {priority.value}: {stream_result.num_received} results "
                      f"({'✅' if stream_result.success else '❌'})")

        except Exception as e:
            print(f"❌ Multi-stream search failed: {e}")
            print("   This may be due to missing SERP_API_KEY or network issues")

    except ImportError as e:
        print(f"❌ Failed to import multi-stream search: {e}")


async def test_intelligent_ranking():
    """Test the intelligent ranking algorithm."""
    print("\n" + "="*60)
    print("TESTING INTELLIGENT RANKING ALGORITHM")
    print("="*60)

    try:
        from multi_agent_research_system.utils.intelligent_ranker import create_master_ranked_list
        from multi_agent_research_system.utils.multi_stream_search import MultiSearchResults, SearchStreamResult, SearchPriority
        from multi_agent_research_system.utils.serp_search_utils import SearchResult

        # Create mock search results for testing
        mock_results = MultiSearchResults(
            stream_results={
                SearchPriority.PRIMARY: SearchStreamResult(
                    priority=SearchPriority.PRIMARY,
                    query="test query primary",
                    results=[
                        SearchResult("Primary Result 1", "https://example1.com", "Snippet 1", 1, relevance_score=0.9),
                        SearchResult("Primary Result 2", "https://example2.com", "Snippet 2", 2, relevance_score=0.8),
                        SearchResult("Primary Result 3", "https://example3.com", "Snippet 3", 3, relevance_score=0.7),
                    ],
                    num_requested=10,
                    num_received=3,
                    success=True
                ),
                SearchPriority.ORTHOGONAL_1: SearchStreamResult(
                    priority=SearchPriority.ORTHOGONAL_1,
                    query="test query orthogonal 1",
                    results=[
                        SearchResult("Orthogonal Result 1", "https://example4.com", "Snippet 4", 1, relevance_score=0.85),
                        SearchResult("Orthogonal Result 2", "https://example5.com", "Snippet 5", 2, relevance_score=0.75),
                    ],
                    num_requested=5,
                    num_received=2,
                    success=True
                )
            },
            total_results=5,
            successful_streams=2,
            failed_streams=0,
            total_execution_time=1.5,
            metadata={"test": True}
        )

        print(f"🔍 Testing with {mock_results.total_results} mock search results")

        # Test ranking
        ranked_results = create_master_ranked_list(mock_results, target_count=10)

        print(f"✅ Ranking completed successfully")
        print(f"   Ranked {len(ranked_results)} results")

        # Show top ranked results
        for i, result in enumerate(ranked_results[:5]):
            print(f"   {i+1}. {result.search_result.title} "
                  f"(Score: {result.composite_score:.3f}, "
                  f"Priority: {result.original_priority.value})")

    except ImportError as e:
        print(f"❌ Failed to import intelligent ranker: {e}")
    except Exception as e:
        print(f"❌ Intelligent ranking test failed: {e}")


async def test_enhanced_url_selector():
    """Test the complete enhanced URL selection system."""
    print("\n" + "="*60)
    print("TESTING ENHANCED URL SELECTOR (COMPLETE SYSTEM)")
    print("="*60)

    try:
        from multi_agent_research_system.utils.serp_search_utils import select_urls_for_crawling_enhanced

        test_query = "machine learning in finance"
        print(f"🔍 Test Query: '{test_query}'")

        try:
            # Test the complete enhanced selection system
            selected_urls = await select_urls_for_crawling_enhanced(
                query=test_query,
                session_id="test_complete_session",
                target_count=20,
                search_type="search",
                use_enhanced_selection=True,
                fallback_on_failure=True
            )

            if selected_urls:
                print(f"✅ Enhanced URL selection successful!")
                print(f"   Selected {len(selected_urls)} URLs:")
                for i, url in enumerate(selected_urls[:10]):
                    print(f"   {i+1}. {url}")
                if len(selected_urls) > 10:
                    print(f"   ... and {len(selected_urls) - 10} more URLs")
            else:
                print(f"❌ Enhanced URL selection returned no URLs")

        except Exception as e:
            print(f"❌ Enhanced URL selection failed: {e}")
            print("   This may be due to missing API keys or network issues")

    except ImportError as e:
        print(f"❌ Failed to import enhanced URL selector: {e}")


def check_environment():
    """Check if required environment variables are set."""
    print("\n" + "="*60)
    print("ENVIRONMENT CHECK")
    print("="*60)

    # Check OpenAI API Key
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print(f"✅ OPENAI_API_KEY: {'*' * 20}...{openai_key[-4:]}")
    else:
        print("❌ OPENAI_API_KEY: Not set (query enhancement will fail)")

    # Check SERP API Key
    serp_key = os.getenv("SERP_API_KEY")
    if serp_key:
        print(f"✅ SERP_API_KEY: {'*' * 20}...{serp_key[-4:]}")
    else:
        print("❌ SERP_API_KEY: Not set (search functionality will fail)")

    print("\n📝 Note: Missing API keys will cause some tests to fail, but the system can still be validated")


async def main():
    """Main test function."""
    print("🚀 ENHANCED URL SELECTION SYSTEM TEST")
    print("="*60)
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Check environment
    check_environment()

    # Run tests
    await test_query_enhancement()
    await test_multi_stream_search()
    await test_intelligent_ranking()
    await test_enhanced_url_selector()

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("✅ Test completed. Check results above for any ❌ indicators.")
    print("📝 If tests failed due to missing API keys, that's expected.")
    print("   The system architecture can still be validated from the import tests.")
    print("\n🎯 Next Steps:")
    print("1. Set OPENAI_API_KEY for query enhancement functionality")
    print("2. Set SERP_API_KEY for search functionality")
    print("3. Run integration tests with real API calls")
    print("4. Compare results with traditional URL selection")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()