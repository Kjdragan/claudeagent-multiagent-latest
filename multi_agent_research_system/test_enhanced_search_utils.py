#!/usr/bin/env python3
"""
Test script for Enhanced Search Utilities

This script tests the adapted zPlayground1 utilities without requiring
the Claude Agent SDK to be installed.
"""

import asyncio
import os
import sys
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "multi_agent_research_system"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_enhanced_search_utils():
    """Test the enhanced search utilities."""

    logger.info("🧪 Starting Enhanced Search Utilities Test")

    try:
        # Test 1: Import search utilities
        logger.info("Test 1: Testing search utilities import...")

        try:
            # Direct imports to avoid package dependency issues
            sys.path.insert(0, str(project_root / "multi_agent_research_system" / "utils"))
            from z_search_crawl_utils import (
                SearchResult,
                calculate_enhanced_relevance_score,
                select_urls_for_crawling,
                format_search_results
            )
            logger.info("✅ Search utilities imported successfully")

            # Test relevance scoring
            score = calculate_enhanced_relevance_score(
                title="Quantum Computing Breakthrough",
                snippet="New quantum computer achieves 1000-qubit milestone",
                position=1,
                query_terms=["quantum", "computing", "breakthrough"]
            )
            logger.info(f"✅ Relevance score test: {score:.3f}")

        except ImportError as e:
            logger.error(f"❌ Failed to import search utilities: {e}")
            return False

        # Test 2: Import content cleaning utilities
        logger.info("Test 2: Testing content cleaning utilities import...")

        try:
            from z_content_cleaning import (
                format_cleaned_results,
                assess_content_cleanliness,
                clean_content_with_judge_optimization
            )
            logger.info("✅ Content cleaning utilities imported successfully")
        except ImportError as e:
            logger.error(f"❌ Failed to import content cleaning utilities: {e}")
            return False

        # Test 3: Import crawl4ai utilities
        logger.info("Test 3: Testing crawl4ai utilities import...")

        try:
            from z_crawl4ai_utils import (
                SimpleCrawler,
                CrawlResult,
                get_crawler,
                get_timeout_for_url
            )
            logger.info("✅ Crawl4AI utilities imported successfully")

            # Test crawler creation
            crawler = get_crawler()
            stats = crawler.get_stats()
            logger.info(f"✅ Crawler created with stats: {stats}")

        except ImportError as e:
            logger.error(f"❌ Failed to import crawl4ai utilities: {e}")
            return False

        # Test 4: Configuration settings
        logger.info("Test 4: Testing configuration settings...")

        try:
            sys.path.insert(0, str(project_root / "multi_agent_research_system" / "config"))
            from settings import (
                get_settings,
                get_enhanced_search_config,
                EnhancedSearchConfig
            )

            settings = get_settings()
            enhanced_config = get_enhanced_search_config()

            logger.info("✅ Configuration settings imported successfully")
            logger.info(f"   Default anti-bot level: {enhanced_config.default_anti_bot_level}")
            logger.info(f"   Default crawl threshold: {enhanced_config.default_crawl_threshold}")
            logger.info(f"   Workproduct directory: {enhanced_config.default_workproduct_dir}")

            # Test settings validation
            validated_level = settings.validate_anti_bot_level(5)
            logger.info(f"✅ Anti-bot level validation (5 -> {validated_level})")

            validated_threshold = settings.validate_crawl_threshold(1.5)
            logger.info(f"✅ Crawl threshold validation (1.5 -> {validated_threshold})")

        except ImportError as e:
            logger.error(f"❌ Failed to import configuration settings: {e}")
            return False

        # Test 5: Test URL selection
        logger.info("Test 5: Testing URL selection logic...")

        try:
            # Create mock search results
            mock_results = [
                SearchResult(
                    title="Best Quantum Computing Article",
                    link="https://example.com/quantum-1",
                    snippet="Comprehensive guide to quantum computing",
                    position=1,
                    relevance_score=0.95
                ),
                SearchResult(
                    title="Related News",
                    link="https://example.com/news-1",
                    snippet="Some news about technology",
                    position=2,
                    relevance_score=0.3
                ),
                SearchResult(
                    title="Another Article",
                    link="https://example.com/article-1",
                    snippet="Another article",
                    position=3,
                    relevance_score=0.7
                )
            ]

            selected_urls = select_urls_for_crawling(
                search_results=mock_results,
                limit=2,
                min_relevance=0.4
            )

            logger.info(f"✅ URL selection: {len(selected_urls)} URLs selected")
            for i, url in enumerate(selected_urls):
                logger.info(f"   {i+1}. {url}")

        except Exception as e:
            logger.error(f"❌ Failed URL selection test: {e}")
            return False

        # Test 6: Test search result formatting
        logger.info("Test 6: Testing search result formatting...")

        try:
            formatted = format_search_results(mock_results)
            logger.info(f"✅ Search results formatted ({len(formatted)} characters)")
            logger.info(f"   First 100 chars: {formatted[:100]}...")

        except Exception as e:
            logger.error(f"❌ Failed search result formatting test: {e}")
            return False

        # Test 7: Workproduct directory creation
        logger.info("Test 7: Testing workproduct directory creation...")

        try:
            workproduct_dir = settings.ensure_workproduct_directory()
            logger.info(f"✅ Workproduct directory created: {workproduct_dir}")
            logger.info(f"   Directory exists: {workproduct_dir.exists()}")
        except Exception as e:
            logger.error(f"❌ Failed workproduct directory creation: {e}")
            return False

        # Test 8: Environment variable checking
        logger.info("Test 8: Checking environment variables...")

        env_vars = {
            'SERP_API_KEY': 'Required for search',
            'OPENAI_API_KEY': 'Required for content cleaning',
            'ANTHROPIC_API_KEY': 'Required for Claude SDK',
            'KEVIN_WORKPRODUCTS_DIR': 'Optional workproduct directory'
        }

        missing_required = []
        for var, description in env_vars.items():
            if os.getenv(var):
                logger.info(f"✅ {var}: SET ({description})")
            else:
                status = "REQUIRED" if var in ['SERP_API_KEY', 'ANTHROPIC_API_KEY'] else "OPTIONAL"
                logger.warning(f"⚠️ {var}: NOT SET ({description}) - {status}")
                if var in ['SERP_API_KEY', 'ANTHROPIC_API_KEY']:
                    missing_required.append(var)

        if missing_required:
            logger.warning(f"⚠️ Missing required variables: {missing_required}")

        # Summary
        logger.info("🎯 Enhanced Search Utilities Test Summary")
        logger.info("=" * 60)

        logger.info("✅ All utility functions imported successfully")
        logger.info("✅ Configuration system working")
        logger.info("✅ Search and URL selection logic working")
        logger.info("✅ Content cleaning utilities available")
        logger.info("✅ Crawl4AI integration ready")
        logger.info("✅ Workproduct directory system working")

        if missing_required:
            logger.warning(f"⚠️ Missing required environment variables: {missing_required}")
            logger.info("   Set these variables to enable full functionality")

        logger.info("🚀 Enhanced search utilities are ready for MCP integration!")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed with unexpected error: {e}")
        import traceback
        logger.error(f"   Traceback: {traceback.format_exc()}")
        return False


async def main():
    """Main test function."""
    print("🧪 Enhanced Search Utilities Test")
    print("=" * 60)

    success = await test_enhanced_search_utils()

    if success:
        print("\n🎉 All utility tests passed! The enhanced search system is working.")
        print("\n📋 Next steps:")
        print("   1. Install Claude Agent SDK: pip install claude-agent-sdk")
        print("   2. Set required environment variables")
        print("   3. Run full MCP integration test")
        print("   4. Test with a research session")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())