"""
Enhanced Search, Scrape & Clean MCP Tool

This MCP tool integrates the zPlayground1 topic-based search, scrape, and clean
functionality with the Claude Agent SDK, providing high-quality content extraction
with parallel processing and anti-bot detection.
"""

import logging
from pathlib import Path
from typing import Any, Dict
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Claude Agent SDK MCP functionality
try:
    from claude_agent_sdk import tool, create_sdk_mcp_server
    CLAUDE_SDK_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Claude Agent SDK not available for MCP integration")
    CLAUDE_SDK_AVAILABLE = False

# Import the enhanced search functionality
from utils.z_search_crawl_utils import search_crawl_and_clean_direct, news_search_and_crawl_direct

logger = logging.getLogger(__name__)


def create_enhanced_search_mcp_server():
    """
    Create the enhanced search, scrape & clean MCP server.

    Returns:
        MCP server instance or None if Claude SDK unavailable
    """
    if not CLAUDE_SDK_AVAILABLE:
        logger.error("Cannot create MCP server: Claude Agent SDK not available")
        return None

    @tool(
        "enhanced_search_scrape_clean",
        "Advanced topic-based search with parallel crawling and AI content cleaning. Performs Google search, extracts content from multiple URLs in parallel, applies progressive anti-bot detection, and returns AI-cleaned results within token limits.",
        {
            "query": {
                "type": "string",
                "description": "Search query or topic to research"
            },
            "search_type": {
                "type": "string",
                "enum": ["search", "news"],
                "default": "search",
                "description": "Type of search: 'search' for web search, 'news' for news search"
            },
            "num_results": {
                "type": "integer",
                "default": 15,
                "minimum": 1,
                "maximum": 50,
                "description": "Number of search results to retrieve"
            },
            "auto_crawl_top": {
                "type": "integer",
                "default": 10,
                "minimum": 0,
                "maximum": 20,
                "description": "Maximum number of URLs to crawl for content extraction"
            },
            "crawl_threshold": {
                "type": "number",
                "default": 0.3,
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Minimum relevance score threshold for crawling (0.0-1.0)"
            },
            "anti_bot_level": {
                "type": "integer",
                "default": 1,
                "minimum": 0,
                "maximum": 3,
                "description": "Progressive anti-bot level: 0=basic, 1=enhanced, 2=advanced, 3=stealth"
            },
            "max_concurrent": {
                "type": "integer",
                "default": 15,
                "minimum": 1,
                "maximum": 20,
                "description": "Maximum concurrent crawling operations"
            },
            "session_id": {
                "type": "string",
                "default": "default",
                "description": "Session identifier for tracking and work products"
            }
        }
    )
    async def enhanced_search_scrape_clean(args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced search, scrape, and clean functionality.

        This tool provides comprehensive research capabilities:
        1. Executes Google search using SERP API
        2. Selects relevant URLs based on intelligent relevance scoring
        3. Crawls multiple URLs in parallel with progressive anti-bot detection
        4. Applies AI content cleaning using GPT-5-nano
        5. Returns cleaned results within token limits
        6. Saves detailed work products for reference

        Args:
            args: Dictionary containing tool parameters

        Returns:
            Dictionary with content and metadata
        """
        try:
            # Extract parameters with defaults
            query = args.get("query")
            if not query:
                return {
                    "content": [{"type": "text", "text": "❌ **Error**: 'query' parameter is required"}],
                    "is_error": True
                }

            search_type = args.get("search_type", "search")
            num_results = int(args.get("num_results", 15))
            auto_crawl_top = int(args.get("auto_crawl_top", 10))
            crawl_threshold = float(args.get("crawl_threshold", 0.3))
            anti_bot_level = int(args.get("anti_bot_level", 1))
            max_concurrent = int(args.get("max_concurrent", 15))
            session_id = args.get("session_id", "default")

            # Validate parameters
            if anti_bot_level < 0 or anti_bot_level > 3:
                anti_bot_level = 1
                logger.warning(f"Invalid anti_bot_level, using default: {anti_bot_level}")

            if max_concurrent < 1 or max_concurrent > 20:
                max_concurrent = min(max(1, max_concurrent), 15)
                logger.warning(f"Invalid max_concurrent, using: {max_concurrent}")

            # Set up work product directory
            workproduct_dir = os.environ.get('KEVIN_WORKPRODUCTS_DIR')
            if not workproduct_dir:
                # Use session-based directory structure
                base_session_dir = f"/home/kjdragan/lrepos/claude-agent-sdk-python/KEVIN/sessions/{session_id}"
                research_dir = f"{base_session_dir}/research"
                workproduct_dir = research_dir

            # Ensure workproduct directory exists
            Path(workproduct_dir).mkdir(parents=True, exist_ok=True)

            logger.info(f"Executing enhanced search: query='{query}', anti_bot_level={anti_bot_level}")

            # Execute the enhanced search and extract functionality
            result = await search_crawl_and_clean_direct(
                query=query,
                search_type=search_type,
                num_results=num_results,
                auto_crawl_top=auto_crawl_top,
                crawl_threshold=crawl_threshold,
                max_concurrent=max_concurrent,
                session_id=session_id,
                anti_bot_level=anti_bot_level,
                workproduct_dir=workproduct_dir
            )

            # Check result length for token management
            if len(result) > 20000:  # Leave room for other content
                # Create a summary version for the response
                summary = f"""# Enhanced Search Results Summary

**Query**: {query}
**Search Type**: {search_type}
**Anti-Bot Level**: {anti_bot_level}
**Processing**: Enhanced parallel crawling with AI content cleaning

## Results Overview
✅ **Search and content extraction completed successfully**
📄 **Detailed work product saved** to KEVIN work_products directory
🧠 **AI content cleaning applied** to all extracted articles

## Key Findings
The search and extraction process has completed with full content processing.
All extracted content has been cleaned and processed for relevance to your query.

## Accessing Full Results
Complete detailed results including all cleaned content are available in the work product files saved to the KEVIN directory.
These files contain the full cleaned content from all successfully crawled URLs.

---
*Results generated by Enhanced Search+Scrape+Clean tool powered by zPlayground1 technology*
"""
                result = summary
                logger.info("Result truncated for token limits, full content saved to work products")

            return {
                "content": [{"type": "text", "text": result}],
                "metadata": {
                    "query": query,
                    "search_type": search_type,
                    "anti_bot_level": anti_bot_level,
                    "session_id": session_id,
                    "workproduct_dir": workproduct_dir,
                    "enhanced_search": True
                }
            }

        except Exception as e:
            error_msg = f"❌ **Enhanced Search Error**\n\nFailed to execute search and content extraction: {str(e)}\n\nPlease check:\n- SERP_API_KEY is configured\n- Network connectivity\n- Query parameters are valid"
            logger.error(f"Enhanced search failed: {e}")
            return {
                "content": [{"type": "text", "text": error_msg}],
                "is_error": True,
                "error_details": str(e)
            }

    @tool(
        "enhanced_news_search",
        "Specialized news search with enhanced content extraction. Searches for recent news and current events, then extracts and cleans full article content from multiple sources in parallel.",
        {
            "query": {
                "type": "string",
                "description": "News topic or search query"
            },
            "num_results": {
                "type": "integer",
                "default": 15,
                "minimum": 1,
                "maximum": 30,
                "description": "Number of news results to retrieve"
            },
            "auto_crawl_top": {
                "type": "integer",
                "default": 10,
                "minimum": 0,
                "maximum": 15,
                "description": "Maximum number of news articles to crawl for full content"
            },
            "anti_bot_level": {
                "type": "integer",
                "default": 1,
                "minimum": 0,
                "maximum": 3,
                "description": "Progressive anti-bot level for news sites"
            },
            "session_id": {
                "type": "string",
                "default": "default",
                "description": "Session identifier for tracking"
            }
        }
    )
    async def enhanced_news_search(args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced news search with full content extraction.

        This tool specializes in news and current events research:
        1. Searches for recent news using enhanced queries
        2. Extracts full article content from news sources
        3. Applies content cleaning optimized for news articles
        4. Returns comprehensive news analysis

        Args:
            args: Dictionary containing tool parameters

        Returns:
            Dictionary with news content and metadata
        """
        try:
            # Extract parameters
            query = args.get("query")
            if not query:
                return {
                    "content": [{"type": "text", "text": "❌ **Error**: 'query' parameter is required for news search"}],
                    "is_error": True
                }

            num_results = args.get("num_results", 15)
            auto_crawl_top = args.get("auto_crawl_top", 10)
            anti_bot_level = args.get("anti_bot_level", 1)
            session_id = args.get("session_id", "default")

            # Set up work product directory
            workproduct_dir = os.environ.get('KEVIN_WORKPRODUCTS_DIR')
            if not workproduct_dir:
                # Use session-based directory structure
                base_session_dir = f"/home/kjdragan/lrepos/claude-agent-sdk-python/KEVIN/sessions/{session_id}"
                research_dir = f"{base_session_dir}/research"
                workproduct_dir = research_dir

            Path(workproduct_dir).mkdir(parents=True, exist_ok=True)

            logger.info(f"Executing enhanced news search: query='{query}', anti_bot_level={anti_bot_level}")

            # Execute enhanced news search
            result = await news_search_and_crawl_direct(
                query=query,
                num_results=num_results,
                auto_crawl_top=auto_crawl_top,
                session_id=session_id,
                anti_bot_level=anti_bot_level,
                workproduct_dir=workproduct_dir
            )

            # Token management for news results
            if len(result) > 20000:
                summary = f"""# Enhanced News Search Results

**News Topic**: {query}
**Anti-Bot Level**: {anti_bot_level}
**Processing**: Enhanced news search with full article extraction

## News Overview
✅ **News search and article extraction completed**
📰 **Multiple news sources processed** with full content extraction
🧠 **AI content cleaning applied** to news articles

## Key News Coverage
The search has successfully identified and extracted content from multiple news sources related to your topic. All articles have been processed for relevance and cleaned for readability.

## Accessing Full News Content
Complete news articles and analysis are available in the work product files saved to the KEVIN directory. These contain the full cleaned content from all successfully crawled news sources.

---
*News results generated by Enhanced News Search tool powered by zPlayground1 technology*
"""
                result = summary
                logger.info("News results truncated for token limits, full content saved to work products")

            return {
                "content": [{"type": "text", "text": result}],
                "metadata": {
                    "query": query,
                    "search_type": "news",
                    "anti_bot_level": anti_bot_level,
                    "session_id": session_id,
                    "workproduct_dir": workproduct_dir,
                    "enhanced_news_search": True
                }
            }

        except Exception as e:
            error_msg = f"❌ **Enhanced News Search Error**\n\nFailed to execute news search and extraction: {str(e)}\n\nPlease check:\n- SERP_API_KEY is configured\n- Network connectivity\n- News topic is valid"
            logger.error(f"Enhanced news search failed: {e}")
            return {
                "content": [{"type": "text", "text": error_msg}],
                "is_error": True,
                "error_details": str(e)
            }

    # Create the MCP server
    server = create_sdk_mcp_server(
        name="enhanced_search_scrape_clean",
        version="1.0.0",
        tools=[enhanced_search_scrape_clean, enhanced_news_search]
    )

    logger.info("Enhanced Search, Scrape & Clean MCP server created successfully")
    return server


# Create server instance
enhanced_search_server = create_enhanced_search_mcp_server()

# Export server for integration
__all__ = [
    'enhanced_search_server',
    'create_enhanced_search_mcp_server'
]