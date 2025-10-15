"""
Enhanced Search, Scrape & Clean MCP Tool

This MCP tool integrates the zPlayground1 topic-based search, scrape, and clean
functionality with the Claude Agent SDK, providing high-quality content extraction
with parallel processing and anti-bot detection.
"""

# ruff: noqa: E402

import logging
import os
import sys
from pathlib import Path
from typing import Any

# Simple chunking statistics tracking
chunking_stats = {
    "total_calls": 0,
    "chunking_triggered": 0,
    "total_content_chars": 0,
    "total_chunks_created": 0,
}

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.resolve()))

# Import the enhanced search functionality
from ..utils.serp_search_utils import expanded_query_search_and_extract
from ..utils.z_search_crawl_utils import (
    news_search_and_crawl_direct,
    search_crawl_and_clean_direct,
)

# Import Claude Agent SDK MCP functionality
try:
    from claude_agent_sdk import create_sdk_mcp_server, tool

    CLAUDE_SDK_AVAILABLE = True
except ImportError:
    CLAUDE_SDK_AVAILABLE = False

logger = logging.getLogger(__name__)

if not CLAUDE_SDK_AVAILABLE:
    logger.warning("Claude Agent SDK not available for MCP integration")


def create_adaptive_chunks(
    content: str, query: str, max_chunk_size: int = 18000
) -> list[dict[str, Any]]:
    """
    Create adaptive chunks for large content to avoid token limits.

    This function intelligently splits content into chunks within the specified size limit,
    preferring logical break points (section headers, article boundaries) but enforcing
    the size limit strictly.

    Args:
        content: Full content to be chunked
        query: Original search query for context
        max_chunk_size: Maximum characters per chunk (conservative limit)

    Returns:
        List of content blocks ready for MCP response
    """
    # Track statistics
    chunking_stats["total_calls"] += 1
    chunking_stats["total_content_chars"] += len(content)

    content_length = len(content)

    if content_length <= max_chunk_size:
        logger.info(
            f"📊 Chunking Stats: Call #{chunking_stats['total_calls']} - "
            f"No chunking needed ({content_length:,} chars <= {max_chunk_size:,} limit)"
        )
        return [{"type": "text", "text": content}]

    # Track when chunking is triggered
    chunking_stats["chunking_triggered"] += 1

    content_blocks = []
    lines = content.split("\n")
    current_chunk = ""
    chunk_number = 1

    # Add header to first chunk
    total_chunks = (len(content) // max_chunk_size) + 1
    header = (
        f"# Expanded Query Search Results - Part {chunk_number} of {total_chunks}\n\n"
    )
    current_chunk = header

    for line in lines:
        # Check if adding this line would exceed the limit
        test_chunk = current_chunk + line + "\n"

        if len(test_chunk) > max_chunk_size:
            # Finalize current chunk
            content_blocks.append({"type": "text", "text": current_chunk.rstrip()})

            # Start new chunk
            chunk_number += 1
            header = f"# Expanded Query Search Results - Part {chunk_number} of {total_chunks}\n\n"
            current_chunk = header + line + "\n"
        else:
            current_chunk += line + "\n"

    # Add the final chunk
    if current_chunk.strip():
        # Remove the header from the last chunk if it's just the header
        if current_chunk.strip() == header.strip():
            current_chunk = f"# Expanded Query Search Results - Part {chunk_number} of {total_chunks}\n\n*Content continued from previous part*"

        content_blocks.append({"type": "text", "text": current_chunk.rstrip()})

    # Add context footer to each chunk except the last
    for i in range(len(content_blocks) - 1):
        content_blocks[i]["text"] += (
            f"\n\n---\n*Part {i + 1} of {len(content_blocks)} - Continued in next part*"
        )

    # Add completion footer to last chunk
    if content_blocks:
        content_blocks[-1]["text"] += (
            f"\n\n---\n*Complete search results for query: '{query}'*"
        )

    # Update statistics
    chunking_stats["total_chunks_created"] += len(content_blocks)

    # Log detailed chunking statistics
    chunking_rate = (
        chunking_stats["chunking_triggered"] / chunking_stats["total_calls"]
    ) * 100
    avg_content_size = (
        chunking_stats["total_content_chars"] / chunking_stats["total_calls"]
    )
    avg_chunks_per_chunking = chunking_stats["total_chunks_created"] / max(
        chunking_stats["chunking_triggered"], 1
    )

    logger.info(
        f"📊 Chunking Triggered: Call #{chunking_stats['total_calls']} - "
        f"Split {content_length:,} chars into {len(content_blocks)} chunks "
        f"(avg {content_length // len(content_blocks):,} chars/chunk)"
    )

    logger.info(
        f"📈 Overall Stats: {chunking_rate:.1f}% chunking rate, "
        f"avg {avg_content_size:,.0f} chars/call, "
        f"avg {avg_chunks_per_chunking:.1f} chunks when chunking"
    )

    return content_blocks


def _rename_workproduct_for_chunking(workproduct_dir: str, session_id: str) -> None:
    """Rename the most recent work product file to indicate chunking was used."""
    try:
        research_dir = Path(workproduct_dir)
        if not research_dir.exists():
            return

        # Find the most recent work product file (multiple patterns)
        patterns = [
            "1-expanded_search_workproduct_*.md",
            "1-search_workproduct_*.md"
        ]

        workproduct_files = []
        for pattern in patterns:
            workproduct_files.extend(research_dir.glob(pattern))

        if not workproduct_files:
            return

        # Sort by modification time and get the most recent
        latest_file = max(workproduct_files, key=lambda f: f.stat().st_mtime)

        # Check if it already has chunking indicator
        if "_chunked" in latest_file.name:
            return

        # Create new filename with chunking indicator
        parts = latest_file.stem.split("_")
        if len(parts) >= 4 and "workproduct" in parts and parts[-1].isdigit():
            # Find the workproduct index and insert "_chunked" before the timestamp parts
            workproduct_idx = parts.index("workproduct")
            if workproduct_idx < len(parts) - 1:
                # Insert "_chunked" after "workproduct" but before timestamp
                new_name = f"{'_'.join(parts[:workproduct_idx+1])}_chunked_{'_'.join(parts[workproduct_idx+1:])}.md"
                new_path = latest_file.parent / new_name

            # Rename the file
            latest_file.rename(new_path)
            logger.info(f"📦 Renamed work product file to indicate chunking: {new_name}")

            # Also add chunking indicator to the file content
            try:
                with open(new_path, encoding='utf-8') as f:
                    content = f.read()

                # Add chunking indicator to the title
                updated_content = content.replace(
                    "# Expanded Query Search Results Work Product",
                    "# Expanded Query Search Results Work Product 📦"
                )

                # Add chunking info to metadata section
                chunking_info = "\n**Response Chunking**: Yes - Content was split for MCP token limits"
                metadata_end = content.find("**Processing Time**:")
                if metadata_end != -1:
                    processing_line_end = content.find("\n", metadata_end)
                    if processing_line_end != -1:
                        updated_content = (
                            updated_content[:processing_line_end] +
                            chunking_info +
                            updated_content[processing_line_end:]
                        )

                with open(new_path, 'w', encoding='utf-8') as f:
                    f.write(updated_content)

                logger.info("📦 Added chunking indicators to work product content")

            except Exception as e:
                logger.warning(f"Could not update work product content with chunking info: {e}")

    except Exception as e:
        logger.warning(f"Could not rename work product file for chunking: {e}")


def get_chunking_stats() -> dict[str, Any]:
    """Get current chunking statistics."""
    if chunking_stats["total_calls"] == 0:
        return {"message": "No chunking calls made yet"}

    chunking_rate = (
        chunking_stats["chunking_triggered"] / chunking_stats["total_calls"]
    ) * 100
    avg_content_size = (
        chunking_stats["total_content_chars"] / chunking_stats["total_calls"]
    )
    avg_chunks_per_chunking = chunking_stats["total_chunks_created"] / max(
        chunking_stats["chunking_triggered"], 1
    )

    return {
        "total_calls": chunking_stats["total_calls"],
        "chunking_triggered": chunking_stats["chunking_triggered"],
        "chunking_rate_percent": round(chunking_rate, 1),
        "total_content_processed": f"{chunking_stats['total_content_chars']:,} chars",
        "average_content_size": f"{avg_content_size:,.0f} chars",
        "total_chunks_created": chunking_stats["total_chunks_created"],
        "average_chunks_when_chunking": round(avg_chunks_per_chunking, 1),
    }


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
                "description": "Search query or topic to research",
            },
            "search_type": {
                "type": "string",
                "enum": ["search", "news"],
                "default": "search",
                "description": "Type of search: 'search' for web search, 'news' for news search",
            },
            "num_results": {
                "type": "integer",
                "default": 30,
                "minimum": 1,
                "maximum": 50,
                "description": "Number of search results to retrieve (increased for better coverage)",
            },
            "auto_crawl_top": {
                "type": "integer",
                "default": 20,
                "minimum": 0,
                "maximum": 50,
                "description": "Maximum number of URLs to crawl for content extraction (increased for comprehensive research)",
            },
            "crawl_threshold": {
                "type": "number",
                "default": 0.3,
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Minimum relevance score threshold for crawling (0.0-1.0)",
            },
            "anti_bot_level": {
                "type": "integer",
                "default": 1,
                "minimum": 0,
                "maximum": 3,
                "description": "Progressive anti-bot level: 0=basic, 1=enhanced, 2=advanced, 3=stealth",
            },
            "max_concurrent": {
                "type": "integer",
                "default": 15,
                "minimum": 1,
                "maximum": 20,
                "description": "Maximum concurrent crawling operations",
            },
            "session_id": {
                "type": "string",
                "default": "default",
                "description": "Session identifier for tracking and work products",
            },
            "workproduct_prefix": {
                "type": "string",
                "default": "",
                "description": "Optional prefix for work product filenames (e.g., 'editor research' for editorial work)",
            },
        },
    )
    async def enhanced_search_scrape_clean(args: dict[str, Any]) -> dict[str, Any]:
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
                    "content": [
                        {
                            "type": "text",
                            "text": "❌ **Error**: 'query' parameter is required",
                        }
                    ],
                    "is_error": True,
                }

            search_type = args.get("search_type", "search")
            num_results = int(args.get("num_results", 15))
            auto_crawl_top = int(args.get("auto_crawl_top", 10))
            crawl_threshold = float(args.get("crawl_threshold", 0.3))
            anti_bot_level = int(args.get("anti_bot_level", 1))
            max_concurrent = int(args.get("max_concurrent", 15))
            session_id = args.get("session_id", "default")

            # Check research threshold before proceeding
            try:
                from ..utils.research_threshold_tracker import check_search_threshold
                intervention = await check_search_threshold(session_id, query, "enhanced_search")
                if intervention:
                    # Return intervention message instead of searching
                    logger.info(f"🎯 Threshold intervention triggered for enhanced search session {session_id}")
                    return {
                        "content": [
                            {
                                "type": "text",
                                "text": intervention
                            }
                        ],
                        "threshold_intervention": True,
                        "search_type": "enhanced_search",
                        "session_id": session_id
                    }
            except ImportError as e:
                logger.warning(f"⚠️  Could not import threshold tracker: {e}")

            # Validate parameters
            if anti_bot_level < 0 or anti_bot_level > 3:
                anti_bot_level = 1
                logger.warning(
                    f"Invalid anti_bot_level, using default: {anti_bot_level}"
                )

            if max_concurrent < 1 or max_concurrent > 20:
                max_concurrent = min(max(1, max_concurrent), 15)
                logger.warning(f"Invalid max_concurrent, using: {max_concurrent}")

            # Set up work product directory
            workproduct_dir = os.environ.get("KEVIN_WORKPRODUCTS_DIR")
            if not workproduct_dir:
                # Use session-based directory structure
                # Use environment-aware path detection
                current_repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                if "claudeagent-multiagent-latest" in current_repo:
                    # Running from claudeagent-multiagent-latest
                    base_session_dir = f"{current_repo}/KEVIN/sessions/{session_id}"
                else:
                    # Fallback to new repository structure
                    base_session_dir = f"/home/kjdragan/lrepos/claudeagent-multiagent-latest/KEVIN/sessions/{session_id}"
                research_dir = f"{base_session_dir}/research"
                workproduct_dir = research_dir

            # Ensure workproduct directory exists
            Path(workproduct_dir).mkdir(parents=True, exist_ok=True)

            logger.info(
                f"Executing enhanced search: query='{query}', anti_bot_level={anti_bot_level}"
            )

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
                workproduct_dir=workproduct_dir,
            )

            # Extract scrape count from result text for budget tracking
            import re
            successful_scrapes = 0
            urls_attempted = 0

            # Pattern 1: "URLs Crawled: X successfully processed"
            match = re.search(r'\*\*URLs Crawled\*\*:\s*(\d+)\s+successfully', result)
            if match:
                successful_scrapes = int(match.group(1))

            # Pattern 2: "Successful crawls: X"
            if not successful_scrapes:
                match = re.search(r'Successful crawls:\s*(\d+)', result)
                if match:
                    successful_scrapes = int(match.group(1))

            # Pattern 3: "URLs selected for crawling: X"
            match = re.search(r'URLs selected for crawling:\s*(\d+)', result)
            if match:
                urls_attempted = int(match.group(1))

            # Check result length for token management
            if len(result) > 20000:  # Leave room for other content
                # Use adaptive chunking to split content into multiple blocks
                content_blocks = create_adaptive_chunks(result, query)
                logger.info(
                    f"Enhanced search content split into {len(content_blocks)} chunks for token management"
                )

                # Rename work product file to indicate chunking was used
                if workproduct_dir:
                    _rename_workproduct_for_chunking(workproduct_dir, session_id)

                return {
                    "content": content_blocks,
                    "metadata": {
                        "query": query,
                        "search_type": search_type,
                        "anti_bot_level": anti_bot_level,
                        "session_id": session_id,
                        "workproduct_dir": workproduct_dir,
                        "enhanced_search": True,
                        "chunked_content": True,
                        "total_chunks": len(content_blocks),
                        "successful_scrapes": successful_scrapes,
                        "urls_attempted": urls_attempted,
                        "search_queries_executed": 1,
                    },
                }
            else:
                return {
                    "content": [{"type": "text", "text": result}],
                    "metadata": {
                        "query": query,
                        "search_type": search_type,
                        "anti_bot_level": anti_bot_level,
                        "session_id": session_id,
                        "workproduct_dir": workproduct_dir,
                        "enhanced_search": True,
                        "successful_scrapes": successful_scrapes,
                        "urls_attempted": urls_attempted,
                        "search_queries_executed": 1,
                    },
                }

        except Exception as e:
            error_msg = f"❌ **Enhanced Search Error**\n\nFailed to execute search and content extraction: {str(e)}\n\nPlease check:\n- SERPER_API_KEY is configured\n- Network connectivity\n- Query parameters are valid"
            logger.error(f"Enhanced search failed: {e}")
            return {
                "content": [{"type": "text", "text": error_msg}],
                "is_error": True,
                "error_details": str(e),
            }

    @tool(
        "enhanced_news_search",
        "Specialized news search with enhanced content extraction. Searches for recent news and current events, then extracts and cleans full article content from multiple sources in parallel.",
        {
            "query": {"type": "string", "description": "News topic or search query"},
            "num_results": {
                "type": "integer",
                "default": 15,
                "minimum": 1,
                "maximum": 30,
                "description": "Number of news results to retrieve",
            },
            "auto_crawl_top": {
                "type": "integer",
                "default": 10,
                "minimum": 0,
                "maximum": 15,
                "description": "Maximum number of news articles to crawl for full content",
            },
            "anti_bot_level": {
                "type": "integer",
                "default": 1,
                "minimum": 0,
                "maximum": 3,
                "description": "Progressive anti-bot level for news sites",
            },
            "session_id": {
                "type": "string",
                "default": "default",
                "description": "Session identifier for tracking",
            },
        },
    )
    async def enhanced_news_search(args: dict[str, Any]) -> dict[str, Any]:
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
                    "content": [
                        {
                            "type": "text",
                            "text": "❌ **Error**: 'query' parameter is required for news search",
                        }
                    ],
                    "is_error": True,
                }

            num_results = args.get("num_results", 15)
            auto_crawl_top = args.get("auto_crawl_top", 10)
            anti_bot_level = args.get("anti_bot_level", 1)
            session_id = args.get("session_id", "default")

            # Check research threshold before proceeding
            try:
                from ..utils.research_threshold_tracker import check_search_threshold
                intervention = await check_search_threshold(session_id, query, "news_search")
                if intervention:
                    # Return intervention message instead of searching
                    logger.info(f"🎯 Threshold intervention triggered for news search session {session_id}")
                    return {
                        "content": [
                            {
                                "type": "text",
                                "text": intervention
                            }
                        ],
                        "threshold_intervention": True,
                        "search_type": "news_search",
                        "session_id": session_id
                    }
            except ImportError as e:
                logger.warning(f"⚠️  Could not import threshold tracker: {e}")

            # Set up work product directory
            workproduct_dir = os.environ.get("KEVIN_WORKPRODUCTS_DIR")
            if not workproduct_dir:
                # Use session-based directory structure
                # Use environment-aware path detection
                current_repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                if "claudeagent-multiagent-latest" in current_repo:
                    # Running from claudeagent-multiagent-latest
                    base_session_dir = f"{current_repo}/KEVIN/sessions/{session_id}"
                else:
                    # Fallback to new repository structure
                    base_session_dir = f"/home/kjdragan/lrepos/claudeagent-multiagent-latest/KEVIN/sessions/{session_id}"
                research_dir = f"{base_session_dir}/research"
                workproduct_dir = research_dir

            Path(workproduct_dir).mkdir(parents=True, exist_ok=True)

            logger.info(
                f"Executing enhanced news search: query='{query}', anti_bot_level={anti_bot_level}"
            )

            # Execute enhanced news search
            result = await news_search_and_crawl_direct(
                query=query,
                num_results=num_results,
                auto_crawl_top=auto_crawl_top,
                session_id=session_id,
                anti_bot_level=anti_bot_level,
                workproduct_dir=workproduct_dir,
            )

            # Token management for news results
            if len(result) > 20000:
                # Use adaptive chunking to split content into multiple blocks
                content_blocks = create_adaptive_chunks(result, query)
                logger.info(
                    f"News search content split into {len(content_blocks)} chunks for token management"
                )

                # Rename work product file to indicate chunking was used
                if workproduct_dir:
                    _rename_workproduct_for_chunking(workproduct_dir, session_id)

                return {
                    "content": content_blocks,
                    "metadata": {
                        "query": query,
                        "search_type": "news",
                        "anti_bot_level": anti_bot_level,
                        "session_id": session_id,
                        "workproduct_dir": workproduct_dir,
                        "enhanced_news_search": True,
                        "chunked_content": True,
                        "total_chunks": len(content_blocks),
                    },
                }
            else:
                return {
                    "content": [{"type": "text", "text": result}],
                    "metadata": {
                        "query": query,
                        "search_type": "news",
                        "anti_bot_level": anti_bot_level,
                        "session_id": session_id,
                        "workproduct_dir": workproduct_dir,
                        "enhanced_news_search": True,
                    },
                }

        except Exception as e:
            error_msg = f"❌ **Enhanced News Search Error**\n\nFailed to execute news search and extraction: {str(e)}\n\nPlease check:\n- SERPER_API_KEY is configured\n- Network connectivity\n- News topic is valid"
            logger.error(f"Enhanced news search failed: {e}")
            return {
                "content": [{"type": "text", "text": error_msg}],
                "is_error": True,
                "error_details": str(e),
            }

    @tool(
        "expanded_query_search_and_extract",
        "Corrected query expansion workflow with master result consolidation. Generates multiple search queries, executes SERP searches for each, deduplicates results into master list, ranks by relevance, and scrapes from master list within budget limits. Eliminates excessive searching while providing comprehensive coverage.",
        {
            "query": {
                "type": "string",
                "description": "Original search query or topic to research",
            },
            "search_type": {
                "type": "string",
                "enum": ["search", "news"],
                "default": "search",
                "description": "Type of search: 'search' for web search, 'news' for news search",
            },
            "num_results": {
                "type": "integer",
                "default": 15,
                "minimum": 1,
                "maximum": 50,
                "description": "Number of search results per expanded query",
            },
            "auto_crawl_top": {
                "type": "integer",
                "default": 10,
                "minimum": 0,
                "maximum": 20,
                "description": "Maximum number of URLs to crawl from master ranked list",
            },
            "crawl_threshold": {
                "type": "number",
                "default": 0.3,
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Minimum relevance score threshold for crawling (0.0-1.0)",
            },
            "session_id": {
                "type": "string",
                "default": "default",
                "description": "Session identifier for tracking and work products",
            },
            "max_expanded_queries": {
                "type": "integer",
                "default": 3,
                "minimum": 1,
                "maximum": 5,
                "description": "Maximum number of expanded queries to generate",
            },
        },
    )
    async def expanded_query_search_and_extract_tool(
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Expanded query search and extract functionality with corrected workflow.

        This tool implements the proper workflow that eliminates excessive searching:
        1. Generate multiple related search queries using query expansion
        2. Execute SERP searches for each expanded query
        3. Collect & deduplicate all results into one master list
        4. Rank results by relevance score
        5. Scrape from master ranked list within budget limits (15 successful scrapes)

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
                    "content": [
                        {
                            "type": "text",
                            "text": "❌ **Error**: 'query' parameter is required",
                        }
                    ],
                    "is_error": True,
                }

            search_type = args.get("search_type", "search")
            num_results = int(args.get("num_results", 15))
            auto_crawl_top = int(args.get("auto_crawl_top", 10))
            crawl_threshold = float(args.get("crawl_threshold", 0.3))
            session_id = args.get("session_id", "default")
            max_expanded_queries = int(args.get("max_expanded_queries", 3))

            # Check research threshold before proceeding
            try:
                from ..utils.research_threshold_tracker import check_search_threshold
                intervention = await check_search_threshold(session_id, query, "expanded_query")
                if intervention:
                    # Return intervention message instead of searching
                    logger.info(f"🎯 Threshold intervention triggered for expanded query search session {session_id}")
                    return {
                        "content": [
                            {
                                "type": "text",
                                "text": intervention
                            }
                        ],
                        "threshold_intervention": True,
                        "search_type": "expanded_query",
                        "session_id": session_id
                    }
            except ImportError as e:
                logger.warning(f"⚠️  Could not import threshold tracker: {e}")

            # Set up work product directory
            workproduct_dir = os.environ.get("KEVIN_WORKPRODUCTS_DIR")
            if not workproduct_dir:
                # Use session-based directory structure
                # Use environment-aware path detection
                current_repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                if "claudeagent-multiagent-latest" in current_repo:
                    # Running from claudeagent-multiagent-latest
                    base_session_dir = f"{current_repo}/KEVIN/sessions/{session_id}"
                else:
                    # Fallback to new repository structure
                    base_session_dir = f"/home/kjdragan/lrepos/claudeagent-multiagent-latest/KEVIN/sessions/{session_id}"
                research_dir = f"{base_session_dir}/research"
                workproduct_dir = research_dir

            # Ensure workproduct directory exists
            Path(workproduct_dir).mkdir(parents=True, exist_ok=True)

            logger.info(
                f"Executing expanded query search: query='{query}', max_expanded_queries={max_expanded_queries}"
            )

            # Execute the corrected expanded query search and extract functionality
            # kevin_dir should be the KEVIN root directory (e.g., /path/to/KEVIN)
            # workproduct_dir is /path/to/KEVIN/sessions/{session_id}/research
            # so we need to go up 3 levels to get to KEVIN root
            kevin_root = Path(workproduct_dir).parent.parent.parent
            result = await expanded_query_search_and_extract(
                query=query,
                search_type=search_type,
                num_results=num_results,
                auto_crawl_top=auto_crawl_top,
                crawl_threshold=crawl_threshold,
                session_id=session_id,
                kevin_dir=kevin_root,
                max_expanded_queries=max_expanded_queries,
            )

            # Extract scrape count from result text for budget tracking
            import re
            successful_scrapes = 0
            urls_attempted = 0
            queries_executed = max_expanded_queries

            # Pattern 1: "Successfully Crawled: X"
            match = re.search(r'\*\*Successfully Crawled\*\*:\s*(\d+)', result)
            if match:
                successful_scrapes = int(match.group(1))

            # Pattern 2: "URLs Extracted: X successfully processed"
            if not successful_scrapes:
                match = re.search(r'\*\*URLs Extracted\*\*:\s*(\d+)\s+successfully', result)
                if match:
                    successful_scrapes = int(match.group(1))

            # Pattern 3: "Total Queries Executed: X"
            match = re.search(r'Total Queries Executed:\s*(\d+)', result)
            if match:
                queries_executed = int(match.group(1))

            # Check result length for token management
            actual_chunking_needed = len(result) > 20000
            if actual_chunking_needed:  # Leave room for other content
                # Use adaptive chunking to split content into multiple blocks
                content_blocks = create_adaptive_chunks(result, query)
                logger.info(
                    f"Content split into {len(content_blocks)} chunks for token management"
                )

                # Rename work product file to indicate chunking was used
                _rename_workproduct_for_chunking(workproduct_dir, session_id)

                return {
                    "content": content_blocks,
                    "metadata": {
                        "query": query,
                        "search_type": search_type,
                        "max_expanded_queries": max_expanded_queries,
                        "session_id": session_id,
                        "workproduct_dir": workproduct_dir,
                        "expanded_query_search": True,
                        "chunked_content": True,
                        "total_chunks": len(content_blocks),
                        "successful_scrapes": successful_scrapes,
                        "urls_attempted": urls_attempted,
                        "search_queries_executed": queries_executed,
                    },
                }
            else:
                return {
                    "content": [{"type": "text", "text": result}],
                    "metadata": {
                        "query": query,
                        "search_type": search_type,
                        "max_expanded_queries": max_expanded_queries,
                        "session_id": session_id,
                        "workproduct_dir": workproduct_dir,
                        "expanded_query_search": True,
                        "successful_scrapes": successful_scrapes,
                        "urls_attempted": urls_attempted,
                        "search_queries_executed": queries_executed,
                    },
                }

        except Exception as e:
            error_msg = f"""❌ **Expanded Query Search Error**

Failed to execute expanded query search and content extraction: {str(e)}

This tool uses the corrected workflow that consolidates searches properly.
Please check:
- SERPER_API_KEY is configured
- Network connectivity
- Query parameters are valid
- Expanded query parameters are within limits
"""
            logger.error(f"Expanded query search failed: {e}")
            return {
                "content": [{"type": "text", "text": error_msg}],
                "is_error": True,
                "error_details": str(e),
            }

    # Create the MCP server
    server = create_sdk_mcp_server(
        name="enhanced_search_scrape_clean",
        version="1.0.0",
        tools=[
            enhanced_search_scrape_clean,
            enhanced_news_search,
            expanded_query_search_and_extract_tool,
        ],
    )

    logger.info("Enhanced Search, Scrape & Clean MCP server created successfully")
    return server


# Create server instance
enhanced_search_server = create_enhanced_search_mcp_server()

# Export server for integration
__all__ = ["enhanced_search_server", "create_enhanced_search_mcp_server"]
