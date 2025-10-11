"""
zPlayground1 Search, Scrape & Clean MCP Tool

This is a single, comprehensive MCP tool that exactly mirrors the zPlayground1
topic-based search, scrape, and clean functionality. No multiple tool calls -
just one complete implementation that handles everything in Python.

Based on the proven zPlayground1 implementation, fully integrated for standalone operation.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Claude Agent SDK MCP functionality
try:
    from claude_agent_sdk import create_sdk_mcp_server, tool

    CLAUDE_SDK_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Claude Agent SDK not available for MCP integration")
    CLAUDE_SDK_AVAILABLE = False

# Import the exact zPlayground1 search functionality
from utils.z_search_crawl_utils import (
    news_search_and_crawl_direct,
    search_crawl_and_clean_direct,
)

logger = logging.getLogger(__name__)


def create_zplayground1_mcp_server():
    """
    Create the single zPlayground1 MCP server.

    This is ONE tool that does everything - search, scrape, and clean.
    No multiple MCP tool calls, just pure Python processing with thin MCP wrapper.

    Returns:
        MCP server instance or None if Claude SDK unavailable
    """
    if not CLAUDE_SDK_AVAILABLE:
        logger.error("Cannot create MCP server: Claude Agent SDK not available")
        return None

    @tool(
        "zplayground1_search_scrape_clean",
        "Complete zPlayground1 topic-based search, scrape, and clean functionality. Performs Google search/SERP news search, extracts content from multiple URLs in parallel using exact zPlayground1 implementation with progressive anti-bot detection, applies AI content cleaning, and returns comprehensive results within token limits. This is a single tool that handles the complete workflow - no multiple tool calls needed.",
        {
            "query": {
                "type": "string",
                "description": "Search query, topic, or news topic to research",
            },
            "search_mode": {
                "type": "string",
                "enum": ["web", "news"],
                "default": "web",
                "description": "Search mode: 'web' for Google search, 'news' for news search",
            },
            "num_results": {
                "type": "integer",
                "default": 15,
                "minimum": 1,
                "maximum": 50,
                "description": "Number of search results to retrieve",
            },
            "auto_crawl_top": {
                "type": "integer",
                "default": 10,
                "minimum": 0,
                "maximum": 20,
                "description": "Maximum number of URLs to crawl for content extraction",
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
    async def zplayground1_search_scrape_clean(args: dict[str, Any]) -> dict[str, Any]:
        """
        Complete zPlayground1 search, scrape, and clean functionality.

        This single tool performs the entire zPlayground1 workflow:
        1. Executes Google search or SERP news search based on search_mode
        2. Selects relevant URLs using exact zPlayground1 relevance scoring
        3. Crawls multiple URLs in parallel using exact zPlayground1 SimpleCrawler
        4. Applies progressive anti-bot detection (levels 0-3)
        5. Uses AI content cleaning (GPT-5-nano) via Pydantic AI
        6. Returns cleaned results within token limits
        7. Saves detailed work products for reference

        NO FALLBACKS - Uses exact zPlayground1 implementation that works effectively.
        If it fails, it fails loudly so we know to fix it properly.

        Args:
            args: Dictionary containing all tool parameters

        Returns:
            Dictionary with content and metadata
        """
        try:
            # Extract parameters with FAIL-FAST validation
            query = args.get("query")
            if not query:
                error_msg = "❌ **CRITICAL ERROR**: 'query' parameter is required and cannot be empty!"
                logger.error(f"FAIL-FAST: {error_msg}")
                return {
                    "content": [{"type": "text", "text": error_msg}],
                    "is_error": True,
                }

            # Extract and normalize numeric parameters immediately after argument extraction
            try:
                # Normalize numeric inputs to ensure consistent types
                num_results_raw = args.get("num_results", 15)
                auto_crawl_top_raw = args.get("auto_crawl_top", 10)
                max_concurrent_raw = args.get("max_concurrent", 15)
                anti_bot_level_raw = args.get("anti_bot_level", 1)
                crawl_threshold_raw = args.get("crawl_threshold", 0.3)

                # Convert numeric strings to integers for all numeric parameters
                def normalize_int_param(value, default, min_val, max_val, param_name):
                    """Normalize integer parameter from string or integer input."""
                    if isinstance(value, int):
                        result = value
                    elif isinstance(value, str) and value.isdigit():
                        result = int(value)
                        logger.info(f"🔄 Converted numeric string '{value}' to integer {result} for {param_name}")
                    elif isinstance(value, str):
                        # Try to parse float and convert to int
                        try:
                            result = int(float(value))
                            logger.info(f"🔄 Converted string '{value}' to integer {result} for {param_name}")
                        except ValueError:
                            logger.warning(f"⚠️ Invalid numeric value '{value}' for {param_name}, using default {default}")
                            result = default
                    else:
                        logger.warning(f"⚠️ Invalid type {type(value)} for {param_name}, using default {default}")
                        result = default

                    # Apply range constraints
                    if not (min_val <= result <= max_val):
                        logger.warning(f"⚠️ {param_name} value {result} out of range [{min_val}, {max_val}], clamping to nearest bound")
                        result = max(min_val, min(result, max_val))

                    return result

                def normalize_float_param(value, default, min_val, max_val, param_name):
                    """Normalize float parameter from string or number input."""
                    if isinstance(value, (int, float)):
                        result = float(value)
                    elif isinstance(value, str):
                        try:
                            result = float(value)
                            logger.info(f"🔄 Converted string '{value}' to float {result} for {param_name}")
                        except ValueError:
                            logger.warning(f"⚠️ Invalid numeric value '{value}' for {param_name}, using default {default}")
                            result = default
                    else:
                        logger.warning(f"⚠️ Invalid type {type(value)} for {param_name}, using default {default}")
                        result = default

                    # Apply range constraints
                    if not (min_val <= result <= max_val):
                        logger.warning(f"⚠️ {param_name} value {result} out of range [{min_val}, {max_val}], clamping to nearest bound")
                        result = max(min_val, min(result, max_val))

                    return result

                # Handle anti_bot_level string mapping BEFORE numeric normalization
                anti_bot_level = None
                anti_bot_level_error = None

                # Try named string mapping for anti_bot_level first
                if isinstance(anti_bot_level_raw, str) and not anti_bot_level_raw.isdigit():
                    level_mapping = {
                        "basic": 0,
                        "enhanced": 1,
                        "advanced": 2,
                        "stealth": 3,
                        "standard": 1,  # Map "standard" to enhanced level
                        "low": 0,
                        "medium": 1,
                        "high": 2,
                        "maximum": 3,
                    }
                    anti_bot_level_str = anti_bot_level_raw.lower().strip()
                    if anti_bot_level_str in level_mapping:
                        anti_bot_level = level_mapping[anti_bot_level_str]
                        logger.info(
                            f"🔄 Converted named string '{anti_bot_level_raw}' to integer {anti_bot_level}"
                        )
                    else:
                        anti_bot_level_error = f"FAIL-FAST PARAMETER ERROR: ❌ **CRITICAL PARAMETER VALIDATION ERROR**: Invalid anti_bot_level parameter '{anti_bot_level_raw}' (type: {type(anti_bot_level_raw)}). Must be an integer between 0 and 3, or one of: {list(level_mapping.keys())}"

                # Apply normalization to all numeric parameters (except anti_bot_level if already handled)
                num_results = normalize_int_param(num_results_raw, 15, 1, 50, "num_results")
                auto_crawl_top = normalize_int_param(auto_crawl_top_raw, 10, 0, 20, "auto_crawl_top")
                max_concurrent = normalize_int_param(max_concurrent_raw, 15, 1, 20, "max_concurrent")
                crawl_threshold = normalize_float_param(crawl_threshold_raw, 0.3, 0.0, 1.0, "crawl_threshold")

                # Handle anti_bot_level numeric normalization if not already handled as string
                if anti_bot_level is None:
                    anti_bot_level = normalize_int_param(anti_bot_level_raw, 1, 0, 3, "anti_bot_level")

                # Extract non-numeric parameters
                search_mode = args.get("search_mode", "web")
                if search_mode not in ["web", "news"]:
                    raise ValueError(
                        f"Invalid search_mode '{search_mode}'. Must be 'web' or 'news'"
                    )

                # Raise error if anti_bot_level string validation failed
                if anti_bot_level_error:
                    logger.error(f"❌ {anti_bot_level_error}")
                    raise ValueError(anti_bot_level_error)

                session_id = args.get("session_id", "default")
                workproduct_prefix = args.get("workproduct_prefix", "")

            except (ValueError, TypeError) as param_error:
                error_msg = f"❌ **CRITICAL PARAMETER VALIDATION ERROR**: {param_error}"
                logger.error(f"FAIL-FAST PARAMETER ERROR: {error_msg}")
                logger.error(
                    "This error would have been silently ignored before - now we fail fast!"
                )
                return {
                    "content": [{"type": "text", "text": error_msg}],
                    "is_error": True,
                    "error_details": str(param_error),
                }

            # Parameter validation is now handled above with FAIL-FAST approach

            # Set up work product directory using session-based structure
            workproduct_dir = os.environ.get("KEVIN_WORKPRODUCTS_DIR")
            if not workproduct_dir:
                # Use session-based directory structure with environment-aware path detection
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
                f"🚀 zPlayground1 executing: query='{query}', search_mode='{search_mode}', anti_bot_level={anti_bot_level}"
            )

            # Execute the exact zPlayground1 search and extract functionality
            if search_mode == "news":
                # Use news search and crawl
                result = await news_search_and_crawl_direct(
                    query=query,
                    num_results=num_results,
                    auto_crawl_top=auto_crawl_top,
                    session_id=session_id,
                    anti_bot_level=anti_bot_level,
                    workproduct_dir=workproduct_dir,
                    workproduct_prefix=workproduct_prefix,
                )
            else:
                # Use web search and crawl
                result = await search_crawl_and_clean_direct(
                    query=query,
                    search_type="search",
                    num_results=num_results,
                    auto_crawl_top=auto_crawl_top,
                    crawl_threshold=crawl_threshold,
                    max_concurrent=max_concurrent,
                    session_id=session_id,
                    anti_bot_level=anti_bot_level,
                    workproduct_dir=workproduct_dir,
                    workproduct_prefix=workproduct_prefix,
                )

            # Apply MCP compliance with multi-level content allocation
            from .mcp_compliance_manager import get_mcp_compliance_manager

            mcp_manager = get_mcp_compliance_manager()

            # Prepare metadata for MCP compliance
            base_metadata = {
                "query": query,
                "search_mode": search_mode,
                "anti_bot_level": anti_bot_level,
                "session_id": session_id,
                "workproduct_dir": workproduct_dir,
                "implementation": "zPlayground1_exact",
                "parallel_processing": True,
                "ai_content_cleaning": True,
            }

            # Context for content allocation
            allocation_context = {
                "query": query,
                "query_terms": query.split(),
                "session_id": session_id,
                "source_count": len(
                    [line for line in result.split("\n") if "URL:" in line]
                ),
                "processing_time": "N/A",  # Would be calculated in production
            }

            # Apply MCP compliance allocation
            allocation = mcp_manager.allocate_content(
                raw_content=result, metadata=base_metadata, context=allocation_context
            )

            logger.info(
                f"MCP compliance applied: {allocation.token_usage['total']:,} tokens "
                f"({allocation.token_usage['utilization']:.1f}% utilization), "
                f"compression: {allocation.compression_applied}"
            )

            # Combine primary content with metadata
            final_content = f"""{allocation.primary_content}

---

{allocation.metadata_content}
"""

            return {
                "content": [{"type": "text", "text": final_content}],
                "metadata": {
                    **base_metadata,
                    "mcp_compliance": True,
                    "token_usage": allocation.token_usage,
                    "allocation_stats": allocation.allocation_stats,
                    "compression_applied": allocation.compression_applied,
                    "priority_distribution": allocation.priority_distribution,
                },
            }

        except Exception as e:
            error_msg = f"""❌ **zPlayground1 Search Error**

Failed to execute search and content extraction using zPlayground1 implementation: {str(e)}

**This uses the exact zPlayground1 implementation that works effectively.**
If this failed, there's an issue with the setup that needs to be addressed.

Please check:
- SERP_API_KEY is configured
- Network connectivity
- Query parameters are valid
- zPlayground1 utilities are properly imported
"""
            logger.error(f"zPlayground1 search failed: {e}")
            return {
                "content": [{"type": "text", "text": error_msg}],
                "is_error": True,
                "error_details": str(e),
            }

    # Create the MCP server with this single comprehensive tool
    server = create_sdk_mcp_server(
        name="zplayground1_search_scrape_clean",
        version="1.0.0",
        tools=[zplayground1_search_scrape_clean],
    )

    logger.info(
        "✅ zPlayground1 Search, Scrape & Clean MCP server created successfully"
    )
    logger.info("📋 Single tool approach - no multiple MCP tool calls needed")
    logger.info("🔧 Exact zPlayground1 implementation with no fallbacks")

    return server


# Create server instance
zplayground1_server = create_zplayground1_mcp_server()

# Export server for integration
__all__ = ["zplayground1_server", "create_zplayground1_mcp_server"]
