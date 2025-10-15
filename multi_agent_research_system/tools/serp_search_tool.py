"""SERP API Search Tool for Multi-Agent Research System

This module provides a high-performance search tool using SERP API
to replace the WebPrime MCP search system.
"""

import os
from pathlib import Path

from claude_agent_sdk import tool

try:
    from ..utils.serp_search_utils import serp_search_and_extract
except ImportError:
    # Fallback for when running as script
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.serp_search_utils import serp_search_and_extract


@tool(
    "serp_search",
    "High-performance Google search using SERP API with automatic content extraction. 10x faster than MCP search with relevance scoring and work product generation.",
    {
        "query": str,
        "search_type": str,
        "num_results": int,
        "auto_crawl_top": int,
        "crawl_threshold": float,
        "session_id": str
    }
)
async def serp_search(args):
    """
    High-performance Google search using SERP API with automatic content extraction.

    This tool provides 10x faster search performance compared to MCP-based search,
    with automatic content extraction and work product generation.
    """
    query = args.get("query")
    search_type = args.get("search_type", "search")
    num_results = args.get("num_results", 15)
    auto_crawl_top = args.get("auto_crawl_top", 5)
    crawl_threshold = args.get("crawl_threshold", 0.3)
    session_id = args.get("session_id", "default")

    try:
        # Get KEVIN directory for work product storage
        kevin_dir = Path.home() / "lrepos" / "claude-agent-sdk-python" / "KEVIN"

        # Execute SERP search with content extraction
        result = await serp_search_and_extract(
            query=query,
            search_type=search_type,
            num_results=num_results,
            auto_crawl_top=auto_crawl_top,
            crawl_threshold=crawl_threshold,
            session_id=session_id,
            kevin_dir=kevin_dir
        )

        return {"content": [{"type": "text", "text": result}]}

    except Exception as e:
        error_msg = f"SERP API search failed: {str(e)}"

        # Check if API key is missing
        if "SERPER_API_KEY" in str(e):
            error_msg += "\n\n⚠️ **SERPER_API_KEY not found in environment**\nPlease add SERPER_API_KEY to your .env file."

        # Check if OpenAI API key is missing (for content cleaning)
        if "OPENAI_API_KEY" in str(e):
            error_msg += "\n\n⚠️ **OPENAI_API_KEY not found in environment**\nPlease add OPENAI_API_KEY to your .env file for content processing."

        return {"content": [{"type": "text", "text": error_msg}], "is_error": True}


# Export the search function
__all__ = ['serp_search']
