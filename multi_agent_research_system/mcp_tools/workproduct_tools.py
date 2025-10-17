"""
Workproduct MCP Tools

This module provides Model Context Protocol (MCP) tools for reading research workproducts.
Replaces the corpus system with direct workproduct access.

Created: October 17, 2025
Purpose: Simple, reliable access to research data without corpus complexity
"""

import logging
from typing import Any, Dict, List, Optional

from claude_agent_sdk import create_sdk_mcp_server, tool
from multi_agent_research_system.utils.workproduct_reader import (
    WorkproductReader,
    find_session_workproduct
)

logger = logging.getLogger(__name__)


@tool("get_workproduct_summary", "Get summary of research workproduct", {
    "session_id": str
})
async def get_workproduct_summary_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get high-level summary of workproduct content.
    
    Returns overview including:
    - Article count
    - Source list
    - Date range
    - Word count
    - File metadata
    
    Args:
        session_id: Session ID to get workproduct for
        
    Returns:
        Summary dictionary with workproduct metadata
    """
    try:
        session_id = args["session_id"]
        logger.info(f"Getting workproduct summary for session {session_id}")
        
        # Find and load workproduct
        reader = WorkproductReader.from_session(session_id)
        summary = reader.get_summary()
        
        return {
            "status": "success",
            "session_id": session_id,
            "summary": summary,
            "article_count": summary["article_count"],
            "sources": summary["sources"],
            "total_words": summary["total_words"]
        }
        
    except FileNotFoundError as e:
        logger.error(f"Workproduct not found for session {session_id}: {e}")
        return {
            "status": "error",
            "error": f"No workproduct found for session {session_id}",
            "session_id": session_id
        }
    except Exception as e:
        logger.error(f"Failed to get workproduct summary: {e}")
        return {
            "status": "error",
            "error": str(e),
            "session_id": args.get("session_id", "unknown")
        }


@tool("get_workproduct_article", "Get specific article from workproduct by index ONLY. Use get_all_workproduct_articles to see available articles first.", {
    "session_id": str,
    "index": int
})
async def get_workproduct_article_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get full content of specific article by index.
    
    IMPORTANT: Call get_all_workproduct_articles FIRST to see available articles.
    Then use the index from that response (1-indexed).
    
    Args:
        session_id: Session ID to get workproduct from
        index: Article position (1-indexed integer)
        
    Returns:
        Article dictionary with full content
        
    Example:
        # Step 1: Get all articles
        all_articles = get_all_workproduct_articles(session_id="abc123")
        # Response: {"articles": [{"index": 1, "title": "...", "url": "..."}, ...]}
        
        # Step 2: Get specific article
        article = get_workproduct_article(session_id="abc123", index=1)
    """
    try:
        session_id = args["session_id"]
        index_raw = args.get("index")
        
        # Convert string to int if needed
        if isinstance(index_raw, str):
            try:
                index = int(index_raw)
                logger.warning(f"Converting string index '{index_raw}' to integer {index}")
            except ValueError:
                return {
                    "status": "error",
                    "error": f"Invalid index '{index_raw}' - must be an integer (1-20)",
                    "hint": "Use get_all_workproduct_articles first to see available indices"
                }
        else:
            index = index_raw
        
        if not index:
            return {
                "status": "error",
                "error": "Must provide 'index' parameter (integer, 1-indexed)",
                "hint": "Call get_all_workproduct_articles first to see available articles"
            }
        
        # Load workproduct
        reader = WorkproductReader.from_session(session_id)
        
        logger.info(f"Getting article by index: {index}")
        article = reader.get_article_by_index(index)
        
        if not article:
            article_count = len(reader.metadata.get("articles", []))
            return {
                "status": "error",
                "error": f"Article not found at index {index}",
                "available_range": f"1-{article_count}",
                "hint": f"This workproduct has {article_count} articles. Use get_all_workproduct_articles to see the list.",
                "session_id": session_id
            }
        
        return {
            "status": "success",
            "session_id": session_id,
            "article": article,
            "title": article["title"],
            "url": article["url"],
            "word_count": article["word_count"]
        }
        
    except FileNotFoundError as e:
        logger.error(f"Workproduct not found: {e}")
        return {
            "status": "error",
            "error": f"No workproduct found for session {args.get('session_id')}",
            "session_id": args.get("session_id", "unknown")
        }
    except Exception as e:
        logger.error(f"Failed to get article: {e}")
        return {
            "status": "error",
            "error": str(e),
            "session_id": args.get("session_id", "unknown")
        }


@tool("get_all_workproduct_articles", "Get metadata + snippets for all articles. Use this FIRST to see what's available and decide which to read.", {
    "session_id": str
})
async def get_all_workproduct_articles_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get list of all articles with metadata and content snippets (NO full content).
    
    This is the PRIMARY tool to call FIRST - shows what articles are available.
    Each article includes:
    - index: Position number (1-indexed) for retrieval
    - title: Article headline
    - url: Source URL
    - source: Publisher name
    - date: Publication date
    - relevance_score: Search relevance (0.0-1.0)
    - snippet: Preview of content (~200 chars)
    
    Use the snippet to decide which articles are most relevant, then:
    - Call get_workproduct_article(index=N) for full content of specific articles
    - Or call read_full_workproduct() to get everything at once
    
    Args:
        session_id: Session ID to get workproduct from
        
    Returns:
        Dictionary with list of article metadata + snippets
        
    Example response:
        {
            "articles": [
                {
                    "index": 1, 
                    "title": "Trump declares end of Gaza war",
                    "url": "https://...",
                    "source": "Reuters",
                    "date": "3 days ago",
                    "relevance_score": 0.79,
                    "snippet": "Hamas freed the last living Israeli hostages..."
                }
            ],
            "count": 20
        }
    """
    try:
        session_id = args["session_id"]
        logger.info(f"Getting all articles for session {session_id}")
        
        # Load workproduct
        reader = WorkproductReader.from_session(session_id)
        
        # Get article METADATA ONLY (no full content)
        metadata_list = reader.metadata.get("articles", [])
        
        # Add index if not present
        for i, article in enumerate(metadata_list, 1):
            if "index" not in article:
                article["index"] = i
        
        sources = list(set(a.get("source", "Unknown") for a in metadata_list if a.get("source")))
        
        # Calculate snippet availability
        with_snippets = sum(1 for a in metadata_list if a.get("snippet"))
        
        return {
            "status": "success",
            "session_id": session_id,
            "articles": metadata_list,  # Includes title, URL, source, date, relevance_score, snippet
            "count": len(metadata_list),
            "sources": sources,
            "articles_with_snippets": with_snippets,
            "message": f"Found {len(metadata_list)} articles ({with_snippets} with content snippets). Review the snippets to decide which articles to read in full.",
            "hint": f"Use article index (1-{len(metadata_list)}) with get_workproduct_article(session_id, index=N) to get full content. DO NOT invent URLs - use the exact index numbers provided.",
            "usage": {
                "selective": "Call get_workproduct_article(session_id, index=N) for specific articles based on snippet relevance",
                "comprehensive": "Call read_full_workproduct(session_id) to get all articles at once"
            }
        }
        
    except FileNotFoundError as e:
        logger.error(f"Workproduct not found: {e}")
        return {
            "status": "error",
            "error": f"No workproduct found for session {session_id}",
            "session_id": session_id
        }
    except Exception as e:
        logger.error(f"Failed to get all articles: {e}")
        return {
            "status": "error",
            "error": str(e),
            "session_id": args.get("session_id", "unknown")
        }


@tool("read_full_workproduct", "Read complete workproduct file", {
    "session_id": str
})
async def read_full_workproduct_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read entire workproduct content as markdown.
    
    Use this if you want the complete file content rather than
    structured article extraction.
    
    Args:
        session_id: Session ID to get workproduct from
        
    Returns:
        Dictionary with full workproduct content and metadata
    """
    try:
        session_id = args["session_id"]
        logger.info(f"Reading full workproduct for session {session_id}")
        
        # Load workproduct
        reader = WorkproductReader.from_session(session_id)
        
        return {
            "status": "success",
            "session_id": session_id,
            "content": reader.get_full_content(),
            "metadata": reader.get_metadata(),
            "summary": reader.get_summary()
        }
        
    except FileNotFoundError as e:
        logger.error(f"Workproduct not found: {e}")
        return {
            "status": "error",
            "error": f"No workproduct found for session {session_id}",
            "session_id": session_id
        }
    except Exception as e:
        logger.error(f"Failed to read workproduct: {e}")
        return {
            "status": "error",
            "error": str(e),
            "session_id": args.get("session_id", "unknown")
        }


def create_workproduct_mcp_server():
    """
    Create MCP server for workproduct tools.
    
    These tools replace the corpus system with simpler, more reliable
    direct workproduct access.
    """
    return create_sdk_mcp_server(
        name="workproduct",
        tools=[
            get_workproduct_summary_tool,
            get_workproduct_article_tool,
            get_all_workproduct_articles_tool,
            read_full_workproduct_tool
        ]
    )


# Create server instance
workproduct_server = create_workproduct_mcp_server()


# Export
__all__ = [
    "workproduct_server",
    "create_workproduct_mcp_server",
    "get_workproduct_summary_tool",
    "get_workproduct_article_tool",
    "get_all_workproduct_articles_tool",
    "read_full_workproduct_tool"
]
