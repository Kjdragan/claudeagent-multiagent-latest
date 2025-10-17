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


@tool("get_workproduct_article", "Get specific article from workproduct", {
    "session_id": str,
    "url": Optional[str],
    "index": Optional[int]
})
async def get_workproduct_article_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get full content of specific article by URL or index.
    
    Provide either url OR index parameter.
    
    Args:
        session_id: Session ID to get workproduct from
        url: Optional URL of article to retrieve
        index: Optional position of article (1-indexed)
        
    Returns:
        Article dictionary with full content
    """
    try:
        session_id = args["session_id"]
        url = args.get("url")
        index = args.get("index")
        
        if not url and not index:
            return {
                "status": "error",
                "error": "Must provide either 'url' or 'index' parameter"
            }
        
        # Load workproduct
        reader = WorkproductReader.from_session(session_id)
        
        # Get article by URL or index
        if url:
            logger.info(f"Getting article by URL: {url}")
            article = reader.get_article_by_url(url)
        else:
            logger.info(f"Getting article by index: {index}")
            article = reader.get_article_by_index(index)
        
        if not article:
            return {
                "status": "error",
                "error": f"Article not found (url={url}, index={index})",
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


@tool("get_all_workproduct_articles", "Get all articles from workproduct", {
    "session_id": str
})
async def get_all_workproduct_articles_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get all articles with full content from workproduct.
    
    This is the primary tool for report generation - it provides all
    research data in one call.
    
    Args:
        session_id: Session ID to get workproduct from
        
    Returns:
        Dictionary with list of all articles and metadata
    """
    try:
        session_id = args["session_id"]
        logger.info(f"Getting all articles for session {session_id}")
        
        # Load workproduct
        reader = WorkproductReader.from_session(session_id)
        
        # Get all articles with content
        articles = reader.get_all_articles()
        
        # Calculate aggregate statistics
        total_words = sum(a["word_count"] for a in articles)
        sources = list(set(a["source"] for a in articles if a.get("source")))
        
        return {
            "status": "success",
            "session_id": session_id,
            "articles": articles,
            "count": len(articles),
            "total_words": total_words,
            "sources": sources,
            "metadata": reader.get_metadata()
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
