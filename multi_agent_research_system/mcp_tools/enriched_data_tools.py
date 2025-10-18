"""
Enriched Data MCP Tools

Provides access to structured research memory objects (enriched_search_metadata.json)
Created: October 18, 2025
Purpose: Give agents direct access to structured research data with salient points
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

from claude_agent_sdk import create_sdk_mcp_server, tool

logger = logging.getLogger(__name__)


@tool("read_enriched_metadata", "Read structured research metadata with salient points", {
    "session_id": str
})
async def read_enriched_metadata(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read enriched_search_metadata.json - the structured memory object.
    
    This is the primary memory object designed for agent consumption.
    Contains:
    - search_metadata: Array of articles with salient_points (bullet summaries)
    - scraped_articles: Full LLM-cleaned article content
    - Structured for direct field access (no markdown parsing)
    
    Args:
        session_id: Session ID to get enriched metadata for
        
    Returns:
        Dictionary with structured research data
    """
    try:
        session_id = args["session_id"]
        logger.info(f"Reading enriched metadata for session {session_id}")
        
        # Find enriched_search_metadata.json
        metadata_file = Path(f"KEVIN/sessions/{session_id}/enriched_search_metadata.json")
        
        if not metadata_file.exists():
            logger.error(f"Enriched metadata not found: {metadata_file}")
            return {
                "status": "error",
                "error": f"enriched_search_metadata.json not found for session {session_id}",
                "session_id": session_id
            }
        
        # Load JSON
        with open(metadata_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded enriched metadata: {data['metadata']['total_articles']} articles, "
                   f"{data['metadata']['articles_with_full_content']} with full content")
        
        return {
            "status": "success",
            "session_id": session_id,
            "data": data,  # Complete structured data
            "summary": {
                "total_articles": data["metadata"]["total_articles"],
                "articles_with_full_content": data["metadata"]["articles_with_full_content"],
                "last_updated": data["metadata"]["last_updated"]
            }
        }
        
    except FileNotFoundError as e:
        logger.error(f"Enriched metadata file not found: {e}")
        return {
            "status": "error",
            "error": f"File not found: {str(e)}",
            "session_id": args.get("session_id", "unknown")
        }
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse enriched metadata JSON: {e}")
        return {
            "status": "error",
            "error": f"Invalid JSON format: {str(e)}",
            "session_id": args.get("session_id", "unknown")
        }
    except Exception as e:
        logger.error(f"Failed to read enriched metadata: {e}")
        return {
            "status": "error",
            "error": str(e),
            "session_id": args.get("session_id", "unknown")
        }


def create_enriched_data_mcp_server():
    """
    Create MCP server for enriched data tools.
    
    These tools provide access to structured memory objects.
    """
    return create_sdk_mcp_server(
        name="enriched_data",
        tools=[
            read_enriched_metadata
        ]
    )


# Export server for integration
enriched_data_server = create_enriched_data_mcp_server()

__all__ = ["enriched_data_server", "create_enriched_data_mcp_server", "read_enriched_metadata"]
