"""Intelligent Research Tool with Z-Playground1 Proven Intelligence

This tool implements the complete sophisticated research system from z-playground1:
- Search 15 URLs with redundancy for expected failures
- Enhanced relevance scoring (position 40% + title 30% + snippet 30%)
- Threshold-based URL selection (default 0.3)
- Parallel crawling with anti-bot escalation
- AI content cleaning with search query filtering
- Smart content compression for MCP compliance
- Complete work product generation

All processing happens internally to stay within MCP token limits while
preserving the proven intelligence of the original system.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse

from claude_agent_sdk import tool

# Import advanced scraping utilities
try:
    from ..utils.crawl4ai_utils import crawl_multiple_urls_with_cleaning
    from ..utils.content_cleaning import format_cleaned_results
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.crawl4ai_utils import crawl_multiple_urls_with_cleaning
    from utils.content_cleaning import format_cleaned_results

# Import existing search utilities for SERP API
try:
    from ..utils.serp_search_utils import execute_serp_search
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.serp_search_utils import execute_serp_search

logger = logging.getLogger(__name__)


class SearchResult:
    """Search result data structure with enhanced relevance scoring"""
    def __init__(self, title: str, link: str, snippet: str, position: int = 0,
                 date: str = None, source: str = None, relevance_score: float = 0.0):
        self.title = title
        self.link = link
        self.snippet = snippet
        self.position = position
        self.date = date
        self.source = source
        self.relevance_score = relevance_score


def calculate_enhanced_relevance_score(
    title: str,
    snippet: str,
    position: int,
    query_terms: List[str]
) -> float:
    """
    Calculate enhanced relevance score based on z-playground1 proven formula.

    Formula:
    - Google position weight: 40%
    - Query term matching in title: 30%
    - Query term matching in snippet: 30%

    Args:
        title: Search result title
        snippet: Search result snippet
        position: Google search position (1-based)
        query_terms: List of query terms to match

    Returns:
        Relevance score between 0.0 and 1.0
    """
    # Normalize inputs
    title_lower = title.lower()
    snippet_lower = snippet.lower()
    query_terms_lower = [term.lower() for term in query_terms if term]

    if not query_terms_lower:
        # Fallback to position-only scoring
        return max(0.05, 1.0 - (position * 0.05))

    # 1. Position score (40% weight) - higher positions get higher scores
    if position <= 10:
        position_score = (11 - position) / 10  # 1.0, 0.9, 0.8, ..., 0.1
    else:
        position_score = max(0.05, 0.1 - ((position - 10) * 0.01))  # Gradual decay, min 0.05

    # 2. Title matching score (30% weight)
    title_matches = sum(1 for term in query_terms_lower if term in title_lower)
    title_score = min(1.0, title_matches / len(query_terms_lower))

    # 3. Snippet matching score (30% weight)
    snippet_matches = sum(1 for term in query_terms_lower if term in snippet_lower)
    snippet_score = min(1.0, snippet_matches / len(query_terms_lower))

    # Combine with weights
    final_score = (
        position_score * 0.40 +
        title_score * 0.30 +
        snippet_score * 0.30
    )

    return round(final_score, 3)


def select_urls_for_crawling(
    search_results: List[SearchResult],
    limit: int,
    min_relevance: float
) -> List[str]:
    """
    Select URLs for crawling based on z-playground1 threshold-based approach.

    Args:
        search_results: List of search results with relevance scores
        limit: Maximum number of URLs to select
        min_relevance: Minimum relevance score threshold

    Returns:
        List of URLs to crawl
    """
    try:
        # Filter by relevance threshold
        filtered_results = [
            result for result in search_results
            if result.relevance_score >= min_relevance and result.link
        ]

        # Sort by relevance score (highest first)
        filtered_results.sort(key=lambda x: x.relevance_score, reverse=True)

        # Extract URLs up to limit
        urls = [result.link for result in filtered_results[:limit]]

        logger.info(f"URL selection with threshold {min_relevance}:")
        logger.info(f"  - Total results: {len(search_results)}")
        logger.info(f"  - Above threshold: {len(filtered_results)}")
        logger.info(f"  - Selected for crawling: {len(urls)}")
        logger.info(f"  - Rejected: {len(search_results) - len(filtered_results)} below threshold")

        return urls

    except Exception as e:
        logger.error(f"Error selecting URLs for crawling: {e}")
        return []


def extract_query_terms(query: str) -> List[str]:
    """Extract meaningful query terms from search query."""
    # Simple term extraction - split by spaces and remove common stop words
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were'}
    words = [word.lower().strip('.,!?;:') for word in query.split() if word.lower() not in stop_words and len(word) > 2]
    return words


def compress_for_mcp_compliance(
    crawl_results: List[Dict[str, Any]],
    search_results: List[SearchResult],
    max_tokens: int = 20000
) -> str:
    """
    Smart content compression to stay within MCP limits while preserving research value.

    Multi-level compression strategy:
    Level 1: High Priority (full detail for top sources)
    Level 2: Medium Priority (summarized for next tier)
    Level 3: Low Priority (references only for remaining)

    Args:
        crawl_results: List of successful crawl results with cleaned content
        search_results: Original search results with relevance scores
        max_tokens: Maximum tokens for MCP compliance (stay under 25K limit)

    Returns:
        Compressed research summary for MCP response
    """
    try:
        if not crawl_results:
            return "❌ No content successfully extracted. Try different search terms or sources."

        # Sort results by relevance (match crawl results with search results)
        successful_results = []
        for crawl_result in crawl_results:
            if crawl_result['success']:
                # Find corresponding search result for relevance score
                relevance_score = 0.0
                for search_result in search_results:
                    if search_result.link == crawl_result['url']:
                        relevance_score = search_result.relevance_score
                        break

                successful_results.append({
                    'url': crawl_result['url'],
                    'cleaned_content': crawl_result.get('cleaned_content', ''),
                    'relevance_score': relevance_score,
                    'char_count': len(crawl_result.get('cleaned_content', '')),
                    'title': crawl_result.get('title', 'Unknown Source')
                })

        # Sort by relevance score (highest first)
        successful_results.sort(key=lambda x: x['relevance_score'], reverse=True)

        # Multi-level content allocation
        response_parts = [
            f"# Intelligent Research Results",
            f"",
            f"**Query Processing**: Searched 15 sources → Filtered by relevance (threshold 0.3) → Parallel crawl → AI cleaning → Smart compression",
            f"**Total Sources**: {len(search_results)} found, {len(successful_results)} successfully processed",
            f"",
            f"## 📊 Research Summary",
            f""
        ]

        current_tokens = 1000  # Base structure tokens

        # Level 1: Top 3 sources (full detail)
        level1_results = successful_results[:3]
        response_parts.append("### 🏆 Top Priority Sources (Full Detail)")
        response_parts.append("")

        for i, result in enumerate(level1_results, 1):
            # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
            content_tokens = len(result['cleaned_content']) // 4

            if current_tokens + content_tokens > max_tokens:
                response_parts.append(f"**{i}. Content truncated due to size limits**")
                response_parts.append(f"See work product file for complete content.")
                break

            response_parts.extend([
                f"**{i}. {result['title']}**",
                f"**URL**: {result['url']}",
                f"**Relevance Score**: {result['relevance_score']:.2f}",
                f"**Content Length**: {result['char_count']} characters",
                f"",
                result['cleaned_content'][:8000],  # Limit content for each source
                "",
                "---",
                ""
            ])

            current_tokens += content_tokens

        # Level 2: Next 3 sources (summarized)
        if current_tokens < max_tokens - 2000 and len(successful_results) > 3:
            level2_results = successful_results[3:6]
            response_parts.append("### 📈 High Priority Sources (Key Insights)")
            response_parts.append("")

            for i, result in enumerate(level2_results, 4):
                if current_tokens > max_tokens - 1500:
                    break

                # Create smart summary (first paragraph + key points)
                content = result['cleaned_content']
                if len(content) > 1000:
                    # Take first paragraph and extract key insights
                    first_para = content.split('\n\n')[0][:500]
                    response_parts.extend([
                        f"**{i}. {result['title']}**",
                        f"**URL**: {result['url']} | **Relevance**: {result['relevance_score']:.2f}",
                        f"",
                        first_para,
                        f"",
                        f"*Full content ({result['char_count']} chars) in work product file*",
                        "",
                        "---",
                        ""
                    ])
                    current_tokens += 800

        # Level 3: Remaining sources (references only)
        if current_tokens < max_tokens - 1000 and len(successful_results) > 6:
            level3_results = successful_results[6:]
            response_parts.append("### 📚 Additional Sources (References)")
            response_parts.append("")

            for i, result in enumerate(level3_results, 7):
                if current_tokens > max_tokens - 500:
                    break

                response_parts.extend([
                    f"**{i}. {result['title']}**",
                    f"**URL**: {result['url']} | **Relevance**: {result['relevance_score']:.2f}",
                    f""
                ])
                current_tokens += 200

        # Add processing summary
        response_parts.extend([
            "## 🔍 Processing Summary",
            "",
            f"**Sources Successfully Processed**: {len(successful_results)}",
            f"**Average Content Length**: {sum(r['char_count'] for r in successful_results) // len(successful_results)} characters",
            f"**Content Cleaning**: AI-powered removal of navigation, ads, and unrelated content",
            f"**Relevance Filtering**: Sources selected with threshold 0.3+ relevance scores",
            "",
            "📄 **Complete work products saved with full content for detailed analysis**",
            ""
        ])

        return "\n".join(response_parts)

    except Exception as e:
        logger.error(f"Error compressing content for MCP compliance: {e}")
        return f"❌ Error processing research results: {str(e)}"


def save_intelligent_work_product(
    search_results: List[SearchResult],
    crawl_results: List[Dict[str, Any]],
    urls_processed: List[str],
    query: str,
    session_id: str
) -> str:
    """
    Save detailed work product with complete research data.

    Args:
        search_results: Original search results with relevance scores
        crawl_results: Complete crawl results with full cleaned content
        urls_processed: URLs that were successfully crawled
        query: Original search query
        session_id: Session identifier

    Returns:
        Path to saved work product file
    """
    try:
        # Create work product directory
        kevin_dir = Path.home() / "lrepos" / "claude-agent-sdk-python" / "KEVIN"
        work_dir = kevin_dir / "work_products" / session_id
        work_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp and filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"intelligent_research_workproduct_{timestamp}.md"
        filepath = work_dir / filename

        # Build comprehensive work product
        content_parts = [
            f"# Intelligent Research Work Product",
            f"",
            f"**Session ID**: {session_id}",
            f"**Export Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Research Query**: {query}",
            f"**Processing Method**: Z-Playground1 Intelligent System (15 → threshold 0.3 → parallel crawl → AI cleaning)",
            f"",
            f"---",
            f"",
            f"## 🔍 Search Results Analysis",
            f""
        ]

        # Add all search results with relevance analysis
        content_parts.append(f"**Total Search Results**: {len(search_results)}")
        content_parts.append(f"**Sources Above Relevance Threshold**: {len(urls_processed)}")
        content_parts.append(f"**Successfully Crawled**: {len([r for r in crawl_results if r['success']])}")
        content_parts.append("")
        content_parts.append("### Search Results with Relevance Scores")
        content_parts.append("")

        for i, result in enumerate(search_results, 1):
            status = "✅ Crawled" if result.link in urls_processed else "❌ Not Processed"
            content_parts.extend([
                f"#### {i}. {result.title}",
                f"**URL**: {result.link}",
                f"**Source**: {result.source or 'Unknown'}",
                f"**Position**: {result.position}",
                f"**Relevance Score**: {result.relevance_score:.3f}",
                f"**Status**: {status}",
                f"**Snippet**: {result.snippet}",
                f"",
                "---",
                f""
            ])

        # Add detailed crawled content
        content_parts.extend([
            f"## 📄 Detailed Crawled Content (AI Cleaned)",
            f""
        ])

        for crawl_result in crawl_results:
            if crawl_result['success']:
                cleaned_content = crawl_result.get('cleaned_content', '')
                if cleaned_content:
                    # Find title from search results
                    title = "Unknown Source"
                    for search_result in search_results:
                        if search_result.link == crawl_result['url']:
                            title = search_result.title
                            break

                    content_parts.extend([
                        f"### 🌐 {title}",
                        f"",
                        f"**URL**: {crawl_result['url']}",
                        f"**Extraction Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        f"**Content Length**: {len(cleaned_content)} characters",
                        f"**Processing**: ✅ Cleaned with GPT-5-nano AI",
                        f"",
                        "### Full Cleaned Content",
                        "",
                        "---",
                        "",
                        cleaned_content,
                        "",
                        "---",
                        ""
                    ])

        # Add processing summary
        content_parts.extend([
            f"## 📊 Intelligent Processing Summary",
            f"",
            f"**Search Strategy**: 15 URLs with redundancy for expected failures",
            f"**Relevance Threshold**: 0.3 minimum score for URL selection",
            f"**Parallel Processing**: Concurrent crawling with anti-bot escalation",
            f"**Content Cleaning**: AI-powered removal of navigation, ads, unrelated content",
            f"**MCP Compliance**: Smart compression to stay within token limits",
            f"**Total Processing Time**: Single tool call with internal optimization",
            f"",
            "*Generated by Intelligent Research Tool - Z-Playground1 Proven Intelligence*"
        ])

        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content_parts))

        logger.info(f"✅ Intelligent work product saved to: {filepath}")
        return str(filepath)

    except Exception as e:
        logger.error(f"Error saving intelligent work product: {e}")
        return ""


@tool(
    "intelligent_research_with_advanced_scraping",
    "Complete intelligent research using proven z-playground1 system: search 15 URLs → relevance threshold filtering → parallel crawl → AI cleaning → MCP-compliant results. Handles failures gracefully, provides work products, and stays within token limits.",
    {
        "query": str,
        "session_id": str,
        "max_urls": int,
        "relevance_threshold": float,
        "max_concurrent": int
    }
)
async def intelligent_research_with_advanced_scraping(args):
    """
    Intelligent research tool implementing complete z-playground1 proven system.

    This tool performs the complete research pipeline internally:
    1. Search 15 URLs with redundancy (expecting some failures)
    2. Apply enhanced relevance scoring (position 40% + title 30% + snippet 30%)
    3. Filter by relevance threshold (default 0.3)
    4. Parallel crawling with anti-bot escalation
    5. AI content cleaning with search query filtering
    6. Smart content compression for MCP compliance
    7. Complete work product generation

    All sophisticated processing happens inside this tool to stay within MCP limits.
    """
    query = args.get("query")
    session_id = args.get("session_id", "default")
    max_urls = args.get("max_urls", 10)
    relevance_threshold = args.get("relevance_threshold", 0.3)
    max_concurrent = args.get("max_concurrent", 10)

    if not query:
        return {
            "content": [{"type": "text", "text": "❌ Error: query parameter is required"}],
            "is_error": True
        }

    logger.info(f"🚀 Starting intelligent research for query: '{query}'")
    logger.info(f"📊 Configuration: max_urls={max_urls}, relevance_threshold={relevance_threshold}, max_concurrent={max_concurrent}")

    try:
        # Phase 1: Search 15 URLs with redundancy
        logger.info("📡 Phase 1: Searching 15 URLs with redundancy for expected failures")

        search_results = await execute_serp_search(
            query=query,
            search_type="search",
            num_results=15
        )

        if not search_results:
            return {
                "content": [{"type": "text", "text": f"❌ No search results found for query: '{query}'"}],
                "is_error": True
            }

        # Phase 2: Apply enhanced relevance scoring
        logger.info("🎯 Phase 2: Applying enhanced relevance scoring (z-playground1 formula)")
        query_terms = extract_query_terms(query)

        # Calculate relevance scores
        for result in search_results:
            result.relevance_score = calculate_enhanced_relevance_score(
                title=result.title,
                snippet=result.snippet,
                position=result.position,
                query_terms=query_terms
            )

        # Phase 3: Apply threshold-based URL selection
        logger.info(f"🔍 Phase 3: Selecting URLs with relevance threshold {relevance_threshold}")
        urls_to_crawl = select_urls_for_crawling(
            search_results=search_results,
            limit=max_urls,
            min_relevance=relevance_threshold
        )

        if not urls_to_crawl:
            # Fallback to basic search results if no URLs meet threshold
            basic_results = f"# Search Results for '{query}'\n\n"
            for i, result in enumerate(search_results[:5], 1):
                basic_results += f"## {i}. {result.title}\n**URL**: {result.link}\n**Relevance**: {result.relevance_score:.2f}\n{result.snippet}\n\n"
            basic_results += f"\n**Note**: No URLs met the relevance threshold ({relevance_threshold}). Search results provided above."

            return {
                "content": [{"type": "text", "text": basic_results}],
                "metadata": {
                    "search_results_found": len(search_results),
                    "urls_selected": 0,
                    "threshold_used": relevance_threshold,
                    "processing_method": "search_only"
                }
            }

        # Phase 4: Parallel crawling with advanced scraping
        logger.info(f"🕷️ Phase 4: Parallel crawling {len(urls_to_crawl)} URLs with advanced scraping")

        crawl_results = await crawl_multiple_urls_with_cleaning(
            urls=urls_to_crawl,
            session_id=session_id,
            search_query=query,
            max_concurrent=min(max_concurrent, len(urls_to_crawl)),
            extraction_mode="article"
        )

        # Phase 5: Save complete work product
        logger.info("💾 Phase 5: Saving complete work product")
        work_product_path = save_intelligent_work_product(
            search_results=search_results,
            crawl_results=crawl_results,
            urls_processed=urls_to_crawl,
            query=query,
            session_id=session_id
        )

        # Phase 6: Smart content compression for MCP compliance
        logger.info("🗜️ Phase 6: Smart content compression for MCP compliance")
        mcp_response = compress_for_mcp_compliance(
            crawl_results=crawl_results,
            search_results=search_results,
            max_tokens=20000  # Stay well under 25K MCP limit
        )

        # Calculate success metrics
        successful_crawls = len([r for r in crawl_results if r['success']])
        total_chars = sum(len(r.get('cleaned_content', '')) for r in crawl_results if r['success'])

        # Build success message
        success_msg = f"""# Intelligent Research Complete ✅

**Query**: {query}
**Processing Method**: Z-Playground1 Proven Intelligence System
**Search Results Found**: {len(search_results)}
**URLs Selected for Crawling**: {len(urls_to_crawl)} (threshold: {relevance_threshold})
**Successfully Crawled**: {successful_crawls}/{len(urls_to_crawl)}
**Total Content Extracted**: {total_chars:,} characters
**Work Product**: {work_product_path}

{mcp_response}

**🎯 Intelligence Applied**:
- ✅ Enhanced relevance scoring (position 40% + title 30% + snippet 30%)
- ✅ Threshold-based URL selection (0.3 minimum relevance)
- ✅ Parallel crawling with anti-bot escalation
- ✅ AI content cleaning with search query filtering
- ✅ Smart compression for MCP compliance
- ✅ Complete work product generation
"""

        logger.info(f"✅ Intelligent research completed successfully")
        logger.info(f"📊 Results: {successful_crawls}/{len(urls_to_crawl)} URLs crawled, {total_chars:,} total characters")

        return {
            "content": [{"type": "text", "text": success_msg}],
            "metadata": {
                "search_results_found": len(search_results),
                "urls_selected": len(urls_to_crawl),
                "successful_crawls": successful_crawls,
                "total_content_chars": total_chars,
                "relevance_threshold": relevance_threshold,
                "work_product_path": work_product_path,
                "processing_method": "z_playground1_intelligent_system"
            }
        }

    except Exception as e:
        error_msg = f"❌ Intelligent research failed: {str(e)}"
        logger.error(error_msg)

        # Check for common issues
        if "OPENAI_API_KEY" in str(e):
            error_msg += "\n\n⚠️ **OPENAI_API_KEY not found** - Add to .env for AI content cleaning"
        elif "crawl4ai" in str(e).lower():
            error_msg += "\n\n⚠️ **Crawl4AI error** - Check advanced scraping installation"

        return {
            "content": [{"type": "text", "text": error_msg}],
            "is_error": True
        }


# Export tool
__all__ = ['intelligent_research_with_advanced_scraping']