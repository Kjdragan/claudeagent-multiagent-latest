"""Advanced web scraping tool using Crawl4AI and AI content cleaning.

This tool provides high-quality content extraction with:
- Browser automation for JavaScript-heavy sites
- Multi-stage extraction with fallback strategies
- AI-powered content cleaning (GPT-5-nano)
- Judge optimization for speed (saves 35-40s per URL)
- Technical content preservation
"""

import logging

from claude_agent_sdk import tool

try:
    from ..utils.content_cleaning import clean_content_with_judge_optimization
    from ..utils.crawl4ai_utils import (
        crawl_multiple_urls_with_cleaning,
        scrape_and_clean_single_url_direct,
    )
except ImportError:
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.crawl4ai_utils import (
        crawl_multiple_urls_with_cleaning,
        scrape_and_clean_single_url_direct,
    )

logger = logging.getLogger(__name__)


@tool(
    "advanced_scrape_url",
    "Advanced web scraping with Crawl4AI browser automation, AI content cleaning, and technical content preservation. Handles JavaScript sites, applies judge optimization for speed, and achieves 70-100% success rates. Returns clean article content with navigation/ads removed.",
    {
        "url": str,
        "session_id": str,
        "search_query": str,
        "preserve_technical": bool
    }
)
async def advanced_scrape_url(args):
    """
    Advanced single URL scraping with multi-stage extraction and AI cleaning.

    Features:
    - Stage 1: Fast CSS selector extraction
    - Stage 2: Robust fallback extraction (universal compatibility)
    - Stage 3: Judge assessment and AI cleaning
    - Technical content preservation (code blocks, installation commands)
    - 30K-58K character content extraction (vs 2K limit in basic scraping)
    """
    url = args.get("url")
    session_id = args.get("session_id", "default")
    search_query = args.get("search_query", None)
    preserve_technical = args.get("preserve_technical", True)

    logger.info(f"Advanced scraping initiated for URL: {url}")

    try:
        # Execute multi-stage scraping with AI cleaning
        result = await scrape_and_clean_single_url_direct(
            url=url,
            session_id=session_id,
            search_query=search_query,
            extraction_mode="article",
            include_metadata=True,
            preserve_technical_content=preserve_technical
        )

        if result['success']:
            # Format response with metadata
            response_text = f"""# Scraped Content from {url}

**Status**: ✅ Success
**Content Length**: {result['char_count']} characters ({result['word_count']} words)
**Processing Stage**: {result.get('stage', 'unknown')}
**Duration**: {result['duration']:.2f}s

---

## Cleaned Content

{result['cleaned_content']}

---

**Metadata**:
- Extraction Mode: {result['extraction_mode']}
- Technical Content Preserved: {preserve_technical}
"""

            # Add cleaning metadata if available
            if result.get('metadata') and result['metadata'].get('cleaning_optimization'):
                cleaning_meta = result['metadata']['cleaning_optimization']
                judge_score = cleaning_meta.get('judge_score', 'N/A')
                latency_saved = cleaning_meta.get('latency_saved', 'N/A')
                response_text += f"\n- Judge Score: {judge_score}"
                response_text += f"\n- Latency Saved: {latency_saved}"

            logger.info(f"Advanced scraping successful for {url}: {result['char_count']} chars extracted")
            return {"content": [{"type": "text", "text": response_text}]}
        else:
            # Extraction failed
            error_msg = f"""# Scraping Failed for {url}

**Status**: ❌ Failed
**Error**: {result['error_message']}
**Duration**: {result['duration']:.2f}s

The multi-stage extraction system attempted to scrape this URL but was unsuccessful.
"""
            logger.warning(f"Advanced scraping failed for {url}: {result['error_message']}")
            return {"content": [{"type": "text", "text": error_msg}], "is_error": True}

    except Exception as e:
        error_msg = f"Advanced scraping failed for {url}: {str(e)}"
        logger.error(error_msg)

        # Check for common issues
        if "playwright" in str(e).lower():
            error_msg += "\n\n⚠️ **Playwright not installed**\nRun: uv run playwright install chromium"

        if "OPENAI_API_KEY" in str(e):
            error_msg += "\n\n⚠️ **OPENAI_API_KEY not found**\nAdd OPENAI_API_KEY to .env for AI content cleaning."

        return {"content": [{"type": "text", "text": error_msg}], "is_error": True}


@tool(
    "advanced_scrape_multiple_urls",
    "Advanced parallel scraping of multiple URLs with Crawl4AI + AI cleaning. Processes URLs concurrently, applies search query filtering to remove unrelated content, and returns cleaned article content. Achieves 70-100% success rates.",
    {
        "urls": list,
        "session_id": str,
        "search_query": str,
        "max_concurrent": int
    }
)
async def advanced_scrape_multiple_urls(args):
    """
    Advanced parallel URL scraping with immediate AI cleaning.

    Features:
    - Parallel crawl+clean processing (reduced latency)
    - Search query context filtering (removes unrelated articles)
    - Progressive anti-bot detection (4-level system)
    - Batch processing with configurable concurrency
    """
    urls = args.get("urls", [])
    session_id = args.get("session_id", "default")
    search_query = args.get("search_query", "")
    max_concurrent = args.get("max_concurrent", 5)

    if not urls:
        return {"content": [{"type": "text", "text": "❌ No URLs provided for scraping."}], "is_error": True}

    logger.info(f"Advanced parallel scraping initiated for {len(urls)} URLs with query: {search_query}")

    try:
        # Execute parallel crawl+clean
        results = await crawl_multiple_urls_with_cleaning(
            urls=urls,
            session_id=session_id,
            search_query=search_query,
            max_concurrent=max_concurrent,
            extraction_mode="article",
            include_metadata=True
        )

        # Format response
        successful_count = sum(1 for r in results if r['success'])
        total_content_length = sum(r.get('char_count', 0) for r in results if r['success'])

        response_text = f"""# Parallel URL Scraping Results

**Total URLs**: {len(urls)}
**Successful**: {successful_count}/{len(urls)} ({successful_count/len(urls)*100:.1f}%)
**Total Content**: {total_content_length} characters
**Search Query Context**: {search_query}

---

"""

        # Add individual results
        for i, result in enumerate(results, 1):
            url = result['url']
            if result['success']:
                content_preview = result.get('cleaned_content', '')[:1000]
                response_text += f"""## ✅ {i}. Success: {url}

**Content Length**: {result['char_count']} characters ({result['word_count']} words)
**Duration**: {result['duration']:.2f}s

### Cleaned Content Preview

{content_preview}{"..." if len(result.get('cleaned_content', '')) > 1000 else ""}

---

"""
            else:
                response_text += f"""## ❌ {i}. Failed: {url}

**Error**: {result['error_message']}
**Duration**: {result['duration']:.2f}s

---

"""

        response_text += f"""
## Processing Summary

- **Parallel Processing**: {max_concurrent} concurrent operations
- **Search Query Filtering**: Applied to remove unrelated content
- **Average Success Rate**: {successful_count/len(urls)*100:.1f}%
- **Average Content Length**: {total_content_length/successful_count if successful_count > 0 else 0:.0f} characters per URL
"""

        logger.info(f"Parallel scraping completed: {successful_count}/{len(urls)} successful")
        return {"content": [{"type": "text", "text": response_text}]}

    except Exception as e:
        error_msg = f"Parallel scraping failed: {str(e)}"
        logger.error(error_msg)
        return {"content": [{"type": "text", "text": error_msg}], "is_error": True}


# Export tools
__all__ = ['advanced_scrape_url', 'advanced_scrape_multiple_urls']
