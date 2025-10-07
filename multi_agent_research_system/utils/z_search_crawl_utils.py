"""
Enhanced search and crawl utilities adapted from zPlayground1.

This module provides optimized search+crawl+clean functionality with
parallel processing, anti-bot detection, and AI content cleaning.
Adapted for integration with the multi-agent research system.
"""

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path

# Performance timer imports
try:
    from .performance_timers import timed_block, get_performance_timer, save_session_performance_report
except ImportError:
    # Fallback no-op if performance_timers not available
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def timed_block(block_name: str, metadata=None):
        yield

    def get_performance_timer():
        return None

    def save_session_performance_report(session_id, working_dir):
        pass

logger = logging.getLogger(__name__)


class SearchResult:
    """Enhanced search result data structure"""

    def __init__(
        self,
        title: str,
        link: str,
        snippet: str,
        position: int = 0,
        date: str = None,
        source: str = None,
        relevance_score: float = 0.0,
    ):
        self.title = title
        self.link = link
        self.snippet = snippet
        self.position = position
        self.date = date
        self.source = source
        self.relevance_score = relevance_score


# Import enhanced relevance scorer with domain authority
from .enhanced_relevance_scorer import (
    calculate_enhanced_relevance_score_with_domain_authority as calculate_enhanced_relevance_score,
)


async def execute_serper_search(
    query: str,
    search_type: str = "search",
    num_results: int = 10,
    country: str = "us",
    language: str = "en",
) -> list[SearchResult]:
    """
    Execute search using Serper API.

    Args:
        query: Search query
        search_type: "search" or "news"
        num_results: Number of results to retrieve
        country: Country code for search
        language: Language code for search

    Returns:
        List of SearchResult objects
    """
    try:
        import httpx

        # FAIL-FAST: Check for correct API key name - must match orchestrator expectations
        serper_api_key = os.getenv(
            "SERP_API_KEY"
        )  # Note: SERP_API_KEY, not SERPER_API_KEY

        if not serper_api_key:
            # During development, fail hard and fast with clear error message
            error_msg = "CRITICAL: SERP_API_KEY not found in environment variables!"
            logger.error(f"❌ {error_msg}")
            logger.error("Search functionality cannot work without SERP_API_KEY!")
            logger.error("Expected environment variable: SERP_API_KEY")
            logger.error("Set with: export SERP_API_KEY='your-serper-api-key'")

            # Check if user has the wrong API key name
            if os.getenv("SERPER_API_KEY"):
                logger.error("🚨 FOUND SERPER_API_KEY but system expects SERP_API_KEY!")
                logger.error(
                    "Please rename your environment variable from SERPER_API_KEY to SERP_API_KEY"
                )

            # During development, fail immediately instead of returning empty results
            raise RuntimeError(f"CRITICAL SEARCH CONFIGURATION FAILURE: {error_msg}")

            # Note: The following return will never be reached due to the RuntimeError above
            # but keeping it for clarity if the fail-fast approach is later softened
            return []

        # Choose endpoint based on search type
        endpoint = "news" if search_type == "news" else "search"
        url = f"https://google.serper.dev/{endpoint}"

        # Build search parameters
        search_params = {
            "q": query,
            "num": min(num_results, 100),  # Serper limit
            "gl": country,
            "hl": language,
        }

        headers = {"X-API-KEY": serper_api_key, "Content-Type": "application/json"}

        logger.info(f"Executing {search_type} search for: {query}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=search_params, headers=headers)

        if response.status_code == 200:
            data = response.json()

            # Extract results based on search type
            if search_type == "news" and "news" in data:
                raw_results = data["news"]
            else:
                raw_results = data.get("organic", [])

            # Parse query terms for enhanced relevance scoring
            query_terms = query.lower().replace("or", " ").replace("and", " ").split()
            query_terms = [
                term.strip() for term in query_terms if len(term.strip()) > 2
            ]

            # Convert to SearchResult objects with enhanced relevance scoring
            search_results = []
            for i, result in enumerate(raw_results):
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                position = i + 1

                # Calculate enhanced relevance score with domain authority
                relevance_score = calculate_enhanced_relevance_score(
                    title=title,
                    snippet=snippet,
                    position=position,
                    query_terms=query_terms,
                    url=result.get("link", ""),
                )

                search_result = SearchResult(
                    title=title,
                    link=result.get("link", ""),
                    snippet=snippet,
                    position=position,
                    date=result.get("date", ""),
                    source=result.get("source", ""),
                    relevance_score=relevance_score,
                )
                search_results.append(search_result)

            logger.info(
                f"Retrieved {len(search_results)} search results for query: '{query}'"
            )
            return search_results

        else:
            logger.error(f"Serper API error: {response.status_code}")
            return []

    except Exception as e:
        # FAIL-FAST: During development, re-raise critical errors instead of swallowing them
        logger.error(f"Error in Serper search: {e}")

        # Check if this is a critical configuration error that should fail fast
        if "CRITICAL" in str(e) or "API_KEY" in str(e) or "Configuration" in str(e):
            logger.error(
                "FAIL-FAST: Critical configuration error detected - re-raising to expose configuration issues!"
            )
            raise  # Re-raise the critical error instead of returning empty results

        # For other errors, return empty list for now
        logger.warning("Non-critical search error - returning empty results")
        return []


def select_urls_for_crawling(
    search_results: list[SearchResult], limit: int = 10, min_relevance: float = 0.4
) -> list[str]:
    """
    Select URLs for crawling based on relevance scores.

    Args:
        search_results: List of search results
        limit: Maximum number of URLs to select
        min_relevance: Minimum relevance score threshold

    Returns:
        List of URLs to crawl
    """
    try:
        # Filter by relevance threshold (ensure type safety)
        filtered_results = [
            result
            for result in search_results
            if float(result.relevance_score) >= float(min_relevance) and result.link
        ]

        # Sort by relevance score (highest first) - ensure float comparison
        filtered_results.sort(key=lambda x: float(x.relevance_score), reverse=True)

        # Extract URLs up to limit
        urls = [result.link for result in filtered_results[:limit]]

        # Enhanced logging for URL selection process
        logger.info(f"URL selection with threshold {min_relevance}:")
        logger.info(f"  - Total results: {len(search_results)}")
        logger.info(f"  - Above threshold: {len(filtered_results)}")
        logger.info(f"  - Selected for crawling: {len(urls)}")
        logger.info(
            f"  - Rejected: {len(search_results) - len(filtered_results)} below threshold"
        )
        return urls

    except Exception as e:
        logger.error(f"Error selecting URLs for crawling: {e}")
        return []


def format_search_results(search_results: list[SearchResult]) -> str:
    """
    Format search results for display.

    Args:
        search_results: List of search results

    Returns:
        Formatted search results string
    """
    if not search_results:
        return "No search results found."

    result_parts = [f"# Search Results ({len(search_results)} found)", ""]

    for i, result in enumerate(search_results, 1):
        result_parts.extend(
            [
                f"## {i}. {result.title}",
                f"**URL**: {result.link}",
                f"**Source**: {result.source}" if result.source else "",
                f"**Date**: {result.date}" if result.date else "",
                f"**Relevance Score**: {result.relevance_score:.2f}",
                "",
                result.snippet,
                "",
                "---",
                "",
            ]
        )

    return "\n".join(result_parts)


def save_work_product(
    search_results: list[SearchResult],
    crawled_content: list[str],
    urls: list[str],
    query: str,
    session_id: str = "default",
    workproduct_dir: str = None,
    workproduct_prefix: str = "",
) -> str:
    """
    Save detailed search and crawl results to work product file.

    Args:
        search_results: List of search results
        crawled_content: List of cleaned content strings
        urls: List of crawled URLs
        query: Original search query
        session_id: Session identifier
        workproduct_dir: Directory to save work products
        workproduct_prefix: Prefix for workproduct naming (e.g., "editor research")

    Returns:
        Path to saved work product file
    """
    try:
        # Determine correct session directory structure
        if workproduct_dir is None:
            # Default to KEVIN sessions directory with proper categorical organization
            # Use environment-aware path detection for sessions directory
            current_repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if "claudeagent-multiagent-latest" in current_repo:
                base_sessions_dir = f"{current_repo}/KEVIN/sessions"
            else:
                base_sessions_dir = "/home/kjdragan/lrepos/claudeagent-multiagent-latest/KEVIN/sessions"
            session_dir = os.path.join(base_sessions_dir, session_id)
            research_dir = os.path.join(session_dir, "research")
            Path(research_dir).mkdir(parents=True, exist_ok=True)

            # Generate timestamp and filename with proper prefix based on workproduct type
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if workproduct_prefix == "editor research":
                # Editorial search workproduct
                filename = f"editor-search-workproduct_{timestamp}.md"
            else:
                # Regular research workproduct
                filename = f"search_workproduct_{timestamp}.md"

            filepath = os.path.join(research_dir, filename)
        else:
            # Custom workproduct directory (updated to use standard naming)
            Path(workproduct_dir).mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if workproduct_prefix == "editor research":
                # Editorial search workproduct
                filename = f"editor-search-workproduct_{timestamp}.md"
            else:
                # Regular research workproduct
                filename = f"search_workproduct_{timestamp}.md"

            filepath = os.path.join(workproduct_dir, filename)

        # Build work product content
        workproduct_content = [
            "# Enhanced Search+Crawl+Clean Workproduct",
            "",
            f"**Session ID**: {session_id}",
            f"**Export Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "**Agent**: Enhanced Search+Crawl Tool (zPlayground1 integration)",
            f"**Search Query**: {query}",
            f"**Total Search Results**: {len(search_results)}",
            f"**Successfully Crawled**: {len(crawled_content)}",
            "",
            "---",
            "",
            "## 🔍 Search Results Summary",
            "",
        ]

        # Add search results overview
        for i, result in enumerate(search_results, 1):
            workproduct_content.extend(
                [
                    f"### {i}. {result.title}",
                    f"**URL**: {result.link}",
                    f"**Source**: {result.source}" if result.source else "",
                    f"**Date**: {result.date}" if result.date else "",
                    f"**Relevance Score**: {result.relevance_score:.2f}",
                    "",
                    f"**Snippet**: {result.snippet}",
                    "",
                    "---",
                    "",
                ]
            )

        workproduct_content.extend(
            ["", "## 📄 Detailed Crawled Content (AI Cleaned)", ""]
        )

        # Add detailed crawled content
        for i, (content, url) in enumerate(zip(crawled_content, urls, strict=False), 1):
            # Find corresponding search result for title
            title = f"Article {i}"
            for result in search_results:
                if result.link == url:
                    title = result.title
                    break

            workproduct_content.extend(
                [
                    f"## 🌐 {i}. {title}",
                    "",
                    f"**URL**: {url}",
                    f"**Extraction Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    f"**Content Length**: {len(content)} characters",
                    "**Processing**: ✅ Cleaned with GPT-5-nano",
                    "",
                    "### 📄 Full Cleaned Content",
                    "",
                    "---",
                    "",
                    content,
                    "",
                    "---",
                    "",
                ]
            )

        # Add footer
        workproduct_content.extend(
            [
                "",
                "## 📊 Processing Summary",
                "",
                f"- **Search Query**: {query}",
                f"- **Search Results Found**: {len(search_results)}",
                f"- **URLs Successfully Crawled**: {len(crawled_content)}",
                "- **Content Cleaning**: GPT-5-nano AI processing",
                "- **Total Processing Time**: Combined search+crawl+clean in single operation",
                "- **Performance**: Parallel processing with anti-bot detection",
                "",
                "*Generated by Enhanced Search+Crawl+Clean Tool - Powered by zPlayground1 technology*",
            ]
        )

        # Write to file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(workproduct_content))

        logger.info(f"✅ Work product saved to: {filepath}")
        return filepath

    except Exception as e:
        logger.error(f"Error saving work product: {e}")
        return ""


async def search_crawl_and_clean_direct(
    query: str,
    search_type: str = "search",
    num_results: int = 15,
    auto_crawl_top: int = 10,
    crawl_threshold: float = 0.3,
    max_concurrent: int = 15,
    session_id: str = "default",
    anti_bot_level: int = 1,
    workproduct_dir: str = None,
    workproduct_prefix: str = "",
) -> str:
    """
    Combined search, crawl, and clean operation using zPlayground1 technology.

    This function:
    1. Performs search + crawl + clean in a single optimized flow
    2. Saves detailed work product to workproducts directory
    3. Returns full detailed data for orchestrator agent analysis
    4. Uses parallel processing and anti-bot detection

    Args:
        query: Search query
        search_type: "search" or "news"
        num_results: Number of search results to retrieve
        auto_crawl_top: Maximum number of URLs to crawl
        crawl_threshold: Minimum relevance threshold for crawling
        max_concurrent: Maximum concurrent crawling operations
        session_id: Session identifier
        anti_bot_level: Progressive anti-bot level (0-3)
        workproduct_dir: Directory for work products
        workproduct_prefix: Prefix for workproduct naming (e.g., "editor research")

    Returns:
        Full detailed content for orchestrator agent processing
    """
    try:
        start_time = datetime.now()
        logger.info(
            f"Starting enhanced search+crawl+clean for query: '{query}' (anti_bot_level: {anti_bot_level})"
        )

        # Initialize performance timer for this session
        timer = get_performance_timer()
        if timer and timer.session_id != session_id:
            timer.start_session(session_id)

        # Step 1: Intelligent search strategy selection
        from utils.search_strategy_selector import get_search_strategy_selector

        strategy_selector = get_search_strategy_selector()

        strategy_analysis = strategy_selector.select_search_strategy(query)
        logger.info(
            f"Search strategy analysis: {strategy_analysis.recommended_strategy.value} "
            f"(confidence: {strategy_analysis.confidence:.2f})"
        )

        # Determine optimal search type based on strategy
        if strategy_analysis.recommended_strategy.value == "news":
            optimal_search_type = "news"
            logger.info("Using SERP News API based on strategy analysis")
        elif strategy_analysis.recommended_strategy.value == "search":
            optimal_search_type = "search"
            logger.info("Using Google Search API based on strategy analysis")
        else:  # hybrid
            # For hybrid, use the search type with higher weight
            optimal_search_type = search_type if search_type else "search"
            logger.info(f"Using hybrid approach - primary: {optimal_search_type}")

        # Step 2: Execute search with optimal strategy (1-2 seconds)
        async with timed_block("search_execution", metadata={"query": query, "search_type": optimal_search_type}):
            search_results = await execute_serper_search(
                query=query, search_type=optimal_search_type, num_results=num_results
            )

        if not search_results:
            return f"❌ **Search Failed**\n\nNo results found for query: '{query}'"

        # Step 2: Select URLs for crawling based on relevance
        urls_to_crawl = select_urls_for_crawling(
            search_results=search_results,
            limit=auto_crawl_top,
            min_relevance=crawl_threshold,
        )

        if not urls_to_crawl:
            # Return standard search results if no URLs meet crawling threshold
            search_section = format_search_results(search_results)
            return f"{search_section}\n\n**Note**: No URLs met the crawling threshold ({crawl_threshold}). Standard search results provided above."

        # Step 3: Process URLs with IMMEDIATE cleaning after each scrape completes
        # This eliminates the sequential bottleneck between scraping and cleaning
        logger.info(
            f"Processing {len(urls_to_crawl)} URLs with immediate cleaning (anti_bot_level: {anti_bot_level})"
        )

        # Import dependencies
        from utils.anti_bot_escalation import get_escalation_manager
        from agents.content_cleaner_agent import ContentCleaningContext, get_content_cleaner
        from urllib.parse import urlparse

        escalation_manager = get_escalation_manager()
        content_cleaner = get_content_cleaner()
        query_terms = query.split()

        # Semaphores for concurrency control
        scrape_semaphore = asyncio.Semaphore(max_concurrent)
        clean_semaphore = asyncio.Semaphore(max_concurrent)  # No rate limiting restrictions

        # Process each URL: scrape then immediately clean
        async def process_url_immediately(url: str):
            """Scrape URL, then immediately clean it without waiting for others."""

            # Scrape with anti-bot escalation
            async with scrape_semaphore:
                scrape_result = await escalation_manager.crawl_with_escalation(
                    url=url,
                    initial_level=anti_bot_level,
                    max_level=3,
                    use_content_filter=False,
                    session_id=session_id,
                )

            # Check if scraping succeeded
            if not scrape_result.success or not scrape_result.content or len(scrape_result.content.strip()) < 200:
                logger.error(
                    f"❌ Scraping failed: {url} [L{scrape_result.final_level}, {scrape_result.attempts_made} attempts] - {scrape_result.error}"
                )
                return {
                    "success": False,
                    "url": url,
                    "scrape_result": scrape_result,
                }

            # Log successful scrape
            escalation_info = f"L{scrape_result.final_level}"
            if scrape_result.escalation_used:
                escalation_info += f" (escalated)"
            logger.info(
                f"✅ Scraped: {len(scrape_result.content)} chars from {url} [{escalation_info}, {scrape_result.duration:.1f}s]"
            )

            # IMMEDIATELY clean the scraped content (don't wait for other URLs!)
            async with clean_semaphore:
                domain = urlparse(url).netloc.lower()
                context = ContentCleaningContext(
                    search_query=query,
                    query_terms=query_terms,
                    url=url,
                    source_domain=domain,
                    session_id=session_id,
                    min_quality_threshold=50,
                    max_content_length=50000,
                )

                clean_result = await content_cleaner.clean_content(
                    scrape_result.content.strip(), context
                )

            return {
                "success": True,
                "url": url,
                "scrape_result": scrape_result,
                "clean_result": clean_result,
                "context": context,
            }

        # Launch all URLs concurrently - each will scrape then immediately clean
        logger.info("🚀 Launching concurrent scrape+clean processing...")
        tasks = [process_url_immediately(url) for url in urls_to_crawl]

        async with timed_block("concurrent_scrape_and_clean", metadata={"url_count": len(urls_to_crawl)}):
            all_results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        # Step 4: Extract successful results
        cleaned_content_list = []
        cleaned_urls = []
        escalation_stats = []
        cleaning_stats = []

        for result in all_results:
            if isinstance(result, Exception):
                logger.error(f"Exception processing URL: {result}")
                continue

            if not result["success"]:
                # Track failed scrape
                escalation_stats.append({
                    "url": result["url"],
                    "success": False,
                    "attempts": result["scrape_result"].attempts_made,
                    "final_level": result["scrape_result"].final_level,
                    "escalation_used": result["scrape_result"].escalation_used,
                    "duration": result["scrape_result"].duration,
                    "word_count": 0,
                    "char_count": 0,
                })
                continue

            # Track successful scrape
            scrape_result = result["scrape_result"]
            escalation_stats.append({
                "url": result["url"],
                "success": True,
                "attempts": scrape_result.attempts_made,
                "final_level": scrape_result.final_level,
                "escalation_used": scrape_result.escalation_used,
                "duration": scrape_result.duration,
                "word_count": scrape_result.word_count,
                "char_count": scrape_result.char_count,
            })

            # Track cleaning result (accept all content regardless of quality score)
            clean_result = result["clean_result"]
            if clean_result.quality_score >= 0:
                cleaned_content_list.append(clean_result.cleaned_content)
                cleaned_urls.append(result["url"])

                cleaning_stats.append({
                    "url": result["url"],
                    "quality_score": clean_result.quality_score,
                    "quality_level": clean_result.quality_level.value,
                    "relevance_score": clean_result.relevance_score,
                    "word_count": clean_result.word_count,
                    "processing_time": clean_result.processing_time,
                })

        if cleaned_content_list:
            # Step 5: Format results for orchestrator
            logger.info(
                f"Formatting {len(cleaned_content_list)} cleaned articles for orchestrator"
            )

            # Build comprehensive research data
            orchestrator_data = f"""# Research Results for: {query}

## Search Overview
- **Query**: {query}
- **Search Type**: {optimal_search_type}
- **Results Found**: {len(search_results)}
- **URLs Selected**: {len(urls_to_crawl)}
- **Successfully Scraped**: {len([s for s in escalation_stats if s['success']])}
- **Successfully Cleaned**: {len(cleaned_content_list)}
- **Processing Mode**: ⚡ Immediate (Concurrent Scrape+Clean)

## Search Results Summary

"""
            # Add top search results
            orchestrator_data += format_search_results(search_results[:5]) + "\n\n"

            # Add cleaned content
            orchestrator_data += "## Extracted and Cleaned Content\n\n"

            for i, (content, url) in enumerate(
                zip(cleaned_content_list, cleaned_urls, strict=False), 1
            ):
                # Find matching search result
                matching_search = next(
                    (sr for sr in search_results if sr.link == url), None
                )
                title = (
                    matching_search.title
                    if matching_search
                    else "Article"
                )
                source = (
                    matching_search.source or "Unknown"
                    if matching_search
                    else "Unknown"
                )

                # Get cleaning stats for this URL
                clean_stat = next(
                    (cs for cs in cleaning_stats if cs["url"] == url), None
                )
                quality_info = (
                    f"{clean_stat['quality_score']}/100 ({clean_stat['quality_level']})"
                    if clean_stat
                    else "N/A"
                )

                orchestrator_data += f"""### Cleaned Article {i}: {title}
**URL**: {url}
**Source**: {source}
**Quality Score**: {quality_info}
**Content Length**: {len(content)} characters
**Processing**: ✅ Immediate scrape→clean

**FULL CLEANED CONTENT**:
{content}

---

"""

            # Add processing summary
            orchestrator_data += f"""
## PROCESSING SUMMARY

- **Search executed**: {optimal_search_type} search for "{query}"
- **Results found**: {len(search_results)} search results
- **URLs selected for crawling**: {len(urls_to_crawl)} (threshold: {crawl_threshold})
- **Successful scrapes**: {len([s for s in escalation_stats if s['success']])}
- **Successful cleans**: {len(cleaned_content_list)}
- **Total execution time**: {total_duration:.2f}s
- **Anti-bot level used**: {anti_bot_level}
- **Processing mode**: Immediate cleaning (no waiting between scrape and clean stages)

### Escalation Statistics
"""
            for stat in escalation_stats:
                if stat["success"]:
                    orchestrator_data += f"""- {stat['url']}: L{stat['final_level']}, {stat['attempts']} attempts, {stat['duration']:.1f}s, {stat['char_count']} chars
"""

            # Save work product
            work_product_path = save_work_product(
                search_results=search_results,
                crawled_content=cleaned_content_list,
                urls=cleaned_urls,
                query=query,
                session_id=session_id,
                workproduct_dir=workproduct_dir,
                workproduct_prefix=workproduct_prefix,
            )

            logger.info(
                f"✅ Immediate scrape+clean completed in {total_duration:.2f}s - Work product saved to {work_product_path}"
            )
            return orchestrator_data

        else:
            # No successful cleaning: Return search results only
            search_section = format_search_results(search_results)

            # Save partial work product
            work_product_path = save_work_product(
                search_results=search_results,
                crawled_content=[],
                urls=[],
                query=query,
                session_id=session_id,
                workproduct_dir=workproduct_dir,
                workproduct_prefix=workproduct_prefix,
            )

            failed_result = f"""{search_section}

---

**Note**: Content extraction and cleaning failed for all selected URLs. Only search results available.
**Anti-Bot Level**: {anti_bot_level}
**Execution Time**: {total_duration:.2f}s
**Work Product Saved**: {work_product_path}
"""

            logger.warning(
                f"All cleaning failed, returning search results only. Duration: {total_duration:.2f}s"
            )
            return failed_result

    except Exception as e:
        logger.error(f"Error in enhanced search+crawl+clean: {e}")
        return f"❌ **Enhanced Search and Crawl Error**\n\nFailed to execute search and crawl operation: {str(e)}"


async def news_search_and_crawl_direct(
    query: str,
    num_results: int = 15,
    auto_crawl_top: int = 10,
    session_id: str = "default",
    anti_bot_level: int = 1,
    workproduct_dir: str = None,
    workproduct_prefix: str = "",
) -> str:
    """
    Specialized news search with content extraction using enhanced technology.

    Args:
        query: News search query
        num_results: Number of news results to retrieve
        auto_crawl_top: Maximum number of articles to crawl
        session_id: Session identifier
        anti_bot_level: Progressive anti-bot level
        workproduct_dir: Directory for work products
        workproduct_prefix: Prefix for workproduct naming (e.g., "editor research")

    Returns:
        Full detailed news content for orchestrator agent processing
    """
    # Add "latest" and time relevance to news queries
    enhanced_query = f"{query} latest news"

    return await search_crawl_and_clean_direct(
        query=enhanced_query,
        search_type="news",
        num_results=num_results,
        auto_crawl_top=auto_crawl_top,
        crawl_threshold=0.3,  # Standard threshold
        max_concurrent=15,  # Increased concurrency for more URLs
        session_id=session_id,
        anti_bot_level=anti_bot_level,
        workproduct_dir=workproduct_dir,
        workproduct_prefix=workproduct_prefix,
    )


async def search_crawl_and_clean_streaming(
    query: str,
    search_type: str = "search",
    num_results: int = 15,
    auto_crawl_top: int = 10,
    crawl_threshold: float = 0.3,
    max_concurrent_scrapes: int = 15,
    session_id: str = "default",
    anti_bot_level: int = 1,
    workproduct_dir: str = None,
    workproduct_prefix: str = "",
) -> str:
    """
    STREAMING VERSION: Combined search, crawl, and clean with parallel processing.

    This is the NEW streaming implementation that processes URLs immediately
    after scraping rather than waiting for all scrapes to complete.

    Performance: ~30-40% faster than search_crawl_and_clean_direct()
    - Old: ~109s (45s scraping + 64s cleaning sequentially)
    - New: ~65-75s (parallel overlap between scraping and cleaning)

    This function is a drop-in replacement for search_crawl_and_clean_direct()
    with identical interface. Use this for better performance!

    Args:
        query: Search query
        search_type: "search" or "news"
        num_results: Number of search results to retrieve
        auto_crawl_top: Maximum number of URLs to crawl
        crawl_threshold: Minimum relevance threshold for crawling
        max_concurrent_scrapes: Maximum concurrent scraping operations
        session_id: Session identifier
        anti_bot_level: Progressive anti-bot level (0-3)
        workproduct_dir: Directory for work products
        workproduct_prefix: Prefix for workproduct naming (e.g., "editor research")

    Returns:
        Full detailed content for orchestrator agent processing
    """
    # DEPRECATED: This function is now identical to search_crawl_and_clean_direct()
    # The immediate processing optimization has been integrated into the main function
    logger.info("Calling search_crawl_and_clean_direct (now includes immediate processing)")
    return await search_crawl_and_clean_direct(
        query=query,
        search_type=search_type,
        num_results=num_results,
        auto_crawl_top=auto_crawl_top,
        crawl_threshold=crawl_threshold,
        max_concurrent=max_concurrent_scrapes,
        session_id=session_id,
        anti_bot_level=anti_bot_level,
        workproduct_dir=workproduct_dir,
        workproduct_prefix=workproduct_prefix,
    )


# Export commonly used functions
__all__ = [
    "search_crawl_and_clean_direct",
    "search_crawl_and_clean_streaming",
    "news_search_and_crawl_direct",
    "save_work_product",
    "select_urls_for_crawling",
    "SearchResult",
]
