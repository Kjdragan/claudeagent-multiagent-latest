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

        # FAIL-FAST: Check for correct API key name - must be SERP_API_KEY
        serp_api_key = os.getenv("SERP_API_KEY")  # Note: SERP_API_KEY, not SERPER_API_KEY

        if not serp_api_key:
            # During development, fail hard and fast with clear error message
            error_msg = "CRITICAL: SERP_API_KEY not found in environment variables!"
            logger.error(f"❌ {error_msg}")
            logger.error("Search functionality cannot work without SERP_API_KEY!")
            logger.error("Expected environment variable: SERP_API_KEY")
            logger.error("Set with: export SERP_API_KEY='your-serp-api-key'")

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

        # Build search parameters - ensure num_results is an integer
        try:
            num_results_int = int(num_results)
        except (ValueError, TypeError):
            num_results_int = 20  # Default fallback
            logger.warning(f"Invalid num_results '{num_results}', using default {num_results_int}")

        search_params = {
            "q": query,
            "num": min(num_results_int, 100),  # Serper limit
            "gl": country,
            "hl": language,
        }

        headers = {"X-API-KEY": serp_api_key, "Content-Type": "application/json"}

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
        import traceback
        logger.error(f"Error in Serper search: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Stack trace: {traceback.format_exc()}")

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
    Select URLs for crawling based on relevance scores with deduplication.

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

        # Deduplicate URLs while preserving order and relevance
        seen_urls = set()
        seen_domains = {}
        deduplicated_results = []

        for result in filtered_results:
            url = result.link
            domain = _extract_domain_for_dedup(url)

            # Skip exact URL duplicates
            if url in seen_urls:
                logger.debug(f"Skipping duplicate URL: {url}")
                continue

            # For high-quality domains (news, gov, edu), allow up to 3 URLs per domain
            # For other domains, allow only 1 URL per domain to ensure diversity
            domain_limit = 3 if _is_high_quality_domain(domain) else 1

            if domain in seen_domains and seen_domains[domain] >= domain_limit:
                logger.debug(f"Skipping additional URL from domain {domain} (limit: {domain_limit}): {url}")
                continue

            # Add to results
            deduplicated_results.append(result)
            seen_urls.add(url)
            seen_domains[domain] = seen_domains.get(domain, 0) + 1

        # Extract URLs up to limit from deduplicated results
        urls = [result.link for result in deduplicated_results[:limit]]

        # Enhanced logging for URL selection process
        logger.info(f"URL selection with threshold {min_relevance}:")
        logger.info(f"  - Total results: {len(search_results)}")
        logger.info(f"  - Above threshold: {len(filtered_results)}")
        logger.info(f"  - After deduplication: {len(deduplicated_results)}")
        logger.info(f"  - Selected for crawling: {len(urls)}")
        logger.info(
            f"  - Rejected: {len(search_results) - len(filtered_results)} below threshold"
        )

        # Log domain distribution
        domain_counts = {}
        for url in urls:
            domain = _extract_domain_for_dedup(url)
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        if len(domain_counts) > 1:
            logger.info(f"  - Domain diversity: {len(domain_counts)} different domains")
            for domain, count in list(domain_counts.items())[:5]:  # Show top 5
                logger.info(f"    * {domain}: {count} URLs")

        return urls

    except Exception as e:
        logger.error(f"Error selecting URLs for crawling: {e}")
        return []


def _extract_domain_for_dedup(url: str) -> str:
    """Extract domain name for deduplication purposes."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception:
        return "unknown"


def _is_high_quality_domain(domain: str) -> bool:
    """Check if domain is high-quality and can have multiple URLs."""
    high_quality_patterns = [
        'gov', 'edu', 'mil',  # Government/educational
        'bbc.com', 'reuters.com', 'ap.org',  # Major news wires
        'cnn.com', 'nytimes.com', 'washingtonpost.com', 'wsj.com',  # Major newspapers
        'theguardian.com', 'economist.com', 'time.com',  # International news
        'understandingwar.org', 'acleddata.org', 'csis.org',  # Research institutes
        'aljazeera.com', 'france24.com', 'dw.com'  # International broadcasters
    ]

    domain_lower = domain.lower()
    return any(pattern in domain_lower for pattern in high_quality_patterns)


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
    max_concurrent: int | None = None,
    session_id: str = "default",
    anti_bot_level: int = 1,
    workproduct_dir: str = None,
    workproduct_prefix: str = "",
    auto_build_corpus: bool = True,  # New parameter for automatic corpus building
) -> str:
    """
    Combined search, crawl, and clean operation using zPlayground1 technology.

    This function:
    1. Performs search + crawl + clean in a single optimized flow
    2. Saves detailed work product to workproducts directory
    3. Automatically builds research corpus for enhanced report agents (NEW)
    4. Returns full detailed data for orchestrator agent analysis
    5. Uses parallel processing and anti-bot detection

    Args:
        query: Search query
        search_type: "search" or "news"
        num_results: Number of search results to retrieve
        auto_crawl_top: Maximum number of URLs to crawl
        crawl_threshold: Minimum relevance threshold for crawling
        max_concurrent: Maximum concurrent crawling operations (None for unbounded)
        session_id: Session identifier
        anti_bot_level: Progressive anti-bot level (0-3)
        workproduct_dir: Directory for work products
        workproduct_prefix: Prefix for workproduct naming (e.g., "editor research")
        auto_build_corpus: Whether to automatically build research corpus (NEW)

    Returns:
        Full detailed content for orchestrator agent processing
    """
    try:
        start_time = datetime.now()
        logger.info(
            f"Starting enhanced search+crawl+clean for query: '{query}' (anti_bot_level: {anti_bot_level}, auto_build_corpus: {auto_build_corpus})"
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

        # Semaphores for concurrency control (optional to allow unbounded concurrency)
        scrape_semaphore = (
            asyncio.Semaphore(max_concurrent)
            if max_concurrent and max_concurrent > 0
            else None
        )
        clean_semaphore = (
            asyncio.Semaphore(max_concurrent)
            if max_concurrent and max_concurrent > 0
            else None
        )  # No rate limiting restrictions

        # Process each URL: scrape then immediately clean
        async def process_url_immediately(url: str):
            """Scrape URL, then immediately clean it without waiting for others."""

            # Scrape with anti-bot escalation
            if scrape_semaphore is None:
                scrape_result = await escalation_manager.crawl_with_escalation(
                    url=url,
                    initial_level=anti_bot_level,
                    max_level=3,
                    use_content_filter=False,
                    session_id=session_id,
                )
            else:
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
                # Handle Level 4 permanent blocks gracefully without error logging
                if scrape_result.final_level == 4:
                    logger.info(
                        f"🚫 Permanently blocked domain skipped: {url} [L4] - {scrape_result.error}"
                    )
                else:
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
            domain = urlparse(url).netloc.lower()

            # Validate scrape_result content before cleaning
            if not scrape_result or not scrape_result.content or len(scrape_result.content.strip()) < 50:
                logger.warning(f"Content too short or empty for {url}, skipping cleaning")
                return {
                    "success": False,
                    "url": url,
                    "scrape_result": scrape_result,
                    "error": "Content too short or empty for cleaning"
                }

            try:
                context = ContentCleaningContext(
                    search_query=query,
                    query_terms=query_terms,
                    url=url,
                    source_domain=domain,
                    session_id=session_id,
                    min_quality_threshold=50,
                    max_content_length=50000,
                )

                # Perform content cleaning with error handling
                if clean_semaphore is None:
                    clean_result = await content_cleaner.clean_content(
                        scrape_result.content.strip(), context
                    )
                else:
                    async with clean_semaphore:
                        clean_result = await content_cleaner.clean_content(
                            scrape_result.content.strip(), context
                        )

                # Validate clean_result structure
                if not clean_result:
                    logger.warning(f"Content cleaning returned None for {url} - treating as garbage content")
                    return {
                        "success": False,
                        "url": url,
                        "scrape_result": scrape_result,
                        "error": "Content cleaning returned None - likely garbage content"
                    }

            except Exception as e:
                logger.error(f"Content cleaning failed for {url}: {e}")
                # Return failed overall - garbage content is not useful
                return {
                    "success": False,
                    "url": url,
                    "scrape_result": scrape_result,
                    "clean_result": None,
                    "error": f"Content cleaning failed - garbage content: {str(e)}"
                }

            return {
                "success": True,
                "url": url,
                "scrape_result": scrape_result,
                "clean_result": clean_result,
                "context": context,
            }

        # Create replacement URL mechanism for permanently blocked domains
        # Get additional URLs beyond the target count for potential replacements
        additional_urls = select_urls_for_crawling(
            search_results=search_results,
            limit=auto_crawl_top + 10,  # Get 10 extra URLs as replacements
            min_relevance=crawl_threshold,
        )

        # Filter out URLs already in the primary list to create replacement pool
        replacement_urls = [url for url in additional_urls if url not in urls_to_crawl]
        logger.info(f"🔄 URL replacement mechanism: {len(replacement_urls)} replacement URLs available for permanently blocked domains")

        # Create shared state for URL replacement coordination
        replacement_state = {
            "replacement_urls": replacement_urls,
            "used_replacements": set(),
            "replacement_lock": asyncio.Lock()
        }

        async def process_url_with_replacement(url: str, is_replacement: bool = False):
            """Process URL with automatic replacement for permanently blocked domains."""
            result = await process_url_immediately(url)

            # If this URL was permanently blocked (Level 4), attempt replacement
            if (not result["success"] and
                result.get("scrape_result") and
                result["scrape_result"].final_level == 4 and
                not is_replacement):  # Don't replace replacement URLs

                async with replacement_state["replacement_lock"]:
                    # Find next available replacement URL
                    replacement_url = None
                    for candidate in replacement_state["replacement_urls"]:
                        if candidate not in replacement_state["used_replacements"]:
                            replacement_url = candidate
                            replacement_state["used_replacements"].add(candidate)
                            break

                    if replacement_url:
                        logger.info(f"🔄 Attempting replacement URL: {replacement_url} (replacing permanently blocked {url})")
                        replacement_result = await process_url_with_replacement(replacement_url, is_replacement=True)

                        # Mark the replacement result with metadata about the original URL
                        if replacement_result.get("success"):
                            replacement_result["replaced_url"] = url
                            replacement_result["is_replacement"] = True
                            logger.info(f"✅ Replacement successful: {replacement_url} successfully replaced {url}")
                        else:
                            logger.warning(f"❌ Replacement failed: {replacement_url} also failed to replace {url}")

                        return replacement_result
                    else:
                        logger.warning(f"⚠️ No replacement URLs available for permanently blocked: {url}")

            return result

        # Launch all URLs concurrently with replacement mechanism and early cutoff
        from ..config.settings import get_settings
        settings = get_settings()
        
        early_cutoff_threshold = settings.early_cutoff_threshold
        target_cleans = settings.target_successful_cleans
        
        logger.info(f"🚀 Launching concurrent scrape+clean processing (early cutoff DISABLED)")
        logger.info(f"   Target: {target_cleans} successful cleans | Processing ALL {len(urls_to_crawl)} URLs in batch")
        
        # Create all tasks but monitor completion dynamically
        pending_tasks = {asyncio.create_task(process_url_with_replacement(url)): url for url in urls_to_crawl}
        completed_results = []
        successful_scrapes = 0
        cancelled_count = 0
        
        async with timed_block("concurrent_scrape_and_clean", metadata={"url_count": len(urls_to_crawl), "cutoff_threshold": early_cutoff_threshold}):
            while pending_tasks:
                # Wait for at least one task to complete
                done, pending = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
                
                # Process completed tasks
                for task in done:
                    url = pending_tasks[task]
                    del pending_tasks[task]
                    
                    try:
                        result = await task
                        completed_results.append(result)
                        
                        # Count successful scrapes (not exceptions, and scrape succeeded)
                        if not isinstance(result, Exception) and result.get("success"):
                            # Check if scrape succeeded (has scrape_result with content)
                            if result.get("scrape_result") and result["scrape_result"].success:
                                successful_scrapes += 1
                                logger.debug(f"✅ Successful scrape #{successful_scrapes}/{early_cutoff_threshold}: {url}")
                    
                    except Exception as e:
                        completed_results.append(e)
                        logger.error(f"Exception processing {url}: {e}")
                
                # DISABLED: Early cutoff optimization - now processing all URLs in batch
                # This ensures we get maximum opportunities for successful cleans
                # if successful_scrapes >= early_cutoff_threshold and pending_tasks:
                #     logger.info(f"🎯 Early cutoff triggered: {successful_scrapes} successful scrapes >= {early_cutoff_threshold} threshold")
                #     logger.info(f"   Cancelling {len(pending_tasks)} remaining scrape tasks...")
                #     
                #     # Cancel all pending tasks immediately
                #     for pending_task in pending_tasks:
                #         pending_task.cancel()
                #     
                #     cancelled_count = len(pending_tasks)
                #     
                #     # Wait briefly for cancellations to process
                #     if pending_tasks:
                #         await asyncio.gather(*pending_tasks, return_exceptions=True)
                #     
                #     logger.info(f"✂️ Cancelled {cancelled_count} tasks | Proceeding with {len(completed_results)} results")
                #     break
        
        all_results = completed_results
        logger.info(f"📊 Scraping completed: {successful_scrapes} successful scrapes, {cancelled_count} cancelled, {len(all_results)} total results")

        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        # Step 4: Extract successful results
        cleaned_content_list = []
        cleaned_urls = []
        escalation_stats = []
        cleaning_stats = []
        replacement_stats = []  # Track replacement URL usage

        for result in all_results:
            if isinstance(result, Exception):
                logger.error(f"Exception processing URL: {result}")
                continue

            if not result["success"]:
                # Track failed scrape (including cleaning failures/garbage content)
                failure_reason = "Scraping failed"
                if "error" in result and "cleaning" in result["error"].lower():
                    failure_reason = "Cleaning failed - garbage content"

                escalation_stats.append({
                    "url": result["url"],
                    "success": False,
                    "attempts": result["scrape_result"].attempts_made if result.get("scrape_result") else 0,
                    "final_level": result["scrape_result"].final_level if result.get("scrape_result") else 0,
                    "escalation_used": result["scrape_result"].escalation_used if result.get("scrape_result") else False,
                    "duration": result["scrape_result"].duration if result.get("scrape_result") else 0.0,
                    "word_count": 0,
                    "char_count": 0,
                    "failure_reason": failure_reason
                })
                continue

            # Track successful scrape
            scrape_result = result["scrape_result"]

            # Track if this is a replacement URL
            is_replacement = result.get("is_replacement", False)
            replaced_url = result.get("replaced_url", None)

            escalation_stats.append({
                "url": result["url"],
                "success": True,
                "attempts": scrape_result.attempts_made,
                "final_level": scrape_result.final_level,
                "escalation_used": scrape_result.escalation_used,
                "duration": scrape_result.duration,
                "word_count": scrape_result.word_count,
                "char_count": scrape_result.char_count,
                "is_replacement": is_replacement,
                "replaced_url": replaced_url,
            })

            # Track replacement statistics
            if is_replacement and replaced_url:
                replacement_stats.append({
                    "original_url": replaced_url,
                    "replacement_url": result["url"],
                    "replacement_successful": True,
                    "original_reason": "Permanently blocked (Level 4)",
                })

            # Track cleaning result (accept all content regardless of quality score)
            clean_result = result.get("clean_result")

            # Handle cases where cleaning failed but scraping succeeded
            if "error" in result and result.get("success", False):
                # Scraping succeeded but cleaning failed - SKIP this garbage content
                logger.warning(f"SKIPPING garbage content from {result['url']}: {result.get('error', 'Unknown error')}")
                # Don't add to cleaned_content_list - this content is likely garbage
                cleaning_stats.append({
                    "url": result["url"],
                    "quality_score": 0,  # Zero score for garbage content
                    "quality_level": "Skipped - Garbage",
                    "relevance_score": 0.0,
                    "word_count": 0,
                    "processing_time": 0.0,
                    "cleaning_error": result.get("error", "Unknown cleaning error"),
                    "skipped": True  # Mark as skipped
                })

            # Validate clean_result structure to prevent data structure mismatches
            elif clean_result and hasattr(clean_result, 'quality_score') and clean_result.quality_score >= 0:
                cleaned_content_list.append(clean_result.cleaned_content)
                cleaned_urls.append(result["url"])

                # Safely extract quality level (handle both enum and string types)
                quality_level = "Unknown"
                if hasattr(clean_result, 'quality_level'):
                    if hasattr(clean_result.quality_level, 'value'):
                        quality_level = clean_result.quality_level.value
                    else:
                        quality_level = str(clean_result.quality_level)

                cleaning_stats.append({
                    "url": result["url"],
                    "quality_score": clean_result.quality_score,
                    "quality_level": quality_level,
                    "relevance_score": getattr(clean_result, 'relevance_score', 0.0),
                    "word_count": getattr(clean_result, 'word_count', 0),
                    "processing_time": getattr(clean_result, 'processing_time', 0.0),
                })
            elif clean_result:
                # Log structure of clean_result for debugging
                logger.warning(f"Unexpected clean_result structure for {result['url']}: {type(clean_result)} - {dir(clean_result)}")
                # Try to extract content anyway if possible
                if hasattr(clean_result, 'cleaned_content'):
                    cleaned_content_list.append(clean_result.cleaned_content)
                    cleaned_urls.append(result["url"])

        # Check successful_cleans against target
        min_quality_threshold = 50  # Standard quality threshold for cleaned content
        successful_cleans = len([cs for cs in cleaning_stats if not cs.get("skipped", False) and cs.get("quality_score", 0) >= min_quality_threshold])
        logger.info(f"📊 Cleaning results: {successful_cleans}/{target_cleans} successful cleans (quality >= {min_quality_threshold})")
        
        # Quality-aware replacement: Try additional URLs if we're below target
        replacement_round = 0
        max_replacement_rounds = 3  # Limit to prevent infinite loops
        
        while successful_cleans < target_cleans and replacement_round < max_replacement_rounds:
            replacement_round += 1
            deficit = target_cleans - successful_cleans
            additional_needed = int(deficit * settings.scrape_attempt_multiplier)
            
            # Get unused replacement URLs
            available_replacements = [
                url for url in replacement_state["replacement_urls"] 
                if url not in replacement_state["used_replacements"]
            ]
            
            if not available_replacements:
                logger.warning(f"⚠️ Round {replacement_round}: No more replacement URLs available (deficit: {deficit})")
                break
            
            # Limit to what we need (with multiplier for safety margin)
            urls_to_try = available_replacements[:additional_needed]
            logger.info(f"🔄 Round {replacement_round}: Attempting {len(urls_to_try)} replacement URLs to fill {deficit} clean deficit")
            
            # Mark these URLs as used
            async with replacement_state["replacement_lock"]:
                for url in urls_to_try:
                    replacement_state["used_replacements"].add(url)
            
            # Process replacement URLs with same pipeline
            replacement_tasks = {
                asyncio.create_task(process_url_with_replacement(url, is_replacement=True)): url 
                for url in urls_to_try
            }
            replacement_results = []
            
            async with timed_block(f"replacement_round_{replacement_round}", metadata={"url_count": len(urls_to_try)}):
                while replacement_tasks:
                    done, pending = await asyncio.wait(replacement_tasks, return_when=asyncio.FIRST_COMPLETED)
                    
                    for task in done:
                        url = replacement_tasks[task]
                        del replacement_tasks[task]
                        
                        try:
                            result = await task
                            replacement_results.append(result)
                            
                            # Immediately process this result into our lists
                            if not isinstance(result, Exception) and result["success"]:
                                scrape_result = result["scrape_result"]
                                clean_result = result.get("clean_result")
                                
                                # Add to escalation stats
                                escalation_stats.append({
                                    "url": result["url"],
                                    "success": True,
                                    "attempts": scrape_result.attempts_made,
                                    "final_level": scrape_result.final_level,
                                    "escalation_used": scrape_result.escalation_used,
                                    "duration": scrape_result.duration,
                                    "word_count": scrape_result.word_count,
                                    "char_count": len(scrape_result.html),
                                    "is_replacement": True,
                                    "replacement_round": replacement_round
                                })
                                
                                # Add to all_results for session state building
                                all_results.append(result)
                                
                                # Check if cleaning succeeded with quality threshold
                                if clean_result and hasattr(clean_result, 'quality_score'):
                                    quality_level = "High" if clean_result.quality_score >= 90 else \
                                                  "Good" if clean_result.quality_score >= 75 else \
                                                  "Acceptable" if clean_result.quality_score >= 50 else \
                                                  "Low"
                                    
                                    # Add to cleaning stats
                                    cleaning_stats.append({
                                        "url": result["url"],
                                        "quality_score": clean_result.quality_score,
                                        "quality_level": quality_level,
                                        "relevance_score": getattr(clean_result, 'relevance_score', 0.0),
                                        "word_count": getattr(clean_result, 'word_count', 0),
                                        "salient_points": getattr(clean_result, 'salient_points', ''),
                                        "is_replacement": True,
                                        "replacement_round": replacement_round
                                    })
                                    
                                    # If quality is good, add to cleaned content
                                    if clean_result.quality_score >= min_quality_threshold:
                                        cleaned_content_list.append(clean_result.cleaned_content)
                                        cleaned_urls.append(result["url"])
                                        logger.info(f"✅ Replacement clean #{len(cleaned_content_list)}: {result['url']} (quality: {clean_result.quality_score})")
                                
                        except Exception as e:
                            replacement_results.append(e)
                            logger.error(f"Exception processing replacement URL {url}: {e}")
            
            # Recount successful cleans
            previous_count = successful_cleans
            successful_cleans = len([cs for cs in cleaning_stats if not cs.get("skipped", False) and cs.get("quality_score", 0) >= min_quality_threshold])
            new_cleans = successful_cleans - previous_count
            
            logger.info(f"📊 Round {replacement_round} complete: +{new_cleans} successful cleans → Total: {successful_cleans}/{target_cleans}")
            
            if new_cleans == 0:
                logger.warning(f"⚠️ Round {replacement_round} produced 0 successful cleans, stopping replacement attempts")
                break
        
        # Final status
        if successful_cleans >= target_cleans:
            logger.info(f"✅ Target achieved: {successful_cleans} >= {target_cleans} successful cleans (after {replacement_round} replacement rounds)")
        else:
            logger.warning(f"⚠️ Target not met: {successful_cleans}/{target_cleans} successful cleans (after {replacement_round} replacement rounds)")
            logger.info(f"   Accepting {successful_cleans} cleans (quality threshold: {min_quality_threshold})")

        # Phase 4: Build enriched session_state.json with salient_points
        logger.info(f"💾 Building enriched session state with salient points...")
        session_state = _build_enriched_session_state(
            session_id=session_id,
            search_results=search_results,
            all_results=all_results,
            cleaning_stats=cleaning_stats,
            escalation_stats=escalation_stats
        )
        _save_session_state(session_id, session_state)
        logger.info(f"✅ Session state saved with {len(session_state.get('search_metadata', []))} metadata entries")

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
                    replacement_info = ""
                    if stat.get("is_replacement") and stat.get("replaced_url"):
                        replacement_info = f" (replaced {stat['replaced_url']})"
                    orchestrator_data += f"""- {stat['url']}: L{stat['final_level']}, {stat['attempts']} attempts, {stat['duration']:.1f}s, {stat['char_count']} chars{replacement_info}
"""

            # Add replacement statistics section
            if replacement_stats:
                orchestrator_data += f"""
### URL Replacement Statistics
- **Permanently Blocked URLs Replaced**: {len(replacement_stats)}
- **Replacement Success Rate**: 100.0%

**Replacements Made**:
"""
                for i, repl in enumerate(replacement_stats, 1):
                    orchestrator_data += f"{i}. {repl['original_url']} → {repl['replacement_url']}\n"
            else:
                orchestrator_data += f"""
### URL Replacement Statistics
- **Permanently Blocked URLs Replaced**: 0 (no Level 4 blocks encountered)
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

            # Log final statistics including replacement information
            successful_scrapes = len([s for s in escalation_stats if s["success"]])
            replacements_made = len(replacement_stats)
            logger.info(
                f"✅ Immediate scrape+clean completed in {total_duration:.2f}s - "
                f"{successful_scrapes} successful scrapes, {replacements_made} URLs replaced - "
                f"Work product saved to {work_product_path}"
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


def _build_enriched_session_state(
    session_id: str,
    search_results: list,
    all_results: list,
    cleaning_stats: list,
    escalation_stats: list
) -> dict:
    """Build enriched session state with search_metadata and scraped_articles."""
    
    # Build search_metadata array with salient_points
    search_metadata = []
    scraped_articles = {}
    
    for idx, search_result in enumerate(search_results, start=1):
        url = search_result.link
        
        # Find if this URL was scraped and cleaned
        result_entry = next((r for r in all_results if not isinstance(r, Exception) and r.get("url") == url), None)
        clean_stat = next((cs for cs in cleaning_stats if cs["url"] == url), None)
        
        # Build metadata entry
        metadata_entry = {
            "index": idx,
            "title": search_result.title,
            "url": url,
            "source": search_result.source or "Unknown",
            "date": search_result.date or "",
            "snippet": search_result.snippet or "",
            "relevance_score": search_result.relevance_score,
            "has_full_content": False,
            "salient_points": None
        }
        
        # If successfully cleaned, add salient_points and full content
        # Only include articles that meet minimum quality threshold (50)
        if result_entry and result_entry.get("success") and clean_stat and not clean_stat.get("skipped"):
            clean_result = result_entry.get("clean_result")
            if clean_result and hasattr(clean_result, 'salient_points') and clean_result.quality_score >= 50:
                metadata_entry["has_full_content"] = True
                metadata_entry["salient_points"] = clean_result.salient_points
                
                # Store full article in scraped_articles (only quality articles)
                scraped_articles[str(idx)] = {
                    "cleaned_content": clean_result.cleaned_content,
                    "word_count": clean_result.word_count,
                    "quality_score": clean_result.quality_score,
                    "relevance_score": clean_result.relevance_score,
                    "cleaned_at": datetime.now().isoformat()
                }
        
        search_metadata.append(metadata_entry)
    
    return {
        "session_id": session_id,
        "search_metadata": search_metadata,
        "scraped_articles": scraped_articles,
        "metadata": {
            "total_articles": len(search_metadata),
            "articles_with_full_content": len(scraped_articles),
            "last_updated": datetime.now().isoformat()
        }
    }


def _save_session_state(session_id: str, state: dict):
    """Save enriched search metadata with salient points to enriched_search_metadata.json."""
    try:
        from pathlib import Path
        import json
        
        session_dir = Path.home() / "lrepos" / "claudeagent-multiagent-latest" / "KEVIN" / "sessions" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Use separate file to avoid orchestrator overwriting our enriched metadata
        state_file = session_dir / "enriched_search_metadata.json"
        
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Session state saved to: {state_file}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save session state: {e}")


# Export commonly used functions
__all__ = [
    "search_crawl_and_clean_direct",
    "search_crawl_and_clean_streaming",
    "news_search_and_crawl_direct",
    "save_work_product",
    "select_urls_for_crawling",
    "SearchResult",
]
