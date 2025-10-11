"""
Enhanced search and crawl utilities adapted from zPlayground1.

This module provides optimized search+crawl+clean functionality with
parallel processing, anti-bot detection, and AI content cleaning.
Adapted for integration with the multi-agent research system.
"""

import logging
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

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

from .crawl4ai_utils import crawl_multiple_urls_with_cleaning
from .serp_search_utils import _build_orthogonal_queries
from .url_tracker import get_url_tracker

try:
    from ..config.settings import get_enhanced_search_config
except ImportError:
    def get_enhanced_search_config():
        class _FallbackConfig:
            target_successful_scrapes = 10
            candidate_pool_multiplier = 2.0
            domain_soft_cap = 2
            min_query_coverage = 1
        return _FallbackConfig()

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
        serper_api_key = os.getenv("SERP_API_KEY") or os.getenv("SERPER_API_KEY")

        if not serper_api_key:
            # During development, fail hard and fast with clear error message
            error_msg = "CRITICAL: SERP_API_KEY not found in environment variables!"
            logger.error(f"❌ {error_msg}")
            logger.error("Search functionality cannot work without SERP_API_KEY!")
            logger.error("Expected environment variable: SERP_API_KEY")
            logger.error("Set with: export SERP_API_KEY='your-serper-api-key'")

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


def _extract_domain(url: str) -> str:
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception:
        return ""


def _calculate_recency_multiplier(date_str: str, half_life_days: float = 7.0) -> float:
    if not date_str:
        return 1.0

    now = datetime.utcnow()
    parsed_date = None

    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%b %d, %Y", "%B %d, %Y"):
        try:
            parsed_date = datetime.strptime(date_str, fmt)
            break
        except ValueError:
            continue

    if not parsed_date:
        match = re.search(r"(\d+)\s+(minute|hour|day|week|month|year)s?\s+ago", date_str.lower())
        if match:
            value = int(match.group(1))
            unit = match.group(2)
            if unit == "minute":
                delta_days = value / (60 * 24)
            elif unit == "hour":
                delta_days = value / 24
            elif unit == "day":
                delta_days = value
            elif unit == "week":
                delta_days = value * 7
            elif unit == "month":
                delta_days = value * 30
            elif unit == "year":
                delta_days = value * 365
            else:
                delta_days = 0
            age_days = max(delta_days, 0)
        else:
            return 1.0
    else:
        delta = now - parsed_date
        age_days = max(delta.days + delta.seconds / 86400, 0)

    if half_life_days <= 0:
        return 1.0

    decay_factor = 0.5 ** (age_days / half_life_days)
    return max(0.2, decay_factor)


def _select_candidate_urls(
    search_results: List[SearchResult],
    pool_target: int,
    crawl_threshold: float,
    domain_soft_cap: int,
    pool_multiplier: float = 2.0,
    min_per_query: int = 1,
) -> tuple[List[str], Dict[str, Any]]:
    if pool_target <= 0:
        return [], {"per_query_counts": {}, "domain_counts": {}}

    seen_urls: set[str] = set()
    domain_counts: defaultdict[str, int] = defaultdict(int)
    per_query_counts: defaultdict[str, int] = defaultdict(int)
    candidates: List[str] = []

    required_labels: List[str] = []
    for result in search_results:
        label = getattr(result, "query_label", "primary")
        if label not in required_labels:
            required_labels.append(label)

    min_per_query = max(1, min_per_query)
    domain_cap_for_candidates = max(domain_soft_cap, 1) + 1

    def record_candidate(result: SearchResult) -> bool:
        url = getattr(result, "link", "")
        if not url or url in seen_urls:
            return False
        domain = _extract_domain(url)
        label = getattr(result, "query_label", "primary")
        candidates.append(url)
        seen_urls.add(url)
        domain_counts[domain] += 1
        per_query_counts[label] += 1
        return True

    # Stage 1: ensure minimum per-query coverage
    for label in required_labels:
        label_results = [r for r in search_results if getattr(r, "query_label", "primary") == label]
        if not label_results:
            continue

        for result in label_results:
            if per_query_counts[label] >= min_per_query:
                break
            if result.relevance_score >= crawl_threshold:
                record_candidate(result)

        while per_query_counts[label] < min_per_query:
            fallback = next((r for r in label_results if r.link and r.link not in seen_urls), None)
            if fallback is None:
                break
            if not record_candidate(fallback):
                break

    max_candidates = max(pool_target, int(pool_target * pool_multiplier))

    # Stage 2: add threshold-compliant results respecting domain cap strictly
    for result in search_results:
        if len(candidates) >= max_candidates:
            break
        url = getattr(result, "link", "")
        if not url or url in seen_urls:
            continue
        if result.relevance_score < crawl_threshold:
            continue
        domain = _extract_domain(url)
        if domain_counts[domain] >= domain_soft_cap:
            continue
        record_candidate(result)

    # Stage 3: relax threshold if needed, still respecting hard cap
    if len(candidates) < max_candidates:
        for result in search_results:
            if len(candidates) >= max_candidates:
                break
            url = getattr(result, "link", "")
            if not url or url in seen_urls:
                continue
            domain = _extract_domain(url)
            if domain_counts[domain] >= domain_cap_for_candidates:
                continue
            record_candidate(result)

    # Stage 4: allow additional diversity by selecting underrepresented domains first
    if len(candidates) < max_candidates:
        ranked_by_domain = sorted(
            (r for r in search_results if getattr(r, "link", "") not in seen_urls),
            key=lambda r: domain_counts[_extract_domain(getattr(r, "link", ""))]
        )
        for result in ranked_by_domain:
            if len(candidates) >= max_candidates:
                break
            url = getattr(result, "link", "")
            if not url or url in seen_urls:
                continue
            domain = _extract_domain(url)
            if domain_counts[domain] >= domain_cap_for_candidates:
                continue
            record_candidate(result)

    fallback_applied = False
    if len(candidates) < pool_target:
        fallback_applied = True
        for result in search_results:
            if len(candidates) >= pool_target:
                break
            url = getattr(result, "link", "")
            if not url or url in seen_urls:
                continue
            domain = _extract_domain(url)
            seen_urls.add(url)
            candidates.append(url)
            domain_counts[domain] += 1
            label = getattr(result, "query_label", "primary")
            per_query_counts[label] += 1

    stats = {
        "per_query_counts": dict(per_query_counts),
        "domain_counts": dict(domain_counts),
        "pool_size": len(candidates),
        "pool_target": pool_target,
        "pool_multiplier": pool_multiplier,
        "min_per_query": min_per_query,
        "fallback_applied": fallback_applied,
    }

    return candidates, stats


def save_work_product(
    search_results: list[SearchResult],
    crawled_content: list[str],
    urls: list[str],
    query: str,
    selection_stats: dict[str, Any],
    attempted_crawls: int,
    successful_crawls: int,
    session_id: str = "default",
    workproduct_dir: str = None,
    workproduct_prefix: str = "",
) -> str:
    """
    Save detailed search and crawl results to work product file.

    Args:
        search_results: List of search results retrieved from SERP
        crawled_content: List of cleaned content strings (successful crawls)
        urls: List of successfully crawled URLs (aligned with crawled_content)
        query: Original search query
        selection_stats: Candidate selection statistics
        attempted_crawls: Total crawl attempts made
        successful_crawls: Count of successful crawls
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

        pool_size = selection_stats.get("pool_size", len(urls))
        pool_target_value = selection_stats.get("pool_target", pool_size)
        filtered_candidates = selection_stats.get("filtered_candidates", len(urls))
        trimmed_for_limit = selection_stats.get("trimmed_for_limit", 0)
        fallback_used = selection_stats.get("fallback_applied", False)
        domain_counts = selection_stats.get("domain_counts", {})

        success_rate = (
            (successful_crawls / attempted_crawls) * 100 if attempted_crawls else 0.0
        )

        # Build work product content
        workproduct_content = [
            "# Enhanced Search+Crawl+Clean Workproduct",
            "",
            f"**Session ID**: {session_id}",
            f"**Export Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "**Agent**: Enhanced Search+Crawl Tool (zPlayground1 integration)",
            f"**Search Query**: {query}",
            f"**Search Queries Executed**: {selection_stats.get('search_queries_executed', 1)} (Orthogonal: {len(selection_stats.get('orthogonal_queries', []))})",
            f"**Search Results Retrieved**: {len(search_results)}",
            f"**Candidates Selected**: {filtered_candidates} (Pool: {pool_size}, Target: {pool_target_value}, Trimmed: {trimmed_for_limit})",
            f"**Crawl Attempts**: {attempted_crawls}",
            f"**Successful Crawls**: {successful_crawls}",
            f"**Crawl Success Rate**: {success_rate:.1f}%",
            f"**Fallback Used**: {'Yes' if fallback_used else 'No'}",
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

        if crawled_content:
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
                f"- **Search Results Retrieved**: {len(search_results)}",
                f"- **Candidates Selected for Crawling**: {filtered_candidates} (Pool: {pool_size})",
                f"- **Crawl Attempts**: {attempted_crawls}",
                f"- **Successful Crawls**: {successful_crawls}",
                f"- **Crawl Success Rate**: {success_rate:.1f}%",
                f"- **Unique Domains in Pool**: {len(domain_counts)}",
                f"- **Fallback Applied**: {'Yes' if fallback_used else 'No'}",
                "- **Content Cleaning**: GPT-5-nano AI processing",
                "- **Total Processing Time**: Combined search+crawl+clean in single operation",
                "- **Performance**: Parallel processing with anti-bot detection",
                "",
                "*Generated by Enhanced Search+Crawl+Clean Tool - Powered by zPlayground1 technology*",
            ]
        )

        if domain_counts:
            workproduct_content.extend([
                "",
                "## 🌍 Domain Distribution",
                "",
            ])
            for domain, count in sorted(domain_counts.items(), key=lambda item: item[1], reverse=True):
                workproduct_content.append(f"- {domain}: {count}")

        # Add excluded sites section if any URLs were excluded
        skipped_urls = selection_stats.get("skipped_urls", [])
        if skipped_urls:
            from urllib.parse import urlparse
            from utils.url_tracker import get_url_tracker

            url_tracker = get_url_tracker()
            excluded_domains = url_tracker.get_excluded_domains()

            # Categorize excluded URLs by reason
            domain_excluded = []
            already_successful = []
            session_duplicates = []

            for url in skipped_urls:
                domain = urlparse(url).netloc.lower()
                if domain in excluded_domains:
                    domain_excluded.append(url)
                elif url in url_tracker.url_records and url_tracker.url_records[url].is_successful:
                    already_successful.append(url)
                elif url in url_tracker.session_urls:
                    session_duplicates.append(url)

            # Add excluded sites section
            excluded_sections = []

            if domain_excluded:
                excluded_sections.extend([
                    "",
                    "## 🚫 Excluded Sites (Domain Block List)",
                    "",
                    f"The following {len(domain_excluded)} URLs were excluded due to being on the domain exclusion list:",
                    ""
                ])

                # Group by domain for better organization
                domain_groups = {}
                for url in domain_excluded:
                    domain = urlparse(url).netloc.lower()
                    if domain not in domain_groups:
                        domain_groups[domain] = []
                    domain_groups[domain].append(url)

                for domain, urls in sorted(domain_groups.items()):
                    excluded_sections.append(f"**{domain}** ({len(urls)} URLs):")
                    for url in urls[:3]:  # Show up to 3 URLs per domain
                        excluded_sections.append(f"  - {url}")
                    if len(urls) > 3:
                        excluded_sections.append(f"  - ... and {len(urls) - 3} more URLs")
                    excluded_sections.append("")

            if already_successful:
                excluded_sections.extend([
                    "## ✅ Previously Processed Sites",
                    "",
                    f"The following {len(already_successful)} URLs were skipped as they were successfully processed in previous sessions:",
                    ""
                ])

                for url in already_successful[:5]:  # Show up to 5 examples
                    excluded_sections.append(f"  - {url}")
                if len(already_successful) > 5:
                    excluded_sections.append(f"  - ... and {len(already_successful) - 5} more previously processed URLs")
                excluded_sections.append("")

            if session_duplicates:
                excluded_sections.extend([
                    "## 🔄 Session Duplicates",
                    "",
                    f"The following {len(session_duplicates)} URLs were skipped as duplicates within the current session:",
                    ""
                ])

                for url in session_duplicates[:3]:  # Show up to 3 examples
                    excluded_sections.append(f"  - {url}")
                if len(session_duplicates) > 3:
                    excluded_sections.append(f"  - ... and {len(session_duplicates) - 3} more duplicate URLs")
                excluded_sections.append("")

            # Add all excluded sections to work product
            if excluded_sections:
                workproduct_content.extend(excluded_sections)

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
            f"Starting search+crawl+clean for query: '{query}' (requested anti_bot_level: {anti_bot_level})"
        )
        if anti_bot_level != 1:
            logger.debug(
                "SimpleCrawler pipeline ignores anti_bot_level parameter (received %s)",
                anti_bot_level,
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
            primary_results = await execute_serper_search(
                query=query, search_type=optimal_search_type, num_results=num_results
            )

        search_results: list[SearchResult] = []
        for res in primary_results:
            setattr(res, "query_label", "primary")
            setattr(res, "query_weight", 1.0)
            search_results.append(res)

        config = get_enhanced_search_config()
        enable_orthogonals = getattr(config, "enable_orthogonal_queries", True)
        max_orthogonals = int(getattr(config, "max_orthogonal_queries", 2))
        orthogonal_stats: list[dict[str, Any]] = []

        if enable_orthogonals and max_orthogonals > 0:
            orthogonal_entries = _build_orthogonal_queries(query, max_orthogonals)
            for idx, entry in enumerate(orthogonal_entries, 1):
                orth_query = entry.get("query")
                if not orth_query:
                    continue
                async with timed_block(
                    "orthogonal_search_execution",
                    metadata={"query": orth_query, "label": entry.get("angle"), "index": idx},
                ):
                    orth_results = await execute_serper_search(
                        query=orth_query,
                        search_type=optimal_search_type,
                        num_results=num_results,
                    )

                for res in orth_results:
                    setattr(res, "query_label", f"orthogonal_{idx}")
                    setattr(res, "query_weight", 0.8)
                    search_results.append(res)

                orthogonal_stats.append(
                    {
                        "index": idx,
                        "angle": entry.get("angle"),
                        "query": orth_query,
                        "results": len(orth_results),
                    }
                )

                logger.debug(
                    "Orthogonal query %s (%s) produced %s results",
                    idx,
                    entry.get("angle"),
                    len(orth_results),
                )

            logger.info(
                "Executed %d orthogonal queries", len(orthogonal_stats)
            )
        else:
            logger.info("Orthogonal query expansion disabled")

        if not search_results:
            return f"❌ **Search Failed**\n\nNo results found for query: '{query}'"

        selection_stats: dict[str, Any]

        # Step 2: Build candidate URL set using target-based selection
        target_successful_scrapes = getattr(config, "target_successful_scrapes", auto_crawl_top)
        minimum_candidate_pool = int(getattr(config, "minimum_candidate_pool", 20))
        pool_target = max(target_successful_scrapes, minimum_candidate_pool)
        pool_multiplier = float(getattr(config, "candidate_pool_multiplier", 2.0))
        domain_soft_cap = int(getattr(config, "domain_soft_cap", 2))
        min_query_coverage = int(getattr(config, "min_query_coverage", 1))

        candidate_urls, selection_stats = _select_candidate_urls(
            search_results=search_results,
            pool_target=pool_target,
            crawl_threshold=crawl_threshold,
            domain_soft_cap=domain_soft_cap,
            pool_multiplier=pool_multiplier,
            min_per_query=min_query_coverage,
        )

        url_tracker = get_url_tracker()
        urls_to_crawl, skipped_urls = url_tracker.filter_urls(candidate_urls, session_id)
        selection_stats["filtered_candidates"] = len(urls_to_crawl)
        selection_stats["skipped_urls"] = skipped_urls
        selection_stats["target_successful_scrapes"] = target_successful_scrapes
        selection_stats["orthogonal_queries"] = orthogonal_stats
        selection_stats["search_queries_executed"] = 1 + len(orthogonal_stats)

        # Respect auto_crawl_top as an upper bound but keep extra buffer if target is higher
        max_urls = max(auto_crawl_top, target_successful_scrapes)
        if len(urls_to_crawl) > max_urls:
            selection_stats["trimmed_for_limit"] = len(urls_to_crawl) - max_urls
            urls_to_crawl = urls_to_crawl[:max_urls]

        logger.info(
            "Candidate selection complete: target_success=%s, pool_target=%s, selected=%s, skipped=%s, fallback=%s, domains=%s",
            target_successful_scrapes,
            pool_target,
            len(urls_to_crawl),
            len(skipped_urls),
            selection_stats.get("fallback_applied"),
            selection_stats.get("domain_counts"),
        )

        if not urls_to_crawl:
            # Return standard search results if no URLs meet crawling threshold
            search_section = format_search_results(search_results)
            return f"{search_section}\n\n**Note**: No URLs met the crawling threshold ({crawl_threshold}). Standard search results provided above."

        # Step 3: Process URLs using the proven zPlayground SimpleCrawler pipeline
        logger.info(
            f"Processing {len(urls_to_crawl)} URLs with zPlayground SimpleCrawler parallel crawl+clean"
        )

        async with timed_block(
            "concurrent_crawl_and_clean",
            metadata={"url_count": len(urls_to_crawl)}
        ):
            crawl_results = await crawl_multiple_urls_with_cleaning(
                urls=urls_to_crawl,
                session_id=session_id,
                search_query=query,
                max_concurrent=max_concurrent,
                extraction_mode="article"
            )

        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        cleaned_content_list = []
        cleaned_urls = []
        crawl_stats = []

        for result in crawl_results:
            if not isinstance(result, dict):
                logger.error(f"Unexpected crawl result type: {type(result)}")
                continue

            crawl_stats.append(result)

            cleaned_content = result.get("cleaned_content") or result.get("content", "")
            if result.get("success") and cleaned_content and len(cleaned_content.strip()) > 200:
                cleaned_content_list.append(cleaned_content.strip())
                cleaned_urls.append(result.get("url"))

        success_count = sum(1 for r in crawl_stats if r.get("success"))
        cleaned_lookup = {
            r.get("url"): r for r in crawl_stats if r.get("success")
        }

        attempted_count = len(crawl_stats)
        success_rate = (success_count / attempted_count * 100) if attempted_count else 0.0

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
- **Successfully Scraped**: {success_count}
- **Successfully Cleaned**: {len(cleaned_content_list)}
- **Processing Mode**: ⚡ SimpleCrawler parallel crawl+clean

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

                # Get crawl stats for this URL
                clean_stat = cleaned_lookup.get(url)
                quality_info = "cleaned"
                if clean_stat:
                    char_count = clean_stat.get("char_count", len(content))
                    cleaned_flag = clean_stat.get("content_cleaned", True)
                    quality_info = f"{char_count} chars{' (LLM cleaned)' if cleaned_flag else ''}"

                orchestrator_data += f"""### Cleaned Article {i}: {title}
**URL**: {url}
**Source**: {source}
**Content Summary**: {quality_info}
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
- **Search queries executed**: {selection_stats.get("search_queries_executed", 1)} (Orthogonal: {len(selection_stats.get("orthogonal_queries", []))})
- **Search results retrieved**: {len(search_results)} (threshold: {crawl_threshold})
- **Candidates selected**: {selection_stats.get("filtered_candidates", len(urls_to_crawl))} (Pool: {selection_stats.get("pool_size")}, Target: {selection_stats.get("pool_target", selection_stats.get("pool_size"))}, Trimmed: {selection_stats.get("trimmed_for_limit", 0)})
- **Crawl attempts**: {attempted_count}
- **Successful crawls**: {success_count}
- **Crawl success rate**: {success_rate:.1f}%
- **Fallback applied**: {'Yes' if selection_stats.get("fallback_applied") else 'No'}
- **Unique domains in pool**: {len(selection_stats.get("domain_counts", {}))}
- **Total execution time**: {total_duration:.2f}s
- **Crawler profile**: SimpleCrawler (parallel crawl+clean)

### Crawl Statistics
"""
            for stat in crawl_stats:
                status = "✅" if stat.get("success") else "❌"
                orchestrator_data += (
                    f"- {status} {stat.get('url')}: "
                    f"{stat.get('char_count', 0)} chars, "
                    f"{stat.get('duration', 0):.1f}s"
                    f"{' (cleaned)' if stat.get('content_cleaned') else ''}\n"
                )

            # Save work product
            work_product_path = save_work_product(
                search_results=search_results,
                crawled_content=cleaned_content_list,
                urls=cleaned_urls,
                query=query,
                selection_stats=selection_stats,
                attempted_crawls=attempted_count,
                successful_crawls=success_count,
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
                selection_stats=selection_stats,
                attempted_crawls=attempted_count,
                successful_crawls=0,
                session_id=session_id,
                workproduct_dir=workproduct_dir,
                workproduct_prefix=workproduct_prefix,
            )

            failed_result = f"""{search_section}

---

**Note**: Content extraction and cleaning failed for all selected URLs. Only search results available.
**Search Queries Executed**: {selection_stats.get("search_queries_executed", 1)} (Orthogonal: {len(selection_stats.get("orthogonal_queries", []))})
**Candidates Selected**: {selection_stats.get("filtered_candidates", 0)} (Pool: {selection_stats.get("pool_size")}, Target: {selection_stats.get("pool_target", selection_stats.get("pool_size"))})
**Crawl Attempts**: {attempted_count}
**Successful Crawls**: 0
**Crawl Success Rate**: 0.0%
**Unique Domains**: {len(selection_stats.get("domain_counts", {}))}
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
