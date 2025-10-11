"""
Enhanced Crawl4AI utility functions adapted from zPlayground1.

This module provides reliable web content extraction with progressive anti-bot
capabilities, parallel processing, and integrated content cleaning.
Adapted for the multi-agent research system.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Try to import crawl4ai components
try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
    from crawl4ai.content_filter_strategy import PruningContentFilter
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
    CRAWL4AI_AVAILABLE = True
except ImportError:
    logger.warning("Crawl4AI not available, install with: pip install crawl4ai>=0.7.4")
    CRAWL4AI_AVAILABLE = False


@dataclass
class CrawlResult:
    """Enhanced result structure for crawl operations."""
    url: str
    success: bool
    content: str | None = None
    cleaned_content: str | None = None
    error: str | None = None
    duration: float = 0.0
    word_count: int = 0
    char_count: int = 0
    anti_bot_level: int = 0
    metadata: dict[str, Any] = None


class SimpleCrawler:
    """
    Enhanced crawler with progressive anti-bot capabilities.

    Based on zPlayground1 research:
    - Basic Crawl4AI achieves 100% success rates
    - Progressive anti-bot levels handle different site protections
    - Parallel processing for efficiency
    """

    def __init__(self, browser_configs: dict | None = None):
        """Initialize with optional browser configurations."""
        self.browser_configs = browser_configs or {}
        self._stats = {
            'total_crawls': 0,
            'successful_crawls': 0,
            'failed_crawls': 0,
            'total_duration': 0.0
        }

    async def crawl_url(
        self,
        url: str,
        anti_bot_level: int = 0,
        use_content_filter: bool = False,
        extraction_mode: str = "article"
    ) -> CrawlResult:
        """
        Crawl a single URL with progressive anti-bot capabilities.

        Args:
            url: URL to crawl
            anti_bot_level: 0=basic, 1=enhanced, 2=advanced, 3=stealth
            use_content_filter: Apply light content filtering
            extraction_mode: Type of content extraction

        Returns:
            CrawlResult with success status and content
        """
        start_time = datetime.now()

        if not CRAWL4AI_AVAILABLE:
            return CrawlResult(
                url=url,
                success=False,
                error="Crawl4AI not available. Install with: pip install crawl4ai>=0.7.4",
                duration=(datetime.now() - start_time).total_seconds()
            )

        try:
            logger.info(f"Crawling {url} with anti-bot level {anti_bot_level}")

            # Progressive anti-bot configuration
            config = self._get_crawl_config(anti_bot_level, use_content_filter, extraction_mode)
            browser_config = self._get_browser_config(anti_bot_level)

            # Perform crawl
            if browser_config:
                async with AsyncWebCrawler(config=browser_config) as crawler:
                    result = await crawler.arun(url, config=config)
            else:
                async with AsyncWebCrawler() as crawler:
                    result = await crawler.arun(url, config=config)

            # Calculate metrics
            duration = (datetime.now() - start_time).total_seconds()
            content = result.markdown if result.success else None
            word_count = len(content.split()) if content else 0
            char_count = len(content) if content else 0

            # Update stats
            self._stats['total_crawls'] += 1
            if result.success:
                self._stats['successful_crawls'] += 1
            else:
                self._stats['failed_crawls'] += 1
            self._stats['total_duration'] += duration

            metadata = {
                'extraction_mode': extraction_mode,
                'anti_bot_level': anti_bot_level,
                'use_content_filter': use_content_filter,
                'cache_mode': getattr(config, 'cache_mode', None)
            }

            return CrawlResult(
                url=url,
                success=result.success,
                content=content,
                error=result.error_message if not result.success else None,
                duration=duration,
                word_count=word_count,
                char_count=char_count,
                anti_bot_level=anti_bot_level,
                metadata=metadata
            )

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self._stats['total_crawls'] += 1
            self._stats['failed_crawls'] += 1
            self._stats['total_duration'] += duration

            logger.error(f"Crawl failed for {url}: {str(e)}")
            return CrawlResult(
                url=url,
                success=False,
                error=str(e),
                duration=duration,
                anti_bot_level=anti_bot_level
            )

    def _get_crawl_config(self, anti_bot_level: int, use_content_filter: bool, extraction_mode: str) -> 'CrawlerRunConfig':
        """Get progressive crawl configuration based on anti-bot level."""

        if not CRAWL4AI_AVAILABLE:
            return None

        if anti_bot_level == 0:
            # Level 0: Basic (works for 6/10 sites)
            config = CrawlerRunConfig(
                cache_mode=CacheMode.ENABLED,
                extraction_mode=extraction_mode
            )

        elif anti_bot_level == 1:
            # Level 1: Enhanced (works for 8/10 sites)
            config = CrawlerRunConfig(
                simulate_user=True,
                magic=True,
                cache_mode=CacheMode.ENABLED,
                extraction_mode=extraction_mode
            )

        elif anti_bot_level == 2:
            # Level 2: Advanced (works for 9/10 sites)
            config = CrawlerRunConfig(
                simulate_user=True,
                magic=True,
                wait_until="domcontentloaded",
                page_timeout=45000,
                cache_mode=CacheMode.ENABLED,
                extraction_mode=extraction_mode
            )

        else:
            # Level 3: Maximum (works for 9.5/10 sites)
            config = CrawlerRunConfig(
                simulate_user=True,
                magic=True,
                wait_until="domcontentloaded",
                page_timeout=60000,
                css_selector="main, article, .content, .article-body",
                cache_mode=CacheMode.ENABLED,
                extraction_mode=extraction_mode
            )

        # Add content filtering if requested
        if use_content_filter:
            md_generator = DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(threshold=0.4)
            )
            config.markdown_generator = md_generator

        return config

    def _get_browser_config(self, anti_bot_level: int) -> Optional['BrowserConfig']:
        """Get browser configuration based on anti-bot level."""

        if not CRAWL4AI_AVAILABLE:
            return None

        if anti_bot_level >= 3:
            # Use stealth browser for maximum anti-bot
            return self.browser_configs.get('stealth_browser_config')
        elif anti_bot_level >= 2:
            # Use enhanced browser for advanced anti-bot
            return self.browser_configs.get('base_browser_config')
        else:
            # Use default browser for basic/enhanced
            return None

    async def crawl_multiple(
        self,
        urls: list[str],
        anti_bot_level: int = 1,
        use_content_filter: bool = False,
        max_concurrent: int | None = None,
        extraction_mode: str = "article"
    ) -> list[CrawlResult]:
        """
        Crawl multiple URLs concurrently with progressive anti-bot.

        Args:
            urls: List of URLs to crawl
            anti_bot_level: Progressive anti-bot level (0-3)
            use_content_filter: Apply content filtering
            max_concurrent: Maximum concurrent crawls
            extraction_mode: Type of content extraction

        Returns:
            List of CrawlResult objects
        """
        if not urls:
            return []

        logger.info(f"Starting crawl of {len(urls)} URLs with anti-bot level {anti_bot_level}")

        # Create semaphore to limit concurrent operations (optional)
        semaphore = (
            asyncio.Semaphore(max_concurrent)
            if max_concurrent and max_concurrent > 0
            else None
        )

        async def crawl_with_semaphore(url: str) -> CrawlResult:
            if semaphore is None:
                return await self.crawl_url(
                    url, anti_bot_level, use_content_filter, extraction_mode
                )

            async with semaphore:
                return await self.crawl_url(url, anti_bot_level, use_content_filter, extraction_mode)

        # Execute crawls concurrently
        tasks = [crawl_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(CrawlResult(
                    url=urls[i],
                    success=False,
                    error=str(result)
                ))
            else:
                final_results.append(result)

        # Log summary
        successful = sum(1 for r in final_results if r.success)
        logger.info(f"Crawl batch completed: {successful}/{len(urls)} successful")

        return final_results

    def get_stats(self) -> dict[str, Any]:
        """Get crawling statistics."""
        return {
            **self._stats,
            'success_rate': self._stats['successful_crawls'] / self._stats['total_crawls'] if self._stats['total_crawls'] > 0 else 0,
            'avg_duration': self._stats['total_duration'] / self._stats['total_crawls'] if self._stats['total_crawls'] > 0 else 0
        }


# Global crawler instance for backward compatibility
_global_crawler: SimpleCrawler | None = None


def get_crawler(browser_configs: dict | None = None) -> SimpleCrawler:
    """Get or create global crawler instance."""
    global _global_crawler
    if _global_crawler is None or browser_configs:
        _global_crawler = SimpleCrawler(browser_configs)
    return _global_crawler


async def crawl_multiple_urls_with_cleaning(
    urls: list[str],
    session_id: str,
    search_query: str = None,
    max_concurrent: int | None = None,
    extraction_mode: str = "article",
    include_metadata: bool = True,
    anti_bot_level: int = 1,
    base_config=None,
    stealth_config=None,
    undetected_config=None
) -> list[dict]:
    """
    Enhanced crawling with integrated content cleaning.

    This function combines crawling with AI-powered content cleaning
    for high-quality, relevant content extraction.

    Args:
        urls: List of URLs to crawl and extract content from
        session_id: Session identifier for tracking
        search_query: Original search query for relevance filtering
        max_concurrent: Maximum concurrent crawling operations (None for unbounded)
        extraction_mode: Type of content extraction (article, etc.)
        include_metadata: Include detailed metadata in results
        anti_bot_level: Progressive anti-bot level (0-3)
        base_config: Base browser configuration
        stealth_config: Stealth browser configuration
        undetected_config: Undetected browser configuration

    Returns:
        List of dictionaries with crawled and cleaned content
    """
    if not CRAWL4AI_AVAILABLE:
        logger.error("Crawl4AI not available")
        return []

    try:
        logger.info(f"Starting enhanced crawling with cleaning for {len(urls)} URLs")

        # Create browser configs from legacy parameters
        browser_configs = {}
        if base_config:
            browser_configs['base_browser_config'] = base_config
        if stealth_config:
            browser_configs['stealth_browser_config'] = stealth_config
        if undetected_config:
            browser_configs['undetected_browser_config'] = undetected_config

        # Get crawler instance
        crawler = get_crawler(browser_configs)

        # Determine anti-bot level from configs if not explicitly set
        if anti_bot_level == 1 and (stealth_config or undetected_config):
            anti_bot_level = 2  # Advanced if stealth configs provided

        # Use content filtering for better quality
        use_content_filter = extraction_mode == "article"

        # Perform crawling
        effective_concurrency = (
            max_concurrent if max_concurrent and max_concurrent > 0 else None
        )

        results = await crawler.crawl_multiple(
            urls=urls,
            anti_bot_level=anti_bot_level,
            use_content_filter=use_content_filter,
            max_concurrent=effective_concurrency,
            extraction_mode=extraction_mode
        )

        # Apply content cleaning to successful results
        cleaned_results = []
        content_cleaning_tasks = []

        for result in results:
            if result.success and result.content and len(result.content.strip()) > 200:
                # Add to cleaning queue
                content_cleaning_tasks.append((result.content, result.url))
            else:
                # Keep as-is without cleaning
                cleaned_results.append({
                    'url': result.url,
                    'success': result.success,
                    'content': result.content or '',
                    'cleaned_content': result.content or '',
                    'markdown': result.content or '',
                    'error_message': result.error,
                    'duration': result.duration,
                    'word_count': result.word_count,
                    'char_count': result.char_count,
                    'extraction_mode': extraction_mode,
                    'session_id': session_id,
                    'anti_bot_level': anti_bot_level,
                    'content_cleaning_performed': False
                })

        # Perform content cleaning in parallel
        if content_cleaning_tasks:
            logger.info(f"Performing AI content cleaning for {len(content_cleaning_tasks)} items")

            try:
                from .z_content_cleaning import clean_content_batch

                cleaned_contents = await clean_content_batch(
                    content_urls=content_cleaning_tasks,
                    search_query=search_query
                )

                # Map cleaned content back to results
                cleaning_index = 0
                for i, result in enumerate(results):
                    if result.success and result.content and len(result.content.strip()) > 200:
                        cleaned_content = cleaned_contents[cleaning_index] if cleaning_index < len(cleaned_contents) else result.content
                        cleaning_index += 1

                        cleaned_results.append({
                            'url': result.url,
                            'success': result.success,
                            'content': result.content or '',
                            'cleaned_content': cleaned_content,
                            'markdown': cleaned_content,
                            'error_message': result.error,
                            'duration': result.duration,
                            'word_count': result.word_count,
                            'char_count': len(cleaned_content) if cleaned_content else 0,
                            'extraction_mode': extraction_mode,
                            'session_id': session_id,
                            'anti_bot_level': anti_bot_level,
                            'content_cleaning_performed': True,
                            'original_length': len(result.content) if result.content else 0,
                            'cleaned_length': len(cleaned_content) if cleaned_content else 0
                        })

            except Exception as e:
                logger.error(f"Content cleaning failed: {e}")
                # Fall back to uncleaned content
                for result in results:
                    if result.success and result.content:
                        cleaned_results.append({
                            'url': result.url,
                            'success': result.success,
                            'content': result.content,
                            'cleaned_content': result.content,
                            'markdown': result.content,
                            'error_message': result.error,
                            'duration': result.duration,
                            'word_count': result.word_count,
                            'char_count': result.char_count,
                            'extraction_mode': extraction_mode,
                            'session_id': session_id,
                            'anti_bot_level': anti_bot_level,
                            'content_cleaning_performed': False,
                            'cleaning_error': str(e)
                        })

        # Add metadata if requested
        if include_metadata:
            for result in cleaned_results:
                result.update({
                    'crawl_timestamp': datetime.now().isoformat(),
                    'use_content_filter': use_content_filter,
                    'enhanced_crawler': True,  # Mark as using enhanced implementation
                    'search_query': search_query
                })

        # Log final statistics
        successful = sum(1 for r in cleaned_results if r['success'])
        cleaned = sum(1 for r in cleaned_results if r.get('content_cleaning_performed', False))
        logger.info(f"Enhanced crawling completed: {successful}/{len(urls)} successful, {cleaned} AI-cleaned")

        return cleaned_results

    except Exception as e:
        logger.error(f"Error in enhanced crawling with cleaning: {e}")
        return []


async def crawl_multiple_urls_direct(
    urls: list[str],
    session_id: str,
    max_concurrent: int | None = None,
    extraction_mode: str = "article"
) -> str:
    """
    Backward compatibility function that returns formatted string results.

    This maintains the original API for components that expect string output.
    """
    results = await crawl_multiple_urls_with_cleaning(
        urls=urls,
        session_id=session_id,
        max_concurrent=max_concurrent,
        extraction_mode=extraction_mode
    )

    # Format as string for backward compatibility
    formatted_results = [
        "# Enhanced Crawl Results",
        f"**URLs Processed**: {len(urls)}",
        f"**Successful**: {sum(1 for r in results if r['success'])}",
        "",
    ]

    for result in results:
        if result['success']:
            formatted_results.extend([
                f"## URL: {result['url']}",
                f"✅ **Success**: Content extracted ({result.get('char_count', 0)} chars)",
                f"**Anti-Bot Level**: {result.get('anti_bot_level', 1)}",
                f"**Content Cleaning**: {'✅ AI-cleaned' if result.get('content_cleaning_performed') else '❌ Not cleaned'}",
                f"**Duration**: {result['duration']:.2f}s",
                "",
                result.get('cleaned_content', result.get('content', ''))[:1000] + ("..." if len(result.get('cleaned_content', result.get('content', ''))) > 1000 else ""),
                "",
                "---",
                ""
            ])
        else:
            formatted_results.extend([
                f"## URL: {result['url']}",
                f"❌ **Failed**: {result.get('error_message', 'Unknown error')}",
                f"**Duration**: {result['duration']:.2f}s",
                "",
                "---",
                ""
            ])

    return "\n".join(formatted_results)


# Legacy timeout configuration compatibility
def get_timeout_for_url(url: str) -> int:
    """
    Legacy timeout function for backward compatibility.

    Returns reasonable default timeout since site categorization
    is handled by anti-bot levels now.
    """
    return 30000  # 30 seconds default timeout


# Export functions for backward compatibility
__all__ = [
    'crawl_multiple_urls_with_cleaning',
    'crawl_multiple_urls_direct',
    'get_crawler',
    'SimpleCrawler',
    'CrawlResult',
    'get_timeout_for_url'
]
