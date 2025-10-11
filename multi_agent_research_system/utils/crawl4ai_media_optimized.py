"""
Crawl4AI utility functions with multimedia content optimization.

This module provides enhanced crawling with complete control over image and multimedia
content to optimize performance for research systems that only need text content.

Key optimizations:
- Complete multimedia exclusion for faster crawling
- Reduced memory usage and bandwidth
- Maintained compatibility with existing anti-bot escalation
- Progressive retry mechanisms with media control
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

# Import Logfire configuration
try:
    from config.logfire_config import configure_logfire, is_logfire_available
    if not is_logfire_available():
        configure_logfire(service_name="crawl4ai-media-optimized")
    import logfire
except ImportError:
    # If logfire config is not available, create a no-op logfire
    class NoOpLogfire:
        def info(self, msg, **kwargs):
            pass
        def warning(self, msg, **kwargs):
            pass
        def error(self, msg, **kwargs):
            pass
        def span(self, msg, **kwargs):
            from contextlib import nullcontext
            return nullcontext()
    logfire = NoOpLogfire()

logger = logging.getLogger(__name__)


@dataclass
class MediaOptimizedCrawlResult:
    """Enhanced result structure with media optimization metrics."""
    url: str
    success: bool
    content: str | None = None
    error: str | None = None
    duration: float = 0.0
    word_count: int = 0
    char_count: int = 0
    anti_bot_level: int = 0
    media_optimization_applied: bool = False
    bandwidth_saved_mb: float = 0.0


class MediaOptimizedCrawler:
    """
    Crawler with complete multimedia content optimization for research systems.

    This crawler is specifically designed for text-only research applications where
    images, videos, and other multimedia content are not needed.
    """

    def __init__(self, browser_configs: dict | None = None):
        """Initialize with optional browser configurations."""
        self.browser_configs = browser_configs or {}
        self._stats = {
            'total_crawls': 0,
            'successful_crawls': 0,
            'failed_crawls': 0,
            'total_duration': 0.0,
            'bandwidth_saved_mb': 0.0,
            'media_optimizations_applied': 0
        }

    def get_media_optimized_config(
        self,
        anti_bot_level: int = 0,
        use_content_filter: bool = False,
        cache_mode: CacheMode = CacheMode.ENABLED
    ) -> CrawlerRunConfig:
        """
        Get crawler configuration with multimedia optimization.

        Args:
            anti_bot_level: Progressive anti-bot level (0-3)
            use_content_filter: Apply content filtering
            cache_mode: Cache mode for crawling

        Returns:
            CrawlerRunConfig with multimedia optimization
        """

        # Base media optimization configuration
        base_config = {
            'text_mode': True,  # Disable images and heavy content
            'exclude_all_images': True,  # Remove all images
            'exclude_external_images': True,  # Block external images
            'light_mode': True,  # Disable background features
            'cache_mode': cache_mode,
            'wait_for': 'body',
            'page_timeout': 30000
        }

        if anti_bot_level == 0:
            # Level 0: Basic media optimization
            config = CrawlerRunConfig(**base_config)

        elif anti_bot_level == 1:
            # Level 1: Enhanced with basic anti-bot
            config = CrawlerRunConfig(
                **base_config,
                simulate_user=True,
                magic=True,
                page_timeout=45000
            )

        elif anti_bot_level == 2:
            # Level 2: Advanced anti-bot with media optimization
            config = CrawlerRunConfig(
                **base_config,
                simulate_user=True,
                magic=True,
                wait_until="domcontentloaded",
                page_timeout=60000,
                delay_before_return_html=1.0
            )

        else:
            # Level 3: Maximum anti-bot with full media optimization
            config = CrawlerRunConfig(
                **base_config,
                simulate_user=True,
                magic=True,
                wait_until="domcontentloaded",
                page_timeout=90000,
                delay_before_return_html=2.0,
                css_selector="main, article, .content, .article-body",
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
            )

        # Add content filtering if requested
        if use_content_filter:
            md_generator = DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(threshold=0.4)
            )
            config.markdown_generator = md_generator

        return config

    def get_browser_config(self, anti_bot_level: int) -> BrowserConfig | None:
        """Get browser configuration based on anti-bot level."""

        if anti_bot_level >= 3:
            return self.browser_configs.get('stealth_browser_config')
        elif anti_bot_level >= 2:
            return self.browser_configs.get('base_browser_config')
        else:
            return None

    async def crawl_url_media_optimized(
        self,
        url: str,
        anti_bot_level: int = 1,
        use_content_filter: bool = False,
        cache_mode: CacheMode = CacheMode.ENABLED
    ) -> MediaOptimizedCrawlResult:
        """
        Crawl a URL with complete multimedia optimization.

        Args:
            url: URL to crawl
            anti_bot_level: Progressive anti-bot level (0-3)
            use_content_filter: Apply content filtering
            cache_mode: Cache mode for crawling

        Returns:
            MediaOptimizedCrawlResult with media optimization metrics
        """
        start_time = datetime.now()

        try:
            with logfire.span("crawl_url_media_optimized",
                             url=url,
                             anti_bot_level=anti_bot_level,
                             media_optimization=True):

                # Get optimized configuration
                config = self.get_media_optimized_config(
                    anti_bot_level, use_content_filter, cache_mode
                )
                browser_config = self.get_browser_config(anti_bot_level)

                # Perform crawl with media optimization
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

                # Estimate bandwidth saved (rough calculation)
                # Average page with images: ~2-5MB, text-only: ~50-200KB
                estimated_bandwidth_saved = 3.0  # Conservative estimate in MB

                # Update statistics
                self._stats['total_crawls'] += 1
                self._stats['bandwidth_saved_mb'] += estimated_bandwidth_saved
                self._stats['media_optimizations_applied'] += 1

                if result.success:
                    self._stats['successful_crawls'] += 1
                else:
                    self._stats['failed_crawls'] += 1
                self._stats['total_duration'] += duration

                return MediaOptimizedCrawlResult(
                    url=url,
                    success=result.success,
                    content=content,
                    error=result.error_message if not result.success else None,
                    duration=duration,
                    word_count=word_count,
                    char_count=char_count,
                    anti_bot_level=anti_bot_level,
                    media_optimization_applied=True,
                    bandwidth_saved_mb=estimated_bandwidth_saved
                )

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self._stats['total_crawls'] += 1
            self._stats['failed_crawls'] += 1
            self._stats['total_duration'] += duration

            logger.error(f"Media optimized crawl failed for {url}: {str(e)}")
            return MediaOptimizedCrawlResult(
                url=url,
                success=False,
                error=str(e),
                duration=duration,
                anti_bot_level=anti_bot_level,
                media_optimization_applied=False,
                bandwidth_saved_mb=0.0
            )

    async def crawl_multiple_media_optimized(
        self,
        urls: list[str],
        anti_bot_level: int = 1,
        use_content_filter: bool = False,
        max_concurrent: int | None = None,
        cache_mode: CacheMode = CacheMode.ENABLED
    ) -> list[MediaOptimizedCrawlResult]:
        """
        Crawl multiple URLs with media optimization and progressive retry.

        Args:
            urls: List of URLs to crawl
            anti_bot_level: Progressive anti-bot level
            use_content_filter: Apply content filtering
            max_concurrent: Maximum concurrent crawls (None for unbounded)
            cache_mode: Cache mode for crawling

        Returns:
            List of MediaOptimizedCrawlResult objects
        """
        if not urls:
            return []

        with logfire.span("crawl_multiple_media_optimized",
                         url_count=len(urls),
                         anti_bot_level=anti_bot_level,
                         max_concurrent=max_concurrent,
                         media_optimization=True):

            # Create semaphore to limit concurrent operations (optional)
            semaphore = (
                asyncio.Semaphore(max_concurrent)
                if max_concurrent and max_concurrent > 0
                else None
            )

            async def crawl_with_semaphore(url: str) -> MediaOptimizedCrawlResult:
                if semaphore is None:
                    return await self.crawl_url_media_optimized(
                        url, anti_bot_level, use_content_filter, cache_mode
                    )

                async with semaphore:
                    return await self.crawl_url_media_optimized(
                        url, anti_bot_level, use_content_filter, cache_mode
                    )

            # Execute crawls concurrently
            tasks = [crawl_with_semaphore(url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle any exceptions
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    final_results.append(MediaOptimizedCrawlResult(
                        url=urls[i],
                        success=False,
                        error=str(result),
                        anti_bot_level=anti_bot_level,
                        media_optimization_applied=False,
                        bandwidth_saved_mb=0.0
                    ))
                else:
                    final_results.append(result)

            # Log summary
            successful = sum(1 for r in final_results if r.success)
            total_bandwidth_saved = sum(r.bandwidth_saved_mb for r in final_results)
            avg_duration = sum(r.duration for r in final_results) / len(final_results)

            logfire.info(
                "Media optimized crawl batch completed",
                total_urls=len(urls),
                successful=successful,
                success_rate=successful / len(urls) if urls else 0,
                avg_duration=avg_duration,
                total_bandwidth_saved_mb=total_bandwidth_saved,
                avg_bandwidth_saved_per_url=total_bandwidth_saved / len(urls) if urls else 0
            )

            logger.info(f"📊 Media optimized crawling completed: {successful}/{len(urls)} successful")
            logger.info(f"💾 Bandwidth saved: {total_bandwidth_saved:.1f}MB total")

            return final_results

    def get_media_optimization_stats(self) -> dict[str, Any]:
        """Get comprehensive media optimization statistics."""
        stats = self._stats.copy()
        if stats['total_crawls'] > 0:
            stats.update({
                'success_rate': stats['successful_crawls'] / stats['total_crawls'],
                'avg_duration': stats['total_duration'] / stats['total_crawls'],
                'avg_bandwidth_saved_per_crawl': stats['bandwidth_saved_mb'] / stats['total_crawls'],
                'media_optimization_effectiveness': stats['media_optimizations_applied'] / stats['total_crawls']
            })
        return stats


# Global media optimized crawler instance
_global_media_optimized_crawler: MediaOptimizedCrawler | None = None


def get_media_optimized_crawler(browser_configs: dict | None = None) -> MediaOptimizedCrawler:
    """Get or create global media optimized crawler instance."""
    global _global_media_optimized_crawler
    if _global_media_optimized_crawler is None or browser_configs:
        _global_media_optimized_crawler = MediaOptimizedCrawler(browser_configs)
    return _global_media_optimized_crawler


# Backward compatibility functions with media optimization

async def crawl_multiple_urls_media_optimized(
    urls: list[str],
    session_id: str,
    max_concurrent: int | None = None,
    extraction_mode: str = "article",
    include_metadata: bool = True,
    base_config=None,
    stealth_config=None,
    undetected_config=None
) -> list[dict]:
    """
    Media optimized version of crawl_multiple_urls_with_results.

    This function provides the same interface but with complete multimedia optimization
    for research systems that only need text content.
    """
    # Create browser configs from legacy parameters
    browser_configs = {}
    if base_config:
        browser_configs['base_browser_config'] = base_config
    if stealth_config:
        browser_configs['stealth_browser_config'] = stealth_config
    if undetected_config:
        browser_configs['undetected_browser_config'] = undetected_config

    # Get media optimized crawler
    crawler = get_media_optimized_crawler(browser_configs)

    # Determine anti-bot level based on configs provided
    anti_bot_level = 1  # Default to enhanced
    if stealth_config or undetected_config:
        anti_bot_level = 2  # Advanced if stealth configs provided

    # Use content filtering for better quality
    use_content_filter = extraction_mode == "article"

    with logfire.span("crawl_multiple_urls_media_optimized",
                     session_id=session_id,
                     url_count=len(urls),
                     extraction_mode=extraction_mode,
                     max_concurrent=max_concurrent,
                     media_optimization=True):

        # Perform media optimized crawling
        effective_concurrency = max_concurrent if max_concurrent and max_concurrent > 0 else None

        results = await crawler.crawl_multiple_media_optimized(
            urls=urls,
            anti_bot_level=anti_bot_level,
            use_content_filter=use_content_filter,
            max_concurrent=effective_concurrency
        )

        # Convert to legacy format with media optimization metadata
        legacy_results = []
        for result in results:
            legacy_result = {
                'url': result.url,
                'success': result.success,
                'content': result.content or '',
                'markdown': result.content or '',  # Legacy field
                'error_message': result.error,
                'duration': result.duration,
                'word_count': result.word_count,
                'char_count': result.char_count,
                'extraction_mode': extraction_mode,
                'session_id': session_id
            }

            # Add metadata if requested
            if include_metadata:
                legacy_result.update({
                    'crawl_timestamp': datetime.now().isoformat(),
                    'anti_bot_level': result.anti_bot_level,
                    'use_content_filter': use_content_filter,
                    'media_optimized': True,
                    'media_optimization_applied': result.media_optimization_applied,
                    'bandwidth_saved_mb': result.bandwidth_saved_mb,
                    'media_optimized_crawler': True
                })

            legacy_results.append(legacy_result)

        return legacy_results


async def scrape_and_clean_single_url_media_optimized(
    url: str,
    session_id: str = "default",
    search_query: str = None,
    extraction_mode: str = "article",
    include_metadata: bool = False,
    preserve_technical_content: bool = True
) -> dict:
    """
    Media optimized single URL scraping with content cleaning.

    This function combines multimedia optimization with your existing content cleaning pipeline.
    """
    try:
        logger.info(f"🚀 Starting media optimized single URL scrape: {url}")

        total_start_time = datetime.now()

        # Get media optimized crawler
        crawler = get_media_optimized_crawler()

        # Execute media optimized crawling
        crawl_result = await crawler.crawl_url_media_optimized(
            url,
            anti_bot_level=1,  # Enhanced anti-bot
            use_content_filter=True  # Always filter for quality
        )

        if not crawl_result.success:
            return {
                'success': False,
                'url': url,
                'content': '',
                'cleaned_content': '',
                'error_message': crawl_result.error,
                'duration': crawl_result.duration,
                'word_count': 0,
                'char_count': 0,
                'extraction_mode': extraction_mode,
                'session_id': session_id,
                'stage': 'failed',
                'metadata': {
                    'media_optimized': True,
                    'media_optimization_applied': crawl_result.media_optimization_applied,
                    'bandwidth_saved_mb': crawl_result.bandwidth_saved_mb,
                    'anti_bot_level': crawl_result.anti_bot_level
                } if include_metadata else None
            }

        # Content cleaning (reuse existing logic)
        cleaned_content = crawl_result.content or ''

        try:
            # Import content cleaning utilities
            from utils.content_cleaning import (
                assess_content_cleanliness,
                clean_content_with_judge_optimization,
            )

            if preserve_technical_content:
                # For technical content, use judge optimization
                logger.info(f"🧠 Judge: Assessing content cleanliness for {url}")
                is_clean, judge_score = await assess_content_cleanliness(cleaned_content, url, 0.75)

                if is_clean:
                    logger.info("✅ Content clean enough - skipping AI cleaning (saving ~35-40 seconds)")
                    cleaning_metadata = {
                        "judge_score": judge_score,
                        "cleaning_performed": False,
                        "optimization_used": True,
                        "latency_saved": "~35-40 seconds"
                    }
                else:
                    logger.info("🧽 Content needs cleaning - running AI cleaning")
                    from utils.content_cleaning import (
                        clean_technical_content_with_gpt5_nano,
                    )
                    cleaned_content = await clean_technical_content_with_gpt5_nano(
                        cleaned_content, url, search_query, session_id
                    )
                    cleaning_metadata = {
                        "judge_score": judge_score,
                        "cleaning_performed": True,
                        "optimization_used": True
                    }
            else:
                # Standard content cleaning
                cleaned_content, cleaning_metadata = await clean_content_with_judge_optimization(
                    cleaned_content, url, search_query, cleanliness_threshold=0.7
                )

        except Exception as clean_error:
            logger.warning(f"Content cleaning failed: {clean_error}, using original content")
            cleaning_metadata = {"cleaning_performed": False, "error": str(clean_error)}

        # Calculate final metrics
        total_duration = (datetime.now() - total_start_time).total_seconds()
        word_count = len(cleaned_content.split()) if cleaned_content else 0
        char_count = len(cleaned_content) if cleaned_content else 0

        logger.info(f"🎉 Media optimized scraping completed: {url} ({char_count} chars in {total_duration:.2f}s)")
        logger.info(f"💾 Bandwidth saved: ~{crawl_result.bandwidth_saved_mb:.1f}MB")

        return {
            'success': True,
            'url': url,
            'content': crawl_result.content or '',
            'cleaned_content': cleaned_content,
            'error_message': None,
            'duration': total_duration,
            'word_count': word_count,
            'char_count': char_count,
            'extraction_mode': extraction_mode,
            'session_id': session_id,
            'stage': 'media_optimized',
            'metadata': {
                'title': '',  # Would be extracted from result if available
                'description': '',
                'status_code': None,
                'extraction_mode': extraction_mode,
                'preserve_technical': preserve_technical_content,
                'cleaning_optimization': cleaning_metadata,
                'media_optimized': True,
                'media_optimization_applied': crawl_result.media_optimization_applied,
                'bandwidth_saved_mb': crawl_result.bandwidth_saved_mb,
                'anti_bot_level': crawl_result.anti_bot_level,
                'media_optimized_crawler': True
            } if include_metadata else None
        }

    except Exception as e:
        logger.error(f"❌ Media optimized scraping error for {url}: {e}")
        return {
            'success': False,
            'url': url,
            'content': '',
            'cleaned_content': '',
            'error_message': f'Media optimized scraping error: {e}',
            'duration': 0,
            'word_count': 0,
            'char_count': 0,
            'extraction_mode': extraction_mode,
            'session_id': session_id,
            'stage': 'error'
        }


# Export media optimized functions
__all__ = [
    'MediaOptimizedCrawler',
    'MediaOptimizedCrawlResult',
    'get_media_optimized_crawler',
    'crawl_multiple_urls_media_optimized',
    'scrape_and_clean_single_url_media_optimized'
]
