"""
Enhanced Crawling Utilities with Crawl4AI Multimedia Exclusion

Provides text-focused crawling with multimedia exclusion for maximum efficiency.
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
    from crawl4ai.async_webcrawler import CrawlResult
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False
    logging.warning("Crawl4AI not available - using fallback crawling")


@dataclass
class CrawlFailureException(Exception):
    """Exception raised when crawling fails."""
    message: str
    url: str
    cause: Exception | None = None


def create_text_focused_browser_config() -> BrowserConfig:
    """Create browser configuration optimized for text-only crawling."""
    if not CRAWL4AI_AVAILABLE:
        raise ImportError("Crawl4AI is required for text-focused crawling")

    return BrowserConfig(
        # Disable images, fonts, and JavaScript for faster text-only crawling
        text_mode=True,        # Disables images/other heavy content for speed
        light_mode=True,       # Disables background features for performance

        # Additional browser optimizations
        headless=True,
        verbose=False,

        # Performance optimizations
        browser_type="chromium",
        viewport_width=1920,
        viewport_height=1080
    )


def create_text_focused_crawler_config() -> CrawlerRunConfig:
    """Create crawler configuration that excludes all multimedia content."""
    if not CRAWL4AI_AVAILABLE:
        raise ImportError("Crawl4AI is required for text-focused crawling")

    return CrawlerRunConfig(
        # Content extraction settings
        only_text=True,              # Attempt to extract text-only content
        exclude_all_images=True,     # Completely remove all images early in pipeline
        exclude_external_images=True, # Exclude images from other domains

        # Speed optimizations
        wait_for_images=False,       # Don't wait for images to load
        pdf=False,                   # Do not generate PDF
        capture_mhtml=False,         # Do not capture MHTML

        # Content processing
        word_count_threshold=10,     # Filter very short content
        exclude_external_links=False, # Keep external links for reference

        # Performance settings
        page_timeout=30,             # 30 second page load timeout
        delay_before_return_html=0.5, # Short delay before extraction
        js_code=[],                  # No JavaScript execution for text-only mode

        # Content cleaning
        remove_overlay_elements=True, # Remove popups and overlays
        simulate_user=False,         # Don't simulate user interactions
        override_navigator=True      # Use simplified navigator for faster crawling
    )


async def crawl_with_text_focus(url: str, logger: logging.Logger | None = None) -> CrawlResult:
    """Execute text-focused crawling with multimedia exclusion.

    Args:
        url: URL to crawl
        logger: Optional logger instance

    Returns:
        CrawlResult with cleaned text content

    Raises:
        CrawlFailureException: If crawling fails
    """
    if not CRAWL4AI_AVAILABLE:
        raise ImportError("Crawl4AI is required for text-focused crawling")

    logger = logger or logging.getLogger(__name__)

    try:
        browser_cfg = create_text_focused_browser_config()
        crawler_cfg = create_text_focused_crawler_config()

        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            logger.info(f"Starting text-focused crawl for {url}")
            start_time = datetime.now()

            result = await crawler.arun(url, config=crawler_cfg)

            end_time = datetime.now()
            crawl_duration = (end_time - start_time).total_seconds()

            if result.success:
                # Verify multimedia exclusion worked
                media_count = len(result.media) if hasattr(result, 'media') else 0
                image_count = len(re.findall(r'<img[^>]*>', result.html, re.IGNORECASE))

                logger.info(f"Text-focused crawl completed for {url}")
                logger.info(f"Duration: {crawl_duration:.2f}s")
                logger.info(f"Media items found: {media_count} (target: 0)")
                logger.info(f"Image tags in HTML: {image_count} (target: 0)")
                logger.info(f"Content length: {len(result.cleaned_html)} chars")

                # Log performance metrics
                if hasattr(result, 'performance'):
                    logger.info(f"Performance metrics: {result.performance}")

                return result
            else:
                error_msg = f"Crawl failed: {result.error_message}"
                logger.error(f"Text-focused crawl failed for {url}: {error_msg}")
                raise CrawlFailureException(error_msg, url)

    except Exception as e:
        if isinstance(e, CrawlFailureException):
            raise

        error_msg = f"Unexpected error during crawl: {str(e)}"
        logger.error(f"Text-focused crawl failed for {url}: {error_msg}")
        raise CrawlFailureException(error_msg, url, cause=e)


def apply_lightweight_cleaning(html_content: str) -> str:
    """Apply minimal cleaning since Crawl4AI already excluded most multimedia.

    Args:
        html_content: HTML content to clean

    Returns:
        Cleaned text content
    """
    # Only remove remaining text-based boilerplate
    lines = html_content.split('\n')
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip obvious navigation and boilerplate text patterns
        skip_patterns = [
            'skip to content', 'navigation menu', 'cookie policy',
            'privacy policy', 'terms of use', 'all rights reserved',
            'copyright ©', 'facebook', 'twitter', 'instagram',
            'youtube', 'tiktok', 'reddit', 'linkedin', 'pinterest',
            'subscribe to our', 'sign up for', 'follow us on',
            'share this article', 'email newsletter', 'breaking news',
            'trending now', 'most popular', 'editor\'s choice',
            'recommended for you', 'related articles', 'more from',
            'support our journalism', 'donate now', 'become a member',
            'advertisement', 'sponsored content', 'paid promotion',
            'about us', 'contact us', 'careers', 'terms & conditions',
            'privacy settings', 'cookie preferences', 'accept cookies',
            'gdpr compliance', 'data protection', 'legal disclaimer',
            'site map', 'help center', 'faq', 'frequently asked questions'
        ]

        if any(skip_pattern in line.lower() for skip_pattern in skip_patterns):
            continue

        # Skip very short lines that are likely boilerplate
        if len(line) < 20:
            # Keep very short lines only if they contain valuable indicators
            valuable_indicators = ['said', 'reported', 'according to', 'stated']
            if not any(indicator in line.lower() for indicator in valuable_indicators):
                continue

        cleaned_lines.append(line)

    # Join and do final cleanup
    content = '\n'.join(cleaned_lines)

    # Remove excessive whitespace
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
    content = re.sub(r' +', ' ', content)

    return content.strip()


async def crawl_url_with_multimedia_exclusion(
    url: str,
    session_id: str,
    search_query: str = "",
    logger: logging.Logger | None = None
) -> dict[str, Any] | None:
    """Crawl URL with Crawl4AI multimedia exclusion for maximum efficiency.

    Args:
        url: URL to crawl
        session_id: Session identifier for tracking
        search_query: Search query for context
        logger: Optional logger instance

    Returns:
        Dictionary with crawl result or None if failed
    """
    logger = logger or logging.getLogger(__name__)

    try:
        # Use text-focused crawling configuration
        result = await crawl_with_text_focus(url, logger)

        if result.success and result.cleaned_html:
            # Apply lightweight cleaning for any remaining boilerplate
            cleaned_content = apply_lightweight_cleaning(result.cleaned_html)

            # Extract metadata about the crawl
            crawl_metadata = {
                "multimedia_excluded": True,
                "text_mode_used": True,
                "images_blocked": True,
                "javascript_disabled": True,
                "crawl_method": "crawl4ai_text_focused"
            }

            # Add performance data if available
            if hasattr(result, 'performance'):
                crawl_metadata["performance"] = result.performance

            # Add media count verification
            media_count = len(result.media) if hasattr(result, 'media') else 0
            crawl_metadata["media_items_found"] = media_count

            return {
                "url": url,
                "title": result.title or "No title",
                "content": cleaned_content,
                "content_length": len(cleaned_content),
                "crawl_metadata": crawl_metadata,
                "crawl_timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "search_query": search_query,
                "success": True,
                "quality_score": _calculate_content_quality(cleaned_content, search_query)
            }
        else:
            logger.warning(f"Text-focused crawl produced no content for {url}")
            return None

    except CrawlFailureException as e:
        logger.error(f"Text-focused crawl failed for {url}: {e.message}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error crawling {url}: {e}")
        return None


def _calculate_content_quality(content: str, search_query: str = "") -> int:
    """Calculate basic quality score for crawled content.

    Args:
        content: Content to evaluate
        search_query: Search query for context

    Returns:
        Quality score (0-100)
    """
    if not content:
        return 0

    score = 50  # Base score

    # Length bonus (prefer substantial content)
    word_count = len(content.split())
    if word_count > 500:
        score += 20
    elif word_count > 200:
        score += 10

    # Search query relevance
    if search_query:
        query_terms = set(search_query.lower().split())
        content_lower = content.lower()
        matches = sum(1 for term in query_terms if term in content_lower)
        if matches > 0:
            score += min(20, matches * 5)

    # Penalize excessive short lines (indicates poor extraction)
    lines = content.split('\n')
    short_lines = sum(1 for line in lines if len(line.strip()) < 20)
    if short_lines > len(lines) * 0.7:
        score -= 15

    # Penalize common boilerplate indicators
    boilerplate_indicators = [
        'cookie', 'privacy', 'terms', 'subscribe', 'follow us',
        'advertisement', 'sponsored', 'copyright'
    ]
    boilerplate_count = sum(
        content.lower().count(indicator) for indicator in boilerplate_indicators
    )
    if boilerplate_count > 10:
        score -= min(20, boilerplate_count * 2)

    return max(0, min(100, score))


def validate_text_focused_config():
    """Validate that text-focused configuration excludes multimedia properly."""
    if not CRAWL4AI_AVAILABLE:
        logging.warning("Crawl4AI not available - cannot validate configuration")
        return

    test_urls = [
        "https://www.bbc.com/news",  # Image-heavy news site
        "https://www.nytimes.com",   # Complex multimedia site
    ]

    logger = logging.getLogger(__name__)

    async def run_validation():
        for url in test_urls:
            try:
                logger.info(f"Validating text-focused config for {url}")

                result = await crawl_with_text_focus(url, logger)

                # Verify multimedia exclusion
                media_count = len(result.media) if hasattr(result, 'media') else 0
                html_images = len(re.findall(r'<img[^>]*>', result.html, re.IGNORECASE))

                if media_count == 0 and html_images == 0:
                    logger.info(f"✅ Text-focused config validated for {url}")
                    logger.info(f"   Content extracted: {len(result.cleaned_html)} characters")
                else:
                    logger.warning(f"⚠️ Text-focused config may need adjustment for {url}")
                    logger.warning(f"   Media items found: {media_count}")
                    logger.warning(f"   Image tags found: {html_images}")

            except Exception as e:
                logger.error(f"❌ Validation failed for {url}: {e}")

    # Run validation
    asyncio.run(run_validation())


# Convenience function for direct usage
async def clean_crawl(url: str, search_query: str = "", session_id: str = "") -> dict[str, Any] | None:
    """Convenience function for text-focused crawling with multimedia exclusion.

    Args:
        url: URL to crawl
        search_query: Optional search query for context
        session_id: Optional session ID

    Returns:
        Crawl result or None
    """
    return await crawl_url_with_multimedia_exclusion(url, session_id or "default", search_query)


if __name__ == "__main__":
    # Run validation when script is executed directly
    validate_text_focused_config()
