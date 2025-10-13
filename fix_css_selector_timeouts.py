"""
Fix for CSS Selector Timeout Errors in Stage 1

This script provides the corrected crawl configuration to resolve timeout issues.
"""

def get_fixed_stage1_config():
    """
    Returns the corrected Stage 1 crawl configuration.

    Key fixes:
    1. More generic CSS selectors that work across sites
    2. PDF URL detection and handling
    3. Adaptive timeout based on content type
    4. Fallback to universal selector if specific ones fail
    """

    def is_pdf_url(url: str) -> bool:
        """Check if URL points to a PDF document."""
        return url.lower().endswith('.pdf') or 'application/pdf' in url.lower()

    def get_adaptive_config(url: str) -> dict:
        """Get crawl configuration adapted for URL type."""

        if is_pdf_url(url):
            # Skip Stage 1 for PDFs - go directly to Stage 2
            return {
                "skip_stage1": True,
                "reason": "PDF document - DOM selectors not applicable"
            }

        # Universal CSS selectors that work on most sites
        universal_selectors = [
            "main",                     # HTML5 main element
            "article",                  # HTML5 article element
            '[role="main"]',           # ARIA main landmark
            ".content",                # Common content class
            ".article-content",        # Article content class
            ".post-content",           # Post content class
            ".entry-content",          # Entry content class
            "#content",                # Content ID
            ".main-content",           # Main content class
            "div.content",             # Fallback div selector
        ]

        return {
            "cache_mode": "disabled",  # Keep disabled for filtering to work
            "wait_for": "domcontentloaded",  # Wait for full DOM instead of just body
            "page_timeout": 90000,     # Increase timeout to 90 seconds
            "css_selector": ", ".join(universal_selectors),
            "markdown_generator": {
                "content_filter": {
                    "threshold": 0.2,      # Less aggressive pruning
                    "min_word_threshold": 10  # Lower minimum word threshold
                }
            },
            "js_code": """
            // Wait for dynamic content to load
            await new Promise(resolve => setTimeout(resolve, 2000));

            // Remove common problematic elements
            const elementsToRemove = [
                'script', 'style', 'nav', 'header', 'footer',
                '.ads', '.advertisement', '.sidebar', '.menu',
                '.navigation', '.social-media', '.comments'
            ];

            elementsToRemove.forEach(selector => {
                document.querySelectorAll(selector).forEach(el => el.remove());
            });
            """,
            "remove_overlay_elements": True,
            "simulate_user": True,
            "magic": True
        }

    return {
        "is_pdf_url": is_pdf_url,
        "get_adaptive_config": get_adaptive_config
    }

# Example usage in crawl4ai_utils.py
def apply_fix_to_scrape_and_clean_single_url_direct():
    """
    Instructions for applying the fix to the main scraping function.

    Replace lines 960-981 in crawl4ai_utils.py with this logic:
    """

    fix_code = '''
    # ===== STAGE 1: ADAPTIVE CSS SELECTOR EXTRACTION =====
    stage1_start = datetime.now()

    # Import the fix
    from fix_css_selector_timeouts import get_fixed_stage1_config
    config_helper = get_fixed_stage1_config()

    # Check if URL should skip Stage 1
    if config_helper["is_pdf_url"](url):
        logger.info(f"📄 PDF detected, skipping Stage 1 for {url}")
        stage2_result = await _robust_extraction_fallback(url, session_id)
        return await _process_stage2_result(stage2_result, url, search_query, extraction_mode, include_metadata, preserve_technical_content, session_id, total_start_time)

    # Get adaptive configuration
    adaptive_config = config_helper["get_adaptive_config"](url)

    # Apply the configuration
    crawl_config = CrawlerRunConfig(
        cache_mode=CacheMode.DISABLED,
        wait_for="domcontentloaded",
        page_timeout=90000,
        css_selector=adaptive_config["css_selector"],
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(
                threshold=0.2,
                min_word_threshold=10
            )
        ),
        js_code=adaptive_config["js_code"],
        remove_overlay_elements=True,
        simulate_user=True,
        magic=True
    )
    '''

    return fix_code

if __name__ == "__main__":
    print("CSS Selector Timeout Fix")
    print("=" * 50)
    print("This fix addresses:")
    print("1. PDF URL detection and handling")
    print("2. More universal CSS selectors")
    print("3. Increased timeout (90s)")
    print("4. Less aggressive content filtering")
    print("5. JavaScript-based content cleaning")
    print("\nApply the fix in crawl4ai_utils.py around line 960")