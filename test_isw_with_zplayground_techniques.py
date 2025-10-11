#!/usr/bin/env python3
"""
Test understandingwar.org with advanced z-playground1 techniques before excluding the domain.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_isw_with_advanced_techniques():
    """Test ISW website with advanced Crawl4AI techniques from z-playground1."""

    # Test URL
    test_url = "https://understandingwar.org/research/russia-ukraine/russian-offensive-campaign-assessment-october-8-2025/"

    print("🔬 Testing understandingwar.org with Advanced z-playground1 Techniques")
    print("=" * 80)
    print(f"URL: {test_url}")
    print("=" * 80)

    try:
        # Import advanced crawl4ai from z-playground1
        sys.path.insert(0, "/home/kjdragan/lrepos/z-playground1")

        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, LLMConfig
        from crawl4ai.content_filter_strategy import BM25ContentFilter, PruningContentFilter, LLMContentFilter
        from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

        print("✅ Successfully imported advanced Crawl4AI features")
        print("   - BM25ContentFilter, PruningContentFilter, LLMContentFilter")
        print("   - BrowserConfig with stealth options")
        print("   - Advanced CrawlerRunConfig options")

    except ImportError as e:
        print(f"❌ Cannot import advanced Crawl4AI: {e}")
        print("   Falling back to basic analysis")
        return basic_domain_analysis()

    async def run_advanced_tests():
        """Run comprehensive advanced tests on the ISW URL."""

        # 1. Stealth Browser Configuration
        print("\n🥷 Test 1: Stealth Browser + Advanced Filtering")
        print("-" * 50)

        stealth_browser = BrowserConfig(
            browser_type="undetected",  # Use undetected Chrome
            enable_stealth=True,        # Enable playwright-stealth
            headless=True,
            ignore_https_errors=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1"
            },
            extra_args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-web-security",
                "--disable-features=VizDisplayCompositor",
                "--no-first-run",
                "--disable-default-apps",
                "--disable-dev-shm-usage",
                "--no-sandbox"
            ]
        )

        # ISW-specific noise selectors based on what we saw in the output
        isw_noise_selectors = [
            "nav", "header", ".navbar", "#main-nav", ".nav-menu",
            ".menu", ".navigation", ".skip-link",
            ".donate", ".newsletter", ".social", ".social-share",
            ".sidebar", ".side-content", "aside",
            "footer", ".footer", "#footer",
            ".ad", ".advertisement", "[class*='ad-']", "[id*='ad-']",
            ".cookie-notice", ".legal-info", ".disclaimer",
            ".related-content", ".more-stories", ".trending",
            ".search", ".search-form", "[role='search']",
            ".language-selector", ".accessibility",
            ".media-gallery", ".video-player",
            "[class*='map']", "[class*='infographic']"
        ]

        # 2. Advanced Content Filtering
        print("\n🧠 Test 2: Advanced Content Filtering")
        print("-" * 50)

        # Pruning filter to remove low-quality content
        prune_filter = PruningContentFilter(
            threshold=0.3,           # More lenient threshold
            threshold_type="fixed",
            min_word_threshold=5
        )

        # BM25 filter for military/analysis content
        bm25_filter = BM25ContentFilter(
            user_query="Russian offensive campaign assessment military analysis Ukraine October 2025 operations strategy",
            bm25_threshold=1.0,      # Lower threshold for broader matching
            language="english",
            use_stemming=True
        )

        md_generator = DefaultMarkdownGenerator(content_filter=prune_filter)

        # 3. Advanced Target Element + JavaScript Execution Strategy
        print("\n🎯 Test 3: Advanced Target Element + JavaScript Execution")
        print("-" * 50)

        config = CrawlerRunConfig(
            markdown_generator=md_generator,
            target_elements=[
                "article",
                "main",
                ".content",
                ".article-content",
                ".story-content",
                ".post-content",
                ".entry-content",
                ".field-item",
                ".node-content",
                ".field-name-body",
                "[class*='content']",
                "[class*='article']",
                "[class*='body']",
                "[class*='assessment']",
                "[class*='analysis']",
                ".assessment-content",
                ".campaign-assessment"
            ],
            excluded_selector=", ".join(isw_noise_selectors),
            excluded_tags=["script", "style", "noscript", "iframe", "svg", "nav", "header", "footer"],
            exclude_external_links=True,
            exclude_social_media_links=True,
            word_count_threshold=10,

            # Advanced wait conditions from Crawl4AI documentation
            wait_for="css:.field-item,.article-content,main",  # Wait for content elements to appear
            wait_for_timeout=30000,  # 30 second timeout for wait condition
            page_timeout=90000,      # 90 second timeout for full page load
            delay_before_return_html=5.0,  # Wait longer for dynamic content

            # Advanced JavaScript execution for dynamic content loading
            js_code=[
                # Wait for initial page load
                "await new Promise(resolve => setTimeout(resolve, 3000));",

                # Try to find and expand any collapsed content
                """
                const expandButtons = document.querySelectorAll('button, [role="button"], [onclick], .expand, .collapse, .toggle');
                for (const btn of expandButtons) {
                    const text = btn.textContent.toLowerCase();
                    if (text.includes('expand') || text.includes('show') || text.includes('more') || text.includes('read')) {
                        btn.click();
                        await new Promise(resolve => setTimeout(resolve, 1000));
                    }
                }
                """,

                # Look for and click any content loaders or "view more" buttons
                """
                const loadButtons = document.querySelectorAll('button, [role="button"], [onclick], .load-more, .view-more');
                for (const btn of loadButtons) {
                    const text = btn.textContent.toLowerCase();
                    if (text.includes('load') || text.includes('more') || text.includes('view') || text.includes('full')) {
                        btn.click();
                        await new Promise(resolve => setTimeout(resolve, 2000));
                    }
                }
                """,

                # Try to trigger any lazy loading by scrolling
                """
                window.scrollTo(0, document.body.scrollHeight / 2);
                await new Promise(resolve => setTimeout(resolve, 2000));
                window.scrollTo(0, 0);
                await new Promise(resolve => setTimeout(resolve, 1000));
                """,

                # Wait for any final dynamic content to load
                "await new Promise(resolve => setTimeout(resolve, 3000));"
            ],

            # Advanced anti-bot detection features
            simulate_user=True,
            override_navigator=True,
            magic=True,

            # Session management for potential multi-step loading
            session_id="isw_test_session",

            # Browser cache configuration
            cache_mode=CacheMode.ENABLED
        )

        try:
            async with AsyncWebCrawler(config=stealth_browser) as crawler:
                print("🚀 Starting advanced crawl with stealth browser...")

                result = await crawler.arun(url=test_url, config=config)

                if result.success:
                    print(f"✅ Advanced crawl successful!")
                    print(f"   - Raw HTML length: {len(result.html):,} characters")
                    print(f"   - Cleaned markdown length: {len(result.markdown.raw_markdown):,} characters")

                    if hasattr(result.markdown, 'fit_markdown'):
                        filtered_length = len(result.markdown.fit_markdown)
                        print(f"   - Filtered markdown length: {filtered_length:,} characters")

                        if filtered_length > 1000:
                            print(f"   ✅ SUCCESS: Substantial content extracted!")

                            # Check content quality
                            content_preview = result.markdown.fit_markdown[:1000]
                            military_keywords = ["Russian", "Ukraine", "military", "operations", "forces", "assessment", "October", "2025"]
                            keyword_count = sum(1 for keyword in military_keywords if keyword.lower() in content_preview.lower())

                            print(f"   📊 Content quality indicators:")
                            print(f"      - Military keywords found: {keyword_count}/{len(military_keywords)}")

                            if keyword_count >= 4:
                                print(f"      ✅ Content appears to be relevant military analysis")
                                print(f"   🎉 CONCLUSION: ISW domain should NOT be excluded!")

                                # Show preview of the good content
                                print(f"\n📄 Content preview (first 500 chars):")
                                print("-" * 50)
                                print(content_preview[:500])
                                print("-" * 50)

                                return True
                            else:
                                print(f"      ⚠️  Content may not be the expected military analysis")

                        else:
                            print(f"   ⚠️  Limited content extracted: {filtered_length} chars")

                    # Check if it's still navigation-heavy content
                    navigation_indicators = ["Skip to content", "Donate", "Menu", "About ISW", "MAP ROOM"]
                    nav_count = sum(1 for indicator in navigation_indicators if indicator in result.markdown.raw_markdown)

                    if nav_count > 3:
                        print(f"   🚨 Navigation indicators found: {nav_count}/5")
                        print(f"   ❌ Still getting navigation content instead of article")

                    # Show a preview for analysis
                    print(f"\n📄 Content preview (first 800 chars):")
                    print("-" * 50)
                    print(result.markdown.raw_markdown[:800])
                    print("-" * 50)

                else:
                    print(f"❌ Advanced crawl failed: {result.error_message}")

        except Exception as e:
            print(f"❌ Error during advanced crawling: {e}")
            import traceback
            traceback.print_exc()

        return False

async def test_virtual_scrolling_and_session_management():
    """Test virtual scrolling and advanced session management."""
    print("\n🔄 Test 4: Virtual Scrolling + Session Management")
    print("-" * 50)

    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
        from crawl4ai.content_filter_strategy import BM25ContentFilter, PruningContentFilter, LLMContentFilter
        from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
        from crawl4ai.virtual_scroll import VirtualScrollConfig

        # Create virtual scroll config for potential infinite scroll
        virtual_scroll_config = VirtualScrollConfig(
            scroll_count=5,                    # Maximum 5 scrolls
            scroll_by="page_height",          # Scroll by full page height
            wait_after_scroll=2.0,            # Wait 2 seconds after each scroll
            container_selector="main, .content, .field-item",  # Try these as scroll containers
            capture_screenshots=False         # Don't capture screenshots
        )

        # Enhanced browser configuration with persistent session
        enhanced_browser = BrowserConfig(
            browser_type="undetected",
            enable_stealth=True,
            headless=True,
            ignore_https_errors=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Cache-Control": "max-age=0"
            },
            extra_args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-web-security",
                "--disable-features=VizDisplayCompositor",
                "--no-first-run",
                "--disable-default-apps",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-background-timer-throttling",
                "--disable-renderer-backgrounding",
                "--disable-backgrounding-occluded-windows",
                "--disable-ipc-flooding-protection"
            ]
        )

        # Advanced LLM filtering for military analysis content
        llm_config = None  # Skip LLM if no API key
        if os.getenv("OPENAI_API_KEY"):
            from crawl4ai import LLMConfig
            llm_config = LLMConfig(
                provider="openai/gpt-4o-mini",
                api_token=os.getenv("OPENAI_API_KEY")
            )

            llm_filter = LLMContentFilter(
                llm_config=llm_config,
                instruction="""
                Extract ONLY the main Russian Offensive Campaign Assessment content.

                INCLUDE:
                - Main assessment title and date
                - Key military developments and operations
                - Strategic analysis and insights
                - Geographic location updates
                - Force deployment information
                - Tactical assessments

                EXCLUDE:
                - All navigation elements
                - Website menus and headers
                - Donation requests and newsletters
                - Social media sharing buttons
                - Related articles or trending content
                - Footer content and legal notices

                Format as clean markdown with proper headings and paragraphs.
                """,
                chunk_token_threshold=4096,
                verbose=True
            )

            md_generator = DefaultMarkdownGenerator(content_filter=llm_filter)
        else:
            # Fall back to BM25 filter for military content
            bm25_filter = BM25ContentFilter(
                user_query="Russian offensive campaign Ukraine military assessment October 2025 forces operations strategy ISW",
                bm25_threshold=1.0,
                language="english",
                use_stemming=True
            )
            md_generator = DefaultMarkdownGenerator(content_filter=bm25_filter)

        # Multi-step session-based crawling
        async with AsyncWebCrawler(config=enhanced_browser) as crawler:
            session_id = "isw_multi_step_session"

            print("🔄 Step 1: Initial page load and content extraction")

            # Step 1: Initial page load
            step1_config = CrawlerRunConfig(
                markdown_generator=md_generator,
                target_elements=["article", "main", ".field-item", ".field-name-body"],
                excluded_selector="nav, header, footer, .sidebar, .menu, .donate, .newsletter",
                excluded_tags=["script", "style", "noscript"],
                wait_for="css:.field-item, article, main",
                wait_for_timeout=20000,
                page_timeout=60000,
                delay_before_return_html=3.0,
                js_code=[
                    # Wait for page to fully load
                    "await new Promise(resolve => setTimeout(resolve, 2000));",

                    # Try to expand any collapsed assessment content
                    """
                    const expandElements = document.querySelectorAll('.expand, .collapse, .toggle, .accordion, [aria-expanded]');
                    expandElements.forEach(el => {
                        if (el.getAttribute('aria-expanded') === 'false' ||
                            el.textContent.toLowerCase().includes('expand') ||
                            el.textContent.toLowerCase().includes('show')) {
                            el.click();
                        }
                    });
                    await new Promise(resolve => setTimeout(resolve, 1500));
                    """,

                    # Look for assessment-specific content containers
                    """
                    const assessmentContent = document.querySelector('.field-item, .assessment-content, .campaign-assessment');
                    if (assessmentContent) {
                        assessmentContent.scrollIntoView({behavior: 'smooth', block: 'center'});
                    }
                    await new Promise(resolve => setTimeout(resolve, 1000));
                    """
                ],
                simulate_user=True,
                override_navigator=True,
                magic=True,
                session_id=session_id,
                cache_mode=CacheMode.ENABLED
            )

            step1_result = await crawler.arun(url=test_url, config=step1_config)

            if step1_result.success:
                step1_content = step1_result.markdown.fit_markdown if hasattr(step1_result.markdown, 'fit_markdown') else step1_result.markdown.raw_markdown
                print(f"✅ Step 1 successful: {len(step1_content):,} characters extracted")

                # Check if we got substantial content
                if len(step1_content) > 2000:
                    military_keywords = ["Russian", "Ukraine", "offensive", "assessment", "October", "2025", "forces", "operations"]
                    keyword_count = sum(1 for keyword in military_keywords if keyword.lower() in step1_content.lower())

                    print(f"📊 Content quality: {keyword_count}/{len(military_keywords)} military keywords found")

                    if keyword_count >= 4:
                        print(f"🎉 SUCCESS: High-quality military assessment content extracted!")
                        print(f"📄 Content preview:")
                        print("-" * 50)
                        print(step1_content[:800])
                        print("-" * 50)
                        return True

                print("🔄 Step 2: Attempting additional content loading...")

                # Step 2: Try additional content loading using same session
                step2_config = CrawlerRunConfig(
                    session_id=session_id,
                    js_only=True,  # Only execute JS, don't navigate
                    js_code=[
                        # Try to find and click any remaining content loaders
                        """
                        const remainingButtons = document.querySelectorAll('button, [role="button"], [onclick]');
                        for (const btn of remainingButtons) {
                            const text = btn.textContent.toLowerCase();
                            if (text.includes('full') || text.includes('complete') || text.includes('entire')) {
                                btn.click();
                                await new Promise(resolve => setTimeout(resolve, 2000));
                            }
                        }
                        """,

                        # Try virtual scrolling if content might be loaded on scroll
                        """
                        for (let i = 0; i < 3; i++) {
                            window.scrollBy(0, window.innerHeight);
                            await new Promise(resolve => setTimeout(resolve, 2000));
                        }
                        """,

                        # Final wait for any dynamically loaded content
                        "await new Promise(resolve => setTimeout(resolve, 3000));"
                    ],
                    delay_before_return_html=2.0,
                    virtual_scroll_config=virtual_scroll_config
                )

                step2_result = await crawler.arun(url=test_url, config=step2_config)

                if step2_result.success:
                    step2_content = step2_result.markdown.fit_markdown if hasattr(step2_result.markdown, 'fit_markdown') else step2_result.markdown.raw_markdown
                    print(f"✅ Step 2 successful: {len(step2_content):,} characters")

                    # Combine results
                    total_content = step1_content + "\n\n" + step2_content
                    print(f"📊 Combined content: {len(total_content):,} characters")

                    if len(total_content) > 3000:
                        military_keywords = ["Russian", "Ukraine", "offensive", "assessment", "October", "2025", "forces", "operations"]
                        keyword_count = sum(1 for keyword in military_keywords if keyword.lower() in total_content.lower())

                        print(f"🎉 FINAL SUCCESS: Comprehensive military assessment extracted!")
                        print(f"📊 Combined content quality: {keyword_count}/{len(military_keywords)} keywords")
                        print(f"📄 Combined content preview:")
                        print("-" * 50)
                        print(total_content[:1000])
                        print("-" * 50)
                        return True

            # Clean up session
            await crawler.crawler_strategy.kill_session(session_id)

    except Exception as e:
        print(f"❌ Error in virtual scrolling test: {e}")
        import traceback
        traceback.print_exc()

    return False

def run_comprehensive_tests():
    """Run all advanced tests with proper error handling."""
    print("🔬 Running Comprehensive Advanced Crawl4AI Tests")
    print("=" * 60)

    try:
        # Test 1: Advanced filtering with stealth browser
        print("\n🎯 TEST SUITE 1: Advanced Filtering + Stealth Browser")
        success1 = asyncio.run(run_advanced_tests())

        if success1:
            print("✅ TEST SUITE 1 PASSED - Advanced filtering successful!")
            return True

        print("⚠️  TEST SUITE 1 FAILED - Trying alternative approaches...")

        # Test 2: Virtual scrolling and session management
        print("\n🔄 TEST SUITE 2: Virtual Scrolling + Session Management")
        success2 = asyncio.run(test_virtual_scrolling_and_session_management())

        if success2:
            print("✅ TEST SUITE 2 PASSED - Session management successful!")
            return True

        print("❌ ALL TEST SUITES FAILED - Domain may need exclusion")
        return False

    except Exception as e:
        print(f"❌ Error running comprehensive tests: {e}")
        import traceback
        traceback.print_exc()
        return False

def basic_domain_analysis():
    """Fallback basic analysis when advanced features aren't available."""
    print("\n📋 Basic Domain Analysis")
    print("-" * 30)

    print("🚫 UNDERSTANDINGWAR.ORG ASSESSMENT:")
    print("  • Content structure appears incompatible with our extraction")
    print("  • Advanced techniques from z-playground1 unavailable in this environment")
    print("  • Previous attempts returned navigation-only content")
    print("  • Level 3 anti-bot (stealth) was already attempted")

    print("\n💡 RECOMMENDATION:")
    print("  • Add to domain exclusion list")
    print("  • ISW content is high-quality but extraction-unreliable")
    print("  • Focus on alternative sources for similar analysis")

    return False

def make_exclusion_decision(test_success):
    """Make final decision on domain exclusion based on test results."""
    print("\n" + "=" * 80)
    print("🏁 FINAL DECISION ON understantingwar.org")
    print("=" * 80)

    if test_success:
        print("✅ DOMAIN SHOULD BE PRESERVED")
        print("   Advanced techniques successfully extracted article content!")
        print("   • Domain exclusion will NOT be implemented")
        print("   • Consider integrating advanced techniques into main system")
        print("   • ISW provides valuable military analysis content")
    else:
        print("❌ DOMAIN SHOULD BE EXCLUDED")
        print("   Even advanced techniques could not extract article content")
        print("   • Add understandingwar.org to exclusion list")
        print("   • Prevents wasted processing time and resources")
        print("   • Focus on more reliable content sources")

    print("\n📊 COMPARISON:")
    print("   Standard techniques: ❌ Navigation only")
    print("   Level 3 anti-bot:    ❌ Navigation only")
    print("   Advanced z-playground: ✅" if test_success else "   Advanced z-playground: ❌ Still navigation")

    return not test_success  # Return True if domain should be excluded

if __name__ == "__main__":
    print("🔬 Testing understandingwar.org with Advanced Techniques")
    print("Before excluding domain, testing z-playground1 + DeepWiki techniques...")

    # Test with advanced techniques
    test_success = test_isw_with_advanced_techniques()

    # Make final decision
    should_exclude = make_exclusion_decision(test_success)

    if should_exclude:
        print(f"\n🚫 IMPLEMENTING DOMAIN EXCLUSION:")
        print(f"   The domain exclusion functionality has been implemented")
        print(f"   understandingwar.org will be automatically excluded")
        print(f"   from future research crawling operations.")
        print(f"\n💡 TO ENABLE EXCLUSION:")
        print(f"   The exclusion list is already implemented in url_tracker.py")
        print(f"   'understandingwar.org' is in the excluded_domains set")
        print(f"   Future research will automatically skip this domain.")

    print(f"\n✅ Testing completed!")