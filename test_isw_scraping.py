#!/usr/bin/env python3
"""
Test different scraping approaches for the problematic understandingwar.org domain.
"""

import asyncio
import sys
import os
import logging

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_crawl4ai_approaches():
    """Test different Crawl4ai approaches for understandingwar.org"""

    # Test URL from the logs
    test_url = "https://understandingwar.org/research/russia-ukraine/russian-offensive-campaign-assessment-october-8-2025/"

    print("🔍 Testing Different Scraping Approaches for understandingwar.org")
    print("=" * 80)
    print(f"URL: {test_url}")
    print("=" * 80)

    try:
        # Import crawl4ai directly
        from crawl4ai import AsyncWebCrawler
        from crawl4ai.extraction_strategy import LLMExtractionStrategy, CssExtractionStrategy
        from crawl4ai.chunking_strategy import RegexChunking

        print("\n📋 Approach 1: Basic Crawl4ai")
        print("-" * 40)

        async with AsyncWebCrawler(verbose=True) as crawler:
            # Basic approach
            result1 = await crawler.arun(
                url=test_url,
                word_count_threshold=10,
                extraction_strategy=None,
                chunking_strategy=None,
                bypass_cache=True
            )

            print(f"✅ Basic approach - Content length: {len(result1.cleaned_html or '')}")
            print(f"📝 Preview (first 500 chars):")
            print((result1.cleaned_html or '')[:500])
            print("...")

        print("\n📋 Approach 2: CSS Extraction Strategy")
        print("-" * 40)

        async with AsyncWebCrawler(verbose=True) as crawler:
            # Try specific CSS selectors for article content
            css_strategy = CssExtractionStrategy(
                css_selectors=[
                    "article",
                    ".field-item",
                    ".content",
                    ".main-content",
                    ".article-content",
                    ".node-content",
                    ".field-name-body",
                    "[class*='content']",
                    "[class*='article']",
                    "[class*='body']"
                ]
            )

            result2 = await crawler.arun(
                url=test_url,
                extraction_strategy=css_strategy,
                bypass_cache=True
            )

            print(f"✅ CSS extraction - Content length: {len(result2.extracted_content or '')}")
            print(f"📝 Preview (first 500 chars):")
            print((result2.extracted_content or '')[:500])
            print("...")

        print("\n📋 Approach 3: LLM Extraction Strategy")
        print("-" * 40)

        async with AsyncWebCrawler(verbose=True) as crawler:
            # Use LLM to extract just the article content
            llm_strategy = LLMExtractionStrategy(
                provider="openai",
                api_token=os.getenv("OPENAI_API_KEY", "dummy"),
                instruction="Extract ONLY the main article content. Remove all navigation, menus, headers, footers, and sidebars. Focus on the actual assessment content about the Russian offensive campaign."
            )

            result3 = await crawler.arun(
                url=test_url,
                extraction_strategy=llm_strategy,
                bypass_cache=True
            )

            print(f"✅ LLM extraction - Content length: {len(result3.extracted_content or '')}")
            print(f"📝 Preview (first 500 chars):")
            print((result3.extracted_content or '')[:500])
            print("...")

        print("\n📋 Approach 4: JavaScript with Wait")
        print("-" * 40)

        async with AsyncWebCrawler(verbose=True) as crawler:
            # Try waiting for JavaScript to load
            result4 = await crawler.arun(
                url=test_url,
                js_code="""
                // Wait for content to load
                await new Promise(resolve => setTimeout(resolve, 3000));

                // Try to find and click any content loaders
                const buttons = document.querySelectorAll('button, [onclick], [role="button"]');
                for (const btn of buttons) {
                    if (btn.textContent.includes('more') || btn.textContent.includes('show') || btn.textContent.includes('load')) {
                        btn.click();
                        break;
                    }
                }

                // Wait a bit more
                await new Promise(resolve => setTimeout(resolve, 2000));
                """,
                wait_for="body",
                bypass_cache=True,
                word_count_threshold=10
            )

            print(f"✅ JavaScript approach - Content length: {len(result4.cleaned_html or '')}")
            print(f"📝 Preview (first 500 chars):")
            print((result4.cleaned_html or '')[:500])
            print("...")

        print("\n📋 Approach 5: User Agent Override")
        print("-" * 40)

        async with AsyncWebCrawler(verbose=True) as crawler:
            # Try with a different user agent
            result5 = await crawler.arun(
                url=test_url,
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                headers={
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Accept-Encoding": "gzip, deflate, br",
                    "DNT": "1",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1"
                },
                bypass_cache=True
            )

            print(f"✅ User agent approach - Content length: {len(result5.cleaned_html or '')}")
            print(f"📝 Preview (first 500 chars):")
            print((result5.cleaned_html or '')[:500])
            print("...")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

async def test_our_utils():
    """Test our existing utility functions"""
    print("\n\n🔧 Testing Our Existing Utils")
    print("=" * 80)

    test_url = "https://understandingwar.org/research/russia-ukraine/russian-offensive-campaign-assessment-october-8-2025/"

    try:
        # Test our current utils
        from multi_agent_research_system.utils.crawl4ai_utils import (
            scrape_and_clean_single_url_direct,
            crawl_multiple_urls_with_cleaning
        )

        print("\n📋 Our Current Direct Scrape Approach")
        print("-" * 40)

        result = await scrape_and_clean_single_url_direct(
            url=test_url,
            search_query="Russian Offensive Campaign Assessment October 8 2025",
            skip_llm_cleaning=True  # Get raw content first
        )

        if hasattr(result, 'content'):
            raw_content = result.content
        else:
            raw_content = result.get('content', '')

        print(f"✅ Our utils approach - Content length: {len(raw_content)}")
        print(f"📝 Preview (first 1000 chars):")
        print(raw_content[:1000])
        print("...")

        # Check if it's the same navigation-heavy content
        navigation_indicators = [
            "Skip to content", "Donate", "Menu", "About ISW",
            "MAP ROOM", "Analysis", "Education", "Get Involved",
            "Research Library", "Careers", "Newsroom"
        ]

        navigation_count = sum(1 for indicator in navigation_indicators if indicator in raw_content)
        print(f"\n🚨 Navigation elements found: {navigation_count}/{len(navigation_indicators)}")

        if navigation_count > len(navigation_indicators) * 0.7:
            print("⚠️  This appears to be mostly navigation content, not article content!")

        # Look for actual article content indicators
        article_indicators = [
            "Russian Offensive Campaign Assessment",
            "October 8, 2025",
            "Ukrainian", "Russian", "forces", "operations",
            "Donetsk", "Kharkiv", "Kremlin", "military"
        ]

        article_content_count = sum(1 for indicator in article_indicators if indicator.lower() in raw_content.lower())
        print(f"📊 Article content indicators: {article_content_count}/{len(article_indicators)}")

    except Exception as e:
        print(f"❌ Error testing our utils: {e}")
        import traceback
        traceback.print_exc()

def analyze_domain_feasibility():
    """Analyze whether this domain is feasible for scraping"""
    print("\n\n📊 Domain Feasibility Analysis")
    print("=" * 80)

    print("🚨 understandingwar.org SCRAPING ISSUES:")
    print("1. Heavy JavaScript-driven content loading")
    print("2. Content appears behind complex navigation structure")
    print("3. Article content may be loaded dynamically after page load")
    print("4. Current extraction is capturing navigation, not actual articles")
    print("5. Domain structure may be using modern web frameworks that block scrapers")

    print("\n💡 RECOMMENDATIONS:")
    print("1. Add understandingwar.org to domain exclusion list")
    print("2. OR invest significant effort in domain-specific extraction logic")
    print("3. OR try to find RSS feeds or alternative content sources")
    print("4. OR use their API if available")

    print("\n⚖️  COST-BENEFIT ANALYSIS:")
    print("• PRO: ISW provides high-quality military analysis")
    print("• CON: Very difficult to extract actual content")
    print("• CON: High failure rate wastes processing time")
    print("• CON: Navigation-heavy results pollute research data")

    print("\n✅ RECOMMENDED ACTION:")
    print("Exclude understandingwar.org from scraping until a domain-specific")
    print("extraction solution can be implemented.")

if __name__ == "__main__":
    asyncio.run(test_crawl4ai_approaches())
    asyncio.run(test_our_utils())
    analyze_domain_feasibility()