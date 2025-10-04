#!/usr/bin/env python3
"""Test script to validate the optimized crawler fixes.

This script tests the optimized Crawl4AI implementation that resolves
the Stage 1 DNS resolution issues identified in the performance analysis.
"""

import asyncio
import sys
import os
from pathlib import Path
import time

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

async def test_optimized_crawler():
    """Test the optimized crawler with problematic URLs."""
    print("\n" + "=" * 80)
    print("TESTING OPTIMIZED CRAWLER - STAGE 1 DNS FIXES")
    print("=" * 80)

    try:
        from utils.crawl4ai_optimized import get_optimized_crawler
        print("✅ Optimized crawler imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import optimized crawler: {e}")
        return False

    # Test URLs that would typically cause DNS issues
    test_urls = [
        "https://example.com",  # Simple test
        "https://httpbin.org/get",  # API endpoint
        "https://github.com/anthropics",  # GitHub (known to have anti-bot)
        "https://stackoverflow.com/questions",  # StackOverflow
    ]

    crawler = get_optimized_crawler()

    print(f"\n🔍 Testing {len(test_urls)} URLs with optimized configuration")
    print(f"📊 Expected: Stage 1 should now succeed for most URLs")

    results = []

    for i, url in enumerate(test_urls, 1):
        print(f"\n--- Test {i}: {url} ---")
        start_time = time.time()

        try:
            result = await crawler.crawl_with_intelligent_fallback(url)
            duration = time.time() - start_time

            print(f"✅ Success: {result.success}")
            print(f"📊 Stage used: {result.stage_used}")
            print(f"📊 Cache mode: {result.cache_mode}")
            print(f"📊 Anti-bot level: {result.anti_bot_level}")
            print(f"📊 DNS resolved: {result.dns_resolved}")
            print(f"📊 Content length: {result.char_count} chars")
            print(f"⏱️  Duration: {duration:.2f}s")

            if result.error:
                print(f"❌ Error: {result.error}")

            results.append({
                'url': url,
                'success': result.success,
                'stage_used': result.stage_used,
                'duration': duration,
                'char_count': result.char_count
            })

        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ Exception: {e}")
            results.append({
                'url': url,
                'success': False,
                'stage_used': 'exception',
                'duration': duration,
                'char_count': 0,
                'error': str(e)
            })

    # Performance analysis
    print(f"\n{'='*80}")
    print("PERFORMANCE ANALYSIS")
    print(f"{'='*80}")

    successful = [r for r in results if r['success']]
    stage1_successful = [r for r in successful if r['stage_used'] == '1_optimized']
    stage2_successful = [r for r in successful if r['stage_used'] == '2_intelligent']

    print(f"📊 Overall Results:")
    print(f"  Total URLs: {len(results)}")
    print(f"  Successful: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"  Stage 1 successes: {len(stage1_successful)}")
    print(f"  Stage 2 successes: {len(stage2_successful)}")

    if successful:
        avg_duration = sum(r['duration'] for r in successful) / len(successful)
        avg_content = sum(r['char_count'] for r in successful) / len(successful)
        print(f"  Average duration: {avg_duration:.2f}s")
        print(f"  Average content: {avg_content:.0f} chars")

    # Stage 1 success rate analysis
    stage1_success_rate = len(stage1_successful) / len(results) * 100
    print(f"\n🎯 Stage 1 Performance:")
    print(f"  Success rate: {stage1_success_rate:.1f}%")

    if stage1_success_rate > 50:
        print("  ✅ GOOD: Stage 1 is working properly")
    elif stage1_success_rate > 25:
        print("  ⚠️  MARGINAL: Stage 1 needs more optimization")
    else:
        print("  ❌ POOR: Stage 1 still has issues")

    # Get crawler statistics
    stats = crawler.get_performance_stats()
    print(f"\n📈 Crawler Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    # Overall assessment
    print(f"\n🏆 Overall Assessment:")
    if stage1_success_rate >= 70:
        print("  ✅ EXCELLENT: Optimization successful")
        print("  🚀 Stage 1 DNS issues resolved")
        return True
    elif stage1_success_rate >= 50:
        print("  ✅ GOOD: Significant improvement achieved")
        print("  🔧 Minor optimizations may help")
        return True
    elif stage1_success_rate >= 25:
        print("  ⚠️  MARGINAL: Some improvement but needs work")
        return False
    else:
        print("  ❌ NEEDS WORK: Optimization ineffective")
        return False


async def test_parallel_crawling():
    """Test parallel crawling with optimized crawler."""
    print(f"\n{'='*80}")
    print("TESTING PARALLEL CRAWLING")
    print(f"{'='*80}")

    try:
        from utils.crawl4ai_optimized import get_optimized_crawler
        crawler = get_optimized_crawler()

        test_urls = [
            "https://example.com",
            "https://httpbin.org/get",
            "https://jsonplaceholder.typicode.com/posts/1",
        ]

        print(f"🚀 Testing parallel crawling with {len(test_urls)} URLs")
        start_time = time.time()

        results = await crawler.crawl_multiple_optimized(test_urls, max_concurrent=3)

        duration = time.time() - start_time
        successful = [r for r in results if r.success]

        print(f"📊 Parallel Results:")
        print(f"  Total URLs: {len(results)}")
        print(f"  Successful: {len(successful)}")
        print(f"  Total duration: {duration:.2f}s")
        print(f"  Avg per URL: {duration/len(results):.2f}s")

        if len(successful) == len(test_urls):
            print("✅ Parallel crawling successful")
            return True
        else:
            print("⚠️  Some parallel crawls failed")
            return False

    except Exception as e:
        print(f"❌ Parallel crawling test failed: {e}")
        return False


async def main():
    """Run all optimized crawler tests."""
    print("🧪 Starting Optimized Crawler Tests")
    print("Testing fixes for Stage 1 DNS resolution issues")

    test1_result = await test_optimized_crawler()
    test2_result = await test_parallel_crawling()

    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")

    print(f"📊 Test Results:")
    print(f"  Optimized crawler test: {'✅ PASSED' if test1_result else '❌ FAILED'}")
    print(f"  Parallel crawling test: {'✅ PASSED' if test2_result else '❌ FAILED'}")

    overall_success = test1_result and test2_result
    print(f"\n🏆 Overall: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")

    if overall_success:
        print("\n🎉 Stage 1 DNS optimization successful!")
        print("📈 System performance improved significantly")
        print("🔧 Ready for production deployment")
    else:
        print("\n⚠️  Optimization needs further work")
        print("🔧 Review configuration and test again")

    return overall_success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)