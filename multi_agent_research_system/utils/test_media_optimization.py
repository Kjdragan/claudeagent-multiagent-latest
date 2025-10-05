#!/usr/bin/env python3
"""
Test script for Crawl4AI media optimization implementation.

This script validates that the media optimization parameters work correctly
and provides performance comparisons with the original implementation.
"""

import asyncio
import logging
import time
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Test result structure for comparison."""
    url: str
    implementation: str
    success: bool
    duration: float
    content_length: int
    bandwidth_saved_mb: float = 0.0
    error: str = None

async def test_media_optimization():
    """Test media optimization against original implementation."""

    # Test URLs (mix of different content types)
    test_urls = [
        "https://example.com",
        "https://httpbin.org/html",  # Simple HTML page
        "https://github.com/unclecode/crawl4ai",  # Documentation site
    ]

    results: list[TestResult] = []

    logger.info("🚀 Starting Crawl4AI media optimization tests")
    logger.info(f"📋 Testing {len(test_urls)} URLs with both implementations")

    try:
        # Import both implementations
        from utils.crawl4ai_media_optimized import (
            crawl_multiple_urls_media_optimized as optimized_crawl,
        )
        from utils.crawl4ai_utils import (
            crawl_multiple_urls_with_results as original_crawl,
        )

        session_id = f"media-test-{int(time.time())}"

        # Test original implementation
        logger.info("📊 Testing original implementation...")
        start_time = time.time()
        original_results = await original_crawl(
            urls=test_urls,
            session_id=f"{session_id}-original",
            max_concurrent=2,
            extraction_mode="article"
        )
        original_duration = time.time() - start_time

        for result in original_results:
            results.append(TestResult(
                url=result['url'],
                implementation="Original",
                success=result['success'],
                duration=result['duration'],
                content_length=len(result.get('content', '')),
                error=result.get('error_message')
            ))

        # Test media optimized implementation
        logger.info("⚡ Testing media optimized implementation...")
        start_time = time.time()
        optimized_results = await optimized_crawl(
            urls=test_urls,
            session_id=f"{session_id}-optimized",
            max_concurrent=2,
            extraction_mode="article"
        )
        optimized_duration = time.time() - start_time

        for result in optimized_results:
            results.append(TestResult(
                url=result['url'],
                implementation="Media Optimized",
                success=result['success'],
                duration=result['duration'],
                content_length=len(result.get('content', '')),
                bandwidth_saved_mb=result.get('bandwidth_saved_mb', 0.0),
                error=result.get('error_message')
            ))

        # Analyze results
        analyze_test_results(results, original_duration, optimized_duration)

    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("Make sure both crawl4ai_utils.py and crawl4ai_media_optimized.py are available")

    except Exception as e:
        logger.error(f"❌ Test execution error: {e}")

def analyze_test_results(results: list[TestResult], original_duration: float, optimized_duration: float):
    """Analyze and compare test results."""

    logger.info("\n" + "="*60)
    logger.info("📊 MEDIA OPTIMIZATION TEST RESULTS")
    logger.info("="*60)

    # Group results by implementation
    original_results = [r for r in results if r.implementation == "Original"]
    optimized_results = [r for r in results if r.implementation == "Media Optimized"]

    # Success rate comparison
    original_success_rate = sum(1 for r in original_results if r.success) / len(original_results) if original_results else 0
    optimized_success_rate = sum(1 for r in optimized_results if r.success) / len(optimized_results) if optimized_results else 0

    # Performance metrics
    original_avg_duration = sum(r.duration for r in original_results) / len(original_results) if original_results else 0
    optimized_avg_duration = sum(r.duration for r in optimized_results) / len(optimized_results) if optimized_results else 0

    original_avg_content = sum(r.content_length for r in original_results) / len(original_results) if original_results else 0
    optimized_avg_content = sum(r.content_length for r in optimized_results) / len(optimized_results) if optimized_results else 0

    total_bandwidth_saved = sum(r.bandwidth_saved_mb for r in optimized_results)

    # Print comparison
    logger.info("\n🎯 OVERALL PERFORMANCE:")
    logger.info("Implementation          | Success Rate | Avg Duration | Avg Content | Total Time")
    logger.info("------------------------|--------------|--------------|-------------|------------")
    logger.info(f"Original                | {original_success_rate:.1%}        | {original_avg_duration:.2f}s        | {original_avg_content:.0f} chars     | {original_duration:.2f}s")
    logger.info(f"Media Optimized         | {optimized_success_rate:.1%}        | {optimized_avg_duration:.2f}s        | {optimized_avg_content:.0f} chars     | {optimized_duration:.2f}s")

    # Calculate improvements
    if original_avg_duration > 0:
        speed_improvement = ((original_avg_duration - optimized_avg_duration) / original_avg_duration) * 100
        logger.info("\n⚡ PERFORMANCE IMPROVEMENTS:")
        logger.info(f"Speed improvement: {speed_improvement:+.1f}%")
        logger.info(f"Total bandwidth saved: {total_bandwidth_saved:.1f} MB")
        logger.info(f"Success rate change: {optimized_success_rate - original_success_rate:+.1%}")

    # Detailed URL results
    logger.info("\n📋 DETAILED URL RESULTS:")
    logger.info("URL                     | Implementation | Success | Duration | Content  | Bandwidth Saved")
    logger.info("------------------------|----------------|---------|----------|----------|----------------")

    for result in results:
        status = "✅" if result.success else "❌"
        bandwidth_info = f"{result.bandwidth_saved_mb:.1f}MB" if result.bandwidth_saved_mb > 0 else "N/A"
        logger.info(f"{result.url[:25]:<25} | {result.implementation:<15} | {status} | {result.duration:7.2f}s | {result.content_length:7d} | {bandwidth_info}")

    # Recommendations
    logger.info("\n💡 RECOMMENDATIONS:")

    if optimized_success_rate >= original_success_rate and optimized_avg_duration < original_avg_duration:
        logger.info("✅ Media optimization is READY FOR PRODUCTION")
        logger.info("   - Success rate maintained or improved")
        logger.info("   - Significant performance gains achieved")
        logger.info("   - Bandwidth savings realized")
    elif optimized_success_rate < original_success_rate:
        logger.info("⚠️  Media optimization needs ADJUSTMENT")
        logger.info("   - Success rate decreased - investigate failures")
        logger.info("   - Consider selective optimization by domain")
    else:
        logger.info("🤔 Media optimization performance is MIXED")
        logger.info("   - Review specific URL results")
        logger.info("   - Consider hybrid approach")

    logger.info("\n🚀 NEXT STEPS:")
    logger.info("1. If results are positive, deploy media optimized crawler")
    logger.info("2. Monitor performance in production environment")
    logger.info("3. Adjust parameters based on real-world usage")
    logger.info("4. Consider domain-specific optimization strategies")

if __name__ == "__main__":
    asyncio.run(test_media_optimization())
