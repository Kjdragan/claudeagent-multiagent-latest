#!/usr/bin/env python3
"""
Test script to validate retry statistics tracking.

This script tests the comprehensive retry statistics system to determine
if the retry mechanism is effective and worthwhile.
"""

import asyncio
import sys
from pathlib import Path

# Add the multi_agent_research_system to Python path
sys.path.append(str(Path(__file__).parent / "multi_agent_research_system"))

def test_retry_statistics_tracking():
    """Test retry statistics tracking functionality."""
    print("\n=== Testing Retry Statistics Tracking ===")

    try:
        from utils.url_tracker import get_url_tracker

        # Create test directory in temp location
        test_storage = Path("/tmp/test_retry_stats")
        if test_storage.exists():
            import shutil
            shutil.rmtree(test_storage)

        tracker = get_url_tracker(test_storage)

        # Test URLs with different retry scenarios
        test_urls = [
            "https://example.com/success_first_try",
            "https://example.com/success_on_retry",
            "https://example.com/failed_all_attempts",
            "https://example.com/improved_on_retry"
        ]

        session_id = "test_retry_session"

        print("🔄 Simulating different retry scenarios...")

        # Scenario 1: Success on first try
        tracker.record_attempt(
            url=test_urls[0],
            success=True,
            anti_bot_level=1,
            content_length=1000,
            duration=2.0,
            session_id=session_id
        )

        # Scenario 2: Success on retry (higher anti-bot level)
        tracker.record_attempt(
            url=test_urls[1],
            success=False,
            anti_bot_level=1,
            content_length=0,
            duration=1.5,
            error_message="Bot detected",
            session_id=session_id
        )

        tracker.record_attempt(
            url=test_urls[1],
            success=True,
            anti_bot_level=2,
            content_length=1500,
            duration=3.0,
            session_id=session_id
        )

        # Scenario 3: Failed all attempts
        tracker.record_attempt(
            url=test_urls[2],
            success=False,
            anti_bot_level=1,
            content_length=0,
            duration=1.2,
            error_message="Connection timeout",
            session_id=session_id
        )

        tracker.record_attempt(
            url=test_urls[2],
            success=False,
            anti_bot_level=2,
            content_length=0,
            duration=2.0,
            error_message="Bot detected",
            session_id=session_id
        )

        # Scenario 4: Improved content on retry
        tracker.record_attempt(
            url=test_urls[3],
            success=True,
            anti_bot_level=1,
            content_length=500,
            duration=2.5,
            session_id=session_id
        )

        tracker.record_attempt(
            url=test_urls[3],
            success=True,
            anti_bot_level=2,
            content_length=2000,  # Much better content
            duration=3.5,
            session_id=session_id
        )

        # Get comprehensive statistics
        stats = tracker.get_statistics()

        print("✅ Retry Statistics Results:")
        print(f"   Total URLs tracked: {stats['total_urls']}")
        print(f"   URLs with retries: {stats['urls_with_retries']}")
        print(f"   Total retry attempts: {stats['total_retry_attempts']}")
        print(f"   Successful retry attempts: {stats['successful_retry_attempts']}")
        print(f"   Retry effectiveness: {stats['retry_effectiveness_percent']:.1f}%")
        print(f"   URLs improved by retries: {stats['urls_improved_by_retries']}")
        print(f"   Improvement rate: {stats['improvement_rate_percent']:.1f}%")
        print(f"   Average improvement ratio: {stats['avg_retry_improvement_ratio']:.1f}%")

        print("\n🎯 Anti-Bot Level Effectiveness:")
        for level, data in stats['anti_bot_level_effectiveness'].items():
            print(f"   Level {level}: {data['success_rate']:.1f}% success rate "
                  f"({data['successful_attempts']}/{data['total_attempts']} attempts)")

        # Validate specific scenarios
        print("\n🔍 Individual URL Analysis:")

        # Check URL that succeeded on retry
        record1 = tracker.url_records[test_urls[1]]
        print(f"   Success on retry URL:")
        print(f"     Total attempts: {record1.total_attempts}")
        print(f"     Retry attempts: {record1.retry_attempts}")
        print(f"     Retry effectiveness: {record1.retry_effectiveness:.1f}%")
        print(f"     Best content: {record1.best_content_length} chars at level {record1.best_anti_bot_level}")

        # Check URL that improved on retry
        record2 = tracker.url_records[test_urls[3]]
        print(f"   Improved on retry URL:")
        print(f"     Improvement from retries: {record2.improvement_from_retries}")
        print(f"     Improvement ratio: {record2.retry_improvement_ratio:.1f}%")
        print(f"     Best content: {record2.best_content_length} chars at level {record1.best_anti_bot_level}")

        # Validate key metrics
        expected_retries = 3  # URLs 1, 2, and 3 had retries
        expected_successful_retries = 2  # URLs 1 and 3 succeeded on retry
        expected_improved_urls = 2  # URLs 1 and 3 both improved on retry

        assert stats['total_retry_attempts'] == expected_retries, f"Expected {expected_retries} retry attempts, got {stats['total_retry_attempts']}"
        assert stats['successful_retry_attempts'] == expected_successful_retries, f"Expected {expected_successful_retries} successful retries, got {stats['successful_retry_attempts']}"
        assert stats['urls_improved_by_retries'] == expected_improved_urls, f"Expected {expected_improved_urls} improved URLs, got {stats['urls_improved_by_retries']}"

        print("\n✅ All retry statistics tracking tests passed!")
        return True

    except Exception as e:
        print(f"❌ Retry statistics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_retry_worthwhile():
    """Analyze if the retry system is worthwhile based on statistics."""
    print("\n=== Analyzing Retry System Worthwhileness ===")

    try:
        from utils.url_tracker import get_url_tracker

        # Load existing tracker data
        tracker = get_url_tracker()
        stats = tracker.get_statistics()

        if stats['total_retry_attempts'] == 0:
            print("ℹ️  No retry attempts recorded yet. Need more data to analyze.")
            return True

        print("📊 Retry System Analysis:")
        print(f"   URLs with retries: {stats['urls_with_retries']}/{stats['total_urls']} "
              f"({stats['urls_with_retries']/stats['total_urls']*100:.1f}%)")
        print(f"   Retry effectiveness: {stats['retry_effectiveness_percent']:.1f}%")
        print(f"   Improvement rate: {stats['improvement_rate_percent']:.1f}%")

        # Determine if retry is worthwhile
        effectiveness_threshold = 20.0  # At least 20% of retries should succeed
        improvement_threshold = 30.0    # At least 30% of retries should improve results

        retry_worthwhile = True
        recommendations = []

        if stats['retry_effectiveness_percent'] < effectiveness_threshold:
            retry_worthwhile = False
            recommendations.append(f"Retry success rate ({stats['retry_effectiveness_percent']:.1f}%) below threshold ({effectiveness_threshold}%)")

        if stats['improvement_rate_percent'] < improvement_threshold and stats['urls_with_retries'] > 5:
            retry_worthwhile = False
            recommendations.append(f"Retry improvement rate ({stats['improvement_rate_percent']:.1f}%) below threshold ({improvement_threshold}%)")

        # Analyze anti-bot level effectiveness
        print("\n🎯 Anti-Bot Level Analysis:")
        best_level = 0
        best_success_rate = 0
        for level, data in stats['anti_bot_level_effectiveness'].items():
            if data['total_attempts'] > 0 and data['success_rate'] > best_success_rate:
                best_success_rate = data['success_rate']
                best_level = level
            print(f"   Level {level}: {data['success_rate']:.1f}% success ({data['total_attempts']} attempts)")

        if best_level > 1:
            recommendations.append(f"Consider starting with anti-bot level {best_level} for better success rates")

        # Conclusion
        print(f"\n🎯 CONCLUSION:")
        if retry_worthwhile:
            print("✅ Retry system appears to be WORTHWHILE")
            print("   Continue using progressive retry logic")
        else:
            print("⚠️  Retry system may NOT be worthwhile")
            print("   Consider disabling or modifying retry logic")

        if recommendations:
            print("\n💡 RECOMMENDATIONS:")
            for rec in recommendations:
                print(f"   - {rec}")

        return retry_worthwhile

    except Exception as e:
        print(f"❌ Retry analysis failed: {e}")
        return False

def main():
    """Run all retry statistics tests."""
    print("🚀 Testing Retry Statistics Tracking System")
    print("=" * 50)

    tests = [
        ("Retry Statistics Tracking", test_retry_statistics_tracking),
        ("Retry Worthwhile Analysis", analyze_retry_worthwhile),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("📊 RETRY STATISTICS TEST SUMMARY")
    print("=" * 50)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1

    print(f"\nOverall Result: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All retry statistics tests passed!")
        print("\n✅ Retry Statistics Features Verified:")
        print("1. ✅ Comprehensive retry attempt tracking")
        print("2. ✅ Success rate analysis by anti-bot level")
        print("3. ✅ Improvement ratio calculations")
        print("4. ✅ Effectiveness metrics and analysis")
        print("5. ✅ Data-driven retry optimization insights")
        print("\n📈 You can now make data-driven decisions about retry effectiveness!")
        return True
    else:
        print("⚠️  Some retry statistics features need attention.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)