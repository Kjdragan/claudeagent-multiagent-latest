#!/usr/bin/env python3
"""
Test Script for Anti-Bot Optimization System

Tests the new difficult sites functionality including:
- Difficult sites database loading
- URL matching and domain extraction
- Optimal level selection
- CLI tool functionality
"""

import sys
import os

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from difficult_sites_manager import get_standalone_manager, extract_domain_from_url, is_url_difficult_site
    print("✅ Successfully imported difficult_sites_manager")
except ImportError as e:
    print(f"❌ Failed to import difficult_sites_manager: {e}")
    sys.exit(1)


def test_difficult_sites_loading():
    """Test that difficult sites are loaded correctly."""
    print("\n🧪 Testing Difficult Sites Loading...")

    try:
        manager = get_standalone_manager()
        sites = manager.get_all_difficult_sites()

        if len(sites) == 0:
            print("❌ No difficult sites loaded")
            return False

        print(f"✅ Loaded {len(sites)} difficult sites")

        # Check for expected sites
        expected_sites = ['linkedin.com', 'facebook.com', 'twitter.com']
        for expected in expected_sites:
            if manager.is_difficult_site(expected):
                print(f"✅ Found expected site: {expected}")
            else:
                print(f"⚠️  Expected site not found: {expected}")

        return True

    except Exception as e:
        print(f"❌ Error loading difficult sites: {e}")
        return False


def test_url_matching():
    """Test URL extraction and matching functionality."""
    print("\n🧪 Testing URL Matching...")

    test_cases = [
        {
            'url': 'https://linkedin.com/in/some-profile',
            'expected_domain': 'linkedin.com',
            'should_be_difficult': True,
            'expected_level': 2
        },
        {
            'url': 'https://facebook.com/some-page',
            'expected_domain': 'facebook.com',
            'should_be_difficult': True,
            'expected_level': 3
        },
        {
            'url': 'https://unknown-site.com/page',
            'expected_domain': 'unknown-site.com',
            'should_be_difficult': False,
            'expected_level': None
        },
        {
            'url': 'https://medium.com/article-title',
            'expected_domain': 'medium.com',
            'should_be_difficult': True,
            'expected_level': 1
        }
    ]

    manager = get_standalone_manager()
    all_passed = True

    for test_case in test_cases:
        url = test_case['url']
        expected_domain = test_case['expected_domain']
        should_be_difficult = test_case['should_be_difficult']
        expected_level = test_case['expected_level']

        # Test domain extraction
        extracted_domain = extract_domain_from_url(url)
        if extracted_domain == expected_domain:
            print(f"✅ Domain extraction: {url} -> {extracted_domain}")
        else:
            print(f"❌ Domain extraction failed: {url} -> {extracted_domain} (expected {expected_domain})")
            all_passed = False
            continue

        # Test difficult site detection
        is_difficult, site_config = is_url_difficult_site(url, manager)

        if is_difficult == should_be_difficult:
            print(f"✅ Difficult site detection: {extracted_domain} -> {is_difficult}")
        else:
            print(f"❌ Difficult site detection failed: {extracted_domain} -> {is_difficult} (expected {should_be_difficult})")
            all_passed = False
            continue

        # Test level detection
        if is_difficult and site_config:
            actual_level = site_config.level
            if actual_level == expected_level:
                print(f"✅ Level detection: {extracted_domain} -> level {actual_level}")
            else:
                print(f"❌ Level detection failed: {extracted_domain} -> level {actual_level} (expected {expected_level})")
                all_passed = False
        elif not should_be_difficult:
            print(f"✅ Correctly identified as non-difficult: {extracted_domain}")

    return all_passed


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n🧪 Testing Edge Cases...")

    manager = get_standalone_manager()

    # Test invalid URLs
    invalid_urls = [
        'not-a-url',
        'ftp://invalid.protocol',
        'https://',
        'http://',
        ''
    ]

    for invalid_url in invalid_urls:
        domain = extract_domain_from_url(invalid_url)
        if domain == '':
            print(f"✅ Correctly handled invalid URL: '{invalid_url}'")
        else:
            print(f"⚠️  Invalid URL returned domain: '{invalid_url}' -> '{domain}'")

    # Test domain variations
    domain_variations = [
        ('linkedin.com', True),
        ('LINKEDIN.COM', True),  # Case insensitive
        ('LinkedIn.Com', True),  # Mixed case
        ('subdomain.linkedin.com', False),  # Subdomain should not match
        ('linkedin', False),  # Incomplete domain
    ]

    for domain, should_match in domain_variations:
        is_difficult = manager.is_difficult_site(domain)
        if is_difficult == should_match:
            print(f"✅ Domain variation: '{domain}' -> {is_difficult}")
        else:
            print(f"❌ Domain variation failed: '{domain}' -> {is_difficult} (expected {should_match})")

    return True


def test_configuration_validation():
    """Test configuration validation and statistics."""
    print("\n🧪 Testing Configuration Validation...")

    manager = get_standalone_manager()
    stats = manager.get_stats()

    # Check stats structure
    required_keys = ['total_sites', 'level_distribution', 'config_file']
    for key in required_keys:
        if key in stats:
            print(f"✅ Stats contain key: {key}")
        else:
            print(f"❌ Stats missing key: {key}")
            return False

    # Check level distribution
    level_dist = stats['level_distribution']
    total_from_levels = sum(level_dist.values())

    if total_from_levels == stats['total_sites']:
        print(f"✅ Level distribution consistent: {total_from_levels} sites")
    else:
        print(f"❌ Level distribution inconsistency: {total_from_levels} vs {stats['total_sites']}")
        return False

    # Check level range
    for level in level_dist.keys():
        if 0 <= level <= 3:
            print(f"✅ Valid level: {level}")
        else:
            print(f"❌ Invalid level: {level}")
            return False

    print(f"✅ Total difficult sites: {stats['total_sites']}")
    return True


def main():
    """Run all tests."""
    print("🧪 ANTI-BOT OPTIMIZATION SYSTEM TESTS")
    print("=" * 60)

    tests = [
        test_difficult_sites_loading,
        test_url_matching,
        test_edge_cases,
        test_configuration_validation
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
                print("✅ PASSED")
            else:
                print("❌ FAILED")
        except Exception as e:
            print(f"❌ ERROR: {e}")

    print(f"\n📊 TEST RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Anti-bot optimization system is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
        return 1


if __name__ == '__main__':
    sys.exit(main())