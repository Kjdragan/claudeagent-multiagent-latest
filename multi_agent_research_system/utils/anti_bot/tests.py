"""
Comprehensive Tests and Validation for Anti-Bot Escalation System

This module provides comprehensive test coverage for the anti-bot escalation system,
including unit tests, integration tests, and validation of core functionality.

Key Features:
- Unit tests for all core components
- Integration tests for escalation workflows
- Performance validation tests
- Configuration validation tests
- Domain learning system tests
- Mock crawling for safe testing

Based on Phase 1.2 implementation specifications
"""

import asyncio
import pytest
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Import the modules we're testing
from . import (
    AntiBotLevel, AntiBotConfig, EscalationResult, DomainProfile,
    EscalationTrigger, extract_domain_from_url, detect_escalation_triggers,
    generate_realistic_user_agent, calculate_escalation_delay,
    validate_anti_bot_level
)

from .escalation_manager import AntiBotEscalationManager
from .config import AntiBotConfigManager, get_anti_bot_config
from .monitoring import AntiBotMonitor, PerformanceMetrics, OptimizationRecommendation


class TestAntiBotCore:
    """Test core anti-bot components."""

    def test_anti_bot_level_enum(self):
        """Test AntiBotLevel enum functionality."""
        # Test level names
        assert AntiBotLevel.get_level_name(0) == "Basic"
        assert AntiBotLevel.get_level_name(1) == "Enhanced"
        assert AntiBotLevel.get_level_name(2) == "Advanced"
        assert AntiBotLevel.get_level_name(3) == "Stealth"
        assert AntiBotLevel.get_level_name(99) == "Unknown"

        # Test success rate estimates
        assert AntiBotLevel.get_success_rate_estimate(0) == 0.6
        assert AntiBotLevel.get_success_rate_estimate(1) == 0.8
        assert AntiBotLevel.get_success_rate_estimate(2) == 0.9
        assert AntiBotLevel.get_success_rate_estimate(3) == 0.95

    def test_anti_bot_config_validation(self):
        """Test AntiBotConfig validation."""
        # Valid configuration
        config = AntiBotConfig(
            level=1,
            headers={'User-Agent': 'test'},
            timeout=30,
            javascript=True,
            stealth=False,
            wait_strategy="smart"
        )
        assert config.level == 1
        assert config.timeout == 30

        # Invalid level
        with pytest.raises(ValueError):
            AntiBotConfig(level=5, headers={}, timeout=10, javascript=False, stealth=False, wait_strategy="")

        # Invalid timeout
        with pytest.raises(ValueError):
            AntiBotConfig(level=1, headers={}, timeout=-5, javascript=False, stealth=False, wait_strategy="")

    def test_escalation_result_calculation(self):
        """Test EscalationResult calculations."""
        result = EscalationResult(
            url="https://example.com",
            domain="example.com",
            success=True,
            duration=10.0,
            attempts_made=2,
            final_level=2,
            escalation_used=True,
            word_count=500,
            char_count=2500
        )

        # Test success rate improvement calculation
        improvement = result.calculate_success_rate_improvement(1)
        assert improvement > 0  # Should show improvement over baseline

        # Test with no improvement needed
        no_improvement = result.calculate_success_rate_improvement(2)
        assert no_improvement == 0.0

    def test_domain_profile_learning(self):
        """Test DomainProfile learning functionality."""
        profile = DomainProfile(
            domain="example.com",
            optimal_level=1,
            last_updated=datetime.now()
        )

        # Test successful attempt
        profile.update_attempt(success=True, level=1, response_time=2.5)
        assert profile.total_attempts == 1
        assert profile.successful_attempts == 1
        assert profile.success_rate == 100.0
        assert profile.consecutive_failures == 0

        # Test failed attempt
        profile.update_attempt(success=False, level=1, response_time=5.0,
                             triggers=[EscalationTrigger.BOT_DETECTION])
        assert profile.total_attempts == 2
        assert profile.successful_attempts == 1
        assert profile.success_rate == 50.0
        assert profile.consecutive_failures == 1

        # Test recommended level calculation
        recommended = profile.get_recommended_level()
        assert isinstance(recommended, int)
        assert 0 <= recommended <= 3

    def test_utility_functions(self):
        """Test utility functions."""
        # Test domain extraction
        assert extract_domain_from_url("https://www.example.com/path") == "www.example.com"
        assert extract_domain_from_url("http://subdomain.example.org") == "subdomain.example.org"
        assert extract_domain_from_url("invalid-url") == ""

        # Test escalation trigger detection
        triggers = detect_escalation_triggers("403 Forbidden", 403)
        assert EscalationTrigger.ACCESS_DENIED in triggers
        assert EscalationTrigger.BOT_DETECTION not in triggers

        triggers = detect_escalation_triggers("captcha challenge", None)
        assert EscalationTrigger.CAPTCHA_CHALLENGE in triggers

        # Test user agent generation
        ua = generate_realistic_user_agent()
        assert isinstance(ua, str)
        assert len(ua) > 50  # Should be a realistic user agent string

        # Test escalation delay calculation
        delay = calculate_escalation_delay(1, 2, 1.0)
        assert delay > 0
        assert delay < 20  # Should be reasonable

        # Test level validation
        assert validate_anti_bot_level(1) == 1
        assert validate_anti_bot_level(-1) == 0
        assert validate_anti_bot_level(5) == 3


class TestAntiBotConfigManager:
    """Test anti-bot configuration management."""

    def test_default_configuration(self):
        """Test default configuration loading."""
        config_manager = AntiBotConfigManager()
        config = config_manager.get_config()

        assert config.max_attempts_per_url == 4
        assert config.base_delay == 1.0
        assert config.enable_domain_learning is True
        assert config.concurrent_limit == 5

    def test_environment_overrides(self):
        """Test environment variable overrides."""
        # Set environment variables
        import os
        os.environ['ANTI_BOT_MAX_ATTEMPTS'] = '8'
        os.environ['ANTI_BOT_ENABLE_LEARNING'] = 'false'
        os.environ['ANTI_BOT_CONCURRENT_LIMIT'] = '10'

        try:
            config_manager = AntiBotConfigManager()
            config = config_manager.get_config()

            assert config.max_attempts_per_url == 8
            assert config.enable_domain_learning is False
            assert config.concurrent_limit == 10

        finally:
            # Clean up environment variables
            for key in ['ANTI_BOT_MAX_ATTEMPTS', 'ANTI_BOT_ENABLE_LEARNING', 'ANTI_BOT_CONCURRENT_LIMIT']:
                if key in os.environ:
                    del os.environ[key]

    def test_custom_configuration(self):
        """Test custom configuration merging."""
        custom_config = {
            'max_attempts_per_url': 6,
            'base_delay': 2.0,
            'enable_domain_learning': False
        }

        config_manager = AntiBotConfigManager(custom_config)
        config = config_manager.get_config()

        assert config.max_attempts_per_url == 6
        assert config.base_delay == 2.0
        assert config.enable_domain_learning is False
        # Should retain defaults for other values
        assert config.concurrent_limit == 5

    def test_level_configuration(self):
        """Test level-specific configuration."""
        config_manager = AntiBotConfigManager()

        # Test each level configuration
        for level in range(4):
            level_config = config_manager.get_level_config(level)
            assert isinstance(level_config, dict)
            assert 'timeout' in level_config
            assert 'javascript' in level_config
            assert 'stealth' in level_config

            # Check that higher levels have longer timeouts
            if level > 0:
                prev_config = config_manager.get_level_config(level - 1)
                assert level_config['timeout'] >= prev_config['timeout']

    def test_configuration_validation(self):
        """Test configuration validation."""
        config_manager = AntiBotConfigManager()
        validation = config_manager.validate_for_environment()

        assert validation['valid'] is True
        assert 'warnings' in validation
        assert 'recommendations' in validation

    def test_invalid_configuration(self):
        """Test handling of invalid configuration."""
        invalid_config = {
            'max_attempts_per_url': 15,  # Invalid: > 10
            'base_delay': 100.0,  # Invalid: > 60
            'learning_confidence_threshold': 2.0  # Invalid: > 1.0
        }

        with pytest.raises(ValueError):
            AntiBotConfigManager(invalid_config)


class TestAntiBotEscalationManager:
    """Test anti-bot escalation manager functionality."""

    @pytest.fixture
    def escalation_manager(self):
        """Create escalation manager for testing."""
        config = {
            'max_attempts_per_url': 3,
            'enable_domain_learning': True,
            'enable_cooldown_management': True,
            'performance_monitoring': True
        }
        return AntiBotEscalationManager(config)

    @pytest.fixture
    def mock_crawl_result(self):
        """Create mock crawl result."""
        return {
            'success': True,
            'markdown': 'Test content with sufficient length.',
            'error_message': None
        }

    def test_initialization(self, escalation_manager):
        """Test escalation manager initialization."""
        assert escalation_manager.max_attempts_per_url == 3
        assert escalation_manager.enable_domain_learning is True
        assert escalation_manager.enable_cooldown_management is True
        assert len(escalation_manager._escalation_configs) == 4

    def test_optimal_start_level(self, escalation_manager):
        """Test optimal start level determination."""
        # Test with unknown domain (should return default)
        level = escalation_manager._get_optimal_start_level("unknown.com")
        assert level == AntiBotLevel.ENHANCED.value

        # Test with known domain profile
        profile = DomainProfile(
            domain="known.com",
            optimal_level=2,
            last_updated=datetime.now()
        )
        escalation_manager.domain_profiles["known.com"] = profile

        level = escalation_manager._get_optimal_start_level("known.com")
        assert level == 2

    def test_escalation_decision(self, escalation_manager):
        """Test escalation decision logic."""
        # Test first failure (should always escalate)
        should_escalate = escalation_manager._should_escalate(
            "test.com", 1, 0, []
        )
        assert should_escalate is True

        # Test high-priority triggers
        triggers = [EscalationTrigger.BOT_DETECTION]
        should_escalate = escalation_manager._should_escalate(
            "test.com", 2, 1, triggers
        )
        assert should_escalate is True

        # Test rate limiting trigger
        triggers = [EscalationTrigger.RATE_LIMIT]
        should_escalate = escalation_manager._should_escalate(
            "test.com", 1, 1, triggers
        )
        assert should_escalate is True

    @pytest.mark.asyncio
    async def test_crawl_with_escalation_success(self, escalation_manager, mock_crawl_result):
        """Test successful crawl with escalation."""
        with patch('multi_agent_research_system.utils.anti_bot.escalation_manager.AsyncWebCrawler') as mock_crawler:
            # Setup mock crawler
            mock_instance = AsyncMock()
            mock_instance.arun.return_value = MagicMock(**mock_crawl_result)
            mock_crawler.return_value.__aenter__.return_value = mock_instance

            result = await escalation_manager.crawl_with_escalation(
                "https://example.com",
                initial_level=1,
                session_id="test-session"
            )

            assert result.success is True
            assert result.domain == "example.com"
            assert result.final_level == 1
            assert result.content is not None
            assert result.attempts_made == 1

    @pytest.mark.asyncio
    async def test_crawl_with_escalation_failure(self, escalation_manager):
        """Test failed crawl with escalation."""
        with patch('multi_agent_research_system.utils.anti_bot.escalation_manager.AsyncWebCrawler') as mock_crawler:
            # Setup mock crawler to always fail
            mock_instance = AsyncMock()
            mock_instance.arun.return_value = MagicMock(
                success=False,
                error_message="Access denied"
            )
            mock_crawler.return_value.__aenter__.return_value = mock_instance

            result = await escalation_manager.crawl_with_escalation(
                "https://blocked.com",
                initial_level=0,
                max_level=1,
                session_id="test-session"
            )

            assert result.success is False
            assert result.domain == "blocked.com"
            assert result.attempts_made > 1  # Should have tried escalation
            assert result.escalation_used is True

    def test_domain_profile_update(self, escalation_manager):
        """Test domain profile updating."""
        domain = "test.com"

        # Update with successful attempt
        escalation_manager._update_domain_profile(
            domain, True, 1, 2.5, []
        )

        assert domain in escalation_manager.domain_profiles
        profile = escalation_manager.domain_profiles[domain]
        assert profile.total_attempts == 1
        assert profile.successful_attempts == 1
        assert profile.success_rate == 100.0

        # Update with failed attempt
        escalation_manager._update_domain_profile(
            domain, False, 2, 5.0, [EscalationTrigger.BOT_DETECTION]
        )

        assert profile.total_attempts == 2
        assert profile.successful_attempts == 1
        assert profile.success_rate == 50.0

    def test_cooldown_management(self, escalation_manager):
        """Test cooldown management."""
        domain = "test.com"

        # Set cooldown
        escalation_manager._set_domain_cooldown(
            domain, 1, 2, [EscalationTrigger.RATE_LIMIT]
        )

        assert domain in escalation_manager.domain_cooldowns
        assert escalation_manager._is_domain_in_cooldown(domain) is True

        # Test cooldown entry
        cooldown = escalation_manager.domain_cooldowns[domain]
        assert cooldown.domain == domain
        assert cooldown.original_level == 1
        assert cooldown.escalated_level == 2
        assert cooldown.is_active() is True

    def test_statistics_tracking(self, escalation_manager):
        """Test statistics tracking."""
        # Create some test results
        results = [
            EscalationResult(
                url="https://success.com",
                domain="success.com",
                success=True,
                duration=2.0,
                attempts_made=1,
                final_level=1
            ),
            EscalationResult(
                url="https://failure.com",
                domain="failure.com",
                success=False,
                duration=5.0,
                attempts_made=3,
                final_level=3,
                escalation_used=True
            )
        ]

        # Simulate statistics updates
        for result in results:
            escalation_manager.stats.total_attempts += 1
            if result.success:
                escalation_manager.stats.successful_crawls += 1
            else:
                escalation_manager.stats.failed_crawls += 1
                escalation_manager.stats.escalations_triggered += 1

        # Check statistics
        stats = escalation_manager.get_statistics()
        assert stats['performance']['total_attempts'] == 2
        assert stats['performance']['successful_crawls'] == 1
        assert stats['performance']['failed_crawls'] == 1
        assert stats['performance']['escalation_rate'] == 50.0

    def test_domain_insights(self, escalation_manager):
        """Test domain insights generation."""
        domain = "test.com"

        # Create a domain profile with some data
        profile = DomainProfile(
            domain=domain,
            optimal_level=2,
            last_updated=datetime.now()
        )
        profile.total_attempts = 10
        profile.successful_attempts = 7
        profile.consecutive_failures = 2
        profile.detection_sophistication = 2

        escalation_manager.domain_profiles[domain] = profile

        insights = escalation_manager.get_domain_insights(domain)

        assert insights['domain'] == domain
        assert insights['profile_exists'] is True
        assert insights['optimal_level'] == 2
        assert insights['success_rate'] == 70.0
        assert len(insights['recommendations']) > 0


class TestAntiBotMonitor:
    """Test anti-bot performance monitoring."""

    @pytest.fixture
    def monitor(self):
        """Create monitor for testing."""
        config = {
            'monitoring_enabled': True,
            'auto_optimization_enabled': True,
            'alert_thresholds': {
                'error_rate': 20.0,
                'avg_response_time': 30.0
            }
        }
        return AntiBotMonitor(config)

    def test_monitor_initialization(self, monitor):
        """Test monitor initialization."""
        assert monitor.monitoring_enabled is True
        assert monitor.auto_optimization_enabled is True
        assert monitor.metrics is not None

    def test_metrics_recording(self, monitor):
        """Test performance metrics recording."""
        # Create test result
        result = EscalationResult(
            url="https://test.com",
            domain="test.com",
            success=True,
            duration=5.0,
            attempts_made=2,
            final_level=2,
            escalation_used=True
        )

        # Record result
        monitor.record_escalation_result(result)

        # Check metrics were updated
        assert monitor.metrics.total_requests == 1
        assert monitor.metrics.successful_requests == 1
        assert monitor.metrics.total_escalations == 1

    def test_domain_metrics_tracking(self, monitor):
        """Test domain-specific metrics tracking."""
        domain = "test.com"

        # Record multiple results for the same domain
        results = [
            EscalationResult(
                url=f"https://{domain}",
                domain=domain,
                success=True,
                duration=2.0,
                attempts_made=1,
                final_level=1
            ),
            EscalationResult(
                url=f"https://{domain}/page2",
                domain=domain,
                success=False,
                duration=8.0,
                attempts_made=3,
                final_level=3,
                escalation_used=True
            )
        ]

        for result in results:
            monitor.record_escalation_result(result)

        # Check domain metrics
        assert domain in monitor.domain_performance
        domain_metrics = monitor.domain_performance[domain]
        assert domain_metrics['total_requests'] == 2
        assert domain_metrics['successful_requests'] == 1
        assert domain_metrics['success_rate'] == 50.0

    def test_optimization_recommendations(self, monitor):
        """Test optimization recommendations generation."""
        # Simulate poor performance to trigger recommendations
        poor_results = []
        for i in range(10):
            result = EscalationResult(
                url=f"https://bad{i}.com",
                domain=f"bad{i}.com",
                success=False,  # All failures
                duration=15.0,
                attempts_made=4,
                final_level=3,
                escalation_used=True
            )
            poor_results.append(result)
            monitor.record_escalation_result(result)

        # Check that recommendations were generated
        assert len(monitor.optimization_recommendations) > 0

        # Check recommendation structure
        recommendation = monitor.optimization_recommendations[0]
        assert isinstance(recommendation, OptimizationRecommendation)
        assert recommendation.category is not None
        assert recommendation.priority is not None
        assert recommendation.title is not None

    def test_performance_summary(self, monitor):
        """Test performance summary generation."""
        # Record some test data
        results = [
            EscalationResult(
                url="https://good.com",
                domain="good.com",
                success=True,
                duration=3.0,
                attempts_made=1,
                final_level=1
            ),
            EscalationResult(
                url="https://bad.com",
                domain="bad.com",
                success=False,
                duration=10.0,
                attempts_made=3,
                final_level=3,
                escalation_used=True
            )
        ]

        for result in results:
            monitor.record_escalation_result(result)

        # Get performance summary
        summary = monitor.get_performance_summary()

        assert 'overall_metrics' in summary
        assert 'recent_performance' in summary
        assert 'domain_performance' in summary
        assert 'level_performance' in summary
        assert 'system_health' in summary

        # Check overall metrics
        overall = summary['overall_metrics']
        assert overall['total_requests'] == 2
        assert overall['success_rate'] == 50.0

    def test_domain_insights(self, monitor):
        """Test domain insights generation."""
        domain = "test.com"

        # Record results for domain
        result = EscalationResult(
            url=f"https://{domain}",
            domain=domain,
            success=True,
            duration=4.0,
            attempts_made=2,
            final_level=2,
            escalation_used=True
        )
        monitor.record_escalation_result(result)

        # Get domain insights
        insights = monitor.get_domain_insights(domain)

        assert insights['domain'] == domain
        assert insights['tracked'] is True
        assert insights['total_requests'] == 1
        assert insights['success_rate'] == 100.0


class TestIntegration:
    """Integration tests for the complete anti-bot system."""

    @pytest.mark.asyncio
    async def test_full_escalation_workflow(self):
        """Test complete escalation workflow with mocked crawling."""
        # Create escalation manager
        config = {
            'max_attempts_per_url': 2,
            'enable_domain_learning': True,
            'enable_cooldown_management': True
        }
        manager = AntiBotEscalationManager(config)

        # Mock crawl responses (first fails, second succeeds at higher level)
        crawl_responses = [
            {
                'success': False,
                'error_message': 'Access denied'
            },
            {
                'success': True,
                'markdown': 'Successfully fetched content after escalation.'
            }
        ]

        with patch('multi_agent_research_system.utils.anti_bot.escalation_manager.AsyncWebCrawler') as mock_crawler:
            mock_instance = AsyncMock()
            mock_instance.arun.side_effect = [MagicMock(**resp) for resp in crawl_responses]
            mock_crawler.return_value.__aenter__.return_value = mock_instance

            # Execute crawl with escalation
            result = await manager.crawl_with_escalation(
                "https://protected-site.com",
                initial_level=0,
                max_level=2,
                session_id="integration-test"
            )

            # Verify escalation occurred
            assert result.success is True
            assert result.escalation_used is True
            assert result.final_level > 0
            assert result.attempts_made == 2

            # Verify domain learning occurred
            domain = "protected-site.com"
            assert domain in manager.domain_profiles
            profile = manager.domain_profiles[domain]
            assert profile.total_attempts == 1
            assert profile.successful_attempts == 1

    @pytest.mark.asyncio
    async def test_batch_escalation_processing(self):
        """Test batch processing with escalation."""
        config = {
            'max_attempts_per_url': 2,
            'concurrent_limit': 3
        }
        manager = AntiBotEscalationManager(config)

        # Create test URLs
        urls = [
            "https://site1.com",
            "https://site2.com",
            "https://site3.com"
        ]

        # Mock successful crawls
        with patch('multi_agent_research_system.utils.anti_bot.escalation_manager.AsyncWebCrawler') as mock_crawler:
            mock_instance = AsyncMock()
            mock_instance.arun.return_value = MagicMock(
                success=True,
                markdown='Test content'
            )
            mock_crawler.return_value.__aenter__.return_value = mock_instance

            # Execute batch crawl
            results = await manager.batch_crawl_with_escalation(
                urls,
                initial_level=1,
                max_concurrent=2,
                session_id="batch-test"
            )

            # Verify results
            assert len(results) == len(urls)
            assert all(result.success for result in results)

    def test_configuration_integration(self):
        """Test configuration integration across components."""
        # Create custom configuration
        custom_config = {
            'max_attempts_per_url': 5,
            'base_delay': 2.0,
            'enable_domain_learning': True,
            'enable_cooldown_management': True
        }

        # Test configuration manager
        config_manager = AntiBotConfigManager(custom_config)
        config = config_manager.get_config()

        # Test escalation manager with custom config
        manager = AntiBotEscalationManager(custom_config)
        assert manager.max_attempts_per_url == 5
        assert manager.base_delay == 2.0

        # Test monitor with custom config
        monitor_config = {
            'monitoring_enabled': True,
            'alert_thresholds': {
                'error_rate': 25.0,
                'avg_response_time': 35.0
            }
        }
        monitor = AntiBotMonitor(monitor_config)
        assert monitor.alert_thresholds['error_rate'] == 25.0

    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        manager = AntiBotEscalationManager()

        # Test with invalid URL
        async def test_invalid_url():
            result = await manager.crawl_with_escalation("not-a-valid-url")
            assert result.success is False
            assert "Invalid URL" in result.error

        # Test domain extraction handling
        domain = extract_domain_from_url("")
        assert domain == ""

        # Test level validation
        invalid_level = validate_anti_bot_level(10)
        assert invalid_level == 3  # Should be clamped to max

    def test_performance_monitoring_integration(self):
        """Test integration between monitoring and escalation."""
        config = {'monitoring_enabled': True, 'auto_optimization_enabled': True}
        manager = AntiBotEscalationManager()
        monitor = AntiBotMonitor(config)

        # Simulate multiple escalation results
        for i in range(20):
            result = EscalationResult(
                url=f"https://site{i}.com",
                domain=f"site{i}.com",
                success=(i % 3 != 0),  # 2/3 success rate
                duration=5.0 + (i % 5),
                attempts_made=1 + (i % 2),
                final_level=i % 4,
                escalation_used=(i % 2 == 1)
            )

            # Record in both systems
            manager.stats.total_attempts += 1
            if result.success:
                manager.stats.successful_crawls += 1
            else:
                manager.stats.failed_crawls += 1

            monitor.record_escalation_result(result)

        # Verify integration
        assert monitor.metrics.total_requests == 20
        performance_summary = monitor.get_performance_summary()
        assert 'overall_metrics' in performance_summary
        assert performance_summary['overall_metrics']['total_requests'] == 20


# Test execution utilities
def run_comprehensive_tests():
    """Run all anti-bot system tests."""
    print("🧪 Running comprehensive anti-bot escalation system tests...")

    # Run pytest programmatically
    import subprocess
    import sys

    test_file = __file__
    result = subprocess.run([
        sys.executable, "-m", "pytest",
        test_file,
        "-v",
        "--tb=short",
        "--no-header"
    ], capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode == 0:
        print("✅ All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False


def validate_system_configuration():
    """Validate system configuration and dependencies."""
    print("🔧 Validating anti-bot system configuration...")

    try:
        # Test configuration loading
        config = get_anti_bot_config()
        print(f"✅ Configuration loaded successfully")
        print(f"   Max attempts per URL: {config.max_attempts_per_url}")
        print(f"   Domain learning enabled: {config.enable_domain_learning}")
        print(f"   Cooldown management enabled: {config.enable_cooldown_management}")

        # Test escalation manager creation
        manager = AntiBotEscalationManager()
        print(f"✅ Escalation manager created successfully")

        # Test monitor creation
        monitor = AntiBotMonitor()
        print(f"✅ Performance monitor created successfully")

        # Test core utilities
        ua = generate_realistic_user_agent()
        print(f"✅ User agent generation working: {len(ua)} characters")

        delay = calculate_escalation_delay(2, 1, 1.0)
        print(f"✅ Delay calculation working: {delay:.2f} seconds")

        return True

    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False


if __name__ == "__main__":
    """Main test execution."""
    print("🚀 Anti-Bot Escalation System - Test Suite")
    print("=" * 50)

    # Validate configuration
    config_valid = validate_system_configuration()
    print()

    if config_valid:
        # Run comprehensive tests
        tests_passed = run_comprehensive_tests()

        if tests_passed:
            print("\n🎉 All anti-bot system tests completed successfully!")
            print("The system is ready for production use.")
        else:
            print("\n⚠️  Some tests failed. Please review the output above.")
            sys.exit(1)
    else:
        print("\n❌ Configuration validation failed. Please fix configuration issues.")
        sys.exit(1)