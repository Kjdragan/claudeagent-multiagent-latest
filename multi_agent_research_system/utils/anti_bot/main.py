"""
Anti-Bot Escalation System - Main Entry Point

This module provides the main entry point and high-level interface for the anti-bot
escalation system, integrating all components into a cohesive, production-ready
solution.

Key Features:
- Unified interface for anti-bot escalation functionality
- Easy integration with existing crawling systems
- Comprehensive monitoring and analytics
- Configuration management with environment overrides
- Performance optimization and domain learning

Phase 1.2 Implementation: Core Anti-Bot Escalation System with 4 levels
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from . import (
    AntiBotLevel, EscalationResult, EscalationStats,
    extract_domain_from_url, validate_anti_bot_level
)

from .escalation_manager import AntiBotEscalationManager
from .config import get_anti_bot_config, get_anti_bot_config_manager
from .monitoring import get_anti_bot_monitor, record_escalation_performance

logger = logging.getLogger(__name__)


class AntiBotEscalationSystem:
    """
    High-level interface for the anti-bot escalation system.

    This class provides a unified interface that integrates:
    - Progressive anti-bot escalation (4 levels)
    - Domain learning and optimization
    - Performance monitoring and analytics
    - Configuration management
    - Cooldown and retry management
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the anti-bot escalation system.

        Args:
            config: Optional configuration dictionary to override defaults
        """
        self.config = get_anti_bot_config(config)
        self.config_manager = get_anti_bot_config_manager()
        self.escalation_manager = AntiBotEscalationManager(config)
        self.monitor = get_anti_bot_monitor(config)

        # System state
        self.initialized_at = datetime.now()
        self.total_requests_processed = 0
        self.last_activity = None

        logger.info(f"Anti-Bot Escalation System initialized with {len(self.config.level_configs)} escalation levels")

    async def crawl_url(
        self,
        url: str,
        initial_level: Optional[int] = None,
        max_level: int = 3,
        session_id: Optional[str] = None,
        **kwargs
    ) -> EscalationResult:
        """
        Crawl a URL with automatic anti-bot escalation.

        This is the main entry point for crawling with anti-bot protection.
        The system will automatically escalate through levels if needed.

        Args:
            url: URL to crawl
            initial_level: Starting anti-bot level (None for auto-detection)
            max_level: Maximum escalation level (0-3)
            session_id: Session ID for correlation tracking
            **kwargs: Additional parameters

        Returns:
            EscalationResult with comprehensive crawl information

        Example:
            ```python
            system = AntiBotEscalationSystem()
            result = await system.crawl_url("https://example.com")
            if result.success:
                print(f"Content: {result.content[:200]}...")
                print(f"Used level: {result.final_level}")
            else:
                print(f"Failed: {result.error}")
            ```
        """
        # Validate inputs
        if not url or not isinstance(url, str):
            return EscalationResult(
                url=url or "",
                domain="",
                success=False,
                error="Invalid URL provided",
                duration=0.0
            )

        # Validate levels
        if initial_level is not None:
            initial_level = validate_anti_bot_level(initial_level)
        max_level = validate_anti_bot_level(max_level)

        if initial_level is not None and initial_level > max_level:
            initial_level = max_level

        # Update system state
        self.total_requests_processed += 1
        self.last_activity = datetime.now()

        # Generate session ID if not provided
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Starting anti-bot crawl: {url} (level {initial_level or 'auto'}-{max_level})")

        # Execute crawl with escalation
        result = await self.escalation_manager.crawl_with_escalation(
            url=url,
            initial_level=initial_level,
            max_level=max_level,
            session_id=session_id,
            **kwargs
        )

        # Record performance metrics
        record_escalation_performance(result)

        # Log completion
        if result.success:
            logger.info(f"✅ Successfully crawled {url} at level {result.final_level} "
                       f"({result.word_count} words, {result.duration:.1f}s)")
        else:
            logger.warning(f"❌ Failed to crawl {url} after {result.attempts_made} attempts "
                          f"(final level: {result.final_level})")

        return result

    async def crawl_urls(
        self,
        urls: List[str],
        initial_level: Optional[int] = None,
        max_level: int = 3,
        max_concurrent: Optional[int] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> List[EscalationResult]:
        """
        Crawl multiple URLs with concurrent anti-bot escalation.

        Args:
            urls: List of URLs to crawl
            initial_level: Starting anti-bot level
            max_level: Maximum escalation level
            max_concurrent: Maximum concurrent crawls (None for system default)
            session_id: Session ID for correlation
            **kwargs: Additional parameters

        Returns:
            List of EscalationResult objects

        Example:
            ```python
            system = AntiBotEscalationSystem()
            urls = ["https://site1.com", "https://site2.com", "https://site3.com"]
            results = await system.crawl_urls(urls, max_concurrent=3)

            successful = sum(1 for r in results if r.success)
            print(f"Successfully crawled {successful}/{len(urls)} URLs")
            ```
        """
        if not urls:
            return []

        # Validate URLs
        valid_urls = [url for url in urls if url and isinstance(url, str)]
        if len(valid_urls) != len(urls):
            logger.warning(f"Filtered {len(urls) - len(valid_urls)} invalid URLs")

        # Use system concurrent limit if not specified
        if max_concurrent is None:
            max_concurrent = self.config.concurrent_limit

        # Generate session ID if not provided
        if session_id is None:
            session_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Starting batch anti-bot crawl: {len(valid_urls)} URLs, "
                   f"max_concurrent={max_concurrent}, session={session_id}")

        # Update system state
        self.total_requests_processed += len(valid_urls)
        self.last_activity = datetime.now()

        # Execute batch crawl
        results = await self.escalation_manager.batch_crawl_with_escalation(
            urls=valid_urls,
            initial_level=initial_level,
            max_level=max_level,
            max_concurrent=max_concurrent,
            session_id=session_id,
            **kwargs
        )

        # Record performance metrics for all results
        for result in results:
            record_escalation_performance(result)

        # Log batch completion
        successful = sum(1 for r in results if r.success)
        escalations = sum(1 for r in results if r.escalation_used)
        avg_attempts = sum(r.attempts_made for r in results) / len(results)

        logger.info(f"✅ Batch crawl completed: {successful}/{len(valid_urls)} successful "
                   f"({successful / len(valid_urls):.1%}), {escalations} escalations, "
                   f"{avg_attempts:.1f} avg attempts")

        return results

    def get_domain_profile(self, domain: str) -> Optional[Dict[str, Any]]:
        """
        Get learned profile for a specific domain.

        Args:
            domain: Domain to get profile for

        Returns:
            Domain profile dictionary or None if not found

        Example:
            ```python
            system = AntiBotEscalationSystem()
            profile = system.get_domain_profile("example.com")
            if profile:
                print(f"Optimal level: {profile['optimal_level']}")
                print(f"Success rate: {profile['success_rate']:.1f}%")
            ```
        """
        profile = self.escalation_manager.get_domain_profile(domain)
        return profile.to_dict() if profile else None

    def get_domain_insights(self, domain: str) -> Dict[str, Any]:
        """
        Get comprehensive insights for a specific domain.

        Args:
            domain: Domain to analyze

        Returns:
            Dictionary with domain insights and recommendations

        Example:
            ```python
            system = AntiBotEscalationSystem()
            insights = system.get_domain_insights("example.com")
            print(f"Domain: {insights['domain']}")
            print(f"Success rate: {insights['success_rate']:.1f}%")
            print(f"Recommendations: {len(insights['recommendations'])}")
            ```
        """
        return self.escalation_manager.get_domain_insights(domain)

    def get_system_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.

        Returns:
            Dictionary with system performance statistics

        Example:
            ```python
            system = AntiBotEscalationSystem()
            stats = system.get_system_statistics()
            print(f"Total requests: {stats['total_requests']}")
            print(f"Success rate: {stats['success_rate']:.1f}%")
            print(f"Domains tracked: {stats['domains_tracked']}")
            ```
        """
        # Get escalation manager statistics
        escalation_stats = self.escalation_manager.get_statistics()

        # Get monitor statistics
        monitor_stats = self.monitor.get_performance_summary()

        # Combine with system-level stats
        system_stats = {
            'system_info': {
                'initialized_at': self.initialized_at.isoformat(),
                'total_requests_processed': self.total_requests_processed,
                'last_activity': self.last_activity.isoformat() if self.last_activity else None,
                'uptime_seconds': (datetime.now() - self.initialized_at).total_seconds()
            },
            'escalation_performance': escalation_stats,
            'monitoring_dashboard': monitor_stats,
            'configuration': {
                'max_attempts_per_url': self.config.max_attempts_per_url,
                'concurrent_limit': self.config.concurrent_limit,
                'domain_learning_enabled': self.config.enable_domain_learning,
                'cooldown_management_enabled': self.config.enable_cooldown_management,
                'performance_monitoring_enabled': self.config.performance_monitoring
            }
        }

        return system_stats

    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get current optimization recommendations.

        Returns:
            List of optimization recommendations

        Example:
            ```python
        system = AntiBotEscalationSystem()
        recommendations = system.get_optimization_recommendations()
        for rec in recommendations:
            print(f"{rec['priority']}: {rec['title']}")
            print(f"  {rec['description']}")
            ```
        """
        return self.monitor.optimization_recommendations

    def configure_level(self, level: int, settings: Dict[str, Any]) -> bool:
        """
        Configure settings for a specific anti-bot level.

        Args:
            level: Anti-bot level (0-3)
            settings: Settings to apply

        Returns:
            True if configuration successful, False otherwise

        Example:
            ```python
            system = AntiBotEscalationSystem()
            success = system.configure_level(2, {
                'timeout': 90,
                'user_agent_rotation': True
            })
            ```
        """
        try:
            # Update configuration
            level_config = self.config_manager.get_level_config(level)
            level_config.update(settings)

            # Reinitialize escalation manager with new config
            self.escalation_manager = AntiBotEscalationManager(self.config.__dict__)

            logger.info(f"Updated configuration for level {level}")
            return True

        except Exception as e:
            logger.error(f"Failed to configure level {level}: {e}")
            return False

    def reset_statistics(self) -> None:
        """Reset all system statistics."""
        self.escalation_manager.reset_statistics()
        self.monitor.reset_metrics()
        self.total_requests_processed = 0
        logger.info("Reset all system statistics")

    def save_state(self) -> bool:
        """Save system state (domain profiles, statistics, etc.).

        Returns:
            True if save successful, False otherwise
        """
        try:
            # Save domain profiles
            self.escalation_manager.save_domain_profiles()

            # Export performance data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = f"KEVIN/data/anti_bot_state_{timestamp}.json"
            self.monitor.export_performance_data(export_path)

            logger.info(f"System state saved to {export_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save system state: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """
        Perform system health check.

        Returns:
            Dictionary with health status information

        Example:
            ```python
            system = AntiBotEscalationSystem()
            health = system.health_check()
            print(f"System health: {health['status']}")
            print(f"Issues: {len(health['issues'])}")
            ```
        """
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'issues': [],
            'warnings': [],
            'metrics': {}
        }

        # Check system activity
        if self.last_activity:
            time_since_activity = (datetime.now() - self.last_activity).total_seconds()
            if time_since_activity > 3600:  # No activity for 1 hour
                health['warnings'].append(f"No activity for {time_since_activity:.0f} seconds")

        # Check performance metrics
        stats = self.get_system_statistics()
        perf = stats['escalation_performance']['performance']

        if perf['error_rate'] > 30:
            health['issues'].append(f"High error rate: {perf['error_rate']:.1f}%")
            health['status'] = 'degraded'

        if perf['avg_duration_per_crawl'] > 60:
            health['warnings'].append(f"High average response time: {perf['avg_duration_per_crawl']:.1f}s")

        # Check configuration
        if not self.config.enable_domain_learning:
            health['warnings'].append("Domain learning is disabled")

        if not self.config.enable_cooldown_management:
            health['warnings'].append("Cooldown management is disabled")

        # Add summary metrics
        health['metrics'] = {
            'total_requests': self.total_requests_processed,
            'success_rate': perf['overall_success_rate'],
            'avg_response_time': perf['avg_duration_per_crawl'],
            'domains_tracked': stats['escalation_performance']['domain_learning']['total_domains_tracked']
        }

        # Determine overall status
        if health['issues']:
            health['status'] = 'unhealthy'
        elif health['warnings']:
            health['status'] = 'warning'

        return health


# Convenience functions for quick usage
async def crawl_with_anti_bot(
    url: str,
    initial_level: Optional[int] = None,
    max_level: int = 3,
    config: Optional[Dict[str, Any]] = None
) -> EscalationResult:
    """
    Quick function to crawl a single URL with anti-bot protection.

    Args:
        url: URL to crawl
        initial_level: Starting anti-bot level
        max_level: Maximum escalation level
        config: Optional configuration

    Returns:
        EscalationResult with crawl information

    Example:
        ```python
        result = await crawl_with_anti_bot("https://example.com")
        if result.success:
            print(f"Got {len(result.content)} characters")
        ```
    """
    system = AntiBotEscalationSystem(config)
    return await system.crawl_url(url, initial_level, max_level)


async def crawl_urls_with_anti_bot(
    urls: List[str],
    initial_level: Optional[int] = None,
    max_level: int = 3,
    max_concurrent: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None
) -> List[EscalationResult]:
    """
    Quick function to crawl multiple URLs with anti-bot protection.

    Args:
        urls: List of URLs to crawl
        initial_level: Starting anti-bot level
        max_level: Maximum escalation level
        max_concurrent: Maximum concurrent crawls
        config: Optional configuration

    Returns:
        List of EscalationResult objects

    Example:
        ```python
        urls = ["https://site1.com", "https://site2.com"]
        results = await crawl_urls_with_anti_bot(urls, max_concurrent=3)
        successful = sum(1 for r in results if r.success)
        print(f"Success: {successful}/{len(urls)}")
        ```
    """
    system = AntiBotEscalationSystem(config)
    return await system.crawl_urls(urls, initial_level, max_level, max_concurrent)


def get_anti_bot_system(config: Optional[Dict[str, Any]] = None) -> AntiBotEscalationSystem:
    """
    Get a configured anti-bot escalation system instance.

    Args:
        config: Optional configuration

    Returns:
        AntiBotEscalationSystem instance

    Example:
        ```python
        system = get_anti_bot_system({
            'max_attempts_per_url': 5,
            'enable_domain_learning': True
        })
        result = await system.crawl_url("https://example.com")
        ```
    """
    return AntiBotEscalationSystem(config)


# Module initialization
def initialize_anti_bot_system(config: Optional[Dict[str, Any]] = None) -> AntiBotEscalationSystem:
    """
    Initialize the anti-bot escalation system for the application.

    This function should be called once during application startup to configure
    and initialize the anti-bot system.

    Args:
        config: Optional configuration dictionary

    Returns:
        Initialized AntiBotEscalationSystem instance

    Example:
        ```python
        # In your application startup
        anti_bot_system = initialize_anti_bot_system({
            'max_attempts_per_url': 4,
            'concurrent_limit': 5,
            'enable_domain_learning': True
        })
        ```
    """
    logger.info("Initializing Anti-Bot Escalation System...")

    system = AntiBotEscalationSystem(config)

    # Perform health check
    health = system.health_check()
    if health['status'] == 'healthy':
        logger.info("✅ Anti-Bot Escalation System initialized successfully")
    else:
        logger.warning(f"⚠️ Anti-Bot system initialized with issues: {health['issues']}")

    return system


# Export main classes and functions
__all__ = [
    'AntiBotEscalationSystem',
    'crawl_with_anti_bot',
    'crawl_urls_with_anti_bot',
    'get_anti_bot_system',
    'initialize_anti_bot_system'
]