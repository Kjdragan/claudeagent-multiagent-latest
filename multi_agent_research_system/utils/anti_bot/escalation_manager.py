"""
Enhanced Anti-Bot Escalation Manager with Domain Learning

This module implements the sophisticated AntiBotEscalationManager with domain learning,
cooldown management, and 4-level progressive escalation as specified in the
technical enhancements document.

Key Features:
- 4-level progressive escalation (Basic → Enhanced → Advanced → Stealth)
- Domain learning and reputation management
- Intelligent cooldown management
- Performance monitoring and optimization
- Integration with enhanced logging from Phase 1.1
- Comprehensive statistics and analytics

Based on Technical Enhancements Section 2: Detailed Anti-Bot Escalation System
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import asdict

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

from . import (
    AntiBotLevel, AntiBotConfig, EscalationResult, EscalationStats,
    DomainProfile, CooldownEntry, EscalationTrigger,
    extract_domain_from_url, detect_escalation_triggers,
    generate_realistic_user_agent, calculate_escalation_delay,
    validate_anti_bot_level
)

# Import enhanced logging from Phase 1.1
try:
    from ...agent_logging.enhanced_logger import get_enhanced_logger, LogLevel, LogCategory, AgentEventType
    ENHANCED_LOGGING_AVAILABLE = True
except ImportError:
    ENHANCED_LOGGING_AVAILABLE = False
    import logging

logger = logging.getLogger(__name__)


class AntiBotEscalationManager:
    """
    Progressive anti-bot escalation system with domain learning capabilities.

    This manager implements intelligent anti-bot detection avoidance through:
    - 4-level escalation with automatic progression
    - Domain-specific learning and optimization
    - Intelligent cooldown management
    - Performance monitoring and analytics
    - Integration with enhanced logging infrastructure
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the anti-bot escalation manager.

        Args:
            config: Configuration dictionary with escalation settings
        """
        self.config = config or {}
        self.stats = EscalationStats()

        # Domain learning and management
        self.domain_profiles: Dict[str, DomainProfile] = {}
        self.domain_cooldowns: Dict[str, CooldownEntry] = {}

        # Escalation settings
        self.max_attempts_per_url = self.config.get('max_attempts_per_url', 4)
        self.base_delay = self.config.get('base_delay', 1.0)
        self.max_delay = self.config.get('max_delay', 30.0)
        self.enable_domain_learning = self.config.get('enable_domain_learning', True)
        self.enable_cooldown_management = self.config.get('enable_cooldown_management', True)

        # Learning settings
        self.min_attempts_for_learning = self.config.get('min_attempts_for_learning', 3)
        self.learning_confidence_threshold = self.config.get('learning_confidence_threshold', 0.7)
        self.auto_optimize_levels = self.config.get('auto_optimize_levels', True)

        # Performance settings
        self.concurrent_limit = self.config.get('concurrent_limit', 5)
        self.cache_enabled = self.config.get('cache_enabled', True)
        self.performance_monitoring = self.config.get('performance_monitoring', True)

        # Setup enhanced logging
        self._setup_logging()

        # Load existing domain profiles
        self._load_domain_profiles()

        # Initialize escalation configurations
        self._escalation_configs = self._create_escalation_configs()

        logger.info(f"AntiBotEscalationManager initialized with domain learning: {self.enable_domain_learning}")

    def _setup_logging(self):
        """Setup enhanced logging integration."""
        if ENHANCED_LOGGING_AVAILABLE:
            self.enhanced_logger = get_enhanced_logger("anti_bot_escalation")
            self.enhanced_logger.log_event(
                LogLevel.INFO,
                LogCategory.SYSTEM,
                AgentEventType.SESSION_START,
                "Anti-Bot Escalation Manager initialized",
                enable_domain_learning=self.enable_domain_learning,
                max_attempts_per_url=self.max_attempts_per_url,
                concurrent_limit=self.concurrent_limit
            )
        else:
            self.enhanced_logger = None
            logger.setLevel(logging.INFO)

    def _create_escalation_configs(self) -> Dict[int, AntiBotConfig]:
        """Create anti-bot configurations for each escalation level."""
        configs = {}

        # Level 0: Basic Configuration
        configs[AntiBotLevel.BASIC.value] = AntiBotConfig(
            level=AntiBotLevel.BASIC.value,
            headers={
                'User-Agent': 'Mozilla/5.0 (compatible; ResearchBot/1.0)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive'
            },
            timeout=10,
            javascript=False,
            stealth=False,
            wait_strategy="minimal",
            retry_delays=[1.0, 2.0],
            detection_markers=["403", "401", "captcha", "bot detected"],
            cool_down_period=0,
            max_redirects=3,
            keep_alive=False,
            verify_ssl=True
        )

        # Level 1: Enhanced Configuration
        configs[AntiBotLevel.ENHANCED.value] = AntiBotConfig(
            level=AntiBotLevel.ENHANCED.value,
            headers={
                'User-Agent': generate_realistic_user_agent(),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5,en-GB;q=0.3',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate'
            },
            timeout=30,
            javascript=True,
            stealth=False,
            wait_strategy="smart",
            retry_delays=[2.0, 5.0, 10.0],
            detection_markers=["403", "401", "captcha", "access denied", "bot detected"],
            cool_down_period=30,
            user_agent_rotation=True,
            max_redirects=5,
            keep_alive=True,
            verify_ssl=True
        )

        # Level 2: Advanced Configuration
        configs[AntiBotLevel.ADVANCED.value] = AntiBotConfig(
            level=AntiBotLevel.ADVANCED.value,
            headers={
                'User-Agent': generate_realistic_user_agent(),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5,en-GB;q=0.3',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Ch-Ua': '"Google Chrome";v="120", "Not;A=Brand";v="99", "Chromium";v="120"',
                'Sec-Ch-Ua-Mobile': '?0',
                'Sec-Ch-Ua-Platform': '"Linux"'
            },
            timeout=60,
            javascript=True,
            stealth=True,
            wait_strategy="adaptive",
            retry_delays=[5.0, 15.0, 30.0, 60.0],
            detection_markers=["403", "401", "captcha", "access denied", "bot detected", "forbidden"],
            cool_down_period=120,
            viewport='width=device-width, initial-scale=1.0',
            user_agent_rotation=True,
            mouse_movements=True,
            scrolling_behavior=True,
            max_redirects=5,
            keep_alive=True,
            verify_ssl=True
        )

        # Level 3: Stealth Configuration
        configs[AntiBotLevel.STEALTH.value] = AntiBotConfig(
            level=AntiBotLevel.STEALTH.value,
            headers={
                'User-Agent': generate_realistic_user_agent(),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5,en-GB;q=0.3',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Ch-Ua': '"Google Chrome";v="120", "Not;A=Brand";v="99", "Chromium";v="120"',
                'Sec-Ch-Ua-Mobile': '?0',
                'Sec-Ch-Ua-Platform': '"Linux"',
                'Cache-Control': 'max-age=0',
                'Sec-Fetch-User': '?1'
            },
            timeout=120,
            javascript=True,
            stealth=True,
            wait_strategy="human_like",
            retry_delays=[10.0, 30.0, 60.0, 120.0, 300.0],
            detection_markers=["403", "401", "captcha", "access denied", "bot detected", "forbidden", "blocked"],
            cool_down_period=300,
            viewport='width=device-width, initial-scale=1.0',
            proxy_rotation=True,
            user_agent_rotation=True,
            mouse_movements=True,
            keyboard_typing=True,
            scrolling_behavior=True,
            max_redirects=10,
            keep_alive=True,
            verify_ssl=True
        )

        return configs

    async def crawl_with_escalation(
        self,
        url: str,
        initial_level: Optional[int] = None,
        max_level: int = 3,
        session_id: Optional[str] = None,
        **kwargs
    ) -> EscalationResult:
        """
        Crawl URL with progressive anti-bot escalation.

        Args:
            url: URL to crawl
            initial_level: Starting anti-bot level (None for auto-detection)
            max_level: Maximum escalation level
            session_id: Session ID for logging correlation
            **kwargs: Additional parameters

        Returns:
            EscalationResult with comprehensive crawl information
        """
        start_time = time.time()
        attempts_made = 0
        escalation_used = False
        final_level = initial_level or 0
        escalation_triggers = []

        # Extract domain for learning and cooldown management
        domain = extract_domain_from_url(url)
        if not domain:
            return EscalationResult(
                url=url,
                domain="",
                success=False,
                error="Invalid URL - could not extract domain",
                duration=0.0
            )

        # Determine optimal starting level
        if initial_level is None:
            initial_level = self._get_optimal_start_level(domain)
        final_level = initial_level

        # Log escalation start
        if self.enhanced_logger:
            self.enhanced_logger.log_event(
                LogLevel.INFO,
                LogCategory.PERFORMANCE,
                AgentEventType.TASK_START,
                f"Starting anti-bot escalation crawl for {domain}",
                url=url,
                domain=domain,
                initial_level=initial_level,
                max_level=max_level,
                session_id=session_id
            )

        logger.info(f"Starting crawl escalation for {url} at level {initial_level} (domain: {domain})")

        # Check if domain is in cooldown
        if self.enable_cooldown_management and self._is_domain_in_cooldown(domain):
            cooldown_entry = self.domain_cooldowns[domain]
            remaining_seconds = cooldown_entry.remaining_seconds()

            if remaining_seconds > 0:
                logger.info(f"Domain {domain} is in cooldown for {remaining_seconds:.1f} seconds")
                await asyncio.sleep(remaining_seconds)

                # Remove expired cooldown
                if not cooldown_entry.is_active():
                    del self.domain_cooldowns[domain]

        # Try crawling with progressive escalation
        current_level = initial_level
        max_attempts = self.max_attempts_per_url

        for attempt in range(max_attempts):
            attempts_made += 1

            # Calculate delay between attempts
            if attempt > 0:
                delay = calculate_escalation_delay(current_level, attempt, self.base_delay)
                delay = min(delay, self.max_delay)
                await asyncio.sleep(delay)

            # Attempt crawl at current level
            crawl_start = time.time()
            success, content, error_message, detection_info = await self._crawl_at_level(
                url, current_level, session_id
            )
            crawl_duration = time.time() - crawl_start

            # Detect escalation triggers from failure
            if not success:
                escalation_triggers = detect_escalation_triggers(error_message)
                if detection_info:
                    escalation_triggers.extend(detection_info)

            # Update domain profile with attempt data
            if self.enable_domain_learning:
                self._update_domain_profile(
                    domain, success, current_level, crawl_duration,
                    escalation_triggers
                )

            if success:
                # Calculate metrics
                total_duration = time.time() - start_time
                word_count = len(content.split()) if content else 0
                char_count = len(content) if content else 0

                # Update global statistics
                self.stats.total_attempts += 1
                self.stats.successful_crawls += 1
                self.stats.total_duration += total_duration
                self.stats.update_level_stats(current_level, True)

                if attempts_made > 1:
                    self.stats.escalations_triggered += 1
                    escalation_used = True

                # Log successful escalation
                if self.enhanced_logger:
                    self.enhanced_logger.log_event(
                        LogLevel.INFO,
                        LogCategory.PERFORMANCE,
                        AgentEventType.TASK_END,
                        f"Successfully crawled {url} at level {current_level}",
                        url=url,
                        domain=domain,
                        final_level=current_level,
                        attempts_made=attempts_made,
                        duration=total_duration,
                        word_count=word_count,
                        escalation_used=escalation_used,
                        session_id=session_id
                    )

                logger.debug(
                    f"Successfully crawled {url}: {char_count} chars in {total_duration:.1f}s "
                    f"(level {current_level}, attempt {attempts_made})"
                )

                return EscalationResult(
                    url=url,
                    domain=domain,
                    success=True,
                    content=content,
                    duration=total_duration,
                    attempts_made=attempts_made,
                    final_level=final_level,
                    escalation_used=escalation_used,
                    escalation_triggers=escalation_triggers,
                    word_count=word_count,
                    char_count=char_count,
                    total_escalation_time=total_duration,
                    average_attempt_duration=total_duration / attempts_made
                )
            else:
                # Update failure statistics
                self.stats.total_attempts += 1
                self.stats.failed_crawls += 1
                self.stats.update_level_stats(current_level, False)

                # Update escalation trigger statistics
                for trigger in escalation_triggers:
                    self.stats.escalation_triggers[trigger] = (
                        self.stats.escalation_triggers.get(trigger, 0) + 1
                    )

                # Determine if we should escalate
                if current_level < max_level and self._should_escalate(
                    domain, current_level, attempt, escalation_triggers
                ):
                    current_level += 1
                    final_level = current_level
                    escalation_used = True
                    logger.info(f"Escalating to level {current_level} for {url} (triggers: {[t.value for t in escalation_triggers]})")

                    # Set cooldown if consistent failures
                    if self.enable_cooldown_management and attempt >= 2:
                        self._set_domain_cooldown(domain, current_level - 1, current_level, escalation_triggers)

                else:
                    logger.debug(f"Retrying {url} at level {current_level} (attempt {attempt + 1})")

        # All attempts failed
        total_duration = time.time() - start_time

        # Set cooldown after complete failure
        if self.enable_cooldown_management:
            self._set_domain_cooldown(domain, final_level, final_level, escalation_triggers)

        # Log failed escalation
        if self.enhanced_logger:
            self.enhanced_logger.log_event(
                LogLevel.WARNING,
                LogCategory.ERROR,
                AgentEventType.ERROR,
                f"All escalation attempts failed for {url}",
                url=url,
                domain=domain,
                final_level=final_level,
                attempts_made=attempts_made,
                duration=total_duration,
                escalation_triggers=[t.value for t in escalation_triggers],
                session_id=session_id
            )

        logger.warning(
            f"Escalation failed for {url}: {attempts_made} attempts across levels {initial_level}-{final_level}"
        )

        return EscalationResult(
            url=url,
            domain=domain,
            success=False,
            error=f"All {attempts_made} attempts failed across levels {initial_level}-{final_level}",
            duration=total_duration,
            attempts_made=attempts_made,
            final_level=final_level,
            escalation_used=escalation_used,
            escalation_triggers=escalation_triggers,
            total_escalation_time=total_duration,
            average_attempt_duration=total_duration / attempts_made if attempts_made > 0 else 0
        )

    async def _crawl_at_level(
        self,
        url: str,
        level: int,
        session_id: Optional[str] = None
    ) -> Tuple[bool, Optional[str], Optional[str], Optional[List[str]]]:
        """Crawl URL at specific anti-bot level.

        Args:
            url: URL to crawl
            level: Anti-bot level to use
            session_id: Session ID for correlation

        Returns:
            Tuple of (success, content, error_message, detection_info)
        """
        try:
            config = self._escalation_configs.get(level)
            if not config:
                return False, None, f"Invalid anti-bot level: {level}", None

            # Create crawl configuration
            crawl_config = self._create_crawl_config(config)
            browser_config = self._create_browser_config(config)

            # Perform crawl
            if browser_config:
                async with AsyncWebCrawler(config=browser_config) as crawler:
                    result = await crawler.arun(url, config=crawl_config)
            else:
                async with AsyncWebCrawler() as crawler:
                    result = await crawler.arun(url, config=crawl_config)

            if result.success and result.markdown:
                # Check for detection markers in content
                detection_info = self._check_for_detection_markers(result.markdown, config.detection_markers)

                if detection_info:
                    return False, result.markdown, "Bot detection markers found in content", detection_info

                return True, result.markdown, None, None
            else:
                error_msg = result.error_message if hasattr(result, 'error_message') else "Unknown crawling error"
                return False, None, error_msg, None

        except Exception as e:
            error_msg = f"Crawling error at level {level}: {str(e)}"
            logger.debug(f"Level {level} crawl failed for {url}: {e}")
            return False, None, error_msg, None

    def _create_crawl_config(self, anti_bot_config: AntiBotConfig) -> CrawlerRunConfig:
        """Create crawl configuration from anti-bot configuration."""
        base_config = {
            'cache_mode': CacheMode.BYPASS,
            'check_robots_txt': False,
            'remove_overlay_elements': True,
            'headers': anti_bot_config.headers,
            'page_timeout': anti_bot_config.timeout * 1000,  # Convert to milliseconds
            'user_agent': anti_bot_config.headers.get('User-Agent')
        }

        # Level-specific configurations
        if anti_bot_config.level == AntiBotLevel.BASIC.value:
            base_config.update({
                'simulate_user': False,
                'magic': False
            })

        elif anti_bot_config.level == AntiBotLevel.ENHANCED.value:
            base_config.update({
                'simulate_user': True,
                'magic': True,
                'wait_for': 'body'
            })

        elif anti_bot_config.level == AntiBotLevel.ADVANCED.value:
            base_config.update({
                'simulate_user': True,
                'magic': True,
                'wait_until': 'domcontentloaded',
                'delay_before_return_html': 1.0
            })

        else:  # STEALTH
            base_config.update({
                'simulate_user': True,
                'magic': True,
                'wait_until': 'domcontentloaded',
                'delay_before_return_html': anti_bot_config.level * 0.5,
                'js_code': [
                    "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})",
                    "Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]})",
                    "Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']})",
                    "Object.defineProperty(screen, 'availHeight', {get: () => 1040})",
                    "Object.defineProperty(screen, 'availWidth', {get: () => 1920})"
                ]
            })

            # Add content filtering for stealth mode
            md_generator = DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(threshold=0.4)
            )
            base_config['markdown_generator'] = md_generator

        return CrawlerRunConfig(**base_config)

    def _create_browser_config(self, anti_bot_config: AntiBotConfig) -> Optional[BrowserConfig]:
        """Create browser configuration from anti-bot configuration."""
        if anti_bot_config.level >= AntiBotLevel.ADVANCED.value:
            return BrowserConfig(
                headless=True,
                browser_type="chromium",
                viewport_width=1920,
                viewport_height=1080,
                user_agent=anti_bot_config.headers.get('User-Agent'),
                java_script_enabled=anti_bot_config.javascript
            )
        return None

    def _check_for_detection_markers(self, content: str, markers: List[str]) -> Optional[List[str]]:
        """Check content for bot detection markers.

        Args:
            content: Content to check
            markers: List of detection markers to look for

        Returns:
            List of found markers, or None if none found
        """
        if not content or not markers:
            return None

        content_lower = content.lower()
        found_markers = []

        for marker in markers:
            if marker.lower() in content_lower:
                found_markers.append(marker)

        return found_markers if found_markers else None

    def _get_optimal_start_level(self, domain: str) -> int:
        """Get optimal starting anti-bot level for a domain.

        Args:
            domain: Domain to get level for

        Returns:
            Optimal starting level (0-3)
        """
        if not self.enable_domain_learning or domain not in self.domain_profiles:
            return AntiBotLevel.ENHANCED.value  # Default to enhanced

        profile = self.domain_profiles[domain]
        return profile.get_recommended_level()

    def _should_escalate(
        self,
        domain: str,
        current_level: int,
        attempt: int,
        triggers: List[EscalationTrigger]
    ) -> bool:
        """Determine if escalation should be attempted.

        Args:
            domain: Domain being crawled
            current_level: Current anti-bot level
            attempt: Current attempt number
            triggers: Detected escalation triggers

        Returns:
            True if should escalate, False otherwise
        """
        # Always escalate after first failure
        if attempt == 0:
            return True

        # Check for high-priority triggers that require immediate escalation
        high_priority_triggers = {
            EscalationTrigger.BOT_DETECTION,
            EscalationTrigger.CAPTCHA_CHALLENGE
        }
        if any(trigger in high_priority_triggers for trigger in triggers):
            return True

        # Check for rate limiting (escalate faster)
        if EscalationTrigger.RATE_LIMIT in triggers:
            return True

        # Check domain profile for escalation patterns
        if self.enable_domain_learning and domain in self.domain_profiles:
            profile = self.domain_profiles[domain]

            # Escalate if domain has high detection sophistication
            if profile.detection_sophistication > current_level:
                return True

            # Escalate if recent success rate is low
            if profile.success_rate < 30:
                return True

        # Default: allow limited attempts per level
        return attempt < 2

    def _update_domain_profile(
        self,
        domain: str,
        success: bool,
        level: int,
        response_time: float,
        triggers: List[EscalationTrigger]
    ):
        """Update domain profile with attempt data.

        Args:
            domain: Domain to update
            success: Whether attempt was successful
            level: Anti-bot level used
            response_time: Response time in seconds
            triggers: Escalation triggers detected
        """
        if not self.enable_domain_learning:
            return

        if domain not in self.domain_profiles:
            self.domain_profiles[domain] = DomainProfile(
                domain=domain,
                optimal_level=AntiBotLevel.ENHANCED.value,
                last_updated=datetime.now()
            )

        profile = self.domain_profiles[domain]
        profile.update_attempt(success, level, response_time, triggers)

        # Auto-optimize level if enabled
        if self.auto_optimize_levels and profile.total_attempts >= self.min_attempts_for_learning:
            self._optimize_domain_level(domain, profile)

        # Log domain profile update
        if self.enhanced_logger:
            self.enhanced_logger.log_event(
                LogLevel.DEBUG,
                LogCategory.PERFORMANCE,
                AgentEventType.MESSAGE_PROCESSED,
                f"Updated domain profile for {domain}",
                domain=domain,
                success=success,
                level=level,
                response_time=response_time,
                optimal_level=profile.optimal_level,
                success_rate=profile.success_rate,
                triggers=[t.value for t in triggers]
            )

    def _optimize_domain_level(self, domain: str, profile: DomainProfile):
        """Optimize anti-bot level for a domain based on learning data.

        Args:
            domain: Domain to optimize
            profile: Domain profile with learning data
        """
        if not profile.escalation_history:
            return

        # Analyze success rates by level
        level_stats = {}
        recent_history = profile.escalation_history[-10:]

        for level in recent_history:
            if level not in level_stats:
                level_stats[level] = {'attempts': 0, 'successes': 0}
            level_stats[level]['attempts'] += 1

            # Estimate success based on consecutive failures
            if profile.consecutive_failures == 0:
                level_stats[level]['successes'] += 1

        # Find best performing level
        best_level = profile.optimal_level
        best_success_rate = 0.0

        for level, stats in level_stats.items():
            if stats['attempts'] >= 2:  # Require minimum attempts
                success_rate = stats['successes'] / stats['attempts']
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    best_level = level

        # Update optimal level if significantly better
        if best_success_rate > self.learning_confidence_threshold:
            old_level = profile.optimal_level
            profile.optimal_level = best_level

            if old_level != best_level:
                logger.info(f"Optimized {domain} anti-bot level: {old_level} → {best_level} "
                          f"(success rate: {best_success_rate:.1%})")

                if self.enhanced_logger:
                    self.enhanced_logger.log_event(
                        LogLevel.INFO,
                        LogCategory.PERFORMANCE,
                        AgentEventType.MESSAGE_PROCESSED,
                        f"Optimized anti-bot level for {domain}",
                        domain=domain,
                        old_level=old_level,
                        new_level=best_level,
                        success_rate=best_success_rate,
                        optimization_type="automatic"
                    )

    def _is_domain_in_cooldown(self, domain: str) -> bool:
        """Check if domain is currently in cooldown.

        Args:
            domain: Domain to check

        Returns:
            True if domain is in cooldown, False otherwise
        """
        if not self.enable_cooldown_management:
            return False

        if domain not in self.domain_cooldowns:
            return False

        cooldown_entry = self.domain_cooldowns[domain]
        if not cooldown_entry.is_active():
            # Remove expired cooldown
            del self.domain_cooldowns[domain]
            return False

        return True

    def _set_domain_cooldown(
        self,
        domain: str,
        original_level: int,
        escalated_level: int,
        triggers: List[EscalationTrigger]
    ):
        """Set cooldown period for a domain.

        Args:
            domain: Domain to set cooldown for
            original_level: Original anti-bot level
            escalated_level: Level escalated to
            triggers: Escalation triggers detected
        """
        if not self.enable_cooldown_management:
            return

        # Calculate cooldown duration based on escalation and triggers
        base_cooldown = 5  # Base 5 minutes

        # Increase cooldown based on level difference
        level_diff = escalated_level - original_level
        level_multiplier = 1 + (level_diff * 0.5)

        # Increase cooldown based on triggers
        trigger_multipliers = {
            EscalationTrigger.RATE_LIMIT: 2.0,
            EscalationTrigger.BOT_DETECTION: 3.0,
            EscalationTrigger.CAPTCHA_CHALLENGE: 4.0,
            EscalationTrigger.ACCESS_DENIED: 2.5
        }

        trigger_multiplier = 1.0
        for trigger in triggers:
            trigger_multiplier = max(trigger_multiplier, trigger_multipliers.get(trigger, 1.0))

        # Calculate final cooldown duration
        cooldown_minutes = int(base_cooldown * level_multiplier * trigger_multiplier)
        cooldown_minutes = min(cooldown_minutes, 60)  # Cap at 1 hour

        # Create cooldown entry
        reason = f"Escalation from level {original_level} to {escalated_level}"
        if triggers:
            reason += f" (triggers: {[t.value for t in triggers]})"

        cooldown_entry = CooldownEntry(
            domain=domain,
            cooldown_until=datetime.now() + timedelta(minutes=cooldown_minutes),
            reason=reason,
            original_level=original_level,
            escalated_level=escalated_level
        )

        self.domain_cooldowns[domain] = cooldown_entry

        logger.info(f"Set cooldown for {domain}: {cooldown_minutes} minutes ({reason})")

        if self.enhanced_logger:
            self.enhanced_logger.log_event(
                LogLevel.INFO,
                LogCategory.PERFORMANCE,
                AgentEventType.MESSAGE_PROCESSED,
                f"Set cooldown for domain {domain}",
                domain=domain,
                cooldown_minutes=cooldown_minutes,
                reason=reason,
                original_level=original_level,
                escalated_level=escalated_level,
                triggers=[t.value for t in triggers]
            )

    def _load_domain_profiles(self):
        """Load existing domain profiles from file."""
        if not self.enable_domain_learning:
            return

        profiles_file = Path("KEVIN/data/anti_bot_domain_profiles.json")
        if not profiles_file.exists():
            profiles_file.parent.mkdir(parents=True, exist_ok=True)
            return

        try:
            with open(profiles_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for domain_data in data.get('domain_profiles', []):
                domain = domain_data['domain']
                profile = DomainProfile(
                    domain=domain,
                    optimal_level=domain_data['optimal_level'],
                    last_updated=datetime.fromisoformat(domain_data['last_updated']),
                    success_rate=domain_data['success_rate'],
                    total_attempts=domain_data['total_attempts'],
                    successful_attempts=domain_data['successful_attempts'],
                    escalation_history=domain_data.get('escalation_history', []),
                    failure_patterns=[EscalationTrigger(t) for t in domain_data.get('failure_patterns', [])],
                    average_response_time=domain_data.get('average_response_time', 0.0),
                    consecutive_failures=domain_data.get('consecutive_failures', 0),
                    domain_reputation_score=domain_data.get('domain_reputation_score', 0.5),
                    detection_sophistication=domain_data.get('detection_sophistication', 0)
                )

                # Restore cooldown if active
                cooldown_data = domain_data.get('cool_down_until')
                if cooldown_data:
                    cooldown_until = datetime.fromisoformat(cooldown_data)
                    if cooldown_until > datetime.now():
                        self.domain_cooldowns[domain] = CooldownEntry(
                            domain=domain,
                            cooldown_until=cooldown_until,
                            reason="Restored from saved state",
                            original_level=profile.optimal_level,
                            escalated_level=profile.optimal_level
                        )

                self.domain_profiles[domain] = profile

            logger.info(f"Loaded {len(self.domain_profiles)} domain profiles from {profiles_file}")

        except Exception as e:
            logger.error(f"Failed to load domain profiles: {e}")

    def save_domain_profiles(self):
        """Save domain profiles to file."""
        if not self.enable_domain_learning:
            return

        profiles_file = Path("KEVIN/data/anti_bot_domain_profiles.json")
        profiles_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            data = {
                'saved_at': datetime.now().isoformat(),
                'total_domains': len(self.domain_profiles),
                'domain_profiles': [profile.to_dict() for profile in self.domain_profiles.values()]
            }

            with open(profiles_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved {len(self.domain_profiles)} domain profiles to {profiles_file}")

        except Exception as e:
            logger.error(f"Failed to save domain profiles: {e}")

    def get_domain_profile(self, domain: str) -> Optional[DomainProfile]:
        """Get domain profile for learning and analysis.

        Args:
            domain: Domain to get profile for

        Returns:
            Domain profile or None if not found
        """
        return self.domain_profiles.get(domain)

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive escalation statistics.

        Returns:
            Dictionary with detailed statistics
        """
        # Calculate success rates by level
        level_rates = {}
        for level in range(4):
            attempts = self.stats.level_attempts.get(level, 0)
            successes = self.stats.level_successes.get(level, 0)
            if attempts > 0:
                level_rates[f'level_{level}'] = (successes / attempts) * 100

        # Domain statistics
        total_domains = len(self.domain_profiles)
        domains_with_failures = sum(1 for p in self.domain_profiles.values() if p.consecutive_failures > 0)
        domains_in_cooldown = len(self.domain_cooldowns)

        return {
            'performance': {
                'total_attempts': self.stats.total_attempts,
                'successful_crawls': self.stats.successful_crawls,
                'failed_crawls': self.stats.failed_crawls,
                'overall_success_rate': self.stats.calculate_success_rate(),
                'escalation_rate': self.stats.calculate_escalation_rate(),
                'avg_attempts_per_url': self.stats.avg_attempts_per_url,
                'avg_duration_per_crawl': (
                    self.stats.total_duration / self.stats.total_attempts
                    if self.stats.total_attempts > 0 else 0
                )
            },
            'level_performance': level_rates,
            'escalation_triggers': {
                trigger.value: count for trigger, count in self.stats.escalation_triggers.items()
            },
            'domain_learning': {
                'total_domains_tracked': total_domains,
                'domains_with_failures': domains_with_failures,
                'domains_in_cooldown': domains_in_cooldown,
                'learning_enabled': self.enable_domain_learning,
                'auto_optimization_enabled': self.auto_optimize_levels
            },
            'configuration': {
                'max_attempts_per_url': self.max_attempts_per_url,
                'base_delay': self.base_delay,
                'max_delay': self.max_delay,
                'concurrent_limit': self.concurrent_limit,
                'cooldown_management_enabled': self.enable_cooldown_management
            }
        }

    def reset_statistics(self):
        """Reset all escalation statistics."""
        self.stats = EscalationStats()
        logger.info("Reset all escalation statistics")

        if self.enhanced_logger:
            self.enhanced_logger.log_event(
                LogLevel.INFO,
                LogCategory.SYSTEM,
                AgentEventType.MESSAGE_PROCESSED,
                "Reset escalation statistics"
            )

    async def batch_crawl_with_escalation(
        self,
        urls: List[str],
        initial_level: Optional[int] = None,
        max_level: int = 3,
        max_concurrent: Optional[int] = None,
        session_id: Optional[str] = None
    ) -> List[EscalationResult]:
        """Crawl multiple URLs with concurrent anti-bot escalation.

        Args:
            urls: List of URLs to crawl
            initial_level: Starting anti-bot level
            max_level: Maximum escalation level
            max_concurrent: Maximum concurrent crawls
            session_id: Session ID for correlation

        Returns:
            List of EscalationResult objects
        """
        if not urls:
            return []

        # Use configured concurrent limit if not specified
        if max_concurrent is None:
            max_concurrent = self.concurrent_limit

        logger.info(f"Starting batch crawl with escalation: {len(urls)} URLs, "
                   f"level {initial_level or 'auto'}-{max_level}, max_concurrent={max_concurrent}")

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def crawl_with_semaphore(url: str) -> EscalationResult:
            async with semaphore:
                return await self.crawl_with_escalation(
                    url, initial_level, max_level, session_id
                )

        # Execute crawls concurrently
        start_time = time.time()
        tasks = [crawl_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions and calculate statistics
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(
                    EscalationResult(
                        url=urls[i],
                        domain=extract_domain_from_url(urls[i]),
                        success=False,
                        error=str(result)
                    )
                )
            else:
                final_results.append(result)

        # Update global statistics
        total_duration = time.time() - start_time
        successful = sum(1 for r in final_results if r.success)
        escalations = sum(1 for r in final_results if r.escalation_used)
        avg_attempts = sum(r.attempts_made for r in final_results) / len(final_results)

        logger.info(f"Batch crawl completed: {successful}/{len(urls)} successful "
                   f"({successful / len(urls):.1%}), {escalations} escalations, "
                   f"{avg_attempts:.1f} avg attempts, {total_duration:.1f}s total")

        # Log batch completion
        if self.enhanced_logger:
            self.enhanced_logger.log_event(
                LogLevel.INFO,
                LogCategory.PERFORMANCE,
                AgentEventType.TASK_END,
                f"Batch crawl with escalation completed",
                total_urls=len(urls),
                successful=successful,
                success_rate=successful / len(urls),
                escalations_triggered=escalations,
                avg_attempts=avg_attempts,
                total_duration=total_duration,
                session_id=session_id
            )

        return final_results

    def get_domain_insights(self, domain: str) -> Dict[str, Any]:
        """Get detailed insights for a specific domain.

        Args:
            domain: Domain to analyze

        Returns:
            Dictionary with domain insights and recommendations
        """
        profile = self.get_domain_profile(domain)
        cooldown = self.domain_cooldowns.get(domain)

        insights = {
            'domain': domain,
            'profile_exists': profile is not None,
            'in_cooldown': cooldown is not None and cooldown.is_active(),
            'recommendations': []
        }

        if profile:
            insights.update({
                'optimal_level': profile.optimal_level,
                'success_rate': profile.success_rate,
                'total_attempts': profile.total_attempts,
                'reputation_score': profile.domain_reputation_score,
                'detection_sophistication': profile.detection_sophistication,
                'consecutive_failures': profile.consecutive_failures,
                'average_response_time': profile.average_response_time,
                'last_attempt': profile.last_attempt_time.isoformat() if profile.last_attempt_time else None
            })

            # Generate recommendations
            if profile.success_rate < 50:
                insights['recommendations'].append(
                    f"Low success rate ({profile.success_rate:.1f}%). Consider using stealth mode."
                )

            if profile.consecutive_failures >= 3:
                insights['recommendations'].append(
                    "Multiple consecutive failures. Domain may have enhanced detection."
                )

            if profile.detection_sophistication >= 2:
                insights['recommendations'].append(
                    f"High detection sophistication ({profile.detection_sophistication}). "
                    "Start at advanced or stealth level."
                )

            if profile.average_response_time > 10:
                insights['recommendations'].append(
                    "Slow response times detected. Consider longer timeouts."
                )

        if cooldown and cooldown.is_active():
            insights['cooldown'] = {
                'active': True,
                'remaining_seconds': cooldown.remaining_seconds(),
                'reason': cooldown.reason,
                'original_level': cooldown.original_level,
                'escalated_level': cooldown.escalated_level
            }
            insights['recommendations'].append(
                f"Domain is in cooldown for {cooldown.remaining_seconds():.0f} more seconds."
            )

        return insights

    def __del__(self):
        """Cleanup when manager is destroyed."""
        # Save domain profiles
        try:
            self.save_domain_profiles()
        except Exception:
            pass  # Ignore errors during cleanup