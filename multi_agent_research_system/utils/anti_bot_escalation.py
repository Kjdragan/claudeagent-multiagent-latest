"""
Progressive Anti-Bot Escalation System

Implements the 4-level progressive anti-bot escalation system from the technical documentation:
- Level 0: Basic SERP API only
- Level 1: Enhanced headers + JavaScript rendering
- Level 2: Advanced proxy rotation + browser automation
- Level 3: Stealth mode with full browser simulation
- Smart retry logic with automatic escalation
- Success rate tracking and optimization
"""

import asyncio
import json
import logging
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

from .performance_timers import async_timed

logger = logging.getLogger(__name__)


class AntiBotLevel(Enum):
    """Anti-bot escalation levels."""

    BASIC = 0  # Basic SERP API and simple crawl
    ENHANCED = 1  # Enhanced headers + JavaScript rendering
    ADVANCED = 2  # Advanced proxy rotation + browser automation
    STEALTH = 3  # Stealth mode with full browser simulation


@dataclass
class EscalationResult:
    """Result of anti-bot escalation attempt."""

    url: str
    success: bool
    content: str | None = None
    error: str | None = None
    duration: float = 0.0
    attempts_made: int = 0
    final_level: int = 0
    escalation_used: bool = False
    word_count: int = 0
    char_count: int = 0


@dataclass
class EscalationStats:
    """Statistics for anti-bot escalation system."""

    total_attempts: int = 0
    successful_crawls: int = 0
    failed_crawls: int = 0
    escalations_triggered: int = 0
    level_success_rates: dict[int, float] = field(default_factory=dict)
    avg_attempts_per_url: float = 0.0
    total_duration: float = 0.0


@dataclass
class DifficultSite:
    """Configuration for a difficult website domain."""

    domain: str
    level: int
    reason: str
    last_updated: str

    def __post_init__(self):
        """Validate the difficult site configuration."""
        if not 0 <= self.level <= 3:
            raise ValueError(f"Anti-bot level must be 0-3, got {self.level}")
        if not self.domain:
            raise ValueError("Domain cannot be empty")


class DifficultSitesManager:
    """Manages difficult website domains and their anti-bot levels."""

    def __init__(self, config_file: str | None = None):
        """Initialize the difficult sites manager.

        Args:
            config_file: Path to the difficult sites JSON config file.
                        If None, uses default location in utils directory.
        """
        if config_file is None:
            # Default location in the utils directory
            self.config_file = Path(__file__).parent / "difficult_sites.json"
        else:
            self.config_file = Path(config_file)

        self._difficult_sites: dict[str, DifficultSite] = {}
        self._config_metadata: dict[str, Any] = {}
        self._load_difficult_sites()

    def _load_difficult_sites(self):
        """Load difficult sites from the JSON configuration file."""
        try:
            if not self.config_file.exists():
                logger.warning(f"Difficult sites config file not found: {self.config_file}")
                return

            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Load difficult sites
            if "difficult_sites" in data:
                for domain, site_data in data["difficult_sites"].items():
                    try:
                        site = DifficultSite(
                            domain=domain,
                            level=site_data["level"],
                            reason=site_data["reason"],
                            last_updated=site_data["last_updated"]
                        )
                        self._difficult_sites[domain.lower()] = site
                    except (KeyError, ValueError) as e:
                        logger.warning(f"Invalid difficult site entry for {domain}: {e}")

            # Load metadata
            self._config_metadata = data.get("metadata", {})

            logger.info(f"Loaded {len(self._difficult_sites)} difficult sites from {self.config_file}")

        except Exception as e:
            logger.error(f"Failed to load difficult sites config: {e}")
            self._difficult_sites = {}
            self._config_metadata = {}

    def get_difficult_site(self, domain: str) -> DifficultSite | None:
        """Get difficult site configuration for a domain.

        Args:
            domain: The domain to look up (case-insensitive)

        Returns:
            DifficultSite configuration if found, None otherwise
        """
        return self._difficult_sites.get(domain.lower())

    def is_difficult_site(self, domain: str) -> bool:
        """Check if a domain is configured as a difficult site.

        Args:
            domain: The domain to check (case-insensitive)

        Returns:
            True if domain is a difficult site, False otherwise
        """
        return domain.lower() in self._difficult_sites

    def add_difficult_site(self, domain: str, level: int, reason: str) -> bool:
        """Add a new difficult site to the configuration.

        Args:
            domain: The domain to add
            level: Anti-bot level (0-3)
            reason: Reason for the difficulty level

        Returns:
            True if added successfully, False otherwise
        """
        try:
            site = DifficultSite(
                domain=domain.lower(),
                level=level,
                reason=reason,
                last_updated=datetime.now().isoformat()
            )

            self._difficult_sites[domain.lower()] = site
            logger.info(f"Added difficult site: {domain.lower()} (level {level}: {reason})")
            return True

        except ValueError as e:
            logger.error(f"Failed to add difficult site {domain}: {e}")
            return False

    def remove_difficult_site(self, domain: str) -> bool:
        """Remove a difficult site from the configuration.

        Args:
            domain: The domain to remove

        Returns:
            True if removed successfully, False otherwise
        """
        domain_lower = domain.lower()
        if domain_lower in self._difficult_sites:
            del self._difficult_sites[domain_lower]
            logger.info(f"Removed difficult site: {domain_lower}")
            return True
        else:
            logger.warning(f"Difficult site not found for removal: {domain}")
            return False

    def get_all_difficult_sites(self) -> dict[str, DifficultSite]:
        """Get all difficult site configurations.

        Returns:
            Dictionary mapping domains to DifficultSite objects
        """
        return self._difficult_sites.copy()

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about difficult sites configuration.

        Returns:
            Dictionary with configuration statistics
        """
        level_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for site in self._difficult_sites.values():
            level_counts[site.level] += 1

        return {
            "total_sites": len(self._difficult_sites),
            "level_distribution": level_counts,
            "config_file": str(self.config_file),
            "last_loaded": datetime.now().isoformat(),
            "metadata": self._config_metadata
        }


class AntiBotEscalationManager:
    """
    Progressive anti-bot escalation system with smart retry logic.

    Features:
    - 4-level escalation with automatic progression
    - Success rate tracking and optimization
    - Smart retry with exponential backoff
    - Domain-specific escalation patterns
    - Performance monitoring and analytics
    - Difficult sites database for optimized starting levels
    """

    def __init__(self):
        """Initialize the anti-bot escalation manager."""
        self.stats = EscalationStats()
        self.domain_success_history: dict[str, list[bool]] = {}
        self.escalation_thresholds = {
            0: 0.7,  # Start escalation at 70% failure rate
            1: 0.5,  # Escalate to level 2 at 50% failure rate
            2: 0.3,  # Escalate to level 3 at 30% failure rate
        }
        self.max_attempts_per_url = 4
        self.base_delay = 1.0  # Base delay in seconds
        self.max_delay = 30.0  # Maximum delay in seconds

        # Initialize difficult sites manager
        self.difficult_sites_manager = DifficultSitesManager()

        # Statistics for difficult sites usage
        self.difficult_sites_hits: dict[str, int] = {}

        # Learning and auto-detection settings
        self.enable_auto_learning = os.getenv("ANTI_BOT_AUTO_LEARNING", "true").lower() == "true"
        self.min_escalations_for_learning = int(os.getenv("ANTI_BOT_MIN_ESCALATIONS", "3"))
        self.escalation_patterns: dict[str, list[int]] = {}  # Track escalation patterns per domain

    @async_timed(metadata={"category": "scraping", "stage": "anti_bot"})
    async def crawl_with_escalation(
        self,
        url: str,
        initial_level: int = 0,
        max_level: int = 3,
        use_content_filter: bool = False,
        session_id: str = "default",
    ) -> EscalationResult:
        """
        Crawl URL with progressive anti-bot escalation.

        Args:
            url: URL to crawl
            initial_level: Starting anti-bot level (0-3)
            max_level: Maximum escalation level (0-3)
            use_content_filter: Apply content filtering
            session_id: Session identifier for tracking

        Returns:
            EscalationResult with comprehensive crawl information
        """
        start_time = datetime.now()
        attempts_made = 0
        escalation_used = False
        final_level = initial_level

        try:
            # Extract domain for tracking
            from urllib.parse import urlparse

            domain = urlparse(url).netloc.lower()

            # Determine optimal starting level based on domain history
            optimal_start = self._get_optimal_start_level(domain, initial_level)
            current_level = optimal_start

            logger.info(f"Starting crawl escalation for {url} at level {current_level}")

            for attempt in range(self.max_attempts_per_url):
                attempts_made += 1
                final_level = current_level

                # Apply delay between attempts (with jitter)
                if attempt > 0:
                    delay = min(
                        self.base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1),
                        self.max_delay,
                    )
                    await asyncio.sleep(delay)

                # Attempt crawl at current level
                crawl_success, crawl_content = await self._crawl_at_level(
                    url, current_level, use_content_filter, session_id
                )

                if crawl_success:
                    # Record successful attempt
                    self._record_attempt(domain, True, current_level)

                    # Calculate metrics
                    duration = (datetime.now() - start_time).total_seconds()
                    word_count = len(crawl_content.split()) if crawl_content else 0
                    char_count = len(crawl_content) if crawl_content else 0

                    logger.debug(
                        f"Successfully crawled {url}: {char_count} chars in {duration:.1f}s (level {current_level})"
                    )

                    return EscalationResult(
                        url=url,
                        success=True,
                        content=crawl_content,
                        duration=duration,
                        attempts_made=attempts_made,
                        final_level=final_level,
                        escalation_used=escalation_used,
                        word_count=word_count,
                        char_count=char_count,
                    )
                else:
                    # Record failed attempt
                    self._record_attempt(domain, False, current_level)

                    # Check if we should escalate
                    if current_level < max_level and self._should_escalate(
                        domain, current_level, attempt
                    ):
                        current_level += 1
                        escalation_used = True
                        logger.info(f"Escalating to level {current_level} for {url}")

                        # Track escalation pattern for learning
                        if self.enable_auto_learning:
                            self._track_escalation_pattern(domain, current_level)
                    else:
                        # Try again at same level if not at max
                        if attempt < self.max_attempts_per_url - 1:
                            logger.debug(f"Retrying {url} at level {current_level}")

            # All attempts failed
            duration = (datetime.now() - start_time).total_seconds()
            self._record_attempt(domain, False, final_level)

            # Distinguish between expected crawling failures and system errors
            error_msg = f"All {attempts_made} attempts failed across levels {initial_level}-{final_level}"
            logger.debug(f"Crawling failed for {url}: {error_msg}")

            return EscalationResult(
                url=url,
                success=False,
                error=error_msg,
                duration=duration,
                attempts_made=attempts_made,
                final_level=final_level,
                escalation_used=escalation_used,
            )

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            # Distinguish between system errors (real bugs) and crawling failures (expected)
            error_msg = f"System error during crawl: {str(e)}"
            logger.error(f"❌ SYSTEM ERROR in crawling infrastructure: {error_msg}")
            logger.debug(f"Full error details for {url}: {e}")

            return EscalationResult(
                url=url,
                success=False,
                error=error_msg,
                duration=duration,
                attempts_made=attempts_made,
                final_level=final_level,
                escalation_used=escalation_used,
            )

    async def crawl_multiple_with_escalation(
        self,
        urls: list[str],
        initial_level: int = 0,
        max_level: int = 3,
        max_concurrent: int = 5,
        use_content_filter: bool = False,
        session_id: str = "default",
    ) -> list[EscalationResult]:
        """
        Crawl multiple URLs with concurrent anti-bot escalation.

        Args:
            urls: List of URLs to crawl
            initial_level: Starting anti-bot level
            max_level: Maximum escalation level
            max_concurrent: Maximum concurrent crawls
            use_content_filter: Apply content filtering
            session_id: Session identifier

        Returns:
            List of EscalationResult objects
        """
        if not urls:
            return []

        logger.info(
            f"Starting batch crawl with escalation: {len(urls)} URLs, "
            f"level {initial_level}-{max_level}, max_concurrent={max_concurrent}"
        )

        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(max_concurrent)

        async def crawl_with_semaphore(url: str) -> EscalationResult:
            async with semaphore:
                return await self.crawl_with_escalation(
                    url, initial_level, max_level, use_content_filter, session_id
                )

        # Execute crawls concurrently
        start_time = datetime.now()
        tasks = [crawl_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions and calculate statistics
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(
                    EscalationResult(url=urls[i], success=False, error=str(result))
                )
            else:
                final_results.append(result)

        # Update global statistics
        self._update_global_stats(final_results, datetime.now() - start_time)

        # Log batch summary
        successful = sum(1 for r in final_results if r.success)
        escalations = sum(1 for r in final_results if r.escalation_used)
        avg_attempts = sum(r.attempts_made for r in final_results) / len(final_results)

        logger.info(
            f"Batch crawl completed: {successful}/{len(urls)} successful "
            f"({successful / len(urls):.1%}), {escalations} escalations, "
            f"{avg_attempts:.1f} avg attempts"
        )

        return final_results

    async def _crawl_at_level(
        self, url: str, level: int, use_content_filter: bool, session_id: str
    ) -> tuple[bool, str | None]:
        """Crawl URL at specific anti-bot level."""
        try:
            # Get configurations for level
            crawl_config = self._get_crawl_config(level, use_content_filter)
            browser_config = self._get_browser_config(level)

            # Perform crawl
            if browser_config:
                async with AsyncWebCrawler(config=browser_config) as crawler:
                    result = await crawler.arun(url, config=crawl_config)
            else:
                async with AsyncWebCrawler() as crawler:
                    result = await crawler.arun(url, config=crawl_config)

            if result.success and result.markdown:
                return True, result.markdown
            else:
                return False, result.error_message if not result.success else None

        except Exception as e:
            logger.debug(f"Level {level} crawl failed for {url}: {e}")
            return False, str(e)

    def _get_crawl_config(
        self, level: int, use_content_filter: bool
    ) -> CrawlerRunConfig:
        """Get crawl configuration for specific anti-bot level."""

        if level == AntiBotLevel.BASIC.value:
            # Level 0: Basic configuration
            config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                check_robots_txt=False,
                remove_overlay_elements=True,
            )

        elif level == AntiBotLevel.ENHANCED.value:
            # Level 1: Enhanced headers + JavaScript rendering
            config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                check_robots_txt=False,
                remove_overlay_elements=True,
                simulate_user=True,
                magic=True,
                wait_for="body",
                page_timeout=30000,
            )

        elif level == AntiBotLevel.ADVANCED.value:
            # Level 2: Advanced proxy rotation + browser automation
            config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                check_robots_txt=False,
                remove_overlay_elements=True,
                simulate_user=True,
                magic=True,
                wait_until="domcontentloaded",
                page_timeout=45000,
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            )

        else:  # STEALTH
            # Level 3: Stealth mode with full browser simulation
            config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                check_robots_txt=False,
                remove_overlay_elements=True,
                simulate_user=True,
                magic=True,
                wait_until="domcontentloaded",
                page_timeout=60000,
                delay_before_return_html=2.0,
                js_code=[
                    "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})",
                    "Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]})",
                    "Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']})",
                ],
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                css_selector="main, article, .content, .article-body, .post-content",
            )

        # Add content filtering if requested
        if use_content_filter:
            md_generator = DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(threshold=0.4)
            )
            config.markdown_generator = md_generator

        return config

    def _get_browser_config(self, level: int) -> BrowserConfig | None:
        """Get browser configuration for specific anti-bot level."""

        if level >= AntiBotLevel.ADVANCED.value:
            # Use enhanced browser for advanced and stealth levels
            return BrowserConfig(
                headless=True,
                browser_type="chromium",
                viewport_width=1920,
                viewport_height=1080,
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            )
        else:
            # Use default for basic and enhanced levels
            return None

    def _get_optimal_start_level(self, domain: str, default_level: int) -> int:
        """Determine optimal starting level based on difficult sites database and domain history."""

        # STEP 1: Check difficult sites database first (highest priority)
        difficult_site = self.difficult_sites_manager.get_difficult_site(domain)
        if difficult_site:
            # Track usage of difficult sites
            if domain not in self.difficult_sites_hits:
                self.difficult_sites_hits[domain] = 0
            self.difficult_sites_hits[domain] += 1

            logger.info(
                f"🎯 Using predefined anti-bot level {difficult_site.level} for difficult site '{domain}' "
                f"(reason: {difficult_site.reason}, hits: {self.difficult_sites_hits[domain]})"
            )
            return difficult_site.level

        # STEP 2: Fall back to domain success history if not in difficult sites
        if domain not in self.domain_success_history:
            return default_level

        history = self.domain_success_history[domain][-10:]  # Last 10 attempts
        if len(history) < 3:
            return default_level

        success_rate = sum(history) / len(history)

        # If domain has low success rate, start at higher level
        if success_rate < 0.3:
            logger.info(f"📊 Using history-based level {min(default_level + 2, 3)} for domain '{domain}' (success rate: {success_rate:.2f})")
            return min(default_level + 2, 3)
        elif success_rate < 0.6:
            logger.info(f"📊 Using history-based level {min(default_level + 1, 3)} for domain '{domain}' (success rate: {success_rate:.2f})")
            return min(default_level + 1, 3)
        else:
            logger.debug(f"📊 Using default level {default_level} for domain '{domain}' (success rate: {success_rate:.2f})")
            return default_level

    def _should_escalate(self, domain: str, current_level: int, attempt: int) -> bool:
        """Determine if escalation should be attempted."""
        # Always escalate after first failed attempt
        if attempt == 0:
            return True

        # Check domain-specific failure rate
        if domain in self.domain_success_history:
            history = self.domain_success_history[domain][-5:]  # Last 5 attempts
            if len(history) >= 2:
                failure_rate = 1 - (sum(history) / len(history))
                threshold = self.escalation_thresholds.get(current_level, 0.5)
                return failure_rate > threshold

        # Default escalation logic
        return attempt < 2  # Allow up to 2 attempts per level

    def _track_escalation_pattern(self, domain: str, escalated_level: int):
        """Track escalation patterns for automatic learning.

        Args:
            domain: The domain being tracked
            escalated_level: The level the system escalated to
        """
        if domain not in self.escalation_patterns:
            self.escalation_patterns[domain] = []

        self.escalation_patterns[domain].append(escalated_level)

        # Keep only last 10 escalations per domain
        if len(self.escalation_patterns[domain]) > 10:
            self.escalation_patterns[domain] = self.escalation_patterns[domain][-10:]

        # Check if we should automatically add this as a difficult site
        self._check_and_auto_add_difficult_site(domain)

    def _check_and_auto_add_difficult_site(self, domain: str):
        """Check if a domain should be automatically added as a difficult site.

        Args:
            domain: The domain to check
        """
        if not self.enable_auto_learning:
            return

        # Skip if already in difficult sites
        if self.difficult_sites_manager.is_difficult_site(domain):
            return

        # Check if we have enough escalation data
        if domain not in self.escalation_patterns:
            return

        escalations = self.escalation_patterns[domain]
        if len(escalations) < self.min_escalations_for_learning:
            return

        # Analyze escalation patterns
        avg_level = sum(escalations) / len(escalations)
        max_level = max(escalations)

        # Only add if consistently requiring higher levels
        if max_level >= 2 and avg_level >= 1.5:  # Consistently needs enhanced or higher
            # Determine appropriate level based on patterns
            if max_level == 3 and len([e for e in escalations if e == 3]) >= 2:
                recommended_level = 3  # Stealth mode
                reason = f"Auto-detected: consistently requires stealth mode (escalations: {escalations})"
            elif avg_level >= 2.0:
                recommended_level = 2  # Advanced
                reason = f"Auto-detected: consistently requires advanced anti-bot (escalations: {escalations})"
            else:
                recommended_level = 1  # Enhanced
                reason = f"Auto-detected: consistently requires enhanced headers (escalations: {escalations})"

            # Add to difficult sites
            success = self.difficult_sites_manager.add_difficult_site(domain, recommended_level, reason)
            if success:
                logger.info(
                    f"🧠 AUTO-LEARNED: Added domain '{domain}' as difficult site (level {recommended_level}) "
                    f"based on {len(escalations)} escalations: {escalations}"
                )

                # Clear escalation patterns for this domain since it's now in difficult sites
                if domain in self.escalation_patterns:
                    del self.escalation_patterns[domain]

    def get_learning_stats(self) -> dict[str, Any]:
        """Get statistics about the auto-learning system.

        Returns:
            Dictionary with learning statistics
        """
        domains_tracking = len(self.escalation_patterns)
        total_patterns = sum(len(patterns) for patterns in self.escalation_patterns.values())

        # Identify potential candidates for auto-addition
        candidates = []
        for domain, escalations in self.escalation_patterns.items():
            if len(escalations) >= self.min_escalations_for_learning:
                avg_level = sum(escalations) / len(escalations)
                max_level = max(escalations)
                if max_level >= 2 and avg_level >= 1.5:
                    candidates.append({
                        "domain": domain,
                        "escalations": escalations,
                        "avg_level": round(avg_level, 2),
                        "max_level": max_level,
                        "recommended_level": 3 if max_level == 3 and len([e for e in escalations if e == 3]) >= 2 else (2 if avg_level >= 2.0 else 1)
                    })

        return {
            "auto_learning_enabled": self.enable_auto_learning,
            "min_escalations_for_learning": self.min_escalations_for_learning,
            "domains_tracking": domains_tracking,
            "total_escalation_patterns": total_patterns,
            "potential_candidates": candidates,
            "patterns_by_domain": {
                domain: {
                    "escalations": patterns,
                    "count": len(patterns),
                    "avg_level": round(sum(patterns) / len(patterns), 2) if patterns else 0,
                    "max_level": max(patterns) if patterns else 0
                }
                for domain, patterns in self.escalation_patterns.items()
            }
        }

    def _record_attempt(self, domain: str, success: bool, level: int):
        """Record crawl attempt for tracking and optimization."""
        # Update domain history
        if domain not in self.domain_success_history:
            self.domain_success_history[domain] = []

        self.domain_success_history[domain].append(success)

        # Keep only last 20 attempts per domain
        if len(self.domain_success_history[domain]) > 20:
            self.domain_success_history[domain] = self.domain_success_history[domain][
                -20:
            ]

        # Update level-specific stats
        if level not in self.stats.level_success_rates:
            self.stats.level_success_rates[level] = []
        self.stats.level_success_rates[level].append(success)

    def _update_global_stats(
        self, results: list[EscalationResult], duration: timedelta
    ):
        """Update global escalation statistics."""
        self.stats.total_attempts += len(results)
        self.stats.successful_crawls += sum(1 for r in results if r.success)
        self.stats.failed_crawls += sum(1 for r in results if not r.success)
        self.stats.escalations_triggered += sum(1 for r in results if r.escalation_used)
        self.stats.total_duration += duration.total_seconds()
        self.stats.avg_attempts_per_url = (
            sum(r.attempts_made for r in results) / len(results) if results else 0
        )

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive escalation statistics."""
        # Calculate level-specific success rates
        level_rates = {}
        for level, attempts in self.stats.level_success_rates.items():
            if attempts:
                level_rates[f"level_{level}"] = sum(attempts) / len(attempts)

        return {
            "total_attempts": self.stats.total_attempts,
            "successful_crawls": self.stats.successful_crawls,
            "failed_crawls": self.stats.failed_crawls,
            "overall_success_rate": (
                self.stats.successful_crawls / self.stats.total_attempts
                if self.stats.total_attempts > 0
                else 0
            ),
            "escalations_triggered": self.stats.escalations_triggered,
            "escalation_rate": (
                self.stats.escalations_triggered / self.stats.total_attempts
                if self.stats.total_attempts > 0
                else 0
            ),
            "avg_attempts_per_url": self.stats.avg_attempts_per_url,
            "avg_duration_per_crawl": (
                self.stats.total_duration / self.stats.total_attempts
                if self.stats.total_attempts > 0
                else 0
            ),
            "level_success_rates": level_rates,
            "domains_tracked": len(self.domain_success_history),
            "difficult_sites_stats": {
                "configured_sites": len(self.difficult_sites_manager.get_all_difficult_sites()),
                "hits": dict(sorted(self.difficult_sites_hits.items(), key=lambda x: x[1], reverse=True)),
                "total_hits": sum(self.difficult_sites_hits.values()),
                "config_details": self.difficult_sites_manager.get_stats()
            },
            "learning_stats": self.get_learning_stats(),
        }

    def reset_stats(self):
        """Reset all escalation statistics."""
        self.stats = EscalationStats()
        self.domain_success_history.clear()
        self.difficult_sites_hits.clear()
        self.escalation_patterns.clear()
        logger.info("Reset all escalation statistics including difficult sites hits and learning patterns")


# Global escalation manager instance
_global_escalation_manager: AntiBotEscalationManager | None = None


def get_escalation_manager() -> AntiBotEscalationManager:
    """Get or create global escalation manager instance."""
    global _global_escalation_manager
    if _global_escalation_manager is None:
        _global_escalation_manager = AntiBotEscalationManager()
    return _global_escalation_manager


def extract_domain_from_url(url: str) -> str:
    """Extract domain from URL for difficult sites matching.

    Args:
        url: The URL to extract domain from

    Returns:
        Domain string in lowercase, or empty string if extraction fails
    """
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception as e:
        logger.warning(f"Failed to extract domain from URL '{url}': {e}")
        return ""


def is_url_difficult_site(url: str, escalation_manager: AntiBotEscalationManager | None = None) -> tuple[bool, DifficultSite | None]:
    """Check if a URL belongs to a difficult site and get the site configuration.

    Args:
        url: The URL to check
        escalation_manager: Optional escalation manager to use (creates one if None)

    Returns:
        Tuple of (is_difficult, site_config)
    """
    if escalation_manager is None:
        escalation_manager = get_escalation_manager()

    domain = extract_domain_from_url(url)
    if not domain:
        return False, None

    difficult_site = escalation_manager.difficult_sites_manager.get_difficult_site(domain)
    return (difficult_site is not None), difficult_site


def get_predefined_anti_bot_level(url: str, escalation_manager: AntiBotEscalationManager | None = None) -> int | None:
    """Get the predefined anti-bot level for a URL if it's a difficult site.

    Args:
        url: The URL to check
        escalation_manager: Optional escalation manager to use (creates one if None)

    Returns:
        Anti-bot level (0-3) if URL is a difficult site, None otherwise
    """
    is_difficult, difficult_site = is_url_difficult_site(url, escalation_manager)
    return difficult_site.level if is_difficult and difficult_site else None
