"""
Anti-Bot Escalation System - Phase 1.2 Implementation

This module provides a comprehensive 4-level progressive anti-bot escalation system
with domain learning, cooldown management, and performance monitoring.

Core Features:
- 4-level progressive escalation (Basic → Enhanced → Advanced → Stealth)
- Domain learning and reputation management
- Intelligent cooldown management
- Performance monitoring and optimization
- Integration with enhanced logging from Phase 1.1

Phase 1.2 Implementation: Core Anti-Bot Escalation System with 4 levels

Quick Start:
```python
from multi_agent_research_system.utils.anti_bot import crawl_with_anti_bot

# Simple usage
result = await crawl_with_anti_bot("https://example.com")
if result.success:
    print(f"Content: {result.content[:200]}...")
    print(f"Anti-bot level used: {result.final_level}")

# Advanced usage with configuration
system = get_anti_bot_system({
    'max_attempts_per_url': 5,
    'enable_domain_learning': True,
    'concurrent_limit': 3
})
result = await system.crawl_url("https://protected-site.com")
```
"""

import time
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urlparse


class AntiBotLevel(Enum):
    """Four-level anti-bot escalation system."""

    BASIC = 0      # Simple headers, 10s timeout, no JavaScript
    ENHANCED = 1   # Realistic headers, 30s timeout, JavaScript enabled
    ADVANCED = 2   # Advanced headers, 60s timeout, stealth mode, viewport
    STEALTH = 3    # Full stealth, 120s timeout, proxy rotation, human-like timing

    @classmethod
    def get_level_name(cls, level: int) -> str:
        """Get human-readable name for level."""
        level_names = {
            0: "Basic",
            1: "Enhanced",
            2: "Advanced",
            3: "Stealth"
        }
        return level_names.get(level, "Unknown")

    @classmethod
    def get_success_rate_estimate(cls, level: int) -> float:
        """Get estimated success rate for level."""
        rates = {
            0: 0.6,   # 6/10 sites success
            1: 0.8,   # 8/10 sites success
            2: 0.9,   # 9/10 sites success
            3: 0.95   # 9.5/10 sites success
        }
        return rates.get(level, 0.5)


class EscalationTrigger(Enum):
    """Reasons for anti-bot escalation."""

    INITIAL_FAILURE = "initial_failure"
    RATE_LIMIT = "rate_limit"
    BOT_DETECTION = "bot_detection"
    CAPTCHA_CHALLENGE = "captcha_challenge"
    ACCESS_DENIED = "access_denied"
    TIMEOUT = "timeout"
    CONSISTENT_FAILURES = "consistent_failures"
    DOMAIN_REPUTATION = "domain_reputation"


@dataclass
class AntiBotConfig:
    """Configuration for a specific anti-bot level."""

    # Core settings
    level: int
    headers: Dict[str, str]
    timeout: int
    javascript: bool
    stealth: bool
    wait_strategy: str

    # Advanced settings
    proxy_rotation: bool = False
    viewport: Optional[str] = None
    retry_delays: List[float] = field(default_factory=list)
    detection_markers: List[str] = field(default_factory=list)
    cool_down_period: int = 0

    # Human-like behavior settings
    user_agent_rotation: bool = False
    mouse_movements: bool = False
    keyboard_typing: bool = False
    scrolling_behavior: bool = False

    # Performance settings
    max_redirects: int = 5
    keep_alive: bool = True
    verify_ssl: bool = True

    def __post_init__(self):
        """Validate configuration after creation."""
        if not 0 <= self.level <= 3:
            raise ValueError(f"Anti-bot level must be 0-3, got {self.level}")

        if self.timeout <= 0:
            raise ValueError(f"Timeout must be positive, got {self.timeout}")

        if not self.wait_strategy:
            self.wait_strategy = "smart"


@dataclass
class EscalationResult:
    """Result of anti-bot escalation attempt."""

    url: str
    domain: str
    success: bool
    content: Optional[str] = None
    error: Optional[str] = None
    duration: float = 0.0
    attempts_made: int = 0
    final_level: int = 0
    escalation_used: bool = False
    escalation_triggers: List[EscalationTrigger] = field(default_factory=list)

    # Content metrics
    word_count: int = 0
    char_count: int = 0
    content_quality_score: Optional[float] = None

    # Performance metrics
    total_escalation_time: float = 0.0
    average_attempt_duration: float = 0.0

    # Detection info
    detection_markers_found: List[str] = field(default_factory=list)
    anti_bot_measures_detected: List[str] = field(default_factory=list)

    def calculate_success_rate_improvement(self, baseline_level: int) -> float:
        """Calculate improvement over baseline level."""
        if not self.success or self.final_level <= baseline_level:
            return 0.0

        baseline_rate = AntiBotLevel.get_success_rate_estimate(baseline_level)
        actual_rate = 1.0  # Since we succeeded

        return ((actual_rate - baseline_rate) / baseline_rate) * 100 if baseline_rate > 0 else 0.0


@dataclass
class EscalationStats:
    """Statistics for anti-bot escalation system."""

    # Core metrics
    total_attempts: int = 0
    successful_crawls: int = 0
    failed_crawls: int = 0
    escalations_triggered: int = 0

    # Level-specific metrics
    level_attempts: Dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0, 2: 0, 3: 0})
    level_successes: Dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0, 2: 0, 3: 0})
    level_success_rates: Dict[int, float] = field(default_factory=dict)

    # Performance metrics
    avg_attempts_per_url: float = 0.0
    total_duration: float = 0.0
    avg_escalation_time: float = 0.0

    # Trigger analysis
    escalation_triggers: Dict[EscalationTrigger, int] = field(default_factory=dict)

    # Domain metrics
    unique_domains_processed: int = 0
    domains_with_escalations: int = 0

    def calculate_success_rate(self) -> float:
        """Calculate overall success rate."""
        if self.total_attempts == 0:
            return 0.0
        return (self.successful_crawls / self.total_attempts) * 100

    def calculate_escalation_rate(self) -> float:
        """Calculate escalation rate."""
        if self.total_attempts == 0:
            return 0.0
        return (self.escalations_triggered / self.total_attempts) * 100

    def get_most_common_trigger(self) -> Optional[EscalationTrigger]:
        """Get most common escalation trigger."""
        if not self.escalation_triggers:
            return None
        return max(self.escalation_triggers.items(), key=lambda x: x[1])[0]

    def update_level_stats(self, level: int, success: bool):
        """Update level-specific statistics."""
        self.level_attempts[level] = self.level_attempts.get(level, 0) + 1
        if success:
            self.level_successes[level] = self.level_successes.get(level, 0) + 1

        # Calculate success rate for this level
        attempts = self.level_attempts.get(level, 0)
        successes = self.level_successes.get(level, 0)
        if attempts > 0:
            self.level_success_rates[level] = (successes / attempts) * 100


@dataclass
class DomainProfile:
    """Learned profile for a specific domain."""

    domain: str
    optimal_level: int
    last_updated: datetime
    success_rate: float = 0.0
    total_attempts: int = 0
    successful_attempts: int = 0

    # Learning data
    escalation_history: List[int] = field(default_factory=list)
    failure_patterns: List[EscalationTrigger] = field(default_factory=list)
    average_response_time: float = 0.0

    # Cooldown management
    last_attempt_time: Optional[datetime] = None
    cool_down_until: Optional[datetime] = None
    consecutive_failures: int = 0

    # Reputation metrics
    domain_reputation_score: float = 0.5  # 0-1, higher is better
    detection_sophistication: int = 0    # 0-3, estimated detection level

    def update_attempt(self, success: bool, level: int, response_time: float,
                      triggers: List[EscalationTrigger] = None):
        """Update domain profile with new attempt data."""
        self.total_attempts += 1
        self.last_attempt_time = datetime.now()

        if success:
            self.successful_attempts += 1
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            if triggers:
                self.failure_patterns.extend(triggers)

        # Update success rate
        self.success_rate = (self.successful_attempts / self.total_attempts) * 100

        # Update escalation history
        self.escalation_history.append(level)
        if len(self.escalation_history) > 20:  # Keep last 20 attempts
            self.escalation_history = self.escalation_history[-20:]

        # Update average response time
        if self.total_attempts == 1:
            self.average_response_time = response_time
        else:
            self.average_response_time = (
                (self.average_response_time * (self.total_attempts - 1) + response_time) /
                self.total_attempts
            )

        # Update optimal level based on success patterns
        self._recalculate_optimal_level()

        # Update domain reputation
        self._update_reputation_score()

        # Update detection sophistication
        self._update_detection_sophistication()

        self.last_updated = datetime.now()

    def _recalculate_optimal_level(self):
        """Recalculate optimal anti-bot level based on history."""
        if not self.escalation_history:
            return

        # Analyze recent success rates by level
        level_success_rates = {0: [], 1: [], 2: [], 3: []}

        # Pair up levels with success indicators (simplified)
        recent_history = self.escalation_history[-10:]  # Last 10 attempts
        for i, level in enumerate(recent_history):
            # Approximate success based on consecutive failures pattern
            success_estimate = 1.0 if self.consecutive_failures == 0 else 0.3
            level_success_rates[level].append(success_estimate)

        # Find level with best success rate
        best_level = 1  # Default to enhanced
        best_success_rate = 0.0

        for level, successes in level_success_rates.items():
            if successes:
                avg_success = sum(successes) / len(successes)
                if avg_success > best_success_rate:
                    best_success_rate = avg_success
                    best_level = level

        # Only increase optimal level if current level is performing poorly
        current_success_rate = level_success_rates.get(self.optimal_level, [])
        if current_success_rate:
            current_avg = sum(current_success_rate) / len(current_success_rate)
            if current_avg < 0.6 and best_level > self.optimal_level:
                self.optimal_level = best_level

    def _update_reputation_score(self):
        """Update domain reputation score based on behavior."""
        # Base score on success rate and consistency
        success_factor = self.success_rate / 100

        # Factor in response time (faster is better)
        time_factor = max(0, 1 - (self.average_response_time / 30))  # 30s as baseline

        # Factor in failure patterns (fewer patterns is better)
        pattern_factor = max(0, 1 - (len(self.failure_patterns) / 10))

        # Combine factors
        self.domain_reputation_score = (success_factor * 0.5 +
                                     time_factor * 0.3 +
                                     pattern_factor * 0.2)

        # Clamp to valid range
        self.domain_reputation_score = max(0, min(1, self.domain_reputation_score))

    def _update_detection_sophistication(self):
        """Update estimated detection sophistication level."""
        if not self.failure_patterns:
            return

        # Analyze failure patterns to estimate detection sophistication
        trigger_scores = {
            EscalationTrigger.INITIAL_FAILURE: 0,
            EscalationTrigger.RATE_LIMIT: 1,
            EscalationTrigger.TIMEOUT: 1,
            EscalationTrigger.ACCESS_DENIED: 2,
            EscalationTrigger.CAPTCHA_CHALLENGE: 2,
            EscalationTrigger.BOT_DETECTION: 3
        }

        max_score = 0
        for trigger in self.failure_patterns[-5:]:  # Consider recent patterns
            score = trigger_scores.get(trigger, 0)
            max_score = max(max_score, score)

        self.detection_sophistication = max_score

    def is_in_cooldown(self) -> bool:
        """Check if domain is currently in cooldown period."""
        if not self.cool_down_until:
            return False
        return datetime.now() < self.cool_down_until

    def set_cooldown(self, duration_minutes: int):
        """Set cooldown period for domain."""
        self.cool_down_until = datetime.now() + timedelta(minutes=duration_minutes)

    def get_recommended_level(self) -> int:
        """Get recommended anti-bot level for this domain."""
        # Start with optimal level
        recommended = self.optimal_level

        # Increase level if in cooldown (recent failures)
        if self.is_in_cooldown():
            recommended = min(3, recommended + 1)

        # Increase level if consecutive failures
        if self.consecutive_failures >= 3:
            recommended = min(3, recommended + 1)

        # Consider detection sophistication
        if self.detection_sophistication > recommended:
            recommended = min(3, self.detection_sophistication)

        return min(3, max(0, recommended))

    def to_dict(self) -> Dict[str, Any]:
        """Convert domain profile to dictionary."""
        return {
            "domain": self.domain,
            "optimal_level": self.optimal_level,
            "last_updated": self.last_updated.isoformat(),
            "success_rate": self.success_rate,
            "total_attempts": self.total_attempts,
            "successful_attempts": self.successful_attempts,
            "average_response_time": self.average_response_time,
            "consecutive_failures": self.consecutive_failures,
            "domain_reputation_score": self.domain_reputation_score,
            "detection_sophistication": self.detection_sophistication,
            "cool_down_until": self.cool_down_until.isoformat() if self.cool_down_until else None,
            "escalation_history": self.escalation_history[-10:],  # Last 10
            "failure_patterns": [t.value for t in self.failure_patterns[-5:]]  # Last 5
        }


@dataclass
class CooldownEntry:
    """Cooldown entry for a domain."""

    domain: str
    cooldown_until: datetime
    reason: str
    original_level: int
    escalated_level: int

    def is_active(self) -> bool:
        """Check if cooldown is still active."""
        return datetime.now() < self.cooldown_until

    def remaining_seconds(self) -> float:
        """Get remaining cooldown seconds."""
        if not self.is_active():
            return 0.0
        return (self.cooldown_until - datetime.now()).total_seconds()


def extract_domain_from_url(url: str) -> str:
    """Extract domain from URL.

    Args:
        url: URL to extract domain from

    Returns:
        Domain string in lowercase, or empty string if extraction fails
    """
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception:
        return ""


def detect_escalation_triggers(error_message: str, status_code: Optional[int] = None) -> List[EscalationTrigger]:
    """Detect escalation triggers from error message and status code.

    Args:
        error_message: Error message to analyze
        status_code: HTTP status code

    Returns:
        List of detected escalation triggers
    """
    triggers = []
    error_lower = (error_message or "").lower()

    # Check status code triggers
    if status_code:
        if status_code == 429:
            triggers.append(EscalationTrigger.RATE_LIMIT)
        elif status_code in [401, 403]:
            triggers.append(EscalationTrigger.ACCESS_DENIED)
        elif status_code >= 500:
            triggers.append(EscalationTrigger.TIMEOUT)

    # Check error message triggers
    if "captcha" in error_lower or "challenge" in error_lower:
        triggers.append(EscalationTrigger.CAPTCHA_CHALLENGE)
    elif "bot" in error_lower or "automated" in error_lower or "suspicious" in error_lower:
        triggers.append(EscalationTrigger.BOT_DETECTION)
    elif "timeout" in error_lower or "timed out" in error_lower:
        triggers.append(EscalationTrigger.TIMEOUT)
    elif "rate limit" in error_lower or "too many requests" in error_lower:
        triggers.append(EscalationTrigger.RATE_LIMIT)
    elif "access denied" in error_lower or "forbidden" in error_lower:
        triggers.append(EscalationTrigger.ACCESS_DENIED)

    # Default trigger if no specific ones found
    if not triggers:
        triggers.append(EscalationTrigger.INITIAL_FAILURE)

    return triggers


def generate_realistic_user_agent() -> str:
    """Generate realistic user agent string.

    Returns:
        Random realistic user agent string
    """
    user_agents = [
        # Chrome
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',

        # Firefox
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0',

        # Safari
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1.2 Safari/605.1.15',
        'Mozilla/5.0 (iPhone; CPU iPhone OS 17_1_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1.2 Mobile/15E148 Safari/604.1',

        # Edge
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0'
    ]

    return random.choice(user_agents)


def calculate_escalation_delay(level: int, attempt: int, base_delay: float = 1.0) -> float:
    """Calculate delay for escalation attempt.

    Args:
        level: Current anti-bot level
        attempt: Attempt number at this level
        base_delay: Base delay in seconds

    Returns:
        Calculated delay in seconds
    """
    # Exponential backoff with jitter
    level_multipliers = {
        0: 1.0,   # Basic - no multiplier
        1: 1.5,   # Enhanced - 1.5x delay
        2: 2.0,   # Advanced - 2x delay
        3: 3.0    # Stealth - 3x delay
    }

    multiplier = level_multipliers.get(level, 1.0)
    exponential_delay = base_delay * (2 ** attempt) * multiplier

    # Add jitter (±25%)
    jitter = exponential_delay * 0.25 * (random.random() - 0.5) * 2

    return max(0, exponential_delay + jitter)


def validate_anti_bot_level(level: int) -> int:
    """Validate and clamp anti-bot level to valid range.

    Args:
        level: Level to validate

    Returns:
        Validated level (0-3)
    """
    return max(0, min(3, level))


# Public API exports
from .main import (
    AntiBotEscalationSystem,
    crawl_with_anti_bot,
    crawl_urls_with_anti_bot,
    get_anti_bot_system,
    initialize_anti_bot_system
)

from .config import (
    get_anti_bot_config,
    get_anti_bot_config_manager,
    configure_anti_bot_system
)

from .monitoring import (
    get_anti_bot_monitor,
    get_performance_dashboard,
    get_optimization_recommendations
)

# Export key classes and functions for easy access
__all__ = [
    # Main system
    'AntiBotEscalationSystem',
    'crawl_with_anti_bot',
    'crawl_urls_with_anti_bot',
    'get_anti_bot_system',
    'initialize_anti_bot_system',

    # Configuration
    'get_anti_bot_config',
    'get_anti_bot_config_manager',
    'configure_anti_bot_system',

    # Monitoring
    'get_anti_bot_monitor',
    'get_performance_dashboard',
    'get_optimization_recommendations',

    # Core components
    'AntiBotLevel',
    'AntiBotConfig',
    'EscalationResult',
    'DomainProfile',
    'EscalationTrigger',
    'EscalationStats',

    # Utilities
    'extract_domain_from_url',
    'detect_escalation_triggers',
    'generate_realistic_user_agent',
    'calculate_escalation_delay',
    'validate_anti_bot_level'
]