"""
URL Tracker for Multi-Agent Research System

This module provides URL deduplication and retry management functionality
to prevent duplicate crawling and implement progressive retry logic.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class URLAttempt:
    """Tracks a single URL crawl attempt."""
    url: str
    timestamp: datetime
    anti_bot_level: int
    success: bool
    content_length: int
    duration: float
    error_message: str | None = None
    session_id: str | None = None


@dataclass
class URLRecord:
    """Complete record for a URL including all attempts."""
    url: str
    domain: str
    first_seen: datetime
    last_attempt: datetime
    attempts: list[URLAttempt]
    final_status: str  # 'success', 'failed', 'pending'
    total_attempts: int
    successful_extractions: int
    retry_attempts: int = 0  # Number of retry attempts (beyond first attempt)
    retry_successes: int = 0  # Number of successful retry attempts
    best_content_length: int = 0  # Best content length achieved
    best_anti_bot_level: int = 0  # Anti-bot level that achieved best result

    @property
    def is_successful(self) -> bool:
        """Check if URL has been successfully crawled."""
        return any(attempt.success for attempt in self.attempts)

    @property
    def last_successful_attempt(self) -> URLAttempt | None:
        """Get the last successful attempt."""
        successful_attempts = [attempt for attempt in self.attempts if attempt.success]
        return successful_attempts[-1] if successful_attempts else None

    @property
    def should_retry(self) -> bool:
        """Determine if URL should be retried."""
        if self.is_successful:
            return False

        # Check if we've exceeded max retry attempts
        max_retries = 3  # Could be configurable
        if self.total_attempts >= max_retries:
            return False

        # Check if enough time has passed since last attempt
        time_since_last = datetime.now() - self.last_attempt
        min_retry_interval = timedelta(minutes=5)  # Could be configurable

        return time_since_last > min_retry_interval

    def get_next_anti_bot_level(self) -> int:
        """Get recommended anti-bot level for next attempt."""
        if not self.attempts:
            return 1  # Start with enhanced

        # Progressive anti-bot escalation
        last_level = max(attempt.anti_bot_level for attempt in self.attempts)
        return min(last_level + 1, 3)  # Cap at stealth level

    @property
    def retry_effectiveness(self) -> float:
        """Calculate retry effectiveness as percentage of retry attempts that succeeded."""
        if self.retry_attempts == 0:
            return 0.0
        return (self.retry_successes / self.retry_attempts) * 100

    @property
    def improvement_from_retries(self) -> bool:
        """Check if retries improved the result beyond first attempt."""
        if len(self.attempts) <= 1:
            return False

        first_attempt = self.attempts[0]
        best_later_attempt = max(self.attempts[1:], key=lambda x: x.content_length)

        return best_later_attempt.content_length > first_attempt.content_length

    @property
    def retry_improvement_ratio(self) -> float:
        """Calculate how much better the best result is compared to the first attempt."""
        if len(self.attempts) <= 1:
            return 0.0

        first_length = self.attempts[0].content_length
        if first_length == 0:
            return float('inf') if self.best_content_length > 0 else 0.0

        return (self.best_content_length - first_length) / first_length * 100


class URLTracker:
    """
    URL deduplication and retry management system.

    Prevents duplicate crawling across sessions and implements
    progressive retry logic with anti-bot level escalation.
    """

    def __init__(self, storage_dir: Path = None):
        """
        Initialize URL tracker.

        Args:
            storage_dir: Directory to store URL tracking data
        """
        if storage_dir is None:
            storage_dir = Path.home() / "lrepos" / "claude-agent-sdk-python" / "KEVIN" / "url_tracking"

        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.url_records: dict[str, URLRecord] = {}
        self.session_urls: set[str] = set()

        # Load existing tracking data
        self._load_tracking_data()

        logger.info(f"URLTracker initialized with {len(self.url_records)} existing URLs")

    def _load_tracking_data(self):
        """Load existing URL tracking data from storage."""
        try:
            tracking_file = self.storage_dir / "url_tracking.json"
            if tracking_file.exists():
                with open(tracking_file, encoding='utf-8') as f:
                    data = json.load(f)

                for url_data in data.get('url_records', []):
                    url_record = self._deserialize_url_record(url_data)
                    self.url_records[url_record.url] = url_record

                logger.info(f"Loaded {len(self.url_records)} URL records from storage")

        except Exception as e:
            logger.warning(f"Error loading URL tracking data: {e}")

    def _save_tracking_data(self):
        """Save URL tracking data to storage."""
        try:
            tracking_file = self.storage_dir / "url_tracking.json"

            data = {
                'last_updated': datetime.now().isoformat(),
                'total_urls': len(self.url_records),
                'url_records': [self._serialize_url_record(record) for record in self.url_records.values()]
            }

            with open(tracking_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error saving URL tracking data: {e}")

    def _serialize_url_record(self, record: URLRecord) -> dict[str, Any]:
        """Serialize URLRecord for JSON storage."""
        return {
            'url': record.url,
            'domain': record.domain,
            'first_seen': record.first_seen.isoformat(),
            'last_attempt': record.last_attempt.isoformat(),
            'attempts': [self._serialize_url_attempt(attempt) for attempt in record.attempts],
            'final_status': record.final_status,
            'total_attempts': record.total_attempts,
            'successful_extractions': record.successful_extractions
        }

    def _serialize_url_attempt(self, attempt: URLAttempt) -> dict[str, Any]:
        """Serialize URLAttempt for JSON storage."""
        return {
            'url': attempt.url,
            'timestamp': attempt.timestamp.isoformat(),
            'anti_bot_level': attempt.anti_bot_level,
            'success': attempt.success,
            'content_length': attempt.content_length,
            'duration': attempt.duration,
            'error_message': attempt.error_message,
            'session_id': attempt.session_id
        }

    def _deserialize_url_record(self, data: dict[str, Any]) -> URLRecord:
        """Deserialize URLRecord from JSON storage."""
        attempts = []
        for attempt_data in data.get('attempts', []):
            attempt_data['timestamp'] = datetime.fromisoformat(attempt_data['timestamp'])
            attempts.append(URLAttempt(**attempt_data))

        return URLRecord(
            url=data['url'],
            domain=data['domain'],
            first_seen=datetime.fromisoformat(data['first_seen']),
            last_attempt=datetime.fromisoformat(data['last_attempt']),
            attempts=attempts,
            final_status=data['final_status'],
            total_attempts=data['total_attempts'],
            successful_extractions=data['successful_extractions']
        )

    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except Exception:
            return "unknown"

    def filter_urls(self, urls: list[str], session_id: str = None) -> tuple[list[str], list[str]]:
        """
        Filter URLs to remove duplicates and already successful URLs.

        Args:
            urls: List of URLs to filter
            session_id: Current session ID for tracking

        Returns:
            Tuple of (urls_to_crawl, skipped_urls)
        """
        urls_to_crawl = []
        skipped_urls = []

        for url in urls:
            # Check if URL is already successful
            if url in self.url_records and self.url_records[url].is_successful:
                skipped_urls.append(url)
                logger.debug(f"Skipping already successful URL: {url}")
                continue

            # Check if URL was already attempted in current session
            if url in self.session_urls:
                skipped_urls.append(url)
                logger.debug(f"Skipping URL already attempted in session: {url}")
                continue

            # Add to crawl list
            urls_to_crawl.append(url)

            # Track as session URL
            self.session_urls.add(url)

        logger.info(f"URL filtering: {len(urls_to_crawl)} to crawl, {len(skipped_urls)} skipped")
        return urls_to_crawl, skipped_urls

    def record_attempt(
        self,
        url: str,
        success: bool,
        anti_bot_level: int,
        content_length: int,
        duration: float,
        error_message: str | None = None,
        session_id: str | None = None
    ):
        """
        Record a crawl attempt for a URL.

        Args:
            url: The URL that was crawled
            success: Whether the crawl was successful
            anti_bot_level: Anti-bot level used
            content_length: Length of content extracted
            duration: Duration of crawl attempt
            error_message: Error message if failed
            session_id: Session identifier
        """
        timestamp = datetime.now()

        # Create attempt record
        attempt = URLAttempt(
            url=url,
            timestamp=timestamp,
            anti_bot_level=anti_bot_level,
            success=success,
            content_length=content_length,
            duration=duration,
            error_message=error_message,
            session_id=session_id
        )

        # Get or create URL record
        if url not in self.url_records:
            self.url_records[url] = URLRecord(
                url=url,
                domain=self._get_domain(url),
                first_seen=timestamp,
                last_attempt=timestamp,
                attempts=[attempt],
                final_status='success' if success else 'failed',
                total_attempts=1,
                successful_extractions=1 if success else 0,
                best_content_length=content_length,
                best_anti_bot_level=anti_bot_level
            )
        else:
            # Update existing record
            record = self.url_records[url]
            record.attempts.append(attempt)
            record.last_attempt=timestamp
            record.total_attempts += 1

            # Track retry statistics (this is a retry if total_attempts > 1)
            if record.total_attempts > 1:
                record.retry_attempts += 1
                if success:
                    record.retry_successes += 1

            # Update best content and anti-bot level if this attempt is better
            if content_length > record.best_content_length:
                record.best_content_length = content_length
                record.best_anti_bot_level = anti_bot_level

            if success:
                record.successful_extractions += 1
                record.final_status = 'success'
            else:
                # Determine final status based on retry eligibility
                record.final_status = 'pending' if record.should_retry else 'failed'

        # Save updated tracking data (async if loop available, sync otherwise)
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            loop.create_task(self._save_tracking_data_async())
        except (ImportError, RuntimeError):
            # No async loop running, use synchronous fallback
            self._save_tracking_data()

        logger.info(f"Recorded crawl attempt for {url}: {'SUCCESS' if success else 'FAILED'} "
                   f"(level: {anti_bot_level}, content: {content_length} chars)")

    def get_retry_candidates(self, urls: list[str]) -> list[str]:
        """
        Get URLs that should be retried with higher anti-bot levels.

        Args:
            urls: List of URLs to check for retry eligibility

        Returns:
            List of URLs that should be retried
        """
        retry_candidates = []

        for url in urls:
            if url in self.url_records:
                record = self.url_records[url]
                if record.should_retry:
                    retry_candidates.append(url)
                    logger.debug(f"URL eligible for retry: {url} "
                               f"(next level: {record.get_next_anti_bot_level()})")

        logger.info(f"Found {len(retry_candidates)} retry candidates")
        return retry_candidates

    def get_retry_anti_bot_level(self, url: str) -> int:
        """
        Get recommended anti-bot level for retrying a URL.

        Args:
            url: URL to get retry level for

        Returns:
            Recommended anti-bot level
        """
        if url not in self.url_records:
            return 1  # Default to enhanced

        return self.url_records[url].get_next_anti_bot_level()

    def get_statistics(self) -> dict[str, Any]:
        """Get URL tracking statistics."""
        total_urls = len(self.url_records)
        successful_urls = sum(1 for record in self.url_records.values() if record.is_successful)
        failed_urls = sum(1 for record in self.url_records.values()
                         if not record.is_successful and not record.should_retry)
        pending_urls = total_urls - successful_urls - failed_urls

        total_attempts = sum(record.total_attempts for record in self.url_records.values())
        total_successful_extractions = sum(record.successful_extractions
                                         for record in self.url_records.values())

        # Calculate retry statistics
        retry_stats = self._calculate_retry_statistics()

        return {
            'total_urls': total_urls,
            'successful_urls': successful_urls,
            'failed_urls': failed_urls,
            'pending_urls': pending_urls,
            'success_rate': successful_urls / total_urls if total_urls > 0 else 0,
            'total_attempts': total_attempts,
            'successful_extractions': total_successful_extractions,
            'extraction_success_rate': total_successful_extractions / total_attempts if total_attempts > 0 else 0,
            'session_urls': len(self.session_urls),
            **retry_stats  # Add all retry statistics
        }

    def _calculate_retry_statistics(self) -> dict[str, Any]:
        """Calculate comprehensive retry statistics."""
        total_urls = len(self.url_records)
        if total_urls == 0:
            return {
                'urls_with_retries': 0,
                'total_retry_attempts': 0,
                'successful_retry_attempts': 0,
                'retry_effectiveness_percent': 0.0,
                'urls_improved_by_retries': 0,
                'improvement_rate_percent': 0.0,
                'avg_retry_improvement_ratio': 0.0,
                'anti_bot_level_effectiveness': {0: {}, 1: {}, 2: {}, 3: {}}
            }

        urls_with_retries = 0
        total_retry_attempts = 0
        successful_retry_attempts = 0
        urls_improved_by_retries = 0
        improvement_ratios = []
        anti_bot_level_success = {0: 0, 1: 0, 2: 0, 3: 0}
        anti_bot_level_total = {0: 0, 1: 0, 2: 0, 3: 0}

        for record in self.url_records.values():
            if record.retry_attempts > 0:
                urls_with_retries += 1
                total_retry_attempts += record.retry_attempts
                successful_retry_attempts += record.retry_successes

                if record.improvement_from_retries:
                    urls_improved_by_retries += 1
                    improvement_ratios.append(record.retry_improvement_ratio)

            # Track anti-bot level effectiveness
            for attempt in record.attempts:
                anti_bot_level_total[attempt.anti_bot_level] += 1
                if attempt.success:
                    anti_bot_level_success[attempt.anti_bot_level] += 1

        # Calculate effectiveness metrics
        retry_effectiveness = (successful_retry_attempts / total_retry_attempts * 100) if total_retry_attempts > 0 else 0.0
        improvement_rate = (urls_improved_by_retries / urls_with_retries * 100) if urls_with_retries > 0 else 0.0
        avg_improvement_ratio = sum(improvement_ratios) / len(improvement_ratios) if improvement_ratios else 0.0

        # Calculate success rates by anti-bot level
        anti_bot_success_rates = {}
        for level in range(4):
            if anti_bot_level_total[level] > 0:
                anti_bot_success_rates[level] = {
                    'success_rate': round(anti_bot_level_success[level] / anti_bot_level_total[level] * 100, 2),
                    'total_attempts': anti_bot_level_total[level],
                    'successful_attempts': anti_bot_level_success[level]
                }
            else:
                anti_bot_success_rates[level] = {
                    'success_rate': 0.0,
                    'total_attempts': 0,
                    'successful_attempts': 0
                }

        return {
            'urls_with_retries': urls_with_retries,
            'total_retry_attempts': total_retry_attempts,
            'successful_retry_attempts': successful_retry_attempts,
            'retry_effectiveness_percent': round(retry_effectiveness, 2),
            'urls_improved_by_retries': urls_improved_by_retries,
            'improvement_rate_percent': round(improvement_rate, 2),
            'avg_retry_improvement_ratio': round(avg_improvement_ratio, 2),
            'anti_bot_level_effectiveness': anti_bot_success_rates
        }

    def cleanup_old_records(self, days_to_keep: int = 30):
        """Clean up old URL records to prevent storage bloat."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        old_urls = []
        for url, record in self.url_records.items():
            if record.last_attempt < cutoff_date and record.is_successful:
                old_urls.append(url)

        for url in old_urls:
            del self.url_records[url]

        if old_urls:
            self._save_tracking_data()
            logger.info(f"Cleaned up {len(old_urls)} old URL records")

    async def _save_tracking_data_async(self):
        """Save URL tracking data asynchronously to prevent blocking."""
        try:
            tracking_file = self.storage_dir / "url_tracking.json"

            data = {
                'last_updated': datetime.now().isoformat(),
                'total_urls': len(self.url_records),
                'url_records': [self._serialize_url_record(record) for record in self.url_records.values()]
            }

            # Use asyncio.to_thread to run file I/O in a separate thread
            await asyncio.to_thread(self._write_json_file, tracking_file, data)

        except Exception as e:
            logger.error(f"Error saving URL tracking data asynchronously: {e}")

    def _write_json_file(self, file_path: Path, data: dict[str, Any]):
        """Synchronous JSON file writer for use in separate thread."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# Global URL tracker instance
_url_tracker: URLTracker | None = None


def get_url_tracker(storage_dir: Path = None) -> URLTracker:
    """Get or create global URL tracker instance."""
    global _url_tracker
    if _url_tracker is None:
        _url_tracker = URLTracker(storage_dir)
    return _url_tracker


def reset_url_tracker():
    """Reset the global URL tracker (for testing)."""
    global _url_tracker
    _url_tracker = None


# Export functions
__all__ = [
    'URLTracker',
    'URLRecord',
    'URLAttempt',
    'get_url_tracker',
    'reset_url_tracker'
]
