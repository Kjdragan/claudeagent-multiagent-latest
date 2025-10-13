"""
Caching and Optimization System for Confidence Scoring

Phase 1.3.3: Implement caching and optimization for confidence scoring

This module provides intelligent caching and performance optimization for
the content cleaning and confidence scoring system.

Key Features:
- LRU caching with TTL support
- Content similarity-based caching
- Performance monitoring and optimization
- Cache warming and preloading
- Memory-efficient storage
- Cache analytics and statistics
"""

import asyncio
import hashlib
import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
import threading

from .fast_confidence_scorer import ConfidenceSignals

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with TTL and metadata."""

    data: Any
    timestamp: datetime = field(default_factory=datetime.now)
    ttl_seconds: int = 3600  # 1 hour default TTL
    access_count: int = 0
    size_bytes: int = 0

    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return datetime.now() > self.timestamp + timedelta(seconds=self.ttl_seconds)

    def touch(self):
        """Update access timestamp and count."""
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache performance statistics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    cache_size: int = 0
    memory_usage_bytes: int = 0
    hit_rate: float = 0.0
    avg_access_time_ms: float = 0.0

    def update_hit_rate(self):
        """Update hit rate calculation."""
        if self.total_requests > 0:
            self.hit_rate = self.hits / self.total_requests


class LRUCache:
    """
    Thread-safe LRU cache with TTL support.

    Provides efficient caching with automatic eviction and performance monitoring.
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats()

        logger.info(f"LRUCache initialized: max_size={max_size}, ttl={default_ttl}s")

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        start_time = time.time()

        with self._lock:
            self._stats.total_requests += 1

            if key not in self._cache:
                self._stats.misses += 1
                self._stats.update_hit_rate()
                return None

            entry = self._cache[key]

            # Check expiration
            if entry.is_expired():
                del self._cache[key]
                self._stats.misses += 1
                self._stats.update_hit_rate()
                logger.debug(f"Cache entry expired: {key}")
                return None

            # Update access
            entry.touch()
            self._cache.move_to_end(key)
            self._stats.hits += 1
            self._stats.update_hit_rate()

            # Update access time
            access_time_ms = (time.time() - start_time) * 1000
            self._update_avg_access_time(access_time_ms)

            logger.debug(f"Cache hit: {key} (access_count: {entry.access_count})")
            return entry.data

    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Put value into cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Custom TTL in seconds

        Returns:
            True if value was cached, False if evicted
        """
        ttl = ttl or self.default_ttl

        # Estimate size
        try:
            import sys
            size_bytes = sys.getsizeof(value)
        except:
            size_bytes = 0

        with self._lock:
            # Check if key exists (update)
            if key in self._cache:
                entry = CacheEntry(
                    data=value,
                    ttl_seconds=ttl,
                    size_bytes=size_bytes
                )
                self._cache[key] = entry
                self._cache.move_to_end(key)
                return True

            # Check if we need to evict
            while len(self._cache) >= self.max_size:
                evicted_key, evicted_entry = self._cache.popitem(last=False)
                self._stats.evictions += 1
                logger.debug(f"Cache eviction: {evicted_key}")

            # Add new entry
            entry = CacheEntry(
                data=value,
                ttl_seconds=ttl,
                size_bytes=size_bytes
            )
            self._cache[key] = entry
            self._stats.cache_size = len(self._cache)

            logger.debug(f"Cache put: {key} (size: {size_bytes} bytes)")
            return True

    def remove(self, key: str) -> bool:
        """
        Remove entry from cache.

        Args:
            key: Cache key

        Returns:
            True if entry was removed
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.cache_size = len(self._cache)
                logger.debug(f"Cache remove: {key}")
                return True
            return False

    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._stats.cache_size = 0
            logger.info("Cache cleared")

    def cleanup_expired(self) -> int:
        """
        Remove expired entries from cache.

        Returns:
            Number of expired entries removed
        """
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]

            for key in expired_keys:
                del self._cache[key]

            self._stats.cache_size = len(self._cache)

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

            return len(expired_keys)

    def get_stats(self) -> CacheStats:
        """Get current cache statistics."""
        with self._lock:
            # Update memory usage
            try:
                import sys
                self._stats.memory_usage_bytes = sum(
                    sys.getsizeof(entry) for entry in self._cache.values()
                )
            except:
                self._stats.memory_usage_bytes = 0

            return self._stats

    def _update_avg_access_time(self, access_time_ms: float):
        """Update average access time."""
        if self._stats.total_requests == 1:
            self._stats.avg_access_time_ms = access_time_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self._stats.avg_access_time_ms = (
                alpha * access_time_ms +
                (1 - alpha) * self._stats.avg_access_time_ms
            )


class ContentSimilarityCache:
    """
    Content similarity-based cache for confidence scoring.

    Provides caching based on content similarity rather than exact matches.
    """

    def __init__(self, similarity_threshold: float = 0.9, max_size: int = 500):
        """
        Initialize similarity cache.

        Args:
            similarity_threshold: Minimum similarity for cache match
            max_size: Maximum number of entries
        """
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self._cache: Dict[str, Tuple[str, ConfidenceSignals]] = {}  # hash -> (content_sample, signals)
        self._lock = threading.RLock()

        logger.info(f"ContentSimilarityCache initialized: threshold={similarity_threshold}, max_size={max_size}")

    def get_similar(self, content: str, content_sample_size: int = 1000) -> Optional[ConfidenceSignals]:
        """
        Get confidence signals for similar content.

        Args:
            content: Content to find similar entry for
            content_sample_size: Size of content sample for similarity comparison

        Returns:
            ConfidenceSignals if similar content found, None otherwise
        """
        content_sample = content[:content_sample_size]
        content_hash = self._generate_content_hash(content_sample)

        with self._lock:
            # Check exact hash match first
            if content_hash in self._cache:
                return self._cache[content_hash][1]

            # Check for similar content
            for stored_hash, (stored_sample, signals) in self._cache.items():
                similarity = self._calculate_similarity(content_sample, stored_sample)
                if similarity >= self.similarity_threshold:
                    logger.debug(f"Found similar content (similarity: {similarity:.3f})")
                    return signals

            return None

    def put(self, content: str, signals: ConfidenceSignals, content_sample_size: int = 1000):
        """
        Put content and signals into similarity cache.

        Args:
            content: Content to cache
            signals: Confidence signals to cache
            content_sample_size: Size of content sample for similarity
        """
        content_sample = content[:content_sample_size]
        content_hash = self._generate_content_hash(content_sample)

        with self._lock:
            # Check if we need to evict
            while len(self._cache) >= self.max_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

            self._cache[content_hash] = (content_sample, signals)
            logger.debug(f"Similarity cache put: {content_hash}")

    def clear(self):
        """Clear similarity cache."""
        with self._lock:
            self._cache.clear()
            logger.info("Similarity cache cleared")

    def _generate_content_hash(self, content: str) -> str:
        """Generate hash for content sample."""
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _calculate_similarity(self, content1: str, content2: str) -> float:
        """
        Calculate similarity between two content samples.

        Simple word-based similarity calculation.
        """
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)


class CachingOptimizer:
    """
    Comprehensive caching and optimization system for content cleaning.

    Coordinates multiple caching strategies and provides performance optimization.
    """

    def __init__(
        self,
        enable_lru_cache: bool = True,
        enable_similarity_cache: bool = True,
        lru_cache_size: int = 1000,
        similarity_cache_size: int = 500,
        cleanup_interval_seconds: int = 300  # 5 minutes
    ):
        """
        Initialize caching optimizer.

        Args:
            enable_lru_cache: Enable LRU caching
            enable_similarity_cache: Enable similarity-based caching
            lru_cache_size: LRU cache maximum size
            similarity_cache_size: Similarity cache maximum size
            cleanup_interval_seconds: Interval for cache cleanup
        """
        self.enable_lru_cache = enable_lru_cache
        self.enable_similarity_cache = enable_similarity_cache

        # Initialize caches
        if enable_lru_cache:
            self.lru_cache = LRUCache(max_size=lru_cache_size, default_ttl=3600)
        else:
            self.lru_cache = None

        if enable_similarity_cache:
            self.similarity_cache = ContentSimilarityCache(
                similarity_threshold=0.9,
                max_size=similarity_cache_size
            )
        else:
            self.similarity_cache = None

        # Cleanup scheduling
        self.cleanup_interval = cleanup_interval_seconds
        self._cleanup_task = None
        self._running = False

        logger.info("CachingOptimizer initialized with LRU and similarity caching")

    async def start(self):
        """Start the caching optimizer (cleanup scheduler)."""
        if self._running:
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("CachingOptimizer started")

    async def stop(self):
        """Stop the caching optimizer."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("CachingOptimizer stopped")

    def get_cached_confidence_signals(
        self,
        content: str,
        url: str,
        search_query: Optional[str] = None
    ) -> Optional[ConfidenceSignals]:
        """
        Get cached confidence signals using multiple strategies.

        Args:
            content: Content to get cached assessment for
            url: Source URL
            search_query: Search query for context

        Returns:
            Cached ConfidenceSignals or None if not found
        """
        # Try LRU cache first (exact match)
        if self.lru_cache:
            cache_key = self._generate_cache_key(content, url, search_query)
            cached_signals = self.lru_cache.get(cache_key)
            if cached_signals:
                logger.debug(f"LRU cache hit for confidence assessment")
                return cached_signals

        # Try similarity cache
        if self.similarity_cache:
            similar_signals = self.similarity_cache.get_similar(content)
            if similar_signals:
                logger.debug(f"Similarity cache hit for confidence assessment")
                return similar_signals

        return None

    def cache_confidence_signals(
        self,
        content: str,
        url: str,
        search_query: Optional[str],
        signals: ConfidenceSignals,
        ttl: Optional[int] = None
    ):
        """
        Cache confidence signals using multiple strategies.

        Args:
            content: Content that was assessed
            url: Source URL
            search_query: Search query for context
            signals: Confidence signals to cache
            ttl: Custom TTL for LRU cache
        """
        # Cache in LRU cache
        if self.lru_cache:
            cache_key = self._generate_cache_key(content, url, search_query)
            self.lru_cache.put(cache_key, signals, ttl=ttl)

        # Cache in similarity cache
        if self.similarity_cache:
            self.similarity_cache.put(content, signals)

    def _generate_cache_key(self, content: str, url: str, search_query: Optional[str]) -> str:
        """Generate cache key for confidence assessment."""
        # Use content hash, URL hash, and query hash
        content_hash = hashlib.md5(content[:1000].encode()).hexdigest()[:8]
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        query_hash = hashlib.md5((search_query or "").encode()).hexdigest()[:8]

        return f"confidence_{content_hash}_{url_hash}_{query_hash}"

    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while self._running:
            try:
                await self._perform_cleanup()
                await asyncio.sleep(self.cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def _perform_cleanup(self):
        """Perform cache cleanup operations."""
        logger.debug("Performing cache cleanup")

        # Cleanup LRU cache
        if self.lru_cache:
            expired_count = self.lru_cache.cleanup_expired()
            if expired_count > 0:
                logger.info(f"Cleaned up {expired_count} expired LRU cache entries")

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive performance statistics.

        Returns:
            Dictionary with performance statistics
        """
        stats = {
            'optimizer_enabled': True,
            'lru_cache_enabled': self.enable_lru_cache,
            'similarity_cache_enabled': self.enable_similarity_cache,
            'cleanup_interval_seconds': self.cleanup_interval
        }

        # LRU cache stats
        if self.lru_cache:
            lru_stats = self.lru_cache.get_stats()
            stats['lru_cache'] = {
                'cache_size': lru_stats.cache_size,
                'hit_rate': lru_stats.hit_rate,
                'total_requests': lru_stats.total_requests,
                'hits': lru_stats.hits,
                'misses': lru_stats.misses,
                'evictions': lru_stats.evictions,
                'memory_usage_bytes': lru_stats.memory_usage_bytes,
                'avg_access_time_ms': lru_stats.avg_access_time_ms
            }

        # Similarity cache stats
        if self.similarity_cache:
            with self.similarity_cache._lock:
                stats['similarity_cache'] = {
                    'cache_size': len(self.similarity_cache._cache),
                    'similarity_threshold': self.similarity_cache.similarity_threshold
                }

        return stats

    def clear_all_caches(self):
        """Clear all caches."""
        if self.lru_cache:
            self.lru_cache.clear()

        if self.similarity_cache:
            self.similarity_cache.clear()

        logger.info("All caches cleared")

    def optimize_for_memory(self, target_memory_mb: int = 100) -> Dict[str, Any]:
        """
        Optimize caches for memory usage.

        Args:
            target_memory_mb: Target memory usage in MB

        Returns:
            Optimization results
        """
        target_memory_bytes = target_memory_mb * 1024 * 1024
        results = {
            'target_memory_mb': target_memory_mb,
            'actions_taken': []
        }

        current_memory = 0
        if self.lru_cache:
            lru_stats = self.lru_cache.get_stats()
            current_memory += lru_stats.memory_usage_bytes

        if current_memory > target_memory_bytes:
            # Need to reduce cache usage
            if self.lru_cache:
                # Reduce LRU cache size
                current_size = len(self.lru_cache._cache)
                target_size = int(current_size * (target_memory_bytes / current_memory))
                target_size = max(target_size, 100)  # Minimum size

                # Clear some entries
                with self.lru_cache._lock:
                    while len(self.lru_cache._cache) > target_size:
                        self.lru_cache._cache.popitem(last=False)
                        self.lru_cache._stats.evictions += 1

                results['actions_taken'].append(
                    f"Reduced LRU cache from {current_size} to {len(self.lru_cache._cache)} entries"
                )

        logger.info(f"Memory optimization completed: {results}")
        return results

    def warm_cache(self, sample_contents: List[Tuple[str, str, str]], confidence_scorer):
        """
        Warm up cache with sample content.

        Args:
            sample_contents: List of (content, url, search_query) tuples
            confidence_scorer: ConfidenceScorer instance for assessments
        """
        logger.info(f"Warming cache with {len(sample_contents)} sample contents")

        async def warm_single(content, url, query):
            try:
                signals = await confidence_scorer.assess_content_confidence(
                    content=content,
                    url=url,
                    search_query=query
                )
                self.cache_confidence_signals(content, url, query, signals)
            except Exception as e:
                logger.warning(f"Cache warming failed for sample: {e}")

        # Run warming concurrently (but limit concurrency)
        semaphore = asyncio.Semaphore(5)  # Limit concurrent warming

        async def warmed_warm_single(content, url, query):
            async with semaphore:
                await warm_single(content, url, query)

        # Create tasks
        tasks = [
            warmed_warm_single(content, url, query)
            for content, url, query in sample_contents
        ]

        # Execute warming
        asyncio.run(asyncio.gather(*tasks, return_exceptions=True))

        logger.info("Cache warming completed")