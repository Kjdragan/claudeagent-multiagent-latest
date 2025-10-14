"""
Message Cache and Optimization System - High-Performance Message Caching

This module provides sophisticated message caching and optimization capabilities with
intelligent cache management, performance optimization, and memory efficiency.

Key Features:
- Multi-level caching with LRU eviction policies
- Intelligent cache key generation and content hashing
- Performance optimization through compression and deduplication
- Cache statistics and monitoring
- TTL (Time To Live) management with automatic expiration
- Cache warming and preloading strategies
- Memory-efficient storage with compression
- Distributed cache support (for future scaling)
"""

import asyncio
import hashlib
import json
import gzip
import time
import logging
import pickle
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from typing import Callable
from datetime import datetime, timedelta
from enum import Enum
from collections import OrderedDict
import threading
import weakref

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from ..core.message_types import RichMessage, EnhancedMessageType


class CacheLevel(Enum):
    """Cache levels for hierarchical caching."""

    MEMORY = "memory"
    DISK = "disk"
    DISTRIBUTED = "distributed"


class EvictionPolicy(Enum):
    """Cache eviction policies."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    SIZE = "size"  # Size-based


@dataclass
class CacheEntry:
    """Cache entry with metadata and content."""

    key: str
    message: RichMessage
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None
    expires_at: Optional[datetime] = None
    compressed_data: Optional[bytes] = None
    compression_ratio: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize cache entry."""
        if self.size_bytes == 0:
            self.size_bytes = self._calculate_size()

        if self.ttl_seconds and not self.expires_at:
            self.expires_at = self.created_at + timedelta(seconds=self.ttl_seconds)

    def _calculate_size(self) -> int:
        """Calculate size of cache entry."""
        # Serialize message to estimate size
        try:
            serialized = json.dumps(self.message.to_dict())
            return len(serialized.encode('utf-8'))
        except:
            # Fallback to content length
            return len(self.message.content.encode('utf-8'))

    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.expires_at:
            return datetime.now() > self.expires_at
        return False

    def touch(self):
        """Update last accessed time and increment access count."""
        self.last_accessed = datetime.now()
        self.access_count += 1

    def get_age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return (datetime.now() - self.created_at).total_seconds()

    def get_idle_seconds(self) -> float:
        """Get idle time since last access."""
        return (datetime.now() - self.last_accessed).total_seconds()


@dataclass
class CacheConfig:
    """Configuration for message cache."""

    max_memory_entries: int = 1000
    max_memory_size_mb: int = 100
    default_ttl_seconds: int = 3600
    enable_compression: bool = True
    compression_threshold_bytes: int = 1024
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    enable_disk_cache: bool = False
    disk_cache_dir: str = "./cache"
    max_disk_size_mb: int = 500
    enable_distributed_cache: bool = False
    redis_url: str = "redis://localhost:6379/0"
    cache_key_prefix: str = "msg_cache"
    enable_statistics: bool = True
    background_cleanup_interval: int = 300
    max_key_length: int = 255


class MemoryCache:
    """In-memory cache implementation with LRU eviction."""

    def __init__(self, max_entries: int, max_size_mb: int, eviction_policy: EvictionPolicy):
        self.max_entries = max_entries
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.eviction_policy = eviction_policy
        self.logger = logging.getLogger(__name__)

        # Use OrderedDict for LRU functionality
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_size_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

        # Thread safety
        self.lock = threading.RLock()

    def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry by key."""
        with self.lock:
            entry = self.cache.get(key)
            if entry is None:
                self.misses += 1
                return None

            # Check expiration
            if entry.is_expired():
                self._remove_entry(key)
                self.misses += 1
                return None

            # Update access information for LRU
            if self.eviction_policy == EvictionPolicy.LRU:
                self.cache.move_to_end(key)  # Mark as recently used
            entry.touch()
            self.hits += 1

            return entry

    def put(self, key: str, entry: CacheEntry) -> bool:
        """Put cache entry with eviction if necessary."""
        with self.lock:
            # Check if entry already exists
            existing_entry = self.cache.get(key)
            if existing_entry:
                self.current_size_bytes -= existing_entry.size_bytes

            # Check if eviction is needed
            while (len(self.cache) >= self.max_entries or
                   self.current_size_bytes + entry.size_bytes > self.max_size_bytes):
                if not self._evict_one():
                    break  # Can't evict more
                self.evictions += 1

            # Add entry
            self.cache[key] = entry
            self.current_size_bytes += entry.size_bytes

            if self.eviction_policy == EvictionPolicy.LRU:
                self.cache.move_to_end(key)

            return True

    def remove(self, key: str) -> bool:
        """Remove cache entry by key."""
        with self.lock:
            entry = self.cache.pop(key, None)
            if entry:
                self.current_size_bytes -= entry.size_bytes
                return True
            return False

    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.current_size_bytes = 0

    def _evict_one(self) -> bool:
        """Evict one entry based on eviction policy."""
        if not self.cache:
            return False

        if self.eviction_policy == EvictionPolicy.LRU:
            # Remove oldest entry (first in OrderedDict)
            key, entry = self.cache.popitem(last=False)
        elif self.eviction_policy == EvictionPolicy.LFU:
            # Remove least frequently used entry
            key_to_remove = min(self.cache.keys(), key=lambda k: self.cache[k].access_count)
            entry = self.cache.pop(key_to_remove)
            key = key_to_remove
        elif self.eviction_policy == EvictionPolicy.TTL:
            # Remove expired entries first, then oldest
            expired_keys = [k for k, v in self.cache.items() if v.is_expired()]
            if expired_keys:
                key = expired_keys[0]
                entry = self.cache.pop(key)
            else:
                key, entry = self.cache.popitem(last=False)
        else:  # SIZE
            # Remove largest entry
            key_to_remove = max(self.cache.keys(), key=lambda k: self.cache[k].size_bytes)
            entry = self.cache.pop(key_to_remove)
            key = key_to_remove

        self.current_size_bytes -= entry.size_bytes
        return True

    def _remove_entry(self, key: str):
        """Remove entry and update size."""
        entry = self.cache.pop(key, None)
        if entry:
            self.current_size_bytes -= entry.size_bytes

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

            return {
                "entries": len(self.cache),
                "size_bytes": self.current_size_bytes,
                "size_mb": self.current_size_bytes / (1024 * 1024),
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "hit_rate": hit_rate,
                "utilization": len(self.cache) / self.max_entries if self.max_entries > 0 else 0.0,
                "memory_utilization": self.current_size_bytes / self.max_size_bytes if self.max_size_bytes > 0 else 0.0
            }


class MessageCache:
    """High-performance message cache with multi-level storage."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize message cache with configuration."""
        self.config = self._create_cache_config(config or {})
        self.logger = logging.getLogger(__name__)

        # Cache levels
        self.memory_cache = MemoryCache(
            self.config.max_memory_entries,
            self.config.max_memory_size_mb,
            self.config.eviction_policy
        )
        self.disk_cache = None  # Future implementation
        self.distributed_cache = None  # Future Redis implementation

        # Initialize distributed cache if available and enabled
        if self.config.enable_distributed_cache and REDIS_AVAILABLE:
            self._initialize_distributed_cache()

        # Background cleanup task
        self.cleanup_task = None
        self._running = False

        # Statistics
        self.cache_stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "compression_saves": 0,
            "total_bytes_saved": 0
        }

    def _create_cache_config(self, config: Dict[str, Any]) -> CacheConfig:
        """Create cache configuration from settings."""
        return CacheConfig(
            max_memory_entries=config.get("max_memory_entries", 1000),
            max_memory_size_mb=config.get("max_memory_size_mb", 100),
            default_ttl_seconds=config.get("default_ttl_seconds", 3600),
            enable_compression=config.get("enable_compression", True),
            compression_threshold_bytes=config.get("compression_threshold_bytes", 1024),
            eviction_policy=EvictionPolicy(config.get("eviction_policy", "lru")),
            enable_disk_cache=config.get("enable_disk_cache", False),
            disk_cache_dir=config.get("disk_cache_dir", "./cache"),
            max_disk_size_mb=config.get("max_disk_size_mb", 500),
            enable_distributed_cache=config.get("enable_distributed_cache", False),
            redis_url=config.get("redis_url", "redis://localhost:6379/0"),
            cache_key_prefix=config.get("cache_key_prefix", "msg_cache"),
            enable_statistics=config.get("enable_statistics", True),
            background_cleanup_interval=config.get("background_cleanup_interval", 300),
            max_key_length=config.get("max_key_length", 255)
        )

    def _initialize_distributed_cache(self):
        """Initialize Redis distributed cache."""
        try:
            import redis
            self.distributed_cache = redis.from_url(self.config.redis_url)
            # Test connection
            self.distributed_cache.ping()
            self.logger.info("Redis distributed cache initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Redis cache: {str(e)}")
            self.distributed_cache = None

    async def get(self, message: RichMessage) -> Optional[Dict[str, Any]]:
        """Get cached message data."""
        cache_key = self._generate_cache_key(message)

        self.cache_stats["total_requests"] += 1

        # Try memory cache first
        entry = self.memory_cache.get(cache_key)
        if entry:
            self.cache_stats["cache_hits"] += 1
            return self._deserialize_entry(entry)

        # Try distributed cache
        if self.distributed_cache:
            try:
                cached_data = self.distributed_cache.get(cache_key)
                if cached_data:
                    self.cache_stats["cache_hits"] += 1
                    entry = self._deserialize_from_bytes(cached_data)
                    # Store in memory cache for faster access
                    self.memory_cache.put(cache_key, entry)
                    return self._deserialize_entry(entry)
            except Exception as e:
                self.logger.warning(f"Distributed cache get failed: {str(e)}")

        self.cache_stats["cache_misses"] += 1
        return None

    async def set(self, message: RichMessage, data: Dict[str, Any], ttl_seconds: Optional[int] = None) -> bool:
        """Cache message data."""
        cache_key = self._generate_cache_key(message)

        # Use provided TTL or default
        if ttl_seconds is None:
            ttl_seconds = self.config.default_ttl_seconds

        # Create cache entry
        entry = CacheEntry(
            key=cache_key,
            message=message,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            ttl_seconds=ttl_seconds,
            metadata=data
        )

        # Apply compression if enabled and beneficial
        if self.config.enable_compression and entry.size_bytes > self.config.compression_threshold_bytes:
            entry = await self._compress_entry(entry)

        # Store in memory cache
        memory_success = self.memory_cache.put(cache_key, entry)

        # Store in distributed cache
        distributed_success = False
        if self.distributed_cache:
            try:
                serialized_data = self._serialize_to_bytes(entry)
                self.distributed_cache.setex(cache_key, ttl_seconds, serialized_data)
                distributed_success = True
            except Exception as e:
                self.logger.warning(f"Distributed cache set failed: {str(e)}")

        return memory_success or distributed_success

    async def delete(self, message: RichMessage) -> bool:
        """Delete cached message."""
        cache_key = self._generate_cache_key(message)

        # Delete from memory cache
        memory_success = self.memory_cache.remove(cache_key)

        # Delete from distributed cache
        distributed_success = False
        if self.distributed_cache:
            try:
                self.distributed_cache.delete(cache_key)
                distributed_success = True
            except Exception as e:
                self.logger.warning(f"Distributed cache delete failed: {str(e)}")

        return memory_success or distributed_success

    def clear(self):
        """Clear all cached messages."""
        self.memory_cache.clear()

        if self.distributed_cache:
            try:
                # Clear keys with our prefix
                pattern = f"{self.config.cache_key_prefix}:*"
                keys = self.distributed_cache.keys(pattern)
                if keys:
                    self.distributed_cache.delete(*keys)
            except Exception as e:
                self.logger.warning(f"Distributed cache clear failed: {str(e)}")

    def _generate_cache_key(self, message: RichMessage) -> str:
        """Generate cache key for message."""
        # Create content hash for content-based caching
        content_hash = hashlib.sha256(message.content.encode('utf-8')).hexdigest()[:16]

        # Include message type and key attributes
        key_components = [
            self.config.cache_key_prefix,
            message.message_type.value,
            message.session_id or "no_session",
            content_hash
        ]

        key = ":".join(key_components)

        # Truncate if too long
        if len(key) > self.config.max_key_length:
            key = key[:self.config.max_key_length]

        return key

    async def _compress_entry(self, entry: CacheEntry) -> CacheEntry:
        """Compress cache entry if beneficial."""
        try:
            # Serialize message
            serialized = json.dumps(entry.message.to_dict()).encode('utf-8')

            # Compress
            compressed = gzip.compress(serialized)

            # Check if compression is beneficial
            if len(compressed) < len(serialized):
                entry.compressed_data = compressed
                entry.compression_ratio = len(compressed) / len(serialized)
                entry.size_bytes = len(compressed)

                self.cache_stats["compression_saves"] += 1
                self.cache_stats["total_bytes_saved"] += (len(serialized) - len(compressed))

                self.logger.debug(f"Compressed entry {entry.key}: {entry.compression_ratio:.2%} ratio")
        except Exception as e:
            self.logger.warning(f"Compression failed for entry {entry.key}: {str(e)}")

        return entry

    def _deserialize_entry(self, entry: CacheEntry) -> Dict[str, Any]:
        """Deserialize cache entry to data."""
        data = entry.metadata.copy()

        # Add compression info
        if entry.compressed_data:
            data["compressed"] = True
            data["compression_ratio"] = entry.compression_ratio

        # Add cache metadata
        data["cache_info"] = {
            "created_at": entry.created_at.isoformat(),
            "last_accessed": entry.last_accessed.isoformat(),
            "access_count": entry.access_count,
            "size_bytes": entry.size_bytes,
            "age_seconds": entry.get_age_seconds(),
            "idle_seconds": entry.get_idle_seconds()
        }

        return data

    def _serialize_to_bytes(self, entry: CacheEntry) -> bytes:
        """Serialize cache entry to bytes for distributed storage."""
        try:
            # Use pickle for full object serialization
            return pickle.dumps(entry)
        except Exception as e:
            self.logger.error(f"Failed to serialize cache entry {entry.key}: {str(e)}")
            raise

    def _deserialize_from_bytes(self, data: bytes) -> CacheEntry:
        """Deserialize cache entry from bytes."""
        try:
            return pickle.loads(data)
        except Exception as e:
            self.logger.error(f"Failed to deserialize cache entry: {str(e)}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = self.cache_stats.copy()
        stats["memory_cache"] = self.memory_cache.get_stats()

        # Calculate overall hit rate
        total_requests = stats["total_requests"]
        if total_requests > 0:
            stats["overall_hit_rate"] = stats["cache_hits"] / total_requests
        else:
            stats["overall_hit_rate"] = 0.0

        # Compression statistics
        if stats["compression_saves"] > 0:
            stats["average_compression_ratio"] = stats["total_bytes_saved"] / (stats["compression_saves"] * 1024)  # Rough estimate
        else:
            stats["average_compression_ratio"] = 0.0

        # Configuration info
        stats["config"] = {
            "max_memory_entries": self.config.max_memory_entries,
            "max_memory_size_mb": self.config.max_memory_size_mb,
            "default_ttl_seconds": self.config.default_ttl_seconds,
            "enable_compression": self.config.enable_compression,
            "eviction_policy": self.config.eviction_policy.value,
            "distributed_cache_enabled": self.distributed_cache is not None
        }

        return stats

    def reset_stats(self):
        """Reset cache statistics."""
        self.cache_stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "compression_saves": 0,
            "total_bytes_saved": 0
        }

        # Reset memory cache stats
        self.memory_cache.hits = 0
        self.memory_cache.misses = 0
        self.memory_cache.evictions = 0

    async def start_background_cleanup(self):
        """Start background cleanup task."""
        if self._running:
            return

        self._running = True
        self.cleanup_task = asyncio.create_task(self._background_cleanup())

    async def stop_background_cleanup(self):
        """Stop background cleanup task."""
        self._running = False
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

    async def _background_cleanup(self):
        """Background cleanup task."""
        while self._running:
            try:
                await asyncio.sleep(self.config.background_cleanup_interval)
                await self._cleanup_expired_entries()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Background cleanup error: {str(e)}")

    async def _cleanup_expired_entries(self):
        """Clean up expired cache entries."""
        # Memory cache cleanup
        expired_keys = [
            key for key, entry in self.memory_cache.cache.items()
            if entry.is_expired()
        ]

        for key in expired_keys:
            self.memory_cache.remove(key)

        if expired_keys:
            self.logger.debug(f"Cleaned up {len(expired_keys)} expired memory cache entries")

        # Distributed cache cleanup (Redis handles TTL automatically)
        # No explicit cleanup needed for Redis with TTL

    # Batch operations
    async def get_batch(self, messages: List[RichMessage]) -> List[Optional[Dict[str, Any]]]:
        """Get multiple cached messages."""
        results = []
        for message in messages:
            result = await self.get(message)
            results.append(result)
        return results

    async def set_batch(self, message_data_pairs: List[Tuple[RichMessage, Dict[str, Any]]],
                       ttl_seconds: Optional[int] = None) -> List[bool]:
        """Set multiple cached messages."""
        results = []
        for message, data in message_data_pairs:
            result = await self.set(message, data, ttl_seconds)
            results.append(result)
        return results

    # Cache warming and preloading
    async def warm_cache(self, messages: List[RichMessage], data_generator: Callable) -> int:
        """Warm cache with precomputed data."""
        warmed_count = 0

        for message in messages:
            # Check if already cached
            cached_data = await self.get(message)
            if cached_data is None:
                # Generate data and cache it
                try:
                    data = await data_generator(message)
                    success = await self.set(message, data)
                    if success:
                        warmed_count += 1
                except Exception as e:
                    self.logger.warning(f"Cache warming failed for message {message.id}: {str(e)}")

        self.logger.info(f"Cache warming completed: {warmed_count}/{len(messages)} entries warmed")
        return warmed_count

    # Cache optimization
    async def optimize_cache(self) -> Dict[str, Any]:
        """Optimize cache performance."""
        optimization_results = {
            "memory_entries_removed": 0,
            "memory_bytes_freed": 0,
            "compression_improved": 0
        }

        # Remove expired entries
        current_size = len(self.memory_cache.cache)
        await self._cleanup_expired_entries()
        new_size = len(self.memory_cache.cache)
        optimization_results["memory_entries_removed"] = current_size - new_size

        # Trigger garbage collection
        import gc
        gc.collect()

        # Log optimization results
        self.logger.info(f"Cache optimization completed: {optimization_results}")

        return optimization_results

    # Export/Import functionality
    def export_cache_state(self) -> Dict[str, Any]:
        """Export cache state for persistence."""
        state = {
            "config": {
                "max_memory_entries": self.config.max_memory_entries,
                "max_memory_size_mb": self.config.max_memory_size_mb,
                "default_ttl_seconds": self.config.default_ttl_seconds,
                "eviction_policy": self.config.eviction_policy.value
            },
            "stats": self.get_stats(),
            "memory_cache_keys": list(self.memory_cache.cache.keys()),
            "memory_cache_size": len(self.memory_cache.cache)
        }
        return state

    def __del__(self):
        """Cleanup when cache is destroyed."""
        if self.cleanup_task:
            self.cleanup_task.cancel()