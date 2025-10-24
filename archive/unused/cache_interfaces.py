"""
Cache interface definitions for dependency inversion.

Provides abstract base classes and interfaces for cache implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Generic, TypeVar

T = TypeVar('T')


@dataclass
class CacheMetrics:
    """Structured cache metrics data."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    corruptions_detected: int = 0
    avg_hit_latency_ms: float = 0.0
    avg_miss_latency_ms: float = 0.0
    cache_size_mb: float = 0.0
    items_count: int = 0
    last_eviction_time: Optional[datetime] = None

    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


@dataclass
class CacheEntry:
    """Metadata for a cached item."""

    key: str
    size_bytes: int
    last_access_time: float
    creation_time: float
    access_count: int = 0
    checksum: Optional[str] = None


class CacheInterface(ABC, Generic[T]):
    """
    Abstract interface for cache implementations.

    Follows Interface Segregation Principle by separating read/write operations.
    """

    @abstractmethod
    def get(self, key: str) -> Optional[T]:
        """
        Retrieve item from cache.

        Args:
            key: Cache key

        Returns:
            Cached item or None if not found
        """
        pass

    @abstractmethod
    def set(self, key: str, value: T) -> None:
        """
        Store item in cache.

        Args:
            key: Cache key
            value: Item to cache
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists, False otherwise
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete item from cache.

        Args:
            key: Cache key

        Returns:
            True if item was deleted, False if not found
        """
        pass


class MetricsInterface(ABC):
    """Interface for cache metrics tracking."""

    @abstractmethod
    def get_metrics(self) -> CacheMetrics:
        """Get current cache metrics."""
        pass

    @abstractmethod
    def reset_metrics(self) -> None:
        """Reset all metrics to initial state."""
        pass

    @abstractmethod
    def record_hit(self, latency_ms: float) -> None:
        """Record a cache hit with its latency."""
        pass

    @abstractmethod
    def record_miss(self, latency_ms: float) -> None:
        """Record a cache miss with its latency."""
        pass


class EvictionPolicy(ABC):
    """Abstract base class for cache eviction policies."""

    @abstractmethod
    def track_access(self, key: str) -> None:
        """Track access to a cache entry."""
        pass

    @abstractmethod
    def select_victims(
            self, entries: list[CacheEntry], bytes_to_free: int
    ) -> list[CacheEntry]:
        """
        Select cache entries to evict.

        Args:
            entries: List of all cache entries
            bytes_to_free: Number of bytes that need to be freed

        Returns:
            List of entries to evict
        """
        pass


class LRUEvictionPolicy(EvictionPolicy):
    """Least Recently Used eviction policy."""

    def __init__(self):
        """Initialize LRU policy."""
        self._access_times: Dict[str, float] = {}

    def track_access(self, key: str) -> None:
        """Track access time for LRU."""
        import time
        self._access_times[key] = time.time()

    def select_victims(
            self, entries: list[CacheEntry], bytes_to_free: int
    ) -> list[CacheEntry]:
        """Select least recently used entries for eviction."""
        # Sort by access time (oldest first)
        sorted_entries = sorted(entries, key=lambda e: e.last_access_time)

        victims = []
        bytes_freed = 0

        for entry in sorted_entries:
            if bytes_freed >= bytes_to_free:
                break
            victims.append(entry)
            bytes_freed += entry.size_bytes

        return victims


class FIFOEvictionPolicy(EvictionPolicy):
    """First In First Out eviction policy."""

    def track_access(self, key: str) -> None:
        """FIFO doesn't need to track access."""
        pass

    def select_victims(
            self, entries: list[CacheEntry], bytes_to_free: int
    ) -> list[CacheEntry]:
        """Select oldest entries for eviction."""
        # Sort by creation time (oldest first)
        sorted_entries = sorted(entries, key=lambda e: e.creation_time)

        victims = []
        bytes_freed = 0

        for entry in sorted_entries:
            if bytes_freed >= bytes_to_free:
                break
            victims.append(entry)
            bytes_freed += entry.size_bytes

        return victims


class CacheManager(CacheInterface[T], MetricsInterface):
    """
    Abstract base class combining cache and metrics interfaces.

    Template method pattern for common cache operations.
    """

    def __init__(self, eviction_policy: Optional[EvictionPolicy] = None):
        """Initialize with optional eviction policy."""
        self.eviction_policy = eviction_policy or LRUEvictionPolicy()
        self.metrics = CacheMetrics()

    @abstractmethod
    def clear(self) -> None:
        """Clear all items from cache."""
        pass

    @abstractmethod
    def get_size_mb(self) -> float:
        """Get total cache size in MB."""
        pass

    def get_metrics(self) -> CacheMetrics:
        """Get current cache metrics."""
        self.metrics.cache_size_mb = self.get_size_mb()
        return self.metrics

    def reset_metrics(self) -> None:
        """Reset metrics to initial state."""
        self.metrics = CacheMetrics()

    def record_hit(self, latency_ms: float) -> None:
        """Record cache hit."""
        self.metrics.hits += 1
        # Update running average
        total_hits = self.metrics.hits
        current_avg = self.metrics.avg_hit_latency_ms
        self.metrics.avg_hit_latency_ms = (
                (current_avg * (total_hits - 1) + latency_ms) / total_hits
        )

    def record_miss(self, latency_ms: float) -> None:
        """Record cache miss."""
        self.metrics.misses += 1
        # Update running average
        total_misses = self.metrics.misses
        current_avg = self.metrics.avg_miss_latency_ms
        self.metrics.avg_miss_latency_ms = (
                (current_avg * (total_misses - 1) + latency_ms) / total_misses
        )
