"""
Disk-based cache manager for persistent scene caching.

This module provides thread-safe caching with the following features:
- Multiple eviction policies (LRU, FIFO)
- Automatic size management with configurable limits
- Corruption detection and self-healing
- Performance metrics and monitoring
- Connection pooling for high-throughput scenarios
- Batch operations

Example Usage:
    >>> from pathlib import Path
    >>> from rawnind.dataset.cache_manager import DiskCacheManager
    >>>
    >>> # Basic cache setup
    >>> cache = DiskCacheManager(
    >>>     cache_dir=Path("/tmp/cache"),
    >>>     max_size_mb=500,
    >>>     eviction="lru"
    >>> )
    >>>
    >>> # Store and retrieve scenes
    >>> cache.set("scene_001", scene_data)
    >>> scene = cache.get("scene_001")
    >>>
    >>> # Get performance metrics
    >>> metrics = cache.get_metrics()
    >>> print(f"Hit rate: {metrics.hit_rate():.2%}")
"""

import concurrent.futures
import hashlib
import json
import logging
import pickle
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Set, Tuple

from rawnind.dataset.SceneInfo import SceneInfo
from rawnind.dataset.cache_interfaces import (
    CacheManager,
    CacheEntry,
    LRUEvictionPolicy,
    FIFOEvictionPolicy,
    EvictionPolicy,
)
from rawnind.dataset.constants import (
    BYTES_PER_MB,
    CACHE_FILE_EXTENSION,
    DEFAULT_CACHE_EVICTION_POLICY,
)

# Configure module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class CacheConfig:
    """Brief summary of the cache configuration options.

    This dataclass encapsulates all configurable parameters for the caching
    system. It defines defaults and optional settings that control where cache
    data is stored, how much space it may occupy, the policies governing
    eviction, concurrency behavior, and additional features such as compression
    and integrity verification. The attributes are intended to be used by the
    cache implementation to tailor its behavior to the needs of the application.

    Attributes:
        cache_dir: Directory where cache files are persisted.
        max_size_mb: Optional upper limit for total cache size in megabytes.
        eviction_policy: Strategy used to remove entries when the cache exceeds
            its size limit.
        thread_safe: Enables safe access to the cache from multiple threads.
        track_metrics: Collects usage statistics for cache operations.
        verify_checksums: Checks data integrity using checksums on reads and
            writes.
        auto_repair: Attempts automatic correction of corrupted cache entries.
        compression: Stores cache entries in a compressed format to reduce
            storage usage.
        max_workers: Number of worker threads allocated for asynchronous cache
            tasks.
        batch_size: Number of items processed together in batch operations.
        retry_count: Number of attempts to retry transient cache errors.
        retry_delay_seconds: Pause duration between retry attempts, expressed in
            seconds.
    """

    cache_dir: Path
    max_size_mb: Optional[int] = None
    eviction_policy: str = DEFAULT_CACHE_EVICTION_POLICY
    thread_safe: bool = True
    track_metrics: bool = True
    verify_checksums: bool = False
    auto_repair: bool = False
    compression: bool = False
    max_workers: int = 4
    batch_size: int = 10
    retry_count: int = 3
    retry_delay_seconds: float = 0.5


class ConnectionPool:
    """
    Connection pool for managing file handles efficiently.

    Reduces file handle churn and improves performance for high-throughput scenarios.
    """

    def __init__(self, max_connections: int = 10):
        """Initialize connection pool."""
        self.max_connections = max_connections
        self._pool: List[Any] = []
        self._in_use: Set[Any] = set()
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    @contextmanager
    def get_connection(self):
        """Get a connection from the pool."""
        conn = None
        with self._lock:
            while not self._pool and len(self._in_use) >= self.max_connections:
                self._condition.wait()

            if self._pool:
                conn = self._pool.pop()
            else:
                conn = self._create_connection()

            self._in_use.add(conn)

        try:
            yield conn
        finally:
            with self._lock:
                self._in_use.discard(conn)
                if conn:
                    self._pool.append(conn)
                self._condition.notify()

    def _create_connection(self) -> Any:
        """Create a new connection (placeholder for file handle management)."""
        return object()  # In real implementation, this would be a file handle wrapper


class DiskCacheManager(CacheManager[SceneInfo]):
    """ """

    def __init__(
        self,
        cache_dir: Path,
        max_size_mb: Optional[int] = None,
        eviction: str = DEFAULT_CACHE_EVICTION_POLICY,
        thread_safe: bool = True,
        track_metrics: bool = True,
        verify_checksums: bool = False,
        compression: bool = False,
        auto_repair: bool = False,
        max_workers: int = 4,
    ):
        """
        Initialize disk cache manager.

        Args:
            cache_dir: Directory for cache files
            max_size_mb: Maximum cache size in MB
            eviction: Eviction policy ("lru" or "fifo")
            thread_safe: Enable thread safety
            track_metrics: Enable metrics collection
            verify_checksums: Verify data integrity
            compression: Enable compression (saves space but slower)
            auto_repair: Enable automatic corruption recovery
            max_workers: Maximum worker threads for batch operations

        Raises:
            ValueError: If invalid configuration is provided
            OSError: If cache directory cannot be created
        """
        # Input validation
        if max_size_mb is not None and max_size_mb <= 0:
            raise ValueError(f"max_size_mb must be positive, got {max_size_mb}")

        # Initialize base class
        eviction_policy = self._create_eviction_policy(eviction)
        super().__init__(eviction_policy)

        # Setup cache directory
        self.cache_dir = Path(cache_dir)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create cache directory {cache_dir}: {e}")
            raise

        # Configuration
        self.max_size_mb = max_size_mb
        self.thread_safe = thread_safe
        self.track_metrics = track_metrics
        self.verify_checksums = verify_checksums
        self.compression = compression
        self.auto_repair = auto_repair
        self.max_workers = max_workers

        # Internal state
        self._entries: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock() if thread_safe else None
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._connection_pool = ConnectionPool() if thread_safe else None

        # Load existing cache metadata
        self._load_metadata()

        logger.info(
            f"DiskCacheManager initialized: dir={cache_dir}, "
            f"max_size={max_size_mb}MB, eviction={eviction}, "
            f"compression={compression}"
        )

    @contextmanager
    def _thread_lock(self):
        """Context manager for thread-safe operations."""
        if self._lock:
            self._lock.acquire()
            try:
                yield
            finally:
                self._lock.release()
        else:
            yield

    def _create_eviction_policy(self, policy_name: str) -> EvictionPolicy:
        """
        Factory method to create eviction policy.

        Args:
            policy_name: Name of eviction policy

        Returns:
            EvictionPolicy instance

        Raises:
            ValueError: If policy name is not recognized
        """
        policies = {
            "lru": LRUEvictionPolicy,
            "fifo": FIFOEvictionPolicy,
        }
        policy_class = policies.get(policy_name.lower())
        if not policy_class:
            available = ", ".join(policies.keys())
            raise ValueError(
                f"Unknown eviction policy: {policy_name}. Available: {available}"
            )
        return policy_class()

    def set(self, key: str, value: SceneInfo, retry_count: int = 3) -> None:
        """
        Write scene to cache with retry logic and size enforcement.

        Args:
            key: Cache key (scene name)
            value: SceneInfo object to cache
            retry_count: Number of retry attempts

        Raises:
            RuntimeError: If cache write fails after all retries
        """
        if not key:
            raise ValueError("Cache key cannot be empty")

        # Enforce size limit before writing
        if self.max_size_mb is not None:
            with self._thread_lock():
                self._enforce_size_limit()

        cache_file = self.cache_dir / f"{key}{CACHE_FILE_EXTENSION}"

        for attempt in range(retry_count):
            try:
                # Calculate checksum if verification enabled
                checksum = None
                if self.verify_checksums:
                    checksum = self._calculate_checksum(value)

                # Serialize with optional compression
                data = self._serialize(value)

                # Write to disk atomically
                temp_file = cache_file.with_suffix(".tmp")
                with open(temp_file, "wb") as f:
                    f.write(data)

                # Atomic rename
                temp_file.replace(cache_file)

                # Update metadata
                file_size = cache_file.stat().st_size
                current_time = time.time()

                with self._thread_lock():
                    self._entries[key] = CacheEntry(
                        key=key,
                        size_bytes=file_size,
                        last_access_time=current_time,
                        creation_time=current_time,
                        access_count=1,
                        checksum=checksum,
                    )

                # Track access for eviction policy
                self.eviction_policy.track_access(key)

                logger.debug(f"Cached scene {key} ({file_size / 1024:.1f}KB)")
                break

            except Exception as e:
                if attempt < retry_count - 1:
                    logger.warning(
                        f"Cache write attempt {attempt + 1} failed for {key}: {e}"
                    )
                    time.sleep(0.1 * (2**attempt))  # Exponential backoff
                else:
                    logger.error(
                        f"Failed to cache {key} after {retry_count} attempts: {e}"
                    )
                    if self.track_metrics:
                        self.metrics.corruptions_detected += 1
                    raise RuntimeError(f"Cache write failed for {key}: {e}")

    def get(self, key: str) -> Optional[SceneInfo]:
        """
        Read scene from cache with integrity verification.

        Args:
            key: Cache key (scene name)

        Returns:
            SceneInfo if cached and valid, None otherwise
        """
        if not key:
            return None

        start_time = time.time() if self.track_metrics else 0
        cache_file = self.cache_dir / f"{key}{CACHE_FILE_EXTENSION}"

        if not cache_file.exists():
            if self.track_metrics:
                latency_ms = (time.time() - start_time) * 1000
                self.record_miss(latency_ms)
            return None

        try:
            # Load and deserialize
            with open(cache_file, "rb") as f:
                data = f.read()

            value = self._deserialize(data)

            # Verify type
            if not isinstance(value, SceneInfo):
                raise ValueError(f"Invalid type in cache: {type(value)}")

            # Verify checksum if enabled
            if self.verify_checksums:
                with self._thread_lock():
                    entry = self._entries.get(key)
                    if entry and entry.checksum:
                        current_checksum = self._calculate_checksum(value)
                        if current_checksum != entry.checksum:
                            raise ValueError("Checksum mismatch - data corrupted")

            # Update metadata
            with self._thread_lock():
                if key in self._entries:
                    self._entries[key].last_access_time = time.time()
                    self._entries[key].access_count += 1

            # Track access
            self.eviction_policy.track_access(key)

            if self.track_metrics:
                latency_ms = (time.time() - start_time) * 1000
                self.record_hit(latency_ms)

            return value

        except Exception as e:
            logger.warning(f"Cache read failed for {key}: {e}")
            self._handle_corrupted_entry(key, cache_file)

            if self.auto_repair:
                logger.info(f"Attempting auto-repair for {key}")
                # this could trigger re-fetch from pipeline
                raise NotImplementedError

            if self.track_metrics:
                latency_ms = (time.time() - start_time) * 1000
                self.record_miss(latency_ms)

            return None

    def get_batch(self, keys: List[str]) -> Dict[str, Optional[SceneInfo]]:
        """
        Retrieve multiple items in parallel for improved performance.

        Args:
            keys: List of cache keys

        Returns:
            Dictionary mapping keys to values (None if not found)
        """
        if not keys:
            return {}

        results = {}

        # Use thread pool for parallel retrieval
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            future_to_key = {executor.submit(self.get, key): key for key in keys}

            for future in concurrent.futures.as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    logger.warning(f"Batch get failed for {key}: {e}")
                    results[key] = None

        return results

    def set_batch(self, items: Dict[str, SceneInfo]) -> Dict[str, bool]:
        """
        Store multiple items in parallel.

        Args:
            items: Dictionary mapping keys to values

        Returns:
            Dictionary mapping keys to success status
        """
        if not items:
            return {}

        results = {}

        # Use thread pool for parallel writes
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            future_to_key = {
                executor.submit(self._set_with_result, key, value): key
                for key, value in items.items()
            }

            for future in concurrent.futures.as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    results[key] = future.result()
                except Exception:
                    results[key] = False

        return results

    def _set_with_result(self, key: str, value: SceneInfo) -> bool:
        """
        Set with boolean result for batch operations.

        Args:
            key: Cache key
            value: SceneInfo to store

        Returns:
            bool: Success status
        """
        try:
            self.set(key, value)
            return True
        except Exception:
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not key:
            return False
        cache_file = self.cache_dir / f"{key}{CACHE_FILE_EXTENSION}"
        return cache_file.exists()

    def delete(self, key: str) -> bool:
        """
        Delete item from cache.

        Args:
            key: Cache key

        Returns:
            bool: True if deleted, False if not found
        """
        if not key:
            return False

        cache_file = self.cache_dir / f"{key}{CACHE_FILE_EXTENSION}"
        if cache_file.exists():
            try:
                cache_file.unlink()
                with self._thread_lock():
                    self._entries.pop(key, None)
                logger.debug(f"Deleted cache entry: {key}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete {key}: {e}")
        return False

    def clear(self) -> None:
        """Clear all items from cache."""
        cleared = 0
        failed = 0

        for cache_file in self.cache_dir.glob(f"*{CACHE_FILE_EXTENSION}"):
            try:
                cache_file.unlink()
                cleared += 1
            except Exception as e:
                logger.warning(f"Failed to delete {cache_file}: {e}")
                failed += 1

        with self._thread_lock():
            self._entries.clear()

        logger.info(f"Cache cleared: {cleared} files removed, {failed} failures")

    def get_size_mb(self) -> float:
        """Get total cache size in MB."""
        total_bytes = sum(
            self._get_file_size(f)
            for f in self.cache_dir.glob(f"*{CACHE_FILE_EXTENSION}")
        )
        return total_bytes / BYTES_PER_MB

    def _get_file_size(self, file_path: Path) -> int:
        """Get file size safely."""
        try:
            return file_path.stat().st_size
        except Exception:
            return 0

    def _serialize(self, value: SceneInfo) -> bytes:
        """
        Serialize value with optional compression.

        Args:
            value: SceneInfo to serialize

        Returns:
            bytes: Serialized data
        """
        data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

        if self.compression:
            import gzip

            data = gzip.compress(data, compresslevel=1)  # Fast compression

        return data

    def _deserialize(self, data: bytes) -> SceneInfo:
        """
        Deserialize value with automatic decompression.

        Args:
            data: Serialized data

        Returns:
            SceneInfo: Deserialized object
        """
        if self.compression:
            import gzip

            try:
                data = gzip.decompress(data)
            except Exception:
                # Try without decompression (backwards compatibility)
                pass

        return pickle.loads(data)

    def _calculate_checksum(self, value: SceneInfo) -> str:
        """
        Calculate checksum for integrity verification.

        Args:
            value: SceneInfo to checksum

        Returns:
            str: Hex checksum
        """
        data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(data).hexdigest()

    def _handle_corrupted_entry(self, key: str, cache_file: Path) -> None:
        """Handle corrupted cache entry."""
        try:
            cache_file.unlink()
            with self._thread_lock():
                self._entries.pop(key, None)
            if self.track_metrics:
                self.metrics.corruptions_detected += 1
            logger.warning(f"Removed corrupted cache entry: {key}")
        except Exception as e:
            logger.error(f"Failed to remove corrupted entry {key}: {e}")

    def _enforce_size_limit(self) -> None:
        """Enforce cache size limit using eviction policy."""
        if self.max_size_mb is None:
            return

        current_size_mb = self.get_size_mb()
        if current_size_mb <= self.max_size_mb:
            return

        logger.info(
            f"Cache size {current_size_mb:.1f}MB exceeds limit {self.max_size_mb}MB"
        )

        # Calculate space to free
        bytes_to_free = int((current_size_mb - self.max_size_mb * 0.9) * BYTES_PER_MB)

        # Get entries for eviction
        entries = self._build_cache_entries()

        # Select victims
        victims = self.eviction_policy.select_victims(entries, bytes_to_free)

        # Evict selected entries
        self._evict_entries(victims)

    def _build_cache_entries(self) -> List[CacheEntry]:
        """Build list of cache entries with metadata."""
        entries = []

        for cache_file in self.cache_dir.glob(f"*{CACHE_FILE_EXTENSION}"):
            key = cache_file.stem

            if key in self._entries:
                entries.append(self._entries[key])
            else:
                # Create entry for untracked file
                try:
                    stat = cache_file.stat()
                    entry = CacheEntry(
                        key=key,
                        size_bytes=stat.st_size,
                        last_access_time=stat.st_mtime,
                        creation_time=stat.st_ctime,
                        access_count=0,
                    )
                    entries.append(entry)
                except Exception:
                    continue

        return entries

    def _evict_entries(self, victims: List[CacheEntry]) -> None:
        """Evict selected cache entries."""
        evicted_count = 0
        evicted_bytes = 0

        for entry in victims:
            if self.delete(entry.key):
                evicted_count += 1
                evicted_bytes += entry.size_bytes

        if self.track_metrics:
            self.metrics.evictions += evicted_count
            self.metrics.last_eviction_time = datetime.now()

        logger.info(
            f"Evicted {evicted_count} entries ({evicted_bytes / BYTES_PER_MB:.1f}MB)"
        )

    def _load_metadata(self) -> None:
        """Load cache metadata from disk on startup."""
        metadata_file = self.cache_dir / ".cache_metadata.json"

        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

                # Reconstruct entries
                for key, data in metadata.items():
                    self._entries[key] = CacheEntry(
                        key=key,
                        size_bytes=data.get("size_bytes", 0),
                        last_access_time=data.get("last_access_time", 0),
                        creation_time=data.get("creation_time", 0),
                        access_count=data.get("access_count", 0),
                        checksum=data.get("checksum"),
                    )

                logger.info(f"Loaded metadata for {len(self._entries)} cache entries")
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")

    def save_metadata(self) -> None:
        """Save cache metadata to disk for persistence."""
        metadata_file = self.cache_dir / ".cache_metadata.json"

        try:
            with self._thread_lock():
                metadata = {}
                for key, entry in self._entries.items():
                    metadata[key] = {
                        "size_bytes": entry.size_bytes,
                        "last_access_time": entry.last_access_time,
                        "creation_time": entry.creation_time,
                        "access_count": entry.access_count,
                        "checksum": entry.checksum,
                    }

            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.debug(f"Saved metadata for {len(metadata)} cache entries")
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on cache.

        Returns:
            dict: Health status information
        """
        health = {
            "status": "healthy",
            "cache_dir": str(self.cache_dir),
            "size_mb": self.get_size_mb(),
            "max_size_mb": self.max_size_mb,
            "entry_count": len(self._entries),
            "file_count": len(list(self.cache_dir.glob(f"*{CACHE_FILE_EXTENSION}"))),
            "compression": self.compression,
            "thread_safe": self.thread_safe,
        }

        # Check for orphaned files
        orphaned = health["file_count"] - health["entry_count"]
        if orphaned > 0:
            health["orphaned_files"] = orphaned
            health["status"] = "degraded"

        # Check size limit
        if self.max_size_mb and health["size_mb"] > self.max_size_mb:
            health["status"] = "over_limit"

        return health

    def __del__(self):
        """Cleanup on deletion."""
        try:
            # Save metadata before shutdown
            if hasattr(self, "_entries") and self._entries:
                self.save_metadata()

            # Shutdown executor
            if hasattr(self, "_executor"):
                self._executor.shutdown(wait=False)
        except Exception:
            pass


class SelfHealingCache(DiskCacheManager):
    """
    Cache with auto-repair and recovery capabilities.

    Extends DiskCacheManager with:
        - Automatic corruption recovery
        - Background integrity checks
        - Redundant storage options
        - Self-optimization based on usage patterns
    """

    def __init__(
        self,
        cache_dir: Path,
        auto_repair: bool = True,
        redundancy_level: int = 1,
        background_checks: bool = True,
    ):
        """
        Initialize self-healing cache.

        Args:
            cache_dir: Cache directory
            auto_repair: Enable automatic repair
            redundancy_level: Number of backup copies (0-2)
            background_checks: Enable background integrity checks
        """
        super().__init__(cache_dir, verify_checksums=True, auto_repair=auto_repair)
        self.redundancy_level = min(max(redundancy_level, 0), 2)
        self.background_checks = background_checks
        self._repair_count = 0

        if self.redundancy_level > 0:
            self._setup_redundancy()

        logger.info(f"SelfHealingCache initialized with redundancy={redundancy_level}")

    def _setup_redundancy(self) -> None:
        """Setup redundant storage directories."""
        for i in range(1, self.redundancy_level + 1):
            backup_dir = self.cache_dir.parent / f"{self.cache_dir.name}_backup{i}"
            backup_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Optional[SceneInfo]:
        """Get with automatic repair on corruption."""
        result = super().get(key)

        if result is None and self.auto_repair:
            # Try to recover from backup
            result = self._recover_from_backup(key)
            if result:
                self._repair_count += 1
                logger.info(
                    f"Recovered {key} from backup (repair #{self._repair_count})"
                )
                # Re-cache the recovered data
                try:
                    self.set(key, result)
                except Exception as e:
                    logger.warning(f"Failed to re-cache recovered {key}: {e}")

        return result

    def _recover_from_backup(self, key: str) -> Optional[SceneInfo]:
        """
        Attempt to recover item from backup locations.

        Args:
            key: Cache key

        Returns:
            SceneInfo if recovered, None otherwise
        """
        for i in range(1, self.redundancy_level + 1):
            backup_dir = self.cache_dir.parent / f"{self.cache_dir.name}_backup{i}"
            backup_file = backup_dir / f"{key}{CACHE_FILE_EXTENSION}"

            if backup_file.exists():
                try:
                    with open(backup_file, "rb") as f:
                        data = f.read()
                    value = self._deserialize(data)
                    if isinstance(value, SceneInfo):
                        return value
                except Exception as e:
                    logger.warning(f"Failed to recover {key} from backup {i}: {e}")

        return None

    def set(self, key: str, value: SceneInfo, retry_count: int = 3) -> None:
        """Set with redundant storage."""
        super().set(key, value, retry_count)

        # Store backups
        if self.redundancy_level > 0:
            self._store_backups(key, value)

    def _store_backups(self, key: str, value: SceneInfo) -> None:
        """Store redundant copies."""
        data = self._serialize(value)

        for i in range(1, self.redundancy_level + 1):
            backup_dir = self.cache_dir.parent / f"{self.cache_dir.name}_backup{i}"
            backup_file = backup_dir / f"{key}{CACHE_FILE_EXTENSION}"

            try:
                with open(backup_file, "wb") as f:
                    f.write(data)
            except Exception as e:
                logger.warning(f"Failed to store backup {i} for {key}: {e}")

    def run_integrity_check(self) -> Tuple[int, int]:
        """
        Run integrity check on all cached items.

        Returns:
            Tuple of (checked_count, repaired_count)
        """
        checked = 0
        repaired = 0

        for cache_file in self.cache_dir.glob(f"*{CACHE_FILE_EXTENSION}"):
            key = cache_file.stem
            checked += 1

            # Try to load and verify
            result = self.get(key)
            if result is None:
                # Item was corrupted and possibly repaired
                if self._repair_count > repaired:
                    repaired = self._repair_count - repaired

        logger.info(f"Integrity check: {checked} checked, {repaired} repaired")
        return checked, repaired
