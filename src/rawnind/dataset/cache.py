"""
Disk-based cache using JSONL format with async I/O. Keeps only an index in memory, not values.
Writes immediately to disk and supports compaction to remove duplicates.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import trio


class StreamingJSONCache:
    """
    Async disk-based cache that writes immediately. Maintains only an index in memory,
    not the actual values.

    Uses trio.Lock for now, but could be optimized with:
    - Semaphore for concurrent reads
    - Channel-based worker pattern for better backpressure
    """

    def __init__(
        self,
        cache_path: Path,
        compact_threshold: Optional[int] = None,
        handle_corruption: bool = False
    ):
        """
        Initialize the streaming cache (call load() after init to load existing cache).

        Args:
            cache_path: Path to the JSONL cache file
            compact_threshold: Auto-compact when duplicate ratio exceeds this
            handle_corruption: If True, skip corrupted lines when loading
        """
        self.cache_path = trio.Path(cache_path)
        self.compact_threshold = compact_threshold
        self.handle_corruption = handle_corruption

        # In-memory index: key -> file position of latest entry
        self._index: Dict[str, int] = {}

        # Track total entries vs unique keys for compaction decision
        self._total_entries = 0

        # Async safety - could use Semaphore for concurrent reads
        self._lock = trio.Lock()

    async def load(self) -> None:
        """Load existing cache if it exists. Call this after __init__."""
        await self._load_existing_cache()

    async def _load_existing_cache(self) -> None:
        """Load existing cache file and build index."""
        if not await self.cache_path.exists():
            return

        async with self._lock:
            try:
                async with await self.cache_path.open('r', encoding='utf-8') as f:
                    line_num = 0
                    while True:
                        # Store position before reading line
                        current_pos = await f.tell()
                        line = await f.readline()

                        if not line:
                            break

                        line_num += 1

                        if line.strip():
                            try:
                                entry = json.loads(line)
                                # Update index to point to latest entry for this key
                                self._index[entry["key"]] = current_pos
                                self._total_entries += 1
                            except (json.JSONDecodeError, KeyError) as e:
                                if self.handle_corruption:
                                    # Skip corrupted lines
                                    print(f"Warning: Skipping corrupted line {line_num}: {e}")
                                else:
                                    raise

            except FileNotFoundError:
                # File was deleted between existence check and open
                pass

    async def put(self, cache_key: str, value: Any) -> None:
        """
        Put a value in the cache, writing immediately to disk.

        Args:
            cache_key: The cache key
            value: The value to cache (must be JSON serializable)
        """
        entry = {
            "key": cache_key,
            "value": value,
            "timestamp": time.time()
        }

        async with self._lock:
            # Ensure directory exists
            await self.cache_path.parent.mkdir(parents=True, exist_ok=True)

            # Append entry to file
            async with await self.cache_path.open('a', encoding='utf-8') as f:
                # Get current file position before writing
                file_size = await f.tell()

                # Update index to point to this entry
                self._index[cache_key] = file_size

                # Write entry
                line = json.dumps(entry, ensure_ascii=False, separators=(',', ':')) + '\n'
                await f.write(line)
                await f.flush()

        self._total_entries += 1

        # Check if auto-compaction is needed
        if self.compact_threshold and self._total_entries > len(self._index) * self.compact_threshold:
            await self.compact()

    async def get(self, cache_key: str) -> Optional[Any]:
        """
        Get a value from the cache.

        Args:
            cache_key: The cache key

        Returns:
            The cached value, or None if not found
        """
        async with self._lock:
            if cache_key not in self._index:
                return None

            # Read the specific line from file
            file_pos = self._index[cache_key]

            try:
                async with await self.cache_path.open('r', encoding='utf-8') as f:
                    await f.seek(file_pos)
                    line = await f.readline()
                    if line:
                        entry = json.loads(line)
                        return entry["value"]
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                # Cache file deleted or corrupted
                del self._index[cache_key]
                return None

        return None

    def __contains__(self, cache_key: str) -> bool:
        """Check if a key exists in the cache (sync - just checks index)."""
        # Just check index - faster than doing a full get()
        # If file is corrupted, we'll handle it on actual get()
        return cache_key in self._index

    def keys(self) -> List[str]:
        """Return all cache keys (sync - just returns index keys)."""
        return list(self._index.keys())

    async def clear(self) -> None:
        """Clear the cache."""
        async with self._lock:
            self._index.clear()
            self._total_entries = 0

            # Truncate or delete the file
            if await self.cache_path.exists():
                try:
                    # Truncate file to 0 bytes
                    async with await self.cache_path.open('w'):
                        pass
                except:
                    # If truncation fails, try to delete
                    try:
                        await self.cache_path.unlink()
                    except:
                        pass

    async def compact(self) -> None:
        """
        Compact the cache file by removing duplicate entries.
        Keeps only the latest entry for each key.
        """
        async with self._lock:
            if not await self.cache_path.exists() or not self._index:
                return

            # Create temporary file
            temp_path = trio.Path(Path(self.cache_path).with_suffix('.tmp'))

            # Write latest entries to temp file
            new_index = {}
            new_pos = 0

            async with await temp_path.open('w', encoding='utf-8') as out_f:
                # Process keys in order they appear in current index
                # Read directly from file to avoid lock re-acquisition
                async with await self.cache_path.open('r', encoding='utf-8') as in_f:
                    for cache_key in self._index:
                        file_pos = self._index[cache_key]
                        
                        # Read value from file
                        await in_f.seek(file_pos)
                        line = await in_f.readline()
                        if line:
                            try:
                                entry_data = json.loads(line)
                                value = entry_data["value"]
                            except (json.JSONDecodeError, KeyError):
                                continue
                        else:
                            continue
                        
                        # Write compacted entry
                        entry = {
                            "key": cache_key,
                            "value": value,
                            "timestamp": time.time()
                        }
                        new_index[cache_key] = new_pos
                        
                        line_out = json.dumps(entry, ensure_ascii=False, separators=(',', ':')) + '\n'
                        await out_f.write(line_out)
                        new_pos += len(line_out.encode('utf-8'))

                await out_f.flush()

            # Atomically replace old file with compacted version
            await temp_path.rename(self.cache_path)

            # Update index and stats
            self._index = new_index
            self._total_entries = len(self._index)

    async def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        async with self._lock:
            file_size = (await self.cache_path.stat()).st_size if await self.cache_path.exists() else 0

            stats = {
                "unique_keys": len(self._index),
                "total_entries": self._total_entries,
                "file_size": file_size,
                "needs_compaction": self._total_entries > len(self._index) * 2,
                "cache_path": str(self.cache_path)
            }

            if self.compact_threshold:
                stats["compact_threshold"] = self.compact_threshold
                stats["auto_compact_enabled"] = True
            else:
                stats["auto_compact_enabled"] = False

            return stats