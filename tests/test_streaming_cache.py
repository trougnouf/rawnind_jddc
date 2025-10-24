"""
Test suite for StreamingJSONCache - async cache streaming results to disk.
"""

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pytest
import trio

from rawnind.dataset.cache import StreamingJSONCache


@pytest.fixture
async def cache_path():
    """Provide temporary cache path."""
    test_dir = tempfile.mkdtemp(prefix="test_cache_")
    cache_file = Path(test_dir) / "test_cache.jsonl"
    yield cache_file
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


@pytest.mark.trio
async def test_basic_initialization(cache_path):
    """Test cache can be initialized with a path."""
    cache = StreamingJSONCache(cache_path)
    await cache.load()
    assert cache.cache_path == trio.Path(cache_path)
    assert cache._index is not None
    assert len(cache._index) == 0


@pytest.mark.trio
async def test_put_single_entry(cache_path):
    """Test putting a single entry writes it to disk immediately."""
    cache = StreamingJSONCache(cache_path)
    await cache.load()

    test_data = {"alignment": [10, 20], "gain": 1.5}
    await cache.put("image_001_sha1", test_data)

    # Verify file exists and contains the entry
    assert cache_path.exists()

    with open(cache_path, "r") as f:
        lines = f.readlines()

    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["key"] == "image_001_sha1"
    assert entry["value"] == test_data
    assert "timestamp" in entry


@pytest.mark.trio
async def test_get_existing_entry(cache_path):
    """Test retrieving an existing entry from cache."""
    cache = StreamingJSONCache(cache_path)
    await cache.load()

    test_data = {"alignment": [5, 10], "mask_mean": 0.95}
    await cache.put("test_key", test_data)

    result = await cache.get("test_key")
    assert result == test_data


@pytest.mark.trio
async def test_get_nonexistent_entry(cache_path):
    """Test retrieving a non-existent entry returns None."""
    cache = StreamingJSONCache(cache_path)
    await cache.load()

    result = await cache.get("nonexistent_key")
    assert result is None


@pytest.mark.trio
async def test_contains_check(cache_path):
    """Test checking if cache contains a key."""
    cache = StreamingJSONCache(cache_path)
    await cache.load()

    await cache.put("exists", {"data": "value"})

    assert "exists" in cache
    assert "not_exists" not in cache


@pytest.mark.trio
async def test_multiple_entries_same_key(cache_path):
    """Test that multiple puts with same key keeps latest value."""
    cache = StreamingJSONCache(cache_path)
    await cache.load()

    await cache.put("key1", {"version": 1})
    await cache.put("key1", {"version": 2})
    await cache.put("key1", {"version": 3})

    result = await cache.get("key1")
    assert result["version"] == 3

    # File should contain all 3 entries (before compaction)
    with open(cache_path, "r") as f:
        lines = f.readlines()
    assert len(lines) == 3


@pytest.mark.trio
async def test_load_existing_cache_file(cache_path):
    """Test loading an existing cache file on initialization."""
    # Pre-create a cache file
    entries = [
        {"key": "key1", "value": {"data": 1}, "timestamp": time.time()},
        {"key": "key2", "value": {"data": 2}, "timestamp": time.time()},
    ]

    with open(cache_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    # Load cache
    cache = StreamingJSONCache(cache_path)
    await cache.load()

    assert await cache.get("key1") == {"data": 1}
    assert await cache.get("key2") == {"data": 2}
    assert len(cache._index) == 2


@pytest.mark.trio
async def test_concurrent_writes(cache_path):
    """Test concurrent writes from multiple tasks are safe."""
    cache = StreamingJSONCache(cache_path)
    await cache.load()

    num_tasks = 10
    writes_per_task = 100

    async def writer(task_id: int):
        for i in range(writes_per_task):
            await cache.put(f"task_{task_id}_item_{i}", {"tid": task_id, "item": i})

    async with trio.open_nursery() as nursery:
        for tid in range(num_tasks):
            nursery.start_soon(writer, tid)

    # Verify all entries were written
    with open(cache_path, "r") as f:
        lines = f.readlines()

    assert len(lines) == num_tasks * writes_per_task

    # Verify we can retrieve all entries
    for tid in range(num_tasks):
        for i in range(writes_per_task):
            key = f"task_{tid}_item_{i}"
            assert key in cache


@pytest.mark.trio
async def test_flush_on_write(cache_path):
    """Test that writes are flushed to disk immediately."""
    cache = StreamingJSONCache(cache_path)
    await cache.load()

    # Write an entry
    await cache.put("test_flush", {"data": "important"})

    # Simulate crash by creating new cache instance
    cache2 = StreamingJSONCache(cache_path)
    await cache2.load()

    result = await cache2.get("test_flush")
    assert result == {"data": "important"}


@pytest.mark.trio
async def test_compact_removes_duplicates(cache_path):
    """Test that compact() removes duplicate entries."""
    cache = StreamingJSONCache(cache_path)
    await cache.load()

    # Write multiple versions of same keys
    for i in range(5):
        await cache.put("key_a", {"version": i})
        await cache.put("key_b", {"version": i * 10})

    # File should have 10 entries
    with open(cache_path, "r") as f:
        lines_before = f.readlines()
    assert len(lines_before) == 10

    # Compact the cache
    await cache.compact()

    # File should now have only 2 entries (latest for each key)
    with open(cache_path, "r") as f:
        lines_after = f.readlines()
    assert len(lines_after) == 2

    # Verify latest values are kept
    assert await cache.get("key_a") == {"version": 4}
    assert await cache.get("key_b") == {"version": 40}


@pytest.mark.trio
async def test_stats_method(cache_path):
    """Test cache statistics reporting."""
    cache = StreamingJSONCache(cache_path)
    await cache.load()

    # Add some entries
    await cache.put("key1", {"size": 100})
    await cache.put("key2", {"size": 200})
    await cache.put("key1", {"size": 150})  # Duplicate

    stats = await cache.stats()

    assert stats["unique_keys"] == 2
    assert stats["total_entries"] == 3
    assert stats["file_size"] == cache_path.stat().st_size
    assert "needs_compaction" in stats


@pytest.mark.trio
async def test_clear_cache(cache_path):
    """Test clearing the cache."""
    cache = StreamingJSONCache(cache_path)
    await cache.load()

    # Add entries
    await cache.put("key1", {"data": 1})
    await cache.put("key2", {"data": 2})

    # Clear cache
    await cache.clear()

    # Verify cache is empty
    assert len(cache._index) == 0
    assert "key1" not in cache
    assert "key2" not in cache

    # File should be empty or not exist
    if cache_path.exists():
        assert cache_path.stat().st_size == 0


@pytest.mark.trio
async def test_keys_method(cache_path):
    """Test getting all cache keys."""
    cache = StreamingJSONCache(cache_path)
    await cache.load()

    keys_to_add = ["alpha", "beta", "gamma", "delta"]
    for key in keys_to_add:
        await cache.put(key, {"name": key})

    cache_keys = cache.keys()
    assert set(cache_keys) == set(keys_to_add)


@pytest.mark.trio
async def test_handle_corrupt_cache_file(cache_path):
    """Test handling of corrupted cache file."""
    # Create corrupted cache file
    with open(cache_path, "w") as f:
        f.write('{"key": "valid", "value": {}, "timestamp": 1234}\n')
        f.write("CORRUPT LINE NOT JSON\n")
        f.write('{"key": "valid2", "value": {}, "timestamp": 1235}\n')

    # Should handle corruption gracefully
    cache = StreamingJSONCache(cache_path, handle_corruption=True)
    await cache.load()

    # Should load valid entries
    assert "valid" in cache
    assert "valid2" in cache


@pytest.mark.trio
async def test_memory_efficiency(cache_path):
    """Test that cache doesn't hold values in memory."""
    cache = StreamingJSONCache(cache_path)
    await cache.load()

    # Create large value
    large_value = {"data": "x" * 1000000}  # 1MB string

    await cache.put("large_key", large_value)

    # Index should only have file position, not the value
    assert "large_key" in cache._index
    index_entry = cache._index["large_key"]

    # Index entry should be small (just metadata)
    assert sys.getsizeof(index_entry) < 1000  # Much smaller than 1MB


@pytest.mark.trio
async def test_auto_compact_threshold(cache_path):
    """Test automatic compaction when threshold is reached."""
    # Set low threshold for testing
    cache = StreamingJSONCache(cache_path, compact_threshold=5)
    await cache.load()

    # Write same key multiple times
    for i in range(10):
        await cache.put("key", {"version": i})

    # Should have auto-compacted
    with open(cache_path, "r") as f:
        lines = f.readlines()

    # Should have fewer lines than total writes due to auto-compaction
    assert len(lines) < 10


@pytest.mark.trio
async def test_with_metadata_enricher_pattern(cache_path):
    """Test cache works with MetadataArtificer access patterns."""
    cache = StreamingJSONCache(cache_path)
    await cache.load()

    # Simulate metadata enricher pattern
    image_sha1s = [f"sha1_{i:06d}" for i in range(1000)]

    # First pass - check cache and compute missing
    computed = 0
    for i, sha1 in enumerate(image_sha1s):
        if sha1 not in cache:
            # Simulate computation
            metadata = {
                "alignment": [i % 10, (i + 5) % 10],
                "gain": 1.0 + (i % 100) / 100,
                "mask_mean": 0.9 + (i % 10) / 100,
            }
            await cache.put(sha1, metadata)
            computed += 1

    assert computed == 1000

    # Second pass - should all be cached
    computed = 0
    for sha1 in image_sha1s:
        if sha1 not in cache:
            computed += 1
        else:
            result = await cache.get(sha1)
            assert result is not None

    assert computed == 0


@pytest.mark.trio
async def test_performance_large_dataset(cache_path):
    """Test performance with dataset (1k entries)."""
    cache = StreamingJSONCache(cache_path)
    await cache.load()

    # Measure write performance
    start_time = time.time()
    for i in range(1000):
        await cache.put(f"key_{i:04d}", {"index": i, "data": f"value_{i}"})
    write_time = time.time() - start_time

    print(f"\nWrite time for 1k entries: {write_time:.2f}s")
    assert write_time < 10  # Should complete within 10 seconds (coverage overhead)

    # Measure read performance (random access)
    import random

    keys = [f"key_{random.randint(0, 999):04d}" for _ in range(100)]

    start_time = time.time()
    for key in keys:
        _ = await cache.get(key)
    read_time = time.time() - start_time

    print(f"Random read time for 100 entries: {read_time:.2f}s")
    assert read_time < 0.5  # Should complete within 0.5 seconds


@pytest.mark.trio
async def test_empty_key(cache_path):
    """Test handling of empty string as key."""
    cache = StreamingJSONCache(cache_path)
    await cache.load()

    await cache.put("", {"empty": "key"})
    result = await cache.get("")
    assert result == {"empty": "key"}


@pytest.mark.trio
async def test_unicode_keys_and_values(cache_path):
    """Test handling of unicode in keys and values."""
    cache = StreamingJSONCache(cache_path)
    await cache.load()

    unicode_key = "图像_处理_🎨"
    unicode_value = {"描述": "图像处理", "emoji": "🎨📷"}

    await cache.put(unicode_key, unicode_value)
    result = await cache.get(unicode_key)
    assert result == unicode_value


@pytest.mark.trio
async def test_very_long_key(cache_path):
    """Test handling of very long cache keys."""
    cache = StreamingJSONCache(cache_path)
    await cache.load()

    long_key = "x" * 10000
    await cache.put(long_key, {"long": "key"})

    result = await cache.get(long_key)
    assert result == {"long": "key"}


@pytest.mark.trio
async def test_none_value(cache_path):
    """Test handling of None as value."""
    cache = StreamingJSONCache(cache_path)
    await cache.load()

    await cache.put("none_key", None)
    result = await cache.get("none_key")
    assert result is None


@pytest.mark.trio
async def test_concurrent_read_during_write(cache_path):
    """Test reading while another task is writing."""
    cache = StreamingJSONCache(cache_path)
    await cache.load()

    results = []

    async def writer():
        for i in range(100):
            await cache.put(f"key_{i}", {"value": i})
            await trio.sleep(0.001)  # Small delay to interleave

    async def reader():
        for _ in range(100):
            # Try to read random keys
            for i in range(0, 100, 10):
                key = f"key_{i}"
                if key in cache:
                    results.append(await cache.get(key))
            await trio.sleep(0.001)

    async with trio.open_nursery() as nursery:
        nursery.start_soon(writer)
        nursery.start_soon(reader)

    # Should have successfully read some values
    assert len(results) > 0
