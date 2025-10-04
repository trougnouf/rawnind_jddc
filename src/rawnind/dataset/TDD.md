### Async Streaming File Discovery with Event-Based Cache Management

## Objective

Implement a true async streaming pipeline for discovering and downloading missing dataset files that:
1. **Uses event-based cache invalidation** for targeted, automatic cache management
2. **Maintains separation of concerns** between filesystem examination and missing file queries
3. **Streams results incrementally** rather than processing entire datasets in batches
4. **Avoids redundant filesystem scans** by trusting cached state
5. **Uses async I/O appropriately** (Trio-compatible) for network operations while keeping filesystem checks synchronous
6. **Preserves the existing API contract** for backward compatibility

## Context

The existing `DatasetIndex` class has clean separation:
- `discover_local_files()` examines the filesystem and updates `img_info.local_path`
- `get_missing_files()` returns a list of items where `local_path is None`

We want to extend this with:
- **Event-based cache invalidation** replacing manual `_invalidate_caches()` calls
- An async method to download missing files with concurrency control
- A streaming interface that yields missing files as they're discovered
- Progress reporting during downloads
- Trio-compatible async implementation

---

## Implementation Requirements

### Phase 0: Event-Based Cache Infrastructure (TDD)

#### Test 0.1: `test_event_emission_on_index_structure_change`
**Given:** A `DatasetIndex` instance  
**When:** `_build_index_from_data()` is called  
**Then:** `INDEX_STRUCTURE_CHANGED` event should be emitted

```python
def test_event_emission_on_index_structure_change():
    """Test that rebuilding index emits appropriate event."""
    index = DatasetIndex()
    
    event_fired = False
    
    def listener():
        nonlocal event_fired
        event_fired = True
    
    index._events.on(CacheEvent.INDEX_STRUCTURE_CHANGED, listener)
    
    # Load index (triggers _build_index_from_data)
    index.load_index()
    
    assert event_fired
```

#### Test 0.2: `test_cache_invalidation_on_event`
**Given:** A cached value that depends on specific events  
**When:** The relevant event is emitted  
**Then:** The cache should be invalidated

```python
def test_cache_invalidation_on_event():
    """Test that caches invalidate when relevant events fire."""
    index = DatasetIndex()
    index.load_index()
    
    # Prime cache
    _ = index.known_extensions
    assert index._known_extensions is not None
    
    # Emit event that should invalidate this cache
    index._emit(CacheEvent.LOCAL_PATHS_UPDATED)
    
    # Cache should be invalidated
    assert index._known_extensions is None
```

#### Test 0.3: `test_targeted_cache_invalidation`
**Given:** Multiple cached values with different dependencies  
**When:** A specific event is emitted  
**Then:** Only caches dependent on that event should be invalidated

```python
def test_targeted_cache_invalidation():
    """Test that only affected caches are invalidated."""
    index = DatasetIndex()
    index.load_index()
    
    # Prime both caches
    extensions = index.known_extensions
    cfa_types = index.sorted_cfa_types
    
    assert index._known_extensions is not None
    assert index._sorted_cfa_types is not None
    
    # Emit LOCAL_PATHS_UPDATED (should not affect sorted_cfa_types)
    index._emit(CacheEvent.LOCAL_PATHS_UPDATED)
    
    # known_extensions should be invalidated
    assert index._known_extensions is None
    
    # sorted_cfa_types should still be cached
    assert index._sorted_cfa_types is not None
    assert index._sorted_cfa_types == cfa_types
```

#### Implementation 0: Event System

Add the following components to `manager.py`:

```python
from enum import Enum, auto
from typing import Callable, Dict, Set, Any
from functools import wraps

class CacheEvent(Enum):
    """Events that can trigger cache invalidation."""
    INDEX_STRUCTURE_CHANGED = auto()  # Scenes added/removed, index rebuilt
    LOCAL_PATHS_UPDATED = auto()      # File discovery updated local_path values
    FILE_VALIDATED = auto()           # File validation state changed
    METADATA_CHANGED = auto()         # Scene metadata updated


class EventEmitter:
    """Simple synchronous event emitter (Trio-safe)."""
    
    def __init__(self):
        self._listeners: Dict[CacheEvent, Set[Callable]] = {
            event: set() for event in CacheEvent
        }
    
    def on(self, event: CacheEvent, callback: Callable) -> None:
        """Register a callback for an event.
        
        Args:
            event: The event to listen for
            callback: Function to call when event fires (takes no arguments)
        """
        self._listeners[event].add(callback)
    
    def off(self, event: CacheEvent, callback: Callable) -> None:
        """Unregister a callback.
        
        Args:
            event: The event to stop listening for
            callback: The callback to remove
        """
        self._listeners[event].discard(callback)
    
    def emit(self, event: CacheEvent, **kwargs: Any) -> None:
        """Emit an event synchronously (safe in Trio context).
        
        Args:
            event: The event to emit
            **kwargs: Optional context data (currently unused)
        """
        for callback in self._listeners[event]:
            callback()


def emits_event(event: CacheEvent) -> Callable:
    """Decorator to emit an event after synchronous method execution.
    
    Args:
        event: The event to emit after method completes
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self: 'DatasetIndex', *args: Any, **kwargs: Any) -> Any:
            result = func(self, *args, **kwargs)
            self._emit(event)
            return result
        return wrapper
    return decorator


def emits_event_async(event: CacheEvent) -> Callable:
    """Decorator to emit an event after async method execution (Trio-compatible).
    
    Args:
        event: The event to emit after method completes
    
    Returns:
        Decorator function for async methods
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(self: 'DatasetIndex', *args: Any, **kwargs: Any) -> Any:
            result = await func(self, *args, **kwargs)
            self._emit(event)  # Synchronous emit is safe in Trio
            return result
        return wrapper
    return decorator
```

Update `DatasetIndex.__init__`:

```python
class DatasetIndex:
    """Canonical index of the RawNIND dataset with event-based cache invalidation."""

    def __init__(self, cache_path: Optional[Path] = None, dataset_root: Optional[Path] = None):
        """Initialize dataset index.

        Args:
            cache_path: Path to cached dataset.yaml (default: DATASET_ROOT/dataset_index.yaml)
            dataset_root: Root directory for dataset files
        """
        self.dataset_root = dataset_root or DATASET_ROOT
        self.cache_path = cache_path or self.dataset_root / "dataset_index.yaml"
        self.scenes: Dict[str, Dict[str, SceneInfo]] = {}
        self._loaded = False
        
        # Cached values
        self._known_extensions: Optional[Set[str]] = None
        self._sorted_cfa_types: Optional[List[str]] = None
        
        # Event system
        self._events = EventEmitter()
        self._setup_cache_listeners()
    
    def _setup_cache_listeners(self) -> None:
        """Register callbacks to invalidate specific caches on relevant events."""
        # known_extensions depends on both local paths and index structure
        self._events.on(
            CacheEvent.LOCAL_PATHS_UPDATED,
            lambda: setattr(self, '_known_extensions', None)
        )
        self._events.on(
            CacheEvent.INDEX_STRUCTURE_CHANGED,
            lambda: setattr(self, '_known_extensions', None)
        )
        
        # sorted_cfa_types only depends on index structure
        self._events.on(
            CacheEvent.INDEX_STRUCTURE_CHANGED,
            lambda: setattr(self, '_sorted_cfa_types', None)
        )
    
    def _emit(self, event: CacheEvent, **kwargs: Any) -> None:
        """Emit an event (internal helper).
        
        Args:
            event: Event to emit
            **kwargs: Optional context
        """
        self._events.emit(event, **kwargs)
```

Update `_build_index_from_data`:

```python
@emits_event(CacheEvent.INDEX_STRUCTURE_CHANGED)
def _build_index_from_data(self, dataset_data: dict) -> None:
    """Build index from parsed dataset YAML data.
    
    Emits INDEX_STRUCTURE_CHANGED event upon completion.
    """
    # ... existing implementation ...
    # (Remove any manual _invalidate_caches() calls)
```

---

### Phase 1: Foundation - Simple Async Download with Events (TDD)

#### Test 1.1: `test_discover_local_files_emits_event`
**Given:** A `DatasetIndex` instance  
**When:** `discover_local_files()` is called  
**Then:** `LOCAL_PATHS_UPDATED` event should be emitted

```python
def test_discover_local_files_emits_event():
    """Test that discover_local_files emits LOCAL_PATHS_UPDATED event."""
    index = DatasetIndex()
    index.load_index()
    
    event_fired = False
    
    def listener():
        nonlocal event_fired
        event_fired = True
    
    index._events.on(CacheEvent.LOCAL_PATHS_UPDATED, listener)
    index.discover_local_files()
    
    assert event_fired
```

#### Test 1.2: `test_async_download_preserves_index_state`
**Given:** A `DatasetIndex` with some files marked as locally available and some missing  
**When:** `async_download_missing_files()` is called  
**Then:** 
- Only files with `local_path is None` should be attempted for download
- Files already marked as available should not be re-checked or re-downloaded
- After download completes, `discover_local_files()` should be called to update state
- `LOCAL_PATHS_UPDATED` event should be emitted

```python
async def test_async_download_preserves_index_state():
    index = DatasetIndex()
    index.load_index()
    
    # Mark some files as available
    initial_available = []
    for scene in index.get_all_scenes()[:3]:
        if scene.clean_images:
            scene.clean_images[0].local_path = Path("/fake/path")
            initial_available.append(scene.clean_images[0])
    
    initial_missing = index.get_missing_files()
    
    event_count = 0
    def listener():
        nonlocal event_count
        event_count += 1
    
    index._events.on(CacheEvent.LOCAL_PATHS_UPDATED, listener)
    
    # Mock download (don't actually download)
    with patch('requests.get'):
        await index.async_download_missing_files(max_concurrent=2)
    
    # Files marked as available should not have been touched
    for img in initial_available:
        assert img.local_path == Path("/fake/path")
    
    # Event should have been emitted at least once
    assert event_count > 0
```

#### Test 1.3: `test_async_download_respects_concurrency_limit`
**Given:** 10 missing files and `max_concurrent=3`  
**When:** Downloads are initiated  
**Then:** No more than 3 concurrent downloads should be active at any time

```python
async def test_async_download_respects_concurrency_limit():
    index = DatasetIndex()
    index.load_index()
    
    active_downloads = []
    max_seen_concurrent = [0]  # Use list to allow modification in closure
    
    async def mock_download(url, path):
        active_downloads.append(path)
        max_seen_concurrent[0] = max(max_seen_concurrent[0], len(active_downloads))
        await trio.sleep(0.1)  # Simulate download time
        active_downloads.remove(path)
    
    with patch('DatasetIndex._download_file', mock_download):
        await index.async_download_missing_files(max_concurrent=3)
    
    assert max_seen_concurrent[0] <= 3
```

#### Implementation 1: `discover_local_files()` with Event Emission

```python
@emits_event(CacheEvent.LOCAL_PATHS_UPDATED)
def discover_local_files(self) -> Tuple[int, int]:
    """Discover local files and update local_path attributes.
    
    Emits LOCAL_PATHS_UPDATED event upon completion.
    
    Returns:
        Tuple of (found_count, total_count)
    """
    if not self._loaded:
        self.load_index()
    
    found_count = 0
    total_count = 0
    
    for cfa_type, scenes in self.scenes.items():
        cfa_dir = self.dataset_root / cfa_type
        if not cfa_dir.exists():
            continue
        
        for scene_name, scene_info in scenes.items():
            for img_info in scene_info.all_images():
                total_count += 1
                
                candidates = self._candidate_paths(cfa_type, scene_name, img_info)
                
                for candidate in candidates:
                    if candidate.exists():
                        img_info.local_path = candidate
                        found_count += 1
                        break
    
    return found_count, total_count
    # Event emitted automatically by decorator
```

#### Implementation 1b: Basic `async_download_missing_files()`

Requirements:
- Accept parameters: `max_concurrent: int = 5`, `progress: bool = True`
- Call `discover_local_files()` **once before** to refresh state (emits event automatically)
- Get missing files via `get_missing_files()` - **do not re-examine filesystem**
- Use `trio.Semaphore` to limit concurrent downloads
- Download files to their designated paths
- Call `discover_local_files()` **once after** to update state (emits event automatically)
- Handle network errors gracefully (log and continue)
- Optional: Display progress bar using `tqdm` if `progress=True`

```python
async def async_download_missing_files(
    self, 
    max_concurrent: int = 5,
    progress: bool = True
) -> Tuple[int, int]:
    """Download missing files with concurrency control.
    
    Args:
        max_concurrent: Maximum concurrent downloads
        progress: Show progress bar
    
    Returns:
        Tuple of (successful_downloads, failed_downloads)
    """
    if not self._loaded:
        self.load_index()
    
    # Refresh state (emits LOCAL_PATHS_UPDATED automatically)
    self.discover_local_files()
    
    missing = self.get_missing_files()
    
    if not missing:
        return 0, 0
    
    successful = 0
    failed = 0
    
    async with trio.open_nursery() as nursery:
        semaphore = trio.Semaphore(max_concurrent)
        
        pbar = tqdm(total=len(missing), desc="Downloading", unit="file") if progress else None
        
        async def download_task(img: ImageInfo):
            nonlocal successful, failed
            async with semaphore:
                try:
                    await self._download_file(img.download_url, img.local_path)
                    successful += 1
                except Exception as e:
                    logger.error(f"Failed to download {img.filename}: {e}")
                    failed += 1
                finally:
                    if pbar:
                        pbar.update(1)
        
        for img_info in missing:
            nursery.start_soon(download_task, img_info)
        
        if pbar:
            pbar.close()
    
    # Refresh to verify downloads (emits LOCAL_PATHS_UPDATED automatically)
    self.discover_local_files()
    
    return successful, failed
```

**Style requirements:**
- Use `async with trio.open_nursery()` for structured concurrency
- Do **not** wrap filesystem operations in `trio.to_thread.run_sync()`
- Let decorators handle event emission automatically

---

### Phase 2: Streaming Discovery (TDD)

#### Test 2.1: `test_iter_missing_files_is_side_effect_free`
**Given:** A `DatasetIndex` with cached state  
**When:** `iter_missing_files()` is called multiple times  
**Then:** 
- Should return the same results each time
- Should not modify `img_info.local_path`
- Should not trigger filesystem checks
- Should not emit events

```python
def test_iter_missing_files_is_side_effect_free():
    index = DatasetIndex()
    index.load_index()
    index.discover_local_files()
    
    event_count = 0
    def listener():
        nonlocal event_count
        event_count += 1
    
    # Listen to all events
    for event in CacheEvent:
        index._events.on(event, listener)
    
    # Get missing files twice
    missing1 = list(index.iter_missing_files())
    missing2 = list(index.iter_missing_files())
    
    # Should be identical
    assert missing1 == missing2
    
    # No events should have been emitted
    assert event_count == 0
    
    # No filesystem operations should occur
    with patch('pathlib.Path.exists') as mock_exists:
        list(index.iter_missing_files())
        mock_exists.assert_not_called()
```

#### Test 2.2: `test_iter_missing_files_yields_incrementally`
**Given:** A large dataset with many missing files  
**When:** `iter_missing_files()` is called  
**Then:** Results should be yielded one at a time (generator behavior)

```python
def test_iter_missing_files_yields_incrementally():
    index = DatasetIndex()
    index.load_index()
    
    # Consume first item without materializing full list
    gen = index.iter_missing_files()
    first = next(gen)
    
    assert isinstance(first, ImageInfo)
    assert first.local_path is None
```

#### Implementation 2: Refactor `iter_missing_files()`

```python
def iter_missing_files(self) -> Generator[ImageInfo, None, None]:
    """Yield files not found locally based on cached state.
    
    This method is side-effect free and relies on cached local_path values.
    Call discover_local_files() first to refresh cached state.
    
    Does not emit any events or modify state.
    
    Yields:
        ImageInfo objects where local_path is None
    """
    if not self._loaded:
        self.load_index()
    
    for cfa_type, scenes in self.scenes.items():
        for scene_name, scene_info in scenes.items():
            for img_info in scene_info.all_images():
                if img_info.local_path is None:
                    yield img_info
```

**Key changes:**
- Remove all filesystem checks
- Remove state updates
- Simple query: yield items where `local_path is None`
- No event emission
- Document that this relies on cached state

---

### Phase 3: Streaming Async Discovery with Events (Advanced TDD)

#### Test 3.1: `test_async_iter_missing_files_streams_during_discovery`
**Given:** A dataset with mixed available/missing files  
**When:** `async_iter_missing_files()` examines filesystem  
**Then:** Missing files should be yielded **before** entire filesystem scan completes

```python
async def test_async_iter_missing_files_streams_during_discovery():
    index = DatasetIndex()
    index.load_index()
    
    yielded_count = 0
    filesystem_scan_complete = False
    
    async for img_info in index.async_iter_missing_files():
        yielded_count += 1
        assert not filesystem_scan_complete, "Should stream before scan completes"
        if yielded_count >= 5:
            break
    
    # Verify we got results before full scan
    assert yielded_count == 5
```

#### Test 3.2: `test_async_iter_missing_files_emits_event`
**Given:** Files that don't exist locally  
**When:** `async_iter_missing_files()` completes  
**Then:** `LOCAL_PATHS_UPDATED` event should be emitted

```python
async def test_async_iter_missing_files_emits_event():
    index = DatasetIndex()
    index.load_index()
    
    event_fired = False
    
    def listener():
        nonlocal event_fired
        event_fired = True
    
    index._events.on(CacheEvent.LOCAL_PATHS_UPDATED, listener)
    
    # Consume generator
    async for img_info in index.async_iter_missing_files():
        break  # Just need to trigger it
    
    # Event should be emitted after completion
    # Note: Event emits after generator exhausts, so consume fully
    async for _ in index.async_iter_missing_files():
        pass
    
    assert event_fired
```

#### Test 3.3: `test_async_iter_missing_files_updates_cached_state`
**Given:** Files that don't exist locally  
**When:** `async_iter_missing_files()` runs  
**Then:** `img_info.local_path` should be set to expected path (for download target)

```python
async def test_async_iter_missing_files_updates_cached_state():
    index = DatasetIndex()
    index.load_index()
    
    # Ensure no local files exist
    for scene in index.get_all_scenes():
        for img in scene.all_images():
            img.local_path = None
    
    missing = []
    async for img_info in index.async_iter_missing_files():
        missing.append(img_info)
        # Should have designated path for download
        assert img_info.local_path is not None
        assert isinstance(img_info.local_path, Path)
        break
    
    # Cached state should be updated
    assert missing[0].local_path is not None
```

#### Implementation 3: `async_iter_missing_files()`

This is the **streaming async discovery** method with event emission:

```python
async def async_iter_missing_files(self) -> AsyncGenerator[ImageInfo, None]:
    """Asynchronously examine filesystem and yield missing files as discovered.
    
    This method:
    - Checks candidate paths for each file
    - Updates img_info.local_path (either to existing file or designated download target)
    - Yields ImageInfo objects that are missing (no existing file found)
    - Emits LOCAL_PATHS_UPDATED event after completion
    
    This is the recommended way to discover and download files in a streaming fashion.
    
    Yields:
        ImageInfo objects for files not found locally
    """
    if not self._loaded:
        self.load_index()
    
    try:
        for cfa_type, scenes in self.scenes.items():
            cfa_dir = self.dataset_root / cfa_type
            # Filesystem check is synchronous (fast, no await needed)
            if not cfa_dir.exists():
                continue
            
            for scene_name, scene_info in scenes.items():
                for img_info in scene_info.all_images():
                    # Get candidate paths
                    candidates = self._candidate_paths(cfa_type, scene_name, img_info)
                    
                    # Check if file exists in any candidate location (synchronous)
                    found = False
                    for candidate in candidates:
                        if candidate.exists():
                            img_info.local_path = candidate
                            found = True
                            break
                    
                    if not found:
                        # Set to primary candidate as download target
                        img_info.local_path = candidates[0] if candidates else None
                        yield img_info
                    
                    # Yield control to Trio scheduler (not for I/O)
                    await trio.sleep(0)
    finally:
        # Emit event after completion (even if interrupted)
        self._emit(CacheEvent.LOCAL_PATHS_UPDATED)
```

**Key features:**
- Streams results incrementally
- Updates `img_info.local_path` as side effect (acceptable for discovery)
- Sets `local_path` even for missing files (download target)
- Uses `await trio.sleep(0)` to yield control (Trio-compatible)
- Emits `LOCAL_PATHS_UPDATED` event in finally block
- Filesystem checks stay synchronous (no `trio.to_thread.run_sync`)

---

### Phase 4: Integration - Streaming Pipeline with Events (TDD)

#### Test 4.1: `test_streaming_download_pipeline`
**Given:** Empty dataset directory  
**When:** `async_download_missing_files()` uses `async_iter_missing_files()`  
**Then:** 
- Files should be downloaded as they're discovered missing
- Progress should be displayed
- No redundant filesystem scans
- Events should be emitted appropriately

```python
async def test_streaming_download_pipeline(tmp_path):
    index = DatasetIndex(dataset_root=tmp_path)
    index.load_index()
    
    downloads_started = []
    event_count = 0
    
    def listener():
        nonlocal event_count
        event_count += 1
    
    index._events.on(CacheEvent.LOCAL_PATHS_UPDATED, listener)
    
    async def mock_download(url, path):
        downloads_started.append(path.name)
        await trio.sleep(0.05)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake")
    
    with patch('DatasetIndex._download_file', mock_download):
        await index.async_download_missing_files(max_concurrent=3)
    
    # Verify downloads happened
    assert len(downloads_started) > 0
    
    # Event should have been emitted (from async_iter_missing_files)
    assert event_count > 0
```

#### Test 4.2: `test_streaming_pipeline_cache_invalidation`
**Given:** Cached values before download  
**When:** `async_download_missing_files()` completes  
**Then:** Caches dependent on `LOCAL_PATHS_UPDATED` should be invalidated

```python
async def test_streaming_pipeline_cache_invalidation():
    index = DatasetIndex()
    index.load_index()
    
    # Prime cache
    _ = index.known_extensions
    assert index._known_extensions is not None
    
    # Mock download
    with patch('DatasetIndex._download_file'):
        await index.async_download_missing_files(max_concurrent=2)
    
    # Cache should have been invalidated by LOCAL_PATHS_UPDATED event
    assert index._known_extensions is None
```

#### Implementation 4: Refactor `async_download_missing_files()` to use streaming

```python
async def async_download_missing_files(
    self, 
    max_concurrent: int = 5,
    progress: bool = True
) -> Tuple[int, int]:
    """Download missing files with concurrency control and progress reporting.
    
    Uses streaming discovery via async_iter_missing_files(), which automatically
    emits LOCAL_PATHS_UPDATED event upon completion.
    
    Args:
        max_concurrent: Maximum concurrent downloads
        progress: Show progress bar
    
    Returns:
        Tuple of (successful_downloads, failed_downloads)
    """
    if not self._loaded:
        self.load_index()
    
    successful = 0
    failed = 0
    
    async with trio.open_nursery() as nursery:
        semaphore = trio.Semaphore(max_concurrent)
        
        # Optional progress bar
        pbar = tqdm(desc="Downloading", unit="file") if progress else None
        
        # Stream missing files as they're discovered
        # Event emission happens automatically in async_iter_missing_files
        async for img_info in self.async_iter_missing_files():
            async def download_task(img: ImageInfo):
                nonlocal successful, failed
                async with semaphore:
                    try:
                        await self._download_file(img.download_url, img.local_path)
                        successful += 1
                    except Exception as e:
                        logger.error(f"Failed to download {img.filename}: {e}")
                        failed += 1
                    finally:
                        if pbar:
                            pbar.update(1)
            
            nursery.start_soon(download_task, img_info)
        
        if pbar:
            pbar.close()
    
    # No need for explicit discover_local_files() here
    # async_iter_missing_files() already emitted LOCAL_PATHS_UPDATED
    # If you want to re-verify downloaded files:
    # self.discover_local_files()  # This would emit the event again
    
    return successful, failed
```

**Key features:**
- Uses `async_iter_missing_files()` for streaming discovery
- Starts downloads as files are discovered (true streaming)
- Semaphore controls concurrency
- Progress bar tracks completion
- Event emission handled automatically by `async_iter_missing_files()`
- No manual cache invalidation needed

---

### Smoke Test

After implementing and testing with unit tests, validate with real dataset:

```python
# smoke_test.py
import trio
from rawnind.dataset.manager import DatasetIndex, CacheEvent

async def smoke_test():
    """Live test with actual RawNIND dataset."""
    index = DatasetIndex()
    
    # Optional: Register event listener for debugging
    def on_cache_event():
        print("  [Event] Cache invalidated")
    
    for event in CacheEvent:
        index._events.on(event, on_cache_event)
    
    print("Loading index...")
    index.load_index()
    
    print("\nInitial state:")
    index.print_summary()
    
    print("\nDiscovering local files...")
    found, total = index.discover_local_files()
    print(f"Found {found}/{total} files")
    
    print("\nChecking for missing files (streaming)...")
    missing_count = 0
    async for img_info in index.async_iter_missing_files():
        missing_count += 1
        if missing_count <= 5:
            print(f"  Missing: {img_info.filename}")
    print(f"Total missing: {missing_count}")
    
    if missing_count > 0:
        print(f"\nDownload first 10 missing files? (y/n)")
        response = input()
        if response.lower() == 'y':
            # Download with streaming pipeline
            successful, failed = await index.async_download_missing_files(
                max_concurrent=3,
                progress=True
            )
            print(f"\nDownloaded: {successful} successful, {failed} failed")
    
    print("\nFinal state:")
    index.print_summary()

if __name__ == "__main__":
    trio.run(smoke_test)
```

---

## Architecture, Style; Patterns & Anti-Patterns

### DO:
1. **Use event-based cache invalidation**: Let decorators and event listeners handle cache management
2. **Emit events from state-changing methods**: Use `@emits_event` and `@emits_event_async`
3. **Separate concerns**: `discover_local_files()` examines filesystem; `get_missing_files()` queries cached state
4. **Stream incrementally**: Use generators/async generators to yield results as they're produced
5. **Trust cached state**: Query methods should not trigger filesystem checks or emit events
6. **Use async for I/O**: Network downloads are async; filesystem checks stay synchronous
7. **Document side effects**: Make it clear when methods update cached state and which events they emit

### DON'T:
1. **Manually invalidate caches**: Don't call `_invalidate_caches()` - use events
2. **Mix responsibilities**: Don't examine filesystem inside query methods
3. **Wrap unnecessary operations**: Don't make filesystem checks async with `trio.to_thread.run_sync()`
4. **Redundant scans**: Don't re-examine filesystem multiple times in a single workflow
5. **Hidden side effects**: Don't update instance state in methods that look like queries
6. **Materialize unnecessarily**: Don't convert generators to lists unless needed
7. **Emit events from queries**: Only state-changing methods should emit events

---

## Trio-Specific Nuances

### ✅ Trio Best Practices:
1. **Synchronous callbacks are safe**: No race conditions in single-threaded Trio
2. **Emit events synchronously**: Even in async methods, use `self._emit(event)`
3. **Keep filesystem checks sync**: `Path.exists()` is fast, don't wrap in async
4. **Use `await trio.sleep(0)`**: Yield control in long-running loops
5. **Use `trio.Semaphore`**: Control concurrency for async operations
6. **Use `async with trio.open_nursery()`**: Structured concurrency for spawning tasks
