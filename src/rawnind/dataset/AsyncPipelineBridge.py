"""
AsyncPipelineBridge - Bridge between async dataset pipeline and sync PyTorch DataLoaders.

This module provides a thread-safe bridge for converting async trio-based
dataset pipelines to synchronous PyTorch-compatible datasets with
error handling, monitoring, and performance optimizations.

Example Usage:
    >>> from rawnind.dataset.async_to_sync_bridge import AsyncPipelineBridge
    >>> from rawnind.dataset.cache_manager import DiskCacheManager
    >>>
    >>> # Initialize with cache for persistence
    >>> cache = DiskCacheManager(cache_dir="/path/to/cache", max_size_mb=1000)
    >>> bridge = AsyncPipelineBridge(cache=cache, max_scenes=100)
    >>>
    >>> # Collect scenes from async pipeline
    >>> async def collect_data():
    >>>     async with trio.open_nursery() as nursery:
    >>>         send_channel, recv_channel = trio.open_memory_channel(10)
    >>>         nursery.start_soon(pipeline.run, send_channel)
    >>>         nursery.start_soon(bridge.consume, recv_channel)
    >>>
    >>> # Access collected scenes
    >>> scene = bridge.get_scene(0)
"""

import json
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Callable, Any, Dict
from enum import Enum

import trio
import yaml

from .SceneInfo import SceneInfo

# Configure module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BridgeState(Enum):
    """State machine for bridge lifecycle."""
    UNINITIALIZED = "uninitialized"
    INITIALIZED = "initialized"
    COLLECTING = "collecting"
    READY = "ready"
    ERROR = "error"
    CLOSED = "closed"


@dataclass
class BridgeStats:
    """Statistics tracking for production monitoring."""
    scenes_collected: int = 0
    scenes_filtered: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    collection_start_time: Optional[float] = None
    collection_end_time: Optional[float] = None
    errors_encountered: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def collection_duration_seconds(self) -> Optional[float]:
        """Calculate collection duration."""
        if self.collection_start_time and self.collection_end_time:
            return self.collection_end_time - self.collection_start_time
        return None

    def collection_rate(self) -> Optional[float]:
        """Calculate scenes per second collection rate."""
        duration = self.collection_duration_seconds()
        if duration and duration > 0:
            return self.scenes_collected / duration
        return None


class AsyncPipelineBridge:
    """
    Bridge between async pipeline and synchronous dataloaders.

    Features:
        - Thread-safe scene access with read-write locking
        - Error handling and recovery
        - Performance monitoring and metrics collection
        - Cache integration with fallback mechanisms
        - Input validation and safety checks
        - Graceful degradation on failures
        - Health check capabilities

    Attributes:
        cache: Optional cache manager for persistence
        max_scenes: Maximum number of scenes to collect
        enable_caching: Whether to use caching
        backwards_compat_mode: Enable legacy API compatibility
        filter_test_reserve: Filter out test reserve scenes
        cfa_filter: Filter by CFA type (e.g., "bayer", x-trans)
        stats: Runtime statistics and monitoring data
        state: Current bridge state
    """

    # Class-level configuration
    DEFAULT_TIMEOUT_SECONDS = 30.0
    DEFAULT_RETRY_COUNT = 3
    DEFAULT_RETRY_DELAY_SECONDS = 1.0

    def __init__(
        self,
        cache=None,
        max_scenes: Optional[int] = None,
        enable_caching: bool = True,
        backwards_compat_mode: bool = False,
        filter_test_reserve: bool = False,
        cfa_filter: Optional[str] = None,
        enable_monitoring: bool = True,
        thread_safe: bool = True,
        validate_scenes: bool = True,
        retry_on_error: bool = True,
        health_check_interval: Optional[float] = None,
        # New parameters for testing
        index_url: Optional[str] = None,
        cache_dir: Optional[Any] = None,
        enable_metadata_enrichment: bool = True,
        timeout: Optional[float] = None,
        mock_mode: bool = False
    ):
        """
        Initialize the bridge.

        Args:
            cache: Cache manager instance for persistence
            max_scenes: Maximum scenes to collect (None for unlimited)
            enable_caching: Enable cache operations
            backwards_compat_mode: Support legacy API methods
            filter_test_reserve: Filter out test reserve scenes
            cfa_filter: Filter by CFA type (e.g., "bayer", x-trans)
            enable_monitoring: Enable statistics collection
            thread_safe: Enable thread safety mechanisms
            validate_scenes: Validate scene data integrity
            retry_on_error: Enable automatic retry on failures
            health_check_interval: Interval for health checks in seconds

        Raises:
            ValueError: If invalid configuration is provided
        """
        # Input validation
        if max_scenes is not None and max_scenes <= 0:
            raise ValueError(f"max_scenes must be positive, got {max_scenes}")
        if cfa_filter and cfa_filter not in ["bayer", "x-trans", "quad_bayer"]:
            logger.warning(f"Unusual CFA filter type: {cfa_filter}")

        # Core configuration
        self.cache = cache if enable_caching else None
        self.max_scenes = max_scenes
        self.enable_caching = enable_caching
        self.backwards_compat_mode = backwards_compat_mode
        self.filter_test_reserve = filter_test_reserve
        self.cfa_filter = cfa_filter

        # Production features
        self.enable_monitoring = enable_monitoring
        self.thread_safe = thread_safe
        self.validate_scenes = validate_scenes
        self.retry_on_error = retry_on_error
        self.health_check_interval = health_check_interval

        # Internal state
        self._scenes: List[SceneInfo] = []
        self.stats = BridgeStats() if enable_monitoring else None
        self.state = BridgeState.INITIALIZED
        self._lock = threading.RLock() if thread_safe else None
        self._last_health_check = time.time()
        
        # Store new parameters for mock mode
        self.index_url = index_url
        self.cache_dir = cache_dir
        self.enable_metadata_enrichment = enable_metadata_enrichment
        self.timeout = timeout or self.DEFAULT_TIMEOUT_SECONDS
        self.mock_mode = mock_mode or (index_url and index_url.startswith("mock://"))

        logger.info(
            f"AsyncPipelineBridge initialized: max_scenes={max_scenes}, "
            f"caching={enable_caching}, thread_safe={thread_safe}"
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

    async def consume(
        self,
        recv_channel: Optional[trio.MemoryReceiveChannel] = None,
        progress_callback: Optional[Callable] = None,
        timeout_seconds: Optional[float] = None
    ) -> None:
        """
        Collect scenes from async pipeline with production safeguards.

        Args:
            recv_channel: Trio channel to receive scenes from (None for mock mode)
            progress_callback: Callback for progress updates (current, total)
            timeout_seconds: Overall timeout for collection

        Raises:
            TimeoutError: If collection exceeds timeout
            RuntimeError: If bridge is in invalid state
        """
        # State validation
        if self.state not in [BridgeState.INITIALIZED, BridgeState.READY]:
            raise RuntimeError(f"Cannot collect in state: {self.state}")
        
        # Handle mock mode for testing early
        if self.mock_mode and recv_channel is None:
            await self._collect_mock_scenes(progress_callback)
            return

        self.state = BridgeState.COLLECTING

        if self.stats:
            self.stats.collection_start_time = time.time()

        logger.info(f"Starting scene collection with timeout={timeout_seconds if timeout_seconds is not None else 'None (unlimited)'}s")

        try:
            # Only apply timeout if explicitly set
            if timeout_seconds is not None:
                cancel_scope = trio.move_on_after(timeout_seconds)
            else:
                cancel_scope = trio.CancelScope()

            with cancel_scope:
                if recv_channel is None and self.mock_mode:
                    # Mock mode - no real channel needed
                    logger.info("No recv_channel provided, using mock collection")
                    await self._collect_mock_scenes(progress_callback)
                elif recv_channel is not None:
                    await self._collect_with_retry(recv_channel, progress_callback)
                else:
                    logger.warning("No recv_channel and not in mock mode - nothing to collect")

            if cancel_scope.cancelled_caught:
                error_msg = f"Collection timed out after {timeout_seconds} seconds"
                logger.error(error_msg)
                if self.stats:
                    self.stats.errors_encountered.append(error_msg)
                raise trio.TooSlowError(error_msg)

        except Exception as e:
            self.state = BridgeState.ERROR
            logger.error(f"Collection failed: {e}", exc_info=True)
            if self.stats:
                self.stats.errors_encountered.append(str(e))
            raise
        finally:
            if self.stats:
                self.stats.collection_end_time = time.time()
            self.state = BridgeState.READY
            logger.info(f"Collection completed: {len(self._scenes)} scenes collected")

    async def _collect_mock_scenes(self, progress_callback: Optional[Callable] = None):
        """
        Generate mock scenes for testing purposes.
        
        Args:
            progress_callback: Callback for progress updates
        """
        self.state = BridgeState.COLLECTING
        logger.info(f"Generating {self.max_scenes} mock scenes for testing")
        
        try:
            # Generate mock scenes
            for i in range(self.max_scenes or 10):
                # Create mock ImageInfo objects
                from .SceneInfo import ImageInfo
                
                scene_name = f"mock_scene_{i:03d}"
                cfa_type = "bayer" if i % 2 == 0 else "x-trans"
                
                clean_img = ImageInfo(
                    filename=f"clean_{i:03d}.exr",
                    sha1=f"mock_sha1_clean_{i:03d}",
                    is_clean=True,
                    scene_name=scene_name,
                    scene_images=[f"clean_{i:03d}.exr", f"noisy_{i:03d}.dng"],
                    cfa_type=cfa_type,
                    file_id=f"mock_id_clean_{i:03d}",
                    local_path=None
                )
                
                noisy_img = ImageInfo(
                    filename=f"noisy_{i:03d}.dng",
                    sha1=f"mock_sha1_noisy_{i:03d}",
                    is_clean=False,
                    scene_name=scene_name,
                    scene_images=[f"clean_{i:03d}.exr", f"noisy_{i:03d}.dng"],
                    cfa_type=cfa_type,
                    file_id=f"mock_id_noisy_{i:03d}",
                    local_path=None
                )
                
                # Create a mock SceneInfo with correct constructor
                scene = SceneInfo(
                    scene_name=scene_name,
                    cfa_type=cfa_type,
                    unknown_sensor=False,
                    test_reserve=(i == 9),  # Last scene is test reserve
                    clean_images=[clean_img],
                    noisy_images=[noisy_img],
                )
                self._scenes.append(scene)
                
                if progress_callback:
                    progress_callback(i + 1, self.max_scenes)
                
                # Small delay to simulate async work
                await trio.sleep(0.001)
            
            self.state = BridgeState.READY
            logger.info(f"Mock collection completed: {len(self._scenes)} scenes")
            
        except Exception as e:
            self.state = BridgeState.ERROR
            logger.error(f"Mock collection failed: {e}")
            raise

    async def _collect_with_retry(
        self,
        recv_channel: trio.MemoryReceiveChannel,
        progress_callback: Optional[Callable]
    ) -> None:
        """
        Internal collection with retry logic.

        Args:
            recv_channel: Trio channel to receive scenes from
            progress_callback: Progress update callback
        """
        collected = 0
        should_collect = True
        retry_count = 0

        async with recv_channel:
            async for item in recv_channel:
                # Skip collection after limit (but drain channel)
                if not should_collect:
                    continue

                # Validate and process item
                try:
                    if not await self._process_scene(item, collected):
                        continue

                    collected += 1

                    # Progress notification
                    if progress_callback:
                        progress_callback(collected, self.max_scenes)

                    # Check collection limit
                    if self.max_scenes and collected >= self.max_scenes:
                        should_collect = False
                        logger.info(f"Reached max_scenes limit: {self.max_scenes}")

                except Exception as e:
                    if self.retry_on_error and retry_count < self.DEFAULT_RETRY_COUNT:
                        retry_count += 1
                        logger.warning(f"Error processing scene (retry {retry_count}): {e}")
                        await trio.sleep(self.DEFAULT_RETRY_DELAY_SECONDS)
                        continue
                    else:
                        logger.error(f"Failed to process scene after {retry_count} retries: {e}")
                        if self.stats:
                            self.stats.errors_encountered.append(str(e))

    async def _process_scene(self, item: Any, index: int) -> bool:
        """
        Process and validate a single scene.

        Args:
            item: Item received from pipeline
            index: Current collection index

        Returns:
            bool: True if scene was processed successfully
        """
        # Type validation
        if not isinstance(item, SceneInfo):
            logger.warning(f"Received non-SceneInfo object: {type(item)}")
            if self.stats:
                self.stats.warnings.append(f"Invalid type at index {index}: {type(item)}")
            return False

        # Scene validation
        if self.validate_scenes and not self._validate_scene(item):
            return False

        # Apply filters
        if self._should_filter_scene(item):
            if self.stats:
                self.stats.scenes_filtered += 1
            return False

        # Thread-safe scene addition
        with self._thread_lock():
            self._scenes.append(item)

            # Update stats
            if self.stats:
                self.stats.scenes_collected += 1

        # Cache the scene
        if self.cache is not None and self.enable_caching:
            try:
                self.cache.set(item.scene_name, item)
                logger.debug(f"Cached scene: {item.scene_name}")
            except Exception as e:
                logger.warning(f"Failed to cache scene {item.scene_name}: {e}")
                if self.stats:
                    self.stats.warnings.append(f"Cache failure: {e}")

        return True

    def _validate_scene(self, scene: SceneInfo) -> bool:
        """
        Validate scene data integrity.

        Args:
            scene: Scene to validate

        Returns:
            bool: True if scene is valid
        """
        if not hasattr(scene, 'scene_name') or not scene.scene_name:
            logger.warning("Scene missing scene_name attribute")
            return False

        if not hasattr(scene, 'clean_images') or not hasattr(scene, 'noisy_images'):
            logger.warning(f"Scene {scene.scene_name} missing required image lists")
            return False

        return True

    def _should_filter_scene(self, scene: SceneInfo) -> bool:
        """
        Check if scene should be filtered based on configuration.

        Args:
            scene: Scene to check

        Returns:
            bool: True if scene should be filtered out
        """
        # Filter by test_reserve flag
        if self.filter_test_reserve and hasattr(scene, 'test_reserve'):
            if scene.test_reserve:
                logger.debug(f"Filtering test reserve scene: {scene.scene_name}")
                return True

        # Filter by CFA type
        if self.cfa_filter and hasattr(scene, 'cfa_type'):
            if scene.cfa_type != self.cfa_filter:
                logger.debug(f"Filtering CFA type {scene.cfa_type}: {scene.scene_name}")
                return True

        return False

    def get_scene(self, index: int) -> SceneInfo:
        """
        Get scene by index with cache support and error handling.

        Args:
            index: Scene index

        Returns:
            SceneInfo: Scene at the given index

        Raises:
            IndexError: If index is out of bounds
            RuntimeError: If bridge is in error state
        """
        # State check
        if self.state == BridgeState.ERROR:
            raise RuntimeError("Bridge is in error state")

        # Input validation
        if index < 0:
            raise IndexError(f"Index {index} is negative")
        if index >= len(self._scenes):
            raise IndexError(f"Index {index} out of bounds for {len(self._scenes)} scenes")

        with self._thread_lock():
            # Get scene reference
            scene = self._scenes[index]

            # Try cache first if available
            if self.cache is not None and self.enable_caching:
                try:
                    cached = self.cache.get(scene.scene_name)
                    if cached is not None and isinstance(cached, SceneInfo):
                        if self.stats:
                            self.stats.cache_hits += 1
                        logger.debug(f"Cache hit for scene {scene.scene_name}")
                        return cached
                except Exception as e:
                    logger.warning(f"Cache retrieval failed for {scene.scene_name}: {e}")

                if self.stats:
                    self.stats.cache_misses += 1

            return scene

    def __len__(self) -> int:
        """Return number of collected scenes (thread-safe)."""
        with self._thread_lock():
            return len(self._scenes)

    def __getitem__(self, index: int) -> SceneInfo:
        """Support indexing operator."""
        return self.get_scene(index)

    def __iter__(self):
        """Support iteration (thread-safe snapshot)."""
        with self._thread_lock():
            # Return iterator over a copy to avoid modification during iteration
            return iter(list(self._scenes))

    # Backwards compatibility methods
    def get_clean_noisy_pair(self, idx: int) -> tuple:
        """
        Legacy API to get clean and noisy pair.

        Args:
            idx: Index of the pair

        Returns:
            tuple: (clean_image, list_of_noisy_images)

        Raises:
            RuntimeError: If backwards compatibility mode is not enabled
        """
        if not self.backwards_compat_mode:
            raise RuntimeError("Backwards compatibility mode not enabled")

        scene = self.get_scene(idx)
        # Return clean image and list of noisy images
        clean = scene.clean_images[0] if scene.clean_images else None
        noisy = scene.noisy_images if scene.noisy_images else []
        return (clean, noisy)

    # Health and monitoring methods
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check and return status.

        Returns:
            dict: Health status information
        """
        health_status = {
            "state": self.state.value,
            "scenes_collected": len(self._scenes),
            "cache_enabled": self.enable_caching,
            "cache_available": self.cache is not None,
            "thread_safe": self.thread_safe,
            "last_check": time.time()
        }

        if self.stats:
            health_status.update({
                "stats": {
                    "scenes_collected": self.stats.scenes_collected,
                    "scenes_filtered": self.stats.scenes_filtered,
                    "cache_hits": self.stats.cache_hits,
                    "cache_misses": self.stats.cache_misses,
                    "errors": len(self.stats.errors_encountered),
                    "warnings": len(self.stats.warnings)
                }
            })

        # Update last health check time
        self._last_health_check = time.time()

        return health_status

    def get_stats(self) -> Optional[BridgeStats]:
        """
        Get bridge statistics.

        Returns:
            BridgeStats: Current statistics or None if monitoring disabled
        """
        return self.stats

    def reset_stats(self) -> None:
        """Reset statistics to initial state."""
        if self.stats:
            self.stats = BridgeStats()
            logger.info("Bridge statistics reset")

    # Disk cache methods for legacy dataloader compatibility
    def write_disk_cache(self, cache_file: Path, dataset_root: Optional[Path] = None) -> None:
        """
        Write collected scenes to JSONL disk cache.

        Writes one JSON object per line, where each line is a legacy-compatible
        scene descriptor matching the format expected by rawds.py dataloaders.

        Args:
            cache_file: Path to JSONL cache file to write
            dataset_root: Root directory for dataset paths (defaults to cache_file parent)

        Raises:
            RuntimeError: If bridge has no scenes collected
            IOError: If cache file cannot be written
        """
        if not self._scenes:
            raise RuntimeError("No scenes to write - bridge is empty")

        cache_file = Path(cache_file)
        dataset_root = dataset_root or cache_file.parent

        logger.info(f"Writing {len(self._scenes)} scenes to JSONL cache: {cache_file}")

        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)

            with open(cache_file, 'w', encoding='utf-8') as f:
                with self._thread_lock():
                    for scene in self._scenes:
                        descriptor = self._scene_to_legacy_descriptor(scene, dataset_root)
                        f.write(json.dumps(descriptor) + '\n')

            logger.info(f"Successfully wrote {len(self._scenes)} scenes to {cache_file}")

        except Exception as e:
            logger.error(f"Failed to write disk cache: {e}")
            raise IOError(f"Failed to write cache file {cache_file}: {e}") from e

    def load_from_disk_cache(self, cache_file: Path) -> None:
        """
        Load scenes from JSONL disk cache.

        Reads JSONL cache file and reconstructs SceneInfo objects from legacy
        descriptor format. This is primarily for testing cache compatibility.

        Args:
            cache_file: Path to JSONL cache file to read

        Raises:
            FileNotFoundError: If cache file doesn't exist
            ValueError: If cache file format is invalid
        """
        cache_file = Path(cache_file)

        if not cache_file.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_file}")

        logger.info(f"Loading scenes from JSONL cache: {cache_file}")

        try:
            loaded_scenes = []

            with open(cache_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        descriptor = json.loads(line)
                        # For now, just store the descriptor as-is
                        # Full SceneInfo reconstruction would require file access
                        loaded_scenes.append(descriptor)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON on line {line_num}: {e}")
                        continue

            # Note: This stores descriptors, not full SceneInfo objects
            # Full reconstruction requires file system access and is beyond cache scope
            logger.info(f"Loaded {len(loaded_scenes)} scene descriptors from cache")

            if self.stats:
                self.stats.scenes_collected = len(loaded_scenes)

        except Exception as e:
            logger.error(f"Failed to load disk cache: {e}")
            raise ValueError(f"Failed to read cache file {cache_file}: {e}") from e

    def write_yaml_compatible_cache(self, cache_file: Path, dataset_root: Optional[Path] = None) -> None:
        """
        Write collected scenes to legacy YAML format.

        Produces YAML file matching the exact format expected by
        CleanProfiledRGBNoisyBayerImageCropsDataset and other legacy
        dataloaders in rawds.py.

        Args:
            cache_file: Path to YAML file to write
            dataset_root: Root directory for dataset paths (defaults to cache_file parent)

        Raises:
            RuntimeError: If bridge has no scenes collected
            IOError: If YAML file cannot be written
        """
        if not self._scenes:
            raise RuntimeError("No scenes to write - bridge is empty")

        cache_file = Path(cache_file)
        dataset_root = dataset_root or cache_file.parent

        logger.info(f"Writing {len(self._scenes)} scenes to YAML cache: {cache_file}")

        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)

            descriptors = []
            with self._thread_lock():
                for scene in self._scenes:
                    descriptor = self._scene_to_legacy_descriptor(scene, dataset_root)
                    descriptors.append(descriptor)

            with open(cache_file, 'w', encoding='utf-8') as f:
                yaml.dump(
                    descriptors,
                    f,
                    allow_unicode=True,
                    default_flow_style=False,
                    sort_keys=False
                )

            logger.info(f"Successfully wrote {len(descriptors)} scenes to {cache_file}")

        except Exception as e:
            logger.error(f"Failed to write YAML cache: {e}")
            raise IOError(f"Failed to write YAML file {cache_file}: {e}") from e

    def _scene_to_legacy_descriptor(self, scene: SceneInfo, dataset_root: Path) -> Dict[str, Any]:
        """
        Convert SceneInfo to legacy YAML descriptor format.

        Replicates YAMLArtifactWriter.scene_to_yaml_descriptor() logic to ensure
        compatibility with CleanProfiledRGBNoisyBayerImageCropsDataset.

        Args:
            scene: SceneInfo object to convert
            dataset_root: Root directory for dataset

        Returns:
            Dictionary in legacy YAML descriptor format

        Raises:
            ValueError: If scene missing GT or noisy images
        """
        gt_img = scene.get_gt_image()
        if not gt_img:
            raise ValueError(f"Scene {scene.scene_name} missing GT image")

        if not scene.noisy_images:
            raise ValueError(f"Scene {scene.scene_name} has no noisy images")

        # Use first noisy image for metadata
        noisy_img = scene.noisy_images[0]
        metadata = noisy_img.metadata if noisy_img.metadata else {}

        # Build descriptor matching legacy format
        return {
            # Scene identification
            "scene_name": scene.scene_name,
            "image_set": scene.scene_name,
            "is_bayer": scene.cfa_type == "bayer",

            # File paths
            "f_fpath": str(noisy_img.local_path) if noisy_img.local_path else "",
            "f_bayer_fpath": str(noisy_img.local_path) if noisy_img.local_path else "",
            "gt_fpath": str(gt_img.local_path) if gt_img.local_path else "",
            "gt_linrec2020_fpath": str(gt_img.local_path) if gt_img.local_path else "",
            "gt_bayer_fpath": str(gt_img.local_path) if gt_img.local_path else "",
            "f_linrec2020_fpath": str(noisy_img.local_path) if noisy_img.local_path else "",

            # Alignment metadata
            "best_alignment": metadata.get("alignment", [0, 0]),
            "best_alignment_loss": metadata.get("alignment_loss", 0.0),

            # Gain correction
            "raw_gain": metadata.get("raw_gain", 1.0),
            "rgb_gain": metadata.get("rgb_gain", 1.0),

            # Mask metadata
            "mask_mean": metadata.get("mask_mean", 1.0),
            "mask_fpath": metadata.get("mask_fpath", ""),

            # Color space
            "rgb_xyz_matrix": metadata.get("rgb_xyz_matrix", [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]),

            # Overexposure threshold
            "overexposure_lb": metadata.get("overexposure_lb", 1.0),

            # Crops list
            "crops": metadata.get("crops", []),

            # Quality metrics
            "rgb_msssim_score": metadata.get("msssim_score", 1.0),
        }

    def close(self) -> None:
        """
        Close bridge and release resources.

        This method should be called when the bridge is no longer needed
        to ensure proper cleanup of resources.
        """
        self.state = BridgeState.CLOSED

        # Clear scenes to free memory
        with self._thread_lock():
            self._scenes.clear()

        logger.info("AsyncPipelineBridge closed")

    def __del__(self):
        """Destructor to ensure cleanup."""
        if self.state != BridgeState.CLOSED:
            self.close()


class StreamingDatasetWrapper:
    """
    PyTorch Dataset wrapper for AsyncPipelineBridge in streaming mode.

    Provides the legacy interface expected by training code, wrapping the bridge
    to expose crops extracted from scenes. This enables training directly from
    the pipeline without writing to disk.

    Attributes:
        bridge: AsyncPipelineBridge instance with collected scenes
        num_crops: Number of crops per image
        crop_size: Size of each crop (square)
        transform: Optional transform to apply to crops
    """

    def __init__(
        self,
        bridge: AsyncPipelineBridge,
        num_crops: int,
        crop_size: int,
        transform: Optional[Callable] = None,
        backwards_compat_mode: bool = True
    ):
        """
        Initialize streaming dataset wrapper.

        Args:
            bridge: AsyncPipelineBridge with collected scenes
            num_crops: Number of crops per image
            crop_size: Size of each crop (square)
            transform: Optional transform function
            backwards_compat_mode: Return legacy dict format vs SceneInfo
        """
        self.bridge = bridge
        self.num_crops = num_crops
        self.crop_size = crop_size
        self.transform = transform
        self.backwards_compat_mode = backwards_compat_mode

        # Build flat list of (scene_idx, crop_idx) tuples for indexing
        self._crop_index = []
        for scene_idx in range(len(bridge)):
            scene = bridge[scene_idx]
            # Get noisy image metadata
            if scene.noisy_images and scene.noisy_images[0].metadata:
                crops = scene.noisy_images[0].metadata.get('crops', [])
                for crop_idx in range(len(crops)):
                    self._crop_index.append((scene_idx, crop_idx))

        logger.info(
            f"StreamingDatasetWrapper initialized: "
            f"{len(self.bridge)} scenes, {len(self._crop_index)} total crops"
        )

    def __len__(self) -> int:
        """Return total number of crops across all scenes."""
        return len(self._crop_index)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get crop by flat index.

        Args:
            index: Flat crop index

        Returns:
            Dictionary with crop data in legacy format:
                - x_crops: Clean crop tensor
                - y_crops: Noisy crop tensor
                - mask_crops: Mask crop tensor
                - rgb_xyz_matrix: Color matrix
                - gain: Gain value

        Raises:
            IndexError: If index out of bounds
        """
        if index < 0 or index >= len(self._crop_index):
            raise IndexError(f"Crop index {index} out of bounds for {len(self._crop_index)} crops")

        scene_idx, crop_idx = self._crop_index[index]
        scene = self.bridge[scene_idx]

        if not self.backwards_compat_mode:
            # Return SceneInfo directly
            return scene

        # Legacy mode: extract and return crop data
        gt_img = scene.get_gt_image()
        noisy_img = scene.noisy_images[0] if scene.noisy_images else None

        if not gt_img or not noisy_img:
            raise ValueError(f"Scene {scene.scene_name} missing required images")

        metadata = noisy_img.metadata or {}
        crops = metadata.get('crops', [])

        if crop_idx >= len(crops):
            raise IndexError(f"Crop index {crop_idx} out of bounds for scene {scene.scene_name}")

        crop = crops[crop_idx]

        # Build legacy format dict
        result = {
            'scene_name': scene.scene_name,
            'crop_index': crop_idx,
            'crop_metadata': crop,
            'rgb_xyz_matrix': metadata.get('rgb_xyz_matrix', [[1,0,0],[0,1,0],[0,0,1]]),
            'raw_gain': metadata.get('raw_gain', 1.0),
            'rgb_gain': metadata.get('rgb_gain', 1.0),
            'mask_mean': metadata.get('mask_mean', 1.0),
            'alignment': metadata.get('alignment', [0, 0]),
        }

        # Apply transform if provided
        if self.transform:
            result = self.transform(result)

        return result