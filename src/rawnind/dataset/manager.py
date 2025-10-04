"""Dataset manager for RawNIND dataset.

This module provides a canonical index of all scenes and images in the dataset,
built from the authoritative dataset.yaml file. It handles discovery of local
files, validation via SHA1 hashes, and downloading of missing files.
"""

import hashlib
import json
import logging
from enum import Enum, auto
from functools import wraps
from pathlib import Path
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    AsyncGenerator,
    Callable,
    Any,
    Set,
)

import httpx
import requests
import trio
import yaml
from tqdm import tqdm

from rawnind.dataset import ImageInfo, SceneInfo

# Logger setup
logger = logging.getLogger(__name__)


class CacheEvent(Enum):
    """Events that can trigger cache invalidation."""

    INDEX_STRUCTURE_CHANGED = auto()  # Scenes added/removed, index rebuilt
    LOCAL_PATHS_UPDATED = auto()  # File discovery updated local_path values
    FILE_VALIDATED = auto()  # File validation state changed
    METADATA_CHANGED = auto()  # Scene metadata updated


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
        def wrapper(self: "DatasetIndex", *args: Any, **kwargs: Any) -> Any:
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
        async def wrapper(self: "DatasetIndex", *args: Any, **kwargs: Any) -> Any:
            result = await func(self, *args, **kwargs)
            self._emit(event)  # Synchronous emit is safe in Trio
            return result

        return wrapper

    return decorator


class DatasetIndex:
    """Canonical index of the RawNIND dataset with event-based cache invalidation."""

    def __init__(
            self,
            cache_paths: Optional[Tuple[Path, Path]] = None,
            dataset_root: Optional[Path] = None,
            dataset_metadata_url: Optional[str] = None,
    ):
        """Initialize dataset index.

        Args:
            cache_paths: Tuple of (yaml_cache_path, metadata_cache_path).
                        Defaults to (dataset_root/dataset_index.yaml, dataset_root/dataset_metadata.json)
            dataset_root: Root directory for dataset files (default: src/rawnind/datasets/RawNIND/src)
            dataset_metadata_url: URL for dataset metadata API (default: Dataverse API URL)
        """
        self.dataset_root = (
            Path(dataset_root)
            if dataset_root
            else Path("src/rawnind/datasets/RawNIND/src")
        )
        self.dataset_metadata_url = dataset_metadata_url or (
            "https://dataverse.uclouvain.be/api/datasets/:persistentId"
            "?persistentId=doi:10.14428/DVN/DEQCIM"
        )
        if cache_paths is None:
            self.cache_paths = (
                self.dataset_root / "dataset_index.yaml",
                self.dataset_root / "dataset_metadata.json",
            )
        else:
            self.cache_paths = cache_paths
        self._scenes: Dict[
            str, Dict[str, SceneInfo]
        ] = {}  # {cfa_type: {scene_name: SceneInfo}}

        # Cached values
        self._known_extensions: Optional[Set[str]] = None
        self._sorted_cfa_types: Optional[List[str]] = None

        # Event system
        self._events = EventEmitter()
        self._setup_cache_listeners()

    @property
    def scenes(self) -> Dict[str, Dict[str, SceneInfo]]:
        """Get the scenes dictionary."""
        if self._scenes == {}:
            self.load_index()
        return self._scenes

    def _setup_cache_listeners(self) -> None:
        """Register callbacks to invalidate specific caches on relevant events."""
        # known_extensions depends on both local paths and index structure
        self._events.on(
            CacheEvent.LOCAL_PATHS_UPDATED,
            lambda: setattr(self, "_known_extensions", None),
        )
        self._events.on(
            CacheEvent.INDEX_STRUCTURE_CHANGED,
            lambda: setattr(self, "_known_extensions", None),
        )

        # sorted_cfa_types only depends on index structure
        self._events.on(
            CacheEvent.INDEX_STRUCTURE_CHANGED,
            lambda: setattr(self, "_sorted_cfa_types", None),
        )

    def _emit(self, event: CacheEvent, **kwargs: Any) -> None:
        """Emit an event (internal helper).

        Args:
            event: Event to emit
            **kwargs: Optional context
        """
        self._events.emit(event, **kwargs)

    @property
    def known_extensions(self) -> set:
        """Get set of all file extensions present in the dataset.

        Computed once and cached until index changes.
        """
        if self._known_extensions is None:
            extensions = set()
            for cfa_type, scenes in self.scenes.items():
                for scene_info in scenes.values():
                    for img_info in scene_info.all_images():
                        if img_info.local_path is not None:
                            extensions.add(img_info.local_path.suffix.lower())
            self._known_extensions = extensions
        return self._known_extensions

    @property
    def sorted_cfa_types(self) -> List[str]:
        """Get sorted list of CFA types.

        Computed once and cached until index changes.
        """
        if self._sorted_cfa_types is None:
            self._sorted_cfa_types = sorted(self.scenes.keys())
        return self._sorted_cfa_types

    def load_index(self) -> None:
        """Load the dataset index from disk, updating from online source if needed."""
        yaml_cache, metadata_cache = self.cache_paths
        if yaml_cache.exists() and metadata_cache.exists():
            # Both cache files exist, load from disk
            with open(yaml_cache, "r") as f:
                dataset_data = yaml.safe_load(f)
        else:
            # Download and cache both files
            dataset_data = self._get_remote_index()
        self._build_index_from_data(dataset_data)

    def _get_remote_index(self) -> dict:
        """Download YAML structure and JSON metadata, caching both.

        Returns:
            dict: Dataset YAML structure (metadata is cached separately)
        """
        yaml_cache, metadata_cache = self.cache_paths

        print("Downloading dataset index from Dataverse...")

        # Download and cache YAML structure
        yaml_url = "https://dataverse.uclouvain.be/api/access/datafile/:persistentId?persistentId=doi:10.14428/DVN/DEQCIM/WWGHOR"
        response = requests.get(yaml_url, timeout=30)
        response.raise_for_status()
        dataset_data = yaml.safe_load(response.text)

        # Ensure parent directory exists before writing
        yaml_cache.parent.mkdir(parents=True, exist_ok=True)
        yaml_cache.write_text(yaml.dump(dataset_data), encoding="utf-8")

        # Download and cache JSON metadata
        print("Downloading file metadata from Dataverse API...")
        response = requests.get(self.dataset_metadata_url, timeout=30)
        response.raise_for_status()

        # Ensure parent directory exists before writing
        metadata_cache.parent.mkdir(parents=True, exist_ok=True)
        metadata_cache.write_text(response.text, encoding="utf-8")

        return dataset_data

    @emits_event(CacheEvent.INDEX_STRUCTURE_CHANGED)
    def _build_index_from_data(self, dataset_data: dict) -> None:
        """Build index from dataset YAML and metadata JSON cache.

        Emits INDEX_STRUCTURE_CHANGED event upon completion.
        """
        yaml_cache, metadata_cache = self.cache_paths

        # Load file ID mapping from cached metadata
        file_id_map = {}
        if metadata_cache.exists():
            with open(metadata_cache, "r") as f:
                metadata = json.load(f)

            if "data" in metadata and "latestVersion" in metadata["data"]:
                for file_entry in metadata["data"]["latestVersion"].get("files", []):
                    if "dataFile" in file_entry:
                        filename = file_entry["dataFile"].get("filename", "")
                        file_id = file_entry["dataFile"].get("id", "")
                        if filename and file_id:
                            file_id_map[filename] = str(file_id)

        # Build index with file IDs from mapping
        _scenes = {}
        for cfa_type, cfa_data in dataset_data.items():
            _scenes[cfa_type] = {}
            for scene_name, scene_data in cfa_data.items():
                clean_images = []
                for img_data in scene_data.get("clean_images", []):
                    filename = img_data["filename"]
                    # Prefer metadata cache, fall back to dataset_data
                    file_id = file_id_map.get(filename, img_data.get("file_id", ""))
                    clean_images.append(
                        ImageInfo(
                            filename=filename,
                            sha1=img_data["sha1"],
                            is_clean=True,
                            file_id=file_id,
                        )
                    )
                noisy_images = []
                for img_data in scene_data.get("noisy_images", []):
                    filename = img_data["filename"]
                    # Prefer metadata cache, fall back to dataset_data
                    file_id = file_id_map.get(filename, img_data.get("file_id", ""))
                    noisy_images.append(
                        ImageInfo(
                            filename=filename,
                            sha1=img_data["sha1"],
                            is_clean=False,
                            file_id=file_id,
                        )
                    )
                scene_info = SceneInfo(
                    scene_name=scene_name,
                    cfa_type=cfa_type,
                    unknown_sensor=scene_data.get("unknown_sensor", False),
                    test_reserve=scene_data.get("test_reserve", False),
                    clean_images=clean_images,
                    noisy_images=noisy_images,
                )

                _scenes[cfa_type][scene_name] = scene_info
        self._scenes = _scenes

    @emits_event(CacheEvent.LOCAL_PATHS_UPDATED)
    def discover(self) -> Tuple[int, int]:
        """Discover which files exist locally and update image paths. Updates image_info.local_path as images are found

        Emits LOCAL_PATHS_UPDATED event upon completion.

        Returns:
            Tuple of (found_count, total_count)
        """

        found_count = 0
        total_count = 0

        for cfa_type, scenes in self.scenes.items():
            cfa_dir = self.dataset_root / cfa_type
            if not cfa_dir.exists():
                continue

            for scene_name, scene_info in scenes.items():
                for img_info in scene_info.all_images():
                    total_count += 1
                    # build list of candidate paths
                    scene_dir = cfa_dir / scene_name
                    if img_info.is_clean:
                        search_dirs = [scene_dir / "gt", scene_dir]
                    else:
                        search_dirs = [scene_dir]

                    # Look for files in candidate paths
                    for search_dir in search_dirs:
                        potential_path = search_dir / img_info.filename
                        if potential_path.exists():
                            img_info.local_path = potential_path
                            found_count += 1
                            break

        return found_count, total_count

    def validate(self, img_info: ImageInfo) -> bool:
        """Validate a file's SHA1 hash.

        Args:
            img_info: Image info with local_path set

        Returns:
            True if file exists and hash matches, False otherwise
        """
        if img_info.local_path is None or not img_info.local_path.exists():
            return False

        computed_hash = hash_sha1(img_info.local_path)
        img_info.validated = computed_hash == img_info.sha1
        return img_info.validated

    async def produce_missing_files(self) -> AsyncGenerator[ImageInfo, None]:
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

        try:
            for cfa_type, scenes in self.scenes.items():
                cfa_dir = self.dataset_root / cfa_type
                cfa_exists = cfa_dir.exists()

                for scene_name, scene_info in scenes.items():
                    for img_info in scene_info.all_images():
                        # Get candidate paths
                        candidates = self._candidate_paths(
                            cfa_type, scene_name, img_info
                        )

                        # Check if file exists in any candidate location (synchronous)
                        found = False
                        if cfa_exists:  # Only check if directory exists
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

    def _candidate_paths(
            self, cfa_type: str, scene_name: str, img_info: ImageInfo
    ) -> List[Path]:
        """Generate candidate paths for an image file.

        Args:
            cfa_type: CFA type (Bayer or X-Trans)
            scene_name: Scene name
            img_info: Image information

        Returns:
            List of candidate paths to check
        """
        cfa_dir = self.dataset_root / cfa_type
        scene_dir = cfa_dir / scene_name

        if img_info.is_clean:
            return [scene_dir / "gt" / img_info.filename, scene_dir / img_info.filename]
        else:
            return [scene_dir / img_info.filename]

    async def _download_file(self, url: str, dest_path: Path) -> None:
        """Download a file asynchronously using Trio and httpx.

        Args:
            url: Download URL
            dest_path: Destination path for the downloaded file

        Raises:
            Exception: If download fails
        """
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()

                with open(dest_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f.write(chunk)

    async def async_download_missing_files(
            self, max_concurrent: int = 5, progress: bool = True
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

        successful = 0
        failed = 0

        async with trio.open_nursery() as nursery:
            semaphore = trio.Semaphore(max_concurrent)

            # Note: Using regular tqdm (not tqdm.asyncio) because we track download
            # completion (via pbar.update() in tasks), not iteration. tqdm.asyncio is
            # for wrapping async iterables to track iteration progress.
            pbar = tqdm(desc="Downloading", unit="file") if progress else None

            # Stream missing files as they're discovered
            # Event emission happens automatically in async_iter_missing_files
            async for img_info in self.produce_missing_files():

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
                            if pbar is not None:
                                pbar.update(1)

                nursery.start_soon(download_task, img_info)

            if pbar is not None:
                pbar.close()

        # No need for explicit discover_local_files() here
        # async_iter_missing_files() already emitted LOCAL_PATHS_UPDATED
        # If you want to re-verify downloaded files:
        # self.discover_local_files()  # This would emit the event again

        return successful, failed

    def print_summary(self) -> None:
        """
        Prints a summary of the dataset index, including information about scenes and image
        statistics.

        This method organizes and displays the dataset information for each type of CFA
        (Color Filter Array), detailing the number of scenes, as well as local and total
        clean and noisy image counts. It helps to summarize the current dataset state in
        a readable manner.

        Args:
            self: The instance of the class containing the dataset index information.

        Returns:
            None: This method does not return any value.

        Raises:
            None
        """

        def count_local_images(images: List[ImageInfo]) -> Tuple[int, int]:
            """Helper to count local/total images."""
            total = len(images)
            local = sum(1 for img in images if img.local_path is not None)
            return local, total

        border = "=" * 80
        print(f"\n{border}")
        print("DATASET INDEX SUMMARY".center(80))
        print(f"{border}\n")

        for cfa_type, scenes in self.scenes.items():
            # Scene count
            print(f"{cfa_type}:")
            print(f"  Scenes: {len(scenes)}")

            # Aggregate image counts
            local_clean, total_clean = (0, 0)
            local_noisy, total_noisy = (0, 0)

            for scene_info in scenes.values():
                lc, tc = count_local_images(scene_info.clean_images)
                ln, tn = count_local_images(scene_info.noisy_images)
                local_clean += lc
                total_clean += tc
                local_noisy += ln
                total_noisy += tn

            # Print image stats
            print(f"  Clean images: {local_clean}/{total_clean} local")
            print(f"  Noisy images: {local_noisy}/{total_noisy} local\n")

        print(f"{border}\n")


def hash_sha1(file_path: Path) -> str:
    """Compute SHA1 hash of a file.

    Args:
        file_path: Path to file

    Returns:
        SHA1 hash as hex string
    """
    sha1 = hashlib.sha1()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha1.update(chunk)
    return sha1.hexdigest()
