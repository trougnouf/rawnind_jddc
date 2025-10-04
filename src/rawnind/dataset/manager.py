"""Dataset manager for RawNIND dataset.

This module provides a canonical index of all scenes and images in the dataset,
built from the authoritative dataset.yaml file. It handles discovery of local
files, validation via SHA1 hashes, and downloading of missing files.
"""

import hashlib
import yaml
import requests
import random
import logging
import trio
import httpx
from tqdm import tqdm
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator, AsyncGenerator, Callable, Any, Set
from dataclasses import dataclass, field
from functools import wraps


# Logger setup
logger = logging.getLogger(__name__)

# Dataset URLs
DATASET_YAML_URL = "https://dataverse.uclouvain.be/api/access/datafile/:persistentId?persistentId=doi:10.14428/DVN/DEQCIM/WWGHOR"
DATASET_ROOT = Path("src/rawnind/datasets/RawNIND/src")


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


def invalidates_cache(func: Callable) -> Callable:
    """Decorator to invalidate DatasetIndex caches after method execution.
    
    DEPRECATED: Use @emits_event decorator instead for event-based invalidation.
    """
    @wraps(func)
    def wrapper(self: 'DatasetIndex', *args: Any, **kwargs: Any) -> Any:
        result = func(self, *args, **kwargs)
        self._invalidate_caches()
        return result
    return wrapper


@dataclass
class ImageInfo:
    """Information about a single image file."""

    filename: str
    sha1: str
    is_clean: bool
    local_path: Optional[Path] = None
    validated: bool = False
    file_id: Optional[str] = None  # Dataverse file ID for downloads
    
    @property
    def download_url(self) -> str:
        """Construct download URL for this file.
        
        Returns:
            Download URL for the file
        """
        if self.file_id:
            return f"https://dataverse.uclouvain.be/api/access/datafile/{self.file_id}"
        # Fallback: use filename with SHA1 (requires lookup in dataset API)
        return f"https://dataverse.uclouvain.be/api/access/datafile/:persistentId?persistentId=doi:10.14428/DVN/DEQCIM&filename={self.filename}"


@dataclass
class SceneInfo:
    """Information about a scene (collection of clean and noisy images)."""

    scene_name: str
    cfa_type: str  # 'Bayer' or 'X-Trans'
    unknown_sensor: bool
    test_reserve: bool
    clean_images: List[ImageInfo] = field(default_factory=list)
    noisy_images: List[ImageInfo] = field(default_factory=list)

    def all_images(self) -> List[ImageInfo]:
        """Get all images (clean + noisy) for this scene."""
        return self.clean_images + self.noisy_images

    def get_gt_image(self) -> Optional[ImageInfo]:
        """Get the first clean (GT) image, or None if none exist."""
        return self.clean_images[0] if self.clean_images else None


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
        self.scenes: Dict[
            str, Dict[str, SceneInfo]
        ] = {}  # {cfa_type: {scene_name: SceneInfo}}
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

    def _invalidate_caches(self) -> None:
        """Invalidate cached computed values when index changes.
        
        DEPRECATED: Event-based invalidation is now preferred.
        """
        self._known_extensions = None
        self._sorted_cfa_types = None

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
        
    def load_index(self, force_update: bool = False) -> None:
        """Load the dataset index.

        Args:
            force_update: If True, download fresh index from online source
        """
        if self._loaded and not force_update:
            return

        # Check if cached index exists and is valid
        if not force_update and self.cache_path.exists():
            try:
                self._load_from_yaml(self.cache_path)
                self._loaded = True
                return
            except Exception as e:
                print(f"Warning: Failed to load cached index: {e}")
                print("Will download fresh index...")

        # Download and cache the index
        self.update_index()

    def update_index(self) -> None:
        """Download dataset.yaml from online source and build index."""
        print(f"Downloading dataset index from {DATASET_YAML_URL}...")

        response = requests.get(DATASET_YAML_URL, timeout=30)
        response.raise_for_status()

        # Parse YAML
        dataset_data = yaml.safe_load(response.text)

        # Build index
        self._build_index_from_data(dataset_data)

        # Save to cache
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            yaml.dump(dataset_data, f)

        print(f"Index cached to {self.cache_path}")
        self._loaded = True

    def _load_from_yaml(self, yaml_path: Path) -> None:
        """Load index from a local YAML file."""
        with open(yaml_path, "r") as f:
            dataset_data = yaml.safe_load(f)
        self._build_index_from_data(dataset_data)

    @emits_event(CacheEvent.INDEX_STRUCTURE_CHANGED)
    def _build_index_from_data(self, dataset_data: dict) -> None:
        """Build index from parsed dataset YAML data.
        
        Emits INDEX_STRUCTURE_CHANGED event upon completion.
        """
        self.scenes = {}

        for cfa_type in ["Bayer", "X-Trans"]:
            if cfa_type not in dataset_data:
                continue

            self.scenes[cfa_type] = {}

            for scene_name, scene_data in dataset_data[cfa_type].items():
                # Create ImageInfo objects for clean images
                clean_images = []
                for img_data in scene_data.get("clean_images", []):
                    clean_images.append(
                        ImageInfo(
                            filename=img_data["filename"],
                            sha1=img_data["sha1"],
                            is_clean=True,
                            file_id=img_data.get("file_id")
                        )
                    )

                # Create ImageInfo objects for noisy images
                noisy_images = []
                for img_data in scene_data.get("noisy_images", []):
                    noisy_images.append(
                        ImageInfo(
                            filename=img_data["filename"],
                            sha1=img_data["sha1"],
                            is_clean=False,
                            file_id=img_data.get("file_id")
                        )
                    )

                # Create SceneInfo
                scene_info = SceneInfo(
                    scene_name=scene_name,
                    cfa_type=cfa_type,
                    unknown_sensor=scene_data.get("unknown_sensor", False),
                    test_reserve=scene_data.get("test_reserve", False),
                    clean_images=clean_images,
                    noisy_images=noisy_images,
                )

                self.scenes[cfa_type][scene_name] = scene_info

    @emits_event(CacheEvent.LOCAL_PATHS_UPDATED)
    def discover_local_files(self) -> Tuple[int, int]:
        """Discover which files exist locally and update image paths.
        
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

                    # Check in scene directory
                    scene_dir = cfa_dir / scene_name

                    # Check in gt subdirectory for clean images
                    if img_info.is_clean:
                        search_dirs = [scene_dir / "gt", scene_dir]
                    else:
                        search_dirs = [scene_dir]

                    # Look for the file
                    for search_dir in search_dirs:
                        potential_path = search_dir / img_info.filename
                        if potential_path.exists():
                            img_info.local_path = potential_path
                            found_count += 1
                            break

        return found_count, total_count

    def validate_file(self, img_info: ImageInfo) -> bool:
        """Validate a file's SHA1 hash.

        Args:
            img_info: Image info with local_path set

        Returns:
            True if file exists and hash matches, False otherwise
        """
        if img_info.local_path is None or not img_info.local_path.exists():
            return False

        computed_hash = compute_sha1(img_info.local_path)
        img_info.validated = computed_hash == img_info.sha1
        return img_info.validated

    def validate_all_local_files(self) -> Tuple[int, int, List[ImageInfo]]:
        """Validate SHA1 hashes of all local files.

        Returns:
            Tuple of (valid_count, total_local_count, invalid_files_list)
        """
        if not self._loaded:
            self.load_index()

        valid_count = 0
        total_local = 0
        invalid_files = []

        for cfa_type, scenes in self.scenes.items():
            for scene_name, scene_info in scenes.items():
                for img_info in scene_info.all_images():
                    if img_info.local_path is not None:
                        total_local += 1
                        if self.validate_file(img_info):
                            valid_count += 1
                        else:
                            invalid_files.append(img_info)

        return valid_count, total_local, invalid_files

    def get_all_scenes(self) -> List[SceneInfo]:
        """Get list of all scenes."""
        if not self._loaded:
            self.load_index()

        all_scenes = []
        for scenes in self.scenes.values():
            all_scenes.extend(scenes.values())
        return all_scenes

    def get_scenes_by_cfa(self, cfa_type: str) -> List[SceneInfo]:
        """Get scenes filtered by CFA type.

        Args:
            cfa_type: 'Bayer' or 'X-Trans'
        """
        if not self._loaded:
            self.load_index()

        return list(self.scenes.get(cfa_type, {}).values())

    def get_scene(self, cfa_type: str, scene_name: str) -> Optional[SceneInfo]:
        """Get a specific scene.

        Args:
            cfa_type: 'Bayer' or 'X-Trans'
            scene_name: Scene name

        Returns:
            SceneInfo or None if not found
        """
        if not self._loaded:
            self.load_index()

        return self.scenes.get(cfa_type, {}).get(scene_name)

    def get_missing_files(self) -> List[ImageInfo]:
        """Get list of files not found locally."""
        if not self._loaded:
            self.load_index()

        missing = []
        for cfa_type, scenes in self.scenes.items():
            for scene_name, scene_info in scenes.items():
                for img_info in scene_info.all_images():
                    if img_info.local_path is None:
                        missing.append(img_info)

        return missing
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
                cfa_exists = cfa_dir.exists()
                
                for scene_name, scene_info in scenes.items():
                    for img_info in scene_info.all_images():
                        # Get candidate paths
                        candidates = self._candidate_paths(cfa_type, scene_name, img_info)
                        
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


    def get_available_scenes(self) -> List[SceneInfo]:
        """Get scenes that have at least one GT image available locally."""
        if not self._loaded:
            self.load_index()

        available = []
        for cfa_type, scenes in self.scenes.items():
            for scene_name, scene_info in scenes.items():
                gt_img = scene_info.get_gt_image()
                if gt_img and gt_img.local_path is not None:
                    available.append(scene_info)

        return available

    def sequential_scene_generator(
        self, extensions: Optional[List[str]] = None
    ) -> Generator[SceneInfo, None, None]:
        """Yield scenes sequentially in order.
        
        Args:
            extensions: Optional list of file extensions to filter by (e.g., ['.arw', '.dng'])
                       If None, all scenes are yielded regardless of extension.
        
        Yields:
            SceneInfo objects in sequential order
        """
        if not self._loaded:
            self.load_index()

        for cfa_type in self.sorted_cfa_types:
            for scene_name in sorted(self.scenes[cfa_type].keys()):
                scene_info = self.scenes[cfa_type][scene_name]
                
                if extensions is None:
                    yield scene_info
                else:
                    gt_img = scene_info.get_gt_image()
                    if gt_img and gt_img.local_path is not None:
                        ext_lower = gt_img.local_path.suffix.lower()
                        if any(ext_lower == ext.lower() for ext in extensions):
                            yield scene_info

    def random_scene_generator(
        self,
        count: Optional[int] = None,
        extensions: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> Generator[SceneInfo, None, None]:
        """Yield random scenes without replacement.
        
        Args:
            count: Number of scenes to yield. If None, yields all scenes.
            extensions: Optional list of file extensions to filter by (e.g., ['.arw', '.dng'])
                       If None, all scenes are considered regardless of extension.
            seed: Random seed for reproducibility
        
        Yields:
            SceneInfo objects in random order
        """
        if not self._loaded:
            self.load_index()

        if seed is not None:
            random.seed(seed)

        all_scenes = []
        for cfa_type, scenes in self.scenes.items():
            for scene_name, scene_info in scenes.items():
                if extensions is None:
                    all_scenes.append(scene_info)
                else:
                    gt_img = scene_info.get_gt_image()
                    if gt_img and gt_img.local_path is not None:
                        if any(gt_img.local_path.suffix.lower() == ext.lower() 
                               for ext in extensions):
                            all_scenes.append(scene_info)

        random.shuffle(all_scenes)
        
        num_to_yield = count if count is not None else len(all_scenes)
        for i, scene_info in enumerate(all_scenes):
            if i >= num_to_yield:
                break
            yield scene_info

    def _candidate_paths(self, cfa_type: str, scene_name: str, img_info: ImageInfo) -> List[Path]:
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
            return [
                scene_dir / "gt" / img_info.filename,
                scene_dir / img_info.filename
            ]
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
            async with client.stream('GET', url) as response:
                response.raise_for_status()
                
                with open(dest_path, 'wb') as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f.write(chunk)

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
                        # Construct destination path
                        scene = None
                        for cfa_scenes in self.scenes.values():
                            for s in cfa_scenes.values():
                                if img in s.all_images():
                                    scene = s
                                    break
                            if scene:
                                break
                        
                        if scene:
                            cfa_dir = self.dataset_root / scene.cfa_type
                            scene_dir = cfa_dir / scene.scene_name
                            if img.is_clean:
                                dest_path = scene_dir / "gt" / img.filename
                            else:
                                dest_path = scene_dir / img.filename
                            
                            await self._download_file(img.download_url, dest_path)
                            successful += 1
                        else:
                            logger.error(f"Could not find scene for {img.filename}")
                            failed += 1
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

    def print_summary(self) -> None:
        """Print a summary of the dataset index."""
        if not self._loaded:
            self.load_index()

        print("\n" + "=" * 80)
        print("DATASET INDEX SUMMARY")
        print("=" * 80)

        for cfa_type, scenes in self.scenes.items():
            print(f"\n{cfa_type}:")
            print(f"  Scenes: {len(scenes)}")

            total_clean = 0
            total_noisy = 0
            local_clean = 0
            local_noisy = 0

            for scene_info in scenes.values():
                total_clean += len(scene_info.clean_images)
                total_noisy += len(scene_info.noisy_images)

                for img in scene_info.clean_images:
                    if img.local_path is not None:
                        local_clean += 1

                for img in scene_info.noisy_images:
                    if img.local_path is not None:
                        local_noisy += 1

            print(f"  Clean images: {local_clean}/{total_clean} local")
            print(f"  Noisy images: {local_noisy}/{total_noisy} local")

        print("\n" + "=" * 80 + "\n")


def compute_sha1(file_path: Path) -> str:
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


# Global dataset index instance
_dataset_index = DatasetIndex()


def get_dataset_index() -> DatasetIndex:
    """Get the global dataset index instance."""
    return _dataset_index
