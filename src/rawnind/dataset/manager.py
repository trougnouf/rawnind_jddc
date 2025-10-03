"""Dataset manager for RawNIND dataset.

This module provides a canonical index of all scenes and images in the dataset,
built from the authoritative dataset.yaml file. It handles discovery of local
files, validation via SHA1 hashes, and downloading of missing files.
"""

import hashlib
import random
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import httpx
import requests
import trio
import yaml
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# Dataset URLs
DATASET_YAML_URL = "https://dataverse.uclouvain.be/api/access/datafile/:persistentId?persistentId=doi:10.14428/DVN/DEQCIM/WWGHOR"
DATASET_API_URL = "https://dataverse.uclouvain.be/api/datasets/:persistentId/versions/:latest/files?persistentId=doi:10.14428/DVN/DEQCIM"
DATASET_ROOT = Path("src/rawnind/datasets/RawNIND/src")

METADATA_TIMEOUT = httpx.Timeout(30.0)
DOWNLOAD_TIMEOUT = httpx.Timeout(300.0)


def invalidates_cache(func: Callable) -> Callable:
    """Decorator to invalidate DatasetIndex caches after method execution."""

    @wraps(func)
    def wrapper(self: "DatasetIndex", *args: Any, **kwargs: Any) -> Any:
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
    """Canonical index of the RawNIND dataset."""

    def __init__(self, cache_path: Optional[Path] = None):
        """Initialize dataset index.

        Args:
            cache_path: Path to cached dataset.yaml (default: DATASET_ROOT/dataset.yaml)
        """
        self.cache_path = cache_path or DATASET_ROOT / "dataset_index.yaml"
        self.scenes: Dict[
            str, Dict[str, SceneInfo]
        ] = {}  # {cfa_type: {scene_name: SceneInfo}}
        self._loaded = False
        self._known_extensions: Optional[set] = None
        self._sorted_cfa_types: Optional[List[str]] = None
        self._remote_file_index: Optional[Dict[str, int]] = None
        self._missing_file_count = 0
        self._dataset_complete = False

    def _invalidate_caches(self) -> None:
        """Invalidate cached computed values when index changes."""
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

    @property
    def missing_file_count(self) -> int:
        """Return the most recently computed number of missing files."""
        return self._missing_file_count

    @property
    def dataset_complete(self) -> bool:
        """Return whether the dataset is currently considered complete."""
        return self._dataset_complete

    async def async_load_index(self, force_update: bool = False) -> None:
        """Asynchronously load the dataset index.

        Args:
            force_update: If True, download fresh index from online source
        """
        if self._loaded and not force_update:
            return

        if not force_update and self.cache_path.exists():
            try:
                await trio.to_thread.run_sync(self._load_from_yaml, self.cache_path)
                self._loaded = True
                return
            except Exception as exc:
                print(f"Warning: Failed to load cached index: {exc}")
                print("Will download fresh index...")

        await self.async_update_index()

    def load_index(self, force_update: bool = False) -> None:
        """Load the dataset index.

        Args:
            force_update: If True, download fresh index from online source
        """
        if self._loaded and not force_update:
            return

        if not force_update and self.cache_path.exists():
            try:
                self._load_from_yaml(self.cache_path)
                self._loaded = True
                return
            except Exception as exc:
                print(f"Warning: Failed to load cached index: {exc}")
                print("Will download fresh index...")

        self.update_index()

    async def async_update_index(self) -> None:
        """Asynchronously download dataset.yaml from online source and build index."""
        print(f"Downloading dataset index from {DATASET_YAML_URL}...")

        async with httpx.AsyncClient(timeout=METADATA_TIMEOUT) as client:
            response = await client.get(DATASET_YAML_URL)
            response.raise_for_status()
            text = response.text

        dataset_data = yaml.safe_load(text)
        self._build_index_from_data(dataset_data)
        await trio.to_thread.run_sync(self._write_cache, dataset_data)

        print(f"Index cached to {self.cache_path}")
        self._loaded = True

    def update_index(self) -> None:
        """Download dataset.yaml from online source and build index."""
        print(f"Downloading dataset index from {DATASET_YAML_URL}...")

        response = requests.get(DATASET_YAML_URL, timeout=30)
        response.raise_for_status()

        dataset_data = yaml.safe_load(response.text)
        self._build_index_from_data(dataset_data)
        self._write_cache(dataset_data)

        print(f"Index cached to {self.cache_path}")
        self._loaded = True

    def _write_cache(self, dataset_data: dict) -> None:
        """Persist dataset metadata to the cache file."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as handle:
            yaml.dump(dataset_data, handle)

    def _load_from_yaml(self, yaml_path: Path) -> None:
        """Load index from a local YAML file."""
        with open(yaml_path, "r") as handle:
            dataset_data = yaml.safe_load(handle)
        self._build_index_from_data(dataset_data)

    @invalidates_cache
    def _build_index_from_data(self, dataset_data: dict) -> None:
        """Build index from parsed dataset YAML data."""
        self.scenes = {}

        for cfa_type in ["Bayer", "X-Trans"]:
            if cfa_type not in dataset_data:
                continue

            self.scenes[cfa_type] = {}

            for scene_name, scene_data in dataset_data[cfa_type].items():
                clean_images = [
                    ImageInfo(
                        filename=img["filename"],
                        sha1=img["sha1"],
                        is_clean=True,
                    )
                    for img in scene_data.get("clean_images", [])
                ]

                noisy_images = [
                    ImageInfo(
                        filename=img["filename"],
                        sha1=img["sha1"],
                        is_clean=False,
                    )
                    for img in scene_data.get("noisy_images", [])
                ]

                scene_info = SceneInfo(
                    scene_name=scene_name,
                    cfa_type=cfa_type,
                    unknown_sensor=scene_data.get("unknown_sensor", False),
                    test_reserve=scene_data.get("test_reserve", False),
                    clean_images=clean_images,
                    noisy_images=noisy_images,
                )

                self.scenes[cfa_type][scene_name] = scene_info

    async def async_discover_local_files(self) -> Tuple[int, int]:
        """Asynchronously discover which files exist locally and update image paths.

        Returns:
            Tuple of (found_count, total_count)
        """
        if not self._loaded:
            await self.async_load_index()

        found_count = 0
        total_count = 0

        for cfa_type, scenes in self.scenes.items():
            cfa_dir = DATASET_ROOT / cfa_type
            if not await trio.to_thread.run_sync(lambda: cfa_dir.exists()):
                continue

            for scene_name, scene_info in scenes.items():
                for img_info in scene_info.all_images():
                    total_count += 1
                    img_info.local_path = None
                    for candidate in self._candidate_paths(cfa_type, scene_name, img_info):
                        if await trio.to_thread.run_sync(lambda path=candidate: path.exists()):
                            img_info.local_path = candidate
                            found_count += 1
                            break

        self._invalidate_caches()
        return found_count, total_count

    @invalidates_cache
    def discover_local_files(self) -> Tuple[int, int]:
        """Discover which files exist locally and update image paths.

        Returns:
            Tuple of (found_count, total_count)
        """
        if not self._loaded:
            self.load_index()

        found_count = 0
        total_count = 0

        for cfa_type, scenes in self.scenes.items():
            cfa_dir = DATASET_ROOT / cfa_type
            if not cfa_dir.exists():
                continue

            for scene_name, scene_info in scenes.items():
                for img_info in scene_info.all_images():
                    total_count += 1
                    img_info.local_path = None
                    for candidate in self._candidate_paths(cfa_type, scene_name, img_info):
                        if candidate.exists():
                            img_info.local_path = candidate
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
        invalid_files: List[ImageInfo] = []

        for scenes in self.scenes.values():
            for scene_info in scenes.values():
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

        all_scenes: List[SceneInfo] = []
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

    def iter_missing_files(self) -> Generator[ImageInfo, None, None]:
        """Yield ImageInfo entries missing from the local filesystem."""
        if not self._loaded:
            self.load_index()

        missing: List[ImageInfo] = []

        for cfa_type in sorted(self.scenes.keys()):
            for scene_name in sorted(self.scenes[cfa_type].keys()):
                scene_info = self.scenes[cfa_type][scene_name]
                for img_info in scene_info.all_images():
                    if img_info.local_path is not None and img_info.local_path.exists():
                        continue

                    candidates = self._candidate_paths(cfa_type, scene_name, img_info)
                    for candidate in candidates:
                        if candidate.exists():
                            img_info.local_path = candidate
                            break
                    else:
                        img_info.local_path = candidates[0]
                        missing.append(img_info)
                        continue

                    if img_info.local_path is None or not img_info.local_path.exists():
                        img_info.local_path = candidates[0]
                        missing.append(img_info)

        self._missing_file_count = len(missing)
        self._dataset_complete = self._missing_file_count == 0

        for img_info in missing:
            yield img_info

    def get_missing_files(self) -> List[ImageInfo]:
        """Get list of files not found locally."""
        return list(self.iter_missing_files())

    def get_available_scenes(self) -> List[SceneInfo]:
        """Get scenes that have at least one GT image available locally."""
        if not self._loaded:
            self.load_index()

        available: List[SceneInfo] = []
        for scenes in self.scenes.values():
            for scene_info in scenes.values():
                gt_img = scene_info.get_gt_image()
                if gt_img and gt_img.local_path is not None and gt_img.local_path.exists():
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
                    if gt_img and gt_img.local_path is not None and gt_img.local_path.exists():
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

        all_scenes: List[SceneInfo] = []
        for scenes in self.scenes.values():
            for scene_info in scenes.values():
                if extensions is None:
                    all_scenes.append(scene_info)
                else:
                    gt_img = scene_info.get_gt_image()
                    if (
                        gt_img
                        and gt_img.local_path is not None
                        and gt_img.local_path.exists()
                        and any(
                            gt_img.local_path.suffix.lower() == ext.lower()
                            for ext in extensions
                        )
                    ):
                        all_scenes.append(scene_info)

        random.shuffle(all_scenes)

        num_to_yield = count if count is not None else len(all_scenes)
        for i, scene_info in enumerate(all_scenes):
            if i >= num_to_yield:
                break
            yield scene_info

    async def async_get_dataset_files(self) -> Dict[str, int]:
        """Asynchronously fetch the list of files in the dataset and their IDs."""
        async with httpx.AsyncClient(timeout=METADATA_TIMEOUT) as client:
            mapping = await self._ensure_remote_file_index(client, force_refresh=True)
        return dict(mapping)

    async def _ensure_remote_file_index(
        self,
        client: httpx.AsyncClient,
        *,
        force_refresh: bool = False,
    ) -> Dict[str, int]:
        """Ensure the remote file index is populated and return it."""
        if force_refresh:
            self._remote_file_index = None

        if self._remote_file_index is None:
            response = await client.get(DATASET_API_URL)
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data", []) if isinstance(payload, dict) else []

            index: Dict[str, int] = {}
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                data_file = entry.get("dataFile")
                if not isinstance(data_file, dict):
                    continue

                filename = data_file.get("filename")
                file_id = data_file.get("id")

                if isinstance(filename, str) and isinstance(file_id, int):
                    index[filename] = file_id

            self._remote_file_index = index

        return self._remote_file_index

    async def _download_and_validate(
        self,
        client: httpx.AsyncClient,
        img_info: ImageInfo,
        file_id: int,
        *,
        timeout: httpx.Timeout,
    ) -> None:
        """Download a single file and validate it."""
        if img_info.local_path is None:
            raise ValueError("Local path must be set before download")

        url = f"https://dataverse.uclouvain.be/api/access/datafile/{file_id}"
        response = await client.get(url, timeout=timeout)
        response.raise_for_status()
        content = await response.aread()

        def _write_file() -> None:
            img_info.local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(img_info.local_path, "wb") as handle:
                handle.write(content)

        await trio.to_thread.run_sync(_write_file)

        valid = await trio.to_thread.run_sync(self.validate_file, img_info)
        if not valid:
            raise ValueError(f"Downloaded file {img_info.filename} failed validation")

    async def async_download_missing_files(
        self,
        max_concurrent: int = 5,
        *,
        limit: Optional[int] = None,
    ) -> None:
        """Asynchronously download missing files with optional concurrency and limit."""
        if not self._loaded:
            await self.async_load_index()

        await self.async_discover_local_files()
        missing = list(self.iter_missing_files())

        if limit is not None:
            if limit <= 0:
                missing = []
            else:
                missing = missing[:limit]

        if not missing:
            print("No missing files to download.")
            return

        console = Console()
        errors: List[str] = []
        limiter = trio.CapacityLimiter(max(1, int(max_concurrent)))

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}", justify="right"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=10,
        )

        with progress:
            async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT) as client:
                await self._ensure_remote_file_index(client, force_refresh=True)

                download_task = progress.add_task(
                    f"[green]Downloading {len(missing)} files", total=len(missing)
                )

                async def worker(image_info: ImageInfo) -> None:
                    async with limiter:
                        try:
                            file_index = await self._ensure_remote_file_index(client)
                            file_id = file_index.get(image_info.filename)
                            if file_id is None:
                                raise FileNotFoundError(
                                    f"File {image_info.filename} not found in remote index"
                                )
                            await self._download_and_validate(
                                client, image_info, file_id, timeout=DOWNLOAD_TIMEOUT
                            )
                            progress.update(download_task, advance=1)
                        except Exception as exc:
                            error_msg = f"Failed to download {image_info.filename}: {exc}"
                            errors.append(error_msg)
                            console.print(f"[red]ERROR:[/red] {error_msg}")
                            progress.update(download_task, advance=1)

                async with trio.open_nursery() as nursery:
                    for img in missing:
                        nursery.start_soon(worker, img)

                progress.update(download_task, completed=len(missing))

        if errors:
            console.print(f"\n[red]Completed with {len(errors)} errors[/red]")
        else:
            console.print("\n[green]All downloads completed successfully![/green]")

        await self.async_discover_local_files()
        list(self.iter_missing_files())  # refresh cached counters  # refresh cached counters  # refresh cached counters

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
                    if img.local_path is not None and img.local_path.exists():
                        local_clean += 1

                for img in scene_info.noisy_images:
                    if img.local_path is not None and img.local_path.exists():
                        local_noisy += 1

            print(f"  Clean images: {local_clean}/{total_clean} local")
            print(f"  Noisy images: {local_noisy}/{total_noisy} local")

        print("\n" + "=" * 80 + "\n")

    def _candidate_paths(
        self,
        cfa_type: str,
        scene_name: str,
        img_info: ImageInfo,
    ) -> List[Path]:
        """Return candidate local paths for an image."""
        scene_dir = DATASET_ROOT / cfa_type / scene_name
        if img_info.is_clean:
            return [scene_dir / "gt" / img_info.filename, scene_dir / img_info.filename]
        return [scene_dir / img_info.filename]


def compute_sha1(file_path: Path) -> str:
    """Compute SHA1 hash of a file.

    Args:
        file_path: Path to file

    Returns:
        SHA1 hash as hex string
    """
    sha1 = hashlib.sha1()
    with open(file_path, "rb") as handle:
        while chunk := handle.read(8192):
            sha1.update(chunk)
    return sha1.hexdigest()


# Global dataset index instance
_dataset_index = DatasetIndex()


def get_dataset_index() -> DatasetIndex:
    """Get the global dataset index instance."""
    return _dataset_index