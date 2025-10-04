import json
from pathlib import Path
from typing import Tuple, Optional

import requests
import trio
import yaml

from . import ImageInfo, SceneInfo


class DataIngestor:
    """Loads cached or remote indexes and produces SceneInfo objects."""

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

    def _create_scene_info(self, cfa_type: str, scene_name: str, scene_data: dict) -> SceneInfo:
        """Create a SceneInfo object from scene data, enriching with file IDs from metadata cache.

        Args:
            cfa_type: CFA type identifier
            scene_name: Scene name
            scene_data: Scene data dictionary containing image lists

        Returns:
            SceneInfo object with enriched ImageInfo objects
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

        # Collect all SHA1s for this scene
        all_sha1s = []
        for img_data in scene_data.get("clean_images", []):
            all_sha1s.append(img_data["sha1"])
        for img_data in scene_data.get("noisy_images", []):
            all_sha1s.append(img_data["sha1"])

        # Build ImageInfo lists with file IDs
        clean_images = []
        for img_data in scene_data.get("clean_images", []):
            filename = img_data["filename"]
            file_id = file_id_map.get(filename, img_data.get("file_id", ""))
            clean_images.append(
                ImageInfo(
                    filename=filename,
                    sha1=img_data["sha1"],
                    is_clean=True,
                    scene_name=scene_name,
                    scene_images=all_sha1s,
                    cfa_type=cfa_type,
                    file_id=file_id,
                )
            )

        noisy_images = []
        for img_data in scene_data.get("noisy_images", []):
            filename = img_data["filename"]
            file_id = file_id_map.get(filename, img_data.get("file_id", ""))
            noisy_images.append(
                ImageInfo(
                    filename=filename,
                    sha1=img_data["sha1"],
                    is_clean=False,
                    scene_name=scene_name,
                    scene_images=all_sha1s,
                    cfa_type=cfa_type,
                    file_id=file_id,
                )
            )

        return SceneInfo(
            scene_name=scene_name,
            cfa_type=cfa_type,
            unknown_sensor=scene_data.get("unknown_sensor", False),
            test_reserve=scene_data.get("test_reserve", False),
            clean_images=clean_images,
            noisy_images=noisy_images,
        )

    async def produce_scenes(
            self,
            send_channel: trio.MemorySendChannel
    ) -> None:
        """Load index and produce SceneInfo objects.

        Args:
            send_channel: Channel to send SceneInfo objects
        """
        async with send_channel:
            dataset_data = await self._load_index()

            for cfa_type, cfa_data in dataset_data.items():
                for scene_name, scene_data in cfa_data.items():
                    scene_info = self._create_scene_info(cfa_type, scene_name, scene_data)
                    await send_channel.send(scene_info)
                    await trio.sleep(0)  # Yield to scheduler

    async def _load_index(self) -> dict:
        """Load from cache or fetch remote."""
        yaml_cache, metadata_cache = self.cache_paths

        if yaml_cache.exists() and metadata_cache.exists():
            with open(yaml_cache, "r") as f:
                dataset_data = yaml.safe_load(f)
        else:
            dataset_data = await self._fetch_remote_index()

        return dataset_data

    async def _fetch_remote_index(self) -> dict:
        """Download and cache YAML structure and JSON metadata."""
        yaml_cache, metadata_cache = self.cache_paths

        # Use trio.to_thread to avoid blocking
        def fetch_yaml():
            yaml_url = "https://dataverse.uclouvain.be/api/access/datafile/:persistentId?persistentId=doi:10.14428/DVN/DEQCIM/WWGHOR"
            response = requests.get(yaml_url, timeout=30)
            response.raise_for_status()
            return yaml.safe_load(response.text)

        def fetch_metadata():
            response = requests.get(self.dataset_metadata_url, timeout=30)
            response.raise_for_status()
            return response.text

        dataset_data = await trio.to_thread.run_sync(fetch_yaml)
        metadata_text = await trio.to_thread.run_sync(fetch_metadata)

        # Cache both files
        yaml_cache.parent.mkdir(parents=True, exist_ok=True)
        yaml_cache.write_text(yaml.dump(dataset_data), encoding="utf-8")
        metadata_cache.write_text(metadata_text, encoding="utf-8")

        return dataset_data
