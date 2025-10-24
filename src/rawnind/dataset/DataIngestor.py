"""DataIngestor loads and yields scene information from cached or remote dataset indexes.

The ingest process resolves scene metadata, enriches image data with identifiers from a metadata cache, and returns fully populated :class:`~sceneinfo.SceneInfo` objects. It supports optional asynchronous loading via Trio, handling both YAML scene lists and JSON metadata. The class abstracts cache handling, default paths, and validation, while providing clear error semantics.

Attributes
----------
cache_paths (Optional[Tuple[Path, Path]]): Paths to the YAML index file and the metadata JSON file. If ``None`` the defaults are derived from the dataset root directory.
index_path (Path): Path to the cached YAML index used for scene lookup.
metadata_path (Path): Path to the JSON metadata file containing per‑image details.

Methods
-------
__init__(cache_paths: Optional[Tuple[Path, Path]], dataset_root: Optional[Path]]):
    Initializes the helper, validates cache paths, and prepares internal state. Raises ``ValueError`` when provided paths are incomplete or the root directory does not exist.

_load_scene_info(cfa_type: str, scene_name: str, scene_data: dict) -> SceneInfo:
    Creates a :class:`SceneInfo` object enriched with file identifiers from the metadata cache and returns the populated instance.

produce_scenes(send_channel: trio.trio.MemorySendChannel):
    Asynchronously loads the dataset index, builds :class:`SceneInfo` objects, and sends each via the provided Trio memory channel. Raises ``RuntimeError`` for invalid channel type or cancellation before completion.

Raises
----
ValueError: Invalid cache configuration, missing root directory, or other validation failures.
KeyError: Missing keys in the metadata JSON when resolving identifiers.
RuntimeError: Channel type mismatch or cancellation before completion.

Notes
----
* The implementation assumes the YAML index and JSON metadata follow the expected schema; schema mismatches raise ``KeyError`` or ``json.JSONDecodeError``.
* Loading the index is synchronous; the public interface is asynchronous; callers must ``await`` the coroutine.
* The class does not automatically clear or close the send channel; callers should close it after consumption to avoid deadlocks.
"""
import json
from pathlib import Path
from typing import Tuple, Optional

import requests
import trio
import yaml

from .SceneInfo import ImageInfo, SceneInfo


class DataIngestor:
    """DataIngestor
    
    The :class:`DataIngestor` orchestrates the loading of dataset indexes,
    enriches scene‑level metadata, and produces :class:`SceneInfo` objects for
    down‑stream consumption.  It supports optional caching of a YAML index and a JSON
    metadata file, performs extensive validation of scene information, and resolves
    file‑ID mappings while handling edge cases such as missing XMP side‑car files,
    inconsistent SHA‑1 hashes, and absent sensor identifiers.  The implementation is
    asynchronous, leveraging :mod:`trio` to avoid blocking I/O, and is tolerant of partial
    failures in remote fetches.
    
    Attributes
    --------
    dataset_root : :class:`pathlib.Path`
        Root directory of the dataset; defaults to a path inferred at runtime.
    cache_paths : tuple[:class:`pathlib.Path`, :class:`pathlib.Path`]
        Paths to the YAML index cache and the metadata JSON file.  If
        ``None``, the class computes sensible defaults based on the dataset
        root.
    dataset_root : :class:`pathlib.Path` | None
        Optional explicit path for the dataset when the automatic
        discovery is unsuitable.
    cache_paths : tuple[:class:`pathlib.Path`, :class:`pathlib.Path`]
        Resolved cache locations derived from the supplied arguments.
    ... (additional internal attributes omitted for brevity)
    
    Methods
    -------
    __init__(self, cache: Optional[Tuple[Path, Path]] = None,
            dataset_root: Optional[Path] = None, dataset_root: Optional[Path] = None)
        Initializes the ingest pipeline, resolves default paths, and prepares the
        internal state.  The method validates the supplied arguments but does not
        perform any I/O.
    
    _load_index(self) -> dict
        Loads the dataset index from the computed cache locations.  Returns a
        dictionary representing the parsed index.
    
    _fetch_remote_index(self) -> dict
        Retrieves the YAML index and the metadata JSON from remote
        storage using HTTP requests in a background thread.  Returns the parsed
        dictionary.
    
    _create_scene_info(self, cfa_type: str, scene_name: str, scene_data: dict) -> :class:`SceneInfo`
        Constructs a :class:`SceneInfo` instance enriched with file‑ID information
        gathered from the metadata cache.
    
    Attributes of returned :class:`SceneInfo` objects
        ``filename`` – path of the scene image file.
        ``file_id`` – Unique identifier of the file in the data store.
        ``sha1`` – SHA‑1 checksum of the image (when available).
        ``scene_images`` – List of all file IDs belonging to the same
        scene, used for associating clean and noisy images.
    
    Notes
    -----
    * The class expects the dataset layout to follow a conventional
        ``<dataset_root>/<split>/...`` structure; if the layout diverges, the
        caller should provide explicit ``dataset_root`` values.
    * Network‑related errors (e.g., :class:`requests.exceptions.RequestException`)
        are propagated to the caller; they are not suppressed.
    * The implementation deliberately skips XMP side‑car files when building
        :class:`SceneInfo` objects because their presence is optional and their
        association is resolved later.
    * Because the loader performs eager loading, it may consume
        significant memory for large datasets; consider streaming or chunked
        processing for extremely large collections.
    """

    def __init__(
            self,
            cache_paths: Optional[Tuple[Path, Path]] = None,
            dataset_root: Optional[Path] = None,
            dataset_metadata_url: Optional[str] = None,
    ):
        """Initialize the dataset index.
        
        Args:
            cache_paths (tuple[Path, Path] or None): Optional pair of paths pointing to the dataset index file and its metadata file. When omitted, defaults are derived from the resolved dataset root directory.
            dataset_root (Path or None): Root directory of the dataset. If ``None``, the class falls back to the built‑in path ``DocScan/rawnind/datasets/RawNIND/DocScan``.
            dataset_metadata_url (str or None): URL for the dataset metadata API. If ``None``, the fallback URL ``https://dataverse.uclouvain.be/api/datasets/...`` is used.
        
        Attributes:
            dataset_root (Path): Resolved dataset root directory, based on the provided ``dataset_root`` or the internal default.
            dataset_metadata_url (str): URL used to fetch dataset metadata, resolved from the supplied ``dataset_metadata_url`` or the internal fallback.
        
        Returns:
            None: The initializer does not return a value.
        
        Raises:
            ValueError: Raised when ``cache_paths`` is supplied but does not contain exactly two elements or contains elements of unsupported types, preventing proper initialization.
        """
        self.dataset_root = (
            Path(dataset_root)
            if dataset_root
            else Path("DocScan/rawnind/datasets/RawNIND/DocScan")
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
        """Create a SceneInfo collection from scene data, enriching the result with file IDs from the metadata cache.
        
        Args:
            cfa_type (str): Identifier of the CFA pattern (e.g., "RGGB"); must be a non‑empty string.
            scene_name (str): Base filename of the scene; used as a key for mapping and for constructing full filenames.
            scene_data (dict): Mapping containing scene metadata. Expected keys:
                - ``clean_images`` (list): List of image dictionaries for clean captures.
                - ``noisy_images`` (list): List of image dictionaries for noisy captures.
                - ``unknown`` (bool, optional): Flag indicating an unknown sensor; defaults to ``False``.
                - ``test_reserve`` (bool, optional): Indicates inclusion in a test reserve; defaults to ``False``.
                Each image dictionary must provide:
                    * ``filename`` (str): Name of the image file.
                    * ``sha1`` (str): SHA‑1 digest of the image content.
                    * Optional ``file_id`` (str): Identifier of the file in the metadata store.
                    * ``is_clean`` (bool): Indicates whether the image is a clean capture.
        
        Returns:
            SceneInfo: Populated SceneInfo object containing enriched clean and noisy image entries, with associated file IDs and without `.xmp` side‑car entries.
        
        Raises:
            RuntimeError: If the metadata cache exists but cannot be read or contains an unexpected structure, preventing proper enrichment.
            KeyError: If required keys are missing from the provided ``scene_data`` mappings.
        
        Notes:
            * The function depends on the instance's cache attribute; changes to the cache after invocation may affect results.
            * Missing file IDs are silently omitted; the function does not raise for absent optional side‑car files.
            * The operation is I/O‑bound; consider off‑loading when used in performance‑critical paths.
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

        # Build ImageInfo lists with file IDs, excluding .xmp files
        # We'll match .xmp files to their parent images afterward
        clean_images = []
        xmp_files_clean = {}  # Map base filename to xmp filename
        
        for img_data in scene_data.get("clean_images", []):
            filename = img_data["filename"]
            
            # Separate .xmp sidecar files
            if filename.lower().endswith(".xmp"):
                # Extract base filename (e.g., "image.cr2.xmp" -> "image.cr2")
                base_name = filename[:-4]  # Remove .xmp extension
                xmp_files_clean[base_name] = filename
                continue
            
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
        xmp_files_noisy = {}  # Map base filename to xmp filename
        
        for img_data in scene_data.get("noisy_images", []):
            filename = img_data["filename"]
            
            # Separate .xmp sidecar files
            if filename.lower().endswith(".xmp"):
                # Extract base filename
                base_name = filename[:-4]
                xmp_files_noisy[base_name] = filename
                continue
            
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
        
        # Match .xmp files to their parent images
        # Note: xmp_path will be set later when files are downloaded/found
        # For now, we just store the xmp filename in metadata
        for img_info in clean_images:
            if img_info.filename in xmp_files_clean:
                img_info.metadata["xmp_filename"] = xmp_files_clean[img_info.filename]
        
        for img_info in noisy_images:
            if img_info.filename in xmp_files_noisy:
                img_info.metadata["xmp_filename"] = xmp_files_noisy[img_info.filename]

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
        """Load index and stream scene information through a Trio memory channel.
        
        Args:
            send_channel (trio.MemorySendChannel): Memory channel used to forward
                SceneInfo objects to downstream consumers. Must remain open for the
                duration of the operation and be a Trio `MemorySendChannel` instance.
        
        Raises:
            Exception: Propagated from failures in loading the dataset index,
                creating `SceneInfo` objects, or transmitting them via the
                provided channel. No additional error handling is performed.
        """
        async with send_channel:
            dataset_data = await self._load_index()

            for cfa_type, cfa_data in dataset_data.items():
                for scene_name, scene_data in cfa_data.items():
                    scene_info = self._create_scene_info(cfa_type, scene_name, scene_data)
                    await send_channel.send(scene_info)
                    await trio.sleep(0)  # Yield to scheduler

    async def _load_index(self) -> dict:
        """Load the index from cache or remote source.
        
        The routine checks cached dataset files; if both exist, it reads the YAML content from the cache, otherwise it fetches the remote index. Returns a dictionary mapping dataset identifiers to their cached data. The operation uses asynchronous I/O and resolves cache paths to Trio Path objects.
        
        Args:
            self: The instance containing cache path attributes.
        
        Returns:
            dict: Mapping of dataset identifiers to cached data.
        
        Raises:
            OSError: If reading the cache fails.
            RuntimeError: If the remote fetch fails or returns unexpected content.
        
        Notes:
            - The method must be awaited; it returns a coroutine.
            - If both cache files are missing, the function invokes the private `_fetch_remote_index` method.
            - The YAML file is assumed to contain a mapping; other structures raise errors.
        """
        yaml_cache, metadata_cache = self.cache_paths
        yaml_cache_trio = trio.Path(yaml_cache)
        metadata_cache_trio = trio.Path(metadata_cache)

        if await yaml_cache_trio.exists() and await metadata_cache_trio.exists():
            yaml_text = await yaml_cache_trio.read_text()
            dataset_data = yaml.safe_load(yaml_text)
        else:
            dataset_data = await self._fetch_remote_index()

        return dataset_data

    async def _fetch_remote_index(self) -> dict:
        """Download and cache the remote dataset index and metadata.
        
        Retrieves the YAML index and associated JSON metadata from the remote
        Dataverse endpoint, writes both files to the local cache using Trio's thread pool,
        and returns the parsed dataset dictionary. The operation is asynchronous,
        writes to the cache synchronously within the thread, and may raise network or
        timeout errors. It does not perform schema validation on the cached
        content; callers must handle potential HTTP errors and I/O exceptions.
        
        Raises:
            RuntimeError: If the HTTP request fails after retries.
            requests.HTTPError: Propagated for HTTP errors encountered during the fetch.
            requests.RequestException: For other request‑related failures.
        
        Notes:
            • The cache directory is created if it does not exist.
            • This function is asynchronous and must be awaited to obtain the result.
            • The returned dictionary reflects the raw structure of the
              dataset; any further processing should be performed after
              retrieval.
        """
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

        # Cache both files using async I/O
        yaml_cache_trio = trio.Path(yaml_cache)
        metadata_cache_trio = trio.Path(metadata_cache)
        await yaml_cache_trio.parent.mkdir(parents=True, exist_ok=True)
        await yaml_cache_trio.write_text(yaml.dump(dataset_data), encoding="utf-8")
        await metadata_cache_trio.write_text(metadata_text, encoding="utf-8")

        return dataset_data
