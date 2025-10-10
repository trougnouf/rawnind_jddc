import logging
import os
from pathlib import Path
from typing import Callable, Awaitable, Optional, Dict, Any, List, Tuple

import trio

from .SceneInfo import SceneInfo, ImageInfo
from .cache import StreamingJSONCache

logger = logging.getLogger(__name__)

VALID_IMAGE_EXTENSIONS = frozenset({
    ".raw", ".nef", ".cr2", ".arw", ".dng", ".rw2", ".orf", ".sr2", ".raf", ".crw",
    ".exr", ".tif", ".tiff", ".npy"
})


class AsyncAligner:
    def __init__(
        self,
        cache_path: Optional[Path] = None,
        dataset_root: Optional[Path] = None,
        max_concurrent: int = 4,
        computation_fn: Optional[
            Callable[[ImageInfo], Awaitable[Dict[str, Any]]]
        ] = None,
        enable_crops_enrichment: bool = True,
        auto_compact_threshold: int = 10,
    ):
        """
        Initialize metadata enricher.

        Args:
            cache_path (Optional[Path]): Path to cache file for metadata. If None, uses default location.
            dataset_root (Optional[Path]): Root path of dataset (for finding crops). If None, uses default.
            max_concurrent (int): Maximum number of concurrent computations
            computation_fn (Optional[Callable[[ImageInfo], Awaitable[Dict[str, Any]]]]): Async function that computes metadata for an image. If None, uses default implementation.
            enable_crops_enrichment (bool): Whether to enrich with crops list metadata
            auto_compact_threshold (int): Auto-compact cache when duplicate ratio exceeds this

        """
        self.cache_path = cache_path or Path(
            "src/rawnind/datasets/RawNIND/metadata_cache.jsonl"
        )
        self.dataset_root = dataset_root or Path("src/rawnind/datasets/RawNIND/src")
        self.max_concurrent = max_concurrent
        self.computation_fn = computation_fn  # Can be None
        self.enable_crops_enrichment = enable_crops_enrichment

        # Use StreamingJSONCache instead of in-memory dict
        self._cache = StreamingJSONCache(
            self.cache_path,
            compact_threshold=auto_compact_threshold,
            handle_corruption=True,
        )
        # Note: cache will be loaded when consume_scenes_produce_enriched() starts

    async def consume_scenes_produce_enriched(
        self,
        scene_recv_channel: trio.MemoryReceiveChannel,
        enriched_send_channel: trio.MemorySendChannel,
    ) -> None:
        """
        Consume scenes, enrich image metadata, and produce enriched scenes.

        Args:
            scene_recv_channel: Receives SceneInfo objects
            enriched_send_channel: Sends enriched SceneInfo objects
        """
        # Load cache before starting enrichment
        await self._cache.load()
        logger.info(
            f"Loaded streaming cache with {len(self._cache.keys())} existing entries"
        )

        async with scene_recv_channel, enriched_send_channel:
            scenes_processed = 0

            async for scene_info in scene_recv_channel:
                # Enrich all images in the scene concurrently
                enriched_scene = await self._enrich_scene(scene_info)
                await enriched_send_channel.send(enriched_scene)

                scenes_processed += 1

                # Log progress periodically
                if scenes_processed % 10 == 0:
                    stats = await self._cache.stats()
                    logger.info(
                        f"Processed {scenes_processed} scenes. "
                        f"Cache: {stats['unique_keys']} unique entries, "
                        f"file size: {stats['file_size'] / 1024 / 1024:.2f} MB"
                    )

            # Log final statistics and compact if needed
            stats = await self._cache.stats()
            logger.info(
                f"Enrichment complete. Processed {scenes_processed} scenes. "
                f"Final cache: {stats['unique_keys']} unique entries, "
                f"{stats['total_entries']} total entries, "
                f"file size: {stats['file_size'] / 1024 / 1024:.2f} MB"
            )

            if stats["needs_compaction"]:
                logger.info("Compacting cache to remove duplicates...")
                await self._cache.compact()
                new_stats = await self._cache.stats()
                logger.info(
                    f"Compaction complete. Reduced file size from "
                    f"{stats['file_size'] / 1024 / 1024:.2f} MB to "
                    f"{new_stats['file_size'] / 1024 / 1024:.2f} MB"
                )

    async def _enrich_scene(self, scene_info: SceneInfo) -> SceneInfo:
        """
        Enriched the scene information by enriching ground truth (GT) image and noisy images with alignment metadata.

        Args:
            scene_info (SceneInfo): The SceneInfo object containing scene data including GT image and noisy images to be enriched.

        Returns:
            SceneInfo: The SceneInfo object after enrichment of GT image and noisy images with respective metadata.
        """
        gt_img = scene_info.get_gt_image()
        if not gt_img or not gt_img.local_path:
            logger.warning(
                f"Scene {scene_info.scene_name} has no valid GT image, skipping enrichment"
            )
            return scene_info

        # Enrich GT image first
        await self._enrich_clean_image(gt_img)

        # Enrich noisy images with alignment metadata
        async with trio.open_nursery() as nursery:
            semaphore = trio.Semaphore(self.max_concurrent)

            async def enrich_noisy_image(noisy_img: ImageInfo):
                """Enrich a noisy image with alignment/gain/mask metadata."""
                async with semaphore:
                    # Skip non-image files (e.g., .xmp metadata files)
                    file_ext = Path(noisy_img.filename).suffix.lower()
                    if file_ext not in VALID_IMAGE_EXTENSIONS:
                        logger.debug(f"Skipping non-image file: {noisy_img.filename}")
                        return

                    if noisy_img.local_path:
                        # Check cache first
                        if noisy_img.sha1 in self._cache:
                            cached_metadata = await self._cache.get(noisy_img.sha1)
                            noisy_img.metadata.update(cached_metadata)
                            logger.debug(
                                f"Using cached metadata for {noisy_img.filename}"
                            )
                        else:
                            try:
                                metadata = await self._compute_alignment_metadata(
                                    gt_img, noisy_img
                                )

                                # Optionally enrich with crops list
                                if self.enable_crops_enrichment:
                                    crops_metadata = await self._compute_crops_list(
                                        scene_info, gt_img, noisy_img
                                    )
                                    metadata["crops"] = crops_metadata

                                noisy_img.metadata.update(metadata)
                                # Cache the computed metadata
                                await self._cache.put(noisy_img.sha1, metadata)
                            except Exception as e:
                                logger.error(
                                    f"Failed to compute metadata for {noisy_img.filename}: {e}"
                                )
                                noisy_img.metadata["enrichment_error"] = str(e)

            # Process all noisy images concurrently
            for noisy_img in scene_info.noisy_images:
                nursery.start_soon(enrich_noisy_image, noisy_img)

        return scene_info

    async def _enrich_clean_image(self, img_info: ImageInfo) -> None:
        """This is an example enrichment function: Enrich a clean (GT) image with basic stats."""
        logger.info(f"Starting _enrich_clean_image for {img_info.filename}")
        # Skip non-image files (e.g., .xmp metadata files)
        file_ext = Path(img_info.filename).suffix.lower()
        if file_ext not in VALID_IMAGE_EXTENSIONS:
            logger.debug(f"Skipping non-image file: {img_info.filename}")
            return

        logger.info(f"Checking cache for {img_info.sha1}")
        if img_info.sha1 in self._cache:
            logger.info(f"Cache hit for {img_info.sha1}")
            img_info.metadata.update(await self._cache.get(img_info.sha1))
            logger.debug(f"Using cached metadata for {img_info.filename}")
        else:
            logger.info(f"Cache miss for {img_info.sha1}, computing stats")
            try:
                metadata = await trio.to_thread.run_sync(
                    self._compute_image_stats, img_info.local_path
                )
                logger.info(f"Computed stats for {img_info.sha1}, putting in cache")
                img_info.metadata.update(metadata)
                await self._cache.put(img_info.sha1, metadata)
                logger.info(f"Cache put complete for {img_info.sha1}")
            except Exception as e:
                logger.error(f"Failed to compute metadata for {img_info.filename}: {e}")
                img_info.metadata["enrichment_error"] = str(e)
        logger.info(f"Finished _enrich_clean_image for {img_info.filename}")

    async def _compute_alignment_metadata(
        self, gt_img: ImageInfo, noisy_img: ImageInfo
    ) -> Dict[str, Any]:
        """
        Compute alignment, gain, and loss mask for a noisy image relative to GT.
        #todo: this is a draft tha needs to be properly integrated - may or may not work properly as is.
        Wraps rawproc.get_best_alignment_compute_gain_and_make_loss_mask but:
        - Takes ImageInfo objects instead of file paths
        - Returns metadata dict without saving masks to disk
        - Runs in thread pool to avoid blocking

        Args:
            gt_img: Ground truth (clean) image
            noisy_img: Noisy image to align

        Returns:
            Dictionary with alignment, gains, mask data
        """
        from rawnind.libs.rawproc import img_fpath_to_np_mono_flt_and_metadata
        from rawnind.libs import rawproc

        def compute_sync() -> Dict[str, Any]:
            """Synchronous computation in thread."""
            # Load images
            gt_np, gt_metadata = img_fpath_to_np_mono_flt_and_metadata(
                str(gt_img.local_path)
            )
            noisy_np, noisy_metadata = img_fpath_to_np_mono_flt_and_metadata(
                str(noisy_img.local_path)
            )

            # Determine if bayer
            is_bayer = gt_np.shape[0] == 4

            # Find alignment
            if is_bayer:
                from rawnind.libs.alignment_backends import find_best_alignment_fft_cfa

                best_alignment, best_alignment_loss = find_best_alignment_fft_cfa(
                    gt_np,
                    noisy_np,
                    gt_metadata,
                    method="median",
                    return_loss_too=True,
                    verbose=False,
                )
                raw_gain = float(rawproc.match_gain(gt_np, noisy_np, return_val=True))
                rgb_gain = None
            else:
                best_alignment, best_alignment_loss = rawproc.find_best_alignment(
                    gt_np,
                    noisy_np,
                    return_loss_too=True,
                    method="auto",
                    verbose=False,
                )
                raw_gain = None
                rgb_gain = float(rawproc.match_gain(gt_np, noisy_np, return_val=True))

            # Shift images
            gt_aligned, noisy_aligned = rawproc.shift_images(
                gt_np, noisy_np, best_alignment
            )

            # Make loss mask
            if is_bayer:
                loss_mask = rawproc.make_loss_mask_bayer(gt_aligned, noisy_aligned)
            else:
                loss_mask = rawproc.make_loss_mask(gt_aligned, noisy_aligned)

            # Compute overexposure mask
            overexposure_mask = rawproc.make_overexposure_mask(
                gt_np, gt_metadata.get("overexposure_lb", 1.0)
            )
            overexposure_mask_shifted = rawproc.shift_mask(
                overexposure_mask, best_alignment
            )

            # Combine masks
            final_mask = loss_mask * overexposure_mask_shifted

            return {
                "alignment": list(best_alignment),
                "alignment_loss": float(best_alignment_loss),
                "raw_gain": raw_gain,
                "rgb_gain": rgb_gain,
                "is_bayer": is_bayer,
                "mask_mean": float(final_mask.mean()),
                "overexposure_lb": gt_metadata.get("overexposure_lb", 1.0),
                "rgb_xyz_matrix": gt_metadata.get("rgb_xyz_matrix", []).tolist()
                if "rgb_xyz_matrix" in gt_metadata
                else None,
            }

        # todo: this is not ideal concurrency; need to actually do this async
        return await trio.to_thread.run_sync(compute_sync)

    async def _compute_crops_list(
        self, scene_info: SceneInfo, gt_img: ImageInfo, noisy_img: ImageInfo
    ) -> List[Dict[str, Any]]:
        """
        Fetch list of pre-computed crops for this image pair.
        #todo: this is a draft that needs to be properly integrated - may or may not work properly as is.
        Adapted from prep_image_dataset.fetch_crops_list.

        Args:
            scene_info: Scene containing the images
            gt_img: Ground truth image
            noisy_img: Noisy image

        Returns:
            List of crop dictionaries with coordinates and file paths
        """

        def fetch_crops_sync() -> List[Dict[str, Any]]:
            """Synchronous crop fetching."""

            def get_coordinates(fn: str) -> Tuple[int, int]:
                """Extract coordinates from filename like 'image_512_256.tif'."""
                return tuple(int(c) for c in fn.split(".")[-2].split("_"))

            crops = []
            gt_basename = os.path.basename(gt_img.filename)
            f_basename = os.path.basename(noisy_img.filename)

            # Determine if bayer based on file extension or metadata
            is_bayer = gt_img.metadata.get(
                "is_bayer", not gt_img.filename.endswith((".exr", ".tif"))
            )

            # Build paths to crops directories
            crops_base = self.dataset_root.parent / "crops"
            prgb_image_set_dpath = (
                crops_base
                / "proc"
                / "lin_rec2020"
                / scene_info.cfa_type
                / scene_info.scene_name
            )

            prgb_gt_dir = prgb_image_set_dpath / "gt"
            prgb_noisy_dir = prgb_image_set_dpath

            if not prgb_gt_dir.exists():
                logger.debug(f"Crops directory not found: {prgb_gt_dir}")
                return []

            # List files in directories
            gt_files = list(prgb_gt_dir.glob("*"))
            noisy_files = list(prgb_noisy_dir.glob("*"))

            # Pre-filter and extract coordinates for GT files
            gt_file_coords = {}
            for gt_file in gt_files:
                if gt_file.name.startswith(gt_basename):
                    try:
                        coords = get_coordinates(gt_file.name)
                        gt_file_coords[coords] = gt_file.name
                    except (ValueError, IndexError):
                        continue

            if is_bayer:
                bayer_image_set_dpath = (
                    crops_base
                    / "src"
                    / "bayer"
                    / scene_info.cfa_type
                    / scene_info.scene_name
                )

            # Process both GT and noisy files
            for f_is_gt in (True, False):
                file_list = gt_files if f_is_gt else noisy_files
                search_dir = prgb_gt_dir if f_is_gt else prgb_noisy_dir

                for f_file in file_list:
                    if f_file.name.startswith(f_basename):
                        try:
                            coordinates = get_coordinates(f_file.name)
                        except (ValueError, IndexError):
                            continue

                        # Check if matching GT coordinates exist
                        if coordinates in gt_file_coords:
                            fn_gt = gt_file_coords[coordinates]

                            crop = {
                                "coordinates": list(coordinates),
                                "f_linrec2020_fpath": str(search_dir / f_file.name),
                                "gt_linrec2020_fpath": str(prgb_gt_dir / fn_gt),
                            }

                            if is_bayer:
                                f_bayer_path = (
                                    bayer_image_set_dpath
                                    / ("gt" if f_is_gt else "")
                                    / f_file.name.replace(".tif", ".npy")
                                )
                                gt_bayer_path = (
                                    bayer_image_set_dpath
                                    / "gt"
                                    / fn_gt.replace(".tif", ".npy")
                                )

                                crop["f_bayer_fpath"] = str(f_bayer_path)
                                crop["gt_bayer_fpath"] = str(gt_bayer_path)

                                # Check if bayer crops exist
                                if (
                                    not f_bayer_path.exists()
                                    or not gt_bayer_path.exists()
                                ):
                                    logger.debug(
                                        f"Missing bayer crop: {f_bayer_path} and/or {gt_bayer_path}"
                                    )
                                    continue

                            crops.append(crop)

            return crops

        # todo: not an ideal asynchronous implementation. Needs to be properly ported.
        return await trio.to_thread.run_sync(fetch_crops_sync)

    @staticmethod
    def _compute_image_stats(image_path: Path) -> Dict[str, Any]:
        """
        Compute image statistics (CPU-intensive, runs in thread).
        Right now it just appends 43
        These are some random basic example computations:
        - Noise statistics
        - Dynamic range
        - Color balance metrics
        - Sharpness measures

        Args:
            image_path: Path to image file

        Returns:
            Dictionary of computed statistics
        """
        # Example: Load image and compute stats
        # Replace with your actual computation logic
        # import rawpy
        #
        # with rawpy.imread(str(image_path)) as raw:
        #     raw_image = raw.raw_image.copy()

        # Compute various statistics
        metadata = {
            # "example_metadata": "example_value"
            # "mean_intensity": float(np.mean(raw_image)),
            # "std_intensity": float(np.std(raw_image)),
            # "min_value": int(np.min(raw_image)),
            # "max_value": int(np.max(raw_image)),
            # "dynamic_range_db": float(20 * np.log10(np.max(raw_image) / (np.std(raw_image) + 1e-10))),
            # "shape": raw_image.shape,
            # "dtype": str(raw_image.dtype),
        }

        return metadata
