"""
Generates Bayer/PRGB crops from enriched scenes.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import trio

from .post_download_worker import PostDownloadWorker
from .SceneInfo import SceneInfo, ImageInfo

logger = logging.getLogger(__name__)


class CropProducerStage(PostDownloadWorker):
    """Generates image crops for training from enriched scenes."""

    def __init__(
        self,
        output_dir: Path,
        crop_size: int = 512,
        num_crops: int = 10,
        max_workers: int = 4,
        save_format: str = "npy",  # "npy" or "tif"
        crop_types: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the crop producer.

        Args:
            output_dir: Directory for saving crops
            crop_size: Size of square crops to extract
            num_crops: Number of crops to extract per image pair
            max_workers: Maximum concurrent workers
            save_format: Format for saving crops ("npy" or "tif")
            crop_types: List of crop types to produce (e.g., ["bayer", "prgb"])
            config: Additional configuration
        """
        super().__init__(output_dir, max_workers, use_process_pool=True, config=config)
        self.crop_size = crop_size
        self.num_crops = num_crops
        self.save_format = save_format
        self.crop_types = crop_types or ["bayer", "prgb"]

        # Create subdirectories for different crop types
        for crop_type in self.crop_types:
            (self.output_dir / crop_type).mkdir(parents=True, exist_ok=True)

    async def process_scene(self, scene: SceneInfo) -> SceneInfo:
        """
        Generate crops for all image pairs in the scene.

        Args:
            scene: Enriched SceneInfo with alignment metadata

        Returns:
            SceneInfo with added crop metadata
        """
        gt_img = scene.get_gt_image()
        if not gt_img or not gt_img.validated:
            logger.warning(f"Scene {scene.scene_name} has no valid GT image, skipping crops")
            return scene

        # Process each noisy image paired with GT
        for noisy_img in scene.noisy_images:
            if not noisy_img.validated:
                continue

            # Check for alignment metadata from enrichment
            if "alignment" not in noisy_img.metadata:
                logger.debug(f"No alignment for {noisy_img.filename}, skipping crops")
                continue

            # Generate crops for this pair
            crop_metadata = await self._generate_crops_for_pair(
                scene, gt_img, noisy_img
            )

            # Add crop metadata to the noisy image
            if "crops" not in noisy_img.metadata:
                noisy_img.metadata["crops"] = []
            noisy_img.metadata["crops"].extend(crop_metadata)

        return scene

    async def _generate_crops_for_pair(
        self,
        scene: SceneInfo,
        gt_img: ImageInfo,
        noisy_img: ImageInfo
    ) -> List[Dict[str, Any]]:
        """
        Generate crops for a clean-noisy image pair.

        Args:
            scene: Scene information
            gt_img: Ground truth image
            noisy_img: Noisy image

        Returns:
            List of crop metadata dictionaries
        """
        alignment = noisy_img.metadata.get("alignment", [0, 0])
        gain = noisy_img.metadata.get("gain", 1.0)

        # Offload CPU-intensive crop extraction to process pool
        crop_metadata = await self.run_cpu_bound(
            self._extract_and_save_crops,
            scene.scene_name,
            gt_img.local_path,
            noisy_img.local_path,
            alignment,
            gain,
            gt_img.sha1,
            noisy_img.sha1
        )

        return crop_metadata

    def _extract_and_save_crops(
        self,
        scene_name: str,
        gt_path: Path,
        noisy_path: Path,
        alignment: List[int],
        gain: float,
        gt_sha1: str,
        noisy_sha1: str
    ) -> List[Dict[str, Any]]:
        """
        Extract and save crops (runs in process pool).

        This is the CPU-bound function that does the actual crop extraction.
        It should be kept as a standalone function to work with ProcessPoolExecutor.

        Args:
            scene_name: Name of the scene
            gt_path: Path to ground truth image
            noisy_path: Path to noisy image
            alignment: [y_offset, x_offset] alignment
            gain: Gain factor for the noisy image
            gt_sha1: SHA1 of ground truth image
            noisy_sha1: SHA1 of noisy image

        Returns:
            List of crop metadata
        """
        # Import here to avoid issues with process pool serialization
        import rawpy
        import numpy as np

        crop_metadata = []

        try:
            # Load images based on type
            if gt_path.suffix.lower() in [".npy"]:
                gt_data = np.load(gt_path)
                noisy_data = np.load(noisy_path)
            elif gt_path.suffix.lower() in [".exr", ".tif", ".tiff"]:
                # Load processed images
                import OpenEXR
                import Imath
                # Simplified - actual implementation would handle EXR properly
                logger.warning("EXR/TIFF loading not fully implemented in example")
                return []
            else:
                # Load RAW images
                with rawpy.imread(str(gt_path)) as gt_raw:
                    gt_data = gt_raw.raw_image_visible.copy()
                with rawpy.imread(str(noisy_path)) as noisy_raw:
                    noisy_data = noisy_raw.raw_image_visible.copy()

            # Apply alignment
            y_offset, x_offset = alignment
            if y_offset != 0 or x_offset != 0:
                # Crop to aligned region
                h, w = gt_data.shape[:2]
                y_start = max(0, y_offset)
                y_end = min(h, h + y_offset)
                x_start = max(0, x_offset)
                x_end = min(w, w + x_offset)

                gt_data = gt_data[y_start:y_end, x_start:x_end]
                noisy_data = noisy_data[
                    y_start - y_offset:y_end - y_offset,
                    x_start - x_offset:x_end - x_offset
                ]

            # Apply gain correction
            if gain != 1.0:
                noisy_data = noisy_data * gain

            # Extract random crops
            h, w = gt_data.shape[:2]
            if h < self.crop_size or w < self.crop_size:
                logger.warning(f"Image too small for {self.crop_size}x{self.crop_size} crops")
                return []

            for i in range(self.num_crops):
                # Random crop position
                y = np.random.randint(0, h - self.crop_size + 1)
                x = np.random.randint(0, w - self.crop_size + 1)

                gt_crop = gt_data[y:y + self.crop_size, x:x + self.crop_size]
                noisy_crop = noisy_data[y:y + self.crop_size, x:x + self.crop_size]

                # Save crops
                crop_id = f"{scene_name}_{i:03d}_{gt_sha1[:8]}_{noisy_sha1[:8]}"

                for crop_type in self.crop_types:
                    crop_dir = self.output_dir / crop_type / scene_name
                    crop_dir.mkdir(parents=True, exist_ok=True)

                    if self.save_format == "npy":
                        gt_path = crop_dir / f"{crop_id}_gt.npy"
                        noisy_path = crop_dir / f"{crop_id}_noisy.npy"
                        np.save(gt_path, gt_crop)
                        np.save(noisy_path, noisy_crop)
                    else:
                        # TIF format - not implemented in this example
                        logger.warning("TIF format not implemented")

                crop_metadata.append({
                    "crop_id": crop_id,
                    "position": [y, x],
                    "size": self.crop_size,
                    "gt_path": str(gt_path),
                    "noisy_path": str(noisy_path)
                })

        except Exception as e:
            logger.error(f"Failed to extract crops: {e}")

        return crop_metadata