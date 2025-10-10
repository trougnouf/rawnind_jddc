"""
Generates bayer/PRGB crops from enriched scenes.
"""

import logging
import os
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import trio

from .PostDownloadWorker import PostDownloadWorker
from .SceneInfo import SceneInfo, ImageInfo
from .pipeline_decorators import stage

logger = logging.getLogger(__name__)


class CropProducerStage(PostDownloadWorker):
    """Generates image crops for training from enriched scenes."""

    def __init__(
        self,
        output_dir: Path,
        crop_size: int = 512,
        num_crops: int = 10,
        max_workers: Optional[int] = None,
        save_format: str = "npy",  # "npy" or "tif"
        crop_types: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        cfa_type: Optional[str] = None,
        stride: Optional[int] = None,
        use_systematic_tiling: bool = True
    ):
        """
        Initialize the crop producer.

        Args:
            output_dir: Directory for saving crops
            crop_size: Size of square crops to extract
            num_crops: Number of crops to extract per image pair (ignored if use_systematic_tiling=True)
            max_workers: Maximum concurrent workers (defaults to 0.75 * cpu_count)
            save_format: Format for saving crops ("npy" or "tif")
            crop_types: List of crop types to produce (e.g., ["bayer", "prgb"])
            config: Additional configuration
            cfa_type: CFA type ('Bayer' or 'X-Trans') for validation. If None, validation is skipped.
            stride: Stride for systematic tiling. If None, defaults to crop_size // 4
            use_systematic_tiling: If True, use overlapping tiling with stride. If False, random sampling
        """
        if max_workers is None:
            max_workers = max(1, int(os.cpu_count() * 0.75))

        super().__init__(output_dir, max_workers, use_process_pool=True, config=config)
        
        # Validate crop_size respects CFA block boundaries
        if cfa_type == "Bayer":
            assert crop_size % 2 == 0, f"Bayer crop_size must be even: {crop_size}"
        elif cfa_type == "X-Trans":
            assert crop_size % 3 == 0, f"X-Trans crop_size must be multiple of 3: {crop_size}"
        
        self.crop_size = crop_size
        self.num_crops = num_crops
        self.save_format = save_format
        self.crop_types = crop_types or ["bayer", "prgb"]
        self.stride = stride if stride is not None else crop_size // 4
        self.use_systematic_tiling = use_systematic_tiling

        # Create subdirectories for different crop types
        for crop_type in self.crop_types:
            (self.output_dir / crop_type).mkdir(parents=True, exist_ok=True)

        # Support for visualizer attachment
        self._viz = None

    def attach_visualizer(self, viz):
        """Attach visualizer for automatic progress tracking."""
        self._viz = viz
        return self

    @stage(progress=("cropping", "cropped"), debug_on_=True)
    async def process_scene(self, scene: SceneInfo) -> SceneInfo:
        """
        Generate crops for all image pairs in the scene.

        Args:
            scene: Enriched SceneInfo with alignment metadata

        Returns:
            SceneInfo with added crop metadata
        """
        logger.info(f"→ CropProducer.process_scene: {scene.scene_name}")
        gt_img = scene.get_gt_image()
        if not gt_img:
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

        total_crops = sum(len(img.metadata.get("crops", [])) for img in scene.noisy_images)
        logger.info(f"← CropProducer.process_scene: {scene.scene_name} ({total_crops} crops)")
        
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
        overexposure_lb = noisy_img.metadata.get("overexposure_lb", 1.0)
        is_bayer = noisy_img.metadata.get("is_bayer", True)

        # Load images using cached tensors (tensor-native architecture)
        # ImageInfo._image_tensor is already loaded by AsyncAligner
        gt_data = await gt_img.load_image(as_torch=True)
        noisy_data = await noisy_img.load_image(as_torch=True)
        
        # Convert to numpy for process pool (torch tensors don't serialize well)
        import torch
        if isinstance(gt_data, torch.Tensor):
            gt_data = gt_data.cpu().numpy()
        if isinstance(noisy_data, torch.Tensor):
            noisy_data = noisy_data.cpu().numpy()
        
        # CRITICAL: Unload immediately after converting to numpy to prevent OOM
        # Main process doesn't need cached tensors while process pool works
        gt_img.unload_image()
        noisy_img.unload_image()

        # Offload CPU-intensive crop extraction to process pool
        crop_metadata = await self.run_cpu_bound(
            CropProducerStage._extract_and_save_crops,
            self.crop_size,
            self.num_crops,
            self.output_dir,
            self.save_format,
            self.crop_types,
            scene.scene_name,
            gt_data,
            noisy_data,
            alignment,
            gain,
            gt_img.sha1,
            noisy_img.sha1,
            gt_img.cfa_type,
            overexposure_lb,
            is_bayer,
            noisy_img.metadata,  # Pass full metadata for PRGB pipeline
            self.stride,
            self.use_systematic_tiling
        )

        return crop_metadata

    @staticmethod
    def _extract_and_save_crops(
        crop_size: int,
        num_crops: int,
        output_dir: Path,
        save_format: str,
        crop_types: List[str],
        scene_name: str,
        gt_data: np.ndarray,
        noisy_data: np.ndarray,
        alignment: List[int],
        gain: float,
        gt_sha1: str,
        noisy_sha1: str,
        cfa_type: str,
        overexposure_lb: float,
        is_bayer: bool,
        metadata: Optional[Dict[str, Any]] = None,
        stride: int = 128,
        use_systematic_tiling: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Extract and save crops (runs in process pool).

        This is the CPU-bound function that does the actual crop extraction.
        It should be kept as a standalone function to work with ProcessPoolExecutor.

        Args:
            scene_name: Name of the scene
            gt_data: Ground truth image data (numpy array)
            noisy_data: Noisy image data (numpy array)
            alignment: [y_offset, x_offset] alignment
            gain: Gain factor for the noisy image
            gt_sha1: SHA1 of ground truth image
            noisy_sha1: SHA1 of noisy image
            cfa_type: CFA type ('bayer' or 'x-trans')
            overexposure_lb: Overexposure threshold
            is_bayer: Whether this is bayer CFA data
            metadata: Full metadata dict for PRGB pipeline (bayer_pattern, rgb_xyz_matrix, etc.)

        Returns:
            List of crop metadata
        """
        # Import here to avoid issues with process pool serialization
        import numpy as np
        from PIL import Image
        import sys
        sys.path.append(str(Path(__file__).parent.parent.parent))
        from rawnind.libs import rawproc

        crop_metadata = []
        MAX_MASKED = 0.5  # Maximum fraction of masked pixels allowed

        # Validate metadata for PRGB crops
        if "prgb" in crop_types and metadata is None:
            raise ValueError(
                "metadata required for PRGB crops (bayer_pattern/RGBG_pattern, rgb_xyz_matrix)"
            )

        try:
            # Images are already loaded from ImageInfo cache (tensor-native architecture)
            # No disk I/O needed here - data comes from memory

            # Snap alignment offsets to CFA block boundaries
            y_offset, x_offset = alignment
            if cfa_type == "bayer":
                y_offset = (y_offset // 2) * 2
                x_offset = (x_offset // 2) * 2
            elif cfa_type == "x-trans":
                y_offset = (y_offset // 3) * 3
                x_offset = (x_offset // 3) * 3
            
            # Apply alignment
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

            # Regenerate masks from cached metadata (temporal locality optimization)
            # Masks are cheap to recompute but expensive to store
            # Detect format: 2D RAW or 3D RGB
            if gt_data.ndim == 2:
                # 2D RAW format (H, W) - bayer or X-Trans
                import torch
                
                if cfa_type == "bayer":
                    # bayer: convert to 4D RGGB for channel-aware mask computation
                    # Strided slicing is cheap (no copy), extracts 4 color channels
                    if isinstance(gt_data, torch.Tensor):
                        gt_rggb = torch.stack([
                            gt_data[0::2, 0::2],  # R
                            gt_data[0::2, 1::2],  # G1
                            gt_data[1::2, 0::2],  # G2
                            gt_data[1::2, 1::2],  # B
                        ])
                        noisy_rggb = torch.stack([
                            noisy_data[0::2, 0::2],
                            noisy_data[0::2, 1::2],
                            noisy_data[1::2, 0::2],
                            noisy_data[1::2, 1::2],
                        ])
                        # Convert to numpy for rawproc functions (they expect numpy)
                        gt_rggb = gt_rggb.cpu().numpy()
                        noisy_rggb = noisy_rggb.cpu().numpy()
                    else:
                        gt_rggb = np.stack([
                            gt_data[0::2, 0::2],
                            gt_data[0::2, 1::2],
                            gt_data[1::2, 0::2],
                            gt_data[1::2, 1::2],
                        ])
                        noisy_rggb = np.stack([
                            noisy_data[0::2, 0::2],
                            noisy_data[0::2, 1::2],
                            noisy_data[1::2, 0::2],
                            noisy_data[1::2, 1::2],
                        ])
                    
                    # Use MS-SSIM + L1 combo (standard for perceptual losses)
                    loss_mask_rggb = rawproc.make_loss_mask_msssim_bayer(gt_rggb, noisy_rggb)
                    # Upsample loss_mask from RGGB resolution (H/2, W/2) back to full (H, W)
                    loss_mask = np.repeat(np.repeat(loss_mask_rggb, 2, axis=0), 2, axis=1)
                    
                    # Overexposure mask works on 2D directly
                    overexposure_mask = rawproc.make_overexposure_mask_bayer(
                        gt_data.cpu().numpy() if isinstance(gt_data, torch.Tensor) else gt_data,
                        overexposure_lb
                    )
                elif cfa_type == "x-trans":
                    # X-Trans: 6x6 pattern, can't easily separate into channels
                    # Use simple 2D masks (conservative but functional)
                    gt_np = gt_data.cpu().numpy() if isinstance(gt_data, torch.Tensor) else gt_data
                    noisy_np = noisy_data.cpu().numpy() if isinstance(noisy_data, torch.Tensor) else noisy_data
                    
                    # Simple L1-based loss mask (no channel separation)
                    loss_mask = (np.abs(gt_np - noisy_np) < 0.3).astype(np.float32)
                    
                    # Overexposure mask on 2D
                    overexposure_mask = (gt_np < overexposure_lb).astype(np.float32)
                else:
                    raise ValueError(f"Unknown CFA type: {cfa_type}")
            elif gt_data.ndim == 3:
                # 3D RGB format (C, H, W)
                if is_bayer:
                    loss_mask = rawproc.make_loss_mask_bayer(gt_data, noisy_data)
                else:
                    loss_mask = rawproc.make_loss_mask(gt_data, noisy_data)
                
                overexposure_mask = rawproc.make_overexposure_mask(gt_data, overexposure_lb)
            else:
                raise ValueError(f"Unexpected data format: {gt_data.shape}")

            final_mask = loss_mask * overexposure_mask

            # Save mask PNG to disk (will be hot in page cache for legacy loaders)
            mask_dir = output_dir / "masks"
            mask_dir.mkdir(parents=True, exist_ok=True)
            mask_path = mask_dir / f"{scene_name}_{noisy_sha1[:8]}_mask.png"

            # Convert mask to uint8 for PNG saving
            mask_uint8 = (final_mask * 255).astype(np.uint8)
            Image.fromarray(mask_uint8).save(mask_path)

            # Extract random crops with vectorized validation
            # Fix dimension extraction for both 2D (H,W) and 3D (C,H,W) formats
            if gt_data.ndim == 2:
                h, w = gt_data.shape
            elif gt_data.ndim == 3:
                _, h, w = gt_data.shape
            else:
                raise ValueError(f"Unexpected tensor shape: {gt_data.shape}")
            if h < crop_size or w < crop_size:
                logger.warning(f"Image too small for {crop_size}x{crop_size} crops")
                return []

            if use_systematic_tiling:
                # Systematic overlapping tiling with stride
                candidates_y = []
                candidates_x = []
                for y in range(0, h - crop_size + 1, stride):
                    for x in range(0, w - crop_size + 1, stride):
                        candidates_y.append(y)
                        candidates_x.append(x)
                candidates_y = np.array(candidates_y)
                candidates_x = np.array(candidates_x)
                num_candidates = len(candidates_y)
            else:
                # Random sampling (legacy behavior)
                num_candidates = num_crops * 20
                candidates_y = np.random.randint(0, h - crop_size + 1, size=num_candidates)
                candidates_x = np.random.randint(0, w - crop_size + 1, size=num_candidates)

            # Snap to CFA boundaries (vectorized)
            if cfa_type == "Bayer":
                candidates_y = (candidates_y // 2) * 2
                candidates_x = (candidates_x // 2) * 2
            elif cfa_type == "X-Trans":
                candidates_y = (candidates_y // 3) * 3
                candidates_x = (candidates_x // 3) * 3

            # Validate all candidates (vectorized)
            valid_mask = np.zeros(num_candidates, dtype=bool)
            for idx in range(num_candidates):
                y, x = candidates_y[idx], candidates_x[idx]
                mask_crop = final_mask[y:y + crop_size, x:x + crop_size]
                valid_mask[idx] = (mask_crop.sum() / (crop_size ** 2)) >= MAX_MASKED

            valid_indices = np.where(valid_mask)[0]

            # For systematic tiling, use all valid crops; for random, sample num_crops
            num_to_sample = len(valid_indices) if use_systematic_tiling else min(num_crops, len(valid_indices))
            
            if len(valid_indices) < num_to_sample and not use_systematic_tiling:
                logger.warning(
                    f"Only found {len(valid_indices)} valid crops out of {num_to_sample} needed "
                    f"for {scene_name}"
                )

            # Select crops
            if use_systematic_tiling:
                # Use all valid positions from systematic grid
                selected_indices = valid_indices
            else:
                # Random sample from valid positions
                selected_indices = np.random.choice(valid_indices, size=num_to_sample, replace=False)

            # Extract and save crops at selected positions
            for i, idx in enumerate(selected_indices):
                # Use pre-validated position
                y = candidates_y[idx]
                x = candidates_x[idx]

                gt_crop = gt_data[y:y + crop_size, x:x + crop_size]
                noisy_crop = noisy_data[y:y + crop_size, x:x + crop_size]

                # Save crops
                crop_id = f"{scene_name}_{i:03d}_{gt_sha1[:8]}_{noisy_sha1[:8]}"

                for crop_type in crop_types:
                    crop_dir = output_dir / crop_type / scene_name
                    crop_dir.mkdir(parents=True, exist_ok=True)

                    if crop_type == "prgb":
                        # PRGB: RAW → demosaic → camera RGB → lin_rec2020 → EXR
                        if metadata is None:
                            raise ValueError("metadata required for PRGB crops (bayer_pattern, rgb_xyz_matrix)")

                        from rawnind.libs import raw
                        import torch

                        # Demosaic to camera RGB using OIIO (supports both bayer and X-Trans)
                        # demosaic() infers pattern type from metadata (is_bayer flag)
                        gt_camrgb = raw.demosaic(gt_crop[np.newaxis, :, :], metadata)
                        noisy_camrgb = raw.demosaic(noisy_crop[np.newaxis, :, :], metadata)

                        # Convert to lin_rec2020
                        rgb_xyz_matrix = metadata.get("rgb_xyz_matrix")
                        if rgb_xyz_matrix is None:
                            raise ValueError("rgb_xyz_matrix required for PRGB crops")

                        rgb_xyz_tensor = torch.tensor([rgb_xyz_matrix], dtype=torch.float32)
                        gt_camrgb_tensor = torch.from_numpy(gt_camrgb).unsqueeze(0).float()
                        noisy_camrgb_tensor = torch.from_numpy(noisy_camrgb).unsqueeze(0).float()

                        gt_lin_rec2020 = rawproc.camRGB_to_lin_rec2020_images(gt_camrgb_tensor, rgb_xyz_tensor)[0]
                        noisy_lin_rec2020 = rawproc.camRGB_to_lin_rec2020_images(noisy_camrgb_tensor, rgb_xyz_tensor)[0]

                        # Save as EXR
                        gt_path = crop_dir / f"{crop_id}_gt.exr"
                        noisy_path = crop_dir / f"{crop_id}_noisy.exr"

                        raw.hdr_nparray_to_file(
                            gt_lin_rec2020.cpu().numpy(), str(gt_path), color_profile="lin_rec2020"
                        )
                        raw.hdr_nparray_to_file(
                            noisy_lin_rec2020.cpu().numpy(), str(noisy_path), color_profile="lin_rec2020"
                        )
                    else:
                        # bayer crops: save as NPY
                        if save_format == "npy":
                            gt_path = crop_dir / f"{crop_id}_gt.npy"
                            noisy_path = crop_dir / f"{crop_id}_noisy.npy"
                            np.save(gt_path, gt_crop)
                            np.save(noisy_path, noisy_crop)
                        else:
                            # TIF format - not implemented in this example
                            logger.warning("TIF format not implemented")

                crop_metadata.append({
                    "crop_id": crop_id,
                    "coordinates": [y, x],
                    "size": crop_size,
                    "gt_linrec2020_fpath": str(gt_path),
                    "f_bayer_fpath": str(noisy_path),
                    "mask_fpath": str(mask_path)
                })

        except Exception as e:
            logger.error(f"Failed to extract crops: {e}")

        return crop_metadata