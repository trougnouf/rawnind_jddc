"""
Async-native crop producer - refactored for fine-grained parallelism.

1. No ProcessPoolExecutor - uses trio.to_thread.run_sync for blocking ops
2. Parallelizes at operation level (MS-SSIM per-channel) instead of scene level
3. Better CPU utilization through trio's scheduler
4. No serialization overhead or venv inheritance issues
"""

import logging
import os
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import trio

from .SceneInfo import SceneInfo, ImageInfo
from .pipeline_decorators import stage

logger = logging.getLogger(__name__)


class CropProducerStageAsync:
    """Async-native crop producer with fine-grained parallelism."""

    def __init__(
        self,
        output_dir: Path,
        crop_size: int = 512,
        num_crops: int = 10,
        max_workers: Optional[int] = None,
        save_format: str = "npy",
        crop_types: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        cfa_type: Optional[str] = None,
        stride: Optional[int] = None,
        use_systematic_tiling: bool = True
    ):
        """Initialize async crop producer."""
        if max_workers is None:
            max_workers = max(1, int(os.cpu_count() * 0.75))

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.config = config or {}

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

        # Create subdirectories
        for crop_type in self.crop_types:
            (self.output_dir / crop_type).mkdir(parents=True, exist_ok=True)

        self._viz = None

    def attach_visualizer(self, viz):
        """Attach visualizer for progress tracking."""
        self._viz = viz
        return self

    async def __aenter__(self):
        """Async context manager entry."""
        logger.info("CropProducerStageAsync starting up")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        logger.info("CropProducerStageAsync shutting down")

    @stage(progress=("cropping", "cropped"), debug_on_=True)
    async def process_scene(self, scene: SceneInfo) -> SceneInfo:
        """Generate crops for all image pairs in the scene (async-native)."""
        logger.info(f"→ CropProducer.process_scene: {scene.scene_name}")
        gt_img = scene.get_gt_image()
        if not gt_img:
            logger.warning(f"Scene {scene.scene_name} has no valid GT image, skipping crops")
            return scene

        # Process each noisy image paired with GT concurrently
        async with trio.open_nursery() as nursery:
            for noisy_img in scene.noisy_images:
                if not noisy_img.validated:
                    continue

                if "alignment" not in noisy_img.metadata:
                    logger.debug(f"No alignment for {noisy_img.filename}, skipping crops")
                    continue

                # Spawn concurrent task for each image pair
                nursery.start_soon(
                    self._generate_crops_for_pair_async,
                    scene,
                    gt_img,
                    noisy_img
                )

        total_crops = sum(len(img.metadata.get("crops", [])) for img in scene.noisy_images)
        logger.info(f"← CropProducer.process_scene: {scene.scene_name} ({total_crops} crops)")

        return scene

    async def _generate_crops_for_pair_async(
        self,
        scene: SceneInfo,
        gt_img: ImageInfo,
        noisy_img: ImageInfo
    ):
        """Generate crops for one image pair (fully async)."""
        try:
            # Extract metadata
            alignment = noisy_img.metadata.get("alignment", [0, 0])
            gain = noisy_img.metadata.get("gain", 1.0)
            overexposure_lb = noisy_img.metadata.get("overexposure_lb", 1.0)
            is_bayer = noisy_img.metadata.get("is_bayer", True)
            cfa_type = gt_img.cfa_type

            # Load images
            gt_data = await gt_img.load_image(as_torch=True)
            noisy_data = await noisy_img.load_image(as_torch=True)

            # Convert to numpy
            import torch
            if isinstance(gt_data, torch.Tensor):
                gt_data = gt_data.cpu().numpy()
            if isinstance(noisy_data, torch.Tensor):
                noisy_data = noisy_data.cpu().numpy()

            # Unload to prevent OOM
            gt_img.unload_image()
            noisy_img.unload_image()

            # Generate crops (async)
            crop_metadata = await self._extract_and_save_crops_async(
                scene.scene_name,
                gt_data,
                noisy_data,
                alignment,
                gain,
                gt_img.sha1,
                noisy_img.sha1,
                cfa_type,
                overexposure_lb,
                is_bayer,
                noisy_img.metadata
            )

            # Store metadata
            if "crops" not in noisy_img.metadata:
                noisy_img.metadata["crops"] = []
            noisy_img.metadata["crops"].extend(crop_metadata)

        except Exception as e:
            logger.error(f"Failed to generate crops for {noisy_img.filename}: {e}", exc_info=True)

    async def _extract_and_save_crops_async(
        self,
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
        metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract and save crops (async-native with fine-grained parallelism)."""

        MAX_MASKED = 0.5

        # Validate metadata for PRGB crops
        if "prgb" in self.crop_types and metadata is None:
            raise ValueError("metadata required for PRGB crops")

        # Snap alignment to CFA boundaries
        y_offset, x_offset = alignment
        if cfa_type == "bayer":
            y_offset = (y_offset // 2) * 2
            x_offset = (x_offset // 2) * 2
        elif cfa_type == "x-trans":
            y_offset = (y_offset // 3) * 3
            x_offset = (x_offset // 3) * 3

        # Apply alignment
        if y_offset != 0 or x_offset != 0:
            h, w = gt_data.shape[-2:]  # Handle both 2D (H,W) and 3D (C,H,W)
            y_start = max(0, y_offset)
            y_end = min(h, h + y_offset)
            x_start = max(0, x_offset)
            x_end = min(w, w + x_offset)

            gt_data = gt_data[..., y_start:y_end, x_start:x_end]
            noisy_data = noisy_data[
                ...,
                y_start - y_offset:y_end - y_offset,
                x_start - x_offset:x_end - x_offset
            ]

        # Apply gain
        if gain != 1.0:
            noisy_data = noisy_data * gain

        # Compute masks (async with parallelism)
        loss_mask, overexposure_mask = await self._compute_masks_async(
            gt_data, noisy_data, cfa_type, overexposure_lb, is_bayer
        )
        final_mask = loss_mask * overexposure_mask

        # Save mask (async I/O)
        mask_path = await self._save_mask_async(
            final_mask, scene_name, noisy_sha1
        )

        # Generate crop positions
        if gt_data.ndim == 2:
            h, w = gt_data.shape
        elif gt_data.ndim == 3:
            _, h, w = gt_data.shape
        else:
            raise ValueError(f"Unexpected tensor shape: {gt_data.shape}")

        if h < self.crop_size or w < self.crop_size:
            logger.warning(f"Image too small for {self.crop_size}x{self.crop_size} crops")
            return []

        candidates_y, candidates_x = self._generate_crop_positions(h, w, cfa_type)

        # Validate candidates
        valid_indices = self._validate_crop_positions(
            candidates_y, candidates_x, final_mask
        )

        if len(valid_indices) == 0:
            logger.warning(f"No valid crops found for {scene_name}")
            return []

        # Select crops
        if self.use_systematic_tiling:
            selected_indices = valid_indices
        else:
            num_to_sample = min(self.num_crops, len(valid_indices))
            selected_indices = np.random.choice(valid_indices, size=num_to_sample, replace=False)

        # Save crops concurrently
        crop_metadata = await self._save_crops_async(
            gt_data, noisy_data, candidates_y, candidates_x, selected_indices,
            scene_name, gt_sha1, noisy_sha1, mask_path, metadata
        )

        return crop_metadata

    async def _compute_masks_async(
        self,
        gt_data: np.ndarray,
        noisy_data: np.ndarray,
        cfa_type: str,
        overexposure_lb: float,
        is_bayer: bool
    ):
        """Compute loss and overexposure masks with async parallelism."""
        from rawnind.libs import rawproc

        if gt_data.ndim == 2:
            # 2D RAW format
            if cfa_type == "bayer":
                # Split into RGGB channels
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

                # Compute MS-SSIM mask with async window-level concurrency
                # This uses Trio's scheduler to parallelize ~9600 window tasks
                # Memory efficient (~400MB overhead vs 1.8GB for sequential PyTorch)
                from rawnind.libs.msssim_async import make_loss_mask_msssim_bayer_async
                loss_mask_rggb = await make_loss_mask_msssim_bayer_async(
                    gt_rggb,
                    noisy_rggb,
                    window_size=192,  # MS-SSIM requires >=176 for 5 scales
                    stride=96          # 50% overlap for robustness
                )

                # Upsample to full resolution
                loss_mask = np.repeat(np.repeat(loss_mask_rggb, 2, axis=0), 2, axis=1)

                # Overexposure mask
                #TODO needs proper async review
                overexposure_mask = await trio.to_thread.run_sync(
                    rawproc.make_overexposure_mask_bayer,
                    gt_data,
                    overexposure_lb
                )

            elif cfa_type == "x-trans":
                # Simple masks for X-Trans
                loss_mask = (np.abs(gt_data - noisy_data) < 0.3).astype(np.float32)
                overexposure_mask = (gt_data < overexposure_lb).astype(np.float32)

            else:
                raise ValueError(f"Unknown CFA type: {cfa_type}")

        elif gt_data.ndim == 3:
            # 3D RGB format
            if is_bayer:
                loss_mask = await trio.to_thread.run_sync(
                    rawproc.make_loss_mask_bayer,
                    gt_data,
                    noisy_data
                )
            else:
                loss_mask = await trio.to_thread.run_sync(
                    rawproc.make_loss_mask,
                    gt_data,
                    noisy_data
                )

            overexposure_mask = await trio.to_thread.run_sync(
                rawproc.make_overexposure_mask,
                gt_data,
                overexposure_lb
            )
        else:
            raise ValueError(f"Unexpected data format: {gt_data.shape}")

        return loss_mask, overexposure_mask

    async def _save_mask_async(
        self,
        final_mask: np.ndarray,
        scene_name: str,
        noisy_sha1: str
    ) -> Path:
        """Save mask to disk (async I/O)."""
        from PIL import Image

        mask_dir = self.output_dir / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask_path = mask_dir / f"{scene_name}_{noisy_sha1[:8]}_mask.png"

        # Save in thread to not block
        def save_mask():
            mask_uint8 = (final_mask * 255).astype(np.uint8)
            Image.fromarray(mask_uint8).save(mask_path)

        await trio.to_thread.run_sync(save_mask)
        return mask_path

    def _generate_crop_positions(
        self,
        h: int,
        w: int,
        cfa_type: str
    ):
        """Generate candidate crop positions."""
        if self.use_systematic_tiling:
            candidates_y = []
            candidates_x = []
            for y in range(0, h - self.crop_size + 1, self.stride):
                for x in range(0, w - self.crop_size + 1, self.stride):
                    candidates_y.append(y)
                    candidates_x.append(x)
            candidates_y = np.array(candidates_y)
            candidates_x = np.array(candidates_x)
        else:
            num_candidates = self.num_crops * 20
            candidates_y = np.random.randint(0, h - self.crop_size + 1, size=num_candidates)
            candidates_x = np.random.randint(0, w - self.crop_size + 1, size=num_candidates)

        # Snap to CFA boundaries
        if cfa_type == "Bayer":
            candidates_y = (candidates_y // 2) * 2
            candidates_x = (candidates_x // 2) * 2
        elif cfa_type == "X-Trans":
            candidates_y = (candidates_y // 3) * 3
            candidates_x = (candidates_x // 3) * 3

        return candidates_y, candidates_x

    def _validate_crop_positions(
        self,
        candidates_y: np.ndarray,
        candidates_x: np.ndarray,
        final_mask: np.ndarray
    ) -> np.ndarray:
        """Validate crop positions based on mask."""
        MAX_MASKED = 0.5
        num_candidates = len(candidates_y)
        valid_mask = np.zeros(num_candidates, dtype=bool)

        for idx in range(num_candidates):
            y, x = candidates_y[idx], candidates_x[idx]
            mask_crop = final_mask[y:y + self.crop_size, x:x + self.crop_size]
            valid_mask[idx] = (mask_crop.sum() / (self.crop_size ** 2)) >= MAX_MASKED

        return np.where(valid_mask)[0]

    async def _save_crops_async(
        self,
        gt_data: np.ndarray,
        noisy_data: np.ndarray,
        candidates_y: np.ndarray,
        candidates_x: np.ndarray,
        selected_indices: np.ndarray,
        scene_name: str,
        gt_sha1: str,
        noisy_sha1: str,
        mask_path: Path,
        metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Save all selected crops concurrently."""
        crop_metadata = []

        # Spawn concurrent tasks for each crop
        async with trio.open_nursery() as nursery:
            for i, idx in enumerate(selected_indices):
                y = candidates_y[idx]
                x = candidates_x[idx]
                crop_id = f"{scene_name}_{i:03d}_{gt_sha1[:8]}_{noisy_sha1[:8]}"

                nursery.start_soon(
                    self._save_one_crop_async,
                    gt_data, noisy_data, y, x, crop_id,
                    scene_name, mask_path, metadata, crop_metadata
                )

        return crop_metadata

    async def _save_one_crop_async(
        self,
        gt_data: np.ndarray,
        noisy_data: np.ndarray,
        y: int,
        x: int,
        crop_id: str,
        scene_name: str,
        mask_path: Path,
        metadata: Optional[Dict[str, Any]],
        crop_metadata: List[Dict[str, Any]]
    ):
        """Save one crop (async I/O)."""
        gt_crop = gt_data[..., y:y + self.crop_size, x:x + self.crop_size]
        noisy_crop = noisy_data[..., y:y + self.crop_size, x:x + self.crop_size]

        for crop_type in self.crop_types:
            crop_dir = self.output_dir / crop_type / scene_name
            crop_dir.mkdir(parents=True, exist_ok=True)

            if crop_type == "prgb":
                gt_path, noisy_path = await self._save_prgb_crop_async(
                    gt_crop, noisy_crop, crop_id, crop_dir, metadata
                )
            else:
                gt_path, noisy_path = await self._save_bayer_crop_async(
                    gt_crop, noisy_crop, crop_id, crop_dir
                )

        # Append metadata (thread-safe since each task writes unique entry)
        crop_metadata.append({
            "crop_id": crop_id,
            "coordinates": [y, x],
            "size": self.crop_size,
            "gt_linrec2020_fpath": str(gt_path),
            "f_bayer_fpath": str(noisy_path),
            "mask_fpath": str(mask_path)
        })

    async def _save_bayer_crop_async(
        self,
        gt_crop: np.ndarray,
        noisy_crop: np.ndarray,
        crop_id: str,
        crop_dir: Path
    ):
        """Save bayer crop to disk (async I/O)."""
        gt_path = crop_dir / f"{crop_id}_gt.npy"
        noisy_path = crop_dir / f"{crop_id}_noisy.npy"

        # Run disk I/O in thread
        await trio.to_thread.run_sync(np.save, gt_path, gt_crop)
        await trio.to_thread.run_sync(np.save, noisy_path, noisy_crop)

        return gt_path, noisy_path

    async def _save_prgb_crop_async(
        self,
        gt_crop: np.ndarray,
        noisy_crop: np.ndarray,
        crop_id: str,
        crop_dir: Path,
        metadata: Dict[str, Any]
    ):
        """Save PRGB crop (demosaic + color transform, async)."""
        from rawnind.libs import raw, rawproc
        import torch

        # Demosaic in thread (CPU-bound)
        # gt_crop already has shape (1, H, W) from ellipsis indexing
        gt_camrgb = await trio.to_thread.run_sync(
            raw.demosaic,
            gt_crop,
            metadata
        )
        noisy_camrgb = await trio.to_thread.run_sync(
            raw.demosaic,
            noisy_crop,
            metadata
        )

        # Color transform
        rgb_xyz_matrix = metadata.get("rgb_xyz_matrix")
        if rgb_xyz_matrix is None:
            raise ValueError("rgb_xyz_matrix required for PRGB crops")

        # Convert to numpy first, then to tensor (avoids nested list warning)
        rgb_xyz_array = np.asarray(rgb_xyz_matrix, dtype=np.float32)
        rgb_xyz_tensor = torch.from_numpy(rgb_xyz_array).unsqueeze(0)
        gt_camrgb_tensor = torch.from_numpy(gt_camrgb).unsqueeze(0).float()
        noisy_camrgb_tensor = torch.from_numpy(noisy_camrgb).unsqueeze(0).float()

        gt_lin_rec2020 = rawproc.camRGB_to_lin_rec2020_images(gt_camrgb_tensor, rgb_xyz_tensor)[0]
        noisy_lin_rec2020 = rawproc.camRGB_to_lin_rec2020_images(noisy_camrgb_tensor, rgb_xyz_tensor)[0]

        # Save EXR (async I/O)
        gt_path = crop_dir / f"{crop_id}_gt.exr"
        noisy_path = crop_dir / f"{crop_id}_noisy.exr"

        await trio.to_thread.run_sync(
            raw.hdr_nparray_to_file,
            gt_lin_rec2020.cpu().numpy(),
            str(gt_path),
            color_profile="lin_rec2020"
        )
        await trio.to_thread.run_sync(
            raw.hdr_nparray_to_file,
            noisy_lin_rec2020.cpu().numpy(),
            str(noisy_path),
            color_profile="lin_rec2020"
        )

        return gt_path, noisy_path

    async def consume_and_produce(
        self,
        input_channel: trio.MemoryReceiveChannel,
        output_channel: Optional[trio.MemorySendChannel] = None
    ):
        """Main processing loop."""
        async with input_channel:
            if output_channel:
                async with output_channel:
                    await self._process_loop(input_channel, output_channel)
            else:
                await self._process_loop(input_channel, None)

    async def _process_loop(
        self,
        input_channel: trio.MemoryReceiveChannel,
        output_channel: Optional[trio.MemorySendChannel]
    ):
        """Process scenes with semaphore-based backpressure."""
        sem = trio.Semaphore(self.max_workers)

        async with trio.open_nursery() as nursery:
            async for scene in input_channel:
                await sem.acquire()
                nursery.start_soon(
                    self._process_one_with_semaphore,
                    scene,
                    output_channel,
                    sem
                )

    async def _process_one_with_semaphore(
        self,
        scene: SceneInfo,
        output_channel: Optional[trio.MemorySendChannel],
        sem: trio.Semaphore
    ):
        """Process one scene and release semaphore."""
        try:
            processed_scene = await self.process_scene(scene)

            if output_channel:
                await output_channel.send(processed_scene)

        except Exception as e:
            logger.error(
                f"Error processing scene {scene.scene_name}: {e}",
                exc_info=True
            )
        finally:
            sem.release()

    @property
    def name(self) -> str:
        """Return worker name for logging."""
        return self.__class__.__name__
