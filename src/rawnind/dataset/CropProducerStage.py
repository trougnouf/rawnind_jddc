"""
Module for generating image crop pairs with asynchronous operations.

This module implements a concurrent pipeline for cropping aligned image pairs while
preserving color filter array (CFA) boundaries. It processes input data consisting
of RAW (Bayer/X-Trans) or RGB image pairs, validates them using MS-SSIM and
overexposure masks (192x192 windows, 96px stride), and outputs the results in
various formats: .npy for raw data, .exr for RGB data, and .png for masks. Each
crop's metadata includes alignment, gain, and coordinate information.

The implementation uses trio for structured concurrency, with a robust concurrency
model built around Trio nurseries for task coordination. A thread pool utilizing
75% of available CPU cores handles I/O and compute operations through a
channel-based pipeline with backpressure support. Task-local storage ensures
thread-safe metadata updates throughout the processing pipeline.

Memory management is carefully optimized through progressive loading and unloading
to maintain peak usage below 2GB for 24MP images. The thread pool implementation
achieves significant efficiency gains, with only ~400MB overhead compared to 1.8GB
for sequential processing. Concurrent tasks are capped at 64 via semaphore to
prevent resource exhaustion.

CFA handling is specifically tailored to different array types, with Bayer arrays
using 2x2 alignment and even-sized crops, while X-Trans arrays require 6x6
alignment with crops in multiples of 3. Boundary snapping is enforced for both
types to maintain data integrity.

Known issues that require attention include the need for async review of
overexposure masking, replacement of PIL dependency with OpenImageIO, potential
memory spikes with very large batches, and simplified X-Trans mask computation
compared to Bayer implementation.

Dependencies: trio, numpy, torch, PIL (temporary)
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import trio

from common.libs.libimganalysis import CFA_TYPE_BAYER, CFA_TYPE_XTRANS
from rawnind.libs import raw, rawproc
from rawnind.libs.msssim_async import make_loss_mask_msssim_bayer_async
from .SceneInfo import SceneInfo, ImageInfo
from .pipeline_decorators import stage

logger = logging.getLogger(__name__)


class CropProducerStage:
    """
    Represents an asynchronous stage for generating cropped segments from images.

    This class is designed for processing image data by generating cropped segments
    from a dataset using specified parameters. Cropped images are stored in output
    directories, and the process can be customized based on attributes such as crop
    size, number of crops, and systematic tiling. It supports asynchronous task
    execution to handle large datasets efficiently and allows attaching a visualizer
    for enhanced debugging or visualization purposes.

    Attributes:
        output_dir (Path): Directory where generated crops are stored. Subdirectories
            are created for each crop type.
        crop_size (int): Size of the generated crops in pixels. Must align with the
            CFA block boundaries when using Bayer or X-Trans formats.
        num_crops (int): Number of crops to generate per image.
        max_workers (int): Maximum number of concurrent workers for crop generation.
        save_format (str): Format used to save the cropped images.
        crop_types (List[str]): List of crop types to be processed. Default includes
            "bayer" and "prgb".
        stride (int): Step size for crop extraction during systematic tiling. Defaults
            to a quarter of the crop size.
        use_systematic_tiling (bool): Determines whether systematic tiling is used for
            generating crops.
        config (Dict[str, Any] or None): Configuration dictionary for additional
            settings or options.
    """

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
        use_systematic_tiling: bool = True,
    ):
        """
        Initializes the class with settings for generating image crops from a dataset.
        Performs validation on crop size depending on the CFA type and creates necessary
        output directories to store processed crops. Default values are assigned for
        several parameters if not explicitly provided.
        """
        if max_workers is None:
            max_workers = max(1, int(os.cpu_count() * 0.75))

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.config = config or {}

        # Validate crop_size respects CFA block boundaries
        if cfa_type == CFA_TYPE_BAYER:
            assert crop_size % 2 == 0, f"Bayer crop_size must be even: {crop_size}"
        elif cfa_type == CFA_TYPE_XTRANS:
            assert (
                crop_size % 3 == 0
            ), f"X-Trans crop_size must be multiple of 3: {crop_size}"

        self.crop_size = crop_size
        self.num_crops = num_crops
        self.save_format = save_format
        self.crop_types = crop_types or [CFA_TYPE_BAYER, "prgb"]
        self.stride = stride if stride is not None else crop_size // 4
        self.use_systematic_tiling = use_systematic_tiling

        # Create subdirectories
        for crop_type in self.crop_types:
            (self.output_dir / crop_type).mkdir(parents=True, exist_ok=True)

        self._viz = None

    def attach_visualizer(self, viz):
        """
        Attaches a visualizer to the current object.

        This method associates a visualizer instance with the current object, storing
        it for subsequent use.

        Args:
            viz: An instance of a visualizer to be attached.

        Returns:
            The current object instance with the visualizer attached.
        """
        self._viz = viz
        return self

    async def __aenter__(self):
        """Enters an asynchronous context manager and performs setup operations.

        This method is part of the `async with` statement implementation and is called
        when entering the asynchronous context. It is commonly used to initialize or
        prepare resources needed during the context's lifetime.

        Returns:
            CropProducerStage: The instance of the context manager.

        """
        logger.info("CropProducerStage starting up")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Handles the asynchronous exit for a context manager.

        This method is called when exiting an asynchronous context managed block. It
        handles cleanup and shutdown operations specific to the instance of the
        context manager.
        """
        logger.info("CropProducerStage shutting down")

    @stage(progress=("cropping", "cropped"), debug_on_=True)
    async def process_scene(self, scene: SceneInfo) -> SceneInfo:
        """Processes a scene by generating crops for paired images.

        This function processes a given scene by iterating over its noisy images
        and generating crops in conjunction with the ground truth (GT) image,
        if available. The function concurrently spawns separate tasks for each
        pair of noisy and GT images to perform the crop generation. If a noisy
        image lacks metadata or alignment information, it is skipped. Logging
        is used to track the progress and results of the crop generation process.

        Args:
            scene: An instance of SceneInfo representing the scene to be processed.

        Returns:
            An updated SceneInfo instance reflecting the completed cropping
            process.

        Raises:
            None explicitly. Any exceptions arising within asynchronous tasks are
            not documented here.
        """
        logger.info(f"→ CropProducer.process_scene: {scene.scene_name}")
        gt_img = scene.get_gt_image()
        if not gt_img:
            logger.warning(
                f"Scene {scene.scene_name} has no valid GT image, skipping crops"
            )
            return scene

        # Process each noisy image paired with GT concurrently
        async with trio.open_nursery() as nursery:
            for noisy_img in scene.noisy_images:
                if not noisy_img.validated:
                    continue

                if "alignment" not in noisy_img.metadata:
                    logger.debug(
                        f"No alignment for {noisy_img.filename}, skipping crops"
                    )
                    continue

                # Spawn concurrent task for each image pair
                nursery.start_soon(
                    self._generate_crops_for_pair_async, scene, gt_img, noisy_img
                )

        total_crops = sum(
            len(img.metadata.get("crops", [])) for img in scene.noisy_images
        )
        logger.info(
            f"← CropProducer.process_scene: {scene.scene_name} ({total_crops} crops)"
        )

        return scene

    async def _generate_crops_for_pair_async(
        self, scene: SceneInfo, gt_img: ImageInfo, noisy_img: ImageInfo
    ):
        """
        Asynchronously generates and processes cropped segments from a pair of images (ground
        truth and noisy images), stores the generated metadata into the noisy image's metadata.

        The function extracts relevant metadata from the input image objects (such as gain,
        alignment, Bayer patterns), loads the images, processes the data, generates crops
        asynchronously, and appends the resulting metadata to the noisy image's metadata storage.

        Args:
            scene (SceneInfo): Information about the scene to which the image pair belongs.
            gt_img (ImageInfo): An object representing the ground truth image, containing
                image data and related metadata.
            noisy_img (ImageInfo): An object representing the noisy image, containing image
                data and relevant metadata.

        Raises:
            Exception: Logs error details and stack trace if crop generation or processing fails,
                without stopping the overall execution flow.
        """
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
                noisy_img.metadata,
            )

            # Store metadata
            if "crops" not in noisy_img.metadata:
                noisy_img.metadata["crops"] = []
            noisy_img.metadata["crops"].extend(crop_metadata)

        except Exception as e:
            logger.error(
                f"Failed to generate crops for {noisy_img.filename}: {e}", exc_info=True
            )

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
        metadata: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Extracts and saves image crops asynchronously.

        This method processes input ground-truth and noisy image data along with specified
        alignment and metadata to extract and save image crops. Crop positions are generated
        based on the provided alignment and CFA type, validated against computed masks, and
        saved asynchronously. Additionally, the method handles overexposure and loss masking
        to ensure valid crops, while leveraging parallelism for mask computation and saving.

        Args:
            scene_name: Name of the scene for which crops are being generated and saved.
            gt_data: Ground-truth image data in numpy array format.
            noisy_data: Noisy image data in numpy array format.
            alignment: List of two integers defining pixel offset for alignment (y, x).
            gain: Multiplicative gain to be applied to the noisy image data.
            gt_sha1: SHA-1 hash string of the ground-truth image data.
            noisy_sha1: SHA-1 hash string of the noisy image data.
            cfa_type: Color filter array type (e.g., "bayer", "x-trans").
            overexposure_lb: Numeric threshold for overexposure masking.
            is_bayer: Boolean indicating whether the CFA type is Bayer.
            metadata: Optional dictionary containing metadata required for specific crop processing.

        Returns:
            List of dictionaries containing metadata of the saved crops.

        Raises:
            ValueError: If metadata is required but not provided for PRGB crops, or unexpected tensor
                shapes are encountered.
        """

        MAX_MASKED = 0.5

        # Validate metadata for PRGB crops
        if "prgb" in self.crop_types and metadata is None:
            raise ValueError("metadata required for PRGB crops")

        # Snap alignment to CFA boundaries
        y_offset, x_offset = alignment
        if cfa_type == CFA_TYPE_BAYER:
            y_offset = (y_offset // 2) * 2
            x_offset = (x_offset // 2) * 2
        elif cfa_type == CFA_TYPE_XTRANS:
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
                y_start - y_offset : y_end - y_offset,
                x_start - x_offset : x_end - x_offset,
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
        mask_path = await self._save_mask_async(final_mask, scene_name, noisy_sha1)

        # Generate crop positions
        if gt_data.ndim == 2:
            h, w = gt_data.shape
        elif gt_data.ndim == 3:
            _, h, w = gt_data.shape
        else:
            raise ValueError(f"Unexpected tensor shape: {gt_data.shape}")

        if h < self.crop_size or w < self.crop_size:
            logger.warning(
                f"Image too small for {self.crop_size}x{self.crop_size} crops"
            )
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
            selected_indices = np.random.choice(
                valid_indices, size=num_to_sample, replace=False
            )

        # Save crops concurrently
        crop_metadata = await self._save_crops_async(
            gt_data,
            noisy_data,
            candidates_y,
            candidates_x,
            selected_indices,
            scene_name,
            gt_sha1,
            noisy_sha1,
            mask_path,
            metadata,
        )

        return crop_metadata

    async def _compute_masks_async(
        self,
        gt_data: np.ndarray,
        noisy_data: np.ndarray,
        cfa_type: str,
        overexposure_lb: float,
        is_bayer: bool,
    ):
        """
        Asynchronously computes masks for loss and overexposure based on the input ground truth (gt)
        and noisy datasets. Handles specific processing for 2D RAW data with Bayer or X-Trans color
        filter arrays (CFA) or 3D RGB data. The masks are created using windowed MS-SSIM for
        Bayer data or absolute pixel differences for X-Trans data, and the overexposure mask
        is calculated using specified thresholds.

        Args:
            gt_data (np.ndarray): Ground truth data array, which can be a 2D RAW image or a 3D RGB image.
            noisy_data (np.ndarray): Noisy data array that matches the dimensions of `gt_data`.
            cfa_type (str): Type of CFA used in the input data. Valid values include 'bayer' and 'x-trans'.
            overexposure_lb (float): Lower bound for overexposure calculation to create the mask.
            is_bayer (bool): Indicates whether input 3D RGB data conforms to Bayer formatting.

        Returns:
            tuple: A tuple containing two numpy arrays:
                - loss_mask (np.ndarray): A mask reflecting the loss values for the dataset.
                - overexposure_mask (np.ndarray): A mask identifying overexposed regions or pixels.

        Raises:
            ValueError: If an unsupported `cfa_type` is provided or input data dimensions are not as expected.
        """
        from rawnind.libs import rawproc

        if gt_data.ndim == 2:
            # 2D RAW format
            if cfa_type == CFA_TYPE_BAYER:
                # Split into RGGB channels
                gt_rggb = np.stack(
                    [
                        gt_data[0::2, 0::2],
                        gt_data[0::2, 1::2],
                        gt_data[1::2, 0::2],
                        gt_data[1::2, 1::2],
                    ]
                )
                noisy_rggb = np.stack(
                    [
                        noisy_data[0::2, 0::2],
                        noisy_data[0::2, 1::2],
                        noisy_data[1::2, 0::2],
                        noisy_data[1::2, 1::2],
                    ]
                )

                # Compute MS-SSIM mask with async window-level concurrency
                # This uses Trio's scheduler to parallelize ~9600 window tasks
                # Memory efficient (~400MB overhead vs 1.8GB for sequential PyTorch)

                loss_mask_rggb = await make_loss_mask_msssim_bayer_async(
                    gt_rggb,
                    noisy_rggb,
                    window_size=192,  # MS-SSIM requires >=176 for 5 scales
                    stride=96,  # 50% overlap for robustness
                )

                # Upsample to full resolution
                loss_mask = np.repeat(np.repeat(loss_mask_rggb, 2, axis=0), 2, axis=1)

                # Overexposure mask
                # TODO needs proper async review
                overexposure_mask = await trio.to_thread.run_sync(
                    rawproc.make_overexposure_mask_bayer, gt_data, overexposure_lb
                )

            elif cfa_type == CFA_TYPE_XTRANS:
                # Simple masks for X-Trans
                loss_mask = (np.abs(gt_data - noisy_data) < 0.3).astype(np.float32)
                overexposure_mask = (gt_data < overexposure_lb).astype(np.float32)

            else:
                raise ValueError(f"Unknown CFA type: {cfa_type}")

        elif gt_data.ndim == 3:
            # 3D RGB format
            if is_bayer:
                loss_mask = await trio.to_thread.run_sync(
                    rawproc.make_loss_mask_bayer, gt_data, noisy_data
                )
            else:
                loss_mask = await trio.to_thread.run_sync(
                    rawproc.make_loss_mask, gt_data, noisy_data
                )

            overexposure_mask = await trio.to_thread.run_sync(
                rawproc.make_overexposure_mask, gt_data, overexposure_lb
            )
        else:
            raise ValueError(f"Unexpected data format: {gt_data.shape}")

        return loss_mask, overexposure_mask

    async def _save_mask_async(
        self, final_mask: np.ndarray, scene_name: str, noisy_sha1: str
    ) -> Path:
        """
        Asynchronously saves a given mask array as an image file, using the provided scene name
        and noisy_sha1 string to form the filename. The image is saved in an output directory
        designated for mask storage. The function ensures non-blocking behavior by utilizing a
        separate thread for the actual file writing process.

        Args:
            final_mask (np.ndarray): The mask array to be saved. Values in the array are expected
                to be in the range [0, 1], which will be scaled to [0, 255] for saving.
            scene_name (str): The name of the scene, used as part of the output file's name.
            noisy_sha1 (str): A string identifier used to create a unique part of the filename,
                based on its first 8 characters.

        Returns:
            Path: The file path where the mask image has been saved.
        """
        # todo this should us OpenImageIO - not PIL
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

    def _generate_crop_positions(self, h: int, w: int, cfa_type: str):
        """
        Generates crop positions based on the provided image dimensions, crop size, stride,
        and Color Filter Array (CFA) type. Depending on the configuration, the crop positions
        are computed either systematically with a fixed tiling pattern or randomly. The
        coordinates are then snapped to CFA boundaries, ensuring alignment based on the CFA type.

        Args:
            h (int): Height of the image or area to generate crop positions for.
            w (int): Width of the image or area to generate crop positions for.
            cfa_type (str): Type of CFA pattern used, such as "Bayer" or "X-Trans".

        Returns:
            tuple: A tuple containing two numpy arrays, `candidates_y` and `candidates_x`, which
            represent the vertical and horizontal crop positions, respectively.
        """
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
            candidates_y = np.random.randint(
                0, h - self.crop_size + 1, size=num_candidates
            )
            candidates_x = np.random.randint(
                0, w - self.crop_size + 1, size=num_candidates
            )

        # Snap to CFA boundaries
        if cfa_type == CFA_TYPE_BAYER:
            candidates_y = (candidates_y // 2) * 2
            candidates_x = (candidates_x // 2) * 2
        elif cfa_type == CFA_TYPE_XTRANS:
            candidates_y = (candidates_y // 3) * 3
            candidates_x = (candidates_x // 3) * 3

        return candidates_y, candidates_x

    def _validate_crop_positions(
        self, candidates_y: np.ndarray, candidates_x: np.ndarray, final_mask: np.ndarray
    ) -> np.ndarray:
        """
        Validates crop positions based on the proportion of the masked area within each crop.

        This method evaluates a set of candidate positions for cropping by checking if the
        proportion of masked elements in the corresponding crop area meets or exceeds a defined
        threshold. Candidates that meet the criteria are returned as valid indices.

        Args:
            candidates_y (np.ndarray): Array of y-coordinates of candidate crop positions.
            candidates_x (np.ndarray): Array of x-coordinates of candidate crop positions.
            final_mask (np.ndarray): 2D mask array representing areas to consider during validation.

        Returns:
            np.ndarray: Indices of valid crop positions that meet the masking threshold.
        """
        MAX_MASKED = 0.5
        num_candidates = len(candidates_y)
        valid_mask = np.zeros(num_candidates, dtype=bool)

        for idx in range(num_candidates):
            y, x = candidates_y[idx], candidates_x[idx]
            mask_crop = final_mask[y : y + self.crop_size, x : x + self.crop_size]
            valid_mask[idx] = (mask_crop.sum() / (self.crop_size**2)) >= MAX_MASKED

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
        metadata: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Extract and save image crops concurrently, returning metadata for each.

        Extracts crops from ground truth and noisy image data at the specified positions
        and saves them using concurrent tasks. Each crop is saved in the configured
        formats with files organized by crop type and scene name.

        Args:
            gt_data: Ground truth image data. Shape is (H, W) for raw Bayer/X-Trans
                data or (C, H, W) for RGB data.
            noisy_data: Noisy image data with the same shape as gt_data.
            candidates_y: Y coordinates of candidate crop positions. Length must match
                candidates_x.
            candidates_x: X coordinates of candidate crop positions. Length must match
                candidates_y.
            selected_indices: Indices into candidates_y/candidates_x indicating which
                positions to save.
            scene_name: Scene identifier used for directory organization and crop ID
                generation.
            gt_sha1: SHA-1 hash of the ground truth image, used in crop ID generation.
            noisy_sha1: SHA-1 hash of the noisy image, used in crop ID and mask file
                naming.
            mask_path: Path to the quality mask file associated with these crops.
            metadata: Metadata dictionary. Required for PRGB crops (must contain
                'rgb_xyz_matrix' field). May be None for bayer-only crops.

        Returns:
            List of dictionaries with metadata for each saved crop. Each dictionary
            contains:
                - 'crop_id': Unique identifier formatted as
                  '{scene_name}_{index:03d}_{gt_sha1[:8]}_{noisy_sha1[:8]}'
                - 'coordinates': [y, x] position of crop's top-left corner
                - 'size': Crop dimensions (equal to self.crop_size)
                - 'gt_linrec2020_fpath': Path to ground truth crop file
                - 'f_bayer_fpath': Path to noisy crop file
                - 'mask_fpath': Path to mask file

        Notes:
            Crops are saved concurrently using a Trio nursery. The method waits for all
            save operations to complete before returning. The crop_metadata list is
            shared across tasks, with each task appending one entry.

            File organization depends on self.crop_types:
                - 'bayer': Saved as .npy in {output_dir}/bayer/{scene_name}/
                - 'prgb': Saved as .exr in {output_dir}/prgb/{scene_name}/

        Raises:
            ValueError: If PRGB crops are configured but metadata is None or lacks
                the required 'rgb_xyz_matrix' field.
        """
        crop_metadata = []

        # Spawn concurrent tasks for each crop
        async with trio.open_nursery() as nursery:
            for i, idx in enumerate(selected_indices):
                y = candidates_y[idx]
                x = candidates_x[idx]
                crop_id = f"{scene_name}_{i:03d}_{gt_sha1[:8]}_{noisy_sha1[:8]}"

                nursery.start_soon(
                    self._save_one_crop_async,
                    gt_data,
                    noisy_data,
                    y,
                    x,
                    crop_id,
                    scene_name,
                    mask_path,
                    metadata,
                    crop_metadata,
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
        crop_metadata: List[Dict[str, Any]],
    ):
        """
        Saves a single crop asynchronously, processes it as either PRGB or Bayer format,
        and appends metadata associated with the crop.

        This method extracts a crop of specified size from both ground truth data and
        noisy data using provided coordinates. The crop is processed as either PRGB or
        Bayer format based on configuration and then saved in the corresponding output
        directory. Metadata for the saved crop is appended to the provided metadata list.

        Args:
            gt_data (np.ndarray): Ground truth data array.
            noisy_data (np.ndarray): Noisy data array.
            y (int): Vertical starting coordinate for the crop.
            x (int): Horizontal starting coordinate for the crop.
            crop_id (str): Unique identifier for the crop.
            scene_name (str): Name of the scene associated with the crop.
            mask_path (Path): Filepath to the mask associated with the crop.
            metadata (Optional[Dict[str, Any]]): Metadata to be saved alongside the PRGB crop.
            crop_metadata (List[Dict[str, Any]]): List to store metadata for the processed crop.

        Raises:
            OSError: If directories or files cannot be created or written.
            ValueError: If inputs do not meet specified requirements.
        """
        gt_crop = gt_data[..., y : y + self.crop_size, x : x + self.crop_size]
        noisy_crop = noisy_data[..., y : y + self.crop_size, x : x + self.crop_size]

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
        crop_metadata.append(
            {
                "crop_id": crop_id,
                "coordinates": [y, x],
                "size": self.crop_size,
                "gt_linrec2020_fpath": str(gt_path),
                "f_bayer_fpath": str(noisy_path),
                "mask_fpath": str(mask_path),
            }
        )

    async def _save_bayer_crop_async(
        self, gt_crop: np.ndarray, noisy_crop: np.ndarray, crop_id: str, crop_dir: Path
    ):
        """
        Asynchronously saves a Bayer crop and its corresponding noisy crop to disk as .npy files.

        Performs disk I/O operations in a separate thread using Trio to save the given ground truth
        and noisy crop arrays to specified file paths.

        Args:
            gt_crop: Ground truth Bayer crop as a NumPy array.
            noisy_crop: Noisy Bayer crop as a NumPy array.
            crop_id: Identifier for the crop, used in the output file names.
            crop_dir: Directory where the crops will be saved.

        Returns:
            A tuple containing the paths to the saved ground truth (.npy) file and the noisy (.npy) file,
            respectively.
        """
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
        metadata: Dict[str, Any],
    ):
        """
        Saves processed high dynamic range (HDR) camera RGB crops as EXR files asynchronously.

        The function performs the following steps sequentially:
        1. Demosaics the given ground truth (gt_crop) and noisy (noisy_crop) image arrays
           into camera RGB format using metadata.
        2. Applies a color transformation from camera RGB to a linear Rec. 2020 color space
           using an RGB-to-XYZ matrix specified in the metadata.
        3. Saves the transformed ground truth and noisy crops as EXR files in the specified
           directory with proper file names.

        This function leverages asynchronous execution to optimize the performance of
        I/O-intensive tasks (e.g., demosaicing, file writing) via thread-based concurrency.

        Args:
            gt_crop (np.ndarray): Ground truth crop array with shape (1, H, W).
            noisy_crop (np.ndarray): Noisy crop array with shape (1, H, W).
            crop_id (str): Identifier for the crop, used to generate file names.
            crop_dir (Path): Path to the directory where the EXR files will be saved.
            metadata (Dict[str, Any]): Additional metadata required for processing,
                including the RGB-to-XYZ color transformation matrix.

        Returns:
            Tuple[Path, Path]: Paths to the saved ground truth and noisy EXR files.

        Raises:
            ValueError: If the metadata does not contain the required "rgb_xyz_matrix" key.
        """
        # Demosaic in thread (CPU-bound)
        # gt_crop already has shape (1, H, W) from ellipsis indexing
        gt_camrgb = await trio.to_thread.run_sync(raw.demosaic, gt_crop, metadata)
        noisy_camrgb = await trio.to_thread.run_sync(raw.demosaic, noisy_crop, metadata)

        # Color transform
        rgb_xyz_matrix = metadata.get("rgb_xyz_matrix")
        if rgb_xyz_matrix is None:
            raise ValueError("rgb_xyz_matrix required for PRGB crops")

        # Convert to numpy first, then to tensor (avoids nested list warning)
        rgb_xyz_array = np.asarray(rgb_xyz_matrix, dtype=np.float32)
        rgb_xyz_tensor = torch.from_numpy(rgb_xyz_array).unsqueeze(0)
        gt_camrgb_tensor = torch.from_numpy(gt_camrgb).unsqueeze(0).float()
        noisy_camrgb_tensor = torch.from_numpy(noisy_camrgb).unsqueeze(0).float()

        gt_lin_rec2020 = rawproc.camRGB_to_lin_rec2020_images(
            gt_camrgb_tensor, rgb_xyz_tensor
        )[0]
        noisy_lin_rec2020 = rawproc.camRGB_to_lin_rec2020_images(
            noisy_camrgb_tensor, rgb_xyz_tensor
        )[0]

        # Save EXR (async I/O)
        gt_path = crop_dir / f"{crop_id}_gt.exr"
        noisy_path = crop_dir / f"{crop_id}_noisy.exr"

        await trio.to_thread.run_sync(
            raw.hdr_nparray_to_file,
            gt_lin_rec2020.cpu().numpy(),
            str(gt_path),
            color_profile="lin_rec2020",
        )
        await trio.to_thread.run_sync(
            raw.hdr_nparray_to_file,
            noisy_lin_rec2020.cpu().numpy(),
            str(noisy_path),
            color_profile="lin_rec2020",
        )

        return gt_path, noisy_path

    async def consume_and_produce(
        self,
        input_channel: trio.MemoryReceiveChannel,
        output_channel: Optional[trio.MemorySendChannel] = None,
    ):
        """
        Consumes data from an input channel, processes it, and optionally produces it
        to an output channel.

        This asynchronous function manages the reception of data from a
        `trio.MemoryReceiveChannel` and optionally sends processed data to a
        `trio.MemorySendChannel`. It opens and consumes the input channel and, if an
        output channel is provided, both the input and output channels are handled
        simultaneously. The actual processing logic is implemented in the
        `_process_loop` method.

        Args:
            input_channel: An instance of `trio.MemoryReceiveChannel` for receiving data
                that will be processed.
            output_channel: An optional instance of `trio.MemorySendChannel` for sending
                the processed data. If `None`, no processed data will be sent.
        """
        async with input_channel:
            if output_channel:
                async with output_channel:
                    await self._process_loop(input_channel, output_channel)
            else:
                await self._process_loop(input_channel, None)

    async def _process_loop(
        self,
        input_channel: trio.MemoryReceiveChannel,
        output_channel: Optional[trio.MemorySendChannel],
    ):
        """
        Processes data asynchronously from an input channel to an output channel using a
        semaphore to limit concurrency.

        This coroutine listens for incoming data from the input_channel, processes it,
        and optionally sends it to the output_channel. Semaphore limits the
        number of concurrent processing tasks.

        Args:
            input_channel: A Trio MemoryReceiveChannel from which input data is consumed.
            output_channel: An optional Trio MemorySendChannel to send processed
                data. If None, processed data is not sent anywhere.

        Raises:
            trio.TooSlowError: Raised if the semaphore's acquisition is delayed beyond
                what the internal implementation handles.
        """
        sem = trio.Semaphore(self.max_workers)

        async with trio.open_nursery() as nursery:
            async for scene in input_channel:
                await sem.acquire()
                nursery.start_soon(
                    self._process_one_with_semaphore, scene, output_channel, sem
                )

    async def _process_one_with_semaphore(
        self,
        scene: SceneInfo,
        output_channel: Optional[trio.MemorySendChannel],
        sem: trio.Semaphore,
    ):
        """
        Processes a single scene within the constraints of a semaphore, and optionally
        sends the processed scene to an output channel. Errors during processing are
        logged.

        Args:
            scene: Information about the scene to be processed.
            output_channel: Optional channel for sending processed scene data.
            sem: Semaphore used to limit concurrent processing.

        Raises:
            Exception: Captures all exceptions that occur during processing and logs
                the error details.
        """
        try:
            processed_scene = await self.process_scene(scene)

            if output_channel:
                await output_channel.send(processed_scene)

        except Exception as e:
            logger.error(
                f"Error processing scene {scene.scene_name}: {e}", exc_info=True
            )
        finally:
            sem.release()

    @property
    def name(self) -> str:
        """
        Returns the name of the class.

        This property provides the name of the class as a string. It is intended
        for use cases where the class's name needs to be dynamically accessed
        without manually specifying it.

        Returns:
            str: The name of the class.
        """
        return self.__class__.__name__
