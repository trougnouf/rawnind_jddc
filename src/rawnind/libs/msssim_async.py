"""Utility functions for asynchronous multi‑scale SSIM computation and loss‑mask
generation for Bayer‑pattern images.

The module provides two public coroutine functions. ``compute_window`` extracts a
square region from a pair of multi‑channel images, selects an appropriate SSIM
variant based on the window size, and adds the resulting similarity score to
shared accumulation arrays while protecting concurrent writes with per‑channel
asynchronous locks. ``make_loss_mask_msssim_bayer_async`` aligns a target RGGB
image to a reference image, computes a per‑pixel MS‑SSIM map (or a single SSIM
value when the image is too small), thresholds the map together with an L1‑
based mask, and applies a binary opening operation to produce a final loss mask
that indicates pixels satisfying both similarity criteria.

Both functions are designed to run under Trio's asynchronous scheduler and rely
on NumPy for array manipulation and SciPy for morphological operations. They
are intended for use in image‑processing pipelines where large Bayer images need
to be compared efficiently on a per‑pixel basis."""

import logging

import numpy as np
import scipy.ndimage
import trio

logger = logging.getLogger(__name__)


async def compute_window(
    ch: int,
    y: int,
    x: int,
    anchor_img: np.ndarray,
    target_matched: np.ndarray,
    ssim_maps: np.ndarray,
    count_maps: np.ndarray,
    locks: list,
    window_size: int,
    data_range: float = 1.0,
):
    """
    Computes the structural similarity index (SSIM) or multi‑scale SSIM (MS‑SSIM) for a
    specific channel and spatial window of two images, then atomically updates the
    corresponding entries in the provided accumulation arrays.

    Args:
        ch: Index of the channel to process.
        y: Vertical start coordinate of the window in the image arrays.
        x: Horizontal start coordinate of the window in the image arrays.
        anchor_img: Source image array from which the reference window is extracted.
        target_matched: Image array aligned with ``anchor_img`` used as the comparison
            target.
        ssim_maps: Array accumulating SSIM scores; the function adds the computed
            score to the region defined by ``y``, ``x``, and ``window_size``.
        count_maps: Array tracking how many scores have contributed to each pixel;
            the function increments the corresponding region by one.
        locks: List of asynchronous locks, one per channel, used to protect concurrent
            writes to ``ssim_maps`` and ``count_maps``.
        window_size: Length of the square window (in pixels) to extract from the
            images.
        data_range: Maximum possible value of the image data (default is ``1.0``).

    Returns:
        None
    """
    from common.libs.msssim_numpy import compute_ssim_numpy, compute_msssim_numpy

    # Extract window (fast, sync - no blocking)
    anchor_win = anchor_img[ch, y : y + window_size, x : x + window_size]
    target_win = target_matched[ch, y : y + window_size, x : x + window_size]

    # Choose SSIM variant based on window size
    # MS-SSIM requires min 176×176 for 5 scales, use single-scale SSIM for smaller windows
    if window_size >= 176:
        score = await trio.to_thread.run_sync(
            compute_msssim_numpy, anchor_win, target_win, data_range
        )
    else:
        # Use single-scale SSIM for small windows
        score = await trio.to_thread.run_sync(
            compute_ssim_numpy, anchor_win, target_win, data_range
        )

    # Accumulate (needs lock - multiple tasks write to shared arrays)
    async with locks[ch]:
        ssim_maps[ch, y : y + window_size, x : x + window_size] += score
        count_maps[ch, y : y + window_size, x : x + window_size] += 1


async def make_loss_mask_msssim_bayer_async(
    anchor_img: np.ndarray,
    target_img: np.ndarray,
    ssim_threshold: float = 0.7,
    l1_threshold: float = 0.4,
    window_size: int = 192,
    stride: int = 96,
    data_range: float = 1.0,
) -> np.ndarray:
    """
    Creates a loss mask for Bayer‑pattern images based on multi‑scale SSIM and L1
    differences.

    The function first aligns the target image to the anchor image using gain
    matching.  If the image dimensions are large enough, it computes a per‑pixel
    MS‑SSIM map by sliding a square window across each of the four RGGB channels,
    accumulating overlapping results, and averaging them.  For smaller images a
    single SSIM value per channel is computed and broadcast to the full image size.
    The MS‑SSIM map is thresholded, then combined with an L1‑based mask derived
    from gamma‑corrected images.  Pixels that satisfy both the SSIM and L1
    conditions are retained, and a binary opening operation is applied to remove
    small isolated regions.  The resulting mask has the same spatial dimensions as
    the input images and is returned as a ``float32`` array.

    Args:
        anchor_img: A NumPy array of shape ``(4, H, W)`` containing the reference
            RGGB image.
        target_img: A NumPy array of the same shape as ``anchor_img`` that will be
            compared against the reference.
        ssim_threshold: Threshold applied to the MS‑SSIM map; pixels with a value
            greater than this are considered similar.
        l1_threshold: Threshold applied to the summed L1 map; pixels with a value
            lower than this are considered similar.
        window_size: Length of the square window (in pixels) used for the MS‑SSIM
            computation.  If the image is smaller than this size, a full‑image SSIM
            is used instead.
        stride: Number of pixels to shift the window between successive
            computations; determines the amount of overlap.
        data_range: The dynamic range of the image data (e.g., ``1.0`` for
            normalized images) used by the SSIM calculation.

    Returns:
        A ``float32`` NumPy array of shape ``(H, W)`` where a value of ``1`` indicates
        that the corresponding pixel satisfies both the SSIM and L1 criteria after
        morphological cleanup, and ``0`` otherwise.

    Raises:
        ValueError: If ``anchor_img`` does not have three dimensions with a leading
            size of 4, or if ``target_img`` does not share the same shape as
            ``anchor_img``.
    """
    from rawnind.libs.rawproc import match_gain, gamma, np_l1

    if anchor_img.ndim != 3 or anchor_img.shape[0] != 4:
        raise ValueError(f"Expected (4, H, W) RGGB, got {anchor_img.shape}")

    if target_img.shape != anchor_img.shape:
        raise ValueError(f"Shape mismatch: {anchor_img.shape} vs {target_img.shape}")

    # Match gain
    target_matched = match_gain(anchor_img, target_img)

    num_channels, H, W = anchor_img.shape

    # Check window size is valid
    if H < window_size or W < window_size:
        logger.warning(
            f"Image too small ({H}×{W}) for window_size={window_size}, "
            f"falling back to full-image SSIM"
        )
        # For small images, compute single SSIM per channel
        from common.libs.msssim_numpy import compute_ssim_numpy

        ssim_scores = []
        for ch in range(num_channels):
            score = compute_ssim_numpy(anchor_img[ch], target_matched[ch], data_range)
            ssim_scores.append(score)
        ssim_map = np.mean(ssim_scores) * np.ones((H, W), dtype=np.float32)
    else:
        # Initialize accumulation arrays
        ssim_maps = np.zeros((num_channels, H, W), dtype=np.float32)
        count_maps = np.zeros((num_channels, H, W), dtype=np.float32)

        # Create per-channel locks for accumulation
        locks = [trio.Lock() for _ in range(num_channels)]

        # Spawn all window tasks (Trio handles scheduling)
        async with trio.open_nursery() as nursery:
            for ch in range(num_channels):
                for y in range(0, H - window_size + 1, stride):
                    for x in range(0, W - window_size + 1, stride):
                        nursery.start_soon(
                            compute_window,
                            ch,
                            y,
                            x,
                            anchor_img,
                            target_matched,
                            ssim_maps,
                            count_maps,
                            locks,
                            window_size,
                            data_range,
                        )

        # Average overlapping windows per channel
        for ch in range(num_channels):
            ssim_maps[ch] = np.divide(
                ssim_maps[ch], count_maps[ch], where=count_maps[ch] > 0
            )

        # Average MS-SSIM across all 4 RGGB channels
        ssim_map = np.mean(ssim_maps, axis=0)

    # Threshold MS-SSIM map
    ssim_mask = (ssim_map > ssim_threshold).astype(np.float32)

    # Combine with L1-based filtering (MS-SSIM + L1 is standard)
    l1_map = np_l1(gamma(anchor_img), gamma(target_matched), avg=False)
    l1_map = l1_map.sum(axis=0)
    l1_mask = (l1_map < l1_threshold).astype(np.float32)

    # Both conditions must be met
    loss_mask = ssim_mask * l1_mask

    # Apply morphological cleanup
    loss_mask = scipy.ndimage.binary_opening(loss_mask.astype(np.uint8)).astype(
        np.float32
    )

    return loss_mask


async def make_loss_mask_msssim_bayer_async_with_progress(
    anchor_img: np.ndarray,
    target_img: np.ndarray,
    ssim_threshold: float = 0.7,
    l1_threshold: float = 0.4,
    window_size: int = 192,
    stride: int = 96,
    data_range: float = 1.0,
    progress_callback: callable = None,
) -> np.ndarray:
    """
    Computes a binary loss mask for a pair of Bayer images using a multi‑scale
    structural similarity index (MS‑SSIM) combined with an L1‑based threshold. The
    function matches the gain of the target image to the anchor, evaluates SSIM
    over overlapping windows (optionally in parallel), and optionally reports
    progress via a callback.

    Args:
        anchor_img: Raw Bayer image with shape (4, H, W) representing RGGB data.
        target_img: Raw Bayer image to compare against the anchor.
        ssim_threshold: Threshold for the averaged SSIM map; pixels with a value
            greater than this threshold are retained in the mask.
        l1_threshold: Threshold for the L1 difference map; pixels with a value
            lower than this threshold are retained in the mask.
        window_size: Size of the square window used for SSIM computation.
        stride: Step size between adjacent windows.
        data_range: Dynamic range of the input data for SSIM calculation.
        progress_callback: Optional asynchronous callable that receives two
            integers—completed windows and total windows—to report progress.

    Returns:
        A 2‑D NumPy array of shape (H, W) containing a binary mask (float32) where
        both the SSIM and L1 thresholds are satisfied. The mask is post‑processed
        with a binary opening operation.

    Raises:
        ValueError: If ``anchor_img`` does not have three dimensions or its first
            dimension is not equal to 4 (expected RGGB layout).
    """
    from rawnind.libs.rawproc import match_gain, gamma, np_l1

    if anchor_img.ndim != 3 or anchor_img.shape[0] != 4:
        raise ValueError(f"Expected (4, H, W) RGGB, got {anchor_img.shape}")

    target_matched = match_gain(anchor_img, target_img)
    num_channels, H, W = anchor_img.shape

    if H < window_size or W < window_size:
        # Fallback for small images
        from common.libs.msssim_numpy import compute_ssim_numpy

        ssim_scores = []
        for ch in range(num_channels):
            score = compute_ssim_numpy(anchor_img[ch], target_matched[ch], data_range)
            ssim_scores.append(score)
        ssim_map = np.mean(ssim_scores) * np.ones((H, W), dtype=np.float32)
    else:
        ssim_maps = np.zeros((num_channels, H, W), dtype=np.float32)
        count_maps = np.zeros((num_channels, H, W), dtype=np.float32)
        locks = [trio.Lock() for _ in range(num_channels)]

        # Count total windows
        total_windows = 0
        for ch in range(num_channels):
            for y in range(0, H - window_size + 1, stride):
                for x in range(0, W - window_size + 1, stride):
                    total_windows += 1

        completed_windows = 0
        progress_lock = trio.Lock()

        async def compute_window_with_progress(*args, **kwargs):
            """
            Computes a single window and updates progress tracking.

            This coroutine forwards all positional and keyword arguments to the
            `compute_window` coroutine. If a progress callback is provided, it
            increments a shared counter that tracks completed windows. The callback
            is invoked after every 100 windows, receiving the number of windows
            processed so far and the total number of windows to be processed.

            Args:
                *args: Positional arguments passed to `compute_window`.
                **kwargs: Keyword arguments passed to `compute_window`.

            Returns:
                None
            """
            nonlocal completed_windows
            await compute_window(*args, **kwargs)

            if progress_callback:
                async with progress_lock:
                    completed_windows += 1
                    if completed_windows % 100 == 0:  # Report every 100 windows
                        await progress_callback(completed_windows, total_windows)

        async with trio.open_nursery() as nursery:
            for ch in range(num_channels):
                for y in range(0, H - window_size + 1, stride):
                    for x in range(0, W - window_size + 1, stride):
                        nursery.start_soon(
                            compute_window_with_progress,
                            ch,
                            y,
                            x,
                            anchor_img,
                            target_matched,
                            ssim_maps,
                            count_maps,
                            locks,
                            window_size,
                            data_range,
                        )

        # Final progress report
        if progress_callback:
            await progress_callback(total_windows, total_windows)

        # Average overlapping windows per channel
        for ch in range(num_channels):
            ssim_maps[ch] = np.divide(
                ssim_maps[ch], count_maps[ch], where=count_maps[ch] > 0
            )

        ssim_map = np.mean(ssim_maps, axis=0)

    # Threshold and combine with L1
    ssim_mask = (ssim_map > ssim_threshold).astype(np.float32)
    l1_map = np_l1(gamma(anchor_img), gamma(target_matched), avg=False).sum(axis=0)
    l1_mask = (l1_map < l1_threshold).astype(np.float32)
    loss_mask = ssim_mask * l1_mask
    loss_mask = scipy.ndimage.binary_opening(loss_mask.astype(np.uint8)).astype(
        np.float32
    )

    return loss_mask
