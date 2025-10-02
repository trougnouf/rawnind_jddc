"""
Image alignment backends for RawNIND dataset preparation.

This module provides alignment backends for noisy RAW images to clean ground truth.

Key insight: Alignment operates directly on RAW/mosaiced data using CFA-aware FFT.
This avoids wasteful demosaicing just to produce shift metadata.
"""
import numpy as np
from typing import Union, Tuple, Optional, List, Literal

try:
    import torch
except ImportError:
    torch = None

# Constants
MAX_SHIFT_SEARCH = 128
NEIGHBORHOOD_SEARCH_WINDOW = 3


def np_l1(img1: np.ndarray, img2: np.ndarray, avg=True) -> Union[float, np.ndarray]:
    """Compute L1 loss between two images."""
    if avg:
        return np.abs(img1 - img2).mean()
    return np.abs(img1 - img2)


def match_gain(
    anchor_img: Union[np.ndarray, torch.Tensor],
    other_img: Union[np.ndarray, torch.Tensor],
    return_val: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """Match gain for a single or batched pair of images; other_img is adapted to anchor_img."""
    if anchor_img.ndim == 4:
        anchor_avg = anchor_img.mean((-1, -2, -3)).reshape(-1, 1, 1, 1)
        other_avg = other_img.mean((-1, -2, -3)).reshape(-1, 1, 1, 1)
    elif anchor_img.ndim == 3:
        anchor_avg = anchor_img.mean()
        other_avg = other_img.mean()
    else:
        raise ValueError(f"{anchor_img.ndim=}")
    if return_val:
        return anchor_avg / other_avg
    return other_img * (anchor_avg / other_avg)


def shift_images(
    anchor_img: Union[np.ndarray, torch.Tensor],
    target_img: Union[np.ndarray, torch.Tensor],
    shift: tuple,
) -> Union[tuple, tuple]:
    """
    Shift images in y,x directions and crop both accordingly.
    
    Handles both RGB and Bayer patterns correctly.
    """
    anchor_img_out = anchor_img
    target_img_out = target_img
    target_is_bayer = target_img.shape[0] == 4
    if anchor_img.shape[0] == 4:
        raise NotImplementedError("shift_images: Bayer anchor_img is not implemented.")
    target_shift_divisor = target_is_bayer + 1
    
    if shift[0] > 0:  # y
        anchor_img_out = anchor_img_out[..., shift[0] :, :]
        target_img_out = target_img_out[
            ..., : -(shift[0] // target_shift_divisor) or None, :
        ]
        if shift[0] % 2:
            anchor_img_out = anchor_img_out[..., :-1, :]
            target_img_out = target_img_out[..., :-1, :]

    elif shift[0] < 0:  # y
        anchor_img_out = anchor_img_out[..., : shift[0], :]
        target_img_out = target_img_out[
            ..., -(shift[0] // target_shift_divisor) :, :
        ]
        if shift[0] % 2:
            anchor_img_out = anchor_img_out[..., 1:, :]
            target_img_out = target_img_out[..., 1:, :]

    if shift[1] > 0:  # x
        anchor_img_out = anchor_img_out[..., shift[1] :]
        target_img_out = target_img_out[
            ..., : -(shift[1] // target_shift_divisor) or None
        ]
        if shift[1] % 2:
            anchor_img_out = anchor_img_out[..., :-1]
            target_img_out = target_img_out[..., :-1]

    elif shift[1] < 0:  # x
        anchor_img_out = anchor_img_out[..., : shift[1]]
        target_img_out = target_img_out[..., -(shift[1] // target_shift_divisor) :]
        if shift[1] % 2:
            anchor_img_out = anchor_img_out[..., 1:]
            target_img_out = target_img_out[..., 1:]

    return anchor_img_out, target_img_out


def find_best_alignment_cpu(
    anchor_raw: np.ndarray,
    target_raw: np.ndarray,
    anchor_metadata: dict,
    method: Literal["median", "mean"] = "median",
    return_loss_too: bool = False,
    verbose: bool = False,
) -> Union[Tuple[int, int], Tuple[Tuple[int, int], float]]:
    """
    CPU-based CFA-aware FFT alignment for RAW images.
    
    This is just an alias for find_best_alignment_fft_cfa() for consistency.
    Hierarchical methods are NOT used - they're fundamentally broken on RAW/CFA data.
    
    See docs/BENCHMARK_FINDINGS.md: "Hierarchical method fundamentally broken on RAW/CFA data"
    """
    return find_best_alignment_fft_cfa(
        anchor_raw, target_raw, anchor_metadata, method, return_loss_too, verbose
    )


def find_best_alignment_fft_cfa(
    anchor_raw: np.ndarray,
    target_raw: np.ndarray,
    anchor_metadata: dict,
    method: Literal["median", "mean"] = "median",
    return_loss_too: bool = False,
    verbose: bool = False,
) -> Union[Tuple[int, int], Tuple[Tuple[int, int], float]]:
    """
    CFA-aware FFT alignment for RAW/mosaiced images.
    
    This uses the production FFT implementation from raw.py that operates directly
    on RAW mosaiced data, avoiding wasteful demosaicing just to get shift metadata.
    
    Args:
        anchor_raw: Reference RAW image [1, H, W]
        target_raw: Target RAW image to align [1, H, W]  
        anchor_metadata: Metadata dict containing 'RGBG_pattern'
        method: 'median' or 'mean' for combining channel shifts
        return_loss_too: If True, compute and return L1 loss after alignment
        verbose: Print per-channel shift detections
        
    Returns:
        shift: (dy, dx) tuple
        OR (shift, loss) if return_loss_too=True
    """
    from rawnind.libs import raw
    
    # Run CFA-aware FFT
    shift, channel_shifts = raw.fft_phase_correlate_cfa(
        anchor_raw, target_raw, anchor_metadata, method=method, verbose=verbose
    )
    
    if return_loss_too:
        # Compute L1 loss after alignment
        anchor_aligned, target_aligned = shift_images(anchor_raw, target_raw, shift)
        loss = np_l1(anchor_aligned, target_aligned)
        return shift, float(loss)
    
    return shift
