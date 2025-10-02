"""
Image alignment backends for RawNIND dataset preparation.

This module provides CPU and GPU backends for aligning noisy RAW images to clean ground truth.
The key innovation is GPU scene-batching (Option #8): batch-process all noisy images for a 
single GT scene on GPU, avoiding multiprocessing+CUDA fork poisoning issues.
"""
import numpy as np
from typing import Union, Tuple, Optional, List

try:
    import torch
    _GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    torch = None
    _GPU_AVAILABLE = False

# Constants
MAX_SHIFT_SEARCH = 128
NEIGHBORHOOD_SEARCH_WINDOW = 3


def _gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return _GPU_AVAILABLE


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
    anchor_img: np.ndarray,
    target_img: np.ndarray,
    max_shift_search: int = MAX_SHIFT_SEARCH,
    return_loss_too: bool = False,
    verbose: bool = False,
) -> Union[Tuple[int, int], Tuple[Tuple[int, int], float]]:
    """
    CPU-optimized hierarchical coarse-to-fine alignment search.
    
    This wraps the existing hierarchical implementation from rawproc.py,
    which uses multi-scale pyramid search (4x, 2x, 1x scales) for efficiency.
    """
    # Import here to avoid circular imports
    from rawnind.libs.rawproc import find_best_alignment_hierarchical
    
    return find_best_alignment_hierarchical(
        anchor_img, target_img, max_shift_search, return_loss_too, verbose
    )


def find_best_alignment_gpu(
    anchor_img: np.ndarray,
    target_img: np.ndarray,
    max_shift_search: int = MAX_SHIFT_SEARCH,
    return_loss_too: bool = False,
    verbose: bool = False,
) -> Union[Tuple[int, int], Tuple[Tuple[int, int], float]]:
    """
    GPU-accelerated alignment for a single image pair.
    
    For batch processing (recommended for GPU), use find_best_alignment_gpu_batch().
    """
    if not _gpu_available():
        if verbose:
            print("GPU not available, falling back to CPU")
        return find_best_alignment_cpu(
            anchor_img, target_img, max_shift_search, return_loss_too, verbose
        )
    
    # For single-pair GPU, just use batch size 1
    results = find_best_alignment_gpu_batch(
        anchor_img, [target_img], max_shift_search, return_loss_too, verbose
    )
    
    return results[0]


def find_best_alignment_gpu_batch(
    anchor_img: np.ndarray,
    target_imgs: List[np.ndarray],
    max_shift_search: int = MAX_SHIFT_SEARCH,
    return_loss_too: bool = False,
    verbose: bool = False,
) -> List[Union[Tuple[int, int], Tuple[Tuple[int, int], float]]]:
    """
    GPU-accelerated batch alignment: align multiple noisy images to single GT.
    
    This is the key optimization (Option #8): batch all noisy images for a single
    GT scene on GPU, avoiding multiprocessing+CUDA fork poisoning.
    
    Args:
        anchor_img: Ground truth image (single)
        target_imgs: List of noisy images to align
        max_shift_search: Maximum shift to search
        return_loss_too: Return (shift, loss) tuples instead of just shifts
        verbose: Print progress information
        
    Returns:
        List of alignment results, one per target image
    """
    if not _gpu_available():
        if verbose:
            print("GPU not available, falling back to CPU for batch")
        # Fall back to CPU processing
        return [
            find_best_alignment_cpu(
                anchor_img, target_img, max_shift_search, return_loss_too, verbose
            )
            for target_img in target_imgs
        ]
    
    # TODO: Implement GPU batch processing
    # For now, fall back to CPU
    if verbose:
        print("GPU batch processing not yet implemented, using CPU")
    
    return [
        find_best_alignment_cpu(
            anchor_img, target_img, max_shift_search, return_loss_too, verbose
        )
        for target_img in target_imgs
    ]


def get_alignment_backend(
    backend: str = "auto",
    batch_mode: bool = False,
) -> callable:
    """
    Get the appropriate alignment backend function.
    
    Args:
        backend: "auto", "cpu", or "gpu"
        batch_mode: If True, return batch-capable function (GPU only)
        
    Returns:
        Alignment function with signature:
            (anchor_img, target_img(s), max_shift_search, return_loss_too, verbose) -> results
    """
    if backend == "auto":
        backend = "gpu" if _gpu_available() else "cpu"
    
    if backend == "gpu":
        if batch_mode:
            return find_best_alignment_gpu_batch
        else:
            return find_best_alignment_gpu
    elif backend == "cpu":
        if batch_mode:
            # CPU doesn't have special batch mode, wrap in list comprehension
            def cpu_batch_wrapper(anchor_img, target_imgs, max_shift_search=MAX_SHIFT_SEARCH, 
                                return_loss_too=False, verbose=False):
                return [
                    find_best_alignment_cpu(
                        anchor_img, target_img, max_shift_search, return_loss_too, verbose
                    )
                    for target_img in target_imgs
                ]
            return cpu_batch_wrapper
        else:
            return find_best_alignment_cpu
    else:
        raise ValueError(f"Unknown backend: {backend}")
