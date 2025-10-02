"""
Image alignment backends for RawNIND dataset preparation.

This module provides CPU and GPU backends for aligning noisy RAW images to clean ground truth.
The key innovation is GPU scene-batching (Option #8): batch-process all noisy images for a 
single GT scene on GPU, avoiding multiprocessing+CUDA fork poisoning issues.

Key insight: Alignment operates directly on RAW/mosaiced data using CFA-aware FFT.
This avoids wasteful demosaicing just to produce shift metadata.
"""
import numpy as np
from typing import Union, Tuple, Optional, List, Literal

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


def find_best_alignment_gpu(
    anchor_raw: np.ndarray,
    target_raw: np.ndarray,
    anchor_metadata: dict,
    method: Literal["median", "mean"] = "median",
    return_loss_too: bool = False,
    verbose: bool = False,
) -> Union[Tuple[int, int], Tuple[Tuple[int, int], float]]:
    """
    GPU-accelerated FFT alignment for a single RAW image pair.
    
    For batch processing (recommended for GPU), use find_best_alignment_fft_cfa_batch().
    """
    # For single-pair, just call the batch function with one image
    results = find_best_alignment_fft_cfa_batch(
        anchor_raw, [target_raw], anchor_metadata, method, return_loss_too, verbose
    )
    
    return results[0]


def find_best_alignment_fft_cfa_batch(
    anchor_raw: np.ndarray,
    target_raws: List[np.ndarray],
    anchor_metadata: dict,
    method: Literal["median", "mean"] = "median",
    return_loss_too: bool = False,
    verbose: bool = False,
    use_gpu: bool = True,
) -> List[Union[Tuple[int, int], Tuple[Tuple[int, int], float]]]:
    """
    Batch CFA-aware FFT alignment for multiple RAW images against single GT.
    
    This implements GPU Hybrid Batching (Option #8): process all noisy images
    for a single GT scene together. Key advantage: natural batching (avg 8 noisy/GT),
    all images in a scene typically same size, avoids multiprocessing+CUDA issues.
    
    Args:
        anchor_raw: Reference RAW image [1, H, W]
        target_raws: List of target RAW images to align
        anchor_metadata: Metadata dict containing 'RGBG_pattern'
        method: 'median' or 'mean' for combining channel shifts
        return_loss_too: If True, compute and return L1 loss after alignment
        verbose: Print progress information
        use_gpu: If True and GPU available, use GPU-accelerated batch processing
        
    Returns:
        List of alignment results, one per target image
    """
    # Try GPU batch processing if requested and available
    if use_gpu and _gpu_available() and len(target_raws) > 1:
        try:
            return _batch_fft_correlation_gpu(
                anchor_raw, target_raws, anchor_metadata, method, return_loss_too, verbose
            )
        except Exception as e:
            if verbose:
                print(f"  GPU batch processing failed ({e}), falling back to CPU loop")
    
    # Fallback: CPU loop processing
    results = []
    for i, target_raw in enumerate(target_raws):
        if verbose and len(target_raws) > 1:
            print(f"  Aligning {i+1}/{len(target_raws)}...")
            
        result = find_best_alignment_fft_cfa(
            anchor_raw, target_raw, anchor_metadata, 
            method=method, return_loss_too=return_loss_too, verbose=False
        )
        results.append(result)
    
    return results


def _batch_fft_correlation_gpu(
    anchor_raw: np.ndarray,
    target_raws: List[np.ndarray],
    anchor_metadata: dict,
    method: Literal["median", "mean"] = "median",
    return_loss_too: bool = False,
    verbose: bool = False,
) -> List[Union[Tuple[int, int], Tuple[Tuple[int, int], float]]]:
    """
    GPU-accelerated batch FFT correlation using PyTorch.
    
    Processes all noisy images for one GT scene in a single GPU batch.
    """
    import torch
    from rawnind.libs import raw
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pattern = anchor_metadata['RGBG_pattern']
    pattern_shape = pattern.shape
    
    # Determine CFA type and scale factor
    if pattern_shape == (2, 2):
        scale_factor = 2
        extract_channels = raw.extract_bayer_channels
    elif pattern_shape == (6, 6):
        scale_factor = 6
        extract_channels = raw.extract_xtrans_channels
    else:
        raise ValueError(f"Unsupported CFA pattern shape: {pattern_shape}")
    
    # Extract anchor channels once (same for all targets)
    channels_anchor = extract_channels(anchor_raw, pattern)
    
    # Convert to torch tensors and move to GPU
    anchor_channels_gpu = {
        color: torch.from_numpy(ch).float().to(device)
        for color, ch in channels_anchor.items()
    }
    
    # Process each target image
    results = []
    for i, target_raw in enumerate(target_raws):
        if verbose:
            print(f"  GPU aligning {i+1}/{len(target_raws)}...")
        
        # Extract target channels
        channels_target = extract_channels(target_raw, pattern)
        
        # Perform FFT correlation on GPU for each color channel
        channel_shifts = []
        for color in anchor_channels_gpu.keys():
            ch_anchor_gpu = anchor_channels_gpu[color]
            ch_target_gpu = torch.from_numpy(channels_target[color]).float().to(device)
            
            # FFT phase correlation on GPU
            dy_down, dx_down = _fft_phase_correlate_single_gpu(ch_anchor_gpu, ch_target_gpu)
            
            # Scale back to full resolution
            dy = dy_down * scale_factor
            dx = dx_down * scale_factor
            
            channel_shifts.append((dy, dx))
        
        # Combine channel results
        if method == "median":
            dy_final = int(np.median([s[0] for s in channel_shifts]))
            dx_final = int(np.median([s[1] for s in channel_shifts]))
        elif method == "mean":
            dy_final = int(np.mean([s[0] for s in channel_shifts]))
            dx_final = int(np.mean([s[1] for s in channel_shifts]))
        else:
            raise ValueError(f"Unknown method: {method}")
        
        shift = (dy_final, dx_final)
        
        # Optionally compute loss
        if return_loss_too:
            # Move to CPU for loss computation to avoid extra GPU memory
            loss = _compute_alignment_loss_cpu(anchor_raw, target_raw, shift)
            results.append((shift, float(loss)))
        else:
            results.append(shift)
    
    return results


def _fft_phase_correlate_single_gpu(
    anchor_ch: torch.Tensor,
    target_ch: torch.Tensor
) -> Tuple[int, int]:
    """
    GPU-accelerated single-channel FFT phase correlation using PyTorch.
    
    Args:
        anchor_ch: Reference channel [H, W] on GPU
        target_ch: Target channel [H, W] on GPU
        
    Returns:
        (dy, dx): Detected shift in pixels
    """
    # Normalize to zero mean
    anchor_norm = anchor_ch - anchor_ch.mean()
    target_norm = target_ch - target_ch.mean()
    
    # FFT-based cross-correlation
    # Cross-correlation = IFFT(FFT(A) * conj(FFT(B)))
    fft_anchor = torch.fft.fft2(anchor_norm)
    fft_target = torch.fft.fft2(target_norm)
    
    # Cross-power spectrum
    cross_spectrum = fft_anchor * torch.conj(fft_target)
    
    # Inverse FFT to get correlation
    correlation = torch.fft.ifft2(cross_spectrum).real
    
    # Shift zero-frequency component to center
    correlation = torch.fft.fftshift(correlation)
    
    # Find peak
    peak_idx_flat = torch.argmax(correlation)
    peak_y = peak_idx_flat // correlation.shape[1]
    peak_x = peak_idx_flat % correlation.shape[1]
    
    # Convert to shift (displacement convention)
    center_y = correlation.shape[0] // 2
    center_x = correlation.shape[1] // 2
    
    dy = int(peak_y.item()) - center_y
    dx = int(peak_x.item()) - center_x
    
    return dy, dx


def _compute_alignment_loss_cpu(
    anchor_raw: np.ndarray,
    target_raw: np.ndarray,
    shift: Tuple[int, int]
) -> float:
    """Compute L1 loss after alignment on CPU."""
    anchor_shifted, target_shifted = shift_images(anchor_raw, target_raw, shift)
    return np_l1(anchor_shifted, target_shifted, avg=True)





def get_alignment_backend(
    backend: str = "auto",
    batch_mode: bool = False,
) -> callable:
    """
    Get the appropriate CFA-aware FFT alignment backend function.
    
    All backends use FFT-based alignment directly on RAW/mosaiced data.
    Hierarchical methods are deprecated (fundamentally broken on RAW data).
    
    Args:
        backend: "auto", "cpu", or "gpu" (currently "gpu" also uses CPU FFT)
        batch_mode: If True, return batch-capable function
        
    Returns:
        Alignment function with signature:
            (anchor_raw, target_raw(s), anchor_metadata, method, return_loss_too, verbose) -> results
    """
    if backend == "auto":
        backend = "gpu" if _gpu_available() else "cpu"
    
    # Note: GPU and CPU both use the same FFT implementation for now
    # GPU batch processing using PyTorch FFT is TODO
    if batch_mode:
        return find_best_alignment_fft_cfa_batch
    else:
        return find_best_alignment_fft_cfa
