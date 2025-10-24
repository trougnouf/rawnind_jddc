"""Pure NumPy/SciPy implementation of Multi-Scale Structural Similarity (MS-SSIM).

This module provides synchronous, CPU-based MS-SSIM metrics without PyTorch dependencies,
suitable for use with trio.to_thread.run_sync() in async contexts or standalone.

References:
- Z. Wang, E. P. Simoncelli, A. C. Bovik, "Multiscale Structural Similarity for
  Image Quality Assessment," 37th Asilomar Conf. on Signals, Systems and Computers, 2003.
- pytorch-msssim: https://github.com/VainF/pytorch-msssim
- fastmetrics: https://github.com/gdemaude/fastmetrics
"""

import numpy as np
from scipy import ndimage
from functools import lru_cache
from typing import Tuple, Optional


# Standard MS-SSIM scale weights from Wang et al. 2003
DEFAULT_WEIGHTS = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=np.float64)


@lru_cache(maxsize=128)
def _create_gaussian_kernel_1d(window_size: int, sigma: float) -> np.ndarray:
    """Create a 1D Gaussian kernel for separable convolution.

    Args:
        window_size: Size of the Gaussian window (must be odd).
        sigma: Standard deviation of the Gaussian.

    Returns:
        1D numpy array of shape (window_size,) containing normalized Gaussian kernel.
    """
    if window_size % 2 == 0:
        window_size += 1  # Ensure odd size

    coords = np.arange(window_size, dtype=np.float64)
    coords -= (window_size - 1) / 2.0

    g = np.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g /= g.sum()

    return g


def _gaussian_filter_separable(img: np.ndarray, window_size: int, sigma: float) -> np.ndarray:
    """Apply separable Gaussian filter (2× 1D convolutions).

    Uses scipy's convolve1d which properly handles boundaries with reflection.

    Args:
        img: Input image of shape (H, W).
        window_size: Size of the Gaussian window.
        sigma: Standard deviation of the Gaussian.

    Returns:
        Filtered image of same shape as input.
    """
    kernel = _create_gaussian_kernel_1d(window_size, sigma)

    # Apply separable filter: first along axis 0 (vertical), then axis 1 (horizontal)
    # Use 'reflect' mode for proper boundary handling (matches PyTorch's reflection padding)
    filtered = ndimage.convolve1d(img, kernel, axis=0, mode='reflect')
    filtered = ndimage.convolve1d(filtered, kernel, axis=1, mode='reflect')

    return filtered


def _compute_ssim_components(
    img1: np.ndarray,
    img2: np.ndarray,
    window_size: int,
    sigma: float,
    K1: float,
    K2: float,
    data_range: float,
    use_separable: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute SSIM luminance and contrast-structure components.

    Args:
        img1: First image (H, W).
        img2: Second image (H, W).
        window_size: Gaussian window size.
        sigma: Gaussian sigma.
        K1: SSIM constant for luminance.
        K2: SSIM constant for contrast-structure.
        data_range: Dynamic range of input images.
        use_separable: Whether to use separable Gaussian filters.

    Returns:
        Tuple of (luminance, contrast_structure) maps.
    """
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    if use_separable:
        filter_fn = lambda x: _gaussian_filter_separable(x, window_size, sigma)
    else:
        filter_fn = lambda x: ndimage.gaussian_filter(x, sigma=sigma, mode='reflect')

    # Compute local means
    mu1 = filter_fn(img1)
    mu2 = filter_fn(img2)

    # Compute local variances and covariance
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = filter_fn(img1 ** 2) - mu1_sq
    sigma2_sq = filter_fn(img2 ** 2) - mu2_sq
    sigma12 = filter_fn(img1 * img2) - mu1_mu2

    # Compute SSIM components
    # Luminance comparison: l = (2*mu1*mu2 + C1) / (mu1^2 + mu2^2 + C1)
    luminance = (2.0 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)

    # Contrast-structure comparison: cs = (2*sigma12 + C2) / (sigma1^2 + sigma2^2 + C2)
    contrast_structure = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)

    return luminance, contrast_structure


def _downsample_2x(img: np.ndarray) -> np.ndarray:
    """Downsample image by factor of 2 using average pooling.

    Args:
        img: Input image of shape (H, W).

    Returns:
        Downsampled image of shape (H//2, W//2).
    """
    # Simple 2x2 average pooling
    h, w = img.shape
    h_new = h // 2
    w_new = w // 2

    # Crop to even dimensions if needed
    img_crop = img[:h_new*2, :w_new*2]

    # Reshape and average
    downsampled = img_crop.reshape(h_new, 2, w_new, 2).mean(axis=(1, 3))

    return downsampled


def compute_ssim_numpy(
    img1: np.ndarray,
    img2: np.ndarray,
    data_range: float = 1.0,
    K1: float = 0.01,
    K2: float = 0.03,
    sigma: float = 1.5,
    window_size: int = 11,
    use_separable: bool = True
) -> float:
    """Compute single-scale SSIM between two images.

    Args:
        img1: First image (H, W) single channel.
        img2: Second image (H, W) single channel.
        data_range: Dynamic range of input images (e.g., 1.0 for [0,1], 255 for [0,255]).
        K1: SSIM constant for luminance (default 0.01).
        K2: SSIM constant for contrast-structure (default 0.03).
        sigma: Gaussian kernel standard deviation (default 1.5).
        window_size: Gaussian window size (default 11).
        use_separable: Use separable Gaussian filters for efficiency (default True).

    Returns:
        SSIM score as a float between 0 and 1.
    """
    assert img1.shape == img2.shape, "Images must have the same shape"
    assert img1.ndim == 2, "Images must be 2D (single channel)"

    luminance, contrast_structure = _compute_ssim_components(
        img1, img2, window_size, sigma, K1, K2, data_range, use_separable
    )

    ssim_map = luminance * contrast_structure
    return float(np.mean(ssim_map))


def compute_msssim_numpy(
    img1: np.ndarray,
    img2: np.ndarray,
    data_range: float = 1.0,
    K1: float = 0.01,
    K2: float = 0.03,
    sigma: float = 1.5,
    window_size: int = 11,
    use_separable: bool = True,
    weights: Optional[np.ndarray] = None
) -> float:
    """Compute Multi-Scale SSIM (MS-SSIM) between two images.

    This implementation matches pytorch_msssim behavior and uses the standard
    5-scale approach with geometric mean combination of scales.

    Args:
        img1: First image (H, W) single channel.
        img2: Second image (H, W) single channel.
        data_range: Dynamic range of input images (default 1.0).
        K1: SSIM constant for luminance (default 0.01).
        K2: SSIM constant for contrast-structure (default 0.03).
        sigma: Gaussian kernel standard deviation (default 1.5).
        window_size: Gaussian window size (default 11).
        use_separable: Use separable Gaussian filters (default True).
        weights: Custom scale weights, shape (num_scales,). If None, uses standard
                 5-scale weights [0.0448, 0.2856, 0.3001, 0.2363, 0.1333].

    Returns:
        MS-SSIM score as a float between 0 and 1.

    Raises:
        ValueError: If images are too small for multi-scale processing.

    Notes:
        - Images must be large enough to support 4 downsampling operations (2x each).
        - Minimum recommended size is 176x176 pixels for 5 scales.
        - The algorithm computes SSIM at 5 scales, combining luminance from the
          finest scale with contrast-structure from all scales.
    """
    assert img1.shape == img2.shape, "Images must have the same shape"
    assert img1.ndim == 2, "Images must be 2D (single channel)"

    if weights is None:
        weights = DEFAULT_WEIGHTS
    else:
        weights = np.asarray(weights, dtype=np.float64)

    num_scales = len(weights)

    # Check minimum size for multi-scale processing
    min_size = 2 ** (num_scales - 1) * window_size
    h, w = img1.shape
    if h < min_size or w < min_size:
        raise ValueError(
            f"Images too small ({h}x{w}) for {num_scales} scales. "
            f"Minimum size is {min_size}x{min_size} pixels."
        )

    # Store contrast-structure values for each scale
    cs_values = []

    # Process each scale
    img1_current = img1.copy()
    img2_current = img2.copy()

    for scale_idx in range(num_scales - 1):
        # Compute SSIM components at current scale
        luminance, contrast_structure = _compute_ssim_components(
            img1_current, img2_current, window_size, sigma, K1, K2,
            data_range, use_separable
        )

        # Store mean contrast-structure for this scale
        cs_values.append(np.mean(contrast_structure))

        # Downsample for next scale
        img1_current = _downsample_2x(img1_current)
        img2_current = _downsample_2x(img2_current)

    # Compute full SSIM at final (coarsest) scale
    luminance_final, contrast_structure_final = _compute_ssim_components(
        img1_current, img2_current, window_size, sigma, K1, K2,
        data_range, use_separable
    )

    ssim_final = luminance_final * contrast_structure_final
    cs_values.append(np.mean(ssim_final))

    # Convert to array for vectorized operations
    cs_values = np.array(cs_values, dtype=np.float64)

    # Ensure non-negative values (can happen due to numerical issues)
    cs_values = np.maximum(cs_values, 0.0)

    # Compute MS-SSIM as weighted geometric mean: product(cs_i^w_i)
    # Equivalent to: exp(sum(w_i * log(cs_i)))
    # Using log space for numerical stability
    ms_ssim = np.prod(cs_values ** weights)

    return float(ms_ssim)


def compute_msssim_multichannel(
    img1: np.ndarray,
    img2: np.ndarray,
    data_range: float = 1.0,
    **kwargs
) -> float:
    """Compute MS-SSIM for multi-channel images (e.g., RGB).

    Computes MS-SSIM independently for each channel and returns the mean.

    Args:
        img1: First image (H, W, C) where C is number of channels.
        img2: Second image (H, W, C).
        data_range: Dynamic range of input images.
        **kwargs: Additional arguments passed to compute_msssim_numpy.

    Returns:
        Mean MS-SSIM score across all channels.
    """
    assert img1.shape == img2.shape, "Images must have the same shape"
    assert img1.ndim == 3, "Images must be 3D (H, W, C)"

    num_channels = img1.shape[2]
    msssim_values = []

    for c in range(num_channels):
        msssim_c = compute_msssim_numpy(
            img1[:, :, c], img2[:, :, c], data_range=data_range, **kwargs
        )
        msssim_values.append(msssim_c)

    return float(np.mean(msssim_values))


if __name__ == "__main__":
    # Quick smoke test
    print("Testing MS-SSIM NumPy implementation...")

    # Create test images
    np.random.seed(42)
    size = 256
    img1 = np.random.rand(size, size).astype(np.float64)
    img2 = img1 + 0.1 * np.random.randn(size, size).astype(np.float64)
    img2 = np.clip(img2, 0, 1)

    # Test SSIM
    ssim_val = compute_ssim_numpy(img1, img2, data_range=1.0)
    print(f"SSIM: {ssim_val:.6f}")

    # Test MS-SSIM
    msssim_val = compute_msssim_numpy(img1, img2, data_range=1.0)
    print(f"MS-SSIM: {msssim_val:.6f}")

    # Test identical images
    msssim_perfect = compute_msssim_numpy(img1, img1, data_range=1.0)
    print(f"MS-SSIM (identical images): {msssim_perfect:.6f}")

    # Test separable vs non-separable
    msssim_sep = compute_msssim_numpy(img1, img2, data_range=1.0, use_separable=True)
    msssim_nonsep = compute_msssim_numpy(img1, img2, data_range=1.0, use_separable=False)
    print(f"Separable vs Non-separable difference: {abs(msssim_sep - msssim_nonsep):.8f}")

    print("Smoke test complete!")
