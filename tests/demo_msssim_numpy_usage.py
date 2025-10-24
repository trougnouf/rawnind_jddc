"""Demonstration of NumPy MS-SSIM usage in various contexts.

Shows how to use the pure NumPy MS-SSIM implementation in:
1. Synchronous code
2. Async workflows with trio
3. Comparison with PyTorch version
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "DocScan"))

from common.libs.msssim_numpy import (
    compute_msssim_numpy,
    compute_ssim_numpy,
    compute_msssim_multichannel
)


def demo_basic_usage():
    """Basic synchronous usage."""
    print("="*70)
    print("DEMO 1: Basic Synchronous Usage")
    print("="*70)

    # Create test images
    np.random.seed(42)
    img1 = np.random.rand(256, 256)
    img2 = img1 + 0.1 * np.random.randn(256, 256)
    img2 = np.clip(img2, 0, 1)

    # Compute SSIM
    ssim = compute_ssim_numpy(img1, img2, data_range=1.0)
    print(f"SSIM score: {ssim:.6f}")

    # Compute MS-SSIM
    msssim = compute_msssim_numpy(img1, img2, data_range=1.0)
    print(f"MS-SSIM score: {msssim:.6f}")

    print()


def demo_multichannel():
    """Multi-channel (RGB) image usage."""
    print("="*70)
    print("DEMO 2: Multi-channel Images (RGB)")
    print("="*70)

    # Create RGB test images
    np.random.seed(42)
    img1_rgb = np.random.rand(256, 256, 3)
    img2_rgb = img1_rgb + 0.05 * np.random.randn(256, 256, 3)
    img2_rgb = np.clip(img2_rgb, 0, 1)

    # Compute MS-SSIM for RGB
    msssim_rgb = compute_msssim_multichannel(img1_rgb, img2_rgb, data_range=1.0)
    print(f"MS-SSIM (RGB): {msssim_rgb:.6f}")

    # Compare to per-channel computation
    msssim_r = compute_msssim_numpy(img1_rgb[:, :, 0], img2_rgb[:, :, 0], data_range=1.0)
    msssim_g = compute_msssim_numpy(img1_rgb[:, :, 1], img2_rgb[:, :, 1], data_range=1.0)
    msssim_b = compute_msssim_numpy(img1_rgb[:, :, 2], img2_rgb[:, :, 2], data_range=1.0)

    print(f"MS-SSIM per channel: R={msssim_r:.6f}, G={msssim_g:.6f}, B={msssim_b:.6f}")
    print(f"Mean: {(msssim_r + msssim_g + msssim_b) / 3:.6f}")

    print()


def demo_custom_parameters():
    """Custom parameters and scale weights."""
    print("="*70)
    print("DEMO 3: Custom Parameters")
    print("="*70)

    np.random.seed(42)
    img1 = np.random.rand(256, 256)
    img2 = img1 + 0.1 * np.random.randn(256, 256)
    img2 = np.clip(img2, 0, 1)

    # Default 5-scale MS-SSIM
    msssim_default = compute_msssim_numpy(img1, img2, data_range=1.0)
    print(f"MS-SSIM (5 scales, default weights): {msssim_default:.6f}")

    # Custom 3-scale MS-SSIM
    weights_3scale = np.array([0.2, 0.3, 0.5])
    msssim_3scale = compute_msssim_numpy(
        img1, img2, data_range=1.0, weights=weights_3scale
    )
    print(f"MS-SSIM (3 scales, custom weights): {msssim_3scale:.6f}")

    # Different Gaussian parameters
    msssim_custom_gauss = compute_msssim_numpy(
        img1, img2, data_range=1.0, sigma=2.0, window_size=15
    )
    print(f"MS-SSIM (custom Gaussian σ=2.0, window=15): {msssim_custom_gauss:.6f}")

    print()


async def demo_async_usage():
    """Async usage with trio."""
    print("="*70)
    print("DEMO 4: Async Usage with Trio")
    print("="*70)

    try:
        import trio

        np.random.seed(42)
        img1 = np.random.rand(256, 256)
        img2 = img1 + 0.1 * np.random.randn(256, 256)
        img2 = np.clip(img2, 0, 1)

        # Run MS-SSIM in thread pool
        msssim = await trio.to_thread.run_sync(
            compute_msssim_numpy, img1, img2, 1.0
        )

        print(f"MS-SSIM (computed in thread pool): {msssim:.6f}")
        print("✓ Async execution successful")

    except ImportError:
        print("Trio not available, skipping async demo")

    print()


def demo_image_quality_assessment():
    """Practical example: image quality assessment."""
    print("="*70)
    print("DEMO 5: Image Quality Assessment")
    print("="*70)

    np.random.seed(42)
    original = np.random.rand(512, 512)

    # Simulate different degradations
    degradations = {
        "Gaussian blur (σ=1.0)": lambda img: ndimage_gaussian_filter(img, 1.0),
        "Gaussian blur (σ=2.0)": lambda img: ndimage_gaussian_filter(img, 2.0),
        "Salt & pepper noise (5%)": lambda img: add_salt_pepper(img, 0.05),
        "Gaussian noise (σ=0.1)": lambda img: np.clip(img + 0.1 * np.random.randn(*img.shape), 0, 1),
    }

    from scipy.ndimage import gaussian_filter as ndimage_gaussian_filter

    def add_salt_pepper(img, prob):
        result = img.copy()
        mask = np.random.rand(*img.shape) < prob
        result[mask] = np.random.choice([0, 1], size=mask.sum())
        return result

    print(f"{'Degradation':<30} {'SSIM':>10} {'MS-SSIM':>10}")
    print("-" * 52)

    for name, degradation_fn in degradations.items():
        degraded = degradation_fn(original)

        ssim = compute_ssim_numpy(original, degraded, data_range=1.0)
        msssim = compute_msssim_numpy(original, degraded, data_range=1.0)

        print(f"{name:<30} {ssim:>10.6f} {msssim:>10.6f}")

    print()


def demo_pytorch_equivalence():
    """Show equivalence with PyTorch implementation."""
    print("="*70)
    print("DEMO 6: PyTorch Equivalence")
    print("="*70)

    try:
        import torch
        from pytorch_msssim import ms_ssim as torch_ms_ssim

        np.random.seed(42)
        torch.manual_seed(42)

        img1_np = np.random.rand(256, 256)
        img2_np = img1_np + 0.1 * np.random.randn(256, 256)
        img2_np = np.clip(img2_np, 0, 1)

        # NumPy version
        msssim_numpy = compute_msssim_numpy(img1_np, img2_np, data_range=1.0)

        # PyTorch version
        img1_torch = torch.from_numpy(img1_np).unsqueeze(0).unsqueeze(0).float()
        img2_torch = torch.from_numpy(img2_np).unsqueeze(0).unsqueeze(0).float()
        msssim_torch = torch_ms_ssim(img1_torch, img2_torch, data_range=1.0).item()

        print(f"NumPy MS-SSIM:   {msssim_numpy:.8f}")
        print(f"PyTorch MS-SSIM: {msssim_torch:.8f}")
        print(f"Difference:      {abs(msssim_numpy - msssim_torch):.8f}")

        if abs(msssim_numpy - msssim_torch) < 0.01:
            print("✓ Within ±0.01 tolerance")
        else:
            print("✗ Outside tolerance")

    except ImportError:
        print("PyTorch or pytorch_msssim not available, skipping comparison")

    print()


if __name__ == "__main__":
    print("\nNumPy MS-SSIM Usage Demonstrations")
    print("="*70)
    print()

    demo_basic_usage()
    demo_multichannel()
    demo_custom_parameters()
    demo_image_quality_assessment()
    demo_pytorch_equivalence()

    # Run async demo if trio is available
    try:
        import trio
        trio.run(demo_async_usage)
    except ImportError:
        print("Skipping async demo (trio not available)")

    print("="*70)
    print("All demonstrations complete!")
    print("="*70)
