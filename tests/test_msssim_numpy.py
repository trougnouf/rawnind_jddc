"""Comprehensive tests for NumPy MS-SSIM implementation.

Tests accuracy against known values, edge cases, and performance benchmarks.
"""

import numpy as np
import pytest
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from common.libs.msssim_numpy import (
    compute_msssim_numpy,
    compute_ssim_numpy,
    _create_gaussian_kernel_1d,
)


class TestGaussianKernel:
    """Tests for Gaussian kernel generation."""

    def test_kernel_normalization(self):
        """Gaussian kernel should sum to 1."""
        kernel = _create_gaussian_kernel_1d(11, 1.5)
        assert np.isclose(kernel.sum(), 1.0, atol=1e-10)

    def test_kernel_symmetry(self):
        """Gaussian kernel should be symmetric."""
        kernel = _create_gaussian_kernel_1d(11, 1.5)
        assert np.allclose(kernel, kernel[::-1])

    def test_kernel_caching(self):
        """Kernel creation should be cached."""
        k1 = _create_gaussian_kernel_1d(11, 1.5)
        k2 = _create_gaussian_kernel_1d(11, 1.5)
        # Should be same object due to caching
        assert k1 is k2


class TestSSIM:
    """Tests for single-scale SSIM."""

    def test_identical_images(self):
        """SSIM of identical images should be 1.0."""
        img = np.random.rand(128, 128)
        ssim = compute_ssim_numpy(img, img, data_range=1.0)
        assert np.isclose(ssim, 1.0, atol=1e-6)

    def test_constant_images(self):
        """SSIM of constant images should be 1.0."""
        img1 = np.ones((128, 128)) * 0.5
        img2 = np.ones((128, 128)) * 0.5
        ssim = compute_ssim_numpy(img1, img2, data_range=1.0)
        assert np.isclose(ssim, 1.0, atol=1e-6)

    def test_different_images(self):
        """SSIM of different images should be < 1.0."""
        np.random.seed(42)
        img1 = np.random.rand(128, 128)
        img2 = np.random.rand(128, 128)
        ssim = compute_ssim_numpy(img1, img2, data_range=1.0)
        assert 0 < ssim < 1.0

    def test_data_range_255(self):
        """SSIM should work with 8-bit image range."""
        np.random.seed(42)
        img1 = (np.random.rand(128, 128) * 255).astype(np.float64)
        img2 = img1 + 10 * np.random.randn(128, 128)
        img2 = np.clip(img2, 0, 255)
        ssim = compute_ssim_numpy(img1, img2, data_range=255.0)
        assert 0 < ssim < 1.0

    def test_separable_vs_nonseparable(self):
        """Separable filter should match non-separable within tolerance."""
        np.random.seed(42)
        img1 = np.random.rand(128, 128)
        img2 = img1 + 0.1 * np.random.randn(128, 128)
        img2 = np.clip(img2, 0, 1)

        ssim_sep = compute_ssim_numpy(img1, img2, data_range=1.0, use_separable=True)
        ssim_nonsep = compute_ssim_numpy(img1, img2, data_range=1.0, use_separable=False)

        # Should be very close (within numerical tolerance)
        assert np.isclose(ssim_sep, ssim_nonsep, atol=1e-5)


class TestMSSSIM:
    """Tests for multi-scale SSIM."""

    def test_identical_images(self):
        """MS-SSIM of identical images should be 1.0."""
        np.random.seed(42)
        img = np.random.rand(256, 256)
        msssim = compute_msssim_numpy(img, img, data_range=1.0)
        assert np.isclose(msssim, 1.0, atol=1e-6)

    def test_constant_images(self):
        """MS-SSIM of constant images should be 1.0."""
        img1 = np.ones((256, 256)) * 0.5
        img2 = np.ones((256, 256)) * 0.5
        msssim = compute_msssim_numpy(img1, img2, data_range=1.0)
        assert np.isclose(msssim, 1.0, atol=1e-6)

    def test_noisy_image(self):
        """MS-SSIM should degrade gracefully with noise."""
        np.random.seed(42)
        img1 = np.random.rand(256, 256)

        noise_levels = [0.01, 0.05, 0.1, 0.2]
        msssim_values = []

        for noise in noise_levels:
            img2 = img1 + noise * np.random.randn(256, 256)
            img2 = np.clip(img2, 0, 1)
            msssim = compute_msssim_numpy(img1, img2, data_range=1.0)
            msssim_values.append(msssim)

        # MS-SSIM should decrease with more noise
        for i in range(len(msssim_values) - 1):
            assert msssim_values[i] > msssim_values[i + 1], \
                f"MS-SSIM should decrease with noise: {msssim_values}"

    def test_custom_weights(self):
        """MS-SSIM should accept custom scale weights."""
        np.random.seed(42)
        img1 = np.random.rand(256, 256)
        img2 = img1 + 0.1 * np.random.randn(256, 256)
        img2 = np.clip(img2, 0, 1)

        # Test with 3 scales instead of 5
        weights_3scale = np.array([0.1, 0.3, 0.6])
        msssim = compute_msssim_numpy(img1, img2, data_range=1.0, weights=weights_3scale)
        assert 0 < msssim < 1.0

    def test_minimum_size_requirement(self):
        """MS-SSIM should raise error for images too small."""
        img1 = np.random.rand(64, 64)  # Too small for 5 scales
        img2 = np.random.rand(64, 64)

        with pytest.raises(ValueError, match="Images too small"):
            compute_msssim_numpy(img1, img2, data_range=1.0)

    def test_data_range_255(self):
        """MS-SSIM should work with 8-bit image range."""
        np.random.seed(42)
        img1 = (np.random.rand(256, 256) * 255).astype(np.float64)
        img2 = img1 + 10 * np.random.randn(256, 256)
        img2 = np.clip(img2, 0, 255)
        msssim = compute_msssim_numpy(img1, img2, data_range=255.0)
        assert 0 < msssim < 1.0

    def test_high_contrast_image(self):
        """MS-SSIM should handle high contrast images."""
        # Create checkerboard pattern
        img1 = np.zeros((256, 256))
        img1[::2, ::2] = 1.0
        img1[1::2, 1::2] = 1.0

        # Slightly blurred version
        from scipy.ndimage import gaussian_filter
        img2 = gaussian_filter(img1, sigma=0.5)

        msssim = compute_msssim_numpy(img1, img2, data_range=1.0)
        assert 0.5 < msssim < 1.0  # Should be fairly similar

    def test_separable_vs_nonseparable(self):
        """Separable filter should match non-separable within tolerance."""
        np.random.seed(42)
        img1 = np.random.rand(256, 256)
        img2 = img1 + 0.1 * np.random.randn(256, 256)
        img2 = np.clip(img2, 0, 1)

        msssim_sep = compute_msssim_numpy(img1, img2, data_range=1.0, use_separable=True)
        msssim_nonsep = compute_msssim_numpy(img1, img2, data_range=1.0, use_separable=False)

        # Should be very close (within numerical tolerance)
        assert np.isclose(msssim_sep, msssim_nonsep, atol=1e-5), \
            f"Separable ({msssim_sep}) != Non-separable ({msssim_nonsep})"


class TestMultiChannel:
    """Tests for multi-channel images."""

    def test_rgb_image(self):
        """MS-SSIM should work with RGB images."""
        from common.libs.msssim_numpy import compute_msssim_multichannel

        np.random.seed(42)
        img1 = np.random.rand(256, 256, 3)
        img2 = img1 + 0.05 * np.random.randn(256, 256, 3)
        img2 = np.clip(img2, 0, 1)

        msssim = compute_msssim_multichannel(img1, img2, data_range=1.0)
        assert 0 < msssim < 1.0


class TestPerformance:
    """Performance benchmarks."""

    def test_ssim_performance(self):
        """Benchmark SSIM computation time."""
        np.random.seed(42)
        img1 = np.random.rand(512, 512)
        img2 = img1 + 0.1 * np.random.randn(512, 512)
        img2 = np.clip(img2, 0, 1)

        # Warm-up
        _ = compute_ssim_numpy(img1, img2, data_range=1.0)

        # Benchmark
        n_iterations = 10
        start = time.time()
        for _ in range(n_iterations):
            _ = compute_ssim_numpy(img1, img2, data_range=1.0)
        elapsed = (time.time() - start) / n_iterations

        print(f"\nSSIM (512x512): {elapsed*1000:.2f} ms per iteration")
        assert elapsed < 0.5  # Should be < 500ms on modern CPU

    def test_msssim_performance(self):
        """Benchmark MS-SSIM computation time."""
        np.random.seed(42)
        img1 = np.random.rand(512, 512)
        img2 = img1 + 0.1 * np.random.randn(512, 512)
        img2 = np.clip(img2, 0, 1)

        # Warm-up
        _ = compute_msssim_numpy(img1, img2, data_range=1.0)

        # Benchmark
        n_iterations = 10
        start = time.time()
        for _ in range(n_iterations):
            _ = compute_msssim_numpy(img1, img2, data_range=1.0)
        elapsed = (time.time() - start) / n_iterations

        print(f"\nMS-SSIM (512x512): {elapsed*1000:.2f} ms per iteration")
        assert elapsed < 1.0  # Should be < 1s on modern CPU

    def test_separable_speedup(self):
        """Separable filters should be faster."""
        np.random.seed(42)
        img1 = np.random.rand(256, 256)
        img2 = img1 + 0.1 * np.random.randn(256, 256)
        img2 = np.clip(img2, 0, 1)

        # Benchmark separable
        n_iterations = 10
        start = time.time()
        for _ in range(n_iterations):
            _ = compute_msssim_numpy(img1, img2, data_range=1.0, use_separable=True)
        time_sep = (time.time() - start) / n_iterations

        # Benchmark non-separable
        start = time.time()
        for _ in range(n_iterations):
            _ = compute_msssim_numpy(img1, img2, data_range=1.0, use_separable=False)
        time_nonsep = (time.time() - start) / n_iterations

        print(f"\nSeparable: {time_sep*1000:.2f} ms")
        print(f"Non-separable: {time_nonsep*1000:.2f} ms")
        print(f"Speedup: {time_nonsep/time_sep:.2f}x")

        # Separable should be at least as fast (often faster)
        assert time_sep <= time_nonsep * 1.2  # Allow 20% margin


class TestPyTorchCompatibility:
    """Tests for compatibility with PyTorch MS-SSIM."""

    @pytest.mark.skipif("pytorch_msssim" not in sys.modules, reason="pytorch_msssim not available")
    def test_pytorch_accuracy(self):
        """Compare accuracy against PyTorch MS-SSIM implementation."""
        import torch
        from pytorch_msssim import ms_ssim as torch_ms_ssim

        np.random.seed(42)
        img1_np = np.random.rand(256, 256).astype(np.float32)
        img2_np = img1_np + 0.1 * np.random.randn(256, 256).astype(np.float32)
        img2_np = np.clip(img2_np, 0, 1)

        # NumPy version
        msssim_numpy = compute_msssim_numpy(img1_np, img2_np, data_range=1.0)

        # PyTorch version
        img1_torch = torch.from_numpy(img1_np).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        img2_torch = torch.from_numpy(img2_np).unsqueeze(0).unsqueeze(0)
        msssim_torch = torch_ms_ssim(img1_torch, img2_torch, data_range=1.0).item()

        print(f"\nNumPy MS-SSIM: {msssim_numpy:.6f}")
        print(f"PyTorch MS-SSIM: {msssim_torch:.6f}")
        print(f"Difference: {abs(msssim_numpy - msssim_torch):.6f}")

        # Should match within ±0.01
        assert np.isclose(msssim_numpy, msssim_torch, atol=0.01), \
            f"NumPy ({msssim_numpy}) != PyTorch ({msssim_torch}), diff = {abs(msssim_numpy - msssim_torch)}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
