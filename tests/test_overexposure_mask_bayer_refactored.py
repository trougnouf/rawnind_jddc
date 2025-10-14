"""Production-quality tests for make_overexposure_mask_bayer.

Tests validate overexposure masking for both mosaiced RAW (2D) and channel-stacked (3D)
image formats used in the RAW image processing pipeline.
"""

import numpy as np
import pytest

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from rawnind.libs.rawproc import make_overexposure_mask_bayer


class TestOverexposureMaskBayer2D:
    """Tests for 2D mosaiced RAW images (Bayer, X-Trans patterns)."""

    def test_all_pixels_below_threshold(self) -> None:
        """When all pixels are below threshold, all should be valid (True)."""
        img = np.full((10, 10), 0.3, dtype=np.float32)
        threshold = 0.5

        mask = make_overexposure_mask_bayer(img, threshold)

        assert mask.shape == (10, 10)
        assert mask.dtype == np.bool_
        assert np.all(mask), "All pixels below threshold should be valid"

    def test_all_pixels_above_threshold(self) -> None:
        """When all pixels are above threshold, all should be invalid (False)."""
        img = np.full((10, 10), 0.8, dtype=np.float32)
        threshold = 0.5

        mask = make_overexposure_mask_bayer(img, threshold)

        assert mask.shape == (10, 10)
        assert mask.dtype == np.bool_
        assert np.all(~mask), "All pixels above threshold should be invalid"

    def test_exact_threshold_boundary(self) -> None:
        """Pixels exactly at threshold should be invalid (< not <=)."""
        img = np.array([[0.4, 0.5, 0.6]], dtype=np.float32)
        threshold = 0.5

        mask = make_overexposure_mask_bayer(img, threshold)

        assert mask[0, 0], "Below threshold should be valid"
        assert not mask[0, 1], "At threshold should be invalid"
        assert not mask[0, 2], "Above threshold should be invalid"

    def test_deterministic_pattern(self) -> None:
        """Test with known deterministic pattern."""
        # Create checkerboard: 0.3 and 0.7 alternating
        img = np.zeros((4, 4), dtype=np.float32)
        img[::2, ::2] = 0.3  # Top-left corners of 2x2 blocks
        img[::2, 1::2] = 0.7
        img[1::2, ::2] = 0.7
        img[1::2, 1::2] = 0.3
        threshold = 0.5

        mask = make_overexposure_mask_bayer(img, threshold)

        # Exactly 8 pixels should be below 0.5
        assert mask.sum() == 8, "Expected 8 valid pixels in checkerboard"
        assert mask[0, 0]
        assert not mask[0, 1]

    def test_single_pixel_image(self) -> None:
        """Single pixel images should work correctly."""
        img_below = np.array([[0.3]], dtype=np.float32)
        img_above = np.array([[0.8]], dtype=np.float32)
        threshold = 0.5

        mask_below = make_overexposure_mask_bayer(img_below, threshold)
        mask_above = make_overexposure_mask_bayer(img_above, threshold)

        assert mask_below.shape == (1, 1)
        assert mask_below[0, 0]
        assert not mask_above[0, 0]

    def test_bayer_dimensions(self) -> None:
        """Test with typical Bayer pattern dimensions (even multiples)."""
        for h, w in [(6, 8), (100, 200), (512, 512)]:
            img = np.random.RandomState(42).uniform(0, 1, (h, w)).astype(np.float32)
            threshold = 0.5

            mask = make_overexposure_mask_bayer(img, threshold)

            assert mask.shape == (h, w)
            assert mask.dtype == np.bool_

    def test_xtrans_dimensions(self) -> None:
        """Test with X-Trans pattern dimensions (multiples of 6)."""
        for size in [6, 12, 18, 24]:
            img = (
                np.random.RandomState(42).uniform(0, 1, (size, size)).astype(np.float32)
            )
            threshold = 0.5

            mask = make_overexposure_mask_bayer(img, threshold)

            assert mask.shape == (size, size)
            assert mask.dtype == np.bool_


class TestOverexposureMaskBayer3D:
    """Tests for 3D channel-stacked images (RGGB, X-Trans channels)."""

    def test_all_channels_below_threshold(self) -> None:
        """When all channels are below threshold, pixel should be valid."""
        img = np.full((4, 10, 10), 0.3, dtype=np.float32)
        threshold = 0.5

        mask = make_overexposure_mask_bayer(img, threshold)

        assert mask.shape == (10, 10)
        assert mask.dtype == np.bool_
        assert np.all(mask), "All channels below threshold → valid pixel"

    def test_any_channel_above_threshold_invalidates_pixel(self) -> None:
        """If ANY channel exceeds threshold, the pixel should be invalid."""
        # All channels at 0.3 except one pixel has one channel at 0.9
        img = np.full((4, 5, 5), 0.3, dtype=np.float32)
        img[0, 2, 2] = 0.9  # Only channel 0 of pixel (2,2) is overexposed
        threshold = 0.5

        mask = make_overexposure_mask_bayer(img, threshold)

        assert not mask[2, 2], "Any channel above threshold should invalidate pixel"
        assert mask[0, 0], "Other pixels should remain valid"
        assert mask.sum() == 24, "Only (2,2) should be invalid"

    def test_multiple_channels_above_threshold(self) -> None:
        """Test pixel with multiple channels above threshold."""
        img = np.full((4, 3, 3), 0.3, dtype=np.float32)
        img[0, 1, 1] = 0.9  # R channel overexposed
        img[1, 1, 1] = 0.8  # G1 channel overexposed
        threshold = 0.5

        mask = make_overexposure_mask_bayer(img, threshold)

        assert not mask[1, 1], "Multiple channels above → invalid"
        assert mask.sum() == 8, "Only center pixel should be invalid"

    def test_threshold_boundary_any_channel(self) -> None:
        """Test exact threshold with one channel at boundary."""
        img = np.full((4, 2, 2), 0.4, dtype=np.float32)
        img[0, 0, 0] = 0.5  # Exactly at threshold
        threshold = 0.5

        mask = make_overexposure_mask_bayer(img, threshold)

        assert not mask[0, 0], "Channel at threshold → invalid"
        assert mask.sum() == 3

    def test_bayer_rggb_shape(self) -> None:
        """Test typical Bayer RGGB stacked format (4, H/2, W/2)."""
        h, w = 18, 24
        img = (
            np.random.RandomState(42)
            .uniform(0, 1, (4, h // 2, w // 2))
            .astype(np.float32)
        )
        threshold = 0.6

        mask = make_overexposure_mask_bayer(img, threshold)

        assert mask.shape == (h // 2, w // 2)
        assert mask.dtype == np.bool_

    def test_xtrans_9channel_shape(self) -> None:
        """Test X-Trans 3x3 color grouping (9, H/3, W/3)."""
        h, w = 18, 24
        img = (
            np.random.RandomState(42)
            .uniform(0, 1, (9, h // 3, w // 3))
            .astype(np.float32)
        )
        threshold = 0.6

        mask = make_overexposure_mask_bayer(img, threshold)

        assert mask.shape == (h // 3, w // 3)
        assert mask.dtype == np.bool_

    def test_equivalence_uniform_channels(self) -> None:
        """2D mosaiced and 3D uniform channels should give equivalent results."""
        # Create 2D image
        img_2d = np.random.RandomState(42).uniform(0, 1, (10, 10)).astype(np.float32)

        # Create 3D image with all channels identical to 2D
        img_3d = np.stack([img_2d, img_2d, img_2d, img_2d], axis=0)

        threshold = 0.5
        mask_2d = make_overexposure_mask_bayer(img_2d, threshold)
        mask_3d = make_overexposure_mask_bayer(img_3d, threshold)

        assert np.array_equal(mask_2d, mask_3d), "2D and uniform 3D should match"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestTorchTensorSupport:
    """Tests for torch.Tensor input compatibility."""

    def test_2d_torch_tensor_returns_numpy(self) -> None:
        """2D torch tensor should be accepted and return numpy array."""
        img = torch.rand((10, 14), dtype=torch.float32)
        threshold = 0.5

        mask = make_overexposure_mask_bayer(img, threshold)

        assert isinstance(mask, np.ndarray), "Should return numpy even for torch input"
        assert mask.dtype == np.bool_
        assert mask.shape == (10, 14)

    def test_3d_torch_tensor_returns_numpy(self) -> None:
        """3D torch tensor should be accepted and return numpy array."""
        img = torch.full((4, 5, 7), 0.4, dtype=torch.float32)
        img[0, 0, 0] = 0.99
        threshold = 0.6

        mask = make_overexposure_mask_bayer(img, threshold)

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.bool_
        assert mask.shape == (5, 7)
        assert not mask[0, 0], "Overexposed pixel should be invalid"
        assert mask[1, 1]

    def test_torch_numpy_equivalence(self) -> None:
        """Torch and numpy inputs should produce identical results."""
        np.random.seed(42)
        img_np = np.random.uniform(0, 1, (4, 10, 10)).astype(np.float32)
        img_torch = torch.from_numpy(img_np)
        threshold = 0.5

        mask_np = make_overexposure_mask_bayer(img_np, threshold)
        mask_torch = make_overexposure_mask_bayer(img_torch, threshold)

        assert np.array_equal(mask_np, mask_torch), (
            "Torch and numpy should be equivalent"
        )

    def test_gpu_tensor_support(self) -> None:
        """GPU tensors should be handled correctly."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        img = torch.rand((10, 10), dtype=torch.float32, device="cuda")
        threshold = 0.5

        mask = make_overexposure_mask_bayer(img, threshold)

        assert isinstance(mask, np.ndarray), "Should convert GPU tensor to numpy"
        assert mask.shape == (10, 10)


class TestInvalidInputs:
    """Tests for error handling and edge cases."""

    def test_rejects_1d_array(self) -> None:
        """1D arrays should be rejected."""
        img = np.array([0.5, 0.6, 0.7], dtype=np.float32)

        with pytest.raises(ValueError, match="Expected 2D or 3D input"):
            make_overexposure_mask_bayer(img, 0.5)

    def test_rejects_4d_array(self) -> None:
        """4D arrays should be rejected."""
        img = np.random.rand(2, 4, 10, 10).astype(np.float32)

        with pytest.raises(ValueError, match="Expected 2D or 3D input"):
            make_overexposure_mask_bayer(img, 0.5)

    def test_rejects_invalid_type(self) -> None:
        """Non-array types should be rejected with clear message."""
        with pytest.raises(TypeError, match="Expected numpy array or torch tensor"):
            make_overexposure_mask_bayer([0.5, 0.6], 0.5)  # type: ignore

    def test_rejects_invalid_threshold(self) -> None:
        """Threshold must be in valid range [0, inf)."""
        img = np.random.rand(10, 10).astype(np.float32)

        with pytest.raises(ValueError, match="threshold must be non-negative"):
            make_overexposure_mask_bayer(img, -0.1)

    def test_handles_nan_values(self) -> None:
        """NaN values should propagate correctly (NaN < threshold is False)."""
        img = np.array([[0.3, np.nan, 0.7]], dtype=np.float32)
        threshold = 0.5

        mask = make_overexposure_mask_bayer(img, threshold)

        assert mask[0, 0], "Normal value below threshold"
        assert not mask[0, 1], "NaN should be invalid (NaN < x is False)"
        assert not mask[0, 2], "Value above threshold"

    def test_handles_inf_values(self) -> None:
        """Infinity values should be handled correctly."""
        img = np.array([[0.3, np.inf, -np.inf]], dtype=np.float32)
        threshold = 0.5

        mask = make_overexposure_mask_bayer(img, threshold)

        assert mask[0, 0]
        assert not mask[0, 1], "+inf should be invalid"
        assert mask[0, 2], "-inf should be valid (< threshold)"

    def test_empty_array(self) -> None:
        """Empty arrays should be handled gracefully."""
        img = np.array([], dtype=np.float32).reshape(0, 0)
        threshold = 0.5

        mask = make_overexposure_mask_bayer(img, threshold)

        assert mask.shape == (0, 0)
        assert mask.dtype == np.bool_


class TestThresholdParameterBehavior:
    """Tests for different threshold values."""

    def test_threshold_zero_accepts_all_negative(self) -> None:
        """threshold=0 should only accept negative values."""
        img = np.array([[-0.5, 0.0, 0.5]], dtype=np.float32)

        mask = make_overexposure_mask_bayer(img, 0.0)

        assert mask[0, 0], "Negative < 0"
        assert not mask[0, 1], "0 is not < 0"
        assert not mask[0, 2], "Positive is not < 0"

    def test_threshold_one_typical_for_raw(self) -> None:
        """threshold=1.0 is typical for normalized RAW images."""
        img = np.array([[0.5, 0.999, 1.0, 1.5]], dtype=np.float32)

        mask = make_overexposure_mask_bayer(img, 1.0)

        assert mask[0, 0]
        assert mask[0, 1]
        assert not mask[0, 2], "Exactly 1.0 should be invalid"
        assert not mask[0, 3]

    def test_very_high_threshold_accepts_most(self) -> None:
        """Very high thresholds should accept normal image values."""
        img = np.random.rand(10, 10).astype(np.float32)

        mask = make_overexposure_mask_bayer(img, 100.0)

        assert np.all(mask), "All normal values should be below 100.0"
