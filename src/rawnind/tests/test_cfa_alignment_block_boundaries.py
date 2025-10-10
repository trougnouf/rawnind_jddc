"""Test that CFA alignment respects block boundaries."""

import pytest
import numpy as np
from rawnind.libs.raw import fft_phase_correlate_cfa


def test_bayer_alignment_snaps_to_even():
    """bayer alignment must return even-numbered shifts (2x2 block boundaries)."""
    # Create synthetic bayer images - use structured pattern not random noise
    h, w = 256, 256
    anchor = np.random.rand(1, h, w).astype(np.float32)

    # Shift by various amounts - FFT should detect and snap to even
    for shift_amount in [3, 5, 7]:
        target = np.zeros((1, h, w), dtype=np.float32)
        target[:, shift_amount:, shift_amount:] = anchor[:, :-shift_amount, :-shift_amount]

        # bayer RGGB pattern
        bayer_pattern = np.array([[0, 1], [1, 2]], dtype=np.uint8)
        metadata = {"RGBG_pattern": bayer_pattern}

        shift, _ = fft_phase_correlate_cfa(anchor, target, metadata, method="median")

        # Both components must be even (key constraint)
        assert shift[0] % 2 == 0, f"Y shift {shift[0]} is not even (violates 2x2 bayer blocks)"
        assert shift[1] % 2 == 0, f"X shift {shift[1]} is not even (violates 2x2 bayer blocks)"


def test_xtrans_alignment_snaps_to_multiple_of_3():
    """x-trans alignment must return shifts that are multiples of 3 (3x3 block boundaries)."""
    # Create synthetic X-Trans images
    h, w = 252, 252  # Multiple of 6 for X-Trans
    anchor = np.random.rand(1, h, w).astype(np.float32)

    # Shift by various amounts - FFT should detect and snap to multiples of 3
    for shift_amount in [4, 5, 7, 8]:
        target = np.zeros((1, h, w), dtype=np.float32)
        target[:, shift_amount:, shift_amount:] = anchor[:, :-shift_amount, :-shift_amount]

        # X-Trans pattern (6x6) - simplified pattern
        xtrans_pattern = np.array([
            [1, 1, 0, 1, 1, 2],
            [1, 2, 1, 1, 0, 1],
            [0, 1, 1, 2, 1, 1],
            [2, 1, 1, 1, 1, 0],
            [1, 0, 1, 1, 2, 1],
            [1, 1, 2, 1, 1, 1],
        ], dtype=np.uint8)
        metadata = {"RGBG_pattern": xtrans_pattern}

        shift, _ = fft_phase_correlate_cfa(anchor, target, metadata, method="median")

        # Both components must be multiples of 3 (key constraint)
        assert shift[0] % 3 == 0, f"Y shift {shift[0]} is not multiple of 3 (violates 3x3 x-trans blocks)"
        assert shift[1] % 3 == 0, f"X shift {shift[1]} is not multiple of 3 (violates 3x3 x-trans blocks)"


def test_bayer_alignment_preserves_block_aligned_shifts():
    """If shift is already block-aligned, it should preserve block alignment."""
    h, w = 256, 256
    anchor = np.random.rand(1, h, w).astype(np.float32)

    # Shift by exactly 4 pixels (already even)
    target = np.zeros((1, h, w), dtype=np.float32)
    target[:, 4:, 4:] = anchor[:, :-4, :-4]

    bayer_pattern = np.array([[0, 1], [1, 2]], dtype=np.uint8)
    metadata = {"RGBG_pattern": bayer_pattern}

    shift, _ = fft_phase_correlate_cfa(anchor, target, metadata, method="median")

    # Should maintain even alignment (magnitude should be 4)
    assert shift[0] % 2 == 0, f"Y shift {shift[0]} lost even alignment"
    assert shift[1] % 2 == 0, f"X shift {shift[1]} lost even alignment"
    assert abs(shift[0]) == 4, f"Y shift magnitude changed: {abs(shift[0])} != 4"
    assert abs(shift[1]) == 4, f"X shift magnitude changed: {abs(shift[1])} != 4"


def test_xtrans_alignment_preserves_block_aligned_shifts():
    """If shift is already block-aligned, it should preserve block alignment."""
    h, w = 252, 252
    anchor = np.random.rand(1, h, w).astype(np.float32)

    # Shift by exactly 6 pixels (multiple of 3)
    target = np.zeros((1, h, w), dtype=np.float32)
    target[:, 6:, 6:] = anchor[:, :-6, :-6]

    xtrans_pattern = np.array([
        [1, 1, 0, 1, 1, 2],
        [1, 2, 1, 1, 0, 1],
        [0, 1, 1, 2, 1, 1],
        [2, 1, 1, 1, 1, 0],
        [1, 0, 1, 1, 2, 1],
        [1, 1, 2, 1, 1, 1],
    ], dtype=np.uint8)
    metadata = {"RGBG_pattern": xtrans_pattern}

    shift, _ = fft_phase_correlate_cfa(anchor, target, metadata, method="median")

    # Should maintain multiple-of-3 alignment (magnitude should be 6)
    assert shift[0] % 3 == 0, f"Y shift {shift[0]} lost multiple-of-3 alignment"
    assert shift[1] % 3 == 0, f"X shift {shift[1]} lost multiple-of-3 alignment"
    assert abs(shift[0]) == 6, f"Y shift magnitude changed: {abs(shift[0])} != 6"
    assert abs(shift[1]) == 6, f"X shift magnitude changed: {abs(shift[1])} != 6"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
