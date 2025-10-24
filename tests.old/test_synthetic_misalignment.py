#!/usr/bin/env python3
"""
Test channel-split FFT with synthetic misalignments.

Strategy: Take aligned pairs, create known misalignments via cropping,
verify that channel-split FFT detects the correct shift.
"""

import numpy as np
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "DocScan"))

from rawnind.libs import raw


def load_raw_image(fpath: str):
    """Load RAW image using the library function."""
    img, metadata = raw.raw_fpath_to_mono_img_and_metadata(fpath, return_float=True)
    return img, metadata


def match_gain(anchor, target):
    """Match gain between images."""
    anchor_mean = anchor.mean()
    target_mean = target.mean()
    gain = anchor_mean / (target_mean + 1e-8)
    return target * gain


def shift_images(anchor, target, shift):
    """Shift images to common area."""
    dy, dx = shift
    h, w = anchor.shape[-2:]

    y1_a = max(0, dy)
    y2_a = h + min(0, dy)
    x1_a = max(0, dx)
    x2_a = w + min(0, dx)

    y1_t = max(0, -dy)
    y2_t = h + min(0, -dy)
    x1_t = max(0, -dx)
    x2_t = w + min(0, -dx)

    anchor_out = anchor[..., y1_a:y2_a, x1_a:x2_a]
    target_out = target[..., y1_t:y2_t, x1_t:x2_t]

    return anchor_out, target_out


def extract_bayer_channel(img, metadata, channel):
    """Extract one bayer channel and downsample."""
    pattern = metadata["RGBG_pattern"]
    h, w = img.shape[-2:]

    # Map channel name to position in 2x2 pattern
    channel_map = {"R": (0, 0), "G1": (0, 1), "G2": (1, 0), "B": (1, 1)}

    row_offset, col_offset = channel_map[channel]

    # Extract positions for this channel
    values = img[0, row_offset::2, col_offset::2].copy()

    return values


def fft_phase_correlate_bayer(anchor, target, anchor_meta, method="median"):
    """bayer channel-split FFT."""
    channels = ["R", "G1", "G2", "B"]
    shifts = []

    for ch in channels:
        anchor_ch = extract_bayer_channel(anchor, anchor_meta, ch)
        target_ch = extract_bayer_channel(target, anchor_meta, ch)

        # Crop to common size (for synthetic misalignments with different sizes)
        min_h = min(anchor_ch.shape[0], target_ch.shape[0])
        min_w = min(anchor_ch.shape[1], target_ch.shape[1])
        anchor_ch = anchor_ch[:min_h, :min_w]
        target_ch = target_ch[:min_h, :min_w]

        # Standard FFT phase correlation
        f_anchor = np.fft.fft2(anchor_ch)
        f_target = np.fft.fft2(target_ch)

        cross_power = f_anchor * np.conj(f_target)
        cross_power /= np.abs(cross_power) + 1e-10

        correlation = np.fft.ifft2(cross_power).real
        correlation = np.fft.fftshift(correlation)

        peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)

        center_y, center_x = correlation.shape[0] // 2, correlation.shape[1] // 2
        shift_y = center_y - peak_y
        shift_x = center_x - peak_x

        # Account for 2x downsampling
        shifts.append((shift_y * 2, shift_x * 2))

    if method == "median":
        median_y = int(np.median([s[0] for s in shifts]))
        median_x = int(np.median([s[1] for s in shifts]))
        final_shift = (median_y, median_x)

    return final_shift, shifts


def create_dense_channel_image(values, mask):
    """Create dense image from sparse channel values by picking representative pixels."""
    h, w = values.shape[-2:]
    pattern_h, pattern_w = mask.shape

    sampled_h = h // pattern_h
    sampled_w = w // pattern_w

    dense = np.zeros((sampled_h, sampled_w), dtype=values.dtype)

    # Find first occurrence of True in mask - this is our representative position
    mask_positions = np.argwhere(mask)
    if len(mask_positions) == 0:
        return dense
    representative_y, representative_x = mask_positions[0]

    for i in range(sampled_h):
        for j in range(sampled_w):
            block_y = i * pattern_h + representative_y
            block_x = j * pattern_w + representative_x

            if block_y < h and block_x < w:
                dense[i, j] = values[block_y, block_x]

    return dense


def extract_xtrans_channels_dense(img, metadata):
    """Extract x-trans channels as dense downsampled arrays."""
    pattern = metadata["RGBG_pattern"]
    color_desc = metadata.get("color_desc", "RGBG")

    h, w = img.shape[-2:]
    img_2d = img[0]

    # Create masks for each color
    masks = {}
    for color_idx, color in enumerate(["R", "G", "B"]):
        if color in color_desc:
            idx = color_desc.index(color)
            masks[color] = pattern == idx

    # Extract dense channels
    channels = {}
    for color in ["R", "G", "B"]:
        if color in masks:
            channels[color] = create_dense_channel_image(img_2d, masks[color])

    return channels["R"], channels["G"], channels["B"]


def fft_phase_correlate_xtrans(anchor, target, anchor_meta, method="median"):
    """x-trans channel-split FFT."""
    anchor_R, anchor_G, anchor_B = extract_xtrans_channels_dense(anchor, anchor_meta)
    target_R, target_G, target_B = extract_xtrans_channels_dense(target, anchor_meta)

    channels_data = [(anchor_R, target_R), (anchor_G, target_G), (anchor_B, target_B)]

    shifts = []
    for anchor_ch, target_ch in channels_data:
        # Crop to common size (for synthetic misalignments with different sizes)
        min_h = min(anchor_ch.shape[0], target_ch.shape[0])
        min_w = min(anchor_ch.shape[1], target_ch.shape[1])
        anchor_ch = anchor_ch[:min_h, :min_w]
        target_ch = target_ch[:min_h, :min_w]

        f_anchor = np.fft.fft2(anchor_ch)
        f_target = np.fft.fft2(target_ch)

        cross_power = f_anchor * np.conj(f_target)
        cross_power /= np.abs(cross_power) + 1e-10

        correlation = np.fft.ifft2(cross_power).real
        correlation = np.fft.fftshift(correlation)

        peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)

        center_y, center_x = correlation.shape[0] // 2, correlation.shape[1] // 2
        shift_y = center_y - peak_y
        shift_x = center_x - peak_x

        # Account for 6x downsampling
        shifts.append((shift_y * 6, shift_x * 6))

    if method == "median":
        median_y = int(np.median([s[0] for s in shifts]))
        median_x = int(np.median([s[1] for s in shifts]))
        final_shift = (median_y, median_x)

    return final_shift, shifts


def create_synthetic_pair(img, offset):
    """
    Create two images with known relative offset by cropping opposite sides.

    Example: offset=5
    - anchor: crop 5px from top and right → [5:, :-5]
    - target: crop 5px from bottom and left → [:-5, 5:]

    Both have same dimensions, with content offset by (offset, -offset).
    Target content appears shifted by (offset, -offset) relative to anchor.

    Returns: (anchor, target, expected_shift)
    """
    _, h, w = img.shape

    # Anchor: remove top and right
    anchor = img[:, offset:, :-offset].copy()

    # Target: remove bottom and left
    target = img[:, :-offset, offset:].copy()

    # Expected shift: target content is DOWN and LEFT relative to anchor
    # In target coordinates, to find anchor content, we look up and right
    # So shift = (offset, -offset)
    expected_shift = (offset, -offset)

    return anchor, target, expected_shift


def test_synthetic_bayer():
    """Test bayer channel-split FFT with synthetic misalignments."""
    print("=" * 80)
    print("TESTING BAYER SYNTHETIC MISALIGNMENTS")
    print("=" * 80)

    # Load a Bayer image pair
    base_path = Path("DocScan/rawnind/datasets/RawNIND/DocScan/bayer/Bark")
    gt_file = (
        base_path
        / "gt"
        / "Bayer_Bark_GT_ISO100_sha1=f15da1140d949ee30c15ce7b251839a7b7a41de7.cr2"
    )

    print(f"\nLoading bayer test image: {gt_file.name}")
    gt_img, gt_meta = load_raw_image(str(gt_file))
    print(f"  Shape: {gt_img.shape}")
    print(f"  Pattern shape: {gt_meta['RGBG_pattern'].shape}")

    # Test different offsets (must be even for Bayer)
    test_offsets = [2, 4, 6, 10, 20]

    results = []

    for offset in test_offsets:
        # Create synthetic pair with known offset
        anchor, target, expected_shift = create_synthetic_pair(gt_img, offset)

        # Match gain
        target_matched = match_gain(anchor, target)

        # Run channel-split FFT
        detected_shift, channel_shifts = fft_phase_correlate_bayer(
            anchor, target_matched, gt_meta, method="median"
        )

        # Calculate error
        error_y = abs(detected_shift[0] - expected_shift[0])
        error_x = abs(detected_shift[1] - expected_shift[1])
        total_error = error_y + error_x

        results.append(
            {
                "offset": offset,
                "expected": expected_shift,
                "detected": detected_shift,
                "error": total_error,
                "channels": channel_shifts,
            }
        )

        status = "✅" if total_error == 0 else f"❌ {total_error}px error"
        print(f"\n  Offset: {offset}px → Expected shift: {expected_shift}")
        print(f"  Detected: {str(detected_shift):>15} {status}")
        if total_error > 0:
            print(f"  Channel shifts: {channel_shifts}")

    # Summary
    perfect = sum(1 for r in results if r["error"] == 0)
    print(f"\n{'=' * 80}")
    print(f"BAYER RESULTS: {perfect}/{len(results)} perfect detections")
    print(f"{'=' * 80}")

    return results


def test_synthetic_xtrans():
    """Test x-trans channel-split FFT with synthetic misalignments."""
    print("\n\n")
    print("=" * 80)
    print("TESTING X-TRANS SYNTHETIC MISALIGNMENTS")
    print("=" * 80)

    # Load an X-Trans image
    base_path = Path("DocScan/rawnind/datasets/RawNIND/DocScan/x-trans/books")
    gt_file = (
        base_path
        / "gt"
        / "X-Trans_books_GT_ISO200_sha1=b2add962f3fd1cd828a1cd5e3c29f7226aebb6df.raf"
    )

    print(f"\nLoading x-trans test image: {gt_file.name}")
    gt_img, gt_meta = load_raw_image(str(gt_file))
    print(f"  Shape: {gt_img.shape}")
    print(f"  Pattern shape: {gt_meta['RGBG_pattern'].shape}")

    # Test different offsets (use multiples of 6 for clean X-Trans alignment)
    test_offsets = [6, 12, 18, 24, 30]

    results = []

    for offset in test_offsets:
        # Create synthetic pair with known offset
        anchor, target, expected_shift = create_synthetic_pair(gt_img, offset)

        # Match gain
        target_matched = match_gain(anchor, target)

        # Run channel-split FFT
        detected_shift, channel_shifts = fft_phase_correlate_xtrans(
            anchor, target_matched, gt_meta, method="median"
        )

        # Calculate error
        error_y = abs(detected_shift[0] - expected_shift[0])
        error_x = abs(detected_shift[1] - expected_shift[1])
        total_error = error_y + error_x

        results.append(
            {
                "offset": offset,
                "expected": expected_shift,
                "detected": detected_shift,
                "error": total_error,
                "channels": channel_shifts,
            }
        )

        status = "✅" if total_error == 0 else f"❌ {total_error}px error"
        print(f"\n  Offset: {offset}px → Expected shift: {expected_shift}")
        print(f"  Detected: {str(detected_shift):>15} {status}")
        if total_error > 0:
            print(f"  Channel shifts: {channel_shifts}")

    # Summary
    perfect = sum(1 for r in results if r["error"] == 0)
    print(f"\n{'=' * 80}")
    print(f"X-TRANS RESULTS: {perfect}/{len(results)} perfect detections")
    print(f"{'=' * 80}")

    return results


if __name__ == "__main__":
    print("Testing channel-split FFT with synthetic misalignments...")
    print("This validates the approach on known shifts before production use.\n")

    bayer_results = test_synthetic_bayer()
    xtrans_results = test_synthetic_xtrans()

    # Overall summary
    bayer_perfect = sum(1 for r in bayer_results if r["error"] == 0)
    xtrans_perfect = sum(1 for r in xtrans_results if r["error"] == 0)

    print("\n\n")
    print("=" * 80)
    print("SYNTHETIC MISALIGNMENT VALIDATION COMPLETE")
    print("=" * 80)
    print(f"bayer:   {bayer_perfect}/{len(bayer_results)} perfect")
    print(f"x-trans: {xtrans_perfect}/{len(xtrans_results)} perfect")
    print()

    if bayer_perfect == len(bayer_results) and xtrans_perfect == len(xtrans_results):
        print("✅ ALL TESTS PASSED - Channel-split FFT validated for both CFA types!")
    else:
        print("⚠️  Some tests had errors - review results above")
