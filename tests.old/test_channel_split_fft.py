"""Test channel-split FFT on Bayer CFA data."""
import numpy as np
import sys
from pathlib import Path
import json
sys.path.append("src")
from rawnind.libs import raw
from rawnind.libs.rawproc import shift_images, match_gain

def load_raw_image(fpath: str):
    """Load RAW image as [1, H, W] mosaiced Bayer."""
    img, metadata = raw.raw_fpath_to_mono_img_and_metadata(fpath)
    return img, metadata

def extract_bayer_channels(img):
    """
    Extract 4 Bayer channels from mosaiced image.
    Assumes RGGB pattern (most common).
    
    Returns: (R, G1, G2, B) channels, each with shape matching their sampling.
    """
    # img shape: [1, H, W]
    img_2d = img[0]
    h, w = img_2d.shape
    
    # RGGB pattern:
    # R  G1    R  G1
    # G2 B     G2 B
    
    # Extract channels (subsample by 2 in each dimension)
    R  = img_2d[0::2, 0::2]  # Top-left
    G1 = img_2d[0::2, 1::2]  # Top-right
    G2 = img_2d[1::2, 0::2]  # Bottom-left
    B  = img_2d[1::2, 1::2]  # Bottom-right
    
    return R, G1, G2, B

def fft_phase_correlate_single_channel(anchor_ch, target_ch):
    """
    FFT phase correlation on a single channel.
    Returns: shift (dy, dx)
    """
    # Mean center
    anchor_centered = anchor_ch - anchor_ch.mean()
    target_centered = target_ch - target_ch.mean()
    
    # FFT
    f1 = np.fft.fft2(anchor_centered)
    f2 = np.fft.fft2(target_centered)
    
    # Cross-power spectrum
    cross_power = (f1 * np.conj(f2)) / (np.abs(f1 * np.conj(f2)) + 1e-10)
    
    # Inverse FFT
    correlation = np.fft.ifft2(cross_power).real
    
    # Find peak
    h, w = correlation.shape
    peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)
    
    # Handle wraparound
    if peak_y > h // 2:
        peak_y -= h
    if peak_x > w // 2:
        peak_x -= w
    
    # Note: peak position IS the shift (no negation)
    return (peak_y, peak_x)

def fft_phase_correlate_channel_split(anchor, target, method="median"):
    """
    FFT phase correlation with Bayer channel splitting.
    
    Args:
        anchor: [1, H, W] mosaiced Bayer image
        target: [1, H, W] mosaiced Bayer image
        method: "median", "mean", or "mode" for combining channel results
    
    Returns:
        shift (dy, dx) in FULL image coordinates
    """
    # Extract Bayer channels
    R_a, G1_a, G2_a, B_a = extract_bayer_channels(anchor)
    R_t, G1_t, G2_t, B_t = extract_bayer_channels(target)
    
    # Run FFT on each channel
    channels = [
        ("R", R_a, R_t),
        ("G1", G1_a, G1_t),
        ("G2", G2_a, G2_t),
        ("B", B_a, B_t),
    ]
    
    shifts_per_channel = []
    
    for name, anchor_ch, target_ch in channels:
        shift = fft_phase_correlate_single_channel(anchor_ch, target_ch)
        # NOTE: Shift is in subsampled coordinates, scale by 2 for full image
        shift_full = (shift[0] * 2, shift[1] * 2)
        shifts_per_channel.append(shift_full)
    
    # Combine results
    if method == "median":
        # Median of each dimension
        dy_median = int(np.median([s[0] for s in shifts_per_channel]))
        dx_median = int(np.median([s[1] for s in shifts_per_channel]))
        final_shift = (dy_median, dx_median)
    elif method == "mean":
        # Mean of each dimension
        dy_mean = int(np.round(np.mean([s[0] for s in shifts_per_channel])))
        dx_mean = int(np.round(np.mean([s[1] for s in shifts_per_channel])))
        final_shift = (dy_mean, dx_mean)
    elif method == "mode":
        # Most common shift
        from collections import Counter
        counter = Counter(shifts_per_channel)
        final_shift = counter.most_common(1)[0][0]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return final_shift, shifts_per_channel

def fft_phase_correlate_standard(anchor, target):
    """Standard FFT on full mosaiced image (for comparison)."""
    anchor_img = anchor[0]
    target_img = target[0]
    
    # Use center crop
    h, w = anchor_img.shape
    crop_h, crop_w = h // 2, w // 2
    y0, x0 = h // 4, w // 4
    
    anchor_crop = anchor_img[y0:y0+crop_h, x0:x0+crop_w]
    target_crop = target_img[y0:y0+crop_h, x0:x0+crop_w]
    
    # Mean center
    anchor_centered = anchor_crop - anchor_crop.mean()
    target_centered = target_crop - target_crop.mean()
    
    # FFT
    f1 = np.fft.fft2(anchor_centered)
    f2 = np.fft.fft2(target_centered)
    
    cross_power = (f1 * np.conj(f2)) / (np.abs(f1 * np.conj(f2)) + 1e-10)
    correlation = np.fft.ifft2(cross_power).real
    
    # Find peak
    peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)
    
    if peak_y > crop_h // 2:
        peak_y -= crop_h
    if peak_x > crop_w // 2:
        peak_x -= crop_w
    
    # No negation (fixed version)
    shift = (peak_y, peak_x)
    return shift

print("="*80)
print("TESTING CHANNEL-SPLIT FFT ON BAYER DATA")
print("="*80)
print()

# Load test dataset
with open("curated_test_dataset.json") as f:
    test_dataset = json.load(f)

base_path = Path("src/rawnind/datasets/RawNIND/src")

# Test on misaligned pairs
for test_case in test_dataset["test_cases"]["misaligned_bayer"]:
    print(f"Scene: {test_case['scene']}")
    print(f"  Expected shift: {tuple(test_case['expected_shift'])}")
    print(f"  Notes: {test_case['notes']}")
    print()
    
    # Load images
    gt_path = base_path / test_case["gt_file"]
    noisy_path = base_path / test_case["noisy_file"]
    
    if not gt_path.exists() or not noisy_path.exists():
        print(f"  ❌ Files not found")
        print()
        continue
    
    gt_img, _ = load_raw_image(str(gt_path))
    noisy_img, _ = load_raw_image(str(noisy_path))
    noisy_matched = match_gain(gt_img, noisy_img)
    
    # Standard FFT
    print("  [1/3] Standard FFT (full mosaiced image)...")
    fft_standard = fft_phase_correlate_standard(gt_img, noisy_matched)
    anchor_out, target_out = shift_images(gt_img, noisy_matched, fft_standard)
    loss_standard = float(np.abs(anchor_out - target_out).mean())
    error_standard = abs(fft_standard[0] - test_case['expected_shift'][0]) + abs(fft_standard[1] - test_case['expected_shift'][1])
    match_standard = "✅" if fft_standard == tuple(test_case['expected_shift']) else "❌"
    print(f"    {match_standard} Shift: {fft_standard}, Loss: {loss_standard:.6f}, Error: {error_standard}px")
    
    # Channel-split FFT with median
    print("  [2/3] Channel-split FFT (median)...")
    fft_split_median, channel_shifts = fft_phase_correlate_channel_split(gt_img, noisy_matched, method="median")
    anchor_out, target_out = shift_images(gt_img, noisy_matched, fft_split_median)
    loss_split_median = float(np.abs(anchor_out - target_out).mean())
    error_split_median = abs(fft_split_median[0] - test_case['expected_shift'][0]) + abs(fft_split_median[1] - test_case['expected_shift'][1])
    match_split_median = "✅" if fft_split_median == tuple(test_case['expected_shift']) else "❌"
    print(f"    Per-channel shifts: R={channel_shifts[0]}, G1={channel_shifts[1]}, G2={channel_shifts[2]}, B={channel_shifts[3]}")
    print(f"    {match_split_median} Median shift: {fft_split_median}, Loss: {loss_split_median:.6f}, Error: {error_split_median}px")
    
    # Channel-split FFT with mean
    print("  [3/3] Channel-split FFT (mean)...")
    fft_split_mean, channel_shifts = fft_phase_correlate_channel_split(gt_img, noisy_matched, method="mean")
    anchor_out, target_out = shift_images(gt_img, noisy_matched, fft_split_mean)
    loss_split_mean = float(np.abs(anchor_out - target_out).mean())
    error_split_mean = abs(fft_split_mean[0] - test_case['expected_shift'][0]) + abs(fft_split_mean[1] - test_case['expected_shift'][1])
    match_split_mean = "✅" if fft_split_mean == tuple(test_case['expected_shift']) else "❌"
    print(f"    {match_split_mean} Mean shift: {fft_split_mean}, Loss: {loss_split_mean:.6f}, Error: {error_split_mean}px")
    
    # Compare
    print()
    print(f"  Comparison:")
    print(f"    Standard:       Error={error_standard}px")
    print(f"    Channel-median: Error={error_split_median}px {'🎯 BETTER' if error_split_median < error_standard else ''}")
    print(f"    Channel-mean:   Error={error_split_mean}px {'🎯 BETTER' if error_split_mean < error_standard else ''}")
    print()

print("="*80)
print("HYPOTHESIS TEST")
print("="*80)
print("If channel-split reduces error on Bark ISO65535 (12,-10),")
print("this confirms CFA pattern interference hypothesis.")
