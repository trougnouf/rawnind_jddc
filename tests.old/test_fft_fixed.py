"""Test the FIXED FFT implementation on all test cases."""

import numpy as np
import sys
from pathlib import Path

sys.path.append("src")
from rawnind.libs import raw
from rawnind.libs.rawproc import shift_images, match_gain


def load_raw_image(fpath: str):
    """Load RAW image as [1, H, W] mosaiced Bayer."""
    img, metadata = raw.raw_fpath_to_mono_img_and_metadata(fpath)
    return img, metadata


def fft_phase_correlate_FIXED(anchor, target, max_shift=128):
    """FIXED FFT phase correlation - NO NEGATION."""
    anchor_img = anchor[0]
    target_img = target[0]

    # Use center crop to reduce computation and avoid edge effects
    h, w = anchor_img.shape
    crop_h, crop_w = h // 2, w // 2
    y0, x0 = h // 4, w // 4

    anchor_crop = anchor_img[y0 : y0 + crop_h, x0 : x0 + crop_w]
    target_crop = target_img[y0 : y0 + crop_h, x0 : x0 + crop_w]

    # Mean center to remove DC component (improves phase correlation)
    anchor_centered = anchor_crop - anchor_crop.mean()
    target_centered = target_crop - target_crop.mean()

    # FFT
    f1 = np.fft.fft2(anchor_centered)
    f2 = np.fft.fft2(target_centered)

    # Cross-power spectrum (phase correlation)
    cross_power = (f1 * np.conj(f2)) / (np.abs(f1 * np.conj(f2)) + 1e-10)

    # Inverse FFT to get correlation surface
    correlation = np.fft.ifft2(cross_power).real

    # Find peak
    peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)

    # Handle FFT wraparound
    if peak_y > crop_h // 2:
        peak_y -= crop_h
    if peak_x > crop_w // 2:
        peak_x -= crop_w

    # FIX: Don't negate! The peak position IS the shift
    shift = (peak_y, peak_x)

    # Clamp to max_shift
    shift = (
        max(-max_shift, min(max_shift, shift[0])),
        max(-max_shift, min(max_shift, shift[1])),
    )

    return shift


def find_alignment_simple(anchor_img, candidate_img, max_shift=128, neighborhood=3):
    """Simple greedy neighbor search for comparison."""
    candidate_img = match_gain(anchor_img, candidate_img)

    best_shift = (0, 0)
    anchor_crop, candidate_crop = shift_images(anchor_img, candidate_img, best_shift)
    best_loss = float(np.abs(anchor_crop - candidate_crop).mean())

    visited = {best_shift}
    improved = True

    while improved and (abs(best_shift[0]) + abs(best_shift[1]) < max_shift):
        improved = False
        for dy in range(-neighborhood, neighborhood + 1):
            for dx in range(-neighborhood, neighborhood + 1):
                test_shift = (best_shift[0] + dy, best_shift[1] + dx)
                if test_shift in visited:
                    continue
                visited.add(test_shift)

                try:
                    anchor_crop, candidate_crop = shift_images(
                        anchor_img, candidate_img, test_shift
                    )
                    loss = float(np.abs(anchor_crop - candidate_crop).mean())

                    if loss < best_loss:
                        best_loss = loss
                        best_shift = test_shift
                        improved = True
                except:
                    pass

        if not improved:
            break

    return best_shift, best_loss


# Test cases from previous runs
test_cases = [
    (
        "Bark",
        "Bayer_Bark_GT_ISO100_sha1=f15da1140d949ee30c15ce7b251839a7b7a41de7.cr2",
        "Bayer_Bark_ISO65535_sha1=6ba8ed5f7fff42c4c900812c02701649f4f2d49e.cr2",
        (12, -10),
    ),
    (
        "Bark",
        "Bayer_Bark_GT_ISO100_sha1=f15da1140d949ee30c15ce7b251839a7b7a41de7.cr2",
        "Bayer_Bark_ISO800_sha1=ba86f1da64a4bb534d9216e96c1c72177ed1e625.cr2",
        (4, -4),
    ),
    (
        "Kortlek",
        "Bayer_Kortlek_GT_ISO100_sha1=b5a564cd9291224caf363b6b03054365d59d316b.cr2",
        "Bayer_Kortlek_ISO51200_sha1=8fed453fdfb162673bb9ed41f9c5f03095331e3b.cr2",
        (0, -4),
    ),
]

base_dir = Path("src/rawnind/datasets/RawNIND/src/Bayer")

print("=" * 80)
print("TESTING FIXED FFT IMPLEMENTATION")
print("=" * 80)
print()

for scene, gt_name, noisy_name, expected_shift in test_cases:
    print(f"Scene: {scene}")
    print(f"  Expected shift: {expected_shift}")

    # Load images
    gt_path = base_dir / scene / "gt" / gt_name
    noisy_path = base_dir / scene / noisy_name

    if not gt_path.exists() or not noisy_path.exists():
        print(f"  ❌ Files not found")
        print()
        continue

    gt_img, _ = load_raw_image(str(gt_path))
    noisy_img, _ = load_raw_image(str(noisy_path))
    noisy_matched = match_gain(gt_img, noisy_img)

    # Test simple method
    simple_shift, simple_loss = find_alignment_simple(gt_img, noisy_img)

    # Test fixed FFT
    fft_shift = fft_phase_correlate_FIXED(gt_img, noisy_matched)
    anchor_out, target_out = shift_images(gt_img, noisy_matched, fft_shift)
    fft_loss = float(np.abs(anchor_out - target_out).mean())

    # Compare
    simple_match = "✅" if simple_shift == expected_shift else "❌"
    fft_match = "✅" if fft_shift == expected_shift else "❌"
    fft_error = abs(fft_shift[0] - expected_shift[0]) + abs(
        fft_shift[1] - expected_shift[1]
    )

    print(
        f"  Simple:     {simple_match} Shift: {simple_shift}, Loss: {simple_loss:.6f}"
    )
    print(
        f"  FFT (FIXED): {fft_match} Shift: {fft_shift}, Loss: {fft_loss:.6f}, Error: {fft_error}px"
    )
    print()

print("=" * 80)
print("RESULT")
print("=" * 80)
print("The fix is simply removing the negation:")
print()
print("BEFORE:")
print("  shift = (-peak_y, -peak_x)")
print()
print("AFTER:")
print("  shift = (peak_y, peak_x)")
print()
print("This is consistent with the FFT shift theorem and phase correlation literature.")
