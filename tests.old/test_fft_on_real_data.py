"""Test FFT on real RAW data with known synthetic shifts."""

import numpy as np
import sys
from pathlib import Path

sys.path.append("DocScan")
from rawnind.libs import raw
from rawnind.libs.rawproc import shift_images, match_gain


def load_raw_image(fpath: str):
    """Load RAW image as [1, H, W] mosaiced bayer."""
    img, metadata = raw.raw_fpath_to_mono_img_and_metadata(fpath)
    return img, metadata


def fft_phase_correlate_v1(anchor, target):
    """Original FFT with negation."""
    anchor_img = anchor[0]
    target_img = target[0]

    h, w = anchor_img.shape
    crop_h, crop_w = h // 2, w // 2
    y0, x0 = h // 4, w // 4

    anchor_crop = anchor_img[y0 : y0 + crop_h, x0 : x0 + crop_w]
    target_crop = target_img[y0 : y0 + crop_h, x0 : x0 + crop_w]

    anchor_centered = anchor_crop - anchor_crop.mean()
    target_centered = target_crop - target_crop.mean()

    f1 = np.fft.fft2(anchor_centered)
    f2 = np.fft.fft2(target_centered)

    cross_power = (f1 * np.conj(f2)) / (np.abs(f1 * np.conj(f2)) + 1e-10)
    correlation = np.fft.ifft2(cross_power).real

    peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)

    if peak_y > crop_h // 2:
        peak_y -= crop_h
    if peak_x > crop_w // 2:
        peak_x -= crop_w

    # Original: negate
    shift = (-peak_y, -peak_x)
    return shift


def fft_phase_correlate_v2(anchor, target):
    """FFT WITHOUT negation."""
    anchor_img = anchor[0]
    target_img = target[0]

    h, w = anchor_img.shape
    crop_h, crop_w = h // 2, w // 2
    y0, x0 = h // 4, w // 4

    anchor_crop = anchor_img[y0 : y0 + crop_h, x0 : x0 + crop_w]
    target_crop = target_img[y0 : y0 + crop_h, x0 : x0 + crop_w]

    anchor_centered = anchor_crop - anchor_crop.mean()
    target_centered = target_crop - target_crop.mean()

    f1 = np.fft.fft2(anchor_centered)
    f2 = np.fft.fft2(target_centered)

    cross_power = (f1 * np.conj(f2)) / (np.abs(f1 * np.conj(f2)) + 1e-10)
    correlation = np.fft.ifft2(cross_power).real

    peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)

    if peak_y > crop_h // 2:
        peak_y -= crop_h
    if peak_x > crop_w // 2:
        peak_x -= crop_w

    # No negation
    shift = (peak_y, peak_x)
    return shift


def fft_phase_correlate_v3(anchor, target):
    """FFT with SWAPPED arguments."""
    anchor_img = anchor[0]
    target_img = target[0]

    h, w = anchor_img.shape
    crop_h, crop_w = h // 2, w // 2
    y0, x0 = h // 4, w // 4

    anchor_crop = anchor_img[y0 : y0 + crop_h, x0 : x0 + crop_w]
    target_crop = target_img[y0 : y0 + crop_h, x0 : x0 + crop_w]

    anchor_centered = anchor_crop - anchor_crop.mean()
    target_centered = target_crop - target_crop.mean()

    # SWAP: f2 conj f1 instead of f1 conj f2
    f1 = np.fft.fft2(anchor_centered)
    f2 = np.fft.fft2(target_centered)

    cross_power = (f2 * np.conj(f1)) / (np.abs(f2 * np.conj(f1)) + 1e-10)
    correlation = np.fft.ifft2(cross_power).real

    peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)

    if peak_y > crop_h // 2:
        peak_y -= crop_h
    if peak_x > crop_w // 2:
        peak_x -= crop_w

    # No negation
    shift = (peak_y, peak_x)
    return shift


print("=" * 80)
print("FFT ON REAL DATA WITH SYNTHETIC SHIFTS")
print("=" * 80)
print()

# Load real RAW image
base_dir = Path("DocScan/rawnind/datasets/RawNIND/DocScan/bayer")
gt_path = (
    base_dir
    / "Bark"
    / "gt"
    / "Bayer_Bark_GT_ISO100_sha1=f15da1140d949ee30c15ce7b251839a7b7a41de7.cr2"
)

print("Loading GT image...")
gt_img, _ = load_raw_image(str(gt_path))
print(f"Image shape: {gt_img.shape}")
print(f"Data range: [{gt_img.min():.3f}, {gt_img.max():.3f}]")
print()

# Apply synthetic shifts using np.roll
test_shifts = [
    (0, 0, "No shift"),
    (-4, -4, "Roll DOWN 4, RIGHT 4"),
    (-10, 5, "Roll DOWN 10, LEFT 5"),
]

print("Testing FFT with synthetic shifts applied to real RAW data:")
print()

for dy, dx, description in test_shifts:
    print(f"Test: {description}")
    print(f"  Applied roll: ({-dy}, {-dx})")
    print(f"  Expected shift: ({dy}, {dx})")
    print()

    # Apply shift
    if dy == 0 and dx == 0:
        shifted_img = gt_img.copy()
    else:
        shifted_img = np.roll(np.roll(gt_img, -dy, axis=1), -dx, axis=2)

    # Gain match
    shifted_matched = match_gain(gt_img, shifted_img)

    # Test FFT variants
    fft_v1 = fft_phase_correlate_v1(gt_img, shifted_matched)
    fft_v2 = fft_phase_correlate_v2(gt_img, shifted_matched)
    fft_v3 = fft_phase_correlate_v3(gt_img, shifted_matched)

    # Compute losses
    def get_loss(shift):
        anchor_out, target_out = shift_images(gt_img, shifted_matched, shift)
        return float(np.abs(anchor_out - target_out).mean())

    loss_v1 = get_loss(fft_v1)
    loss_v2 = get_loss(fft_v2)
    loss_v3 = get_loss(fft_v3)
    expected_loss = get_loss((dy, dx))

    print(
        f"  V1 (original with negation):  {fft_v1}, loss={loss_v1:.6f} {'✅' if fft_v1 == (dy, dx) else '❌'}"
    )
    print(
        f"  V2 (no negation):             {fft_v2}, loss={loss_v2:.6f} {'✅' if fft_v2 == (dy, dx) else '❌'}"
    )
    print(
        f"  V3 (swapped args):            {fft_v3}, loss={loss_v3:.6f} {'✅' if fft_v3 == (dy, dx) else '❌'}"
    )
    print(f"  Expected:                     ({dy}, {dx}), loss={expected_loss:.6f}")
    print()

print("=" * 80)
print("NOW TEST ON ACTUAL MISALIGNED PAIR")
print("=" * 80)
print()

noisy_path = (
    base_dir
    / "Bark"
    / "Bayer_Bark_ISO800_sha1=ba86f1da64a4bb534d9216e96c1c72177ed1e625.cr2"
)
noisy_img, _ = load_raw_image(str(noisy_path))
noisy_matched = match_gain(gt_img, noisy_img)

print("Bark GT vs ISO800 (known misalignment)")
print("Expected shift from simple method: (4, -4)")
print()

fft_v1 = fft_phase_correlate_v1(gt_img, noisy_matched)
fft_v2 = fft_phase_correlate_v2(gt_img, noisy_matched)
fft_v3 = fft_phase_correlate_v3(gt_img, noisy_matched)

loss_v1 = get_loss(fft_v1)
loss_v2 = get_loss(fft_v2)
loss_v3 = get_loss(fft_v3)
expected_loss = get_loss((4, -4))

print(
    f"V1 (original with negation):  {fft_v1}, loss={loss_v1:.6f} {'✅' if fft_v1 == (4, -4) else '❌'}"
)
print(
    f"V2 (no negation):             {fft_v2}, loss={loss_v2:.6f} {'✅' if fft_v2 == (4, -4) else '❌'}"
)
print(
    f"V3 (swapped args):            {fft_v3}, loss={loss_v3:.6f} {'✅' if fft_v3 == (4, -4) else '❌'}"
)
print(f"Expected:                     (4, -4), loss={expected_loss:.6f}")
print()

print("=" * 80)
print("ANALYSIS")
print("=" * 80)
print("If FFT works on synthetic shifts but fails on real misalignment,")
print("this suggests the real images have more than just translation:")
print("  - Possible rotation")
print("  - Possible scale change")
print("  - CFA pattern interference")
print("  - Content differences (noise, motion blur, etc.)")
