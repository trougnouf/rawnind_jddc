"""Test alignment on more scenes with debug output to find misaligned pairs."""

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


def find_alignment_simple(anchor_img, candidate_img, max_shift=128, neighborhood=3):
    """Simple greedy neighbor search for alignment."""
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


# Find all Bayer scenes
bayer_dir = Path("src/rawnind/datasets/RawNIND/src/Bayer")
scene_dirs = sorted([d for d in bayer_dir.iterdir() if d.is_dir()])

print(f"Found {len(scene_dirs)} total scenes")
print()

# Sample scenes from different cameras/sources
test_scenes = []
for scene_dir in scene_dirs[:30]:  # Test first 30
    gt_dir = scene_dir / "gt"
    if not gt_dir.exists():
        continue

    gt_files = list(gt_dir.glob("*.cr2")) + list(gt_dir.glob("*.CR2"))
    if not gt_files:
        continue

    noisy_files = list(scene_dir.glob("*.cr2")) + list(scene_dir.glob("*.CR2"))
    if len(noisy_files) >= 2:
        test_scenes.append(
            {
                "scene": scene_dir.name,
                "gt": str(gt_files[0]),
                "noisy": [str(f) for f in noisy_files[:3]],
            }
        )

print(f"Testing {len(test_scenes)} scenes with GT and noisy pairs")
print("=" * 80)

non_zero_shifts = []

for i, scene in enumerate(test_scenes[:20]):  # Test first 20
    print(f"\n[{i + 1}/{len(test_scenes)}] Scene: {scene['scene']}")
    print(f"  GT: {Path(scene['gt']).name}")

    try:
        gt_img, _ = load_raw_image(scene["gt"])

        for noisy_fpath in scene["noisy"]:
            noisy_name = Path(noisy_fpath).name
            print(f"  Testing: {noisy_name}")

            noisy_img, _ = load_raw_image(noisy_fpath)

            # Check if images are same size
            if gt_img.shape != noisy_img.shape:
                print(
                    f"    ⚠️  Size mismatch: GT {gt_img.shape} vs noisy {noisy_img.shape}"
                )
                continue

            # Find alignment
            shift, loss = find_alignment_simple(gt_img, noisy_img)

            if shift != (0, 0):
                marker = "🔴"
                non_zero_shifts.append(
                    {
                        "scene": scene["scene"],
                        "gt": Path(scene["gt"]).name,
                        "noisy": noisy_name,
                        "shift": shift,
                        "loss": loss,
                    }
                )
            else:
                marker = "✓"

            print(f"    {marker} shift={shift}, loss={loss:.6f}")

    except Exception as e:
        print(f"  ❌ Error: {e}")

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Tested {len(test_scenes[:20])} scenes")
print(f"Found {len(non_zero_shifts)} pairs with non-zero shifts:")
print()

if non_zero_shifts:
    for item in non_zero_shifts:
        print(f"  Scene: {item['scene']}")
        print(f"    GT: {item['gt']}")
        print(f"    Noisy: {item['noisy']}")
        print(f"    Shift: {item['shift']}, Loss: {item['loss']:.6f}")
        print()
else:
    print("  ⚠️  ALL TESTED PAIRS HAD SHIFT=(0,0)")
    print("  This suggests either:")
    print("    1. Images are pre-aligned (as stated)")
    print("    2. Need to test more scenes")
    print("    3. Need to look at specific problem cases")
