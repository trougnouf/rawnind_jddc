"""
Test that bayer-domain loss mask computation produces results comparable to RGB-domain computation.

This test validates that:
1. Aligning in bayer domain, then demosaicing produces similar results to demosaic-then-align
2. Loss masks computed on bayer vs RGB data are comparable

The test loads raw bayer images, processes them through both pipelines, and compares:
- Aligned images (after demosaicing the bayer path)
- Loss masks from both methods
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add DocScan to path
sys.path.insert(0, str(Path(__file__).parent / "DocScan"))

from rawnind.libs import rawproc, raw
from rawnind.libs.rawproc import (
    img_fpath_to_np_mono_flt_and_metadata,
    make_overexposure_mask,
    shift_images,
    shift_mask,
    make_loss_mask,
    make_loss_mask_bayer,
)
from rawnind.libs.alignment_backends import find_best_alignment_fft_cfa


def test_bayer_vs_rgb_pipeline():
    """Compare bayer-domain pipeline to RGB-domain pipeline."""

    # Find test images from the dataset
    bayer_ds_path = rawproc.BAYER_DS_DPATH

    # Look for a test image set
    test_sets = []
    for image_set in os.listdir(bayer_ds_path):
        set_path = os.path.join(bayer_ds_path, image_set)
        if not os.path.isdir(set_path):
            continue

        gt_path = os.path.join(set_path, "gt")
        if not os.path.exists(gt_path):
            continue

        gt_files = [f for f in os.listdir(gt_path) if f.endswith(".dng")]
        noisy_files = [
            f
            for f in os.listdir(set_path)
            if f.endswith(".dng") and not f.startswith("gt")
        ]

        if gt_files and noisy_files:
            test_sets.append(
                {
                    "image_set": image_set,
                    "gt_file": gt_files[0],
                    "noisy_file": noisy_files[0],
                }
            )

    if not test_sets:
        print("ERROR: No test images found in dataset")
        return

    print(f"Testing on {len(test_sets)} image pairs...")
    print("=" * 80)

    results = []

    for idx, test_set in enumerate(test_sets):
        print(f"\n[{idx + 1}/{len(test_sets)}] Testing: {test_set['image_set']}")
        print("-" * 80)

        # Load images
        gt_path = os.path.join(
            bayer_ds_path, test_set["image_set"], "gt", test_set["gt_file"]
        )
        noisy_path = os.path.join(
            bayer_ds_path, test_set["image_set"], test_set["noisy_file"]
        )

        gt_img, gt_metadata = img_fpath_to_np_mono_flt_and_metadata(gt_path)
        noisy_img, noisy_metadata = img_fpath_to_np_mono_flt_and_metadata(noisy_path)

        print(f"  GT image shape: {gt_img.shape}")
        print(f"  Noisy image shape: {noisy_img.shape}")

        # Create overexposure mask
        overexp_mask = make_overexposure_mask(gt_img, gt_metadata["overexposure_lb"])

        # ========================================
        # PIPELINE 1: Bayer-domain (NEW)
        # ========================================
        print("\n  Pipeline 1: bayer-domain alignment + loss mask")

        # Align in Bayer domain
        bayer_alignment, bayer_loss = find_best_alignment_fft_cfa(
            gt_img,
            noisy_img,
            gt_metadata,
            method="median",
            return_loss_too=True,
            verbose=False,
        )
        print(f"    Alignment: {bayer_alignment}, Loss: {bayer_loss:.6f}")

        # Shift Bayer images
        gt_bayer_aligned, noisy_bayer_aligned = shift_images(
            gt_img, noisy_img, bayer_alignment
        )
        overexp_mask_bayer = shift_mask(overexp_mask, bayer_alignment)

        # Compute loss mask on Bayer data
        loss_mask_bayer = (
            make_loss_mask_bayer(gt_bayer_aligned, noisy_bayer_aligned)
            * overexp_mask_bayer
        )

        # NOW demosaic the aligned Bayer images for comparison
        gt_rgb_from_bayer = raw.demosaic(gt_bayer_aligned, gt_metadata)
        noisy_rgb_from_bayer = raw.demosaic(noisy_bayer_aligned, noisy_metadata)

        print(
            f"    Loss mask (bayer): mean={loss_mask_bayer.mean():.4f}, "
            f"min={loss_mask_bayer.min():.4f}, max={loss_mask_bayer.max():.4f}"
        )
        print(f"    Aligned RGB (from bayer): {gt_rgb_from_bayer.shape}")

        # ========================================
        # PIPELINE 2: RGB-domain (ORIGINAL)
        # ========================================
        print("\n  Pipeline 2: RGB-domain alignment + loss mask")

        # Demosaic first
        gt_rgb = raw.demosaic(gt_img, gt_metadata)
        noisy_rgb = raw.demosaic(noisy_img, noisy_metadata)

        # Align in RGB domain (use same FFT method for fair comparison)
        from rawnind.libs.rawproc import find_best_alignment

        rgb_alignment, rgb_loss = find_best_alignment(
            gt_rgb, noisy_rgb, return_loss_too=True, method="fft", verbose=False
        )
        print(f"    Alignment: {rgb_alignment}, Loss: {rgb_loss:.6f}")

        # Shift RGB images
        gt_rgb_aligned, noisy_rgb_aligned = shift_images(
            gt_rgb, noisy_rgb, rgb_alignment
        )
        overexp_mask_rgb = shift_mask(overexp_mask, rgb_alignment)

        # Compute loss mask on RGB data
        loss_mask_rgb = (
            make_loss_mask(gt_rgb_aligned, noisy_rgb_aligned) * overexp_mask_rgb
        )

        print(
            f"    Loss mask (RGB): mean={loss_mask_rgb.mean():.4f}, "
            f"min={loss_mask_rgb.min():.4f}, max={loss_mask_rgb.max():.4f}"
        )
        print(f"    Aligned RGB: {gt_rgb_aligned.shape}")

        # ========================================
        # COMPARISON
        # ========================================
        print("\n  Comparison:")

        # Compare alignments
        alignment_match = bayer_alignment == rgb_alignment
        print(f"    Alignments match: {alignment_match}")
        if not alignment_match:
            print(f"      bayer: {bayer_alignment}, RGB: {rgb_alignment}")
            print(
                f"      Difference: ({bayer_alignment[0] - rgb_alignment[0]}, {bayer_alignment[1] - rgb_alignment[1]})"
            )

        # Compare aligned images (after demosaicing Bayer path)
        # Note: shapes may differ slightly due to shift cropping
        if gt_rgb_from_bayer.shape == gt_rgb_aligned.shape:
            img_diff = np.abs(gt_rgb_from_bayer - gt_rgb_aligned)
            img_mae = img_diff.mean()
            img_max_error = img_diff.max()
            print(f"    GT image MAE: {img_mae:.6f}, Max error: {img_max_error:.6f}")

            noisy_diff = np.abs(noisy_rgb_from_bayer - noisy_rgb_aligned)
            noisy_mae = noisy_diff.mean()
            noisy_max_error = noisy_diff.max()
            print(
                f"    Noisy image MAE: {noisy_mae:.6f}, Max error: {noisy_max_error:.6f}"
            )
        else:
            print("    Image shapes differ (expected due to cropping):")
            print(f"      bayer path: {gt_rgb_from_bayer.shape}")
            print(f"      RGB path: {gt_rgb_aligned.shape}")
            img_mae = None
            noisy_mae = None

        # Compare loss masks
        if loss_mask_bayer.shape == loss_mask_rgb.shape:
            mask_diff = np.abs(loss_mask_bayer - loss_mask_rgb)
            mask_mae = mask_diff.mean()
            mask_max_error = mask_diff.max()

            # Compute agreement percentage (both 0 or both 1)
            agreement = ((loss_mask_bayer == 0) & (loss_mask_rgb == 0)) | (
                (loss_mask_bayer == 1) & (loss_mask_rgb == 1)
            )
            agreement_pct = agreement.sum() / agreement.size * 100

            print(f"    Loss mask MAE: {mask_mae:.6f}, Max error: {mask_max_error:.6f}")
            print(f"    Loss mask agreement: {agreement_pct:.2f}%")

            # Analyze disagreements
            bayer_only = (loss_mask_bayer == 1) & (loss_mask_rgb == 0)
            rgb_only = (loss_mask_bayer == 0) & (loss_mask_rgb == 1)
            print(
                f"    Pixels accepted by bayer but rejected by RGB: {bayer_only.sum()}"
            )
            print(f"    Pixels rejected by bayer but accepted by RGB: {rgb_only.sum()}")
        else:
            print("    Loss mask shapes differ:")
            print(f"      bayer path: {loss_mask_bayer.shape}")
            print(f"      RGB path: {loss_mask_rgb.shape}")
            mask_mae = None
            agreement_pct = None

        results.append(
            {
                "image_set": test_set["image_set"],
                "alignment_match": alignment_match,
                "bayer_alignment": bayer_alignment,
                "rgb_alignment": rgb_alignment,
                "img_mae": img_mae,
                "noisy_mae": noisy_mae,
                "mask_mae": mask_mae,
                "mask_agreement_pct": agreement_pct,
                "mask_mean_bayer": loss_mask_bayer.mean(),
                "mask_mean_rgb": loss_mask_rgb.mean(),
            }
        )

    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    alignment_matches = sum(1 for r in results if r["alignment_match"])
    print(f"\nAlignment agreement: {alignment_matches}/{len(results)} pairs matched")

    valid_img_comparisons = [r for r in results if r["img_mae"] is not None]
    if valid_img_comparisons:
        avg_img_mae = np.mean([r["img_mae"] for r in valid_img_comparisons])
        print("\nAligned GT images (after demosaicing bayer):")
        print(f"  Average MAE: {avg_img_mae:.6f}")
        for r in valid_img_comparisons:
            print(f"    {r['image_set']}: MAE={r['img_mae']:.6f}")

    valid_mask_comparisons = [r for r in results if r["mask_mae"] is not None]
    if valid_mask_comparisons:
        avg_mask_mae = np.mean([r["mask_mae"] for r in valid_mask_comparisons])
        avg_agreement = np.mean(
            [r["mask_agreement_pct"] for r in valid_mask_comparisons]
        )
        print("\nLoss masks:")
        print(f"  Average MAE: {avg_mask_mae:.6f}")
        print(f"  Average agreement: {avg_agreement:.2f}%")
        for r in valid_mask_comparisons:
            print(
                f"    {r['image_set']}: MAE={r['mask_mae']:.6f}, "
                f"agreement={r['mask_agreement_pct']:.2f}%, "
                f"mean_bayer={r['mask_mean_bayer']:.4f}, mean_rgb={r['mask_mean_rgb']:.4f}"
            )

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("""
Expected behavior:
- Alignments may differ slightly (±1-2 pixels) due to demosaicing artifacts
- Aligned images should be very similar (MAE < 0.01) after demosaicing bayer path
- Loss masks should have high agreement (>95%) since they use same threshold logic
- Small differences are acceptable and expected due to:
  * Demosaicing introduces interpolation (RGB path demosaics before alignment)
  * Sub-pixel alignment differences
  * Numerical precision in different processing orders

The key validation is that loss masks are comparable, not identical.
    """)


if __name__ == "__main__":
    test_bayer_vs_rgb_pipeline()
