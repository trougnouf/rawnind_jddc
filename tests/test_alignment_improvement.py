"""Test #2: Alignment Improvement

Verify that applying the detected shift actually reduces error compared to
unaligned images (MAE/MSE before alignment > MAE/MSE after alignment).
"""

import sys
import pytest
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, "src")
from rawnind.libs import alignment_backends, raw


def compute_mae_mse(img1: np.ndarray, img2: np.ndarray) -> tuple[float, float]:
    """Compute MAE and MSE between two images using PyTorch.

    Args:
        img1: First image (numpy array)
        img2: Second image (numpy array)

    Returns:
        Tuple of (MAE, MSE)
    """
    tensor1 = torch.from_numpy(img1).float()
    tensor2 = torch.from_numpy(img2).float()

    mae = F.l1_loss(tensor1, tensor2).item()
    mse = F.mse_loss(tensor1, tensor2).item()

    return mae, mse


def test_fft_cfa_improves_alignment(random_scenes_with_synthetic_shifts):
    """Test that FFT-CFA detected shift reduces alignment error.
    
    Tests until at least one scene of each CFA type (Bayer, X-Trans) passes.
    """
    scenes = random_scenes_with_synthetic_shifts

    print(f"\n{'=' * 80}")
    print(f"Testing FFT-CFA Alignment Improvement on {len(scenes)} scenes")
    print(f"{'=' * 80}\n")

    bayer_passed = False
    xtrans_passed = False
    failures = []

    for i, scene in enumerate(scenes, 1):
        cfa_type = scene["cfa_type"]
        
        # Skip if we've already validated this CFA type
        if (cfa_type == "Bayer" and bayer_passed) or (cfa_type == "X-Trans" and xtrans_passed):
            print(f"[{i:2d}] {scene['subject']:<25} ({cfa_type:<7}) | Skipped (already validated {cfa_type})")
            continue
        
        original = scene["original_img"]
        shifted = scene["shifted_img"]
        true_shift = scene["true_shift"]
        metadata = scene["metadata"]
        subject = scene["subject"]
        cfa_type = scene["cfa_type"]

        # Step 1: Compute initial MAE (before detection/alignment)
        mae_initial, _ = compute_mae_mse(original, shifted)
        print(
            f"[{i:2d}] {subject:<25} ({cfa_type:<7}) | Initial MAE: {mae_initial:.6f}"
        )

        # Step 2: Detect shift using FFT-CFA
        detected_shift = alignment_backends.find_best_alignment_fft_cfa(
            original, shifted, metadata, verbose=False
        )
        print(f"     └─ Detected shift: {detected_shift} (true: {true_shift})")

        # Step 3: Apply detected shift to align images
        original_aligned, shifted_aligned = alignment_backends.shift_images(
            original, shifted, detected_shift
        )

        # Step 4: Compute MAE after alignment
        mae_after, _ = compute_mae_mse(original_aligned, shifted_aligned)
        print(f"     └─ After alignment: {mae_after:.6f}")

        # Check if alignment improved (reduced error)
        mae_improved = mae_after < mae_initial

        status = "✓" if mae_improved else "✗"
        improvement_pct = (
            ((mae_initial - mae_after) / mae_initial * 100) if mae_initial > 0 else 0
        )
        print(f"     └─ {status} Improvement: {improvement_pct:+.1f}%\n")

        if mae_improved:
            if cfa_type == "Bayer":
                bayer_passed = True
            else:
                xtrans_passed = True
        else:
            failures.append(
                {
                    "scene": subject,
                    "cfa_type": cfa_type,
                    "detected_shift": detected_shift,
                    "true_shift": true_shift,
                    "mae_initial": mae_initial,
                    "mae_after": mae_after,
                }
            )

    print(f"{'=' * 80}")
    if bayer_passed and xtrans_passed:
        print("PASSED: Successfully validated both Bayer and X-Trans scenes")
    elif bayer_passed:
        print("PARTIAL: Validated Bayer but not X-Trans")
    elif xtrans_passed:
        print("PARTIAL: Validated X-Trans but not Bayer")
    else:
        print("FAILED: No scenes passed alignment improvement test")

    if failures:
        print("\nFailures (alignment did not improve error):")
        for f in failures:
            print(
                f"  - {f['scene']} ({f['cfa_type']}): "
                f"Detected {f['detected_shift']} vs True {f['true_shift']}"
            )
            print(f"    MAE: {f['mae_initial']:.6f} → {f['mae_after']:.6f} (WORSE)")

    print(f"{'=' * 80}\n")

    assert bayer_passed and xtrans_passed, (
        f"Failed to validate both CFA types (Bayer: {bayer_passed}, X-Trans: {xtrans_passed})"
    )


def test_bruteforce_rgb_improves_alignment(random_scenes_with_synthetic_shifts):
    """Test that Bruteforce-RGB detected shift reduces alignment error.
    
    Note: Bruteforce-RGB only works on Bayer images (requires RGB demosaic).
    X-Trans images are skipped as there's no X-Trans RGB demosaic in the codebase.
    Tests until at least one Bayer scene passes, then skips remaining scenes.
    """
    scenes = random_scenes_with_synthetic_shifts
    
    # Filter to Bayer only - bruteforce RGB requires RGB demosaic which only exists for Bayer
    bayer_scenes = [s for s in scenes if s["cfa_type"] == "Bayer"]
    
    if len(bayer_scenes) == 0:
        pytest.skip("No Bayer scenes available for bruteforce RGB test")

    print(f"\n{'=' * 80}")
    print(f"Testing Bruteforce-RGB Alignment Improvement")
    print(f"(Testing until one Bayer scene passes)")
    print(f"{'=' * 80}\n")

    failures = []
    bayer_passed = False

    for i, scene in enumerate(bayer_scenes, 1):
        if bayer_passed:
            print(f"[{i:2d}] {scene['subject']:<25} (Bayer  ) | Skipped (already validated Bayer)")
            continue
        original = scene["original_img"]
        shifted = scene["shifted_img"]
        true_shift = scene["true_shift"]
        metadata = scene["metadata"]
        subject = scene["subject"]
        cfa_type = scene["cfa_type"]

        # Demosaic both images to RGB
        original_rgb = raw.demosaic(original, metadata)
        shifted_rgb = raw.demosaic(shifted, metadata)

        # Step 1: Compute initial MAE (before alignment)
        mae_initial, _ = compute_mae_mse(original_rgb, shifted_rgb)
        print(
            f"[{i:2d}] {subject:<25} ({cfa_type:<7}) | Initial MAE: {mae_initial:.6f}"
        )

        # Step 2: Detect shift using Bruteforce-RGB
        detected_shift = alignment_backends.find_best_alignment_bruteforce_rgb(
            original_rgb, shifted_rgb, verbose=False
        )
        print(f"     └─ Detected shift: {detected_shift} (true: {true_shift})")

        # Step 3: Apply detected shift to align images
        original_aligned, shifted_aligned = alignment_backends.shift_images(
            original_rgb, shifted_rgb, detected_shift
        )

        # Step 4: Compute MAE after alignment
        mae_after, _ = compute_mae_mse(original_aligned, shifted_aligned)
        print(f"     └─ After alignment: {mae_after:.6f}")

        # Check if alignment improved (reduced error)
        mae_improved = mae_after < mae_initial

        status = "✓" if mae_improved else "✗"
        improvement_pct = (
            ((mae_initial - mae_after) / mae_initial * 100) if mae_initial > 0 else 0
        )
        print(f"     └─ {status} Improvement: {improvement_pct:+.1f}%\n")

        if mae_improved:
            bayer_passed = True
        else:
            failures.append(
                {
                    "scene": subject,
                    "cfa_type": cfa_type,
                    "detected_shift": detected_shift,
                    "true_shift": true_shift,
                    "mae_initial": mae_initial,
                    "mae_after": mae_after,
                }
            )

    print(f"{'=' * 80}")
    if bayer_passed:
        print(f"PASSED: Successfully validated at least one Bayer scene")
    else:
        print(f"FAILED: No Bayer scenes passed alignment improvement test")

    if failures:
        print("\nFailures (alignment did not improve error):")
        for f in failures:
            print(
                f"  - {f['scene']} ({f['cfa_type']}): "
                f"Detected {f['detected_shift']} vs True {f['true_shift']}"
            )
            print(f"    MAE: {f['mae_initial']:.6f} → {f['mae_after']:.6f} (WORSE)")

    print(f"{'=' * 80}\n")

    assert bayer_passed, "Failed to validate alignment improvement on any Bayer scene"
