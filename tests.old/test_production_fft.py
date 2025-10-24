#!/usr/bin/env python3
"""Test production CFA-aware FFT implementation on real RawNIND pairs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "DocScan"))

from rawnind.libs import raw
import numpy as np


def compute_mae(img1, img2):
    """Compute Mean Absolute Error."""
    return np.mean(np.abs(img1 - img2))


def compute_mse(img1, img2):
    """Compute Mean Squared Error."""
    return np.mean((img1 - img2) ** 2)


def shift_images(anchor, target, shift):
    """Shift images to common aligned region."""
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


# Test cases from curated dataset
TEST_CASES = [
    # Bayer - known misalignments
    {
        "name": "Bark ISO65535 (bayer)",
        "anchor": "DocScan/rawnind/datasets/RawNIND/DocScan/bayer/Bark/gt/Bayer_Bark_GT_ISO100_sha1=f15da1140d949ee30c15ce7b251839a7b7a41de7.cr2",
        "target": "DocScan/rawnind/datasets/RawNIND/DocScan/bayer/Bark/Bayer_Bark_ISO65535_sha1=6ba8ed5f7fff42c4c900812c02701649f4f2d49e.cr2",
        "expected": (12, -10),
    },
    {
        "name": "Bark ISO800 (bayer)",
        "anchor": "DocScan/rawnind/datasets/RawNIND/DocScan/bayer/Bark/gt/Bayer_Bark_GT_ISO100_sha1=f15da1140d949ee30c15ce7b251839a7b7a41de7.cr2",
        "target": "DocScan/rawnind/datasets/RawNIND/DocScan/bayer/Bark/Bayer_Bark_ISO800_sha1=ba86f1da64a4bb534d9216e96c1c72177ed1e625.cr2",
        "expected": (4, -4),
    },
    {
        "name": "Kortlek ISO51200 (bayer)",
        "anchor": "DocScan/rawnind/datasets/RawNIND/DocScan/bayer/Kortlek/gt/Bayer_Kortlek_GT_ISO100_sha1=b5a564cd9291224caf363b6b03054365d59d316b.cr2",
        "target": "DocScan/rawnind/datasets/RawNIND/DocScan/bayer/Kortlek/Bayer_Kortlek_ISO51200_sha1=8fed453fdfb162673bb9ed41f9c5f03095331e3b.cr2",
        "expected": (0, -4),
    },
    # X-Trans - already aligned
    {
        "name": "MuseeL-pedestal ISO6400 (x-trans)",
        "anchor": "DocScan/rawnind/datasets/RawNIND/DocScan/x-trans/MuseeL-pedestal/gt/X-Trans_MuseeL-pedestal_GT_ISO200_sha1=161b21f545c4c4ed7fc4fce014f637bb7040d8aa.raf",
        "target": "DocScan/rawnind/datasets/RawNIND/DocScan/x-trans/MuseeL-pedestal/X-Trans_MuseeL-pedestal_ISO6400_sha1=693af8d1f36f89ad3c4cb8c0eb0e41c924684140.raf",
        "expected": (0, 0),
    },
]


def test_production_implementation():
    """Test production FFT function on real RawNIND pairs."""
    print("=" * 80)
    print("TESTING PRODUCTION CFA-AWARE FFT IMPLEMENTATION")
    print("=" * 80)

    passed = 0
    failed = 0

    for test_case in TEST_CASES:
        print(f"\n{test_case['name']}")
        print(f"  Expected: {test_case['expected']}")

        # Load images
        anchor, anchor_meta = raw.raw_fpath_to_mono_img_and_metadata(
            test_case["anchor"], return_float=True
        )
        target, target_meta = raw.raw_fpath_to_mono_img_and_metadata(
            test_case["target"], return_float=True
        )

        # Apply production FFT function
        shift, channel_shifts = raw.fft_phase_correlate_cfa(
            anchor, target, anchor_meta, method="median", verbose=False
        )

        print(f"  Detected: {shift}")
        print(f"  Channels: {channel_shifts}")

        # Compute alignment error metrics
        anchor_aligned, target_aligned = shift_images(anchor, target, shift)
        mae = compute_mae(anchor_aligned, target_aligned)
        mse = compute_mse(anchor_aligned, target_aligned)

        print(f"  MAE after alignment: {mae:.6f}")
        print(f"  MSE after alignment: {mse:.6f}")

        # Check result
        if shift == test_case["expected"]:
            print("  ✅ PASS")
            passed += 1
        else:
            print("  ❌ FAIL")
            failed += 1

    print("\n" + "=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(TEST_CASES)} tests")
    print("=" * 80)

    return failed == 0


def test_alignment_quality_comparison():
    """Compare alignment quality (MAE/MSE) between FFT and bruteforce methods."""
    print("=" * 80)
    print("ALIGNMENT QUALITY COMPARISON: FFT vs BRUTEFORCE")
    print("=" * 80)

    # Import alignment backends
    sys.path.insert(0, "DocScan")
    from rawnind.libs import alignment_backends

    comparison_results = []

    for test_case in TEST_CASES:
        print(f"\n{test_case['name']}")
        print("-" * 80)

        # Load images
        anchor, anchor_meta = raw.raw_fpath_to_mono_img_and_metadata(
            test_case["anchor"], return_float=True
        )
        target, target_meta = raw.raw_fpath_to_mono_img_and_metadata(
            test_case["target"], return_float=True
        )

        case_result = {"name": test_case["name"], "methods": {}}

        # Test both alignment methods
        for method_name in ["fft", "original"]:
            print(f"\n  Method: {method_name}")

            # Call the appropriate alignment function
            if method_name == "fft":
                shift = alignment_backends.find_best_alignment_fft_cfa(
                    anchor, target, anchor_meta, verbose=False
                )
            else:  # original/bruteforce
                shift = alignment_backends.find_best_alignment_cpu(
                    anchor, target, anchor_meta, verbose=False
                )

            print(f"    Detected shift: {shift}")

            # Compute alignment error metrics
            anchor_aligned, target_aligned = shift_images(anchor, target, shift)
            mae = compute_mae(anchor_aligned, target_aligned)
            mse = compute_mse(anchor_aligned, target_aligned)

            print(f"    MAE: {mae:.6f}")
            print(f"    MSE: {mse:.6f}")

            # Check if shift matches expected
            correct = shift == test_case["expected"]
            print(f"    Expected: {test_case['expected']} {'✅' if correct else '❌'}")

            case_result["methods"][method_name] = {
                "shift": shift,
                "mae": mae,
                "mse": mse,
                "correct": correct,
            }

        comparison_results.append(case_result)

        # Show comparison
        fft_mae = case_result["methods"]["fft"]["mae"]
        orig_mae = case_result["methods"]["original"]["mae"]
        fft_mse = case_result["methods"]["fft"]["mse"]
        orig_mse = case_result["methods"]["original"]["mse"]

        print("\n  Comparison:")
        print(
            f"    MAE - FFT: {fft_mae:.6f}, Original: {orig_mae:.6f} (Δ: {abs(fft_mae - orig_mae):.6f})"
        )
        print(
            f"    MSE - FFT: {fft_mse:.6f}, Original: {orig_mse:.6f} (Δ: {abs(fft_mse - orig_mse):.6f})"
        )

        if fft_mae < orig_mae:
            print("    ✅ FFT produces better alignment (lower MAE)")
        elif fft_mae > orig_mae:
            print("    ⚠️  Original produces better alignment (lower MAE)")
        else:
            print("    ➡️  Both methods produce same MAE")

    # Summary table
    print("\n\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Test Case':<40} {'Method':<10} {'MAE':<12} {'MSE':<12} {'Correct'}")
    print("-" * 80)

    for result in comparison_results:
        name = result["name"][:38]
        for method in ["fft", "original"]:
            m = result["methods"][method]
            correct = "✅" if m["correct"] else "❌"
            print(
                f"{name:<40} {method:<10} {m['mae']:<12.6f} {m['mse']:<12.6f} {correct}"
            )

    print("=" * 80)

    return comparison_results


if __name__ == "__main__":
    # Run original test
    print("\n\n")
    success = test_production_implementation()

    # Run comparison test
    print("\n\n")
    comparison_results = test_alignment_quality_comparison()

    sys.exit(0 if success else 1)
