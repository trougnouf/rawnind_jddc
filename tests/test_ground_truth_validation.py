"""Test #1: Ground Truth Validation

Test on synthetically shifted images where the true shift is known,
verifying both methods detect the correct shift.
"""

import sys

sys.path.insert(0, "DocScan")
from rawnind.libs import alignment_backends


def test_fft_cfa_detects_known_shifts(random_scenes_with_synthetic_shifts):
    """Test that FFT-CFA correctly detects known synthetic shifts."""
    scenes = random_scenes_with_synthetic_shifts

    print(f"\n{'=' * 80}")
    print(f"Testing FFT-CFA on {len(scenes)} random scenes with synthetic shifts")
    print(f"{'=' * 80}\n")

    failures = []

    for i, scene in enumerate(scenes, 1):
        original = scene["original_img"]
        shifted = scene["shifted_img"]
        true_shift = scene["true_shift"]
        metadata = scene["metadata"]
        subject = scene["subject"]
        cfa_type = scene["cfa_type"]

        # Detect shift using FFT-CFA
        detected_shift = alignment_backends.find_best_alignment_fft_cfa(
            original, shifted, metadata, verbose=False
        )

        # Check if detected shift matches ground truth
        match = detected_shift == true_shift
        status = "✓" if match else "✗"

        print(
            f"[{i:2d}] {status} {subject:<25} ({cfa_type:<7}) | "
            f"True: {true_shift}, Detected: {detected_shift}"
        )

        if not match:
            failures.append(
                {
                    "scene": subject,
                    "cfa_type": cfa_type,
                    "true_shift": true_shift,
                    "detected_shift": detected_shift,
                    "error": (
                        detected_shift[0] - true_shift[0],
                        detected_shift[1] - true_shift[1],
                    ),
                }
            )

    print(f"\n{'=' * 80}")
    print(f"Results: {len(scenes) - len(failures)}/{len(scenes)} passed")

    if failures:
        print("\nFailures:")
        for f in failures:
            print(
                f"  - {f['scene']} ({f['cfa_type']}): "
                f"Expected {f['true_shift']}, got {f['detected_shift']} "
                f"(error: {f['error']})"
            )

    print(f"{'=' * 80}\n")

    assert len(failures) == 0, f"{len(failures)} scenes failed to detect correct shift"


def test_bruteforce_rgb_detects_known_shifts(random_scenes_with_synthetic_shifts):
    """Test that Bruteforce-RGB correctly detects known synthetic shifts."""
    scenes = random_scenes_with_synthetic_shifts

    print(f"\n{'=' * 80}")
    print(
        f"Testing Bruteforce-RGB on {len(scenes)} random scenes with synthetic shifts"
    )
    print(f"{'=' * 80}\n")

    failures = []

    for i, scene in enumerate(scenes, 1):
        original = scene["original_img"]
        shifted = scene["shifted_img"]
        true_shift = scene["true_shift"]
        metadata = scene["metadata"]
        subject = scene["subject"]
        cfa_type = scene["cfa_type"]

        # Import raw module for demosaicing
        from rawnind.libs import raw

        # Demosaic both images to RGB
        original_rgb = raw.demosaic(original, metadata)
        shifted_rgb = raw.demosaic(shifted, metadata)

        # Detect shift using Bruteforce-RGB
        detected_shift = alignment_backends.find_best_alignment_bruteforce_rgb(
            original_rgb, shifted_rgb, verbose=False
        )

        # Check if detected shift matches ground truth
        match = detected_shift == true_shift
        status = "✓" if match else "✗"

        print(
            f"[{i:2d}] {status} {subject:<25} ({cfa_type:<7}) | "
            f"True: {true_shift}, Detected: {detected_shift}"
        )

        if not match:
            failures.append(
                {
                    "scene": subject,
                    "cfa_type": cfa_type,
                    "true_shift": true_shift,
                    "detected_shift": detected_shift,
                    "error": (
                        detected_shift[0] - true_shift[0],
                        detected_shift[1] - true_shift[1],
                    ),
                }
            )

    print(f"\n{'=' * 80}")
    print(f"Results: {len(scenes) - len(failures)}/{len(scenes)} passed")

    if failures:
        print("\nFailures:")
        for f in failures:
            print(
                f"  - {f['scene']} ({f['cfa_type']}): "
                f"Expected {f['true_shift']}, got {f['detected_shift']} "
                f"(error: {f['error']})"
            )

    print(f"{'=' * 80}\n")

    assert len(failures) == 0, f"{len(failures)} scenes failed to detect correct shift"
