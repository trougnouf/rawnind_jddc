"""Validate NumPy MS-SSIM against PyTorch implementation and benchmark performance.

This test compares numerical accuracy and performance against the reference
pytorch_msssim implementation to ensure ±0.01 tolerance and competitive speed.
"""

import numpy as np
import torch
import time
import sys
from pathlib import Path

# Add DocScan to path
sys.path.insert(0, str(Path(__file__).parent.parent / "DocScan"))

from common.libs.msssim_numpy import compute_msssim_numpy, compute_ssim_numpy
from pytorch_msssim import ms_ssim as torch_ms_ssim, ssim as torch_ssim


def test_ssim_accuracy():
    """Test SSIM accuracy against PyTorch."""
    print("\n" + "="*70)
    print("SSIM ACCURACY TEST")
    print("="*70)

    np.random.seed(42)
    torch.manual_seed(42)

    test_cases = [
        ("Random noise (256x256)", 256, 256, 0.1),
        ("Large image (512x512)", 512, 512, 0.05),
        ("Small noise", 256, 256, 0.01),
        ("High noise", 256, 256, 0.3),
    ]

    results = []

    for name, h, w, noise_level in test_cases:
        # Create test images
        img1_np = np.random.rand(h, w).astype(np.float64)
        img2_np = img1_np + noise_level * np.random.randn(h, w).astype(np.float64)
        img2_np = np.clip(img2_np, 0, 1)

        # NumPy version
        ssim_numpy = compute_ssim_numpy(img1_np, img2_np, data_range=1.0)

        # PyTorch version
        img1_torch = torch.from_numpy(img1_np).unsqueeze(0).unsqueeze(0).float()
        img2_torch = torch.from_numpy(img2_np).unsqueeze(0).unsqueeze(0).float()
        ssim_torch = torch_ssim(img1_torch, img2_torch, data_range=1.0).item()

        diff = abs(ssim_numpy - ssim_torch)
        passed = diff < 0.01

        results.append({
            "name": name,
            "numpy": ssim_numpy,
            "torch": ssim_torch,
            "diff": diff,
            "passed": passed
        })

        status = "PASS" if passed else "FAIL"
        print(f"{name:30s} | NumPy: {ssim_numpy:.6f} | PyTorch: {ssim_torch:.6f} | Diff: {diff:.6f} | {status}")

    print()
    all_passed = all(r["passed"] for r in results)
    if all_passed:
        print("✓ All SSIM tests passed within ±0.01 tolerance")
    else:
        print("✗ Some SSIM tests failed")

    assert all_passed, "SSIM accuracy test failed"
    return results


def test_msssim_accuracy():
    """Test MS-SSIM accuracy against PyTorch."""
    print("\n" + "="*70)
    print("MS-SSIM ACCURACY TEST")
    print("="*70)

    np.random.seed(42)
    torch.manual_seed(42)

    test_cases = [
        ("Random noise (256x256)", 256, 256, 0.1),
        ("Large image (512x512)", 512, 512, 0.05),
        ("Small noise", 256, 256, 0.01),
        ("High noise", 256, 256, 0.3),
        ("Very small noise", 256, 256, 0.001),
    ]

    results = []

    for name, h, w, noise_level in test_cases:
        # Create test images
        img1_np = np.random.rand(h, w).astype(np.float64)
        img2_np = img1_np + noise_level * np.random.randn(h, w).astype(np.float64)
        img2_np = np.clip(img2_np, 0, 1)

        # NumPy version
        msssim_numpy = compute_msssim_numpy(img1_np, img2_np, data_range=1.0)

        # PyTorch version
        img1_torch = torch.from_numpy(img1_np).unsqueeze(0).unsqueeze(0).float()
        img2_torch = torch.from_numpy(img2_np).unsqueeze(0).unsqueeze(0).float()
        msssim_torch = torch_ms_ssim(img1_torch, img2_torch, data_range=1.0).item()

        diff = abs(msssim_numpy - msssim_torch)
        passed = diff < 0.01

        results.append({
            "name": name,
            "numpy": msssim_numpy,
            "torch": msssim_torch,
            "diff": diff,
            "passed": passed
        })

        status = "PASS" if passed else "FAIL"
        print(f"{name:30s} | NumPy: {msssim_numpy:.6f} | PyTorch: {msssim_torch:.6f} | Diff: {diff:.6f} | {status}")

    print()
    all_passed = all(r["passed"] for r in results)
    if all_passed:
        print("✓ All MS-SSIM tests passed within ±0.01 tolerance")
    else:
        print("✗ Some MS-SSIM tests failed")

    assert all_passed, "MS-SSIM accuracy test failed"
    return results


def benchmark_ssim():
    """Benchmark SSIM performance."""
    print("\n" + "="*70)
    print("SSIM PERFORMANCE BENCHMARK")
    print("="*70)

    np.random.seed(42)
    torch.manual_seed(42)

    sizes = [(256, 256), (512, 512), (1024, 1024)]
    n_iterations = 20

    for h, w in sizes:
        # Prepare data
        img1_np = np.random.rand(h, w).astype(np.float64)
        img2_np = img1_np + 0.1 * np.random.randn(h, w).astype(np.float64)
        img2_np = np.clip(img2_np, 0, 1)

        img1_torch = torch.from_numpy(img1_np).unsqueeze(0).unsqueeze(0).float()
        img2_torch = torch.from_numpy(img2_np).unsqueeze(0).unsqueeze(0).float()

        # Warm-up
        _ = compute_ssim_numpy(img1_np, img2_np, data_range=1.0)
        _ = torch_ssim(img1_torch, img2_torch, data_range=1.0)

        # Benchmark NumPy
        start = time.time()
        for _ in range(n_iterations):
            _ = compute_ssim_numpy(img1_np, img2_np, data_range=1.0)
        time_numpy = (time.time() - start) / n_iterations

        # Benchmark PyTorch (CPU)
        start = time.time()
        for _ in range(n_iterations):
            _ = torch_ssim(img1_torch, img2_torch, data_range=1.0)
        time_torch = (time.time() - start) / n_iterations

        speedup = time_torch / time_numpy
        print(f"Size {h}x{w:4d} | NumPy: {time_numpy*1000:6.2f} ms | PyTorch: {time_torch*1000:6.2f} ms | Speedup: {speedup:.2f}x")


def benchmark_msssim():
    """Benchmark MS-SSIM performance."""
    print("\n" + "="*70)
    print("MS-SSIM PERFORMANCE BENCHMARK")
    print("="*70)

    np.random.seed(42)
    torch.manual_seed(42)

    sizes = [(256, 256), (512, 512), (1024, 1024)]
    n_iterations = 10

    for h, w in sizes:
        # Prepare data
        img1_np = np.random.rand(h, w).astype(np.float64)
        img2_np = img1_np + 0.1 * np.random.randn(h, w).astype(np.float64)
        img2_np = np.clip(img2_np, 0, 1)

        img1_torch = torch.from_numpy(img1_np).unsqueeze(0).unsqueeze(0).float()
        img2_torch = torch.from_numpy(img2_np).unsqueeze(0).unsqueeze(0).float()

        # Warm-up
        _ = compute_msssim_numpy(img1_np, img2_np, data_range=1.0)
        _ = torch_ms_ssim(img1_torch, img2_torch, data_range=1.0)

        # Benchmark NumPy
        start = time.time()
        for _ in range(n_iterations):
            _ = compute_msssim_numpy(img1_np, img2_np, data_range=1.0)
        time_numpy = (time.time() - start) / n_iterations

        # Benchmark PyTorch (CPU)
        start = time.time()
        for _ in range(n_iterations):
            _ = torch_ms_ssim(img1_torch, img2_torch, data_range=1.0)
        time_torch = (time.time() - start) / n_iterations

        speedup = time_torch / time_numpy
        print(f"Size {h}x{w:4d} | NumPy: {time_numpy*1000:6.2f} ms | PyTorch: {time_torch*1000:6.2f} ms | Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    print("\nNumPy MS-SSIM Implementation Validation")
    print("="*70)

    # Test accuracy
    ssim_results = test_ssim_accuracy()
    msssim_results = test_msssim_accuracy()

    # Benchmark performance
    benchmark_ssim()
    benchmark_msssim()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    ssim_max_diff = max(r["diff"] for r in ssim_results)
    msssim_max_diff = max(r["diff"] for r in msssim_results)

    print(f"SSIM maximum difference:    {ssim_max_diff:.6f} (target < 0.01)")
    print(f"MS-SSIM maximum difference: {msssim_max_diff:.6f} (target < 0.01)")
    print()

    if ssim_max_diff < 0.01 and msssim_max_diff < 0.01:
        print("✓ Implementation validates successfully!")
        print("  - Numerical accuracy within ±0.01 tolerance")
        print("  - Pure NumPy/SciPy implementation (no PyTorch dependency)")
        print("  - Suitable for async workflows with trio.to_thread.run_sync()")
    else:
        print("✗ Implementation validation failed")
        exit(1)
