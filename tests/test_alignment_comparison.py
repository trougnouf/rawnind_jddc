"""Test to compare FFT-CFA vs Bruteforce-RGB alignment methods."""

import sys
import numpy as np

sys.path.insert(0, 'src')
from rawnind.libs import raw, alignment_backends


def compute_mae(img1, img2):
    """Compute Mean Absolute Error between two images."""
    return np.mean(np.abs(img1.astype(np.float64) - img2.astype(np.float64)))


def compute_mse(img1, img2):
    """Compute Mean Squared Error between two images."""
    diff = img1.astype(np.float64) - img2.astype(np.float64)
    return np.mean(diff ** 2)


def shift_images(anchor, target, shift):
    """Apply shift to align images, cropping to valid overlap region.
    
    Handles both (C, H, W) and (H, W) formats.
    """
    sy, sx = shift
    
    # Detect format: if 3D and first dim is small (1-4), it's (C, H, W)
    if anchor.ndim == 3 and anchor.shape[0] <= 4:
        # Format: (C, H, W)
        c, h, w = anchor.shape
        
        # Calculate valid overlap region
        y1_anchor = max(0, sy)
        y2_anchor = min(h, h + sy)
        x1_anchor = max(0, sx)
        x2_anchor = min(w, w + sx)
        
        y1_target = max(0, -sy)
        y2_target = min(h, h - sy)
        x1_target = max(0, -sx)
        x2_target = min(w, w - sx)
        
        return (anchor[:, y1_anchor:y2_anchor, x1_anchor:x2_anchor],
                target[:, y1_target:y2_target, x1_target:x2_target])
    else:
        # Format: (H, W) or (H, W, C)
        h, w = anchor.shape[:2]
        
        # Calculate valid overlap region
        y1_anchor = max(0, sy)
        y2_anchor = min(h, h + sy)
        x1_anchor = max(0, sx)
        x2_anchor = min(w, w + sx)
        
        y1_target = max(0, -sy)
        y2_target = min(h, h - sy)
        x1_target = max(0, -sx)
        x2_target = min(w, w - sx)
        
        if anchor.ndim == 3:
            return (anchor[y1_anchor:y2_anchor, x1_anchor:x2_anchor, :],
                    target[y1_target:y2_target, x1_target:x2_target, :])
        else:
            return (anchor[y1_anchor:y2_anchor, x1_anchor:x2_anchor],
                    target[y1_target:y2_target, x1_target:x2_target])


def test_alignment_comparison():
    """Compare FFT-CFA (raw) vs Bruteforce-RGB (demosaic first) alignment."""
    
    print("\n" + "="*80)
    print("ALIGNMENT METHOD COMPARISON: FFT-CFA vs Bruteforce-RGB")
    print("="*80)
    print("\nFFT-CFA: Fast phase correlation on raw Bayer/X-Trans data")
    print("Bruteforce-RGB: Slow exhaustive search after demosaicing to RGB\n")
    
    test_cases = [
        {
            "name": "Bark ISO65535 (Bayer)",
            "gt": "src/rawnind/datasets/RawNIND/src/Bayer/Bark/gt/Bayer_Bark_GT_ISO100_sha1=f15da1140d949ee30c15ce7b251839a7b7a41de7.cr2",
            "noisy": "src/rawnind/datasets/RawNIND/src/Bayer/Bark/Bayer_Bark_ISO65535_sha1=6ba8ed5f7fff42c4c900812c02701649f4f2d49e.cr2",
        },
        {
            "name": "Bark ISO800 (Bayer)",
            "gt": "src/rawnind/datasets/RawNIND/src/Bayer/Bark/gt/Bayer_Bark_GT_ISO100_sha1=f15da1140d949ee30c15ce7b251839a7b7a41de7.cr2",
            "noisy": "src/rawnind/datasets/RawNIND/src/Bayer/Bark/Bayer_Bark_ISO800_sha1=ba86f1da64a4bb534d9216e96c1c72177ed1e625.cr2",
        },
    ]
    
    results = []
    
    for test in test_cases:
        print(f"\n{test['name']}")
        print("-" * 80)
        
        # Load RAW images
        gt_raw, gt_meta = raw.raw_fpath_to_mono_img_and_metadata(test["gt"], return_float=True)
        noisy_raw, noisy_meta = raw.raw_fpath_to_mono_img_and_metadata(test["noisy"], return_float=True)
        
        # ===== METHOD 1: FFT-CFA on RAW data =====
        print("\n  [1] FFT-CFA (operates on raw Bayer data)")
        
        shift_fft = alignment_backends.find_best_alignment_fft_cfa(
            gt_raw, noisy_raw, gt_meta, verbose=False
        )
        print(f"      Detected shift: {shift_fft}")
        
        # Compute MAE/MSE on aligned RAW data
        gt_raw_aligned, noisy_raw_aligned = shift_images(gt_raw, noisy_raw, shift_fft)
        mae_fft_raw = compute_mae(gt_raw_aligned, noisy_raw_aligned)
        mse_fft_raw = compute_mse(gt_raw_aligned, noisy_raw_aligned)
        
        print(f"      MAE (RAW): {mae_fft_raw:.6f}")
        print(f"      MSE (RAW): {mse_fft_raw:.6f}")
        
        # Also compute in RGB domain (demosaic the aligned raw images)
        gt_rgb_fft = raw.demosaic(gt_raw_aligned, gt_meta)
        noisy_rgb_fft = raw.demosaic(noisy_raw_aligned, noisy_meta)
        mae_fft_rgb = compute_mae(gt_rgb_fft, noisy_rgb_fft)
        mse_fft_rgb = compute_mse(gt_rgb_fft, noisy_rgb_fft)
        
        print(f"      MAE (RGB): {mae_fft_rgb:.6f}")
        print(f"      MSE (RGB): {mse_fft_rgb:.6f}")
        
        # ===== METHOD 2: Bruteforce-RGB on demosaiced data =====
        print("\n  [2] Bruteforce-RGB (demosaics first, then searches)")
        
        # Demosaic both images
        gt_rgb = raw.demosaic(gt_raw, gt_meta)
        noisy_rgb = raw.demosaic(noisy_raw, noisy_meta)
        
        print(f"      Demosaiced: {gt_raw.shape} -> {gt_rgb.shape}")
        
        # Find alignment using bruteforce on RGB
        shift_bruteforce = alignment_backends.find_best_alignment_bruteforce_rgb(
            gt_rgb, noisy_rgb, verbose=False
        )
        print(f"      Detected shift: {shift_bruteforce}")
        
        # Compute MAE/MSE on aligned RGB data
        gt_rgb_aligned, noisy_rgb_aligned = shift_images(gt_rgb, noisy_rgb, shift_bruteforce)
        mae_bruteforce_rgb = compute_mae(gt_rgb_aligned, noisy_rgb_aligned)
        mse_bruteforce_rgb = compute_mse(gt_rgb_aligned, noisy_rgb_aligned)
        
        print(f"      MAE (RGB): {mae_bruteforce_rgb:.6f}")
        print(f"      MSE (RGB): {mse_bruteforce_rgb:.6f}")
        
        # Also compute in RAW domain (apply shift to original raw images)
        gt_raw_bruteforce_aligned, noisy_raw_bruteforce_aligned = shift_images(gt_raw, noisy_raw, shift_bruteforce)
        mae_bruteforce_raw = compute_mae(gt_raw_bruteforce_aligned, noisy_raw_bruteforce_aligned)
        mse_bruteforce_raw = compute_mse(gt_raw_bruteforce_aligned, noisy_raw_bruteforce_aligned)
        
        print(f"      MAE (RAW): {mae_bruteforce_raw:.6f}")
        print(f"      MSE (RAW): {mse_bruteforce_raw:.6f}")
        
        # ===== COMPARISON =====
        if shift_fft == shift_bruteforce:
            print(f"\n  ✅ SHIFTS MATCH: {shift_fft}")
        else:
            print(f"\n  ⚠️  SHIFTS DIFFER:")
            print(f"      FFT-CFA:       {shift_fft}")
            print(f"      Bruteforce-RGB: {shift_bruteforce}")
            print(f"      Difference:     ({shift_fft[0]-shift_bruteforce[0]}, {shift_fft[1]-shift_bruteforce[1]})")
        
        results.append({
            'name': test['name'],
            'fft_shift': shift_fft,
            'bruteforce_shift': shift_bruteforce,
            'fft_mae_raw': mae_fft_raw,
            'fft_mse_raw': mse_fft_raw,
            'fft_mae_rgb': mae_fft_rgb,
            'fft_mse_rgb': mse_fft_rgb,
            'bruteforce_mae_raw': mae_bruteforce_raw,
            'bruteforce_mse_raw': mse_bruteforce_raw,
            'bruteforce_mae_rgb': mae_bruteforce_rgb,
            'bruteforce_mse_rgb': mse_bruteforce_rgb,
        })
    
    # Print summary
    print("\n" + "="*120)
    print("SUMMARY TABLE")
    print("="*120)
    print(f"{'Test':<30} {'Method':<15} {'Shift':<12} {'MAE(RAW)':<12} {'MSE(RAW)':<12} {'MAE(RGB)':<12} {'MSE(RGB)':<12}")
    print("-" * 120)
    
    for r in results:
        print(f"{r['name']:<30} {'FFT-CFA':<15} {str(r['fft_shift']):<12} "
              f"{r['fft_mae_raw']:<12.6f} {r['fft_mse_raw']:<12.6f} "
              f"{r['fft_mae_rgb']:<12.6f} {r['fft_mse_rgb']:<12.6f}")
        print(f"{'':<30} {'Bruteforce-RGB':<15} {str(r['bruteforce_shift']):<12} "
              f"{r['bruteforce_mae_raw']:<12.6f} {r['bruteforce_mse_raw']:<12.6f} "
              f"{r['bruteforce_mae_rgb']:<12.6f} {r['bruteforce_mse_rgb']:<12.6f}")
        print()


if __name__ == "__main__":
    test_alignment_comparison()
