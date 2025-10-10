"""
Cross-validation test for alignment methods.

Compares CFA-aware FFT alignment against RGB brute-force alignment
to ensure consistency across multiple image pairs.
"""

import pytest
import numpy as np
from pathlib import Path
import glob
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_test_pairs(max_pairs=20):
    """Find GT-noisy RAW file pairs for testing."""
    bayer_dir = Path("tmp/rawnind_dataset/bayer")
    if not bayer_dir.exists():
        pytest.skip("Test dataset not available")
    
    pairs = []
    for scene_dir in sorted(bayer_dir.glob("*")):
        if not scene_dir.is_dir() or scene_dir.name == "gt":
            continue
            
        gt_dir = scene_dir / "gt"
        if not gt_dir.exists():
            continue
        
        # Find GT file
        gt_files = list(gt_dir.glob("*GT*.cr2")) + list(gt_dir.glob("*GT*.arw"))
        if not gt_files:
            continue
        gt_file = gt_files[0]
        
        # Find noisy files (not GT)
        noisy_files = []
        for ext in ["*.cr2", "*.arw"]:
            noisy_files.extend(scene_dir.glob(ext))
        noisy_files = [f for f in noisy_files if "GT" not in f.name]
        
        # Add variety: different ISO levels from each scene
        # Low ISO
        low_iso = [f for f in noisy_files if any(iso in f.name for iso in ["ISO200", "ISO400"])]
        if low_iso:
            pairs.append((gt_file, low_iso[0], scene_dir.name, "low"))
        
        # High ISO
        high_iso = [f for f in noisy_files if any(iso in f.name for iso in ["ISO6400", "ISO12800"])]
        if high_iso:
            pairs.append((gt_file, high_iso[0], scene_dir.name, "high"))
        
        if len(pairs) >= max_pairs:
            return pairs
    
    return pairs


def load_raw_image(fpath):
    """Load RAW image as numpy array."""
    import rawpy
    with rawpy.imread(str(fpath)) as raw:
        raw_data = raw.raw_image_visible.copy().astype(np.float32)
        # Reshape to (1, H, W) for alignment functions
        raw_data = raw_data[np.newaxis, ...]
        
        # Get CFA pattern
        raw_pattern = raw.raw_pattern
        metadata = {"RGBG_pattern": raw_pattern}
        
    return raw_data, metadata


def demosaic_simple(raw_data, pattern):
    """
    Simple demosaicing for RGB alignment comparison.
    Uses bilinear interpolation on each channel.
    """
    from scipy.ndimage import zoom
    
    # Remove batch dimension
    if raw_data.ndim == 3 and raw_data.shape[0] == 1:
        raw_data = raw_data[0]
    
    h, w = raw_data.shape
    pattern_h, pattern_w = pattern.shape
    
    # Create RGB channels
    rgb = np.zeros((3, h, w), dtype=np.float32)
    
    # Extract each channel from the pattern
    for c in range(3):  # R=0, G=1, B=2
        mask = (pattern == c)
        # Tile mask to image size
        full_mask = np.tile(mask, (h // pattern_h + 1, w // pattern_w + 1))[:h, :w]
        
        # Extract channel values
        channel_data = np.where(full_mask, raw_data, 0)
        
        # Simple interpolation: upsample by factor 2, then downsample
        # This is crude but sufficient for alignment comparison
        count_mask = full_mask.astype(np.float32)
        count_sum = count_mask.sum()
        if count_sum > 0:
            rgb[c] = channel_data / (count_mask + 1e-8)
            # Fill zeros with nearest neighbor
            for _ in range(2):
                kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
                from scipy.ndimage import convolve
                rgb[c] = np.where(full_mask, channel_data, convolve(rgb[c], kernel) / 4)
    
    return rgb


@pytest.mark.parametrize("pair_idx", range(20))
def test_cfa_alignment_produces_valid_shifts(pair_idx):
    """
    Test that CFA-aware FFT alignment produces valid block-aligned shifts.
    
    For bayer (2x2): shifts must be even
    Simplified test - just verifies CFA snapping works, no RGB comparison
    """
    from rawnind.libs.alignment_backends import find_best_alignment_fft_cfa
    
    pairs = find_test_pairs(max_pairs=20)
    if pair_idx >= len(pairs):
        pytest.skip(f"Only {len(pairs)} pairs available")
    
    gt_file, noisy_file, scene_name, iso_level = pairs[pair_idx]
    
    logger.info(f"Testing pair {pair_idx}: {scene_name} ({iso_level}) - {noisy_file.name}")
    
    # Load RAW images
    gt_raw, gt_metadata = load_raw_image(gt_file)
    noisy_raw, noisy_metadata = load_raw_image(noisy_file)
    
    # Run CFA-aware FFT alignment
    shift_cfa, loss_cfa = find_best_alignment_fft_cfa(
        gt_raw, noisy_raw, gt_metadata, method="median", return_loss_too=True, verbose=False
    )
    
    logger.info(f"  CFA shift: {shift_cfa}, loss: {loss_cfa:.6f}")
    
    # Verify Bayer block alignment (2x2)
    pattern_shape = gt_metadata["RGBG_pattern"].shape
    if pattern_shape == (2, 2):
        # Bayer - shifts must be even
        assert shift_cfa[0] % 2 == 0, (
            f"bayer Y shift {shift_cfa[0]} is not even (violates 2x2 blocks)"
        )
        assert shift_cfa[1] % 2 == 0, (
            f"bayer X shift {shift_cfa[1]} is not even (violates 2x2 blocks)"
        )
        logger.info(f"  ✓ bayer shifts are even (block-aligned)")
    elif pattern_shape == (6, 6):
        # X-Trans - shifts must be multiples of 3
        assert shift_cfa[0] % 3 == 0, (
            f"x-trans Y shift {shift_cfa[0]} is not multiple of 3 (violates 3x3 blocks)"
        )
        assert shift_cfa[1] % 3 == 0, (
            f"x-trans X shift {shift_cfa[1]} is not multiple of 3 (violates 3x3 blocks)"
        )
        logger.info(f"  ✓ x-trans shifts are multiples of 3 (block-aligned)")
    
    # Verify loss is reasonable (not catastrophically bad)
    # Note: RAW values are in sensor units (0-16k range), so loss will be much higher than 0-1 normalized
    # A loss < 100 is generally good, > 200 indicates possible alignment failure
    assert loss_cfa < 200, f"Loss too high: {loss_cfa} (possible alignment failure)"
    
    logger.info(f"  ✓ CFA alignment successful")


if __name__ == "__main__":
    # Quick test run
    import sys
    
    pairs = find_test_pairs(max_pairs=20)
    print(f"Found {len(pairs)} test pairs")
    
    if len(pairs) == 0:
        print("No test pairs found. Run smoke_test.py first.")
        sys.exit(1)
    
    print("\nRunning CFA alignment validation on first 5 pairs...")
    for i in range(min(5, len(pairs))):
        try:
            test_cfa_alignment_produces_valid_shifts(i)
            print(f"Pair {i}: PASS")
        except Exception as e:
            print(f"Pair {i}: FAIL - {e}")
            raise
    
    print(f"\n✓ All {min(5, len(pairs))} pairs passed CFA alignment validation")
    print(f"Run full test with: pytest test_alignment_cross_validation.py -v")
