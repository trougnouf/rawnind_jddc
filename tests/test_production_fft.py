#!/usr/bin/env python3
"""Test production CFA-aware FFT implementation on real RawNIND pairs."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rawnind.libs import raw

# Test cases from curated dataset
TEST_CASES = [
    # Bayer - known misalignments
    {
        "name": "Bark ISO65535 (Bayer)",
        "anchor": "src/rawnind/datasets/RawNIND/src/Bayer/Bark/gt/Bayer_Bark_GT_ISO100_sha1=f15da1140d949ee30c15ce7b251839a7b7a41de7.cr2",
        "target": "src/rawnind/datasets/RawNIND/src/Bayer/Bark/Bayer_Bark_ISO65535_sha1=6ba8ed5f7fff42c4c900812c02701649f4f2d49e.cr2",
        "expected": (12, -10)
    },
    {
        "name": "Bark ISO800 (Bayer)",
        "anchor": "src/rawnind/datasets/RawNIND/src/Bayer/Bark/gt/Bayer_Bark_GT_ISO100_sha1=f15da1140d949ee30c15ce7b251839a7b7a41de7.cr2",
        "target": "src/rawnind/datasets/RawNIND/src/Bayer/Bark/Bayer_Bark_ISO800_sha1=ba86f1da64a4bb534d9216e96c1c72177ed1e625.cr2",
        "expected": (4, -4)
    },
    {
        "name": "Kortlek ISO51200 (Bayer)", 
        "anchor": "src/rawnind/datasets/RawNIND/src/Bayer/Kortlek/gt/Bayer_Kortlek_GT_ISO100_sha1=b5a564cd9291224caf363b6b03054365d59d316b.cr2",
        "target": "src/rawnind/datasets/RawNIND/src/Bayer/Kortlek/Bayer_Kortlek_ISO51200_sha1=8fed453fdfb162673bb9ed41f9c5f03095331e3b.cr2",
        "expected": (0, -4)
    },
    # X-Trans - already aligned
    {
        "name": "MuseeL-pedestal ISO6400 (X-Trans)",
        "anchor": "src/rawnind/datasets/RawNIND/src/X-Trans/MuseeL-pedestal/gt/X-Trans_MuseeL-pedestal_GT_ISO200_sha1=161b21f545c4c4ed7fc4fce014f637bb7040d8aa.raf",
        "target": "src/rawnind/datasets/RawNIND/src/X-Trans/MuseeL-pedestal/X-Trans_MuseeL-pedestal_ISO6400_sha1=693af8d1f36f89ad3c4cb8c0eb0e41c924684140.raf",
        "expected": (0, 0)
    }
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
            test_case['anchor'], return_float=True
        )
        target, target_meta = raw.raw_fpath_to_mono_img_and_metadata(
            test_case['target'], return_float=True
        )
        
        # Apply production FFT function
        shift, channel_shifts = raw.fft_phase_correlate_cfa(
            anchor, target, anchor_meta, method="median", verbose=False
        )
        
        print(f"  Detected: {shift}")
        print(f"  Channels: {channel_shifts}")
        
        # Check result
        if shift == test_case['expected']:
            print(f"  ✅ PASS")
            passed += 1
        else:
            print(f"  ❌ FAIL")
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(TEST_CASES)} tests")
    print("=" * 80)
    
    return failed == 0

if __name__ == "__main__":
    success = test_production_implementation()
    sys.exit(0 if success else 1)
