#!/usr/bin/env python3
"""
Test script to verify the pairing logic works with RawNIND filename patterns.
"""

import sys
import os
sys.path.append('src')

# Import the functions from the prep_image_dataset module
from rawnind.tools.prep_image_dataset import extract_scene_identifier, files_match_same_scene, validate_image_compatibility

def test_scene_extraction():
    """Test scene identifier extraction with RawNIND patterns."""
    print("=== Testing Scene Identifier Extraction ===")
    
    test_cases = [
        # RawNIND patterns
        ("Bayer_2pilesofplates_GT_ISO100_sha1=854554a34b339413462eb1538d4cb0fa95d468b5.arw", "2pilesofplates"),
        ("Bayer_2pilesofplates_ISO1250_sha1=9121efbd50e2f2392665cb17b435b2526df8e9ae.arw", "2pilesofplates"),
        ("Bayer_7D-1_GT_ISO100_sha1=22aef4a5b4038e241082741117827f364ce6a5ac.cr2", "7d-1"),
        ("Bayer_7D-1_ISO12800_sha1=5322313f9650a7b94028d69567c1f6c66bac8765.cr2", "7d-1"),
        # Legacy patterns (should still work)
        ("gt/scene001_iso100.dng", "scene001"),
        ("iso3200/scene001_iso3200.dng", "scene001"),
    ]
    
    all_passed = True
    for filename, expected in test_cases:
        result = extract_scene_identifier(filename)
        status = "✓" if result == expected else "✗"
        print(f"{status} {filename[:50]}... -> {result} (expected: {expected})")
        if result != expected:
            all_passed = False
    
    return all_passed

def test_scene_matching():
    """Test scene matching between GT and noisy files."""
    print("\n=== Testing Scene Matching ===")
    
    test_cases = [
        # Should match
        ("Bayer_2pilesofplates_GT_ISO100_sha1=854554a34b339413462eb1538d4cb0fa95d468b5.arw",
         "Bayer_2pilesofplates_ISO1250_sha1=9121efbd50e2f2392665cb17b435b2526df8e9ae.arw", True),
        ("Bayer_7D-1_GT_ISO100_sha1=22aef4a5b4038e241082741117827f364ce6a5ac.cr2",
         "Bayer_7D-1_ISO12800_sha1=5322313f9650a7b94028d69567c1f6c66bac8765.cr2", True),
        # Should not match
        ("Bayer_2pilesofplates_GT_ISO100_sha1=854554a34b339413462eb1538d4cb0fa95d468b5.arw",
         "Bayer_7D-1_ISO12800_sha1=5322313f9650a7b94028d69567c1f6c66bac8765.cr2", False),
    ]
    
    all_passed = True
    for gt_file, noisy_file, expected in test_cases:
        result = files_match_same_scene(gt_file, noisy_file)
        status = "✓" if result == expected else "✗"
        gt_short = gt_file.split('_')[1] if 'Bayer_' in gt_file else gt_file[:20]
        noisy_short = noisy_file.split('_')[1] if 'Bayer_' in noisy_file else noisy_file[:20]
        print(f"{status} {gt_short} <-> {noisy_short}: {result} (expected: {expected})")
        if result != expected:
            all_passed = False
    
    return all_passed

def test_pairing_simulation():
    """Simulate the pairing process with mock file lists."""
    print("\n=== Testing Pairing Simulation ===")
    
    # Mock file lists similar to what would be found in the dataset
    gt_files = [
        "Bayer_2pilesofplates_GT_ISO100_sha1=854554a34b339413462eb1538d4cb0fa95d468b5.arw",
        "Bayer_7D-1_GT_ISO100_sha1=22aef4a5b4038e241082741117827f364ce6a5ac.cr2",
        "Bayer_scene3_GT_ISO100_sha1=abcdef1234567890.dng",
    ]
    
    noisy_files = [
        "Bayer_2pilesofplates_ISO1250_sha1=9121efbd50e2f2392665cb17b435b2526df8e9ae.arw",
        "Bayer_2pilesofplates_ISO3200_sha1=1111111111111111111111111111111111111111.arw",
        "Bayer_7D-1_ISO12800_sha1=5322313f9650a7b94028d69567c1f6c66bac8765.cr2",
        "Bayer_scene4_ISO800_sha1=fedcba0987654321.dng",  # No matching GT
    ]
    
    # Simulate pairing logic
    pairs = []
    for gt_file in gt_files:
        for noisy_file in noisy_files:
            if files_match_same_scene(gt_file, noisy_file):
                pairs.append((gt_file, noisy_file))
    
    print(f"Found {len(pairs)} valid pairs:")
    for i, (gt, noisy) in enumerate(pairs, 1):
        gt_scene = extract_scene_identifier(gt)
        noisy_scene = extract_scene_identifier(noisy)
        print(f"  {i}. {gt_scene}: GT vs {noisy.split('_')[2]} ISO")
    
    # Expected pairs: 2pilesofplates (2 pairs), 7D-1 (1 pair)
    expected_pairs = 3
    success = len(pairs) == expected_pairs
    status = "✓" if success else "✗"
    print(f"{status} Expected {expected_pairs} pairs, found {len(pairs)}")
    
    return success

def main():
    """Run all tests."""
    print("Testing RawNIND Pairing Logic")
    print("=" * 50)
    
    test1 = test_scene_extraction()
    test2 = test_scene_matching()
    test3 = test_pairing_simulation()
    
    print("\n" + "=" * 50)
    if test1 and test2 and test3:
        print("✓ All tests passed! The pairing logic should work correctly with RawNIND dataset.")
        return 0
    else:
        print("✗ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())