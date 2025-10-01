#!/usr/bin/env python3
"""
Test script to verify the corrected pairing logic works with the expected RawNIND directory structure.
"""

import os
import sys
import tempfile
import shutil

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rawnind.tools.prep_image_dataset import extract_scene_identifier, files_match_same_scene

def create_test_directory_structure():
    """Create a test directory structure matching the expected RawNIND organization."""
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"Created test directory: {temp_dir}")
    
    # Create the expected structure: src/Bayer/scene_name/[gt/]filename
    bayer_dir = os.path.join(temp_dir, "src", "Bayer")
    
    # Scene 1: 2pilesofplates
    scene1_dir = os.path.join(bayer_dir, "2pilesofplates")
    scene1_gt_dir = os.path.join(scene1_dir, "gt")
    os.makedirs(scene1_gt_dir, exist_ok=True)
    
    # Create test files for scene 1
    gt_file1 = "Bayer_2pilesofplates_GT_ISO100_sha1=854554a34b339413462eb1538d4cb0fa95d468b5.arw"
    noisy_file1a = "Bayer_2pilesofplates_ISO200_sha1=123456789abcdef.arw"
    noisy_file1b = "Bayer_2pilesofplates_ISO400_sha1=987654321fedcba.arw"
    
    # Create empty files
    open(os.path.join(scene1_gt_dir, gt_file1), 'w').close()
    open(os.path.join(scene1_dir, noisy_file1a), 'w').close()
    open(os.path.join(scene1_dir, noisy_file1b), 'w').close()
    
    # Scene 2: TitusToys
    scene2_dir = os.path.join(bayer_dir, "TitusToys")
    scene2_gt_dir = os.path.join(scene2_dir, "gt")
    os.makedirs(scene2_gt_dir, exist_ok=True)
    
    # Create test files for scene 2
    gt_file2 = "Bayer_TitusToys_GT_ISO50_sha1=85c54ed5174f0ca97385984e6301a1fab08cbdbe.arw"
    noisy_file2 = "Bayer_TitusToys_ISO800_sha1=abcdef123456789.arw"
    
    open(os.path.join(scene2_gt_dir, gt_file2), 'w').close()
    open(os.path.join(scene2_dir, noisy_file2), 'w').close()
    
    return temp_dir, {
        'scene1_dir': scene1_dir,
        'scene1_gt_dir': scene1_gt_dir,
        'scene2_dir': scene2_dir,
        'scene2_gt_dir': scene2_gt_dir,
        'files': {
            'gt1': os.path.join("gt", gt_file1),
            'noisy1a': noisy_file1a,
            'noisy1b': noisy_file1b,
            'gt2': os.path.join("gt", gt_file2),
            'noisy2': noisy_file2,
        }
    }

def test_pairing_logic():
    """Test the pairing logic with the expected directory structure."""
    
    temp_dir, structure = create_test_directory_structure()
    
    try:
        print("\n=== Testing Scene Matching Logic ===")
        
        files = structure['files']
        
        # Test cases that should match
        test_cases = [
            (files['gt1'], files['noisy1a'], True, "GT and noisy from same scene (2pilesofplates)"),
            (files['gt1'], files['noisy1b'], True, "GT and noisy from same scene (2pilesofplates, different ISO)"),
            (files['gt2'], files['noisy2'], True, "GT and noisy from same scene (TitusToys)"),
            (files['gt1'], files['noisy2'], False, "GT and noisy from different scenes"),
            (files['gt2'], files['noisy1a'], False, "GT and noisy from different scenes"),
        ]
        
        print(f"Testing with files:")
        for key, path in files.items():
            scene = extract_scene_identifier(path)
            print(f"  {key}: {path} -> scene: {scene}")
        
        print(f"\nRunning pairing tests:")
        all_passed = True
        
        for gt_path, noisy_path, expected, description in test_cases:
            result = files_match_same_scene(gt_path, noisy_path)
            status = "✓ PASS" if result == expected else "✗ FAIL"
            print(f"  {status}: {description}")
            print(f"    GT: {gt_path} -> {extract_scene_identifier(gt_path)}")
            print(f"    Noisy: {noisy_path} -> {extract_scene_identifier(noisy_path)}")
            print(f"    Expected: {expected}, Got: {result}")
            
            if result != expected:
                all_passed = False
            print()
        
        # Test directory structure simulation
        print("=== Testing Directory Structure Simulation ===")
        
        scene1_dir = structure['scene1_dir']
        print(f"Scene directory: {scene1_dir}")
        print(f"Contents: {os.listdir(scene1_dir)}")
        
        gt_dir = structure['scene1_gt_dir']
        print(f"GT directory: {gt_dir}")
        print(f"GT files: {os.listdir(gt_dir)}")
        
        # Simulate the corrected pairing logic
        gt_files_endpaths = [os.path.join("gt", fn) for fn in os.listdir(gt_dir)]
        noisy_files_endpaths = os.listdir(scene1_dir)
        noisy_files_endpaths.remove("gt")  # Remove gt directory from list
        
        print(f"\nGT files (with gt/ prefix): {gt_files_endpaths}")
        print(f"Noisy files (direct in scene dir): {noisy_files_endpaths}")
        
        pairs_found = 0
        for gt_file_endpath in gt_files_endpaths:
            for noisy_file in noisy_files_endpaths:
                # Skip directories and unwanted files
                noisy_file_path = os.path.join(scene1_dir, noisy_file)
                if os.path.isdir(noisy_file_path):
                    continue
                    
                f_endpath = noisy_file  # noisy files are directly in scene directory
                
                if files_match_same_scene(gt_file_endpath, f_endpath):
                    pairs_found += 1
                    print(f"  Found pair: {gt_file_endpath} <-> {f_endpath}")
        
        print(f"\nTotal pairs found: {pairs_found}")
        
        if all_passed and pairs_found > 0:
            print("\n🎉 All tests passed! The corrected pairing logic should work.")
        else:
            print(f"\n❌ Some tests failed or no pairs found. all_passed={all_passed}, pairs_found={pairs_found}")
            
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up test directory: {temp_dir}")

if __name__ == "__main__":
    test_pairing_logic()