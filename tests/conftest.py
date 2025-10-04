"""Pytest fixtures for alignment tests."""

import sys
import random
from typing import Tuple
import numpy as np
import pytest

sys.path.insert(0, "src")
from rawnind.libs import raw
from rawnind.dataset import get_dataset_index


def apply_synthetic_shift(
    img: np.ndarray, 
    shift: Tuple[int, int]
) -> np.ndarray:
    """Apply a synthetic shift to an image by cropping and padding.
    
    Args:
        img: Input image, shape (H, W) or (C, H, W)
        shift: (dy, dx) shift to apply
        
    Returns:
        Shifted image with same shape as input
    """
    dy, dx = shift
    
    if img.ndim == 2:
        h, w = img.shape
        shifted = np.zeros_like(img)
        
        src_y_start = max(0, -dy)
        src_y_end = min(h, h - dy)
        src_x_start = max(0, -dx)
        src_x_end = min(w, w - dx)
        
        dst_y_start = max(0, dy)
        dst_y_end = min(h, h + dy)
        dst_x_start = max(0, dx)
        dst_x_end = min(w, w + dx)
        
        shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
            img[src_y_start:src_y_end, src_x_start:src_x_end]
            
    elif img.ndim == 3:
        c, h, w = img.shape
        shifted = np.zeros_like(img)
        
        src_y_start = max(0, -dy)
        src_y_end = min(h, h - dy)
        src_x_start = max(0, -dx)
        src_x_end = min(w, w - dx)
        
        dst_y_start = max(0, dy)
        dst_y_end = min(h, h + dy)
        dst_x_start = max(0, dx)
        dst_x_end = min(w, w + dx)
        
        shifted[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
            img[:, src_y_start:src_y_end, src_x_start:src_x_end]
    else:
        raise ValueError(f"Unsupported image dimensions: {img.ndim}")
    
    return shifted


@pytest.fixture
def random_scenes_with_synthetic_shifts(request):
    """Fixture that provides n random scenes with synthetic shifts applied.
    
    Usage in test:
        def test_something(random_scenes_with_synthetic_shifts):
            scenes = random_scenes_with_synthetic_shifts
            # scenes is a list of dicts with keys:
            # - 'original_img': original RAW image
            # - 'shifted_img': synthetically shifted image
            # - 'true_shift': the known ground truth shift (dy, dx)
            # - 'metadata': image metadata
            # - 'subject': scene name
            # - 'cfa_type': 'Bayer' or 'X-Trans'
    
    The fixture defaults to 10 scenes.
    """
    num_scenes = getattr(request, "param", 10)
    
    # Get dataset index
    index = get_dataset_index()
    index.load_index()
    index.discover_local_files()
    
    # Get available scenes (those with GT images)
    available_scenes = index.get_available_scenes()
    
    if len(available_scenes) == 0:
        pytest.skip("No scenes found in dataset")
    
    # Select random scenes
    num_scenes = min(num_scenes, len(available_scenes))
    selected_scenes = random.sample(available_scenes, num_scenes)
    
    # Load images and apply synthetic shifts
    synthetic_data = []
    
    for scene_info in selected_scenes:
        gt_img_info = scene_info.get_gt_image()
        if gt_img_info is None or gt_img_info.local_path is None:
            continue
        
        # Load the GT image
        img, metadata = raw.raw_fpath_to_mono_img_and_metadata(
            str(gt_img_info.local_path), 
            return_float=True
        )
        
        # Generate a random shift (even values only for Bayer compatibility)
        # Range: -20 to +20 pixels, even values only
        dy = random.choice(range(-20, 22, 2))
        dx = random.choice(range(-20, 22, 2))
        true_shift = (dy, dx)
        
        # Apply synthetic shift
        shifted_img = apply_synthetic_shift(img, true_shift)
        
        synthetic_data.append({
            "original_img": img,
            "shifted_img": shifted_img,
            "true_shift": true_shift,
            "metadata": metadata,
            "subject": scene_info.scene_name,
            "cfa_type": scene_info.cfa_type,
        })
    
    return synthetic_data
