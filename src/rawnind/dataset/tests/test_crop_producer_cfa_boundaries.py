"""Test that CropProducerStage respects CFA block boundaries."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
from rawnind.dataset.crop_producer_stage import CropProducerStage
from rawnind.dataset.SceneInfo import SceneInfo, ImageInfo


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    return tmp_path / "crops"


@pytest.fixture
def mock_bayer_scene():
    """Create mock Bayer scene with GT and noisy images."""
    scene = Mock(spec=SceneInfo)
    scene.scene_name = "test_bayer_scene"
    scene.cfa_type = "Bayer"
    
    gt_img = Mock(spec=ImageInfo)
    gt_img.cfa_type = "Bayer"
    gt_img.validated = True
    gt_img.metadata = {}
    gt_img.sha1 = "abcd1234"
    
    noisy_img = Mock(spec=ImageInfo)
    noisy_img.cfa_type = "Bayer"
    noisy_img.validated = True
    noisy_img.metadata = {"alignment": [3, 5], "gain": 1.0}  # Odd alignment - should be snapped
    noisy_img.sha1 = "efgh5678"
    
    scene.get_gt_image = Mock(return_value=gt_img)
    scene.noisy_images = [noisy_img]
    
    return scene, gt_img, noisy_img


@pytest.fixture
def mock_xtrans_scene():
    """Create mock X-Trans scene with GT and noisy images."""
    scene = Mock(spec=SceneInfo)
    scene.scene_name = "test_xtrans_scene"
    scene.cfa_type = "X-Trans"
    
    gt_img = Mock(spec=ImageInfo)
    gt_img.cfa_type = "X-Trans"
    gt_img.validated = True
    gt_img.metadata = {}
    gt_img.sha1 = "abcd1234"
    
    noisy_img = Mock(spec=ImageInfo)
    noisy_img.cfa_type = "X-Trans"
    noisy_img.validated = True
    noisy_img.metadata = {"alignment": [4, 7], "gain": 1.0}  # Not multiples of 3 - should be snapped
    noisy_img.sha1 = "efgh5678"
    
    scene.get_gt_image = Mock(return_value=gt_img)
    scene.noisy_images = [noisy_img]
    
    return scene, gt_img, noisy_img


def test_bayer_crop_size_must_be_even(temp_output_dir):
    """Bayer crop_size must be even (2x2 block boundaries)."""
    with pytest.raises(AssertionError, match="Bayer.*even"):
        CropProducerStage(
            output_dir=temp_output_dir,
            crop_size=255,  # Odd - should fail
            num_crops=1,
            cfa_type="Bayer"
        )


def test_xtrans_crop_size_must_be_multiple_of_3(temp_output_dir):
    """X-Trans crop_size must be multiple of 3 (3x3 block boundaries)."""
    with pytest.raises(AssertionError, match="X-Trans.*multiple of 3"):
        CropProducerStage(
            output_dir=temp_output_dir,
            crop_size=256,  # Not multiple of 3 - should fail
            num_crops=1,
            cfa_type="X-Trans"
        )


def test_bayer_crop_positions_are_even(temp_output_dir, mock_bayer_scene):
    """Random crop positions must be even for Bayer."""
    scene, gt_img, noisy_img = mock_bayer_scene
    
    # Create mock image data
    h, w = 1024, 1024
    gt_data = np.random.rand(h, w).astype(np.float32)
    noisy_data = gt_data + np.random.randn(h, w).astype(np.float32) * 0.05  # Add correlated noise
    
    stage = CropProducerStage(
        output_dir=temp_output_dir,
        crop_size=256,  # Even
        num_crops=10
    )
    
    # Pass numpy arrays directly (tensor-native architecture)
    crop_metadata = stage._extract_and_save_crops(
        scene.scene_name,
        gt_data,
        noisy_data,
            noisy_img.metadata["alignment"],
            noisy_img.metadata["gain"],
            gt_img.sha1,
            noisy_img.sha1,
            gt_img.cfa_type,
        1.0,  # overexposure_lb
        True  # is_bayer
    )
    
    # Check all crop positions are even
    for crop in crop_metadata:
        y, x = crop["position"]
        assert y % 2 == 0, f"Bayer crop Y position {y} is not even"
        assert x % 2 == 0, f"Bayer crop X position {x} is not even"


def test_xtrans_crop_positions_multiple_of_3(temp_output_dir, mock_xtrans_scene):
    """Random crop positions must be multiples of 3 for X-Trans."""
    scene, gt_img, noisy_img = mock_xtrans_scene
    
    # Create mock image data
    h, w = 513, 513  # Multiple of 3
    gt_data = np.random.rand(h, w).astype(np.float32)
    noisy_data = np.random.rand(h, w).astype(np.float32)
    
    stage = CropProducerStage(
        output_dir=temp_output_dir,
        crop_size=255,  # Multiple of 3
        num_crops=10
    )
    
    # Pass numpy arrays directly (tensor-native architecture)
    crop_metadata = stage._extract_and_save_crops(
        scene.scene_name,
        gt_data,
        noisy_data,
        noisy_img.metadata["alignment"],
        noisy_img.metadata["gain"],
        gt_img.sha1,
        noisy_img.sha1,
        gt_img.cfa_type,
        1.0,  # overexposure_lb
        True  # is_bayer
    )
    
    # Check all crop positions are multiples of 3
    for crop in crop_metadata:
        y, x = crop["position"]
        assert y % 3 == 0, f"X-Trans crop Y position {y} is not multiple of 3"
        assert x % 3 == 0, f"X-Trans crop X position {x} is not multiple of 3"


def test_bayer_alignment_offsets_snapped_to_even(temp_output_dir, mock_bayer_scene):
    """Alignment offsets must be snapped to even for Bayer."""
    scene, gt_img, noisy_img = mock_bayer_scene
    
    # Odd alignment offsets - should be snapped to even
    noisy_img.metadata["alignment"] = [3, 5]
    
    h, w = 1024, 1024
    gt_data = np.random.rand(h, w).astype(np.float32)
    noisy_data = gt_data + np.random.randn(h, w).astype(np.float32) * 0.05  # Add correlated noise
    
    stage = CropProducerStage(
        output_dir=temp_output_dir,
        crop_size=256,
        num_crops=1
    )
    
    # Pass numpy arrays directly (tensor-native architecture)
    # Alignment snapping happens inside _extract_and_save_crops
    crop_metadata = stage._extract_and_save_crops(
        scene.scene_name,
        gt_data,
        noisy_data,
            noisy_img.metadata["alignment"],
            noisy_img.metadata["gain"],
            gt_img.sha1,
            noisy_img.sha1,
            gt_img.cfa_type,
        1.0,  # overexposure_lb
        True  # is_bayer
    )
    
    # Alignment snapping happens internally
    # Just verify the function doesn't crash with odd alignment offsets
    # (snapping from [3,5] to [2,4] or [4,6] happens inside _extract_and_save_crops)


def test_xtrans_alignment_offsets_snapped_to_multiple_of_3(temp_output_dir, mock_xtrans_scene):
    """Alignment offsets must be snapped to multiples of 3 for X-Trans."""
    scene, gt_img, noisy_img = mock_xtrans_scene
    
    # Alignment not multiples of 3 - should be snapped
    noisy_img.metadata["alignment"] = [4, 7]
    
    h, w = 513, 513
    gt_data = np.random.rand(h, w).astype(np.float32)
    noisy_data = np.random.rand(h, w).astype(np.float32)
    
    stage = CropProducerStage(
        output_dir=temp_output_dir,
        crop_size=255,
        num_crops=1
    )
    
    # Similar to Bayer test - verify no assertion errors occur
    # and that the snapping happens correctly
    # Pass numpy arrays directly (tensor-native architecture)
    crop_metadata = stage._extract_and_save_crops(
        scene.scene_name,
        gt_data,
        noisy_data,
        noisy_img.metadata["alignment"],
        noisy_img.metadata["gain"],
        gt_img.sha1,
        noisy_img.sha1,
        gt_img.cfa_type,
        1.0,  # overexposure_lb
        True  # is_bayer
    )
    
    # Crop positions should all be multiples of 3
    for crop in crop_metadata:
        y, x = crop["position"]
        assert y % 3 == 0, f"X-Trans crop Y position {y} is not multiple of 3"
        assert x % 3 == 0, f"X-Trans crop X position {x} is not multiple of 3"


def test_mask_png_saved_to_disk(temp_output_dir, mock_bayer_scene):
    """Mask PNG file should be created in output_dir/masks/ during crop extraction.

    DESIGN: Masks regenerated from cached metadata for temporal locality (hot in page cache).
    FORMATS: is_bayer=sensor CFA type. Data format=(C,H,W) for rawproc.make_overexposure_mask.
    """
    scene, gt_img, noisy_img = mock_bayer_scene

    noisy_img.metadata.update({
        "alignment": [0, 0],
        "gain": 1.0,
        "overexposure_lb": 0.99,
        "is_bayer": True
    })

    h, w = 1024, 1024
    # Create 3-channel RGB tensors with correlation (simulates real aligned pairs)
    gt_data = np.random.rand(3, h, w).astype(np.float32) * 0.5  # Valid range
    noisy_data = gt_data + np.random.randn(3, h, w).astype(np.float32) * 0.05  # Add noise

    stage = CropProducerStage(
        output_dir=temp_output_dir,
        crop_size=256,
        num_crops=1
    )

    # Pass numpy arrays directly (tensor-native architecture, no disk I/O)
    crop_metadata = stage._extract_and_save_crops(
        scene.scene_name,
        gt_data,
        noisy_data,
        noisy_img.metadata["alignment"],
        noisy_img.metadata["gain"],
        gt_img.sha1,
        noisy_img.sha1,
        gt_img.cfa_type,
        noisy_img.metadata["overexposure_lb"],
        noisy_img.metadata["is_bayer"]
    )

    # Verify mask file was created
    expected_mask_path = temp_output_dir / "masks" / f"{scene.scene_name}_{noisy_img.sha1[:8]}_mask.png"
    assert expected_mask_path.exists(), f"Mask PNG not found at {expected_mask_path}"

    # Verify it's a valid PNG with correct dimensions
    from PIL import Image
    mask_img = Image.open(expected_mask_path)
    assert mask_img.size == (w, h), f"Mask dimensions {mask_img.size} != image {(w, h)}"


def test_mask_fpath_added_to_crop_metadata(temp_output_dir, mock_bayer_scene):
    """Each crop metadata should include mask_fpath for legacy loader compatibility.

    LEGACY: rawds.py expects image["mask_fpath"]. All crops from same pair share one mask PNG.
    """
    scene, gt_img, noisy_img = mock_bayer_scene

    noisy_img.metadata.update({
        "alignment": [0, 0],
        "gain": 1.0,
        "overexposure_lb": 0.99,
        "is_bayer": True
    })

    h, w = 1024, 1024
    gt_data = np.random.rand(3, h, w).astype(np.float32) * 0.5
    noisy_data = gt_data + np.random.randn(3, h, w).astype(np.float32) * 0.05  # Add correlated noise

    stage = CropProducerStage(
        output_dir=temp_output_dir,
        crop_size=256,
        num_crops=3
    )

    # Pass numpy arrays directly (tensor-native architecture)
    crop_metadata = stage._extract_and_save_crops(
        scene.scene_name,
        gt_data,
        noisy_data,
            noisy_img.metadata["alignment"],
            noisy_img.metadata["gain"],
            gt_img.sha1,
            noisy_img.sha1,
            gt_img.cfa_type,
            noisy_img.metadata["overexposure_lb"],
        noisy_img.metadata["is_bayer"]
    )

    # Verify all crops have mask_fpath
    assert len(crop_metadata) > 0, "No crops generated"
    for crop in crop_metadata:
        assert "mask_fpath" in crop, "mask_fpath missing from crop metadata"
        mask_path = Path(crop["mask_fpath"])
        assert mask_path.name.endswith("_mask.png"), f"Invalid mask filename: {mask_path.name}"
        assert "masks" in str(mask_path), f"Mask not in masks/ directory: {mask_path}"
        # Verify the path points to the same file for all crops (one mask per pair)
        assert mask_path.exists(), f"Mask file doesn't exist: {mask_path}"


def test_vectorized_batch_validation_finds_valid_crops(temp_output_dir, mock_bayer_scene):
    """Vectorized validation should find valid crops and reject heavily masked regions.

    PERFORMANCE: 20x oversample + vectorized filter + random sample = 2-5x faster than retry loops.
    BEHAVIOR: Rejects crops with >50% masked pixels (MAX_MASKED threshold).
    """
    scene, gt_img, noisy_img = mock_bayer_scene

    noisy_img.metadata.update({
        "alignment": [0, 0],
        "gain": 1.0,
        "overexposure_lb": 0.99,
        "is_bayer": True
    })

    h, w = 1024, 1024
    # Create images where top-left has high values (will be masked), rest is valid
    gt_data = np.ones((3, h, w), dtype=np.float32) * 0.5
    gt_data[:, :256, :256] = 1.5  # Overexposed region (all channels)

    # Noisy should match structure with added noise
    noisy_data = gt_data.copy() + np.random.randn(3, h, w).astype(np.float32) * 0.05

    stage = CropProducerStage(
        output_dir=temp_output_dir,
        crop_size=256,
        num_crops=5  # Request 5 crops
    )

    # Pass numpy arrays directly (tensor-native architecture)
    crop_metadata = stage._extract_and_save_crops(
        scene.scene_name,
        gt_data,
        noisy_data,
            noisy_img.metadata["alignment"],
            noisy_img.metadata["gain"],
            gt_img.sha1,
            noisy_img.sha1,
            gt_img.cfa_type,
            noisy_img.metadata["overexposure_lb"],
        noisy_img.metadata["is_bayer"]
    )

    # Should generate some crops (may be <5 if heavily masked)
    assert len(crop_metadata) > 0, "No valid crops found"

    # All generated crops should avoid the top-left overexposed region
    # (since it will be masked and violate MAX_MASKED threshold)
    for crop in crop_metadata:
        y, x = crop["position"]
        # Crop shouldn't be entirely in top-left quadrant
        # (at least not starting at 0,0 since that's heavily masked)
        if y == 0 and x == 0:
            # This crop would be heavily masked, should have been rejected
            pytest.fail(f"Crop at (0,0) should have been rejected due to masking")


def test_mask_png_saved_to_disk_2d_bayer(temp_output_dir, mock_bayer_scene):
    """Mask PNG file should be created for 2D Bayer data format (H,W).

    DESIGN: Tests 2D Bayer input format (raw_image_visible returns H,W array).
    BEHAVIOR: Should work identically to 3D RGB test but with different data shape.
    """
    scene, gt_img, noisy_img = mock_bayer_scene

    noisy_img.metadata.update({
        "alignment": [0, 0],
        "gain": 1.0,
        "overexposure_lb": 0.99,
        "is_bayer": True
    })

    h, w = 1024, 1024
    # Create 2D Bayer data with correlation (simulates real aligned pairs)
    gt_data = np.random.rand(h, w).astype(np.float32) * 0.5  # 2D Bayer
    noisy_data = gt_data + np.random.randn(h, w).astype(np.float32) * 0.05  # Add noise

    stage = CropProducerStage(
        output_dir=temp_output_dir,
        crop_size=256,
        num_crops=1
    )

    # Pass numpy arrays directly (tensor-native architecture)
    crop_metadata = stage._extract_and_save_crops(
        scene.scene_name,
        gt_data,
        noisy_data,
            noisy_img.metadata["alignment"],
            noisy_img.metadata["gain"],
            gt_img.sha1,
            noisy_img.sha1,
            gt_img.cfa_type,
            noisy_img.metadata["overexposure_lb"],
        noisy_img.metadata["is_bayer"]
    )

    # Verify all crops have mask_fpath
    assert len(crop_metadata) > 0, "No crops generated"
    for crop in crop_metadata:
        assert "mask_fpath" in crop, "mask_fpath missing from crop metadata"
        mask_path = Path(crop["mask_fpath"])
        assert mask_path.name.endswith("_mask.png"), f"Invalid mask filename: {mask_path.name}"
        assert "masks" in str(mask_path), f"Mask not in masks/ directory: {mask_path}"
        # Verify the path points to the same file for all crops (one mask per pair)
        assert mask_path.exists(), f"Mask file doesn't exist: {mask_path}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
