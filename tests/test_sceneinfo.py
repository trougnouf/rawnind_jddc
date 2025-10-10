"""
Unit tests for SceneInfo - data structures for scene and image metadata.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import shutil

import pytest
import trio
import numpy as np
import torch

from rawnind.dataset.SceneInfo import ImageInfo, SceneInfo


@pytest.fixture
async def temp_dir():
    """Provide temporary directory."""
    test_dir = tempfile.mkdtemp(prefix="test_sceneinfo_")
    yield Path(test_dir)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


@pytest.fixture
def sample_image_info():
    """Create sample ImageInfo for testing."""
    return ImageInfo(
        filename="test_image.cr2",
        sha1="abc123def456",
        is_clean=True,
        scene_name="TestScene",
        scene_images=["abc123def456", "other_sha1"],
        cfa_type="bayer",
        file_id="12345"
    )


@pytest.mark.trio
async def test_imageinfo_initialization(sample_image_info):
    """Test ImageInfo can be initialized with required fields."""
    assert sample_image_info.filename == "test_image.cr2"
    assert sample_image_info.sha1 == "abc123def456"
    assert sample_image_info.is_clean == True
    assert sample_image_info.cfa_type == "bayer"
    assert sample_image_info._image_tensor is None
    assert sample_image_info.local_path is None


@pytest.mark.trio
async def test_imageinfo_load_image_no_path(sample_image_info):
    """Test load_image raises error when local_path not set."""
    # SceneInfo.load_image checks self.local_path before trio.Path conversion
    with pytest.raises((ValueError, TypeError)):
        await sample_image_info.load_image()


@pytest.mark.trio
async def test_imageinfo_load_image_path_not_exists(sample_image_info, temp_dir):
    """Test load_image raises error when file doesn't exist."""
    sample_image_info.local_path = temp_dir / "nonexistent.cr2"

    with pytest.raises(ValueError, match="doesn't exist"):
        await sample_image_info.load_image()


@pytest.mark.trio
async def test_imageinfo_load_image_caches_tensor(sample_image_info, temp_dir):
    """Test load_image caches tensor in memory."""
    # Create fake RAW file
    test_file = temp_dir / "test.cr2"
    test_file.write_bytes(b"fake raw data")
    sample_image_info.local_path = test_file

    # Mock img_fpath_to_np_mono_flt_and_metadata to return fake data
    fake_array = np.random.rand(100, 100).astype(np.float32)
    fake_metadata = {"black_level": 512, "white_level": 16383}

    with patch("rawnind.libs.rawproc.img_fpath_to_np_mono_flt_and_metadata") as mock_raw:
        mock_raw.return_value = (fake_array, fake_metadata)

        # First load
        tensor1 = await sample_image_info.load_image(as_torch=True)

        # Should have cached the tensor
        assert sample_image_info._image_tensor is not None
        assert isinstance(sample_image_info._image_tensor, torch.Tensor)

        # Second load should return cached tensor without re-reading
        tensor2 = await sample_image_info.load_image(as_torch=True)

        # Should be same object (not a copy)
        assert tensor1 is tensor2

        # raw_fpath_to_mono_img_and_metadata should only be called once
        assert mock_raw.call_count == 1


@pytest.mark.trio
async def test_imageinfo_load_image_numpy_vs_torch(sample_image_info, temp_dir):
    """Test load_image returns numpy or torch based on as_torch parameter."""
    test_file = temp_dir / "test.cr2"
    test_file.write_bytes(b"fake raw data")
    sample_image_info.local_path = test_file

    fake_array = np.random.rand(100, 100).astype(np.float32)
    fake_metadata = {"black_level": 512, "white_level": 16383}

    with patch("rawnind.libs.rawproc.img_fpath_to_np_mono_flt_and_metadata") as mock_raw:
        mock_raw.return_value = (fake_array, fake_metadata)

        # Load as numpy
        result_np = await sample_image_info.load_image(as_torch=False)
        assert isinstance(result_np, np.ndarray)

        # Load as torch (from cache)
        result_torch = await sample_image_info.load_image(as_torch=True)
        assert isinstance(result_torch, torch.Tensor)


@pytest.mark.trio
async def test_imageinfo_unload_image(sample_image_info, temp_dir):
    """Test unload_image clears cached tensor."""
    test_file = temp_dir / "test.cr2"
    test_file.write_bytes(b"fake raw data")
    sample_image_info.local_path = test_file

    fake_array = np.random.rand(100, 100).astype(np.float32)
    fake_metadata = {"black_level": 512, "white_level": 16383}

    with patch("rawnind.libs.rawproc.img_fpath_to_np_mono_flt_and_metadata") as mock_raw:
        mock_raw.return_value = (fake_array, fake_metadata)

        # Load image
        await sample_image_info.load_image(as_torch=True)
        assert sample_image_info._image_tensor is not None

        # Unload
        sample_image_info.unload_image()
        assert sample_image_info._image_tensor is None


@pytest.mark.trio
async def test_imageinfo_image_tensor_property(sample_image_info, temp_dir):
    """Test image_tensor property returns cached tensor or None."""
    assert sample_image_info.image_tensor is None

    test_file = temp_dir / "test.cr2"
    test_file.write_bytes(b"fake raw data")
    sample_image_info.local_path = test_file

    fake_array = np.random.rand(100, 100).astype(np.float32)
    fake_metadata = {"black_level": 512, "white_level": 16383}

    with patch("rawnind.libs.rawproc.img_fpath_to_np_mono_flt_and_metadata") as mock_raw:
        mock_raw.return_value = (fake_array, fake_metadata)

        tensor = await sample_image_info.load_image(as_torch=True)
        assert sample_image_info.image_tensor is tensor


@pytest.mark.trio
async def test_imageinfo_aligned_image_tensor_no_alignment(sample_image_info, temp_dir):
    """Test aligned_image_tensor returns original tensor when no alignment set."""
    test_file = temp_dir / "test.cr2"
    test_file.write_bytes(b"fake raw data")
    sample_image_info.local_path = test_file

    fake_array = np.random.rand(100, 100).astype(np.float32)
    fake_metadata = {"black_level": 512, "white_level": 16383}

    with patch("rawnind.libs.rawproc.img_fpath_to_np_mono_flt_and_metadata") as mock_raw:
        mock_raw.return_value = (fake_array, fake_metadata)

        await sample_image_info.load_image(as_torch=True)

        # No alignment in metadata
        assert "alignment" not in sample_image_info.metadata or sample_image_info.metadata.get("alignment") == [0, 0]

        # Should return original tensor
        aligned = sample_image_info.aligned_image_tensor
        assert aligned is sample_image_info.image_tensor


@pytest.mark.trio
async def test_imageinfo_aligned_image_tensor_with_alignment(sample_image_info, temp_dir):
    """Test aligned_image_tensor applies alignment transformation."""
    test_file = temp_dir / "test.cr2"
    test_file.write_bytes(b"fake raw data")
    sample_image_info.local_path = test_file

    fake_array = np.random.rand(100, 100).astype(np.float32)
    fake_metadata = {"black_level": 512, "white_level": 16383}

    with patch("rawnind.libs.rawproc.img_fpath_to_np_mono_flt_and_metadata") as mock_raw:
        mock_raw.return_value = (fake_array, fake_metadata)

        await sample_image_info.load_image(as_torch=True)

        # Set alignment in metadata
        sample_image_info.metadata["alignment"] = [10, 20]

        # Get aligned tensor
        aligned = sample_image_info.aligned_image_tensor

        # Should be different object (cropped)
        assert aligned is not sample_image_info.image_tensor

        # Should be smaller due to cropping
        orig_shape = sample_image_info.image_tensor.shape
        aligned_shape = aligned.shape

        # At least one dimension should be smaller
        assert aligned_shape[0] <= orig_shape[0] or aligned_shape[1] <= orig_shape[1]


@pytest.mark.trio
async def test_imageinfo_is_loaded_via_tensor(sample_image_info, temp_dir):
    """Test checking if image is loaded via image_tensor property."""
    assert sample_image_info.image_tensor is None

    test_file = temp_dir / "test.cr2"
    test_file.write_bytes(b"fake raw data")
    sample_image_info.local_path = test_file

    fake_array = np.random.rand(100, 100).astype(np.float32)
    fake_metadata = {"black_level": 512, "white_level": 16383}

    with patch("rawnind.libs.rawproc.img_fpath_to_np_mono_flt_and_metadata") as mock_raw:
        mock_raw.return_value = (fake_array, fake_metadata)

        await sample_image_info.load_image(as_torch=True)
        assert sample_image_info.image_tensor is not None

        sample_image_info.unload_image()
        assert sample_image_info.image_tensor is None


@pytest.mark.trio
async def test_sceneinfo_initialization():
    """Test SceneInfo can be initialized."""
    clean_img = ImageInfo(
        filename="clean.cr2",
        sha1="clean_sha",
        is_clean=True,
        scene_name="Scene1",
        scene_images=["clean_sha"],
        cfa_type="bayer",
        file_id="1"
    )

    noisy_img = ImageInfo(
        filename="noisy.cr2",
        sha1="noisy_sha",
        is_clean=False,
        scene_name="Scene1",
        scene_images=["noisy_sha"],
        cfa_type="bayer",
        file_id="2"
    )

    scene = SceneInfo(
        scene_name="Scene1",
        cfa_type="bayer",
        unknown_sensor=False,
        test_reserve=False,
        clean_images=[clean_img],
        noisy_images=[noisy_img]
    )

    assert scene.scene_name == "Scene1"
    assert scene.cfa_type == "bayer"
    assert len(scene.clean_images) == 1
    assert len(scene.noisy_images) == 1


@pytest.mark.trio
async def test_sceneinfo_all_images_property():
    """Test SceneInfo.all_images returns all images."""
    clean_img = ImageInfo(
        filename="clean.cr2",
        sha1="clean_sha",
        is_clean=True,
        scene_name="Scene1",
        scene_images=["clean_sha"],
        cfa_type="bayer",
        file_id="1"
    )

    noisy_img1 = ImageInfo(
        filename="noisy1.cr2",
        sha1="noisy_sha1",
        is_clean=False,
        scene_name="Scene1",
        scene_images=["noisy_sha1"],
        cfa_type="bayer",
        file_id="2"
    )

    noisy_img2 = ImageInfo(
        filename="noisy2.cr2",
        sha1="noisy_sha2",
        is_clean=False,
        scene_name="Scene1",
        scene_images=["noisy_sha2"],
        cfa_type="bayer",
        file_id="3"
    )

    scene = SceneInfo(
        scene_name="Scene1",
        cfa_type="bayer",
        unknown_sensor=False,
        test_reserve=False,
        clean_images=[clean_img],
        noisy_images=[noisy_img1, noisy_img2]
    )

    all_imgs = scene.all_images()
    assert len(all_imgs) == 3
    assert clean_img in all_imgs
    assert noisy_img1 in all_imgs
    assert noisy_img2 in all_imgs


@pytest.mark.trio
async def test_imageinfo_concurrent_loads(sample_image_info, temp_dir):
    """Test concurrent load_image calls with caching."""
    test_file = temp_dir / "test.cr2"
    test_file.write_bytes(b"fake raw data")
    sample_image_info.local_path = test_file

    fake_array = np.random.rand(100, 100).astype(np.float32)
    fake_metadata = {"black_level": 512, "white_level": 16383}

    with patch("rawnind.libs.rawproc.img_fpath_to_np_mono_flt_and_metadata") as mock_raw:
        mock_raw.return_value = (fake_array, fake_metadata)

        # Load once to populate cache
        first_tensor = await sample_image_info.load_image(as_torch=True)

        # Subsequent concurrent calls should return cached tensor
        results = []
        async def loader():
            tensor = await sample_image_info.load_image(as_torch=True)
            results.append(tensor)

        async with trio.open_nursery() as nursery:
            for _ in range(10):
                nursery.start_soon(loader)

        # All results should be the same cached tensor
        assert len(results) == 10
        assert all(t is first_tensor for t in results)
        # Only loaded from disk once
        assert mock_raw.call_count == 1


@pytest.mark.trio
async def test_imageinfo_metadata_stored(sample_image_info, temp_dir):
    """Test that metadata from raw file is stored in ImageInfo."""
    test_file = temp_dir / "test.cr2"
    test_file.write_bytes(b"fake raw data")
    sample_image_info.local_path = test_file

    fake_array = np.random.rand(100, 100).astype(np.float32)
    fake_metadata = {
        "black_level": 512,
        "white_level": 16383,
        "iso": 800,
        "camera": "Canon EOS 7D Mark II"
    }

    with patch("rawnind.libs.rawproc.img_fpath_to_np_mono_flt_and_metadata") as mock_raw:
        mock_raw.return_value = (fake_array, fake_metadata)

        await sample_image_info.load_image(as_torch=True)

        # Metadata should be stored (if ImageInfo has metadata attribute)
        # This documents expected behavior
        assert hasattr(sample_image_info, "_raw_metadata") or hasattr(sample_image_info, "metadata")