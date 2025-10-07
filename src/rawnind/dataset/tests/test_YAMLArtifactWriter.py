"""
Comprehensive tests for YAMLArtifactWriter.

Tests cover:
- YAML descriptor generation from SceneInfo
- File writing and persistence
- Integration with PostDownloadWorker base class
- Trio channel consumption
- Edge cases and error handling

Target coverage: >90%
"""

from pathlib import Path
from typing import List, Dict, Any

import pytest
import trio
import yaml

from rawnind.dataset.SceneInfo import SceneInfo, ImageInfo
# This will fail initially (RED) - we're writing tests first
from rawnind.dataset.YAMLArtifactWriter import YAMLArtifactWriter, scene_to_yaml_descriptor

pytestmark = pytest.mark.dataset


# ============================================================================
# Fixtures for Test Data
# ============================================================================

@pytest.fixture
def minimal_scene():
    """Create a minimal valid SceneInfo for testing."""
    gt_img = ImageInfo(
        filename="gt.exr",
        sha1="abc123",
        is_clean=True,
        scene_name="test_scene",
        scene_images=["gt.exr", "noisy.arw"],
        cfa_type="Bayer",
        local_path=Path("/fake/dataset/test_scene/gt/gt.exr"),
        validated=True,
    )

    noisy_img = ImageInfo(
        filename="noisy.arw",
        sha1="def456",
        is_clean=False,
        scene_name="test_scene",
        scene_images=["gt.exr", "noisy.arw"],
        cfa_type="Bayer",
        local_path=Path("/fake/dataset/test_scene/noisy.arw"),
        validated=True,
        metadata={
            "alignment": [2, -3],
            "alignment_loss": 0.05,
            "mask_mean": 0.92,
            "raw_gain": 1.5,
            "rgb_gain": None,
            "is_bayer": True,
            "overexposure_lb": 0.98,
            "rgb_xyz_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "crops": [
                {
                    "coordinates": [512, 256],
                    "gt_linrec2020_fpath": "/fake/crops/gt_512_256.exr",
                    "f_bayer_fpath": "/fake/crops/noisy_512_256.npy",
                    "gt_bayer_fpath": "/fake/crops/gt_512_256.npy",
                    "f_linrec2020_fpath": "/fake/crops/noisy_512_256.exr",
                }
            ],
        }
    )

    return SceneInfo(
        scene_name="test_scene",
        cfa_type="Bayer",
        unknown_sensor=False,
        test_reserve=False,
        clean_images=[gt_img],
        noisy_images=[noisy_img],
    )


@pytest.fixture
def scene_with_multiple_noisy(minimal_scene):
    """Scene with multiple noisy images."""
    noisy2 = ImageInfo(
        filename="noisy2.arw",
        sha1="ghi789",
        is_clean=False,
        scene_name="test_scene",
        scene_images=["gt.exr", "noisy.arw", "noisy2.arw"],
        cfa_type="Bayer",
        local_path=Path("/fake/dataset/test_scene/noisy2.arw"),
        validated=True,
        metadata={
            "alignment": [1, 1],
            "alignment_loss": 0.03,
            "mask_mean": 0.95,
            "raw_gain": 1.2,
        }
    )
    minimal_scene.noisy_images.append(noisy2)
    return minimal_scene


@pytest.fixture
def scene_missing_optional_metadata(minimal_scene):
    """Scene with minimal required metadata only."""
    minimal_scene.noisy_images[0].metadata = {
        "alignment": [0, 0],
        "alignment_loss": 0.0,
        "mask_mean": 1.0,
        # Missing: raw_gain, crops, rgb_xyz_matrix, etc.
    }
    return minimal_scene


# ============================================================================
# Tests for scene_to_yaml_descriptor()
# ============================================================================

def test_scene_to_yaml_descriptor_basic(minimal_scene, tmp_path):
    """Test basic YAML descriptor generation."""
    descriptor = scene_to_yaml_descriptor(minimal_scene, tmp_path)

    # Core fields present
    assert descriptor["scene_name"] == "test_scene"
    assert descriptor["image_set"] == "test_scene"
    assert descriptor["is_bayer"] is True

    # Alignment data
    assert descriptor["best_alignment"] == [2, -3]
    assert descriptor["best_alignment_loss"] == 0.05

    # Mask data
    assert descriptor["mask_mean"] == 0.92

    # Gain data
    assert descriptor["raw_gain"] == 1.5
    assert descriptor["rgb_gain"] is None

    # File paths
    assert str(descriptor["f_fpath"]) == "/fake/dataset/test_scene/noisy.arw"
    assert str(descriptor["gt_fpath"]) == "/fake/dataset/test_scene/gt/gt.exr"

    # Crops
    assert len(descriptor["crops"]) == 1
    assert descriptor["crops"][0]["coordinates"] == [512, 256]


def test_scene_to_yaml_descriptor_handles_missing_optional(scene_missing_optional_metadata, tmp_path):
    """Test descriptor generation with minimal metadata."""
    descriptor = scene_to_yaml_descriptor(scene_missing_optional_metadata, tmp_path)

    # Required fields still present
    assert descriptor["best_alignment"] == [0, 0]
    assert descriptor["mask_mean"] == 1.0

    # Optional fields have defaults
    assert "raw_gain" in descriptor
    assert "crops" in descriptor
    assert "rgb_xyz_matrix" in descriptor


def test_scene_to_yaml_descriptor_no_gt_image():
    """Test error when scene has no GT image."""
    scene = SceneInfo(
        scene_name="bad_scene",
        cfa_type="Bayer",
        unknown_sensor=False,
        test_reserve=False,
        clean_images=[],  # No GT!
        noisy_images=[ImageInfo(
            filename="noisy.arw",
            sha1="abc",
            is_clean=False,
            scene_name="bad_scene",
            scene_images=["noisy.arw"],
            cfa_type="Bayer",
        )],
    )

    with pytest.raises((ValueError, AttributeError)):  # Should raise error
        scene_to_yaml_descriptor(scene, Path("/fake"))


def test_scene_to_yaml_descriptor_no_noisy_images():
    """Test error when scene has no noisy images."""
    scene = SceneInfo(
        scene_name="bad_scene",
        cfa_type="Bayer",
        unknown_sensor=False,
        test_reserve=False,
        clean_images=[ImageInfo(
            filename="gt.exr",
            sha1="abc",
            is_clean=True,
            scene_name="bad_scene",
            scene_images=["gt.exr"],
            cfa_type="Bayer",
        )],
        noisy_images=[],  # No noisy!
    )

    with pytest.raises((ValueError, IndexError)):  # Should raise error
        scene_to_yaml_descriptor(scene, Path("/fake"))


def test_scene_to_yaml_descriptor_uses_first_noisy(scene_with_multiple_noisy, tmp_path):
    """Test that descriptor uses first noisy image when multiple exist."""
    descriptor = scene_to_yaml_descriptor(scene_with_multiple_noisy, tmp_path)

    # Should use first noisy image's metadata
    assert descriptor["best_alignment"] == [2, -3]  # From first noisy
    assert descriptor["best_alignment_loss"] == 0.05


# ============================================================================
# Tests for YAMLArtifactWriter Initialization
# ============================================================================

def test_yaml_artifact_writer_init(tmp_path):
    """Test YAMLArtifactWriter initialization."""
    writer = YAMLArtifactWriter(
        output_dir=tmp_path,
        output_filename="test_output.yaml",
    )

    assert writer.output_dir == tmp_path
    assert writer.output_filename == "test_output.yaml"
    assert writer.yaml_path == tmp_path / "test_output.yaml"
    assert tmp_path.exists()  # Output dir should be created


def test_yaml_artifact_writer_creates_output_dir(tmp_path):
    """Test that writer creates non-existent output directory."""
    output_dir = tmp_path / "nested" / "dir"
    assert not output_dir.exists()

    YAMLArtifactWriter(output_dir=output_dir)

    assert output_dir.exists()


# ============================================================================
# Tests for process_scene()
# ============================================================================

def test_process_scene_buffers_descriptor(minimal_scene, tmp_path):
    """Test that process_scene buffers descriptors."""
    writer = YAMLArtifactWriter(output_dir=tmp_path)

    # Should start empty
    assert len(writer.descriptors) == 0

    # Process a scene (synchronous for testing)
    import asyncio
    result = asyncio.run(writer.process_scene(minimal_scene))

    # Should buffer the descriptor
    assert len(writer.descriptors) == 1
    assert writer.descriptors[0]["scene_name"] == "test_scene"

    # Should return the same scene (pass-through)
    assert result is minimal_scene


@pytest.mark.trio
async def test_process_scene_async(minimal_scene, tmp_path):
    """Test process_scene in async context."""
    writer = YAMLArtifactWriter(output_dir=tmp_path)

    result = await writer.process_scene(minimal_scene)

    assert len(writer.descriptors) == 1
    assert result is minimal_scene


def test_process_scene_accumulates_multiple(minimal_scene, tmp_path):
    """Test that multiple scenes accumulate."""
    writer = YAMLArtifactWriter(output_dir=tmp_path)

    import asyncio

    # Process multiple scenes
    scene2 = SceneInfo(
        scene_name="scene2",
        cfa_type="Bayer",
        unknown_sensor=False,
        test_reserve=False,
        clean_images=[minimal_scene.clean_images[0]],
        noisy_images=[minimal_scene.noisy_images[0]],
    )

    asyncio.run(writer.process_scene(minimal_scene))
    asyncio.run(writer.process_scene(scene2))

    assert len(writer.descriptors) == 2
    assert writer.descriptors[0]["scene_name"] == "test_scene"
    assert writer.descriptors[1]["scene_name"] == "scene2"


# ============================================================================
# Tests for shutdown() and YAML Writing
# ============================================================================

@pytest.mark.trio
async def test_shutdown_writes_yaml(minimal_scene, tmp_path):
    """Test that shutdown writes buffered descriptors to YAML."""
    writer = YAMLArtifactWriter(output_dir=tmp_path)

    await writer.process_scene(minimal_scene)

    # File should not exist yet
    assert not writer.yaml_path.exists()

    # Shutdown should write
    await writer.shutdown()

    # File should now exist
    assert writer.yaml_path.exists()

    # Verify contents
    with open(writer.yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["scene_name"] == "test_scene"


@pytest.mark.trio
async def test_shutdown_writes_valid_yaml_format(minimal_scene, tmp_path):
    """Test that written YAML is valid and correctly formatted."""
    writer = YAMLArtifactWriter(output_dir=tmp_path)

    await writer.process_scene(minimal_scene)
    await writer.shutdown()

    # Load and verify all expected fields
    with open(writer.yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    entry = data[0]

    # Check all required fields for legacy dataloader compatibility
    required_fields = [
        "scene_name", "image_set", "is_bayer",
        "best_alignment", "best_alignment_loss",
        "mask_mean", "mask_fpath",
        "raw_gain", "rgb_gain",
        "f_fpath", "gt_fpath",
        "crops", "rgb_xyz_matrix",
    ]

    for field in required_fields:
        assert field in entry, f"Missing required field: {field}"


@pytest.mark.trio
async def test_shutdown_empty_descriptors(tmp_path):
    """Test shutdown with no descriptors buffered."""
    writer = YAMLArtifactWriter(output_dir=tmp_path)

    # Shutdown without processing any scenes
    await writer.shutdown()

    # Should write empty list
    assert writer.yaml_path.exists()
    with open(writer.yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    assert data == []


@pytest.mark.trio
async def test_shutdown_overwrites_existing_file(minimal_scene, tmp_path):
    """Test that shutdown overwrites existing YAML file."""
    writer = YAMLArtifactWriter(output_dir=tmp_path)

    # Create pre-existing file
    with open(writer.yaml_path, 'w') as f:
        yaml.dump([{"old": "data"}], f)

    await writer.process_scene(minimal_scene)
    await writer.shutdown()

    # Should overwrite with new data
    with open(writer.yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    assert len(data) == 1
    assert data[0]["scene_name"] == "test_scene"
    assert "old" not in data[0]


# ============================================================================
# Tests for consume_and_produce() Integration
# ============================================================================

@pytest.mark.trio
async def test_consume_and_produce_single_scene(minimal_scene, tmp_path):
    """Test full consume_and_produce workflow with single scene."""
    writer = YAMLArtifactWriter(output_dir=tmp_path)

    send_channel, receive_channel = trio.open_memory_channel(10)

    async with trio.open_nursery() as nursery:
        # Start writer consuming from channel
        nursery.start_soon(writer.consume_and_produce, receive_channel, None)

        # Send a scene
        await send_channel.send(minimal_scene)
        await send_channel.aclose()

    # After nursery exits, shutdown should have run
    assert writer.yaml_path.exists()

    with open(writer.yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    assert len(data) == 1
    assert data[0]["scene_name"] == "test_scene"


@pytest.mark.trio
async def test_consume_and_produce_multiple_scenes(minimal_scene, tmp_path):
    """Test consume_and_produce with multiple scenes."""
    writer = YAMLArtifactWriter(output_dir=tmp_path)

    send_channel, receive_channel = trio.open_memory_channel(10)

    # Create multiple scenes
    scenes = []
    for i in range(5):
        scene = SceneInfo(
            scene_name=f"scene_{i}",
            cfa_type="Bayer",
            unknown_sensor=False,
            test_reserve=False,
            clean_images=[minimal_scene.clean_images[0]],
            noisy_images=[minimal_scene.noisy_images[0]],
        )
        scenes.append(scene)

    async with trio.open_nursery() as nursery:
        nursery.start_soon(writer.consume_and_produce, receive_channel, None)

        for scene in scenes:
            await send_channel.send(scene)

        await send_channel.aclose()

    # Verify all scenes written
    with open(writer.yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    assert len(data) == 5
    for i in range(5):
        assert data[i]["scene_name"] == f"scene_{i}"


@pytest.mark.trio
async def test_consume_and_produce_no_scenes(tmp_path):
    """Test consume_and_produce with empty channel."""
    writer = YAMLArtifactWriter(output_dir=tmp_path)

    send_channel, receive_channel = trio.open_memory_channel(10)

    async with trio.open_nursery() as nursery:
        nursery.start_soon(writer.consume_and_produce, receive_channel, None)
        await send_channel.aclose()

    # Should write empty YAML
    assert writer.yaml_path.exists()
    with open(writer.yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    assert data == []


# ============================================================================
# Tests for Edge Cases
# ============================================================================

def test_scene_without_metadata_raises_error(tmp_path):
    """Test handling of scene without required metadata."""
    scene = SceneInfo(
        scene_name="incomplete_scene",
        cfa_type="Bayer",
        unknown_sensor=False,
        test_reserve=False,
        clean_images=[ImageInfo(
            filename="gt.exr",
            sha1="abc",
            is_clean=True,
            scene_name="incomplete_scene",
            scene_images=["gt.exr"],
            cfa_type="Bayer",
            local_path=Path("/fake/gt.exr"),
        )],
        noisy_images=[ImageInfo(
            filename="noisy.arw",
            sha1="def",
            is_clean=False,
            scene_name="incomplete_scene",
            scene_images=["noisy.arw"],
            cfa_type="Bayer",
            local_path=Path("/fake/noisy.arw"),
            metadata={},  # Empty metadata!
        )],
    )

    # Should either raise error or handle gracefully
    # Let's test that it at least doesn't crash completely
    try:
        descriptor = scene_to_yaml_descriptor(scene, tmp_path)
        # If it doesn't raise, verify required fields have defaults
        assert "best_alignment" in descriptor
        assert "mask_mean" in descriptor
    except (KeyError, ValueError):
        # Acceptable to raise error for missing required data
        pass


def test_scene_with_empty_crops_list(minimal_scene, tmp_path):
    """Test scene with empty crops list."""
    minimal_scene.noisy_images[0].metadata["crops"] = []

    descriptor = scene_to_yaml_descriptor(minimal_scene, tmp_path)

    assert descriptor["crops"] == []


@pytest.mark.trio
async def test_writer_with_custom_filename(minimal_scene, tmp_path):
    """Test writer with custom output filename."""
    writer = YAMLArtifactWriter(
        output_dir=tmp_path,
        output_filename="custom_name.yaml",
    )

    await writer.process_scene(minimal_scene)
    await writer.shutdown()

    custom_path = tmp_path / "custom_name.yaml"
    assert custom_path.exists()


# ============================================================================
# Tests for PostDownloadWorker Base Class Compatibility
# ============================================================================

def test_inherits_from_post_download_worker(tmp_path):
    """Test that YAMLArtifactWriter properly inherits from PostDownloadWorker."""
    from rawnind.dataset.post_download_worker import PostDownloadWorker

    writer = YAMLArtifactWriter(output_dir=tmp_path)

    assert isinstance(writer, PostDownloadWorker)
    assert hasattr(writer, 'consume_and_produce')
    assert hasattr(writer, 'startup')
    assert hasattr(writer, 'shutdown')
    assert hasattr(writer, 'process_scene')


@pytest.mark.trio
async def test_async_context_manager(minimal_scene, tmp_path):
    """Test YAMLArtifactWriter as async context manager."""
    async with YAMLArtifactWriter(output_dir=tmp_path) as writer:
        await writer.process_scene(minimal_scene)

    # After context exit, shutdown should have run
    assert writer.yaml_path.exists()


# ============================================================================
# Integration Test with Real File Paths
# ============================================================================

@pytest.mark.trio
async def test_full_integration_with_file_paths(tmp_path):
    """Integration test with realistic file paths."""
    # Create fake directory structure
    dataset_root = tmp_path / "datasets" / "RawNIND"
    scene_dir = dataset_root / "src" / "Bayer" / "test_scene"
    scene_dir.mkdir(parents=True)
    gt_dir = scene_dir / "gt"
    gt_dir.mkdir()

    # Create fake image files
    gt_file = gt_dir / "gt.exr"
    gt_file.touch()
    noisy_file = scene_dir / "noisy.arw"
    noisy_file.touch()

    # Create scene with real paths
    gt_img = ImageInfo(
        filename="gt.exr",
        sha1="abc123",
        is_clean=True,
        scene_name="test_scene",
        scene_images=["gt.exr", "noisy.arw"],
        cfa_type="Bayer",
        local_path=gt_file,
        validated=True,
    )

    noisy_img = ImageInfo(
        filename="noisy.arw",
        sha1="def456",
        is_clean=False,
        scene_name="test_scene",
        scene_images=["gt.exr", "noisy.arw"],
        cfa_type="Bayer",
        local_path=noisy_file,
        validated=True,
        metadata={
            "alignment": [0, 0],
            "alignment_loss": 0.01,
            "mask_mean": 0.99,
            "raw_gain": 1.0,
            "crops": [],
        }
    )

    scene = SceneInfo(
        scene_name="test_scene",
        cfa_type="Bayer",
        unknown_sensor=False,
        test_reserve=False,
        clean_images=[gt_img],
        noisy_images=[noisy_img],
    )

    # Write YAML
    output_dir = tmp_path / "output"
    writer = YAMLArtifactWriter(output_dir=output_dir)

    await writer.process_scene(scene)
    await writer.shutdown()

    # Verify YAML has correct paths
    with open(writer.yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    assert str(gt_file) in data[0]["gt_fpath"]
    assert str(noisy_file) in data[0]["f_fpath"]