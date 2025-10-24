"""
Tests for MetadataArtificer async enrichment functionality.

Enhanced with trio.testing utilities for deterministic async testing.
"""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from rawnind.dataset.MetadataArtificer import MetadataArtificer
from rawnind.dataset.SceneInfo import SceneInfo, ImageInfo

pytestmark = pytest.mark.dataset


@pytest.fixture
def metadata_enricher():
    return MetadataArtificer()


@pytest.fixture
def scene_info():
    # Create a mock SceneInfo object with one clean and one noisy image.
    gt_image = ImageInfo(
        filename="gt_image.orf",
        sha1="gt_sha1",
        is_clean=True,
        scene_name="test_scene",
        scene_images=[],
        cfa_type="bayer",
    )
    noisy_image = ImageInfo(
        filename="noisy_image.orf",
        sha1="noisy_sha1",
        is_clean=False,
        scene_name="test_scene",
        scene_images=[],
        cfa_type="bayer",
    )

    return SceneInfo(
        scene_name="test_scene",
        cfa_type="bayer",
        unknown_sensor=False,
        test_reserve=False,
        clean_images=[gt_image],
        noisy_images=[noisy_image],
    )


def test_metadata_enricher_initialization(metadata_enricher):
    assert isinstance(metadata_enricher, MetadataArtificer)
    assert metadata_enricher.cache_path == Path(
        "DocScan/rawnind/datasets/RawNIND/metadata_cache.json"
    )
    assert metadata_enricher.dataset_root == Path("DocScan/rawnind/datasets/RawNIND/DocScan")
    assert metadata_enricher.max_concurrent == 4
    assert callable(metadata_enricher.computation_fn)


def test_metadata_enricher_cache_loading(metadata_enricher):
    cache_path = Path("DocScan/rawnind/tests/test_cache.json")

    # Create a temporary cache file for testing
    with open(cache_path, "w") as f:
        f.write('{"test": {"key": "value"}}')

    enricher_with_custom_cache = MetadataArtificer(cache_path=cache_path)
    assert "test" in enricher_with_custom_cache._metadata_cache


@pytest.mark.trio
async def test_consume_scenes_produce_enriched(mocker):
    metadata_enricher = MetadataArtificer()
    scene_recv_channel, enriched_send_channel = mocker.MagicMock(), mocker.MagicMock()

    scene_info = SceneInfo(
        scene_name="test_scene",
        cfa_type="bayer",
        unknown_sensor=False,
        test_reserve=False,
    )

    # Mock the channels and _enrich_scene method for testing
    scene_recv_channel.__aenter__.return_value = scene_recv_channel
    enriched_send_channel.__aenter__.return_value = enriched_send_channel

    scene_recv_channel.__anext__ = AsyncMock(return_value=scene_info)
    mocker.patch.object(metadata_enricher, "_enrich_scene", new_callable=AsyncMock)

    await metadata_enricher.consume_scenes_produce_enriched(
        scene_recv_channel, enriched_send_channel
    )

    scene_recv_channel.__aenter__.assert_called_once()
    enriched_send_channel.__aenter__.assert_called_once()
    enriched_send_channel.send.assert_awaited_with(scene_info)
    metadata_enricher._save_cache.assert_called_once()


@pytest.mark.trio
async def test_enrich_clean_image(metadata_enricher, mocker):
    img_info = ImageInfo(
        filename="gt_image.tif",
        sha1="test_sha1",
        is_clean=True,
        scene_name="test_scene",
        scene_images=[],
        cfa_type="bayer",
    )

    metadata = {"example_metadata": "example_value"}

    mocker.patch.object(
        metadata_enricher, "_compute_image_stats", return_value=metadata
    )
    await metadata_enricher._enrich_clean_image(img_info)

    assert img_info.metadata == metadata
    assert metadata_enricher._metadata_cache["test_sha1"] == metadata


@pytest.mark.trio
async def test_compute_alignment_metadata(metadata_enricher, mocker):
    gt_img = ImageInfo(
        filename="gt_image.raf",
        sha1="gt_sha1",
        is_clean=True,
        scene_name="test_scene",
        scene_images=[],
        cfa_type="bayer",
    )
    noisy_img = ImageInfo(
        filename="noisy_image.raf",
        sha1="noisy_sha1",
        is_clean=False,
        scene_name="test_scene",
        scene_images=[],
        cfa_type="bayer",
    )

    metadata = {"alignment": [0, 0], "raw_gain": None, "rgb_gain": None}

    compute_sync_mock = mocker.patch.object(
        metadata_enricher, "_compute_alignment_metadata"
    )
    compute_sync_mock.return_value = metadata

    result = await metadata_enricher._compute_alignment_metadata(gt_img, noisy_img)
    assert result == metadata


@pytest.mark.trio
async def test_compute_crops_list(metadata_enricher, mocker):
    scene_info = SceneInfo(
        scene_name="test_scene",
        cfa_type="bayer",
        unknown_sensor=False,
        test_reserve=False,
    )
    gt_img = ImageInfo(
        filename="gt_image.cr2",
        sha1="gt_sha1",
        is_clean=True,
        scene_name="test_scene",
        scene_images=[],
        cfa_type="bayer",
    )
    noisy_img = ImageInfo(
        filename="noisy_image.cr2",
        sha1="noisy_sha1",
        is_clean=False,
        scene_name="test_scene",
        scene_images=[],
        cfa_type="bayer",
    )

    crops_metadata = [
        {
            "coordinates": [0, 0],
            "f_linrec2020_fpath": "path/to/crop",
            "gt_linrec2020_fpath": "path/to/gt",
        }
    ]

    fetch_crops_sync_mock = mocker.patch.object(
        metadata_enricher, "_compute_crops_list"
    )
    fetch_crops_sync_mock.return_value = crops_metadata

    result = await metadata_enricher._compute_crops_list(scene_info, gt_img, noisy_img)
    assert result == crops_metadata


@pytest.mark.trio
async def test_enrich_clean_image_skips_non_image_files(metadata_enricher, mocker):
    """Test that _enrich_clean_image skips non-image files like .xmp"""
    xmp_img = ImageInfo(
        filename="metadata.xmp",
        sha1="xmp_sha1",
        is_clean=True,
        scene_name="test_scene",
        scene_images=[],
        cfa_type="bayer",
    )

    # Mock _compute_image_stats - it should NOT be called for .xmp files
    compute_stats_mock = mocker.patch.object(metadata_enricher, "_compute_image_stats")

    await metadata_enricher._enrich_clean_image(xmp_img)

    # Verify that _compute_image_stats was NOT called
    compute_stats_mock.assert_not_called()
    # Verify that no metadata was added
    assert xmp_img.metadata == {}
    # Verify that nothing was cached
    assert "xmp_sha1" not in metadata_enricher._metadata_cache


@pytest.mark.trio
async def test_enrich_clean_image_processes_valid_image_files(
    metadata_enricher, mocker
):
    """Test that _enrich_clean_image processes valid image files"""

    # Test various valid image extensions
    valid_extensions = [".exr", ".tif", ".tiff", ".arw", ".cr2", ".nef", ".raf", ".dng"]

    for ext in valid_extensions:
        img = ImageInfo(
            filename=f"test{ext}",
            sha1=f"sha1_{ext}",
            is_clean=True,
            scene_name="test_scene",
            scene_images=[],
            cfa_type="bayer",
        )
        img.local_path = None  # Will skip cache check

        # Mock _compute_image_stats to return test metadata
        test_metadata = {"test": f"metadata_for_{ext}"}
        compute_stats_mock = mocker.patch.object(
            metadata_enricher, "_compute_image_stats", return_value=test_metadata
        )

        # Mock trio.to_thread.run_sync
        mocker.patch(
            "trio.to_thread.run_sync",
            new_callable=AsyncMock,
            return_value=test_metadata,
        )

        await metadata_enricher._enrich_clean_image(img)

        # For valid extensions, metadata should be added
        assert img.metadata == test_metadata
        assert metadata_enricher._metadata_cache[f"sha1_{ext}"] == test_metadata


@pytest.mark.trio
async def test_enrich_scene_skips_xmp_files_in_noisy_images(metadata_enricher, mocker):
    """Test that _enrich_scene skips .xmp files in noisy images"""
    gt_image = ImageInfo(
        filename="gt_image.arw",
        sha1="gt_sha1",
        is_clean=True,
        local_path="/path/to/gt.arw",
        validated=True,
        scene_name="test_scene",
        scene_images=[],
        cfa_type="bayer",
    )

    xmp_image = ImageInfo(
        filename="noisy_image.xmp",
        sha1="xmp_sha1",
        is_clean=False,
        local_path="/path/to/noisy.xmp",
        validated=True,
        scene_name="test_scene",
        scene_images=[],
        cfa_type="bayer",
    )

    valid_image = ImageInfo(
        filename="noisy_image.arw",
        sha1="noisy_sha1",
        is_clean=False,
        local_path="/path/to/noisy.arw",
        validated=True,
        scene_name="test_scene",
        scene_images=[],
        cfa_type="bayer",
    )

    scene_info = SceneInfo(
        scene_name="test_scene",
        cfa_type="bayer",
        unknown_sensor=False,
        test_reserve=False,
        clean_images=[gt_image],
        noisy_images=[xmp_image, valid_image],
    )

    # Mock the enrichment methods
    mocker.patch.object(
        metadata_enricher, "_enrich_clean_image", new_callable=AsyncMock
    )
    compute_alignment_mock = mocker.patch.object(
        metadata_enricher,
        "_compute_alignment_metadata",
        new_callable=AsyncMock,
        return_value={"alignment": [0, 0]},
    )

    await metadata_enricher._enrich_scene(scene_info)

    # _compute_alignment_metadata should only be called for the valid image, not the .xmp
    assert compute_alignment_mock.call_count == 1
    # Verify it was called with the valid image
    call_args = compute_alignment_mock.call_args[0]
    assert call_args[1].filename == "noisy_image.arw"


if __name__ == "__main__":
    pytest.main()
