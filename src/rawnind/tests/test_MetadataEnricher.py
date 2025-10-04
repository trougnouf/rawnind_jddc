from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from rawnind.dataset.MetadataEnricher import MetadataEnricher
from rawnind.dataset.SceneInfo import SceneInfo, ImageInfo


@pytest.fixture
def metadata_enricher():
    return MetadataEnricher()


@pytest.fixture
def scene_info():
    # Create a mock SceneInfo object with one clean and one noisy image.
    gt_image = ImageInfo(
        filename="gt_image.orf",
        sha1="gt_sha1",
        is_clean=True,
        scene_name="test_scene",
        scene_images=[],
        cfa_type="Bayer"
    )
    noisy_image = ImageInfo(
        filename="noisy_image.orf",
        sha1="noisy_sha1",
        is_clean=False,
        scene_name="test_scene",
        scene_images=[],
        cfa_type="Bayer"
    )

    return SceneInfo(
        scene_name="test_scene",
        cfa_type="Bayer",
        unknown_sensor=False,
        test_reserve=False,
        clean_images=[gt_image],
        noisy_images=[noisy_image]
    )


def test_metadata_enricher_initialization(metadata_enricher):
    assert isinstance(metadata_enricher, MetadataEnricher)
    assert metadata_enricher.cache_path == Path("src/rawnind/datasets/RawNIND/metadata_cache.json")
    assert metadata_enricher.dataset_root == Path("src/rawnind/datasets/RawNIND/src")
    assert metadata_enricher.max_concurrent == 4
    assert callable(metadata_enricher.computation_fn)


def test_metadata_enricher_cache_loading(metadata_enricher):
    cache_path = Path("src/rawnind/tests/test_cache.json")

    # Create a temporary cache file for testing
    with open(cache_path, "w") as f:
        f.write('{"test": {"key": "value"}}')

    enricher_with_custom_cache = MetadataEnricher(cache_path=cache_path)
    assert "test" in enricher_with_custom_cache._metadata_cache


async def test_consume_scenes_produce_enriched(mocker):
    metadata_enricher = MetadataEnricher()
    scene_recv_channel, enriched_send_channel = mocker.MagicMock(), mocker.MagicMock()

    scene_info = SceneInfo(
        scene_name="test_scene",
        cfa_type="Bayer",
        unknown_sensor=False,
        test_reserve=False
    )

    # Mock the channels and _enrich_scene method for testing
    scene_recv_channel.__aenter__.return_value = scene_recv_channel
    enriched_send_channel.__aenter__.return_value = enriched_send_channel

    scene_recv_channel.__anext__ = AsyncMock(return_value=scene_info)
    mocker.patch.object(metadata_enricher, '_enrich_scene', new_callable=AsyncMock)

    await metadata_enricher.consume_scenes_produce_enriched(scene_recv_channel, enriched_send_channel)

    scene_recv_channel.__aenter__.assert_called_once()
    enriched_send_channel.__aenter__.assert_called_once()
    enriched_send_channel.send.assert_awaited_with(scene_info)
    metadata_enricher._save_cache.assert_called_once()


async def test_enrich_clean_image(metadata_enricher, mocker):
    img_info = ImageInfo(
        filename="gt_image.tif",
        sha1="test_sha1",
        is_clean=True,
        scene_name="test_scene",
        scene_images=[],
        cfa_type="Bayer"
    )

    metadata = {"example_metadata": "example_value"}

    mocker.patch.object(metadata_enricher, '_compute_image_stats', return_value=metadata)
    await metadata_enricher._enrich_clean_image(img_info)

    assert img_info.metadata == metadata
    assert metadata_enricher._metadata_cache["test_sha1"] == metadata


async def test_compute_alignment_metadata(metadata_enricher, mocker):
    gt_img = ImageInfo(
        filename="gt_image.raf",
        sha1="gt_sha1",
        is_clean=True,
        scene_name="test_scene",
        scene_images=[],
        cfa_type="Bayer"
    )
    noisy_img = ImageInfo(
        filename="noisy_image.raf",
        sha1="noisy_sha1",
        is_clean=False,
        scene_name="test_scene",
        scene_images=[],
        cfa_type="Bayer"
    )

    metadata = {
        "alignment": [0, 0],
        "raw_gain": None,
        "rgb_gain": None
    }

    compute_sync_mock = mocker.patch.object(metadata_enricher, '_compute_alignment_metadata')
    compute_sync_mock.return_value = metadata

    result = await metadata_enricher._compute_alignment_metadata(gt_img, noisy_img)
    assert result == metadata


async def test_compute_crops_list(metadata_enricher, mocker):
    scene_info = SceneInfo(
        scene_name="test_scene",
        cfa_type="Bayer",
        unknown_sensor=False,
        test_reserve=False
    )
    gt_img = ImageInfo(
        filename="gt_image.cr2",
        sha1="gt_sha1",
        is_clean=True,
        scene_name="test_scene",
        scene_images=[],
        cfa_type="Bayer"
    )
    noisy_img = ImageInfo(
        filename="noisy_image.cr2",
        sha1="noisy_sha1",
        is_clean=False,
        scene_name="test_scene",
        scene_images=[],
        cfa_type="Bayer"
    )

    crops_metadata = [
        {"coordinates": [0, 0], "f_linrec2020_fpath": "path/to/crop", "gt_linrec2020_fpath": "path/to/gt"}]

    fetch_crops_sync_mock = mocker.patch.object(metadata_enricher, '_compute_crops_list')
    fetch_crops_sync_mock.return_value = crops_metadata

    result = await metadata_enricher._compute_crops_list(scene_info, gt_img, noisy_img)
    assert result == crops_metadata


if __name__ == "__main__":
    pytest.main()
