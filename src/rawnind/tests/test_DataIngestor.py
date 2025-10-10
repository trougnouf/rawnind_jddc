from pathlib import Path

import pytest
import trio
import yaml

from rawnind.dataset.DataIngestor import DataIngestor
from rawnind.dataset.SceneInfo import SceneInfo

pytestmark = pytest.mark.dataset


@pytest.mark.trio
async def test_produce_scenes(tmp_path):
    """Test that produce_scenes yields SceneInfo objects."""
    # Arrange
    yaml_cache = tmp_path / "test_yaml_cache.yaml"
    metadata_cache = tmp_path / "test_metadata_cache.json"

    # Create mock cache data
    dataset_data = {
        "bayer": {
            "scene1": {
                "clean_images": [{"filename": "img1.png", "sha1": "abc123"}],
                "noisy_images": [{"filename": "img2.png", "sha1": "def456"}],
                "unknown_sensor": False,
                "test_reserve": False
            }
        }
    }

    with open(yaml_cache, "w") as f:
        yaml.dump(dataset_data, f)

    with open(metadata_cache, "w") as f:
        f.write('{"data": {"latestVersion": {"files": []}}}')

    send_channel, receive_channel = trio.open_memory_channel(10)
    data_ingestor = DataIngestor(
        cache_paths=(yaml_cache, metadata_cache),
        dataset_root=tmp_path
    )

    # Act & Assert
    async with trio.open_nursery() as nursery:
        nursery.start_soon(data_ingestor.produce_scenes, send_channel)

        async with receive_channel:
            scene_info = await receive_channel.receive()

            assert scene_info.scene_name == "scene1"
            assert scene_info.cfa_type == "bayer"
            assert len(scene_info.clean_images) == 1
            assert len(scene_info.noisy_images) == 1
            assert scene_info.clean_images[0].filename == "img1.png"
            assert scene_info.noisy_images[0].filename == "img2.png"

            nursery.cancel_scope.cancel()


@pytest.mark.trio
async def test_load_index_from_cache(tmp_path):
    """Test that _load_index loads from cache when available."""
    # Arrange
    yaml_cache = tmp_path / "test_yaml_cache.yaml"
    metadata_cache = tmp_path / "test_metadata_cache.json"

    dataset_data = {"bayer": {"scene1": {"clean_images": [], "noisy_images": []}}}

    with open(yaml_cache, "w") as f:
        yaml.dump(dataset_data, f)

    with open(metadata_cache, "w") as f:
        f.write('{}')

    data_ingestor = DataIngestor(
        cache_paths=(yaml_cache, metadata_cache),
        dataset_root=tmp_path
    )

    # Act
    index_data = await data_ingestor._load_index()

    # Assert
    assert index_data == dataset_data


@pytest.mark.trio
async def test_fetch_remote_index(tmp_path, mocker):
    """Test that _fetch_remote_index downloads and caches data."""
    # Arrange
    yaml_cache = tmp_path / "test_yaml_cache.yaml"
    metadata_cache = tmp_path / "test_metadata_cache.json"

    mock_yaml_data = {"bayer": {"scene1": {"clean_images": [], "noisy_images": []}}}
    mock_metadata = '{"data": {"latestVersion": {"files": []}}}'

    def mock_fetch_yaml():
        import yaml
        return mock_yaml_data

    def mock_fetch_metadata():
        return mock_metadata

    mocker.patch("trio.to_thread.run_sync", side_effect=[mock_yaml_data, mock_metadata])

    data_ingestor = DataIngestor(
        cache_paths=(yaml_cache, metadata_cache),
        dataset_root=tmp_path
    )

    # Act
    index_data = await data_ingestor._fetch_remote_index()

    # Assert
    assert index_data == mock_yaml_data
    assert yaml_cache.exists()
    assert metadata_cache.exists()


@pytest.mark.trio
async def test_xmp_files_excluded_from_image_lists(tmp_path):
    """Test that .xmp files are excluded from clean_images and noisy_images lists."""
    # Arrange
    yaml_cache = tmp_path / "test_yaml_cache.yaml"
    metadata_cache = tmp_path / "test_metadata_cache.json"

    dataset_data = {
        "bayer": {
            "scene1": {
                "clean_images": [
                    {"filename": "gt_image.cr2", "sha1": "abc123"},
                    {"filename": "gt_image.cr2.xmp", "sha1": "xmp111"}
                ],
                "noisy_images": [
                    {"filename": "noisy_image.cr2", "sha1": "def456"},
                    {"filename": "noisy_image.cr2.xmp", "sha1": "xmp222"}
                ],
                "unknown_sensor": False,
                "test_reserve": False
            }
        }
    }

    with open(yaml_cache, "w") as f:
        yaml.dump(dataset_data, f)

    with open(metadata_cache, "w") as f:
        f.write('{"data": {"latestVersion": {"files": []}}}')

    send_channel, receive_channel = trio.open_memory_channel(10)
    data_ingestor = DataIngestor(
        cache_paths=(yaml_cache, metadata_cache),
        dataset_root=tmp_path
    )

    # Act & Assert
    async with trio.open_nursery() as nursery:
        nursery.start_soon(data_ingestor.produce_scenes, send_channel)

        async with receive_channel:
            scene_info = await receive_channel.receive()

            # .xmp files should NOT be in the image lists
            assert len(scene_info.clean_images) == 1
            assert len(scene_info.noisy_images) == 1
            assert scene_info.clean_images[0].filename == "gt_image.cr2"
            assert scene_info.noisy_images[0].filename == "noisy_image.cr2"
            
            # .xmp filename should be stored in metadata
            assert scene_info.clean_images[0].metadata.get("xmp_filename") == "gt_image.cr2.xmp"
            assert scene_info.noisy_images[0].metadata.get("xmp_filename") == "noisy_image.cr2.xmp"

            nursery.cancel_scope.cancel()


@pytest.mark.trio
async def test_xmp_files_matched_to_parent_images(tmp_path):
    """Test that .xmp files are correctly matched to their parent images."""
    # Arrange
    yaml_cache = tmp_path / "test_yaml_cache.yaml"
    metadata_cache = tmp_path / "test_metadata_cache.json"

    dataset_data = {
        "bayer": {
            "scene1": {
                "clean_images": [
                    {"filename": "image1.arw", "sha1": "sha1_1"},
                    {"filename": "image1.arw.xmp", "sha1": "xmp_1"},
                    {"filename": "image2.arw", "sha1": "sha1_2"},
                    # image2 has no .xmp file
                ],
                "noisy_images": [
                    {"filename": "noisy1.arw", "sha1": "sha1_3"},
                    {"filename": "noisy1.arw.xmp", "sha1": "xmp_3"},
                ],
                "unknown_sensor": False,
                "test_reserve": False
            }
        }
    }

    with open(yaml_cache, "w") as f:
        yaml.dump(dataset_data, f)

    with open(metadata_cache, "w") as f:
        f.write('{"data": {"latestVersion": {"files": []}}}')

    send_channel, receive_channel = trio.open_memory_channel(10)
    data_ingestor = DataIngestor(
        cache_paths=(yaml_cache, metadata_cache),
        dataset_root=tmp_path
    )

    # Act & Assert
    async with trio.open_nursery() as nursery:
        nursery.start_soon(data_ingestor.produce_scenes, send_channel)

        async with receive_channel:
            scene_info = await receive_channel.receive()

            # Should have 2 clean images and 1 noisy image (no .xmp in lists)
            assert len(scene_info.clean_images) == 2
            assert len(scene_info.noisy_images) == 1
            
            # image1.arw should have xmp metadata
            img1 = next(img for img in scene_info.clean_images if img.filename == "image1.arw")
            assert img1.metadata.get("xmp_filename") == "image1.arw.xmp"
            
            # image2.arw should NOT have xmp metadata
            img2 = next(img for img in scene_info.clean_images if img.filename == "image2.arw")
            assert "xmp_filename" not in img2.metadata
            
            # noisy1.arw should have xmp metadata
            noisy1 = scene_info.noisy_images[0]
            assert noisy1.metadata.get("xmp_filename") == "noisy1.arw.xmp"

            nursery.cancel_scope.cancel()
