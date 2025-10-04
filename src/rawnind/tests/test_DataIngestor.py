from pathlib import Path

import pytest
import trio
import yaml

from rawnind.dataset.DataIngestor import DataIngestor
from rawnind.dataset.SceneInfo import SceneInfo


@pytest.mark.trio
async def test_produce_scenes(tmp_path):
    """Test that produce_scenes yields SceneInfo objects."""
    # Arrange
    yaml_cache = tmp_path / "test_yaml_cache.yaml"
    metadata_cache = tmp_path / "test_metadata_cache.json"

    # Create mock cache data
    dataset_data = {
        "Bayer": {
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
            assert scene_info.cfa_type == "Bayer"
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

    dataset_data = {"Bayer": {"scene1": {"clean_images": [], "noisy_images": []}}}

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

    mock_yaml_data = {"Bayer": {"scene1": {"clean_images": [], "noisy_images": []}}}
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
