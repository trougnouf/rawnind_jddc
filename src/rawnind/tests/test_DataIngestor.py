from pathlib import Path

import pytest
import trio

from rawnind.dataset.DataIngestor import DataIngestor


@pytest.mark.asyncio
async def test_produce_scenes(mocker):
    # Arrange
    cache_paths = (Path("test_yaml_cache.yaml"), Path("test_metadata_cache.json"))
    dataset_root = Path("test_dataset")
    mocker.patch(
        "rawnind.dataset.DataIngestor._load_index",
        new_callable=mocker.AsyncMock,
        return_value={
            "Bayer": {
                "scene1": {
                    "clean_images": [{"filename": "img1.png", "sha1": "abc"}],
                    "noisy_images": [{"filename": "img2.png", "sha1": "def"}],
                }
            }
        },
    )

    send_channel, receive_channel = trio.open_memory_channel(0)

    data_ingestor = DataIngestor(cache_paths=cache_paths, dataset_root=dataset_root)
    produce_task = asyncio.create_task(data_ingestor.produce_scenes(send_channel))

    # Act
    scene_info = await receive_channel.receive()

    # Assert
    assert scene_info.scene_name == "scene1"
    assert len(scene_info.clean_images) == 1
    assert len(scene_info.noisy_images) == 1

    await produce_task


@pytest.mark.asyncio
async def test_load_index_from_cache(mocker):
    # Arrange
    cache_paths = (Path("test_yaml_cache.yaml"), Path("test_metadata_cache.json"))
    dataset_root = Path("test_dataset")
    mocker.patch(
        "rawnind.dataset.DataIngestor._fetch_remote_index",
        new_callable=mocker.AsyncMock,
        return_value={"Bayer": {"scene1": {}}},
    )

    data_ingestor = DataIngestor(cache_paths=cache_paths, dataset_root=dataset_root)
    with open("test_yaml_cache.yaml", "w") as f:
        yaml.dump({"Bayer": {"scene1": {}}}, f)

    # Act
    index_data = await data_ingestor._load_index()

    # Assert
    assert index_data == {"Bayer": {"scene1": {}}}


@pytest.mark.asyncio
async def test_fetch_remote_index(mocker):
    # Arrange
    cache_paths = (Path("test_yaml_cache.yaml"), Path("test_metadata_cache.json"))
    dataset_root = Path("test_dataset")
    mock_yaml_data = {"Bayer": {"scene1": {}}}
    mocker.patch(
        "rawnind.dataset.DataIngestor.fetch_yaml",
        return_value=mock_yaml_data,
    )
    mocker.patch(
        "rawnind.dataset.DataIngestor.fetch_metadata",
        return_value='{"data": {"latestVersion": {"files": []}}}',
    )

    data_ingestor = DataIngestor(cache_paths=cache_paths, dataset_root=dataset_root)

    # Act
    index_data = await data_ingestor._fetch_remote_index()

    # Assert
    assert index_data == mock_yaml_data
