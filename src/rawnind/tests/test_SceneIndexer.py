from pathlib import Path

import pytest
import trio

from rawnind.dataset.SceneIndexer import SceneIndexer
from rawnind.dataset.SceneInfo import ImageInfo, SceneInfo


@pytest.mark.asyncio
async def test_consume_images_produce_scenes():
    dataset_root = Path("/fake/path")
    indexer = SceneIndexer(dataset_root)

    image_send_channel, image_recv_channel = trio.open_memory_channel(10)
    scene_recv_channel, scene_send_channel = trio.open_memory_channel(10)

    # Create a mock ImageInfo object
    img_info = ImageInfo(
        filename="test_image.png",
        sha1="abcdef1234567890",
        is_clean=True,
        scene_name="scene_1",
        scene_images=["abcdef1234567890", "fedcba0987654321"],
        cfa_type="Bayer"
    )

    # Run the consumer-producer task
    async def run_indexer():
        await indexer.consume_images_produce_scenes(image_recv_channel, scene_send_channel)

    async with trio.open_nursery() as nursery:
        nursery.start_soon(run_indexer)
        await image_send_channel.send(img_info)

        # Check if the scene is sent
        scene = await scene_recv_channel.receive()
        assert isinstance(scene, SceneInfo)
        assert scene.scene_name == "scene_1"
        assert len(scene.clean_images) == 1


@pytest.mark.asyncio
async def test_iter_complete_scenes():
    dataset_root = Path("/fake/path")
    indexer = SceneIndexer(dataset_root)

    image_send_channel, image_recv_channel = trio.open_memory_channel(10)
    scene_recv_channel, scene_send_channel = trio.open_memory_channel(10)

    # Create mock ImageInfo objects
    img_info_1 = ImageInfo(
        filename="test_image_1.png",
        sha1="abcdef1234567890",
        is_clean=True,
        scene_name="scene_1",
        scene_images=["abcdef1234567890", "fedcba0987654321"],
        cfa_type="Bayer"
    )

    img_info_2 = ImageInfo(
        filename="test_image_2.png",
        sha1="fedcba0987654321",
        is_clean=False,
        scene_name="scene_1",
        scene_images=["abcdef1234567890", "fedcba0987654321"],
        cfa_type="Bayer"
    )

    # Run the consumer-producer task
    async def run_indexer():
        await indexer.consume_images_produce_scenes(image_recv_channel, scene_send_channel)

    async with trio.open_nursery() as nursery:
        nursery.start_soon(run_indexer)
        await image_send_channel.send(img_info_1)
        await image_send_channel.send(img_info_2)

        # Check if the scenes are complete and sent
        scene = await scene_recv_channel.receive()
        assert isinstance(scene, SceneInfo)
        assert scene.scene_name == "scene_1"
        assert len(scene.clean_images) == 1

    # Iterate over complete scenes
    scenes = list(indexer.iter_complete_scenes())
    assert len(scenes) == 1
    assert scenes[0].scene_name == "scene_1"


@pytest.mark.asyncio
async def test_iter_incomplete_images():
    dataset_root = Path("/fake/path")
    indexer = SceneIndexer(dataset_root)

    image_send_channel, image_recv_channel = trio.open_memory_channel(10)
    scene_recv_channel, scene_send_channel = trio.open_memory_channel(10)

    # Create a mock ImageInfo object for an incomplete scene
    img_info_1 = ImageInfo(
        filename="test_image_incomplete.png",
        sha1="abcdef1234567890",
        is_clean=True,
        scene_name="scene_incomplete",
        scene_images=["abcdef1234567890", "fedcba0987654321"],
        cfa_type="Bayer"
    )

    # Run the consumer-producer task
    async def run_indexer():
        await indexer.consume_images_produce_scenes(image_recv_channel, scene_send_channel)

    async with trio.open_nursery() as nursery:
        nursery.start_soon(run_indexer)
        await image_send_channel.send(img_info_1)

    # Check incomplete images
    incomplete_images = list(indexer.iter_incomplete_images())
    assert len(incomplete_images) == 1
    assert incomplete_images[0].filename == "test_image_incomplete.png"


@pytest.mark.asyncio
async def test_scene_completion_tracking():
    dataset_root = Path("/fake/path")
    indexer = SceneIndexer(dataset_root)

    image_send_channel, image_recv_channel = trio.open_memory_channel(10)
    scene_recv_channel, scene_send_channel = trio.open_memory_channel(10)

    # Create mock ImageInfo objects
    img_info_1 = ImageInfo(
        filename="test_image_1.png",
        sha1="abcdef1234567890",
        is_clean=True,
        scene_name="scene_tracking",
        scene_images=["abcdef1234567890", "fedcba0987654321"],
        cfa_type="Bayer"
    )

    img_info_2 = ImageInfo(
        filename="test_image_2.png",
        sha1="fedcba0987654321",
        is_clean=False,
        scene_name="scene_tracking",
        scene_images=["abcdef1234567890", "fedcba0987654321"],
        cfa_type="Bayer"
    )

    # Run the consumer-producer task
    async def run_indexer():
        await indexer.consume_images_produce_scenes(image_recv_channel, scene_send_channel)

    async with trio.open_nursery() as nursery:
        nursery.start_soon(run_indexer)
        await image_send_channel.send(img_info_1)
        await image_send_channel.send(img_info_2)

        # Check if the scenes are complete and sent
        scene = await scene_recv_channel.receive()
        assert isinstance(scene, SceneInfo)
        assert scene.scene_name == "scene_tracking"
        assert len(scene.clean_images) == 1

    # Check tracking of scene completion
    assert (img_info_1.cfa_type, img_info_1.scene_name) in indexer._scene_completion_tracker
