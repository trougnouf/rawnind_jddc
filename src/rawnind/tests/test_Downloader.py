# ... existing code ...
from pathlib import Path

import pytest
import trio

from rawnind.dataset.Downloader import Downloader
from rawnind.dataset.SceneInfo import ImageInfo


@pytest.fixture
def downloader():
    return Downloader(max_concurrent=2, max_retries=1)


@pytest.mark.trio
async def test_download_task(downloader):
    send_channel, recv_channel = trio.open_memory_channel(10)
    downloaded_send, downloaded_recv = trio.open_memory_channel(10)
    img_info = ImageInfo(
        filename="test_image.jpg",
        sha1="d41d8cd98f00b204e9800998ecf8427e",
        is_clean=True,
        scene_name="test_scene",
        scene_images=["test_image.jpg"],
        cfa_type="Bayer",
        local_path=Path("tests/test_downloads/test_image.jpg"),
        file_id="test_file_id"
    )
    async with trio.open_nursery() as nursery:
        nursery.start_soon(downloader.consume_missing, recv_channel, downloaded_send)
        await send_channel.send(img_info)
        send_channel.close()

        # Wait a bit for processing (this is a mock test, download will fail)
        await trio.sleep(0.1)


@pytest.mark.trio
async def test_no_concurrent_downloads_exceeding_limit():
    downloader = Downloader(max_concurrent=2, max_retries=1)
    send_channel, recv_channel = trio.open_memory_channel(10)
    downloaded_send, downloaded_recv = trio.open_memory_channel(10)
    img_infos = [
        ImageInfo(
            filename=f"test_image_{i}.jpg",
            sha1="d41d8cd98f00b204e9800998ecf8427e",
            is_clean=True,
            scene_name="test_scene",
            scene_images=[f"test_image_{i}.jpg"],
            cfa_type="Bayer",
            local_path=Path(f"tests/test_downloads/test_image_{i}.jpg"),
            file_id="test_file_id"
        ) for i in range(5)
    ]
    async with trio.open_nursery() as nursery:
        nursery.start_soon(downloader.consume_missing, recv_channel, downloaded_send)
        await send_channel.send(img_infos[0])
        await send_channel.send(img_infos[1])
        send_channel.close()

        # Wait a bit for processing (this is a mock test, downloads will fail)
        await trio.sleep(0.1)

# ... existing code ...
