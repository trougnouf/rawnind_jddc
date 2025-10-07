"""
Unit tests for Downloader - async file downloader with retry logic.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import shutil

import pytest
import trio
import httpx

from rawnind.dataset.Downloader import Downloader
from rawnind.dataset.SceneInfo import ImageInfo


@pytest.fixture
async def temp_dir():
    """
    Temporary directory fixture for testing.

    This fixture creates a temporary directory, yields it for use in tests,
    and then cleans up by removing the directory and its contents afterward.
    It is particularly useful for tests that require a temporary working
    directory to store files or other data during their execution.

    :return:
        Path: The path to the created temporary directory.
    """
    test_dir = tempfile.mkdtemp(prefix="test_download_")
    yield Path(test_dir)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


@pytest.fixture
def mock_image_info(temp_dir):
    """Create a mock ImageInfo for testing."""
    img_info = ImageInfo(
        filename="test_image.cr2",
        sha1="abc123",
        is_clean=True,
        scene_name="test_scene",
        scene_images=["abc123"],
        cfa_type="Bayer",
        file_id="12345"
    )
    img_info.local_path = temp_dir / "test_image.cr2"
    return img_info


@pytest.mark.trio
async def test_basic_initialization():
    """Test downloader can be initialized."""
    downloader = Downloader(max_concurrent=5, max_retries=3, progress=False)
    assert downloader.max_concurrent == 5
    assert downloader.max_retries == 3
    assert downloader.progress == False


@pytest.mark.trio
async def test_successful_download(temp_dir, mock_image_info):
    """Test successful file download."""
    downloader = Downloader(max_concurrent=1, max_retries=1, progress=False)

    # Mock httpx response
    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.aiter_bytes = AsyncMock(return_value=iter([b"fake image data"]))

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.stream = MagicMock(return_value=mock_response)
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    # Mock load_image to avoid actual RAW decoding
    with patch('httpx.AsyncClient', return_value=mock_client):
        with patch.object(ImageInfo, 'load_image', new_callable=AsyncMock) as mock_load:
            recv_send, recv_recv = trio.open_memory_channel(10)
            down_send, down_recv = trio.open_memory_channel(10)

            async with recv_send, down_recv:
                await recv_send.send(mock_image_info)

                async with trio.open_nursery() as nursery:
                    nursery.start_soon(downloader.consume_missing, recv_recv, down_send)

                    downloaded = await down_recv.receive()
                    assert downloaded.filename == "test_image.cr2"
                    assert downloaded.local_path.exists()

                    # Verify load_image was called
                    mock_load.assert_called_once()


@pytest.mark.trio
async def test_download_retry_on_failure(temp_dir, mock_image_info):
    """Test retry logic on download failure."""
    downloader = Downloader(max_concurrent=1, max_retries=3, progress=False)

    # Mock httpx to fail twice then succeed
    call_count = 0

    def create_mock_response():
        nonlocal call_count
        call_count += 1
        mock_response = AsyncMock()
        if call_count < 3:
            # Fail first 2 attempts
            mock_response.raise_for_status = MagicMock(side_effect=httpx.HTTPError("Fail"))
        else:
            # Succeed on 3rd attempt
            mock_response.raise_for_status = MagicMock()
            mock_response.aiter_bytes = AsyncMock(return_value=iter([b"success"]))

        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        return mock_response

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.stream = MagicMock(side_effect=create_mock_response)

    with patch('httpx.AsyncClient', return_value=mock_client):
        with patch.object(ImageInfo, 'load_image', new_callable=AsyncMock):
            recv_send, recv_recv = trio.open_memory_channel(10)
            down_send, down_recv = trio.open_memory_channel(10)

            async with recv_send, down_recv:
                await recv_send.send(mock_image_info)

                async with trio.open_nursery() as nursery:
                    nursery.start_soon(downloader.consume_missing, recv_recv, down_send)

                    downloaded = await down_recv.receive()
                    assert downloaded.local_path.exists()
                    assert call_count == 3


@pytest.mark.trio
async def test_download_max_retries_exceeded(temp_dir, mock_image_info):
    """Test that download fails after max retries."""
    downloader = Downloader(max_concurrent=1, max_retries=2, progress=False)

    # Mock httpx to always fail
    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock(side_effect=httpx.HTTPError("Always fail"))
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.stream = MagicMock(return_value=mock_response)

    with patch('httpx.AsyncClient', return_value=mock_client):
        recv_send, recv_recv = trio.open_memory_channel(10)
        down_send, down_recv = trio.open_memory_channel(10)

        async with recv_send:
            await recv_send.send(mock_image_info)

        async with trio.open_nursery() as nursery:
            nursery.start_soon(downloader.consume_missing, recv_recv, down_send)

        # No successful downloads sent
        async with down_recv:
            with trio.fail_after(0.1):
                try:
                    await down_recv.receive()
                    assert False, "Should not receive any downloads"
                except trio.EndOfChannel:
                    pass  # Expected


@pytest.mark.trio
async def test_concurrent_downloads(temp_dir):
    """Test concurrent downloads with semaphore."""
    downloader = Downloader(max_concurrent=3, max_retries=1, progress=False)

    # Create multiple mock images
    images = []
    for i in range(10):
        img = ImageInfo(
            filename=f"image_{i}.cr2",
            sha1=f"sha{i}",
            is_clean=True,
            scene_name="test",
            scene_images=[f"sha{i}"],
            cfa_type="Bayer",
            file_id=str(i)
        )
        img.local_path = temp_dir / f"image_{i}.cr2"
        images.append(img)

    # Mock httpx
    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.aiter_bytes = AsyncMock(return_value=iter([b"data"]))
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.stream = MagicMock(return_value=mock_response)

    with patch('httpx.AsyncClient', return_value=mock_client):
        with patch.object(ImageInfo, 'load_image', new_callable=AsyncMock):
            recv_send, recv_recv = trio.open_memory_channel(100)
            down_send, down_recv = trio.open_memory_channel(100)

            async with recv_send:
                for img in images:
                    await recv_send.send(img)

            async with trio.open_nursery() as nursery:
                nursery.start_soon(downloader.consume_missing, recv_recv, down_send)

                # Collect all downloads
                downloaded = []
                async with down_recv:
                    async for img in down_recv:
                        downloaded.append(img)

            assert len(downloaded) == 10


@pytest.mark.trio
async def test_download_creates_parent_directories(temp_dir, mock_image_info):
    """Test that download creates parent directories if they don't exist."""
    # Set path with non-existent parent
    mock_image_info.local_path = temp_dir / "subdir" / "nested" / "test.cr2"

    downloader = Downloader(max_concurrent=1, max_retries=1, progress=False)

    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.aiter_bytes = AsyncMock(return_value=iter([b"data"]))
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.stream = MagicMock(return_value=mock_response)

    with patch('httpx.AsyncClient', return_value=mock_client):
        with patch.object(ImageInfo, 'load_image', new_callable=AsyncMock):
            recv_send, recv_recv = trio.open_memory_channel(10)
            down_send, down_recv = trio.open_memory_channel(10)

            async with recv_send, down_recv:
                await recv_send.send(mock_image_info)

                async with trio.open_nursery() as nursery:
                    nursery.start_soon(downloader.consume_missing, recv_recv, down_send)

                    downloaded = await down_recv.receive()
                    assert downloaded.local_path.exists()
                    assert downloaded.local_path.parent.exists()


@pytest.mark.trio
async def test_download_calls_load_image(temp_dir, mock_image_info):
    """Test that download calls load_image for opportunistic caching."""
    downloader = Downloader(max_concurrent=1, max_retries=1, progress=False)

    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.aiter_bytes = AsyncMock(return_value=iter([b"data"]))
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.stream = MagicMock(return_value=mock_response)

    with patch('httpx.AsyncClient', return_value=mock_client):
        with patch.object(ImageInfo, 'load_image', new_callable=AsyncMock) as mock_load:
            recv_send, recv_recv = trio.open_memory_channel(10)
            down_send, down_recv = trio.open_memory_channel(10)

            async with recv_send, down_recv:
                await recv_send.send(mock_image_info)

                async with trio.open_nursery() as nursery:
                    nursery.start_soon(downloader.consume_missing, recv_recv, down_send)

                    await down_recv.receive()

                    # Verify load_image was called with as_torch=True
                    mock_load.assert_called_once_with(as_torch=True)