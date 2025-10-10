"""
Simple unit tests for Downloader - testing core functionality without complex mocking.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import shutil

import pytest
import trio

from rawnind.dataset.Downloader import Downloader
from rawnind.dataset.SceneInfo import ImageInfo


@pytest.fixture
async def temp_dir():
    """Provide temporary directory."""
    test_dir = tempfile.mkdtemp(prefix="test_download_")
    yield Path(test_dir)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


@pytest.mark.trio
async def test_basic_initialization():
    """Test downloader can be initialized with parameters."""
    downloader = Downloader(max_concurrent=5, max_retries=3, progress=False)
    assert downloader.max_concurrent == 5
    assert downloader.max_retries == 3
    assert downloader.progress == False


@pytest.mark.trio
async def test_download_with_retry_mock(temp_dir):
    """Test _download_with_retry method directly."""
    downloader = Downloader(max_concurrent=1, max_retries=3, progress=False)

    img_info = ImageInfo(
        filename="test.cr2",
        sha1="abc",
        is_clean=True,
        scene_name="test",
        scene_images=["abc"],
        cfa_type="bayer",
        file_id="123"
    )
    img_info.local_path = temp_dir / "test.cr2"

    #Mock _download_file to succeed immediately
    with patch.object(downloader, '_download_file', new_callable=AsyncMock):
        with patch.object(ImageInfo, 'load_image', new_callable=AsyncMock):
            result = await downloader._download_with_retry(img_info)
            assert result == True


@pytest.mark.trio
async def test_download_retry_logic(temp_dir):
    """Test that retry logic attempts multiple times."""
    downloader = Downloader(max_concurrent=1, max_retries=3, progress=False)

    img_info = ImageInfo(
        filename="test.cr2",
        sha1="abc",
        is_clean=True,
        scene_name="test",
        scene_images=["abc"],
        cfa_type="bayer",
        file_id="123"
    )
    img_info.local_path = temp_dir / "test.cr2"

    # Mock _download_file to fail twice then succeed
    call_count = [0]
    async def mock_download(*args):
        call_count[0] += 1
        if call_count[0] < 3:
            raise Exception("Fail")
        return None

    with patch.object(downloader, '_download_file', side_effect=mock_download):
        with patch.object(ImageInfo, 'load_image', new_callable=AsyncMock):
            result = await downloader._download_with_retry(img_info)
            assert result == True
            assert call_count[0] == 3


@pytest.mark.trio
async def test_download_max_retries_fail(temp_dir):
    """Test that download fails after max retries."""
    downloader = Downloader(max_concurrent=1, max_retries=2, progress=False)

    img_info = ImageInfo(
        filename="test.cr2",
        sha1="abc",
        is_clean=True,
        scene_name="test",
        scene_images=["abc"],
        cfa_type="bayer",
        file_id="123"
    )
    img_info.local_path = temp_dir / "test.cr2"

    # Mock to always fail
    async def mock_download(*args):
        raise Exception("Always fail")

    with patch.object(downloader, '_download_file', side_effect=mock_download):
        result = await downloader._download_with_retry(img_info)
        assert result == False


@pytest.mark.trio
async def test_download_calls_load_image(temp_dir):
    """Test that successful download calls load_image."""
    downloader = Downloader(max_concurrent=1, max_retries=1, progress=False)

    img_info = ImageInfo(
        filename="test.cr2",
        sha1="abc",
        is_clean=True,
        scene_name="test",
        scene_images=["abc"],
        cfa_type="bayer",
        file_id="123"
    )
    img_info.local_path = temp_dir / "test.cr2"

    with patch.object(downloader, '_download_file', new_callable=AsyncMock):
        with patch.object(ImageInfo, 'load_image', new_callable=AsyncMock) as mock_load:
            result = await downloader._download_with_retry(img_info)
            assert result == True
            mock_load.assert_called_once_with(as_torch=True)


@pytest.mark.trio
async def test_download_file_creates_directories(temp_dir):
    """Test that _download_file creates parent directories."""
    downloader = Downloader(max_concurrent=1, max_retries=1, progress=False)

    dest_path = temp_dir / "nested" / "dir" / "test.cr2"
    url = "https://example.com/test.cr2"

    # Create a proper async generator for aiter_bytes
    async def fake_bytes(chunk_size=8192):
        yield b"fake data"

    # Create properly structured mocks
    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.aiter_bytes = fake_bytes
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_client = AsyncMock()
    mock_client.stream = MagicMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch('rawnind.dataset.Downloader.httpx.AsyncClient', return_value=mock_client):
        await downloader._download_file(url, dest_path)
        assert dest_path.exists()
        assert dest_path.parent.exists()


@pytest.mark.trio
async def test_consume_missing_basic(temp_dir):
    """Test consume_missing processes images through channels."""
    downloader = Downloader(max_concurrent=2, max_retries=1, progress=False)

    # Create test images
    images = []
    for i in range(3):
        img = ImageInfo(
            filename=f"image_{i}.cr2",
            sha1=f"sha{i}",
            is_clean=True,
            scene_name="test",
            scene_images=[f"sha{i}"],
            cfa_type="bayer",
            file_id=str(i)
        )
        img.local_path = temp_dir / f"image_{i}.cr2"
        images.append(img)

    # Mock download methods
    with patch.object(downloader, '_download_file', new_callable=AsyncMock):
        with patch.object(ImageInfo, 'load_image', new_callable=AsyncMock):
            # Setup channels
            recv_send, recv_recv = trio.open_memory_channel(10)
            down_send, down_recv = trio.open_memory_channel(10)

            # Send images
            async with recv_send:
                for img in images:
                    await recv_send.send(img)

            # Process downloads
            async with trio.open_nursery() as nursery:
                nursery.start_soon(downloader.consume_missing, recv_recv, down_send)

                # Collect downloaded images
                downloaded = []
                async with down_recv:
                    async for img in down_recv:
                        downloaded.append(img)

            # Should have downloaded all 3
            assert len(downloaded) == 3
            assert set(img.filename for img in downloaded) == {f"image_{i}.cr2" for i in range(3)}


@pytest.mark.trio
async def test_consume_missing_with_progress(temp_dir):
    """Test consume_missing with progress bar enabled."""
    downloader = Downloader(max_concurrent=1, max_retries=1, progress=True)

    img = ImageInfo(
        filename="test.cr2",
        sha1="abc",
        is_clean=True,
        scene_name="test",
        scene_images=["abc"],
        cfa_type="bayer",
        file_id="123"
    )
    img.local_path = temp_dir / "test.cr2"

    with patch.object(downloader, '_download_file', new_callable=AsyncMock):
        with patch.object(ImageInfo, 'load_image', new_callable=AsyncMock):
            recv_send, recv_recv = trio.open_memory_channel(10)
            down_send, down_recv = trio.open_memory_channel(10)

            async with recv_send:
                await recv_send.send(img)

            async with trio.open_nursery() as nursery:
                nursery.start_soon(downloader.consume_missing, recv_recv, down_send)

                downloaded = []
                async with down_recv:
                    async for img_result in down_recv:
                        downloaded.append(img_result)

            assert len(downloaded) == 1


@pytest.mark.trio
async def test_consume_missing_concurrent_limiting(temp_dir):
    """Test that consume_missing respects max_concurrent limit."""
    downloader = Downloader(max_concurrent=2, max_retries=1, progress=False)

    # Track concurrent downloads
    active_downloads = []
    max_concurrent_seen = 0

    async def mock_download(*args):
        active_downloads.append(1)
        nonlocal max_concurrent_seen
        max_concurrent_seen = max(max_concurrent_seen, len(active_downloads))
        await trio.sleep(0.05)  # Simulate download time
        active_downloads.pop()

    # Create 5 images
    images = []
    for i in range(5):
        img = ImageInfo(
            filename=f"image_{i}.cr2",
            sha1=f"sha{i}",
            is_clean=True,
            scene_name="test",
            scene_images=[f"sha{i}"],
            cfa_type="bayer",
            file_id=str(i)
        )
        img.local_path = temp_dir / f"image_{i}.cr2"
        images.append(img)

    with patch.object(downloader, '_download_file', side_effect=mock_download):
        with patch.object(ImageInfo, 'load_image', new_callable=AsyncMock):
            recv_send, recv_recv = trio.open_memory_channel(10)
            down_send, down_recv = trio.open_memory_channel(10)

            async with recv_send:
                for img in images:
                    await recv_send.send(img)

            async with trio.open_nursery() as nursery:
                nursery.start_soon(downloader.consume_missing, recv_recv, down_send)

                downloaded = []
                async with down_recv:
                    async for img in down_recv:
                        downloaded.append(img)

            # Should have downloaded all 5
            assert len(downloaded) == 5

            # Should never exceed max_concurrent of 2
            assert max_concurrent_seen <= 2