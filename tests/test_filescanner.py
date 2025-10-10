"""
Unit tests for FileScanner - async filesystem scanner for dataset files.
"""

import os
import tempfile
from pathlib import Path
import shutil

import pytest
import trio

from rawnind.dataset.FileScanner import FileScanner
from rawnind.dataset.SceneInfo import ImageInfo, SceneInfo


@pytest.fixture
async def temp_dataset():
    """Create temporary dataset structure."""
    test_dir = tempfile.mkdtemp(prefix="test_dataset_")
    dataset_root = Path(test_dir)

    # Create Bayer scene structure
    bayer_scene = dataset_root / "bayer" / "TestScene"
    bayer_gt = bayer_scene / "gt"
    bayer_gt.mkdir(parents=True, exist_ok=True)

    # Create some test files
    (bayer_gt / "clean_001.cr2").write_bytes(b"fake clean")
    (bayer_scene / "noisy_001.cr2").write_bytes(b"fake noisy")
    (bayer_scene / "noisy_002.cr2").write_bytes(b"fake noisy 2")

    yield dataset_root

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


@pytest.mark.trio
async def test_basic_initialization(temp_dataset):
    """Test scanner can be initialized."""
    scanner = FileScanner(temp_dataset)
    assert scanner.dataset_root == temp_dataset


@pytest.mark.trio
async def test_scan_finds_existing_files(temp_dataset):
    """Test scanner finds existing files and sends them to new_file_send."""
    scanner = FileScanner(temp_dataset)

    # Create scene with images that exist
    clean_img = ImageInfo(
        filename="clean_001.cr2",
        sha1="abc",
        is_clean=True,
        scene_name="TestScene",
        scene_images=["abc"],
        cfa_type="bayer",
        file_id="123"
    )

    noisy_img = ImageInfo(
        filename="noisy_001.cr2",
        sha1="def",
        is_clean=False,
        scene_name="TestScene",
        scene_images=["abc", "def"],
        cfa_type="bayer",
        file_id="456"
    )

    scene = SceneInfo(
        scene_name="TestScene",
        cfa_type="bayer",
        unknown_sensor=False,
        test_reserve=False,
        clean_images=[clean_img],
        noisy_images=[noisy_img]
    )

    # Setup channels
    recv_send, recv_recv = trio.open_memory_channel(10)
    new_send, new_recv = trio.open_memory_channel(10)
    missing_send, missing_recv = trio.open_memory_channel(10)

    async with recv_send:
        await recv_send.send(scene)

    async with trio.open_nursery() as nursery:
        nursery.start_soon(scanner.consume_new_items, recv_recv, new_send, missing_send)

    # Collect results
    found = []
    async with new_recv:
        async for img in new_recv:
            found.append(img)

    missing = []
    async with missing_recv:
        async for img in missing_recv:
            missing.append(img)

    # Both files exist
    assert len(found) == 2
    assert len(missing) == 0

    # Verify local_path is set
    for img in found:
        assert img.local_path is not None
        assert img.local_path.exists()


@pytest.mark.trio
async def test_scan_reports_missing_files(temp_dataset):
    """Test scanner reports missing files to missing_send."""
    scanner = FileScanner(temp_dataset)

    # Create scene with image that doesn't exist
    missing_img = ImageInfo(
        filename="does_not_exist.cr2",
        sha1="xyz",
        is_clean=False,
        scene_name="TestScene",
        scene_images=["xyz"],
        cfa_type="bayer",
        file_id="789"
    )

    scene = SceneInfo(
        scene_name="TestScene",
        cfa_type="bayer",
        unknown_sensor=False,
        test_reserve=False,
        clean_images=[],
        noisy_images=[missing_img]
    )

    recv_send, recv_recv = trio.open_memory_channel(10)
    new_send, new_recv = trio.open_memory_channel(10)
    missing_send, missing_recv = trio.open_memory_channel(10)

    async with recv_send:
        await recv_send.send(scene)

    async with trio.open_nursery() as nursery:
        nursery.start_soon(scanner.consume_new_items, recv_recv, new_send, missing_send)

    found = []
    async with new_recv:
        async for img in new_recv:
            found.append(img)

    missing = []
    async with missing_recv:
        async for img in missing_recv:
            missing.append(img)

    assert len(found) == 0
    assert len(missing) == 1
    assert missing[0].filename == "does_not_exist.cr2"


@pytest.mark.trio
async def test_scan_multiple_candidates_prefers_gt_dir(temp_dataset):
    """Test that clean images prefer gt/ directory."""
    scanner = FileScanner(temp_dataset)

    # Create file in both gt/ and scene root
    scene_dir = temp_dataset / "bayer" / "TestScene"
    (scene_dir / "clean_001.cr2").write_bytes(b"in root")

    clean_img = ImageInfo(
        filename="clean_001.cr2",
        sha1="abc",
        is_clean=True,
        scene_name="TestScene",
        scene_images=["abc"],
        cfa_type="bayer",
        file_id="123"
    )

    scene = SceneInfo(
        scene_name="TestScene",
        cfa_type="bayer",
        unknown_sensor=False,
        test_reserve=False,
        clean_images=[clean_img],
        noisy_images=[]
    )

    recv_send, recv_recv = trio.open_memory_channel(10)
    new_send, new_recv = trio.open_memory_channel(10)
    missing_send, missing_recv = trio.open_memory_channel(10)

    async with recv_send:
        await recv_send.send(scene)

    async with trio.open_nursery() as nursery:
        nursery.start_soon(scanner.consume_new_items, recv_recv, new_send, missing_send)

    found = []
    async with new_recv:
        async for img in new_recv:
            found.append(img)

    async with missing_recv:
        pass  # Drain

    assert len(found) == 1
    # Should prefer gt/ directory
    assert found[0].local_path == scene_dir / "gt" / "clean_001.cr2"


@pytest.mark.trio
async def test_scan_noisy_images_only_check_scene_root(temp_dataset):
    """Test that noisy images only check scene root directory."""
    scanner = FileScanner(temp_dataset)

    noisy_img = ImageInfo(
        filename="noisy_001.cr2",
        sha1="def",
        is_clean=False,
        scene_name="TestScene",
        scene_images=["def"],
        cfa_type="bayer",
        file_id="456"
    )

    scene = SceneInfo(
        scene_name="TestScene",
        cfa_type="bayer",
        unknown_sensor=False,
        test_reserve=False,
        clean_images=[],
        noisy_images=[noisy_img]
    )

    recv_send, recv_recv = trio.open_memory_channel(10)
    new_send, new_recv = trio.open_memory_channel(10)
    missing_send, missing_recv = trio.open_memory_channel(10)

    async with recv_send:
        await recv_send.send(scene)

    async with trio.open_nursery() as nursery:
        nursery.start_soon(scanner.consume_new_items, recv_recv, new_send, missing_send)

    found = []
    async with new_recv:
        async for img in new_recv:
            found.append(img)

    async with missing_recv:
        pass

    assert len(found) == 1
    assert found[0].local_path == temp_dataset / "bayer" / "TestScene" / "noisy_001.cr2"


@pytest.mark.trio
async def test_scan_processes_multiple_scenes(temp_dataset):
    """Test scanner handles multiple scenes."""
    scanner = FileScanner(temp_dataset)

    # Create second scene
    scene2_dir = temp_dataset / "bayer" / "Scene2"
    scene2_dir.mkdir(parents=True, exist_ok=True)
    (scene2_dir / "noisy_005.cr2").write_bytes(b"scene 2")

    img1 = ImageInfo(
        filename="noisy_001.cr2",
        sha1="a",
        is_clean=False,
        scene_name="TestScene",
        scene_images=["a"],
        cfa_type="bayer",
        file_id="1"
    )

    img2 = ImageInfo(
        filename="noisy_005.cr2",
        sha1="b",
        is_clean=False,
        scene_name="Scene2",
        scene_images=["b"],
        cfa_type="bayer",
        file_id="2"
    )

    scene1 = SceneInfo("TestScene", "bayer", False, False, [], [img1])
    scene2 = SceneInfo("Scene2", "bayer", False, False, [], [img2])

    recv_send, recv_recv = trio.open_memory_channel(10)
    new_send, new_recv = trio.open_memory_channel(10)
    missing_send, missing_recv = trio.open_memory_channel(10)

    async with recv_send:
        await recv_send.send(scene1)
        await recv_send.send(scene2)

    async with trio.open_nursery() as nursery:
        nursery.start_soon(scanner.consume_new_items, recv_recv, new_send, missing_send)

    found = []
    async with new_recv:
        async for img in new_recv:
            found.append(img)

    async with missing_recv:
        pass

    assert len(found) == 2
    filenames = {img.filename for img in found}
    assert filenames == {"noisy_001.cr2", "noisy_005.cr2"}