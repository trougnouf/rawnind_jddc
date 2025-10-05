from pathlib import Path

import pytest
import trio

from rawnind.dataset.FileScanner import FileScanner
from rawnind.dataset.SceneInfo import SceneInfo, ImageInfo

pytestmark = pytest.mark.dataset


@pytest.fixture
def dataset_root(tmp_path):
    return tmp_path


@pytest.fixture
async def file_scanner(dataset_root):
    scanner = FileScanner(dataset_root)
    yield scanner


class TestFileScanner:

    async def test_consume_new_items(self, file_scanner, dataset_root):
        # Setup
        scene_info = SceneInfo(
            cfa_type="test_cfa",
            scene_name="test_scene"
        )
        image_info = ImageInfo(
            filename="test_image.jpg",
            sha1="dummy_sha1",
            is_clean=True,
            file_id="dummy_file_id"
        )

        async def producer(channel):
            await channel.send(scene_info)

        # Create channels
        recv_channel, new_file_send = trio.open_memory_channel(0)
        _, missing_send = trio.open_memory_channel(0)

        # Start consumer task
        async with trio.open_nursery() as nursery:
            nursery.start_soon(file_scanner.consume_new_items, recv_channel, new_file_send, missing_send)
            nursery.start_soon(producer, recv_channel)

            # Check if the file is processed properly

            # Mock receiving ImageInfo from channel
            async with recv_channel, new_file_send, missing_send:
                scene_dir = dataset_root / "test_cfa" / "test_scene"
                gt_dir = scene_dir / "gt"

                await trio.sleep(0.1)  # Give some time for the nursery to process

                # Assert file exists check (this will be mocked in real tests)
                candidates = [
                    gt_dir / image_info.filename,
                    scene_dir / image_info.filename
                ]

                assert any(candidate.exists() for candidate in candidates)

    async def test_candidate_paths_for_scene(self, file_scanner):
        # Setup
        scene_dir = Path("/test/scene")
        gt_dir = scene_dir / "gt"
        image_info_clean = ImageInfo(
            filename="clean_image.jpg",
            sha1="dummy_sha1",
            is_clean=True,
            file_id="dummy_file_id"
        )
        image_info_noisy = ImageInfo(
            filename="noisy_image.jpg",
            sha1="dummy_sha1",
            is_clean=False,
            file_id="dummy_file_id"
        )

        # Test clean image
        candidates_clean = file_scanner._candidate_paths_for_scene(scene_dir, gt_dir, image_info_clean)
        assert len(candidates_clean) == 2
        assert (gt_dir / "clean_image.jpg") in candidates_clean
        assert (scene_dir / "clean_image.jpg") in candidates_clean

        # Test noisy image
        candidates_noisy = file_scanner._candidate_paths_for_scene(scene_dir, gt_dir, image_info_noisy)
        assert len(candidates_noisy) == 1
        assert (scene_dir / "noisy_image.jpg") in candidates_noisy
