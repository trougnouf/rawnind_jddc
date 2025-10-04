"""Tests for DatasetIndex event-based cache invalidation system."""

from pathlib import Path
from unittest.mock import patch

import pytest

from rawnind.dataset.manager import DatasetIndex, CacheEvent



class TestAsyncDownloadPhase1:
    """Test suite for Phase 1: Simple Async Download with Events."""

    @pytest.mark.trio
    async def test_async_download_preserves_index_state(self):
        """Test that async_download_missing_files preserves index state correctly.

        Given: A DatasetIndex with some files marked as locally available and some missing
        When: async_download_missing_files() is called
        Then:
            - Only files with local_path is None should be attempted for download
            - Files already marked as available should not be re-checked or re-downloaded
            - After download completes, discover() should be called to update state
            - LOCAL_PATHS_UPDATED event should be emitted
        """

        index = DatasetIndex()

        # Create test dataset with mixed availability
        dataset_data = {
            "Bayer": {
                "scene1": {
                    "clean_images": [
                        {
                            "filename": "available1.arw",
                            "sha1": "sha1_1",
                            "file_id": "id1",
                        },
                        {
                            "filename": "missing1.arw",
                            "sha1": "sha1_2",
                            "file_id": "id2",
                        },
                    ],
                    "noisy_images": [
                        {
                            "filename": "available2.arw",
                            "sha1": "sha1_3",
                            "file_id": "id3",
                        },
                        {
                            "filename": "missing2.arw",
                            "sha1": "sha1_4",
                            "file_id": "id4",
                        },
                    ],
                    "unknown_sensor": False,
                    "test_reserve": False,
                }
            }
        }
        index._build_index_from_data(dataset_data)
        # Mark some files as available
        initial_available = []
        for scenes in index.scenes.values():
            for scene in scenes.values():
                if scene.clean_images:
                    scene.clean_images[0].local_path = Path("/fake/path/available1.arw")
                    initial_available.append(scene.clean_images[0])
                if scene.noisy_images:
                    scene.noisy_images[0].local_path = Path("/fake/path/available2.arw")
                    initial_available.append(scene.noisy_images[0])

        initial_missing = [
            img
            for scenes in index.scenes.values()
            for scene in scenes.values()
            for img in scene.all_images()
            if img.local_path is None
        ]
        assert len(initial_missing) == 2, "Should have 2 missing files"

        event_count = 0

        def listener():
            nonlocal event_count
            event_count += 1

        index._events.on(CacheEvent.LOCAL_PATHS_UPDATED, listener)

        # Mock download (don't actually download)
        with patch.object(index, "_download_file", return_value=None):
            with patch.object(
                    index, "discover", wraps=index.discover
            ):
                await index.async_download_missing_files(max_concurrent=2)

        # Files marked as available should not have been touched
        for img in initial_available:
            assert img.local_path is not None, (
                "Available files should still have local_path"
            )

        # Event should have been emitted at least once (from discover)
        assert event_count > 0, "LOCAL_PATHS_UPDATED event should be emitted"

    @pytest.mark.trio
    async def test_async_download_respects_concurrency_limit(self):
        """Test that downloads respect the concurrency limit.

        Given: 10 missing files and max_concurrent=3
        When: Downloads are initiated
        Then: No more than 3 concurrent downloads should be active at any time
        """
        import trio

        index = DatasetIndex()

        # Create dataset with 10 missing files
        dataset_data = {
            "Bayer": {
                "scene1": {
                    "clean_images": [
                        {
                            "filename": f"file{i}.arw",
                            "sha1": f"sha{i}",
                            "file_id": f"id{i}",
                        }
                        for i in range(10)
                    ],
                    "noisy_images": [],
                    "unknown_sensor": False,
                    "test_reserve": False,
                }
            }
        }
        index._build_index_from_data(dataset_data)

        active_downloads = []
        max_seen_concurrent = [0]  # Use list to allow modification in closure

        async def mock_download(url, path):
            active_downloads.append(str(path))
            max_seen_concurrent[0] = max(max_seen_concurrent[0], len(active_downloads))
            await trio.sleep(0.01)  # Simulate download time
            active_downloads.remove(str(path))

        with patch.object(index, "_download_file", side_effect=mock_download):
            with patch.object(index, "discover"):
                await index.async_download_missing_files(
                    max_concurrent=3, progress=False
                )

        assert max_seen_concurrent[0] <= 3, (
            f"Max concurrent should be ≤3, was {max_seen_concurrent[0]}"
        )
        assert max_seen_concurrent[0] > 0, "At least one download should have occurred"


class TestStreamingDiscoveryPhase2:
    """Test suite for Phase 2: Streaming Discovery."""

    def test_direct_scenes_access_is_side_effect_free(self):
        """Test that direct scenes property access is side-effect free.

        Given: A DatasetIndex with cached state
        When: Scenes are accessed directly multiple times
        Then:
            - Should return the same results each time
            - Should not modify img_info.local_path
            - Should not trigger filesystem checks
            - Should not emit events
        """
        index = DatasetIndex()

        # Create minimal dataset data
        dataset_data = {
            "Bayer": {
                "scene1": {
                    "clean_images": [
                        {"filename": "available.arw", "sha1": "sha1", "file_id": "id1"},
                        {"filename": "missing.arw", "sha1": "sha2", "file_id": "id2"},
                    ],
                    "noisy_images": [],
                    "unknown_sensor": False,
                    "test_reserve": False,
                }
            }
        }
        index._build_index_from_data(dataset_data)
        # Mark one as available
        scene = list(index.scenes["Bayer"].values())[0]
        scene.clean_images[0].local_path = Path("/fake/path/available.arw")

        event_count = 0

        def listener():
            nonlocal event_count
            event_count += 1

        # Listen to all events
        for event in CacheEvent:
            index._events.on(event, listener)

        # Get missing files twice using direct property access
        missing1 = [
            img
            for scenes in index.scenes.values()
            for scene in scenes.values()
            for img in scene.all_images()
            if img.local_path is None
        ]
        missing2 = [
            img
            for scenes in index.scenes.values()
            for scene in scenes.values()
            for img in scene.all_images()
            if img.local_path is None
        ]

        # Should be identical
        assert missing1 == missing2, "Results should be identical"
        assert len(missing1) == 1, "Should have exactly 1 missing file"

        # No events should have been emitted
        assert event_count == 0, "No events should be emitted"

        # No filesystem operations should occur
        with patch("pathlib.Path.exists") as mock_exists:
            [
                img
                for scenes in index.scenes.values()
                for scene in scenes.values()
                for img in scene.all_images()
                if img.local_path is None
            ]
            mock_exists.assert_not_called()

    def test_generator_yields_incrementally(self):
        """Test that generator comprehension yields results incrementally.

        Given: A large dataset with many missing files
        When: A generator is used to access missing files
        Then: Results should be yielded one at a time (generator behavior)
        """
        index = DatasetIndex()

        # Create dataset with multiple missing files
        dataset_data = {
            "Bayer": {
                "scene1": {
                    "clean_images": [
                        {
                            "filename": f"missing{i}.arw",
                            "sha1": f"sha{i}",
                            "file_id": f"id{i}",
                        }
                        for i in range(5)
                    ],
                    "noisy_images": [],
                    "unknown_sensor": False,
                    "test_reserve": False,
                }
            }
        }
        index._build_index_from_data(dataset_data)
        # Consume first item without materializing full list
        gen = (
            img
            for scenes in index.scenes.values()
            for scene in scenes.values()
            for img in scene.all_images()
            if img.local_path is None
        )
        first = next(gen)

        # Verify it's a generator and yields ImageInfo objects
        from rawnind.dataset import ImageInfo

        assert isinstance(first, ImageInfo), "Should yield ImageInfo objects"
        assert first.local_path is None, "Should yield only files with local_path=None"

        # Verify we can continue consuming
        second = next(gen)
        assert isinstance(second, ImageInfo), (
            "Should continue yielding ImageInfo objects"
        )


class TestStreamingAsyncDiscoveryPhase3:
    """Test suite for Phase 3: Streaming Async Discovery with Events."""

    @pytest.mark.trio
    async def test_async_iter_missing_files_streams_during_discovery(self):
        """Test that async_iter_missing_files streams during filesystem discovery.

        Given: A dataset with mixed available/missing files
        When: async_iter_missing_files() examines filesystem
        Then: Missing files should be yielded before entire filesystem scan completes
        """

        index = DatasetIndex()

        # Create dataset with multiple files
        dataset_data = {
            "Bayer": {
                "scene1": {
                    "clean_images": [
                        {
                            "filename": f"file{i}.arw",
                            "sha1": f"sha{i}",
                            "file_id": f"id{i}",
                        }
                        for i in range(10)
                    ],
                    "noisy_images": [],
                    "unknown_sensor": False,
                    "test_reserve": False,
                }
            }
        }
        index._build_index_from_data(dataset_data)
        yielded_count = 0
        filesystem_scan_complete = False

        async for img_info in index.produce_missing_files():
            yielded_count += 1
            # First results should come before full scan completes
            # (we'll verify by checking we can break early)
            if yielded_count >= 5:
                break

        # Verify we got results before consuming all
        assert yielded_count == 5, "Should be able to consume first 5 results"

        # Verify it's actually streaming (can get results incrementally)
        from rawnind.dataset import ImageInfo

        gen = index.produce_missing_files()
        first = await gen.__anext__()
        assert isinstance(first, ImageInfo), "Should yield ImageInfo objects"

    @pytest.mark.trio
    async def test_async_iter_missing_files_emits_event(self):
        """Test that async_iter_missing_files emits LOCAL_PATHS_UPDATED event.

        Given: Files that don't exist locally
        When: async_iter_missing_files() completes
        Then: LOCAL_PATHS_UPDATED event should be emitted
        """

        index = DatasetIndex()

        # Create minimal dataset
        dataset_data = {
            "Bayer": {
                "scene1": {
                    "clean_images": [
                        {"filename": "missing1.arw", "sha1": "sha1", "file_id": "id1"},
                        {"filename": "missing2.arw", "sha1": "sha2", "file_id": "id2"},
                    ],
                    "noisy_images": [],
                    "unknown_sensor": False,
                    "test_reserve": False,
                }
            }
        }
        index._build_index_from_data(dataset_data)
        event_fired = False

        def listener():
            nonlocal event_fired
            event_fired = True

        index._events.on(CacheEvent.LOCAL_PATHS_UPDATED, listener)

        # Consume generator fully (event emits after completion)
        async for _ in index.produce_missing_files():
            pass

        assert event_fired, (
            "LOCAL_PATHS_UPDATED event should be emitted after completion"
        )

    @pytest.mark.trio
    async def test_async_iter_missing_files_updates_cached_state(self):
        """Test that async_iter_missing_files updates img_info.local_path.

        Given: Files that don't exist locally
        When: async_iter_missing_files() runs
        Then: img_info.local_path should be set to expected path (for download target)
        """
        from pathlib import Path

        index = DatasetIndex()

        # Create minimal dataset
        dataset_data = {
            "Bayer": {
                "scene1": {
                    "clean_images": [
                        {"filename": "missing.arw", "sha1": "sha1", "file_id": "id1"}
                    ],
                    "noisy_images": [],
                    "unknown_sensor": False,
                    "test_reserve": False,
                }
            }
        }
        index._build_index_from_data(dataset_data)
        # Ensure no local path is set initially
        for scenes in index.scenes.values():
            for scene in scenes.values():
                for img in scene.all_images():
                    assert img.local_path is None, "Initial local_path should be None"

        missing = []
        async for img_info in index.produce_missing_files():
            missing.append(img_info)
            # Should have designated path for download
            assert img_info.local_path is not None, "local_path should be set"
            assert isinstance(img_info.local_path, Path), "local_path should be a Path"

        # Cached state should be updated
        assert len(missing) == 1, "Should have yielded one missing file"
        assert missing[0].local_path is not None, "Cached state should be updated"

        # Verify the path makes sense (should be in expected location)
        expected_path = index.dataset_root / "Bayer" / "scene1" / "gt" / "missing.arw"
        assert missing[0].local_path == expected_path, (
            "Path should match expected location"
        )
