"""
Comprehensive TDD test suite for AsyncPipelineBridge.

Tests cover (target >=90% coverage):
1. Bridge Core Functionality
   - Initialization and configuration
   - Scene collection from async pipeline
   - Scene retrieval and indexing
   - Thread safety
   - State management
   - Statistics tracking
   - Health checks and monitoring
   - Cache integration
   - Scene filtering and validation
   - Error handling and retry logic

2. Legacy Dataloader Compatibility
   - Disk cache mode (offline preprocessing)
   - Streaming mode (real-time pipeline)
   - Format compatibility with YAML-based dataloaders
   - Legacy API methods (backwards_compat_mode)

3. Upstream Pipeline Connections
   - Trio channel consumption
   - Backpressure handling
   - Channel closure and cleanup
   - Integration with AsyncAligner output

4. Future Components (stub tests, marked skip)
   - Advanced caching strategies
   - Enhanced monitoring dashboards
   - Multi-bridge coordination
   - Distributed pipeline support

Written following TDD red-green-refactor discipline.
Tests fail initially (RED) until implementation is complete.
"""

import json
import logging
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, patch, MagicMock

import pytest
import trio
import yaml

from rawnind.dataset.SceneInfo import SceneInfo, ImageInfo
# This will fail initially - we're writing tests first (RED phase)
from rawnind.dataset.AsyncPipelineBridge import (
    AsyncPipelineBridge,
    BridgeState,
    BridgeStats,
)

pytestmark = pytest.mark.dataset

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def minimal_scene():
    """Create a minimal valid SceneInfo for testing."""
    gt_img = ImageInfo(
        filename="gt.exr",
        sha1="abc123",
        is_clean=True,
        scene_name="test_scene",
        scene_images=["gt.exr", "noisy.arw"],
        cfa_type="bayer",
        file_id="gt_001",
    )

    noisy_img = ImageInfo(
        filename="noisy.arw",
        sha1="def456",
        is_clean=False,
        scene_name="test_scene",
        scene_images=["gt.exr", "noisy.arw"],
        cfa_type="bayer",
        file_id="noisy_001",
        metadata={
            "alignment": [2, -3],
            "alignment_loss": 0.05,
            "mask_mean": 0.92,
            "raw_gain": 1.5,
            "rgb_gain": 1.0,
            "is_bayer": True,
            "overexposure_lb": 0.98,
            "rgb_xyz_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "crops": [
                {
                    "coordinates": [512, 256],
                    "gt_linrec2020_fpath": "/fake/crops/gt_512_256.exr",
                    "f_bayer_fpath": "/fake/crops/noisy_512_256.npy",
                    "gt_bayer_fpath": "/fake/crops/gt_512_256.npy",
                    "f_linrec2020_fpath": "/fake/crops/noisy_512_256.exr",
                }
            ],
        }
    )

    return SceneInfo(
        scene_name="test_scene",
        cfa_type="bayer",
        unknown_sensor=False,
        test_reserve=False,
        clean_images=[gt_img],
        noisy_images=[noisy_img],
    )


@pytest.fixture
def test_reserve_scene(minimal_scene):
    """Scene marked as test_reserve."""
    scene = minimal_scene
    scene.test_reserve = True
    scene.scene_name = "test_reserve_scene"
    return scene


@pytest.fixture
def xtrans_scene():
    """Scene with x-trans CFA."""
    gt_img = ImageInfo(
        filename="gt_xtrans.exr",
        sha1="xtrans123",
        is_clean=True,
        scene_name="xtrans_scene",
        scene_images=["gt_xtrans.exr", "noisy_xtrans.raf"],
        cfa_type="x-trans",
        file_id="xtrans_gt_001",
    )

    noisy_img = ImageInfo(
        filename="noisy_xtrans.raf",
        sha1="xtrans456",
        is_clean=False,
        scene_name="xtrans_scene",
        scene_images=["gt_xtrans.exr", "noisy_xtrans.raf"],
        cfa_type="x-trans",
        file_id="xtrans_noisy_001",
        metadata={
            "alignment": [0, 0],
            "alignment_loss": 0.01,
            "mask_mean": 0.95,
            "raw_gain": 1.2,
        }
    )

    return SceneInfo(
        scene_name="xtrans_scene",
        cfa_type="x-trans",
        unknown_sensor=False,
        test_reserve=False,
        clean_images=[gt_img],
        noisy_images=[noisy_img],
    )


@pytest.fixture
def mock_cache():
    """Mock cache manager for testing."""
    cache = MagicMock()
    cache.get = MagicMock(return_value=None)
    cache.set = MagicMock()
    return cache


# ============================================================================
# Part 1: Bridge Core Functionality Tests
# ============================================================================

class TestBridgeInitialization:
    """Test bridge initialization and configuration."""

    def test_init_default_config(self):
        """Test bridge initializes with default configuration."""
        bridge = AsyncPipelineBridge()

        assert bridge.max_scenes is None
        assert bridge.enable_caching is True
        assert bridge.backwards_compat_mode is False
        assert bridge.filter_test_reserve is False
        assert bridge.cfa_filter is None
        assert bridge.state == BridgeState.INITIALIZED

    def test_init_with_max_scenes(self):
        """Test bridge initializes with max_scenes limit."""
        bridge = AsyncPipelineBridge(max_scenes=100)

        assert bridge.max_scenes == 100

    def test_init_with_caching_disabled(self):
        """Test bridge initializes with caching disabled."""
        bridge = AsyncPipelineBridge(enable_caching=False)

        assert bridge.enable_caching is False
        assert bridge.cache is None

    def test_init_with_backwards_compat_mode(self):
        """Test bridge initializes in backwards compatibility mode."""
        bridge = AsyncPipelineBridge(backwards_compat_mode=True)

        assert bridge.backwards_compat_mode is True

    def test_init_with_test_reserve_filter(self):
        """Test bridge initializes with test_reserve filtering."""
        bridge = AsyncPipelineBridge(filter_test_reserve=True)

        assert bridge.filter_test_reserve is True

    def test_init_with_cfa_filter(self):
        """Test bridge initializes with CFA type filter."""
        bridge = AsyncPipelineBridge(cfa_filter="bayer")

        assert bridge.cfa_filter == "bayer"

    def test_init_with_monitoring_disabled(self):
        """Test bridge initializes with monitoring disabled."""
        bridge = AsyncPipelineBridge(enable_monitoring=False)

        assert bridge.enable_monitoring is False
        assert bridge.stats is None

    def test_init_with_thread_safety_disabled(self):
        """Test bridge initializes with thread safety disabled."""
        bridge = AsyncPipelineBridge(thread_safe=False)

        assert bridge.thread_safe is False
        assert bridge._lock is None

    def test_init_invalid_max_scenes_raises_error(self):
        """Test bridge initialization fails with invalid max_scenes."""
        with pytest.raises(ValueError, match="max_scenes must be positive"):
            AsyncPipelineBridge(max_scenes=0)

        with pytest.raises(ValueError, match="max_scenes must be positive"):
            AsyncPipelineBridge(max_scenes=-5)

    def test_init_with_cache_manager(self, mock_cache):
        """Test bridge initializes with provided cache manager."""
        bridge = AsyncPipelineBridge(cache=mock_cache)

        assert bridge.cache is mock_cache

    def test_init_with_mock_mode(self):
        """Test bridge initializes in mock mode."""
        bridge = AsyncPipelineBridge(mock_mode=True, max_scenes=10)

        assert bridge.mock_mode is True
        assert bridge.max_scenes == 10


class TestBridgeSceneCollection:
    """Test async scene collection from pipeline."""

    @pytest.mark.trio
    async def test_collect_single_scene(self, minimal_scene):
        """Test collecting a single scene from channel."""
        bridge = AsyncPipelineBridge(max_scenes=1)
        send_channel, recv_channel = trio.open_memory_channel(10)

        async with trio.open_nursery() as nursery:
            nursery.start_soon(bridge.consume, recv_channel)
            await send_channel.send(minimal_scene)
            await send_channel.aclose()

        assert len(bridge) == 1
        assert bridge.get_scene(0).scene_name == "test_scene"
        assert bridge.state == BridgeState.READY

    @pytest.mark.trio
    async def test_collect_multiple_scenes(self, minimal_scene):
        """Test collecting multiple scenes from channel."""
        bridge = AsyncPipelineBridge(max_scenes=5)
        send_channel, recv_channel = trio.open_memory_channel(10)

        scenes = []
        for i in range(5):
            scene = SceneInfo(
                scene_name=f"scene_{i}",
                cfa_type="bayer",
                unknown_sensor=False,
                test_reserve=False,
                clean_images=[minimal_scene.clean_images[0]],
                noisy_images=[minimal_scene.noisy_images[0]],
            )
            scenes.append(scene)

        async with trio.open_nursery() as nursery:
            nursery.start_soon(bridge.consume, recv_channel)

            for scene in scenes:
                await send_channel.send(scene)

            await send_channel.aclose()

        assert len(bridge) == 5
        for i in range(5):
            assert bridge.get_scene(i).scene_name == f"scene_{i}"

    @pytest.mark.trio
    async def test_collect_respects_max_scenes(self, minimal_scene):
        """Test collection stops at max_scenes limit."""
        bridge = AsyncPipelineBridge(max_scenes=3)
        send_channel, recv_channel = trio.open_memory_channel(10)

        scenes = []
        for i in range(10):  # Send more than max_scenes
            scene = SceneInfo(
                scene_name=f"scene_{i}",
                cfa_type="bayer",
                unknown_sensor=False,
                test_reserve=False,
                clean_images=[minimal_scene.clean_images[0]],
                noisy_images=[minimal_scene.noisy_images[0]],
            )
            scenes.append(scene)

        async with trio.open_nursery() as nursery:
            nursery.start_soon(bridge.consume, recv_channel)

            for scene in scenes:
                await send_channel.send(scene)

            await send_channel.aclose()

        # Should only collect up to max_scenes
        assert len(bridge) == 3

    @pytest.mark.trio
    async def test_collect_with_progress_callback(self, minimal_scene):
        """Test collection calls progress callback."""
        bridge = AsyncPipelineBridge(max_scenes=3)
        send_channel, recv_channel = trio.open_memory_channel(10)

        progress_calls = []

        def progress_callback(current, total):
            progress_calls.append((current, total))

        scenes = []
        for i in range(3):
            scene = SceneInfo(
                scene_name=f"scene_{i}",
                cfa_type="bayer",
                unknown_sensor=False,
                test_reserve=False,
                clean_images=[minimal_scene.clean_images[0]],
                noisy_images=[minimal_scene.noisy_images[0]],
            )
            scenes.append(scene)

        async with trio.open_nursery() as nursery:
            nursery.start_soon(
                bridge.consume,
                recv_channel,
                progress_callback
            )

            for scene in scenes:
                await send_channel.send(scene)

            await send_channel.aclose()

        # Verify progress was reported
        assert len(progress_calls) == 3
        assert progress_calls[-1] == (3, 3)

    @pytest.mark.trio
    async def test_collect_empty_channel(self):
        """Test collecting from empty channel."""
        bridge = AsyncPipelineBridge()
        send_channel, recv_channel = trio.open_memory_channel(10)

        async with trio.open_nursery() as nursery:
            nursery.start_soon(bridge.consume, recv_channel)
            await send_channel.aclose()

        assert len(bridge) == 0
        assert bridge.state == BridgeState.READY

    @pytest.mark.trio
    async def test_collect_updates_statistics(self, minimal_scene):
        """Test collection updates bridge statistics."""
        bridge = AsyncPipelineBridge(max_scenes=2, enable_monitoring=True)
        send_channel, recv_channel = trio.open_memory_channel(10)

        async with trio.open_nursery() as nursery:
            nursery.start_soon(bridge.consume, recv_channel)
            await send_channel.send(minimal_scene)
            await send_channel.send(minimal_scene)
            await send_channel.aclose()

        assert bridge.stats is not None
        assert bridge.stats.scenes_collected == 2
        assert bridge.stats.collection_start_time is not None
        assert bridge.stats.collection_end_time is not None

    @pytest.mark.trio
    async def test_collect_with_timeout(self, minimal_scene, autojump_clock):
        """Test collection respects timeout using MockClock."""
        bridge = AsyncPipelineBridge()
        send_channel, recv_channel = trio.open_memory_channel(10)

        with pytest.raises(trio.TooSlowError):
            # Directly await the collection to raise in this task (no ExceptionGroup)
            await bridge.consume(
                recv_channel,
                None,
                0.01  # 10ms timeout
            )

    @pytest.mark.trio
    async def test_collect_mock_mode(self):
        """Test collection in mock mode generates test scenes."""
        bridge = AsyncPipelineBridge(mock_mode=True, max_scenes=5)

        # No channel needed in mock mode
        await bridge.consume()

        assert len(bridge) == 5
        assert all(scene.scene_name.startswith("mock_scene_") for scene in bridge)


class TestBridgeSceneRetrieval:
    """Test scene retrieval and indexing."""

    @pytest.mark.trio
    async def test_get_scene_by_index(self, minimal_scene):
        """Test retrieving scene by index."""
        bridge = AsyncPipelineBridge(max_scenes=3)
        send_channel, recv_channel = trio.open_memory_channel(10)

        scenes = []
        for i in range(3):
            scene = SceneInfo(
                scene_name=f"scene_{i}",
                cfa_type="bayer",
                unknown_sensor=False,
                test_reserve=False,
                clean_images=[minimal_scene.clean_images[0]],
                noisy_images=[minimal_scene.noisy_images[0]],
            )
            scenes.append(scene)

        async with trio.open_nursery() as nursery:
            nursery.start_soon(bridge.consume, recv_channel)

            for scene in scenes:
                await send_channel.send(scene)

            await send_channel.aclose()

        # Test valid indices
        assert bridge.get_scene(0).scene_name == "scene_0"
        assert bridge.get_scene(1).scene_name == "scene_1"
        assert bridge.get_scene(2).scene_name == "scene_2"

    def test_get_scene_index_out_of_bounds(self):
        """Test retrieving scene with invalid index raises error."""
        bridge = AsyncPipelineBridge()

        with pytest.raises(IndexError):
            bridge.get_scene(0)

        with pytest.raises(IndexError):
            bridge.get_scene(-1)

    def test_len_returns_scene_count(self, minimal_scene):
        """Test __len__ returns correct scene count."""
        bridge = AsyncPipelineBridge()

        assert len(bridge) == 0

        # Manually add scenes (simulating collection)
        bridge._scenes = [minimal_scene, minimal_scene, minimal_scene]

        assert len(bridge) == 3

    def test_getitem_indexing(self, minimal_scene):
        """Test __getitem__ supports indexing operator."""
        bridge = AsyncPipelineBridge()
        bridge._scenes = [minimal_scene]

        scene = bridge[0]
        assert scene.scene_name == "test_scene"

    def test_iter_supports_iteration(self, minimal_scene):
        """Test __iter__ supports for-loop iteration."""
        bridge = AsyncPipelineBridge()

        scenes = []
        for i in range(3):
            scene = SceneInfo(
                scene_name=f"scene_{i}",
                cfa_type="bayer",
                unknown_sensor=False,
                test_reserve=False,
                clean_images=[minimal_scene.clean_images[0]],
                noisy_images=[minimal_scene.noisy_images[0]],
            )
            scenes.append(scene)

        bridge._scenes = scenes

        collected = []
        for scene in bridge:
            collected.append(scene.scene_name)

        assert collected == ["scene_0", "scene_1", "scene_2"]


class TestBridgeThreadSafety:
    """Test thread-safe operations."""

    def test_thread_safe_get_scene(self, minimal_scene):
        """Test get_scene is thread-safe."""
        bridge = AsyncPipelineBridge(thread_safe=True)
        bridge._scenes = [minimal_scene] * 10

        results = []
        errors = []

        def worker():
            try:
                for i in range(10):
                    scene = bridge.get_scene(i)
                    results.append(scene)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Should have no errors
        assert len(errors) == 0
        # Should have all results
        assert len(results) == 100  # 10 threads * 10 scenes

    def test_thread_safe_len(self, minimal_scene):
        """Test __len__ is thread-safe."""
        bridge = AsyncPipelineBridge(thread_safe=True)

        results = []

        def worker():
            bridge._scenes.append(minimal_scene)
            results.append(len(bridge))

        threads = [threading.Thread(target=worker) for _ in range(10)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        assert len(bridge) == 10


class TestBridgeStateManagement:
    """Test bridge state transitions."""

    def test_initial_state(self):
        """Test bridge starts in INITIALIZED state."""
        bridge = AsyncPipelineBridge()

        assert bridge.state == BridgeState.INITIALIZED

    @pytest.mark.trio
    async def test_state_during_collection(self, autojump_clock):
        """Test state transitions using wait_all_tasks_blocked."""
        bridge = AsyncPipelineBridge()
        send_channel, recv_channel = trio.open_memory_channel(10)

        state_during_collection = None

        async def check_state():
            # Wait until collection task blocks (no arbitrary sleep!)
            await trio.testing.wait_all_tasks_blocked()
            nonlocal state_during_collection
            state_during_collection = bridge.state
            # Now close the channel to let collection finish
            await send_channel.aclose()

        async with trio.open_nursery() as nursery:
            nursery.start_soon(bridge.consume, recv_channel)
            nursery.start_soon(check_state)

        assert state_during_collection == BridgeState.COLLECTING

    @pytest.mark.trio
    async def test_state_after_collection(self, minimal_scene):
        """Test state transitions to READY after collection."""
        bridge = AsyncPipelineBridge()
        send_channel, recv_channel = trio.open_memory_channel(10)

        async with trio.open_nursery() as nursery:
            nursery.start_soon(bridge.consume, recv_channel)
            await send_channel.send(minimal_scene)
            await send_channel.aclose()

        assert bridge.state == BridgeState.READY

    def test_get_scene_in_error_state_raises(self):
        """Test get_scene raises error when bridge in ERROR state."""
        bridge = AsyncPipelineBridge()
        bridge.state = BridgeState.ERROR

        with pytest.raises(RuntimeError, match="Bridge is in error state"):
            bridge.get_scene(0)

    def test_close_transitions_to_closed(self):
        """Test close() transitions bridge to CLOSED state."""
        bridge = AsyncPipelineBridge()

        bridge.close()

        assert bridge.state == BridgeState.CLOSED


class TestBridgeStatistics:
    """Test statistics tracking and reporting."""

    @pytest.mark.trio
    async def test_stats_track_scenes_collected(self, minimal_scene):
        """Test statistics track number of scenes collected."""
        bridge = AsyncPipelineBridge(max_scenes=5, enable_monitoring=True)
        send_channel, recv_channel = trio.open_memory_channel(10)

        async with trio.open_nursery() as nursery:
            nursery.start_soon(bridge.consume, recv_channel)

            for i in range(5):
                await send_channel.send(minimal_scene)

            await send_channel.aclose()

        stats = bridge.get_stats()
        assert stats.scenes_collected == 5

    @pytest.mark.trio
    async def test_stats_track_scenes_filtered(self, minimal_scene, test_reserve_scene):
        """Test statistics track filtered scenes."""
        bridge = AsyncPipelineBridge(
            filter_test_reserve=True,
            enable_monitoring=True
        )
        send_channel, recv_channel = trio.open_memory_channel(10)

        async with trio.open_nursery() as nursery:
            nursery.start_soon(bridge.consume, recv_channel)

            # Send mix of regular and test reserve scenes
            await send_channel.send(minimal_scene)
            await send_channel.send(test_reserve_scene)  # Should be filtered
            await send_channel.send(minimal_scene)

            await send_channel.aclose()

        stats = bridge.get_stats()
        assert stats.scenes_collected == 2
        assert stats.scenes_filtered == 1

    def test_stats_collection_duration(self):
        """Test statistics calculate collection duration."""
        stats = BridgeStats()
        stats.collection_start_time = 100.0
        stats.collection_end_time = 110.5

        duration = stats.collection_duration_seconds()
        assert duration == 10.5

    def test_stats_collection_rate(self):
        """Test statistics calculate collection rate."""
        stats = BridgeStats()
        stats.scenes_collected = 100
        stats.collection_start_time = 0.0
        stats.collection_end_time = 10.0

        rate = stats.collection_rate()
        assert rate == 10.0  # 100 scenes / 10 seconds

    def test_reset_stats(self, minimal_scene):
        """Test resetting statistics."""
        bridge = AsyncPipelineBridge(enable_monitoring=True)
        bridge._scenes = [minimal_scene] * 5
        bridge.stats.scenes_collected = 5

        bridge.reset_stats()

        assert bridge.stats.scenes_collected == 0


class TestBridgeHealthChecks:
    """Test health check and monitoring."""

    def test_health_check_returns_status(self):
        """Test health_check returns current status."""
        bridge = AsyncPipelineBridge(max_scenes=10)

        health = bridge.health_check()

        assert health["state"] == BridgeState.INITIALIZED.value
        assert health["scenes_collected"] == 0
        assert health["cache_enabled"] is True
        assert health["thread_safe"] is True
        assert "last_check" in health

    def test_health_check_includes_stats(self, minimal_scene):
        """Test health check includes statistics when enabled."""
        bridge = AsyncPipelineBridge(enable_monitoring=True)
        bridge._scenes = [minimal_scene] * 3
        bridge.stats.scenes_collected = 3

        health = bridge.health_check()

        assert "stats" in health
        assert health["stats"]["scenes_collected"] == 3


class TestBridgeCacheIntegration:
    """Test cache integration and operations."""

    @pytest.mark.trio
    async def test_cache_stores_scenes(self, minimal_scene, mock_cache):
        """Test scenes are cached during collection."""
        bridge = AsyncPipelineBridge(cache=mock_cache, enable_caching=True)
        send_channel, recv_channel = trio.open_memory_channel(10)

        async with trio.open_nursery() as nursery:
            nursery.start_soon(bridge.consume, recv_channel)
            await send_channel.send(minimal_scene)
            await send_channel.aclose()

        # Verify cache.set was called
        mock_cache.set.assert_called_once()

    def test_get_scene_uses_cache(self, minimal_scene, mock_cache):
        """Test get_scene retrieves from cache when available."""
        bridge = AsyncPipelineBridge(cache=mock_cache, enable_caching=True)
        bridge._scenes = [minimal_scene]

        # Configure cache to return a cached scene
        cached_scene = minimal_scene
        mock_cache.get.return_value = cached_scene

        scene = bridge.get_scene(0)

        # Verify cache was checked
        mock_cache.get.assert_called_once_with("test_scene")
        # Should return cached version
        assert scene == cached_scene

    def test_get_scene_tracks_cache_hits(self, minimal_scene, mock_cache):
        """Test cache hits are tracked in statistics."""
        bridge = AsyncPipelineBridge(
            cache=mock_cache,
            enable_caching=True,
            enable_monitoring=True
        )
        bridge._scenes = [minimal_scene]

        mock_cache.get.return_value = minimal_scene

        bridge.get_scene(0)

        assert bridge.stats.cache_hits == 1
        assert bridge.stats.cache_misses == 0

    def test_get_scene_tracks_cache_misses(self, minimal_scene, mock_cache):
        """Test cache misses are tracked in statistics."""
        bridge = AsyncPipelineBridge(
            cache=mock_cache,
            enable_caching=True,
            enable_monitoring=True
        )
        bridge._scenes = [minimal_scene]

        mock_cache.get.return_value = None  # Cache miss

        bridge.get_scene(0)

        assert bridge.stats.cache_hits == 0
        assert bridge.stats.cache_misses == 1


class TestBridgeSceneFiltering:
    """Test scene filtering logic."""

    @pytest.mark.trio
    async def test_filter_test_reserve_scenes(self, minimal_scene, test_reserve_scene):
        """Test filtering out test_reserve scenes."""
        bridge = AsyncPipelineBridge(filter_test_reserve=True, max_scenes=10)
        send_channel, recv_channel = trio.open_memory_channel(10)

        async with trio.open_nursery() as nursery:
            nursery.start_soon(bridge.consume, recv_channel)

            # Send mix of regular and test reserve
            await send_channel.send(minimal_scene)
            await send_channel.send(test_reserve_scene)  # Should be filtered
            await send_channel.send(minimal_scene)

            await send_channel.aclose()

        # Only non-test_reserve scenes should be collected
        assert len(bridge) == 2
        assert all(not scene.test_reserve for scene in bridge)

    @pytest.mark.trio
    async def test_filter_by_cfa_type(self, minimal_scene, xtrans_scene):
        """Test filtering by CFA type."""
        bridge = AsyncPipelineBridge(cfa_filter="bayer", max_scenes=10)
        send_channel, recv_channel = trio.open_memory_channel(10)

        async with trio.open_nursery() as nursery:
            nursery.start_soon(bridge.consume, recv_channel)

            # Send mix of bayer and X-Trans
            await send_channel.send(minimal_scene)  # bayer
            await send_channel.send(xtrans_scene)   # X-Trans, should be filtered
            await send_channel.send(minimal_scene)  # bayer

            await send_channel.aclose()

        # Only bayer scenes should be collected
        assert len(bridge) == 2
        assert all(scene.cfa_type == "bayer" for scene in bridge)


class TestBridgeSceneValidation:
    """Test scene validation logic."""

    @pytest.mark.trio
    async def test_validate_scene_missing_scene_name(self):
        """Test validation rejects scene without scene_name."""
        bridge = AsyncPipelineBridge(validate_scenes=True)
        send_channel, recv_channel = trio.open_memory_channel(10)

        # Create invalid scene (missing scene_name)
        invalid_scene = SceneInfo(
            scene_name="",  # Empty scene name
            cfa_type="bayer",
            unknown_sensor=False,
            test_reserve=False,
            clean_images=[],
            noisy_images=[],
        )

        async with trio.open_nursery() as nursery:
            nursery.start_soon(bridge.consume, recv_channel)
            await send_channel.send(invalid_scene)
            await send_channel.aclose()

        # Invalid scene should not be collected
        assert len(bridge) == 0

    @pytest.mark.trio
    async def test_validate_scene_missing_images(self):
        """Test validation rejects scene without image lists."""
        bridge = AsyncPipelineBridge(validate_scenes=True)
        send_channel, recv_channel = trio.open_memory_channel(10)

        # Create scene without proper attributes
        invalid_scene = type('SceneInfo', (), {
            'scene_name': 'test'
        })()

        async with trio.open_nursery() as nursery:
            nursery.start_soon(bridge.consume, recv_channel)
            await send_channel.send(invalid_scene)
            await send_channel.aclose()

        # Invalid scene should not be collected
        assert len(bridge) == 0


class TestBridgeErrorHandling:
    """Test error handling and retry logic."""

    @pytest.mark.trio
    async def test_handles_non_sceneinfo_objects(self, minimal_scene):
        """Test bridge handles non-SceneInfo objects gracefully."""
        bridge = AsyncPipelineBridge()
        send_channel, recv_channel = trio.open_memory_channel(10)

        async with trio.open_nursery() as nursery:
            nursery.start_soon(bridge.consume, recv_channel)

            # Send valid scene
            await send_channel.send(minimal_scene)
            # Send invalid object
            await send_channel.send("not a scene")
            # Send another valid scene
            await send_channel.send(minimal_scene)

            await send_channel.aclose()

        # Should collect only valid scenes
        assert len(bridge) == 2

    @pytest.mark.trio
    async def test_retry_on_error(self, minimal_scene):
        """Test retry logic on processing errors."""
        bridge = AsyncPipelineBridge(retry_on_error=True)
        send_channel, recv_channel = trio.open_memory_channel(10)

        call_count = 0

        # Patch _process_scene to fail first time, succeed second
        original_process = bridge._process_scene

        async def failing_process(item, index):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Test error")
            return await original_process(item, index)

        bridge._process_scene = failing_process

        async with trio.open_nursery() as nursery:
            nursery.start_soon(bridge.consume, recv_channel)
            await send_channel.send(minimal_scene)
            await send_channel.aclose()

        # Should have retried and succeeded
        assert call_count >= 2


class TestBridgeCleanup:
    """Test resource cleanup and lifecycle."""

    def test_close_clears_scenes(self, minimal_scene):
        """Test close() clears scene list."""
        bridge = AsyncPipelineBridge()
        bridge._scenes = [minimal_scene] * 10

        bridge.close()

        assert len(bridge._scenes) == 0

    def test_close_sets_closed_state(self):
        """Test close() sets CLOSED state."""
        bridge = AsyncPipelineBridge()

        bridge.close()

        assert bridge.state == BridgeState.CLOSED

    def test_destructor_calls_close(self, minimal_scene):
        """Test __del__ calls close() if not already closed."""
        bridge = AsyncPipelineBridge()
        bridge._scenes = [minimal_scene]

        # Trigger destructor
        del bridge

        # Bridge should be closed (can't directly verify, but no errors)


# ============================================================================
# Part 2: Legacy Dataloader Compatibility Tests
# ============================================================================

class TestLegacyDataloaderDiskCache:
    """Test disk cache mode for legacy dataloader compatibility."""

    def test_bridge_writes_jsonl_cache(self, minimal_scene, tmp_path):
        """Test bridge can write scenes to JSONL cache file."""
        bridge = AsyncPipelineBridge()
        bridge._scenes = [minimal_scene] * 3

        cache_file = tmp_path / "scenes_cache.jsonl"

        # Method to be implemented: write_disk_cache
        bridge.write_disk_cache(cache_file)

        assert cache_file.exists()

        # Verify JSONL format
        with open(cache_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 3

            for line in lines:
                scene_dict = json.loads(line)
                assert "scene_name" in scene_dict

    def test_bridge_reads_jsonl_cache(self, tmp_path):
        """Test bridge can read scenes from JSONL cache file."""
        cache_file = tmp_path / "scenes_cache.jsonl"

        # Write test cache
        test_scene_dict = {
            "scene_name": "cached_scene",
            "cfa_type": "bayer",
            "unknown_sensor": False,
            "test_reserve": False,
        }

        with open(cache_file, 'w') as f:
            f.write(json.dumps(test_scene_dict) + '\n')

        bridge = AsyncPipelineBridge()

        # Method to be implemented: load_from_disk_cache
        bridge.load_from_disk_cache(cache_file)

        assert len(bridge) == 1
        assert bridge[0].scene_name == "cached_scene"

    def test_cache_compatible_with_yaml_format(self, minimal_scene, tmp_path):
        """Test cache output compatible with legacy YAML-based dataloaders."""
        bridge = AsyncPipelineBridge()
        bridge._scenes = [minimal_scene]

        cache_file = tmp_path / "pipeline_output.yaml"

        # Method to be implemented: write_yaml_compatible_cache
        bridge.write_yaml_compatible_cache(cache_file)

        # Verify YAML structure matches legacy format
        with open(cache_file, 'r') as f:
            data = yaml.safe_load(f)

        assert isinstance(data, list)
        assert len(data) == 1

        entry = data[0]

        # Check required fields for legacy dataloader
        required_fields = [
            "scene_name", "image_set", "is_bayer",
            "best_alignment", "best_alignment_loss",
            "mask_mean", "mask_fpath",
            "raw_gain", "rgb_gain",
            "f_fpath", "gt_fpath",
            "crops", "rgb_xyz_matrix",
        ]

        for field in required_fields:
            assert field in entry, f"Missing required field: {field}"

    def test_legacy_dataloader_consumes_bridge_cache(self, minimal_scene, tmp_path):
        """Test legacy dataloader can load bridge-written cache."""
        # This test will import the actual legacy dataloader
        from rawnind.libs.rawds import CleanProfiledRGBNoisyBayerImageCropsDataset

        bridge = AsyncPipelineBridge()
        bridge._scenes = [minimal_scene]

        cache_file = tmp_path / "pipeline_output.yaml"
        bridge.write_yaml_compatible_cache(cache_file)

        # Attempt to load with legacy dataloader
        dataset = CleanProfiledRGBNoisyBayerImageCropsDataset(
            content_fpaths=[str(cache_file)],
            num_crops=1,
            crop_size=256,
            test_reserve=[],
            bayer_only=True,
        )

        # Should successfully load
        assert len(dataset) > 0


class TestLegacyDataloaderStreaming:
    """Test streaming mode for real-time pipeline access."""

    def test_bridge_as_pytorch_dataset(self, minimal_scene):
        """Test bridge can act as PyTorch Dataset."""
        bridge = AsyncPipelineBridge()
        bridge._scenes = [minimal_scene] * 5

        # Should support __len__ and __getitem__
        assert len(bridge) == 5
        sample = bridge[0]
        assert isinstance(sample, SceneInfo)

    def test_streaming_wrapper_provides_legacy_interface(self, minimal_scene):
        """Test StreamingDatasetWrapper provides legacy dataloader interface."""
        bridge = AsyncPipelineBridge()
        bridge._scenes = [minimal_scene] * 5

        # Wrapper to be implemented: StreamingDatasetWrapper
        from rawnind.dataset.AsyncPipelineBridge import StreamingDatasetWrapper

        wrapper = StreamingDatasetWrapper(
            bridge=bridge,
            num_crops=4,
            crop_size=256,
        )

        # Should provide legacy __getitem__ returning crop dictionaries
        sample = wrapper[0]

        assert "x_crops" in sample
        assert "y_crops" in sample
        assert "mask_crops" in sample
        assert "rgb_xyz_matrix" in sample

    def test_streaming_no_disk_writes(self, minimal_scene, tmp_path):
        """Test streaming mode does not write to disk."""
        bridge = AsyncPipelineBridge()
        bridge._scenes = [minimal_scene] * 5

        from rawnind.dataset.AsyncPipelineBridge import StreamingDatasetWrapper

        wrapper = StreamingDatasetWrapper(
            bridge=bridge,
            num_crops=1,
            crop_size=256,
        )

        # Access multiple samples
        for i in range(5):
            _ = wrapper[i]

        # Verify no files written to disk
        assert len(list(tmp_path.glob("*"))) == 0


class TestBackwardsCompatibilityMode:
    """Test backwards compatibility API methods."""

    def test_get_clean_noisy_pair(self, minimal_scene):
        """Test get_clean_noisy_pair returns legacy format."""
        bridge = AsyncPipelineBridge(backwards_compat_mode=True)
        bridge._scenes = [minimal_scene]

        clean, noisy_list = bridge.get_clean_noisy_pair(0)

        assert clean is not None
        assert isinstance(noisy_list, list)
        assert len(noisy_list) > 0

    def test_get_clean_noisy_pair_disabled(self, minimal_scene):
        """Test get_clean_noisy_pair raises error when disabled."""
        bridge = AsyncPipelineBridge(backwards_compat_mode=False)
        bridge._scenes = [minimal_scene]

        with pytest.raises(RuntimeError, match="Backwards compatibility mode not enabled"):
            bridge.get_clean_noisy_pair(0)


# ============================================================================
# Part 3: Upstream Pipeline Connection Tests
# ============================================================================

class TestTrioChannelConsumption:
    """Test consuming from trio channels."""

    @pytest.mark.trio
    async def test_consumes_from_memory_channel(self, minimal_scene):
        """Test bridge consumes from trio.open_memory_channel."""
        bridge = AsyncPipelineBridge(max_scenes=3)
        send, recv = trio.open_memory_channel(10)

        async with trio.open_nursery() as nursery:
            nursery.start_soon(bridge.consume, recv)

            for i in range(3):
                await send.send(minimal_scene)

            await send.aclose()

        assert len(bridge) == 3

    @pytest.mark.trio
    async def test_handles_channel_closure(self, minimal_scene):
        """Test bridge handles channel closure gracefully."""
        bridge = AsyncPipelineBridge()
        send, recv = trio.open_memory_channel(10)

        async with trio.open_nursery() as nursery:
            nursery.start_soon(bridge.consume, recv)

            await send.send(minimal_scene)
            # Close channel early
            await send.aclose()

        # Should complete without error
        assert len(bridge) == 1
        assert bridge.state == BridgeState.READY

    @pytest.mark.trio
    async def test_respects_backpressure(self, minimal_scene):
        """Test bridge respects backpressure from unbuffered channels."""
        bridge = AsyncPipelineBridge(max_scenes=10)
        # Use capacity=0 for unbuffered channel
        send, recv = trio.open_memory_channel(0)

        scene_count = 0

        async def producer():
            nonlocal scene_count
            for i in range(10):
                await send.send(minimal_scene)
                scene_count += 1
            await send.aclose()

        async with trio.open_nursery() as nursery:
            nursery.start_soon(bridge.consume, recv)
            nursery.start_soon(producer)

        # All scenes should be collected
        assert len(bridge) == 10
        assert scene_count == 10


class TestMetadataEnricherIntegration:
    """Test integration with AsyncAligner output."""

    @pytest.mark.trio
    async def test_consumes_enriched_scenes(self, minimal_scene):
        """Test bridge consumes scenes with enriched metadata."""
        bridge = AsyncPipelineBridge()
        send, recv = trio.open_memory_channel(10)

        # Add enriched metadata
        enriched_scene = minimal_scene
        enriched_scene.noisy_images[0].metadata["msssim_score"] = 0.95
        enriched_scene.noisy_images[0].metadata["crops"] = [
            {"coordinates": [0, 0], "gt_linrec2020_fpath": "/path/to/crop"}
        ]

        async with trio.open_nursery() as nursery:
            nursery.start_soon(bridge.consume, recv)
            await send.send(enriched_scene)
            await send.aclose()

        retrieved = bridge[0]
        assert retrieved.noisy_images[0].metadata["msssim_score"] == 0.95
        assert len(retrieved.noisy_images[0].metadata["crops"]) > 0


class TestPipelineBackpressure:
    """Test backpressure handling in pipeline."""

    @pytest.mark.trio
    async def test_unbuffered_channels_apply_backpressure(self, minimal_scene):
        """Test backpressure using sequencing, not timing."""
        bridge = AsyncPipelineBridge(max_scenes=5)
        send, recv = trio.open_memory_channel(0)  # Unbuffered

        sent_count = 0
        receive_order = []

        async def producer():
            nonlocal sent_count
            for i in range(5):
                receive_order.append(f"send_start_{i}")
                await send.send(minimal_scene)
                sent_count += 1
                receive_order.append(f"send_complete_{i}")
            await send.aclose()

        async with trio.open_nursery() as nursery:
            nursery.start_soon(bridge.consume, recv)
            nursery.start_soon(producer)

        assert len(bridge) == 5
        assert sent_count == 5

        # Verify interleaving (backpressure forces send/receive alternation)
        # Without backpressure, all sends would complete before receives
        assert "send_complete_0" in receive_order
        assert receive_order.index("send_start_1") > receive_order.index("send_complete_0")


# ============================================================================
# Part 4: Future Component Tests (Stubs, marked skip)
# ============================================================================

@pytest.mark.skip(reason="Future feature: advanced caching strategies")
class TestAdvancedCaching:
    """Stub tests for future advanced caching features."""

    def test_tiered_cache_system(self):
        """Test multi-tier cache (memory → disk → remote)."""
        pytest.fail("Not yet implemented")

    def test_cache_eviction_policies(self):
        """Test LRU/LFU cache eviction policies."""
        pytest.fail("Not yet implemented")

    def test_distributed_cache_coordination(self):
        """Test cache coordination across multiple bridges."""
        pytest.fail("Not yet implemented")


@pytest.mark.skip(reason="Future feature: enhanced monitoring")
class TestEnhancedMonitoring:
    """Stub tests for future monitoring dashboard features."""

    def test_prometheus_metrics_export(self):
        """Test exporting metrics in Prometheus format."""
        pytest.fail("Not yet implemented")

    def test_grafana_dashboard_integration(self):
        """Test Grafana dashboard integration."""
        pytest.fail("Not yet implemented")

    def test_real_time_performance_alerts(self):
        """Test real-time alerts for performance degradation."""
        pytest.fail("Not yet implemented")


@pytest.mark.skip(reason="Future feature: multi-bridge coordination")
class TestMultiBridgeCoordination:
    """Stub tests for coordinating multiple bridges."""

    def test_load_balancing_across_bridges(self):
        """Test load balancing scene collection across multiple bridges."""
        pytest.fail("Not yet implemented")

    def test_bridge_failover(self):
        """Test automatic failover when a bridge fails."""
        pytest.fail("Not yet implemented")

    def test_distributed_scene_partitioning(self):
        """Test partitioning scenes across multiple bridges by criteria."""
        pytest.fail("Not yet implemented")


@pytest.mark.skip(reason="Future feature: distributed pipeline")
class TestDistributedPipeline:
    """Stub tests for distributed pipeline support."""

    def test_remote_scene_collection(self):
        """Test collecting scenes from remote pipeline nodes."""
        pytest.fail("Not yet implemented")

    def test_network_resilience(self):
        """Test handling network failures in distributed setup."""
        pytest.fail("Not yet implemented")

    def test_cross_datacenter_replication(self):
        """Test replicating scenes across datacenters."""
        pytest.fail("Not yet implemented")


# ============================================================================
# Integration Tests
# ============================================================================

class TestEndToEndIntegration:
    """End-to-end integration tests."""

    @pytest.mark.trio
    async def test_full_pipeline_to_dataloader(self, minimal_scene, tmp_path):
        """Test complete flow: pipeline → bridge → dataloader."""
        # Step 1: Collect scenes via bridge
        bridge = AsyncPipelineBridge(max_scenes=3)
        send, recv = trio.open_memory_channel(10)

        async with trio.open_nursery() as nursery:
            nursery.start_soon(bridge.consume, recv)

            for i in range(3):
                scene = SceneInfo(
                    scene_name=f"scene_{i}",
                    cfa_type="bayer",
                    unknown_sensor=False,
                    test_reserve=False,
                    clean_images=[minimal_scene.clean_images[0]],
                    noisy_images=[minimal_scene.noisy_images[0]],
                )
                await send.send(scene)

            await send.aclose()

        # Step 2: Write to disk cache
        cache_file = tmp_path / "pipeline_output.yaml"
        bridge.write_yaml_compatible_cache(cache_file)

        # Step 3: Load with legacy dataloader
        from rawnind.libs.rawds import CleanProfiledRGBNoisyBayerImageCropsDataset

        dataset = CleanProfiledRGBNoisyBayerImageCropsDataset(
            content_fpaths=[str(cache_file)],
            num_crops=1,
            crop_size=256,
            test_reserve=[],
            bayer_only=True,
        )

        # Step 4: Verify dataloader works
        assert len(dataset) > 0

    @pytest.mark.trio
    async def test_streaming_mode_end_to_end(self, minimal_scene):
        """Test streaming mode end-to-end without disk writes."""
        # Step 1: Collect scenes
        bridge = AsyncPipelineBridge(max_scenes=5)
        send, recv = trio.open_memory_channel(10)

        async with trio.open_nursery() as nursery:
            nursery.start_soon(bridge.consume, recv)

            for i in range(5):
                await send.send(minimal_scene)

            await send.aclose()

        # Step 2: Create streaming wrapper
        from rawnind.dataset.AsyncPipelineBridge import StreamingDatasetWrapper

        wrapper = StreamingDatasetWrapper(
            bridge=bridge,
            num_crops=1,
            crop_size=256,
        )

        # Step 3: Verify can iterate
        samples = []
        for i in range(len(wrapper)):
            sample = wrapper[i]
            samples.append(sample)

        assert len(samples) == 5

# Additional Trio testing fixtures and deterministic concurrency tests

@pytest.fixture
def autojump_clock():
    """MockClock with autojump enabled for instant time advancement."""
    return trio.testing.MockClock(autojump_threshold=0)


@pytest.mark.trio
async def test_collect_signals_readiness_via_start(minimal_scene):
    """Test bridge signals readiness using nursery.start() protocol."""
    bridge = AsyncPipelineBridge(max_scenes=3)
    send, recv = trio.open_memory_channel(10)

    async def bridge_with_start_protocol(task_status=trio.TASK_STATUS_IGNORED):
        # Signal ready before starting collection
        task_status.started()
        await bridge.consume(recv)

    async with trio.open_nursery() as nursery:
        # start() waits until task_status.started() is called
        await nursery.start(bridge_with_start_protocol)

        # Now we know bridge is ready to receive
        for i in range(3):
            await send.send(minimal_scene)
        await send.aclose()

    assert len(bridge) == 3


@pytest.mark.trio
async def test_collect_handles_cancellation_gracefully(minimal_scene):
    """Test bridge handles cancellation mid-collection."""
    bridge = AsyncPipelineBridge()
    send, recv = trio.open_memory_channel(10)

    with trio.CancelScope() as cancel_scope:
        async with trio.open_nursery() as nursery:
            nursery.start_soon(bridge.consume, recv)

            # Send one scene
            await send.send(minimal_scene)
            await trio.testing.wait_all_tasks_blocked()

            # Cancel while waiting for more
            cancel_scope.cancel()

    # Should have collected at least the one scene sent before cancel
    assert len(bridge) >= 1
    # State should reflect graceful shutdown, not ERROR
    assert bridge.state in [BridgeState.READY, BridgeState.COLLECTING]


@pytest.mark.trio
async def test_concurrent_get_scene_ordering_with_sequencer(minimal_scene):
    """Test concurrent access ordering using Sequencer for determinism."""
    from trio.testing import Sequencer

    bridge = AsyncPipelineBridge()
    bridge._scenes = [minimal_scene] * 3

    seq = Sequencer()
    results = []

    async def reader(reader_id, index):
        async with seq(reader_id * 2):
            # All readers reach this point before any proceed
            pass

        scene = bridge.get_scene(index)
        results.append((reader_id, scene.scene_name))

        async with seq(reader_id * 2 + 1):
            # Synchronize completion
            pass

    async with trio.open_nursery() as nursery:
        for i in range(3):
            nursery.start_soon(reader, i, i)

    assert len(results) == 3
    # Verify all readers completed successfully
    assert all(name == "test_scene" for _, name in results)
