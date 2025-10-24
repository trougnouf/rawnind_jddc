"""
Tests for PostDownloadWorker base class and implementation.
#todo: deprecate these
"""

import tempfile
from pathlib import Path
from unittest import TestCase, main

import trio

from src.rawnind.dataset.PostDownloadWorker import PostDownloadWorker
from src.rawnind.dataset.SceneInfo import SceneInfo


class MockPostDownloadWorker(PostDownloadWorker):
    """Mock worker for testing base class functionality."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processed_scenes = []
        self.startup_called = False
        self.shutdown_called = False

    async def startup(self):
        await super().startup()
        self.startup_called = True

    async def shutdown(self):
        await super().shutdown()
        self.shutdown_called = True

    async def process_scene(self, scene: SceneInfo) -> SceneInfo:
        """Record that scene was processed."""
        self.processed_scenes.append(scene.scene_name)
        # Simulate some work
        await trio.sleep(0.01)
        return scene


class TestPostDownloadWorker(TestCase):
    """Tests for PostDownloadWorker base class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp(prefix="test_worker_")
        self.output_dir = Path(self.test_dir) / "output"

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        if Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test worker initialization creates output directory."""
        worker = MockPostDownloadWorker(self.output_dir, max_workers=2)
        self.assertTrue(self.output_dir.exists())
        self.assertEqual(worker.max_workers, 2)
        self.assertFalse(worker.use_process_pool)

    def test_initialization_with_process_pool(self):
        """Test worker initialization with process pool."""
        worker = MockPostDownloadWorker(
            self.output_dir, max_workers=4, use_process_pool=True
        )
        self.assertTrue(worker.use_process_pool)
        self.assertIsNotNone(worker._executor)

    def test_context_manager(self):
        """Test worker lifecycle with context manager."""

        async def test():
            worker = MockPostDownloadWorker(self.output_dir)

            async with worker:
                self.assertTrue(worker.startup_called)
                self.assertFalse(worker.shutdown_called)

            self.assertTrue(worker.shutdown_called)

        trio.run(test)

    def test_process_single_scene(self):
        """Test processing a single scene."""

        async def test():
            worker = MockPostDownloadWorker(self.output_dir)

            scene = SceneInfo(
                scene_name="test_scene",
                cfa_type="bayer",
                unknown_sensor=False,
                test_reserve=False,
            )
            result = await worker.process_scene(scene)

            self.assertEqual(result.scene_name, "test_scene")
            self.assertIn("test_scene", worker.processed_scenes)

        trio.run(test)

    def test_consume_and_produce(self):
        """Test consuming scenes from channel and producing to output."""

        async def test():
            worker = MockPostDownloadWorker(self.output_dir, max_workers=2)

            # Create test scenes
            scenes = [
                SceneInfo(
                    scene_name=f"scene_{i}",
                    cfa_type="bayer",
                    unknown_sensor=False,
                    test_reserve=False,
                )
                for i in range(5)
            ]

            # Create channels
            send, recv = trio.open_memory_channel(10)
            output_send, output_recv = trio.open_memory_channel(10)

            async with trio.open_nursery() as nursery:
                # Start worker
                nursery.start_soon(worker.consume_and_produce, recv, output_send)

                # Send scenes
                async with send:
                    for scene in scenes:
                        await send.send(scene)

                # Collect output
                results = []
                async with output_recv:
                    async for scene in output_recv:
                        results.append(scene)

            # Verify all scenes processed
            self.assertEqual(len(results), 5)
            self.assertEqual(len(worker.processed_scenes), 5)

        trio.run(test)

    def test_consume_without_output(self):
        """Test consuming scenes without output channel (sink)."""

        async def test():
            worker = MockPostDownloadWorker(self.output_dir)

            scenes = [
                SceneInfo(
                    scene_name=f"scene_{i}",
                    cfa_type="bayer",
                    unknown_sensor=False,
                    test_reserve=False,
                )
                for i in range(3)
            ]
            send, recv = trio.open_memory_channel(10)

            async with trio.open_nursery() as nursery:
                nursery.start_soon(worker.consume_and_produce, recv, None)

                async with send:
                    for scene in scenes:
                        await send.send(scene)

            self.assertEqual(len(worker.processed_scenes), 3)

        trio.run(test)

    def test_concurrency_control(self):
        """Test that worker respects max_workers limit."""

        async def test():
            worker = MockPostDownloadWorker(self.output_dir, max_workers=2)

            # Track concurrent processing
            concurrent_count = 0
            max_concurrent = 0
            lock = trio.Lock()

            class ConcurrencyTrackingWorker(MockPostDownloadWorker):
                async def process_scene(self, scene):
                    nonlocal concurrent_count, max_concurrent

                    async with lock:
                        concurrent_count += 1
                        max_concurrent = max(max_concurrent, concurrent_count)

                    await trio.sleep(0.05)

                    async with lock:
                        concurrent_count -= 1

                    return scene

            worker = ConcurrencyTrackingWorker(self.output_dir, max_workers=2)

            scenes = [
                SceneInfo(
                    scene_name=f"scene_{i}",
                    cfa_type="bayer",
                    unknown_sensor=False,
                    test_reserve=False,
                )
                for i in range(10)
            ]
            send, recv = trio.open_memory_channel(10)

            async with trio.open_nursery() as nursery:
                nursery.start_soon(worker.consume_and_produce, recv, None)

                async with send:
                    for scene in scenes:
                        await send.send(scene)

            # Max concurrent should not exceed max_workers
            self.assertLessEqual(max_concurrent, 2)

        trio.run(test)

    def test_get_artifact_path(self):
        """Test artifact path generation."""
        worker = MockPostDownloadWorker(self.output_dir)
        scene = SceneInfo(
            scene_name="test_scene",
            cfa_type="bayer",
            unknown_sensor=False,
            test_reserve=False,
        )

        path = worker.get_artifact_path(scene)
        self.assertEqual(path, self.output_dir / "test_scene")
        self.assertTrue(path.exists())

        path_with_suffix = worker.get_artifact_path(scene, "metadata.yaml")
        self.assertEqual(
            path_with_suffix, self.output_dir / "test_scene" / "metadata.yaml"
        )

    def test_worker_name_property(self):
        """Test worker name property returns class name."""
        worker = MockPostDownloadWorker(self.output_dir)
        self.assertEqual(worker.name, "MockPostDownloadWorker")

    def test_error_handling(self):
        """Test that errors in processing don't crash the pipeline."""

        class FailingWorker(PostDownloadWorker):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.error_count = 0

            async def process_scene(self, scene):
                self.error_count += 1
                raise ValueError(f"Processing failed for {scene.scene_name}")

        async def test():
            worker = FailingWorker(self.output_dir)

            scenes = [
                SceneInfo(
                    scene_name=f"scene_{i}",
                    cfa_type="bayer",
                    unknown_sensor=False,
                    test_reserve=False,
                )
                for i in range(3)
            ]
            send, recv = trio.open_memory_channel(10)
            output_send, output_recv = trio.open_memory_channel(10)

            async with trio.open_nursery() as nursery:
                nursery.start_soon(worker.consume_and_produce, recv, output_send)

                async with send:
                    for scene in scenes:
                        await send.send(scene)

                # Should not receive any output due to errors
                results = []
                async with output_recv:
                    async for scene in output_recv:
                        results.append(scene)

            # All scenes should have been attempted
            self.assertEqual(worker.error_count, 3)
            # But no successful outputs
            self.assertEqual(len(results), 0)

        trio.run(test)


if __name__ == "__main__":
    main()
