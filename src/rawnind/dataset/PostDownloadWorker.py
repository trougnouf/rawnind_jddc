"""
Base class for post-processing stages that produce artifacts from enriched scenes.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any
from concurrent.futures import ProcessPoolExecutor
import trio

from .SceneInfo import SceneInfo

logger = logging.getLogger(__name__)


class PostDownloadWorker(ABC):
    """
    Base class for post-processing stages.

    Consumes enriched SceneInfo objects and produces artifacts on disk.
    CPU-heavy operations use trio.to_thread or ProcessPoolExecutor.
    """

    def __init__(
        self,
        output_dir: Path,
        max_workers: int = 4,
        use_process_pool: bool = False,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the post-download worker.

        Args:
            output_dir: Base directory for writing artifacts
            max_workers: Maximum concurrent workers for CPU-bound tasks
            use_process_pool: If True, use ProcessPoolExecutor instead of threads
            config: Additional configuration for the specific worker
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.use_process_pool = use_process_pool
        self.config = config or {}
        self._executor = None

        if use_process_pool:
            # Initialize process pool for CPU-heavy operations
            self._executor = ProcessPoolExecutor(max_workers=max_workers)

    async def __aenter__(self):
        """Async context manager entry."""
        await self.startup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.shutdown()
        if self._executor:
            # Shutdown the ProcessPoolExecutor with wait=True
            await trio.to_thread.run_sync(lambda: self._executor.shutdown(wait=True))

    async def startup(self):
        """
        Initialize any resources needed by the worker.
        Override in subclasses for custom initialization.
        """
        logger.info(f"{self.__class__.__name__} starting up")

    async def shutdown(self):
        """
        Clean up resources used by the worker.
        Override in subclasses for custom cleanup.
        """
        logger.info(f"{self.__class__.__name__} shutting down")

    @abstractmethod
    async def process_scene(self, scene: SceneInfo) -> SceneInfo:
        """
        Process a single scene and produce artifacts.

        Args:
            scene: Enriched SceneInfo object

        Returns:
            The same or modified SceneInfo for downstream stages
        """
        raise NotImplementedError

    async def consume_and_produce(
        self,
        input_channel: trio.MemoryReceiveChannel,
        output_channel: Optional[trio.MemorySendChannel] = None
    ):
        """
        Main processing loop that consumes scenes and produces artifacts.

        Args:
            input_channel: Channel receiving enriched SceneInfo objects
            output_channel: Optional channel to forward processed scenes
        """
        async with input_channel:
            if output_channel:
                async with output_channel:
                    await self._process_loop(input_channel, output_channel)
            else:
                await self._process_loop(input_channel, None)

    async def _process_loop(
        self,
        input_channel: trio.MemoryReceiveChannel,
        output_channel: Optional[trio.MemorySendChannel]
    ):
        """Internal processing loop with proper backpressure control."""
        # Create semaphore to limit concurrent scene processing
        # This prevents unbounded task accumulation in the nursery
        sem = trio.Semaphore(self.max_workers)

        async with trio.open_nursery() as nursery:
            async for scene in input_channel:
                # Acquire semaphore BEFORE spawning task
                # This blocks if we already have max_workers tasks running
                await sem.acquire()
                nursery.start_soon(
                    self._process_one_with_semaphore,
                    scene,
                    output_channel,
                    sem
                )

    async def _process_one_with_semaphore(
        self,
        scene: SceneInfo,
        output_channel: Optional[trio.MemorySendChannel],
        sem: trio.Semaphore
    ):
        """Process a single scene and release semaphore when done."""
        try:
            processed_scene = await self.process_scene(scene)

            if output_channel:
                await output_channel.send(processed_scene)

        except Exception as e:
            logger.error(
                f"Error processing scene {scene.scene_name}: {e}",
                exc_info=True
            )
        finally:
            # Always release semaphore, even if processing failed
            sem.release()

    async def _process_one(
        self,
        scene: SceneInfo,
        output_channel: Optional[trio.MemorySendChannel]
    ):
        """Process a single scene (legacy method for compatibility)."""
        try:
            processed_scene = await self.process_scene(scene)

            if output_channel:
                await output_channel.send(processed_scene)

        except Exception as e:
            logger.error(
                f"Error processing scene {scene.scene_name}: {e}",
                exc_info=True
            )

    async def run_cpu_bound(self, func, *args, **kwargs):
        """
        Run a CPU-bound function in a thread or process pool.

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result from the function
        """
        if self.use_process_pool and self._executor:
            # Run in process pool for true parallelism
            future = await trio.to_thread.run_sync(
                self._executor.submit, func, *args, **kwargs
            )
            return await trio.to_thread.run_sync(future.result)
        else:
            # Run in thread pool (default)
            return await trio.to_thread.run_sync(func, *args, **kwargs)

    def get_artifact_path(self, scene: SceneInfo, suffix: str = "") -> Path:
        """
        Generate a consistent artifact path for a scene.

        Args:
            scene: SceneInfo object
            suffix: Additional suffix for the artifact

        Returns:
            Path object for the artifact
        """
        scene_dir = self.output_dir / scene.scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)

        if suffix:
            return scene_dir / suffix
        return scene_dir

    @property
    def name(self) -> str:
        """Return the worker's name for logging and metrics."""
        return self.__class__.__name__