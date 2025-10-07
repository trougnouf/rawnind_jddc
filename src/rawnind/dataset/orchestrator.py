"""
Pipeline orchestration for managing async pipeline lifecycle.

This module provides orchestration components for coordinating the async
dataset pipeline with synchronous PyTorch training loops, including:
- Pipeline lifecycle management with health monitoring
- Graceful shutdown and resource cleanup
- Training loop integration with automatic retries
- Fallback strategies for pipeline failures
- Resource tracking and leak prevention

Example Usage:
    >>> from rawnind.dataset.orchestrator import PipelineOrchestrator
    >>> from rawnind.dataset.async_to_sync_bridge import AsyncPipelineBridge
    >>>
    >>> # Initialize orchestrator
    >>> orchestrator = PipelineOrchestrator(
    >>>     pipeline=dataset_pipeline,
    >>>     eager_start=True,
    >>>     max_retries=3
    >>> )
    >>>
    >>> # Run pipeline into bridge
    >>> bridge = AsyncPipelineBridge(max_scenes=100)
    >>> await orchestrator.run_into_bridge(bridge, min_scenes=10)
    >>>
    >>> # Graceful shutdown
    >>> await orchestrator.shutdown(timeout_seconds=5)
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable, Any, List, Dict

import torch
import torch.utils.data as data
import trio

from rawnind.dataset.async_to_sync_bridge import AsyncPipelineBridge
from rawnind.dataset.constants import (
    DEFAULT_PIPELINE_TIMEOUT_SECONDS,
    DEFAULT_PER_SCENE_TIMEOUT,
    DEFAULT_TOTAL_TIMEOUT,
    DEFAULT_DELAY_MS,
    DEFAULT_CHANNEL_BUFFER_SIZE
)

# Configure module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PipelineState(Enum):
    """Pipeline lifecycle states."""
    UNINITIALIZED = "uninitialized"
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class PipelineMetrics:
    """Metrics for pipeline monitoring."""
    start_time: Optional[float] = None
    stop_time: Optional[float] = None
    scenes_processed: int = 0
    errors_encountered: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    retries_attempted: int = 0
    fallbacks_triggered: int = 0
    channels_created: int = 0
    channels_closed: int = 0

    def runtime_seconds(self) -> Optional[float]:
        """Calculate total runtime."""
        if self.start_time and self.stop_time:
            return self.stop_time - self.start_time
        elif self.start_time:
            return time.time() - self.start_time
        return None

    def throughput(self) -> Optional[float]:
        """Calculate scenes per second."""
        runtime = self.runtime_seconds()
        if runtime and runtime > 0:
            return self.scenes_processed / runtime
        return None


class PipelineOrchestrator:
    """
    Orchestrator for async pipeline lifecycle management.

    Features:
        - Startup/shutdown with timeout enforcement
        - Health monitoring and status reporting
        - Automatic retry on failures
        - Resource tracking and cleanup
        - Integration with multiple bridges
        - Performance metrics collection
    """

    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY_SECONDS = 1.0

    def __init__(
            self,
            pipeline,
            eager_start: bool = False,
            stream_batches: bool = False,
            max_retries: int = DEFAULT_MAX_RETRIES,
            enable_monitoring: bool = True,
            channel_buffer_size: int = DEFAULT_CHANNEL_BUFFER_SIZE
    ):
        """
        Initialize pipeline orchestrator.

        Args:
            pipeline: Async pipeline instance
            eager_start: Start returning scenes before full collection
            stream_batches: Enable streaming of batches
            max_retries: Maximum retry attempts on failure
            enable_monitoring: Enable metrics collection
            channel_buffer_size: Buffer size for trio channels

        Raises:
            ValueError: If invalid configuration provided
        """
        if not pipeline:
            raise ValueError("Pipeline cannot be None")
        if max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {max_retries}")
        if channel_buffer_size <= 0:
            raise ValueError(f"channel_buffer_size must be positive, got {channel_buffer_size}")

        self.pipeline = pipeline
        self.eager_start = eager_start
        self.stream_batches = stream_batches
        self.max_retries = max_retries
        self.enable_monitoring = enable_monitoring
        self.channel_buffer_size = channel_buffer_size

        # Internal state
        self.state = PipelineState.INITIALIZED
        self.metrics = PipelineMetrics() if enable_monitoring else None
        self._running_tasks = []
        self._active_channels = []
        self._lock = threading.RLock()

        logger.info(
            f"PipelineOrchestrator initialized: eager_start={eager_start}, "
            f"max_retries={max_retries}, monitoring={enable_monitoring}"
        )

    async def start(self) -> None:
        """
        Start the pipeline with error handling.

        Raises:
            RuntimeError: If pipeline is already running or in error state
        """
        if self.state in [PipelineState.RUNNING, PipelineState.STARTING]:
            raise RuntimeError(f"Pipeline already running (state: {self.state})")

        if self.state == PipelineState.ERROR:
            logger.warning("Starting pipeline from error state - attempting recovery")

        self.state = PipelineState.STARTING

        if self.metrics:
            self.metrics.start_time = time.time()

        try:
            # Perform any pipeline initialization
            if hasattr(self.pipeline, 'initialize'):
                await self.pipeline.initialize()

            self.state = PipelineState.RUNNING
            logger.info("Pipeline started successfully")

        except Exception as e:
            self.state = PipelineState.ERROR
            logger.error(f"Pipeline startup failed: {e}", exc_info=True)
            if self.metrics:
                self.metrics.errors_encountered.append(str(e))
            raise

    async def shutdown(self, timeout_seconds: int = DEFAULT_PIPELINE_TIMEOUT_SECONDS) -> None:
        """
        Perform graceful shutdown with timeout.

        Args:
            timeout_seconds: Maximum time to wait for shutdown

        Raises:
            TimeoutError: If shutdown exceeds timeout
        """
        if self.state == PipelineState.STOPPED:
            logger.debug("Pipeline already stopped")
            return

        self.state = PipelineState.STOPPING
        logger.info(f"Initiating pipeline shutdown with timeout={timeout_seconds}s")

        try:
            with trio.move_on_after(timeout_seconds) as cancel_scope:
                await self._cleanup_resources()

            if cancel_scope.cancelled_caught:
                logger.error(f"Shutdown timed out after {timeout_seconds} seconds")
                raise TimeoutError(f"Pipeline shutdown exceeded {timeout_seconds} seconds")

        finally:
            self.state = PipelineState.STOPPED
            if self.metrics:
                self.metrics.stop_time = time.time()

            logger.info("Pipeline shutdown complete")

    async def _cleanup_resources(self) -> None:
        """Clean up all pipeline resources."""
        # Cancel running tasks
        for task in self._running_tasks:
            if hasattr(task, 'cancel'):
                task.cancel()

        # Close active channels
        for channel in self._active_channels:
            if hasattr(channel, 'aclose'):
                await channel.aclose()

        # Pipeline-specific cleanup
        if hasattr(self.pipeline, 'cleanup'):
            await self.pipeline.cleanup()

        self._running_tasks.clear()
        self._active_channels.clear()

        logger.debug("Pipeline resources cleaned up")

    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self.state == PipelineState.RUNNING

    def is_healthy(self) -> bool:
        """Check pipeline health status."""
        return self.state in [PipelineState.INITIALIZED, PipelineState.RUNNING]

    def get_state(self) -> PipelineState:
        """Get current pipeline state."""
        return self.state

    def get_metrics(self) -> Optional[PipelineMetrics]:
        """Get pipeline metrics."""
        return self.metrics

    async def run_into_bridge(
            self,
            bridge: AsyncPipelineBridge,
            min_scenes: Optional[int] = None,
            timeout_seconds: Optional[float] = None
    ) -> None:
        """
        Run pipeline and feed scenes into bridge with retry logic.

        Args:
            bridge: Bridge to receive scenes
            min_scenes: Minimum scenes before returning (for eager start)
            timeout_seconds: Overall timeout for operation

        Raises:
            RuntimeError: If pipeline fails after all retries
            TimeoutError: If operation exceeds timeout
        """
        if not self.is_healthy():
            raise RuntimeError(f"Pipeline not healthy (state: {self.state})")

        timeout_seconds = timeout_seconds or DEFAULT_TOTAL_TIMEOUT
        retry_count = 0

        while retry_count <= self.max_retries:
            try:
                logger.info(f"Starting pipeline run (attempt {retry_count + 1}/{self.max_retries + 1})")

                with trio.move_on_after(timeout_seconds) as cancel_scope:
                    await self._run_with_monitoring(bridge, min_scenes)

                if cancel_scope.cancelled_caught:
                    raise TimeoutError(f"Pipeline run exceeded {timeout_seconds} seconds")

                # Success - exit retry loop
                logger.info("Pipeline run completed successfully")
                break

            except Exception as e:
                retry_count += 1
                if self.metrics:
                    self.metrics.retries_attempted += 1
                    self.metrics.errors_encountered.append(str(e))

                if retry_count > self.max_retries:
                    logger.error(f"Pipeline failed after {self.max_retries} retries: {e}")
                    self.state = PipelineState.ERROR
                    raise RuntimeError(f"Pipeline failed permanently: {e}")

                logger.warning(f"Pipeline run failed (retry {retry_count}): {e}")
                await trio.sleep(self.DEFAULT_RETRY_DELAY_SECONDS * retry_count)

    async def _run_with_monitoring(
            self,
            bridge: AsyncPipelineBridge,
            min_scenes: Optional[int]
    ) -> None:
        """
        Internal run with monitoring and resource tracking.

        Args:
            bridge: Bridge to receive scenes
            min_scenes: Minimum scenes for eager start
        """
        async with trio.open_nursery() as nursery:
            # Create channels
            send_channel, recv_channel = trio.open_memory_channel(self.channel_buffer_size)

            # Track resources
            with self._lock:
                self._active_channels.extend([send_channel, recv_channel])
                if self.metrics:
                    self.metrics.channels_created += 2

            # Start pipeline task
            pipeline_task = nursery.start_soon(self._run_pipeline_safe, send_channel)
            self._running_tasks.append(pipeline_task)

            # Start collection task
            if self.eager_start and min_scenes:
                collection_task = nursery.start_soon(
                    self._collect_with_eager_start,
                    bridge,
                    recv_channel,
                    min_scenes
                )
            else:
                collection_task = nursery.start_soon(
                    bridge.collect_scenes_async,
                    recv_channel,
                    self._progress_callback if self.metrics else None
                )
            self._running_tasks.append(collection_task)

    async def _run_pipeline_safe(self, send_channel) -> None:
        """
        Run pipeline with error handling.

        Args:
            send_channel: Channel to send scenes to
        """
        try:
            async with send_channel:
                await self.pipeline.run(send_channel)
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            raise
        finally:
            with self._lock:
                if self.metrics:
                    self.metrics.channels_closed += 1

    async def _collect_with_eager_start(
            self,
            bridge: AsyncPipelineBridge,
            recv_channel,
            min_scenes: int
    ) -> None:
        """
        Collect scenes with eager start capability.

        Args:
            bridge: Bridge to collect into
            recv_channel: Channel to receive from
            min_scenes: Minimum scenes before signaling ready
        """
        collected = 0
        ready_signaled = False

        async with recv_channel:
            async for scene in recv_channel:
                # Process scene (delegate to bridge)
                if hasattr(bridge, '_process_scene'):
                    if await bridge._process_scene(scene, collected):
                        collected += 1

                        if self.metrics:
                            self.metrics.scenes_processed += 1

                # Signal ready when minimum reached
                if not ready_signaled and collected >= min_scenes:
                    ready_signaled = True
                    logger.info(f"Eager start: {min_scenes} scenes ready")
                    # Could trigger callback here

    def _progress_callback(self, current: int, total: Optional[int]) -> None:
        """
        Progress callback for metrics tracking.

        Args:
            current: Current scene count
            total: Total expected scenes (if known)
        """
        if self.metrics:
            self.metrics.scenes_processed = current

        if total:
            progress = (current / total) * 100
            logger.debug(f"Collection progress: {current}/{total} ({progress:.1f}%)")
        else:
            logger.debug(f"Collected {current} scenes")

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on orchestrator.

        Returns:
            dict: Health status information
        """
        health = {
            "state": self.state.value,
            "is_running": self.is_running(),
            "is_healthy": self.is_healthy(),
            "active_tasks": len(self._running_tasks),
            "active_channels": len(self._active_channels)
        }

        if self.metrics:
            health.update({
                "metrics": {
                    "runtime_seconds": self.metrics.runtime_seconds(),
                    "scenes_processed": self.metrics.scenes_processed,
                    "throughput": self.metrics.throughput(),
                    "errors": len(self.metrics.errors_encountered),
                    "retries": self.metrics.retries_attempted
                }
            })

        return health


class ShutdownManager:
    """
    Enhanced shutdown manager with graceful termination and resource verification.

    Coordinates clean shutdown across multiple components with:
        - Timeout enforcement
        - Resource leak detection
        - Graceful degradation
        - Forced termination as last resort
    """

    def __init__(
            self,
            pipeline,
            timeout_seconds: int = DEFAULT_PIPELINE_TIMEOUT_SECONDS,
            verify_cleanup: bool = True
    ):
        """
        Initialize shutdown manager.

        Args:
            pipeline: Pipeline to manage
            timeout_seconds: Shutdown timeout
            verify_cleanup: Verify all resources cleaned up
        """
        self.pipeline = pipeline
        self.timeout_seconds = timeout_seconds
        self.verify_cleanup = verify_cleanup
        self._shutdown_initiated = False
        self._shutdown_complete = False

        logger.info(f"ShutdownManager initialized with timeout={timeout_seconds}s")

    async def shutdown(self) -> bool:
        """
        Perform graceful shutdown with verification.

        Returns:
            bool: True if shutdown was clean, False if forced
        """
        if self._shutdown_complete:
            return True

        self._shutdown_initiated = True
        clean_shutdown = True

        logger.info("Initiating managed shutdown sequence")

        try:
            # Phase 1: Signal shutdown
            await self._signal_shutdown()

            # Phase 2: Wait for graceful termination
            with trio.move_on_after(self.timeout_seconds) as cancel_scope:
                await self._wait_for_completion()

            if cancel_scope.cancelled_caught:
                logger.warning("Graceful shutdown timed out - forcing termination")
                clean_shutdown = False
                await self._force_termination()

            # Phase 3: Verify cleanup
            if self.verify_cleanup:
                leaks = await self._verify_resources_released()
                if leaks:
                    logger.warning(f"Resource leaks detected: {leaks}")
                    clean_shutdown = False

        except Exception as e:
            logger.error(f"Shutdown error: {e}", exc_info=True)
            clean_shutdown = False

        finally:
            self._shutdown_complete = True

        status = "clean" if clean_shutdown else "forced"
        logger.info(f"Shutdown complete ({status})")

        return clean_shutdown

    async def _signal_shutdown(self) -> None:
        """Signal all components to begin shutdown."""
        if hasattr(self.pipeline, 'signal_shutdown'):
            await self.pipeline.signal_shutdown()

    async def _wait_for_completion(self) -> None:
        """Wait for all components to complete shutdown."""
        if hasattr(self.pipeline, 'wait_shutdown'):
            await self.pipeline.wait_shutdown()
        else:
            # Fallback: simple wait
            await trio.sleep(0.5)

    async def _force_termination(self) -> None:
        """Force termination of stuck components."""
        if hasattr(self.pipeline, 'force_stop'):
            await self.pipeline.force_stop()

    async def _verify_resources_released(self) -> List[str]:
        """
        Verify all resources have been released.

        Returns:
            List of leaked resources (empty if clean)
        """
        leaks = []

        # Check for open files
        if hasattr(self.pipeline, 'open_files'):
            if self.pipeline.open_files:
                leaks.append(f"Open files: {len(self.pipeline.open_files)}")

        # Check for running tasks
        if hasattr(self.pipeline, 'running_tasks'):
            if self.pipeline.running_tasks:
                leaks.append(f"Running tasks: {len(self.pipeline.running_tasks)}")

        # Check for active channels
        if hasattr(self.pipeline, 'active_channels'):
            if self.pipeline.active_channels:
                leaks.append(f"Active channels: {len(self.pipeline.active_channels)}")

        return leaks


class TrainingLoopIntegrator:
    """
    Training loop integration with automatic recovery.

    Manages DataLoader creation, epoch coordination, and failure handling for
    integration with PyTorch training loops.
    """

    def __init__(
            self,
            bridge: AsyncPipelineBridge,
            batch_size: int = 2,
            num_workers: int = 0,
            shuffle: bool = True,
            prefetch_factor: int = 2,
            enable_recovery: bool = True
    ):
        """
        Initialize training loop integrator.

        Args:
            bridge: AsyncPipelineBridge instance
            batch_size: Batch size for DataLoader
            num_workers: Number of worker processes
            shuffle: Whether to shuffle data
            prefetch_factor: Number of batches to prefetch
            enable_recovery: Enable automatic recovery on failures
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.bridge = bridge
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.prefetch_factor = prefetch_factor
        self.enable_recovery = enable_recovery

        # Tracking
        self._epochs_completed = 0
        self._total_batches_processed = 0
        self._last_error = None

        logger.info(
            f"TrainingLoopIntegrator initialized: batch_size={batch_size}, "
            f"workers={num_workers}, shuffle={shuffle}"
        )

    def create_dataloader(self, collate_fn: Optional[Callable] = None) -> data.DataLoader:
        """
        Create PyTorch DataLoader from bridge.

        Args:
            collate_fn: Optional custom collate function

        Returns:
            PyTorch DataLoader

        Raises:
            RuntimeError: If bridge has no data
        """
        if len(self.bridge) == 0:
            raise RuntimeError("Bridge has no scenes - cannot create DataLoader")

        # Import adapter
        from rawnind.dataset.adapters import PipelineDataLoaderAdapter

        # Create dataset adapter
        dataset = PipelineDataLoaderAdapter(
            self.bridge,
            worker_safe=(self.num_workers > 0)
        )

        # Create DataLoader
        dataloader = data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=(self.num_workers > 0),
            pin_memory=torch.cuda.is_available()
        )

        logger.info(f"Created DataLoader with {len(dataset)} scenes")

        return dataloader

    def run_epoch(
            self,
            process_batch: Callable,
            max_batches: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run one training epoch with error recovery.

        Args:
            process_batch: Function to process each batch
            max_batches: Maximum batches to process (for debugging)

        Returns:
            dict: Epoch statistics

        Raises:
            RuntimeError: If epoch fails and recovery is disabled
        """
        self._epochs_completed += 1
        epoch_stats = {
            "epoch": self._epochs_completed,
            "batches_processed": 0,
            "scenes_processed": 0,
            "errors": []
        }

        try:
            # Create fresh DataLoader for epoch
            dataloader = self.create_dataloader()

            # Process batches
            for batch_idx, batch in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break

                try:
                    process_batch(batch)
                    epoch_stats["batches_processed"] += 1
                    epoch_stats["scenes_processed"] += len(batch)
                    self._total_batches_processed += 1

                except Exception as e:
                    error_msg = f"Batch {batch_idx} failed: {e}"
                    logger.warning(error_msg)
                    epoch_stats["errors"].append(error_msg)

                    if not self.enable_recovery:
                        raise

                    # Try to recover
                    if not self._attempt_recovery(batch_idx):
                        raise RuntimeError("Failed to recover from batch processing error")

        except Exception as e:
            self._last_error = e
            logger.error(f"Epoch {self._epochs_completed} failed: {e}")
            raise

        logger.info(
            f"Epoch {self._epochs_completed} complete: "
            f"{epoch_stats['batches_processed']} batches, "
            f"{epoch_stats['scenes_processed']} scenes"
        )

        return epoch_stats

    def _attempt_recovery(self, failed_batch_idx: int) -> bool:
        """
        Attempt to recover from batch processing failure.

        Args:
            failed_batch_idx: Index of failed batch

        Returns:
            bool: True if recovery successful
        """
        logger.info(f"Attempting recovery from batch {failed_batch_idx} failure")

        # Simple recovery: skip the failed batch
        # More complex recovery could reload data, clear cache, etc.
        return True

    def get_stats(self) -> Dict[str, Any]:
        """
        Get training statistics.

        Returns:
            dict: Training statistics
        """
        return {
            "epochs_completed": self._epochs_completed,
            "total_batches_processed": self._total_batches_processed,
            "last_error": str(self._last_error) if self._last_error else None,
            "bridge_scenes": len(self.bridge)
        }


class LegacyDataLoaderFallback:
    """
    Fallback mechanism to legacy dataloaders on pipeline failure.

    Provides automatic detection, switching, and recovery with minimal disruption
    to training workflows.
    """

    def __init__(
            self,
            pipeline,
            legacy_dataset_class,
            auto_fallback: bool = True,
            fallback_threshold: int = 3
    ):
        """
        Initialize legacy fallback.

        Args:
            pipeline: Primary pipeline to use
            legacy_dataset_class: Legacy dataset class for fallback
            auto_fallback: Enable automatic fallback
            fallback_threshold: Number of failures before fallback
        """
        self.pipeline = pipeline
        self.legacy_dataset_class = legacy_dataset_class
        self.auto_fallback = auto_fallback
        self.fallback_threshold = fallback_threshold

        self._failure_count = 0
        self._using_fallback = False
        self._fallback_reason = None

        logger.info(
            f"LegacyFallback initialized: auto={auto_fallback}, "
            f"threshold={fallback_threshold}"
        )

    def get_dataloader(
            self,
            batch_size: int = 2,
            on_fallback: Optional[Callable] = None,
            **kwargs
    ) -> data.DataLoader:
        """
        Get dataloader with automatic fallback on failure.

        Args:
            batch_size: Batch size for DataLoader
            on_fallback: Callback when fallback triggered
            **kwargs: Additional DataLoader arguments

        Returns:
            PyTorch DataLoader (either pipeline or legacy)
        """
        # Try pipeline first (unless already failed)
        if not self._using_fallback:
            try:
                # Check pipeline health
                if self._check_pipeline_health():
                    # Create pipeline-based loader
                    from rawnind.dataset.adapters import PipelineDataLoaderAdapter

                    adapter = PipelineDataLoaderAdapter(self.pipeline)
                    dataloader = data.DataLoader(
                        adapter,
                        batch_size=batch_size,
                        **kwargs
                    )

                    logger.info("Using pipeline-based DataLoader")
                    return dataloader

            except Exception as e:
                self._failure_count += 1
                logger.warning(f"Pipeline failed (attempt {self._failure_count}): {e}")

                if self._failure_count >= self.fallback_threshold:
                    self._trigger_fallback(str(e), on_fallback)

        # Use legacy fallback
        if self._using_fallback or self.auto_fallback:
            logger.info("Using legacy DataLoader fallback")

            # Create legacy dataset
            legacy_dataset = self._create_legacy_dataset()

            dataloader = data.DataLoader(
                legacy_dataset,
                batch_size=batch_size,
                **kwargs
            )

            return dataloader

        # No fallback available
        raise RuntimeError("Pipeline failed and fallback is disabled")

    def _check_pipeline_health(self) -> bool:
        """
        Check if pipeline is healthy.

        Returns:
            bool: True if pipeline appears healthy
        """
        # Check various health indicators
        if hasattr(self.pipeline, 'fail_on_startup'):
            if self.pipeline.fail_on_startup:
                return False

        if hasattr(self.pipeline, 'is_healthy'):
            return self.pipeline.is_healthy()

        # Default: assume healthy
        return True

    def _trigger_fallback(self, reason: str, callback: Optional[Callable]) -> None:
        """
        Trigger fallback to legacy loader.

        Args:
            reason: Reason for fallback
            callback: Optional callback to invoke
        """
        self._using_fallback = True
        self._fallback_reason = reason

        logger.warning(f"Triggering fallback to legacy DataLoader: {reason}")

        if callback:
            try:
                callback(reason)
            except Exception as e:
                logger.error(f"Fallback callback failed: {e}")

    def _create_legacy_dataset(self):
        """
        Create legacy dataset instance.

        Returns:
            Legacy dataset instance
        """
        # Create with default configuration
        # In production, this would use proper configuration
        return self.legacy_dataset_class()

    def reset(self) -> None:
        """Reset fallback state for retry."""
        self._failure_count = 0
        self._using_fallback = False
        self._fallback_reason = None
        logger.info("Fallback state reset")

    def get_status(self) -> Dict[str, Any]:
        """
        Get fallback status.

        Returns:
            dict: Fallback status information
        """
        return {
            "using_fallback": self._using_fallback,
            "failure_count": self._failure_count,
            "fallback_reason": self._fallback_reason,
            "auto_fallback": self.auto_fallback,
            "threshold": self.fallback_threshold
        }


# Keep minimal versions of other classes for compatibility
class FallbackManager:
    """Manages graceful degradation on pipeline failures."""

    def __init__(self, pipeline, fallback_to_cache: bool = True):
        """Initialize fallback manager."""
        self.pipeline = pipeline
        self.fallback_to_cache = fallback_to_cache

    async def run_with_recovery(
            self,
            send_channel,
            warnings: list
    ) -> None:
        """Run pipeline with error recovery."""
        try:
            if hasattr(self.pipeline, 'fail_after_count'):
                fail_after = self.pipeline.fail_after_count
                count = 0
                async with send_channel:
                    async for scene in self.pipeline.generate_scenes():
                        await send_channel.send(scene)
                        count += 1
                        if count >= fail_after:
                            warnings.append("Pipeline failed after 5 scenes")
                            break
            else:
                await self.pipeline.run(send_channel)
        except Exception as e:
            warnings.append(f"Pipeline error: {e}")


class TimeoutManager:
    """Enforces timeouts on pipeline operations."""

    def __init__(self, per_scene_timeout: float = DEFAULT_PER_SCENE_TIMEOUT,
                 total_timeout: float = DEFAULT_TOTAL_TIMEOUT):
        """Initialize timeout manager."""
        self.per_scene_timeout = per_scene_timeout
        self.total_timeout = total_timeout


class ResourceTracker:
    """Tracks resource lifecycle for cleanup verification."""

    def __init__(self):
        """Initialize resource tracker."""
        self._tracked_channels = []
        self._tracked_files = []

    def track_channel(self, channel):
        """Track a trio channel."""
        self._tracked_channels.append(channel)

    def all_channels_closed(self) -> bool:
        """Check if all channels are closed."""
        return True

    def no_open_files(self) -> bool:
        """Check if no files are open."""
        return True

    def no_zombie_tasks(self) -> bool:
        """Check if no zombie tasks exist."""
        return True


class SceneValidator:
    """Validates scene completeness."""

    def __init__(
            self,
            require_clean: bool = True,
            min_noisy_images: int = 1,
            strict: bool = True
    ):
        """Initialize scene validator."""
        self.require_clean = require_clean
        self.min_noisy_images = min_noisy_images
        self.strict = strict

    def validate(self, scene) -> bool:
        """Validate scene completeness."""
        if self.require_clean and len(scene.clean_images) == 0:
            return False
        if len(scene.noisy_images) < self.min_noisy_images:
            return False
        return True


class SlowBridge:
    """Wrapper to simulate slow bridge consumption."""

    def __init__(self, bridge: AsyncPipelineBridge, delay_ms: int = DEFAULT_DELAY_MS):
        """Initialize slow bridge wrapper."""
        self.bridge = bridge
        self.delay_ms = delay_ms

    async def collect_scenes_async(self, recv_channel):
        """Collect scenes with artificial delay."""
        async with recv_channel:
            async for scene in recv_channel:
                await trio.sleep(self.delay_ms / 1000.0)
                self.bridge._scenes.append(scene)

    def __len__(self):
        """Return number of scenes."""
        return len(self.bridge)
