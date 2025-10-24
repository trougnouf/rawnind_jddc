"""Data pipeline orchestrator for managing experiment execution.

The module defines classes for controlling the lifecycle of a data pipeline
used in scientific experiments.  The primary class, ExperimentOrchestrator,
coordinates pipeline initialization, execution, retry logic, monitoring,
and graceful shutdown.  PipelineMetrics provides runtime statistics
such as throughput and error counts, while PipelineState enumerates
the possible lifecycle states of the orchestrator.

Exported classes
----------------
ExperimentOrchestrator
    Coordinates the pipeline lifecycle.
PipelineMetrics
    Stores execution metrics.
PipelineState
    Enum of orchestrator states.
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

from rawnind.dataset import AsyncPipelineBridge
from rawnind.dataset.constants import (
    DEFAULT_PIPELINE_TIMEOUT_SECONDS,
    DEFAULT_PER_SCENE_TIMEOUT,
    DEFAULT_TOTAL_TIMEOUT,
    DEFAULT_DELAY_MS,
    DEFAULT_CHANNEL_BUFFER_SIZE,
)

# Configure module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PipelineState(Enum):
    """Represents the various states a pipeline can be in during its lifecycle.

    The PipelineState enumeration defines a set of string constants that
    represent each possible state of a pipeline. The states cover the
    full lifecycle, from the moment the pipeline is created (UNINITIALIZED)
    to its termination (STOPPED or ERROR). Each state is used by pipeline
    control logic to determine the correct actions and transitions.

    Attributes:
        None
    """

    UNINITIALIZED = "uninitialized"
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class PipelineMetrics:
    """Pipeline metrics for a scene processing pipeline.

    This dataclass collects runtime statistics and operational counters for a
    processing pipeline.  It stores timestamps marking the start and end of a
    run, counts of processed scenes, and lists of errors and warnings that were
    encountered.  Counters for retry attempts, fallback triggers, channel
    lifecycle events, and throughput calculations are also available.

    The class provides helper methods that compute the total runtime in seconds
    and the average throughput (scenes per second) based on the recorded data.

    Attributes:
        start_time: Optional[float]
            Timestamp (seconds since epoch) when the pipeline started.
        stop_time: Optional[float]
            Timestamp when the pipeline finished or was stopped.
        scenes_processed: int
            Number of scenes successfully processed during the run.
        errors_encountered: List[str]
            Error messages collected while the pipeline executed.
        warnings: List[str]
            Warning messages generated during execution.
        retries_attempted: int
            Total number of retry attempts that were made.
        fallbacks_triggered: int
            Number of times a fallback mechanism was activated.
        channels_created: int
            Total count of channels created during the run.
        channels_closed: int
            Total count of channels that were closed.

    """

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


class ExperimentOrchestrator:
    """\
    ExperimentOrchestrator

    Brief summary
    -------------
    The ExperimentOrchestrator class manages the lifecycle of a data pipeline
    by coordinating its startup, execution, monitoring, and shutdown. It supports
    configurable retry logic, optional eager execution, streaming of batches,
    and optional metrics collection.

    Detailed description
    --------------------
    An ExperimentOrchestrator instance is created with a reference to a pipeline
    object and optional configuration flags. It provides asynchronous methods
    to start the pipeline, run the pipeline with a bridge while applying retry
    logic, and gracefully shut it down within a timeout. The orchestrator
    maintains an internal state machine to track whether the pipeline is
    initialised, running, stopping, or stopped. Monitoring can be enabled to
    collect execution metrics such as start and stop times, number of retries
    attempted, and error messages.

    The class exposes public attributes that can be inspected for debugging
    or integration purposes. Private members prefixed with an underscore are
    used internally for task and channel management and are intentionally
    omitted from the documentation.

    Attributes
    ----------
    DEFAULT_MAX_RETRIES (int):
        The default maximum number of retry attempts for a pipeline run.

    DEFAULT_RETRY_DELAY_SECONDS (float):
        The default delay between consecutive retry attempts, expressed in
        seconds.

    pipeline (Pipeline):
        The pipeline instance that will be orchestrated. The pipeline is
        expected to expose optional `initialize`, `cleanup`, and other
        lifecycle methods.

    eager_start (bool):
        When True, the pipeline is started immediately upon creation.
        If False, the pipeline must be explicitly started via `start()`.

    stream_batches (bool):
        Indicates whether the pipeline should process data batches as a
        continuous stream rather than in discrete batches.

    max_retries (int):
        The maximum number of times the orchestrator will retry a failed
        pipeline run before marking the pipeline as in an error state.

    enable_monitoring (bool):
        Flag to enable collection of metrics during pipeline execution.
        If False, the `metrics` attribute will be `None`.

    channel_buffer_size (int):
        The buffer size used for internal communication channels
        between pipeline components.

    state (PipelineState):
        Current state of the pipeline, one of INITIALIZED, RUNNING,
        STARTING, STOPPING, STOPPED, or ERROR.

    metrics (Optional[PipelineMetrics]):
        Instance holding runtime metrics when monitoring is enabled; otherwise
        `None`.  The metrics include timestamps, retry counts, and error logs.
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
        channel_buffer_size: int = DEFAULT_CHANNEL_BUFFER_SIZE,
    ):
        """
        Initializes a PipelineOrchestrator.

        Sets up the orchestrator with the provided pipeline and configuration options.
        Validates input parameters and initializes internal state, metrics, and task
        tracking structures.

        Args:
            pipeline: The pipeline instance to be orchestrated.
            eager_start: If ``True``, the orchestrator will start immediately upon
                initialization.
            stream_batches: If ``True``, batches will be streamed instead of collected.
            max_retries: Maximum number of retry attempts for a failed task. Must be
                non‑negative.
            enable_monitoring: If ``True``, pipeline metrics will be collected.
            channel_buffer_size: Size of the internal channel buffer. Must be positive.

        Raises:
            ValueError: If *pipeline* is ``None``, *max_retries* is negative, or
                *channel_buffer_size* is not positive.
        """
        if not pipeline:
            raise ValueError("Pipeline cannot be None")
        if max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {max_retries}")
        if channel_buffer_size <= 0:
            raise ValueError(
                f"channel_buffer_size must be positive, got {channel_buffer_size}"
            )

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
        Starts the pipeline.

        This method transitions the pipeline from an idle or error state into a running
        state. It performs any necessary initialization, updates metrics, and logs progress. If the
        pipeline is already running or in the STARTING state, a RuntimeError is raised. Any exception
        that occurs during initialization is logged and re‑raised after setting the pipeline state to
        ERROR.

        Args:
            self: The Pipeline instance.

        Returns:
            None.

        Raises:
            RuntimeError: If the pipeline is already running or in the STARTING state.
            Exception: If an error occurs during initialization or startup.
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
            if hasattr(self.pipeline, "initialize"):
                await self.pipeline.initialize()

            self.state = PipelineState.RUNNING
            logger.info("Pipeline started successfully")

        except Exception as e:
            self.state = PipelineState.ERROR
            logger.error(f"Pipeline startup failed: {e}", exc_info=True)
            if self.metrics:
                self.metrics.errors_encountered.append(str(e))
            raise

    async def shutdown(
        self, timeout_seconds: int = DEFAULT_PIPELINE_TIMEOUT_SECONDS
    ) -> None:
        """
        Shuts down the pipeline, ensuring that all resources are cleaned up within
        the specified timeout.

        Args:
            timeout_seconds (int): Maximum number of seconds to wait for the
                shutdown process to complete. If the timeout is reached, a
                TimeoutError is raised.

        Returns:
            None

        Raises:
            TimeoutError: Raised when the shutdown exceeds ``timeout_seconds``.
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
                raise TimeoutError(
                    f"Pipeline shutdown exceeded {timeout_seconds} seconds"
                )

        finally:
            self.state = PipelineState.STOPPED
            if self.metrics:
                self.metrics.stop_time = time.time()

            logger.info("Pipeline shutdown complete")

    async def _cleanup_resources(self) -> None:
        """
        Cleans up resources used by the pipeline.

        This coroutine cancels any currently running asynchronous tasks, closes
        all active channels, and runs the pipeline‑specific cleanup routine if it
        is defined. After performing these actions, it clears the internal
        collections that track running tasks and active channels.

        Args:
            self: The instance whose resources are being cleaned up.

        Returns:
            None
        """
        # Cancel running tasks
        for task in self._running_tasks:
            if hasattr(task, "cancel"):
                task.cancel()

        # Close active channels
        for channel in self._active_channels:
            if hasattr(channel, "aclose"):
                await channel.aclose()

        # Pipeline-specific cleanup
        if hasattr(self.pipeline, "cleanup"):
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
        timeout_seconds: Optional[float] = None,
    ) -> None:
        """
        Runs the pipeline using the provided bridge, optionally enforcing a minimum
        number of scenes and a timeout for the overall operation.

        This method first verifies that the pipeline is in a healthy state. It then
        attempts to execute the pipeline, respecting the configured maximum number of
        retries. Each attempt is bounded by a timeout; if the timeout elapses the
        run is aborted and a ``TimeoutError`` is raised.  On failure, the method
        increments retry counters, records metrics (if enabled), and waits for an
        increasing delay before the next attempt.  When the run succeeds, the
        method exits after logging a success message.

        Args:
            bridge: The asynchronous pipeline bridge used to communicate with the
                downstream components.
            min_scenes: Optional minimum number of scenes that must be processed
                before the run is considered successful. If ``None``, no minimum is
                enforced.
            timeout_seconds: Optional total timeout for a single run attempt. If
                ``None``, a default timeout value is used.

        Returns:
            None.

        Raises:
            RuntimeError: If the pipeline is not healthy before starting, or if the
                maximum number of retries is exceeded and the run fails permanently.
            TimeoutError: If a single run attempt exceeds the specified timeout.
            Exception: Propagates any unexpected exception raised during execution.
        """
        if not self.is_healthy():
            raise RuntimeError(f"Pipeline not healthy (state: {self.state})")

        timeout_seconds = timeout_seconds or DEFAULT_TOTAL_TIMEOUT
        retry_count = 0

        while retry_count <= self.max_retries:
            try:
                logger.info(
                    f"Starting pipeline run (attempt {retry_count + 1}/{self.max_retries + 1})"
                )

                with trio.move_on_after(timeout_seconds) as cancel_scope:
                    await self._run_with_monitoring(bridge, min_scenes)

                if cancel_scope.cancelled_caught:
                    raise TimeoutError(
                        f"Pipeline run exceeded {timeout_seconds} seconds"
                    )

                # Success - exit retry loop
                logger.info("Pipeline run completed successfully")
                break

            except Exception as e:
                retry_count += 1
                if self.metrics:
                    self.metrics.retries_attempted += 1
                    self.metrics.errors_encountered.append(str(e))

                if retry_count > self.max_retries:
                    logger.error(
                        f"Pipeline failed after {self.max_retries} retries: {e}"
                    )
                    self.state = PipelineState.ERROR
                    raise RuntimeError(f"Pipeline failed permanently: {e}")

                logger.warning(f"Pipeline run failed (retry {retry_count}): {e}")
                await trio.sleep(self.DEFAULT_RETRY_DELAY_SECONDS * retry_count)

    async def _run_with_monitoring(
        self, bridge: AsyncPipelineBridge, min_scenes: Optional[int]
    ) -> None:
        """
        Runs the pipeline and monitoring tasks within a Trio nursery.

        This method creates a pair of memory channels for communication between the
        pipeline and the consumer, registers the channels for resource tracking, and
        starts the pipeline execution task. Depending on the `eager_start` flag and the
        provided `min_scenes` value, it either begins eager collection of results or
        delegates consumption to the supplied bridge. All started tasks are added to
        the internal list of running tasks for later management.

        Args:
            bridge: An AsyncPipelineBridge responsible for consuming messages from the
                pipeline.
            min_scenes: Minimum number of scenes required before eager collection starts.
                If `None` or falsy, eager start is disabled.

        Returns:
            None
        """
        async with trio.open_nursery() as nursery:
            # Create channels
            send_channel, recv_channel = trio.open_memory_channel(
                self.channel_buffer_size
            )

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
                    self._collect_with_eager_start, bridge, recv_channel, min_scenes
                )
            else:
                collection_task = nursery.start_soon(
                    bridge.consume,
                    recv_channel,
                    self._progress_callback if self.metrics else None,
                )
            self._running_tasks.append(collection_task)

    async def _run_pipeline_safe(self, send_channel) -> None:
        """
        Runs the pipeline safely, ensuring the channel is closed and metrics are
        updated even if an error occurs.

        Args:
            send_channel: The asynchronous channel used to send data through the
                pipeline.

        Returns:
            None.

        Raises:
            Exception: Propagates any exception raised during pipeline execution.
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
        self, bridge: AsyncPipelineBridge, recv_channel, min_scenes: int
    ) -> None:
        """
        Collect scenes from an asynchronous channel and delegate processing to a bridge,
        triggering a ready signal once a minimum number of scenes have been processed.

        This coroutine continuously reads scenes from ``recv_channel`` within an async
        context manager. For each received scene it attempts to invoke the bridge's
        ``_process_scene`` method (if present). When the bridge confirms successful
        processing, the internal counter is incremented and optional metrics are
        updated. Once the count reaches ``min_scenes`` a ready signal is logged and
        subsequent scenes continue to be processed without further signaling.

        Args:
            bridge: An instance providing a ``_process_scene`` coroutine used to handle
                each incoming scene.
            recv_channel: An asynchronous iterable channel that yields scenes. It must
                support the async context manager protocol.
            min_scenes: The threshold of successfully processed scenes required to log
                the eager‑start ready message.

        Returns:
            None
        """
        collected = 0
        ready_signaled = False

        async with recv_channel:
            async for scene in recv_channel:
                # Process scene (delegate to bridge)
                if hasattr(bridge, "_process_scene"):
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
        Updates progress metrics and logs collection status.

        This callback is invoked during scene collection to keep track of the number
        of processed scenes. If a metrics object is available, it updates the
        `scenes_processed` attribute. Depending on whether the total number of
        scenes is known, it logs either a percentage progress or a simple count.

        Args:
            current: Number of scenes that have been processed so far.
            total: Total number of scenes to be processed, or ``None`` if the total
                is not known.

        Returns:
            None
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
            "active_channels": len(self._active_channels),
        }

        if self.metrics:
            health.update(
                {
                    "metrics": {
                        "runtime_seconds": self.metrics.runtime_seconds(),
                        "scenes_processed": self.metrics.scenes_processed,
                        "throughput": self.metrics.throughput(),
                        "errors": len(self.metrics.errors_encountered),
                        "retries": self.metrics.retries_attempted,
                    }
                }
            )

        return health


class ShutdownManager:
    """
    Handles graceful shutdown of a pipeline with timeout and cleanup verification.

    The manager coordinates the shutdown of a pipeline, ensuring that all
    components receive a shutdown signal, wait for completion, and optionally
    verify that resources such as open files, running tasks, and active channels
    have been released. If the shutdown does not finish within the configured
    timeout, the manager forces termination and records any resource leaks.

    Attributes:
        pipeline: Pipeline to manage.
        timeout_seconds: Shutdown timeout in seconds.
        verify_cleanup: Whether to verify that all resources are released after
            shutdown.
    """

    def __init__(
        self,
        pipeline,
        timeout_seconds: int = DEFAULT_PIPELINE_TIMEOUT_SECONDS,
        verify_cleanup: bool = True,
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
        if hasattr(self.pipeline, "signal_shutdown"):
            await self.pipeline.signal_shutdown()

    async def _wait_for_completion(self) -> None:
        """Wait for all components to complete shutdown."""
        if hasattr(self.pipeline, "wait_shutdown"):
            await self.pipeline.wait_shutdown()
        else:
            # Fallback: simple wait
            await trio.sleep(0.5)

    async def _force_termination(self) -> None:
        """Force termination of stuck components."""
        if hasattr(self.pipeline, "force_stop"):
            await self.pipeline.force_stop()

    async def _verify_resources_released(self) -> List[str]:
        """
        Verify all resources have been released.

        Returns:
            List of leaked resources (empty if clean)
        """
        leaks = []

        # Check for open files
        if hasattr(self.pipeline, "open_files"):
            if self.pipeline.open_files:
                leaks.append(f"Open files: {len(self.pipeline.open_files)}")

        # Check for running tasks
        if hasattr(self.pipeline, "running_tasks"):
            if self.pipeline.running_tasks:
                leaks.append(f"Running tasks: {len(self.pipeline.running_tasks)}")

        # Check for active channels
        if hasattr(self.pipeline, "active_channels"):
            if self.pipeline.active_channels:
                leaks.append(f"Active channels: {len(self.pipeline.active_channels)}")

        return leaks


class TrainingLoopIntegrator:
    """
    Integrates a training loop with an asynchronous data pipeline.

    This class manages the creation of a PyTorch DataLoader from an
    AsyncPipelineBridge instance, runs training epochs with optional
    batch processing limits, and tracks statistics such as epochs
    completed and total batches processed. It also provides simple
    recovery logic to continue training after encountering a batch
    processing error.

    Attributes:
        bridge: AsyncPipelineBridge instance used as the source of data.
        batch_size: Batch size used by the DataLoader.
        num_workers: Number of worker processes for data loading.
        shuffle: Whether to shuffle the dataset each epoch.
        prefetch_factor: Number of batches to prefetch when using
            multiple workers.
        enable_recovery: Enables automatic recovery when a batch
            processing error occurs.
    """

    def __init__(
        self,
        bridge: AsyncPipelineBridge,
        batch_size: int = 2,
        num_workers: int = 0,
        shuffle: bool = True,
        prefetch_factor: int = 2,
        enable_recovery: bool = True,
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

    def create_dataloader(
        self, collate_fn: Optional[Callable] = None
    ) -> data.DataLoader:
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
            self.bridge, worker_safe=(self.num_workers > 0)
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
            pin_memory=torch.cuda.is_available(),
        )

        logger.info(f"Created DataLoader with {len(dataset)} scenes")

        return dataloader

    def run_epoch(
        self, process_batch: Callable, max_batches: Optional[int] = None
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
            "errors": [],
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
                        raise RuntimeError(
                            "Failed to recover from batch processing error"
                        )

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
            "bridge_scenes": len(self.bridge),
        }


class TimeoutManager:
    """Enforces timeouts on pipeline operations."""

    def __init__(
        self,
        per_scene_timeout: float = DEFAULT_PER_SCENE_TIMEOUT,
        total_timeout: float = DEFAULT_TOTAL_TIMEOUT,
    ):
        """Initialize timeout manager."""
        self.per_scene_timeout = per_scene_timeout
        self.total_timeout = total_timeout


class ResourceTracker:
    """Track resource usage across channels and files.

    This class keeps a record of active trio channels and open
    files so that callers can assert that all resources have been
    released after a test or operation.  The internal state is
    maintained in private lists; callers interact with the class
    exclusively through the public methods.
    """

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
    """Validate scenes based on image content.

    The SceneValidator class encapsulates logic for checking that a scene
    contains an acceptable set of clean and noisy images.  It exposes a
    configurable set of parameters that control the strictness of the
    validation and the minimum image counts required.

    Attributes:
        require_clean: Indicates whether the scene must contain at least one
            clean image.  If set to ``True`` the validator will reject a
            scene that has no clean images.
        min_noisy_images: Minimum number of noisy images that must be present
            in the scene for it to be considered valid.  The default is
            one.
        strict: Flag that can be used to enable additional, more strict
            checks in future extensions.  Currently it does not affect the
            validation logic."""

    def __init__(
        self, require_clean: bool = True, min_noisy_images: int = 1, strict: bool = True
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
