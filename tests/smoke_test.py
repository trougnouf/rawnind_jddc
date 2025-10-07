# smoke_test.py
"""Smoke test for the new pipeline architecture."""
import argparse
import logging
import os
import sys
from pathlib import Path

import trio

from rawnind.dataset import (
    DataIngestor,
    FileScanner,
    Downloader,
    Verifier,
    SceneIndexer,
    MetadataEnricher,
)
from rawnind.dataset.alignment_artifact_writer import AlignmentArtifactWriter
from rawnind.dataset.crop_producer_stage import CropProducerStage
from rawnind.dataset.YAMLArtifactWriter import YAMLArtifactWriter

# Default channel buffer multiplier (relative to download concurrency)
# We don't need huge buffers - 2.5x download concurrency is enough to avoid blocking
# while preventing excessive RAM usage from queued ImageInfo objects
CHANNEL_BUFFER_MULTIPLIER = 2.5

# Default download concurrency: 75% of CPU cores (downloads are I/O bound)
DEFAULT_DOWNLOAD_CONCURRENCY = max(1, int(os.cpu_count() * 0.75))

# Default worker pool size: 75% of CPU cores (for CPU-bound image processing)
DEFAULT_MAX_WORKERS = max(1, int(os.cpu_count() * 0.75))

# Configure file logging to capture detailed debug info
log_file = Path('/tmp/smoke_test.log')
file_handler = logging.FileHandler(log_file, mode='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logging.root.addHandler(file_handler)
logging.root.setLevel(logging.DEBUG)

# Suppress download error messages that mess up the visualization
logging.getLogger('rawnind.dataset.Downloader').setLevel(logging.CRITICAL)
# Enable debug logging for MetadataEnricher to diagnose blocking
logging.getLogger('rawnind.dataset.MetadataEnricher').setLevel(logging.INFO)

print(f"Logging to: {log_file}")


class PipelineVisualizer:
    """Live visualization of pipeline stage counters with color coding."""

    def __init__(self, total_items=None):
        # Detect if we should use colors (only when stdout is a TTY)
        self._use_colors = sys.stdout.isatty()
        
        # ANSI color codes (will be empty strings if colors disabled)
        if self._use_colors:
            self.RESET = '\033[0m'
            self.GREEN = '\033[92m'
            self.YELLOW = '\033[93m'
            self.RED = '\033[91m'
            self.BLUE = '\033[94m'
            self.BOLD = '\033[1m'
        else:
            self.RESET = ''
            self.GREEN = ''
            self.YELLOW = ''
            self.RED = ''
            self.BLUE = ''
            self.BOLD = ''
        
        self.counters = {
            'scanned': 0,
            'found': 0,
            'missing': 0,
            'queued': 0,
            'active': 0,
            'finished': 0,
            'verifying': 0,
            'verified': 0,
            'failed': 0,
            'errors': 0,
            'Indexed': 0,
            'complete': 0,
            'enriching': 0,
            'enriched': 0,
            'aligning': 0,
            'aligned': 0,
            'cropping': 0,
            'cropped': 0,
            'yaml_writing': 0,
            'yaml_written': 0,
        }
        # Track when counters become zero (for yellow/red transitions)
        self.zero_since = {}
        self.lock = trio.Lock()
        self._last_lines = 0
        self._start_time = None
        self.total_items = total_items
        # Download progress tracking
        self.download_start_time = None
        self.total_downloads = 0
        
        # Status bar tracking for bottom of display
        self.active_downloads = 0
        self.completed_downloads = 0

    def _get_color(self, counter_name, has_downstream=True):
        """Get color for a counter based on its value and stall time."""
        import trio
        value = self.counters[counter_name]

        # Green if active (counter > 0)
        if value > 0:
            # Clear zero timestamp if we're active again
            if counter_name in self.zero_since:
                del self.zero_since[counter_name]
            return self.GREEN

        # Track how long this counter has been at zero
        current_time = trio.current_time()
        if counter_name not in self.zero_since:
            self.zero_since[counter_name] = current_time

        time_at_zero = current_time - self.zero_since[counter_name]

        # Blue if at zero for > 1.5 seconds (shows stage is idle/done)
        if time_at_zero > 1.5:
            return self.BLUE
        # Otherwise stay white (neutral)
        else:
            return self.RESET

    def _format_time(self, seconds):
        """Format seconds into HH:MM:SS or MM:SS."""
        if seconds < 0:
            return "--:--"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _estimate_time_remaining(self):
        """Estimate time remaining based on progress."""
        import trio
        if self._start_time is None or self.total_items is None or self.total_items == 0:
            return None

        completed = self.counters['complete']
        if completed == 0:
            return None

        elapsed = trio.current_time() - self._start_time
        rate = completed / elapsed  # items per second
        remaining_items = self.total_items - completed

        if rate > 0:
            return remaining_items / rate
        return None

    def _render(self):
        """Render the pipeline diagram with current counters and color coding."""
        import trio

        # Initialize start time on first render
        if self._start_time is None:
            self._start_time = trio.current_time()

        # Stage colors
        scanner_color = self._get_color('scanned', has_downstream=True)
        downloader_color = self._get_color('active', has_downstream=True)
        verifier_color = self._get_color('verifying', has_downstream=True)
        indexer_color = self._get_color('Indexed', has_downstream=True)
        aligning_color = self._get_color('aligning', has_downstream=True)
        cropping_color = self._get_color('cropping', has_downstream=True)
        yaml_writing_color = self._get_color('yaml_writing', has_downstream=True)

        # Time tracking
        elapsed = trio.current_time() - self._start_time
        elapsed_str = self._format_time(elapsed)

        pct = (self.counters['complete'] / self.total_items * 100) if self.total_items else 0
        eta_str = self._format_time(self._estimate_time_remaining()) if self._estimate_time_remaining() else '--:--'
        
        # Format header with proper padding (box is 62 chars, need space for "║ " and " ║")
        header_content = f"{elapsed_str} │ {pct:4.1f}% │ ETA {eta_str}"
        header_padding = 58 - len(header_content)  # 62 total - "║ " (2) - " ║" (2) = 58 for content+padding
        
        return f"""
╔════════════════════════════════════════════════════════════╗
║ {header_content}{' ' * header_padding} ║
╚════════════════════════════════════════════════════════════╝
│ DataIngestor │ {scanner_color}Scanned: {self.counters['scanned']:3d}{self.RESET}
└──▼
│ FileScanner  │ Found: {self.counters['found']:3d} │ Missing: {self.counters['missing']:3d}
└──┬───────────┴─────────────┐
   │ (exists)        (missing)│
   ▼                          ▼
   │                  ┌───Downloader───┐ Queued: {self.counters['queued']:3d} │ {downloader_color}Active: {self.counters['active']:3d}{self.RESET} │ Finished: {self.counters['finished']:3d}
   │                  └───────┬────────┘
   │◄─────────────────────────┘
   ▼
│ Verifier     │ {verifier_color}Verifying: {self.counters['verifying']:3d}{self.RESET} │ OK: {self.counters['verified']:3d} │ Fail: {self.counters['failed']:3d} │ Err: {self.counters['errors']:3d}
└──┬────────────────┐
   │         (retry)│
   ▼                │
│ SceneIndexer │ {indexer_color}Indexed: {self.counters['Indexed']:3d}{self.RESET}
└──▼
│MetadataEnrich│ Enriching: {self.counters['enriching']:3d} │ Enriched: {self.counters['enriched']:3d}
└──▼
│AlignArtifact │ {aligning_color}Aligning: {self.counters['aligning']:3d}{self.RESET} │ Aligned: {self.counters['aligned']:3d}
└──▼
│ CropProducer │ {cropping_color}Cropping: {self.counters['cropping']:3d}{self.RESET} │ Cropped: {self.counters['cropped']:3d}
└──▼
│ YAMLArtifact │ {yaml_writing_color}Writing: {self.counters['yaml_writing']:3d}{self.RESET} │ Written: {self.counters['yaml_written']:3d}
└──▼
│FinalConsumer │ {self.BOLD}{self.GREEN}Complete: {self.counters['complete']:3d}{self.RESET}
{self._render_status_bars()}"""

    def _render_status_bars(self):
        """Render status bars at the bottom of the display."""
        import trio
        
        bars = []
        
        # Download progress bar
        if self.counters['finished'] > 0 or self.counters['active'] > 0:
            current_time = trio.current_time()
            
            if self.download_start_time is None:
                elapsed = 0
            else:
                elapsed = current_time - self.download_start_time
            
            completed = self.counters['finished']
            active = self.counters['active']
            total = self.total_downloads
            
            # Calculate rate
            if elapsed > 0 and completed > 0:
                rate = completed / elapsed
                time_per_file = elapsed / completed
            else:
                rate = 0
                time_per_file = 0
            
            # Format elapsed time
            elapsed_str = self._format_time(elapsed)
            
            # Format rate
            if time_per_file > 0:
                rate_str = f"{time_per_file:.2f}s/file"
            else:
                rate_str = "?s/file"
            
            # Progress bar
            if total > 0:
                progress = f"Downloads: {completed}/{total} [{elapsed_str}, {rate_str}]"
            else:
                progress = f"Downloads: {completed} [{elapsed_str}, {rate_str}]"
            
            if active > 0:
                progress += f" ({active} active)"
            
            bars.append(progress)
        
        # Verification progress bar
        verified_total = self.counters['verified'] + self.counters['failed']
        if verified_total > 0:
            verified = self.counters['verified']
            failed = self.counters['failed']
            verifying = self.counters['verifying']
            
            progress = f"Verification: {verified} verified, {failed} failed"
            if verifying > 0:
                progress += f" ({verifying} in progress)"
            
            bars.append(progress)
        
        # Enrichment progress bar
        enriched_total = self.counters['enriched']
        if enriched_total > 0 or self.counters['enriching'] > 0:
            enriched = self.counters['enriched']
            enriching = self.counters['enriching']
            
            progress = f"Enrichment: {enriched} complete"
            if enriching > 0:
                progress += f" ({enriching} in progress)"
            
            bars.append(progress)
        
        if bars:
            return "\n" + "\n".join(bars)
        return ""
    
    async def update(self, **kwargs):
        """Update counters and refresh display."""
        async with self.lock:
            # Update counters first
            for key, value in kwargs.items():
                if key in self.counters:
                    self.counters[key] += value
            
            # Track download start time and total downloads
            if 'active' in kwargs and kwargs['active'] > 0:
                if self.download_start_time is None:
                    import trio
                    self.download_start_time = trio.current_time()
                # Track the cumulative total of downloads requested
                self.total_downloads += kwargs['active']
            
            self._display()

    def _display(self):
        """Display the current state to terminal."""
        # Move cursor up to overwrite previous output
        if self._last_lines > 0:
            sys.stdout.write(f'\033[{self._last_lines}A')

        output = self._render()
        lines = output.count('\n')
        self._last_lines = lines

        sys.stdout.write(output)
        sys.stdout.flush()

    def clear(self):
        """Initial display."""
        self._display()


# NOTE: we must NOT open a nursery inside an async generator that yields,
# because Trio forbids doing async work while an async generator is being
# finalized. Instead, start the producer task from a regular async function
# and read from an internal channel.

async def _start_producer_and_limit(ingestor, max_scenes, outbound_send, channel_buffer_size):
    """Start ingestor.produce_scenes into an internal channel then forward up to max_scenes to outbound_send."""
    internal_send, internal_recv = trio.open_memory_channel(channel_buffer_size)

    async with trio.open_nursery() as prod_nursery:
        # Start the real producer feeding the internal_send channel.
        prod_nursery.start_soon(ingestor.produce_scenes, internal_send)

        # Read up to max_scenes from the internal_recv and forward to outbound_send.
        scene_count = 0
        async with internal_recv, outbound_send:
            async for scene_info in internal_recv:
                await outbound_send.send(scene_info)
                scene_count += 1
                if scene_count >= max_scenes:
                    # Stop the background producer and exit
                    prod_nursery.cancel_scope.cancel()
                    break


async def limited_smoke_test(max_images=None, timeout_seconds=None, debug=False, download_concurrency=None):
    """
    Smoke test for the pipeline architecture.

    Args:
        max_images: Maximum number of complete scenes to process. If None, processes all scenes in dataset.
        timeout_seconds: Maximum time to run in seconds. If None, runs until completion.
        debug: If True, print debug information about enrichment process.
        download_concurrency: Maximum number of concurrent downloads. If None, defaults to 75% of CPU cores.
    """
    # Set default download concurrency if not specified
    if download_concurrency is None:
        download_concurrency = DEFAULT_DOWNLOAD_CONCURRENCY

    dataset_root = Path("tmp/rawnind_dataset")

    # Create visualizer with expected number of scenes to complete
    # Note: We don't know exactly how many scenes we'll complete ahead of time
    # Use max_images if provided, otherwise None to disable percentage
    viz = PipelineVisualizer(total_items=max_images)
    viz.clear()

    async with trio.open_nursery() as nursery:
        # Set timeout if specified
        if timeout_seconds is not None:
            nursery.cancel_scope.deadline = trio.current_time() + timeout_seconds
        
        # Calculate channel buffer size based on download concurrency
        # Use 2.5x multiplier to allow some queueing without excessive RAM usage
        channel_buffer_size = int(download_concurrency * CHANNEL_BUFFER_MULTIPLIER)

        # Create channels with dynamic buffer size
        (scene_send, scene_recv) = trio.open_memory_channel(channel_buffer_size)
        (new_file_send, new_file_recv) = trio.open_memory_channel(channel_buffer_size)
        (missing_send, missing_recv) = trio.open_memory_channel(channel_buffer_size)
        (downloaded_send, downloaded_recv) = trio.open_memory_channel(channel_buffer_size)
        (verified_send, verified_recv) = trio.open_memory_channel(channel_buffer_size)
        (complete_scene_send, complete_scene_recv) = trio.open_memory_channel(channel_buffer_size)
        (enriched_send, enriched_recv) = trio.open_memory_channel(channel_buffer_size)
        (aligned_send, aligned_recv) = trio.open_memory_channel(channel_buffer_size)
        (cropped_send, cropped_recv) = trio.open_memory_channel(channel_buffer_size)
        (yaml_send, yaml_recv) = trio.open_memory_channel(channel_buffer_size)

        # Initialize components
        ingestor = DataIngestor(dataset_root=dataset_root)
        scanner = FileScanner(dataset_root)
        downloader = Downloader(max_concurrent=download_concurrency, progress=False)
        verifier = Verifier(max_retries=2)
        indexer = SceneIndexer(dataset_root)
        enricher = MetadataEnricher(dataset_root=dataset_root, enable_crops_enrichment=False)
        aligner = AlignmentArtifactWriter(
            output_dir=dataset_root / "alignment_artifacts",
            max_workers=DEFAULT_MAX_WORKERS
        )
        cropper = CropProducerStage(
            output_dir=dataset_root / "crops",
            crop_size=256,
            num_crops=5,
            max_workers=DEFAULT_MAX_WORKERS
        )
        yaml_writer = YAMLArtifactWriter(
            output_dir=dataset_root,
            output_filename="pipeline_output.yaml"
        )

        # Producer - limits scenes if max_images (max_scenes) is specified
        async def limited_produce_scenes(send_channel):
            if max_images is not None:
                # Use the helper function to limit scenes
                await _start_producer_and_limit(ingestor, max_scenes=max_images, outbound_send=send_channel, channel_buffer_size=channel_buffer_size)
            else:
                # No limit - produce all scenes
                await ingestor.produce_scenes(send_channel)

        # Scanner with stats
        async def scanner_with_stats(recv, new_file, missing):
            async with recv, new_file:
                async for scene_info in recv:
                    await viz.update(scanned=len(scene_info.all_images()))
                    # Manually route images
                    scene_dir = dataset_root / scene_info.cfa_type / scene_info.scene_name
                    gt_dir = scene_dir / "gt"

                    for img_info in scene_info.all_images():
                        candidates = []
                        if img_info.is_clean:
                            candidates = [gt_dir / img_info.filename, scene_dir / img_info.filename]
                        else:
                            candidates = [scene_dir / img_info.filename]

                        found = False
                        for candidate in candidates:
                            if candidate.exists():
                                img_info.local_path = candidate
                                await viz.update(found=1)
                                await new_file.send(img_info)
                                found = True
                                break

                        if not found:
                            img_info.local_path = candidates[0] if candidates else None
                            # Clear any cached image data to save RAM while in download queue
                            img_info.unload_image()
                            await viz.update(missing=1)
                            await missing.send(img_info)

        # Downloader with stats tracking
        async def run_downloader(recv, send):
            # Create internal channels for tracking
            internal_missing_send, internal_missing_recv = trio.open_memory_channel(channel_buffer_size)
            internal_downloaded_send, internal_downloaded_recv = trio.open_memory_channel(channel_buffer_size)

            # Track total items in the download pipeline
            total_in_pipeline = 0
            pipeline_lock = trio.Lock()

            async def track_downloads():
                """Track download requests and completions"""
                nonlocal total_in_pipeline

                async with trio.open_nursery() as track_nursery:
                    # Track incoming download requests
                    async def track_incoming():
                        nonlocal total_in_pipeline
                        async with recv:
                            async for img_info in recv:
                                async with pipeline_lock:
                                    total_in_pipeline += 1
                                # Track download start time and total count
                                if viz.download_start_time is None:
                                    viz.download_start_time = trio.current_time()
                                viz.total_downloads += 1
                                await internal_missing_send.send(img_info)
                        await internal_missing_send.aclose()

                    # Track completed downloads
                    async def track_outgoing():
                        nonlocal total_in_pipeline
                        async with internal_downloaded_recv:
                            async for img_info in internal_downloaded_recv:
                                async with pipeline_lock:
                                    total_in_pipeline -= 1
                                await viz.update(finished=1)
                                await send.send(img_info)
                        await send.aclose()

                    # Periodically update queued/active from actual state
                    async def monitor_pipeline():
                        nonlocal total_in_pipeline
                        prev_queued = 0
                        prev_active = 0

                        while True:
                            await trio.sleep(0.1)  # Update every 100ms
                            stats = internal_missing_send.statistics()
                            queued_count = stats.current_buffer_used

                            async with pipeline_lock:
                                # Active = total in pipeline - queued in buffer
                                # (capped at download_concurrency since that's max possible)
                                active_count = min(total_in_pipeline - queued_count, download_concurrency)
                                active_count = max(0, active_count)

                                # Calculate deltas and update via proper method
                                queued_delta = queued_count - prev_queued
                                active_delta = active_count - prev_active

                                if queued_delta != 0 or active_delta != 0:
                                    await viz.update(queued=queued_delta, active=active_delta)

                                prev_queued = queued_count
                                prev_active = active_count

                    # Run the actual downloader
                    async def run_actual_downloader():
                        await downloader.consume_missing(internal_missing_recv, internal_downloaded_send)

                    track_nursery.start_soon(track_incoming)
                    track_nursery.start_soon(monitor_pipeline)
                    track_nursery.start_soon(run_actual_downloader)
                    track_nursery.start_soon(track_outgoing)

            await track_downloads()

        # Merge channels
        merged_send, merged_recv = trio.open_memory_channel(channel_buffer_size)

        async def merge_inputs():
            async with merged_send:
                async with trio.open_nursery() as merge_nursery:
                    async def forward(recv):
                        async with recv:
                            async for item in recv:
                                await merged_send.send(item)

                    merge_nursery.start_soon(forward, new_file_recv)
                    merge_nursery.start_soon(forward, downloaded_recv)

        # Verifier with stats
        async def verifier_with_stats(recv, verified, missing):
            async with recv, verified:
                async for img_info in recv:
                    await viz.update(verifying=1)
                    if img_info.local_path and img_info.local_path.exists():
                        # Compute hash
                        from rawnind.dataset import hash_sha1
                        computed = await trio.to_thread.run_sync(hash_sha1, img_info.local_path)

                        if computed == img_info.sha1:
                            img_info.validated = True
                            await viz.update(verifying=-1, verified=1)
                            await verified.send(img_info)
                        else:
                            await viz.update(verifying=-1, failed=1)
                            if img_info.retry_count < 2:
                                img_info.retry_count += 1
                                img_info.local_path.unlink()
                                img_info.local_path = None
                                await missing.send(img_info)
                            else:
                                # Max retries exceeded - count as error
                                await viz.update(errors=1)
                    else:
                        await viz.update(verifying=-1, failed=1)
                        # File doesn't exist - count as error if max retries exceeded
                        if img_info.retry_count >= 2:
                            await viz.update(errors=1)
            # Close missing channel after verification is done
            await missing.aclose()

        # Indexer with stats
        async def indexer_with_stats(recv, send):
            async with recv, send:
                async for img_info in recv:
                    await viz.update(Indexed=1)
                    indexer._add_image_to_index(img_info)

                    scene_key = (img_info.cfa_type, img_info.scene_name)
                    if scene_key not in indexer._scene_completion_tracker:
                        if indexer._is_scene_complete(img_info):
                            scene_info = indexer._construct_scene(img_info)
                            indexer._scene_completion_tracker.add(scene_key)
                            indexer._move_scene_to_complete(scene_info)
                            await send.send(scene_info)

        # Enricher with stats
        async def enricher_with_stats(recv, send):
            # Load cache before enrichment loop
            await enricher._cache.load()
            logging.info(f"Enricher cache loaded with {len(enricher._cache.keys())} entries")
            
            try:
                async with recv:
                    async for scene_info in recv:
                        await viz.update(enriching=1)
                        try:
                            if debug:
                                gt_img = scene_info.get_gt_image()
                                print(f"DEBUG: Enriching scene {scene_info.scene_name}")
                                print(f"  GT image: {gt_img.filename if gt_img else 'None'}")
                                if gt_img:
                                    print(f"  GT local_path: {gt_img.local_path}")
                                    print(f"  GT validated: {gt_img.validated}")
                                print(f"  Cache size before: {len(enricher._cache.keys())}")
                            
                            enriched_scene = await enricher._enrich_scene(scene_info)
                            
                            if debug:
                                print(f"  Cache size after: {len(enricher._cache.keys())}")
                            
                            await viz.update(enriching=-1, enriched=1)
                            await send.send(enriched_scene)
                        except Exception as e:
                            if debug:
                                print(f"  ERROR during enrichment: {e}")
                            await viz.update(enriching=-1, errors=1)
                            # Still send the scene even if enrichment fails
                            await send.send(scene_info)
            finally:
                await send.aclose()

        # Aligner with stats
        async def aligner_with_stats(recv, send):
            # Start aligner
            await aligner.startup()
            try:
                async with recv, send:
                    async for scene_info in recv:
                        await viz.update(aligning=1)
                        try:
                            aligned_scene = await aligner.process_scene(scene_info)
                            await viz.update(aligning=-1, aligned=1)
                            await send.send(aligned_scene)
                        except Exception as e:
                            if debug:
                                print(f"  ERROR during alignment: {e}")
                            await viz.update(aligning=-1, errors=1)
                            await send.send(scene_info)
            finally:
                await aligner.shutdown()

        # Cropper with stats
        async def cropper_with_stats(recv, send):
            # Start cropper
            await cropper.startup()
            try:
                async with recv, send:
                    async for scene_info in recv:
                        await viz.update(cropping=1)
                        try:
                            cropped_scene = await cropper.process_scene(scene_info)
                            await viz.update(cropping=-1, cropped=1)
                            await send.send(cropped_scene)
                        except Exception as e:
                            if debug:
                                print(f"  ERROR during cropping: {e}")
                            await viz.update(cropping=-1, errors=1)
                            await send.send(scene_info)
            finally:
                await cropper.shutdown()

        # YAML writer with stats
        async def yaml_writer_with_stats(recv, send):
            # Start writer
            await yaml_writer.startup()
            try:
                async with recv, send:
                    async for scene_info in recv:
                        await viz.update(yaml_writing=1)
                        try:
                            yaml_scene = await yaml_writer.process_scene(scene_info)
                            await viz.update(yaml_writing=-1, yaml_written=1)
                            await send.send(yaml_scene)
                        except Exception as e:
                            if debug:
                                print(f"  ERROR during YAML writing: {e}")
                            await viz.update(yaml_writing=-1, errors=1)
                            await send.send(scene_info)
            finally:
                # Ensure YAML is written on completion
                await yaml_writer.shutdown()

        # Final consumer
        async def final_consumer(recv):
            async with recv:
                completed = 0
                async for scene_info in recv:
                    await viz.update(complete=1)
                    completed += 1
                    # If we've reached max_images, cancel the entire pipeline
                    if max_images is not None and completed >= max_images:
                        nursery.cancel_scope.cancel()

        # Start pipeline stages
        nursery.start_soon(limited_produce_scenes, scene_send)
        nursery.start_soon(scanner_with_stats, scene_recv, new_file_send, missing_send)
        nursery.start_soon(run_downloader, missing_recv, downloaded_send)
        nursery.start_soon(merge_inputs)
        nursery.start_soon(verifier_with_stats, merged_recv, verified_send, missing_send)
        nursery.start_soon(indexer_with_stats, verified_recv, complete_scene_send)
        nursery.start_soon(enricher_with_stats, complete_scene_recv, enriched_send)
        nursery.start_soon(aligner_with_stats, enriched_recv, aligned_send)
        nursery.start_soon(cropper_with_stats, aligned_recv, cropped_send)
        nursery.start_soon(yaml_writer_with_stats, cropped_recv, yaml_send)
        nursery.start_soon(final_consumer, yaml_recv)

    # Final display is already shown by the visualizer
    print("\n\n" + "="*60)
    print("Pipeline completed!")
    print("="*60)

    # Show summary of errors if any
    total_errors = viz.counters['errors']
    failed_verifications = viz.counters['failed']

    if total_errors > 0 or failed_verifications > 0:
        print("\n⚠️  ISSUES DETECTED:")
        if failed_verifications > 0:
            print(f"  • {failed_verifications} files failed verification (hash mismatch)")
        if total_errors > 0:
            print(f"  • {total_errors} errors occurred during processing")
        print("\n💡 Check logs for details:")
        print("  - Re-run with --debug flag for enrichment details")
        print("  - Check dataset logs in tmp/rawnind_dataset/")
    else:
        print("\n✅ No errors detected!")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline smoke test with live visualization")
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        dest="max_images",  # Keep internal name for compatibility
        help="Maximum number of complete scenes to process (default: process all scenes)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Maximum time to run in seconds (default: no timeout)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information about enrichment process"
    )
    parser.add_argument(
        "--download-concurrency",
        type=int,
        default=None,
        help=f"Maximum number of concurrent downloads (default: {DEFAULT_DOWNLOAD_CONCURRENCY}, which is 75%% of {os.cpu_count()} CPU cores)"
    )
    
    args = parser.parse_args()
    
    trio.run(limited_smoke_test, args.max_images, args.timeout, args.debug, args.download_concurrency)
