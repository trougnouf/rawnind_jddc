# smoke_test.py
"""Smoke test for the new pipeline architecture."""
import argparse
import logging
import sys
from pathlib import Path

import trio

from rawnind.dataset import DataIngestor, FileScanner, Downloader, Verifier, SceneIndexer, MetadataEnricher

# Channel buffer size - independent of number of images to process
CHANNEL_BUFFER_SIZE = 10

# Suppress download error messages that mess up the visualization
logging.getLogger('rawnind.dataset.Downloader').setLevel(logging.CRITICAL)


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
            self.BOLD = '\033[1m'
        else:
            self.RESET = ''
            self.GREEN = ''
            self.YELLOW = ''
            self.RED = ''
            self.BOLD = ''
        
        self.counters = {
            'scanned': 0,
            'found': 0,
            'missing': 0,
            'active': 0,
            'finished': 0,
            'verifying': 0,
            'verified': 0,
            'failed': 0,
            'errors': 0,
            'indexing': 0,
            'complete': 0,
            'enriching': 0,
            'enriched': 0,
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

        # If there's no downstream consumer waiting, stay white
        if not has_downstream:
            return self.RESET

        # Track how long this counter has been at zero
        current_time = trio.current_time()
        if counter_name not in self.zero_since:
            self.zero_since[counter_name] = current_time

        time_at_zero = current_time - self.zero_since[counter_name]

        # Red if stalled for > 10 seconds
        if time_at_zero > 10:
            return self.RED
        # Yellow if at zero with downstream waiting
        else:
            return self.YELLOW

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
        indexer_color = self._get_color('indexing', has_downstream=True)

        # Time tracking
        elapsed = trio.current_time() - self._start_time
        elapsed_str = self._format_time(elapsed)

        eta = self._estimate_time_remaining()
        if eta is not None:
            eta_str = self._format_time(eta)
            progress_pct = (self.counters['complete'] / self.total_items * 100) if self.total_items else 0
            time_info = f"│ Elapsed: {elapsed_str}     │\n│ ETA: {eta_str}         │\n│ Progress: {progress_pct:5.1f}%  │"
        else:
            time_info = f"│ Elapsed: {elapsed_str}     │\n│ ETA: --:--         │\n│ Progress: --.--%  │"

        return f"""
╔═════════════════════════════════════════════════════════════════════╗
║                    PIPELINE PROGRESS MONITOR                        ║
╚═════════════════════════════════════════════════════════════════════╝

┌─────────────────┐                                      ┌──────────────────┐
│  DataIngestor   │  {scanner_color}Scanned: {self.counters['scanned']:3d}{self.RESET}                 │ Elapsed: {self._format_time(trio.current_time() - self._start_time) if self._start_time else '00:00'}     │
└────────┬────────┘                                      │ ETA: {self._format_time(self._estimate_time_remaining()) if self._estimate_time_remaining() else '--:--'}         │
         │                                               │ Progress: {(self.counters['complete'] / self.total_items * 100) if self.total_items else 0:5.1f}%  │
         ▼                                               └──────────────────┘
┌─────────────────┐
│  FileScanner    │  Found: {self.counters['found']:3d}  Missing: {self.counters['missing']:3d}
└────┬────────┬───┘
     │        │
     │        └────────────────┐
     │                         ▼
     │                ┌─────────────────┐
     │                │  Downloader     │  {downloader_color}Active: {self.counters['active']:3d}{self.RESET}
     │                └────────┬────────┘  Finished: {self.counters['finished']:3d}
     │                         │
     │          ┌──────────────┘
     │          │
     ▼          ▼
┌─────────────────────┐
│  Merge Channels     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────┐
│   Verifier      │  {verifier_color}Verifying: {self.counters['verifying']:3d}{self.RESET}
└────┬────────┬───┘  Verified:  {self.counters['verified']:3d}
     │        │       Failed:    {self.counters['failed']:3d}
     │        │       Errors:    {self.counters['errors']:3d}
     │        └────────────────────┐
     │                      (retry loop)
     ▼
┌─────────────────┐
│ SceneIndexer    │  {indexer_color}Indexing: {self.counters['indexing']:3d}{self.RESET}
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│MetadataEnricher │  Enriching: {self.counters['enriching']:3d}
└────────┬────────┘  Enriched:  {self.counters['enriched']:3d}
         │
         ▼
┌─────────────────┐
│ Final Consumer  │  Complete: {self.counters['complete']:3d}
└─────────────────┘

{self._render_status_bars()}
"""

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

async def _start_producer_and_limit(ingestor, max_scenes, outbound_send):
    """Start ingestor.produce_scenes into an internal channel then forward up to max_scenes to outbound_send."""
    internal_send, internal_recv = trio.open_memory_channel(CHANNEL_BUFFER_SIZE)

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


async def limited_smoke_test(max_images=None, timeout_seconds=None, debug=False, download_concurrency=2):
    """
    Smoke test for the pipeline architecture.

    Args:
        max_images: Maximum number of complete scenes to process. If None, processes all scenes in dataset.
        timeout_seconds: Maximum time to run in seconds. If None, runs until completion.
        debug: If True, print debug information about enrichment process.
        download_concurrency: Maximum number of concurrent downloads (default: 2).
    """
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
        
        # Create channels with fixed buffer size
        (scene_send, scene_recv) = trio.open_memory_channel(CHANNEL_BUFFER_SIZE)
        (new_file_send, new_file_recv) = trio.open_memory_channel(CHANNEL_BUFFER_SIZE)
        (missing_send, missing_recv) = trio.open_memory_channel(CHANNEL_BUFFER_SIZE)
        (downloaded_send, downloaded_recv) = trio.open_memory_channel(CHANNEL_BUFFER_SIZE)
        (verified_send, verified_recv) = trio.open_memory_channel(CHANNEL_BUFFER_SIZE)
        (complete_scene_send, complete_scene_recv) = trio.open_memory_channel(CHANNEL_BUFFER_SIZE)
        (enriched_send, enriched_recv) = trio.open_memory_channel(CHANNEL_BUFFER_SIZE)

        # Initialize components
        ingestor = DataIngestor(dataset_root=dataset_root)
        scanner = FileScanner(dataset_root)
        downloader = Downloader(max_concurrent=download_concurrency, progress=False)
        verifier = Verifier(max_retries=2)
        indexer = SceneIndexer(dataset_root)
        enricher = MetadataEnricher(dataset_root=dataset_root, enable_crops_enrichment=False)

        # Producer - limits scenes if max_images (max_scenes) is specified
        async def limited_produce_scenes(send_channel):
            if max_images is not None:
                # Use the helper function to limit scenes
                await _start_producer_and_limit(ingestor, max_scenes=max_images, outbound_send=send_channel)
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
                            await viz.update(missing=1)
                            await missing.send(img_info)

        # Downloader with stats tracking
        async def run_downloader(recv, send):
            # Create internal channels for tracking
            internal_missing_send, internal_missing_recv = trio.open_memory_channel(CHANNEL_BUFFER_SIZE)
            internal_downloaded_send, internal_downloaded_recv = trio.open_memory_channel(CHANNEL_BUFFER_SIZE)
            
            async def track_downloads():
                """Track download requests and completions"""
                async with trio.open_nursery() as track_nursery:
                    # Track incoming download requests
                    async def track_incoming():
                        async with recv:
                            async for img_info in recv:
                                await viz.update(active=1)
                                await internal_missing_send.send(img_info)
                        await internal_missing_send.aclose()
                    
                    # Track completed downloads
                    async def track_outgoing():
                        async with internal_downloaded_recv:
                            async for img_info in internal_downloaded_recv:
                                await viz.update(active=-1, finished=1)
                                await send.send(img_info)
                        await send.aclose()
                    
                    # Run the actual downloader
                    async def run_actual_downloader():
                        await downloader.consume_missing(internal_missing_recv, internal_downloaded_send)
                    
                    track_nursery.start_soon(track_incoming)
                    track_nursery.start_soon(run_actual_downloader)
                    track_nursery.start_soon(track_outgoing)
            
            await track_downloads()

        # Merge channels
        merged_send, merged_recv = trio.open_memory_channel(CHANNEL_BUFFER_SIZE)

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
                    await viz.update(indexing=1)
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
            try:
                async with recv:
                    async for scene_info in recv:
                        await viz.update(enriching=1)
                        try:
                            if debug:
                                gt_img = scene_info.get_gt_image()
                                print(f"\nDEBUG: Enriching scene {scene_info.scene_name}")
                                print(f"  GT image: {gt_img.filename if gt_img else 'None'}")
                                if gt_img:
                                    print(f"  GT local_path: {gt_img.local_path}")
                                    print(f"  GT validated: {gt_img.validated}")
                                print(f"  Cache size before: {len(enricher._metadata_cache)}")
                            
                            enriched_scene = await enricher._enrich_scene(scene_info)
                            
                            if debug:
                                print(f"  Cache size after: {len(enricher._metadata_cache)}")
                            
                            await viz.update(enriching=-1, enriched=1)
                            await send.send(enriched_scene)
                        except Exception as e:
                            if debug:
                                print(f"  ERROR during enrichment: {e}")
                            await viz.update(enriching=-1, errors=1)
                            # Still send the scene even if enrichment fails
                            await send.send(scene_info)
            finally:
                # Save cache when done
                if debug:
                    print(f"\nDEBUG: Saving cache with {len(enricher._metadata_cache)} entries")
                enricher._save_cache()
                await send.aclose()

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
        nursery.start_soon(final_consumer, enriched_recv)

    # Final display is already shown by the visualizer
    print("\n\nPipeline completed successfully!")


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
        default=2,
        help="Maximum number of concurrent downloads (default: 2)"
    )
    
    args = parser.parse_args()
    
    trio.run(limited_smoke_test, args.max_images, args.timeout, args.debug, args.download_concurrency)
