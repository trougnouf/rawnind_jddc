# smoke_test.py
"""Smoke test for the new pipeline architecture."""
import argparse
import sys
from pathlib import Path

import trio

from rawnind.dataset import DataIngestor, FileScanner, Downloader, Verifier, SceneIndexer

# Channel buffer size - independent of number of images to process
CHANNEL_BUFFER_SIZE = 10


class PipelineVisualizer:
    """Live visualization of pipeline stage counters with color coding."""

    # ANSI color codes
    RESET = '\033[0m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'

    def __init__(self, total_items=None):
        self.counters = {
            'scanned': 0,
            'found': 0,
            'missing': 0,
            'downloading': 0,
            'downloaded': 0,
            'verifying': 0,
            'verified': 0,
            'failed': 0,
            'indexing': 0,
            'complete': 0,
        }
        # Track when counters become zero (for yellow/red transitions)
        self.zero_since = {}
        self.lock = trio.Lock()
        self._last_lines = 0
        self._start_time = None
        self.total_items = total_items

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
        downloader_color = self._get_color('downloading', has_downstream=True)
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
│  DataIngestor   │  {scanner_color}Scanned: {self.counters['scanned']:3d}{self.RESET}                 {time_info}
└────────┬────────┘                                      └──────────────────┘
         │
         ▼
┌─────────────────┐
│  FileScanner    │  Found: {self.counters['found']:3d}  Missing: {self.counters['missing']:3d}
└────┬────────┬───┘
     │        │
     │        └────────────────┐
     │                         ▼
     │                ┌─────────────────┐
     │                │  Downloader     │  {downloader_color}Downloading: {self.counters['downloading']:3d}{self.RESET}
     │                └────────┬────────┘  Downloaded:  {self.counters['downloaded']:3d}
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
     │        └────────────────────┐
     │                      (retry loop)
     ▼
┌─────────────────┐
│ SceneIndexer    │  {indexer_color}Indexing: {self.counters['indexing']:3d}{self.RESET}
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Final Consumer  │  Complete: {self.counters['complete']:3d}
└─────────────────┘
"""

    async def update(self, **kwargs):
        """Update counters and refresh display."""
        async with self.lock:
            for key, value in kwargs.items():
                if key in self.counters:
                    self.counters[key] += value
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


async def limited_smoke_test(max_images=None, timeout_seconds=None):
    """
    Smoke test for the pipeline architecture.

    Args:
        max_images: Maximum number of complete scenes to process. If None, processes all scenes in dataset.
        timeout_seconds: Maximum time to run in seconds. If None, runs until completion.
    """
    dataset_root = Path("tmp/rawnind_dataset")

    # Create visualizer with expected number of scenes to complete
    # Note: We don't know exactly how many scenes we'll complete,
    # so we'll estimate conservatively (typically 1 scene has multiple images)
    viz = PipelineVisualizer(total_items=1)
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

        # Initialize components
        ingestor = DataIngestor(dataset_root=dataset_root)
        scanner = FileScanner(dataset_root)
        downloader = Downloader(max_concurrent=2, progress=True)
        verifier = Verifier(max_retries=2)
        indexer = SceneIndexer(dataset_root)

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

        # Downloader - just call the actual downloader
        # Note: We lose detailed stats tracking for now, but downloads will actually happen
        async def run_downloader(recv, send):
            await downloader.consume_missing(recv, send)

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
                        await viz.update(verifying=-1, failed=1)
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
        nursery.start_soon(final_consumer, complete_scene_recv)

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
    
    args = parser.parse_args()
    
    trio.run(limited_smoke_test, args.max_images, args.timeout)
