#!/usr/bin/env python3
"""Smoke test for the pipeline architecture with live visualization (Async Crop Producer variant)."""

import argparse
import logging
import os
from pathlib import Path

import trio
import tracemalloc
import psutil

from rawnind.dataset import (
    DataIngestor,
    Downloader,
    SceneIndexer,
    AsyncAligner,
)
from rawnind.dataset.Aligner import MetadataArtificer
from rawnind.dataset.crop_producer_stage_async import CropProducerStageAsync
from rawnind.dataset.AsyncPipelineBridge import AsyncPipelineBridge
from rawnind.dataset.visualizer import PipelineVisualizer
from rawnind.dataset.channel_utils import (
    create_channel_dict,
    merge_channels,
    limit_producer,
)

# Constants
CHANNEL_BUFFER_MULTIPLIER = 2.5
DEFAULT_DOWNLOAD_CONCURRENCY = max(1, int(os.cpu_count() * 0.75))
DEFAULT_MAX_WORKERS = max(1, int(os.cpu_count() * 0.75))
LOG_FILE = Path('/tmp/smoke_test_async_crops.log')
MEMORY_LOG_FILE = Path('/tmp/smoke_test_async_crops_memory.log')


class MemoryProfiler:
    """Lightweight memory profiler for smoke test."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.snapshots = []
        self.log_file = MEMORY_LOG_FILE.open('w')

    def snapshot(self, label: str):
        snap = tracemalloc.take_snapshot()
        rss_mb = self.process.memory_info().rss / (1024 * 1024)
        self.snapshots.append({'label': label, 'snapshot': snap, 'rss_mb': rss_mb})
        msg = f"[{label}] RSS: {rss_mb:.1f} MB"
        self.log_file.write(msg + '\n')
        self.log_file.flush()

    def compare_last(self, top_n=10):
        if len(self.snapshots) < 2:
            return
        s1, s2 = self.snapshots[-2], self.snapshots[-1]
        delta = s2['rss_mb'] - s1['rss_mb']
        msg = f"\n{s1['label']} → {s2['label']}: Δ{delta:+.1f} MB"
        self.log_file.write(msg + '\n')

        top = s2['snapshot'].compare_to(s1['snapshot'], 'lineno')
        self.log_file.write(f"Top {top_n} increases:\n")
        for stat in top[:top_n]:
            self.log_file.write(f"  {stat}\n")
        self.log_file.flush()

    def report(self):
        if not self.snapshots:
            return
        peak = max(s['rss_mb'] for s in self.snapshots)
        growth = self.snapshots[-1]['rss_mb'] - self.snapshots[0]['rss_mb']
        msg = f"\n{'='*60}\nMemory Summary: Peak={peak:.1f} MB, Growth={growth:+.1f} MB\n{'='*60}\n"
        self.log_file.write(msg)
        self.log_file.close()
        print(f"Memory profiling logged to: {MEMORY_LOG_FILE}")


def setup_logging():
    """Configure logging for the smoke test."""
    file_handler = logging.FileHandler(LOG_FILE, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logging.root.addHandler(file_handler)
    logging.root.setLevel(logging.DEBUG)

    logging.getLogger('rawnind.dataset.Downloader').setLevel(logging.CRITICAL)
    logging.getLogger('rawnind.dataset.AsyncAligner').setLevel(logging.INFO)
    logging.getLogger('rawnind.dataset.MetadataArtificer').setLevel(logging.DEBUG)
    logging.getLogger('rawnind.dataset.crop_producer_stage_async').setLevel(logging.DEBUG)

    print(f"Logging to: {LOG_FILE}")


async def scanner_with_stats(dataset_root, recv, new_file_send, missing_send, viz):
    """Scanner stage with metrics tracking."""
    async with recv, new_file_send:
        async for scene_info in recv:
            await viz.update(scanned=len(scene_info.all_images()))

            scene_dir = dataset_root / scene_info.cfa_type / scene_info.scene_name
            gt_dir = scene_dir / "gt"

            for img_info in scene_info.all_images():
                candidates = (
                    [gt_dir / img_info.filename, scene_dir / img_info.filename]
                    if img_info.is_clean
                    else [scene_dir / img_info.filename]
                )

                found = False
                for candidate in candidates:
                    if candidate.exists():
                        img_info.local_path = candidate
                        await viz.update(found=1)
                        await new_file_send.send(img_info)
                        found = True
                        break

                if not found:
                    img_info.local_path = candidates[0] if candidates else None
                    img_info.unload_image()
                    await viz.update(missing=1)
                    await missing_send.send(img_info)


async def downloader_with_stats(downloader, recv, send, viz):
    """Downloader stage with metrics tracking."""
    async with recv, send:
        async with trio.open_nursery() as nursery:
            sem = trio.Semaphore(downloader.max_concurrent)

            async def download_one(img_info):
                await viz.update(queued=1)
                async with sem:
                    await viz.update(queued=-1, active=1)
                    try:
                        success = await downloader._download_with_retry(img_info)
                        if success:
                            await viz.update(active=-1, finished=1)
                            await send.send(img_info)
                            logging.debug(f"Downloaded {img_info.filename}")
                        else:
                            await viz.update(active=-1, errors=1)
                            logging.error(f"Failed to download {img_info.filename}")
                    except Exception as e:
                        await viz.update(active=-1, errors=1)
                        logging.error(f"Download error for {img_info.filename}: {e}")

            async for img_info in recv:
                nursery.start_soon(download_one, img_info)


async def verifier_with_stats(recv, verified_send, missing_send, max_retries, viz, max_concurrent=None):
    """Verifier stage with metrics tracking, retry logic, and concurrent processing."""
    from rawnind.dataset import hash_sha1

    max_concurrent = max_concurrent or DEFAULT_MAX_WORKERS
    sem = trio.Semaphore(max_concurrent)

    async def verify_one(img_info):
        async with sem:
            await viz.update(verifying=1)

            if img_info.local_path and img_info.local_path.exists():
                computed = await trio.to_thread.run_sync(hash_sha1, img_info.local_path)

                if computed == img_info.sha1:
                    img_info.validated = True
                    await viz.update(verifying=-1, verified=1)
                    await verified_send.send(img_info)
                    logging.debug(f"✓ Verified {img_info.filename}")
                else:
                    await viz.update(verifying=-1, failed=1)
                    if img_info.retry_count < max_retries:
                        img_info.retry_count += 1
                        logging.warning(
                            f"Hash mismatch for {img_info.filename}: "
                            f"expected {img_info.sha1[:8]}, got {computed[:8]} "
                            f"(retry {img_info.retry_count}/{max_retries})"
                        )
                        img_info.local_path.unlink(missing_ok=True)
                        img_info.local_path = None
                        await missing_send.send(img_info)
                    else:
                        await viz.update(errors=1)
                        logging.error(
                            f"Hash mismatch for {img_info.filename} after {max_retries} retries, giving up"
                        )
            else:
                await viz.update(verifying=-1, failed=1)
                logging.warning(f"File not found: {img_info.local_path}")
                if img_info.retry_count >= max_retries:
                    await viz.update(errors=1)
                    logging.error(f"File {img_info.filename} missing after {max_retries} retries")

    async with recv, verified_send, trio.open_nursery() as nursery:
        async for img_info in recv:
            nursery.start_soon(verify_one, img_info)

    await missing_send.aclose()


async def indexer_with_stats(indexer, recv, send, viz):
    """Indexer stage with metrics tracking."""
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


async def enricher_with_stats(enricher, recv, send, viz, debug, max_concurrent=None):
    """Enricher stage with metrics tracking and concurrent processing."""
    max_concurrent = max_concurrent or DEFAULT_MAX_WORKERS
    await enricher._cache.load()

    sem = trio.Semaphore(max_concurrent)

    async def process_one(scene_info):
        async with sem:
            await viz.update(enriching=1)
            try:
                enriched = await enricher._enrich_scene(scene_info)
                await viz.update(enriching=-1, enriched=1)
                await send.send(enriched)
            except Exception as e:
                if debug:
                    print(f"ERROR during enrichment: {e}")
                await viz.update(enriching=-1, errors=1)
                await send.send(scene_info)

    try:
        async with recv, send, trio.open_nursery() as nursery:
            async for scene_info in recv:
                nursery.start_soon(process_one, scene_info)
    finally:
        await send.aclose()








async def test_dataloader_integration(dataset_root, viz):
    """Test that YAML → Dataset → DataLoader → Model works."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from rawnind.libs.rawds import CleanProfiledRGBNoisyBayerImageCropsDataset
    from rawnind.models.raw_denoiser import UtNet2
    import torch.nn as nn
    import torch.utils.data

    yaml_path = dataset_root / "pipeline_output.yaml"

    if not yaml_path.exists():
        print("\n⚠️  YAML file not found, skipping dataloader test")
        await viz.update(dataloader_skip=1)
        return

    print("\n" + "=" * 60)
    print("Testing DataLoader + Model Integration")
    print(f"YAML: {yaml_path}")
    print("=" * 60)

    try:
        # Create dataset from YAML
        await viz.update(dataloader_init=1)
        dataset = CleanProfiledRGBNoisyBayerImageCropsDataset(num_crops=2, crop_size=256, test_reserve=[],
                                                              bayer_only=True, content_fpaths=[str(yaml_path)],
                                                              test=False)

        print(f"✓ Dataset created: {len(dataset)} scenes")
        await viz.update(dataloader_images=len(dataset))

        # Create DataLoader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0  # Single-threaded for smoke test
        )

        print("✓ DataLoader created")
        await viz.update(dataloader_created=1)

        # Try to load one batch
        batch = next(iter(dataloader))
        await viz.update(dataloader_batches=1)

        print("✓ Loaded batch:")
        print(f"  - x_crops shape: {batch['x_crops'].shape}")
        print(f"  - y_crops shape: {batch['y_crops'].shape}")
        print(f"  - mask_crops shape: {batch['mask_crops'].shape}")
        print(f"  - rgb_xyz_matrix shape: {batch['rgb_xyz_matrix'].shape}")

        # Create model and run one training step
        print("\n✓ Creating model...")
        model = UtNet2(in_channels=4, funit=16)  # Small model for smoke test
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

        # Run one training step
        print("✓ Running one training step...")
        model.train()

        input_tensor = batch['y_crops']  # Noisy bayer crops
        target_tensor = batch['x_crops']  # Clean bayer crops

        # Forward pass
        output = model(input_tensor)

        # Handle size mismatch if model upsamples
        if target_tensor.shape[-2:] != output.shape[-2:]:
            target_tensor = nn.functional.interpolate(
                target_tensor,
                size=output.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        # Compute loss
        loss = loss_fn(output, target_tensor)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("✓ Training step completed:")
        print(f"  - Loss: {loss.item():.4f}")
        print(f"  - Output shape: {output.shape}")

        print("\n✅ DataLoader + Model integration successful!")
        await viz.update(dataloader_success=1)

    except Exception as e:
        print(f"\n❌ DataLoader/Model test failed: {e}")
        await viz.update(dataloader_errors=1)
        import traceback
        traceback.print_exc()


async def run_smoke_test(max_scenes=None, timeout_seconds=None, debug=False, download_concurrency=None):
    """
    Run the pipeline smoke test.

    Args:
        max_scenes:Maximum scenes allowed 'in flight' through the pipeline during processing.
        timeout_seconds: Maximum time to run in seconds
        debug: If True, print debug information
        download_concurrency: Maximum number of concurrent downloads
    """
    download_concurrency = download_concurrency or DEFAULT_DOWNLOAD_CONCURRENCY
    buffer_size = int(download_concurrency * CHANNEL_BUFFER_MULTIPLIER)
    dataset_root = Path("tmp/rawnind_dataset")

    # Memory profiler
    profiler = MemoryProfiler()
    profiler.snapshot("0_startup")

    viz = PipelineVisualizer(total_items=max_scenes)
    viz.clear()

    # Initialize components
    ingestor = DataIngestor(dataset_root=dataset_root)
    downloader = Downloader(max_concurrent=download_concurrency, progress=False)
    indexer = SceneIndexer(dataset_root)
    enricher = AsyncAligner(dataset_root=dataset_root, enable_crops_enrichment=False)

    # Initialize PostDownloadWorker stages with decorator-based progress tracking
    # Use unbuffered channels (0) for stages that load images to prevent OOM
    aligner = MetadataArtificer(
        output_dir=dataset_root / "alignment_artifacts",
        max_workers=DEFAULT_MAX_WORKERS,
        write_masks=True,
        write_metadata=True,
    ).attach_visualizer(viz)

    cropper = CropProducerStageAsync(
        output_dir=dataset_root / "crops",
        crop_size=256,
        num_crops=5,
        max_workers=DEFAULT_MAX_WORKERS
    ).attach_visualizer(viz)

    bridge = AsyncPipelineBridge(
        max_scenes=max_scenes,
        backwards_compat_mode=True,
    )


    #TODO Seperation of concerns: the following needs to be encapsulated and moved into PipelineBuilder
    async with aligner, cropper:
        async with trio.open_nursery() as nursery:
            if timeout_seconds is not None:
                nursery.cancel_scope.deadline = trio.current_time() + timeout_seconds

            # Create channels
            channels = create_channel_dict(
                ['scene', 'new_file', 'missing', 'downloaded', 'verified',
                 'complete_scene'],
                buffer_size
            )
            # Create unbuffered channels for image-heavy stages to prevent OOM
            enriched_send, enriched_recv = trio.open_memory_channel(0)
            aligned_send, aligned_recv = trio.open_memory_channel(0)
            cropped_send, cropped_recv = trio.open_memory_channel(0)
            channels.update({
                'enriched_send': enriched_send, 'enriched_recv': enriched_recv,
                'aligned_send': aligned_send, 'aligned_recv': aligned_recv,
                'cropped_send': cropped_send, 'cropped_recv': cropped_recv,
            })
            merged_send, merged_recv = trio.open_memory_channel(buffer_size)

            # Start pipeline stages
            nursery.start_soon(
                limit_producer,
                ingestor.produce_scenes, max_scenes,
                channels['scene_send'], buffer_size
            )
            nursery.start_soon(
                scanner_with_stats,
                dataset_root, channels['scene_recv'], channels['new_file_send'],
                channels['missing_send'], viz
            )
            nursery.start_soon(
                downloader_with_stats,
                downloader, channels['missing_recv'], channels['downloaded_send'], viz
            )
            nursery.start_soon(
                merge_channels,
                channels['new_file_recv'], channels['downloaded_recv'], merged_send
            )
            nursery.start_soon(
                verifier_with_stats,
                merged_recv, channels['verified_send'], channels['missing_send'], 2, viz, DEFAULT_MAX_WORKERS
            )
            nursery.start_soon(
                indexer_with_stats,
                indexer, channels['verified_recv'], channels['complete_scene_send'], viz
            )
            nursery.start_soon(
                enricher_with_stats,
                enricher, channels['complete_scene_recv'], channels['enriched_send'], viz, debug, DEFAULT_MAX_WORKERS
            )
            # Decorated workers handle metrics automatically
            nursery.start_soon(
                aligner.consume_and_produce,
                channels['enriched_recv'], channels['aligned_send']
            )

            # AsyncCropProducer uses consume_and_produce pattern (handles its own unloading)
            nursery.start_soon(
                cropper.consume_and_produce,
                channels['aligned_recv'], channels['cropped_send']
            )
            # Bridge consumes cropped scenes and exposes legacy interface in-memory
            nursery.start_soon(
                bridge.consume,
                channels['cropped_recv']
            )
            # Watch bridge progress and update visualizer
            # Also trigger dataloader test as soon as YAML is ready
            async def _watch_bridge():
                last = 0
                yaml_tested = False
                while True:
                    await trio.sleep(0.1)
                    n = len(bridge)
                    if n > last:
                        logging.info(f"Bridge progress: {n} scenes collected (was {last})")
                        await viz.update(complete=(n - last))
                        last = n
                    elif n == 0 and viz.counters.get('cropped', 0) > 0:
                        # Debug: cropping happened but bridge is empty
                        logging.warning(f"Bridge empty despite {viz.counters['cropped']} scenes cropped. Bridge state: {bridge.state.value}")

                    #TODO: This is not how this should work/ be triggered. I think the key issue is that we need to
                    # have  a `BatchAccumulator the the next stage should _pull_ from - no push/trigger but natural
                    # demand pressure. The training loop needs to be able to trigger for a single batch but I think
                    # we are basically there.

                    # Trigger dataloader test as soon as we have at least 2 scenes (batch_size=2)
                    if n >= 2 and not yaml_tested:
                        yaml_tested = True
                        yaml_path = dataset_root / "pipeline_output.yaml"
                        print(f"\nWriting {n} scene(s) to {yaml_path} for early dataloader test...")
                        bridge.write_yaml_compatible_cache(yaml_path, dataset_root)
                        print("✓ YAML written, testing dataloader...")
                        nursery.start_soon(test_dataloader_integration, dataset_root, viz)

                    if max_scenes and n >= max_scenes:
                        nursery.cancel_scope.cancel()
                        break
            nursery.start_soon(_watch_bridge)

    # Display summary
    print("\n\n" + "=" * 60)
    print("Pipeline completed!")
    print("=" * 60)

    # Write YAML cache for legacy dataloader compatibility
    if len(bridge) > 0:
        yaml_path = dataset_root / "pipeline_output.yaml"
        print(f"\nWriting {len(bridge)} scenes to {yaml_path}...")
        bridge.write_yaml_compatible_cache(yaml_path, dataset_root)
        print("✓ YAML written successfully")

    # Test dataloader integration
    await test_dataloader_integration(dataset_root, viz)

    # Memory profiling report
    profiler.snapshot("9_complete")
    profiler.report()

    total_errors = viz.counters['errors']
    failed_verifications = viz.counters['failed']

    if total_errors > 0 or failed_verifications > 0:
        print("\n⚠️  ISSUES DETECTED:")
        if failed_verifications > 0:
            print(f"  • {failed_verifications} files failed verification")
        if total_errors > 0:
            print(f"  • {total_errors} errors during processing")
    else:
        print("\n✅ No errors detected!")

    print()


if __name__ == "__main__":
    import tracemalloc
    tracemalloc.start()

    setup_logging()

    parser = argparse.ArgumentParser(
        description="Pipeline smoke test with live visualization (Async Crop Producer variant)"
    )
    parser.add_argument(
        "--max-scenes", type=int, default=None,
        help="Maximum scenes allowed 'in flight' through the pipeline during processing."
    )
    parser.add_argument(
        "--timeout", type=int, default=None,
        help="Maximum time to run in seconds"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print debug information"
    )
    parser.add_argument(
        "--download-concurrency", type=int, default=None,
        help=f"Maximum concurrent downloads (default: {DEFAULT_DOWNLOAD_CONCURRENCY})"
    )

    args = parser.parse_args()
    trio.run(run_smoke_test, None, args.timeout, args.debug, args.download_concurrency)