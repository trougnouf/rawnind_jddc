#!/usr/bin/env python3
"""Smoke test for the pipeline architecture with live visualization."""

import argparse
import logging
import os
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
from rawnind.dataset.Aligner import Aligner
from rawnind.dataset.crop_producer_stage import CropProducerStage
from rawnind.dataset.YAMLArtifactWriter import YAMLArtifactWriter
from rawnind.dataset.visualizer import PipelineVisualizer
from rawnind.dataset.channel_utils import (
    create_channel_dict,
    merge_channels,
    limit_producer,
    consume_until,
)

# Constants
CHANNEL_BUFFER_MULTIPLIER = 2.5
DEFAULT_DOWNLOAD_CONCURRENCY = max(1, int(os.cpu_count() * 0.75))
DEFAULT_MAX_WORKERS = max(1, int(os.cpu_count() * 0.75))
LOG_FILE = Path('/tmp/smoke_test.log')


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
    logging.getLogger('rawnind.dataset.MetadataEnricher').setLevel(logging.INFO)

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
                else:
                    await viz.update(verifying=-1, failed=1)
                    if img_info.retry_count < max_retries:
                        img_info.retry_count += 1
                        img_info.local_path.unlink(missing_ok=True)
                        img_info.local_path = None
                        await missing_send.send(img_info)
                    else:
                        await viz.update(errors=1)
            else:
                await viz.update(verifying=-1, failed=1)
                if img_info.retry_count >= max_retries:
                    await viz.update(errors=1)

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








async def run_smoke_test(max_scenes=None, timeout_seconds=None, debug=False, download_concurrency=None):
    """
    Run the pipeline smoke test.

    Args:
        max_scenes: Maximum number of complete scenes to process
        timeout_seconds: Maximum time to run in seconds
        debug: If True, print debug information
        download_concurrency: Maximum number of concurrent downloads
    """
    download_concurrency = download_concurrency or DEFAULT_DOWNLOAD_CONCURRENCY
    buffer_size = int(download_concurrency * CHANNEL_BUFFER_MULTIPLIER)
    dataset_root = Path("tmp/rawnind_dataset")

    viz = PipelineVisualizer(total_items=max_scenes)
    viz.clear()

    # Initialize components
    ingestor = DataIngestor(dataset_root=dataset_root)
    downloader = Downloader(max_concurrent=download_concurrency, progress=False)
    indexer = SceneIndexer(dataset_root)
    enricher = MetadataEnricher(dataset_root=dataset_root, enable_crops_enrichment=False)

    # Initialize PostDownloadWorker stages with decorator-based progress tracking
    aligner = Aligner(
        output_dir=dataset_root / "alignment_artifacts",
        max_workers=DEFAULT_MAX_WORKERS
    ).attach_visualizer(viz)

    cropper = CropProducerStage(
        output_dir=dataset_root / "crops",
        crop_size=256,
        num_crops=5,
        max_workers=DEFAULT_MAX_WORKERS
    ).attach_visualizer(viz)

    yaml_writer = YAMLArtifactWriter(
        output_dir=dataset_root,
        output_filename="pipeline_output.yaml"
    ).attach_visualizer(viz)

    async with trio.open_nursery() as nursery:
        if timeout_seconds is not None:
            nursery.cancel_scope.deadline = trio.current_time() + timeout_seconds

        # Create channels
        channels = create_channel_dict(
            ['scene', 'new_file', 'missing', 'downloaded', 'verified',
             'complete_scene', 'enriched', 'aligned', 'cropped', 'yaml'],
            buffer_size
        )
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
            downloader.consume_missing,
            channels['missing_recv'], channels['downloaded_send']
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
        nursery.start_soon(
            cropper.consume_and_produce,
            channels['aligned_recv'], channels['cropped_send']
        )
        nursery.start_soon(
            yaml_writer.consume_and_produce,
            channels['cropped_recv'], channels['yaml_send']
        )
        nursery.start_soon(
            consume_until,
            channels['yaml_recv'], max_scenes, nursery,
            lambda scene: viz.update(complete=1)
        )

    # Display summary
    print("\n\n" + "=" * 60)
    print("Pipeline completed!")
    print("=" * 60)

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
        description="Pipeline smoke test with live visualization"
    )
    parser.add_argument(
        "--timeout", type=float, default=None,
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