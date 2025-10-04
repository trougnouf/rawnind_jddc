# smoke_test.py
"""Smoke test for the new pipeline architecture."""
import logging
from pathlib import Path

import trio

from rawnind.dataset import DataIngestor, FileScanner, Downloader, Verifier, SceneIndexer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def limited_smoke_test():
    """
    Smoke test that processes only the first 5 images from the dataset.

    Uses a modified ingestor that limits output to test pipeline stages
    without processing the entire dataset.
    """
    dataset_root = Path("tmp/rawnind_dataset")

    print("\n" + "=" * 70)
    print("SMOKE TEST: Processing first 5 images")
    print("=" * 70 + "\n")

    # Track statistics
    stats = {
        'scenes_ingested': 0,
        'images_scanned': 0,
        'files_found': 0,
        'files_missing': 0,
        'downloads_attempted': 0,
        'downloads_successful': 0,
        'verified': 0,
        'verification_failed': 0,
        'scenes_completed': 0
    }

    async with trio.open_nursery() as nursery:
        # Create channels
        (scene_send, scene_recv) = trio.open_memory_channel(10)
        (new_file_send, new_file_recv) = trio.open_memory_channel(10)
        (missing_send, missing_recv) = trio.open_memory_channel(10)
        (downloaded_send, downloaded_recv) = trio.open_memory_channel(10)
        (verified_send, verified_recv) = trio.open_memory_channel(10)
        (complete_scene_send, complete_scene_recv) = trio.open_memory_channel(10)

        # Initialize components
        ingestor = DataIngestor(dataset_root=dataset_root)
        scanner = FileScanner(dataset_root)
        downloader = Downloader(max_concurrent=2, progress=True)
        verifier = Verifier(max_retries=2)
        indexer = SceneIndexer(dataset_root)

        # Limited ingestor that stops after 5 images
        async def limited_produce_scenes(send_channel):
            """Produce scenes but stop after 5 total images."""
            async with send_channel:
                image_count = 0
                async for scene_info in _limited_scene_producer(ingestor, max_images=5):
                    stats['scenes_ingested'] += 1
                    image_count += len(scene_info.all_images())
                    logger.info(f"Ingested scene: {scene_info.scene_name} ({len(scene_info.all_images())} images)")
                    await send_channel.send(scene_info)
                    if image_count >= 5:
                        break

        # Scanner with stats
        async def scanner_with_stats(recv, new_file, missing):
            async with recv:
                async for scene_info in recv:
                    for img in scene_info.all_images():
                        stats['images_scanned'] += 1
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
                                stats['files_found'] += 1
                                await new_file.send(img_info)
                                logger.info(f"  Found: {img_info.filename}")
                                found = True
                                break

                        if not found:
                            img_info.local_path = candidates[0] if candidates else None
                            stats['files_missing'] += 1
                            await missing.send(img_info)
                            logger.info(f"  Missing: {img_info.filename}")

        # Downloader with stats
        async def downloader_with_stats(recv, send):
            async with recv:
                async for img_info in recv:
                    stats['downloads_attempted'] += 1
                    logger.info(f"  Downloading: {img_info.filename}")
                    # Forward to actual downloader (it handles the download)
                    await send.send(img_info)

        # Merge channels
        merged_send, merged_recv = trio.open_memory_channel(10)

        async def merge_inputs():
            async with new_file_recv, downloaded_recv, merged_send:
                async with trio.open_nursery() as merge_nursery:
                    async def forward(recv):
                        async with recv:
                            async for item in recv:
                                await merged_send.send(item)

                    merge_nursery.start_soon(forward, new_file_recv)
                    merge_nursery.start_soon(forward, downloaded_recv)

        # Verifier with stats
        async def verifier_with_stats(recv, verified, missing):
            async with recv:
                async for img_info in recv:
                    if img_info.local_path and img_info.local_path.exists():
                        # Compute hash
                        from rawnind.dataset import hash_sha1
                        computed = await trio.to_thread.run_sync(hash_sha1, img_info.local_path)

                        if computed == img_info.sha1:
                            img_info.validated = True
                            stats['verified'] += 1
                            logger.info(f"  Verified: {img_info.filename}")
                            await verified.send(img_info)
                        else:
                            stats['verification_failed'] += 1
                            logger.warning(f"  Verification failed: {img_info.filename}")
                            if img_info.retry_count < 2:
                                img_info.retry_count += 1
                                img_info.local_path.unlink()
                                img_info.local_path = None
                                await missing.send(img_info)
                    else:
                        stats['verification_failed'] += 1

        # Indexer with stats
        async def indexer_with_stats(recv, send):
            async with recv, send:
                async for img_info in recv:
                    indexer._add_image_to_index(img_info)

                    scene_key = (img_info.cfa_type, img_info.scene_name)
                    if scene_key not in indexer._scene_completion_tracker:
                        if indexer._is_scene_complete(img_info):
                            scene_info = indexer._construct_scene(img_info)
                            indexer._scene_completion_tracker.add(scene_key)
                            indexer._move_scene_to_complete(scene_info)
                            stats['scenes_completed'] += 1
                            logger.info(f"Scene complete: {scene_info.scene_name}")
                            await send.send(scene_info)

        # Final consumer
        async def final_consumer(recv):
            async with recv:
                async for scene_info in recv:
                    logger.info(f"Final: Scene {scene_info.scene_name} ready")

        # Start pipeline stages
        nursery.start_soon(limited_produce_scenes, scene_send)
        nursery.start_soon(scanner_with_stats, scene_recv, new_file_send, missing_send)
        nursery.start_soon(downloader_with_stats, missing_recv, downloaded_send)
        nursery.start_soon(merge_inputs)
        nursery.start_soon(verifier_with_stats, merged_recv, verified_send, missing_send)
        nursery.start_soon(indexer_with_stats, verified_recv, complete_scene_send)
        nursery.start_soon(final_consumer, complete_scene_recv)

    # Print statistics
    print("\n" + "=" * 70)
    print("SMOKE TEST RESULTS")
    print("=" * 70)
    print(f"Scenes ingested:         {stats['scenes_ingested']}")
    print(f"Images scanned:          {stats['images_scanned']}")
    print(f"Files found locally:     {stats['files_found']}")
    print(f"Files missing:           {stats['files_missing']}")
    print(f"Download attempts:       {stats['downloads_attempted']}")
    print(f"Files verified:          {stats['verified']}")
    print(f"Verification failures:   {stats['verification_failed']}")
    print(f"Complete scenes:         {stats['scenes_completed']}")
    print("=" * 70 + "\n")


async def _limited_scene_producer(ingestor, max_images):
    """Helper generator that limits total images produced."""
    image_count = 0

    # Create internal channel
    send_channel, recv_channel = trio.open_memory_channel(10)

    async def produce():
        await ingestor.produce_scenes(send_channel)

    async with trio.open_nursery() as nursery:
        nursery.start_soon(produce)

        async with recv_channel:
            async for scene_info in recv_channel:
                yield scene_info
                image_count += len(scene_info.all_images())
                if image_count >= max_images:
                    nursery.cancel_scope.cancel()
                    break


if __name__ == "__main__":
    trio.run(limited_smoke_test)
