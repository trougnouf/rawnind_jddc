#!/usr/bin/env python3
"""
Memory profiling for CropProducer - minimal working version.

Wraps a simplified pipeline with memory snapshots.
"""

import argparse
import logging
import tracemalloc
from pathlib import Path
import psutil
import os

import trio

from rawnind.dataset import DataIngestor, MetadataArtificer
from rawnind.dataset.Aligner import MetadataArtificer
from rawnind.dataset.CropProducerStage import CropProducerStage

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MemoryProfiler:
    """Track memory with tracemalloc and psutil."""

    def __init__(self, top_n=15):
        self.top_n = top_n
        self.snapshots = []
        self.process = psutil.Process(os.getpid())

    def start(self):
        tracemalloc.start()
        logger.info("Memory profiling started")

    def snapshot(self, label: str):
        snap = tracemalloc.take_snapshot()
        rss_mb = self.process.memory_info().rss / (1024 * 1024)

        self.snapshots.append({"label": label, "snapshot": snap, "rss_mb": rss_mb})
        logger.info(f"[{label}] RSS: {rss_mb:.1f} MB")

    def compare(self, idx1: int, idx2: int):
        s1, s2 = self.snapshots[idx1], self.snapshots[idx2]
        logger.info(f"\n{'='*60}")
        logger.info(f"{s1['label']} → {s2['label']}")
        logger.info(
            f"RSS: {s1['rss_mb']:.1f} → {s2['rss_mb']:.1f} MB (Δ{s2['rss_mb']-s1['rss_mb']:+.1f})"
        )
        logger.info(f"{'='*60}")

        top = s2["snapshot"].compare_to(s1["snapshot"], "lineno")
        logger.info(f"\nTop {self.top_n} increases:")
        for stat in top[: self.top_n]:
            logger.info(f"  {stat}")

    def report(self):
        logger.info(f"\n{'='*60}")
        logger.info("Memory Profile Summary")
        logger.info(f"{'='*60}")

        for i, s in enumerate(self.snapshots):
            logger.info(f"{i}. [{s['label']}] {s['rss_mb']:.1f} MB")

        if len(self.snapshots) >= 2:
            peak = max(s["rss_mb"] for s in self.snapshots)
            growth = self.snapshots[-1]["rss_mb"] - self.snapshots[0]["rss_mb"]
            logger.info(f"\nPeak: {peak:.1f} MB | Total growth: {growth:+.1f} MB")


async def run_profiled_pipeline(max_scenes: int):
    """Run minimal pipeline with memory profiling."""
    profiler = MemoryProfiler()
    profiler.start()
    profiler.snapshot("0_startup")

    dataset_root = Path("datasets/RawNIND")
    output_dir = Path("tmp/rawnind_profile")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Channels
    ingest_send, ingest_recv = trio.open_memory_channel(0)
    indexed_send, indexed_recv = trio.open_memory_channel(0)
    enriched_send, enriched_recv = trio.open_memory_channel(0)
    artifact_send, artifact_recv = trio.open_memory_channel(0)
    cropped_send, cropped_recv = trio.open_memory_channel(0)

    profiler.snapshot("1_channels_created")

    async with trio.open_nursery() as nursery:
        # DataIngestor - limit to max_scenes
        async def produce_limited():
            # Create internal channel to limit output
            internal_send, internal_recv = trio.open_memory_channel(0)

            async def run_ingestor():
                ingestor = DataIngestor(dataset_root=dataset_root)
                await ingestor.produce_scenes(internal_send)

            async with trio.open_nursery() as limit_nursery:
                limit_nursery.start_soon(run_ingestor)

                count = 0
                async with internal_recv, ingest_send:
                    async for scene in internal_recv:
                        await ingest_send.send(scene)
                        count += 1
                        if count >= max_scenes:
                            limit_nursery.cancel_scope.cancel()
                            break

        nursery.start_soon(produce_limited)
        profiler.snapshot("2_ingestor_started")

        # SceneIndexer (only processes scenes with all files found)
        async def index_local_only():
            async with ingest_recv, indexed_send:
                async for scene in ingest_recv:
                    # Only index if all files exist locally
                    all_exist = all(
                        (
                            dataset_root
                            / scene.cfa_type
                            / scene.scene_name
                            / img.filename
                        ).exists()
                        or (
                            dataset_root
                            / scene.cfa_type
                            / scene.scene_name
                            / "gt"
                            / img.filename
                        ).exists()
                        for img in scene.all_images()
                    )

                    if all_exist:
                        # Set local paths
                        for img in scene.all_images():
                            candidates = [
                                dataset_root
                                / scene.cfa_type
                                / scene.scene_name
                                / "gt"
                                / img.filename,
                                dataset_root
                                / scene.cfa_type
                                / scene.scene_name
                                / img.filename,
                            ]
                            for path in candidates:
                                if path.exists():
                                    img.local_path = path
                                    img.validated = True
                                    break

                        await indexed_send.send(scene)

        nursery.start_soon(index_local_only)
        profiler.snapshot("3_indexer_started")

        # MetadataArtificer
        async def align_scenes():
            async with indexed_recv, enriched_send:
                aligner = MetadataArtificer()
                async for scene in indexed_recv:
                    enriched = await aligner.process_scene(scene)
                    await enriched_send.send(enriched)

        nursery.start_soon(align_scenes)
        profiler.snapshot("4_aligner_started")

        # MetadataArtificer
        async def artifact_scenes():
            async with enriched_recv, artifact_send:
                artificer = MetadataArtificer(output_dir=output_dir / "metadata")
                async for scene in enriched_recv:
                    processed = await artificer.process_scene(scene)
                    await artifact_send.send(processed)

        nursery.start_soon(artifact_scenes)
        profiler.snapshot("5_artificer_started")

        # CropProducer with memory snapshots
        async def crop_with_profiling():
            count = 0
            async with artifact_recv, cropped_send:
                cropper = CropProducerStage(
                    output_dir=output_dir / "crops",
                    crop_size=256,
                    num_crops=5,
                    max_workers=2,
                    use_systematic_tiling=False,
                )

                async for scene in artifact_recv:
                    profiler.snapshot(f"6a_before_crop_{count}_{scene.scene_name[:15]}")

                    processed = await cropper.process_scene(scene)

                    profiler.snapshot(f"6b_after_crop_{count}_{scene.scene_name[:15]}")

                    await cropped_send.send(processed)
                    count += 1

                    # Compare memory delta
                    if len(profiler.snapshots) >= 2:
                        profiler.compare(-2, -1)

        nursery.start_soon(crop_with_profiling)
        profiler.snapshot("6_cropper_started")

        # Bridge - collect results
        async def collect():
            scenes = []
            async with cropped_recv:
                async for scene in cropped_recv:
                    scenes.append(scene)
            return scenes

        result = await collect()

    profiler.snapshot("7_complete")
    profiler.report()

    logger.info(f"\n{len(result)} scenes processed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile CropProducer memory")
    parser.add_argument("--max-scenes", type=int, default=3)
    args = parser.parse_args()

    trio.run(run_profiled_pipeline, args.max_scenes)
