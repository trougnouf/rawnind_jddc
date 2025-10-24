import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import trio

from .Aligner import MetadataArtificer
from .DataIngestor import DataIngestor
from .Downloader import Downloader
from .FileScanner import FileScanner
from .MetadataArtificer import MetadataArtificer
from .PostDownloadWorker import PostDownloadWorker
from .SceneIndexer import SceneIndexer
from .Verifier import Verifier
from .CropProducerStage import CropProducerStage

logger = logging.getLogger(__name__)


class PipelineBuilder:
    """Pipeline builder that orchestrates dataset ingestion, downloading, verification,
    indexing, optional enrichment, and optional post‑processing.

    The `PipelineBuilder` class creates a set of components that together
    form an end‑to‑end pipeline for preparing a dataset.  The pipeline can
    operate in several modes:

    1. **With enrichment** – after downloading and verifying images,
       metadata is enriched by aligning scenes and optionally adding
       crop information.
    2. **Without enrichment** – the pipeline simply downloads and verifies
       images and then indexes them.
    3. **With post‑processing** – additional stages such as crop
       generation or metadata artifact creation are executed after the
       core pipeline completes.

    The builder exposes the core components as attributes so they can be
    inspected or replaced in tests.  All components are started and run
    concurrently inside an asynchronous Trio nursery.

    Attributes:
        dataset_root (Path): Root directory for dataset files.
        enable_enrichment (bool): Whether to run the enrichment stage.
        postprocessor_config (Dict[str, Any]): Configuration for post‑processors.
        postprocessors (List[PostDownloadWorker]): List of post‑processing stages.
        ingestor (DataIngestor): Component that produces scenes from dataset
            metadata.
        scanner (FileScanner): Component that watches for new files in the
            dataset root.
        downloader (Downloader): Component that downloads missing files.
        verifier (Verifier): Component that verifies downloaded files.
        indexer (SceneIndexer): Component that indexes verified images.
        enricher (MetadataArtificer): Component that enriches scenes with
            alignment and crop data."""

    def __init__(
        self,
        dataset_root: Path,
        cache_paths: Optional[tuple[Path, Path]] = None,
        dataset_metadata_url: Optional[str] = None,
        max_concurrent_downloads: int = 5,
        max_concurrent_enrichment: int = 4,
        enable_enrichment: bool = True,
        enable_crops_enrichment: bool = True,
        postprocessor_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize pipeline.

        Args:
            dataset_root: Root directory for dataset files
            cache_paths: Tuple of (yaml_cache, metadata_cache) paths
            dataset_metadata_url: URL for dataset metadata API
            max_concurrent_downloads: Max concurrent downloads
            max_concurrent_enrichment: Max concurrent enrichment tasks
            enable_enrichment: Whether to run enrichment stage
            enable_crops_enrichment: Whether to enrich with crops list
            postprocessor_config: Configuration for post-processors
        """
        self.dataset_root = dataset_root
        self.enable_enrichment = enable_enrichment
        self.postprocessor_config = postprocessor_config or {}
        self.postprocessors: List[PostDownloadWorker] = []

        self.ingestor = DataIngestor(cache_paths, dataset_root, dataset_metadata_url)
        self.scanner = FileScanner(dataset_root)
        self.downloader = Downloader(max_concurrent=max_concurrent_downloads)
        self.verifier = Verifier(max_retries=3)
        self.indexer = SceneIndexer(dataset_root)
        self.enricher = MetadataArtificer(
            dataset_root=dataset_root,
            max_concurrent=max_concurrent_enrichment,
            enable_crops_enrichment=enable_crops_enrichment,
        )

        self._init_postprocessors()

    def _init_postprocessors(self):
        """Initialize post-processing stages."""
        artifacts_dir = self.dataset_root / "artifacts"

        if self.postprocessor_config.get("enable_crops", True):
            crop_config = self.postprocessor_config.get("crops", {})
            self.postprocessors.append(
                CropProducerStage(
                    output_dir=artifacts_dir / "crops",
                    crop_size=crop_config.get("size", 512),
                    num_crops=crop_config.get("num_crops", 10),
                    max_workers=crop_config.get("max_workers", 4),
                    save_format=crop_config.get("format", "npy"),
                )
            )

        if self.postprocessor_config.get("enable_alignment_artifacts", True):
            self.postprocessors.append(
                MetadataArtificer(
                    output_dir=artifacts_dir / "alignment",
                    write_masks=True,
                    write_metadata=True,
                    max_workers=4,
                )
            )

    def add_postprocessor(self, worker: PostDownloadWorker):
        """Add a post-processing stage."""
        self.postprocessors.append(worker)
        logger.info(f"Added post-processor: {worker.name}")

    async def run(self):
        """Run the pipeline."""
        if self.postprocessors:
            await self._run_with_postprocessing()
        elif self.enable_enrichment:
            await self._run_with_enrichment()
        else:
            await self._run_without_enrichment()

    async def _merge_channels(self, recv1, recv2, send):
        """Merge two receive channels into one send channel."""
        async with recv1, recv2, send:
            async with trio.open_nursery() as nursery:

                async def forward(recv):
                    async with recv:
                        async for item in recv:
                            await send.send(item)

                nursery.start_soon(forward, recv1)
                nursery.start_soon(forward, recv2)

    async def _run_with_enrichment(self):
        """Run pipeline with metadata enrichment stage."""
        async with trio.open_nursery() as nursery:
            (scene_send, scene_recv) = trio.open_memory_channel(100)
            (new_file_send, new_file_recv) = trio.open_memory_channel(100)
            (missing_send, missing_recv) = trio.open_memory_channel(100)
            (downloaded_send, downloaded_recv) = trio.open_memory_channel(100)
            (verified_send, verified_recv) = trio.open_memory_channel(100)
            (complete_scene_send, complete_scene_recv) = trio.open_memory_channel(100)
            (enriched_send, enriched_recv) = trio.open_memory_channel(100)

            merged_send, merged_recv = trio.open_memory_channel(100)

            nursery.start_soon(
                self._merge_channels, new_file_recv, downloaded_recv, merged_send
            )
            nursery.start_soon(self.ingestor.produce_scenes, scene_send)
            nursery.start_soon(
                self.scanner.consume_new_items, scene_recv, new_file_send, missing_send
            )
            nursery.start_soon(
                self.downloader.consume_missing, missing_recv, downloaded_send
            )
            nursery.start_soon(
                self.verifier.consume_new_files,
                merged_recv,
                verified_send,
                missing_send,
            )
            nursery.start_soon(
                self.indexer.consume_images_produce_scenes,
                verified_recv,
                complete_scene_send,
            )
            nursery.start_soon(
                self.enricher.consume_scenes_produce_enriched,
                complete_scene_recv,
                enriched_send,
            )
            nursery.start_soon(self._final_consumer, enriched_recv)

    async def _run_without_enrichment(self):
        """Run pipeline without enrichment stage."""
        async with trio.open_nursery() as nursery:
            (scene_send, scene_recv) = trio.open_memory_channel(100)
            (new_file_send, new_file_recv) = trio.open_memory_channel(100)
            (missing_send, missing_recv) = trio.open_memory_channel(100)
            (downloaded_send, downloaded_recv) = trio.open_memory_channel(100)
            (verified_send, verified_recv) = trio.open_memory_channel(100)
            (complete_scene_send, complete_scene_recv) = trio.open_memory_channel(100)

            merged_send, merged_recv = trio.open_memory_channel(100)

            nursery.start_soon(
                self._merge_channels, new_file_recv, downloaded_recv, merged_send
            )
            nursery.start_soon(self.ingestor.produce_scenes, scene_send)
            nursery.start_soon(
                self.scanner.consume_new_items, scene_recv, new_file_send, missing_send
            )
            nursery.start_soon(
                self.downloader.consume_missing, missing_recv, downloaded_send
            )
            nursery.start_soon(
                self.verifier.consume_new_files,
                merged_recv,
                verified_send,
                missing_send,
            )
            nursery.start_soon(
                self.indexer.consume_images_produce_scenes,
                verified_recv,
                complete_scene_send,
            )
            nursery.start_soon(self._final_consumer, complete_scene_recv)

    async def _final_consumer(self, recv_channel: trio.MemoryReceiveChannel):
        """Final consumer for complete/enriched scenes."""
        async with recv_channel:
            async for scene_info in recv_channel:
                logger.info(f"Pipeline completed scene: {scene_info.scene_name}")

    async def _run_with_postprocessing(self):
        """Run pipeline with post-processing stages."""
        async with trio.open_nursery() as nursery:
            (scene_send, scene_recv) = trio.open_memory_channel(100)
            (new_file_send, new_file_recv) = trio.open_memory_channel(100)
            (missing_send, missing_recv) = trio.open_memory_channel(100)
            (downloaded_send, downloaded_recv) = trio.open_memory_channel(100)
            (verified_send, verified_recv) = trio.open_memory_channel(100)
            (complete_scene_send, complete_scene_recv) = trio.open_memory_channel(100)

            postprocessor_channels = []
            for i in range(len(self.postprocessors) + 1):
                send, recv = trio.open_memory_channel(100)
                postprocessor_channels.append((send, recv))

            merged_send, merged_recv = trio.open_memory_channel(100)

            nursery.start_soon(
                self._merge_channels, new_file_recv, downloaded_recv, merged_send
            )
            nursery.start_soon(self.ingestor.produce_scenes, scene_send)
            nursery.start_soon(
                self.scanner.consume_new_items, scene_recv, new_file_send, missing_send
            )
            nursery.start_soon(
                self.downloader.consume_missing, missing_recv, downloaded_send
            )
            nursery.start_soon(
                self.verifier.consume_new_files,
                merged_recv,
                verified_send,
                missing_send,
            )
            nursery.start_soon(
                self.indexer.consume_images_produce_scenes,
                verified_recv,
                complete_scene_send,
            )

            if self.enable_enrichment:
                nursery.start_soon(
                    self.enricher.consume_scenes_produce_enriched,
                    complete_scene_recv,
                    postprocessor_channels[0][0],
                )
            else:
                nursery.start_soon(
                    self._forward_channel,
                    complete_scene_recv,
                    postprocessor_channels[0][0],
                )

            for i, worker in enumerate(self.postprocessors):
                input_recv = postprocessor_channels[i][1]
                output_send = (
                    postprocessor_channels[i + 1][0]
                    if i < len(self.postprocessors) - 1
                    else None
                )

                nursery.start_soon(
                    self._run_postprocessor, worker, input_recv, output_send
                )

            final_recv = postprocessor_channels[-1][1]
            nursery.start_soon(self._final_consumer, final_recv)

    async def _run_postprocessor(
        self,
        worker: PostDownloadWorker,
        input_channel: trio.MemoryReceiveChannel,
        output_channel: Optional[trio.MemorySendChannel],
    ):
        """Run a post-processor."""
        async with worker:
            await worker.consume_and_produce(input_channel, output_channel)

    async def _forward_channel(
        self,
        input_channel: trio.MemoryReceiveChannel,
        output_channel: trio.MemorySendChannel,
    ):
        """Forward items from one channel to another."""
        async with input_channel, output_channel:
            async for item in input_channel:
                await output_channel.send(item)
