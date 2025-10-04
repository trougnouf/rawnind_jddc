import trio
from pathlib import Path
from typing import Optional
import logging

from .DataIngestor import DataIngestor
from .SceneIndexer import SceneIndexer
from .FileScanner import FileScanner
from .Downloader import Downloader
from .Verifier import Verifier
from .MetadataEnricher import MetadataEnricher

logger = logging.getLogger(__name__)


class PipelineBuilder:
    """Orchestrates the complete dataset pipeline with all stages."""

    def __init__(
        self,
        dataset_root: Path,
        cache_paths: Optional[tuple[Path, Path]] = None,
        dataset_metadata_url: Optional[str] = None,
        max_concurrent_downloads: int = 5,
        max_concurrent_enrichment: int = 4,
        enable_enrichment: bool = True,
        enable_crops_enrichment: bool = True
    ):
        """
        Initialize pipeline builder.

        Args:
            dataset_root: Root directory for dataset files
            cache_paths: Tuple of (yaml_cache, metadata_cache) paths
            dataset_metadata_url: URL for dataset metadata API
            max_concurrent_downloads: Max concurrent downloads
            max_concurrent_enrichment: Max concurrent enrichment tasks
            enable_enrichment: Whether to run enrichment stage
            enable_crops_enrichment: Whether to enrich with crops list
        """
        self.dataset_root = dataset_root
        self.enable_enrichment = enable_enrichment

        # Initialize pipeline components
        self.ingestor = DataIngestor(cache_paths, dataset_root, dataset_metadata_url)
        self.scanner = FileScanner(dataset_root)
        self.downloader = Downloader(max_concurrent=max_concurrent_downloads)
        self.verifier = Verifier(max_retries=3)
        self.indexer = SceneIndexer(dataset_root)
        self.enricher = MetadataEnricher(
            dataset_root=dataset_root,
            max_concurrent=max_concurrent_enrichment,
            enable_crops_enrichment=enable_crops_enrichment
        )

    async def run(self):
        """Run the complete pipeline."""
        if self.enable_enrichment:
            await self._run_with_enrichment()
        else:
            await self._run_without_enrichment()

    async def _run_with_enrichment(self):
        """Run pipeline with metadata enrichment stage."""
        async with trio.open_nursery() as nursery:
            # Create channels for inter-stage communication
            (scene_send, scene_recv) = trio.open_memory_channel(100)
            (new_file_send, new_file_recv) = trio.open_memory_channel(100)
            (missing_send, missing_recv) = trio.open_memory_channel(100)
            (downloaded_send, downloaded_recv) = trio.open_memory_channel(100)
            (verified_send, verified_recv) = trio.open_memory_channel(100)
            (complete_scene_send, complete_scene_recv) = trio.open_memory_channel(100)
            (enriched_send, enriched_recv) = trio.open_memory_channel(100)

            # Merge new_file and downloaded channels for verifier
            merged_send, merged_recv = trio.open_memory_channel(100)

            async def merge_inputs():
                async with new_file_recv, downloaded_recv, merged_send:
                    async with trio.open_nursery() as merge_nursery:
                        async def forward(recv):
                            async with recv:
                                async for item in recv:
                                    await merged_send.send(item)

                        merge_nursery.start_soon(forward, new_file_recv)
                        merge_nursery.start_soon(forward, downloaded_recv)

            # Start pipeline stages
            nursery.start_soon(self.ingestor.produce_scenes, scene_send)
            nursery.start_soon(self.scanner.consume_new_items, scene_recv, new_file_send, missing_send)
            nursery.start_soon(self.downloader.consume_missing, missing_recv, downloaded_send)
            nursery.start_soon(merge_inputs)
            nursery.start_soon(
                self.verifier.consume_new_files,
                merged_recv,
                verified_send,
                missing_send
            )
            nursery.start_soon(
                self.indexer.consume_images_produce_scenes,
                verified_recv,
                complete_scene_send
            )
            nursery.start_soon(
                self.enricher.consume_scenes_produce_enriched,
                complete_scene_recv,
                enriched_send
            )
            nursery.start_soon(self._final_consumer, enriched_recv)

    async def _run_without_enrichment(self):
        """Run pipeline without enrichment stage."""
        async with trio.open_nursery() as nursery:
            # Create channels
            (scene_send, scene_recv) = trio.open_memory_channel(100)
            (new_file_send, new_file_recv) = trio.open_memory_channel(100)
            (missing_send, missing_recv) = trio.open_memory_channel(100)
            (downloaded_send, downloaded_recv) = trio.open_memory_channel(100)
            (verified_send, verified_recv) = trio.open_memory_channel(100)
            (complete_scene_send, complete_scene_recv) = trio.open_memory_channel(100)

            # Merge new_file and downloaded channels for verifier
            merged_send, merged_recv = trio.open_memory_channel(100)

            async def merge_inputs():
                async with new_file_recv, downloaded_recv, merged_send:
                    async with trio.open_nursery() as merge_nursery:
                        async def forward(recv):
                            async with recv:
                                async for item in recv:
                                    await merged_send.send(item)

                        merge_nursery.start_soon(forward, new_file_recv)
                        merge_nursery.start_soon(forward, downloaded_recv)

            # Start pipeline stages
            nursery.start_soon(self.ingestor.produce_scenes, scene_send)
            nursery.start_soon(self.scanner.consume_new_items, scene_recv, new_file_send, missing_send)
            nursery.start_soon(self.downloader.consume_missing, missing_recv, downloaded_send)
            nursery.start_soon(merge_inputs)
            nursery.start_soon(
                self.verifier.consume_new_files,
                merged_recv,
                verified_send,
                missing_send
            )
            nursery.start_soon(
                self.indexer.consume_images_produce_scenes,
                verified_recv,
                complete_scene_send
            )
            nursery.start_soon(self._final_consumer, complete_scene_recv)

    async def _final_consumer(self, recv_channel: trio.MemoryReceiveChannel):
        """Final consumer for complete/enriched scenes."""
        async with recv_channel:
            async for scene_info in recv_channel:
                logger.info(f"Pipeline completed scene: {scene_info.scene_name}")
                # Add custom logic here (e.g., database storage, exports, etc.)
