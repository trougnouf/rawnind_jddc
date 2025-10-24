"""
Meilisearch Indexer for Pipeline Scene Metadata.

Indexes enriched SceneInfo objects to Meilisearch for fast querying during
development, debugging, and quality analysis.

Optional stage - can be disabled in production without affecting pipeline.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import time

import httpx
import trio

from .PostDownloadWorker import PostDownloadWorker
from .SceneInfo import SceneInfo

logger = logging.getLogger(__name__)


def scene_to_meilisearch_document(scene: SceneInfo) -> Dict[str, Any]:
    """
    Convert SceneInfo to Meilisearch document format.

    Args:
        scene: Enriched SceneInfo object

    Returns:
        Dictionary suitable for Meilisearch indexing
    """
    gt_img = scene.get_gt_image()
    noisy_images = scene.noisy_images

    # Aggregate metadata from noisy images
    avg_alignment_loss = 0.0
    avg_mask_mean = 0.0
    total_crops = 0
    has_metadata = False

    if noisy_images:
        alignment_losses = []
        mask_means = []

        for noisy_img in noisy_images:
            if noisy_img.metadata:
                has_metadata = True
                alignment_losses.append(noisy_img.metadata.get("alignment_loss", 0.0))
                mask_means.append(noisy_img.metadata.get("mask_mean", 0.0))
                crops = noisy_img.metadata.get("crops", [])
                total_crops += len(crops) if crops else 0

        if alignment_losses:
            avg_alignment_loss = sum(alignment_losses) / len(alignment_losses)
        if mask_means:
            avg_mask_mean = sum(mask_means) / len(mask_means)

    # Build searchable document
    document = {
        # Primary key
        "id": scene.scene_name,

        # Scene identification
        "scene_name": scene.scene_name,
        "cfa_type": scene.cfa_type,
        "unknown_sensor": scene.unknown_sensor,
        "test_reserve": scene.test_reserve,

        # Image counts
        "clean_image_count": len(scene.clean_images),
        "noisy_image_count": len(noisy_images),

        # Quality metrics (aggregated)
        "avg_alignment_loss": avg_alignment_loss,
        "avg_mask_mean": avg_mask_mean,
        "total_crops": total_crops,
        "has_metadata": has_metadata,

        # File info
        "gt_filename": gt_img.filename if gt_img else None,
        "noisy_filenames": [img.filename for img in noisy_images],

        # Timestamps
        "indexed_at": time.time(),
    }

    return document


class MeilisearchIndexer(PostDownloadWorker):
    """
    Pipeline stage that indexes enriched scenes to Meilisearch.

    Enables fast querying and filtering of scene metadata for:
    - Development/debugging
    - Quality analysis
    - Dataset exploration
    - Smoke test validation
    """

    def __init__(
        self,
        output_dir: Path,
        meilisearch_url: Optional[str] = None,
        index_name: str = "rawnind_scenes",
        api_key: Optional[str] = None,
        batch_size: int = 100,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Meilisearch indexer.

        Args:
            output_dir: Base directory (for PostDownloadWorker compatibility)
            meilisearch_url: Meilisearch server URL (defaults to MEILISEARCH_URL env var,
                           or http://localhost:7700 for local development)
            index_name: Name of index to create/update
            api_key: Optional API key for authentication (can use MEILISEARCH_API_KEY env var)
            batch_size: Number of documents to buffer before sending
            config: Additional configuration

        Environment Variables:
            MEILISEARCH_URL: Meilisearch server URL
            MEILISEARCH_API_KEY: API key for authentication
        """
        super().__init__(
            output_dir=output_dir,
            max_workers=1,
            use_process_pool=False,
            config=config
        )

        # Read URL from environment if not provided
        if meilisearch_url is None:
            meilisearch_url = os.getenv("MEILISEARCH_URL", "http://localhost:7700")
        
        # Read API key from environment if not provided
        if api_key is None:
            api_key = os.getenv("MEILISEARCH_API_KEY")

        self.meilisearch_url = meilisearch_url.rstrip("/")
        self.index_name = index_name
        self.api_key = api_key
        self.batch_size = batch_size

        self.document_buffer: List[Dict[str, Any]] = []
        self.total_indexed = 0
        self.client: Optional[httpx.AsyncClient] = None

        logger.info(
            f"MeilisearchIndexer initialized: url={self.meilisearch_url}, "
            f"index={self.index_name}"
        )

    async def startup(self):
        """Initialize HTTP client and ensure index exists."""
        await super().startup()

        # Create async HTTP client
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self.client = httpx.AsyncClient(
            base_url=self.meilisearch_url,
            headers=headers,
            timeout=30.0
        )

        # Ensure index exists
        try:
            await self._ensure_index_exists()
            logger.info(f"Meilisearch index '{self.index_name}' ready")
        except Exception as e:
            logger.warning(f"Failed to verify Meilisearch index: {e}")
            logger.warning("Indexing will continue but may fail")

    async def _ensure_index_exists(self):
        """Create index if it doesn't exist."""
        if not self.client:
            return

        try:
            # Try to get index info
            response = await self.client.get(f"/indexes/{self.index_name}")
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                # Index doesn't exist, create it
                logger.info(f"Creating Meilisearch index: {self.index_name}")
                response = await self.client.post(
                    "/indexes",
                    json={
                        "uid": self.index_name,
                        "primaryKey": "id"
                    }
                )
                response.raise_for_status()
            else:
                raise

    async def process_scene(self, scene: SceneInfo) -> SceneInfo:
        """
        Process a single scene and buffer its document for indexing.

        Args:
            scene: Enriched SceneInfo object

        Returns:
            The same SceneInfo for downstream stages (pass-through)
        """
        try:
            document = scene_to_meilisearch_document(scene)
            self.document_buffer.append(document)

            # Flush buffer if it reaches batch size
            if len(self.document_buffer) >= self.batch_size:
                await self._flush_buffer()

        except Exception as e:
            logger.warning(
                f"Failed to create document for scene {scene.scene_name}: {e}"
            )

        return scene

    async def _flush_buffer(self):
        """Send buffered documents to Meilisearch."""
        if not self.document_buffer:
            return

        if not self.client:
            logger.warning("Meilisearch client not initialized, skipping flush")
            self.document_buffer.clear()
            return

        try:
            response = await self.client.post(
                f"/indexes/{self.index_name}/documents",
                json=self.document_buffer
            )
            response.raise_for_status()

            self.total_indexed += len(self.document_buffer)
            logger.info(
                f"Indexed {len(self.document_buffer)} documents "
                f"(total: {self.total_indexed})"
            )

            self.document_buffer.clear()

        except Exception as e:
            logger.error(f"Failed to index documents to Meilisearch: {e}")
            # Clear buffer to avoid retry storms
            self.document_buffer.clear()

    async def consume_and_produce(
        self,
        input_channel: trio.MemoryReceiveChannel,
        output_channel: Optional[trio.MemorySendChannel] = None
    ):
        """
        Override to ensure buffer flush on completion.

        Args:
            input_channel: Channel receiving enriched SceneInfo objects
            output_channel: Optional channel to forward processed scenes
        """
        try:
            await super().consume_and_produce(input_channel, output_channel)
        finally:
            # Flush any remaining documents
            await self._flush_buffer()
            await self.shutdown()

    async def shutdown(self):
        """Flush remaining documents and close HTTP client."""
        await self._flush_buffer()

        if self.client:
            await self.client.aclose()
            self.client = None

        logger.info(
            f"MeilisearchIndexer shutdown: {self.total_indexed} total documents indexed"
        )

        await super().shutdown()


# Utility functions for querying (optional, for development use)

async def search_scenes(
    meilisearch_url: str,
    index_name: str = "rawnind_scenes",
    query: str = "",
    filters: Optional[str] = None,
    limit: int = 20,
    api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search indexed scenes.

    Args:
        meilisearch_url: Meilisearch server URL
        index_name: Index to search
        query: Search query string
        filters: Optional Meilisearch filter expression
        limit: Max results to return
        api_key: Optional API key

    Returns:
        List of matching scene documents

    Example:
        # Find bayer scenes with good alignment
        results = await search_scenes(
            url,
            filters="cfa_type = 'bayer' AND avg_alignment_loss < 0.1"
        )
    """
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient(base_url=meilisearch_url, headers=headers) as client:
        response = await client.post(
            f"/indexes/{index_name}/search",
            json={
                "q": query,
                "filter": filters,
                "limit": limit,
            }
        )
        response.raise_for_status()
        data = response.json()
        return data.get("hits", [])