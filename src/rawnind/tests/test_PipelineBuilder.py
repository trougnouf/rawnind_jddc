from pathlib import Path
from unittest.mock import patch, AsyncMock

import pytest

from rawnind.dataset.PipelineBuilder import PipelineBuilder


@pytest.mark.asyncio
async def test_pipeline_builder_initialization():
    dataset_root = Path("/fake/dataset/root")
    cache_paths = (Path("/fake/cache"), Path("/fake/metadata"))
    pipeline = PipelineBuilder(
        dataset_root=dataset_root,
        cache_paths=cache_paths,
        dataset_metadata_url="http://example.com/api",
        max_concurrent_downloads=5,
        max_concurrent_enrichment=4,
        enable_enrichment=True,
        enable_crops_enrichment=False
    )

    assert pipeline.dataset_root == dataset_root
    assert pipeline.enable_enrichment is True


@pytest.mark.asyncio
async def test_pipeline_builder_run_without_enrichment():
    with patch("rawnind.dataset.PipelineBuilder.trio.open_nursery") as mock_open_nursery:
        dataset_root = Path("/fake/dataset/root")
        pipeline = PipelineBuilder(dataset_root=dataset_root, enable_enrichment=False)

        await pipeline.run()

        assert mock_open_nursery.called
        # Additional assertions for tasks started by the nursery


@pytest.mark.asyncio
async def test_pipeline_builder_run_with_enrichment():
    with patch("rawnind.dataset.PipelineBuilder.trio.open_nursery") as mock_open_nursery:
        dataset_root = Path("/fake/dataset/root")
        pipeline = PipelineBuilder(dataset_root=dataset_root, enable_enrichment=True)

        await pipeline.run()

        assert mock_open_nursery.called
        # Additional assertions for tasks started by the nursery


@pytest.mark.asyncio
async def test_final_consumer():
    with patch("rawnind.dataset.PipelineBuilder.logger") as mock_logger:
        dataset_root = Path("/fake/dataset/root")
        pipeline = PipelineBuilder(dataset_root=dataset_root)

        recv_channel = AsyncMock()
        recv_channel.__aenter__ = AsyncMock(return_value=recv_channel)
        recv_channel.__aexit__ = AsyncMock(return_value=None)

        scene_info = type('SceneInfo', (object,), {'scene_name': 'test_scene'})
        recv_channel.__aiter__.return_value = [scene_info]

        await pipeline._final_consumer(recv_channel)

        mock_logger.info.assert_called_with("Pipeline completed scene: test_scene")
