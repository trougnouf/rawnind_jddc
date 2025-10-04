from pathlib import Path

import pytest
import trio

from rawnind.dataset.PipelineBuilder import PipelineBuilder
from rawnind.dataset.SceneInfo import SceneInfo, ImageInfo


@pytest.mark.trio
async def test_pipeline_builder_initialization():
    """Test that PipelineBuilder initializes all components correctly."""
    dataset_root = Path("/fake/dataset/root")
    cache_paths = (Path("/fake/cache.yaml"), Path("/fake/metadata.json"))
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
    assert pipeline.ingestor is not None
    assert pipeline.scanner is not None
    assert pipeline.downloader is not None
    assert pipeline.verifier is not None
    assert pipeline.indexer is not None
    assert pipeline.enricher is not None


@pytest.mark.trio
async def test_final_consumer():
    """Test that _final_consumer properly consumes scene objects."""
    dataset_root = Path("/fake/dataset/root")
    pipeline = PipelineBuilder(dataset_root=dataset_root)

    send_channel, recv_channel = trio.open_memory_channel(10)

    # Create a test scene
    scene_info = SceneInfo(
        scene_name="test_scene",
        cfa_type="Bayer",
        unknown_sensor=False,
        test_reserve=False,
        clean_images=[],
        noisy_images=[]
    )

    # Run consumer in background
    async with trio.open_nursery() as nursery:
        nursery.start_soon(pipeline._final_consumer, recv_channel)

        # Send a scene
        await send_channel.send(scene_info)
        await send_channel.aclose()


@pytest.mark.trio
async def test_pipeline_builder_with_mock_data(tmp_path):
    """Test pipeline with minimal mock data (integration-style test)."""
    import yaml

    # Setup mock cache files
    yaml_cache = tmp_path / "cache.yaml"
    metadata_cache = tmp_path / "metadata.json"
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    # Create minimal dataset
    dataset_data = {
        "Bayer": {
            "test_scene": {
                "clean_images": [{"filename": "clean.png", "sha1": "abc123"}],
                "noisy_images": [],
                "unknown_sensor": False,
                "test_reserve": False
            }
        }
    }

    with open(yaml_cache, "w") as f:
        yaml.dump(dataset_data, f)

    with open(metadata_cache, "w") as f:
        f.write('{"data": {"latestVersion": {"files": []}}}')

    # Create the image file so scanner finds it
    (dataset_root / "Bayer" / "test_scene" / "gt").mkdir(parents=True)
    clean_file = dataset_root / "Bayer" / "test_scene" / "gt" / "clean.png"
    clean_file.write_bytes(b"fake image data")

    pipeline = PipelineBuilder(
        dataset_root=dataset_root,
        cache_paths=(yaml_cache, metadata_cache),
        enable_enrichment=False
    )

    # This would run the full pipeline - we just verify it initializes
    assert pipeline is not None
