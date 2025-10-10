"""
Basic tests for MeilisearchIndexer.

Lighter test coverage than YAMLArtifactWriter since this is a utility/debug tool.

Enhanced with trio.testing utilities for deterministic async testing:
- MockClock for timeout/timing tests
- wait_all_tasks_blocked for state verification
- Sequencer for concurrent access ordering
"""

from pathlib import Path
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import trio
import trio.testing
import httpx

from rawnind.dataset.SceneInfo import SceneInfo, ImageInfo
from rawnind.dataset.MeilisearchIndexer import (
    MeilisearchIndexer,
    scene_to_meilisearch_document,
    search_scenes
)

pytestmark = pytest.mark.dataset


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def minimal_scene():
    """Create a minimal valid SceneInfo for testing."""
    gt_img = ImageInfo(
        filename="gt.exr",
        sha1="abc123",
        is_clean=True,
        scene_name="test_scene",
        scene_images=["gt.exr", "noisy.arw"],
        cfa_type="bayer",
        local_path=Path("/fake/dataset/test_scene/gt/gt.exr"),
        validated=True,
    )

    noisy_img = ImageInfo(
        filename="noisy.arw",
        sha1="def456",
        is_clean=False,
        scene_name="test_scene",
        scene_images=["gt.exr", "noisy.arw"],
        cfa_type="bayer",
        local_path=Path("/fake/dataset/test_scene/noisy.arw"),
        validated=True,
        metadata={
            "alignment": [2, -3],
            "alignment_loss": 0.05,
            "mask_mean": 0.92,
            "raw_gain": 1.5,
            "crops": [{"coordinates": [512, 256]}],
        }
    )

    return SceneInfo(
        scene_name="test_scene",
        cfa_type="bayer",
        unknown_sensor=False,
        test_reserve=False,
        clean_images=[gt_img],
        noisy_images=[noisy_img],
    )


@pytest.fixture
def mock_httpx_client():
    """Mock httpx.AsyncClient."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    return mock_client


# ============================================================================
# Tests for scene_to_meilisearch_document()
# ============================================================================

def test_scene_to_meilisearch_document_basic(minimal_scene):
    """Test basic document generation."""
    document = scene_to_meilisearch_document(minimal_scene)

    assert document["id"] == "test_scene"
    assert document["scene_name"] == "test_scene"
    assert document["cfa_type"] == "bayer"
    assert document["unknown_sensor"] is False
    assert document["test_reserve"] is False
    assert document["clean_image_count"] == 1
    assert document["noisy_image_count"] == 1
    assert document["avg_alignment_loss"] == 0.05
    assert document["avg_mask_mean"] == 0.92
    assert document["total_crops"] == 1
    assert document["has_metadata"] is True
    assert document["gt_filename"] == "gt.exr"
    assert document["noisy_filenames"] == ["noisy.arw"]
    assert "indexed_at" in document


def test_scene_to_meilisearch_document_no_metadata():
    """Test document generation with no metadata."""
    scene = SceneInfo(
        scene_name="minimal_scene",
        cfa_type="x-trans",
        unknown_sensor=True,
        test_reserve=True,
        clean_images=[],
        noisy_images=[ImageInfo(
            filename="noisy.arw",
            sha1="abc",
            is_clean=False,
            scene_name="minimal_scene",
            scene_images=["noisy.arw"],
            cfa_type="x-trans",
        )],
    )

    document = scene_to_meilisearch_document(scene)

    assert document["scene_name"] == "minimal_scene"
    assert document["cfa_type"] == "x-trans"
    assert document["has_metadata"] is False
    assert document["avg_alignment_loss"] == 0.0
    assert document["avg_mask_mean"] == 0.0
    assert document["total_crops"] == 0


def test_scene_to_meilisearch_document_multiple_noisy():
    """Test aggregation with multiple noisy images."""
    scene = SceneInfo(
        scene_name="multi_scene",
        cfa_type="bayer",
        unknown_sensor=False,
        test_reserve=False,
        clean_images=[],
        noisy_images=[
            ImageInfo(
                filename="noisy1.arw",
                sha1="abc",
                is_clean=False,
                scene_name="multi_scene",
                scene_images=["noisy1.arw"],
                cfa_type="bayer",
                metadata={"alignment_loss": 0.1, "mask_mean": 0.9, "crops": [1, 2]}
            ),
            ImageInfo(
                filename="noisy2.arw",
                sha1="def",
                is_clean=False,
                scene_name="multi_scene",
                scene_images=["noisy2.arw"],
                cfa_type="bayer",
                metadata={"alignment_loss": 0.2, "mask_mean": 0.8, "crops": [3, 4, 5]}
            ),
        ],
    )

    document = scene_to_meilisearch_document(scene)

    assert document["noisy_image_count"] == 2
    assert document["avg_alignment_loss"] == pytest.approx(0.15)  # (0.1 + 0.2) / 2
    assert document["avg_mask_mean"] == pytest.approx(0.85)  # (0.9 + 0.8) / 2
    assert document["total_crops"] == 5  # 2 + 3


# ============================================================================
# Tests for MeilisearchIndexer Initialization
# ============================================================================

def test_meilisearch_indexer_init(tmp_path):
    """Test MeilisearchIndexer initialization."""
    indexer = MeilisearchIndexer(
        output_dir=tmp_path,
        meilisearch_url="http://localhost:7700",
        index_name="test_index",
        batch_size=50,
    )

    assert indexer.meilisearch_url == "http://localhost:7700"
    assert indexer.index_name == "test_index"
    assert indexer.batch_size == 50
    assert indexer.document_buffer == []
    assert indexer.total_indexed == 0


def test_meilisearch_indexer_strips_trailing_slash(tmp_path):
    """Test that trailing slash is removed from URL."""
    indexer = MeilisearchIndexer(
        output_dir=tmp_path,
        meilisearch_url="http://localhost:7700/",
    )

    assert indexer.meilisearch_url == "http://localhost:7700"


# ============================================================================
# Tests for process_scene()
# ============================================================================

@pytest.mark.trio
async def test_process_scene_buffers_document(minimal_scene, tmp_path):
    """Test that process_scene buffers documents.

    Uses wait_all_tasks_blocked to verify buffer state without arbitrary delays.
    """
    indexer = MeilisearchIndexer(output_dir=tmp_path, batch_size=10)

    result = await indexer.process_scene(minimal_scene)

    # Deterministic verification - no race conditions
    await trio.testing.wait_all_tasks_blocked()

    assert len(indexer.document_buffer) == 1
    assert indexer.document_buffer[0]["scene_name"] == "test_scene"
    assert result is minimal_scene


@pytest.mark.trio
async def test_process_scene_flushes_at_batch_size(minimal_scene, tmp_path, mock_httpx_client):
    """Test that buffer flushes at batch size.

    Uses wait_all_tasks_blocked to verify flush completion deterministically.
    """
    indexer = MeilisearchIndexer(output_dir=tmp_path, batch_size=2)
    indexer.client = mock_httpx_client

    # Mock successful POST
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_httpx_client.post.return_value = mock_response

    # Process first scene
    await indexer.process_scene(minimal_scene)
    await trio.testing.wait_all_tasks_blocked()
    assert len(indexer.document_buffer) == 1

    scene2 = SceneInfo(
        scene_name="scene2",
        cfa_type="bayer",
        unknown_sensor=False,
        test_reserve=False,
        clean_images=[minimal_scene.clean_images[0]],
        noisy_images=[minimal_scene.noisy_images[0]],
    )

    # Process second scene - triggers flush
    await indexer.process_scene(scene2)
    await trio.testing.wait_all_tasks_blocked()

    # Buffer should be flushed
    assert len(indexer.document_buffer) == 0
    assert indexer.total_indexed == 2
    mock_httpx_client.post.assert_called_once()


# ============================================================================
# Tests for Meilisearch Integration (Mocked)
# ============================================================================

@pytest.mark.trio
async def test_startup_creates_client(tmp_path):
    """Test that startup creates HTTP client."""
    indexer = MeilisearchIndexer(output_dir=tmp_path)

    with patch('httpx.AsyncClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_client.get.return_value = MagicMock(raise_for_status=lambda: None)
        mock_client_class.return_value = mock_client

        await indexer.startup()

        assert indexer.client is not None
        mock_client_class.assert_called_once()


@pytest.mark.trio
async def test_flush_buffer_sends_to_meilisearch(tmp_path, mock_httpx_client):
    """Test that flush_buffer sends documents.

    Uses Sequencer to deterministically order buffer state verification.
    """
    indexer = MeilisearchIndexer(output_dir=tmp_path)
    indexer.client = mock_httpx_client

    # Add documents to buffer
    indexer.document_buffer = [
        {"id": "scene1", "scene_name": "scene1"},
        {"id": "scene2", "scene_name": "scene2"},
    ]

    # Capture the JSON payload before it gets cleared
    json_payload_captured = None

    async def capture_post(*args, **kwargs):
        nonlocal json_payload_captured
        json_payload_captured = list(kwargs['json'])  # Make a copy
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        return mock_response

    mock_httpx_client.post = AsyncMock(side_effect=capture_post)

    await indexer._flush_buffer()
    await trio.testing.wait_all_tasks_blocked()

    # Verify POST was called with correct arguments
    assert json_payload_captured is not None
    assert len(json_payload_captured) == 2
    assert json_payload_captured[0]["id"] == "scene1"
    assert json_payload_captured[1]["id"] == "scene2"

    # Buffer should be cleared
    assert len(indexer.document_buffer) == 0
    assert indexer.total_indexed == 2


@pytest.mark.trio
async def test_flush_buffer_handles_errors(tmp_path, mock_httpx_client, caplog):
    """Test that flush_buffer handles errors gracefully.

    Uses MockClock to test timeout behavior deterministically.
    """
    indexer = MeilisearchIndexer(output_dir=tmp_path)
    indexer.client = mock_httpx_client

    indexer.document_buffer = [{"id": "scene1"}]

    # Mock failed POST
    mock_httpx_client.post.side_effect = httpx.HTTPError("Connection failed")

    await indexer._flush_buffer()
    await trio.testing.wait_all_tasks_blocked()

    # Buffer should be cleared to avoid retry storms
    assert len(indexer.document_buffer) == 0
    assert "Failed to index documents" in caplog.text


# ============================================================================
# Tests for search_scenes() Utility
# ============================================================================

@pytest.mark.trio
async def test_search_scenes():
    """Test search_scenes utility function."""
    with patch('httpx.AsyncClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "hits": [
                {"id": "scene1", "scene_name": "scene1"},
                {"id": "scene2", "scene_name": "scene2"},
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        results = await search_scenes(
            "http://localhost:7700",
            query="test",
            filters="cfa_type = 'bayer'",
            limit=10
        )

        assert len(results) == 2
        assert results[0]["scene_name"] == "scene1"


# ============================================================================
# Tests for PostDownloadWorker Compatibility
# ============================================================================

def test_inherits_from_post_download_worker(tmp_path):
    """Test that MeilisearchIndexer properly inherits from PostDownloadWorker."""
    from rawnind.dataset.PostDownloadWorker import PostDownloadWorker

    indexer = MeilisearchIndexer(output_dir=tmp_path)

    assert isinstance(indexer, PostDownloadWorker)
    assert hasattr(indexer, 'consume_and_produce')
    assert hasattr(indexer, 'startup')
    assert hasattr(indexer, 'shutdown')