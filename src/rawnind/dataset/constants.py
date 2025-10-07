"""
Constants and configuration for async pipeline bridge components.

This module centralizes all magic numbers and configuration values
used across the async pipeline integration components.
"""

from dataclasses import dataclass
from typing import Optional

# Tensor dimensions
DEFAULT_CHANNELS = 3
DEFAULT_CROP_SIZE = 256
DEFAULT_RESIZE_SIZE = 512
DEFAULT_BATCH_SIZE = 2

# Cache configuration
BYTES_PER_KB = 1024
BYTES_PER_MB = BYTES_PER_KB * 1024
CACHE_FILE_EXTENSION = ".cache"
DEFAULT_CACHE_EVICTION_POLICY = "lru"

# Pipeline configuration
DEFAULT_CHANNEL_BUFFER_SIZE = 10
DEFAULT_PIPELINE_TIMEOUT_SECONDS = 5
DEFAULT_PER_SCENE_TIMEOUT = 2.0
DEFAULT_TOTAL_TIMEOUT = 30.0

# Performance thresholds
MIN_SCENES_FOR_EAGER_START = 5
DEFAULT_DELAY_MS = 100
MAX_CONCURRENT_WORKERS = 5


@dataclass
class CacheConfig:
    """Configuration for disk cache manager."""

    cache_dir: str
    max_size_mb: Optional[int] = None
    eviction_policy: str = DEFAULT_CACHE_EVICTION_POLICY
    thread_safe: bool = False
    track_metrics: bool = False
    verify_checksums: bool = False
    auto_repair: bool = False


@dataclass
class CollateConfig:
    """Configuration for collate functions."""

    crop_size: int = DEFAULT_CROP_SIZE
    resize_size: int = DEFAULT_RESIZE_SIZE
    channels: int = DEFAULT_CHANNELS
    enable_flip: bool = True
    enable_rotate: bool = True


@dataclass
class PipelineConfig:
    """Configuration for pipeline orchestration."""

    eager_start: bool = False
    stream_batches: bool = False
    channel_buffer_size: int = DEFAULT_CHANNEL_BUFFER_SIZE
    timeout_seconds: int = DEFAULT_PIPELINE_TIMEOUT_SECONDS
    min_scenes: Optional[int] = None


@dataclass
class AdapterConfig:
    """Configuration for PyTorch DataLoader adapters."""

    worker_safe: bool = False
    indexable: bool = True
    memory_efficient: bool = False
    prefetch: bool = False
    batch_size: int = DEFAULT_BATCH_SIZE


@dataclass
class ValidationConfig:
    """Configuration for scene validation."""

    require_clean: bool = True
    min_noisy_images: int = 1
    strict: bool = True


# Mock data values (to be replaced with real data loading)
MOCK_CLEAN_IMAGE_ID = "clean_image"
MOCK_NOISY_IMAGE_ID = "noisy_image"
MOCK_DATASET_SIZE = 10
