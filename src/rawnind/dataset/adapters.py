"""
Dataset adapter framework for raw image processing pipelines.

Provides abstractions for connecting an AsyncPipelineBridge to PyTorch datasets,
backward‑compatible legacy adapters, and a factory to instantiate the
appropriate adapter type.

Exported classes:
    DatasetAdapter
    PipelineDataLoaderAdapter
    LegacyAdapter
    BackwardsCompatAdapter
    AdapterFactory
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List, Tuple

import torch.utils.data as data

from rawnind.dataset.SceneInfo import SceneInfo
from rawnind.dataset import AsyncPipelineBridge
from rawnind.dataset.constants import (
    AdapterConfig,
    MOCK_CLEAN_IMAGE_ID,
    MOCK_NOISY_IMAGE_ID,
    MOCK_DATASET_SIZE,
)


class DatasetAdapter(ABC, data.Dataset):
    """
    A base adapter class for dataset objects.

    This abstract base class extends the standard Dataset interface and
    provides a template for adapters that convert raw data into a form
    suitable for model consumption. Concrete implementations must
    implement the dataset protocol methods as well as a configuration
    retrieval method.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return dataset size."""
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """Get item by index."""
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get adapter configuration."""
        pass


class PipelineDataLoaderAdapter(DatasetAdapter):
    """
    PyTorch Dataset adapter for AsyncPipelineBridge.

    Wraps bridge scenes to provide standard PyTorch Dataset interface
    with improved configuration management.
    """

    def __init__(
            self,
            bridge: AsyncPipelineBridge,
            config: Optional[AdapterConfig] = None,
            # Legacy keyword arguments for backwards compatibility
            worker_safe: Optional[bool] = None,
            indexable: Optional[bool] = None,
            memory_efficient: Optional[bool] = None,
            prefetch: Optional[bool] = None
    ):
        """
        Initialize adapter with bridge and configuration.

        Args:
            bridge: AsyncPipelineBridge instance
            config: Adapter configuration (uses defaults if None)
            worker_safe: Legacy - use config.worker_safe instead
            indexable: Legacy - use config.indexable instead
            memory_efficient: Legacy - use config.memory_efficient instead
            prefetch: Legacy - use config.prefetch instead
        """
        self.bridge = bridge

        # Handle legacy arguments by creating config if needed
        if any(arg is not None for arg in [worker_safe, indexable, memory_efficient, prefetch]):
            if config is None:
                config = AdapterConfig()
            # Override config with legacy arguments
            if worker_safe is not None:
                config.worker_safe = worker_safe
            if indexable is not None:
                config.indexable = indexable
            if memory_efficient is not None:
                config.memory_efficient = memory_efficient
            if prefetch is not None:
                config.prefetch = prefetch

        self.config = config or AdapterConfig()

        # Pre-cache scenes if indexable for better performance
        if self.config.indexable and self.config.prefetch:
            self._prefetch_scenes()

    def __len__(self) -> int:
        """Return number of scenes in bridge."""
        return len(self.bridge)

    def __getitem__(self, index: int) -> SceneInfo:
        """
        Get scene by index.

        Args:
            index: Scene index

        Returns:
            SceneInfo object at the given index
        """
        if not self.config.indexable:
            raise RuntimeError("Adapter is configured as non-indexable")

        return self.bridge.get_scene(index)

    def get_config(self) -> Dict[str, Any]:
        """Get adapter configuration as dictionary."""
        return {
            "worker_safe": self.config.worker_safe,
            "indexable": self.config.indexable,
            "memory_efficient": self.config.memory_efficient,
            "prefetch": self.config.prefetch,
            "batch_size": self.config.batch_size,
        }

    def _prefetch_scenes(self) -> None:
        """Prefetch scenes for improved performance."""
        # In a real implementation, this would trigger
        # background loading of scenes
        pass


class LegacyAdapter(ABC):
    """
    Abstract base class for legacy compatibility adapters.

    Provides interface that matches legacy dataset APIs.
    """

    @abstractmethod
    def get_clean_noisy_pair(self, idx: int) -> Tuple[Any, List[Any]]:
        """Get clean and noisy image pair (legacy API)."""
        pass

    @abstractmethod
    def random_crops(self, *args, **kwargs) -> Any:
        """Apply random crops (legacy API)."""
        pass


class BackwardsCompatAdapter(LegacyAdapter, DatasetAdapter):
    """
    Backwards compatibility adapter for legacy training scripts.

    Provides both modern and legacy API methods for migration.
    """

    def __init__(
            self,
            cache_dir: Optional[str] = None,
            dataset_type: str = "bayer2prgb",
            bridge: Optional[AsyncPipelineBridge] = None
    ):
        """
        Initialize backwards compatibility adapter.

        Args:
            cache_dir: Cache directory path (for compatibility)
            dataset_type: Type of dataset (bayer2prgb, prgb2prgb, etc.)
            bridge: Optional bridge for real data access
        """
        self.cache_dir = cache_dir
        self.dataset_type = dataset_type
        self.bridge = bridge
        self._scenes: List[SceneInfo] = []

        # Load scenes from bridge if available
        if self.bridge:
            self._load_scenes_from_bridge()

    def get_clean_noisy_pair(self, idx: int) -> Tuple[Any, List[Any]]:
        """
        Legacy API to get clean and noisy pair.

        Args:
            idx: Index of the pair

        Returns:
            Tuple of (clean_image, list_of_noisy_images)
        """
        if self._scenes and idx < len(self._scenes):
            scene = self._scenes[idx]
            # Extract clean and noisy from SceneInfo
            clean = scene.clean_images[0] if scene.clean_images else MOCK_CLEAN_IMAGE_ID
            noisy = scene.noisy_images if scene.noisy_images else [MOCK_NOISY_IMAGE_ID]
            return (clean, noisy)

        # Fallback to mock data
        return (MOCK_CLEAN_IMAGE_ID, [MOCK_NOISY_IMAGE_ID])

    def random_crops(self, *args, **kwargs) -> None:
        """Legacy random crops method (placeholder for compatibility)."""
        #todo
        pass

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self._scenes) if self._scenes else MOCK_DATASET_SIZE

    def __getitem__(self, index: int) -> Optional[SceneInfo]:
        """
        Get item by index.

        Args:
            index: Item index

        Returns:
            SceneInfo at index or None if out of bounds
        """
        if 0 <= index < len(self._scenes):
            return self._scenes[index]
        return None

    def get_config(self) -> Dict[str, Any]:
        """Get adapter configuration."""
        return {
            "cache_dir": self.cache_dir,
            "dataset_type": self.dataset_type,
            "has_bridge": self.bridge is not None,
            "scene_count": len(self._scenes),
        }

    def _load_scenes_from_bridge(self) -> None:
        """Load scenes from bridge if available."""
        if self.bridge:
            self._scenes = list(self.bridge)


class AdapterFactory:
    """Adapter factory for creating dataset adapters.

    The factory encapsulates the logic required to instantiate the appropriate
    adapter class based on a string identifier.  It accepts an optional
    `AsyncPipelineBridge` and additional keyword arguments that are passed
    directly to the adapter constructor.

    Adapters supported:

    * ``pipeline`` – returns a ``PipelineDataLoaderAdapter`` configured
      with an ``AdapterConfig`` instance.
    * ``legacy`` and ``compat`` – both map to
      ``BackwardsCompatAdapter`` and accept the same keyword arguments.

    If an unknown type is supplied, the factory raises a ``ValueError`` with
    a clear message indicating the valid options.

    Attributes
    ----------
    None
    """

    @staticmethod
    def create(
            adapter_type: str,
            bridge: Optional[AsyncPipelineBridge] = None,
            **kwargs
    ) -> DatasetAdapter:
        """
        Create a dataset adapter by type.

        Args:
            adapter_type: Type of adapter ("pipeline", "legacy", "compat")
            bridge: AsyncPipelineBridge instance
            **kwargs: Additional adapter-specific arguments

        Returns:
            DatasetAdapter instance

        Raises:
            ValueError: If adapter_type is not recognized
        """
        adapters = {
            "pipeline": PipelineDataLoaderAdapter,
            "legacy": BackwardsCompatAdapter,
            "compat": BackwardsCompatAdapter,
        }

        adapter_class = adapters.get(adapter_type.lower())
        if not adapter_class:
            available = ", ".join(adapters.keys())
            raise ValueError(
                f"Unknown adapter type: {adapter_type}. Available: {available}"
            )

        if adapter_type == "pipeline":
            config = AdapterConfig(**kwargs)
            return adapter_class(bridge, config)
        else:
            return adapter_class(bridge=bridge, **kwargs)
