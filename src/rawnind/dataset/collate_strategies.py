"""
Strategy pattern implementation for collate functions.

Provides a clean, extensible architecture for different batching strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

import torch

from rawnind.dataset.SceneInfo import SceneInfo
from rawnind.dataset.constants import (
    DEFAULT_CROP_SIZE,
    DEFAULT_RESIZE_SIZE,
    CollateConfig,
)


class CollateStrategy(ABC):
    """Abstract base class for collate strategies."""

    def __init__(self, config: Optional[CollateConfig] = None):
        """Initialize with optional configuration."""
        self.config = config or CollateConfig()

    @abstractmethod
    def collate(self, scenes: List[SceneInfo]) -> Dict[str, Any]:
        """
        Convert list of SceneInfo objects into training batch.

        Args:
            scenes: List of SceneInfo objects

        Returns:
            Dictionary with batched tensors and metadata
        """
        pass

    def _create_base_tensors(
            self, batch_size: int, height: int, width: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Helper method to create base clean and noisy tensors.

        Args:
            batch_size: Number of samples in batch
            height: Tensor height
            width: Tensor width

        Returns:
            Tuple of (clean_tensor, noisy_tensor)
        """
        channels = self.config.channels
        clean = torch.zeros(batch_size, channels, height, width)
        noisy = torch.zeros(batch_size, channels, height, width)
        return clean, noisy

    def _extract_scene_names(self, scenes: List[SceneInfo]) -> List[str]:
        """Extract scene names from list of scenes."""
        return [scene.scene_name for scene in scenes]


class BasicCollateStrategy(CollateStrategy):
    """Basic collate strategy with fixed-size tensors."""

    def collate(self, scenes: List[SceneInfo]) -> Dict[str, Any]:
        """Create basic batch with default dimensions."""
        batch_size = len(scenes)
        clean, noisy = self._create_base_tensors(
            batch_size, DEFAULT_CROP_SIZE, DEFAULT_CROP_SIZE
        )

        return {
            "clean": clean,
            "noisy": noisy,
            "scene_names": self._extract_scene_names(scenes),
            "metadata": {"batch_size": batch_size},
        }


class RandomCropCollateStrategy(CollateStrategy):
    """Collate strategy with random cropping."""

    def __init__(self, crop_size: int = DEFAULT_CROP_SIZE):
        """Initialize with crop size."""
        config = CollateConfig(crop_size=crop_size)
        super().__init__(config)

    def collate(self, scenes: List[SceneInfo]) -> Dict[str, Any]:
        """Create batch with random crops."""
        batch_size = len(scenes)
        crop_size = self.config.crop_size
        clean, noisy = self._create_base_tensors(batch_size, crop_size, crop_size)

        # In a real implementation, this would apply random cropping
        # to the actual image data from scenes

        return {
            "clean": clean,
            "noisy": noisy,
            "scene_names": self._extract_scene_names(scenes),
            "metadata": {"batch_size": batch_size, "crop_size": crop_size},
        }


class ResizeCollateStrategy(CollateStrategy):
    """Collate strategy with resizing."""

    def __init__(self, target_size: int = DEFAULT_RESIZE_SIZE):
        """Initialize with target size."""
        config = CollateConfig(resize_size=target_size)
        super().__init__(config)

    def collate(self, scenes: List[SceneInfo]) -> Dict[str, Any]:
        """Create batch with resized images."""
        batch_size = len(scenes)
        size = self.config.resize_size
        clean, noisy = self._create_base_tensors(batch_size, size, size)

        # In a real implementation, this would resize the actual
        # image data from scenes

        return {
            "clean": clean,
            "noisy": noisy,
            "scene_names": self._extract_scene_names(scenes),
            "metadata": {"batch_size": batch_size, "target_size": size},
        }


class AugmentationCollateStrategy(CollateStrategy):
    """Collate strategy with data augmentation."""

    def __init__(self, flip: bool = True, rotate: bool = True):
        """Initialize with augmentation options."""
        config = CollateConfig(enable_flip=flip, enable_rotate=rotate)
        super().__init__(config)

    def collate(self, scenes: List[SceneInfo]) -> Dict[str, Any]:
        """Create batch with augmented data."""
        batch_size = len(scenes)
        clean, noisy = self._create_base_tensors(
            batch_size, DEFAULT_CROP_SIZE, DEFAULT_CROP_SIZE
        )

        # In a real implementation, this would apply augmentations
        # like flipping and rotation to the actual image data

        metadata = {
            "batch_size": batch_size,
            "augmentations": {
                "flip": self.config.enable_flip,
                "rotate": self.config.enable_rotate,
            },
        }

        return {
            "clean": clean,
            "noisy": noisy,
            "scene_names": self._extract_scene_names(scenes),
            "metadata": metadata,
        }


class CollateStrategyFactory:
    """Factory for creating collate strategies."""

    _strategies = {
        "basic": BasicCollateStrategy,
        "random_crop": RandomCropCollateStrategy,
        "resize": ResizeCollateStrategy,
        "augment": AugmentationCollateStrategy,
    }

    @classmethod
    def create(cls, strategy_type: str, **kwargs) -> CollateStrategy:
        """
        Create a collate strategy by type.

        Args:
            strategy_type: Type of strategy to create
            **kwargs: Additional arguments for strategy constructor

        Returns:
            CollateStrategy instance

        Raises:
            ValueError: If strategy_type is not recognized
        """
        if strategy_type not in cls._strategies:
            available = ", ".join(cls._strategies.keys())
            raise ValueError(
                f"Unknown strategy type: {strategy_type}. Available: {available}"
            )

        strategy_class = cls._strategies[strategy_type]
        return strategy_class(**kwargs)

    @classmethod
    def register(cls, name: str, strategy_class: type):
        """
        Register a custom collate strategy.

        Args:
            name: Name for the strategy
            strategy_class: Class implementing CollateStrategy
        """
        if not issubclass(strategy_class, CollateStrategy):
            raise TypeError(f"{strategy_class} must inherit from CollateStrategy")
        cls._strategies[name] = strategy_class
