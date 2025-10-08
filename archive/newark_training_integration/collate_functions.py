"""
Collate functions for batching SceneInfo objects into training batches.

Refactored to use Strategy pattern while maintaining backward compatibility.
"""

from typing import List, Dict, Any, Callable

from rawnind.dataset.SceneInfo import SceneInfo
from rawnind.dataset.collate_strategies import (
    BasicCollateStrategy,
    RandomCropCollateStrategy,
    ResizeCollateStrategy,
    AugmentationCollateStrategy,
    CollateStrategyFactory,
)


def scene_batch_collate_fn(scenes: List[SceneInfo]) -> Dict[str, Any]:
    """
    Basic collate function to convert SceneInfo list into training batch.

    Args:
        scenes: List of SceneInfo objects

    Returns:
        Dictionary with clean/noisy tensors and metadata
    """
    strategy = BasicCollateStrategy()
    return strategy.collate(scenes)


def random_crop_collate_fn(crop_size: int = 256) -> Callable:
    """
    Factory function for random crop collate function.

    Args:
        crop_size: Size of random crops

    Returns:
        Collate function that crops to specified size
    """
    strategy = RandomCropCollateStrategy(crop_size=crop_size)
    return strategy.collate


def resize_collate_fn(target_size: int = 512) -> Callable:
    """
    Factory function for resize collate function.

    Args:
        target_size: Target size for resizing

    Returns:
        Collate function that resizes to specified size
    """
    strategy = ResizeCollateStrategy(target_size=target_size)
    return strategy.collate


def augment_collate_fn(flip: bool = True, rotate: bool = True) -> Callable:
    """
    Factory function for augmentation collate function.

    Args:
        flip: Enable random flipping
        rotate: Enable random rotation

    Returns:
        Collate function that applies augmentations
    """
    strategy = AugmentationCollateStrategy(flip=flip, rotate=rotate)
    return strategy.collate


def create_collate_fn(strategy_type: str = "basic", **kwargs) -> Callable:
    """
    Create a collate function using the specified strategy.

    Args:
        strategy_type: Type of collate strategy to use
        **kwargs: Additional arguments for the strategy

    Returns:
        Collate function using the specified strategy
    """
    strategy = CollateStrategyFactory.create(strategy_type, **kwargs)
    return strategy.collate
