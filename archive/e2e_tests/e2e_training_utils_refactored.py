"""Refactored utilities for end-to-end training smoke test.

This module provides supporting classes and functions for the E2E training test,
with improved code quality following SOLID principles.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, Protocol
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import logging
from enum import Enum

from .async_to_sync_bridge import AsyncPipelineBridge
from .SceneInfo import SceneInfo

logger = logging.getLogger(__name__)

# ==================== Constants ====================
DEFAULT_CACHE_DIR = Path("/tmp/e2e_cache")
DEFAULT_CHECKPOINT_DIR = Path("/tmp/e2e_checkpoints")
DEFAULT_CONFIG_DIR = Path("/tmp/e2e_config")

# Model defaults
DEFAULT_FUNIT = 8
DEFAULT_BAYER_CHANNELS = 4
DEFAULT_RGB_CHANNELS = 3

# Training defaults
DEFAULT_NUM_ITERATIONS = 5
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_BATCH_SIZE = 2
DEFAULT_CROP_SIZE_TRAIN = 64
DEFAULT_CROP_SIZE_VAL = 128

# Dataset splits
TRAIN_SCENES = 7
VAL_SCENES = 2
TEST_SCENES = 1

# Performance thresholds
MAX_GRADIENT_NORM = 1.0
MEMORY_GROWTH_TOLERANCE = 1.2
LOSS_INCREASE_TOLERANCE = 1.5

# Image dimensions for synthetic data
SYNTHETIC_IMAGE_HEIGHT = 256
SYNTHETIC_IMAGE_WIDTH = 256
SYNTHETIC_NOISE_LEVEL = 0.1
SYNTHETIC_MEAN_LEVEL = 0.5


# ==================== Enums ====================
class DatasetSplit(Enum):
    """Dataset split types."""
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class OptimizerType(Enum):
    """Supported optimizer types."""
    ADAM = "Adam"
    SGD = "SGD"


class LossType(Enum):
    """Supported loss function types."""
    MSE = "MSE"
    L1 = "L1"


# ==================== Protocols ====================
class TensorProcessor(Protocol):
    """Protocol for tensor processing operations."""
    def process(self, tensor: torch.Tensor) -> torch.Tensor:
        """Process a tensor."""
        ...


# ==================== Configuration ====================
@dataclass
class E2ETrainingConfig:
    """Improved configuration for end-to-end training smoke test with validation."""

    # Dataset configuration
    num_scenes_train: int = TRAIN_SCENES
    num_scenes_val: int = VAL_SCENES
    num_scenes_test: int = TEST_SCENES
    crop_size_train: int = DEFAULT_CROP_SIZE_TRAIN
    crop_size_val: int = DEFAULT_CROP_SIZE_VAL
    batch_size_train: int = DEFAULT_BATCH_SIZE
    batch_size_val: int = 1

    # Model configuration
    model_type: str = "UtNet2"
    model_params: Dict[str, Any] = field(default_factory=lambda: {
        "funit": DEFAULT_FUNIT,
        "input_channels": DEFAULT_BAYER_CHANNELS,
        "output_channels": DEFAULT_RGB_CHANNELS,
    })

    # Training configuration
    num_iterations: int = DEFAULT_NUM_ITERATIONS
    learning_rate: float = DEFAULT_LEARNING_RATE
    optimizer: str = OptimizerType.ADAM.value
    loss_function: str = LossType.MSE.value

    # Paths
    cache_dir: Path = field(default_factory=lambda: DEFAULT_CACHE_DIR)
    checkpoint_dir: Path = field(default_factory=lambda: DEFAULT_CHECKPOINT_DIR)
    config_dir: Path = field(default_factory=lambda: DEFAULT_CONFIG_DIR)

    # Performance
    num_workers: int = 2
    prefetch_factor: int = 2
    pin_memory: bool = False

    # Validation
    validate_every: int = 3
    save_checkpoint_every: int = 5

    # Timeouts
    pipeline_timeout: float = 30.0
    iteration_timeout: float = 10.0

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters."""
        if self.num_iterations <= 0:
            raise ValueError(f"num_iterations must be positive, got {self.num_iterations}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.batch_size_train <= 0:
            raise ValueError(f"batch_size_train must be positive, got {self.batch_size_train}")
        if self.crop_size_train <= 0:
            raise ValueError(f"crop_size_train must be positive, got {self.crop_size_train}")

    @property
    def total_scenes(self) -> int:
        """Total number of scenes across all splits."""
        return self.num_scenes_train + self.num_scenes_val + self.num_scenes_test


# ==================== Image Processing ====================
class ImageProcessor(ABC):
    """Abstract base class for image processing operations."""

    @abstractmethod
    def process(self, data: Any) -> torch.Tensor:
        """Process input data into a tensor."""
        pass


class BayerProcessor(ImageProcessor):
    """Process scene data into bayer tensor format."""

    def __init__(self, height: int = SYNTHETIC_IMAGE_HEIGHT,
                 width: int = SYNTHETIC_IMAGE_WIDTH):
        self.height = height
        self.width = width

    def process(self, scene: SceneInfo) -> torch.Tensor:
        """Convert scene to bayer tensor (4 channels: RGGB).

        Args:
            scene: SceneInfo containing image data

        Returns:
            bayer tensor with shape (4, H, W)
        """
        # For testing, create synthetic Bayer data
        bayer = torch.randn(DEFAULT_BAYER_CHANNELS, self.height, self.width)
        bayer = bayer * SYNTHETIC_NOISE_LEVEL + SYNTHETIC_MEAN_LEVEL
        return bayer.clamp(0, 1)


class RGBProcessor(ImageProcessor):
    """Process scene data into RGB tensor format."""

    def __init__(self, height: int = SYNTHETIC_IMAGE_HEIGHT,
                 width: int = SYNTHETIC_IMAGE_WIDTH):
        self.height = height
        self.width = width

    def process(self, scene: SceneInfo) -> torch.Tensor:
        """Convert scene to RGB tensor (3 channels).

        Args:
            scene: SceneInfo containing image data

        Returns:
            RGB tensor with shape (3, H, W)
        """
        # For testing, create synthetic RGB data
        rgb = torch.randn(DEFAULT_RGB_CHANNELS, self.height, self.width)
        rgb = rgb * SYNTHETIC_NOISE_LEVEL + SYNTHETIC_MEAN_LEVEL
        return rgb.clamp(0, 1)


# ==================== Tensor Operations ====================
class TensorOperations:
    """Utility class for common tensor operations."""

    @staticmethod
    def random_crop(input_tensor: torch.Tensor,
                   target_tensor: torch.Tensor,
                   crop_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply synchronized random crop to input and target tensors.

        Args:
            input_tensor: Input tensor (C, H, W)
            target_tensor: Target tensor (C, H, W)
            crop_size: Size of square crop

        Returns:
            Tuple of cropped input and target tensors

        Raises:
            ValueError: If crop size exceeds tensor dimensions
        """
        _, h, w = input_tensor.shape

        if crop_size > h or crop_size > w:
            raise ValueError(f"Crop size {crop_size} exceeds tensor dimensions ({h}, {w})")

        # Random crop coordinates
        top = np.random.randint(0, h - crop_size + 1)
        left = np.random.randint(0, w - crop_size + 1)

        # Apply crop to both tensors
        input_crop = input_tensor[:, top:top+crop_size, left:left+crop_size]
        target_crop = target_tensor[:, top:top+crop_size, left:left+crop_size]

        return input_crop, target_crop

    @staticmethod
    def resize_to_match(source: torch.Tensor,
                       target_shape: Tuple[int, int],
                       mode: str = 'bilinear') -> torch.Tensor:
        """Resize tensor to match target shape.

        Args:
            source: Source tensor to resize
            target_shape: Target (H, W) shape
            mode: Interpolation mode

        Returns:
            Resized tensor
        """
        if source.dim() == 3:
            source = source.unsqueeze(0)
            squeeze_after = True
        else:
            squeeze_after = False

        resized = F.interpolate(
            source,
            size=target_shape,
            mode=mode,
            align_corners=False if mode == 'bilinear' else None
        )

        if squeeze_after:
            resized = resized.squeeze(0)

        return resized


# ==================== Dataset Split Manager ====================
class DatasetSplitManager:
    """Manages dataset splitting logic."""

    def __init__(self, total_scenes: int,
                 train_scenes: int = TRAIN_SCENES,
                 val_scenes: int = VAL_SCENES,
                 test_scenes: int = TEST_SCENES):
        """Initialize split manager.

        Args:
            total_scenes: Total number of available scenes
            train_scenes: Number of training scenes
            val_scenes: Number of validation scenes
            test_scenes: Number of test scenes
        """
        self.total_scenes = total_scenes
        self.train_scenes = train_scenes
        self.val_scenes = val_scenes
        self.test_scenes = test_scenes

        self._validate_splits()

    def _validate_splits(self):
        """Validate that splits don't exceed total scenes."""
        required = self.train_scenes + self.val_scenes + self.test_scenes
        if required > self.total_scenes:
            raise ValueError(
                f"Split requirements ({required}) exceed total scenes ({self.total_scenes})"
            )

    def get_split_indices(self, split: DatasetSplit) -> Tuple[int, int]:
        """Get start and end indices for a dataset split.

        Args:
            split: Dataset split type

        Returns:
            Tuple of (start_idx, end_idx)
        """
        if split == DatasetSplit.TRAIN:
            return 0, self.train_scenes
        elif split == DatasetSplit.VAL:
            return self.train_scenes, self.train_scenes + self.val_scenes
        elif split == DatasetSplit.TEST:
            start = self.train_scenes + self.val_scenes
            return start, start + self.test_scenes
        else:
            raise ValueError(f"Unknown split: {split}")


# ==================== Improved Dataset Wrapper ====================
class E2EDatasetWrapper(Dataset):
    """Improved PyTorch Dataset wrapper with single responsibility."""

    def __init__(
        self,
        bridge: AsyncPipelineBridge,
        split: str = "train",
        crop_size: int = DEFAULT_CROP_SIZE_TRAIN,
        transform: Optional[Any] = None,
    ):
        """Initialize dataset wrapper.

        Args:
            bridge: AsyncPipelineBridge with collected scenes
            split: Dataset split ("train", "val", or "test")
            crop_size: Size of random crops to extract
            transform: Optional transforms to apply
        """
        self.bridge = bridge
        self.split = DatasetSplit(split)
        self.crop_size = crop_size
        self.transform = transform

        # Initialize processors
        self.bayer_processor = BayerProcessor()
        self.rgb_processor = RGBProcessor()

        # Initialize split manager
        total_scenes = len(bridge) if bridge else 10
        self.split_manager = DatasetSplitManager(total_scenes)

        # Get split indices
        self.start_idx, self.end_idx = self.split_manager.get_split_indices(self.split)
        self.num_scenes = self.end_idx - self.start_idx

        logger.info(f"Created {split} dataset with {self.num_scenes} scenes")

    def __len__(self) -> int:
        """Return number of scenes in this split."""
        return self.num_scenes

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset.

        Args:
            idx: Index within this split

        Returns:
            Dictionary with "input" (bayer) and "target" (RGB) tensors
        """
        # Map split index to global scene index
        global_idx = self.start_idx + idx

        # Get scene from bridge
        scene = self.bridge[global_idx]

        # Process scene to tensors
        input_tensor = self.bayer_processor.process(scene)
        target_tensor = self.rgb_processor.process(scene)

        # Apply random crop
        input_tensor, target_tensor = TensorOperations.random_crop(
            input_tensor, target_tensor, self.crop_size
        )

        # Apply additional transforms if provided
        if self.transform:
            input_tensor = self.transform(input_tensor)
            target_tensor = self.transform(target_tensor)

        return {
            "input": input_tensor,
            "target": target_tensor,
        }


# ==================== Metrics Tracking ====================
class MetricsTracker:
    """Handles gradient and training metrics tracking."""

    @staticmethod
    def verify_gradient_flow(model: nn.Module) -> Dict[str, float]:
        """Verify that gradients are flowing through the model.

        Args:
            model: PyTorch model with computed gradients

        Returns:
            Dictionary mapping parameter names to gradient L2 norms
        """
        gradient_norms = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                gradient_norms[name] = grad_norm

                # Log warnings for problematic gradients
                if grad_norm == 0:
                    logger.warning(f"Zero gradient in parameter: {name}")
                elif np.isnan(grad_norm):
                    logger.error(f"NaN gradient in parameter: {name}")
                elif np.isinf(grad_norm):
                    logger.error(f"Inf gradient in parameter: {name}")

        return gradient_norms

    @staticmethod
    def compute_metrics(
        loss: float,
        gradient_norms: Dict[str, float],
        learning_rate: float,
        batch_time: float,
        memory_mb: Optional[float] = None,
    ) -> Dict[str, float]:
        """Compute and format training metrics.

        Args:
            loss: Training loss value
            gradient_norms: Dictionary of parameter gradient norms
            learning_rate: Current learning rate
            batch_time: Time taken for batch in seconds
            memory_mb: Memory usage in MB (optional)

        Returns:
            Dictionary of aggregated metrics
        """
        grad_values = list(gradient_norms.values())

        metrics = {
            "loss": loss,
            "avg_grad_norm": np.mean(grad_values) if grad_values else 0.0,
            "max_grad_norm": np.max(grad_values) if grad_values else 0.0,
            "min_grad_norm": np.min(grad_values) if grad_values else 0.0,
            "learning_rate": learning_rate,
            "batch_time": batch_time,
        }

        if memory_mb is not None:
            metrics["memory_mb"] = memory_mb

        return metrics

    @staticmethod
    def log_metrics(metrics: Dict[str, float], iteration: Optional[int] = None):
        """Log metrics in a formatted way.

        Args:
            metrics: Dictionary of metrics to log
            iteration: Optional iteration number
        """
        prefix = f"Iter {iteration}: " if iteration is not None else ""

        log_msg = (
            f"{prefix}"
            f"loss={metrics['loss']:.4f}, "
            f"avg_grad={metrics['avg_grad_norm']:.4f}, "
            f"max_grad={metrics['max_grad_norm']:.4f}, "
            f"lr={metrics['learning_rate']:.6f}, "
            f"time={metrics['batch_time']:.2f}s"
        )

        if "memory_mb" in metrics:
            log_msg += f", mem={metrics['memory_mb']:.1f}MB"

        logger.info(log_msg)


# ==================== Config Generator ====================
class ConfigGenerator:
    """Generates configuration files for training."""

    @staticmethod
    def create_yaml_config(config: E2ETrainingConfig) -> Dict[str, Any]:
        """Create YAML configuration dictionary.

        Args:
            config: E2E training configuration

        Returns:
            Dictionary suitable for YAML serialization
        """
        return {
            "experiment_name": "e2e_training_smoke_test",

            # Dataset configuration
            "dataset": {
                "type": "AsyncPipelineDataset",
                "num_scenes_train": config.num_scenes_train,
                "num_scenes_val": config.num_scenes_val,
                "num_scenes_test": config.num_scenes_test,
                "crop_size_train": config.crop_size_train,
                "crop_size_val": config.crop_size_val,
                "cache_dir": str(config.cache_dir),
            },

            # Model configuration
            "model": {
                "type": config.model_type,
                "params": config.model_params,
            },

            # Training configuration
            "training": {
                "num_iterations": config.num_iterations,
                "batch_size_train": config.batch_size_train,
                "batch_size_val": config.batch_size_val,
                "learning_rate": config.learning_rate,
                "optimizer": config.optimizer,
                "loss_function": config.loss_function,
                "num_workers": config.num_workers,
                "prefetch_factor": config.prefetch_factor,
                "pin_memory": config.pin_memory,
            },

            # Checkpointing
            "checkpointing": {
                "checkpoint_dir": str(config.checkpoint_dir),
                "save_every": config.save_checkpoint_every,
                "validate_every": config.validate_every,
            },

            # Logging
            "logging": {
                "level": "INFO",
                "log_every": 1,
            },
        }


# ==================== Backward Compatibility ====================
# Keep old function names for compatibility
def create_minimal_config_yaml(config: E2ETrainingConfig) -> Dict[str, Any]:
    """Create minimal YAML configuration (backward compatibility wrapper)."""
    return ConfigGenerator.create_yaml_config(config)


def verify_gradient_flow(model: nn.Module) -> Dict[str, float]:
    """Verify gradient flow (backward compatibility wrapper)."""
    return MetricsTracker.verify_gradient_flow(model)


def track_training_metrics(
    loss: float,
    gradient_norms: Dict[str, float],
    learning_rate: float,
    batch_time: float,
    memory_mb: Optional[float] = None,
) -> Dict[str, float]:
    """Track training metrics (backward compatibility wrapper)."""
    metrics = MetricsTracker.compute_metrics(
        loss, gradient_norms, learning_rate, batch_time, memory_mb
    )
    MetricsTracker.log_metrics(metrics)
    return metrics