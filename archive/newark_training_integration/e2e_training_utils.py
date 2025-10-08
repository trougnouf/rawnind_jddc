"""Utilities for end-to-end training smoke test.

This module provides supporting classes and functions for the E2E training test,
including configuration, dataset wrapper, and validation utilities.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import logging

from .async_to_sync_bridge import AsyncPipelineBridge
from .SceneInfo import SceneInfo

logger = logging.getLogger(__name__)


@dataclass
class E2ETrainingConfig:
    """Configuration for end-to-end training smoke test."""
    
    # Dataset configuration
    num_scenes_train: int = 7
    num_scenes_val: int = 2
    num_scenes_test: int = 1
    crop_size_train: int = 64
    crop_size_val: int = 128
    batch_size_train: int = 2
    batch_size_val: int = 1
    
    # Model configuration
    model_type: str = "UtNet2"
    model_params: Dict[str, Any] = field(default_factory=lambda: {
        "funit": 8,
        "input_channels": 4,
        "output_channels": 3,
    })
    
    # Training configuration
    num_iterations: int = 5
    learning_rate: float = 0.001
    optimizer: str = "Adam"
    loss_function: str = "MSE"
    
    # Paths
    cache_dir: Path = field(default_factory=lambda: Path("/tmp/e2e_cache"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("/tmp/e2e_checkpoints"))
    config_dir: Path = field(default_factory=lambda: Path("/tmp/e2e_config"))
    
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


class E2EDatasetWrapper(Dataset):
    """PyTorch Dataset wrapper for AsyncPipelineBridge.
    
    This wrapper provides a PyTorch-compatible interface to the async pipeline,
    handling scene splitting, image loading, and tensor conversion.
    """
    
    def __init__(
        self,
        bridge: AsyncPipelineBridge,
        split: str = "train",
        crop_size: int = 64,
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
        self.split = split
        self.crop_size = crop_size
        self.transform = transform
        
        # Split scenes based on configuration
        total_scenes = len(bridge) if bridge else 10
        
        if split == "train":
            self.start_idx = 0
            self.end_idx = 7
        elif split == "val":
            self.start_idx = 7
            self.end_idx = 9
        elif split == "test":
            self.start_idx = 9
            self.end_idx = 10
        else:
            raise ValueError(f"Unknown split: {split}")
        
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
            Dictionary with "input" (Bayer) and "target" (RGB) tensors
        """
        # Map split index to global scene index
        global_idx = self.start_idx + idx
        
        # Get scene from bridge
        scene = self.bridge[global_idx]
        
        # Convert scene to tensors (simplified for testing)
        input_tensor = self._scene_to_bayer_tensor(scene)
        target_tensor = self._scene_to_rgb_tensor(scene)
        
        # Apply random crop
        input_tensor, target_tensor = self._random_crop(
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
    
    def _scene_to_bayer_tensor(self, scene: SceneInfo) -> torch.Tensor:
        """Convert scene to Bayer tensor (4 channels: RGGB).
        
        This is a placeholder - actual implementation would load
        and process the raw Bayer data.
        """
        # For testing, create synthetic Bayer data
        h, w = 256, 256  # Full size before cropping
        bayer = torch.randn(4, h, w) * 0.1 + 0.5  # Centered around 0.5
        return bayer.clamp(0, 1)
    
    def _scene_to_rgb_tensor(self, scene: SceneInfo) -> torch.Tensor:
        """Convert scene to RGB tensor (3 channels).
        
        This is a placeholder - actual implementation would load
        the ground truth RGB data.
        """
        # For testing, create synthetic RGB data
        h, w = 256, 256  # Full size before cropping
        rgb = torch.randn(3, h, w) * 0.1 + 0.5  # Centered around 0.5
        return rgb.clamp(0, 1)
    
    def _random_crop(
        self,
        input_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
        crop_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random crop to input and target tensors.
        
        Args:
            input_tensor: Input tensor (C, H, W)
            target_tensor: Target tensor (C, H, W)
            crop_size: Size of square crop
            
        Returns:
            Cropped input and target tensors
        """
        _, h, w = input_tensor.shape
        
        # Random crop coordinates
        top = np.random.randint(0, h - crop_size + 1)
        left = np.random.randint(0, w - crop_size + 1)
        
        # Apply crop
        input_crop = input_tensor[
            :, top:top+crop_size, left:left+crop_size
        ]
        target_crop = target_tensor[
            :, top:top+crop_size, left:left+crop_size
        ]
        
        return input_crop, target_crop


def create_minimal_config_yaml(config: E2ETrainingConfig) -> Dict[str, Any]:
    """Create minimal YAML configuration for training.
    
    Args:
        config: E2E training configuration
        
    Returns:
        Dictionary suitable for YAML serialization
    """
    yaml_config = {
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
    
    return yaml_config


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
            
            if grad_norm == 0:
                logger.warning(f"Zero gradient in parameter: {name}")
            elif np.isnan(grad_norm):
                logger.error(f"NaN gradient in parameter: {name}")
            elif np.isinf(grad_norm):
                logger.error(f"Inf gradient in parameter: {name}")
    
    return gradient_norms


def track_training_metrics(
    loss: float,
    gradient_norms: Dict[str, float],
    learning_rate: float,
    batch_time: float,
    memory_mb: Optional[float] = None,
) -> Dict[str, float]:
    """Track and log training metrics.
    
    Args:
        loss: Training loss value
        gradient_norms: Dictionary of parameter gradient norms
        learning_rate: Current learning rate
        batch_time: Time taken for batch in seconds
        memory_mb: Memory usage in MB (optional)
        
    Returns:
        Dictionary of aggregated metrics
    """
    metrics = {
        "loss": loss,
        "avg_grad_norm": np.mean(list(gradient_norms.values())),
        "max_grad_norm": np.max(list(gradient_norms.values())),
        "learning_rate": learning_rate,
        "batch_time": batch_time,
    }
    
    if memory_mb is not None:
        metrics["memory_mb"] = memory_mb
    
    # Log metrics
    logger.info(
        f"Metrics: loss={metrics['loss']:.4f}, "
        f"avg_grad={metrics['avg_grad_norm']:.4f}, "
        f"max_grad={metrics['max_grad_norm']:.4f}, "
        f"lr={metrics['learning_rate']:.6f}, "
        f"time={metrics['batch_time']:.2f}s"
    )
    
    return metrics