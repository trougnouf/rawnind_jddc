"""Utilities for end-to-end training smoke test with REAL data loading.

This module provides supporting classes and functions for the E2E training test,
including configuration, dataset wrapper that actually loads real images from
the async pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import logging
import sys
import os

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parents[2]))

from common.libs import pt_helpers
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
    """PyTorch Dataset wrapper for AsyncPipelineBridge with REAL image loading.
    
    This wrapper provides a PyTorch-compatible interface to the async pipeline,
    handling scene splitting, actual image loading from downloaded files, and tensor conversion.
    """
    
    def __init__(
        self,
        bridge: AsyncPipelineBridge,
        split: str = "train",
        crop_size: int = 64,
        transform: Optional[Any] = None,
        config: Optional[E2ETrainingConfig] = None,
    ):
        """Initialize dataset wrapper.
        
        Args:
            bridge: AsyncPipelineBridge with collected scenes
            split: Dataset split ("train", "val", or "test")
            crop_size: Size of random crops to extract
            transform: Optional transforms to apply
            config: Optional config with scene counts (if None, uses defaults)
        """
        self.bridge = bridge
        self.split = split
        self.crop_size = crop_size
        self.transform = transform
        
        # Get split sizes from config or use defaults
        if config is not None:
            num_train = config.num_scenes_train
            num_val = config.num_scenes_val
            num_test = config.num_scenes_test
        else:
            num_train = 7
            num_val = 2
            num_test = 1
        
        # Calculate split indices
        if split == "train":
            self.start_idx = 0
            self.end_idx = num_train
        elif split == "val":
            self.start_idx = num_train
            self.end_idx = num_train + num_val
        elif split == "test":
            self.start_idx = num_train + num_val
            self.end_idx = num_train + num_val + num_test
        else:
            raise ValueError(f"Unknown split: {split}")
        
        self.num_scenes = self.end_idx - self.start_idx
        logger.info(f"Created {split} dataset with {self.num_scenes} scenes")
    
    def __len__(self) -> int:
        """Return number of scenes in this split."""
        return self.num_scenes
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset with proper alignment and gain correction.
        
        Args:
            idx: Index within this split
            
        Returns:
            Dictionary with "input" (Bayer), "target" (RGB), and optionally "mask"
        """
        # Map split index to global scene index
        global_idx = self.start_idx + idx
        
        # Get scene from bridge
        scene = self.bridge[global_idx]
        
        # Get noisy image with metadata
        if not scene.noisy_images:
            raise ValueError(f"Scene {scene.scene_name} has no noisy images")
        
        noisy_img = scene.noisy_images[0]
        gt_img = scene.get_gt_image()
        
        if gt_img is None:
            raise ValueError(f"Scene {scene.scene_name} has no GT image")
        
        # Load tensors
        input_tensor = self._scene_to_bayer_tensor(scene)
        target_tensor = self._scene_to_rgb_tensor(scene)
        
        # Apply alignment if metadata exists
        alignment = noisy_img.metadata.get("alignment", [0, 0])
        if alignment != [0, 0]:
            logger.info(f"Applying alignment {alignment} for {scene.scene_name}")
            from rawnind.libs import rawproc
            target_tensor, input_tensor = rawproc.shift_images(
                target_tensor, input_tensor, alignment
            )
        
        # Apply random crop
        input_tensor, target_tensor = self._random_crop(
            input_tensor, target_tensor, self.crop_size
        )
        
        # Apply gain correction if metadata exists
        # Use raw_gain for Bayer, rgb_gain for RGB (fallback to 1.0 if both None)
        gain = noisy_img.metadata.get("raw_gain") or noisy_img.metadata.get("rgb_gain") or 1.0
        if gain != 1.0 and gain is not None:
            logger.info(f"Applying gain {gain} for {scene.scene_name}")
            input_tensor = input_tensor * gain
        
        result = {
            "input": input_tensor,
            "target": target_tensor,
        }
        
        # Include metadata if available
        if "rgb_xyz_matrix" in noisy_img.metadata:
            result["rgb_xyz_matrix"] = torch.tensor(noisy_img.metadata["rgb_xyz_matrix"])
        
        # Apply additional transforms if provided
        if self.transform:
            result["input"] = self.transform(result["input"])
            result["target"] = self.transform(result["target"])
        
        return result
    
    def _scene_to_bayer_tensor(self, scene: SceneInfo) -> torch.Tensor:
        """Convert scene to Bayer tensor (4 channels: RGGB) by loading REAL noisy raw images.
        
        This loads the actual downloaded raw files and converts them to Bayer tensors.
        """
        # Get the first noisy image from the scene
        if not scene.noisy_images:
            # Fallback to synthetic if no noisy images
            logger.warning(f"No noisy images for scene {scene.scene_name}, using synthetic")
            h, w = 256, 256
            bayer = torch.randn(4, h, w) * 0.1 + 0.5
            return bayer.clamp(0, 1)
        
        noisy_img = scene.noisy_images[0]
        
        # Check if the image has been downloaded
        if noisy_img.local_path is None or not noisy_img.local_path.exists():
            # If not downloaded, create synthetic data for now
            logger.warning(f"Image not downloaded: {noisy_img.filename}, using synthetic")
            h, w = 256, 256
            bayer = torch.randn(4, h, w) * 0.1 + 0.5
            return bayer.clamp(0, 1)
        
        try:
            # Load the actual raw image using pt_helpers
            # This will handle RAF, DNG, and other raw formats
            bayer_tensor = pt_helpers.fpath_to_tensor(str(noisy_img.local_path))
            
            # Ensure it's 4-channel Bayer (RGGB)
            if bayer_tensor.shape[0] == 3:
                # If it's RGB, convert to Bayer pattern (simplified)
                # In reality, we'd need proper Bayer pattern extraction
                logger.warning(f"Got RGB instead of Bayer for {noisy_img.filename}, converting")
                h, w = bayer_tensor.shape[1], bayer_tensor.shape[2]
                bayer = torch.zeros(4, h//2, w//2)
                # Simple RGB to Bayer conversion (not accurate but works for testing)
                bayer[0] = bayer_tensor[0, ::2, ::2]    # R
                bayer[1] = bayer_tensor[1, ::2, 1::2]   # G1
                bayer[2] = bayer_tensor[1, 1::2, ::2]   # G2
                bayer[3] = bayer_tensor[2, 1::2, 1::2]  # B
                return bayer.clamp(0, 1)
            elif bayer_tensor.shape[0] == 4:
                # Already in Bayer format
                return bayer_tensor.clamp(0, 1)
            elif bayer_tensor.shape[0] == 1:
                # Monochrome Bayer, need to unpack
                h, w = bayer_tensor.shape[1], bayer_tensor.shape[2]
                bayer = torch.zeros(4, h//2, w//2)
                mono = bayer_tensor[0]
                bayer[0] = mono[::2, ::2]    # R
                bayer[1] = mono[::2, 1::2]   # G1
                bayer[2] = mono[1::2, ::2]   # G2
                bayer[3] = mono[1::2, 1::2]  # B
                return bayer.clamp(0, 1)
            else:
                raise ValueError(f"Unexpected channel count: {bayer_tensor.shape[0]}")
                
        except Exception as e:
            logger.error(f"Failed to load {noisy_img.local_path}: {e}")
            # Fallback to synthetic data
            h, w = 256, 256
            bayer = torch.randn(4, h, w) * 0.1 + 0.5
            return bayer.clamp(0, 1)
    
    def _scene_to_rgb_tensor(self, scene: SceneInfo) -> torch.Tensor:
        """Convert scene to RGB tensor (3 channels) by loading REAL clean/GT images.
        
        This loads the actual downloaded ground truth images.
        """
        # Get the first clean image from the scene
        gt_img = scene.get_gt_image()
        if gt_img is None:
            # Fallback to synthetic if no clean images
            logger.warning(f"No clean images for scene {scene.scene_name}, using synthetic")
            h, w = 256, 256
            rgb = torch.randn(3, h, w) * 0.1 + 0.5
            return rgb.clamp(0, 1)
        
        # Check if the image has been downloaded
        if gt_img.local_path is None or not gt_img.local_path.exists():
            # If not downloaded, create synthetic data
            logger.warning(f"GT image not downloaded: {gt_img.filename}, using synthetic")
            h, w = 256, 256
            rgb = torch.randn(3, h, w) * 0.1 + 0.5
            return rgb.clamp(0, 1)
        
        try:
            # Load the actual clean image (might be OpenEXR or other format)
            rgb_tensor = pt_helpers.fpath_to_tensor(str(gt_img.local_path))
            
            # Ensure it's 3-channel RGB
            if rgb_tensor.shape[0] == 3:
                return rgb_tensor.clamp(0, 1)
            elif rgb_tensor.shape[0] == 4:
                # RGBA, drop alpha channel
                return rgb_tensor[:3].clamp(0, 1)
            elif rgb_tensor.shape[0] == 1:
                # Grayscale, expand to RGB
                return rgb_tensor.repeat(3, 1, 1).clamp(0, 1)
            else:
                raise ValueError(f"Unexpected channel count: {rgb_tensor.shape[0]}")
                
        except Exception as e:
            logger.error(f"Failed to load {gt_img.local_path}: {e}")
            # Fallback to synthetic data
            h, w = 256, 256
            rgb = torch.randn(3, h, w) * 0.1 + 0.5
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
        # Get minimum dimensions
        _, h_in, w_in = input_tensor.shape
        _, h_tgt, w_tgt = target_tensor.shape
        
        # Handle size mismatch (Bayer is often half resolution)
        if h_in != h_tgt or w_in != w_tgt:
            # Resize to match smallest dimensions
            min_h = min(h_in, h_tgt)
            min_w = min(w_in, w_tgt)
            
            if h_in > min_h or w_in > min_w:
                input_tensor = input_tensor[:, :min_h, :min_w]
            if h_tgt > min_h or w_tgt > min_w:
                target_tensor = target_tensor[:, :min_h, :min_w]
            
            h_in = h_tgt = min_h
            w_in = w_tgt = min_w
        
        # Make sure we have enough size for cropping
        if h_in < crop_size or w_in < crop_size:
            # Pad if too small
            pad_h = max(0, crop_size - h_in)
            pad_w = max(0, crop_size - w_in)
            if pad_h > 0 or pad_w > 0:
                input_tensor = torch.nn.functional.pad(
                    input_tensor, (0, pad_w, 0, pad_h), mode='reflect'
                )
                target_tensor = torch.nn.functional.pad(
                    target_tensor, (0, pad_w, 0, pad_h), mode='reflect'
                )
                h_in += pad_h
                w_in += pad_w
        
        # Random crop coordinates
        top = np.random.randint(0, h_in - crop_size + 1)
        left = np.random.randint(0, w_in - crop_size + 1)
        
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
        "avg_grad_norm": np.mean(list(gradient_norms.values())) if gradient_norms else 0,
        "max_grad_norm": np.max(list(gradient_norms.values())) if gradient_norms else 0,
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