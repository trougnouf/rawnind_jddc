#!/usr/bin/env python3
"""Refactored end-to-end training smoke test executable.

This script runs a complete training loop using the async dataset pipeline,
with improved code quality following SOLID principles.
"""

import argparse
import logging
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import trio
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    # Try absolute imports first (when run as module)
    from src.rawnind.dataset.async_to_sync_bridge import AsyncPipelineBridge
    from src.rawnind.dataset.e2e_training_utils_refactored import (
        E2ETrainingConfig,
        E2EDatasetWrapper,
        ConfigGenerator,
        MetricsTracker,
        TensorOperations,
        DatasetSplit,
        OptimizerType,
        LossType,
        MAX_GRADIENT_NORM,
        LOSS_INCREASE_TOLERANCE,
    )
    from src.rawnind.models.raw_denoiser import UtNet2
except ImportError:
    # Fall back to relative imports (when run as script)
    from async_to_sync_bridge import AsyncPipelineBridge
    from e2e_training_utils_refactored import (
        E2ETrainingConfig,
        E2EDatasetWrapper,
        ConfigGenerator,
        MetricsTracker,
        TensorOperations,
        DatasetSplit,
        OptimizerType,
        LossType,
        MAX_GRADIENT_NORM,
        LOSS_INCREASE_TOLERANCE,
    )
    sys.path.append(str(Path(__file__).parent.parent))
    from models.raw_denoiser import UtNet2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ==================== Abstract Base Classes ====================
class ModelFactory(ABC):
    """Abstract factory for creating models."""

    @abstractmethod
    def create_model(self, config: E2ETrainingConfig) -> nn.Module:
        """Create a model based on configuration."""
        pass


class OptimizerFactory(ABC):
    """Abstract factory for creating optimizers."""

    @abstractmethod
    def create_optimizer(self, model: nn.Module, config: E2ETrainingConfig) -> torch.optim.Optimizer:
        """Create an optimizer based on configuration."""
        pass


class LossFactory(ABC):
    """Abstract factory for creating loss functions."""

    @abstractmethod
    def create_loss(self, config: E2ETrainingConfig) -> nn.Module:
        """Create a loss function based on configuration."""
        pass


# ==================== Concrete Factories ====================
class UtNetModelFactory(ModelFactory):
    """Factory for creating UtNet models."""

    def create_model(self, config: E2ETrainingConfig) -> nn.Module:
        """Create a UtNet2 model.

        Args:
            config: Training configuration

        Returns:
            UtNet2 model instance
        """
        model = UtNet2(
            in_channels=config.model_params["input_channels"],
            funit=config.model_params["funit"],
        )

        # Log model information
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Created {config.model_type} with {num_params:,} parameters")

        return model


class StandardOptimizerFactory(OptimizerFactory):
    """Factory for creating standard PyTorch optimizers."""

    def create_optimizer(self, model: nn.Module, config: E2ETrainingConfig) -> torch.optim.Optimizer:
        """Create an optimizer based on configuration.

        Args:
            model: PyTorch model
            config: Training configuration

        Returns:
            Configured optimizer

        Raises:
            ValueError: If optimizer type is not supported
        """
        optimizer_type = OptimizerType(config.optimizer)

        if optimizer_type == OptimizerType.ADAM:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
            )
        elif optimizer_type == OptimizerType.SGD:
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

        logger.info(f"Using {optimizer_type.value} optimizer with lr={config.learning_rate}")
        return optimizer


class StandardLossFactory(LossFactory):
    """Factory for creating standard loss functions."""

    def create_loss(self, config: E2ETrainingConfig) -> nn.Module:
        """Create a loss function based on configuration.

        Args:
            config: Training configuration

        Returns:
            Loss function module

        Raises:
            ValueError: If loss type is not supported
        """
        loss_type = LossType(config.loss_function)

        if loss_type == LossType.MSE:
            loss_fn = nn.MSELoss()
        elif loss_type == LossType.L1:
            loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_type}")

        logger.info(f"Using {loss_type.value} loss function")
        return loss_fn


# ==================== Training Components ====================
class CheckpointManager:
    """Manages checkpoint saving and loading."""

    def __init__(self, checkpoint_dir: Path):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save training checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer state to save
            iteration: Current iteration number
            train_loss: Current training loss
            val_loss: Optional validation loss
            metadata: Optional additional metadata

        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_iter{iteration}.pt"

        checkpoint = {
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "timestamp": time.time(),
        }

        if metadata:
            checkpoint.update(metadata)

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """Load checkpoint from file.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to load state into

        Returns:
            Checkpoint metadata dictionary
        """
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint


class TrainingLoop:
    """Encapsulates training loop logic."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        config: E2ETrainingConfig,
    ):
        """Initialize training loop.

        Args:
            model: Model to train
            optimizer: Optimizer to use
            loss_fn: Loss function
            config: Training configuration
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config
        self.metrics_tracker = MetricsTracker()

    def train_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train on a single batch.

        Args:
            batch: Batch data dictionary

        Returns:
            Dictionary of batch metrics
        """
        start_time = time.time()

        # Forward pass
        output = self.model(batch["input"])

        # Handle size mismatch if model upsamples
        target = self._match_target_size(batch["target"], output)

        # Compute loss
        loss = self.loss_fn(output, target)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=MAX_GRADIENT_NORM)

        # Optimizer step
        self.optimizer.step()

        # Compute metrics
        batch_time = time.time() - start_time
        gradient_norms = self.metrics_tracker.verify_gradient_flow(self.model)

        metrics = self.metrics_tracker.compute_metrics(
            loss=loss.item(),
            gradient_norms=gradient_norms,
            learning_rate=self.config.learning_rate,
            batch_time=batch_time,
        )

        return metrics

    def validate_batch(self, batch: Dict[str, torch.Tensor]) -> float:
        """Validate on a single batch.

        Args:
            batch: Batch data dictionary

        Returns:
            Validation loss
        """
        with torch.no_grad():
            output = self.model(batch["input"])
            target = self._match_target_size(batch["target"], output)
            loss = self.loss_fn(output, target)
            return loss.item()

    def _match_target_size(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Resize target to match output size if needed.

        Args:
            target: Target tensor
            output: Output tensor

        Returns:
            Resized target tensor
        """
        if target.shape[-2:] != output.shape[-2:]:
            return TensorOperations.resize_to_match(
                target,
                output.shape[-2:],
                mode='bilinear'
            )
        return target


class DataLoaderManager:
    """Manages creation and configuration of data loaders."""

    def __init__(self, bridge: AsyncPipelineBridge, config: E2ETrainingConfig):
        """Initialize data loader manager.

        Args:
            bridge: AsyncPipelineBridge with scenes
            config: Training configuration
        """
        self.bridge = bridge
        self.config = config

    def create_train_loader(self) -> DataLoader:
        """Create training data loader.

        Returns:
            Configured training DataLoader
        """
        dataset = E2EDatasetWrapper(
            bridge=self.bridge,
            split=DatasetSplit.TRAIN.value,
            crop_size=self.config.crop_size_train,
        )

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size_train,
            shuffle=True,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=self.config.pin_memory,
            drop_last=True,
        )

    def create_val_loader(self) -> DataLoader:
        """Create validation data loader.

        Returns:
            Configured validation DataLoader
        """
        dataset = E2EDatasetWrapper(
            bridge=self.bridge,
            split=DatasetSplit.VAL.value,
            crop_size=self.config.crop_size_val,
        )

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size_val,
            shuffle=False,
            num_workers=0,  # Single worker for validation
        )


# ==================== Main Training Runner ====================
class E2ETrainingSmoke:
    """Refactored end-to-end training smoke test runner."""

    def __init__(self, config: E2ETrainingConfig):
        """Initialize training smoke test.

        Args:
            config: Training configuration
        """
        self.config = config

        # Initialize factories
        self.model_factory = UtNetModelFactory()
        self.optimizer_factory = StandardOptimizerFactory()
        self.loss_factory = StandardLossFactory()

        # Initialize managers
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir)

        # Components to be initialized
        self.bridge: Optional[AsyncPipelineBridge] = None
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.loss_fn: Optional[nn.Module] = None
        self.training_loop: Optional[TrainingLoop] = None
        self.dataloader_manager: Optional[DataLoaderManager] = None

        # Setup directories and save config
        self._setup_directories()
        self._save_config()

    def _setup_directories(self):
        """Create necessary directories."""
        for directory in [self.config.cache_dir, self.config.checkpoint_dir, self.config.config_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def _save_config(self):
        """Save configuration to YAML file."""
        yaml_path = self.config.config_dir / "e2e_smoke_config.yaml"
        yaml_content = ConfigGenerator.create_yaml_config(self.config)

        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

        logger.info(f"Saved configuration to {yaml_path}")

    async def setup_async_pipeline(self):
        """Set up async pipeline and collect scenes."""
        logger.info("Setting up async pipeline...")

        self.bridge = AsyncPipelineBridge(
            index_url="mock://test",
            cache_dir=self.config.cache_dir,
            max_scenes=self.config.total_scenes,
            enable_metadata_enrichment=False,
            timeout=self.config.pipeline_timeout,
        )

        # Collect scenes
        logger.info("Collecting scenes...")
        start_time = time.time()

        await self.bridge.collect_scenes_async()

        collection_time = time.time() - start_time
        logger.info(f"Collected {len(self.bridge)} scenes in {collection_time:.2f}s")

        # Validate scene count
        if len(self.bridge) != self.config.total_scenes:
            raise ValueError(
                f"Expected {self.config.total_scenes} scenes, got {len(self.bridge)}"
            )

    def setup_model_and_training(self):
        """Set up model, optimizer, and training components."""
        # Create model
        self.model = self.model_factory.create_model(self.config)

        # Create optimizer
        self.optimizer = self.optimizer_factory.create_optimizer(self.model, self.config)

        # Create loss function
        self.loss_fn = self.loss_factory.create_loss(self.config)

        # Initialize training loop
        self.training_loop = TrainingLoop(
            self.model,
            self.optimizer,
            self.loss_fn,
            self.config,
        )

        # Initialize data loader manager
        self.dataloader_manager = DataLoaderManager(self.bridge, self.config)

    def run_training(self) -> Tuple[List[float], List[float]]:
        """Run the complete training loop.

        Returns:
            Tuple of (train_losses, val_losses)
        """
        logger.info("=" * 60)
        logger.info("Starting training loop...")
        logger.info("=" * 60)

        # Create data loaders
        train_loader = self.dataloader_manager.create_train_loader()
        val_loader = self.dataloader_manager.create_val_loader()

        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")

        # Initialize metrics
        train_losses = []
        val_losses = []

        # Training loop
        for iteration in range(self.config.num_iterations):
            # Training phase
            self.model.train()
            iteration_losses = []

            for batch_idx, batch in enumerate(train_loader):
                metrics = self.training_loop.train_batch(batch)
                iteration_losses.append(metrics["loss"])

                # Log batch progress
                if batch_idx % 2 == 0:
                    MetricsTracker.log_metrics(metrics, iteration=iteration+1)

                # Limit batches for testing
                if batch_idx >= 2:
                    break

            # Record iteration metrics
            avg_train_loss = np.mean(iteration_losses)
            train_losses.append(avg_train_loss)

            logger.info(
                f"Iteration {iteration+1}/{self.config.num_iterations}: "
                f"train_loss={avg_train_loss:.4f}"
            )

            # Validation phase
            if (iteration + 1) % self.config.validate_every == 0:
                val_loss = self._run_validation(val_loader)
                val_losses.append(val_loss)
                logger.info(f"Validation loss: {val_loss:.4f}")

            # Save checkpoint
            if (iteration + 1) % self.config.save_checkpoint_every == 0:
                self.checkpoint_manager.save_checkpoint(
                    self.model,
                    self.optimizer,
                    iteration + 1,
                    avg_train_loss,
                    val_losses[-1] if val_losses else None,
                )

        # Final validation
        final_val_loss = self._run_validation(val_loader)
        val_losses.append(final_val_loss)

        # Log summary
        self._log_training_summary(train_losses, val_losses, final_val_loss)

        # Validate results
        self._validate_training_results(train_losses)

        return train_losses, val_losses

    def _run_validation(self, val_loader: DataLoader) -> float:
        """Run validation loop.

        Args:
            val_loader: Validation data loader

        Returns:
            Average validation loss
        """
        self.model.eval()
        val_losses = []

        for batch in val_loader:
            loss = self.training_loop.validate_batch(batch)
            val_losses.append(loss)

        return np.mean(val_losses)

    def _log_training_summary(
        self,
        train_losses: List[float],
        val_losses: List[float],
        final_val_loss: float,
    ):
        """Log training summary statistics.

        Args:
            train_losses: List of training losses
            val_losses: List of validation losses
            final_val_loss: Final validation loss
        """
        logger.info("=" * 60)
        logger.info("Training completed!")
        logger.info(f"Final train loss: {train_losses[-1]:.4f}")
        logger.info(f"Final val loss: {final_val_loss:.4f}")
        logger.info(f"Avg train loss: {np.mean(train_losses):.4f}")
        logger.info(f"Loss reduction: {train_losses[0]:.4f} -> {train_losses[-1]:.4f}")
        logger.info("=" * 60)

    def _validate_training_results(self, train_losses: List[float]):
        """Validate that training was successful.

        Args:
            train_losses: List of training losses

        Raises:
            ValueError: If training resulted in invalid losses
        """
        if np.isnan(train_losses[-1]) or np.isinf(train_losses[-1]):
            raise ValueError("Training resulted in NaN or Inf loss")

        if train_losses[-1] > train_losses[0] * LOSS_INCREASE_TOLERANCE:
            logger.warning("Loss increased significantly during training!")

    async def run(self) -> bool:
        """Run the complete end-to-end training smoke test.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Step 1: Set up async pipeline
            await self.setup_async_pipeline()

            # Step 2: Set up model and training components
            self.setup_model_and_training()

            # Step 3: Run training
            train_losses, val_losses = self.run_training()

            logger.info("E2E training smoke test completed successfully!")
            return True

        except Exception as e:
            logger.error(f"E2E training smoke test failed: {e}")
            raise


# ==================== CLI Entry Point ====================
def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="End-to-end training smoke test"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file (optional)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/tmp/e2e_cache"),
        help="Cache directory for dataset",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate",
    )

    return parser.parse_args()


def load_or_create_config(args: argparse.Namespace) -> E2ETrainingConfig:
    """Load configuration from file or create from arguments.

    Args:
        args: Command-line arguments

    Returns:
        Training configuration
    """
    if args.config and args.config.exists():
        # Load from file
        with open(args.config) as f:
            config_dict = yaml.safe_load(f)
        return E2ETrainingConfig(**config_dict)
    else:
        # Use defaults with command-line overrides
        return E2ETrainingConfig(
            cache_dir=args.cache_dir,
            num_iterations=args.iterations,
            batch_size_train=args.batch_size,
            learning_rate=args.learning_rate,
        )


def main():
    """Main entry point."""
    args = parse_arguments()
    config = load_or_create_config(args)

    # Run training smoke test
    runner = E2ETrainingSmoke(config)
    success = trio.run(runner.run)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()