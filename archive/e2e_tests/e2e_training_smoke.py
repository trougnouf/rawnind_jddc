#!/usr/bin/env python3
"""End-to-end training smoke test executable.

This script runs a complete training loop using the async dataset pipeline,
with minimal data and a tiny model for fast execution and validation.

Usage:
    python e2e_training_smoke.py [--config CONFIG_PATH]
"""

import argparse
import logging
import sys
import time
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import trio
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    # Try absolute imports first (when run as module)
    from src.rawnind.dataset.async_to_sync_bridge import AsyncPipelineBridge
    from src.rawnind.dataset.e2e_training_utils import (
        E2ETrainingConfig,
        E2EDatasetWrapper,
        create_minimal_config_yaml,
        verify_gradient_flow,
        track_training_metrics,
    )
    from src.rawnind.models.raw_denoiser import UtNet2
except ImportError:
    # Fall back to relative imports (when run as script)
    from async_to_sync_bridge import AsyncPipelineBridge
    from e2e_training_utils import (
        E2ETrainingConfig,
        E2EDatasetWrapper,
        create_minimal_config_yaml,
        verify_gradient_flow,
    )
    sys.path.append(str(Path(__file__).parent.parent))
    from models.raw_denoiser import UtNet2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class E2ETrainingSmoke:
    """End-to-end training smoke test runner."""
    
    def __init__(self, config: E2ETrainingConfig):
        """Initialize training smoke test.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.bridge = None
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.train_loader = None
        self.val_loader = None
        
        # Create directories
        config.cache_dir.mkdir(parents=True, exist_ok=True)
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        self.save_config()
    
    def save_config(self):
        """Save configuration to YAML file."""
        yaml_path = self.config.config_dir / "e2e_smoke_config.yaml"
        yaml_content = create_minimal_config_yaml(self.config)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        logger.info(f"Saved configuration to {yaml_path}")
    
    async def setup_async_pipeline(self):
        """Set up async pipeline and collect scenes."""
        logger.info("Setting up async pipeline...")
        
        # For testing, use mock data
        # In production, this would use real RawNIND index URL
        self.bridge = AsyncPipelineBridge(
            index_url="mock://test",  # Will use mock implementation
            cache_dir=self.config.cache_dir,
            max_scenes=10,  # Total scenes: 7 train + 2 val + 1 test
            enable_metadata_enrichment=False,  # Disable for speed
            timeout=self.config.pipeline_timeout,
        )
        
        # Collect scenes asynchronously
        logger.info("Collecting scenes...")
        start_time = time.time()
        
        await self.bridge.collect_scenes_async()
        
        collection_time = time.time() - start_time
        logger.info(f"Collected {len(self.bridge)} scenes in {collection_time:.2f}s")
        
        if len(self.bridge) != 10:
            raise ValueError(f"Expected 10 scenes, got {len(self.bridge)}")
    
    def setup_datasets(self):
        """Set up training and validation datasets."""
        logger.info("Setting up datasets...")
        
        # Training dataset
        train_dataset = E2EDatasetWrapper(
            bridge=self.bridge,
            split="train",
            crop_size=self.config.crop_size_train,
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size_train,
            shuffle=True,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=self.config.pin_memory,
            drop_last=True,
        )
        
        # Validation dataset
        val_dataset = E2EDatasetWrapper(
            bridge=self.bridge,
            split="val",
            crop_size=self.config.crop_size_val,
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size_val,
            shuffle=False,
            num_workers=0,  # Single worker for validation
        )
        
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Val batches: {len(self.val_loader)}")
    
    def setup_model(self):
        """Set up model, optimizer, and loss function."""
        logger.info("Setting up model...")
        
        # Create model
        self.model = UtNet2(
            in_channels=self.config.model_params["input_channels"],
            funit=self.config.model_params["funit"],
        )
        
        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {num_params:,}")
        
        # Create optimizer
        if self.config.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
            )
        elif self.config.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        # Create loss function
        if self.config.loss_function == "MSE":
            self.loss_fn = nn.MSELoss()
        elif self.config.loss_function == "L1":
            self.loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss function: {self.config.loss_function}")
        
        logger.info(f"Using {self.config.optimizer} optimizer with lr={self.config.learning_rate}")
        logger.info(f"Using {self.config.loss_function} loss function")
    
    def train_iteration(self, iteration: int) -> float:
        """Run one training iteration.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            Average loss for the iteration
        """
        self.model.train()
        iteration_losses = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Forward pass
            output = self.model(batch["input"])
            loss = self.loss_fn(output, batch["target"])
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (optional)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Track loss
            iteration_losses.append(loss.item())
            
            # Log batch progress
            if batch_idx % 2 == 0:
                grad_norms = verify_gradient_flow(self.model)
                avg_grad = np.mean(list(grad_norms.values()))
                logger.info(
                    f"  Batch {batch_idx+1}/{len(self.train_loader)}: "
                    f"loss={loss.item():.4f}, avg_grad={avg_grad:.4f}"
                )
            
            # Limit batches per iteration for testing
            if batch_idx >= 2:  # Process 3 batches per iteration
                break
        
        return np.mean(iteration_losses)
    
    def validate(self) -> float:
        """Run validation loop.
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                output = self.model(batch["input"])
                loss = self.loss_fn(output, batch["target"])
                val_losses.append(loss.item())
        
        return np.mean(val_losses)
    
    def save_checkpoint(self, iteration: int, train_loss: float, val_loss: float):
        """Save training checkpoint.
        
        Args:
            iteration: Current iteration number
            train_loss: Training loss
            val_loss: Validation loss
        """
        checkpoint_path = self.config.checkpoint_dir / f"checkpoint_iter{iteration}.pt"
        
        checkpoint = {
            "iteration": iteration,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": self.config,
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def run_training(self):
        """Run the training loop."""
        logger.info("=" * 60)
        logger.info("Starting training loop...")
        logger.info("=" * 60)
        
        train_losses = []
        val_losses = []
        
        for iteration in range(self.config.num_iterations):
            # Training iteration
            start_time = time.time()
            train_loss = self.train_iteration(iteration)
            train_time = time.time() - start_time
            train_losses.append(train_loss)
            
            logger.info(
                f"Iteration {iteration+1}/{self.config.num_iterations}: "
                f"train_loss={train_loss:.4f}, time={train_time:.2f}s"
            )
            
            # Validation
            if (iteration + 1) % self.config.validate_every == 0:
                val_loss = self.validate()
                val_losses.append(val_loss)
                logger.info(f"Validation loss: {val_loss:.4f}")
            
            # Save checkpoint
            if (iteration + 1) % self.config.save_checkpoint_every == 0:
                self.save_checkpoint(
                    iteration + 1,
                    train_loss,
                    val_losses[-1] if val_losses else train_loss,
                )
        
        # Final validation
        final_val_loss = self.validate()
        val_losses.append(final_val_loss)
        
        # Summary statistics
        logger.info("=" * 60)
        logger.info("Training completed!")
        logger.info(f"Final train loss: {train_losses[-1]:.4f}")
        logger.info(f"Final val loss: {final_val_loss:.4f}")
        logger.info(f"Avg train loss: {np.mean(train_losses):.4f}")
        logger.info(f"Loss reduction: {train_losses[0]:.4f} -> {train_losses[-1]:.4f}")
        logger.info("=" * 60)
        
        # Verify training was successful
        if np.isnan(train_losses[-1]) or np.isinf(train_losses[-1]):
            raise ValueError("Training resulted in NaN or Inf loss")
        
        # Check if loss decreased (with tolerance)
        if train_losses[-1] > train_losses[0] * 1.5:
            logger.warning("Loss increased during training!")
        
        return train_losses, val_losses
    
    async def run(self):
        """Run the complete end-to-end training smoke test."""
        try:
            # Step 1: Set up async pipeline
            await self.setup_async_pipeline()
            
            # Step 2: Set up datasets
            self.setup_datasets()
            
            # Step 3: Set up model
            self.setup_model()
            
            # Step 4: Run training
            train_losses, val_losses = self.run_training()
            
            logger.info("E2E training smoke test completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"E2E training smoke test failed: {e}")
            raise


def main():
    """Main entry point."""
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
    
    args = parser.parse_args()
    
    # Create configuration
    if args.config and args.config.exists():
        # Load from file
        with open(args.config) as f:
            config_dict = yaml.safe_load(f)
        config = E2ETrainingConfig(**config_dict)
    else:
        # Use defaults with command-line overrides
        config = E2ETrainingConfig(
            cache_dir=args.cache_dir,
            num_iterations=args.iterations,
            batch_size_train=args.batch_size,
            learning_rate=args.learning_rate,
        )
    
    # Run training smoke test
    runner = E2ETrainingSmoke(config)
    
    # Use trio to run async portions
    success = trio.run(runner.run)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()