"""
YAML Artifact Writer for Legacy Dataloader Compatibility.

Consumes enriched SceneInfo objects from MetadataEnricher and produces
YAML manifest files that legacy dataloaders (rawds.py) can consume.

This bridges the async pipeline → YAML descriptors → production dataset classes.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import trio
import yaml

from .PostDownloadWorker import PostDownloadWorker
from .SceneInfo import SceneInfo, ImageInfo
from .pipeline_decorators import stage

logger = logging.getLogger(__name__)


def scene_to_yaml_descriptor(scene: SceneInfo, dataset_root: Path) -> Dict[str, Any]:
    """
    Convert an enriched SceneInfo to YAML descriptor format.

    This creates the format expected by CleanProfiledRGBNoisyBayerImageCropsDataset
    and other production dataset classes in rawds.py.

    Args:
        scene: Enriched SceneInfo with metadata
        dataset_root: Root directory for dataset

    Returns:
        Dictionary in YAML descriptor format

    Raises:
        ValueError: If scene missing GT image or noisy images
        AttributeError: If required metadata missing
    """
    gt_img = scene.get_gt_image()
    if not gt_img:
        raise ValueError(f"Scene {scene.scene_name} missing GT image")

    if not scene.noisy_images:
        raise ValueError(f"Scene {scene.scene_name} has no noisy images")

    # Use first noisy image for metadata
    noisy_img = scene.noisy_images[0]

    # Extract enriched metadata with defaults
    metadata = noisy_img.metadata if noisy_img.metadata else {}

    # Build descriptor matching legacy format
    descriptor = {
        # Scene identification
        "scene_name": scene.scene_name,
        "image_set": scene.scene_name,  # Used for test/train splitting
        "is_bayer": scene.cfa_type == "Bayer",

        # File paths
        "f_fpath": str(noisy_img.local_path) if noisy_img.local_path else "",
        "f_bayer_fpath": str(noisy_img.local_path) if noisy_img.local_path else "",
        "gt_fpath": str(gt_img.local_path) if gt_img.local_path else "",
        "gt_linrec2020_fpath": str(gt_img.local_path) if gt_img.local_path else "",
        "gt_bayer_fpath": str(gt_img.local_path) if gt_img.local_path else "",  # For x_x pairing
        "f_linrec2020_fpath": str(noisy_img.local_path) if noisy_img.local_path else "",  # For y_y pairing

        # Alignment metadata (from enricher)
        "best_alignment": metadata.get("alignment", [0, 0]),
        "best_alignment_loss": metadata.get("alignment_loss", 0.0),

        # Gain correction (from enricher)
        "raw_gain": metadata.get("raw_gain", 1.0),
        "rgb_gain": metadata.get("rgb_gain", 1.0),

        # Mask metadata (from enricher)
        "mask_mean": metadata.get("mask_mean", 1.0),
        "mask_fpath": metadata.get("mask_fpath", ""),

        # Color space (from enricher)
        "rgb_xyz_matrix": metadata.get("rgb_xyz_matrix", [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]),

        # Overexposure threshold
        "overexposure_lb": metadata.get("overexposure_lb", 1.0),

        # Crops list (from enricher)
        "crops": metadata.get("crops", []),

        # Quality metrics (optional)
        "rgb_msssim_score": metadata.get("msssim_score", 1.0),
    }

    return descriptor


class YAMLArtifactWriter(PostDownloadWorker):
    """
    Pipeline stage that writes enriched scenes to YAML artifacts.

    Bridges new async pipeline with legacy PyTorch dataloaders by producing
    YAML files in the format expected by rawds.py dataset classes.

    This enables:
    - "Preprocess once, train many times" workflow
    - Separation of preprocessing (pipeline) from training (dataloaders)
    - Gradual migration from artifacts to real-time pipeline
    """

    def __init__(
        self,
        output_dir: Path,
        output_filename: str = "pipeline_output.yaml",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize YAML writer.

        Args:
            output_dir: Directory for YAML output
            output_filename: Name of YAML file to write
            config: Additional configuration
        """
        super().__init__(
            output_dir=output_dir,
            max_workers=1,  # Single-threaded writer
            use_process_pool=False,  # No need for process pool
            config=config
        )

        self.output_filename = output_filename
        self.yaml_path = self.output_dir / self.output_filename
        self.descriptors: List[Dict[str, Any]] = []

        logger.info(
            f"YAMLArtifactWriter initialized: output={self.yaml_path}"
        )

        # Support for visualizer attachment
        self._viz = None

    def attach_visualizer(self, viz):
        """Attach visualizer for automatic progress tracking."""
        self._viz = viz
        return self

    async def startup(self):
        """Initialize writer resources."""
        await super().startup()
        self.descriptors = []
        logger.info("YAMLArtifactWriter ready")

    @stage(progress=("yaml_writing", "yaml_written"), skip_on_error=True)
    async def process_scene(self, scene: SceneInfo) -> SceneInfo:
        """
        Process a single scene and buffer its descriptor.

        Args:
            scene: Enriched SceneInfo object

        Returns:
            The same SceneInfo for downstream stages (pass-through)
        """
        try:
            descriptor = scene_to_yaml_descriptor(scene, self.output_dir)
            self.descriptors.append(descriptor)

            if len(self.descriptors) % 10 == 0:
                logger.debug(f"Buffered {len(self.descriptors)} descriptors")

        except (ValueError, AttributeError, KeyError) as e:
            logger.warning(
                f"Failed to create descriptor for scene {scene.scene_name}: {e}"
            )

        # Pass through unchanged
        return scene

    async def consume_and_produce(
        self,
        input_channel: trio.MemoryReceiveChannel,
        output_channel: Optional[trio.MemorySendChannel] = None
    ):
        """
        Override to ensure shutdown is called after processing completes.

        Args:
            input_channel: Channel receiving enriched SceneInfo objects
            output_channel: Optional channel to forward processed scenes
        """
        try:
            await super().consume_and_produce(input_channel, output_channel)
        finally:
            # Always write YAML on completion
            await self.shutdown()

    async def shutdown(self):
        """Write all buffered descriptors to YAML file."""
        logger.info(f"Writing {len(self.descriptors)} descriptors to {self.yaml_path}")

        try:
            with open(self.yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(
                    self.descriptors,
                    f,
                    allow_unicode=True,
                    default_flow_style=False,
                    sort_keys=False
                )

            logger.info(f"Successfully wrote YAML to {self.yaml_path}")

        except Exception as e:
            logger.error(f"Failed to write YAML: {e}")
            raise

        await super().shutdown()