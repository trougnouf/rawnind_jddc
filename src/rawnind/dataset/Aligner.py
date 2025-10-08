"""
Writes alignment masks (PNG) and metadata (YAML) to disk.
"""

import logging
import yaml
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import trio

from .PostDownloadWorker import PostDownloadWorker
from .SceneInfo import SceneInfo, ImageInfo
from .pipeline_decorators import stage

logger = logging.getLogger(__name__)


class Aligner(PostDownloadWorker):
    """Writes alignment masks and metadata to disk."""

    def __init__(
        self,
        output_dir: Path,
        write_masks: bool = True,
        write_metadata: bool = True,
        max_workers: int = 4,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the alignment artifact writer.

        Args:
            output_dir: Directory for saving artifacts
            write_masks: Whether to write mask PNG files
            write_metadata: Whether to write metadata YAML files
            max_workers: Maximum concurrent workers
            config: Additional configuration
        """
        super().__init__(output_dir, max_workers, use_process_pool=False, config=config)
        self.write_masks = write_masks
        self.write_metadata = write_metadata

        # Create subdirectories
        if self.write_masks:
            (self.output_dir / "masks").mkdir(parents=True, exist_ok=True)
        if self.write_metadata:
            (self.output_dir / "metadata").mkdir(parents=True, exist_ok=True)

        # Support for visualizer attachment
        self._viz = None

    def attach_visualizer(self, viz):
        """Attach visualizer for automatic progress tracking."""
        self._viz = viz
        return self

    @stage(progress=("aligning", "aligned"), skip_on_error=True)
    async def process_scene(self, scene: SceneInfo) -> SceneInfo:
        """
        Write alignment artifacts for the scene.

        Args:
            scene: Enriched SceneInfo with alignment metadata

        Returns:
            SceneInfo with artifact paths added to metadata
        """
        gt_img = scene.get_gt_image()
        if not gt_img or not gt_img.validated:
            logger.warning(f"Scene {scene.scene_name} has no valid GT image")
            return scene

        scene_artifacts = []

        # Process each noisy image
        for noisy_img in scene.noisy_images:
            if not noisy_img.validated:
                continue

            # Check for alignment metadata
            if "alignment" not in noisy_img.metadata:
                logger.debug(f"No alignment for {noisy_img.filename}")
                continue

            # Write artifacts for this pair
            artifacts = await self._write_alignment_artifacts(
                scene, gt_img, noisy_img
            )

            if artifacts:
                scene_artifacts.append(artifacts)
                # Add artifact paths to image metadata
                noisy_img.metadata["alignment_artifacts"] = artifacts

        # Write scene-level summary if we have artifacts
        if scene_artifacts and self.write_metadata:
            summary_path = await self._write_scene_summary(scene, scene_artifacts)
            scene.metadata["alignment_summary"] = str(summary_path)

        return scene

    async def _write_alignment_artifacts(
        self,
        scene: SceneInfo,
        gt_img: ImageInfo,
        noisy_img: ImageInfo
    ) -> Optional[Dict[str, Any]]:
        """
        Write alignment artifacts for an image pair.

        Args:
            scene: Scene information
            gt_img: Ground truth image
            noisy_img: Noisy image with alignment metadata

        Returns:
            Dictionary with artifact paths
        """
        artifacts = {}

        # Extract metadata
        alignment = noisy_img.metadata.get("alignment", [0, 0])
        gain = noisy_img.metadata.get("gain", 1.0)
        mask_path_from_metadata = noisy_img.metadata.get("mask_path")

        pair_id = f"{scene.scene_name}_{gt_img.sha1[:8]}_{noisy_img.sha1[:8]}"

        # Write mask if available
        if self.write_masks and mask_path_from_metadata:
            mask_dest = await self._write_mask_file(
                pair_id, Path(mask_path_from_metadata)
            )
            if mask_dest:
                artifacts["mask_path"] = str(mask_dest)

        # Write metadata YAML
        if self.write_metadata:
            metadata_path = await self._write_metadata_file(
                pair_id,
                scene.scene_name,
                gt_img,
                noisy_img,
                alignment,
                gain
            )
            artifacts["metadata_path"] = str(metadata_path)

        return artifacts if artifacts else None

    async def _write_mask_file(self, pair_id: str, source_mask_path: Path) -> Optional[Path]:
        """
        Copy mask file using async I/O.

        Args:
            pair_id: Identifier for the image pair
            source_mask_path: Path to existing mask file

        Returns:
            Path to written mask file
        """
        source = trio.Path(source_mask_path)
        if not await source.exists():
            logger.warning(f"Mask file not found: {source_mask_path}")
            return None

        mask_dir = self.output_dir / "masks"
        mask_dest = trio.Path(mask_dir) / f"{pair_id}_mask.png"

        try:
            content = await source.read_bytes()
            await mask_dest.write_bytes(content)
            return Path(mask_dest)

        except Exception as e:
            logger.error(f"Failed to write mask: {e}")
            return None

    async def _write_metadata_file(
        self,
        pair_id: str,
        scene_name: str,
        gt_img: ImageInfo,
        noisy_img: ImageInfo,
        alignment: list,
        gain: float
    ) -> Path:
        """
        Write alignment metadata to YAML file using async I/O.

        Args:
            pair_id: Identifier for the image pair
            scene_name: Name of the scene
            gt_img: Ground truth image info
            noisy_img: Noisy image info
            alignment: Alignment offset [y, x]
            gain: Gain correction factor

        Returns:
            Path to written metadata file
        """
        metadata_dir = self.output_dir / "metadata"
        metadata_path = trio.Path(metadata_dir) / f"{pair_id}_alignment.yaml"

        metadata = {
            "scene_name": scene_name,
            "pair_id": pair_id,
            "ground_truth": {
                "filename": gt_img.filename,
                "sha1": gt_img.sha1,
                "path": str(gt_img.local_path) if gt_img.local_path else None
            },
            "noisy": {
                "filename": noisy_img.filename,
                "sha1": noisy_img.sha1,
                "path": str(noisy_img.local_path) if noisy_img.local_path else None,
                "iso": noisy_img.metadata.get("iso"),
                "exposure_time": noisy_img.metadata.get("exposure_time")
            },
            "alignment": {
                "offset_y": alignment[0],
                "offset_x": alignment[1],
                "gain": gain,
                "method": noisy_img.metadata.get("alignment_method", "unknown")
            }
        }

        # Add any additional metadata
        if "mask_mean" in noisy_img.metadata:
            metadata["alignment"]["mask_mean"] = noisy_img.metadata["mask_mean"]

        # Serialize to YAML string, then write async
        yaml_str = yaml.dump(metadata, default_flow_style=False, sort_keys=False)
        await metadata_path.write_text(yaml_str)

        return Path(metadata_path)

    async def _write_scene_summary(
        self,
        scene: SceneInfo,
        artifacts: list
    ) -> Path:
        """
        Write scene-level summary of all alignments using async I/O.

        Args:
            scene: Scene information
            artifacts: List of artifact dictionaries

        Returns:
            Path to summary file
        """
        summary_path = trio.Path(self.output_dir / "metadata" / f"{scene.scene_name}_summary.yaml")

        summary = {
            "scene_name": scene.scene_name,
            "num_pairs": len(artifacts),
            "timestamp": trio.current_time(),
            "pairs": artifacts
        }

        # Serialize to YAML string, then write async
        yaml_str = yaml.dump(summary, default_flow_style=False, sort_keys=False)
        await summary_path.write_text(yaml_str)

        return Path(summary_path)