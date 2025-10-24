"""
Writes alignment masks (PNG) and metadata (YAML) to disk.

REFACTOR (2025-01-10):
Separated alignment artifact computation from disk I/O to enable:
- Artifacts cached on ImageInfo.metadata['alignment_artifacts'] before disk writes
- Disk I/O becomes optional/conditional (e.g., skip writes for high alignment_loss)
- Downstream consumers can access artifacts without waiting for disk writes
- SceneInfo.compiled_alignment_artifacts provides legacy-compatible artifact list

Key changes:
1. _compute_alignments(): assembles artifact dict, stores on ImageInfo, returns dict
2. process_scene(): calls compute → optional write (respects ALIGNMENT_TOLERANCE)
3. _write_alignment_artifacts(): reuses cached pair_id, returns only disk paths
4. SceneInfo gains metadata field and compiled_alignment_artifacts property
"""

import logging
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List, cast
import trio

from .PostDownloadWorker import PostDownloadWorker
from .SceneInfo import SceneInfo, ImageInfo
from .pipeline_decorators import stage

logger = logging.getLogger(__name__)

ALIGNMENT_TOLERANCE = 0.035

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
        self._viz: Any = None

    def attach_visualizer(self, viz: Any):
        """Attach visualizer for automatic progress tracking."""
        self._viz = viz
        return self

    async def _compute_alignments(
        self,
        scene: SceneInfo,
        gt_img: ImageInfo,
        noisy_img: ImageInfo,
    ) -> Dict[str, Any]:
        """
        Compute alignment artifact metadata independent of disk I/O and
        cache it on the ImageInfo for downstream consumers.
        """
        pair_id = f"{scene.scene_name}_{gt_img.sha1[:8]}_{noisy_img.sha1[:8]}"
        md = cast(Dict[str, Any], noisy_img.metadata)
        artifact: Dict[str, Any] = {
            "pair_id": pair_id,
            "alignment": md.get("alignment", [0, 0]),
            "gain": md.get("gain", 1.0),
        }
        # Pass-through non-I/O details if present (do not include paths here)
        if "alignment_loss" in md:
            artifact["alignment_loss"] = md["alignment_loss"]
        if "alignment_method" in md:
            artifact["alignment_method"] = md["alignment_method"]
        if "mask_mean" in md:
            artifact["mask_mean"] = md["mask_mean"]
        # Store on the image metadata (cached view for later write/consumers)
        noisy_img.metadata["alignment_artifacts"] = artifact
        return artifact

    @stage(progress=("aligning", "aligned"), debug_on_=True)
    async def process_scene(self, scene: SceneInfo) -> SceneInfo:
        """
        Write alignment artifacts for the scene.

        Args:
            scene: Enriched SceneInfo with alignment metadata

        Returns:
            SceneInfo with artifact paths added to metadata
        """
        logger.info(f"→ MetadataArtificer.process_scene: {scene.scene_name}")
        gt_img = scene.get_gt_image()
        if not gt_img:
            logger.warning(f"Scene {scene.scene_name} has no valid GT image")
            return scene

        scene_artifacts: List[Dict[str, Any]] = []

        # Process each noisy image
        for noisy_img in scene.noisy_images:
            # Step 1: compute in-memory artifact and cache it on the image
            artifact = await self._compute_alignments(scene, gt_img, noisy_img)
            # Optional: honor loss tolerance before doing any I/O
            loss_val = cast(Dict[str, Any], noisy_img.metadata).get("alignment_loss", None)
            if isinstance(loss_val, (int, float)) and loss_val > ALIGNMENT_TOLERANCE:
                logger.debug(
                    f"Skipping write for {noisy_img.filename} due to alignment_loss={loss_val:.4f} > {ALIGNMENT_TOLERANCE}"
                )
                scene_artifacts.append(dict(artifact))
                continue
            # Step 2: write disk artifacts and merge paths back into cached artifact
            write_results = await self._write_alignment_artifacts(scene, gt_img, noisy_img)
            if not write_results:
                logger.warning(f"Scene {scene.scene_name} has no artifacts for {noisy_img.filename}")
            else:
                for k, v in write_results.items():
                    if v is not None:
                        artifact[k] = v
                noisy_img.metadata["alignment_artifacts"] = artifact
                scene_artifacts.append(dict(artifact))

        # Write scene-level summary if we have artifacts
        if scene_artifacts and self.write_metadata:
            summary_path = await self._write_scene_summary(scene, scene_artifacts)
            scene.metadata["alignment_summary"] = str(summary_path)

        logger.info(f"← MetadataArtificer.process_scene: {scene.scene_name} ({len(scene_artifacts)} artifacts)")
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
        artifacts: Dict[str, Any] = {}

        # Extract metadata
        md = cast(Dict[str, Any], noisy_img.metadata)
        alignment = md.get("alignment", [0, 0])
        gain = md.get("gain", 1.0)
        mask_path_from_metadata = md.get("mask_path")

        # Prefer cached pair_id from compute step, fallback to recompute
        cached_art = md.get("alignment_artifacts", {}) if isinstance(md.get("alignment_artifacts"), dict) else {}
        pair_id = cast(Dict[str, Any], cached_art).get(
            "pair_id", f"{scene.scene_name}_{gt_img.sha1[:8]}_{noisy_img.sha1[:8]}"
        )

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