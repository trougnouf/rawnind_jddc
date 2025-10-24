"""
Tests for MetadataArtificer refactoring - alignment artifacts stored on ImageInfo.metadata.

Enhanced with trio.testing utilities for deterministic async testing.
"""

import trio.testing as tt
from pathlib import Path

from rawnind.dataset.SceneInfo import SceneInfo, ImageInfo
from rawnind.dataset.Aligner import MetadataArtificer

# ... existing code ...

@tt.trio_test
async def test_compute_alignments_stores_artifact_and_scene_view(tmp_path: Path):
    # Arrange: minimal scene with GT and one noisy image
    scene_name = "scene1"
    cfa_type = "Bayer"

    gt_img = ImageInfo(
        filename="gt.exr",
        sha1="gtsha1abcdef1234",
        is_clean=True,
        scene_name=scene_name,
        scene_images=[],
        cfa_type=cfa_type,
        local_path=tmp_path / cfa_type / scene_name / "gt.exr",
        validated=True,
    )
    noisy_img = ImageInfo(
        filename="noisy.arw",
        sha1="noisysha1deadbeef",
        is_clean=False,
        scene_name=scene_name,
        scene_images=[],
        cfa_type=cfa_type,
        local_path=tmp_path / cfa_type / scene_name / "noisy.arw",
        validated=True,
    )
    # Pre-populate metadata produced by enricher
    noisy_img.metadata.update({
        "alignment": [2, 4],
        "gain": 1.1,
        "mask_path": str(tmp_path / "masks" / "mask.png"),
        "alignment_loss": 0.01,
        "alignment_method": "fft_cfa",
        "mask_mean": 0.95,
    })

    scene = SceneInfo(
        scene_name=scene_name,
        cfa_type=cfa_type,
        unknown_sensor=False,
        test_reserve=False,
        clean_images=[gt_img],
        noisy_images=[noisy_img],
    )

    aligner = MetadataArtificer(output_dir=tmp_path / "alignment_artifacts")

    # Act: compute artifacts only (no disk writes)
    artifact = await aligner._compute_alignments(scene, gt_img, noisy_img)

    # Assert: artifact stored on ImageInfo and contains expected content
    assert "alignment_artifacts" in noisy_img.metadata, "artifact should be cached on ImageInfo.metadata"
    cached = noisy_img.metadata["alignment_artifacts"]
    assert cached is artifact
    assert cached["alignment"] == [2, 4]
    assert cached["gain"] == 1.1
    assert "pair_id" in cached and cached["pair_id"].startswith(f"{scene_name}_gtsha1ab")
    # No disk-output fields yet
    assert "metadata_path" not in cached and "mask_path" not in cached

    # Optional: scene-level compiled view (to be implemented in SceneInfo)
    # When implemented, uncomment:
    # compiled = scene.compiled_alignment_artifacts
    # assert isinstance(compiled, list) and compiled and compiled[0] is cached

# ... existing code ...

@tt.trio_test
async def test_process_scene_calls_compute_then_write(tmp_path: Path, monkeypatch):
    # Arrange
    scene_name = "scene2"
    cfa_type = "Bayer"
    gt_img = ImageInfo(
        filename="gt2.exr",
        sha1="gt2sha112345678",
        is_clean=True,
        scene_name=scene_name,
        scene_images=[],
        cfa_type=cfa_type,
        local_path=tmp_path / cfa_type / scene_name / "gt2.exr",
        validated=True,
    )
    noisy_img = ImageInfo(
        filename="noisy2.arw",
        sha1="noisy2sha1cafebabe",
        is_clean=False,
        scene_name=scene_name,
        scene_images=[],
        cfa_type=cfa_type,
        local_path=tmp_path / cfa_type / scene_name / "noisy2.arw",
        validated=True,
        metadata={"alignment": [0, 0], "gain": 1.0},
    )
    scene = SceneInfo(
        scene_name=scene_name,
        cfa_type=cfa_type,
        unknown_sensor=False,
        test_reserve=False,
        clean_images=[gt_img],
        noisy_images=[noisy_img],
    )

    aligner = MetadataArtificer(output_dir=tmp_path / "alignment_artifacts")

    # Patch compute to store artifact and write to augment with file paths
    async def fake_compute(self, sc, gt, nz):
        nz.metadata["alignment_artifacts"] = {
            "pair_id": f"{sc.scene_name}_{gt.sha1[:8]}_{nz.sha1[:8]}",
            "alignment": nz.metadata.get("alignment", [0, 0]),
            "gain": nz.metadata.get("gain", 1.0),
        }
        return nz.metadata["alignment_artifacts"]

    async def fake_write(self, sc, gt, nz):
        # Simulate IO results by mutating the cached artifact
        art = nz.metadata.get("alignment_artifacts", {})
        art["metadata_path"] = str(tmp_path / "metadata" / f"{art['pair_id']}_alignment.yaml")
        art["mask_path"] = str(tmp_path / "masks" / f"{art['pair_id']}_mask.png")
        return {"metadata_path": art["metadata_path"], "mask_path": art["mask_path"]}

    monkeypatch.setattr(MetadataArtificer, "_compute_alignments", fake_compute, raising=False)
    monkeypatch.setattr(MetadataArtificer, "_write_alignment_artifacts", fake_write, raising=False)

    # Act
    out_scene = await aligner.process_scene(scene)

    # Assert
    assert out_scene is scene
    assert "alignment_artifacts" in noisy_img.metadata
    art = noisy_img.metadata["alignment_artifacts"]
    assert "metadata_path" in art and art["metadata_path"].endswith("_alignment.yaml")
    assert "mask_path" in art and art["mask_path"].endswith("_mask.png")