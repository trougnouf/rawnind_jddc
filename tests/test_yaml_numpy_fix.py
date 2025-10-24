#!/usr/bin/env python3
"""Test that AsyncPipelineBridge correctly converts numpy arrays to lists for YAML serialization."""

import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "DocScan"))

from rawnind.dataset.AsyncPipelineBridge import AsyncPipelineBridge
from rawnind.dataset.SceneInfo import SceneInfo, ImageInfo


def test_numpy_to_yaml_conversion():
    """Test that numpy arrays in metadata are converted to lists before YAML serialization."""

    # Create mock SceneInfo with numpy arrays in metadata (simulating crop producer output)
    clean_img = ImageInfo(
        filename="clean_test.exr",
        sha1="mock_sha1_clean",
        is_clean=True,
        scene_name="test_scene",
        scene_images=["clean_test.exr", "noisy_test.dng"],
        cfa_type="bayer",
        file_id="mock_id_clean",
        local_path=Path("/tmp/clean_test.exr"),
    )

    noisy_img = ImageInfo(
        filename="noisy_test.dng",
        sha1="mock_sha1_noisy",
        is_clean=False,
        scene_name="test_scene",
        scene_images=["clean_test.exr", "noisy_test.dng"],
        cfa_type="bayer",
        file_id="mock_id_noisy",
        local_path=Path("/tmp/noisy_test.dng"),
        metadata={
            # These are numpy arrays as they would come from CropProducerStage.py
            "alignment": np.array([10, 20]),
            "alignment_loss": 0.05,
            "rgb_xyz_matrix": np.array(
                [[0.9, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.95]]
            ),
            "crops": [
                {
                    "crop_id": 0,
                    "position": np.array([100, 200]),  # Numpy array in crop dict
                    "size": 256,
                }
            ],
            "raw_gain": 1.5,
            "rgb_gain": 1.2,
        },
    )

    scene = SceneInfo(
        scene_name="test_scene",
        cfa_type="bayer",
        unknown_sensor=False,
        test_reserve=False,
        clean_images=[clean_img],
        noisy_images=[noisy_img],
    )

    # Create bridge and add scene
    bridge = AsyncPipelineBridge(max_scenes=1, enable_caching=False)
    bridge._scenes = [scene]  # Directly add scene to bypass async collection

    # Write YAML using the fixed method
    yaml_path = Path("/tmp/test_yaml_numpy_fix.yaml")
    print(f"Writing YAML to {yaml_path}...")
    bridge.write_yaml_compatible_cache(yaml_path, Path("/tmp"))

    print("✓ YAML written successfully")

    # Now try to load it with yaml.safe_load() (this would fail before the fix)
    print("Loading YAML with yaml.safe_load()...")
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    print(f"✓ Loaded {len(data)} scenes successfully with yaml.safe_load()")

    # Verify the data was converted correctly
    scene_data = data[0]
    print(f"\nVerifying data types:")
    print(
        f"  best_alignment: {scene_data['best_alignment']} (type: {type(scene_data['best_alignment'])})"
    )
    print(
        f"  rgb_xyz_matrix: {scene_data['rgb_xyz_matrix']} (type: {type(scene_data['rgb_xyz_matrix'])})"
    )
    print(
        f"  crops[0]['position']: {scene_data['crops'][0]['position']} (type: {type(scene_data['crops'][0]['position'])})"
    )

    # Assert they're lists, not numpy arrays or pickled objects
    assert isinstance(scene_data["best_alignment"], list), "alignment should be a list"
    assert isinstance(
        scene_data["rgb_xyz_matrix"], list
    ), "rgb_xyz_matrix should be a list"
    assert isinstance(
        scene_data["crops"][0]["position"], list
    ), "crop position should be a list"

    # Verify values match
    assert scene_data["best_alignment"] == [10, 20], "alignment values should match"
    assert len(scene_data["rgb_xyz_matrix"]) == 3, "rgb_xyz_matrix should have 3 rows"
    assert scene_data["crops"][0]["position"] == [
        100,
        200,
    ], "crop position should match"

    print("\n✅ All assertions passed! Numpy arrays were correctly converted to lists.")
    yaml_path.unlink()  # Cleanup


if __name__ == "__main__":
    test_numpy_to_yaml_conversion()
