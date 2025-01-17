"""
Run the same test procedure as in training.
Required argument : --config <path to training config file>.yaml
Launch with --debug_options output_valtest_images to output images.
"""

import sys
import os

sys.path.append("..")

from rawnind.libs import rawds_cleancleantest
from rawnind.libs import rawtestlib


if __name__ == "__main__":
    preset_args = {"test_only": True, "init_step": None}
    if "--load_path" not in sys.argv:
        preset_args["load_path"] = None
    denoiserTraining = rawtestlib.DenoiseTestCustomDataloaderBayerToProfiledRGB(
        preset_args=preset_args
    )
    if (
        "playraw_msssim_loss.None" in denoiserTraining.json_saver.results["best_val"]
        or "playraw_msssim_loss" in denoiserTraining.json_saver.results["best_val"]
    ):
        print(f"Skipping test, best_val is known")
        sys.exit(0)
    dataset = rawds_cleancleantest.CleanProfiledRGBCleanBayerImageCropsTestDataloader(
        content_fpaths=[
            os.path.join(
                "..",
                "..",
                "datasets",
                "extraraw",
                "play_raw_test",
                "crops_metadata.yaml",
            )
        ]
    )
    dataloader = dataset.batched_iterator()

    denoiserTraining.offline_custom_test(
        dataloader=dataloader, test_name="playraw", save_individual_images=True
    )
