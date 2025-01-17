"""
Run the same test procedure as in training.
Required argument : --config <path to training config file>.yaml
Launch with --debug_options output_valtest_images to output images.
"""

import sys
import os

sys.path.append("..")

from rawnind.libs import rawds_ext_paired_test
from rawnind.libs import rawtestlib


if __name__ == "__main__":
    preset_args = {"test_only": True, "init_step": None}
    if "--load_path" not in sys.argv:
        preset_args["load_path"] = None
    denoiserTraining = rawtestlib.DenoiseTestCustomDataloaderBayerToProfiledRGB(
        preset_args=preset_args
    )
    dataset = (
        rawds_ext_paired_test.CleanProfiledRGBNoisyBayerImageCropsExtTestDataloader(
            content_fpaths=[
                os.path.join(
                    "..",
                    "..",
                    "datasets",
                    "ext_raw_denoise_test",
                    "ext_raw_denoise_test_masks_and_alignments.yaml",
                )
            ]
        )
    )
    dataloader = dataset.batched_iterator()

    denoiserTraining.offline_custom_test(
        dataloader=dataloader,
        test_name="ext_raw_denoise_test",
        save_individual_images=True,
    )
