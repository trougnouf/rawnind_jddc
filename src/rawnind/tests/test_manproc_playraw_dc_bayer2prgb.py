"""
Run the same test procedure as in training.
Required argument : --config <path to training config file>.yaml
Launch with --debug_options output_valtest_images to output images.
"""

import sys
import os

sys.path.append("..")

from rawnind.libs import rawds_manproc
from rawnind.libs import rawtestlib


if __name__ == "__main__":
    preset_args = {"test_only": True, "init_step": None}
    if "--load_path" not in sys.argv:
        preset_args["load_path"] = None
    denoiserTraining = rawtestlib.DCTestCustomDataloaderBayerToProfiledRGB(
        preset_args=preset_args
    )
    if any(
        akey in denoiserTraining.json_saver.results["best_val"]
        for akey in [
            "manproc_playraw_combined.None",
            "manproc_playraw_combined",
            "manproc_playraw_combined.gamma22",
        ]
    ):
        print(f"Skipping test, best_val is known")
        sys.exit(0)
    # dataset = rawds_cleancleantest.CleanProfiledRGBCleanBayerImageCropsTestDataloader(
    #     content_fpaths=[
    #         os.path.join(
    #             "..",
    #             "..",
    #             "datasets",
    #             "extraraw",
    #             "play_raw_test",
    #             "crops_metadata.yaml",
    #         )
    #     ]
    # )
    dataset = rawds_manproc.ManuallyProcessedImageTestDataHandler(
        net_input_type="bayer",
        test_descriptor_fpath="../../datasets/extraraw/play_raw_test/manproc_test_descriptor.yaml",
    )

    dataloader = dataset.batched_iterator()

    denoiserTraining.offline_custom_test(
        dataloader=dataloader, test_name="manproc_playraw", save_individual_images=True
    )
