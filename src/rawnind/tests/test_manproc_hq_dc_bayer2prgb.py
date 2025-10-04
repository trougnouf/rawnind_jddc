"""
Run the same test procedure as in training.
Required argument : --config <path to training config file>.yaml
Launch with --debug_options output_valtest_images to output images.
"""

import sys

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
    if (
        "manproc_hq_msssim_loss.None" in denoiserTraining.json_saver.results["best_val"]
        or "manproc_hq_msssim_loss" in denoiserTraining.json_saver.results["best_val"]
        or "manproc_hq_msssim_loss.gamma22"
        in denoiserTraining.json_saver.results["best_val"]
    ):
        print("Skipping test, manproc_msssim_loss is known")
        sys.exit(0)
    dataset = rawds_manproc.ManuallyProcessedImageTestDataHandler(
        net_input_type="bayer", min_msssim_score=0.95
    )
    dataloader = dataset.batched_iterator()

    denoiserTraining.offline_custom_test(
        dataloader=dataloader, test_name="manproc_hq", save_individual_images=True
    )
