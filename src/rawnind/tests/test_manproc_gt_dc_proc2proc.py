"""
Run the same test procedure as in training.
Required argument : --config <path to training config file>.yaml
Launch with --debug_options output_valtest_images to output images.
"""

import configargparse
import sys
import torch

sys.path.append("..")
from rawnind import train_dc_prgb2prgb
from rawnind.libs import abstract_trainer
from rawnind.libs import rawds_manproc


class DCTestCustomDataloaderProfiledRGBToProfiledRGB(
    train_dc_prgb2prgb.DCTrainingProfiledRGBToProfiledRGB
):
    def __init__(self, launch=False, **kwargs) -> None:
        super().__init__(launch=launch, **kwargs)

    def get_dataloaders(self) -> None:
        return None


if __name__ == "__main__":
    preset_args = {"test_only": True, "init_step": None}
    if "--load_path" not in sys.argv:
        preset_args["load_path"] = None
    denoiserTraining = DCTestCustomDataloaderProfiledRGBToProfiledRGB(
        preset_args=preset_args
    )
    if (
        "manproc_gt_msssim_loss.arbitraryproc"
        in denoiserTraining.json_saver.results["best_val"]
    ):
        print(f"Skipping test, manproc_msssim_loss is known")
        sys.exit(0)
    dataset = rawds_manproc.ManuallyProcessedImageTestDataHandler(
        net_input_type="proc", min_msssim_score=1.00
    )
    dataloader = dataset.batched_iterator()

    denoiserTraining.offline_custom_test(
        dataloader=dataloader, test_name="manproc_gt", save_individual_images=True
    )
