"""
Run the same test procedure as in training.
Required argument : --config <path to training config file>.yaml
Launch with --debug_options output_valtest_images to output images.
"""

import sys
import os

sys.path.append("..")

from rawnind.libs import rawds
from rawnind.libs import rawtestlib

MS_SSSIM_VALUES = {
    "le": {0.85, 0.9, 0.97, 0.99},
    "ge": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.00],
}

if __name__ == "__main__":
    preset_args = {"test_only": True, "init_step": None}
    if "--load_path" not in sys.argv:
        preset_args["load_path"] = None
    denoiserTraining = rawtestlib.DenoiseTestCustomDataloaderProfiledRGBToProfiledRGB(
        preset_args=preset_args
    )
    for operator, msssim_values in MS_SSSIM_VALUES.items():
        for msssim_value in msssim_values:
            if operator == "le":
                kwargs = {"max_msssim_score": msssim_value}
            elif operator == "ge":
                kwargs = {"min_msssim_score": msssim_value}
            dataloader = rawds.CleanProfiledRGBNoisyProfiledRGBImageCropsTestDataloader(
                content_fpaths=[
                    "../../datasets/RawNIND/RawNIND_masks_and_alignments.yaml"
                ],  # denoiserTraining.noise_dataset_yamlfpaths,
                crop_size=denoiserTraining.test_crop_size,
                test_reserve=denoiserTraining.test_reserve,
                bayer_only=True,
                match_gain="input",
                **kwargs,
            )
            # dataset = (
            #     rawds_ext_paired_test.CleanProfiledRGBNoisyBayerImageCropsExtTestDataloader(
            #         content_fpaths=[
            #             os.path.join(
            #                 "..",
            #                 "..",
            #                 "datasets",
            #                 "ext_raw_denoise_test",
            #                 "ext_raw_denoise_test_masks_and_alignments.yaml",
            #             )
            #         ]
            #     )
            # )
            # dataloader = dataset.batched_iterator()

            denoiserTraining.offline_custom_test(
                dataloader=dataloader,
                test_name=f"progressive_test_msssim_{operator}_{msssim_value}",
                save_individual_images=True,
            )
