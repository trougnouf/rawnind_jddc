"""
Run the same test procedure as in training.
Required argument : --config <path to training config file>.yaml
Launch with --debug_options output_valtest_images to output images.
"""

import configargparse
import sys

sys.path.append("..")
from rawnind import train_dc_bayer2prgb

if __name__ == "__main__":
    preset_args = {"test_only": True, "init_step": None}
    if "--load_path" not in sys.argv:
        preset_args["load_path"] = None
    preset_args['noise_dataset_yamlfpaths'] = ['../../datasets/RawNIND/RawNIND_masks_and_alignments.yaml']
    denoiserTraining = train_dc_bayer2prgb.DCTrainingBayerToProfiledRGB(
        preset_args=preset_args
    )
    denoiserTraining.offline_validation()
    denoiserTraining.offline_std_test()
