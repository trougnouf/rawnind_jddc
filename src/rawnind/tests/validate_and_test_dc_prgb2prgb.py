import configargparse
import sys

sys.path.append("..")
from rawnind import train_dc_prgb2prgb

if __name__ == "__main__":
    preset_args = {"test_only": True, "init_step": None}
    if "--load_path" not in sys.argv:
        preset_args["load_path"] = None
    preset_args["noise_dataset_yamlfpaths"] = [
        "../../datasets/RawNIND/RawNIND_masks_and_alignments.yaml"
    ]
    denoiserTraining = train_dc_prgb2prgb.DCTrainingProfiledRGBToProfiledRGB(
        preset_args=preset_args
    )
    denoiserTraining.offline_validation()
    denoiserTraining.offline_std_test()
