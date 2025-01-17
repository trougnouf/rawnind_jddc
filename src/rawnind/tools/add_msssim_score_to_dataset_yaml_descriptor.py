import sys
import os
import yaml
import tqdm
import torch
import argparse

sys.path.append("..")
from common.libs import utilities
from common.libs import pt_losses
from common.libs import pt_helpers
from rawnind.libs import rawproc
from rawnind.libs import raw


DATASET_YAML_DESCRIPTOR_FPATH = os.path.join(
    "..", "..", "datasets", "RawNIND", "RawNIND_masks_and_alignments.yaml"
)


def add_msssim_to_dataset_descriptor(
    dataset_descriptor_fpath: str = DATASET_YAML_DESCRIPTOR_FPATH,
):
    dataset = utilities.load_yaml(dataset_descriptor_fpath, error_on_404=True)
    for image in tqdm.tqdm(dataset):
        try:
            if image["is_bayer"]:
                gt_img, gt_metadata = raw.raw_fpath_to_mono_img_and_metadata(
                    image["gt_fpath"]
                )
                gt_img = raw.demosaic(gt_img, gt_metadata)
                gt_img = raw.camRGB_to_profiledRGB_img(
                    gt_img, gt_metadata, output_color_profile=raw.OUTPUT_COLOR_PROFILE
                )
                gt_img = torch.from_numpy(gt_img).unsqueeze(0)
                f_img, f_metadata = raw.raw_fpath_to_mono_img_and_metadata(
                    image["f_fpath"]
                )
                f_img = raw.demosaic(f_img, f_metadata)
                f_img = raw.camRGB_to_profiledRGB_img(
                    f_img, f_metadata, output_color_profile=raw.OUTPUT_COLOR_PROFILE
                )
                f_img = torch.from_numpy(f_img).unsqueeze(0)
            else:
                # load images
                f_img = pt_helpers.fpath_to_tensor(
                    image["f_fpath"], incl_metadata=False
                ).unsqueeze(0)
                gt_img = pt_helpers.fpath_to_tensor(
                    image["gt_fpath"], incl_metadata=False
                ).unsqueeze(0)
            # align images
            gt_img, f_img = rawproc.shift_images(gt_img, f_img, image["best_alignment"])
            # match exposures
            f_img *= image["rgb_gain"]
            # compute msssim loss
            score = pt_losses.MS_SSIM_metric()(f_img, gt_img)
            image["rgb_msssim_score"] = score.item()
        except ValueError as e:
            print(e)
            print("Error while processing image {}".format(image["gt_fpath"]))
            image["rgb_msssim_score"] = None

    # save dataset
    with open(dataset_descriptor_fpath, "w") as file:
        yaml.dump(dataset, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset_descriptor_fpath",
        type=str,
        default=DATASET_YAML_DESCRIPTOR_FPATH,
        help="Path to the dataset descriptor yaml file",
    )
    args = parser.parse_args()
    add_msssim_to_dataset_descriptor(args.dataset_descriptor_fpath)
