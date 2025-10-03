import os
import sys
from typing import Literal, Optional
import torch
import tqdm
import yaml
import argparse

sys.path.append("..")
from rawnind.libs import rawds
from rawnind.libs import rawproc
from rawnind.libs import raw
from rawnind.tools import make_hdr_rawnind_files, make_hdr_extraraw_files
from common.libs import utilities
from common.libs import pt_helpers
from common.libs import pt_ops

TEST_DESCRIPTOR_FPATH = os.path.join(
    rawproc.DS_BASE_DPATH, "manproc_test_descriptor.yaml"
)
# MANPROC_DS_DPATH = os.path.join(rawproc.DS_BASE_DPATH, "proc", "dt")
TMPDIR = f"tmp_{os.uname()[1]}"
REMOVE_TMP_IMAGES = False
"""
"""


# class ManuallyProcessedImageDataloader(rawds.TestDataLoader):
#     def __init__(self, test_descriptor_fpath=TEST_DESCRIPTOR_FPATH):
#         super().__init__()
#         self._dataset = utilities.load_yaml(test_descriptor_fpath)

#     def __len__(self):
#         return len(self._dataset)


class ManuallyProcessedImageTestDataHandler(rawds.TestDataLoader):
    OUTPUTS_IMAGE_FILES = True

    def __init__(
        self,
        # net_output_proc_dpath: str,
        net_input_type: Literal["bayer", "lin_rec2020", "proc"],
        test_descriptor_fpath=TEST_DESCRIPTOR_FPATH,
        always_output_processed_net_output: bool = True,
        min_msssim_score: Optional[float] = 0.0,
        max_msssim_score: Optional[float] = 1.0,
    ):
        # super().__init__(test_descriptor_fpath=test_descriptor_fpath)
        self._dataset = utilities.load_yaml(test_descriptor_fpath, error_on_404=True)
        # check for msssim scores
        if min_msssim_score is not None or max_msssim_score is not None:
            new_dataset = []
            for image in self._dataset:
                if "rgb_msssim_score" not in image:
                    raise ValueError(
                        f"ManuallyProcessedImageTestDataHandler: {image['f_fpath']} does not have an MSSSIM score"
                    )
                if (
                    min_msssim_score is None
                    or min_msssim_score <= image["rgb_msssim_score"]
                ) and (
                    max_msssim_score is None
                    or image["rgb_msssim_score"] <= max_msssim_score
                ):
                    new_dataset.append(image)
            self._dataset = new_dataset

        self.net_input_type = net_input_type
        # self.net_output_proc_dpath = net_output_proc_dpath
        self.always_output_processed_net_output = always_output_processed_net_output

    @staticmethod
    def process_lin_rec2020_img(
        linrec_img, output_fpath: str, xmp_fpath: str, src_fpath: str
    ) -> None:
        assert os.path.isfile(xmp_fpath), f"{xmp_fpath} does not exist"
        # make paths
        os.makedirs(os.path.dirname(output_fpath), exist_ok=True)
        os.makedirs(TMPDIR, exist_ok=True)
        tmp_fpath = os.path.join(TMPDIR, os.path.basename(output_fpath))
        # make tmp image
        raw.hdr_nparray_to_file(
            linrec_img,
            tmp_fpath,
            color_profile="lin_rec2020",
            bit_depth=16,
            src_fpath=src_fpath,
        )
        # remove dest_fpath if it exists (always overwrite)
        if os.path.isfile(output_fpath):
            print(
                f"ManuallyProcessedImageTestDataHandler.process_lin_rec2020_img: removing {output_fpath} to overwrite it"
            )
            os.remove(output_fpath)
        # make final image
        rawproc.dt_proc_img(
            src_fpath=tmp_fpath,
            dest_fpath=output_fpath,
            xmp_fpath=xmp_fpath,
            compression=True,
        )
        # remove tmp gt image
        if REMOVE_TMP_IMAGES:
            os.remove(tmp_fpath)  # FIXME DBG recomment

    def __getitem__(self, i: int):
        image = self._dataset[i]
        # figure out output_fpath
        # net_output_processed_fpath = os.path.join(
        #     self.net_output_proc_dpath,
        #     f'{os.path.basename(image["f_fpath"])}_aligned_to_{os.path.basename(image["gt_fpath"])}',
        # )

        def process_net_output(
            net_output: torch.Tensor, output_fpath: str
        ) -> torch.Tensor:
            if self.net_input_type == "proc":
                if self.always_output_processed_net_output:
                    raw.hdr_nparray_to_file(
                        net_output.squeeze(0).cpu(),
                        output_fpath,
                        color_profile="lin_rec2020",
                        bit_depth=16,
                        src_fpath=image["f_fpath"],
                    )
                return net_output
            elif self.net_input_type == "lin_rec2020" or self.net_input_type == "bayer":
                # first match gain wrt image['gt_rgb_mean']
                net_output = net_output * image["gt_rgb_mean"] / net_output.mean()

                self.process_lin_rec2020_img(
                    linrec_img=net_output.squeeze(0).cpu(),
                    output_fpath=output_fpath,
                    xmp_fpath=image["dt_xmp"],
                    src_fpath=image["f_fpath"],
                )
                return (
                    pt_helpers.fpath_to_tensor(
                        output_fpath, incl_metadata=False, device=net_output.device
                    )
                    .unsqueeze(0)
                    .to(net_output.dtype)
                )  # type: ignore
            else:
                raise ValueError(f"Invalid {self.net_input_type=}")

        manproc_gt = pt_helpers.fpath_to_tensor(image["gt_manproc_fpath"])
        res = {
            "x_crops": manproc_gt,
            "gt_fpath": image["gt_manproc_fpath"],
            "gain": 1.0,
            "net_output_processor_fun": process_net_output,
            "image_set": image["image_set"],
            "gt_rgb_mean": image["gt_rgb_mean"],
            # "net_output_processed_fpath": net_output_processed_fpath,
        }
        if "rgb_msssim_score" in image:
            res["rgb_msssim_score"] = image["rgb_msssim_score"]
        if self.net_input_type == "bayer":
            noisy_img, metadata = pt_helpers.fpath_to_tensor(
                image["f_fpath"], incl_metadata=True
            )
            res["y_fpath"] = image["f_fpath"]
            res["rgb_xyz_matrix"] = torch.from_numpy(metadata["rgb_xyz_matrix"])
            # align noisy image
            dummy_linrec_gt = noisy_img.new_empty(
                (3, noisy_img.size(-2) * 2, noisy_img.size(-1) * 2)
            )
            if "best_alignment" in image:
                _, noisy_img = rawproc.shift_images(
                    dummy_linrec_gt, noisy_img, image["best_alignment"]
                )
            noisy_img = pt_ops.crop_to_multiple(noisy_img, 16)
            noisy_img *= image.get("raw_gain", 1.0)
        elif self.net_input_type == "lin_rec2020":
            res["y_fpath"] = image["f_manproc_fpath"]
            if (
                "f_linrec2020_fpath" not in image and "linrec2020_fpath" in image
            ):  # workaround for unpaired images w/ incomplete metadata, this should probably be more consistent / cleaner but oh well
                image["f_linrec2020_fpath"] = image["linrec2020_fpath"]
            noisy_img: torch.Tensor = pt_helpers.fpath_to_tensor(
                image["f_linrec2020_fpath"], incl_metadata=False
            )  # type: ignore
            dummy_linrec_gt = torch.empty_like(noisy_img)
            if "best_alignment" in image:
                _, noisy_img = rawproc.shift_images(
                    dummy_linrec_gt, noisy_img, image["best_alignment"]
                )
            noisy_img = pt_ops.crop_to_multiple(noisy_img, 32)
            noisy_img *= image.get("rgb_gain", 1.0)
        elif self.net_input_type == "proc":
            res["y_fpath"] = image["f_manproc_fpath"]
            noisy_img = pt_helpers.fpath_to_tensor(
                image["f_manproc_fpath"], incl_metadata=False, crop_to_multiple=16
            )  # type: ignore
            res["x_crops"] = pt_ops.crop_to_multiple(res["x_crops"], 16)
        else:
            raise ValueError(f"Invalid {self.net_input_type=}")
        res["mask_crops"] = torch.ones_like(res["x_crops"])
        res["y_crops"] = noisy_img
        return res

    def get_images(self):
        for i in range(len(self._dataset)):
            yield self.__getitem__(i)

    def __len__(self):
        return len(self._dataset)


# class CleanManuallyProcessedNoisyBayerImageTestDataloader:
#     def __init__(self):
#         pass


def prep_manproc_dataset(
    test_descriptor_fpath: str = TEST_DESCRIPTOR_FPATH,
    rawnind_content_fpath=rawproc.RAWNIND_CONTENT_FPATH,
    test_reserve_fpath=os.path.join("config", "test_reserve.yaml"),
    bayer_only=True,
    alignment_max_loss=rawds.ALIGNMENT_MAX_LOSS,
    mask_mean_min=rawds.MASK_MEAN_MIN,
    unpaired_images=False,
):
    """
    Generate the manually processed images and the manproc dataset descriptor.

    This must be run after the training dataset has been generated, and before using the above data loaders.
    """
    print("Generating TIFF HDR ground-truth images")
    data_dpath = os.path.dirname(rawnind_content_fpath)  # ugly / FIXME
    if unpaired_images:
        make_hdr_extraraw_files.proc_dataset(dataset=data_dpath)
    else:
        make_hdr_rawnind_files.proc_dataset(
            gt_only=False,
            test_images_only=True,
            data_dpath=data_dpath,
            test_reserve_fpath=test_reserve_fpath,
        )
    # Load dataset descriptor containing alignment information
    if not test_reserve_fpath:
        print("Warning: missing test_reserve_fpath")
        test_reserve = []
    else:
        test_reserve = utilities.load_yaml(test_reserve_fpath, error_on_404=True)[
            "test_reserve"
        ]
    print(f"Using {test_reserve=}")
    print(f"Loading {rawnind_content_fpath}...")
    rawnind_content = utilities.load_yaml(rawnind_content_fpath, error_on_404=True)
    print(f"Loaded {rawnind_content_fpath}")
    test_dataset_descriptor = []
    # Get images that are reserved for testing
    for image in tqdm.tqdm(rawnind_content):
        if unpaired_images:
            image["is_bayer"] = image.get("is_bayer", "bayer_fpath" in image)
            image["image_set"] = os.path.basename(image["bayer_fpath"])
            image["gt_bayer_fpath"] = image["gt_fpath"] = image["f_fpath"] = image[
                "bayer_fpath"
            ]
            image["rgb_msssim_score"] = 1.0
        # Check that the image is reserved for testing
        if (
            (image["image_set"] not in test_reserve and not unpaired_images)
            or bayer_only
            and not image["is_bayer"]
        ):
            continue
        # Check that image is good enough
        if (not unpaired_images) and (
            image["best_alignment_loss"] > alignment_max_loss
            or image["mask_mean"] < mask_mean_min
        ):
            print(f"prep_manproc_dataset: rejected {image['f_fpath']}")
            continue
        del image["crops"]  # Not needed here
        # Add the relevant paths to metadata
        if unpaired_images:
            image["gt_linrec2020_fpath"] = image["linrec2020_fpath"]
        else:
            image["f_linrec2020_fpath"] = (
                image["f_fpath"].replace("src/Bayer", "proc/lin_rec2020") + ".tif"
            )
            image["gt_linrec2020_fpath"] = (
                image["gt_fpath"].replace("src/Bayer", "proc/lin_rec2020") + ".tif"
            )
        image["dt_xmp"] = image["gt_linrec2020_fpath"] + ".xmp"
        # figure out new and temporary paths
        os.makedirs(TMPDIR, exist_ok=True)
        gt_fn = os.path.basename(image["gt_linrec2020_fpath"])
        gt_dt_dpath = os.path.dirname(image["linrec2020_fpath"]).replace(
            "proc/lin_rec2020", "proc/dt"
        )

        if unpaired_images:
            image["gt_manproc_fpath"] = os.path.join(gt_dt_dpath, gt_fn)
            image["manproc_fpath"] = image["gt_manproc_fpath"]
            image["f_manproc_fpath"] = image["gt_manproc_fpath"]
        else:
            noisy_fn = os.path.basename(image["f_linrec2020_fpath"])
            noisy_dpath = os.path.dirname(image["f_linrec2020_fpath"]).replace(
                "proc/lin_rec2020", "proc/dt"
            )
            # "noisy" image can't be in the gt directory even if it's clean, otherwise
            # there will be a conflict when the actual gt xmp is used
            noisy_dpath = noisy_dpath.replace("/gt", "")
            image["gt_manproc_fpath"] = os.path.join(
                gt_dt_dpath, f"{image['image_set']}_{gt_fn}_aligned_to_{noisy_fn}"
            )
            image["f_manproc_fpath"] = os.path.join(
                noisy_dpath, f"{image['image_set']}_{noisy_fn}_aligned_to_{gt_fn}"
            )
            print(f"prep_manproc_dataset: processing {image['f_manproc_fpath']}")
        # Previously done; nothing to do
        if os.path.isfile(image["gt_manproc_fpath"]) and (
            os.path.isfile(image["f_manproc_fpath"]) and "gt_rgb_mean" in image
        ):
            print(
                f"prep_manproc_dataset: skipping existing files ({image['f_manproc_fpath']})"
            )
            test_dataset_descriptor.append(image)
            continue
        # Partially done; remove the existing files and start over
        if os.path.isfile(image["gt_manproc_fpath"]):
            os.remove(image["gt_manproc_fpath"])
        if os.path.isfile("f_manproc_fpath"):
            os.remove(image["f_manproc_fpath"])
        ### Generate the manually processed image
        # load the images
        gt_img: torch.Tensor = pt_helpers.fpath_to_tensor(image["gt_linrec2020_fpath"])  # type: ignore
        if unpaired_images:  # pretty useless
            pass
            # noisy_img = gt_img
        else:
            noisy_img: torch.Tensor = pt_helpers.fpath_to_tensor(
                image["f_linrec2020_fpath"]
            )  # type: ignore
            # align x, y
            gt_img, noisy_img = rawproc.shift_images(
                gt_img, noisy_img, image["best_alignment"]
            )
        # crop images to multiple of 16
        gt_img = pt_ops.crop_to_multiple(gt_img, 32)
        image["gt_rgb_mean"] = float(gt_img.float().mean().item())
        if not unpaired_images:
            noisy_img = pt_ops.crop_to_multiple(noisy_img, 32)
            noisy_img = noisy_img * image["rgb_gain"]
        # Make GT, noisy images
        # if 'capt0016.arw' in image['f_fpath'] or 'capt0016.arw' in image['gt_fpath']:
        #     breakpoint()
        ManuallyProcessedImageTestDataHandler.process_lin_rec2020_img(
            linrec_img=gt_img,
            output_fpath=image["gt_manproc_fpath"],
            xmp_fpath=image["dt_xmp"],
            src_fpath=image["gt_fpath"],
        )
        # repeat with noisy image if it is different from the gt image
        if image["gt_manproc_fpath"] != image["f_manproc_fpath"]:
            ManuallyProcessedImageTestDataHandler.process_lin_rec2020_img(
                linrec_img=noisy_img,
                output_fpath=image["f_manproc_fpath"],
                xmp_fpath=image["dt_xmp"],
                src_fpath=image["f_fpath"],
            )
        # Add the image to the test dataset descriptor
        assert "gt_rgb_mean" in image
        test_dataset_descriptor.append(image)
        print(f"prep_manproc_dataset: added {image['f_manproc_fpath']}")
    # save the test dataset descriptor
    with open(test_descriptor_fpath, "w") as f:
        yaml.dump(test_dataset_descriptor, f, sort_keys=False, allow_unicode=True)
    print(f"prep_manproc_dataset: saved {test_descriptor_fpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--test_descriptor_fpath",
        type=str,
        default=TEST_DESCRIPTOR_FPATH,
        help="Path to the test descriptor file",
    )
    parser.add_argument(
        "--rawnind_content_fpath",
        type=str,
        default=rawproc.RAWNIND_CONTENT_FPATH,
        help="Path to the RawNIND content file",
    )
    parser.add_argument(
        "--test_reserve_fpath",
        type=str,
        default=os.path.join("config", "test_reserve.yaml"),
        help="Path to the test reserve file",
    )
    parser.add_argument(
        "--unpaired_images", action="store_true", help="Use unpaired images"
    )

    args = parser.parse_args()
    prep_manproc_dataset(
        test_descriptor_fpath=args.test_descriptor_fpath,
        rawnind_content_fpath=args.rawnind_content_fpath,
        test_reserve_fpath=args.test_reserve_fpath,
        unpaired_images=args.unpaired_images,
    )
