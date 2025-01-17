"""
Raw dataset handlers.

Returns x (clean), y (noisy). Bayer (black-white point) or
ProfiledRGB (Lin. Rec2020).

Loaders from whole images are deprecated (too slow). To bring up to date they would need to handle:
- bayer_only
"""

import random
import logging
import os
import sys
import math
import time
import unittest
from typing import Literal, NamedTuple, Optional, Union
from typing import TypedDict
import tqdm

import torch

sys.path.append("..")
from common.libs import pt_helpers, utilities
from rawnind.libs import raw
from rawnind.libs import rawproc
from rawnind.libs import arbitrary_proc_fun

BREAKPOINT_ON_ERROR = True

COLOR_PROFILE = "lin_rec2020"
LOG_FPATH = os.path.join("logs", os.path.basename(__file__) + ".log")

MAX_MASKED: float = (
    0.5  # Must ensure that we don't send a crop with this more than this many masked pixels
)
MAX_RANDOM_CROP_ATTEMPS = 10

MASK_MEAN_MIN = 0.8  # 14+11+1 = 26 images out of 1145 = 2.3 %
ALIGNMENT_MAX_LOSS = (
    0.035  # eliminates 6+3+2 + 1+4+2+6+3 = 27 images out of 1145 = 2.4 %
)
OVEREXPOSURE_LB = 0.99

TOY_DATASET_LEN = 25  # debug option

# Abstract classes:


class RawDatasetOutput(TypedDict):
    x_crops: torch.Tensor
    y_crops: Optional[torch.Tensor]
    mask_crops: torch.BoolTensor
    rgb_xyz_matrix: Optional[torch.Tensor]
    gain: float


class RawImageDataset:
    def __init__(self, num_crops: int, crop_size: int):
        self.num_crops = num_crops
        assert crop_size % 2 == 0
        self.crop_size = crop_size

    def random_crops(
        self,
        ximg: torch.Tensor,
        yimg: Optional[torch.Tensor],
        whole_img_mask: torch.BoolTensor,
    ):  # -> Union[tuple[torch.Tensor, Optional[torch.Tensor], torch.BoolTensor], bool]:  # Python 3.8 incompat :(
        """
        Crop an image into num_crops cs*cs crops without exceeding MAX_MASKED threshold.

        Returns x_crops, (optionally) y_crops, mask_crops
        """
        vdim, hdim = ximg.shape[-2:]
        max_start_v, max_start_h = vdim - self.crop_size, hdim - self.crop_size
        x_crops_dims = (self.num_crops, ximg.shape[-3], self.crop_size, self.crop_size)
        x_crops = torch.empty(x_crops_dims)
        mask_crops = torch.BoolTensor(x_crops.shape)
        if yimg is not None:
            assert rawproc.shape_is_compatible(
                ximg.shape, yimg.shape
            ), f"ximg and yimg should already be aligned. {ximg.shape=}, {yimg.shape=}"
            y_crops_dims = (
                self.num_crops,
                yimg.shape[-3],
                self.crop_size // ((yimg.shape[-3] == 4) + 1),
                self.crop_size // ((yimg.shape[-3] == 4) + 1),
            )
            y_crops = torch.empty(y_crops_dims)
        else:
            y_crops = None
        # mask_crops = torch.BoolTensor(self.num_crops, self.crop_size, self.crop_size)
        for crop_i in range(self.num_crops):
            # try a random crop
            self.make_a_random_crop(
                crop_i,
                x_crops,
                y_crops,
                mask_crops,
                max_start_v,
                max_start_h,
                ximg,
                yimg,
                whole_img_mask,
            )
            # ensure there are sufficient valid pixels
            attempts: int = 0
            while mask_crops[crop_i].sum() / self.crop_size**2 < MAX_MASKED:
                if attempts >= MAX_RANDOM_CROP_ATTEMPS:
                    return False
                self.make_a_random_crop(
                    crop_i,
                    x_crops,
                    y_crops,
                    mask_crops,
                    max_start_v,
                    max_start_h,
                    ximg,
                    yimg,
                    whole_img_mask,
                )
                attempts += 1
        if yimg is not None:
            return x_crops, y_crops, mask_crops
        return x_crops, mask_crops

    def make_a_random_crop(
        self,
        crop_i: int,
        x_crops: torch.Tensor,
        y_crops: Optional[torch.Tensor],
        mask_crops: torch.BoolTensor,
        max_start_v: int,
        max_start_h: int,
        ximg: torch.Tensor,
        yimg: Optional[torch.Tensor],
        whole_img_mask: torch.BoolTensor,
    ) -> None:
        """
        Make a random crop at specified index of ximg without validity check.

        Modifies x_crops[crop_i] and mask_crops[crop_i] in-place.
        """
        hstart = random.randrange(max_start_h)
        vstart = random.randrange(max_start_v)
        hstart -= hstart % 2  # maintain Bayer pattern
        vstart -= vstart % 2  # maintain Bayer pattern
        # print(
        #    f"{x_crops.shape=}, {ximg.shape=}, {vstart=}, {hstart=}, {self.crop_size=}, {max_start_h=}, {max_start_v=}"
        # )  # dbg
        x_crops[crop_i] = ximg[
            ..., vstart : vstart + self.crop_size, hstart : hstart + self.crop_size
        ]
        if yimg is not None:
            yimg_divisor = (yimg.shape[0] == 4) + 1
            y_crops[crop_i] = yimg[
                ...,
                vstart // yimg_divisor : vstart // yimg_divisor
                + self.crop_size // yimg_divisor,
                hstart // yimg_divisor : hstart // yimg_divisor
                + self.crop_size // yimg_divisor,
            ]
        mask_crops[crop_i] = whole_img_mask[
            ..., vstart : vstart + self.crop_size, hstart : hstart + self.crop_size
        ]

    def center_crop(
        self,
        ximg: torch.Tensor,
        yimg: Optional[torch.Tensor],
        mask: torch.BoolTensor,
    ):  # -> Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]: # python3.8 incompat :(
        height, width = ximg.shape[-2:]
        ystart = height // 2 - (self.crop_size // 2)
        xstart = width // 2 - (self.crop_size // 2)
        ystart -= ystart % 2
        xstart -= xstart % 2
        xcrop = ximg[
            ...,
            ystart : ystart + self.crop_size,
            xstart : xstart + self.crop_size,
        ]
        mask_crop = mask[
            ...,
            ystart : ystart + self.crop_size,
            xstart : xstart + self.crop_size,
        ]
        if yimg is not None:
            if yimg.size(-3) == 4:
                shape_divisor = 2
            elif yimg.size(-3) == 3:
                shape_divisor = 1
            else:
                raise ValueError(
                    f"center_crop: invalid number of channels: {yimg.size(-3)=}"
                )
            ycrop = yimg[
                ...,
                ystart // shape_divisor : ystart // shape_divisor
                + self.crop_size // shape_divisor,
                xstart // shape_divisor : xstart // shape_divisor
                + self.crop_size // shape_divisor,
            ]
            return xcrop, ycrop, mask_crop
        return xcrop, mask_crop


class ProfiledRGBBayerImageDataset(RawImageDataset):
    def __init__(self, num_crops: int, crop_size: int):
        super().__init__(num_crops=num_crops, crop_size=crop_size)

    @staticmethod
    def camRGB_to_profiledRGB_img(
        camRGB_img: torch.Tensor,
        metadata: dict,
        output_color_profile=COLOR_PROFILE,
    ) -> torch.Tensor:
        return raw.camRGB_to_profiledRGB_img(camRGB_img, metadata, output_color_profile)

    # @staticmethod
    # def crop_rgb_to_bayer(
    #     rgb_img: torch.Tensor, metadata: dict
    # ) -> torch.Tensor:
    #     """Crop an RGB image to match the crop applied to get a universal RGGB Bayer pattern."""
    #     assert rgb_img.dim() == 3
    #     if metadata.get("cropped_y"):
    #         rgb_img = rgb_img[:, 1:-1]
    #     if metadata.get("cropped_x"):
    #         rgb_img = rgb_img[:, :, 1:-1]
    #     return rgb_img


class ProfiledRGBProfiledRGBImageDataset(RawImageDataset):
    def __init__(self, num_crops: int, crop_size: int):
        super().__init__(num_crops=num_crops, crop_size=crop_size)


class CleanCleanImageDataset(RawImageDataset):
    def __init__(self, num_crops: int, crop_size: int):
        super().__init__(num_crops=num_crops, crop_size=crop_size)

    def get_mask(self, ximg: torch.Tensor, metadata: dict) -> torch.BoolTensor:
        # we only ever apply the mask to RGB images so interpolate if Bayer
        if ximg.shape[0] == 4:
            ximg = torch.nn.functional.interpolate(
                ximg.unsqueeze(0), scale_factor=2
            ).squeeze(0)
            return (
                (ximg.max(0).values < metadata["overexposure_lb"])
                .unsqueeze(0)
                .repeat(3, 1, 1)
            )
        # because color transform has already been applied we can mask individual channels
        return ximg < metadata["overexposure_lb"]  # .all(-3)


class CleanNoisyDataset(RawImageDataset):
    """Dataset of clean-noisy images."""

    def __init__(self, num_crops: int, crop_size: int):
        super().__init__(num_crops=num_crops, crop_size=crop_size)
        self._dataset = []

    def __len__(self):
        return len(self._dataset)


class TestDataLoader:
    OUTPUTS_IMAGE_FILES = False

    def __init__(self, **kwargs):
        pass

    def __getitem__(self, i):
        raise TypeError(
            f"{type(self).__name__} is its own data loader: "
            "call get_images instead of __getitem__ (or use built-in __iter__)."
        )

    def __iter__(self):
        return self.get_images()

    def batched_iterator(self):
        single_to_batch = lambda x: torch.unsqueeze(x, 0)
        identity = lambda x: x
        if hasattr(
            self, "get_images"
        ):  # TODO should combine this ifelse with an iterator selection
            for res in self.get_images():
                batch_fun = single_to_batch if res["y_crops"].dim() == 3 else identity
                res["y_crops"] = batch_fun(res["y_crops"]).float()
                res["x_crops"] = batch_fun(res["x_crops"]).float()
                res["mask_crops"] = batch_fun(res["mask_crops"])
                if "rgb_xyz_matrix" in res:
                    res["rgb_xyz_matrix"] = batch_fun(res["rgb_xyz_matrix"])
                yield res
        else:
            for i in range(len(self._dataset)):
                res = self.__getitem__(i)
                res["y_crops"] = batch_fun(res["y_crops"], 0).float()
                res["x_crops"] = batch_fun(res["x_crops"], 0).float()
                res["mask_crops"] = batch_fun(res["mask_crops"])
                if "rgb_xyz_matrix" in res:
                    res["rgb_xyz_matrix"] = batch_fun(res["rgb_xyz_matrix"])
                yield res

    @staticmethod
    def _content_fpaths_to_test_reserve(content_fpaths: list[str]) -> list[str]:
        # add all images to test_reserve:
        test_reserve = []
        for content_fpath in content_fpaths:
            for image in utilities.load_yaml(content_fpath, error_on_404=True):
                # get the directory name of the image (not the full path)
                dn = os.path.basename(os.path.dirname(image["f_fpath"]))
                if dn == "gt":
                    continue
                test_reserve.append(dn)
        return test_reserve


# Actual training classes:


## Pre-cropped images


class _ds_item(NamedTuple):
    overexposure_lb: float
    rgb_xyz_matrix: torch.Tensor
    crops: list[dict[str, str]]


class CleanProfiledRGBCleanBayerImageCropsDataset(
    CleanCleanImageDataset, ProfiledRGBBayerImageDataset
):
    """Dataloader for pre-cropped unpaired images generated by tools/crop_dataset.py w/ metadata from tools/prep_image_dataset_extraraw.py."""

    def __init__(
        self,
        content_fpaths: list[str],
        num_crops: int,
        crop_size: int,
        toy_dataset: bool = False,
    ):
        super().__init__(num_crops=num_crops, crop_size=crop_size)
        self.num_crops = num_crops
        # self._dataset_xy_fpaths: list[tuple[str, str]] = []  # (gt_fpath, src_fpath)  # python 3.8 incompat
        self._dataset: list[_ds_item] = []  # (gt_fpath, src_fpath)
        for content_fpath in content_fpaths:
            logging.info(
                f"CleanProfiledRGBCleanBayerImageCropsDataset.__init__: loading {content_fpath}"
            )
            ds_content = utilities.load_yaml(content_fpath, error_on_404=True)
            for all_metadata in tqdm.tqdm(ds_content):
                if toy_dataset and len(self._dataset) >= TOY_DATASET_LEN:
                    break
                useful_metadata = {
                    "overexposure_lb": all_metadata["overexposure_lb"],
                    "rgb_xyz_matrix": torch.tensor(all_metadata["rgb_xyz_matrix"]),
                    "crops": all_metadata["crops"],
                }
                if not useful_metadata["crops"]:
                    logging.warning(
                        f"CleanProfiledRGBCleanBayerImageCropsDataset.__init__: image {all_metadata} has no useful crops; not adding to dataset."
                    )
                else:
                    self._dataset.append(useful_metadata)
        logging.info(f"initialized {type(self).__name__} with {len(self)} images.")
        if len(self) == 0:
            if BREAKPOINT_ON_ERROR:
                breakpoint()
            else:
                exit(-1)

    def __getitem__(self, i: int) -> RawDatasetOutput:
        metadata = self._dataset[i]
        crop: dict[str, str] = random.choice(metadata["crops"])
        try:
            gt = pt_helpers.fpath_to_tensor(crop["gt_linrec2020_fpath"]).float()
            rgbg_img = pt_helpers.fpath_to_tensor(crop["gt_bayer_fpath"]).float()
        except ValueError as e:
            logging.error(e)
            return self.__getitem__(random.randrange(len(self)))
        mask = self.get_mask(rgbg_img, metadata)
        try:
            x_crops, y_crops, mask_crops = self.random_crops(gt, rgbg_img, mask)
        except AssertionError as e:
            logging.info(crop)
            raise AssertionError(f"{self} {e} with {crop=}")
        except RuntimeError as e:
            logging.error(e)
            logging.error(f"{gt.shape=}, {rgbg_img.shape=}, {mask.shape=}")
            raise RuntimeError(f"{self} {e} with {crop=}")
        except TypeError:
            logging.warning(
                f"{crop} does not contain sufficient valid pixels; removing from dataset"
            )
            self._dataset[i]["crops"].remove(crop)
            if len(self._dataset[i]["crops"]) == 0:
                logging.warning(
                    f"{self._dataset[i]} does not contain anymore valid crops. Removing whole image from dataset."
                )
                self._dataset.remove(self._dataset[i])
            return self.__getitem__(i)
        return {
            "x_crops": x_crops,
            "y_crops": y_crops,
            "mask_crops": mask_crops,
            "rgb_xyz_matrix": metadata["rgb_xyz_matrix"],
            "gain": 1.0,
        }

    def __len__(self) -> int:
        return len(self._dataset)


class CleanProfiledRGBCleanProfiledRGBImageCropsDataset(
    CleanCleanImageDataset, ProfiledRGBProfiledRGBImageDataset
):
    """Dataloader for pre-cropped unpaired images generated by tools/crop_dataset.py w/ metadata from tools/prep_image_dataset_extraraw.py."""

    def __init__(
        self,
        content_fpaths: list[str],
        num_crops: int,
        crop_size: int,
        toy_dataset: bool = False,
        arbitrary_proc_method: bool = False,
    ):
        super().__init__(num_crops=num_crops, crop_size=crop_size)
        self.arbitrary_proc_method = arbitrary_proc_method
        self.num_crops = num_crops
        # self._dataset_xy_fpaths: list[tuple[str, str]] = []  # (gt_fpath, src_fpath)  # python 3.8 incompat
        self._dataset: list[_ds_item] = []  # (gt_fpath, src_fpath)
        for content_fpath in content_fpaths:
            logging.info(
                f"CleanProfiledRGBCleanProfiledRGBImageCropsDataset.__init__: loading {content_fpath}"
            )
            ds_content = utilities.load_yaml(content_fpath, error_on_404=True)
            for all_metadata in tqdm.tqdm(ds_content):
                if toy_dataset and len(self._dataset) >= TOY_DATASET_LEN:
                    break
                useful_metadata = {
                    "overexposure_lb": all_metadata["overexposure_lb"],
                    "crops": all_metadata["crops"],
                }
                if not useful_metadata["crops"]:
                    logging.warning(
                        f"CleanProfiledRGBCleanProfiledRGBImageCropsDataset.__init__: image {all_metadata} has no useful crops; not adding to dataset."
                    )
                else:
                    self._dataset.append(useful_metadata)
        logging.info(f"initialized {type(self).__name__} with {len(self)} images.")
        if len(self) == 0:
            if BREAKPOINT_ON_ERROR:
                breakpoint()
            else:
                exit(-1)

    def __getitem__(self, i: int) -> RawDatasetOutput:
        metadata = self._dataset[i]
        crop: dict[str, str] = random.choice(metadata["crops"])
        try:
            gt = pt_helpers.fpath_to_tensor(crop["gt_linrec2020_fpath"]).float()
            rgbg_img = pt_helpers.fpath_to_tensor(
                crop["gt_bayer_fpath"]
            ).float()  # used to compute the overexposure mask
        except ValueError as e:
            logging.error(e)
            return self.__getitem__(random.randrange(len(self)))
        mask = self.get_mask(rgbg_img, metadata)
        if self.arbitrary_proc_method:
            gt = arbitrary_proc_fun.arbitrarily_process_images(
                gt,
                randseed=crop["gt_linrec2020_fpath"],
                method=self.arbitrary_proc_method,
            )
        try:
            x_crops, mask_crops = self.random_crops(gt, None, mask)
        except AssertionError as e:
            logging.info(crop)
            raise AssertionError(f"{self} {e} with {crop=}")
        except RuntimeError as e:
            logging.error(e)
            logging.error(f"{gt.shape=}, {rgbg_img.shape=}, {mask.shape=}")
            raise RuntimeError(f"{self} {e} with {crop=}")
        except TypeError:
            logging.warning(
                f"{crop} does not contain sufficient valid pixels; removing from dataset"
            )
            self._dataset[i]["crops"].remove(crop)
            if len(self._dataset[i]["crops"]) == 0:
                logging.warning(
                    f"{self._dataset[i]} does not contain anymore valid crops. Removing whole image from dataset."
                )
                self._dataset.remove(self._dataset[i])
            return self.__getitem__(i)
        return {"x_crops": x_crops, "mask_crops": mask_crops, "gain": 1.0}

    def __len__(self) -> int:
        return len(self._dataset)


class CleanProfiledRGBNoisyBayerImageCropsDataset(
    CleanNoisyDataset, ProfiledRGBBayerImageDataset
):
    """
    Dataset of clean-noisy raw images from rawNIND.

    Load from raw files using rawpy.
    Returns float crops, (highlight and anomaly) mask, metadata

    Alignment and masks are pre-computed.
    Output metadata contains color_matrix.
    """

    def __init__(
        self,
        content_fpaths: list[str],
        num_crops: int,
        crop_size: int,
        # test_reserve: list[str],  # python 3.8 incompat
        test_reserve: list,
        bayer_only: bool = True,
        alignment_max_loss: float = ALIGNMENT_MAX_LOSS,
        mask_mean_min: float = MASK_MEAN_MIN,
        test: bool = False,
        toy_dataset: bool = False,
        data_pairing: Literal["x_y", "x_x", "y_y"] = "x_y",  # x_y, x_x, y_y
        match_gain: bool = False,
        min_msssim_score: Optional[float] = 0.0,
        max_msssim_score: Optional[float] = 1.0,
    ):
        """
        content_fpath points to a yaml file containing:
            - best_alignment
            - f_bayer_fpath
            - gt_linrec2020_fpath
            - mask_fpath
            - best_alignment_loss
            - mask_mean

        return_data
        """
        super().__init__(num_crops=num_crops, crop_size=crop_size)
        self.match_gain = match_gain
        assert bayer_only
        # contents: list[dict] = utilities.load_yaml(content_fpath)
        for content_fpath in content_fpaths:
            contents = utilities.load_yaml(
                content_fpath, error_on_404=True
            )  # python 3.8 incompat
            for image in contents:
                if toy_dataset and len(self._dataset) >= TOY_DATASET_LEN:
                    break
                if not image["is_bayer"]:
                    continue

                # check that the image is (/not) reserved for testing
                if (not test and image["image_set"] in test_reserve) or (
                    test and image["image_set"] not in test_reserve
                ):
                    continue
                try:
                    if (
                        min_msssim_score
                        and min_msssim_score > image["rgb_msssim_score"]
                    ):
                        print(
                            f"Skipping {image['f_fpath']} with {image['rgb_msssim_score']} < {min_msssim_score}"
                        )
                        continue
                    if (
                        max_msssim_score
                        and max_msssim_score != 1.0
                        and max_msssim_score < image["rgb_msssim_score"]
                    ):
                        print(
                            f"Skipping {image['f_fpath']} with {image['rgb_msssim_score']} > {max_msssim_score}"
                        )
                        continue
                except KeyError:
                    raise KeyError(
                        f"{image} does not contain msssim score (required with {min_msssim_score=})"
                    )
                if (
                    image["best_alignment_loss"] > alignment_max_loss
                    or image["mask_mean"] < mask_mean_min
                ):
                    logging.info(
                        f'{type(self).__name__}.__init__: rejected {image["f_fpath"]}'
                    )
                    continue
                image["crops"] = sorted(
                    image["crops"], key=lambda d: d["coordinates"]
                )  # for testing
                if len(image["crops"]) > 0:
                    self._dataset.append(image)
                else:
                    logging.warning(
                        f'{type(self).__name__}.__init__: {image["f_fpath"]} has no crops.'
                    )
        logging.info(f"initialized {type(self).__name__} with {len(self)} images.")
        assert (
            len(self) > 0
        ), f"{type(self).__name__} has no images. {content_fpaths=}, {test_reserve=}"
        self.data_pairing = data_pairing

    def __getitem__(self, i: int) -> RawDatasetOutput:
        image: dict = self._dataset[i]
        # load x, y, mask
        crop = random.choice(image["crops"])
        if self.data_pairing == "x_y":
            gt_img = pt_helpers.fpath_to_tensor(crop["gt_linrec2020_fpath"])
            noisy_img = pt_helpers.fpath_to_tensor(crop["f_bayer_fpath"])
            # gt_img = self.crop_rgb_to_bayer(gt_img, metadata)

            # align x, y

            gt_img, noisy_img = rawproc.shift_images(
                gt_img, noisy_img, image["best_alignment"]
            )

            whole_img_mask = pt_helpers.fpath_to_tensor(image["mask_fpath"])[
                :,
                crop["coordinates"][1] : crop["coordinates"][1] + gt_img.shape[1],
                crop["coordinates"][0] : crop["coordinates"][0] + gt_img.shape[2],
            ]
            whole_img_mask = whole_img_mask.expand(gt_img.shape)
        elif self.data_pairing == "x_x":
            gt_img = pt_helpers.fpath_to_tensor(crop["gt_linrec2020_fpath"])
            noisy_img = pt_helpers.fpath_to_tensor(crop["gt_bayer_fpath"])
            whole_img_mask = torch.ones_like(gt_img)
        elif self.data_pairing == "y_y":
            gt_img = pt_helpers.fpath_to_tensor(crop["f_linrec2020_fpath"])
            noisy_img = pt_helpers.fpath_to_tensor(crop["f_bayer_fpath"])
            whole_img_mask = torch.ones_like(gt_img)
        else:
            raise ValueError(f"return_data={self.data_pairing} not supported")

        # crop x, y, mask, add alignment to mask
        try:
            x_crops, y_crops, mask_crops = self.random_crops(
                gt_img, noisy_img, whole_img_mask
            )
        except TypeError:
            logging.warning(
                f"{crop} does not contain sufficient valid pixels; removing from dataset"
            )
            self._dataset[i]["crops"].remove(crop)
            if len(self._dataset[i]["crops"]) == 0:
                logging.warning(
                    f"{self._dataset[i]} does not contain anymore valid crops. Removing whole image from dataset."
                )
                self._dataset.remove(self._dataset[i])
            return self.__getitem__(i)
        # hardcoded_rgbm = torch.tensor(
        #     [
        #         [0.7034, -0.0804, -0.1014],
        #         [-0.4420, 1.2564, 0.2058],
        #         [-0.0851, 0.1994, 0.5758],
        #         [0.0000, 0.0000, 0.0000],
        #     ]
        # )
        output = {
            "x_crops": x_crops.float(),
            "y_crops": y_crops.float(),
            "mask_crops": mask_crops,
            # "rgb_xyz_matrix": hardcoded_rgbm  # TODO RM DBG
            "rgb_xyz_matrix": torch.tensor(image["rgb_xyz_matrix"]),
        }
        if self.match_gain:
            output["y_crops"] *= image["raw_gain"]
            output["gain"] = 1.0
        else:
            output["gain"] = image["raw_gain"]
        return output


class CleanProfiledRGBNoisyProfiledRGBImageCropsDataset(
    CleanNoisyDataset, ProfiledRGBProfiledRGBImageDataset
):
    """
    Dataset of clean-noisy demosaiced images from rawNIND.

    Load from OpenEXR files.
    Returns aligned float crops, (highlight and anomaly) mask
    """

    def __init__(
        self,
        content_fpaths: list[str],
        num_crops: int,
        crop_size: int,
        # test_reserve: list[str],
        test_reserve,  # python 38 incompat
        bayer_only: bool,
        alignment_max_loss: float = ALIGNMENT_MAX_LOSS,
        mask_mean_min: float = MASK_MEAN_MIN,
        test: bool = False,
        toy_dataset: bool = False,
        data_pairing: Literal["x_y", "x_x", "y_y"] = "x_y",
        match_gain: bool = False,
        arbitrary_proc_method: bool = False,
        min_msssim_score: Optional[float] = 0.0,
        max_msssim_score: Optional[float] = 1.0,
    ):
        """
        content_fpath points to a yaml file containing:
            - best_alignment
            - f_linrec2020_fpath
            - gt_linrec2020_fpath
            - mask_fpath
            - best_alignment_loss
            - mask_mean
        """
        super().__init__(num_crops=num_crops, crop_size=crop_size)
        self.match_gain = match_gain
        self.arbitrary_proc_method = arbitrary_proc_method
        if self.arbitrary_proc_method:
            assert (
                self.match_gain
            ), f"{type(self).__name__}: arbitrary_proc_method requires match_gain"
        self.data_pairing = data_pairing
        # contents: list[dict] = utilities.load_yaml(content_fpath)
        for content_fpath in content_fpaths:
            contents = utilities.load_yaml(
                content_fpath, error_on_404=True
            )  # python38 incompat
            for image in contents:
                if toy_dataset and len(self._dataset) >= TOY_DATASET_LEN:
                    break

                # check that the image is (/not) reserved for testing
                if (
                    (not test and image["image_set"] in test_reserve)
                    or (test and image["image_set"] not in test_reserve)
                    or (  # check that there is a bayer version available if bayer_only is True
                        bayer_only and not image["is_bayer"]
                    )
                ):
                    # print(f'Image is (/not) reserved for testing: {image["image_set"]}')
                    continue
                try:
                    if (
                        min_msssim_score
                        and min_msssim_score > image["rgb_msssim_score"]
                    ):
                        continue
                    if (
                        max_msssim_score
                        and max_msssim_score != 1.0
                        and max_msssim_score < image["rgb_msssim_score"]
                    ):
                        print(
                            f"Skipping {image['f_fpath']} with {image['rgb_msssim_score']} > {max_msssim_score}"
                        )
                        continue
                except KeyError:
                    raise KeyError(
                        f"{image} does not contain msssim score (required with {min_msssim_score=})"
                    )

                if (
                    image["best_alignment_loss"] > alignment_max_loss
                    or image["mask_mean"] < mask_mean_min
                ):
                    logging.info(
                        f'{type(self).__name__}.__init__: rejected {image["f_fpath"]}'
                    )
                    continue
                image["crops"] = sorted(
                    image["crops"], key=lambda d: d["coordinates"]
                )  # for testing
                if len(image["crops"]) > 0:
                    self._dataset.append(image)
                else:
                    logging.warning(
                        f"{type(self).__name__}.__init__: {image['f_fpath']} has no crops."
                    )
        logging.info(f"initialized {type(self).__name__} with {len(self)} images.")
        if len(self) == 0:
            if BREAKPOINT_ON_ERROR:
                breakpoint()
            else:
                exit(-1)

    def __getitem__(
        self,
        i: int,
        # ) -> tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
    ):  # python 3.8 incompat
        """Returns a random crop triplet (ximage, yimage, mask).

        Args:
            i (int): Image index

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]: random crop triplet
        """
        image = self._dataset[i]
        crop = random.choice(image["crops"])
        if self.data_pairing == "x_y":
            # load x, y, mask
            gt_img = pt_helpers.fpath_to_tensor(crop["gt_linrec2020_fpath"])
            noisy_img = pt_helpers.fpath_to_tensor(crop["f_linrec2020_fpath"])

            # align x, y
            gt_img, noisy_img = rawproc.shift_images(
                gt_img, noisy_img, image["best_alignment"]
            )
            whole_img_mask = pt_helpers.fpath_to_tensor(image["mask_fpath"])[
                :,
                crop["coordinates"][1] : crop["coordinates"][1] + gt_img.shape[1],
                crop["coordinates"][0] : crop["coordinates"][0] + gt_img.shape[2],
            ]
            whole_img_mask = whole_img_mask.expand(gt_img.shape)
        elif self.data_pairing == "x_x":
            gt_img = pt_helpers.fpath_to_tensor(crop["gt_linrec2020_fpath"])
            noisy_img = pt_helpers.fpath_to_tensor(crop["gt_linrec2020_fpath"])
            whole_img_mask = torch.ones_like(gt_img)
        elif self.data_pairing == "y_y":
            gt_img = pt_helpers.fpath_to_tensor(crop["f_linrec2020_fpath"])
            noisy_img = pt_helpers.fpath_to_tensor(crop["f_linrec2020_fpath"])
            whole_img_mask = torch.ones_like(gt_img)
        output = {}
        if self.match_gain:
            noisy_img *= image["rgb_gain"]
            output["gain"] = 1.0
        else:
            output["gain"] = image["rgb_gain"]
        if self.arbitrary_proc_method:
            # print(f"{self.__class__.__name__}.__getitem__ dbg: {crop['gt_linrec2020_fpath']=}, {self.arbitrary_proc_method=}")
            gt_img = arbitrary_proc_fun.arbitrarily_process_images(
                gt_img,
                randseed=crop["gt_linrec2020_fpath"],
                method=self.arbitrary_proc_method,
            )
            # print(f"{self.__class__.__name__}.__getitem__ dbg: {crop['f_linrec2020_fpath']=}, {self.arbitrary_proc_method=}")
            noisy_img = arbitrary_proc_fun.arbitrarily_process_images(
                noisy_img,
                randseed=crop["gt_linrec2020_fpath"],
                method=self.arbitrary_proc_method,
            )
        # crop x, y, mask
        try:
            x_crops, y_crops, mask_crops = self.random_crops(
                gt_img, noisy_img, whole_img_mask
            )
        except TypeError:
            logging.warning(
                f"{crop} does not contain sufficient valid pixels; removing from dataset"
            )
            self._dataset[i]["crops"].remove(crop)
            if len(self._dataset[i]["crops"]) == 0:
                logging.warning(
                    f"{self._dataset[i]} does not contain anymore valid crops. Removing whole image from dataset."
                )
                self._dataset.remove(self._dataset[i])
            return self.__getitem__(i)
        # mask_crops = mask_crops.unsqueeze(1).expand(x_crops.shape)
        output["x_crops"] = x_crops.float()
        output["y_crops"] = y_crops.float()
        output["mask_crops"] = mask_crops

        return output
        return {
            "x_crops": x_crops.float(),
            "y_crops": y_crops.float(),
            "mask_crops": mask_crops,
            "gain": image["rgb_gain"],
        }


# Test / Validation datasets

## From image crops


class CleanProfiledRGBNoisyProfiledRGBImageCropsValidationDataset(
    CleanProfiledRGBNoisyProfiledRGBImageCropsDataset
):
    """Validation dataset for noisy profiled RGB to clean profiled RGB."""

    def __init__(
        self,
        content_fpaths: list[str],
        crop_size: int,
        # test_reserve: list[str],
        test_reserve,  # python38 incompat
        bayer_only: bool,
        alignment_max_loss: float = ALIGNMENT_MAX_LOSS,
        mask_mean_min: float = MASK_MEAN_MIN,
        toy_dataset: bool = False,
        match_gain: bool = False,
        arbitrary_proc_method: bool = False,
        data_pairing: Literal["x_y", "x_x", "y_y"] = "x_y",
    ):
        super().__init__(
            content_fpaths=content_fpaths,
            num_crops=1,
            crop_size=crop_size,
            test_reserve=test_reserve,
            alignment_max_loss=alignment_max_loss,
            mask_mean_min=mask_mean_min,
            test=True,
            bayer_only=bayer_only,
            toy_dataset=toy_dataset,
            match_gain=match_gain,
            arbitrary_proc_method=arbitrary_proc_method,
            data_pairing=data_pairing,
        )

    def __getitem__(
        self, i: int
    ):  # -> tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:  # python 3.8 incompat
        """Returns a center crop triplet (ximage, yimage, mask).

        Args:
            i (int): Image index

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]: center crop triplet
        """
        image: dict = self._dataset[i]
        crop_n = len(image["crops"]) // 2
        crop = image["crops"][crop_n]
        # load x, y, mask
        if self.data_pairing == "x_y":
            gt_img = pt_helpers.fpath_to_tensor(crop["gt_linrec2020_fpath"])
            noisy_img = pt_helpers.fpath_to_tensor(crop["f_linrec2020_fpath"])

            # align x, y
            gt_img, noisy_img = rawproc.shift_images(
                gt_img, noisy_img, image["best_alignment"]
            )
            whole_img_mask = pt_helpers.fpath_to_tensor(image["mask_fpath"])[
                :,
                crop["coordinates"][1] : crop["coordinates"][1] + gt_img.shape[1],
                crop["coordinates"][0] : crop["coordinates"][0] + gt_img.shape[2],
            ]
            try:
                whole_img_mask = whole_img_mask.expand(gt_img.shape)
            except RuntimeError as e:
                logging.error(e)
                breakpoint()
        elif self.data_pairing == "x_x":
            gt_img = pt_helpers.fpath_to_tensor(crop["gt_linrec2020_fpath"])
            noisy_img = pt_helpers.fpath_to_tensor(crop["gt_linrec2020_fpath"])
            whole_img_mask = torch.ones_like(gt_img)
        elif self.data_pairing == "y_y":
            gt_img = pt_helpers.fpath_to_tensor(crop["f_linrec2020_fpath"])
            noisy_img = pt_helpers.fpath_to_tensor(crop["f_linrec2020_fpath"])
            whole_img_mask = torch.ones_like(gt_img)
        # crop x, y, mask, add alignment to mask
        if self.crop_size == 0:
            height, width = gt_img.shape[-2:]
            height = height - height % 256
            width = width - width % 256
            min_crop_size = 256
            x_crop = gt_img[..., :height, :width]
            noisy_img = y_crop = noisy_img[..., :height, :width]
            whole_img_mask = mask_crop = whole_img_mask[..., :height, :width]
        else:
            min_crop_size = self.crop_size
            x_crop, y_crop, mask_crop = self.center_crop(
                gt_img, noisy_img, whole_img_mask
            )
        if x_crop.shape[-1] < min_crop_size or x_crop.shape[-2] < min_crop_size:
            logging.warning(
                f"CleanProfiledRGBNoisyProfiledRGBImageCropsValidationDataset.__getitem__: not enough pixels in {crop['gt_linrec2020_fpath']}; deleting from dataset"
            )
            self._dataset[i]["crops"].remove(crop)
            return self.__getitem__(i)

        output = {
            "x_crops": x_crop.float(),
            "y_crops": y_crop.float(),
            "mask_crops": mask_crop,
            "gt_fpath": crop["gt_linrec2020_fpath"],
            "y_fpath": crop["f_linrec2020_fpath"],
        }
        if self.match_gain:
            output["y_crops"] *= image["rgb_gain"]
            output["gain"] = 1.0
        else:
            output["gain"] = image["rgb_gain"]
        if self.arbitrary_proc_method:
            output["x_crops"] = arbitrary_proc_fun.arbitrarily_process_images(
                output["x_crops"],
                randseed=crop["gt_linrec2020_fpath"],
                method=self.arbitrary_proc_method,
            )
            output["y_crops"] = arbitrary_proc_fun.arbitrarily_process_images(
                output["y_crops"],
                randseed=crop["gt_linrec2020_fpath"],
                method=self.arbitrary_proc_method,
            )
        return output


class CleanProfiledRGBNoisyBayerImageCropsValidationDataset(
    CleanProfiledRGBNoisyBayerImageCropsDataset
):
    """Dataset of clean (profiled RGB) - noisy (Bayer) images from rawNIND."""

    def __init__(
        self,
        content_fpaths: list[str],
        crop_size: int,
        # test_reserve: list[str],
        test_reserve,  # python 38 incompat
        bayer_only: bool,
        alignment_max_loss: float = ALIGNMENT_MAX_LOSS,
        mask_mean_min: float = MASK_MEAN_MIN,
        toy_dataset=False,
        match_gain: bool = False,
        data_pairing: Literal["x_y", "x_x", "y_y"] = "x_y",
    ):
        super().__init__(
            content_fpaths=content_fpaths,
            num_crops=1,
            crop_size=crop_size,
            test_reserve=test_reserve,
            alignment_max_loss=alignment_max_loss,
            mask_mean_min=mask_mean_min,
            test=True,
            bayer_only=bayer_only,
            toy_dataset=toy_dataset,
            match_gain=match_gain,
            data_pairing=data_pairing,
        )

    def __getitem__(self, i):
        image: dict = self._dataset[i]
        crop_n = len(image["crops"]) // 2
        crop = image["crops"][crop_n]
        # load x, y, mask
        if self.data_pairing == "x_y":
            gt_img = pt_helpers.fpath_to_tensor(crop["gt_linrec2020_fpath"])
            noisy_img = pt_helpers.fpath_to_tensor(crop["f_bayer_fpath"])
            # print(f"{metadata=}")
            # align x, y
            gt_img, noisy_img = rawproc.shift_images(
                gt_img, noisy_img, image["best_alignment"]
            )
            whole_img_mask = pt_helpers.fpath_to_tensor(image["mask_fpath"])[
                :,
                crop["coordinates"][1] : crop["coordinates"][1] + gt_img.shape[1],
                crop["coordinates"][0] : crop["coordinates"][0] + gt_img.shape[2],
            ]
            whole_img_mask = whole_img_mask.expand(gt_img.shape)
        elif self.data_pairing == "x_x":
            gt_img = pt_helpers.fpath_to_tensor(crop["gt_linrec2020_fpath"])
            noisy_img = pt_helpers.fpath_to_tensor(crop["f_bayer_fpath"])
            whole_img_mask = torch.ones_like(gt_img)
        elif self.data_pairing == "y_y":
            gt_img = pt_helpers.fpath_to_tensor(crop["f_linrec2020_fpath"])
            noisy_img = pt_helpers.fpath_to_tensor(crop["f_bayer_fpath"])
            whole_img_mask = torch.ones_like(gt_img)

        # crop x, y, mask, add alignment to mask
        if self.crop_size == 0:
            height, width = gt_img.shape[-2:]
            height = height - height % 256
            width = width - width % 256
            min_crop_size = 256
            x_crop = gt_img[..., :height, :width]
            noisy_img = y_crop = noisy_img[..., : height // 2, : width // 2]
            whole_img_mask = mask_crop = whole_img_mask[..., :height, :width]
        else:
            min_crop_size = self.crop_size
            x_crop, y_crop, mask_crop = self.center_crop(
                gt_img, noisy_img, whole_img_mask
            )
        if x_crop.shape[-1] < min_crop_size or x_crop.shape[-2] < min_crop_size:
            logging.warning(
                f"CleanProfiledRGBNoisyBayerImageCropsValidationDataset.__getitem__: not enough pixels in {crop['gt_linrec2020_fpath']}; deleting from dataset"
            )
            self._dataset[i]["crops"].remove(crop)
            return self.__getitem__(i)
        output = {
            "x_crops": x_crop.float(),
            "y_crops": y_crop.float(),
            "mask_crops": mask_crop,
            "rgb_xyz_matrix": torch.tensor(image["rgb_xyz_matrix"]),
            "gt_fpath": crop["gt_linrec2020_fpath"],
            "y_fpath": crop["f_bayer_fpath"],
        }
        if self.match_gain:
            output["y_crops"] *= image["raw_gain"]
            output["gain"] = 1.0
        else:
            output["gain"] = image["raw_gain"]
        return output


class CleanProfiledRGBNoisyBayerImageCropsTestDataloader(
    CleanProfiledRGBNoisyBayerImageCropsDataset, TestDataLoader
):
    """Dataloader of clean (profiled RGB) - noisy (Bayer) images crops from rawNIND."""

    def __init__(
        self,
        content_fpaths: list[str],
        crop_size: int,
        # test_reserve: list[str],
        test_reserve,  # python38 incompat
        bayer_only: bool,
        alignment_max_loss: float = ALIGNMENT_MAX_LOSS,
        mask_mean_min: float = MASK_MEAN_MIN,
        toy_dataset=False,
        match_gain: bool = False,
        min_msssim_score: Optional[float] = 0.0,
        max_msssim_score: Optional[float] = 1.0,
    ):
        super().__init__(
            content_fpaths=content_fpaths,
            num_crops=1,
            crop_size=crop_size,
            test_reserve=test_reserve,
            alignment_max_loss=alignment_max_loss,
            mask_mean_min=mask_mean_min,
            test=True,
            bayer_only=bayer_only,
            toy_dataset=toy_dataset,
            match_gain=match_gain,
            min_msssim_score=min_msssim_score,
            max_msssim_score=max_msssim_score,
        )
        # take dataset of images and add coordinates

    def get_images(self):
        """Yield test images one crop at a time. Replaces __getitem__ s.t. the image is not re-loaded many times."""
        for image in self._dataset:
            rgb_xyz_matrix = torch.tensor(image["rgb_xyz_matrix"])
            for crop in image["crops"]:
                gt_img = pt_helpers.fpath_to_tensor(crop["gt_linrec2020_fpath"]).float()
                noisy_img = pt_helpers.fpath_to_tensor(crop["f_bayer_fpath"]).float()
                # gt_img = self.crop_rgb_to_bayer(gt_img, metadata)

                gt_img, noisy_img = rawproc.shift_images(
                    gt_img, noisy_img, image["best_alignment"]
                )
                whole_img_mask = pt_helpers.fpath_to_tensor(image["mask_fpath"])[
                    :,
                    crop["coordinates"][1] : crop["coordinates"][1] + gt_img.shape[1],
                    crop["coordinates"][0] : crop["coordinates"][0] + gt_img.shape[2],
                ].expand(gt_img.shape)

                height, width = gt_img.shape[-2:]
                if self.match_gain:
                    noisy_img *= image["raw_gain"]
                    out_gain = 1.0
                else:
                    out_gain = image["raw_gain"]
                if self.crop_size == 0:
                    height = height - height % 256
                    width = width - width % 256
                    if height == 0 or width == 0:
                        continue
                    yield (
                        {
                            "x_crops": gt_img[
                                ...,
                                :height,
                                :width,
                            ].unsqueeze(0),
                            "y_crops": noisy_img[
                                ...,
                                : height // 2,
                                : width // 2,
                            ].unsqueeze(0),
                            "mask_crops": whole_img_mask[
                                ...,
                                :height,
                                :width,
                            ].unsqueeze(0),
                            "rgb_xyz_matrix": rgb_xyz_matrix.unsqueeze(0),
                            "gt_fpath": crop["gt_linrec2020_fpath"],
                            "y_fpath": crop["f_bayer_fpath"],
                            "gain": torch.tensor(out_gain),
                        }
                    )
                else:
                    y = x = 0
                    while y < height:
                        while x < width:
                            if (
                                y + self.crop_size <= height
                                and x + self.crop_size <= width
                            ):
                                yield (
                                    {
                                        "x_crops": gt_img[
                                            ...,
                                            y : y + self.crop_size,
                                            x : x + self.crop_size,
                                        ].unsqueeze(0),
                                        "y_crops": noisy_img[
                                            ...,
                                            y // 2 : y // 2 + self.crop_size // 2,
                                            x // 2 : x // 2 + self.crop_size // 2,
                                        ].unsqueeze(0),
                                        "mask_crops": whole_img_mask[
                                            ...,
                                            y : y + self.crop_size,
                                            x : x + self.crop_size,
                                        ].unsqueeze(0),
                                        "rgb_xyz_matrix": rgb_xyz_matrix.unsqueeze(0),
                                        "gt_fpath": crop["gt_linrec2020_fpath"],
                                        "y_fpath": crop["f_bayer_fpath"],
                                        "gain": torch.tensor(out_gain),
                                    }
                                )
                            x += self.crop_size
                        x = 0
                        y += self.crop_size


class CleanProfiledRGBNoisyProfiledRGBImageCropsTestDataloader(
    CleanProfiledRGBNoisyProfiledRGBImageCropsDataset, TestDataLoader
):
    """Dataloader of clean (profiled RGB) - noisy (profiled RGB) images crops from rawNIND."""

    def __init__(
        self,
        content_fpaths: list[str],
        crop_size: int,
        # test_reserve: list[str],
        test_reserve,  # python38 incompat
        bayer_only: bool,
        alignment_max_loss: float = ALIGNMENT_MAX_LOSS,
        mask_mean_min: float = MASK_MEAN_MIN,
        toy_dataset=False,
        match_gain: bool = False,
        arbitrary_proc_method: bool = False,
        min_msssim_score: Optional[float] = 0.0,
        max_msssim_score: Optional[float] = 1.0,
    ):
        super().__init__(
            content_fpaths=content_fpaths,
            num_crops=1,
            crop_size=crop_size,
            test_reserve=test_reserve,
            alignment_max_loss=alignment_max_loss,
            mask_mean_min=mask_mean_min,
            test=True,
            bayer_only=bayer_only,
            toy_dataset=toy_dataset,
            match_gain=match_gain,
            arbitrary_proc_method=arbitrary_proc_method,
            min_msssim_score=min_msssim_score,
            max_msssim_score=max_msssim_score,
        )

    def get_images(self):
        """Yield test images one crop at a time. Replaces __getitem__ s.t. the image is not re-loaded many times."""
        for image in self._dataset:
            for crop in image["crops"]:
                gt_img = pt_helpers.fpath_to_tensor(crop["gt_linrec2020_fpath"]).float()
                noisy_img = pt_helpers.fpath_to_tensor(
                    crop["f_linrec2020_fpath"]
                ).float()

                gt_img, noisy_img = rawproc.shift_images(
                    gt_img, noisy_img, image["best_alignment"]
                )
                whole_img_mask = pt_helpers.fpath_to_tensor(image["mask_fpath"])[
                    :,
                    crop["coordinates"][1] : crop["coordinates"][1] + gt_img.shape[1],
                    crop["coordinates"][0] : crop["coordinates"][0] + gt_img.shape[2],
                ].expand(gt_img.shape)
                height, width = gt_img.shape[-2:]
                if self.match_gain:
                    noisy_img *= image["rgb_gain"]
                    out_gain = 1.0
                else:
                    out_gain = image["rgb_gain"]

                if self.arbitrary_proc_method:
                    gt_img = arbitrary_proc_fun.arbitrarily_process_images(
                        gt_img,
                        randseed=crop["gt_linrec2020_fpath"],
                        method=self.arbitrary_proc_method,
                    )
                    noisy_img = arbitrary_proc_fun.arbitrarily_process_images(
                        noisy_img,
                        randseed=crop["gt_linrec2020_fpath"],
                        method=self.arbitrary_proc_method,
                    )
                if self.crop_size == 0:
                    height = height - height % 256
                    width = width - width % 256
                    if height == 0 or width == 0:
                        continue
                    yield (
                        {
                            "x_crops": gt_img[
                                ...,
                                :height,
                                :width,
                            ].unsqueeze(0),
                            "y_crops": noisy_img[
                                ...,
                                :height,
                                :width,
                            ].unsqueeze(0),
                            "mask_crops": whole_img_mask[
                                ...,
                                :height,
                                :width,
                            ].unsqueeze(0),
                            "gt_fpath": crop["gt_linrec2020_fpath"],
                            "y_fpath": crop["f_linrec2020_fpath"],
                            "gain": torch.tensor(out_gain),
                        }
                    )
                else:
                    x = y = 0
                    while y < height:
                        while x < width:
                            if (
                                y + self.crop_size <= height
                                and x + self.crop_size <= width
                            ):
                                yield (
                                    {
                                        "x_crops": gt_img[
                                            ...,
                                            y : y + self.crop_size,
                                            x : x + self.crop_size,
                                        ].unsqueeze(0),
                                        "y_crops": noisy_img[
                                            ...,
                                            y : y + self.crop_size,
                                            x : x + self.crop_size,
                                        ].unsqueeze(0),
                                        "mask_crops": whole_img_mask[
                                            ...,
                                            y : y + self.crop_size,
                                            x : x + self.crop_size,
                                        ].unsqueeze(0),
                                        "gt_fpath": crop["gt_linrec2020_fpath"],
                                        "y_fpath": crop["f_linrec2020_fpath"],
                                        "gain": torch.tensor(out_gain),
                                    }
                                )
                            x += self.crop_size
                        x = 0
                        y += self.crop_size


class DataLoadersUnitTests(unittest.TestCase):
    def test_CleanProfiledRGBNoisyBayerImageCropsDataset(self):
        pretime = time.time()
        ds = CleanProfiledRGBNoisyBayerImageCropsDataset(
            content_fpath=rawproc.RAWNIND_CONTENT_FPATH,
            num_crops=4,
            crop_size=256,
            test_reserve=["MuseeL-Bobo-alt-A7C", "MuseeL-yombe-A7C"],
        )
        for i in (0, -1):
            image = ds[i]
            self.assertEqual(image["x_crops"].shape, (4, 3, 256, 256))
            self.assertEqual(image["y_crops"].shape, (4, 4, 128, 128))
            self.assertEqual(image["mask_crops"].shape, (4, 3, 256, 256))
            self.assertEqual(image["rgb_xyz_matrix"].shape, (4, 3))
            self.assertNotEqual(image["gain"], 1.0)
        print(
            f"Time to load CleanProfiledRGBNoisyBayerImageCropsDataset dataset: {time.time() - pretime}"
        )

    def test_CleanProfiledRGBNoisyProfiledRGBImageCropsDataset(self):
        pretime = time.time()
        ds = CleanProfiledRGBNoisyProfiledRGBImageCropsDataset(
            content_fpaths=[rawproc.RAWNIND_CONTENT_FPATH],
            num_crops=4,
            crop_size=256,
            test_reserve=["MuseeL-Bobo-alt-A7C", "MuseeL-yombe-A7C"],
        )
        for i in (0, -1):
            image = ds[i]
            self.assertEqual(image["x_crops"].shape, (4, 3, 256, 256))
            self.assertEqual(image["y_crops"].shape, (4, 3, 256, 256))
            self.assertEqual(image["mask_crops"].shape, (4, 3, 256, 256))
            self.assertNotEqual(image["gain"], 1.0)
        print(
            f"Time to load CleanProfiledRGBNoisyProfiledRGBImageCropsDataset dataset: {time.time() - pretime}"
        )

    def test_CleanProfiledRGBCleanBayerImageCropsDataset(self):
        pretime = time.time()
        ds = CleanProfiledRGBCleanBayerImageCropsDataset(
            content_fpaths=rawproc.EXTRARAW_CONTENT_FPATHS, num_crops=4, crop_size=256
        )
        print(
            f"Time to load CleanProfiledRGBCleanBayerImageCropsDataset dataset: {time.time() - pretime}"
        )
        pretime = time.time()
        for i in (0, -1):
            image = ds[i]
            self.assertEqual(image["x_crops"].shape, (4, 3, 256, 256))
            self.assertEqual(image["y_crops"].shape, (4, 4, 128, 128))
            self.assertEqual(image["mask_crops"].shape, (4, 3, 256, 256))
            self.assertEqual(image["rgb_xyz_matrix"].shape, (4, 3))
            self.assertEqual(image["gain"], 1.0)
        for i in range(len(ds)):
            self.assertGreater(
                len(ds._dataset[i]["crops"]), 0, f"{ds._dataset[i]} has no crops."
            )
            acrop = random.choice(ds._dataset[i]["crops"])
        print(
            f"Time to check CleanProfiledRGBCleanBayerImageCropsDataset dataset: {time.time() - pretime}"
        )

    def test_CleanProfiledRGBCleanProfiledRGBImageCropsDataset(self):
        pretime = time.time()
        ds = CleanProfiledRGBCleanProfiledRGBImageCropsDataset(
            content_fpaths=rawproc.EXTRARAW_CONTENT_FPATHS, num_crops=4, crop_size=256
        )
        for i in (0, -1):
            image = ds[i]
            self.assertEqual(image["x_crops"].shape, (4, 3, 256, 256))
            self.assertEqual(image["mask_crops"].shape, (4, 3, 256, 256))
            self.assertEqual(image["gain"], 1.0)
        print(
            f"Time to load CleanProfiledRGBCleanProfiledRGBImageCropsDataset dataset: {time.time() - pretime}"
        )

    def test_CleanProfiledRGBNoisyProfiledRGBImageCropsValidationDataset(self):
        pretime = time.time()
        ds = CleanProfiledRGBNoisyProfiledRGBImageCropsValidationDataset(
            content_fpaths=[rawproc.RAWNIND_CONTENT_FPATH],
            crop_size=256,
            test_reserve=["MuseeL-Bobo-alt-A7C", "MuseeL-yombe-A7C"],
        )
        for i in (0, -1):
            image = ds[i]
            self.assertEqual(image["x_crops"].shape, (3, 256, 256))
            self.assertEqual(image["y_crops"].shape, (3, 256, 256))
            self.assertEqual(image["mask_crops"].shape, (3, 256, 256))
            self.assertNotEqual(image["gain"], 1.0)
        print(
            f"Time to load CleanProfiledRGBNoisyProfiledRGBImageCropsValidationDataset dataset: {time.time() - pretime}"
        )

    def test_CleanProfiledRGBNoisyBayerImageCropsValidationDataset(self):
        pretime = time.time()
        test_reserve = [
            "ursulines-red",
            "stefantiek",
            "ursulines-building",
            "MuseeL-Bobo",
            "CourtineDeVillersDebris",
            "Vaxt-i-trad",
            "Pen-pile",
            "MuseeL-vases",
        ]
        ds = CleanProfiledRGBNoisyBayerImageCropsValidationDataset(
            content_fpath=rawproc.RAWNIND_CONTENT_FPATH,
            crop_size=256,
            test_reserve=test_reserve,
        )
        print(
            f"Time to load CleanProfiledRGBNoisyBayerImageCropsValidationDataset dataset: {time.time() - pretime}"
        )
        pretime = time.time()
        for i in (0, -1):
            image = ds[i]
            self.assertEqual(image["x_crops"].shape, (3, 256, 256))
            self.assertEqual(image["y_crops"].shape, (4, 128, 128))
            self.assertEqual(image["mask_crops"].shape, (3, 256, 256))
            self.assertEqual(image["rgb_xyz_matrix"].shape, (4, 3))
        for imagedict in ds:
            self.assertGreaterEqual(imagedict["x_crops"].shape[-1], 256)
            self.assertGreaterEqual(imagedict["x_crops"].shape[-2], 256)
            self.assertNotEqual(image["gain"], 1.0)
        print(
            f"Time to check CleanProfiledRGBNoisyBayerImageCropsValidationDataset dataset: {time.time() - pretime}"
        )

    def test_CleanProfiledRGBNoisyBayerImageCropsTestDataloader(self):
        MAX_ITERS = 20
        pretime = time.time()
        ds = CleanProfiledRGBNoisyBayerImageCropsTestDataloader(
            content_fpath=rawproc.RAWNIND_CONTENT_FPATH,
            crop_size=256,
            test_reserve=["MuseeL-Bobo-alt-A7C", "MuseeL-yombe-A7C"],
        )
        for i, output in enumerate(ds.get_images()):
            self.assertEqual(output["x_crops"].shape, (1, 3, 256, 256))
            self.assertEqual(output["y_crops"].shape, (1, 4, 128, 128))
            self.assertEqual(output["mask_crops"].shape, (1, 3, 256, 256))
            self.assertEqual(output["rgb_xyz_matrix"].shape, (1, 4, 3))
            self.assertNotEqual(image["gain"], 1.0)
            if i >= MAX_ITERS:
                break
        print(
            f"Time to run {min(MAX_ITERS, i)} iterations of CleanProfiledRGBNoisyBayerImageCropsTestDataloader dataset: {time.time() - pretime}"
        )

    def test_CleanProfiledRGBNoisyProfiledRGBImageCropsTestDataloader(self):
        MAX_ITERS = 20
        pretime = time.time()
        ds = CleanProfiledRGBNoisyProfiledRGBImageCropsTestDataloader(
            content_fpath=rawproc.RAWNIND_CONTENT_FPATH,
            crop_size=256,
            test_reserve=["MuseeL-Bobo-alt-A7C", "MuseeL-yombe-A7C"],
        )
        for i, output in enumerate(ds.get_images()):
            self.assertEqual(output["x_crops"].shape, (1, 3, 256, 256))
            self.assertEqual(output["mask_crops"].shape, (1, 3, 256, 256))
            self.assertNotEqual(image["gain"], 1.0)
            if i >= MAX_ITERS:
                break
        print(
            f"Time to run {min(MAX_ITERS, i)} iterations of CleanProfiledRGBNoisyProfiledRGBImageCropsTestDataloader dataset: {time.time() - pretime}"
        )


if __name__ == "__main__":
    # the usual logging init
    logging.basicConfig(
        filename=LOG_FPATH,
        format="%(message)s",
        level=logging.INFO,
        filemode="w",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f'# python {" ".join(sys.argv)}')

    cleanRGB_noisyBayer_ds = CleanProfiledRGBNoisyBayerImageDataset(
        content_fpaths=[rawproc.RAWNIND_CONTENT_FPATH], num_crops=4, crop_size=256
    )
    cleanRGB_noisyRGB_ds = CleanProfiledRGBNoisyProfiledRGBImageDataset(
        content_fpaths=[rawproc.RAWNIND_CONTENT_FPATH], num_crops=4, crop_size=256
    )
