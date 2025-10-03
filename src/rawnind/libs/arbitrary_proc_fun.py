import os
import random

# cv2.setNumThreads(0)
import time
import unittest
from typing import Literal

# import multiprocessing
# multiprocessing.set_start_method('spawn')
import cv2
import numpy as np
import torch
import torchvision

# sys.path.append("..")
from rawnind.libs import rawproc, raw

TONEMAPPING_FUN: Literal["reinhard", "drago", "log"] = "log"
# Reinhard tonemapping parameters range
TONEMAPPING_PARAMS_RANGES = {
    "reinhard": {
        "min": {"gamma": 0.9, "intensity": -0.25, "light_adapt": 1, "color_adapt": 0},
        "max": {"gamma": 1.1, "intensity": 0.25, "light_adapt": 1, "color_adapt": 0},
    },
    "drago": {
        "min": {"gamma": 0.9, "saturation": 0.75, "bias": 0.7},
        "max": {"gamma": 1.1, "saturation": 1.4, "bias": 0.9},
    },
    "log": {"min": {}, "max": {}},
}
TONEMAPPING_PARAMS_RANGE = TONEMAPPING_PARAMS_RANGES[TONEMAPPING_FUN]

EDGES_ENHANCEMENT_PARAMS_RANGE = {"min": {"alpha": 0.01}, "max": {"alpha": 0.5}}
GAMMA_CORRECTION_PARAMS_RANGE = {"min": {"gamma": 1.8}, "max": {"gamma": 2.2}}
CONTRAST_PARAMS_RANGE = {
    "min": {"clipLimit": 2.5, "tileGridSize": 16},
    "max": {"clipLimit": 3.5, "tileGridSize": 48},
}
SIGMOID_CONTRAST_ENHANCEMENT_PARAMS_RANGE = {
    "min": {"gain": 5, "cutoff": 0.3},
    "max": {"gain": 20, "cutoff": 0.7},
}
SHARPEN_PARAMS_RANGE = {
    "min": {"kernel_size": 3, "sigma": 0.5},
    "max": {"kernel_size": 5, "sigma": 1.5},
}
ARBITRARY_PROC_PARAMS_RANGE = {
    "edges_enhancement": EDGES_ENHANCEMENT_PARAMS_RANGE,
    "sharpen": SHARPEN_PARAMS_RANGE,
    "tonemapping": TONEMAPPING_PARAMS_RANGE,
    "gamma_correction": GAMMA_CORRECTION_PARAMS_RANGE,
    "contrast": CONTRAST_PARAMS_RANGE,
    "sigmoid_contrast_enhancement": SIGMOID_CONTRAST_ENHANCEMENT_PARAMS_RANGE,
}


# TONEMAPPING_PARAMS_RANGE = {'min': {'gamma': 0.8, 'intensity': -0.25, 'light_adapt': 1, 'color_adapt': 0},
#                            'max': {'gamma': 1, 'intensity': 0.25, 'light_adapt': 1, 'color_adapt': 0.4}}
# EDGES_ENHANCEMENT_PARAMS_RANGE = {'min': {'alpha': 0.5}, 'max': {'alpha': 1.5}}
# GAMMA_CORRECTION_PARAMS_RANGE = {'min': {'gamma': 1.0}, 'max': {'gamma': 1.2}}
# CONTRAST_PARAMS_RANGE = {'min': {'clipLimit': 2.5, 'tileGridSize': 16}, 'max': {'clipLimit': 3.5, 'tileGridSize': 48}}
# SIGMOID_CONTRAST_ENHANCEMENT_PARAMS_RANGE = {'min': {'gain': 5, 'cutoff': 0.3}, 'max': {'gain': 20, 'cutoff': 0.7}}
# SHARPEN_PARAMS_RANGE = {'min': {'kernel_size': 3, 'sigma': 0.5}, 'max': {'kernel_size': 5, 'sigma': 1.5}}
# ARBITRARY_PROC_PARAMS_RANGE = {'tonemapping': TONEMAPPING_PARAMS_RANGE,
#                                'edges_enhancement': EDGES_ENHANCEMENT_PARAMS_RANGE,
#                                'gamma_correction': GAMMA_CORRECTION_PARAMS_RANGE,
#                                'contrast': CONTRAST_PARAMS_RANGE,
#                                'sigmoid_contrast_enhancement': SIGMOID_CONTRAST_ENHANCEMENT_PARAMS_RANGE,
#                                'sharpen': SHARPEN_PARAMS_RANGE}


def correct_white_balance(img):
    """
    Correct white balance using the Gray World assumption for an image in float32 format, normalized to [0, 1].
    """
    # print(f"pre-wb: min: {np.min(img)}, max: {np.max(img)}, mean: {np.mean(img)}")
    # Calculate the mean of each channel
    mean_r = np.mean(img[:, :, 2].clip(0, 1))
    mean_g = np.mean(img[:, :, 1].clip(0, 1))
    mean_b = np.mean(img[:, :, 0].clip(0, 1))

    # Compute the gain for each channel
    K = (mean_r + mean_g + mean_b) / 3
    Kr = K / mean_r
    Kg = K / mean_g
    Kb = K / mean_b

    # # Apply the gains
    # img[:, :, 2] = np.clip(img[:, :, 2] * Kr, 0, 1)
    # img[:, :, 1] = np.clip(img[:, :, 1] * Kg, 0, 1)
    # img[:, :, 0] = np.clip(img[:, :, 0] * Kb, 0, 1)
    img[:, :, 2] = img[:, :, 2] * Kr
    img[:, :, 1] = img[:, :, 1] * Kg
    img[:, :, 0] = img[:, :, 0] * Kb
    return img


def apply_tone_mapping_reinhard(
    img, gamma=1.0, intensity=0.0, light_adapt=1.0, color_adapt=0.0
):
    """
    Apply tone mapping to the image. Since the image is already in float32 format, we can directly use it.
    """
    # Apply tone mapping
    # print range of values before
    # print(f"pre-tm: min: {np.min(img)}, max: {np.max(img)}, mean: {np.mean(img)}")
    tonemap = cv2.createTonemapReinhard(
        gamma=gamma,
        intensity=intensity,
        light_adapt=light_adapt,
        color_adapt=color_adapt,
    )
    # tonemap = cv2.TonemapDrago(gamma=gamma, saturation=saturation, bias=bias)
    print(img.dtype)
    # img = (img * 255).clip(0, 255).astype(np.uint8)
    ldr = tonemap.process(img)  # This is already normalized to [0, 1]
    # img = ldr.astype(np.float32) / 255
    # print(f"min: {np.min(img)}, max: {np.max(img)}, mean: {np.mean(img)}")
    return ldr


def apply_tone_mapping_drago(img, gamma=1.0, saturation=1.0, bias=0.85):
    """
    Apply tone mapping to the image. Since the image is already in float32 format, we can directly use it.
    """
    # Apply tone mapping
    # print range of values before
    # print(f"pre-tm: min: {np.min(img)}, max: {np.max(img)}, mean: {np.mean(img)}")

    tonemap = cv2.TonemapDrago(gamma=gamma, saturation=saturation, bias=bias)
    print(img.dtype)
    # img = (img * 255).clip(0, 255).astype(np.uint8)
    # img = img.clip(0, 1)
    ldr = tonemap.process(img)  # This is already normalized to [0, 1]
    # img = ldr.astype(np.float32) / 255
    # print(f"min: {np.min(img)}, max: {np.max(img)}, mean: {np.mean(img)}")
    return ldr


# def apply_tone_mapping_log(img, epsilon = 0.001):
#     yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
#     L_max = np.max(yuv_img[:,:,0])
#     tone_mapped_Y = np.log(1+yuv_img[:,:,0]) / np.log(1+L_max+epsilon)
#     tone_mapped_yuv = np.stack([tone_mapped_Y, yuv_img[:,:,1], yuv_img[:,:,2]], axis=-1)
#     return cv2.cvtColor(tone_mapped_yuv, cv2.COLOR_YUV2BGR)
def apply_tone_mapping_log(img, epsilon=0.001):
    yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    L_max = np.max(yuv_img[:, :, 0])
    tone_mapped_Y = np.log((1 + yuv_img[:, :, 0]).clip(epsilon)) / np.log(
        (1 + L_max + epsilon).clip(epsilon)
    )
    # Normalize tone-mapped Y to the range [0, 1]
    tone_mapped_Y = (tone_mapped_Y - np.min(tone_mapped_Y)) / (
        np.max(tone_mapped_Y) - np.min(tone_mapped_Y)
    )
    tone_mapped_yuv = np.stack(
        [tone_mapped_Y, yuv_img[:, :, 1], yuv_img[:, :, 2]], axis=-1
    )
    return cv2.cvtColor(tone_mapped_yuv, cv2.COLOR_YUV2BGR)


def apply_gamma_correction_inplace(img, gamma=2.2):
    """
    Apply gamma correction to the image in float32 format.
    """
    img[img > 0] = img[img > 0] ** (1.0 / gamma)
    return img
    # return np.clip(img_corrected, 0, 1)


def adjust_contrast(img, clipLimit=2.0, tileGridSize=8):
    """
    Apply CLAHE for contrast adjustment with proper data type handling.
    """
    tileGridSize = (int(tileGridSize), int(tileGridSize))
    # Ensure the image is in the correct format and range
    if img.dtype != np.uint8:
        img_8bit = np.clip(img * 255, 0, 255).astype("uint8")
    else:
        img_8bit = img

    # Convert to YUV color space to apply CLAHE on the Luminance channel
    lab = cv2.cvtColor(img_8bit, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    # Convert back to RGB
    adjusted_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Convert back to float32 if the original was float32
    if img.dtype == np.float32:
        adjusted_img = adjusted_img.astype("float32") / 255

    return adjusted_img


def sigmoid_contrast_enhancement(img, gain=10, cutoff=0.5):
    """
    Enhance contrast using a sigmoid function for smooth and controlled adjustment.
    """
    # img_float = img.astype(np.float32) / 255.0  # ensure image is in float format
    # Apply the sigmoid correction
    return 1 / (1 + np.exp(np.clip(gain * (cutoff - img), -50, 50)))
    # corrected = np.clip(corrected * 255, 0, 255).astype(np.uint8)


# def sharpen_image(img, kernel_size=5, sigma=1.0):
#     """
#     Sharpen the image. The image needs to be converted to 8-bit format temporarily for Gaussian blur.
#     """
#     kernel_size = (int(kernel_size), int(kernel_size))
#     img_8bit = np.clip(img * 255, 0, 255).astype("uint8")
#     blurred = cv2.GaussianBlur(img_8bit, kernel_size, sigma)
#     sharpened = cv2.addWeighted(img_8bit, 1.5, blurred, -0.5, 0)

#     return sharpened.astype("float32") / 255


def sharpen_image(img, kernel_size=5, sigma=1.0):
    """
    Sharpen the image using floating point precision throughout.
    """
    kernel_size = (int(kernel_size), int(kernel_size))
    blurred = cv2.GaussianBlur(img, kernel_size, sigma)
    sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    return sharpened
    # Ensure output is still in the float format
    return np.clip(sharpened, 0, 1).astype("float32")


# def enhance_edges(img, alpha=1.0):
#     """
#     Enhance edges in the image using Laplacian for edge detection and then blending with the original.

#     :param img: Input image in float32 format normalized to [0, 1].
#     :param alpha: Controls the amount of edge enhancement. Higher values mean more enhancement.
#     :return: Edge-enhanced image, still in float32 format normalized to [0, 1].
#     """
#     # Convert to 8-bit format for the Laplacian operation
#     img_8bit = np.clip(img * 255, 0, 255).astype("uint8")

#     # Apply Laplacian filter to detect edges
#     laplacian = cv2.Laplacian(img_8bit, cv2.CV_64F)

#     # Convert back to float32 and normalize to [0, 1]
#     laplacian_normalized = np.clip((laplacian / 255), 0, 1).astype("float32")

#     # Enhance edges by adding the Laplacian (scaled by alpha) back to the original image
#     enhanced = img + alpha * laplacian_normalized

#     # Clip the result to ensure it's within [0, 1]
#     enhanced_clipped = np.clip(enhanced, 0, 1)

#     return enhanced_clipped


def enhance_edges(img, alpha=1.0):
    """
    Enhance edges in the image using Laplacian for edge detection and then blending with the original.
    All operations are done in floating point precision.
    """
    yuv_img = cv2.cvtColor(img.clip(0, 1), cv2.COLOR_BGR2YUV)
    # Apply Laplacian filter to detect edges
    laplacian_y = cv2.Laplacian(img[:, :, 0], cv2.CV_32F)

    # Enhance edges by adding the Laplacian (scaled by alpha) back to the original image
    yuv_img[:, :, 0] += alpha * laplacian_y

    return cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)
    # Clip the result to ensure it's within [0, 1]
    return np.clip(enhanced, 0, 1).astype("float32")


def replace_nan_with_nearest(output_img):
    # Check if the tensor has a batch dimension
    has_batch = output_img.dim() == 4

    # If there's no batch dimension, add one
    if not has_batch:
        output_img = output_img.unsqueeze(0)

    # Create a mask where the NaNs are True
    nan_mask = torch.isnan(output_img)

    # Replace NaNs temporarily with 0 for computation
    temp_img = output_img.clone()
    temp_img[nan_mask] = 0

    # Kernel for convolution
    kernel_size = 3
    kernel = torch.ones(
        (1, 1, kernel_size, kernel_size),
        device=output_img.device,
        dtype=output_img.dtype,
    )

    # Expand the kernel to operate across all channels as a group
    kernel = kernel.repeat(output_img.shape[1], 1, 1, 1)

    # Count valid (non-NaN) neighbors for each element
    valid_neighbors = torch.nn.functional.conv2d(
        temp_img, kernel, padding=1, groups=output_img.shape[1]
    )

    # Sum of valid neighbors' values
    neighbor_sum = torch.nn.functional.conv2d(
        temp_img, kernel, padding=1, groups=output_img.shape[1]
    )

    # Compute the mean from neighbor sum and valid neighbor count
    neighbor_mean = neighbor_sum / valid_neighbors

    # Place computed means where there were NaNs
    output_img[nan_mask] = neighbor_mean[nan_mask]

    # Remove batch dimension if it was added
    if not has_batch:
        output_img = output_img.squeeze(0)

    return output_img


def arbitrarily_process_images_opencv(
    lin_rgb_img: torch.Tensor, randseed=None, enable_all=False
) -> torch.Tensor:
    if randseed:
        random.seed(randseed)
    # ensure a batch
    if len(lin_rgb_img.shape) == 3:
        lin_rgb_img = lin_rgb_img.unsqueeze(0)
        is_batched = False
    else:
        is_batched = True
    orig_dtype = lin_rgb_img.dtype
    output_img = torch.empty_like(lin_rgb_img)
    # convert pytorch tensor image batch to opencv-compatible
    lin_rgb_img = lin_rgb_img.permute(0, 2, 3, 1).cpu().numpy()
    # convert to float
    lin_rgb_img = lin_rgb_img.astype(np.float32)
    for i, img in enumerate(lin_rgb_img):
        # convert RGB format
        # pretime = time.time()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # print(f"convert RGB format: {time.time() - pretime}")
        # white balance
        # pretime = time.time()
        assert not np.isnan(img).any(), "nan pre-wb"  # DBG
        if random.getrandbits(3) or enable_all:
            img = correct_white_balance(img)
            # check if there are no nan values in the numpy array
            assert not np.isnan(img).any(), "nan post-wb"  # DBG
        # print(f"correct white balance: {time.time() - pretime}")
        # tone mapping
        # pretime = time.time()
        if random.getrandbits(3) or enable_all:
            if TONEMAPPING_FUN == "drago":
                img = apply_tone_mapping_drago(
                    img,
                    gamma=random.uniform(
                        TONEMAPPING_PARAMS_RANGE["min"]["gamma"],
                        TONEMAPPING_PARAMS_RANGE["max"]["gamma"],
                    ),
                    saturation=random.uniform(
                        TONEMAPPING_PARAMS_RANGE["min"]["saturation"],
                        TONEMAPPING_PARAMS_RANGE["max"]["saturation"],
                    ),
                    bias=random.uniform(
                        TONEMAPPING_PARAMS_RANGE["min"]["bias"],
                        TONEMAPPING_PARAMS_RANGE["max"]["bias"],
                    ),
                )
            elif TONEMAPPING_FUN == "reinhard":
                img = apply_tone_mapping_reinhard(
                    img,
                    gamma=random.uniform(
                        TONEMAPPING_PARAMS_RANGE["min"]["gamma"],
                        TONEMAPPING_PARAMS_RANGE["max"]["gamma"],
                    ),  # 0.5-1, 1
                    intensity=random.uniform(
                        TONEMAPPING_PARAMS_RANGE["min"]["intensity"],
                        TONEMAPPING_PARAMS_RANGE["max"]["intensity"],
                    ),  # -0.25 to 0.25, 0
                    light_adapt=random.uniform(
                        TONEMAPPING_PARAMS_RANGE["min"]["light_adapt"],
                        TONEMAPPING_PARAMS_RANGE["max"]["light_adapt"],
                    ),  # 1
                    color_adapt=random.uniform(
                        TONEMAPPING_PARAMS_RANGE["min"]["color_adapt"],
                        TONEMAPPING_PARAMS_RANGE["max"]["color_adapt"],
                    ),
                )
            elif TONEMAPPING_FUN == "log":
                img = apply_tone_mapping_log(img)
            else:
                raise ValueError(f"Unknown tonemapping function {TONEMAPPING_FUN}")
            if np.isnan(img).any():
                print(np.isnan(img).sum())
                breakpoint()
            assert not np.isnan(img).any(), "nan post-tonemapping"  # DBG
        # print(f"apply tone mapping: {time.time() - pretime}")
        # edges enhancement
        # pretime = time.time()
        if random.getrandbits(3) or enable_all:
            img = enhance_edges(
                img,
                alpha=random.uniform(
                    EDGES_ENHANCEMENT_PARAMS_RANGE["min"]["alpha"],
                    EDGES_ENHANCEMENT_PARAMS_RANGE["max"]["alpha"],
                ),
            )  # 1 random.uniform(0.1, 1.5))
            assert not np.isnan(img).any(), "nan post-edges"  # DBG
        # print(f"enhance edges: {time.time() - pretime}")
        # gamma
        # pretime = time.time()
        if random.getrandbits(3) or enable_all:
            img = apply_gamma_correction_inplace(
                img,
                gamma=random.uniform(
                    GAMMA_CORRECTION_PARAMS_RANGE["min"]["gamma"],
                    GAMMA_CORRECTION_PARAMS_RANGE["max"]["gamma"],
                ),
            )  # random.uniform(1.8, 2.4)), maybe 1.0-1.8
            assert not np.isnan(img).any(), "nan post-gamma"  # DBG
        # print(f"apply gamma correction: {time.time() - pretime}")
        # contrast
        # pretime = time.time()
        # img = adjust_contrast(
        #     img,
        #     clipLimit=2,  # random.uniform(2.0, 4.0),
        #     tileGridSize=64,  # int(random.uniform(8, 16)),
        # )
        if random.getrandbits(3) or enable_all:
            img = sigmoid_contrast_enhancement(
                img,
                gain=int(
                    random.uniform(
                        SIGMOID_CONTRAST_ENHANCEMENT_PARAMS_RANGE["min"]["gain"],
                        SIGMOID_CONTRAST_ENHANCEMENT_PARAMS_RANGE["max"]["gain"],
                    )
                ),
                cutoff=random.uniform(
                    SIGMOID_CONTRAST_ENHANCEMENT_PARAMS_RANGE["min"]["cutoff"],
                    SIGMOID_CONTRAST_ENHANCEMENT_PARAMS_RANGE["max"]["cutoff"],
                ),
            )  # 5 to 20, 0.3 to 0.7, def 10,.5
            assert not np.isnan(img).any(), "nan post-sigmoid"  # DBG
        # print(f"adjust contrast: {time.time() - pretime}")
        # sharpen
        # pretime = time.time()

        if random.getrandbits(3) or enable_all:
            img = sharpen_image(
                img,
                kernel_size=random.choice((3, 5)),
                sigma=random.uniform(
                    SHARPEN_PARAMS_RANGE["min"]["sigma"],
                    SHARPEN_PARAMS_RANGE["max"]["sigma"],
                ),  # random.choice((3, 5)), sigma=#random.uniform(0.5, 1.5), def 5,1
            )
            assert not np.isnan(img).any(), "nan post-sharpen"  # DBG
        # print(f"sharpen image: {time.time() - pretime}")
        # convert back to RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.clip(0, 1)
        output_img[i] = torch.from_numpy(img).permute(2, 0, 1)

    # output_img = replace_nan_with_nearest(output_img)
    # assert that there are no nan values
    assert not torch.isnan(output_img).any()
    # check for inf and -inf values
    assert not torch.isinf(output_img).any()

    if not is_batched:
        output_img = output_img.squeeze(0)
    output_img = output_img.to(orig_dtype)

    return output_img


def arbitrarily_process_images_naive_python(
    lin_rgb_img: torch.Tensor,
    equalize_weight=0.33,
    contrast=1.25,
    saturation=1.5,
    gamma_val=2,
    sharpness=2,
    randseed=None,
) -> torch.Tensor:
    """
    Arbitrarily process an image (or a batch) to emulate typical raw-rgb conversion
    white balance, exposure, (color calibration), () contrast?, sharpen, color, gamma, contrast?

    Adjust brightness (0.5, 1.5)
    autocontrast
    gamma 2.2
    saturation (1, 1.5)
    adjust_sharpness (1, 2)
    """
    if randseed:
        random.seed(randseed)
        # adjust every parameter by +/- 25%
        equalize_weight = random.uniform(equalize_weight * 0.75, equalize_weight * 1.25)
        contrast = random.uniform(contrast * 0.75, contrast * 1.25)
        saturation = random.uniform(saturation * 0.75, saturation * 1.25)
        gamma_val = random.uniform(gamma_val * 0.75, gamma_val * 1.25)
        sharpness = random.uniform(sharpness * 0.75, sharpness * 1.25)
    img = lin_rgb_img
    img = torchvision.transforms.functional.equalize((img * 255).to(torch.uint8)).to(
        torch.float32
    ) / 255 * equalize_weight + img * (1 - equalize_weight)
    img = torchvision.transforms.functional.adjust_contrast(img, contrast)
    img = torchvision.transforms.functional.adjust_saturation(img, saturation)
    img = rawproc.gamma_pt(img, gamma_val)
    img = torchvision.transforms.functional.adjust_sharpness(img, sharpness)
    img = img.clip(0, 1)
    return img


def arbitrarily_process_images(
    lin_rgb_img: torch.Tensor,
    method: Literal["naive", "opencv"],
    randseed=None,
    **kwargs,
):
    if method == "naive":
        return arbitrarily_process_images_naive_python(
            lin_rgb_img, randseed=randseed, **kwargs
        )
    elif method == "opencv":
        return arbitrarily_process_images_opencv(lin_rgb_img, randseed=randseed)
    else:
        raise ValueError(f"Unknown method {method}")


####### tests


def arbitrary_proc_visual_test():
    from common.libs import pt_helpers

    TEST_IMAGES_FPATHS = (
        "/orb/benoit_phd/datasets/RawNIND/proc/lin_rec2020/D60-1/gt/CRW_6994.CRW.tif",
        "/orb/benoit_phd/datasets/RawNIND/proc/lin_rec2020/Vaxt-i-trad/gt/IMG_8865.CR2.tif",
        "/orb/benoit_phd/datasets/RawNIND/proc/lin_rec2020/7D-2/gt/_M7D5418.CR2.tif",
        "/orb/benoit_phd/datasets/RawNIND/proc/lin_rec2020/MuseeL-vases-A7C/gt/ISO50_capt0014.arw.tif",
        "/orb/benoit_phd/datasets/RawNIND/proc/lin_rec2020/TitusToys/gt/ISO50_capt0001.arw.tif",
        "/orb/benoit_phd/datasets/RawNIND/proc/lin_rec2020/Laura_Lemons_platformer/gt/ISO50_capt0001.arw.tif",
        "/orb/benoit_phd/datasets/RawNIND/proc/lin_rec2020/boardgames_top/gt/ISO50_capt0001.arw.tif",
    )
    OUTPUT_DPATH = os.path.join("tmp", "arbitrary_proc_visual_test")
    os.makedirs(OUTPUT_DPATH, exist_ok=True)
    for img_fpath in TEST_IMAGES_FPATHS:
        pretime = time.time()
        print(f"arbitrary_proc_visual_test: processing {img_fpath}")
        img = pt_helpers.fpath_to_tensor(img_fpath)
        print(f"arbitrary_proc_visual_test: loading time {time.time() - pretime}")
        pretime = time.time()
        # proc_img_naive = arbitrarily_process_images(
        #     img, method="naive", randseed=img_fpath
        # )
        # print(
        #     f"arbitrary_proc_visual_test: naive processing time {time.time() - pretime}"
        # )
        # pretime = time.time()
        proc_img_opencv = arbitrarily_process_images(
            img, method="opencv", randseed=img_fpath
        )
        print(
            f"arbitrary_proc_visual_test: opencv processing time {time.time() - pretime}"
        )

        # pt_helpers.sdr_pttensor_to_file(
        #     proc_img_naive,
        #     os.path.join(OUTPUT_DPATH, f"{os.path.basename(img_fpath)}_naive.tif"),
        # )
        # pt_helpers.sdr_pttensor_to_file(
        #     proc_img_opencv,
        #     os.path.join(OUTPUT_DPATH, f"{os.path.basename(img_fpath)}_opencv.tif"),
        # )
        raw.hdr_nparray_to_file(
            proc_img_opencv,
            os.path.join(OUTPUT_DPATH, f"{os.path.basename(img_fpath)}_opencv.tif"),
            color_profile="lin_rec2020",
        )


def test_arbitrarily_process_images_opencv_values(infpath, outdpath):
    """Test all possible values for the arbitrary processing functions by modifying the global variables defining ranges."""
    # create a backup of the default values
    import hashlib
    import copy
    from common.libs import pt_helpers

    global ARBITRARY_PROC_PARAMS_RANGE
    ARBITRARY_PROC_PARAMS_RANGE_bak = copy.deepcopy(ARBITRARY_PROC_PARAMS_RANGE)

    with open(infpath, "rb") as f:
        file_content = f.read()
        file_hash = hashlib.md5(file_content).hexdigest()

    # First, test with all parameters set to their midpoints
    for param_type, param_ranges in ARBITRARY_PROC_PARAMS_RANGE_bak.items():
        for param_name in param_ranges["min"].keys():
            midpoint = (
                param_ranges["min"][param_name] + param_ranges["max"][param_name]
            ) / 2
            ARBITRARY_PROC_PARAMS_RANGE[param_type]["min"][param_name] = midpoint
            ARBITRARY_PROC_PARAMS_RANGE[param_type]["max"][param_name] = midpoint

    # Process the image with all parameters set to midpoints and save the result
    output_fpath = os.path.join(
        outdpath, f"{os.path.basename(infpath)}_{file_hash}_midpoints.tif"
    )
    img = pt_helpers.fpath_to_tensor(infpath)
    proc_img = arbitrarily_process_images_opencv(img, enable_all=True)
    # pt_helpers.sdr_pttensor_to_file(proc_img, output_fpath)
    raw.hdr_nparray_to_file(proc_img, output_fpath, color_profile="lin_rec2020")
    print(f"Saved midpoints to {output_fpath}")

    # Then, test each parameter's min and max while keeping other parameters at their midpoints
    for param_type, param_ranges in ARBITRARY_PROC_PARAMS_RANGE_bak.items():
        for param_name in param_ranges["min"].keys():
            for other_param in param_ranges["min"].keys():
                if other_param != param_name:
                    mid_min = (
                        param_ranges["min"][other_param]
                        + param_ranges["max"][other_param]
                    ) / 2
                    ARBITRARY_PROC_PARAMS_RANGE[param_type]["min"][other_param] = (
                        mid_min
                    )
                    ARBITRARY_PROC_PARAMS_RANGE[param_type]["max"][other_param] = (
                        mid_min
                    )

            # Test the parameter's min and max
            for min_or_max in ("min", "max"):
                value = param_ranges[min_or_max][param_name]
                ARBITRARY_PROC_PARAMS_RANGE[param_type]["min"][param_name] = value
                ARBITRARY_PROC_PARAMS_RANGE[param_type]["max"][param_name] = value
                output_fpath = os.path.join(
                    outdpath,
                    f"{os.path.basename(infpath)}_{file_hash}_{param_type}_{param_name}_{min_or_max}={value}.tif",
                )
                img = pt_helpers.fpath_to_tensor(infpath)
                proc_img = arbitrarily_process_images_opencv(img, enable_all=True)
                # pt_helpers.sdr_pttensor_to_file(proc_img, output_fpath)
                raw.hdr_nparray_to_file(
                    proc_img, output_fpath, color_profile="lin_rec2020"
                )

                print(f"Saved {param_name} {min_or_max} to {output_fpath}")

    # Restore original parameter ranges
    ARBITRARY_PROC_PARAMS_RANGE = ARBITRARY_PROC_PARAMS_RANGE_bak


class Test_Rawproc(unittest.TestCase):
    def test_arbitrary_proc_single_and_batch_equivalence(self):
        from common.libs import pt_helpers

        TEST_IMAGES_FPATHS = (
            "../../datasets/RawNIND/proc/lin_rec2020/gnome/gt/DSCF1051.RAF.exr",
            "/orb/benoit_phd/datasets/RawNIND/proc/lin_rec2020/stefantiek/gt/DSCF3916.RAF.exr",
        )
        img1 = pt_helpers.fpath_to_tensor(TEST_IMAGES_FPATHS[0])
        img2 = pt_helpers.fpath_to_tensor(TEST_IMAGES_FPATHS[1])
        batch_proc_img = arbitrarily_process_images(torch.stack((img1, img2)))
        proc_img1 = arbitrarily_process_images(img1)
        proc_img2 = arbitrarily_process_images(img2)
        self.assertTrue(torch.allclose(proc_img1, batch_proc_img[0]))
        self.assertTrue(torch.allclose(proc_img2, batch_proc_img[1]))


if __name__ == "__main__":
    # unittest.main()
    # arbitrary_proc_visual_test()
    TEST_IMAGES_FPATHS = (
        "/orb/benoit_phd/datasets/RawNIND/proc/lin_rec2020/D60-1/gt/CRW_6994.CRW.tif",
        "/orb/benoit_phd/datasets/RawNIND/proc/lin_rec2020/MuseeL-vases-A7C/gt/ISO50_capt0014.arw.tif",
        "/orb/benoit_phd/datasets/RawNIND/proc/lin_rec2020/Vaxt-i-trad/gt/IMG_8865.CR2.tif",
        "/orb/benoit_phd/datasets/RawNIND/proc/lin_rec2020/7D-2/gt/_M7D5418.CR2.tif",
        "/orb/benoit_phd/datasets/RawNIND/proc/lin_rec2020/TitusToys/gt/ISO50_capt0001.arw.tif",
        "/orb/benoit_phd/datasets/RawNIND/proc/lin_rec2020/Laura_Lemons_platformer/gt/ISO50_capt0001.arw.tif",
        "/orb/benoit_phd/datasets/RawNIND/proc/lin_rec2020/boardgames_top/gt/ISO50_capt0001.arw.tif",
    )
    OUTPUT_DPATH = os.path.join("tmp", "arbitrary_proc_visual_test")
    for fpath in TEST_IMAGES_FPATHS:
        test_arbitrarily_process_images_opencv_values(fpath, OUTPUT_DPATH)
