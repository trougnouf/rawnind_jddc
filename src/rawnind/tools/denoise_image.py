"""
Denoise a single image using a trained model.

By default save the denoised image in a directory named "denoised_images" in the model's save_dpath,
with the same filename as the input image and the extension ".denoised.exr".
Also save the metrics (and the bitrate if applicable) in a directory named "denoised_images_metrics",
with the same filename as the input image and the extension ".metrics.yaml". 

egrun python tools/denoise_image.py -i /orb/Pictures/ITookAPicture/2023/05/29_LucieHikeLustinYvoirGR126/DSC04011.ARW --config ../../models/rawnind_dc/DCTrainingProfiledRGBToProfiledRGB_3ch_L4096.0_Balle_Balle_dc_prgb_1_/args.yaml  --device
"""

import os
import sys
from typing import Literal, Optional, Tuple, Union
import torch
import configargparse
import yaml

sys.path.append("..")
from rawnind.libs import abstract_trainer
from common.libs import np_imgops
from common.libs import pt_ops
from common.libs import pt_losses
from common.libs import pt_helpers
from rawnind.models import raw_denoiser
from rawnind.libs import rawproc
from rawnind.libs import raw

DENOISED_DN = "denoised_images"
METRICS_DN = "denoised_images_metrics"


def add_arguments(parser):
    parser.add_argument(
        "--config",
        dest="config",
        required=True,
        help="trained model's config file in yaml format",
    )
    parser.add_argument(
        "-i",
        "--in_img_fpath",
        required=True,
        help="Path of the image to denoise",
    )
    parser.add_argument(
        "--gt_img_fpath",
        help="Path to the ground-truth image (optional)",
    )
    parser.add_argument(
        "-o", "--out_img_fpath", help="Optional path to save the denoised image"
    )
    # parser.add_argument("--device", type=int, help="CUDA device number (-1 for CPU)")
    parser.add_argument(
        "--metrics",
        nargs=("*"),
        help=f"Validation and test metrics: {pt_losses.metrics}",
        default=["mse", "msssim_loss"],
    )
    parser.add_argument(
        "--nonlinearities",
        nargs=("*"),
        help=f"Nonlinearities used to compute the metrics, as defined in abstract_trainer.ImageToImageNN.get_transfer_function (ie pq, gamma22)",
        default=["pq", "gamma22"],
    )
    # parser.add_argument(  # rm?
    #     "--skip_metrics",
    #     action="store_true",
    #     help="Skip computing metrics",
    # )


"""
GT should be loaded once
Image should be denoised once
Metrics should be computed multiple times
:
    - load GT
    - denoise image
    - compute metrics
    - save denoised image
    - save metrics with gt in fn

    
    
denoise_image(model, in_img) -> {denoised_image, bpp (opt)}

process_image_base(model, out_img, gt_img (opt), xyz_rgb_matrix (opt)) -> linrec2020_img

apply_nonlinearity(model, img, nonlinearity: str) -> nl_img

compute_metrics(img, gt, metrics, xyz_rgb_matrix (opt), prefix: str = None) -> {metrics_results}
    calls model's process_net_output
    loss pre/post non-linearity

save_image(image, fpath)

save_metrics(metrics_results, fpath)


"""


def load_image(fpath, device) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    img, metadata = pt_helpers.fpath_to_tensor(
        fpath, incl_metadata=True, device=device, crop_to_multiple=16
    )
    rgb_xyz_matrix = metadata.get("rgb_xyz_matrix", None)
    if rgb_xyz_matrix is not None:
        rgb_xyz_matrix = torch.tensor(rgb_xyz_matrix).unsqueeze(0)
    return img, rgb_xyz_matrix  # type: ignore


def process_image_base(
    test_obj: abstract_trainer.ImageToImageNN,
    out_img: torch.Tensor,
    gt_img: Optional[torch.Tensor] = None,
    in_img: Optional[torch.Tensor] = None,
    rgb_xyz_matrix: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if gt_img is not None:
        ref_img = gt_img
    elif in_img is not None and in_img.shape[-3] == out_img.shape[-3]:
        ref_img = in_img
    elif in_img is not None and in_img.shape[-3] == 4 and out_img.shape[-3] == 3:
        # demosaic image to get input mean for match_gain
        ref_img = rawproc.demosaic(in_img).unsqueeze(0)
        ref_img = rawproc.camRGB_to_lin_rec2020_images(ref_img, rgb_xyz_matrix)
    else:
        ref_img = None
    if hasattr(test_obj, "process_net_output"):
        print(f"Mean of network output: {out_img.mean()}")
        out_img = test_obj.process_net_output(out_img, rgb_xyz_matrix, ref_img)
        print(f"Mean after process_net_output: {out_img.mean()}")
    elif ref_img is not None:
        print(f"Mean of network output: {out_img.mean()}")
        out_img = rawproc.match_gain(ref_img, out_img)
        print(f"Mean after match_gain: {out_img.mean()}")
    else:
        pass
    if out_img.mean() > 1 or out_img.mean() < 0:
        print(
            f"WARNING: mean of output image is outside of valid range ({out_img.mean()=})"
        )
        out_img = rawproc.match_gain(ref_img, out_img)
        print(f"Mean after matching gain: {out_img.mean()}")
    return out_img


def apply_nonlinearity(img: torch.Tensor, nonlinearity: str) -> torch.Tensor:
    return abstract_trainer.ImageToImageNN.get_transfer_function(nonlinearity)(
        img.clone()
    )


def compute_metrics(
    in_img: torch.Tensor,
    gt_img: torch.Tensor,
    metrics: list[str] = [],
    prefix=None,
) -> dict:
    metrics_results = {}
    for metric in metrics:
        metrics_results[metric] = float(pt_losses.metrics[metric](in_img, gt_img))
    if prefix is not None:
        metrics_results = {f"{prefix}_{k}": v for k, v in metrics_results.items()}
    return metrics_results


def save_image(image, fpath: str, src_fpath: Optional[str] = None):
    assert image.shape[-3] == 3
    if len(image.shape) == 4:
        image = image.squeeze(0)
    raw.hdr_nparray_to_file(image.numpy(), fpath, "lin_rec2020", src_fpath=src_fpath)


def save_metrics(metrics: dict, fpath: str):
    with open(fpath, "w") as f:
        yaml.dump(metrics, f)


def denoise_image_from_to_fpath(
    in_img_fpath: str, out_img_fpath: str, test_obj: abstract_trainer.ImageToImageNN
):
    img, rgb_xyz_matrix = load_image(in_img_fpath, device=test_obj.device)

    model_results = test_obj.infer(img, return_dict=False)
    processed_image = process_image_base(
        test_obj, model_results, rgb_xyz_matrix=rgb_xyz_matrix
    )
    save_image(processed_image, out_img_fpath, src_fpath=in_img_fpath)


def bayer_to_prgb(image, rgb_xyz_matrix):
    """Used when trying to input a bayer image to a pRGB model."""
    if image.shape[-3] == 3:
        return image
    image = rawproc.demosaic(image).unsqueeze(0)
    image = rawproc.camRGB_to_lin_rec2020_images(image, rgb_xyz_matrix)
    return image


#############


def denoise_image_compute_metrics(
    in_img,
    test_obj: abstract_trainer.ImageToImageNN,
    rgb_xyz_matrix: Optional[torch.Tensor] = None,
    gt_img: Optional[torch.Tensor] = None,
    metrics: list[str] = [],
    nonlinearities: list[str] = [],
) -> tuple[torch.Tensor, dict]:
    # if model is pRGB and img is bayer, debayer
    if test_obj.in_channels == 3:
        in_img = bayer_to_prgb(in_img, rgb_xyz_matrix)
    # denoise and proc
    model_results = test_obj.infer(in_img, return_dict=True)
    processed_image = process_image_base(
        test_obj,
        model_results["reconstructed_image"],
        gt_img,
        in_img,
        rgb_xyz_matrix=rgb_xyz_matrix,
    )
    # init metrics
    metrics_results = {}
    if "bpp" in model_results:
        metrics_results["bpp"] = float(model_results["bpp"])
    # compute metrics
    if gt_img and metrics:
        metrics_results.update(compute_metrics(processed_image, gt_img, metrics))
        for nonlinearity in nonlinearities:
            if str(nonlinearity) == "None":
                continue
            nl_gt = apply_nonlinearity(gt_img, nonlinearity)
            nl_out = apply_nonlinearity(processed_image, nonlinearity)
            nl_out = rawproc.match_gain(nl_gt, nl_out)
            metrics_results.update(
                compute_metrics(nl_out, nl_gt, metrics, nonlinearity)
            )
    return processed_image, metrics_results


def denoise_image_from_fpath_compute_metrics_and_export(
    in_img_fpath: str,
    # test_obj: Union[abstract_trainer.ImageToImageNN, str],
    test_obj: Optional[abstract_trainer.ImageToImageNN] = None,
    gt_img_fpath: Optional[str] = None,
    metrics: list[str] = [],
    nonlinearities: list[str] = [],
    out_img_fpath=None,
):
    # if isinstance(test_obj, str):
    if test_obj is None:
        test_obj = abstract_trainer.get_and_load_test_object()
    in_img, rgb_xyz_matrix = load_image(in_img_fpath, test_obj.device)
    if gt_img_fpath:
        gt_img = load_image(gt_img_fpath, test_obj.device)
    else:
        gt_img = None
    processed_image, metrics_results = denoise_image_compute_metrics(
        in_img=in_img,
        test_obj=test_obj,
        gt_img=gt_img,
        metrics=metrics,
        nonlinearities=nonlinearities,
        rgb_xyz_matrix=rgb_xyz_matrix,
    )
    # output
    model_fn = os.path.basename(test_obj.load_path)
    if out_img_fpath is None:
        out_img_dpath = os.path.join(test_obj.save_dpath, DENOISED_DN + "_" + model_fn)
        os.makedirs(out_img_dpath, exist_ok=True)
        out_img_fpath = os.path.join(
            out_img_dpath, os.path.basename(in_img_fpath) + ".denoised.tif"
        )
    save_image(processed_image, out_img_fpath, src_fpath=in_img_fpath)
    if gt_img_fpath or metrics_results:
        metrics_dpath = os.path.join(test_obj.save_dpath, METRICS_DN + "_" + model_fn)
        os.makedirs(metrics_dpath, exist_ok=True)
        if gt_img_fpath:
            metrics_fn = f"{os.path.basename(in_img_fpath)}--{os.path.basename(gt_img_fpath)}.metrics.yaml"
        else:
            metrics_fn = f"{os.path.basename(in_img_fpath)}.metrics.yaml"
        save_metrics(metrics_results, os.path.join(metrics_dpath, metrics_fn))
        print(
            f"{metrics_results=} written to {os.path.join(metrics_dpath, metrics_fn)}"
        )


# def denoise_image_from_fpath_and_compute_metrics(
#     test_obj, in_img_fpath: str, gt_img_fpath: str, gt_img: Optional[torch.Tensor] = None, metrics: list[str] = [], skip_metrics: bool = False
# ):
#     img, metadata = pt_helpers.fpath_to_tensor(in_img_fpath, incl_metadata=True, device=test_obj.device)
#     if gt_img is None:
#         gt_img = pt_helpers.fpath_to_tensor(gt_img_fpath, incl_metadata=False, device=test_obj.device)
#     rgb_xyz_matrix = metadata.get("rgb_xyz_matrix", None)
#     model_results = test_obj.infer(img, rgb_xyz_matrix)

#     metrics_results = {}
#     if "bpp" in model_results:
#         metrics_results["bpp"] = model_results["bpp"]

#     if not skip_metrics and (metrics or metrics_results):
#         for metric in metrics:
#             metrics_results[metric] = pt_losses.metrics[metric](metrics_results["proc_img"], img)
#     if metrics_results:
#         print(f"{in_img_fpath=}: {metrics_results}")
#     return model_results, metrics_results


# def save_denoising_results(
#     proc_img: torch.Tensor,
#     metrics_results: Optional[dict] = None,
#     test_obj: Optional[abstract_trainer.ImageToImageNN] = None,
#     in_img_fpath: Optional[str] = None,
#     out_img_fpath: Optional[str] = None,
#     out_metrics_fpath: Optional[str] = None,
# ) -> None:
#     if out_img_fpath is None:
#         assert test_obj is not None and in_img_fpath is not None
#         out_img_dpath = os.path.join(test_obj.save_dpath, DENOISED_DN)
#         os.makedirs(out_img_dpath, exist_ok=True)
#         out_img_fpath = os.path.join(
#             out_img_dpath, os.path.basename(in_img_fpath) + ".denoised.exr"
#         )
#     proc_img: np.ndarray = proc_img.squeeze(0).numpy()  # maybe need from "__future__ import annotations"?
#     raw.hdr_nparray_to_file(proc_img, out_img_fpath, "lin_rec2020")
#     print(f"output written to {out_img_fpath}")
#     if not metrics_results:
#         return
#     if not out_metrics_fpath:
#         assert test_obj is not None and in_img_fpath is not None
#         out_metrics_dpath = os.path.join(test_obj.save_dpath, METRICS_DN)
#         os.makedirs(out_metrics_dpath, exist_ok=True)
#         out_metrics_fpath = os.path.join(
#             out_metrics_dpath, os.path.basename(in_img_fpath) + ".metrics.yaml"
#         )
#     with open(out_metrics_fpath, "w") as f:
#         yaml.dump(metrics_results, f)
#     print(f"metrics written to {out_metrics_fpath}")


if __name__ == "__main__":
    parser = configargparse.argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_arguments(parser)
    args = parser.parse_known_args()[0]
    # call denoise_image_compute_metrics with all args except config
    vars(args).pop("config")
    denoise_image_from_fpath_compute_metrics_and_export(**vars(args))


# if __name__ == "__main__":
#     test_obj = abstract_trainer.get_and_load_test_object()
#     model = test_obj.model

#     # args =
#     model_fpath = sys.argv[1]
#     infpath = sys.argv[2]
#     outfpath = (
#         sys.argv[2] + ".denoised.exr" if len(sys.argv) == 3 else sys.argv[3]
#     )  # sys.argv[3]
#     in_channels: Literal[3, 4] = 3 if infpath.endswith(".exr") else 4
#     model: raw_denoiser.Denoiser = raw_denoiser.UtNet2(in_channels=in_channels)
#     model.load_state_dict(torch.load(model_fpath, map_location="cpu"))
#     model = model.eval()
#     with torch.no_grad():
#         if in_channels == 3:
#             img: torch.Tensor = torch.from_numpy(
#                 np_imgops.img_fpath_to_np_flt(infpath)
#             ).unsqueeze(0)
#         else:
#             img, metadata = np_imgops.img_fpath_to_np_flt(infpath, incl_metadata=True)
#             img = torch.from_numpy(img).unsqueeze(0)
#             rgb_xyz_matrix = metadata["rgb_xyz_matrix"]
#         img = pt_ops.crop_to_multiple(img, 16)
#         output: torch.Tensor = model(img)
#         output = rawproc.match_gain(img, output)
#         if in_channels == 4:
#             output = rawproc.camRGB_to_lin_rec2020_images(
#                 output, torch.from_numpy(rgb_xyz_matrix).unsqueeze(0)
#             )
#         output_np = output.squeeze(0).numpy()
#     raw.hdr_nparray_to_file(output_np, outfpath, "lin_rec2020")
#     print(f"output written to {outfpath}")
