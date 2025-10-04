import argparse
import os
import sys
import time

import torch

sys.path.append("..")
import rawnind.models
from rawnind.tools import denoise_image
from rawnind.libs import rawproc

MODEL_FPATH = os.path.join(os.path.abspath(os.path.curdir),
                           "src/rawnind/models/rawnind_denoise/DenoiserTrainingBayerToProfiledRGB_4ch_2024-11-22-bayer_ms-ssim_mgout_notrans_valeither_noowwnpics_mgdef_-1/saved_models/iter_1245000.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_fpath", default=MODEL_FPATH)
    parser.add_argument("-i", "--input_fpath", required=True)
    parser.add_argument("-o", "--output_fpath")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    # TODO debayer if prgb model
    model_is_bayer = 'bayer' in args.model_fpath.lower()
    if rawnind.libs.raw.is_xtrans(args.input_fpath):
        infile = args.input_fpath.replace('RAF', 'exr')
        rawnind.libs.raw.xtrans_fpath_to_OpenEXR(args.input_fpath, infile)
    else:
        infile = args.input_fpath
    if args.output_fpath:
        output_fpath = args.output_fpath
    else:
        # {input_fpath}_{model grandparent directory}_{model_fn}.tif
        model_fpath = os.path.abspath(args.model_fpath)
        model_parent_dir = os.path.basename(os.path.dirname(model_fpath))
        model_grandparent_dir = os.path.basename(
            os.path.dirname(os.path.dirname(model_fpath))
        )
        model_fn = os.path.basename(model_fpath).replace(".", "-")
        output_fpath = f"{args.input_fpath}_{model_grandparent_dir}_{model_fn}.tif"

    if not args.cpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    with torch.no_grad():
        input_image, rgb_xyz_matrix = denoise_image.load_image(
            infile, device=device
        )
        input_image = input_image.unsqueeze(0)
        model = rawnind.models.raw_denoiser.UtNet2(
            in_channels=4 if model_is_bayer and not infile.endswith(".exr") else 3, funit=32
        )
        model.load_state_dict(
            torch.load(
                args.model_fpath, map_location=torch.device("cpu") if args.cpu else None
            )
        )
        model.eval()
        model = model.to(device)
        input_image = input_image.to(device)
        # time it
        start = time.time()
        out_image = model(input_image)
        end = time.time()
        out_image = rawproc.match_gain(anchor_img=input_image, other_img=out_image)
        out_image = rawproc.camRGB_to_lin_rec2020_images(out_image, rgb_xyz_matrix)
        out_image = out_image.cpu()
        print(f"Saving to {output_fpath}. Processing time: {end - start:.2f} s")
        denoise_image.save_image(out_image, output_fpath, src_fpath=args.input_fpath)
