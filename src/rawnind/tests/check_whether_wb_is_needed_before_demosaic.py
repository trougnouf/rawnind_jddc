import os
import statistics
import sys
import numpy as np
import cv2

sys.path.append("..")
from rawnind.libs import raw

IMAGE_SETS_DPATH = os.path.join("..", "..", "datasets", "rawNIND", "DocScan", "bayer")

if __name__ == "__main__":
    losses = []
    for demosaic_algorithm_name, demosaic_algorithm in {
        "COLOR_BayerRGGB2RGB_EA": cv2.COLOR_BayerRGGB2RGB_EA,
        "COLOR_BayerRGGB2RGB": cv2.COLOR_BayerRGGB2RGB,
    }.items():
        print(f"{demosaic_algorithm_name=}")
        for aset in os.listdir(IMAGE_SETS_DPATH):
            dpath = os.path.join(IMAGE_SETS_DPATH, aset, "gt")
            for fn in os.listdir(dpath):
                fpath = os.path.join(dpath, fn)
                mono_img, metadata = raw.raw_fpath_to_mono_img_and_metadata(fpath)
                # without whitebalance
                camRGB_img_nowb = raw.demosaic(
                    mono_img, metadata, method=demosaic_algorithm
                )  # NN GT
                gamma_sRGB_img_nowb = raw.camRGB_to_profiledRGB_img(
                    camRGB_img_nowb, metadata, "gamma_sRGB"
                )
                # with whitebalance
                mono_img_wb = raw.apply_whitebalance(
                    mono_img, metadata, wb_type="daylight", in_place=False
                )
                camRGB_img_wb = raw.demosaic(mono_img_wb, metadata)  # NN GT
                camRGB_img_wb = raw.apply_whitebalance(
                    camRGB_img_wb,
                    metadata,
                    wb_type="daylight",
                    in_place=False,
                    reverse=True,
                )
                gamma_sRGB_img_wb = raw.camRGB_to_profiledRGB_img(
                    camRGB_img_wb, metadata, "gamma_sRGB"
                )
                loss = np.abs(gamma_sRGB_img_nowb - gamma_sRGB_img_wb).mean()
                print(f"{fn=}, {loss=}")
                losses.append(loss)
        print(f"{statistics.mean(losses)=}")
