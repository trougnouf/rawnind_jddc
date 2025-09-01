#!/bin/python3
"""Capture a scene with different ISO values. If an argument is set, it will be used as the scene name."""

# TODO validate by computing difference between first and last image
# TODO set silent shooting
import subprocess
import os
import logging
from typing import Iterable, Literal
import gphoto2 as gp
import sys
import datetime
import time
import random
import torch
import sys

sys.path.append("..")
sys.path.append("../..")
from rawnind.libs import raw

MAX_NOISY: int = 12
CAPTURED_DPATH: str = os.path.join("..", "..", "datasets", "RawNIND", "src", "Bayer")


def get_gt_noisy_iso_values(cfg: gp.widget.CameraWidget) -> tuple[str, list[str]]:
    all_isos: list[str] = list(cfg.get_child_by_name("iso").get_choices())
    lowest_iso: str = ""
    noisy_isos: list[str] = []
    for iso_value in all_isos:
        if iso_value.isnumeric():
            if not lowest_iso or int(iso_value) < int(lowest_iso):
                lowest_iso = iso_value
            else:
                noisy_isos.append(iso_value)
    return lowest_iso, noisy_isos


def empty_cache(camera, folder="/"):
    if not list(camera.folder_list_files(folder)):
        return
    breakpoint()


def capture_image(camera, out_dpath: str, fn_prefix: str, delay: int = 0) -> str:
    empty_cache(camera)
    try:
        file_path = camera.capture(
            gp.GP_CAPTURE_IMAGE
        )  # trigger_capture should work when this does not
    except gp.GPhoto2Error as e:
        print(f"capture_image: error {e}, trying again in {delay=} ...")
        time.sleep(delay)
        return capture_image(camera, out_dpath, fn_prefix, delay + 1)
    ("Camera file path: {0}/{1}".format(file_path.folder, file_path.name))
    if out_dpath is None and fn_prefix is None:
        camera.file_delete(file_path.folder, file_path.name)
        return
    target = os.path.join(out_dpath, fn_prefix + "_" + file_path.name)
    print("Copying image to", target)
    camera_file = camera.file_get(
        file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL
    )
    camera_file.save(target)
    camera.file_delete(file_path.folder, file_path.name)
    return target


def set_iso(
    camera: gp.camera.Camera,
    cfg: gp.widget.CameraWidget,
    iso_value: str,
    first_attempt=True,
) -> None:
    cfg.get_child_by_name("iso").set_value(iso_value)
    camera.set_config(cfg)
    time.sleep(2)  # otherwise the setting is often not applied
    assert cfg.get_child_by_name("iso").get_value() == iso_value


def set_focus(camera, cfg, mode: Literal["auto", "manual"]) -> None:
    if mode == "auto":
        cfg.get_child_by_name("focusmode").set_value("AF-A")
    elif mode == "manual":
        cfg.get_child_by_name("focusmode").set_value("Manual")
    camera.set_config(cfg)


if __name__ == "__main__":
    start_time: float = time.time()
    scene_name: str = (
        datetime.datetime.now().isoformat() if len(sys.argv) == 1 else sys.argv[1]
    )
    print(f"Using {scene_name=}")
    camera: gp.camera.Camera = gp.Camera()
    camera.init()
    # print('Camera summary:')
    # print(camera.get_summary())
    cfg: gp.widget.CameraWidget = camera.get_config()
    gt_iso, noisy_isos = get_gt_noisy_iso_values(cfg)
    print(f"{gt_iso=}, {noisy_isos=}")

    out_dpath = os.path.join(CAPTURED_DPATH, scene_name)
    os.makedirs(os.path.join(out_dpath, "gt"), exist_ok=True)
    set_focus(camera, cfg, "auto")
    set_iso(camera, cfg, gt_iso)
    capture_image(camera, out_dpath=None, fn_prefix=None)  # set focus
    time.sleep(5)
    set_focus(camera, cfg, "manual")
    gt1_fpath = capture_image(
        camera, out_dpath=os.path.join(out_dpath, "gt"), fn_prefix="ISO" + gt_iso
    )
    for iso_value in random.sample(noisy_isos, min(MAX_NOISY, len(noisy_isos))):
        set_iso(camera, cfg, iso_value)
        capture_image(camera, out_dpath=out_dpath, fn_prefix="ISO" + iso_value)
    set_iso(camera, cfg, gt_iso)
    gt2_fpath = capture_image(
        camera, out_dpath=os.path.join(out_dpath, "gt"), fn_prefix="ISO" + gt_iso
    )
    print(f"Total capture time: {time.time() - start_time} seconds")
    set_focus(camera, cfg, "auto")
    gt1_tensor, _ = raw.raw_fpath_to_mono_img_and_metadata(gt1_fpath)
    gt2_tensor, _ = raw.raw_fpath_to_mono_img_and_metadata(gt2_fpath)
    image_set_quality = torch.nn.MSELoss()(
        torch.tensor(gt1_tensor), torch.tensor(gt2_tensor)
    )
    print(f"{image_set_quality=}")

    # file_path = camera.capture(gp.GP_CAPTURE_IMAGE)

    # print(file_path)
    # print('Camera file path: {0}/{1}'.format(file_path.folder, file_path.name))
    # # target = os.path.join('/tmp', file_path.name)
    # # print('Copying image to', target)
    # # camera_file = camera.file_get(
    # # file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL)
    # # camera_file.save(target)
    # # subprocess.call(['xdg-open', target])
    # camera.exit()
