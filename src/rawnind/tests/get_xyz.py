import os
import rawpy
import numpy as np
import doctest


def print_attributes(fpath):
    rawpy_img = rawpy.imread(fpath)
    attributes = dict()
    attributes["camera_whitebalance"] = rawpy_img.camera_whitebalance
    attributes["black_level_per_channel"] = rawpy_img.black_level_per_channel
    attributes["white_level"] = rawpy_img.white_level  # imgdata.rawdata.color.maximum
    attributes["camera_white_level_per_channel"] = (
        rawpy_img.camera_white_level_per_channel
    )  # imgdata.rawdata.color.linear_max
    attributes["daylight_whitebalance"] = (
        rawpy_img.daylight_whitebalance
    )  # imgdata.rawdata.color.pre_mul
    attributes["rgb_xyz_matrix"] = rawpy_img.rgb_xyz_matrix
    print(attributes)


if __name__ == "__main__":
    # print('rgb_uncompressed')
    # print_attributes('/orb/Pictures/ITookAPicture/2022/03/24/DSC01550.ARW')
    # print('rgb_compressed')
    # print_attributes('/orb/Pictures/ITookAPicture/2022/03/24/DSC01551.ARW')
    # print('adobe_uncompressed')
    # print_attributes('/orb/Pictures/ITookAPicture/2022/03/24/_DSC1552.ARW')
    # print('adobe_compressed')
    # print_attributes('/orb/Pictures/ITookAPicture/2022/03/24/_DSC1553.ARW')

    print("arw")
    print_attributes(
        "/orb/Pictures/ITookAPicture/2022/03/19_16dAlleeDesCharmes/DSC01526.ARW"
    )
    print("dng")
    print_attributes(
        "/orb/Pictures/ITookAPicture/2022/03/19_16dAlleeDesCharmes/DSC01526.dng"
    )
