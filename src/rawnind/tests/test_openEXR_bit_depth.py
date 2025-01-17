import sys
import numpy as np

sys.path.append("..")
from rawnind.libs import raw

if __name__ == "__main__":
    image: np.ndarray = np.random.random((3, 128, 128))
    raw.hdr_nparray_to_file(
        image,
        "tests_output/test_openEXR_bit_depth_32.exr",
        bit_depth=32,
        color_profile="lin_rec2020",
    )
    raw.hdr_nparray_to_file(
        image,
        "tests_output/test_openEXR_bit_depth_16.exr",
        bit_depth=16,
        color_profile="lin_rec2020",
    )
