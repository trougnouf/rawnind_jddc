import os
import sys

sys.path.append("..")
from rawnind.libs import rawproc
from common.libs import np_imgops

FP1: str = os.path.join("test_data", "Moor_frog_bl.jpg")
FP2: str = os.path.join("test_data", "Moor_frog_tr.jpg")
img1 = np_imgops.img_fpath_to_np_flt(FP1)
img2 = np_imgops.img_fpath_to_np_flt(FP2)
best_alignment = rawproc.find_best_alignment(
    anchor_img=img1, target_img=img2, verbose=True
)
print(best_alignment)
img1_aligned, img2_aligned = rawproc.shift_images(img1, img2, best_alignment)
np_imgops.np_to_img(
    img1_aligned, os.path.join("tests_output", "Moor_frog_bl_aligned.png")
)
np_imgops.np_to_img(
    img2_aligned, os.path.join("tests_output", "Moor_frog_tr_aligned.png")
)
