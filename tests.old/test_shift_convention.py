"""Test to understand shift_images convention."""

import numpy as np
import sys

sys.path.append("src")
from rawnind.libs.rawproc import shift_images

# Create a simple test image with a clear pattern
img = np.zeros((1, 10, 10), dtype=np.float32)
img[0, 2:4, 2:4] = 1.0  # A 2x2 bright square at position (2,2)

print("Original image (showing where bright square is):")
print(img[0, :5, :5])
print()

# Create a shifted version by rolling the array
shifted_down_right = np.roll(np.roll(img, 2, axis=1), 2, axis=2)
print("Shifted down-right by 2 (bright square now at 4,4):")
print(shifted_down_right[0, :7, :7])
print()

# Now test what shift_images does with different shift values
print("Testing shift_images(anchor=original, target=shifted_down_right, shift):")
print()

for test_shift in [(0, 0), (2, 2), (-2, -2), (3, 3)]:
    anchor_out, target_out = shift_images(img, shifted_down_right, test_shift)
    loss = np.abs(anchor_out - target_out).mean()
    print(f"shift={test_shift}: anchor_out.shape={anchor_out.shape}, loss={loss:.6f}")

print()
print(
    "Expected: shift=(2,2) should give lowest loss since target is shifted down-right by 2"
)
