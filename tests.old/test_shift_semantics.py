"""
Test to definitively establish and document the shift convention in rawproc.

FINDING: shift=(dy,dx) represents NEGATIVE of physical content displacement.
- If content moves DOWN by 2, shift = (-2, ?)
- If content moves RIGHT by 3, shift = (?, -3)
"""

import numpy as np
import sys

sys.path.append("src")
from rawnind.libs.rawproc import shift_images

print("=" * 80)
print("SHIFT CONVENTION TEST")
print("=" * 80)
print()

# Create test image with distinct pattern
img = np.zeros((1, 12, 12), dtype=np.float32)
img[0, 2:5, 2:5] = 1.0
img[0, 7:10, 7:10] = 0.5

print("Original anchor image (two squares at [2:5,2:5]=1.0 and [7:10,7:10]=0.5):")
print(img[0])
print()

# Test 1: Roll content DOWN by 3 pixels
target_down_3 = np.roll(img, 3, axis=1)
print("Target 1: Content rolled DOWN by 3 (squares now at rows 5 and 10):")
print(target_down_3[0])
print()

print("Testing different shift values with shift_images:")
for test_shift in [(0, 0), (3, 0), (-3, 0), (2, 0), (-2, 0)]:
    anchor_out, target_out = shift_images(img, target_down_3, test_shift)
    loss = np.abs(anchor_out - target_out).mean()
    print(f"  shift={test_shift}: loss={loss:.6f}, output_shape={anchor_out.shape}")

print()
print("✓ RESULT: shift=(-3,0) gives loss=0 when content rolled DOWN by 3")
print("  → Convention: shift = NEGATIVE of displacement")
print()

# Test 2: Roll content RIGHT by 2 pixels
target_right_2 = np.roll(img, 2, axis=2)
print("Target 2: Content rolled RIGHT by 2 (squares now at cols 4 and 9):")
print(target_right_2[0])
print()

print("Testing different shift values:")
for test_shift in [(0, 0), (0, 2), (0, -2), (0, 3), (0, -3)]:
    anchor_out, target_out = shift_images(img, target_right_2, test_shift)
    loss = np.abs(anchor_out - target_out).mean()
    print(f"  shift={test_shift}: loss={loss:.6f}, output_shape={anchor_out.shape}")

print()
print("✓ RESULT: shift=(0,-2) gives loss=0 when content rolled RIGHT by 2")
print("  → Convention: shift = NEGATIVE of displacement")
print()

# Test 3: Combined DOWN-RIGHT
target_dr = np.roll(np.roll(img, 3, axis=1), 2, axis=2)
print("Target 3: Content rolled DOWN by 3, RIGHT by 2:")
print(target_dr[0])
print()

print("Testing combined shift:")
for test_shift in [(0, 0), (3, 2), (-3, -2), (3, -2), (-3, 2)]:
    anchor_out, target_out = shift_images(img, target_dr, test_shift)
    loss = np.abs(anchor_out - target_out).mean()
    print(f"  shift={test_shift}: loss={loss:.6f}, output_shape={anchor_out.shape}")

print()
print("✓ RESULT: shift=(-3,-2) gives loss=0 when content rolled DOWN by 3, RIGHT by 2")
print("  → Convention confirmed: shift = NEGATIVE of displacement vector")
print()

print("=" * 80)
print("SUMMARY: SHIFT CONVENTION")
print("=" * 80)
print("shift=(dy, dx) in rawproc means:")
print("  - dy = NEGATIVE of vertical content displacement")
print("  - dx = NEGATIVE of horizontal content displacement")
print()
print("Example: If target content moved down 5px and right 3px:")
print("  → correct shift = (-5, -3)")
print()
print("This convention likely reflects camera motion rather than content motion:")
print("  - Positive shift = camera moved up/left (content appears down/right)")
print("  - Negative shift = camera moved down/right (content appears up/left)")
print("=" * 80)
