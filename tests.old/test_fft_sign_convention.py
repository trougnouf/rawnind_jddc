"""Test FFT phase correlation on synthetic data to understand sign convention."""
import numpy as np
import sys
sys.path.append("src")
from rawnind.libs.rawproc import shift_images

print("="*80)
print("FFT SIGN CONVENTION TEST")
print("="*80)
print()

# Create synthetic test image
img = np.zeros((1, 100, 100), dtype=np.float32)
img[0, 20:40, 20:40] = 1.0
img[0, 60:80, 60:80] = 0.5

print("Created synthetic image with two squares")
print()

# Apply known shift using np.roll (content displacement)
# Roll DOWN by 4, RIGHT by 4
shifted_img = np.roll(np.roll(img, 4, axis=1), 4, axis=2)

print("Applied shift: content rolled DOWN by 4, RIGHT by 4")
print("In shift_images convention: shift = (-4, -4)")
print()

# Verify with shift_images
test_shifts = [(-4, -4), (4, 4), (-4, 4), (4, -4), (0, 0)]
print("Testing shift_images with different shifts:")
for test_shift in test_shifts:
    anchor_out, target_out = shift_images(img, shifted_img, test_shift)
    loss = np.abs(anchor_out - target_out).mean()
    marker = "✅" if loss < 0.001 else "  "
    print(f"  {marker} shift={test_shift}: loss={loss:.6f}")
print()
print("Confirmed: correct shift is (-4, -4)")
print()

# Now test FFT phase correlation
print("="*80)
print("FFT PHASE CORRELATION TEST")
print("="*80)
print()

def fft_phase_correlate(anchor, target):
    """Standard FFT phase correlation."""
    anchor_img = anchor[0]
    target_img = target[0]
    
    # Mean center to remove DC component
    anchor_centered = anchor_img - anchor_img.mean()
    target_centered = target_img - target_img.mean()
    
    # FFT
    f1 = np.fft.fft2(anchor_centered)
    f2 = np.fft.fft2(target_centered)
    
    # Cross-power spectrum (phase correlation)
    cross_power = (f1 * np.conj(f2)) / (np.abs(f1 * np.conj(f2)) + 1e-10)
    
    # Inverse FFT to get correlation
    correlation = np.fft.ifft2(cross_power).real
    
    # Find peak
    h, w = correlation.shape
    peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)
    
    print(f"Raw peak position: ({peak_y}, {peak_x}) out of shape ({h}, {w})")
    
    # The peak tells us the displacement from target to anchor
    # But we need to handle FFT wraparound
    if peak_y > h // 2:
        peak_y -= h
    if peak_x > w // 2:
        peak_x -= w
    
    print(f"After wraparound handling: ({peak_y}, {peak_x})")
    
    return (peak_y, peak_x), correlation

shift_from_fft, corr = fft_phase_correlate(img, shifted_img)

print()
print(f"FFT returned: {shift_from_fft}")
print(f"Correct shift: (-4, -4)")
print(f"FFT sign: {'CORRECT' if shift_from_fft == (-4, -4) else 'WRONG'}")
print()

# The issue: FFT tells us where target is displaced FROM anchor
# But we need to know where target is POSITIONED relative to anchor

print("="*80)
print("UNDERSTANDING THE SIGN CONVENTION")
print("="*80)
print()

print("1. We rolled content DOWN by 4, RIGHT by 4")
print("   → target[0,0] now contains what was at anchor[96,96] (wraparound)")
print("   → target[4,4] now contains what was at anchor[0,0]")
print()

print("2. FFT phase correlation finds displacement:")
print("   → It finds that anchor is at position (4,4) in target's coordinate system")
print("   → Or equivalently: target's origin is at (-4,-4) in anchor's coordinate system")
print()

print("3. shift_images convention:")
print("   → shift=(dy,dx) means 'target is at position (dy,dx) relative to anchor'")
print("   → We want shift=(-4,-4)")
print()

print("4. FFT peak interpretation:")
print(f"   → Peak at {shift_from_fft}")
print("   → This is the displacement of anchor relative to target")
print("   → We need to NEGATE this to get target's position relative to anchor")
print()

# Test with negation
fft_shift_negated = (-shift_from_fft[0], -shift_from_fft[1])
print(f"5. Negated FFT result: {fft_shift_negated}")
print(f"   Expected shift: (-4, -4)")
print(f"   Match: {'YES ✅' if fft_shift_negated == (-4, -4) else 'NO ❌'}")
print()

# Verify
anchor_out, target_out = shift_images(img, shifted_img, fft_shift_negated)
loss = np.abs(anchor_out - target_out).mean()
print(f"   Loss with negated shift: {loss:.6f} {'✅' if loss < 0.001 else '❌'}")
print()

print("="*80)
print("CONCLUSION")
print("="*80)
print("The FFT implementation is NEGATING when it should NOT, or vice versa.")
print("FFT returns displacement, but we're interpreting it incorrectly.")
print()
print("In current code:")
print("  shift = (-peak_y, -peak_x)  ← This is WRONG")
print()
print("Should be:")
print("  shift = (peak_y, peak_x)    ← Correct interpretation")
print()
print("OR the issue is in how we're computing the cross-power spectrum.")
