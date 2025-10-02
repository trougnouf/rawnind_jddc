"""Implement and test X-Trans channel-split FFT."""
import numpy as np
import sys
from pathlib import Path
sys.path.append("src")
from rawnind.libs import raw
from rawnind.libs.rawproc import shift_images, match_gain

def load_raw_image(fpath: str):
    """Load RAW image as [1, H, W] mosaiced image."""
    img, metadata = raw.raw_fpath_to_mono_img_and_metadata(fpath)
    return img, metadata

def extract_xtrans_channels(img):
    """
    Extract 3 color channels from X-Trans mosaiced image.
    
    X-Trans pattern (6x6):
    G B G G R G
    R G R B G B
    G B G G R G
    G R G G B G
    B G B R G R
    G R G G B G
    
    Instead of extracting all 6 unique positions, we extract by COLOR:
    - R channel: all R positions
    - G channel: all G positions  
    - B channel: all B positions
    
    Returns: (R, G, B) masks and values
    """
    # X-Trans pattern (one 6x6 block)
    # Simplified: we'll identify positions by their color
    # This is the standard X-Trans pattern used by Fujifilm
    xtrans_pattern = np.array([
        [1, 2, 1, 1, 0, 1],  # G B G G R G
        [0, 1, 0, 2, 1, 2],  # R G R B G B
        [1, 2, 1, 1, 0, 1],  # G B G G R G
        [1, 0, 1, 1, 2, 1],  # G R G G B G
        [2, 1, 2, 0, 1, 0],  # B G B R G R
        [1, 0, 1, 1, 2, 1],  # G R G G B G
    ])  # 0=R, 1=G, 2=B
    
    img_2d = img[0]
    h, w = img_2d.shape
    
    # Create full pattern by tiling
    pattern_h = (h // 6 + 1) * 6
    pattern_w = (w // 6 + 1) * 6
    full_pattern = np.tile(xtrans_pattern, (pattern_h // 6, pattern_w // 6))
    full_pattern = full_pattern[:h, :w]
    
    # Extract each color channel
    R_mask = (full_pattern == 0)
    G_mask = (full_pattern == 1)
    B_mask = (full_pattern == 2)
    
    # Get values at each color position
    R_values = img_2d[R_mask]
    G_values = img_2d[G_mask]
    B_values = img_2d[B_mask]
    
    # Create dense arrays by extracting subsampled positions
    # For R: extract positions where pattern == 0
    R_positions = np.argwhere(R_mask)
    G_positions = np.argwhere(G_mask)
    B_positions = np.argwhere(B_mask)
    
    return (R_mask, R_values, R_positions), (G_mask, G_values, G_positions), (B_mask, B_values, B_positions)

def create_dense_channel_image(mask, values, shape):
    """
    Create a dense image from sparse channel data.
    Strategy: Create a smaller dense array by regular sampling.
    """
    h, w = shape
    
    # Find bounding grid of the channel
    positions = np.argwhere(mask)
    
    if len(positions) == 0:
        return np.zeros((h // 6, w // 6), dtype=np.float32)
    
    # Sample every 6th pixel (approximate)
    # For X-Trans, each color appears multiple times in 6x6 block
    # We'll create a downsampled version
    sampled_h = h // 6
    sampled_w = w // 6
    
    dense = np.zeros((sampled_h, sampled_w), dtype=np.float32)
    
    # Fill by averaging nearby values in 6x6 blocks
    for i in range(sampled_h):
        for j in range(sampled_w):
            block_y = i * 6
            block_x = j * 6
            
            # Get values in this 6x6 block for this channel
            block_values = values[block_y:min(block_y+6, h), block_x:min(block_x+6, w)]
            block_mask = mask[block_y:min(block_y+6, h), block_x:min(block_x+6, w)]
            
            # Extract only the values where mask is True
            channel_values = block_values[block_mask]
            if len(channel_values) > 0:
                dense[i, j] = channel_values.mean()
    
    return dense

def extract_xtrans_channels_dense(img):
    """
    Extract X-Trans channels as dense downsampled arrays.
    Returns: (R_dense, G_dense, B_dense)
    """
    img_2d = img[0]
    h, w = img_2d.shape
    
    # X-Trans pattern
    xtrans_pattern = np.array([
        [1, 2, 1, 1, 0, 1],  # G B G G R G
        [0, 1, 0, 2, 1, 2],  # R G R B G B
        [1, 2, 1, 1, 0, 1],  # G B G G R G
        [1, 0, 1, 1, 2, 1],  # G R G G B G
        [2, 1, 2, 0, 1, 0],  # B G B R G R
        [1, 0, 1, 1, 2, 1],  # G R G G B G
    ])  # 0=R, 1=G, 2=B
    
    # Create full pattern
    pattern_h = (h // 6 + 1) * 6
    pattern_w = (w // 6 + 1) * 6
    full_pattern = np.tile(xtrans_pattern, (pattern_h // 6, pattern_w // 6))
    full_pattern = full_pattern[:h, :w]
    
    # Create masks
    R_mask = (full_pattern == 0)
    G_mask = (full_pattern == 1)
    B_mask = (full_pattern == 2)
    
    # Get values
    R_values = img_2d[R_mask]
    G_values = img_2d[G_mask]
    B_values = img_2d[B_mask]
    
    # Create dense channels
    R_dense = create_dense_channel_image(R_mask, img_2d, (h, w))
    G_dense = create_dense_channel_image(G_mask, img_2d, (h, w))
    B_dense = create_dense_channel_image(B_mask, img_2d, (h, w))
    
    return R_dense, G_dense, B_dense

def fft_phase_correlate_single_channel(anchor_ch, target_ch):
    """FFT phase correlation on a single channel."""
    if anchor_ch.size == 0 or target_ch.size == 0:
        return (0, 0)
    
    # Mean center
    anchor_centered = anchor_ch - anchor_ch.mean()
    target_centered = target_ch - target_ch.mean()
    
    # FFT
    f1 = np.fft.fft2(anchor_centered)
    f2 = np.fft.fft2(target_centered)
    
    # Cross-power spectrum
    cross_power = (f1 * np.conj(f2)) / (np.abs(f1 * np.conj(f2)) + 1e-10)
    
    # Inverse FFT
    correlation = np.fft.ifft2(cross_power).real
    
    # Find peak
    h, w = correlation.shape
    peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)
    
    # Handle wraparound
    if peak_y > h // 2:
        peak_y -= h
    if peak_x > w // 2:
        peak_x -= w
    
    return (peak_y, peak_x)

def fft_phase_correlate_xtrans(anchor, target, method="median"):
    """
    FFT phase correlation with X-Trans channel splitting.
    
    Args:
        anchor: [1, H, W] mosaiced X-Trans image
        target: [1, H, W] mosaiced X-Trans image
        method: "median", "mean", or "mode" for combining channel results
    
    Returns:
        shift (dy, dx) in FULL image coordinates
    """
    # Extract X-Trans channels (downsampled by 6)
    R_a, G_a, B_a = extract_xtrans_channels_dense(anchor)
    R_t, G_t, B_t = extract_xtrans_channels_dense(target)
    
    # Run FFT on each channel
    channels = [
        ("R", R_a, R_t),
        ("G", G_a, G_t),
        ("B", B_a, B_t),
    ]
    
    shifts_per_channel = []
    
    for name, anchor_ch, target_ch in channels:
        shift = fft_phase_correlate_single_channel(anchor_ch, target_ch)
        # Scale by 6 for full image (X-Trans is 6x6 pattern)
        shift_full = (shift[0] * 6, shift[1] * 6)
        shifts_per_channel.append(shift_full)
    
    # Combine results
    if method == "median":
        dy_median = int(np.median([s[0] for s in shifts_per_channel]))
        dx_median = int(np.median([s[1] for s in shifts_per_channel]))
        final_shift = (dy_median, dx_median)
    elif method == "mean":
        dy_mean = int(np.round(np.mean([s[0] for s in shifts_per_channel])))
        dx_mean = int(np.round(np.mean([s[1] for s in shifts_per_channel])))
        final_shift = (dy_mean, dx_mean)
    elif method == "mode":
        from collections import Counter
        counter = Counter(shifts_per_channel)
        final_shift = counter.most_common(1)[0][0]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return final_shift, shifts_per_channel

print("="*80)
print("TESTING X-TRANS CHANNEL-SPLIT FFT")
print("="*80)
print()

# First, let's just try to load an X-Trans image
base_path = Path("src/rawnind/datasets/RawNIND/src/X-Trans")

# Find a valid X-Trans scene
test_scenes = ["books", "banana", "beads", "bloop"]

for scene_name in test_scenes:
    scene_dir = base_path / scene_name
    if not scene_dir.exists():
        continue
    
    print(f"Testing scene: {scene_name}")
    
    # Find GT
    gt_dir = scene_dir / "gt"
    if not gt_dir.exists():
        print(f"  ⚠️  No gt/ directory")
        continue
    
    gt_files = list(gt_dir.glob("*.raf"))
    if not gt_files:
        print(f"  ⚠️  No GT .raf files")
        continue
    
    gt_file = gt_files[0]
    print(f"  GT file: {gt_file.name}")
    
    # Try to load
    try:
        print("  Loading GT...")
        gt_img, gt_meta = load_raw_image(str(gt_file))
        print(f"    ✓ Loaded: shape={gt_img.shape}, dtype={gt_img.dtype}")
        print(f"    Camera: {gt_meta.get('make', 'Unknown')} {gt_meta.get('model', 'Unknown')}")
        
        # Test channel extraction
        print("  Extracting X-Trans channels...")
        R_dense, G_dense, B_dense = extract_xtrans_channels_dense(gt_img)
        print(f"    R channel: {R_dense.shape}")
        print(f"    G channel: {G_dense.shape}")
        print(f"    B channel: {B_dense.shape}")
        
        # Find a noisy image
        noisy_files = list(scene_dir.glob("*.raf"))
        noisy_files = [f for f in noisy_files if f.parent.name != "gt"]
        
        if not noisy_files:
            print(f"  ⚠️  No noisy files")
            continue
        
        noisy_file = noisy_files[0]
        print(f"  Noisy file: {noisy_file.name}")
        
        print("  Loading noisy...")
        noisy_img, _ = load_raw_image(str(noisy_file))
        print(f"    ✓ Loaded: shape={noisy_img.shape}")
        
        # Gain match
        noisy_matched = match_gain(gt_img, noisy_img)
        
        # Test X-Trans FFT
        print("  Running X-Trans channel-split FFT...")
        shift, channel_shifts = fft_phase_correlate_xtrans(gt_img, noisy_matched, method="median")
        
        print(f"    Per-channel shifts:")
        print(f"      R: {channel_shifts[0]}")
        print(f"      G: {channel_shifts[1]}")
        print(f"      B: {channel_shifts[2]}")
        print(f"    Final shift (median): {shift}")
        
        # Compute loss
        anchor_out, target_out = shift_images(gt_img, noisy_matched, shift)
        loss = float(np.abs(anchor_out - target_out).mean())
        print(f"    Loss: {loss:.6f}")
        
        # Also check (0,0) for comparison
        anchor_out_0, target_out_0 = shift_images(gt_img, noisy_matched, (0, 0))
        loss_0 = float(np.abs(anchor_out_0 - target_out_0).mean())
        print(f"    Loss at (0,0): {loss_0:.6f}")
        
        if shift != (0, 0):
            print(f"    ✓ Detected misalignment: {shift}")
        else:
            print(f"    - Images appear aligned")
        
        print()
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print()
        continue

print("="*80)
print("X-TRANS IMPLEMENTATION STATUS")
print("="*80)
print("If successful, X-Trans channel-split FFT is working!")
print("The 6x6 pattern is more complex than Bayer 2x2, but the same principle applies.")
