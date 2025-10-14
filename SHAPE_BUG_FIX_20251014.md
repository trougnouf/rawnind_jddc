# Shape Bug Fix - 2025-10-14

## Summary

Fixed shape handling bugs in `crop_producer_stage.py` that were causing `cannot reshape array` errors when processing 3D (C, H, W) image data through the PRGB crop pipeline.

## Root Cause

The crop extraction code assumed 2D (H, W) arrays but was receiving 3D (C, H, W) arrays from the image loading pipeline. Two distinct bugs:

1. **Crop indexing bug (lines 351-356, 516-517)**: Using 2D slicing `[y:y+size, x:x+size]` on 3D arrays produced wrong shapes
2. **Extra dimension bug (lines 540-541)**: Adding `np.newaxis` to already-3D crops created 4D arrays that demosaic couldn't process

## Errors Before Fix

```
SHAPE DEBUG: gt_data.shape=(1, 3460, 5200)
SHAPE DEBUG: gt_crop.shape=(1, 256, 6024)  ← WRONG: took slice from first two dims
SHAPE DEBUG: gt_crop[np.newaxis,:,:].shape=(1, 1, 256, 6024)  ← WORSE
ERROR: cannot reshape array of size 1542144 into shape (1,256,1)
```

After first fix:
```
SHAPE DEBUG: gt_crop.shape=(1, 256, 256)  ← CORRECT
SHAPE DEBUG: gt_crop[np.newaxis,:,:].shape=(1, 1, 256, 256)  ← Still wrong
ERROR: cannot reshape array of size 65536 into shape (1,256,1)
```

## Fixes Applied

### Fix 1: Ellipsis Indexing for Crop Extraction

**File**: `src/rawnind/dataset/crop_producer_stage.py`
**Lines**: 345-356 (alignment cropping), 516-517 (crop extraction)

Changed from 2D-only indexing:
```python
gt_data[y_start:y_end, x_start:x_end]
```

To dimension-agnostic ellipsis indexing:
```python
gt_data[..., y_start:y_end, x_start:x_end]
```

The ellipsis (`...`) means "all preceding dimensions," handling both:
- 2D case: `data[..., y:y+size, x:x+size]` → `(H, W)` slicing → `(crop_size, crop_size)`
- 3D case: `data[..., y:y+size, x:x+size]` → `(C, H, W)` slicing → `(C, crop_size, crop_size)`

Also updated dimension extraction (line 345):
```python
h, w = gt_data.shape[-2:]  # Last two dims are always H, W
```

### Fix 2: Remove Redundant np.newaxis

**File**: `src/rawnind/dataset/crop_producer_stage.py`
**Lines**: 536-541 (demosaic calls)

Changed from adding extra dimension:
```python
gt_camrgb = raw.demosaic(gt_crop[np.newaxis, :, :], metadata)
```

To passing crop directly:
```python
gt_camrgb = raw.demosaic(gt_crop, metadata)  # Already (1, H, W)
```

The ellipsis fix ensures `gt_crop` has shape `(1, 256, 256)` which is exactly what `demosaic()` expects. Adding `np.newaxis` created `(1, 1, 256, 256)` which failed the reshape at `raw.py:598`.

## Verification

Created unit tests (`/tmp/test_crop_shape_fix.py`) verifying:
- ✓ 2D crop extraction produces `(256, 256)`
- ✓ 3D crop extraction produces `(1, 256, 256)`
- ✓ Alignment cropping handles both 2D and 3D
- ✓ Demosaic receives correct `(1, H, W)` shape
- ✓ Demosaic's internal reshape `(H, W, 1)` succeeds

All tests pass.

## Impact

Combined with the memory leak fix (removing `@lru_cache` from `rawproc.py:914`), the pipeline can now:
- Process scenes without accumulating memory (RSS oscillates, doesn't monotonically grow)
- Successfully extract PRGB crops (demosaic path now works)
- Handle both 2D bayer and 3D profiled RGB data formats

## Related Fixes

This completes the work started in `MEMORY_LEAK_SOLUTION_20251014.md`:
1. ✅ Memory leak fixed (@lru_cache removed)
2. ✅ Shape mismatch fixed (ellipsis indexing + remove np.newaxis)
3. ⏳ Smoke test needs full run to confirm end-to-end (downloads still in progress)

## Files Modified

- `src/rawnind/dataset/crop_producer_stage.py` (lines 345, 351-356, 516-517, 536-541)

## Technical Notes

**Why ellipsis indexing?** Python's ellipsis (`...`) expands to as many `:` slices as needed to fill the indexing tuple. For arrays:
- `arr[..., y:y+h, x:x+w]` is equivalent to:
  - `arr[y:y+h, x:x+w]` for 2D `(H, W)`
  - `arr[:, y:y+h, x:x+w]` for 3D `(C, H, W)`
  - `arr[:, :, y:y+h, x:x+w]` for 4D `(B, C, H, W)`

This makes the code dimension-agnostic without isinstance checks or conditional branches.

**Why did the bug appear now?** The PRGB crop type was likely not exercised in previous test runs, or the code path was using cached data that bypassed the shape-sensitive logic. Removing `@lru_cache` forced all code paths to execute with freshly loaded data, exposing latent bugs.
