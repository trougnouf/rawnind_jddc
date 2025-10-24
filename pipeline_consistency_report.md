# Pipeline Implementation Consistency Report

**Date:** 2025-01-10  
**Analysis:** Comparison of preprocessing analysis and architecture diagram against code implementation

## Executive Summary

The implementation is largely consistent with the analysis. Core preprocessing operations, loss masking, color transformations, and training infrastructure match specifications. Two key divergences remain: demosaicing algorithm choice and patch generation strategy. Initial alignment domain difference is intentional per design.

## Consistencies

### Sensor RAW Preprocessing
- Empty border removal: [`raw.rm_empty_borders()`](src/rawnind/libs/raw.py:212)
- Black-level subtraction and white-level normalization: [`raw.scale_img_to_bw_points()`](src/rawnind/libs/raw.py:470)
- Bayer pattern standardization to RGGB via edge cropping: [`raw.mono_any_to_mono_rggb()`](src/rawnind/libs/raw.py:141)
- Pattern-size-constrained dimensions (4 for Bayer, 6 for X-Trans): [`raw.ensure_correct_shape()`](src/rawnind/libs/raw.py:274)

### Color Space Transformations
- EXIF matrix extraction (rgb_xyz_matrix): [`raw.raw_fpath_to_mono_img_and_metadata():323`](src/rawnind/libs/raw.py:323)
- Matrix inversion and composition: [`raw.get_camRGB_to_profiledRGB_img_matrix()`](src/rawnind/libs/raw.py:1196)
- CamRGB → Rec.2020 transformation: [`raw.camRGB_to_profiledRGB_img()`](src/rawnind/libs/raw.py:1218)
- PRGB pipeline applies transformation at crop generation: [`CropProducerStage.py:430`](src/rawnind/dataset/CropProducerStage.py:430)

### Loss Masking Pipeline
- **Binary opening operation implemented**: 
  - [`rawproc.make_loss_mask():662`](src/rawnind/libs/rawproc.py:662)
  - [`rawproc.make_loss_mask_bayer():517`](src/rawnind/libs/rawproc.py:517)
  - [`rawproc.make_loss_mask_msssim_bayer():623`](src/rawnind/libs/rawproc.py:623)
- Loss thresholds (0.4, 99.99th percentile): [`rawproc.py:34,36,656`](src/rawnind/libs/rawproc.py:34)
- Overexposure threshold via `metadata['overexposure_lb']`: [`raw.scale_img_to_bw_points():479`](src/rawnind/libs/raw.py:479)
- **Crop validation correct** (keeps crops with ≥50% valid pixels): [`CropProducerStage.py:378`](src/rawnind/dataset/CropProducerStage.py:378)

### Training Infrastructure
- **Developed image track implemented**: [`arbitrary_proc_fun.py`](src/rawnind/libs/arbitrary_proc_fun.py)
  - Tone mapping (Reinhard/Drago/Log): lines 99-160
  - Edge enhancement (Laplacian): line 265
  - Gamma correction: line 163
  - Contrast (CLAHE + sigmoid): lines 198, 433
  - Sharpening: lines 44-46
  - Randomized parameters: [`ARBITRARY_PROC_PARAMS_RANGE`](src/rawnind/libs/arbitrary_proc_fun.py:48)

- **Clean data augmentation implemented**:
  - Unpaired clean datasets: [`rawds.CleanProfiledRGBCleanBayerImageCropsDataset`](src/rawnind/libs/rawds.py:1356)
  - Extraraw pipeline: [`make_hdr_extraraw_files.py`](src/rawnind/tools/make_hdr_extraraw_files.py)
  - Training integration: [`abstract_trainer.py:1370-1424`](src/rawnind/libs/abstract_trainer.py:1370)

### Training-Time Operations
- Gain normalization: [`alignment_backends.match_gain()`](src/rawnind/libs/alignment_backends.py:33)
- Bayer shift halving and odd-shift trimming: [`alignment_backends.shift_images()`](src/rawnind/libs/alignment_backends.py:52)
- Alignment loss filtering (0.035 threshold): [`MetadataArtificer.process_scene():123`](src/rawnind/dataset/Aligner.py:123)

## Divergences

### 1. Alignment Domain (Intentional Design Choice)

**Analysis specification:**
- RGB domain after demosaicing
- 3×3 neighborhood iterative search
- Constraint: "individual Bayer pixels contain only one color channel and cannot be meaningfully compared"

**Actual implementation:**
- CFA-aware FFT on mosaiced RAW: [`alignment_backends.find_best_alignment_fft_cfa()`](src/rawnind/libs/alignment_backends.py:125)
- Per-channel extraction then FFT correlation: [`raw.fft_phase_correlate_cfa()`](src/rawnind/libs/raw.py:1474)
- RGB 3×3 search exists as fallback: [`alignment_backends.find_best_alignment_bruteforce_rgb()`](src/rawnind/libs/alignment_backends.py:167)

**Status:** User confirmed intentional - CFA-aware extraction resolves the constraint differently

### 2. Demosaicing Algorithm

**Analysis specification:**
- Li's algorithm for Bayer
- Markesteijn for X-Trans

**Actual implementation:**
- OpenImageIO with `algorithm="linear"` (bilinear): [`raw.demosaic():740-746`](src/rawnind/libs/raw.py:740)
- Markesteijn implementation present but not default: [`raw.markesteijn_demosaic():807`](src/rawnind/libs/raw.py:807)

**Impact:** Simpler interpolation vs edge-aware algorithms. May affect RGB track quality, particularly edge preservation and color fringing.

### 3. Patch Generation Strategy

**Analysis specification:**
- Systematic overlapping tiling
- Bayer: 512×512, stride 128
- RGB: 1024×1024, stride 256

**Actual implementation:**
- Random crop selection: [`CropProducerStage.py:360-390`](src/rawnind/dataset/CropProducerStage.py:360)
- Default: `crop_size=512`, `num_crops=10` per pair
- No stride-based systematic tiling

**Impact:** Different spatial coverage and dataset diversity. Random may provide more varied contexts but less systematic boundary conditioning.

## Components Confirmed Present

### Not Previously Verified
- **Binary opening**: Confirmed in all loss mask functions
- **Crop validation**: Confirmed correct (not inverted)
- **Developed image track**: Confirmed in arbitrary_proc_fun.py
- **Clean augmentation**: Confirmed via extraraw datasets and training integration
- **Alignment loss thresholding**: Confirmed at 0.035 in MetadataArtificer

### Still Not Fully Traced
- Training loop PixelShuffle → CamRGB→Rec.2020 timing for Bayer models
- Complete YAML artifact → Dataset → DataLoader → Model integration path
- Evaluation pipeline asymmetry implementation

## Risk Assessment

**Medium Risk:**
- Demosaicing algorithm difference affects RGB track reconstruction quality
- Random vs systematic patching alters dataset characteristics and boundary artifact exposure

**Low Risk:**
- All core preprocessing operations implemented correctly
- Loss masking with morphological cleanup present
- Color transformations mathematically sound
- Clean augmentation and developed image tracks functional

## References

- Analysis: [`preprocessing_analysis_20251010.md`](preprocessing_analysis_20251010.md)
- Architecture: [`PIPELINE_ARCHITECTURE_DIAGRAM_LHS.md`](PIPELINE_ARCHITECTURE_DIAGRAM_LHS.md)
- Key implementation:
  - [`src/rawnind/libs/raw.py`](src/rawnind/libs/raw.py) - RAW I/O, demosaicing, color transforms
  - [`src/rawnind/libs/rawproc.py`](src/rawnind/libs/rawproc.py) - Alignment, loss masking
  - [`src/rawnind/libs/alignment_backends.py`](src/rawnind/libs/alignment_backends.py) - CFA-aware FFT alignment
  - [`src/rawnind/dataset/CropProducerStage.py`](src/rawnind/dataset/CropProducerStage.py) - Crop extraction
  - [`src/rawnind/libs/arbitrary_proc_fun.py`](src/rawnind/libs/arbitrary_proc_fun.py) - Developed image track
  - [`src/rawnind/libs/rawds.py`](src/rawnind/libs/rawds.py) - Dataset classes with clean augmentation