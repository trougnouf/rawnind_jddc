# common.libs — Shared Utilities and PyTorch Helpers

## Backgrounder

This module provides utility functions that recur across multiple components of the codebase but don't belong to any specific domain. It's the accumulation of helper functions extracted during refactoring when the same pattern appeared in three different files. Some utilities are generic (file I/O, serialization, directory operations), others are specific to PyTorch workflows (device management, loss functions, tensor conversions).

The organization reflects practical evolution rather than principled taxonomy. A function lives here if it's used by both training code and evaluation scripts, or by both Bayer and RGB processing paths, or by multiple model architectures. Functions used in only one place stay in that file; functions used everywhere migrate here.

### The Tension Between Abstraction and Specificity

Generic utility libraries face a fundamental tradeoff. Make them too specific and you end up duplicating logic because the abstraction doesn't quite fit a new use case. Make them too generic and you write elaborate frameworks that impose cognitive overhead for simple tasks.

This codebase leans toward specificity. The utilities assume PyTorch is available, that images are NumPy arrays or PyTorch tensors, that files are YAML or JSON or OpenEXR. They don't attempt to abstract over different deep learning frameworks or support arbitrary serialization formats. This specificity makes individual functions simpler at the cost of portability.

The bet is that for research code, portability matters less than clarity. A utility function that does exactly what this codebase needs, clearly and correctly, is more valuable than a generic function that handles edge cases irrelevant to this project.

## pt_helpers.py — PyTorch Utilities

Helper functions for PyTorch tensors, devices, and model operations.

### Device Management

**`get_device(device_spec)`** translates device specifications ("cuda:0", "cpu", "auto") into PyTorch device objects. The "auto" option selects CUDA if available, falling back to CPU. This encapsulates the pattern of checking `torch.cuda.is_available()` that appears in every training script.

AMD ROCm (for AMD GPUs) identifies as CUDA in PyTorch's API, so the same logic works for both NVIDIA and AMD hardware. The function doesn't attempt to handle multi-GPU topology or specialized accelerators (TPUs, NPUs)—those scenarios are outside this project's scope.

### Tensor I/O

**`fpath_to_tensor(file_path)`** loads an image file (TIFF, OpenEXR, PNG) and returns a PyTorch tensor. It handles the conversion from disk format (may be integer, may be gamma-encoded, may have alpha channels) to the standard representation expected by the models (float32, linear, RGB only).

The function wraps OpenImageIO or OpenCV depending on which can read the format. OpenCV is faster but supports fewer formats; OpenImageIO handles exotic HDR formats but has installation quirks. The abstraction hides this decision from calling code.

**`sdr_pttensor_to_file(tensor, file_path)`** writes a PyTorch tensor to an image file, applying tone mapping if the tensor contains HDR values. "SDR" (standard dynamic range) indicates the output is display-ready, not linear scene-referred data.

This asymmetry (reading preserves linear encoding, writing applies tone mapping) reflects typical usage: models train on linear data but humans inspect outputs in gamma space.

### Model Utilities

**`freeze_model(model)`** disables gradient computation for all parameters. Used when loading a pretrained model as a fixed feature extractor or when evaluating without training.

**`get_lossclass(loss_name)`** looks up loss functions by string name, enabling configuration files to specify losses without importing specific classes.

**`bits_per_value(tensor)`** computes the information content of a tensor (useful for compression models). It estimates the entropy by counting unique values and computing the theoretical minimum bitrate.

## pt_losses.py — Loss Functions and Metrics

Defines loss functions for training and metrics for evaluation.

### MS-SSIM (Multi-Scale Structural Similarity)

**`MS_SSIM_loss`** computes the MS-SSIM distance between two images, suitable for backpropagation during training. MS-SSIM correlates better with human perception than MSE, capturing structural similarity across multiple scales.

The implementation uses Gaussian filters to compute local statistics (mean, variance, covariance) in windows across the image. Multi-scale means computing this at multiple resolutions—original size, 2× downsampled, 4× downsampled, etc. Each scale captures different frequency content; humans are sensitive to structure at all scales.

For training, MS-SSIM must be differentiable. The Gaussian filters and downsampling are implemented as convolutions, making gradients well-defined. The loss is 1 - MS-SSIM, so minimizing the loss maximizes similarity.

**`MS_SSIM_metric`** computes the same value but without gradient tracking, used during validation.

### Loss and Metric Registries

The `losses` and `metrics` dictionaries map names to classes, enabling runtime lookup. Training configurations specify `loss: "MS_SSIM"` in YAML, and the trainer uses `losses["MS_SSIM"]()` to instantiate it.

This indirection allows experimenting with different losses without editing code. A researcher trying L1 vs. L2 vs. MS-SSIM losses can change a configuration parameter rather than modifying Python files.

## utilities.py — General-Purpose Helpers

Miscellaneous functions for file operations, serialization, and data manipulation.

### Serialization

**`load_yaml(file_path)`**, **`dict_to_yaml(data, file_path)`** — YAML is the standard configuration format for this codebase. Human-readable, supports comments, widely used in ML research.

**`jsonfpath_load(file_path)`**, **`dict_to_json(data, file_path)`** — JSON for structured data exchange and logging. More restricted than YAML (no comments, limited types) but guaranteed parseable by any system.

**`picklefpath_to_dict(file_path)`**, **`dict_to_pickle(data, file_path)`** — Python pickle for complex objects that don't serialize to JSON/YAML. Pickle is Python-specific and version-fragile but handles arbitrary data structures.

The diversity of formats reflects pragmatic choices: YAML for human-edited configs, JSON for interoperability, pickle for internal caching.

### File System Operations

**`listfpaths(directory, pattern)`** — List files matching a glob pattern. Wraps `glob.glob` with sensible defaults.

**`get_last_modified_file(directory)`** — Find the most recently modified file, useful for finding the latest checkpoint in a training run.

**`filesize(file_path)`** — Return file size in bytes. Used for logging and progress estimation.

**`backup(file_path)`** — Create a timestamped backup copy before overwriting. Prevents accidental data loss during development.

**`touch(file_path)`** — Create an empty file or update modification time, used for synchronization markers.

These are thin wrappers around standard library functions, adding error handling and consistent behavior. Their value is that calling code doesn't need to remember `os.path.getsize` vs. `os.stat().st_size` or handle exceptions inline.

### Data Structure Manipulation

**`freeze_dict(data)`**, **`unfreeze_dict(data)`** — Convert dictionaries to/from frozen (immutable) form for use as dictionary keys or set elements.

**`shuffle_dictionary(data)`**, **`sort_dictionary(data)`** — Reorder dictionary items, useful for randomization or deterministic iteration.

**`avg_listofdicts(list_of_dicts)`** — Compute average values across a list of dictionaries with matching keys. Used for aggregating metrics across batches or runs.

### Compression Helpers

**`compress_lzma(data)`**, **`decompress_lzma(data)`** — LZMA compression for text data (logs, configurations). High compression ratio at the cost of speed.

**`compress_png(image_array)`** — Lossless PNG compression for images, used when storing intermediate results that may be inspected later.

### Miscellaneous

**`checksum(file_path)`** — Compute file hash for integrity verification, complementing the SHA-1 hashes in the dataset pipeline.

**`mt_runner(function, arguments, num_threads)`** — Simple multiprocessing pool for embarrassingly parallel tasks. Convenience wrapper around `multiprocessing.Pool`.

**`get_date()`** — Return current timestamp in a standard format for logs and filenames.

**`std_bpp(height, width, file_size)`** — Standard bits-per-pixel calculation for image compression metrics.

## np_imgops.py — NumPy Image Operations

Image manipulation functions operating on NumPy arrays.

### Image Loading

**`img_fpath_to_np_flt(file_path)`** loads an image file and returns a float32 NumPy array. It handles format detection, color space conversion, and normalization.

The function tries OpenImageIO first (supports more formats), falling back to OpenCV if unavailable. It normalizes integer images to [0, 1] range and converts color formats to RGB.

This abstraction lets the rest of the codebase assume "images are float32 NumPy arrays in RGB format" without worrying about source file formats.

### Image Manipulation

**`np_pad_img_pair(image1, image2)`** pads two images to the same dimensions, useful when processing pairs with slightly different sizes (due to cropping or alignment).

**`np_crop_img_pair(image1, image2, crop_spec)`** extracts matching crops from two images. The `crop_spec` defines the crop location and size; both images are cropped identically.

**`np_to_img(array, file_path)`** writes a NumPy array to an image file, handling format-specific requirements (clipping to valid ranges, converting float to integer, etc.).

### CropMethod Enumeration

Defines standard cropping strategies: center crop, random crop, specific coordinates. Used by dataset loaders to ensure consistent cropping behavior.

## Design Patterns and Conventions

### Error Handling

Utility functions generally prefer raising exceptions over returning error codes or None. The assumption is that calling code should handle errors at appropriate granularity—a missing file during dataset loading might be recoverable (skip that image), but a missing configuration file should terminate the program.

Functions document exceptions they raise in docstrings. Common patterns: `FileNotFoundError` for missing inputs, `ValueError` for invalid parameters, `RuntimeError` for unexpected conditions.

### Type Annotations

Type hints are incomplete—the codebase predates widespread Python type annotation adoption. Some functions have hints, others don't. Where present, they document intent but aren't enforced by type checkers.

Modern additions include type hints; legacy code is gradually being annotated during refactoring. No systematic effort to achieve full coverage exists—type hints are added when they clarify interfaces, skipped when they add noise.

### Logging vs. Printing

Functions that report progress or warnings use Python's `logging` module rather than print statements. This allows calling code to configure log levels (suppress debug messages in production, enable them during development).

The `Printer` class provides a context-manager-based alternative for cases where structured logging is overkill—simple progress indicators that shouldn't appear in log files.

## Testing and Validation

Some modules include unittest test classes (`Test_utilities`, `TestImgOps`). Coverage is incomplete—tests exist for functions with subtle edge cases or those that caused bugs in the past.

Run tests with `python -m unittest common.libs.utilities.Test_utilities` or similar. These are lightweight smoke tests, not comprehensive suites.

## Performance Considerations

Utility functions prioritize correctness over performance. Most operations (file I/O, JSON parsing) are I/O-bound anyway, making optimization pointless.

Where performance matters (image processing in `np_imgops`, tensor operations in `pt_helpers`), functions use efficient libraries (NumPy, PyTorch) rather than pure Python loops.

The `mt_runner` multiprocessing wrapper provides basic parallelism but has overhead—only use for tasks taking >100ms per item.

## Limitations and Caveats

These utilities assume a Linux-like environment. Windows path handling may have issues (hardcoded `/` separators in some places). MacOS works but is less tested.

Serialization functions don't handle circular references or exotic Python types gracefully. They're sufficient for dictionaries of primitives and NumPy arrays, not general object graphs.

Image I/O functions assume images fit in memory. Processing 100MP images might exhaust RAM. No streaming or tiled processing exists.

## Related Documentation

- `src/rawnind/libs/` — Domain-specific utilities for raw image processing
- `src/rawnind/models/` — Neural networks that use these loss functions
- `src/rawnind/libs/abstract_trainer.py` — Training framework that uses these utilities extensively
