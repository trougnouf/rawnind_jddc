# rawnind.libs — Core Processing and Training Libraries

## Backgrounder

This module implements the foundational infrastructure for working with raw camera sensor data in the context of deep learning research. Understanding these libraries requires familiarity with several distinct but interconnected domains: digital camera sensor technology, color science, image quality assessment, and the practical constraints of training neural networks on high-dynamic-range imagery.

### The Raw Sensor Data Problem

Digital cameras capture light through a single sensor plane covered by a color filter array—most commonly the bayer pattern, which arranges red, green, and blue filters in a specific mosaic (RGGB in a 2×2 repeating unit). This means each pixel location records only one color channel; the camera's internal processor must infer the missing color information through demosaicing algorithms. Consumer cameras perform this demosaicing, apply color correction, tone mapping, noise reduction, and compression automatically, producing JPEG or similar outputs.

For research purposes, this pipeline is problematic. Each processing step introduces artifacts and discards information. More critically, the noise characteristics of the final image bear little resemblance to the actual sensor noise, making it unsuitable for training denoising networks that must operate on real-world data. This motivates working with raw sensor data directly.

However, raw sensor data presents its own challenges. The mosaiced structure means that standard image processing operations (convolution, downsampling, etc.) must either respect the color filter pattern or perform demosaicing first. The linear sensor response spans a high dynamic range, requiring careful normalization. Each camera model has a unique color filter spectral response, necessitating per-device color correction matrices. And the noise is signal-dependent, non-Gaussian, and varies across the sensor.

### The RawNIND Dataset Context

This codebase supports research using the Raw Natural Image Noise Dataset (RawNIND), which provides paired clean and noisy captures of identical scenes. Creating such pairs requires solving a fundamental alignment problem: even when using a tripod, minute camera movements between the clean (long exposure) and noisy (short exposure) captures introduce sub-pixel misalignments. These shifts must be detected and corrected, or they will dominate the training signal and teach the network to perform spatial shifting rather than denoising.

The alignment problem is complicated by the high dynamic range of the data. Small exposure differences between captures require gain normalization. Regions where the images cannot be precisely aligned—due to motion blur, scene changes, or intrinsic noise—must be masked out. Overexposed regions in either image must be excluded because clipped highlights provide no useful gradient information. The result is that a significant fraction of each image pair may be unusable, and some pairs must be discarded entirely.

### Color Space Transformations and Standardization

Raw sensor values exist in a device-specific "camera RGB" color space determined by the spectral sensitivities of that particular sensor's color filters. To train networks that generalize across cameras, these values must be transformed to a standardized, perceptually-relevant color space. This codebase uses linear Rec.2020 as the canonical "profiled RGB" representation.

The transformation requires a 3×3 color correction matrix derived from the camera's white balance multipliers and its factory-calibrated response under a standard illuminant. This matrix is not a simple scaling; it represents a change of basis between the camera's color primaries and the Rec.2020 primaries, accounting for the fact that no real camera's red filter captures only "red" light in a perceptual sense.

The linearity of this representation is crucial for neural network training. Gamma-encoded (nonlinear) representations compress the signal space in ways that interact poorly with standard loss functions. A mean squared error in gamma-encoded space does not correspond to perceptual differences. Working in linear space with perceptual loss functions (like MS-SSIM computed on gamma-encoded versions) provides better optimization landscapes.

However, linear representations present visualization challenges. Human vision is approximately logarithmic; linear values must be tone-mapped for display. This codebase supports several transfer functions (gamma 2.2, PQ, etc.) for visualization while maintaining linear representations during training.

### Dataset Architecture and Crop Extraction

Training neural networks on full-resolution raw images (typically 24+ megapixels) is computationally infeasible. The standard approach extracts random crops during training, effectively augmenting the dataset while fitting within GPU memory constraints. But random cropping of aligned image pairs with validity masks introduces subtle complexities.

Each crop must maintain bayer pattern alignment—if the top-left corner lands on an odd-numbered row or column, the color filter pattern shifts, and the network sees corrupted data. Crops must sample regions with sufficient valid (unmasked) pixels; a crop that is 80% masked provides little training signal. The cropping strategy must balance between seeing all regions of the dataset and sampling preferentially from high-quality regions.

For validation and testing, the cropping strategy changes fundamentally. Random crops would make metrics non-reproducible. Instead, validation typically uses deterministic crops from specific locations, or processes full images when memory permits. This dichotomy means dataset classes must support multiple modes of operation.

### The Training Abstraction Hierarchy

Training image-to-image networks follows a consistent pattern regardless of the specific task: load batches, compute loss, backpropagate, validate periodically, checkpoint when validation improves. The abstract trainer classes factor out this common machinery while allowing specialization for different input/output formats.

The key insight is that "bayer → profiled RGB" training differs from "profiled RGB → profiled RGB" training primarily in data loading and preprocessing, not in the core optimization loop. By abstracting the model instantiation and data pipeline while providing a complete training implementation, these base classes let researchers focus on architecture design rather than boilerplate.

The separation between configuration (YAML files), model architecture (subclass-specific), and training loop (base class) reflects a pragmatic division of concerns. Hyperparameters that affect multiple training runs (learning rate schedules, validation frequency) live in configuration files. Architecture choices (number of layers, attention mechanisms) live in model constructors. The interaction between them—when to decay learning rate based on validation plateaus, how to handle gradient accumulation—lives in the base trainer.

### Mask-Aware Training and the Valid Pixel Problem

The validity masks generated during preprocessing represent a form of missing data. Standard neural network training assumes complete data; introducing masks requires careful handling. The most straightforward approach—zeroing the loss for masked pixels—works but introduces bias. A batch where 50% of pixels are masked effectively uses half the batch size, increasing gradient noise.

More subtly, the mask pattern is not random. Masked regions cluster near motion boundaries, overexposed highlights, and image edges where alignment is uncertain. This spatial correlation means that even if 20% of an image is masked, the unmasked regions may not represent the full diversity of image content. A training run might undersample challenging regions.

The current implementation accepts this bias as inevitable, focusing instead on ensuring sufficient valid pixels per crop. The threshold (50% valid) represents a compromise: too strict and many crops are rejected, slowing training; too lenient and gradient noise increases. The optimal threshold likely varies by dataset and architecture.

### Alignment Algorithms and the Subpixel Problem

Aligning two noisy images with subpixel precision is a classical computer vision problem with no perfect solution. FFT-based phase correlation works well for integer pixel shifts and provides a closed-form solution, but degrades in the presence of noise and struggles with subpixel precision. Exhaustive spatial search optimizing a pixelwise loss function can find subpixel shifts through interpolation but scales poorly with search window size.

This codebase implements both approaches and composites them: coarse FFT alignment followed by fine spatial refinement within a small window. The choice of loss function for the spatial search (L1 vs. L2 vs. perceptual) affects which alignments are preferred. L1 is more robust to outliers from motion or extreme noise, while L2 converges faster in smooth regions.

The fundamental limitation is that for some image pairs, no single global shift produces good alignment. Local motion (tree branches moving in wind) cannot be corrected by global translation. These cases motivate the masking system: regions where no plausible shift yields good alignment get masked out rather than corrupting the training data with misaligned pairs.

### The Profiled RGB Convention and Why It Matters

The choice of linear Rec.2020 as the standard color space reflects several considerations. Rec.2020 has wider gamut than sRGB, reducing clipping of saturated colors from modern cameras. Linearity preserves radiometric relationships, meaning that a pixel value of 0.5 represents half the radiant energy of a pixel value of 1.0. This matters for denoising because noise variance scales with signal intensity; a network trained on nonlinear data learns incorrect noise models.

But linearity comes at a cost. Most natural images concentrate values in the lower 20% of the range (corresponding to typical scene reflectances), with bright highlights sparsely populating the upper range. This imbalanced distribution challenges network training. Some architectures benefit from normalizing the input distribution or applying a mild companding curve before processing, then inverting it afterward.

The codebase supports multiple transfer functions precisely to explore these tradeoffs. Research questions about whether to train on linear vs. gamma-encoded data, or whether to use perceptual quantizer (PQ) encoding, require the ability to switch representations without rewriting data loading code.

### Performance Considerations and the Preprocessing Tradeoff

Loading raw files, demosaicing, color-correcting, and extracting crops at runtime would bottleneck training. Preprocessing—converting raw files to OpenEXR, aligning pairs, computing masks—trades disk space and upfront computation for faster iteration during training. A single preprocessing pass might take hours, but subsequent training runs read preprocessed data at GPU-limited speeds.

The preprocessed format (OpenEXR) stores linear floating-point HDR data losslessly. File sizes are large (50-200 MB per image), but modern SSDs can stream them fast enough to saturate GPU training. The alternative—storing compressed raw files and processing on-the-fly—would save disk space but shift the bottleneck to CPU-bound demosaicing and color correction.

This design assumes researchers will run many training experiments on the same dataset, amortizing preprocessing cost across runs. For one-off processing, the tradeoff favors runtime processing. The codebase supports both modes, though the preprocessed path is more thoroughly tested.

### The X-Trans Exception

Fujifilm cameras use a 6×6 non-bayer color filter pattern called X-Trans, claimed to reduce moiré artifacts without an optical low-pass filter. For this codebase, X-Trans creates special-case handling throughout. Standard bayer demosaicing algorithms fail. The mosaiced representation has 9 channels instead of 4, requiring architectural changes if processing pre-demosaiced data.

Alignment and crop extraction for X-Trans require respecting 3×3 block boundaries, not the naive expectation of 6×6 tile boundaries suggested by the pattern periodicity. This insight comes from vkdt (a line-based RAW processing pipeline) and the JCGT 2021 paper "Fast Temporal Reprojection without Motion Vectors": X-Trans demosaicing processes three lines at a time, and alignment shifts that don't preserve 3-line chunks introduce artifacts. Concretely, alignment offsets must be multiples of 3, and crop positions must land on coordinates divisible by 3. Since 3 divides 6, this automatically respects tile boundaries while allowing finer-grained alignment than 6-pixel quantization would permit. For bayer sensors, the analogous constraint is 2×2 block alignment—all coordinates must be even. These constraints are enforced in `raw.py` (for alignment) and `CropProducerStage.py` (for crop extraction).

The pragmatic solution for training: convert X-Trans files to profiled RGB via external demosaicing (using camera manufacturer algorithms or high-quality open-source alternatives) during preprocessing. This sidesteps the X-Trans complexity at the cost of preventing research on X-Trans-specific processing. Given that bayer sensors dominate the market, this compromise seems reasonable for a research codebase focused on general principles rather than vendor-specific optimization.

## Module Organization

### raw.py — Sensor Data to Color Images
Handles the lowest-level transformations: reading proprietary RAW formats, extracting mosaiced sensor values, applying black/white point normalization, demosaicing, color correction, and writing standardized output. This is where color science meets practical file format handling.

### rawproc.py — Image Alignment and Quality Control
Implements the alignment algorithms, gain matching, and mask generation that make paired training data usable. This is where computer vision techniques address the data quality problems inherent in real-world captures.

### rawds.py — Dataset Loaders for Neural Network Training
Bridges the gap between preprocessed image files and PyTorch training loops. Handles the combinatorics of different input/output format combinations (bayer/RGB × clean/noisy) and the mechanics of crop extraction with mask awareness.

### abstract_trainer.py — Training Loop Abstraction
Factors out the common machinery of training, validation, checkpointing, and logging. Subclasses implement model-specific logic; the base class ensures consistent experiment tracking and reproducibility.

### rawtestlib.py — Model Evaluation Infrastructure
Standardizes the process of loading trained models and computing metrics on test sets, ensuring that comparisons between different approaches use identical evaluation procedures.

### alignment_backends.py — Alignment Algorithm Implementations
Separates alignment strategy (FFT vs. spatial search) from the alignment interface, allowing experimentation with different approaches without changing downstream code.

## Audience and Prerequisites

Working effectively with these libraries assumes familiarity with:

- **Color science fundamentals**: CIE XYZ, RGB color spaces, chromatic adaptation, gamma encoding
- **Camera sensor technology**: bayer patterns, RAW file formats, demosaicing algorithms
- **Image quality metrics**: PSNR, SSIM, MS-SSIM, perceptual losses
- **PyTorch dataset conventions**: `__getitem__`, batching, data loaders, preprocessing pipelines
- **High dynamic range imaging**: Linear vs. logarithmic encoding, tone mapping, exposure fusion

The code assumes readers can distinguish between "this pixel is overexposed in the camera sensor" (a physical fact requiring masking) and "this pixel is clipped in the file format" (a data representation issue). It assumes understanding that demosaicing is interpolation, not deconvolution, and that no amount of clever processing recovers information destroyed by color filter projection.

## Related Documentation

- `src/rawnind/dataset/` — Newer async pipeline architecture for data acquisition
- `src/rawnind/models/` — Neural network architectures that consume data from these loaders
- `src/common/libs/` — Shared PyTorch utilities and loss functions
