# rawnind.models — Neural Network Architectures

## Backgrounder

This module contains the neural network architectures designed for processing raw camera sensor data. The models address three fundamental tasks that are traditionally handled separately in camera image signal processors: denoising (removing sensor and photon shot noise), demosaicing (reconstructing full-color images from mosaiced sensor data), and compression (reducing file size while preserving perceptual quality).

The key research question this codebase explores is whether jointly optimizing these tasks produces better results than the traditional pipeline approach. Classical camera processing applies each step sequentially—demosaic first, then denoise, then compress—with each stage operating independently. This creates several problems: demosaicing amplifies noise in the interpolated color channels, denoising blurs fine detail that compression could preserve more efficiently, and compression artifacts interact poorly with residual noise. By training end-to-end networks that see raw sensor data and produce final outputs, the models can learn to navigate these tradeoffs jointly.

### The Denoising Problem

Real sensor noise is complicated. At low light levels, photon shot noise dominates—a Poisson process where variance equals the mean. At higher intensities, read noise from the sensor electronics becomes significant—approximately Gaussian but with fixed variance. Dark current contributes temperature-dependent noise. Manufacturing variations create fixed-pattern noise unique to each sensor. The combination is signal-dependent, spatially correlated, and not well-modeled by simple additive Gaussian noise.

Classical denoising algorithms like BM3D make explicit assumptions about noise statistics and operate on fully-demosaiced RGB images. They work well when their assumptions match reality, but struggle with the complex noise characteristics of raw sensor data. Learned approaches can adapt to real noise distributions through data, but require seeing enough examples of clean/noisy pairs to generalize.

The RawNIND dataset provides such pairs: long-exposure "clean" captures paired with short-exposure "noisy" captures of identical scenes. The noise in the short exposures is real camera noise, not synthetic. Training on this data teaches networks the actual statistical structure of sensor noise rather than idealized models.

### The Demosaicing Problem

Bayer demosaicing is fundamentally an ill-posed problem. Each pixel location measures only one color channel; the other two must be inferred from neighbors. Near edges, this inference fails—interpolating across a sharp boundary produces color fringing and zipper artifacts. In textured regions with high-frequency detail, naïve interpolation causes moiré patterns.

Classical algorithms use hand-crafted heuristics: detect edges, interpolate along edges rather than across them, apply different kernels to different frequency bands. They work remarkably well for natural images but fail on adversarial cases. More importantly, they make no use of the semantic content of the image—a classical algorithm treats skin tones identically to foliage, despite the different statistical regularities.

Learned demosaicing can exploit image semantics. A network that has seen thousands of faces learns that skin has certain smoothness properties; one that has seen foliage learns the characteristic spectral signatures of chlorophyll. By training on diverse natural images, the network builds implicit priors about what color patterns are plausible.

### The Compression Problem

Lossy image compression is a rate-distortion optimization: given a bitrate budget, maximize perceptual quality, or equivalently, given a quality target, minimize bitrate. JPEG and its successors solve this through transform coding: convert to frequency domain (DCT), quantize coefficients (losing information preferentially in high frequencies where human vision is less sensitive), and entropy code the quantized values.

This approach has limitations. The DCT assumes images are smooth within 8×8 blocks, failing on sharp edges. The quantization is fixed and cannot adapt to local image content—it allocates bits uniformly rather than spending more bits on visually complex regions. And the quality metric (MSE in DCT space) correlates imperfectly with human perception.

Learned compression uses neural networks as both the transform (encoder) and inverse transform (decoder). The encoder maps images to a latent representation optimized for compression rather than frequency localization. The decoder reconstructs images from this representation. A separate entropy model estimates the probability distribution of latent codes, enabling arithmetic coding to approach the theoretical minimum bitrate.

The key advantage is adaptivity. The network can learn to allocate latent capacity to perceptually important features—spending bits on texture that matters while simplifying imperceptible detail. By training with perceptual loss functions rather than MSE, the network optimizes directly for human vision.

### Joint Optimization and Why It Helps

Consider the traditional pipeline: demosaic raw sensor data, denoise the RGB result, compress the denoised image. Each step makes irrevocable decisions. Demosaicing interpolates color channels, creating correlations that didn't exist in the sensor data. Denoising smooths these interpolated values, discarding signal that the next stage could use. Compression quantizes both signal and residual noise, wasting bits on noise that should have been removed earlier.

A joint model can make better global decisions. It can leave noise in the latent representation where it compresses efficiently (high-frequency noise has low entropy after transform coding) while preserving signal details. It can defer demosaicing decisions until after compression, potentially transmitting mosaiced data and demosaicing at the decoder. It can learn which artifacts matter perceptually—color fringing near edges might be less objectionable than texture loss in smooth regions.

The tradeoff is complexity. Joint models are harder to train (multiple conflicting objectives), harder to interpret (no clean separation of concerns), and harder to deploy (non-standard pipeline incompatible with existing infrastructure). Whether joint optimization's benefits outweigh these costs is an empirical question this research investigates.

## Architecture Families

### Denoising-Only Networks (raw_denoiser.py)

These models take noisy images and produce denoised outputs, ignoring compression. They serve as baselines for evaluating whether joint models sacrifice denoising quality for compression efficiency.

**UtNet2 and UtNet3** are U-Net variants with architectural modifications for raw sensor data. The U-Net architecture—encoder-decoder with skip connections at matching resolutions—has proven effective for image-to-image tasks. The "Ut" variants adapt this for raw data processing, handling Bayer patterns and high dynamic range appropriately.

The key design choice is whether to operate on mosaiced Bayer data directly or demosaic first then denoise in RGB space. Operating on Bayer data potentially preserves more information but requires specialized convolution layers that respect the color filter pattern. Demosaicing first is simpler but commits early to interpolation decisions. Both approaches appear in the architectures, allowing empirical comparison.

**Passthrough** is a trivial baseline that outputs its input unchanged. It establishes the "do nothing" performance level.

### Compression Networks (compression_autoencoders.py)

These implement learned image compression: encoder maps images to latent codes, decoder reconstructs from latents, entropy model enables arithmetic coding.

**Ballé Encoder/Decoder** are based on the architecture from Ballé et al.'s seminal work on variational image compression. The encoder uses strided convolutions to downsample while increasing channel count, concentrating information into a compact latent representation. The decoder uses transposed convolutions to upsample. Generalized divisive normalization (GDN) layers replace standard normalization, motivated by models of biological vision.

The latent representation is quantized during training using additive uniform noise (straight-through estimator), simulating quantization while maintaining differentiability. At test time, actual rounding replaces the noise. An entropy model (typically a Gaussian mixture or autoregressive model) estimates latent statistics for the arithmetic coder.

**BayerPSDecoder and BayerTCDecoder** are decoder variants that output Bayer-pattern data rather than RGB. "PS" (pixel shuffle) and "TC" (transposed convolution) refer to different upsampling strategies. These enable end-to-end compression of raw sensor data without demosaicing, potentially preserving information lost in traditional pipelines.

### Joint Denoise-Compress Models (denoise_then_compress.py)

**DenoiseThenCompress** chains a denoiser and compressor in sequence. Despite the name suggesting separate stages, the entire pipeline is trained end-to-end. Gradients flow from the compression loss back through the denoiser, allowing the denoiser to make decisions informed by downstream compression.

This represents a middle ground between fully separate pipelines and architectures designed from scratch for joint optimization. It reuses existing denoiser and compressor architectures, gaining interpretability (the denoiser output can be inspected) at the cost of potentially suboptimal information flow.

### Multi-Scale Compression (manynets_compression.py)

**ManyPriors_RawImageCompressor** extends the basic compression architecture with multiple scales of latent representations. The encoder produces representations at multiple resolutions; the decoder fuses them during reconstruction. This allows the model to capture both global structure (coarse scales) and local detail (fine scales), similar to wavelet-based compression but with learned rather than fixed transforms.

The "many priors" refers to having separate entropy models for each scale. Coarser scales are compressed first, then used as context when compressing finer scales—a form of conditional coding that exploits inter-scale dependencies.

### Baseline Compressors (standard_compressor.py)

These wrap standard codecs (JPEG, BPG, JPEG XL) in a common interface, enabling direct comparison with learned methods. They write images to temporary files, invoke external compression tools, and read back the results—an admitted hack, but pragmatic for research purposes.

**JPEG_ImageCompressor**, **BPG_ImageCompressor**, **JPEGXS_ImageCompressor**, **JPEGXL_ImageCompressor** each implement quality-level parameterization appropriate to that codec. JPEG uses quality factors (0-100), JPEG XL uses distance parameters, etc.

**Passthrough_ImageCompressor** returns the input unchanged, establishing uncompressed baselines.

### BM3D Baseline (bm3d_denoiser.py)

**BM3D_Denoiser** wraps the BM3D algorithm (Block-Matching and 3D filtering), a classical denoising method that achieved state-of-art results before deep learning. It serves as a strong traditional baseline.

BM3D operates by finding similar patches throughout the image, stacking them into 3D arrays, applying collaborative filtering (3D transform → threshold → inverse transform), and aggregating the results. It assumes additive white Gaussian noise with known variance.

For raw data with signal-dependent noise, BM3D requires preprocessing to stabilize the noise variance (variance-stabilizing transform) and post-processing to invert it. The wrapper handles this, making BM3D comparable to learned methods.

## Training Considerations

### Loss Functions and Perceptual Quality

Mean squared error (MSE) is a poor proxy for perceptual quality. An image shifted by one pixel has large MSE relative to the reference despite looking nearly identical to humans. Conversely, images with subtle texture loss may have low MSE but appear obviously degraded.

Perceptual loss functions address this by comparing images in feature spaces that correlate with human vision. MS-SSIM (multi-scale structural similarity) captures luminance, contrast, and structure across scales. Losses based on pretrained VGG networks compare high-level features rather than raw pixels.

For compression, rate-distortion optimization requires balancing bitrate (quantified via entropy model) against reconstruction quality (quantified via perceptual loss). The tradeoff is parameterized by a Lagrange multiplier: minimize distortion + λ × rate. Different λ values trace out the rate-distortion curve.

### Handling High Dynamic Range

Raw sensor data spans a dynamic range far exceeding standard image formats (12-16 bits vs. 8 bits). Training networks on this data requires care with numerical precision, loss function scaling, and activation functions.

Many architectures assume input values in [0,1] or [-1,1]. Naïvely scaling 14-bit sensor data to [0,1] concentrates most values in the lower 10% of the range, wasting representation capacity. Some form of companding (logarithmic or PQ curve) balances the distribution, though this conflicts with training on linear data for radiometric correctness.

The current approach preprocesses data to linear Rec.2020 with values typically in [0,1] for normal scenes, occasionally exceeding 1.0 for bright specular highlights. Networks must handle this gracefully, either through clipping, saturation-handling activation functions, or expanding the representational range.

### Bayer Pattern Alignment

When processing Bayer data directly, convolution kernels must not mix color channels inappropriately. A naïve 3×3 convolution on mosaiced data would combine red, green, and blue pixels that measure different scene locations, creating false correlations.

Two solutions exist: use 1×1 convolutions (no spatial mixing, limiting receptive field) or reshape Bayer data into 4-channel images where each channel contains only one color (separating R, G1, G2, B). The latter enables standard convolutions but requires care with downsampling/upsampling to maintain alignment.

### Data Augmentation

Image-to-image models benefit from augmentation: random flips, rotations (by 90° to preserve Bayer alignment), crops. For supervised tasks with paired inputs (clean/noisy), augmentations must be applied consistently to both images.

Augmentation is particularly important for small datasets. RawNIND provides ~1000 image pairs—sufficient to prevent gross overfitting, but more diversity helps. Random crops effectively multiply the dataset size, though crops from the same image are not truly independent.

## Model Instantiation and Configuration

Each model class defines an `__init__` method accepting architecture-specific parameters: number of layers, channel counts, activation functions, etc. Training scripts instantiate models by name, looking up constructors in architecture dictionaries.

This indirection allows experiments to be configured via YAML files rather than code changes. A configuration specifying `architecture: "UtNet3"` and parameters `{num_layers: 5, channels: 64}` constructs the appropriate model without editing Python files.

The pattern trades explicitness for flexibility. Understanding what model is being trained requires consulting both the configuration file and the architecture implementation. But running ablation studies (varying depth, width, etc.) becomes straightforward.

## Evaluation and Comparison

Model comparison requires controlling for computational cost. A model with 10× more parameters may achieve better quality, but is that a fair comparison? The research literature lacks consensus on how to account for complexity.

This codebase computes parameter counts and FLOPs for each model, logging them alongside quality metrics. Post-hoc analysis can construct Pareto frontiers trading off quality against efficiency. A model strictly dominated by another (worse quality at higher cost) is clearly inferior; models on the frontier represent different points in the tradeoff space.

For compression specifically, bitrate provides a natural normalization. Comparing denoising quality at equal bitrate controls for one axis of the tradeoff, though computational cost at inference still varies.

## Limitations and Caveats

These architectures reflect research explorations rather than production-optimized implementations. Performance has not been systematically profiled. Memory usage during training may be suboptimal. Inference speed is adequate for research but may not meet real-time requirements.

The architectures assume PyTorch's conventions: batch-first tensors (N×C×H×W), floating-point computation, GPU acceleration. Adapting them for other frameworks or deployment constraints (quantized inference, mobile devices) would require non-trivial effort.

Some models include hardcoded assumptions specific to RawNIND: particular image dimensions, channel counts, dynamic ranges. Generalizing to other datasets may require parameter tuning or architectural modifications.

## Related Documentation

- `src/rawnind/libs/abstract_trainer.py` — Training infrastructure that instantiates and optimizes these models
- `src/common/libs/pt_losses.py` — Loss functions used during training
- `CLAUDE.md` — Entry points for training different model types
