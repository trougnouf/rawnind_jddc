## Multi‑Scale Structural Similarity (MS‑SSIM) – an ASCII‑art walk‑through  

Below is a **step‑by‑step visualisation** of how MS‑SSIM is computed between two images  
`I₁` (reference) and `I₂` (distorted).  
All operations are **pixel‑wise**; the only "magic" is that we look at the images **at several resolutions (scales)** and combine the results.

### TL;DR (in plain English)
1. **Downsample both images iteratively** using 2×2 average pooling to create a pyramid of scales.  
2. **At each scale** compute the two SSIM components: **luminance** (l) and **contrast-structure** (cs).  
3. **Store mean(cs)** for scales 0 to N-2; store **mean(l·cs)** for the coarsest scale N-1.  
4. **Combine using weighted geometric mean**: multiply all stored values raised to their respective weights.  
5. The result is a single number between 0 and 1 that tells you *how structurally similar* the two images are **across multiple scales**.


```
          +-------------------+       +-------------------+
          |   Full‑resolution |       |   Full‑resolution |
          |   (scale 0)       |       |   (scale 0)       |
          |   I₁   I₂         |       |   I₁   I₂         |
          +-------------------+       +-------------------+
                     |                         |
                     |  ↓   Gaussian blur & ↓   |
                     |   down‑sample (×2)      |
                     v                         v
          +-------------------+       +-------------------+
          |   ½‑resolution    |       |   ½‑resolution    |
          |   (scale 1)       |       |   (scale 1)       |
          +-------------------+       +-------------------+
                     |                         |
                     |  ↓   Gaussian blur & ↓   |
                     |   down‑sample (×2)      |
                     v                         v
          +-------------------+       +-------------------+
          |   ¼‑resolution    | …   … |   ¼‑resolution    |
          |   (scale 2)       |       |   (scale 2)       |
          +-------------------+       +-------------------+
                     .
                     .
                     .
          +-------------------+       +-------------------+
          |   1/2ⁿ‑resolution |       |   1/2ⁿ‑resolution |
          |   (scale N‑1)     |       |   (scale N‑1)     |
          +-------------------+       +-------------------+
```

*Each level (scale) is a **low‑pass filtered & down‑sampled** version of the original images.*  
The standard choice is **5 scales** (N = 5) but any number works.

---

### 1️⃣ What is computed *inside* each scale?

At a given scale `k` we calculate the local statistics needed for SSIM:

```
      μ₁  =  GaussianBlur(I₁)          (local mean of I₁)
      μ₂  =  GaussianBlur(I₂)          (local mean of I₂)

      σ₁² =  GaussianBlur(I₁²) - μ₁²   (local variance of I₁)
      σ₂² =  GaussianBlur(I₂²) - μ₂²   (local variance of I₂)

      σ₁₂ =  GaussianBlur(I₁·I₂) - μ₁·μ₂   (local covariance)
```

From these we build two *comparisons*:

```
          Luminance           (l)  = (2·μ₁·μ₂ + C₁) / (μ₁² + μ₂² + C₁)
          Contrast‑Structure  (cs) = (2·σ₁₂ + C₂) / (σ₁² + σ₂² + C₂)
```

* `C₁` and `C₂` are tiny constants that avoid division by zero.  
* In our implementation: `C₁ = (K₁ × data_range)²` and `C₂ = (K₂ × data_range)²`
* Default constants: `K₁ = 0.01`, `K₂ = 0.03`, `data_range = 1.0` (for normalized images)
* Gaussian kernel parameters: `window_size = 11`, `sigma = 1.5`

**ASCII‑style sketch of one local patch (scale k):**

```
   ┌───────────────────────┐
   │   I₁ patch (window)    │   ← Gaussian blur →  μ₁, σ₁
   └───────────────────────┘
          ↓   (same window)
   ┌───────────────────────┐
   │   I₂ patch (window)    │   ← Gaussian blur →  μ₂, σ₂
   └───────────────────────┘
   (multiply the two patches element‑wise → σ₁₂)
```

The **SSIM map** for that scale is simply  

```
SSIMₖ(x,y) = l(x,y) · cs(x,y)
```

where both the luminance and contrast‑structure terms contribute equally at each pixel.

---

### 2️⃣ How do we *merge* the scales?

At each scale from **finest to coarsest‑1**, we compute and store the mean **contrast‑structure** term `cs`.  
At the **coarsest scale** (the smallest image), we compute the full **SSIM** (luminance × contrast‑structure).

```
MS‑SSIM = ∏_{k=0}^{N‑1} (csₖ) ^ wₖ
```

where:
* For scales 0 to N‑2: `csₖ` is the mean contrast‑structure value
* For scale N‑1 (coarsest): `csₙ₋₁` is the mean of the full SSIM map (l·cs)

* `w₀ … w_{N‑1}` are **pre‑defined weights** that sum to 1.  
  The standard 5-scale weights are:

```
w = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
```

* In words:  
  - At the **finest scale (k = 0)** we focus on *contrast‑structure* details.  
  - At the **coarsest scale (k = N‑1)** we include both *luminance* and *contrast‑structure*.

**ASCII diagram of the final combination**

```
   cs₀ ^ w₀      cs₁ ^ w₁      cs₂ ^ w₂      cs₃ ^ w₃     (l·cs)₄ ^ w₄
   ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐   ┌──────────┐
   │Scale 0│   │Scale 1│   │Scale 2│   │Scale 3│   │ Scale 4  │
   │(finest)│   │       │   │       │   │       │   │(coarsest)│
   └───────┘   └───────┘   └───────┘   └───────┘   └──────────┘
        \          |          |          |          /
         \         |          |          |         /
          \        |          |          |        /
           \       |          |          |       /
            \      |          |          |      /
                Weighted geometric mean → MS‑SSIM
```

The **product** of all those weighted terms yields a **single scalar** in `[0, 1]`.  
- `1` means *perfect similarity* (identical images).  
- Values near `0` indicate *severe distortion*.

---

## 3️⃣ Putting it all together – Pseudo‑code (for clarity)

```python
def compute_msssim(img1, img2, scales=5, weights=[0.0448,0.2856,0.3001,0.2363,0.1333]):
    """Compute MS-SSIM between two images using 2x downsampling pyramid."""
    
    cs_values = []  # Will hold cs or (l·cs) for all scales
    
    current1, current2 = img1.copy(), img2.copy()
    
    # Process scales 0 to N-2: compute and store contrast-structure only
    for k in range(scales - 1):
        luminance, contrast_structure = compute_ssim_components(current1, current2)
        cs_values.append(mean(contrast_structure))
        
        # Downsample by 2x using average pooling for next scale
        current1 = downsample_2x(current1)
        current2 = downsample_2x(current2)
    
    # Process final scale N-1: compute full SSIM (l·cs)
    luminance_final, cs_final = compute_ssim_components(current1, current2)
    ssim_final = luminance_final * cs_final
    cs_values.append(mean(ssim_final))
    
    # Combine using weighted geometric mean: ∏(cs_k^w_k)
    ms_ssim = product(cs_values[k] ** weights[k] for k in range(scales))
    
    return ms_ssim
```

---

## 4️⃣ Quick visual summary (ASCII)

```
Scale 0 (full res)           Scale 1 (½ res)          …   Scale N‑1 (coarsest)
┌───────────────┐            ┌───────────────┐            ┌───────────────┐
│  I₁  ───────► │   ↓2×2   │  I₁  ───────► │   ↓2×2   │  I₁  ───────► │
│  I₂  ───────► │  pool    │  I₂  ───────► │  pool    │  I₂  ───────► │
│  Compute cs   │  ──────►  │  Compute cs   │  ──────►  │ Compute l·cs  │
│  mean(cs₀)    │            │  mean(cs₁)    │            │ mean(l·cs)    │
└───────────────┘            └───────────────┘            └───────────────┘
        │                           │                           │
        └──────────┬────────┬───────┴──────┬─────────┬─────────┘
                   │        │              │         │
                   ▼        ▼              ▼         ▼
             cs₀^w₀  ×  cs₁^w₁  × … × cs₃^w₃ × (l·cs)₄^w₄  → MS‑SSIM
```

---

### Notes

**Minimum Image Size:** For 5-scale MS-SSIM with window_size=11, images must be at least 176×176 pixels (calculated as 2^(N-1) × window_size = 2^4 × 11 = 176). Smaller images will raise a `ValueError`.

**Downsampling Method:** Uses 2×2 average pooling (not Gaussian blur before downsampling), which reshapes the image into 2×2 blocks and averages each block.

**Boundary Handling:** Gaussian filtering uses reflection padding (`mode='reflect'`) to properly handle image edges, matching PyTorch's behavior.

**Reference Implementation:** `src/common/libs/msssim_numpy.py` provides the production implementation with full documentation.
