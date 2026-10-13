# Literature Review: Structure-Guided Restoration of High-Density Islamic Geometric Patterns

## 1. Core Inpainting Methods

### 1.1 EdgeConnect (Nazeri et al., 2019)
- **Paper:** *EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning*
- **Key Idea:** Two-stage approach: Edge Generator → Image Completion Network
- **Relevance:** The "structure-first" philosophy is ideal for geometric patterns where line preservation is critical.
- **Link:** [arXiv:1901.00212](https://arxiv.org/abs/1901.00212)

### 1.2 LaMa (Suvorov et al., 2022)
- **Paper:** *Resolution-robust Large Mask Inpainting with Fourier Convolutions*
- **Key Idea:** Fast Fourier Convolutions (FFC) for image-wide receptive field.
- **Relevance:** Excellent for periodic structures but may prioritize global context over local geometric precision.
- **Link:** [arXiv:2109.07161](https://arxiv.org/abs/2109.07161)

### 1.3 MAT (Li et al., 2022)
- **Paper:** *MAT: Mask-Aware Transformer for Large Hole Image Inpainting*
- **Key Idea:** Transformer-based attention for long-range dependencies.
- **Relevance:** Future work candidate for handling very large damage areas.
- **Link:** [CVPR 2022](https://arxiv.org/abs/2203.15270)

---

## 2. Heritage & Cultural Preservation

### 2.1 Digital Heritage Preservation
- **CyArk Initiative:** 3D laser scanning of Islamic cultural sites.
- **Key Techniques:** LiDAR, Photogrammetry, AR/VR visualization.
- **Gap:** Focus on documentation, not restoration of damaged patterns.

### 2.2 AI for Heritage (2024)
- **Source:** islamonweb.net, reelmind.ai
- **Key Insight:** Growing use of AI for artifact restoration, but primarily for faces and paintings—not high-density geometric patterns.

---

## 3. Islamic Geometric Pattern Analysis

### 3.1 Pattern Classification
- **Paper:** Classification of Islamic Geometric Patterns using CNNs
- **Key Insight:** Pre-trained CNNs achieve 94% accuracy in classifying patterns (frieze, wallpaper, rosette).
- **Gap:** Classification ≠ Restoration. No work on reconstructing *damaged* classified patterns.

### 3.2 Pattern Generation
- **Source:** ResearchGate, Cumincad
- **Key Insight:** DCGANs can synthesize *new* authentic-looking patterns.
- **Gap:** Generation ≠ Restoration. Creating new art is different from fixing old art.

---

## 4. Evaluation Metrics (Comprehensive Framework)

### Tier 1: Standard Image Quality Metrics
| Metric | What it Measures | Limitation |
|--------|-----------------|------------|
| **PSNR** | Pixel-level error | Ignores perceptual quality |
| **SSIM** | Local structural similarity | Limited to single scale |
| **MS-SSIM** | Multi-scale structural similarity | More robust than SSIM |
| **LPIPS** | Perceptual similarity (Deep Learning) | Requires pretrained network |

### Tier 2: Distribution Metrics (Generative Quality)
| Metric | What it Measures | Use Case |
|--------|-----------------|----------|
| **FID** (Fréchet Inception Distance) | Distance between real & generated distributions | Overall realism (requires >10k images) |
| **KID** (Kernel Inception Distance) | Similar to FID, unbiased | Small sample sizes |

### Tier 3: Geometry-Specific Metrics ★ (Domain-Specific Contribution)
| Metric | What it Measures | Why It Matters for Islamic Patterns |
|--------|-----------------|-------------------------------------|
| **Edge IoU** | Overlap of Canny edges (original vs. restored) | Measures if geometric *lines* are preserved |
| **Symmetry Score** | Bilateral/radial symmetry consistency | Islamic patterns are mathematically symmetric |
| **Hough Line Accuracy** | Difference in detected straight lines | Checks if lines remain straight, not curved |
| **Figure of Merit (FOM)** | Edge accuracy with tolerance for slight offsets | Standard edge detection benchmark |

### Tier 4: Human Evaluation
| Metric | What it Measures | Protocol |
|--------|-----------------|----------|
| **MOS** (Mean Opinion Score) | Human quality rating (1-5 scale) | Survey with 20+ participants |
| **A/B Preference Test** | Pairwise comparison ("Which looks better?") | Blind test comparing methods |

> **Note:** Tier 3 metrics are our unique contribution—they evaluate *geometric correctness*, which standard metrics ignore.

---

## 5. Research Gap Summary
1. **No standard dataset** for High-Density Islamic Geometric Pattern restoration.
2. **No benchmark** comparing structure-guided vs. global-context methods for this niche.
3. **No geometric-specific loss functions** (e.g., Symmetry Loss) applied to this domain.
