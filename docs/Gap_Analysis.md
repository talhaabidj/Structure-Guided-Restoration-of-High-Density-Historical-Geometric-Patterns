# Gap Analysis: Structure-Guided Restoration of High-Density Islamic Geometric Patterns

This document identifies the specific research gaps that our work aims to address, providing the scientific justification for the novelty of this research.

---

## 1. The Dataset Gap

### Current State of Art
- **CelebA/Places2:** Widely used benchmarks for inpainting (faces, landscapes).
- **Edge Density:** Typically 8-15 (low complexity).
- **Islamic Patterns:** No standardized, publicly available benchmark exists.

### Our Contribution
- **Dataset Size:** 4,000+ curated images from India, Iran, Morocco, Transoxiana.
- **Edge Density:** 45.96 (3x-4x higher than standard benchmarks).
- **Significance:** First quantitatively verified "High-Density" geometric pattern benchmark.

---

## 2. The Methodology Gap

### Current State of Art
- **LaMa/DeepFill/MAT:** Designed for natural scenes; treat patterns as "texture."
- **EdgeConnect:** Structure-guided, but never tested on high-density geometric patterns.
- **Result:** Standard models produce blurry, geometrically invalid results on complex patterns.

### Our Contribution
- **Hypothesis:** Structure-guided methods (EdgeConnect) will outperform global-context methods (LaMa) for geometric fidelity (SSIM).
- **Experiment:** Direct comparison on the same High-Density dataset.
- **Finding:** EdgeConnect achieves SSIM of 0.88 vs. LaMa's 0.83 (6% improvement in structural accuracy).

---

## 3. The Evaluation Gap

### Current State of Art
- **PSNR/SSIM:** Standard metrics, but don't capture "geometric correctness."
- **Problem:** A line shifted by 1 pixel may have good PSNR but is geometrically wrong.

### Our Contribution
- **Proposed Metric:** "Symmetry Score" – measuring if restored patterns maintain bilateral/radial symmetry.
- **LPIPS:** Adding perceptual evaluation to align with human judgment.

---

## 4. The Loss Function Gap

### Current State of Art
- **L1/L2 Loss:** Standard pixel-level reconstruction.
- **Adversarial Loss:** Encourages realism but not geometric precision.
- **Problem:** No penalty for asymmetry or broken lines.

### Our Contribution
- **Proposed Addition:** "Symmetry Consistency Loss" – penalizing deviations from expected symmetry axes.
- **Impact:** Forces the model to learn the *mathematical rules* of Islamic geometry, not just pixel patterns.

---

## 5. The Data Augmentation Gap

### Current State of Art
- **Random Crops/Flips:** Standard augmentations.
- **Problem:** Don't account for the rotational/reflective symmetry inherent in Islamic patterns.

### Our Contribution
- **Proposed Augmentation:** 90°/180°/270° rotations to teach the model that patterns are orientation-invariant.
- **Impact:** More robust learning of geometric invariants.

---

## 6. The "Realistic Damage" Gap

### Current State of Art
- **Synthetic Masks:** Random rectangles or free-form strokes.
- **Problem:** Real damage is cracks, erosion, water stains—not clean boxes.

### Our Contribution
- **Proposed Addition:** "Crack Mask Generator" – procedurally generating realistic deterioration patterns.
- **Impact:** More realistic test scenarios that match real-world heritage restoration needs.

---

## Summary Table

| Gap Category | Current State | Our Contribution | Priority |
|--------------|---------------|------------------|----------|
| Dataset | No High-Density benchmark | 4,000+ images @ 46 edge density | **Critical** |
| Methodology | Untested on HD patterns | EdgeConnect vs. LaMa comparison | **Critical** |
| Evaluation | PSNR/SSIM only | Symmetry Score + LPIPS | High |
| Loss Function | No geometric penalty | Symmetry Consistency Loss | Medium |
| Augmentation | Basic transforms | Geometric rotations | Medium |
| Damage Simulation | Synthetic boxes | Realistic crack masks | Medium |
