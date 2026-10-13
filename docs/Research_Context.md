# Research Context: Structure-Guided Restoration of High-Density Islamic Geometric Patterns

**Date:** December 30, 2025
**Status:** Ground Level Restart

---

## 1. The Research Topic
**Title:** *Structure-Guided Restoration of High-Density Islamic Geometric Patterns in Deteriorating Heritage*

**The Core Problem:**
Islamic Geometric Patterns (IGPs) are characterized by "High-Density" (Edge Density > 45%) and strict mathematical symmetry. Standard Deep Learning inpainting models (like LaMa or Stable Diffusion) treat these patterns as random textures, resulting in blurry, geometrically invalid restorations when filling large holes.

**The Objective:**
To demonstrate that a **Structure-Guided Approach** (explicitly reconstructing edges before colors) is superior to Global Context approaches for this specific niche of high-density geometric heritage.

### 1.1 Scope Definition: Restoration vs. Recreation
It is critical to distinguish between these two concepts to define our research boundary:

| Feature | **Restoration (Our Scope)** | **Recreation (Out of Scope)** |
| :--- | :--- | :--- |
| **Goal** | Recover the *exact* original state of a damaged image. | Create a *new* or *plausible* pattern where none exists. |
| **Input** | Partial image + **Ground Truth** (hidden). | Partial image (or text) + **No Ground Truth**. |
| **Success** | Measured by accuracy (PSNR/SSIM) against the original. | Measured by aesthetics (User Preference). |
| **The "Stop" Point** | We stop when we can no longer verify the result against a real photo. | N/A (Open-ended generation). |

> **Rule of Thumb:** If we cannot calculate PSNR because we don't have the original photo, it is Recreation. We strictly stick to Restoration where Ground Truth exists.

### 1.2 The Paradox: How to Fix What We Don't Know?
The user asks: *"What if we don't have the original pattern?"*
This is the core of our research methodology:

1.  **In the Lab (Training/Testing):** We use **Simulated Damage**. We take perfect images and *intentionally* break them. Because we have the original, we can grade the AI's work.
    *   *Logic:* "If the AI can fix a hole in a pattern we *do* know, it learns the mathematical rules of geometry."
2.  **In the Field (Real Application):** We give the AI a *truly* damaged wall where the original is lost forever.
    *   *Logic:* "Because the AI learned the rules of symmetry and geometry in the Lab, we trust it to hallucinate the *mathematically correct* missing piece, even though we can't verify it."

**Conclusion:** We need Ground Truth for *Research* (to prove the model works) so that we can use it for *Restoration* (where Ground Truth is missing).

---

## 2. State of the Art (Literature Review Summary)
### What Exists:
1.  **Digital Heritage:** Extensive work on 3D scanning (LiDAR) and AR/VR visualization of sites.
2.  **Pattern Generation:** Use of GANs to *create* new Islamic patterns from scratch.
3.  **General Inpainting:** Strong models for natural scenes (Places2) and faces (CelebA), typically achieving 25-28 dB PSNR.

### What is Missing (The Research Gap):
1.  **No "ImageNet" for IGPs:** There is no standard, large-scale benchmark dataset for Islamic Geometric Patterns. Most research uses small, synthetic, or private datasets.
2.  **The "Density" Blind Spot:** Existing models are not evaluated on patterns with Edge Density > 40%. They fail to preserve high-frequency geometry.
3.  **Restoration vs. Generation:** Very little work focuses on the *restoration* of damaged high-density patterns using Deep Learning.

---

## 3. Our Unique Contribution (The "Dot")
We are filling the gap by:
1.  **Defining "High-Density":** Quantitatively proving that our dataset (Edge Density ~46) is 3x-4x more complex than standard benchmarks (CelebA ~12).
2.  **Benchmarking:** Comparing **EdgeConnect** (Structure-First) vs. **LaMa** (Global Context) on this specific task.
3.  **Key Finding:** While LaMa may have slightly higher PSNR (pixel accuracy), **EdgeConnect achieves significantly higher SSIM (0.88 vs 0.83)**, proving that structural guidance is essential for geometric fidelity.

---

## 4. The Dataset
*   **Name:** Islamic Patterns Dataset
*   **Source:** Curated from `guest-contributions` (India, Iran, Morocco, etc.).
*   **Size:** ~4,000+ images.
*   **Complexity:** High Entropy (7.47), High Edge Density (45.96).
*   **Status:** Currently raw. Needs formal splitting into `Train`, `Validation`, and `Test` sets.

---

## 5. The Roadmap (Execution Plan)
### Phase 1: Foundation (Current)
*   [ ] **Data Formalization:** Split raw data into standard folders.
*   [ ] **Baseline Establishment:** Re-verify OpenCV baseline (21.27 dB).

### Phase 2: Experimentation
*   [ ] **Ablation Study:** Test EdgeConnect *without* the Edge Generator to prove its value.
*   [ ] **Geometric Augmentation:** Implement random rotations (90°, 180°) to force geometric learning.
*   [ ] **Realistic Damage:** Generate "Crack Masks" to simulate real-world heritage deterioration.

### Phase 3: Advanced Optimization (The "Missing Pieces")
*   [ ] **Symmetry Loss:** Implement a loss function that penalizes asymmetry.
*   [ ] **LPIPS Evaluation:** Use perceptual metrics to better align with human visual judgment.

---

## 6. Pragmatic Assessment & Feasibility
### 6.1 Achievability Check
*   **Must Do (Core Research):** Dataset split, OpenCV baseline, EdgeConnect & LaMa training, PSNR/SSIM/LPIPS evaluation. (Est: 2-3 weeks)
*   **Should Do (Strengthens Paper):** Edge IoU metric, Ablation study (EdgeConnect w/o edges), Geometric Augmentation, Realistic "Crack" masks. (Est: +1 week)
*   **Dream Big (Future Work):** Symmetry Loss, Symmetry Score, Human User Study, MAT/Stable Diffusion (too heavy/complex for current scope).

### 6.2 Model Selection Rationale
| Model | Role | Why Selected |
|-------|------|--------------|
| **OpenCV (Telea)** | Baseline | Represents classical computer vision methods. |
| **EdgeConnect** | Structure-Guided | Tests the hypothesis that "Edges First" is better for geometry. |
| **LaMa** | Global Context | Current SOTA for large masks; tests "Fourier Convolutions" approach. |
| **DeepFill v2** | *Optional* | Additional baseline if time permits. |

> **Note:** We are excluding MAT and Stable Diffusion due to hardware constraints (requires A100/H100) and scope (restoration vs. generation).

### 6.3 Data Augmentation Strategy
We will use **Online Augmentation** (applied on-the-fly during training) to expand the effective dataset size by 8x without consuming disk space.
*   **Rotations:** 90°, 180°, 270° (Valid due to Islamic geometric symmetry).
*   **Flips:** Horizontal, Vertical.
*   **Crops:** Random resized crops to focus on local details.

### 6.4 Hardware Feasibility
*   **Kaggle GPU (T4 x2):** Used for training EdgeConnect and LaMa (~12-16 hours total).
*   **MacBook M1 Pro:** Used for dataset prep, baseline testing, and final evaluation metrics.

---

## 7. References & Sources
*   *EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning* (Nazeri et al., 2019)
*   *Resolution-robust Large Mask Inpainting with Fourier Convolutions (LaMa)* (Suvorov et al., 2022)
*   *Image Inpainting for Cultural Heritage Preservation* (Various surveys, 2020-2024)
