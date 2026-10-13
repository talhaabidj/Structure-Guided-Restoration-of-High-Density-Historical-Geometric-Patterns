# Structure-Guided Restoration of High-Density Historical Geometric Patterns

[![License: CC BY-NC 4.0](https://img.shields.io/badge/Docs-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![License: GPL v3](https://img.shields.io/badge/Code-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

> **Status:** ongoing research at the University of Central Punjab, Lahore, under the supervision of
> **Mr. Muhammad Yasin Nasir** (Faculty of Information Technology & Computer Science).

Can deep learning repair damaged Islamic geometric patterns, the tile work, plaster and woodwork of
heritage sites, without breaking their geometry? This project builds a dataset of geometric patterns,
damages them in controlled ways, and measures how well image-repair (inpainting) models put them back.

## The question

General inpainting models are trained on faces and landscapes and treat everything as texture. Geometric
patterns are different: a star or a strapwork line has to continue exactly, or the repair is visibly
wrong. We compare two families of models:

- **EdgeConnect** (structure first): predicts the missing lines, then fills colour around them.
- **LaMa** (global context): uses Fourier convolutions to see repeating structure across the whole image.

Our hypothesis is that a structure-first model keeps the geometry more accurate on dense patterns, even when
a global-context model scores slightly higher on raw pixel error.

## Dataset

- **4,176 pattern images**, de-duplicated from 14,740 collected, about 95% pure geometric patterns.
  Regions include Egypt, India (Mughal), Iran, Morocco, Spain, Syria, Central Asia and Turkey.
- Split **3,340 / 417 / 419** (train / validation / test), processed at **512 × 512**.
- Training images are augmented on the fly with 90° rotations and flips, so every pattern is seen in
  **8 orientations** (about 33,000 image variants from the 4,176 originals). Geometric patterns are
  symmetric, so these copies stay valid examples.
- Damage is simulated with irregular, rectangular and crack-shaped masks on known originals, so every
  repair can be scored against ground truth.
- The images are **not redistributed here**: they are copyrighted (© David Wade, *Pattern in Islamic Art*,
  educational use). `scripts/prepare_dataset.py` builds the masked dataset from your own image folder.

## Results

Final runs on the 419-image test set, 512 × 512, 90 epochs each.

| Model | Hardware | PSNR (dB) ↑ | SSIM ↑ |
|---|---|---|---|
| LaMa (global context) | Tesla P100 | **23.59** ± 3.21 | 0.833 ± 0.050 |
| EdgeConnect (structure first) | TPU v5e | 23.36 ± 3.86 | **0.886** ± 0.052 |

- LaMa is marginally better on pixel error (PSNR), while **EdgeConnect keeps structure noticeably better
  (SSIM 0.886 vs 0.833)**, which supports the structure-first hypothesis for geometric patterns.
- Both clear the classical OpenCV inpainting reference of **21.27 dB** PSNR. That reference comes from our
  first-phase benchmark on a smaller, mixed dataset and was not re-measured on this one.

### What changed from the first phase

The first phase used a public 1,615-image Islamic-art set (pottery, carpets and calligraphy mixed in) at
256 × 256. There, classical OpenCV inpainting beat both deep models (21.27 dB vs 18.99 dB for the best LaMa
run). Building a clean, pattern-only dataset 2.6× larger, adding a geometric consistency loss (Sobel edges +
Laplacian detail) and training at 512 × 512 is what moved both models past the classical baseline. The loss
curves from that first phase are in [`results/`](results).

## Repository

```
├── docs/
│   ├── Research_Context.md     methodology, scope and roadmap
│   ├── Literature_Review.md    survey of inpainting methods and metrics
│   └── Gap_Analysis.md         the research gaps this work addresses
├── notebooks/
│   ├── training/               EdgeConnect (TPU v5e) and LaMa training notebooks (Kaggle)
│   └── final_runs/             the two runs reported above, with their logs (images removed)
├── scripts/
│   ├── prepare_dataset.py      resizing, splits and synthetic damage masks
│   ├── analyze_dataset.py      size, resolution and edge-density statistics
│   ├── evaluate.py             PSNR, SSIM and MAE over a results folder
│   ├── evaluate_post_processing.py, local_refinement.py   optional sharpening / CLAHE refinement
├── results/                    first-phase loss curves
└── requirements.txt
```

## Running it

```bash
pip install -r requirements.txt
python scripts/prepare_dataset.py --input ./your_images --output ./dataset --size 512
# train with the notebooks in notebooks/training on Kaggle (TPU v5e for EdgeConnect, GPU for LaMa)
python scripts/evaluate.py --gt ./dataset/test/images --pred ./results --output evaluation.json
```

## Author

**Talha Abid**, BS Artificial Intelligence, University of Central Punjab, Lahore.
Supervised by **Mr. Muhammad Yasin Nasir**, Lecturer, Faculty of Information Technology & Computer Science.

## References

1. Nazeri et al., *EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning*, ICCV Workshops 2019.
2. Suvorov et al., *Resolution-robust Large Mask Inpainting with Fourier Convolutions* (LaMa), WACV 2022.

## License

- **Documentation:** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
- **Code:** [GPLv3](https://www.gnu.org/licenses/gpl-3.0)

## Citation

```bibtex
@misc{abid2026geometric,
  author    = {Talha Abid},
  title     = {Structure-Guided Restoration of High-Density Historical Geometric Patterns},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/talhaabidj/Structure-Guided-Restoration-of-High-Density-Historical-Geometric-Patterns}
}
```
