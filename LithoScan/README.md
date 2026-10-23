# LithoScan

**Automated Drill-Core Lithology Logging System**

> ⚠️ **Research Prototype** — This is an academic research system.  
> The primary dataset (DCID) is licensed under **CC BY-NC 4.0** — academic / non-commercial use only.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset — DCID](#dataset--dcid)
   - [How to Obtain the Dataset](#how-to-obtain-the-dataset)
   - [Directory Layout after Extraction](#directory-layout-after-extraction)
   - [The Four DCID Sub-directories](#the-four-dcid-sub-directories)
   - [Official Train/Test Split Rule](#official-traintest-split-rule)
3. [Project Structure](#project-structure)
4. [Setup](#setup)
5. [Running Dataset Reconnaissance](#running-dataset-reconnaissance)
6. [Running Visualizations](#running-visualizations)
7. [Running Tests](#running-tests)
8. [Configuration Files](#configuration-files)
9. [Planned Experiments](#planned-experiments)
10. [Citation](#citation)
11. [License](#license)

---

## Project Overview

LithoScan is a research prototype for automated drill-core lithology logging.  
It combines:

| Component | Description |
|-----------|-------------|
| **Server Edition** | Texture-aware CNN + Transformer fusion model |
| **Field Edition** | Lightweight Spiking Neural Network (SNN) |
| **Conformal Prediction** | Triage layer with calibrated confidence/error control |
| **Federated Learning** | Simulation with Byzantine-robust aggregation |
| **Bias Mitigation** | Artifact-bias correction for cracks, mud, scale-bar regions |

> **Status**: Phase 1 (infrastructure & dataset reconnaissance) is complete.  
> Model training has **not** been started yet.

---

## Dataset — DCID

### How to Obtain the Dataset

The DCID dataset is available directly on **Hugging Face Hub**:
- **Dataset Viewer & Hub:** [huggingface.co/datasets/168sir/drill-core-image-dataset](https://huggingface.co/datasets/168sir/drill-core-image-dataset)
- **Direct Zip Archive (3.8 GB):** [Download DCID.zip](https://huggingface.co/datasets/168sir/drill-core-image-dataset/resolve/main/DCID.zip)
- **Original Authors GitHub:** [github.com/JiayuLi1120/drill-core-image-dataset](https://github.com/JiayuLi1120/drill-core-image-dataset)

**Automated 1-command download:**
From the repository root, run:
```bash
python download_dataset.py
```
This automatically downloads `DCID.zip` and extracts it to `DCID/DCID`.

### Directory Layout after Extraction

Place (or symlink) the extracted archive at:

```
Software_lithology/
└── DCID/
    └── DCID/                        ← this is DCID_ROOT
        ├── DCID-512-7/
        │   ├── train/
        │   │   ├── 1.Red sandstone/
        │   │   ├── 2.Light sandstone/
        │   │   └── ...
        │   └── test/
        │       └── ...
        ├── DCID-512-35/
        │   ├── train/
        │   └── test/
        ├── noise-512-7/
        │   ├── train/
        │   └── test/
        ├── noise-512-35/
        │   ├── train/
        │   └── test/
        └── build_dcid_dataset.py    ← original helper from the authors
```

### The Four DCID Sub-directories

| Directory | Classes | Images/class | Format | Description |
|-----------|--------:|-------------:|--------|-------------|
| `DCID-512-7`  |  7 | 4,000 train + 1,000 test | JPG 512×512 | 7 coarse lithology classes — clean images |
| `DCID-512-35` | 35 | varies (see manifest)    | JPG 512×512 | 35 fine-grained classes — clean images |
| `noise-512-7` |  7 | mirrors DCID-512-7       | JPG 512×512 | Real-world defective variants (cracks, mud, scale-bars) — 7 classes |
| `noise-512-35`| 35 | mirrors DCID-512-35      | JPG 512×512 | Real-world defective variants — 35 classes |

**DCID-7 Class Names** (in label order):

| ID | Class |
|----|-------|
| 0 | Red sandstone |
| 1 | Light sandstone |
| 2 | Gray siltstone |
| 3 | Mudstone |
| 4 | Granite |
| 5 | Basalt |
| 6 | Marble |

### Official Train/Test Split Rule

> ❗ **Critical**: Never randomly re-split the DCID dataset.  
> Always use the official `train/` and `test/` sub-directories exactly as provided.  
> Any attempt to override the split raises a `SplitViolationError` in the code.

---

## Project Structure

```
LithoScan/
├── data/
│   ├── raw/DCID/          ← symlink or copy of DCID_ROOT (optional)
│   ├── processed/         ← pre-processed tensors (future)
│   ├── manifests/         ← dcid_manifest.csv (auto-generated)
│   ├── calibration/       ← conformal prediction calibration data
│   ├── site_data/         ← federated learning site partitions
│   │   ├── site_A/
│   │   ├── site_B/
│   │   └── site_C/
│   └── final_test/        ← held-out final evaluation set
├── notebooks/             ← Jupyter notebooks for exploration
├── src/
│   ├── data/
│   │   ├── explore_dcid.py      ← reconnaissance script
│   │   ├── dcid_dataset.py      ← PyTorch Dataset + DataLoader factory
│   │   ├── split_utils.py       ← split safety utilities
│   │   └── visualize_dataset.py ← visualization suite
│   ├── models/            ← TODO: CNN+Transformer, SNN
│   ├── training/          ← TODO: training loops
│   ├── evaluation/        ← TODO: evaluation + metrics
│   ├── calibration/       ← TODO: conformal prediction
│   └── federated/         ← TODO: FL simulation + Byzantine-robust aggregation
├── configs/
│   ├── dcid7.yaml         ← DCID-7 experiment configs (A, C–H)
│   └── dcid35.yaml        ← DCID-35 experiment configs (B, C–H)
├── reports/
│   ├── dataset_report.md  ← auto-generated reconnaissance report
│   └── figures/           ← auto-generated plots
├── checkpoints/           ← model checkpoints (future)
├── tests/
│   └── test_dataset.py    ← pytest test suite
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Set DCID root if different from default
set DCID_ROOT=C:\path\to\DCID\DCID      # Windows CMD
$env:DCID_ROOT="C:\path\to\DCID\DCID"  # PowerShell
```

---

## Running Dataset Reconnaissance

```bash
# From the LithoScan/ project root:
python src/data/explore_dcid.py

# Override paths:
python src/data/explore_dcid.py \
    --dcid-root "C:\Users\punee\Desktop\Software_lithology\DCID\DCID" \
    --project-root "."
```

**Outputs:**
- `data/manifests/dcid_manifest.csv` — one row per image with filepath, class, split, dimensions, MD5, corruption flag
- `reports/dataset_report.md` — full Markdown report with class distributions and quality summary

---

## Running Visualizations

```bash
python src/data/visualize_dataset.py
```

**Outputs** (in `reports/figures/`):
- `class_dist_dcid7.png` — class distribution bar chart for DCID-7
- `class_dist_dcid35.png` — class distribution bar chart for DCID-35
- `samples_dcid7.png` — per-class sample image grid (DCID-7)
- `samples_dcid35.png` — per-class sample image grid (DCID-35)
- `clean_vs_noise.png` — side-by-side clean vs defective comparison

---

## Running Tests

```bash
# From LithoScan/ project root:
python -m pytest tests/test_dataset.py -v

# With short traceback:
python -m pytest tests/test_dataset.py -v --tb=short
```

Tests verify: dataset discovery, label consistency, image loading, no train/test leakage, class counts, and split safety.

---

## Configuration Files

| File | Covers |
|------|--------|
| `configs/dcid7.yaml`  | Experiments A, C, D, E, F, G, H (7-class) |
| `configs/dcid35.yaml` | Experiments B, C, D, E, F, G, H (35-class) |

Load a config in Python:
```python
import yaml
with open("configs/dcid7.yaml") as f:
    cfg = yaml.safe_load(f)
```

---

## Planned Experiments

| Exp | Name | Status |
|-----|------|--------|
| A | DCID-7 baseline classifier (centralized, clean) | ⏳ TODO |
| B | DCID-35 fine-grained classifier | ⏳ TODO |
| C | Clean vs noisy robustness | ⏳ TODO |
| D | Artifact-bias mitigation | ⏳ TODO |
| E | Server Edition (CNN+Transformer) vs Field Edition (SNN) | ⏳ TODO |
| F | Centralized vs federated training | ⏳ TODO |
| G | FedAvg vs Byzantine-robust aggregation | ⏳ TODO |
| H | Conformal prediction calibration | ⏳ TODO |

> No accuracy numbers are reported until experiments have actually been run.

---

## Citation

If you use DCID in your research, please cite:

```bibtex
@article{li2025dcid,
  author  = {Jia-Yu Li and Ji-Zhou Tang and Xian-Zheng Zhao and Bo Fan and
             Wen-Ya Jiang and Shun-Yao Song and Jian-Bing Li and
             Kai-Da Chen and Zheng-Guang Zhao},
  title   = {A large-scale, high-quality dataset for lithology identification:
             Construction and applications},
  journal = {Petroleum Science},
  year    = {2025},
  issn    = {1995-8226},
  doi     = {10.1016/j.petsci.2025.04.013}
}
```

---

## License

- **DCID Dataset**: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — academic / non-commercial use only.
- **LithoScan code**: See `LICENSE` file (to be added).
