# LithoScan: Automated Drill-Core Lithology Logging System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-orange.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Git LFS](https://img.shields.io/badge/Git-LFS%20Enabled-green.svg)](https://git-lfs.github.com)
[![Dataset CC BY-NC 4.0](https://img.shields.io/badge/Dataset-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

An end-to-end research framework and interactive deployment platform for automated drill-core lithology classification, out-of-distribution robustness evaluation, and calibrated conformal prediction.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Interactive Streamlit Dashboard](#interactive-streamlit-dashboard)
3. [Dataset: DCID (Links & Setup)](#dataset-dcid-links--setup)
   - [Official Dataset Download Links](#official-dataset-download-links)
   - [Automated Download Script](#automated-download-script)
   - [Directory Structure After Extraction](#directory-structure-after-extraction)
   - [Official Train/Test Split Rules](#official-traintest-split-rules)
4. [Pretrained Model Checkpoints](#pretrained-model-checkpoints)
   - [Available Models in the Repository](#available-models-in-the-repository)
   - [Git LFS Setup for Collaborators](#git-lfs-setup-for-collaborators)
   - [Environment & Model Verification](#environment--model-verification)
5. [Quickstart Guide (Step-by-Step)](#quickstart-guide-step-by-step)
6. [Reproducing Paper Benchmarks](#reproducing-paper-benchmarks)
7. [Repository Layout (A to Z)](#repository-layout-a-to-z)
8. [Scientific Citation & License](#scientific-citation--license)

---

## Project Overview

LithoScan addresses the operational and scientific challenges of automated core logging in subsurface geology and mineral exploration:

| Core Component | Technical Implementation | Purpose |
| :--- | :--- | :--- |
| **Server Edition** | Texture-aware ResNet-50 CNN + Transformer Fusion | High-precision server/cloud inference across 7 and 35 lithologies |
| **Field Edition** | Spiking Neural Network (`ConvSNN4`) | Ultra-low power, neuromorphic edge inference for drilling rigs |
| **Conformal Prediction** | Split conformal prediction & calibrated error control | Guaranteed coverage & prediction set triage under uncertainty |
| **Robustness Audit** | Real-World Data Augmentation (RWDA) + dHash Near-Duplicate Filter | Quantifies spatial memorization vs genuine geological generalization |
| **Artifact Mitigation** | Border masking & Grad-CAM visual attention intervention | Eliminates bias from drill cracks, core trays, and scale bars |

---

## Interactive Streamlit Dashboard

LithoScan includes an interactive web application (`app.py`) featuring 7 dedicated analytics pages:

1. **Overview**: Architectural summary, dataset specs, and active phase status.
2. **Model Comparison**: Side-by-side performance comparison (DCID-7 vs DCID-35, Server Fusion vs CNN-Only, accuracy, macro-F1, and latency).
3. **Image Prediction**: Real-time drill core upload, live CNN/Transformer inference, class probabilities, and conformal confidence sets.
4. **Robustness Analysis**: Degradation curves under RWDA defective core conditions (mud, scale bars, physical fractures).
5. **Confusion Matrices**: High-resolution interactive confusion matrices with per-class breakdown.
6. **Experiment Timeline**: Phase-by-phase development history from Phase 1 reconnaissance to Phase 6 SNN edge pilots.
7. **Artifact Explorer**: Direct viewer for publication-ready figures, benchmark JSONs, and audit reports.

### Launching the Dashboard:
```bash
python start.py
```
*(Automatically detects your virtual environment and opens `http://localhost:8501` in your browser).*

---

## Dataset: DCID (Links & Setup)

The system is evaluated on the **Drill Core Image Dataset (DCID)** published in *Petroleum Science* (2025):
- **98,000** high-resolution ($512 \times 512$) core photos.
- **DCID-7**: 7 coarse lithology classes (Red sandstone, Light sandstone, Gray siltstone, Mudstone, Granite, Basalt, Marble).
- **DCID-35**: 35 fine-grained lithology classes.
- **Defective Cohorts (`noise-512-7` & `noise-512-35`)**: Real-world core artifacts including fractures, mud smear, and optical scale markers.

### Official Dataset Download Links

* **Hugging Face Hub (Direct Download & Dataset Viewer):**  
  [https://huggingface.co/datasets/168sir/drill-core-image-dataset](https://huggingface.co/datasets/168sir/drill-core-image-dataset)  
  *Direct Archive Link (3.8 GB):* [DCID.zip Download](https://huggingface.co/datasets/168sir/drill-core-image-dataset/resolve/main/DCID.zip)

* **Original Dataset GitHub Repository:**  
  [https://github.com/JiayuLi1120/drill-core-image-dataset](https://github.com/JiayuLi1120/drill-core-image-dataset)

### Automated Download Script

We provide a built-in automated downloader that streams and extracts the dataset directly:

```bash
python download_dataset.py
```

### Directory Structure After Extraction

Ensure the extracted folder matches the layout expected by all configs and loaders:

```
software_litho/
├── DCID/
│   └── DCID/                          <-- DCID_ROOT
│       ├── DCID-512-7/
│       │   ├── train/                 (4,000 images / class)
│       │   └── test/                  (1,000 images / class)
│       ├── DCID-512-35/
│       │   ├── train/
│       │   └── test/
│       ├── noise-512-7/
│       │   ├── train/
│       │   └── test/
│       ├── noise-512-35/
│       │   ├── train/
│       │   └── test/
│       └── build_dcid_dataset.py
├── LithoScan/
└── README.md
```

> **Note on Custom Dataset Locations:**  
> If your dataset is stored in an external directory or drive, simply set the environment variable:
> ```bash
> # Windows PowerShell
> $env:DCID_ROOT = "D:\path\to\DCID\DCID"
> # Windows CMD
> set DCID_ROOT=D:\path\to\DCID\DCID
> # Linux / macOS
> export DCID_ROOT=/path/to/DCID/DCID
> ```

### Official Train/Test Split Rules

> ⚠️ **CRITICAL SPLIT SAFETY:**  
> Never randomly re-partition the DCID images. The benchmark requires using the official `train/` and `test/` subdirectories. Overriding the official partition raises an enforced `SplitViolationError` to prevent scientific data leakage.

---

## Pretrained Model Checkpoints

All core trained models are included directly in this repository and managed via **Git LFS**:

### Available Models in the Repository

| Model Checkpoint Path | Architecture | Benchmark | Classes | Status |
| :--- | :--- | :--- | :---: | :--- |
| `LithoScan/checkpoints/phase5/dcid35_server/best_model.pt` | ResNet-50 + Transformer | DCID-35 | 35 | Production Server Model |
| `LithoScan/checkpoints/phase5/dcid35_cnn_only/best_model.pt` | ResNet-50 (CNN-Only) | DCID-35 | 35 | CNN Baseline |
| `LithoScan/checkpoints/phase4/dcid7_server/best_model.pt` | ResNet-50 + Transformer | DCID-7 | 7 | Coarse Server Model |
| `LithoScan/checkpoints/phase4/dcid7_cnn_only/best_model.pt` | ResNet-50 (CNN-Only) | DCID-7 | 7 | Coarse Baseline |
| `LithoScan/checkpoints/exp_A/best_model.pt` | ResNet-50 Baseline | DCID-7 | 7 | Centralized Baseline A |
| `LithoScan/checkpoints/exp_B/best_model.pt` | ResNet-50 Baseline | DCID-35 | 35 | Centralized Baseline B |
| `LithoScan/checkpoints/phase6_snn_pilot/best_model.pt` | ConvSNN4 (Neuromorphic) | DCID-7 | 7 | Field Edition Edge Pilot |

### Git LFS Setup for Collaborators

When cloning the repository, Git LFS downloads the actual model weights rather than small pointer text files:

```bash
# 1. Install Git LFS once on your system (if not already installed)
git lfs install

# 2. Pull all checkpoint weights
git lfs pull
```

### Environment & Model Verification

Run the built-in diagnostic script to verify your Python environment, GPU/CUDA acceleration, Git LFS checkpoint integrity, and DCID dataset path:

```bash
python verify_environment.py
```

Expected diagnostic output:
```text
======================================================================
  LithoScan Environment & Checkpoint Integrity Audit
======================================================================
[*] Python Version: 3.11.x
    [OK] Python version OK.
[*] PyTorch Version: 2.2+
[*] CUDA Acceleration: Available (NVIDIA GPU detected)
    [OK] Streamlit and Plotly are installed.

[*] Auditing Pretrained Model Checkpoints:
    [OK] checkpoints/phase5/dcid35_server/best_model.pt (118.6 MB)
    [OK] checkpoints/phase5/dcid35_cnn_only/best_model.pt (94.1 MB)
    [OK] checkpoints/phase4/dcid7_server/best_model.pt (118.6 MB)
    [OK] checkpoints/phase4/dcid7_cnn_only/best_model.pt (94.0 MB)
    [OK] checkpoints/exp_A/best_model.pt (90.0 MB)
    [OK] checkpoints/exp_B/best_model.pt (90.3 MB)
    [OK] checkpoints/phase6_snn_pilot/best_model.pt (1.5 MB)
    [OK] All core model weights verified and ready for inference!
```

---

## Quickstart Guide (Step-by-Step)

### 1. Clone the Repository
```bash
git clone https://github.com/Puneethreddy2530/software_litho.git
cd software_litho
```

### 2. Pull Git LFS Weights
```bash
git lfs install
git lfs pull
```

### 3. Create & Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Verify Setup
```bash
python verify_environment.py
```

### 6. Launch the Interactive Dashboard
```bash
python start.py
```

---

## Reproducing Paper Benchmarks

All reproduction scripts run from `LithoScan/`:

```bash
cd LithoScan
```

### 1. Generalization Stress Test & Deduplication Sensitivity
Evaluates performance after systematically filtering out spatial training neighbors ($d\text{Hash} \le 2, 5, 10$):
```bash
python src/scripts/evaluate_generalization.py
```

### 2. Full Paper B Uncertainty & Calibration Evaluation
Runs split conformal prediction, empirical coverage analysis, and stratified bootstrap testing:
```bash
python src/scripts/run_paperB_experiments.py
```

### 3. Generate Manuscript Tables & Publication Figures
```bash
python src/scripts/generate_paperB_tables.py
python src/scripts/generate_paperB_figures.py
```

All generated tables (`.tex` and `.csv`) and figures (`.png` and `.pdf`) are output directly to `LithoScan/manuscript/` and `LithoScan/results/`.

---

## Repository Layout (A to Z)

```
software_litho/
├── .gitattributes             # Git LFS rules for *.pt weights
├── .gitignore                # Production ignore rules (excludes raw data, venvs)
├── README.md                 # Master documentation & guide
├── requirements.txt          # Root Python dependencies
├── start.py                  # 1-click launcher for the Streamlit dashboard
├── download_dataset.py       # Automated DCID 3.8GB dataset downloader from Hugging Face
├── verify_environment.py     # Diagnostic audit for Python, CUDA, LFS weights, and dataset
├── LithoScan/                # Core application & research module
│   ├── app.py                # 7-page interactive Streamlit analytics dashboard
│   ├── start.py              # Sub-launcher for Streamlit app
│   ├── requirements.txt      # Module dependencies
│   ├── configs/              # Reproducible experiment configurations
│   │   ├── dcid7.yaml
│   │   ├── dcid35.yaml
│   │   ├── phase4_dcid7_server.yaml
│   │   ├── phase4_dcid7_cnn_only.yaml
│   │   ├── phase5_dcid35_server.yaml
│   │   ├── phase5_dcid35_cnn_only.yaml
│   │   └── phase6_snn_pilot.yaml
│   ├── checkpoints/          # Pretrained PyTorch models (Git LFS)
│   │   ├── exp_A/            # Baseline 7-class ResNet-50
│   │   ├── exp_B/            # Baseline 35-class ResNet-50
│   │   ├── phase4/           # DCID-7 Server Fusion & CNN-Only models
│   │   ├── phase5/           # DCID-35 Server Fusion & CNN-Only models
│   │   └── phase6_snn_pilot/ # ConvSNN4 Neuromorphic Field Pilot
│   ├── data/                 # Manifests and dataset audit metadata
│   │   └── manifests/        # Auto-generated image manifests (dcid_manifest.csv)
│   ├── reports/              # Audit reports, dHash neighbor tables, and figures
│   │   ├── phase1/           # Reconnaissance summaries
│   │   ├── phase3/           # Near-duplicate pairs (compressed gz) & visual stats
│   │   ├── phase4/           # Generalization stress test results & confusion matrices
│   │   └── research_readiness_audit_v2/
│   ├── results/              # Authentic benchmark JSONs & evaluated metrics
│   │   └── paperB_uncertainty/
│   │       ├── metrics/      # Full evaluation results
│   │       ├── logits/       # Calibrated output logits
│   │       └── figures/      # Performance plots
│   ├── manuscript/           # Research paper assets & publication tables
│   ├── src/                  # Core library source code
│   │   ├── data/             # DCID Dataset class, transforms, split guards
│   │   ├── models/           # Server Fusion (CNN+Transformer) & ConvSNN architectures
│   │   ├── training/         # Training loops & early stopping
│   │   ├── evaluation/       # Grad-CAM, artifact intervention, robustness
│   │   ├── calibration/      # Conformal prediction algorithms
│   │   ├── analysis/         # Background bias, dHash duplicate audits
│   │   └── scripts/          # Benchmark and paper execution pipelines
│   └── tests/                # Pytest test suite ensuring zero train/test leakage
```

---

## Scientific Citation & License

### Dataset Citation
If you utilize the DCID dataset, please cite the original authors:

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

### Licensing
- **DCID Dataset**: Licensed under [Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) — Strictly for academic and non-commercial research use.
- **LithoScan Codebase**: MIT License.
