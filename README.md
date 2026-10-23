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

## Dataset: DCID (Drill Core Image Dataset)

The **Drill Core Image Dataset (DCID)** is a large-scale, high-resolution geological benchmark constructed to accelerate automated borehole logging and subsurface lithology characterization. Released in *Petroleum Science* (2025), it contains **97,999** verified $512 \times 512$ RGB images captured under standardized optical imaging conditions.

### Official Publication & Dataset Access Links

* **Primary Research Article:**  
  *A large-scale, high-quality dataset for lithology identification: Construction and applications.*  
  **Petroleum Science**, 2025. ISSN: 1995-8226.  
  DOI: [10.1016/j.petsci.2025.04.013](https://doi.org/10.1016/j.petsci.2025.04.013)

* **Hugging Face Hub (Dataset Viewer & Download):**  
  [https://huggingface.co/datasets/168sir/drill-core-image-dataset](https://huggingface.co/datasets/168sir/drill-core-image-dataset)  
  *Direct Download Archive (3.82 GB):* [DCID.zip (3,819,926,174 bytes)](https://huggingface.co/datasets/168sir/drill-core-image-dataset/resolve/main/DCID.zip)

* **Original Authors GitHub Repository:**  
  [https://github.com/JiayuLi1120/drill-core-image-dataset](https://github.com/JiayuLi1120/drill-core-image-dataset)

---

### How to Acquire the Dataset (3 Options)

#### Option 1: Automated 1-Command Python Downloader (Recommended)
From the root of this repository, run:
```bash
python download_dataset.py
```
This script streams `DCID.zip` directly from Hugging Face with a live progress bar, checks data integrity, and extracts it directly into `DCID/DCID/`.

#### Option 2: Manual Download & Extraction
1. Download `DCID.zip` (3.82 GB) from the direct link above or via `curl`:
   ```bash
   curl -L -O https://huggingface.co/datasets/168sir/drill-core-image-dataset/resolve/main/DCID.zip
   ```
2. Extract the archive into the repository root so the layout matches `DCID/DCID/`:
   ```bash
   unzip DCID.zip -d DCID
   ```

#### Option 3: External Dataset Directory via Environment Variable
If you already have DCID stored on another drive (e.g., an SSD or server partition), you do not need to duplicate files. Set `DCID_ROOT`:
```bash
# Windows PowerShell
$env:DCID_ROOT = "D:\Datasets\DCID\DCID"

# Windows CMD
set DCID_ROOT=D:\Datasets\DCID\DCID

# Linux / macOS
export DCID_ROOT=/data/DCID/DCID
```
All training and evaluation scripts automatically detect and prioritize `DCID_ROOT`.

---

### The Four DCID Sub-Cohorts

DCID provides four parallel directories separating coarse vs. fine-grained lithologies and clean vs. defective core states:

| Directory Name | Target Classes | Train Split | Test Split | Total Images | Format & Resolution | Characteristics |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`DCID-512-7`** | 7 coarse | 28,000 | 7,000 | **35,000** | RGB JPEG (512×512) | Standard clean core sections (4,000 train / 1,000 test per class) |
| **`DCID-512-35`** | 35 fine | 27,999 | 7,000 | **34,999** | RGB JPEG (512×512) | Fine-grained petrographic classes (~800 train / 200 test per class) |
| **`noise-512-7`** | 7 coarse | 11,200 | 2,800 | **14,000** | RGB JPEG (512×512) | Defective core images (cracks, mud smears, scale bars, core box borders) |
| **`noise-512-35`** | 35 fine | 11,200 | 2,800 | **14,000** | RGB JPEG (512×512) | Defective variants across all 35 fine-grained classes (320 train / 80 test) |
| **Total Benchmark** | — | **78,399** | **19,600** | **97,999** | — | **3.56 GB uncompressed (3.82 GB zip)** |

---

### Complete Class Taxonomy & Distribution

#### 1. DCID-7: Coarse Lithology Classification (7 Classes)

Balanced distribution: exactly **4,000 train** and **1,000 test** images per class.

| Class ID | Lithology Name | Rock Group | Train | Test | Total | Visual Characteristics |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **0** | `1.Red sandstone` | Sedimentary (Siliciclastic) | 4,000 | 1,000 | 5,000 | Ferruginous red/orange, medium-to-fine grain quartz texture |
| **1** | `2.Light sandstone` | Sedimentary (Siliciclastic) | 4,000 | 1,000 | 5,000 | Buff/cream to pale gray, quartzose-feldspathic framework |
| **2** | `3.Gray siltstone` | Sedimentary (Fine-grained) | 4,000 | 1,000 | 5,000 | Dark gray/charcoal, microcrystalline, laminated silty fabric |
| **3** | `4.Mudstone` | Sedimentary (Argillaceous) | 4,000 | 1,000 | 5,000 | Homogeneous dark argillaceous matrix, non-fissile |
| **4** | `5.Granite` | Igneous (Plutonic Felsic) | 4,000 | 1,000 | 5,000 | Phaneritic felsic intergrowth: quartz, K-feldspar, biotite |
| **5** | `6.Basalt` | Igneous (Volcanic Mafic) | 4,000 | 1,000 | 5,000 | Aphanitic melanocratic dark gray/black mafic matrix |
| **6** | `7.Marble` | Metamorphic (Non-foliated) | 4,000 | 1,000 | 5,000 | Recrystallized calcite/dolomite, saccharoidal granoblastic |

---

#### 2. DCID-35: Fine-Grained Lithology Classification (35 Classes)

Uniform distribution: **800 train** and **200 test** images per class (Total: 1,000 images/class; Class 28 has 799 train + 200 test = 999).

| ID | Class Directory Name | Rock Supergroup | Primary Mineralogy & Structural Description |
| :---: | :--- | :--- | :--- |
| **1** | `1.Red sandstone` | Sedimentary | Hematite-cemented quartz grains, fine-to-medium sand |
| **2** | `2.Light sandstone` | Sedimentary | Pale feldspathic/quartzose sandstone, low clay content |
| **3** | `3.Gray-green sandstone` | Sedimentary | Chloritic/glauconitic greenish sandstone |
| **4** | `4.Gray siltstone` | Sedimentary | Intermediate grain-size between sandstone and shale |
| **5** | `5.Brown siltstone` | Sedimentary | Limonitic/ferruginous oxidized siltstone |
| **6** | `6.Carbonaceous siltstone`| Sedimentary | Organic-rich dark silty laminae |
| **7** | `7.Mudstone` | Sedimentary | Massive micro-grained argillaceous sediment |
| **8** | `8.Granite` | Igneous (Plutonic) | Equigranular quartz + alkali feldspar + plagioclase |
| **9** | `9.Megacrystic granite` | Igneous (Plutonic) | Porphyritic with large K-feldspar phenocrysts |
| **10** | `10.Monzogranite` | Igneous (Plutonic) | Equal proportions of alkali feldspar and plagioclase |
| **11** | `11.Fine-grained granite` | Igneous (Plutonic) | Microcrystalline felsic intrusive rock |
| **12** | `12.Pegmatite` | Igneous (Intrusive) | Giant crystalline texture (>1 cm crystals) |
| **13** | `13.Basalt` | Igneous (Extrusive) | Mafic volcanic rock with plagioclase and pyroxene |
| **14** | `14.Fine-grained basalt` | Igneous (Extrusive) | Rapidly chilled dense aphanitic basalt |
| **15** | `15.Gabbro` | Igneous (Plutonic) | Coarse-grained mafic intrusive (plagioclase + pyroxene) |
| **16** | `16.Dolerite` | Igneous (Subvolcanic) | Medium-grained diabase/dolerite dike rock |
| **17** | `17.White marble` | Metamorphic | Pure recrystallized calcite limestone |
| **18** | `18.Blue marble` | Metamorphic | Calcite marble with trace graphite/serpentine tinting |
| **19** | `19.Dolomite` | Sedimentary/Meta | Magnesium-rich carbonate $(CaMg(CO_3)_2)$ |
| **20** | `20.Black Shale` | Sedimentary | Highly fissile, sapropelic, high-TOC source rock |
| **21** | `21.Red shale` | Sedimentary | Oxidized fissile argillaceous shale |
| **22** | `22.Dark green calc-silicate` | Metamorphic | Contact metamorphic skarn/calc-silicate (epidote/diopside) |
| **23** | `23.Brown calc-silicate` | Metamorphic | Grossular garnet-vesuvianite contact rock |
| **24** | `24.Skarn` | Metamorphic | Hydrothermal calc-silicate replacement ore host |
| **25** | `25.Red-brown dacite` | Igneous (Extrusive) | Intermediate volcanic with plagioclase phenocrysts |
| **26** | `26.Black dacite` | Igneous (Extrusive) | Melanocratic glassy dacitic volcanic rock |
| **27** | `27.Rhyolite` | Igneous (Extrusive) | Felsic extrusive volcanic with flow banding |
| **28** | `28.Banded iron formation`| Sedimentary/Meta | Alternating silica (chert) and iron oxide (hematite/magnetite) bands |
| **29** | `29.Hematite breccia` | Tectonic/Hydrothermal | Angular rock fragments cemented by hematite |
| **30** | `30.Altered breccia` | Hydrothermal | Hydrothermally sericitized/argillized cataclastic breccia |
| **31** | `31.Schist` | Metamorphic (Foliated) | Strong planar foliation with visible mica flakes |
| **32** | `32.Gneiss` | Metamorphic (Banded) | Alternating felsic and mafic compositional banding |
| **33** | `33.Porphyry` | Igneous (Subvolcanic) | Distinct phenocrysts embedded in fine groundmass |
| **34** | `34.Brecciated mudstone` | Sedimentary/Tectonic | Mechanically fractured mudstone clasts |
| **35** | `35.Shale fragments` | Sedimentary/Drilling | Drill-damaged or mechanically disintegrated shale |

---

### Real-World Defect Cohorts (RWDA: `noise-512-7` & `noise-512-35`)

Real drill cores photographed at drilling rigs and storage core sheds contain severe environmental and mechanical interference:
1. **Physical Fracturing & Disking:** Transverse drilling cracks, mechanical core disking, and fragmentation.
2. **Drilling Mud Smear:** Bentonite and chemical drilling mud cakes coating the core surfaces.
3. **Optical Artifacts:** Physical plastic/wooden millimeter and centimeter scale-bar rulers placed next to the core.
4. **Core Box Framing:** Wooden core tray slats, foam spacers, and dark shadow gaps between core intervals.

Our research framework includes automated artifact interventions:
- **Border Masking (`mask_image_border`):** Crops the outer 8%–12% margin of images to strip away scale-bar fonts and tray edges, forcing deep networks to learn mineral crystallization rather than tray wood.
- **dHash Near-Duplicate Audit:** Discovered **700,216** near-duplicate image pairs between train and test sets caused by consecutive camera shots along unbroken core intervals. Our evaluation suite benchmarks models under systematic neighbor exclusion ($d\text{Hash} \le 2, 5, 10$).

---

### Strict Split Safety (`SplitViolationError`)

> ⚠️ **METHODOLOGICAL WARNING:**  
> Standard random shuffling (e.g., `train_test_split`) on drill core images causes severe spatial autocorrelation leakage: adjacent 512×512 patches cut from the same 1-meter drill-core run share identical mineral textures and lighting.  
> 
> In LithoScan, all data loaders strictly enforce the official `train/` and `test/` splits. Any attempt to re-shuffle or randomly re-split raises `SplitViolationError`, guaranteeing that benchmark results reflect genuine out-of-distribution evaluation.

---

### Directory Layout After Extraction

```
software_litho/
├── DCID/
│   └── DCID/                          <-- DCID_ROOT
│       ├── DCID-512-7/
│       │   ├── train/                 (28,000 images: 4,000 / class)
│       │   └── test/                  (7,000 images: 1,000 / class)
│       ├── DCID-512-35/
│       │   ├── train/                 (27,999 images: ~800 / class)
│       │   └── test/                  (7,000 images: 200 / class)
│       ├── noise-512-7/
│       │   ├── train/                 (11,200 images: 1,600 / class)
│       │   └── test/                  (2,800 images: 400 / class)
│       ├── noise-512-35/
│       │   ├── train/                 (11,200 images: 320 / class)
│       │   └── test/                  (2,800 images: 80 / class)
│       └── build_dcid_dataset.py      (Authors' dataset utility)
├── LithoScan/
│   ├── data/
│   │   └── manifests/
│   │       └── dcid_manifest.csv      (Full 98k image index with MD5 hashes)
│   ├── reports/
│   │   └── phase3/
│   │       ├── near_duplicate_pairs.csv.gz  (700k dHash pairs audit)
│   │       └── class_visual_statistics.csv  (Color, contrast, entropy metrics)
...
```

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
