# 04: LITERATURE GAP UPDATE & NOVELTY POSITIONING

**Audit Gate:** Source Provenance & Claim Correction  
**Date:** October 22, 2026  
**Auditor:** Skeptic Senior Research Scientist & Computer Vision Reviewer  
**Context:** Positioning LithoScan within the updated 2025–2026 scientific and technical landscape.

---

## 1. Landmark Literature Analysis

### A. The DCID Foundation Paper: Li et al. (2025)
- **Citation:** Jia-Yu Li, Ji-Zhou Tang, Xian-Zheng Zhao, et al. (2025). *A large-scale, high-quality dataset for lithology identification: Construction and applications.* **Petroleum Science**, 22(8), 3207–3228. DOI: [10.1016/j.petsci.2025.04.013](https://doi.org/10.1016/j.petsci.2025.04.013).
- **Core Findings:** Introduced the 98,000-image DCID benchmark (7 and 35 classes, clean and RWDA noise subsets). Benchmarked multiple vision architectures: ResNet, VGG, DenseNet, MobileNet, Vision Transformers (ViT), and MLP-Mixer. Observed near-perfect classification on clean test sets (>98–99%) and documented performance degradation under real-world defective data augmentation.
- **Impact on LithoScan Novelty:**  
  **Li et al. completely eliminates any claim that evaluating standard CNNs or Vision Transformers on DCID is novel.** A paper presenting a ResNet-50 vs. CNN-Transformer comparison on DCID is an incremental rerun of Li et al.'s experiments. To establish a legitimate contribution, LithoScan must critically examine what Li et al. took for granted: the validity of the random split, the presence of cross-split perceptual duplicates, and the lack of calibrated uncertainty estimation.

---

### B. The State-of-the-Art Review: Günther, Simán, et al. (2025)
- **Citation:** C. Günther, Filip Simán, et al. (2025). *Machine learning for drill core image analysis: A review.* **Ore Geology Reviews**, Vol. 187, Article 106974.
- **Core Findings:** This definitive survey structures computer vision on drill-core imagery around three core tasks: (1) lithology prediction, (2) geotechnical analysis (RQD, fractures, structural planes), and (3) mineralogical prediction. Crucially, the authors identify two unresolved industry-wide bottlenecks:
  1. **Domain Generalization Failure:** Models trained on single deposits or uniform imaging setups experience catastrophic drops when transferred across boreholes or facilities.
  2. **The Need for Context-Aware Human-in-the-Loop Triage:** Fully automated systems that output overconfident single labels fail operational safety standards. The review calls for ML systems that quantify predictive uncertainty and actively triage difficult, fractured, or ambiguous core intervals to human geologists.
- **Impact on LithoScan Novelty:**  
  **This review provides the exact theoretical motivation for LithoScan's selective prediction pivot.** Rather than trying to achieve 99.9% autonomous accuracy (which the literature recognizes as unrealistic in real mining operations), developing a mathematically calibrated triage framework (conformal prediction / risk-coverage thresholding) directly addresses the open challenge identified by Günther et al.

---

### C. The Open-Source Industry Benchmark: CoreLog Vision (September 2026)
- **Citation:** Felipe Santibáñez-Leal (September 2026). *CoreLog Vision: Browser-Native Drill-Core Logging and Lithology Classification Workbench.* Technical Report / Open-Source System Portfolio, Faena Mining Analytics (`fasl-work.com`).
- **Nature & Architecture:** A non-peer-reviewed interactive web workbench (MIT License). Features sliding-window patch extraction across continuous core-tray channels, CNN-based lithology prediction, depth-stitched strip logging, interactive confidence threshold sliders, and out-of-distribution (OOD) flagging.
- **Distinction from Peer-Reviewed Journal Research:**  
  | Feature | CoreLog Vision (Technical Report / Demo) | Required Journal Research Standard (LithoScan) |
  | :--- | :--- | :--- |
  | **Format** | Browser application & interactive demonstration | Rigorous peer-reviewed empirical methodology |
  | **Uncertainty Quantification** | Heuristic confidence slider based on raw softmax | Distribution-free split conformal prediction with guaranteed $1 - \alpha$ marginal coverage |
  | **Validation Rigor** | Visual demonstration on sample core trays | Formal statistical testing (bootstrap 95% CIs, McNemar's tests) |
  | **Data Leakage Treatment** | Standard sliding-window cropping without spatial leakage audit | Explicit pairwise dHash audit and spatial cross-validation critique |
- **Impact on LithoScan Novelty:**  
  CoreLog Vision proves that interactive, uncertainty-aware core logging is in high demand. However, because it relies on uncalibrated heuristic thresholds, **LithoScan must provide the missing mathematical rigor**: proving that raw softmax confidence is miscalibrated under acquisition noise, and demonstrating that conformal prediction sets provide true finite-sample error guarantees.

---

### D. Recent Literature on Spatial Validation Design in Geosciences
- **Key References:**
  - Pierre Ploton, Maxime Réjou-Méchain, et al. (2020). *Spatial validation reveals poor predictive performance of large-scale ecological mapping models.* **Nature Communications**, 11, 4580.
  - David R. Roberts, Volker Bahn, et al. (2017). *Cross-validation strategies for data with temporal, spatial, hierarchical or phylogenetic structure.* **Ecography**, 40(8), 913–929.
  - Recent subsurface ML literature on Leave-One-Well-Out (LOWO) and spatial block cross-validation in *Computers & Geosciences* and *Earth Science Informatics*.
- **Core Principles:** Spatial autocorrelation (Tobler's First Law) means samples close in space share physical characteristics. Random k-fold cross-validation allows models to memorize localized textures from the same borehole, severely underestimating true generalization error. Blocked or well-grouped splitting is mandatory for honest error estimation.
- **Impact on LithoScan Novelty:**  
  Arms LithoScan with a peer-reviewed methodology to critique DCID's random 80/20 split. Demonstrating that DCID's >99.7% benchmark is heavily influenced by spatial autocorrelation elevates the paper from a routine dataset evaluation to an impactful methodological critique.

---

## 2. Updated Novelty Matrix & Strategic Positioning

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           THE LITERATURE LANDSCAPE                          │
├────────────────────────────────┬────────────────────────────────────────────┤
│ What is ALREADY FULLY SOLVED:  │ • Patch-based CNN core classification      │
│ (Do NOT claim as novel)        │ • ResNet / ViT baseline training on DCID   │
│                                │ • Sliding-window depth-stitching UI demos  │
├────────────────────────────────┼────────────────────────────────────────────┤
│ What the LITERATURE IDENTIFIES │ • Fragility of >99% benchmarks in the field│
│ AS THE OPEN GAP:               │ • Spatial data leakage in core benchmarks  │
│ (LITHOSCAN'S EXACT TARGET)     │ • Mathematically guaranteed triage layers  │
│                                │   (Conformal Prediction / Risk-Coverage)   │
└────────────────────────────────┴────────────────────────────────────────────┘
```

### Strategic Paper Framing
By synthesizing these four literature pillars, LithoScan is positioned at the intersection of **methodological rigor and operational reality**:
> *"While Li et al. (2025) established DCID and CoreLog Vision (2026) demonstrated interactive core logging, Günther et al. (2025) and Ploton et al. (2020) emphasize that real-world deployment fails due to spatial leakage and lack of uncertainty bounds. LithoScan conducts the first forensic audit of the DCID benchmark, proves that clean accuracy is flattered by spatial redundancy, exposes an 11% collapse under real-world defects, and establishes a mathematically grounded conformal triage layer that guarantees operational precision."*
