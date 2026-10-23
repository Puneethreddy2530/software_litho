# 06: PAPER DIRECTION RANKING & SCIENTIFIC DEFENSE — LITHOSCAN

**Audit Date:** October 22, 2026  
**Auditor:** Skeptic Senior Research Scientist & Computer Vision Reviewer  

---

## 1. Candidate Paper Directions Overview

Three candidate publication trajectories have been formulated, scrutinized, and ranked based on their **scientific validity**, **novelty potential**, **reviewer defensibility**, and **alignment with existing LithoScan assets**.

```
Rank 1 (RECOMMENDED) ──► Direction A: Benchmark Fragility, Leakage & Selective Prediction
Rank 2 (VIABLE)      ──► Direction B: Uncertainty-Aware Selective Lithology Logging (Conformal Triage)
Rank 3 (HIGH RISK)   ──► Direction C: CNN-Transformer Architecture vs. Baselines (Ablation Paper)
```

---

## 2. Detailed Evaluation of Candidate Directions

### 🥇 Direction A (RECOMMENDED): The Illusion of 99% Accuracy: Leakage, Corruption Robustness, and Selective Prediction in Drill-Core Logging

- **Proposed Title:**  
  *The Illusion of 99% Accuracy: Dissecting Spatial Leakage, Acquisition Corruption, and Selective Prediction in Automated Drill-Core Lithology Benchmarks*
- **Core Research Question:**  
  *Why do standard deep vision models easily achieve >99.7% test accuracy on public drill-core benchmarks while dropping precipitously under real-world core defects, and can distribution-free selective prediction restore trustworthy operational reliability?*
- **Testable Hypothesis:**  
  Near-perfect classification accuracy on the DCID drill-core benchmark is an artifact of cross-split perceptual similarity and spatial autocorrelation; under real-world acquisition corruptions (RWDA defects), accuracy degrades by over 11%, but a calibrated selective classification mechanism (conformal prediction / risk-coverage thresholding) can guarantee human-level error bounds ($\le 1\%$) by selectively abstaining and triaging defective cores to expert geologists.
- **Intended Contribution:**  
  1. A rigorous forensic audit of the newly released DCID benchmark, exposing that linear probing on frozen ImageNet features already reaches 99.28% on 35 classes due to class separability and cross-split perceptual duplicates (dHash $\le 2$).
  2. A systematic robustness evaluation of standard CNNs and hybrid architectures against real-world physical core defects (RWDA noise: cracks, mud, scale bars) and occlusion interventions.
  3. The introduction of an uncertainty-aware selective classification framework (risk-coverage curves and split conformal prediction) specifically tailored for automated borehole logging.
- **Relevant Related Work:**  
  - Li et al. (2025) *Petroleum Science* (DCID benchmark)
  - Ploton et al. (2020) *Nature Communications* (Spatial validation and inflated ML metrics)
  - Geirhos et al. (2020) *Nature Machine Intelligence* (Shortcut learning)
  - Hendrycks & Dietterich (2019) *ICLR* (Image corruption benchmarking)
  - Geifman & El-Yaniv (2017) *NeurIPS* (Selective classification / risk-coverage)
  - Angelopoulos & Bates (2021) (Conformal prediction)
- **Available Preliminary Evidence:**  
  - Clean test accuracy, RWDA noise metrics, and 40px border occlusion metrics across ResNet-50, CNN-only, and CNN-Transformer on both DCID-7 and DCID-35.
  - Linear probe results proving 99.28% separability on frozen features.
  - Phase 3 dHash matrix proving hundreds of cross-split near-duplicate pairs.
  - Phase 4D stress-test scripts.
- **Missing Evidence:**  
  - Implementation of post-hoc calibration (temperature scaling, conformal prediction, risk-coverage curves).
  - Validation of selective coverage on DCID-35 under RWDA noise.
  - Formal statistical confidence intervals (bootstrap 95% CIs) on all metrics.
- **Minimum Required Experiments:**  
  1. Extract predicted softmax probabilities across clean and noise test sets for existing checkpoints.
  2. Implement split conformal prediction and risk-coverage thresholding.
  3. Compute error vs. coverage trade-offs: measure what percentage of defective core samples must be referred to a human geologist to maintain 99.5% operational precision.
  4. Perform Phase 4D similarity-filtering stress test on DCID-35 (remedying the DCID-7-only limitation).
- **Compute and Hardware Requirements:**  
  **Near zero GPU training required.** All required checkpoints are already trained and saved. Experiments require only forward-pass probability extraction and CPU-level calibration analysis (~2–4 hours total execution time).
- **Main Limitations:**  
  DCID lacks explicit borehole and depth identifiers, precluding a true Leave-One-Well-Out split. Must be transparently framed as a study on public benchmark dynamics.
- **Likely Rejection Reasons & Rebuttal:**  
  - *Rejection Objection:* "DCID was already benchmarked by Li et al. (2025)."  
    *Rebuttal:* Li et al. merely published the dataset and reported high numbers without questioning split leakage, perceptual duplication, or providing uncertainty bounds. This paper is the critical audit that provides the necessary selective prediction solution for practical adoption.
- **Target Venues:**  
  *Computers & Geosciences*, *Geoenergy Science and Engineering*, or *IEEE Transactions on Geoscience and Remote Sensing (TGRS)*.

---

### 🥈 Direction B (VIABLE): Uncertainty-Aware Selective Lithology Prediction via Conformal Triage

- **Proposed Title:**  
  *Distribution-Free Conformal Prediction and Risk-Coverage Calibration for Robust Autonomous Drill-Core Logging*
- **Core Research Question:**  
  *Can distribution-free conformal prediction provide rigorous, user-specified error guarantees for fine-grained lithology classification in the presence of surface defects and operational noise?*
- **Testable Hypothesis:**  
  By calibrating prediction sets using split conformal prediction and adaptive conformal inference on drill-core imagery, an automated logging system can bound misclassification risk to a user-defined rate $\alpha \in \{0.01, 0.05\}$, producing multi-class candidate sets only when visual ambiguity arises (e.g., distinguishing interbedded siltstone and mudstone).
- **Intended Contribution:**  
  1. Formulation of drill-core lithology logging as a selective classification / set-prediction problem.
  2. Demonstration that standard softmax confidence is miscalibrated under acquisition noise (cracks/mud).
  3. Operational triage protocol: clean cores are logged autonomously; ambiguous/defective cores output calibrated prediction sets for human geologist review.
- **Why Ranked #2:**  
  Highly actionable and computationally inexpensive, but narrower in scope than Direction A. Works best as the central methodology within Direction A.

---

### 🥉 Direction C (HIGH RISK / NOT RECOMMENDED): CNN-Transformer Fusion for Fine-Grained Drill-Core Classification

- **Proposed Title:**  
  *LithoScan: A Texture-Aware Hybrid CNN-Transformer Architecture for Fine-Grained Drill-Core Lithology Classification*
- **Core Research Question:**  
  *Does combining a CNN spatial backbone with a self-attention Transformer encoder improve fine-grained lithology classification over pure convolutional baselines?*
- **Why This Direction Must Be Rejected:**  
  1. **Empirical Failure:** The experimental results gathered in Phase 4 and Phase 5 **falsify** the core hypothesis. On DCID-7, ResNet-50 outperforms the CNN-Transformer on clean data, noisy data, and border-masked data. On DCID-35, the CNN-Transformer is 0.03% *worse* than CNN-only on clean data and edges out ResNet-50 on noise by a trivial 0.29%, while doubling training time and adding 6.4M parameters.
  2. **Zero Novelty:** The architecture is an off-the-shelf ResNet-50 connected to a standard 2-layer `nn.TransformerEncoderLayer`. Reviewers will immediately note that dozens of papers (ViT, BoTNet, TransUNet, Li et al.) have already evaluated this exact hybrid paradigm.
  3. **Guaranteed Rejection:** Reviewers at any reputable journal (*IEEE TGRS*, *Computers & Geosciences*, *Pattern Recognition*) will reject a paper claiming architectural superiority when the experimental table shows microscopic differences (<0.3%) and a standard baseline performs just as well with far fewer parameters.

---

## 3. Final Recommendation and Selection Justification

### **SELECTED DIRECTION: DIRECTION A**

**Justification:**  
1. **Academic Honesty:** It embraces the reality of the experimental data rather than trying to hide the fact that the CNN-Transformer is not superior. Negative and critical findings regarding benchmark saturation and leakage are highly cited and respected in modern machine learning literature.
2. **Maximum Asset Reusability:** All heavy GPU training is already complete. The checkpoints (`best_model.pt` for ResNet-50, CNN-only, and CNN-Transformer on both DCID-7 and DCID-35) can be used directly without running expensive retraining.
3. **High Reviewer Appeal:** Reviewers in earth science and computer vision are increasingly skeptical of papers reporting "99.9% accuracy" on rock images. A paper that critically investigates *why* accuracy is so high, shows where it breaks down (11% drop under RWDA noise), and provides a mathematically grounded selective prediction solution (conformal prediction) directly answers real-world geological concerns.
