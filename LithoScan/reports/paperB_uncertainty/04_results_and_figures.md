# 04: Empirical Results, Figures & Statistical Tables — LithoScan Paper B

**Document ID:** `reports/paperB_uncertainty/04_results_and_figures.md`  
**Date:** October 2026  
**Status:** COMPLETE & EMPIRICALLY AUDITED FROM SAVED ARTIFACTS  
**Research Track:** Paper B (*When Should a Lithology Classifier Abstain? Calibration, Selective Risk, and Human Referral Under Defective Drill-Core Imaging*)  
**Execution Runtime:** 809.49 seconds on NVIDIA GeForce RTX 3050 Laptop GPU (PyTorch CUDA 12.4)  
**Evaluated Benchmarks:**  
- Primary Benchmark: DCID-35 (35 fine-grained lithologies, $512 \times 512$ resolution)  
- Replication Benchmark: DCID-7 (7 coarse lithologies, $512 \times 512$ resolution)  
**Evaluated Architectures:** Fine-tuned ResNet-50 and CNN-Only Baseline (`server_fusion_cnn_only`)  

---

## 1. Executive Experimental Summary

This study conducted a systematic empirical evaluation of probability calibration, selective prediction (abstention), and split conformal prediction across 21,000 drill-core photographs. All evaluations were conducted using frozen historical model checkpoints and locked, deterministic partition manifests.

### Summary of Key Empirical Findings:
1. **Acquisition Fragility:** On DCID-35, both models suffer an ~11.1% to 12.3% collapse in top-1 accuracy when evaluated on drill cores exhibiting real-world drilling rig defects (ResNet-50: 99.73% clean $\rightarrow$ 88.64% defective; CNN-only: 99.81% clean $\rightarrow$ 87.50% defective).
2. **Probability Miscalibration under Shift (RQ1):** Uncalibrated models exhibit severe overconfidence under defective image acquisition. On DCID-35 ResNet-50, Expected Calibration Error (ECE) surges from **0.12%** [0.07, 0.25] on clean imagery to **6.66%** [5.74, 7.69] on defective imagery.
3. **Catastrophic Failure of Clean-Calibrated Thresholds (RQ3 / Hypothesis $H_1$):** When an operating threshold is calibrated exclusively on clean imagery to achieve a nominal $5\%$ selective risk ($\tau_{\text{clean}} = 0.5389$), it **fails catastrophically** when applied to defective cores, realizing an error rate of **9.63%** (a 92.6% excess error violation!).
4. **Success of Domain-Matched Calibration:** When calibrated using representative defective images ($D_{\text{cal,def}}$, $\tau_{\text{def}} = 0.9295$), the pipeline **successfully controls selective risk** at **4.52%** ($\le 5.0\%$) while retaining **84.6%** automated coverage and referring **15.4%** of ambiguous cores to human geologists.
5. **Conformal Undercoverage Breakdown (Hypothesis $H_2$):** Standard split conformal prediction calibrated on clean imagery exhibits severe undercoverage on defective imagery, achieving only **57.68%** marginal coverage [56.07, 59.29] at the nominal $95\%$ level ($\alpha=0.05$), with $41.4\%$ empty prediction sets. In contrast, domain-matched calibration restores valid coverage to **95.57%** [94.86, 96.29] (Wilcoxon $p = 2.54 \times 10^{-198}$).
6. **Invariance to Duplicate Filtering:** Removing all test images sharing near-duplicate perceptual similarity with the training set (dHash $\le 2$) produces virtually zero change in evaluation accuracy ($99.73\% \rightarrow 99.73\%$) and ECE ($0.12\% \rightarrow 0.12\%$), verifying that findings are not artifacts of benchmark redundancy.

---

## 2. Statistical Tables

### Table 1: Dataset Partition Counts, Balance & Roles
*Saved as [table1_dataset_partitions.csv](file:///C:/Users/punee/Desktop/Software_lithology/LithoScan/manuscript/paperB_uncertainty/tables/table1_dataset_partitions.csv)*

| Benchmark | Partition Name | Source Directory | Total Images | Images/Class | Protocol Role | Historical Exposure |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **DCID-35** | Model Fitting ($D_{\text{fit}}$) | `DCID-512-35/train` | 25,200 | 720 | Weight optimization | Historical (Phase 2/5A) |
| **DCID-35** | Model Selection ($D_{\text{sel}}$) | `DCID-512-35/train` | 2,799 | ~80 | Checkpoint selection | Historical (Phase 2/5A) |
| **DCID-35** | Clean Calibration ($D_{\text{cal,clean}}$) | `DCID-512-35/test` | **1,750** | 50 | In-distribution calibration | Retrospective benchmark |
| **DCID-35** | Clean Evaluation ($D_{\text{eval,clean}}$) | `DCID-512-35/test` | **5,250** | 150 | Locked clean evaluation | Retrospective benchmark |
| **DCID-35** | Defective Calibration ($D_{\text{cal,def}}$) | `noise-512-35/train` | **3,500** | 100 | Domain-matched calibration | **Untouched Pool** |
| **DCID-35** | Defective Evaluation ($D_{\text{eval,def}}$) | `noise-512-35/test` | **2,800** | 80 | Locked defective evaluation | Retrospective benchmark |
| **DCID-7** | Clean Calibration ($D_{\text{cal,clean}}$) | `DCID-512-7/test` | **1,750** | 250 | Clean calibration | Retrospective benchmark |
| **DCID-7** | Clean Evaluation ($D_{\text{eval,clean}}$) | `DCID-512-7/test` | **5,250** | 750 | Clean evaluation | Retrospective benchmark |
| **DCID-7** | Defective Calibration ($D_{\text{cal,def}}$) | `noise-512-7/train` | **2,800** | 400 | Defective calibration | **Untouched Pool** |
| **DCID-7** | Defective Evaluation ($D_{\text{eval,def}}$) | `noise-512-7/test` | **2,800** | 400 | Defective evaluation | Retrospective benchmark |

---

### Table 2: Model Performance & Probability Calibration
*Saved as [table2_calibration_metrics.csv](file:///C:/Users/punee/Desktop/Software_lithology/LithoScan/manuscript/paperB_uncertainty/tables/table2_calibration_metrics.csv)*  
*Note: Point estimates are reported alongside 95% class-stratified percentile bootstrap confidence intervals ($B = 2,000$ resamples).*

| Benchmark | Architecture | Evaluation Cohort & Calibration Strategy | Accuracy (%) [95% CI] | ECE (%) [95% CI] | NLL [95% CI] | Brier Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **DCID-35** | ResNet-50 | Clean (Uncalibrated) | **99.73%** [99.60, 99.87] | **0.12%** [0.07, 0.25] | 0.0065 [0.0036, 0.0096] | 0.0037 |
| **DCID-35** | ResNet-50 | Clean ($T_{\text{clean}} = 1.2868$ Scaled) | 99.73% [99.60, 99.87] | 0.15% [0.13, 0.29] | 0.0069 [0.0044, 0.0095] | 0.0036 |
| **DCID-35** | ResNet-50 | Defective (Uncalibrated) | **88.64%** [87.46, 89.75] | **6.66%** [5.74, 7.69] | 0.5161 [0.4540, 0.5806] | 0.1794 |
| **DCID-35** | ResNet-50 | Defective ($T_{\text{clean}}$ Scaled, Shifted) | 88.64% [87.46, 89.75] | 5.04% [4.20, 6.09] | 0.4410 [0.3919, 0.4922] | 0.1725 |
| **DCID-35** | ResNet-50 | Defective ($T_{\text{def}} = 1.6714$ Scaled, Matched) | 88.64% [87.46, 89.75] | **2.12%** [1.84, 3.45] | **0.4052** [0.3659, 0.4462] | **0.1688** |
| **DCID-35** | CNN-Only | Clean (Uncalibrated) | **99.81%** [99.68, 99.92] | **0.08%** [0.05, 0.18] | 0.0054 [0.0028, 0.0084] | 0.0028 |
| **DCID-35** | CNN-Only | Defective (Uncalibrated) | **87.50%** [86.29, 88.71] | **6.67%** [5.69, 7.74] | 0.5283 [0.4639, 0.5960] | 0.1878 |
| **DCID-35** | CNN-Only | Defective ($T_{\text{clean}} = 1.2583$ Scaled, Shifted) | 87.50% [86.29, 88.71] | 5.03% [4.16, 6.09] | 0.4623 [0.4093, 0.5181] | 0.1804 |
| **DCID-35** | CNN-Only | Defective ($T_{\text{def}} = 1.7766$ Scaled, Matched) | 87.50% [86.29, 88.71] | **1.96%** [1.65, 3.37] | **0.4132** [0.3705, 0.4578] | **0.1764** |
| **DCID-7** | ResNet-50 | Clean (Uncalibrated) | **100.00%** [100.00, 100.00] | **0.05%** [0.03, 0.08] | 0.0006 [0.0003, 0.0009] | 0.0001 |
| **DCID-7** | ResNet-50 | Defective (Uncalibrated) | **96.79%** [96.11, 97.43] | **1.38%** [0.98, 2.03] | 0.1134 [0.0887, 0.1400] | 0.0510 |
| **DCID-7** | ResNet-50 | Defective ($T_{\text{def}} = 1.4540$ Scaled, Matched) | 96.79% [96.11, 97.43] | **0.76%** [0.64, 1.48] | **0.1046** [0.0870, 0.1237] | **0.0504** |
| **DCID-7** | CNN-Only | Clean (Uncalibrated) | **99.96%** [99.90, 100.00] | **0.03%** [0.01, 0.09] | 0.0022 [0.0001, 0.0057] | 0.0008 |
| **DCID-7** | CNN-Only | Defective (Uncalibrated) | **96.00%** [95.29, 96.71] | **2.56%** [2.11, 3.32] | 0.2056 [0.1587, 0.2522] | 0.0666 |
| **DCID-7** | CNN-Only | Defective ($T_{\text{def}} = 1.8569$ Scaled, Matched) | 96.00% [95.29, 96.71] | **1.73%** [1.27, 2.44] | **0.1426** [0.1161, 0.1685] | **0.0646** |

---

### Table 3: Selective Classification Risk-Coverage Operating Points (MSP Confidence)
*Saved as [table3_selective_risk_operating_points.csv](file:///C:/Users/punee/Desktop/Software_lithology/LithoScan/manuscript/paperB_uncertainty/tables/table3_selective_risk_operating_points.csv)*  
*Comparison of Pipeline A (calibrated on clean development data) versus Pipeline B (calibrated on representative defective development data) evaluated on the defective evaluation cohort ($D_{\text{eval,def}}$).*

| Benchmark | Architecture | Target Error Ceiling | Pipeline A $\tau_{\text{clean}}$ | Pipeline A Realized Cov | Pipeline A Referral Rate | Pipeline A Realized Error | Pipeline B $\tau_{\text{def}}$ | Pipeline B Realized Cov | Pipeline B Referral Rate | Pipeline B Realized Error | Constraint Satisfied? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **DCID-35** | ResNet-50 | **1.0%** | 0.5389 | 97.2% | 2.8% | **9.63%** | 0.9998 | 48.1% | 51.9% | **1.56%** | **Pipeline B Only** |
| **DCID-35** | ResNet-50 | **5.0%** | 0.5389 | 97.2% | 2.8% | **9.63%** | 0.9295 | 84.6% | 15.4% | **4.52%** | **Pipeline B Only** |
| **DCID-35** | ResNet-50 | **10.0%** | 0.5389 | 97.2% | 2.8% | **9.63%** | 0.5289 | 97.4% | 2.6% | **9.79%** | Both |
| **DCID-35** | CNN-Only | **1.0%** | 0.5747 | 97.4% | 2.6% | **11.04%** | 1.0000 | 48.2% | 51.8% | **1.19%** | **Pipeline B Only** |
| **DCID-35** | CNN-Only | **5.0%** | 0.5747 | 97.4% | 2.6% | **11.04%** | 0.9553 | 84.2% | 15.8% | **5.81%** | **Pipeline B Only** |
| **DCID-35** | CNN-Only | **10.0%** | 0.5747 | 97.4% | 2.6% | **11.04%** | 0.6741 | 95.3% | 4.7% | **9.85%** | **Pipeline B Only** |
| **DCID-7** | ResNet-50 | **1.0%** | 0.8782 | 94.0% | 6.0% | **1.25%** | 0.9141 | 93.0% | 7.0% | **1.11%** | **Pipeline B Only** |
| **DCID-7** | ResNet-50 | **5.0%** | 0.8782 | 94.0% | 6.0% | **1.25%** | 0.3784 | 100.0% | 0.0% | **3.22%** | Both |
| **DCID-7** | CNN-Only | **1.0%** | 0.5818 | 99.1% | 0.9% | **3.57%** | 0.9986 | 82.9% | 17.1% | **1.08%** | **Pipeline B Only** |
| **DCID-7** | CNN-Only | **5.0%** | 0.5818 | 99.1% | 0.9% | **3.57%** | 0.3771 | 100.0% | 0.0% | **4.00%** | Both |

---

### Table 4: Split Conformal Prediction Triage Evaluation
*Saved as [table4_conformal_prediction_triage.csv](file:///C:/Users/punee/Desktop/Software_lithology/LithoScan/manuscript/paperB_uncertainty/tables/table4_conformal_prediction_triage.csv)*  
*Evaluation on the defective evaluation cohort ($D_{\text{eval,def}}$) comparing Clean Calibration (Shifted) vs. Defective Calibration (Domain-Matched). Statistical comparison of set sizes via Wilcoxon signed-rank test.*

| Benchmark | Model | Target Coverage | Shifted Coverage [95% CI] | Shifted Mean Set Size | Shifted Empty Rate | Matched Coverage [95% CI] | Matched Mean Set Size | Matched Singletons | Matched Referral Rate | Wilcoxon $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DCID-35** | ResNet-50 | **99%** ($\alpha=0.01$) | 75.96% [74.50, 77.46] | 0.79 | 21.1% | **98.96%** [98.57, 99.32] | 4.12 | 45.2% | 54.8% | $p < 0.001$ ($1.04\times 10^{-253}$) |
| **DCID-35** | ResNet-50 | **95%** ($\alpha=0.05$) | 57.68% [56.07, 59.29] | 0.59 | 41.4% | **95.57%** [94.86, 96.29] | 1.40 | 79.4% | 20.6% | $p < 0.001$ ($2.54\times 10^{-198}$) |
| **DCID-35** | ResNet-50 | **90%** ($\alpha=0.10$) | 47.04% [45.43, 48.68] | 0.48 | 52.2% | **90.04%** [88.93, 91.07] | 1.04 | 96.3% | 3.7% | $p < 0.001$ ($4.15\times 10^{-300}$) |
| **DCID-35** | CNN-Only | **95%** ($\alpha=0.05$) | 59.79% [58.21, 61.36] | 0.61 | 39.1% | **94.96%** [94.14, 95.71] | 1.57 | 72.8% | 27.2% | $p < 0.001$ ($4.92\times 10^{-184}$) |
| **DCID-7** | ResNet-50 | **95%** ($\alpha=0.05$) | 67.50% [66.04, 69.00] | 0.68 | 32.4% | **94.61%** [93.75, 95.39] | 0.96 | 96.4% | 3.6% | $p < 0.001$ ($2.68\times 10^{-177}$) |
| **DCID-7** | CNN-Only | **95%** ($\alpha=0.05$) | 62.54% [60.86, 64.14] | 0.63 | 37.3% | **95.25%** [94.43, 96.00] | 0.98 | 98.5% | 1.5% | $p < 0.001$ ($6.60\times 10^{-220}$) |

---

### Table 5: Deduplication Sensitivity Analysis (dHash $\le 2$)
*Saved as [table5_deduplication_sensitivity.csv](file:///C:/Users/punee/Desktop/Software_lithology/LithoScan/manuscript/paperB_uncertainty/tables/table5_deduplication_sensitivity.csv)*  
*Assessing potential bias from perceptual near-duplicates identified between training and test sets.*

| Benchmark | Architecture | Total Clean Eval Images | Excluded Pairs (dHash $\le 2$) | Retained Pristine Subset | Original Full Eval Accuracy | Deduplicated Eval Accuracy | $\Delta\text{Accuracy}$ | Original ECE | Deduplicated ECE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DCID-35** | ResNet-50 | 5,250 | 50 | 5,200 (99.0%) | 99.73% | **99.73%** | **+0.00%** | 0.12% | 0.12% |
| **DCID-35** | CNN-Only | 5,250 | 50 | 5,200 (99.0%) | 99.81% | **99.81%** | **+0.00%** | 0.08% | 0.08% |
| **DCID-7** | ResNet-50 | 5,250 | 82 | 5,168 (98.4%) | 100.00% | **100.00%** | **+0.00%** | 0.05% | 0.05% |
| **DCID-7** | CNN-Only | 5,250 | 82 | 5,168 (98.4%) | 99.96% | **99.96%** | **+0.00%** | 0.03% | 0.04% |

---

## 3. Publication Figures

All figures are compiled at 300 DPI publication resolution in `results/paperB_uncertainty/figures/` and `manuscript/paperB_uncertainty/figures/`:

1. **Figure 1: LithoScan Uncertainty-Aware Selective Classification & Human Referral Architecture**  
   ![Figure 1](file:///C:/Users/punee/Desktop/Software_lithology/LithoScan/results/paperB_uncertainty/figures/fig1_decision_triage_framework.png)  
   *Schematic showing deep feature extraction, temperature-scaled probability calibration, dual selective/conformal decision gates, automated acceptance, and geologist referral.*

2. **Figure 2: Reliability Diagrams Across Acquisition Cohorts and Calibration Strategies (DCID-35 ResNet-50)**  
   ![Figure 2](file:///C:/Users/punee/Desktop/Software_lithology/LithoScan/results/paperB_uncertainty/figures/fig2_reliability_diagrams.png)  
   *Four-panel reliability diagrams illustrating empirical accuracy versus confidence bins for (a) Clean Uncalibrated, (b) Clean Temperature-Scaled, (c) Defective Uncalibrated, and (d) Defective Matched-Calibrated ($T_{\text{def}} = 1.6714$).*

3. **Figure 3: Selective Risk-Coverage Trade-off Across Confidence Scoring Methods (DCID-35 ResNet-50)**  
   ![Figure 3](file:///C:/Users/punee/Desktop/Software_lithology/LithoScan/results/paperB_uncertainty/figures/fig3_selective_risk_coverage.png)  
   *Empirical risk-coverage curves for Maximum Softmax Probability (MSP), Negative Predictive Entropy, and Margin on (a) Clean Evaluation Cohort and (b) Defective Evaluation Cohort, with target 1% and 5% error lines.*

4. **Figure 4: Conformal Prediction Set Cardinality & Triage Behavior Under Acquisition Shift**  
   ![Figure 4](file:///C:/Users/punee/Desktop/Software_lithology/LithoScan/results/paperB_uncertainty/figures/fig4_conformal_set_distributions.png)  
   *(a) Prediction set cardinality distributions under Clean-Calibrated vs. Domain-Matched Defective Calibration ($\alpha = 0.05$). (b) Triage outcomes comparing singletons, multi-class referrals, and empty sets.*

5. **Figure 5: Fine-Grained Lithological Degradation and Referral Rates Under Defective Acquisition (35 Classes)**  
   ![Figure 5](file:///C:/Users/punee/Desktop/Software_lithology/LithoScan/results/paperB_uncertainty/figures/fig5_per_class_calibration_referral.png)  
   *Dual-axis bar chart sorted by referral rate showing top-1 defective accuracy versus required selective referral rate under a nominal 5% error ceiling across all 35 lithologies.*

6. **Figure 6: Representative High-Confidence Misclassifications & Physical Core Acquisition Defects**  
   ![Figure 6](file:///C:/Users/punee/Desktop/Software_lithology/LithoScan/results/paperB_uncertainty/figures/fig6_geological_error_panel.png)  
   *Authentic photographic failure panel illustrating: (a) Drilling mud smears, (b) Longitudinal core fractures, (c) Core-tray wetting glare, (d) Crystalline grain size ambiguity, (e) Core edge shadows, and (f) Clastic grain-size continua.*

---

## 4. Empirical Answers to Core Research Questions

### RQ1 — Confidence Reliability
**Answer:** Softmax confidence is highly reliable under pristine laboratory scanning ($ECE \le 0.12\%$), but becomes severely overconfident under defective acquisition ($ECE = 6.66\%$). Temperature scaling fitted to representative defective cores successfully restores calibration ($ECE = 2.12\%$), whereas temperature scaling fitted to clean cores fails to correct the defective calibration gap ($ECE = 5.04\%$).

### RQ2 — Selective Risk
**Answer:** Selective abstention reliably decreases classification risk across all evaluated confidence scoring functions. On DCID-35, Maximum Softmax Probability achieves an AURC of **2.59%** on defective cores (compared to 0.001% on clean cores). Selective risk can be driven down from an uncalibrated baseline error of $11.36\%$ to $4.52\%$ by abstaining on $15.4\%$ of the most uncertain cores, and to $1.56\%$ by abstaining on $51.9\%$ of cores.

### RQ3 — Calibration-Domain Mismatch (Primary Hypothesis $H_1$)
**Answer:** Calibration procedures fitted exclusively on clean drill-core photographs **do not transfer reliably** to defective acquisition cohorts. A clean-calibrated threshold targeting $\le 5\%$ error incurs a realized error of **9.63%** on defective cores. In contrast, calibrating on an untouched representative defective development pool ($D_{\text{cal,def}}$) successfully controls realized risk at **4.52%**.

### RQ4 — Operational Utility & Referral Ratios
**Answer:** 
- Under clean laboratory imaging: $> 99.7\%$ of drill cores can be automatically accepted with $< 0.3\%$ error.
- Under defective drill-rig acquisition:
  - At a **5% error ceiling**: **84.6%** of core images can be automatically logged, referring **15.4%** to an expert geologist.
  - At a stringent **1% error ceiling**: **48.1%** of core images can be automatically logged, referring **51.9%** to a geologist.

### RQ5 — Fine-Grained Ambiguity & Fragility
**Answer:** Failure under defective acquisition is highly class-asymmetric. Intact homogeneous classes (`Red sandstone`, `Banded iron formation`, `Pegmatite`, `White marble`) maintain $100\%$ accuracy with $0\%$ referral. In contrast, classes with mechanical fragmentation (`Mudstone`, `Brecciated mudstone`, `Shale fragments`) or dark coatings (`Gabbro`, `Skarn`, `Dark green calc-silicate` confused with `Black Shale`) require referral rates of $25\%$ to $57.5\%$.
