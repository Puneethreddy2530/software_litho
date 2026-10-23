# 01: SCIENTIFIC CLAIMS AND EVIDENCE CORRECTION REGISTER

**Audit Gate:** Source Provenance & Claim Correction  
**Date:** October 22, 2026  
**Auditor:** Skeptic Senior Research Scientist, Vision Reviewer & Methodology Auditor  
**Scope:** Re-evaluation of prior empirical interpretations, decoupling verified observations from speculative inferences across Phases 1 through 5A.

---

## 1. Epistemological Framework

To maintain scientific integrity, all statements within LithoScan documentation are categorized according to this strict evidentiary hierarchy:
- **[VERIFIED FACT]**: Directly observed, mathematically recomputed, or verified in raw filesystem artifacts.
- **[OBSERVED RESULT]**: Empirical output recorded by an executed software run under specific constraints.
- **[INFERENCE]**: A logical deduction supported by observations, but requiring further testing.
- **[UNVERIFIED HYPOTHESIS]**: A plausible explanation lacking definitive proof.
- **[ERRONEOUS CLAIM]**: An assertion demonstrated to be factually incorrect or unsupported by the evidence.

---

## 2. Claim Corrections

### Correction 1: Perceptual Similarity (dHash) Does Not Prove Spatial Leakage

- **Previous Working Claim:**  
  *"The presence of 492 pairs with dHash $\le 2$ in DCID-7 and 203 pairs in DCID-35 between train and test proves cross-split spatial data leakage."*
- **Verified Fact:**  
  Independent calculation of 64-bit difference hashes confirms that cross-split pairs share near-identical binary hash values (Hamming distance $\le 2$). Zero identical MD5 file hashes exist between splits.
- **Scientific Correction:**  
  **dHash similarity measures perceptual pattern resemblance, NOT geographic or physical core identity.** In visually homogeneous rock classes (e.g., fine-grained basalt, massive mudstone, uniform red sandstone), two distinct core samples extracted from different boreholes miles apart can exhibit near-identical pixel gradient distributions. While high perceptual similarity demonstrates benchmark redundancy and limits the diversity of the test set, **it does not constitute mathematical proof that the images were cropped from adjacent intervals of the same physical drill core**. Spatial leakage remains an **UNVERIFIED HYPOTHESIS** until parent borehole and depth metadata are established.

---

### Correction 2: Unchanged Accuracy in Phase 4D Does Not Prove Independent Generalization

- **Previous Working Claim:**  
  *"Model accuracy remained unchanged after filtering out near-duplicate images up to dHash $\le 10$, proving the models learn robust geological features that generalize independently of perceptual similarity."*
- **Verified Fact:**  
  When test images with training neighbors within dHash distance $\le 10$ were removed from DCID-7, ResNet-50 retained 1.0000 accuracy, CNN-only retained 0.9995, and CNN-Transformer retained 0.9993.
- **Scientific Correction:**  
  **Pruning perceptually similar test samples does not prove true out-of-domain generalization.** Because the underlying borehole origin of the images is unknown, the retained test images may still originate from the same parent core boxes as the training images (differing only by surface texture, lighting, or fracture orientation). True independent generalization can only be established by evaluating on an independently drilled borehole or external geographic site. Retained accuracy on a filtered subpopulation within the same dataset is an **OBSERVED RESULT** of subset filtering, not proof of independent generalization.

---

### Correction 3: Severe Class Imbalance in Phase 4D Filtered Subsets

- **Previous Working Claim:**  
  *"Phase 4D demonstrates consistent model robustness across all filtered subsets (TEST_FILTERED_2, TEST_FILTERED_5, TEST_FILTERED_10)."*
- **Verified Fact:**  
  Inspection of `reports/phase4/generalization_stress_test/class_distribution.csv` reveals extreme class distortion at $k=10$:
  - *6.Basalt* collapsed from 1,000 images down to **115 images (an 88.5% loss)**.
  - *2.Light sandstone* dropped from 1,000 to **355 images (a 64.5% loss)**.
  - *4.Mudstone* retained **932 images (only a 6.8% loss)**.
- **Scientific Correction:**  
  **The filtered test cohort at $k=10$ is severely class-imbalanced.** The aggregate accuracy of 99.93% is heavily dominated by classes that suffered minimal filtering (*Mudstone*, *Marble*). Reporting aggregate accuracy without explicitly declaring this population distortion creates a misleading impression of uniform robustness across all lithologies.

---

### Correction 4: Phase 4D Findings on DCID-7 Must Not Be Transferred to DCID-35

- **Previous Working Claim:**  
  *"The generalization robustness demonstrated in Phase 4D applies to the LithoScan classification system broadly."*
- **Verified Fact:**  
  Phase 4D was executed **strictly on DCID-7**. Not a single similarity-filtered inference pass has ever been run on the 35 fine-grained classes of DCID-35.
- **Scientific Correction:**  
  **No conclusion regarding similarity filtering on DCID-35 currently exists.** Fine-grained classes share much tighter visual margins; filtering at $k=5$ or $k=10$ may eradicate entire fine-grained categories or induce severe performance drops. Claiming Phase 4D robustness for DCID-35 is an **UNVERIFIED CLAIM**.

---

### Correction 5: The CNN-Transformer Architecture Is Not a Demonstrated Novel Contribution

- **Previous Working Claim:**  
  *"LithoScan Server Edition introduces a novel, texture-aware CNN + Transformer fusion model that outperforms baselines."*
- **Verified Fact:**  
  1. The code in `src/models/server_fusion.py` implements a standard Torchvision ResNet-50 backbone coupled to a standard 2-layer `nn.TransformerEncoderLayer`. No texture-specific filtering (Gabor, LBP) or multi-scale fusion modules exist.
  2. On DCID-7: ResNet-50 baseline outperforms the CNN-Transformer across clean data (1.0000 vs 0.9994), noise data (0.9679 vs 0.9664), and 40px border occlusion (0.9994 vs 0.9984).
  3. On DCID-35: Clean accuracy of CNN-Transformer (0.9974) is lower than CNN-only (0.9977). Its noise accuracy advantage over ResNet-50 baseline is a negligible 0.29% (0.8893 vs 0.8864), while requiring 6.4M additional parameters and nearly quadruple the training compute of ResNet-50.
- **Scientific Correction:**  
  **The CNN-Transformer is a conventional architecture adaptation, not a novel research contribution.** The empirical results falsify the claim that it provides a superior operating model for drill-core classification.

---

### Correction 6: Clean vs. Noisy Performance Is Not a Paired Causal Corruption Effect

- **Previous Working Claim:**  
  *"Evaluating clean vs. noisy test sets isolates the causal degradation effect of field noise on core classification."*
- **Verified Fact:**  
  1. Clean test set contains 7,000 images (DCID-7: 1,000/class; DCID-35: 200/class).
  2. Noise test set contains 2,800 images (DCID-7: 400/class; DCID-35: 80/class).
  3. Analysis of `build_dcid_dataset.py` proves that `noise-512-7` and `noise-512-35` are an independent pool of naturally defective images sampled at a 40% ratio, not bit-wise corruptions generated by adding synthetic noise to the clean test images.
- **Scientific Correction:**  
  **The observed accuracy drop (11.0% to 12.3% on DCID-35) represents evaluation across two distinct sample populations, NOT a paired treatment effect on identical images.** While it successfully demonstrates that models perform worse on defective/damaged core populations, it cannot be reported as a per-image paired sensitivity curve.

---

### Correction 7: Conformal Coverage Guarantees Do Not Hold Under Uncalibrated Distribution Shift

- **Previous Working Claim:**  
  *"Conformal prediction guarantees a 95% confidence error bound across field deployment and noise conditions."*
- **Verified Fact:**  
  `src/calibration/conformal.py` is currently an unexecuted 19-line stub.
- **Scientific Correction:**  
  Standard split conformal prediction guarantees marginal coverage $1 - \alpha$ **only under the exchangeability assumption** (calibration and test samples drawn i.i.d. from the same distribution). If a conformal model is calibrated on clean training/validation data and deployed on defective or out-of-domain core images (covariate shift), the coverage guarantee is formally violated and empirical error rates will exceed $\alpha$ unless weighted conformal prediction or explicit domain-shift calibration (e.g., Tibshirani et al., 2019) is implemented.

---

### Correction 8: Disentangling Fine-Tuned ResNet-50 from Frozen Linear Probe

- **Previous Working Claim (in Phase 5A reports and `app.py`):**  
  *"ResNet-50 baseline on DCID-35 achieved Val F1 0.9882 and Test F1 0.9928, with noise performance untested."*
- **Verified Fact:**  
  There are two completely separate ResNet-50 evaluations in the project archives:
  1. **Fine-Tuned ResNet-50 (Phase 2B, `checkpoints/exp_B/metrics.json`):**  
     - Best Validation Accuracy: 0.9982  
     - Best Validation Macro-F1: 0.9982 (Epoch 3)  
     - Official Clean Test Accuracy: 0.9969  
     - Official Clean Test Macro-F1: 0.9969  
     - Official Noise Test Accuracy: 0.8864 (`reports/exp_B_robustness/robustness_summary.json`)  
     - Official Noise Test Macro-F1: 0.8922  
  2. **Frozen-Feature Linear Probe (Phase 3, `reports/phase3/linear_baseline_metrics.csv`):**  
     - Scikit-learn Logistic Regression on frozen ImageNet features.  
     - Validation Accuracy: 0.9885, Validation Macro-F1: 0.9882  
     - Test Accuracy: 0.9928, Test Macro-F1: 0.9928  
     - Noise evaluation: Not performed (N/A)  
- **Scientific Correction:**  
  **Phase 5A comparison tables and dashboard code mistakenly substituted the Phase 3 frozen linear probe for the Phase 2B fine-tuned baseline.** When the true fine-tuned ResNet-50 is compared against Phase 5A models, the CNN-Transformer's clean test accuracy (0.9974) is virtually identical to ResNet-50 (0.9969), and its noise accuracy advantage is only 0.29% (0.8893 vs 0.8864).
