# 07: MINIMUM VIABLE PUBLICATION ROADMAP — LITHOSCAN

**Audit Date:** October 22, 2026  
**Auditor:** Skeptic Senior Research Scientist & Computer Vision Reviewer  
**Target Venue:** *Computers & Geosciences* or *Geoenergy Science and Engineering*  
**Paper Direction:** Direction A (*The Illusion of 99% Accuracy: Dissecting Spatial Leakage, Acquisition Corruption, and Selective Prediction in Automated Drill-Core Lithology Benchmarks*)

---

## 1. Roadmap Architecture and Decision Gates

```
[ Stage 0: Forensic Reconciliation ] ──► GATE 0: Data Integrity Locked
                 │
                 ▼
[ Stage 1: Diagnostic Leakage & Intervention Protocol ] ──► GATE 1: Stress-Test Matrix Complete
                 │
                 ▼
[ Stage 2: Selective Prediction & Conformal Calibration ] ──► GATE 2: Triage Guarantees Verified
                 │
                 ▼
[ Stage 3: Statistical Significance & Calibration Metrics ] ──► GATE 3: Significance Verified
                 │
                 ▼
[ Stage 4: Geological Error Analysis & Domain Context ] ──► GATE 4: Petrological Validity Checked
                 │
                 ▼
[ Stage 5: Manuscript Preparation & Reproducibility Bundle ] ──► SUBMISSION READY
```

---

## 2. Stage-by-Stage Implementation Plan

### Stage 0: Baseline Reconciliation & Provenance Freeze
- **Goal:** Completely eliminate historical metric contradictions between Phase 2, Phase 3, Phase 5A, and `app.py`. Document immutable dataset provenance.
- **Exact Deliverables:**
  1. Updated Phase 5A summary table that correctly lists the true fine-tuned ResNet-50 baseline (`Clean Acc/F1: 0.9969`, `Noise Acc: 0.8864`, `Noise F1: 0.8922`).
  2. Deprecation note attached to the Phase 3 frozen linear probe metric (0.9928) in `reports/phase5/` and `app.py` explaining its historical misattribution.
  3. Immutable dataset provenance declaration documenting the lack of spatial/well coordinates in DCID.
- **Acceptance Criteria:** Every reported metric in all summary tables reconciles 1-to-1 with its underlying `metrics.json` and confusion matrix.
- **Compute Cost:** 0 GPU hours (CPU file updates only).
- **Dependencies:** Research readiness audit deliverables.
- **Risks:** None.
- **Estimated Effort:** ~4–6 hours.

---

### Stage 1: Diagnostic Leakage & Intervention Protocol
- **Goal:** Fill the critical missing empirical cells across DCID-35 and evaluate the perceptual similarity filter on 35 classes.
- **Exact Deliverables:**
  1. Evaluate the saved Phase 2B ResNet-50 DCID-35 checkpoint under the 40px border occlusion mask to complete the three-way intervention table.
  2. Extend `src/scripts/evaluate_generalization.py` to run on DCID-35 across thresholds $k \in \{2, 5, 10\}$.
  3. Generate class-retention statistics and evaluate all three DCID-35 models (ResNet-50, CNN-only, CNN-Transformer) on the similarity-filtered subsets.
- **Acceptance Criteria:** A completed, non-empty three-way comparison table across clean, noise, and border-mask conditions for all models on both DCID-7 and DCID-35, plus filtered test results on DCID-35.
- **Compute Cost:** ~1.5 GPU hours on RTX 3050 (inference only; no training).
- **Dependencies:** Existing checkpoints in `checkpoints/exp_B/` and `checkpoints/phase5/`.
- **Risks:** dHash filtering on DCID-35 may cause severe sample collapse for specific fine-grained classes.
- **Estimated Effort:** ~1–2 days.

---

### Stage 2: Selective Prediction & Conformal Calibration
- **Goal:** Transform the project from a passive benchmark evaluation into a functional, uncertainty-aware logging framework.
- **Exact Deliverables:**
  1. Standalone prediction extractor script that saves test-set logits and softmax probabilities (`test_probs.npy`, `targets.npy`) for clean and noisy sets across all architectures.
  2. Implementation of **Split Conformal Prediction** in [src/calibration/conformal.py](file:///C:/Users/punee/Desktop/Software_lithology/LithoScan/src/calibration/conformal.py) using nonconformity scores $s_i = 1 - \hat{P}(y_i | x_i)$.
  3. Construction of **Risk-Coverage Curves**: Plot empirical error rate vs. sample coverage (abstention rate) across clean and noisy test sets.
  4. Measurement of the **Human Referral Rate**: Determine the percentage of defective/noisy core samples that must be referred to a human geologist to maintain an operational error bound $\le 0.5\%$ and $\le 1.0\%$.
- **Acceptance Criteria:** Conformal prediction sets provide valid marginal coverage $1 - \alpha$ for $\alpha \in \{0.01, 0.05, 0.10\}$ on both clean and noise evaluation sets.
- **Compute Cost:** ~0.5 GPU hours (forward passes only) + CPU numerical processing.
- **Dependencies:** Stage 1 checkpoints and outputs.
- **Risks:** Under severe noise, conformal prediction sets may become excessively large (e.g., set size > 5 classes), indicating high epistemic uncertainty. (This is an informative finding).
- **Estimated Effort:** ~2–3 days.

---

### Stage 3: Statistical Significance & Calibration Evaluation
- **Goal:** Provide rigorous statistical validation beyond raw point estimates.
- **Exact Deliverables:**
  1. **Bootstrap 95% Confidence Intervals:** Compute percentile bootstrap intervals ($B=2,000$ iterations) for clean accuracy, noise accuracy, macro-F1, and performance drop across all models.
  2. **Paired Statistical Tests:** Execute McNemar's test and paired Wilcoxon signed-rank tests between CNN-only and CNN-Transformer on the test predictions to formally determine whether the 0.03% clean deficit or 1.43% noise gain is statistically significant.
  3. **Calibration Metrics:** Compute Expected Calibration Error (ECE), Brier score, and plot reliability diagrams (calibration curves) before and after temperature scaling.
- **Acceptance Criteria:** All reported comparison tables include explicit $95\%$ confidence intervals (e.g., $0.9974 \pm 0.0012$) and $p$-values.
- **Compute Cost:** 0 GPU hours (CPU statistical computation).
- **Dependencies:** Logits and predictions extracted in Stage 2.
- **Risks:** $p$-values will likely show that CNN-only and CNN-Transformer are statistically indistinguishable on clean data. (This reinforces the paper's thesis).
- **Estimated Effort:** ~1–2 days.

---

### Stage 4: Geological Failure Analysis & Domain Grounding
- **Goal:** Bridge pure computer vision metrics with real-world geological interpretation.
- **Exact Deliverables:**
  1. Extraction and visual inspection of the **top 20 consistent failure cases** (images misclassified by both CNN and CNN-Transformer on clean and noise sets).
  2. Categorization of errors into:
     - *Petrologically adjacent confusions* (e.g., *4.Gray siltstone* vs. *7.Mudstone* — natural transitional lithologies).
     - *Artifact-driven failures* (prominent cracks or mud mistaken for bedding).
     - *Grain-size resolution limits* (optical resolution unable to resolve microscopic crystal boundaries).
  3. Synthesis section articulating why optical-only classification has physical limits that necessitate multi-modal logging (e.g., combining vision with XRF, gamma ray, or scratch tests).
- **Acceptance Criteria:** A structured qualitative failure taxonomy illustrated with representative core image panels.
- **Compute Cost:** 0 GPU hours.
- **Dependencies:** Confusion matrices and prediction indices from Stage 2.
- **Risks:** Lack of access to raw rock samples for physical hardness/acid testing. (Mitigated by focusing on visible image features).
- **Estimated Effort:** ~2 days.

---

### Stage 5: Publication Package & Manuscript Assembly
- **Goal:** Draft the complete, defensible manuscript and prepare an open reproducibility package.
- **Exact Deliverables:**
  1. Complete manuscript structured for *Computers & Geosciences* or *Geoenergy Science and Engineering*:
     - Abstract & Introduction: The paradox of >99% benchmarks vs. field fragility.
     - Dataset & Leakage Audit: Empirical proof of perceptual duplication and spatial autocorrelation.
     - Experimental Protocol: Clean, RWDA noise, and border occlusion evaluations.
     - Uncertainty-Aware Selective Prediction: Conformal triage results and risk-coverage operating curves.
     - Geological Discussion & Limitations: Why automated logging requires selective abstention.
  2. Publication-quality vector figures (matplotlib/seaborn PDF/SVG):
     - Figure 1: Pipeline diagram showing core patch ingestion and selective triage.
     - Figure 2: dHash perceptual similarity distribution across splits.
     - Figure 3: Clean vs. RWDA noise degradation across models.
     - Figure 4: Risk-coverage curves and ECE reliability diagrams.
     - Figure 5: Qualitative failure analysis panel.
  3. Reproducibility package: Clean repository release with frozen conda/pip environment, trained checkpoint download links, and evaluation scripts.
- **Acceptance Criteria:** Manuscript passes internal co-author review with zero unsupported architectural or novelty claims.
- **Compute Cost:** Minimal.
- **Dependencies:** Stages 0 through 4 complete.
- **Estimated Effort:** ~7–10 days.

---

## 3. Decision Gates and Stop/Go Criteria

| Gate | Stage Boundary | Condition to PROCEED | Condition to STOP / PIVOT |
| :--- | :--- | :--- | :--- |
| **GATE 0** | After Stage 0 | All historical discrepancies reconciled; clean test metrics agreed across all files. | Any untraceable metric in baseline runs. |
| **GATE 1** | After Stage 1 | ResNet-50 border mask evaluated; DCID-35 similarity filtering successfully produces retained cohorts. | If dHash filtering on DCID-35 eliminates >95% of classes (in which case dHash threshold must be widened). |
| **GATE 2** | After Stage 2 | Conformal prediction achieves $\ge 95\%$ empirical coverage on noise test set with average set size $\le 3$ classes. | If prediction sets span >10 classes under noise (indicating base model total failure, requiring temperature scaling). |
| **GATE 3** | After Stage 3 | Statistical significance tests completed; confidence intervals narrow enough to draw clear empirical conclusions. | None. |
| **GATE 4** | After Stage 4 | Geological failure analysis provides sensible petrological rationale for errors. | None. |

---

## 4. Aggregate Resource Summary

- **Total Estimated Effort:** ~13–19 working days (approximated for an experienced researcher).
- **Total GPU Training Required:** **0 hours** (No retraining required; leverages existing checkpoints).
- **Total GPU Inference Required:** **~2.5 hours** on the local RTX 3050 Laptop GPU.
- **Financial / Cloud Cost:** **$0.00** (Entire plan executes locally on existing hardware).
