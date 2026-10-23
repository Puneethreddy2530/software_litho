# 05: NEXT EXPERIMENT DECISION FRAMEWORK & GO/NO-GO RECOMMENDATION

**Audit Gate:** Source Provenance & Claim Correction  
**Date:** October 22, 2026  
**Auditor:** Skeptic Senior Research Scientist & Methodology Auditor  
**Decision Objective:** Establish the single most scientifically defensible, resource-efficient next experimental route for LithoScan.

---

## 1. The Three Mutually Exclusive Routes

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STRATEGIC EXPERIMENT ROUTE SELECTION                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ ROUTE A: Source-Group Metadata Recoverable (Grouped DCID LOWO Split)        │
│          Status: BLOCKED / PENDING AUTHOR COOPERATION                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ ROUTE B: External-Domain Evaluation (DCID -> GSSA HyLogger Borehole Transfer)│
│          Status: HIGHLY VIABLE ALTERNATIVE (Requires External Data Intake)  │
├─────────────────────────────────────────────────────────────────────────────┤
│ ROUTE C: Benchmark Fragility, Corruption & Conformal Triage (DCID-Only)     │
│          Status: IMMEDIATELY ACTIONABLE (Zero GPU Training Required)        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Detailed Specification of Each Route

### Route A: Source-Group Metadata Is Recoverable (Grouped Leakage-Resistant Study)

- **Scientific Question:**  
  *How significantly does classification performance on DCID degrade when evaluated under a strict Leave-One-Well-Out (LOWO) or core-box grouped split compared to the official random i.i.d. split?*
- **Required Data:**  
  DCID images paired with an authoritative author-provided metadata table containing borehole identifiers (`well_id`), depth intervals (`depth_m`), or continuous core box numbers (`core_tray_id`).
- **Minimum Experiments:**  
  1. Re-split DCID-35 using `GroupKFold` on the recovered well or tray IDs (holding out 20% of independent drill holes for testing).
  2. Retrain ResNet-50 and CNN-only on the grouped split.
  3. Compare grouped test accuracy against the random test accuracy (0.9969) to measure the true spatial leakage penalty.
- **Analysis & Statistical Plan:**  
  - Measure performance degradation $\Delta = \text{Acc}_{\text{random}} - \text{Acc}_{\text{grouped}}$.
  - Paired $t$-test / Wilcoxon signed-rank test across folds.
  - Variogram or Moran's $I$ spatial autocorrelation analysis downhole.
- **What Supports vs. Falsifies Hypothesis:**  
  - *Supports:* Grouped accuracy drops by $>5\%$ to $15\%$, proving that random splitting masked significant spatial autocorrelation and leakage.
  - *Falsifies:* Grouped accuracy remains $>99\%$, proving that rock lithology features are sufficiently distinct that spatial autocorrelation played no meaningful role in benchmark performance.
- **Hardware & Effort Requirements:**  
  - Compute: ~15–25 GPU hours on RTX 3050 (full retraining across folds).
  - Effort: ~2–3 weeks (heavily dependent on author response latency).
- **Novelty & Feasibility Assessment:**  
  Extremely high novelty for earth science journals, but **FEASIBILITY IS CURRENTLY BLOCKED** because the authors have not provided metadata, and public archives contain zero well tags.

---

### Route B: External-Domain Transfer (DCID $\to$ GSSA HyLogger Boreholes)

- **Scientific Question:**  
  *Do deep representations trained on the curated DCID benchmark transfer effectively to continuous, uncurated drill-core imagery from an independent government survey library across unseen boreholes?*
- **Required Data:**  
  1. Existing DCID-trained checkpoints.
  2. GSSA HyLogger Lithology Dataset (Alzubaidi et al., Mendeley Data, DOI: 10.17632/b6rj29mrss.1, CC BY 4.0). Contains 224px core patches from South Australian exploration holes labeled across Sandstone, Shale, Limestone, and Tray artifacts.
- **Minimum Experiments:**  
  1. Extract frozen representations from the Phase 2B ResNet-50 and Phase 5A CNN-only models on the GSSA dataset.
  2. Evaluate zero-shot transfer on overlapping classes (*Sandstone*, *Shale*, *Marble/Limestone*).
  3. Train a linear probe on Borehole A and evaluate out-of-domain generalization on held-out Borehole B.
- **Analysis & Statistical Plan:**  
  - Out-of-domain generalization gap: $\text{Acc}_{\text{in-domain}} - \text{Acc}_{\text{out-of-domain}}$.
  - Confusion matrix analysis to identify whether external failures stem from lighting shifts or true mineralogical variance.
- **What Supports vs. Falsifies Hypothesis:**  
  - *Supports:* Models experience substantial degradation ($>20\%$ drop) on unseen boreholes, proving that benchmark-trained models fail to generalize without domain adaptation.
  - *Falsifies:* Models achieve $>95\%$ zero-shot transfer to GSSA cores, proving that ImageNet/DCID features represent universal macroscopic petrological signatures.
- **Hardware & Effort Requirements:**  
  - Compute: ~2–4 GPU hours (feature extraction and linear probing; no deep retraining).
  - Effort: ~1–2 weeks (data ingestion, label verification, evaluation).
- **Novelty & Feasibility Assessment:**  
  High scientific credibility. Solves the missing borehole provenance issue without waiting for DCID authors.

---

### Route C: DCID-Only Empirical Audit, Defect Fragility & Conformal Triage

- **Scientific Question:**  
  *How fragile is near-perfect (>99.7%) benchmark accuracy under real-world core defects, and can distribution-free selective prediction guarantee human-level reliability by triaging damaged samples?*
- **Required Data:**  
  **Existing project data only.** Checkpoints in `checkpoints/exp_B/` and `checkpoints/phase5/`, clean test set (7,000 images), and official RWDA noise test set (2,800 images).
- **Minimum Experiments:**  
  1. Run inference on saved checkpoints to extract clean and noise prediction logits.
  2. Implement Split Conformal Prediction in `src/calibration/conformal.py` using nonconformity scores $s_i = 1 - \hat{P}(y_i | x_i)$.
  3. Construct Risk-Coverage curves: measure empirical error as a function of sample coverage.
  4. Quantify the Human Referral Rate: determine the exact fraction of defective samples that must be triaged to human geologists to bound operational error $\le 0.5\%$.
  5. Run Phase 4D similarity stress-testing on DCID-35 to measure fine-grained class retention dynamics.
- **Analysis & Statistical Plan:**  
  - Percentile bootstrap 95% confidence intervals ($B=2,000$) on all metrics.
  - McNemar's test comparing CNN-only and CNN-Transformer.
  - Marginal coverage validation ($1 - \alpha \in \{0.90, 0.95, 0.99\}$).
  - Expected Calibration Error (ECE) and reliability diagrams.
- **What Supports vs. Falsifies Hypothesis:**  
  - *Supports:* RWDA noise causes an 11–12% drop, but selective abstention (triaging the bottom 15–20% most uncertain samples) restores accuracy to $>99.5\%$, proving the viability of an autonomous-plus-human triage pipeline.
  - *Falsifies:* Conformal prediction sets span excessively large class subsets (>8 classes) or fail to bound error under noise, proving that standard calibration collapses under acquisition shift.
- **Hardware & Effort Requirements:**  
  - Compute: **0 GPU training hours**; ~1.5 GPU hours of inference on RTX 3050.
  - Effort: ~3–5 working days.
- **Limitations & Novelty Risk:**  
  - Cannot evaluate Leave-One-Well-Out spatial transfer.
  - Novelty rests on methodology (auditing benchmark fragility and proving selective prediction triage), not on multi-site field proof.

---

## 3. Evidence-Based Go/No-Go Recommendation

### **RECOMMENDED ROUTE: ROUTE C (IMMEDIATE) WITH ROUTE B CONTINGENCY**

### Decisive Rationale:
1. **Zero External Blocker:** Route A is paralyzed because the DCID authors have not provided well metadata and public files contain no spatial tags. Waiting for external communication violates forward research momentum.
2. **Immediate Execution with Existing Assets:** Route C leverages the exact checkpoints already trained in Phase 2 and Phase 5A. It requires **zero expensive retraining**, zero cloud compute, and resolves the core research questions within days.
3. **High Publication Defensibility:** A paper demonstrating that (1) clean 99.7% benchmark scores are brittle, (2) RWDA field noise induces an 11% drop, and (3) conformal prediction provides a mathematically guaranteed human-triage operating point directly answers the open challenge articulated in the 2025 *Ore Geology Reviews* survey.
4. **Strategic Escalation to Route B:** If peer reviewers subsequent to submission demand out-of-dataset validation, the project can immediately execute Route B (GSSA HyLogger transfer) using pre-existing feature embeddings.

---

## 4. Next Decision Gate

```
[ DECISION GATE 1.0 ]
  ├── Check: Has user authorized immediate implementation of Route C?
  ├── If YES: Execute Stage 2 (Conformal Prediction & Risk-Coverage Extraction).
  └── If NO / External Data Preferred: Approve Route B dataset intake (GSSA HyLogger).
```
