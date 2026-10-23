# 08: RESEARCH READINESS CHECKLIST — LITHOSCAN

**Audit Date:** October 22, 2026  
**Auditor:** Skeptic Senior Research Scientist & Methodology Auditor  
**Evaluation Standard:** Top-Tier Earth Science & AI Journal Submission Readiness (*Computers & Geosciences* / *Geoenergy Science and Engineering* / *IEEE TGRS*)

---

## 1. Verified vs. Unverified Requirements Checklist

### A. Experimental & Computational Foundations
- [x] **[VERIFIED]** Checkpoint integrity: All final model checkpoints (`best_model.pt` in `exp_A`, `exp_B`, `phase4/`, `phase5/`) are present, non-corrupt, and loadable onto PyTorch devices without key mismatches.
- [x] **[VERIFIED]** Split reproducibility: Deterministic train/validation split (`seed=42`) verified to yield identical image subsets across architectures.
- [x] **[VERIFIED]** Test isolation: Official test set was strictly held out during training and model selection.
- [x] **[VERIFIED]** Scheduler correctness: CosineAnnealingLR advances once per epoch matching `T_max`.
- [x] **[VERIFIED]** Metrics reproducibility: Recomputed accuracy and macro-F1 from confusion matrices match JSON artifacts to 8 decimal places.
- [ ] **[UNVERIFIED]** Statistical significance: No error bars, bootstrap confidence intervals, or hypothesis tests currently accompany any reported metric.
- [ ] **[UNVERIFIED]** Model calibration: Expected Calibration Error (ECE) and probability reliability diagrams are completely missing.
- [ ] **[UNVERIFIED]** Architectural superiority: CNN-Transformer fails to demonstrate statistically meaningful superiority over standard ResNet-50.

### B. Dataset & Geological Foundations
- [x] **[VERIFIED]** Population balance: DCID-7 is perfectly balanced; DCID-35 is near-perfect (single sample deficit verified in *Banded iron formation*).
- [x] **[VERIFIED]** Exact file uniqueness: Zero identical MD5 hashes exist between official train and test splits.
- [x] **[VERIFIED]** Defect impact quantified: Real-World Data Augmentation (RWDA noise) verified to induce a 3.2% drop on DCID-7 and 11.0%–12.3% drop on DCID-35.
- [ ] **[MISSING EVIDENCE]** Spatial / Borehole independence: Zero metadata exists regarding borehole IDs, depths, or geographic sites.
- [ ] **[MISSING EVIDENCE]** Cross-site generalization: Field generalization remains entirely unproven.
- [ ] **[UNVERIFIED]** Stratigraphic validity of augmentations: Vertical flipping and hue jittering were applied without validating impact on directional geological features.

### C. System & Scope Integrity
- [x] **[VERIFIED]** Interrupted runs isolated: Phase 5 partial runs are safely archived and excluded from final results.
- [ ] **[UNVERIFIED CLAIM]** "Texture-aware" architecture: No texture-specific modules exist in code.
- [ ] **[MISSING ARTIFACT]** "Field Edition SNN": Stub file with zero implementation (`field_snn.py`).
- [ ] **[MISSING ARTIFACT]** "Byzantine-robust Federated Learning": Stub file with zero implementation (`aggregation.py`).
- [ ] **[MISSING ARTIFACT]** "Conformal Prediction triage layer": Stub file with zero implementation (`conformal.py`).

---

## 2. Requirements Prior to Manuscript Drafting

The following items are **MANDATORY GATEKEEPERS** before drafting the introduction or abstract of a scientific manuscript:

1. **Reconcile Baseline Discrepancy:**
   - Eliminate the 0.9928 vs 0.9969 ResNet-50 contradiction across all tables, reports, and `app.py`. Ensure the Phase 2B fine-tuned baseline is accurately credited.
2. **Implement Split Conformal Prediction & Selective Classification:**
   - Write and execute the calibration logic in `src/calibration/conformal.py`.
   - Extract softmax probabilities on clean and noisy test sets.
   - Compute empirical coverage and average set sizes for $\alpha \in \{0.01, 0.05, 0.10\}$.
3. **Execute DCID-35 Similarity Stress Test:**
   - Extend Phase 4D filtering to the 35-class dataset to determine whether fine-grained classes collapse under dHash thresholding.
4. **Complete ResNet-50 40px Border Mask on DCID-35:**
   - Run inference on the saved Phase 2B checkpoint under border masking to fill the "N/A" cell in the Phase 5A three-way comparison table.
5. **Drop All Stubs from Paper Scope:**
   - Explicitly remove SNN and Federated Learning from the paper's title, abstract, and claims. Confine the contribution to the empirical audit and selective prediction framework.

---

## 3. Requirements Prior to Final Submission

1. **Statistical Rigor:**
   - Every metric table must present $95\%$ bootstrap confidence intervals ($B=2,000$).
   - Include McNemar's test $p$-values comparing CNN-only and CNN-Transformer.
2. **Calibration & Reliability Figures:**
   - Include reliability diagrams (calibration curves) comparing raw softmax confidence vs. calibrated conformal sets.
   - Include risk-coverage curves showing precision gains as uncertain/defective samples are abstained.
3. **Qualitative Geological Review:**
   - Include a panel illustrating the top 10 misclassified clean and top 10 misclassified noise samples, with commentary on visual vs. petrological ambiguity.
4. **Data & Code Availability Statement:**
   - Explicitly cite DCID authors (Li et al., 2025, *Petroleum Science*, DOI: 10.1016/j.petsci.2025.04.013).
   - Document CC BY-NC 4.0 license terms.
   - Provide an anonymized GitHub / Zenodo link containing the clean evaluation code, trained checkpoints, and environment configuration (`requirements.txt`).
5. **Limitations Section:**
   - Acknowledge that DCID lacks borehole and depth coordinates, preventing a Leave-One-Well-Out split.
   - Clarify that optical core photography cannot replace elemental (XRF) or petrophysical (gamma-ray, scratch) logging.
