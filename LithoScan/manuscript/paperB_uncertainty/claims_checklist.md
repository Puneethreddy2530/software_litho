# Claims Checklist — LithoScan Paper B
**Title:** When Should a Lithology Classifier Abstain? Calibration, Selective Risk, and Human Referral Under Defective Drill-Core Imaging
**Target Venue:** Computers & Geosciences (Elsevier)
**Checklist Version:** 1.0 — October 2026
**Status:** LOCKED — Full Scientific Integrity PASS

---

## Purpose

This checklist maps every quantitative claim in the manuscript (main.md / main.tex) to its supporting artifact, verifies that no claim exceeds the available evidence, and records all required scientific disclosures.

---

## Section 1 — Abstract Claims

| Claim | Exact Value | Supporting Artifact | Verified? | Notes |
|:---|:---|:---|:---:|:---|
| Clean accuracy (ResNet-50, DCID-35) | 99.73% | metrics/dcid35_resnet50_full_results.json | YES | Eval partition Deval,clean |
| Defective accuracy (ResNet-50, DCID-35) | 88.64% | metrics/dcid35_resnet50_full_results.json | YES | Eval partition Deval,def |
| Accuracy drop | 11.09 pp | Derived from above two | YES | Consistent with CNN-only 12.3 pp |
| Uncalibrated ECE (clean) | 0.12% [0.07, 0.25] | metrics/dcid35_resnet50_full_results.json | YES | M=15 fixed bins |
| Uncalibrated ECE (defective) | 6.66% [5.74, 7.69] | metrics/dcid35_resnet50_full_results.json | YES | 2000 bootstrap resamples |
| Clean-calibrated tau (5% target) | 0.5389 | metrics/dcid35_resnet50_full_results.json | YES | Derived on Dcal,clean only |
| Realized error (clean-cal, defective eval) | 9.63% | metrics/dcid35_resnet50_full_results.json | YES | PRIMARY HYPOTHESIS H1 CONFIRMED |
| Domain-matched tau (5% target) | 0.9295 | metrics/dcid35_resnet50_full_results.json | YES | Derived on Dcal,def only |
| Realized error (domain-matched, def eval) | 4.52% | metrics/dcid35_resnet50_full_results.json | YES | <=5.0% ceiling PASS |
| Coverage (domain-matched, 5% target) | 84.6% | metrics/dcid35_resnet50_full_results.json | YES | 15.4% referral rate |
| Conformal coverage (clean-cal, defective) | 57.68% [56.07,59.29] | metrics/dcid35_resnet50_full_results.json | YES | alpha=0.05; 37.32pp undercoverage |
| Empty set fraction (clean-cal, defective) | 41.4% | metrics/dcid35_resnet50_full_results.json | YES | Coverage collapse |
| Conformal coverage (domain-matched) | 95.57% [94.86,96.29] | metrics/dcid35_resnet50_full_results.json | YES | H2 CONFIRMED |
| Wilcoxon p-value | 2.54e-198 | metrics/dcid35_resnet50_full_results.json | YES | |
| Mean conformal set size (domain-matched) | 1.40 | metrics/dcid35_resnet50_full_results.json | YES | |

---

## Section 2 — FORBIDDEN CLAIMS

| Forbidden Claim | Status |
|:---|:---:|
| Spatial generalization across boreholes | FORBIDDEN — no borehole metadata |
| Temperature scaling improves accuracy | FORBIDDEN — post-hoc calibration property |
| SNN/neuromorphic energy advantages | FORBIDDEN — Phase 6 isolated from Paper B |
| Defective cohort = synthetic paired images | FORBIDDEN — independent physical photographs |
| Historical test set was pristine | FORBIDDEN — retrospective benchmark disclosed |
| Near-duplicate audit proves spatial leakage | FORBIDDEN — no source metadata |
| Conformal guarantees transfer across cohorts | FORBIDDEN — exchangeability violation disclosed |

---

## Section 3 — Required Disclosures

| Disclosure | Section in manuscript | Present? |
|:---|:---|:---:|
| Absence of borehole/depth metadata | Section 3.1 and 7.2 | YES |
| Independent (not paired) defective cohort | Section 3.1 and 5.1 | YES |
| Retrospective benchmark acknowledgment | Section 7.2 | YES |
| Near-duplicate contamination risk | Section 5.4 | YES |
| Protocol locked and SHA-256 hashed | Section 3.2 | YES |
| Marginal coverage != operational safety | Section 4.3 | YES |
| Human expertise required for referrals | Section 7.4 | YES |

---

## Section 4 — Geological Error Claims

| Claim | Value | Artifact | Verified? |
|:---|:---|:---|:---:|
| Total persistent errors audited | 217 | 03_geological_failure_analysis.md | YES |
| Drilling sludge masking | 38.2% | 03_geological_failure_analysis.md | YES |
| Mechanical fracture/brecciation | 24.4% | 03_geological_failure_analysis.md | YES |
| Igneous cooling continua | 19.8% | 03_geological_failure_analysis.md | YES |

---

## OVERALL VERDICT: APPROVED FOR SUBMISSION
