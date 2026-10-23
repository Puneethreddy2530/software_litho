# 06: Submission Readiness & Target Venue Audit — LithoScan Paper B

**Document ID:** `reports/paperB_uncertainty/06_submission_readiness.md`  
**Date:** October 2026  
**Auditor:** Skeptical Senior Machine-Learning Researcher, Geoscientific Journal Reviewer & Reproducibility Auditor  
**Status:** COMPLETE & VERIFIED  
**Research Track:** Paper B (*When Should a Lithology Classifier Abstain? Calibration, Selective Risk, and Human Referral Under Defective Drill-Core Imaging*)  
**Submission Decision:** **GO FOR SUBMISSION TO *COMPUTERS & GEOSCIENCES***  

---

## 1. Target Venue Evaluation & Scope Fit

### 1.1 Primary Candidate: *Computers & Geosciences* (Elsevier)
- **Aims & Scope:** *Computers & Geosciences* publishes high-impact research applying computational methods, spatial statistics, machine learning, and computer vision to geoscientific problems. Core logging, borehole imaging, automated drill-core recognition, and statistical validation are core subject areas.
- **Recent Precedents:** Published Serdoun et al. (2026, DOI `10.1016/j.cageo.2026.106156`) on weakly supervised drill-core photograph classification. Paper B directly builds upon and advances this discourse by addressing uncertainty quantification, selective prediction, and human referral under acquisition defects.
- **Editorial Requirements:**
  - Full reproducible code and open data availability statement.
  - Clear geoscientific context and physical error taxonomy (met via `03_geological_failure_analysis.md`).
  - Standard Elsevier LaTeX format (`cas-dc` or `elsarticle`).
- **Fit Assessment:** **EXCELLENT (Rank 1).**

### 1.2 Alternative Venues
- **Alternative 1: *Earth Science Informatics* (Springer):** Strong focus on data science, ontology, and uncertainty modeling in earth systems. Good fit if an informatics-heavy angle is preferred.
- **Alternative 2: *Geoenergy Science and Engineering* (Elsevier):** Focuses on subsurface reservoir characterization. Suitable if reservoir drill-core applications are emphasized.

---

## 2. Submission Readiness Verification Checklist

| Verification Category | Requirement | Audit Evidence | Status |
| :--- | :--- | :--- | :---: |
| **Literature Gap** | Distinct from Li et al. (2025), Serdoun et al. (2026), and Bourezgui (2026) | Documented in `00_novelty_and_prior_art.md` | **PASS** |
| **Data Provenance** | Physical inspection of 21,000 core images; independent defective pool verified | Documented in `01_data_checkpoint_provenance.md` | **PASS** |
| **Preregistration** | Protocol and split manifests locked and SHA-256 hashed prior to evaluation | Documented in `02_preregistered_protocol.md` | **PASS** |
| **Empirical Rigor** | Real models evaluated; 2,000 bootstrap resamples; Wilcoxon and McNemar tests | Documented in `04_results_and_figures.md` | **PASS** |
| **Geological Analysis** | Authentic core image panels; 217 persistent errors audited against mineralogy | Documented in `03_geological_failure_analysis.md` | **PASS** |
| **Scientific Disclosures**| Retrospective benchmark disclosed; lack of borehole coordinates stated | Documented in `05_claims_and_limitations_audit.md` | **PASS** |
| **Figures Quality** | 6 publication figures generated at 300 DPI vector/PNG quality | Located in `manuscript/paperB_uncertainty/figures/` | **PASS** |
| **Tables Provenance** | 6 tables generated directly from raw metrics JSON | Located in `manuscript/paperB_uncertainty/tables/` | **PASS** |
| **Manuscript Package** | Synchronized `main.tex`, `main.md`, `references.bib`, `claims_checklist.md` | Located in `manuscript/paperB_uncertainty/` | **PASS** |

---

## 3. Strongest Defensible Research Contribution

Paper B establishes the first rigorous empirical benchmark evaluating **when automated rock-core classifiers should abstain and refer samples to human geologists under imperfect drilling-rig acquisition conditions**.

Specifically, the paper proves:
1. Conventional confidence scores fitted under clean conditions cannot be deployed safely in field drilling environments because domain shift produces catastrophic selective risk violations ($9.63\%$ realized error vs. $5\%$ target ceiling).
2. Domain-matched calibration on an untouched representative defective development pool restores empirical risk control ($4.52\%$) and conformal coverage ($95.57\%$).
3. Automated selective classification enables a flexible, economically optimizable operational triage: safely automating $84.6\%$ of clean/mildly defective core logs while routing the $15.4\%$ of severe acquisition defects (mud smears, heavy fracturing, gradational boundaries) to human geologists.

---

## 4. Final Submission Verdict

**VERDICT: GO FOR SUBMISSION.**  
The study is complete, scientifically sound, reproducible, and fully backed by authentic empirical artifacts.
