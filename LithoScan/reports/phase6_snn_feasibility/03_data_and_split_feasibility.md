# Phase 6 — Data and Split Feasibility Report

**Document ID:** `reports/phase6_snn_feasibility/03_data_and_split_feasibility.md`  
**Date:** October 2026  
**Status:** COMPLETE & VERIFIED  
**Objective:** Audit data partitions, provenance metadata, and borehole group-separation feasibility for DCID and external datasets (GSSA HyLogger) to ensure honest, reproducible scientific claims.

---

## 1. DCID Dataset Evaluation Protocol & Limitations

### A. Current Structure and Inventory
- **Location:** `C:\Users\punee\Desktop\Software_lithology\DCID\DCID`
- **Partitions:**
  - `DCID-512-7`: 7 major lithological classes, $512 \times 512$ resolution. 28,000 train images (4,000/class), 7,000 test images (1,000/class).
  - `noise-512-7`: Defective/imperfect cohort, 11,200 train images (1,600/class), 2,800 test images (400/class).
  - `DCID-512-35` and `noise-512-35`: 35 fine-grained classes (140,000 clean total, 56,000 noisy total).

### B. Mandatory Partitioning Rules for Phase 6
1. **Hold-Out Integrity:**
   - The official test sets (`DCID-512-7/test` and `noise-512-7/test`) are held out completely from model development, architecture selection, hyperparameter tuning, and pilot verification.
   - Any evaluation during Phase 6 pilot activities is restricted exclusively to the validation split.
2. **Deterministic Validation Split:**
   - A deterministic 10% stratified validation split is created from the official training directory using random seed `42` (`torch.Generator().manual_seed(42)`):
     - For `DCID-512-7`: 25,200 training samples (3,600/class) and 2,800 validation samples (400/class).
     - For `noise-512-7`: 10,080 training samples (1,440/class) and 1,120 validation samples (160/class).
   - This exact partition must be used across all architectures (SNN and ANN baselines) to ensure comparability.
3. **Classification Level:**
   - **MANDATORY LABEL:** All evaluations conducted on DCID must be designated strictly as **"Image-Level Random-Split Evaluation"**.
   - Because DCID lacks released crop-to-borehole and depth-interval identifiers, **it must NEVER be described as "source-separated", "well-separated", or "out-of-domain"**.

---

## 2. GSSA HyLogger Dataset Audit & Source Separation Feasibility

### A. Metadata and Access Inspection
- **Source:** Alzubaidi et al., *Lithology dataset*, Mendeley Data, V1 (2022), DOI: `10.17632/b6rj29mrss.1`.
- **License:** Creative Commons Attribution 4.0 International (**CC BY 4.0**).
- **Format:** 12,000 pre-cropped patches ($224 \times 224$ pixels) across 4 categories: Sandstone, Limestone, Shale, and Non-core/Garbage.
- **Borehole Identifiers in Public Mendeley Release:**
  - Inspection of the Mendeley Data file structure indicates that images are packaged into class folders (`Sandstone/`, `Limestone/`, `Shale/`, `Garbage/`) with numeric index filenames (e.g., `sandstone_001.png` to `sandstone_3000.png`).
  - **CRITICAL FINDING:** In the public Mendeley zip release, individual image filenames **do not encode original well names, drill-hole ENO numbers, or downhole depth coordinates**.
  - To recover borehole IDs, one must cross-reference image MD5 hashes or coordinate stamps against the raw South Australian SARIG/HyLogger database, which requires dedicated geospatial data mining.

### B. Overlap and Lineage Analysis with DCID
- While DCID (Li et al., 2025) contains drill-core imagery from international geological repositories, its authors noted the inclusion of open-file government core collections.
- Because both datasets incorporate imagery scanned from South Australian core archives (SARIG / GSSA), there is an unquantified risk that certain core intervals overlap.
- **SCIENTIFIC DETERMINATION:** GSSA HyLogger cannot be treated as an independent external out-of-domain test until borehole-level exclusion is confirmed.

### C. Label Mapping Requirements
If GSSA HyLogger is used for a future external evaluation:
- **DCID Classes:** 7 or 35 classes.
- **GSSA Classes:** Sandstone (Class 0), Limestone (Class 1), Shale (Class 2), Non-core (Class 3).
- **Exact Mapping:**
  - DCID `1.Red sandstone`, `3.Light gray fine sandstone`, `5.Gray sandstone`, `6.Sandstone` $\to$ GSSA `Sandstone`.
  - DCID `2.Gray-black mudstone`, `7.Gray mudstone` $\to$ GSSA `Shale` (mudrock group).
  - DCID does not have a native "tray garbage" class; non-core images in GSSA must either be filtered out or mapped to an OOD/background rejection class.
- **Constraint:** Arbitrary label forced-fitting must not be performed. Cross-dataset evaluation must evaluate either representation transfer (linear probing on frozen embeddings) or be confined to the exact shared lithological subsets.

---

## 3. Generalization Claim Summary

| Dataset Asset | Current Provenance Level | Source Separation Feasibility | Permitted Evaluation Claim |
| :--- | :--- | :--- | :--- |
| **DCID-512-7** | Image-level only (no well IDs) | **Not Feasible** | *"Image-level random-split classification"* |
| **noise-512-7** | Image-level only (defective cohort) | **Not Feasible** | *"Inter-cohort distribution shift under natural drilling defects"* |
| **GSSA HyLogger (Mendeley)** | Class-level only (well IDs stripped in zip) | **Conditional** (Requires raw SARIG borehole recovery) | *"Cross-dataset transfer on shared lithology classes"* |
| **AuScope NVCL (Raw)** | Borehole IDs, depths, coordinates verified | **Feasible (Gold Standard)** | *"Leave-One-Well-Out (LOWO) source-separated spatial generalization"* |

### Explicit Scientific Rule for Phase 6
Source-level generalization **CANNOT** be claimed with currently downloaded datasets. Phase 6 SNN evaluation must strictly report performance as **image-level evaluation under natural acquisition distribution shift**, avoiding any unsubstantiated spatial separation claims.
