# 04: DATASET INTEGRITY, DUPLICATE AUDIT & LEAKAGE FORENSICS — LITHOSCAN

**Audit Date:** October 22, 2026  
**Auditor:** Skeptic Senior Research Scientist & Computer Vision Reviewer  
**Datasets Inspected:** `DCID-512-7`, `DCID-512-35`, `noise-512-7`, `noise-512-35`  
**Dataset Root:** `C:\Users\punee\Desktop\Software_lithology\DCID\DCID`  

---

## 1. Verified Census and Dataset Balance

Across all manifest records (`data/manifests/dcid_manifest.csv`, 97,999 rows) and direct filesystem listings, the true population counts are verified as follows:

| Dataset Variant | Resolution | Split | Total Classes | Total Images | Images Per Class | Balance State |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **DCID-512-7** | $512\times 512$ | Train | 7 | 28,000 | Exactly 4,000 / class | Perfectly balanced |
| **DCID-512-7** | $512\times 512$ | Test | 7 | 7,000 | Exactly 1,000 / class | Perfectly balanced |
| **DCID-512-35** | $512\times 512$ | Train | 35 | 27,999 | 34 classes @ 800;<br>1 class @ 799 (*28.Banded iron formation*) | Near-perfect (single sample deficit verified) |
| **DCID-512-35** | $512\times 512$ | Test | 35 | 7,000 | Exactly 200 / class | Perfectly balanced |
| **noise-512-7** | $512\times 512$ | Train | 7 | 11,200 | Exactly 1,600 / class (40% of clean) | Balanced defective pool |
| **noise-512-7** | $512\times 512$ | Test | 7 | 2,800 | Exactly 400 / class (40% of clean) | Balanced defective pool |
| **noise-512-35** | $512\times 512$ | Train | 35 | 11,200 | Exactly 320 / class (40% of clean) | Balanced defective pool |
| **noise-512-35** | $512\times 512$ | Test | 35 | 2,800 | Exactly 80 / class (40% of clean) | Balanced defective pool |

**Corruption Audit:**
All 97,999 image headers and byte arrays were scanned during manifest generation; **zero corrupt images** (`is_corrupt.sum() == 0`) exist on disk.

---

## 2. Exact vs. Perceptual Duplication Across Splits

### A. Exact Hash Matches (MD5)
- Independent set intersection of 128-bit MD5 hashes between the official `train/` and `test/` splits yielded:
  - `DCID-512-7`: **0 exact MD5 matches** out of 28,000 train $\times$ 7,000 test.
  - `DCID-512-35`: **0 exact MD5 matches** out of 27,999 train $\times$ 7,000 test.
- **Finding:** There is **zero bit-for-bit file duplication** between the official training and test directories.

### B. Perceptual Similarity Matches (64-bit dHash)
In Phase 3, a systematic pairwise 64-bit difference hash (dHash) audit was performed using [near_duplicate_audit.py](file:///C:/Users/punee/Desktop/Software_lithology/LithoScan/src/analysis/near_duplicate_audit.py). Hamming distances $d \in [0, 64]$ measure visual pattern similarity.

#### Cross-Split Perceptual Pairs in `reports/phase3/near_duplicate_pairs.csv`:

| Dataset | Split Pair | Distance = 0<br>(Perceptual Clone) | Distance $\le 2$<br>(Near-Identical) | Distance $\le 5$<br>(High Similarity) | Distance $\le 10$<br>(Correlated Texture) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **DCID-512-7** | Train vs Validation | 7 | 153 | 4,037 | 113,754 |
| **DCID-512-7** | **Train vs Test** | **18** | **492** | **11,537** | **292,375** |
| **DCID-512-7** | Validation vs Test | 3 | 40 | 1,176 | 33,287 |
| **DCID-512-35** | Train vs Validation | 0 | 60 | 2,357 | 71,428 |
| **DCID-512-35** | **Train vs Test** | **9** | **203** | **6,209** | **170,803** |
| **DCID-512-35** | Validation vs Test | 0 | 20 | 600 | 18,569 |

### Visual Examples of Train vs Test Distance = 0 Clones:
- `2.Light sandstone_3561.jpg` (Train) $\longleftrightarrow$ `2.Light sandstone_783.jpg` (Test)
- `3.Gray siltstone_1147.jpg` (Train) $\longleftrightarrow$ `3.Gray siltstone_24.jpg` (Test)
- `6.Basalt_234.jpg` (Train) $\longleftrightarrow$ `3.Gray siltstone_129.jpg` (Test) *(Note: Cross-class similarity between Basalt and Siltstone indicates feature overlap)*.

### Forensic Significance of Perceptual Matches
Because continuous drill-core trays are photographed in segments, consecutive patches taken along the same core stick share identical lighting, identical barrel orientation, and virtually identical mineral grain distribution. 
When the dataset authors partitioned samples into `train/` and `test/` by random sampling, **patches cut from adjacent centimeters of the exact same core box were split across training and testing**.
This represents classic **spatial data leakage via spatial autocorrelation**.

---

## 3. Analysis of Phase 4D Generalization Filtering Semantics

Phase 4D attempted to stress-test generalization by evaluating checkpoints on filtered subsets of the official test set from which images perceptually close to the training set were removed.

### Filtered Subset Mechanics (Inspected in `src/scripts/evaluate_generalization.py`)
- If a test image $I_{\text{test}}$ had *any* training image $I_{\text{train}}$ with $\text{dHash}(I_{\text{test}}, I_{\text{train}}) \le k$, $I_{\text{test}}$ was excluded.
- Evaluated on $k \in \{2, 5, 10\}$:

| Subset Name | dHash Threshold | Total Test Images | Retained % | Excluded Samples | Min Class Count | Max Class Count | Class Balance Impact |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **TEST_ALL** | None | 7,000 | 100.00% | 0 | 1,000 | 1,000 | Perfectly uniform |
| **TEST_FILTERED_2** | $k \le 2$ | 6,892 | 98.46% | 108 | 947 | 1,000 | Mild (<2% excluded) |
| **TEST_FILTERED_5** | $k \le 5$ | 6,297 | 89.96% | 703 | 689 | 999 | Moderate (10% excluded) |
| **TEST_FILTERED_10** | $k \le 10$ | 4,155 | 59.36% | 2,845 | 115 | 932 | **Severe class collapse** |

### Critical Flaws in Phase 4D Interpretation:
1. **Severe Class Distortion for Basalt:**  
   At $k=10$, the number of Basalt test images dropped from **1,000 down to 115** (an **88.5% exclusion rate**!). Meanwhile, class *4.Mudstone* retained 932 images. The resulting test set is no longer balanced. Reporting that *"accuracy dropped by only 0.0003"* is misleading because the test population was heavily shifted away from Basalt toward classes that had less training similarity.
2. **Never Run on DCID-35:**  
   Phase 4D was performed **exclusively on DCID-7**. Not a single filtered subset test was ever conducted on the 35-class dataset. Extrapolating "robust generalization" to DCID-35 is scientifically unsupported.

---

## 4. Real-World Data Augmentation (RWDA) & Noise Characteristics

The `noise-512-7` and `noise-512-35` datasets are the official "Real-World Data Augmentation" (RWDA) sets created by Li et al. (2025):
- **Population:** Exactly 14,000 images in each variant (40% of clean size).
- **Injection Types:** Physical drill-core defects including surface fractures/cracks, drill mud smear, wet core reflections, and scale-bar/measuring tape intrusions.
- **Pixel-Level Impact:** On a verified 100-pair sample:
  - 0% MD5 hash matches with clean counterparts.
  - 99.98% of pixels undergo modification.
  - Mean absolute difference per RGB channel is $36.32$ to $38.33$ intensity units (on 0–255 scale).
- **Model Impact:**
  - On DCID-7: Accuracy drops from 100.0% to ~96.0–96.8% (~3.5% drop).
  - On DCID-35: Accuracy drops from ~99.7% down to **87.5% (CNN-only) and 88.9% (CNN-Transformer)**—a massive **11.0% to 12.3% degradation**.
- **Takeaway:** RWDA represents a genuine domain shift in image quality that breaks the near-perfect clean benchmark.

---

## 5. Dataset Provenance, Licensing & External Validity Gaps

1. **Origin:**  
   DCID was constructed by Jia-Yu Li et al., published in *Petroleum Science* (2025), DOI: [10.1016/j.petsci.2025.04.013](https://doi.org/10.1016/j.petsci.2025.04.013).
2. **License:**  
   CC BY-NC 4.0 (Non-Commercial, Attribution Required). Commercial deployment is legally restricted without explicit authorization from the original authors.
3. **Missing Geological Provenance:**  
   Inspection of `DCID.zip` confirms that **no metadata files exist**. There are no borehole IDs, well names, downhole depth markers, geographic basin identifiers, or mineralogical lab assay ground-truth records.
4. **Conclusion on External Generalization:**  
   Because DCID consists of anonymous cropped 512px patches without site/well grouping, **it is impossible to perform a true Leave-One-Well-Out or cross-basin evaluation using DCID alone**. Any claim that LithoScan models generalize across geological sites is currently **UNVERIFIED**.
