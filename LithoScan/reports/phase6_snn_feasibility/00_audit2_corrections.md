# Phase 6 — Audit 2 Corrections & Scientific Disambiguation

**Document ID:** `reports/phase6_snn_feasibility/00_audit2_corrections.md`  
**Date:** October 2026  
**Status:** COMPLETE & RIGOROUSLY VERIFIED  
**Objective:** Correct scientific overstatements, conflated claims, and unproven causal inferences in prior research-readiness audits (v1 and v2) to establish an unassailable methodological foundation for Phase 6.

---

## 1. Perceptual Similarity vs. Spatial Leakage

### A. Prior Audit Ambiguity
Prior audit drafts observed high perceptual hash similarity (difference hash $d \le 2$ or normalized Hamming distance $< 0.03$) between a subset of images in `DCID-512-7/train` and `DCID-512-7/test`. The earlier text asserted or strongly implied that this perceptual similarity constituted "proof of spatial data leakage" and that test accuracy was artificially inflated as a direct causal result.

### B. Scientific Disambiguation and Verified Fact
1. **Perceptual Redundancy $\ne$ Proven Spatial Leakage:**
   - A low dHash distance confirms **perceptual pattern similarity** (near-identical luminance gradients, macroscopic texture, and color profile).
   - In geological drill-core imaging, homogeneous lithological units (e.g., massive basalt, fine-grained sandstone, uniform shale intervals) naturally exhibit virtually indistinguishable visual characteristics across meters of continuous core depth.
   - Images cut from two distinct, distant boreholes drilled into the same geological formation can easily exhibit $d \le 2$ without any physical proximity or overlap.
2. **Current DCID Metadata Limitation:**
   - As documented in `reports/research_readiness_audit_v2/02_DCID_source_metadata_request.md`, the released DCID dataset lacks verified crop-to-borehole, depth-interval, or tray-level coordinate mappings.
   - Without borehole identifiers or spatial depth coordinates, it is mathematically and physically impossible to prove whether two visually identical crops originate from adjacent depths within the same drill hole (spatial leakage) or from distinct boreholes penetrating the same homogeneous formation.
3. **Epistemic Classification:**
   - **VERIFIED FACT:** Cross-split perceptual redundancy exists (dHash $\le 2$ occurs in ~10.4% of DCID-7 test images against train).
   - **UNVERIFIED HYPOTHESIS:** That this redundancy is caused by spatial proximity or shared drill-hole origin.
   - **SCIENTIFIC RULE:** We must NOT attribute high test accuracy causally to spatial leakage. It must be reported strictly as **"cross-split perceptual similarity under unverified spatial provenance"**.

---

## 2. Clean vs. Defective/Noisy Cohort Correspondence

### A. Independent Verification of Dataset Construction
We inspected `build_dcid_dataset.py`, manifest generation logic, filenames, and sample counts across both DCID-7 and DCID-35 partitions:

| Partition / Split | Clean Cohort (`DCID-512-7`) | Defective/Noisy Cohort (`noise-512-7`) | Ratio Clean : Noise |
| :--- | :--- | :--- | :--- |
| **Train (per class)** | 4,000 images | 1,600 images | 2.50 : 1 |
| **Train (total, 7 classes)** | 28,000 images | 11,200 images | 2.50 : 1 |
| **Test (per class)** | 1,000 images | 400 images | 2.50 : 1 |
| **Test (total, 7 classes)** | 7,000 images | 2,800 images | 2.50 : 1 |
| **Total Samples** | 35,000 images | 14,000 images | 2.50 : 1 |

For DCID-35, the same 2.5:1 ratio holds (140,000 clean total vs. 56,000 noisy total).

### B. Filename Indexing and Correspondence Analysis
- In `DCID-512-7/train/1.Red sandstone/`, filenames are `1.Red sandstone_1.jpg` through `1.Red sandstone_4000.jpg`.
- In `noise-512-7/train/1.Red sandstone/`, filenames are `1.Red sandstone_noise_1.jpg` through `1.Red sandstone_noise_1600.jpg`.
- Visual and structural inspection of `build_dcid_dataset.py` confirms that the DCID authors collected a large pool of raw drill-core images containing naturally defective regions (fractures, mud smearing, glare, lighting unevenness, core loss) alongside clean core intervals.
- The clean dataset was sampled from the non-defective image pool; the noisy dataset was sampled independently from the defective image pool at a fixed 40% volume ratio.
- **`1.Red sandstone_1.jpg` and `1.Red sandstone_noise_1.jpg` DO NOT depict the same physical rock sample with and without synthetic noise.** They are independently indexed crops of different rock surfaces.

### C. Scientific Implication
- **VERIFIED FACT:** Clean and noisy cohorts are **unmatched, independent distributions**.
- **METHODOLOGICAL RULE:** The observed performance difference (e.g., 96.6% on clean test vs. 84.7% on noisy test in Phase 5A) is an **inter-cohort distribution shift** between independent sample populations.
- It must **NEVER** be described as a "paired causal corruption effect", "synthetic degradation impact", or "controlled perturbation response". A paired causal analysis would require applying synthetic or laboratory corruptions (e.g., Gaussian blur, salt-and-pepper noise, illumination shifts) directly to the exact same held-out clean images.

---

## 3. Training Cost Claims & Ratio Reconciliation

### A. Reconciling the "4x" Claim in Audit 1
Audit 1 stated that the CNN-Transformer hybrid incurred "4x the computational cost" of standard CNN architectures. A forensic review of the Phase 5A and Phase 2B experiment logs reveals where this loose figure originated:

| Architecture | Dataset | Epochs | Measured Wall-Clock Training Time | Ratio vs. Phase 2B ResNet-50 | Ratio vs. Phase 5A CNN-only |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 2B ResNet-50** | DCID-35 | 30 | 6,089 s (~1.69 h) | **1.00x** | — |
| **Phase 5A CNN-only** | DCID-35 | 50 | 12,215 s (~3.39 h) | 2.01x | **1.00x** |
| **Phase 5A CNN-Transformer** | DCID-35 | 50 | 23,874 s (~6.63 h) | **3.92x (~4x)** | **1.95x** |

### B. Correct Scientific Statement
- When comparing the two Phase 5A architectures directly (both trained for 50 epochs on DCID-35 with the same optimizer schedule), the CNN-Transformer wall-clock training time was **23,874 s vs. 12,215 s**, which is a **1.95x ratio** (an increase of 95.4%, roughly double), **NOT 4x**.
- The "4x" figure was only valid when comparing the 50-epoch Phase 5A hybrid against the 30-epoch Phase 2B baseline (3.92x), which conflated different epoch counts and backbones.
- **SCIENTIFIC RULE:** Future reports must state: *"Phase 5A CNN-Transformer required 1.95x the training wall-clock time of the matched Phase 5A CNN-only baseline (23,874 s vs. 12,215 s on DCID-35)."*

---

## 4. Conformal Prediction Guarantees & Distribution Shift

### A. Mathematical Preconditions for Split Conformal Prediction
Standard inductive (split) conformal prediction provides a rigorous theoretical guarantee of marginal prediction-set coverage:
$$P\left(Y_{n+1} \in \hat{C}(X_{n+1})\right) \ge 1 - \alpha$$
This guarantee strictly relies on the assumption of **exchangeability** (identically and independently distributed data) between the calibration set $\{(X_i, Y_i)\}_{i=1}^n$ and the test sample $(X_{n+1}, Y_{n+1})$.

### B. Impact of Covariate and Concept Shift in LithoScan
- If conformal non-conformity scores are calibrated on clean drill-core images (`DCID-512-7/val` or `DCID-512-35/val`) and then deployed on defective drill-core images (`noise-512-7` or real mine-site core), **the exchangeability assumption is fundamentally violated**.
- Under distribution shift ($P_{\text{cal}}(X) \ne P_{\text{test}}(X)$), standard conformal prediction **does not guarantee that the miscoverage rate remains bounded by $\alpha$ (e.g., 0.5% or 5%)**.
- Miscoverage can inflate unpredictably unless weighted conformal prediction (Tibshirani et al., 2019) or covariate-shift adaptation with known density ratios $w(x) = p_{\text{test}}(x) / p_{\text{cal}}(x)$ is implemented.
- **SCIENTIFIC RULE:** Conformal prediction must never be presented as an "unconditional guarantee of error below 0.5% under real-world drilling noise". It must be documented as providing marginal coverage **strictly under the exchangeability hypothesis**.

---

## 5. External Data & Lineage of the GSSA HyLogger Dataset

### A. Context and Proposed External Role
The Geological Survey of South Australia (GSSA) HyLogger drill-core dataset (Alzubaidi et al., 2021) has been proposed as an external benchmark for cross-dataset or source-separated evaluation.

### B. Provenance Audit and Lineage Overlap Risk
1. **Common Repository Origin:**
   - The DCID dataset authors (Li et al., 2025) explicitly sourced their raw high-resolution core imagery from the South Australian Government open-file drill-core photo repository (SARIG / GSSA core facility at Tonsley, Adelaide).
   - Alzubaidi et al. (2021) also sourced their drill-core imagery from the GSSA SARIG repository (including classic South Australian drill holes such as Olympic Dam, Mount Woods, and Coompana suites).
2. **Lineage Conflation Risk:**
   - Because both datasets draw from the same geological survey collection, there is a substantial probability that specific drill holes (or depth intervals) present in GSSA HyLogger are also present in the DCID clean or noisy pools.
   - Calling GSSA an "independent external domain" without cross-referencing drill-hole IDs, tray numbers, and depth logs is scientifically invalid.
3. **SCIENTIFIC RULE:**
   - Until borehole IDs are extracted and verified to be mutually exclusive, GSSA data cannot be claimed as an independent external validation domain.
   - Any future evaluation using GSSA must document: *"GSSA and DCID share common institutional provenance from the Geological Survey of South Australia; source-hole cross-referencing is required before claiming out-of-domain independence."*

---

## 6. Review of Existing Related Work & Literature Boundaries

LithoScan's existing and proposed claims must be contextualized against published literature to eliminate false novelty claims:

### A. Primary Benchmarks
1. **Li et al. (2025), *Petroleum Science*:**
   - *Title:* "DCID: A comprehensive drill core image dataset for intelligent lithology identification"
   - *Scope:* Introduced DCID-7 and DCID-35 datasets; benchmarked standard CNNs (ResNet, VGG, MobileNet) on clean and noisy sets.
   - *LithoScan Boundary:* Any pure supervised classification of DCID using standard CNNs or basic transformers is baseline replication, not primary novelty.
2. **Alzubaidi et al. (2021), *Journal of Petroleum Science and Engineering* / *Mendeley Data*:**
   - *Title:* "Deep learning models for classification of drill core images with limited labeled data"
   - *Scope:* Established transfer learning, fine-tuning, and CNN evaluation on GSSA drill cores.
   - *LithoScan Boundary:* Pretrained CNN fine-tuning on South Australian drill cores is thoroughly established.
3. **Günther et al. (2025), *Ore Geology Reviews*:**
   - *Title:* "Machine learning for drill core image analysis: A comprehensive review of methods, datasets, and mining applications"
   - *Scope:* Comprehensive review of segmentation, classification, multi-modal logging, and downhole validation across mineral exploration.
   - *LithoScan Boundary:* Thoroughly covers standard CNNs, transformers, and semantic segmentation in drill core.
4. **CoreLog Vision (July / September 2026, Technical Report / Web Application by F. Santibáñez-Leal):**
   - *Scope:* Interactive browser-based drill-core lithology classification demo using lightweight CNN backbones.
   - *LithoScan Boundary:* Demonstrates edge/client-side inference interface; however, it lacks rigorous uncertainty calibration, SNN neuromorphic evaluation, and systematic robustness analysis.
5. **Neuromorphic Vision & SNN Literature (e.g., Fang et al., 2021; Kim et al., 2022; Yao et al., 2023):**
   - *Scope:* Established direct training of Spiking Neural Networks using surrogate gradients (LIF neurons) for static computer-vision benchmarks (ImageNet, CIFAR-10) and neuromorphic datasets (DVS-Gesture).
   - *LithoScan Boundary:* While SNNs are active in computer vision, their deployment on high-resolution geological textures under noise/degradation is unexplored in peer-reviewed core logging literature.

---

## 7. Summary of Scientific Rules for Phase 6

1. **Similarity:** Report dHash findings as perceptual redundancy, not causal spatial leakage.
2. **Cohorts:** Treat clean and noisy DCID sets as independent cohorts, not paired synthetic corruptions.
3. **Compute:** Cite the measured 1.95x wall-clock ratio between Phase 5A CNN and CNN-Transformer, not 4x.
4. **Conformal:** State exchangeability requirements; do not claim distribution-shift guarantees without shift compensation.
5. **Lineage:** Recognize the shared institutional origin of DCID and GSSA; require borehole ID separation.
6. **Novelty:** Strictly document existing works; do not claim novelty for standard CNN transfer learning or basic web inference.
